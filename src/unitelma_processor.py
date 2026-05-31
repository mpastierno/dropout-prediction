import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.config_unitelma import MACRO_MAPPING, CLASSIFIERS


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, dropout_rate=0.2):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout_rate)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1] 
        dropped_hidden = self.dropout(last_hidden)
        logits = self.fc(dropped_hidden).squeeze(-1)
        return logits


class UnitelmaProcessor:
    """
    Processor class that handle the pipeline for preprocessing, training and explaination with kernel
    shap of unitelma's dataset
    """
    def __init__(self, train_path, test_path, output_path, use_macro: bool):
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.output_path = output_path
        self.macro_mapping = MACRO_MAPPING
        self.use_macro = use_macro
        
        self.device = self._get_torch_device()
        print(f"[INFO] Device PyTorch -> {self.device}")


    def _get_torch_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")

    def _apply_macro_areas(self, df) -> pd.DataFrame:
        metadata_cols = ['global_id', 'day', 'dropout']
        df_macro = df[metadata_cols].copy()
        
        mapping = self.macro_mapping.copy()
        mapped_features = [act for actions in mapping.values() for act in actions]
        action_cols = [c for c in df.columns if c not in metadata_cols]
        others = [c for c in action_cols if c not in mapped_features]
        mapping['others'] = others

        for macro_name, actions in mapping.items():
            valid_actions = [col for col in actions if col in df.columns]
            if valid_actions:
                df_macro[macro_name] = df[valid_actions].sum(axis=1)
            else:
                df_macro[macro_name] = 0
                
        return df_macro

    def _calculate_metrics(self, y_true, y_pred, y_prob):
        pr_auc_1 = average_precision_score(y_true, y_prob)
        pr_auc_0 = average_precision_score(1 - y_true, 1 - y_prob)
        
        support_1 = (y_true == 1).sum()
        support_0 = (y_true == 0).sum()
        total_samples = len(y_true)
        
        weighted_pr_auc = (pr_auc_1 * support_1 + pr_auc_0 * support_0) / total_samples

        return {
            'accuracy': round(accuracy_score(y_true, y_pred), 4),
            'precision': round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4),
            'recall': round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4),
            'f1': round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
            'roc_auc': round(roc_auc_score(y_true, y_prob), 4),
            'pr_auc': round(weighted_pr_auc, 4)
        }


    def prepare_ml_data(self, lag, time_aggregation='flatten'):
        def _process_split(df):
            df_lag = df[df['day'] <= lag].copy()
            if self.use_macro:
                df_lag = self._apply_macro_areas(df_lag)
                
            metadata_cols = ['global_id', 'day', 'dropout']
            features = [c for c in df_lag.columns if c not in metadata_cols]
            
            if time_aggregation == 'flatten':
                X = df_lag.pivot_table(index=['global_id'], columns='day', values=features, fill_value=0)
                X.columns = [f"{act}_day_{d}" for act, d in X.columns]
            elif time_aggregation == 'sum':
                X = df_lag.groupby('global_id')[features].sum()
                X.columns = [f"{col}_sum" for col in X.columns]
            
            y = df_lag.groupby('global_id')['dropout'].first().astype(int).reset_index(drop=True)
            return X.reset_index(drop=True), y

        X_train, y_train = _process_split(self.train_df)
        X_test, y_test = _process_split(self.test_df)

        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        return X_train, y_train, X_test, y_test

    def prepare_dl_data(self, lag):
        def _process_split_dl(df):
            df_lag = df[df['day'] <= lag].copy()
            assert df_lag['day'].astype(int).max() == lag, f"[CRITICAL] Days greater than {lag} found"

            if self.use_macro:
                df_lag = self._apply_macro_areas(df_lag)
                
            metadata_cols = ['global_id', 'day', 'dropout']
            features = [c for c in df_lag.columns if c not in metadata_cols]

            df_lag['dropout'] = (df_lag['dropout'] >= 0.5).astype(int)
            df_lag = df_lag.sort_values(by=['global_id', 'day']).reset_index(drop=True)

            X_numpy = df_lag[features].values
            X_3d = X_numpy.reshape(-1, lag, len(features))

            X_tensor = torch.tensor(X_3d, dtype=torch.float32)
            y_tensor = torch.tensor(df_lag.groupby('global_id')['dropout'].first().values, dtype=torch.float32)

            return X_tensor, y_tensor, features

        X_train_t, y_train_t, features = _process_split_dl(self.train_df)
        X_test_t, y_test_t, _ = _process_split_dl(self.test_df)

        return X_train_t, y_train_t, X_test_t, y_test_t, features


    def apply_feature_selection(self, X_train, y_train, X_test, corr_threshold=0.75):
        print("[INFO] Starting Feature Selection...")
        
        # Dead Actions
        zero_cols = [col for col in X_train.columns if X_train[col].sum() == 0]
        X_train_fs = X_train.drop(columns=zero_cols)
        X_test_fs = X_test.drop(columns=zero_cols)
        print(f"   -> Removed {len(zero_cols)} zero-variance features")

        # Collinearity
        corr_matrix = X_train_fs.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
        X_train_fs = X_train_fs.drop(columns=to_drop)
        X_test_fs = X_test_fs.drop(columns=to_drop)
        print(f"   -> Removed {len(to_drop)} correlated features")

        # RFECV
        rf_estimator = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42, n_jobs=-1)
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rfecv = RFECV(estimator=rf_estimator, step=0.1, cv=cv_strategy, scoring='average_precision', min_features_to_select=5, n_jobs=-1)
        
        rfecv.fit(X_train_fs, y_train)
        selected_features = X_train_fs.columns[rfecv.support_]
        
        print(f"[SUCCESS] Features to keep: {len(selected_features)}")
        return X_train_fs[selected_features], X_test_fs[selected_features], selected_features

    def evaluate_ml_model(self, model_config, X_train, y_train, X_test, y_test):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model_config['estimator'])
        ])
        
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(
            pipeline, 
            param_grid=model_config['params'], 
            cv=list(cv_strategy.split(X_train, y_train)),
            scoring='average_precision',
            n_jobs=-1
        )
        
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_metrics(y_test, y_pred, y_prob)
        metrics['best_params'] = grid.best_params_
        
        return metrics, best_model


    def evaluate_dl_model(self, X_train, y_train, X_test, y_test, input_dim, hidden_dim=64, lr=0.001, batch_size=64, epochs=30):
        num_persist = (y_train == 0).sum().item()
        num_dropout = (y_train == 1).sum().item()

        pos_weight_val = num_persist / num_dropout
        pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(self.device)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = LSTMClassifier(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        # Training Loop
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
        # Eval Loop
        model.eval()
        with torch.no_grad():
            X_test_dev = X_test.to(self.device)
            logits = model(X_test_dev)
            y_prob_tensor = torch.sigmoid(logits)
            
            y_prob = y_prob_tensor.cpu().numpy()
            y_pred = (y_prob >= 0.5).astype(int)
            y_true = y_test.numpy()

        metrics = self._calculate_metrics(y_true, y_pred, y_prob)
        return metrics, model


    def generate_ml_shap(self, model, X_train, X_test, model_name, lag, output_dir):
        def predict_fn(x):
            x_df = pd.DataFrame(x, columns=X_train.columns)
            return model.predict_proba(x_df)[:, 1]

        background = shap.sample(X_train, 100, random_state=42)
        explainer = shap.KernelExplainer(predict_fn, background)
        
        X_test_sample = X_test.sample(n=min(200, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(X_test_sample, silent=True)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_sample, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_ml_{model_name}_lag{lag}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def generate_dl_shap(self, model, X_train_tensor, X_test_tensor, feature_names, lag, model_name, output_dir):
        model.eval()

        def predict_fn(x_2d_numpy):
            x_3d = x_2d_numpy.reshape(-1, lag, len(feature_names))
            x_tensor = torch.tensor(x_3d, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                preds = torch.sigmoid(model(x_tensor)).cpu().numpy().flatten()
            return preds
        
        X_train_2d = X_train_tensor.numpy().reshape(X_train_tensor.shape[0], -1)
        X_test_2d = X_test_tensor.numpy().reshape(X_test_tensor.shape[0], -1)

        flat_feature_names = [f"{feat}_day_{d+1}" for d in range(lag) for feat in feature_names]

        background = shap.sample(X_train_2d, 50, random_state=42)
        explainer = shap.KernelExplainer(predict_fn, background)

        X_test_sample = shap.sample(X_test_2d, min(50, X_test_2d.shape[0]), random_state=42)
        shap_values = explainer.shap_values(X_test_sample, silent=True)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_sample, feature_names=flat_feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_dl_{model_name}_lag{lag}.png"), dpi=300, bbox_inches='tight')
        plt.close()