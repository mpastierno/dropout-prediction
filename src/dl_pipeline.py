import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import shap
import matplotlib.pyplot as plt
from datetime import datetime

from src.base_pipeline import BasePipeline

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.2):
        super(LSTMClassifier, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        logits = self.fc(self.dropout(h_n[-1])).squeeze(-1)
        return logits


class DLPipeline(BasePipeline):
    def __init__(self, train_path, test_path, output_path, use_macro: bool):
        super().__init__(train_path, test_path, output_path, use_macro)

    def prepare_data(self, lag):
        def _process_split_dl(df, train_features=None):
            df_lag = df[df['day'] <= lag].copy()
            if self.use_macro:
                df_lag = self._apply_macro_areas(df_lag)
                
            metadata_cols = ['global_id', 'day', 'dropout']
            
            if train_features is None:
                features = [c for c in df_lag.columns if c not in metadata_cols]
            else:
                features = train_features
                for f in features:
                    if f not in df_lag.columns:
                        df_lag[f] = 0
            
            df_lag['dropout'] = (df_lag['dropout'] >= 0.5).astype(int)
            df_lag = df_lag.sort_values(by=['global_id', 'day']).reset_index(drop=True)

            X_3d = df_lag[features].values.reshape(-1, lag, len(features))
            y_tensor = torch.tensor(df_lag.groupby('global_id')['dropout'].first().values, dtype=torch.float32)

            return torch.tensor(X_3d, dtype=torch.float32), y_tensor, features

        X_train_t, y_train_t, features = _process_split_dl(self.train_df)
        X_test_t, y_test_t, _ = _process_split_dl(self.test_df, train_features=features)

        return X_train_t, y_train_t, X_test_t, y_test_t, features

    def evaluate_model(self, X_train, y_train, X_test, y_test, input_dim, hidden_dim, lr, batch_size, epochs):
        pos_weight_val = (y_train == 0).sum().item() / (y_train == 1).sum().item()
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32).to(self.device))
        
        model = LSTMClassifier(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                
        model.eval()
        with torch.no_grad():
            logits = model(X_test.to(self.device))
            y_prob = torch.sigmoid(logits).cpu().numpy()
            y_pred = (y_prob >= 0.5).astype(int)

        metrics = self._calculate_metrics(y_test.numpy(), y_pred, y_prob)
        return metrics, model

    def generate_shap(self, model, X_train_t, X_test_t, feature_names, lag, output_dir):
        model.eval()
        def predict_fn(x_2d_numpy):
            x_3d = x_2d_numpy.reshape(-1, lag, len(feature_names))
            with torch.no_grad():
                return torch.sigmoid(model(torch.tensor(x_3d, dtype=torch.float32).to(self.device))).cpu().numpy().flatten()
        
        X_train_2d = X_train_t.numpy().reshape(X_train_t.shape[0], -1)
        X_test_2d = X_test_t.numpy().reshape(X_test_t.shape[0], -1)

        explainer = shap.KernelExplainer(predict_fn, shap.sample(X_train_2d, 50, random_state=42))
        X_test_sample = shap.sample(X_test_2d, min(50, X_test_2d.shape[0]), random_state=42)
        shap_values = explainer.shap_values(X_test_sample, silent=True)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_sample, feature_names=[f"{feat}_d{d+1}" for d in range(lag) for feat in feature_names], show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_dl_lstm_lag{lag}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def run(self, lags: list, epochs: int = 30, batch_size: int = 64, hidden_dim: int = 64, lr: float = 0.001, apply_clipping: bool = False):
        results_list = []
        plots_dir = os.path.join(self.output_path, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        for current_lag in lags:
            print(f"\n[INFO] PIPELINE DL (LSTM) | Lag: {current_lag}")
            X_train, y_train, X_test, y_test, features = self.prepare_data(lag=current_lag)

            if apply_clipping:
                lim = torch.quantile(X_train, 0.95, dim=0)
                X_train = torch.min(X_train, lim)
                X_test = torch.min(X_test, lim)

            print(f"[TRAIN] LSTM (Epochs: {epochs}, Hidden: {hidden_dim})")
            metrics, best_model = self.evaluate_model(X_train, y_train, X_test, y_test, len(features), hidden_dim, lr, batch_size, epochs)
            
            self.generate_shap(best_model, X_train, X_test, features, current_lag, plots_dir)

            metrics.update({
                'lag': current_lag,
                'algorithm': 'LSTM',
                'use_macro': self.use_macro,
                'epochs': epochs,
                'batch_size': batch_size,
                'hidden_dim': hidden_dim,
                'lr': lr
            })
            results_list.append(metrics)

        if results_list:
            df_results = pd.DataFrame(results_list)
            csv_path = os.path.join(self.output_path, f"metrics_dl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df_results.to_csv(csv_path, index=False)
            print(f"\n[SUCCESS] CSV saved in: {csv_path}")
        return pd.DataFrame(results_list)