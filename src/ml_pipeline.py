import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

from src.base_pipeline import BasePipeline

class MLPipeline(BasePipeline):
    def __init__(self, train_path, test_path, output_path, use_macro: bool):
        super().__init__(train_path, test_path, output_path, use_macro)

    def prepare_data(self, lag, time_aggregation):
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

    def apply_feature_selection(self, X_train, y_train, X_test, corr_threshold=0.75):
        zero_cols = [col for col in X_train.columns if X_train[col].sum() == 0]
        X_train_fs = X_train.drop(columns=zero_cols)
        X_test_fs = X_test.drop(columns=zero_cols)

        corr_matrix = X_train_fs.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
        X_train_fs = X_train_fs.drop(columns=to_drop)
        X_test_fs = X_test_fs.drop(columns=to_drop)

        rf_estimator = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42, n_jobs=-1)
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rfecv = RFECV(estimator=rf_estimator, step=0.1, cv=cv_strategy, scoring='average_precision', min_features_to_select=5, n_jobs=-1)
        
        rfecv.fit(X_train_fs, y_train)
        selected_features = X_train_fs.columns[rfecv.support_]
        
        return X_train_fs[selected_features], X_test_fs[selected_features]

    def evaluate_model(self, model_config, X_train, y_train, X_test, y_test):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model_config['estimator'])
        ])
        
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(pipeline, param_grid=model_config['params'], cv=list(cv_strategy.split(X_train, y_train)), scoring='average_precision', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_metrics(y_test, y_pred, y_prob)
        return metrics, best_model

    def generate_shap(self, model, X_train, X_test, model_name, lag, output_dir):
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

    def run(self, lags: list, classifiers: dict, apply_clipping: bool, use_fs: bool, time_aggregation: str, corr_threshold: float = 0.75):
        results_list = []
        plots_dir = os.path.join(self.output_path, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        for current_lag in lags:
            print(f"\n[INFO] PIPELINE ML | Lag: {current_lag}")
            X_train, y_train, X_test, y_test = self.prepare_data(lag=current_lag, time_aggregation=time_aggregation)

            if apply_clipping:
                upper_limits = X_train.quantile(0.95)
                X_train = X_train.clip(upper=upper_limits, axis=1)
                X_test = X_test.clip(upper=upper_limits, axis=1)

            if use_fs and not self.use_macro:
                print("[PREPROCESSING] Feature Selection...")
                X_train, X_test = self.apply_feature_selection(X_train, y_train, X_test, corr_threshold)

            for algo_name, model_config in classifiers.items():
                print(f"[TRAIN] {algo_name}...")
                metrics, best_model = self.evaluate_model(model_config, X_train, y_train, X_test, y_test)
                self.generate_shap(best_model, X_train, X_test, algo_name, current_lag, plots_dir)

                metrics.update({
                    'lag': current_lag,
                    'algorithm': algo_name,
                    'time_agg': time_aggregation,
                    'use_macro': self.use_macro,
                    'use_fs': use_fs
                })
                results_list.append(metrics)

        if results_list:
            df_results = pd.DataFrame(results_list)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = os.path.join(self.output_path, f"metrics_ml_{timestamp}.csv")
            df_results.to_csv(csv_path, index=False)
            print(f"\n[SUCCESS] CSV saved in: {csv_path}")
        return pd.DataFrame(results_list)