import torch
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from src.config_unitelma import MACRO_MAPPING

class BasePipeline:
    def __init__(self, train_path, test_path, output_path, use_macro: bool):
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.output_path = output_path
        self.macro_mapping = MACRO_MAPPING
        self.use_macro = use_macro

        os.makedirs(self.output_path, exist_ok=True)
        
        self.device = self._get_torch_device()
        print(f"[INFO] Device: {self.device}")

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