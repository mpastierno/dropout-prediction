import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from utils.soa_helpers import apply_macro_areas

def check_for_mps():
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because PyTorch was not built with MPS enabled.")
        else:
            print("MPS not available because macOS version/device does not support it.")
        return torch.device("cpu")
    return torch.device("mps")

# NOTE time_aggregation removed for LSTM, because the purpose of using this network is to
# catch timeseries patterns
def prepare_3d_global_data(file_path, lag, use_macro=True):
    """
    Function that process the train set into a 3d torch tensor, ready to train the model LSTM
    """
    df = pd.read_csv(file_path)
    df_lag = df[df['day'] <= lag].copy()

    assert df_lag['day'].astype(int).max() == lag, f"[CRITICAL] Days greather than {lag} found"

    if use_macro:
        df_lag = apply_macro_areas(df_lag)
        
    metadata_cols = ['global_id', 'day', 'dropout']
    features = [c for c in df_lag.columns if c not in metadata_cols]

    df_lag['dropout'] = (df_lag['dropout'] >= 0.5).astype(int)

    df_lag = df_lag.sort_values(by=['global_id', 'day']).reset_index(drop=True)

    X_numpy = df_lag[features].values
    X_3d = X_numpy.reshape(-1, lag, len(features))

    num_students = df_lag['global_id'].nunique()
    num_features = len(features)

    expected_shape = (num_students, lag, num_features)
    assert X_3d.shape == expected_shape, f"[CRITICAL] Wrong shape, expected: \n{expected_shape}, found: {X_3d.shape}"

    X_tensor = torch.tensor(X_3d, dtype=torch.float32)
    y_series = df_lag.groupby('global_id')['dropout'].first()
    y_tensor = torch.tensor(y_series.values, dtype=torch.float32)

    assert X_tensor.shape[0] == y_tensor.shape[0], "[CRITICAL] Disalligned students between X and y"
    if use_macro == True:
        assert len(features) == 10, f"[CRITICAL] Data leak, features {len(features)}/10, missing {len(features) - 10} features"
    else:
        assert len(features) == 97, f"[CRITICAL] Data leak, features {len(features)}/97, missing {len(features) - 97} features"

    return X_tensor, y_tensor, features


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
    
def evaluate_dl_models(X_train, y_train, X_test, y_test, input_dim, hidden_dim=64, lr=0.001, batch_size=64, epochs=30):
    device = check_for_mps()
    print(f"[INFO] Device in use: {device}")

    num_persist = (y_train == 0).sum().item()
    num_dropout = (y_train == 1).sum().item()

    # NOTE to penalize errors on class 0 more than 1 (1/4 on dropout), passed in criterion
    pos_weight_val = num_persist / num_dropout
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = LSTMClassifier(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) # TODO test 1e-3

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_X) # now LOGITS
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
            
    model.eval()
    with torch.no_grad():
        X_test_dev = X_test.to(device)
        logits = model(X_test_dev)
        y_prob_tensor = torch.sigmoid(logits)
        
        y_prob = y_prob_tensor.cpu().numpy()
        y_pred = (y_prob >= 0.5).astype(int)
        y_true = y_test.numpy()

    metrics = {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'roc_auc': round(roc_auc_score(y_true, y_prob), 4),
        'pr_auc': round(average_precision_score(y_true, y_prob), 4)
    }
    
    return metrics, model

def generate_shap_summary(model, X_train_tensor, X_test_tensor, feature_names, lag, model_name, output_dir):
    device = check_for_mps()
    model.eval()

    def predict_fn(x_2d_numpy):
        x_3d = x_2d_numpy.reshape(-1, lag, len(feature_names))
        x_tensor = torch.tensor(x_3d, dtype=torch.float32).to(device)
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

    plot_path = os.path.join(output_dir, f"shap_{model_name}_lag{lag}_{datetime.now()}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

