import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, average_precision_score
)


def prepare_sequential_data(file_path, lag=7):
    """
    Carica i dati esplosi e crea un tensore 3D: (Num_Studenti, LAG, Num_Feature).
    Questo è il formato nativo richiesto da RNN/LSTM/GRU.
    """
    df = pd.read_csv(file_path)
    df_lag = df[df['day'] <= lag].copy()
    
    metadata_cols = ['student_id', 'course_id', 'day', 'dropout']
    action_cols = [col for col in df_lag.columns if col not in metadata_cols]
    
    # Binarizzazione (soglia 0.5)
    df_lag['dropout'] = (df_lag['dropout'] >= 0.5).astype(int)
    
    # Raggruppamento per identificare gli studenti
    if 'course_id' in df_lag.columns:
        group_keys = ['course_id', 'student_id']
    else:
        group_keys = ['student_id']
        
    y_series = df_lag.groupby(group_keys)['dropout'].first()
    
    num_students = len(y_series)
    num_days = lag
    num_features = len(action_cols)
    
    # Inizializziamo il tensore 3D con zeri
    X_3d = np.zeros((num_students, num_days, num_features), dtype=np.float32)
    
    # Mappature veloci
    student_mapping = {k: i for i, k in enumerate(y_series.index)}
    action_mapping = {a: i for i, a in enumerate(action_cols)}
    
    # Popoliamo il tensore
    for _, row in df_lag.iterrows():
        k = (row['course_id'], row['student_id']) if 'course_id' in df_lag.columns else row['student_id']
        s_idx = student_mapping[k]
        d_idx = int(row['day']) - 1  # I giorni partono da 1, gli indici da 0
        
        for act in action_cols:
            a_idx = action_mapping[act]
            X_3d[s_idx, d_idx, a_idx] = row[act]
            
    return X_3d, y_series.values.astype(np.float32), action_cols


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        # batch_first=True -> input e output aspettano formato (batch, seq, feature)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # out: (batch, seq, hidden_dim), (h_n, c_n): stati finali
        out, (h_n, c_n) = self.lstm(x)
        
        # Prendiamo l'ultimo hidden state: h_n ha shape (num_layers, batch, hidden_dim)
        last_hidden = h_n[-1] 
        
        # Logits e probabilità
        logits = self.fc(last_hidden)
        probs = torch.sigmoid(logits).squeeze(-1) # shape: (batch,)
        return probs


def evaluate_dl_model(X, y, epochs=50, batch_size=16, lr=1e-3, n_splits=10, hidden_dim=100):
    """
    Esegue la K-Fold cross-validation stratificata su modelli PyTorch.
    Addestra il modello usando iperparametri standard del paper.
    """
    if np.count_nonzero(y == 0) < n_splits or np.count_nonzero(y == 1) < n_splits:
        print(f"[WARNING] Classi sbilanciate (0: {np.count_nonzero(y==0)}, 1: {np.count_nonzero(y==1)}). Skipped.")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=42)
    
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': [], 'pr_auc': []}
    
    input_dim = X.shape[2]  # numero di feature

    for fold, (train_idx, test_idx) in enumerate(sss.split(X, y)):
        # Preparazione Dati fold corrente
        X_train, y_train = torch.tensor(X[train_idx]), torch.tensor(y[train_idx])
        X_test, y_test = torch.tensor(X[test_idx]), torch.tensor(y[test_idx])
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Inizializzazione Modello e Ottimizzatore da zero ad ogni fold
        model = LSTMClassifier(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        
        # Training Loop
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                probs = model(batch_X)
                loss = criterion(probs, batch_y)
                loss.backward()
                optimizer.step()
                
        # Eval Loop
        model.eval()
        with torch.no_grad():
            X_test = X_test.to(device)
            y_prob_tensor = model(X_test)
            y_prob = y_prob_tensor.cpu().numpy()
            y_pred = (y_prob >= 0.5).astype(int)
            y_true = y_test.numpy()
            
        metrics['accuracy'].append(accuracy_score(y_true, y_pred))
        metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_true, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_true, y_pred, zero_division=0))
        metrics['roc_auc'].append(roc_auc_score(y_true, y_prob))
        metrics['pr_auc'].append(average_precision_score(y_true, y_prob))

    return {k: np.mean(v) for k, v in metrics.items()}