import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    average_precision_score
)

def prepare_flattened_data(file_path, lag):
    """
    Carica i dati esplosi, prende solo i primi 'lag' giorni, e appiattisce
    tutte le feature temporali su un'unica riga per studente.
    """
    df = pd.read_csv(file_path)
    
    df_lag = df[df['day'] <= lag].copy()
    
    metadata_cols = ['student_id', 'course_id', 'day', 'dropout']
    action_cols = [col for col in df_lag.columns if col not in metadata_cols]
    
    # Binarizzazione rigorosa (soglia 0.5)
    df_lag['dropout'] = (df_lag['dropout'] >= 0.5).astype(int)
    
    # Estrazione target (costante per studente)
    if 'course_id' in df_lag.columns:
        group_keys = ['course_id', 'student_id']
    else:
        group_keys = ['student_id']
        
    y_series = df_lag.groupby(group_keys)['dropout'].first()
    
    # Flattening tramite pivot_table
    df_pivot = df_lag.pivot_table(
        index=group_keys, 
        columns='day', 
        values=action_cols, 
        fill_value=0
    )
    
    # Rinomina colonne (es: 'view_day_1')
    df_pivot.columns = [f"{action}_day_{day}" for action, day in df_pivot.columns]
    
    X = df_pivot.values
    y = y_series.values
    
    return X, y, df_pivot.columns.tolist()

def evaluate_model(model_name, model, X, y, n_splits=10):
    """
    Esegue cross-validation stratificata e calcola le metriche medie su test set.
    Mantiene l'aderenza alla baseline senza feature scaling.
    """
    if np.count_nonzero(y == 0) < n_splits or np.count_nonzero(y == 1) < n_splits:
        print(f"[WARNING] Classi sbilanciate (0: {np.count_nonzero(y==0)}, 1: {np.count_nonzero(y==1)}). Skipped.")
        return None

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=42)
    metrics = {
        'accuracy': [], 'precision': [], 'recall': [], 
        'f1': [], 'roc_auc': [], 'pr_auc': []
    }
    
    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
            
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        metrics['roc_auc'].append(roc_auc_score(y_test, y_prob))
        metrics['pr_auc'].append(average_precision_score(y_test, y_prob))

    # Media delle metriche
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return avg_metrics