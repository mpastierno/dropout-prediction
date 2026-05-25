import os
import pandas as pd
from datetime import datetime
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

MACRO_MAPPING = {
    'content_view': ['view', 'view all', 'launch', 'launched', 'open', 'pre-view', 'launch recent', 'recent', 'downloaded'],
    'forum_interaction': ['view forum', 'view forums', 'view discussion', 'add discussion', 'delete discussion', 'add post', 'update post', 'delete post', 'subscribe', 'subscribeall', 'unsubscribe', 'unsubscribeall', 'view subscriber'],
    'assessment': ['attempt', 'continue attempt', 'close attempt', 'review', 'reviewed', 'view summary', 'graded', 'submitted', 'started', 'reset', 'failed', 'abandoned'],
    'content_creation': ['write', 'upload', 'uploaded', 'add', 'added', 'update', 'updated', 'edit', 'created', 'delete', 'deleted', 'editvideos', 'add page', 'add entry', 'edit entry'],
    'social_interaction': ['add contact', 'remove contact', 'block contact', 'unblock contact', 'talk', 'called', 'sent', 'comment', 'mail blocked'],
    'scorm_tracking': ['trk: l2lscorm at: 1', 'trk: l2lscorm at: 2', 'trk: l2lscorm at: 3', 'trk: l2lscorm at: 4', 'trk: l2lscorm at: 9', 'stop tracking', 'start tracking'],
    'system_profile': ['enrol', 'history', 'assign', 'assigned', 'unassigned', 'accepted', 'restored', 'mark read', 'flag', 'map', 'searched', 'search', 'diff', 'disabled', 'removed'],
    'reports': ['user report', 'report', 'report outline', 'report log'],
    'backend_api': ['automatically create user token', 'sending requested user token', 'core_webservice_get_site_info', 'ltol_get_categories', 'core_enrol_get_users_courses', 'ltol_ws_ltol_get_courses', 'core_course_get_contents', 'ltol_ws_get_asset', 'ltol_ws_put_track', 'ltol_get_users_courses', 'error']
}

def apply_macro_areas(df):
    metadata_cols = ['global_id', 'day', 'dropout']
    df_macro = df[metadata_cols].copy()
    
    mapping = MACRO_MAPPING.copy()
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

def prepare_global_data(file_path, lag, use_macro=True, time_aggregation='flatten'):
    df = pd.read_csv(file_path)
    df_lag = df[df['day'] <= lag].copy()

    if use_macro:
        df_lag = apply_macro_areas(df_lag)
        
    metadata_cols = ['global_id', 'day', 'dropout']
    features = [c for c in df_lag.columns if c not in metadata_cols]
    
    if time_aggregation == 'flatten':
        X = df_lag.pivot_table(index=['global_id'], columns='day', values=features, fill_value=0)
        X.columns = [f"{act}_day_{d}" for act, d in X.columns]
    elif time_aggregation == 'sum':
        X = df_lag.groupby('global_id')[features].sum()
        X.columns = [f"{col}_sum" for col in X.columns]
    
    y_info = df_lag.groupby('global_id').agg({'dropout': 'first'})
    y = y_info['dropout'].astype(int)
    
    y_stratify = y.copy() 
    
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    y_stratify = y_stratify.reset_index(drop=True)
    
    return X, y, y_stratify

def evaluate_model_global(model_config, X_train, y_train, y_strat_train, X_test, y_test):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model_config['estimator'])
    ])
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid = GridSearchCV(
        pipeline, 
        param_grid=model_config['params'], 
        cv=list(cv_strategy.split(X_train, y_strat_train)),
        scoring='average_precision',
        n_jobs=-1
    )
    
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'best_params': grid.best_params_,
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
        'f1': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'roc_auc': round(roc_auc_score(y_test, y_prob), 4),
        'pr_auc': round(average_precision_score(y_test, y_prob), 4)
    }
    
    return metrics, best_model

def generate_shap_summary(model, X_train, X_test, model_name, lag, output_dir):
    def predict_fn(x):
        x_df = pd.DataFrame(x, columns=X_train.columns)
        return model.predict_proba(x_df)[:, 1]

    background = shap.sample(X_train, 100, random_state=42)
    explainer = shap.KernelExplainer(predict_fn, background)
    
    sample_size = min(200, len(X_test))
    X_test_sample = X_test.sample(n=sample_size, random_state=42)
    
    shap_values = explainer.shap_values(X_test_sample)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"shap_{model_name}_lag{lag}_{datetime.now()}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()