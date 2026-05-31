from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier

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

CLASSIFIERS = {
    'LogisticRegression': {
        'estimator': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'params': {
            'clf__penalty': ['l2', 'l1'],
            'clf__C': [0.01, 0.1, 1.0, 10.0], 
            'clf__solver': ['liblinear']}
    },
    'RandomForest': {
        'estimator': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        'params': {
            'scaler': ['passthrough'],
            'clf__n_estimators': [200, 500], 
            'clf__max_depth': [10, 15, None], 
            'clf__min_samples_leaf': [1, 2],
            'clf__max_features': ['sqrt', 'log2']}
    }
}