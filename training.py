"""training.py
Train a RandomForestClassifier (suitable for churn classification).
Produces a saved model file under models/best_model.pkl
"""
import joblib, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_and_save(X_train, y_train, model_path='models/best_model.pkl', do_grid=False):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if do_grid:
        param_grid = {
            'n_estimators': [50,100],
            'max_depth': [None, 10, 20]
        }
        clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, scoring='f1')
        clf.fit(X_train, y_train)
        best = clf.best_estimator_
    else:
        best = RandomForestClassifier(n_estimators=100, random_state=42)
        best.fit(X_train, y_train)
    joblib.dump(best, model_path)
    print(f'Model saved to: {model_path}')
    return best
