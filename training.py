# training.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import joblib

from data_loading import load_data
from preprocessing import Preprocessor
from evaluation import evaluate_model

# Step 1: Load data
df = load_data("telecom_churn.csv")
# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Step 2: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
print(f"Churn rate in training set: {y_train.mean():.2f}, in test set: {y_test.mean():.2f}")

# Step 3: Handle class imbalance on the training set using SMOTE
print("Before SMOTE: Class distribution in training:", pd.Series(y_train).value_counts().to_dict())
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("After SMOTE: Class distribution in training:", pd.Series(y_train_res).value_counts().to_dict())

# Step 4: Preprocess the data (fit on training, transform train and test)
preprocessor = Preprocessor()
# Fit on the resampled training data and transform it
X_train_proc = preprocessor.fit_transform(X_train_res)
# Transform the test set using the same preprocessing (note: do NOT fit on test!)
# We need to ensure test has same features as train (especially if any dummies)
X_test_proc = X_test.copy()
# If any categorical features exist, add missing dummy columns as in Preprocessor.transform
# (We can reuse Preprocessor.transform here for consistency)
X_test_proc = preprocessor.transform(X_test_proc)

# Align test set columns to training set columns (if any difference due to encoding)
# Ensure X_test_proc has the same columns as X_train_proc
for col in X_train_proc.columns:
    if col not in X_test_proc.columns:
        X_test_proc[col] = 0
X_test_proc = X_test_proc[X_train_proc.columns]

# Step 5: Define logistic regression model and hyperparameter grid for tuning
model = LogisticRegression(max_iter=1000)  # base model
param_grid = [
    {'solver': ['lbfgs'], 'penalty': ['l2'], 'C': [0.01, 0.1, 1.0, 10.0]},
    {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1.0, 10.0]}
]
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_proc, y_train_res)
print(f"Best parameters from GridSearch: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# Evaluate on the test set
print("Evaluation on test set:")
evaluate_model(best_model, X_test_proc, y_test)

# Step 6: Save the trained model and preprocessor (scaler) for deployment
joblib.dump(best_model, "model.pkl")
joblib.dump(preprocessor.scaler, "scaler.pkl")
print("Model and scaler saved for deployment.")
