# training.py
"""
Train ensemble models (Logistic, RandomForest, XGBoost) and a Stacking ensemble.
Saves the final best pipeline (preprocessor + smote + model) to models/best_pipeline.joblib
Also saves a JSON with feature lists used for the UI.
"""
import os
import joblib
import json
import numpy as np
from pprint import pprint

from data_loading import load_data
from preprocessing import build_preprocessing_pipeline, infer_feature_lists_from_df

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

OUT_DIR = "models"
MODEL_PATH = os.path.join(OUT_DIR, "best_pipeline.joblib")
FEATURE_META = os.path.join(OUT_DIR, "features.json")

def train_pipeline(csv_path: str, target_col: str = "Churn", test_size=0.2, random_state=42):
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_data(csv_path)
    numeric_feats, categorical_feats = infer_feature_lists_from_df(df, target_col)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessing_pipeline(numeric_feats, categorical_feats)

    # Define candidate models
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=random_state)
    lr = LogisticRegression(max_iter=1000, solver="liblinear")

    # Train each base model inside an imbalanced pipeline (preprocess -> SMOTE -> classifier)
    base_models = {"LogisticRegression": lr, "RandomForest": rf, "XGBoost": xgb}
    base_results = {}
    fitted_models = {}

    for name, estimator in base_models.items():
        pipe = ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=random_state)),
            ("model", estimator)
        ])
        print(f"Training {name} ...")
        pipe.fit(X_train, y_train)
        # predict probabilities if available, else predict labels
        try:
            preds = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            preds = pipe.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        base_results[name] = rmse
        fitted_models[name] = pipe

    # Build stacking using fitted base estimators (use un-fitted clones in estimator list)
    print("Training Stacking ensemble ...")
    estimators_for_stack = [
        ("lr", LogisticRegression(max_iter=1000, solver="liblinear")),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=random_state)),
        ("xgb", XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=random_state))
    ]
    # For stacking we create a pipeline: preprocessor -> smote -> stacking
    stack = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=random_state)),
        ("stacking", StackingClassifier(estimators=estimators_for_stack, final_estimator=LogisticRegression(), cv=3, n_jobs=-1))
    ])
    stack.fit(X_train, y_train)
    try:
        stack_preds = stack.predict_proba(X_test)[:, 1]
    except Exception:
        stack_preds = stack.predict(X_test)
    stack_rmse = float(np.sqrt(mean_squared_error(y_test, stack_preds)))
    base_results["Stacking"] = stack_rmse
    fitted_models["Stacking"] = stack

    print("\nModel RMSEs (lower is better):")
    pprint(base_results)

    # Choose best (lowest RMSE)
    best_name = min(base_results, key=base_results.get)
    best_pipeline = fitted_models[best_name]
    print(f"\nBest model: {best_name}  RMSE={base_results[best_name]:.4f}")

    # Save best pipeline and feature metadata
    joblib.dump(best_pipeline, MODEL_PATH)
    meta = {"numeric_features": numeric_feats, "categorical_features": categorical_feats, "target": target_col, "best_model": best_name}
    with open(FEATURE_META, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved pipeline to: {MODEL_PATH}")
    print(f"Saved features metadata to: {FEATURE_META}")

    return MODEL_PATH, FEATURE_META, base_results

if __name__ == "__main__":
    # Example: adjust path if your CSV lives somewhere else
    csv_path = "telecom_churn.csv"
    train_pipeline(csv_path)
