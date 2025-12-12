# evaluation.py

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    """Compute and print evaluation metrics for the model on the test set."""
    # Predict classes and probabilities
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]  # probability of class 1 (churn)
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Print the results
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-score:  {f1:.3f}")
    if roc_auc is not None:
        print(f"ROC-AUC:   {roc_auc:.3f}")
    else:
        print("ROC-AUC:   N/A (model does not support probability predictions)")
    # Optionally, print detailed classification report
    print("\nDetailed classification report:")
    print(classification_report(y_test, y_pred, digits=3))
