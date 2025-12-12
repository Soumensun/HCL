# evaluation.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

def evaluate_and_save(model, X_test, y_test, outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)

    try:
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
    except Exception:
        preds = model.predict(X_test)
        probs = None

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0))
    }
    if probs is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, probs))
        except Exception:
            metrics["roc_auc"] = None

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title("Confusion matrix")
    plt.colorbar()
    ticks = [0, 1]
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confusion_matrix.png"))
    plt.close()

    if probs is not None:
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f"AUC = {metrics.get('roc_auc'):.3f}")
        plt.plot([0,1],[0,1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "roc_curve.png"))
        plt.close()

    return metrics
