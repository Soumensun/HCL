"""evaluation.py
Evaluate classification model and produce metrics and plots.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_model(model, X_test, y_test, outdir='outputs'):
    os.makedirs(outdir, exist_ok=True)
    preds = model.predict(X_test)
    probs = None
    try:
        probs = model.predict_proba(X_test)[:,1]
    except Exception:
        pass

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs) if probs is not None else None

    print('Accuracy:', acc)
    print('Precision:', prec)
    print('Recall:', rec)
    print('F1 score:', f1)
    if auc is not None:
        print('ROC AUC:', auc)

    # confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title('Confusion matrix')
    plt.colorbar()
    ticks = [0,1]
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'confusion_matrix.png'))
    plt.close()

    metrics = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': auc}
    return metrics
