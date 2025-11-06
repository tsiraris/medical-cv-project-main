from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np

def basic_metrics(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = 0.0
    return {"auc": float(auc), "sens": float(sens), "spec": float(spec)}
