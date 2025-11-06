import argparse
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
from src.utils.io import ensure_dir

parser = argparse.ArgumentParser()
# Loads models/model.joblib and data/processed
parser.add_argument("--model", default="models/model.joblib")
parser.add_argument("--data", default="data/processed")
parser.add_argument("--out", default="artifacts")

if __name__ == "__main__":
    args = parser.parse_args()
    Xte = np.load(Path(args.data)/"Xte.npy")
    yte = np.load(Path(args.data)/"yte.npy")
    model = joblib.load(args.model)

    out = ensure_dir(args.out)

    # We don’t have probabilistic outputs for all models; use decision function if available
    # Try predict_proba -> decision_function -> fallback to predicted labels
    yscore = None
    try:
        proba = model.predict_proba(Xte)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            yscore = proba[:, 1]  # positive class prob for binary
    except Exception:
        try:
            yscore = model.decision_function(Xte)
        except Exception:
            yscore = None

    # Computes ROC curve, confusion matrix for visualization
    # Confusion matrix (always available)
    yhat = model.predict(Xte)
    fig = ConfusionMatrixDisplay.from_predictions(yte, yhat)
    plt.title("Confusion")
    plt.tight_layout()
    plt.savefig(Path(out)/"confusion.png", bbox_inches="tight", dpi=180)
    plt.close()

    # ROC: only for binary + score-like outputs
    roc_path = Path(out)/"roc.png"
    if yscore is not None and len(np.unique(yte)) == 2:
        try:
            RocCurveDisplay.from_predictions(yte, yscore)
            plt.title("ROC")
            plt.tight_layout()
            plt.savefig(roc_path, bbox_inches="tight", dpi=180)
            plt.close()
        except Exception:
            # Fallback simple ROC using manual fpr/tpr
            fpr, tpr, thr = roc_curve(yte, yscore)
            plt.figure()
            plt.plot(fpr, tpr)
            plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
            plt.tight_layout()
            plt.savefig(roc_path, bbox_inches="tight", dpi=180)
            plt.close()
    else:
        # If we can't compute ROC (multiclass or no scores), write a placeholder
        plt.figure()
        plt.text(0.5, 0.5, "ROC unavailable (no scores or multiclass)", ha="center", va="center")
        plt.axis("off")
        plt.savefig(roc_path, bbox_inches="tight", dpi=180)
        plt.close()

    # Writes plots to artifacts/ (tracked by DVC as outputs)
    print(f"Evaluation plots saved to: {out}")
    print("Evaluation Done.")

    # For debugging
    # plt.show()