import argparse
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
from src.utils.io import ensure_dir

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/model.joblib")
parser.add_argument("--data", default="data/processed")
parser.add_argument("--out", default="artifacts")

if __name__ == "__main__":
    args = parser.parse_args()
    Xte = np.load(Path(args.data)/"Xte.npy")
    yte = np.load(Path(args.data)/"yte.npy")
    model = joblib.load(args.model)

    out = ensure_dir(args.out)

    # score/proba if available
    yscore = None
    try:
        proba = model.predict_proba(Xte)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            yscore = proba[:, 1]
    except Exception:
        try:
            yscore = model.decision_function(Xte)
        except Exception:
            yscore = None

    # Confusion matrix
    yhat = model.predict(Xte)
    ConfusionMatrixDisplay.from_predictions(yte, yhat)
    plt.title("Confusion")
    plt.tight_layout()
    (Path(out)/"confusion.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(out)/"confusion.png", bbox_inches="tight", dpi=180)
    plt.close()

    # ROC (binary only)
    roc_path = Path(out)/"roc.png"
    if yscore is not None and len(np.unique(yte)) == 2:
        try:
            RocCurveDisplay.from_predictions(yte, yscore)
            plt.title("ROC")
            plt.tight_layout()
            plt.savefig(roc_path, bbox_inches="tight", dpi=180)
            plt.close()
        except Exception:
            fpr, tpr, thr = roc_curve(yte, yscore)
            plt.figure()
            plt.plot(fpr, tpr)
            plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
            plt.tight_layout()
            plt.savefig(roc_path, bbox_inches="tight", dpi=180)
            plt.close()
    else:
        plt.figure()
        plt.text(0.5, 0.5, "ROC unavailable (no scores or multiclass)", ha="center", va="center")
        plt.axis("off")
        plt.savefig(roc_path, bbox_inches="tight", dpi=180)
        plt.close()

    print(f"Evaluation plots saved to: {out}")
    print("Evaluation Done.")

# --- Attach plots to the SAME MLflow run as train (handoff via metrics/active_run_id)
try:
    import os, mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    run_id_file = Path("metrics") / "active_run_id"
    if run_id_file.exists():
        run_id = run_id_file.read_text(encoding="utf-8").strip()
        # ✅ use local_dir (MLflow 2.x), not local_path
        MlflowClient().log_artifacts(run_id=run_id, local_dir=str(out), artifact_path="plots")
        print(f"[eval] Logged plots to MLflow run {run_id} under artifact path 'plots/'")
    else:
        print("[eval] Skipping MLflow plots upload (no metrics/active_run_id found).")
except Exception as e:
    print(f"[eval] Non-fatal: failed to upload plots to MLflow: {e}")
