# Trains a simple sklearn model, logs params/metrics to MLflow,
# saves a local artifact for the pipeline (models/model.joblib),
# and (only when using an HTTP MLflow server) attempts model registration.

import os
import yaml
import joblib
from pathlib import Path
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from src.utils.io import ensure_dir, save_json
from src.utils.mlflow_ready import wait_for_mlflow

import argparse

parser = argparse.ArgumentParser()
# args: --params (YAML with hyperparams), --out (model path)
parser.add_argument("--params", default="params.yaml")
parser.add_argument("--out", default="models/model.joblib")


def _is_http_uri(uri: str) -> bool:
    """Returns True if the MLflow tracking URI is an HTTP(s) endpoint (i.e., server-backed)."""
    return uri.startswith("http://") or uri.startswith("https://")


if __name__ == "__main__":
    args = parser.parse_args()

    # --- Load pipeline config (hyperparameters, etc.) ---
    with open(args.params, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # --- Load processed data produced by the 'preprocess' stage ---
    Xtr = np.load("data/processed/Xtr.npy")
    Xte = np.load("data/processed/Xte.npy")
    ytr = np.load("data/processed/ytr.npy")
    yte = np.load("data/processed/yte.npy")

    # --- MLflow setup ---
    # In CI we set MLFLOW_TRACKING_URI=file:./mlruns (no server).
    # Locally you might run the server via docker-compose and use http://127.0.0.1:5500
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    wait_for_mlflow(tracking_uri)  # no-op for file backend, waits only for http(s)

    # Use env experiment if provided; otherwise default
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "demo-cv"))

    # --- Train + Log ---
    params = cfg.get("model", {})
    seed = cfg.get("seed", 42)

    with mlflow.start_run() as run:
        # Train a simple classifier
        clf = LogisticRegression(
            max_iter=params.get("max_iter", 1000),
            random_state=seed
        )
        clf.fit(Xtr, ytr)

        # Predictions + metrics
        y_pred = clf.predict(Xte)
        acc = accuracy_score(yte, y_pred)
        f1 = f1_score(yte, y_pred)

        # Log params/metrics
        mlflow.log_params({
            "arch": cfg["model"].get("arch", "logistic"),
            "max_iter": params.get("max_iter", 1000),
            "seed": seed,
        })
        mlflow.log_metric("acc", float(acc))
        mlflow.log_metric("f1", float(f1))

        # --- Persist the model to the repo for DVC (matches dvc.yaml: outs: models/model.joblib) ---
        outp = Path(args.out)  # e.g., models/model.joblib
        outp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, outp)
        print(f"[train] Saved model to: {outp.resolve()}")

        # --- Log model artifact to MLflow ---
        # Add signature + input_example to silence MLflow warning and improve lineage.
        signature = infer_signature(Xtr, clf.predict(Xtr))
        input_example = Xtr[:2]

        if _is_http_uri(tracking_uri):
            # On server-backed tracking, we can also register a model version
            reg_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "cxr-demo")
            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model",
                registered_model_name=reg_name,
                signature=signature,
                input_example=input_example,
            )
        else:
            # On file backend (CI), registry APIs aren’t available; just log the model
            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )

        # --- Emit metrics file for DVC & CI gate ---
        ensure_dir("metrics")
        save_json({"acc": float(acc), "f1": float(f1)}, "metrics/val.json")

        print({"acc": acc, "f1": f1, "run_id": run.info.run_id})
