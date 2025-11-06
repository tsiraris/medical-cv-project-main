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

from src.utils.io import ensure_dir, save_json
from src.utils.mlflow_ready import wait_for_mlflow

import argparse

parser = argparse.ArgumentParser()
# args: --params (YAML with hyperparams), --out (model path)
parser.add_argument("--params", default="params.yaml")
parser.add_argument("--out", default="models/model.joblib")


def _is_http_uri(uri: str) -> bool:
    return uri.startswith("http://") or uri.startswith("https://")


if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.params, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load processed data (written by the preprocess stage)
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

    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "demo-cv"))

    params = cfg.get("model", {})
    with mlflow.start_run() as run:  # Start an MLflow run
        clf = LogisticRegression(max_iter=params.get("max_iter", 1000), random_state=cfg.get("seed", 42))
        clf.fit(Xtr, ytr)
        y_pred = clf.predict(Xte)

        acc = accuracy_score(yte, y_pred)
        f1 = f1_score(yte, y_pred)

        # log params/metrics
        mlflow.log_params({
            "arch": cfg["model"].get("arch", "logistic"),
            "max_iter": params.get("max_iter", 1000),
            "seed": cfg.get("seed", 42),
        })
        mlflow.log_metric("acc", float(acc))
        mlflow.log_metric("f1", float(f1))

        # Save model to repo (models/model.joblib) for pipeline continuity
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, outp)

        # Log model artifact to MLflow.
        # If we're on an HTTP server, you can also register;
        # with 'file:' backend, registry APIs are not available.
        if _is_http_uri(tracking_uri):
            reg_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "cxr-demo")
            mlflow.sklearn.log_model(sk_model=clf, artifact_path="model", registered_model_name=reg_name)
        else:
            mlflow.sklearn.log_model(sk_model=clf, artifact_path="model")

        # Emit a simple val metrics json for DVC (tracked metric to diff across runs)
        ensure_dir("metrics")
        save_json({"acc": float(acc), "f1": float(f1)}, "metrics/val.json")

        print({"acc": acc, "run_id": run.info.run_id})
