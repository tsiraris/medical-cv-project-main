import os
import yaml
import joblib
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from src.utils.io import ensure_dir, save_json
from src.utils.mlflow_ready import get_tracking_uri, wait_for_mlflow


import argparse
parser = argparse.ArgumentParser()
# args: --params (YAML with hyperparams), --out (model path)
parser.add_argument("--params", default="params.yaml")
parser.add_argument("--out", default="models/model.joblib")

def build_model(arch: str, model_cfg: dict, seed: int):
    """
    Small factory so you can later switch model.arch in params.yaml
    without changing code.
    """
    arch = (arch or "logistic").lower()
    if arch == "logistic":
        return LogisticRegression(max_iter=int(model_cfg.get("max_iter", 1000)), random_state=seed)
    elif arch in {"rf", "random_forest", "random-forest"}:
        return RandomForestClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 150)),
            max_depth=model_cfg.get("max_depth"),
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported model.arch: {arch}")

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.params, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load processed data
    Xtr = np.load("data/processed/Xtr.npy")
    Xte = np.load("data/processed/Xte.npy")
    ytr = np.load("data/processed/ytr.npy")
    yte = np.load("data/processed/yte.npy")

    # Read MLflow URI flexibly + wait until server is ready
    uri = get_tracking_uri()
    wait_for_mlflow(uri)  # retries until MLflow is up
    mlflow.set_tracking_uri(uri)  # no need to export envs in shell anymore
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "demo-cv"))  # still overridable via env

    # Pull hyperparams from your params.yaml schema
    seed = int(cfg.get("seed", 42))
    model_cfg = cfg.get("model", {}) or {}
    arch = model_cfg.get("arch", "logistic")
    max_iter = int(model_cfg.get("max_iter", 1000))
    train_cfg = cfg.get("train", {}) or {}
    test_size = float(train_cfg.get("test_size", 0.2))
    features = train_cfg.get("features", ["f1", "f2", "f3", "f4"])  # kept for provenance; arrays already precomputed

    with mlflow.start_run() as run:  # Start an MLflow run (goes to MLflow server)
        # Build + fit model
        clf = build_model(arch, model_cfg, seed)
        clf.fit(Xtr, ytr)

        # Evaluate
        ypred = clf.predict(Xte)
        acc = accuracy_score(yte, ypred)
        f1 = f1_score(yte, ypred, average="macro")

        # log params/metrics (kept your originals, added a few helpful ones)
        mlflow.log_params({
            "arch": arch,                          # from cfg["model"].get("arch", ...)
            "max_iter": max_iter,                  # for logistic
            "seed": seed,
            "train.test_size": test_size,
            "train.features": ",".join(map(str, features))
        })
        mlflow.log_metrics({
            "acc": float(acc),
            "val_f1_macro": float(f1)
        })

        # Save model to repo (models/model.joblib) for pipeline continuity
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, outp)

        # Always log the model to the current run
        mlflow.sklearn.log_model(sk_model=clf, artifact_path="model")

        # ✅ Conditionally register the model:
        # Only if we're talking to a real MLflow server (http/https) AND a name is provided.
        reg_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "").strip()
        tracking_uri = mlflow.get_tracking_uri() or ""
        if reg_name and tracking_uri.startswith(("http://", "https://")):
            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/model"
            client = MlflowClient()
            # Create registered model if missing
            if reg_name not in [m.name for m in client.search_registered_models()]:
                client.create_registered_model(reg_name)
            client.create_model_version(name=reg_name, source=model_uri, run_id=run_id)

        # emit DVC metrics
        ensure_dir("metrics")
        save_json({"acc": float(acc), "val_f1_macro": float(f1)}, "metrics/val.json")

        print({"acc": acc, "val_f1_macro": f1, "run_id": run.info.run_id})

    print("Training Done.")