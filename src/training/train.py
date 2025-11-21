# Trains a simple sklearn model, logs params/metrics to MLflow,
# saves a local artifact for the pipeline (models/model.joblib),
# and (only when using an HTTP MLflow server) attempts model registration.
# Also writes metrics/active_run_id so eval.py can attach plots to the same run.

import os
import yaml
import joblib
from pathlib import Path
import numpy as np
import argparse
import urllib.parse

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Optional ONNX export (for KServe)
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    HAVE_SKL2ONNX = True
except Exception:
    HAVE_SKL2ONNX = False

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.exceptions import MlflowException

from src.utils.io import ensure_dir, save_json
from src.utils.mlflow_ready import wait_for_mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--params", default="params.yaml")
parser.add_argument("--out", default="models/model.joblib")


def _is_http_uri(uri: str) -> bool:
    try:
        scheme = urllib.parse.urlparse(uri).scheme.lower()
        return scheme in {"http", "https"}
    except Exception:
        return uri.startswith("http://") or uri.startswith("https://")


def log_model_compat(sk_model, artifact_path: str, signature=None, input_example=None, registered_model_name: str | None = None) -> str:
    """
    Try modern mlflow.sklearn.log_model(). If the server doesn't support the
    'logged-models' API (404), fall back to uploading models/model.joblib as a plain artifact.
    """
    import mlflow.sklearn as mls
    try:
        if registered_model_name:
            mls.log_model(
                sk_model=sk_model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example,
            )
        else:
            mls.log_model(
                sk_model=sk_model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=input_example,
            )
        return "mlflow.sklearn.log_model"
    except MlflowException as e:
        msg = str(e).lower()
        if "logged-models" in msg or "404" in msg or "not found" in msg:
            local_model_path = Path("models") / "model.joblib"
            if not local_model_path.exists():
                local_model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(sk_model, local_model_path)
            mlflow.log_artifact(str(local_model_path), artifact_path=artifact_path)
            return "artifact-fallback"
        raise


if __name__ == "__main__":
    args = parser.parse_args()

    # --- Load pipeline config ---
    with open(args.params, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # --- Load processed data produced by preprocess ---
    Xtr = np.load("data/processed/Xtr.npy")
    Xte = np.load("data/processed/Xte.npy")
    ytr = np.load("data/processed/ytr.npy")
    yte = np.load("data/processed/yte.npy")

    # --- MLflow setup ---
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    wait_for_mlflow(tracking_uri)  # robust for http(s), no-op for file

    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "demo-cv"))

    # --- Train + Log ---
    params = cfg.get("model", {})
    seed = cfg.get("seed", 42)

    with mlflow.start_run() as run:
        # expose run_id to eval via small handoff file
        ensure_dir("metrics")
        Path("metrics/active_run_id").write_text(run.info.run_id, encoding="utf-8")
        print(f"[train] Run ID: {run.info.run_id}")

        # Train
        clf = LogisticRegression(max_iter=params.get("max_iter", 1000), random_state=seed)
        clf.fit(Xtr, ytr)

        # Metrics
        y_pred = clf.predict(Xte)
        acc = accuracy_score(yte, y_pred)
        f1 = f1_score(yte, y_pred)

        mlflow.log_params({
            "arch": cfg.get("model", {}).get("arch", "logistic"),
            "max_iter": params.get("max_iter", 1000),
            "seed": seed,
        })
        mlflow.log_metric("acc", float(acc))
        mlflow.log_metric("f1", float(f1))

        # Persist for DVC
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, outp)
        print(f"[train] Saved model to: {outp.resolve()}")

        # Log model to MLflow (with compat)
        signature = infer_signature(Xtr, clf.predict(Xtr))
        input_example = Xtr[:2]

        reg_name = None
        if _is_http_uri(tracking_uri):
            reg_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME")  # optional

        how = log_model_compat(
            sk_model=clf,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=reg_name,
        )
        print(f"[train] model logged via: {how}")

        # ------------------------------------------------------------------
        # Optional: export ONNX model for KServe and tag storage URI
        # ------------------------------------------------------------------
        if HAVE_SKL2ONNX:
            try:
                n_features = Xtr.shape[1]
                initial_types = [("input", FloatTensorType([None, n_features]))]

                onnx_model = convert_sklearn(clf, initial_types=initial_types)

                onnx_dir = Path("tmp_artifacts") / "onnx"
                onnx_dir.mkdir(parents=True, exist_ok=True)
                onnx_path = onnx_dir / "model.onnx"
                with open(onnx_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())

                # Log ONNX file under "onnx" sub-artifact
                mlflow.log_artifact(str(onnx_path), artifact_path="onnx")

                # This is the URI KServe can use as storageUri
                onnx_storage_uri = f"{run.info.artifact_uri}/onnx/model.onnx"
                mlflow.set_tag("deploy.storage_uri", onnx_storage_uri)

                print(
                    f"[train] Exported ONNX model to {onnx_path} and "
                    f"logged artifact. storageUri={onnx_storage_uri}"
                )
            except Exception as e:
                print(f"[train] Warning: could not export/log ONNX model: {e}")
        else:
            print("[train] skl2onnx not available; skipping ONNX export")

        # Emit metrics file for DVC & CI gate
        ensure_dir("metrics")
        save_json({"acc": float(acc), "f1": float(f1)}, "metrics/val.json")

        print({"acc": acc, "f1": f1, "run_id": run.info.run_id})
        save_json({"run_id": run.info.run_id}, "metrics/run_info.json")
