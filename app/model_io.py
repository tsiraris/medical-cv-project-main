# app/model_io.py
import os
import tempfile
from pathlib import Path
from typing import Optional

import joblib
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts

MODEL_NAME_ENV = "MLFLOW_REGISTERED_MODEL_NAME"
DEFAULT_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "Production")

# simple in-process cache
_model_cache = {"path": None, "obj": None, "version": None}


def _get_tracking_client() -> MlflowClient:
    """
    Return an MlflowClient bound to the configured tracking URI.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


def _resolve_model_version_by_alias(
    client: MlflowClient, model_name: str, alias: str
) -> Optional[str]:
    """
    Given a model name and alias, return the underlying version number (as string),
    or None if not found.
    """
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
        return mv.version
    except Exception:
        return None


def load_model():
    """
    Load model from MLflow Model Registry and cache it in-process.

    Uses:
    - MLFLOW_TRACKING_URI
    - MLFLOW_REGISTERED_MODEL_NAME (e.g. "cxr-demo")
    - MLFLOW_MODEL_ALIAS (e.g. "Production" / "Candidate")
    """
    global _model_cache

    model_name = os.getenv(MODEL_NAME_ENV)
    if not model_name:
        raise RuntimeError(
            f"{MODEL_NAME_ENV} env var is not set; cannot load model from MLflow."
        )

    alias = os.getenv("MLFLOW_MODEL_ALIAS", DEFAULT_ALIAS)
    client = _get_tracking_client()

    version = _resolve_model_version_by_alias(client, model_name, alias)
    if version is None:
        raise RuntimeError(
            f"Could not find alias '{alias}' for model '{model_name}' "
            "in the MLflow Model Registry."
        )

    # MLflow model URI for that alias
    model_uri = f"models:/{model_name}@{alias}"

    # Download to a temp dir, then look for a joblib file (your training code
    # saves a sklearn model as joblib in the MLflow artifact). :contentReference[oaicite:3]{index=3}
    tmp_dir = tempfile.mkdtemp(prefix="mlflow_model_")
    download_artifacts(artifact_uri=model_uri, dst_path=tmp_dir)

    # Find a .joblib file in that directory
    joblib_path = None
    for p in Path(tmp_dir).rglob("*.joblib"):
        joblib_path = p
        break

    if joblib_path is None:
        raise RuntimeError(
            f"No .joblib file found in MLflow model artifacts for {model_uri}"
        )

    model_obj = joblib.load(joblib_path)

    _model_cache = {
        "path": str(joblib_path),
        "obj": model_obj,
        "version": f"{model_name}@{alias}",
    }
    return model_obj


def predict_batch(X):
    """Predict with cached model; loads if empty."""
    if _model_cache["obj"] is None:
        load_model()
    return _model_cache["obj"].predict(X)


def model_short_sha():
    """Return a short identifier for the loaded model (name@alias or 'unloaded')."""
    return _model_cache.get("version") or "unloaded"
