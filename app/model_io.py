# app/model_io.py
import os
import tempfile
from pathlib import Path
import joblib

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts

MODEL_NAME_ENV = "MLFLOW_REGISTERED_MODEL_NAME"
DEFAULT_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "Production")

# simple in-process cache
_model_cache = {"path": None, "obj": None, "version": None}


def _tracking_setup():
    uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(uri)


def _download_by_alias(model_name: str, alias: str) -> Path:
    """
    Try to resolve <model_name>@<alias> to a model version, download model/model.joblib.
    Fallback: latest numeric version if alias isn't set.
    """
    _tracking_setup()
    client = MlflowClient()

    mv = None
    # 1) Preferred: alias
    try:
        mv = client.get_model_version_by_alias(name=model_name, alias=alias)
    except Exception:
        mv = None

    # 2) Fallback: newest by version number
    if mv is None:
        all_vs = client.search_model_versions(f"name='{model_name}'")
        if not all_vs:
            raise RuntimeError(f"No versions of '{model_name}' exist in the registry")
        mv = sorted(all_vs, key=lambda v: int(v.version))[-1]

    tmpdir = Path(tempfile.mkdtemp(prefix=f"{model_name}-v{mv.version}-"))
    # Try exact file first
    try:
        p = download_artifacts(
            run_id=mv.run_id,
            artifact_path="model/model.joblib",
            dst_path=str(tmpdir),
        )
        return Path(p)
    except Exception:
        # Fallback: download whole model dir, then find a .joblib
        p = download_artifacts(
            run_id=mv.run_id,
            artifact_path="model",
            dst_path=str(tmpdir),
        )
        joblibs = list(Path(p).rglob("*.joblib"))
        if not joblibs:
            raise RuntimeError("Downloaded model dir has no .joblib")
        return joblibs[0]


def load_model():
    """Load (and cache) the current model pointed to by alias (default: Production)."""
    global _model_cache
    model_name = os.getenv(MODEL_NAME_ENV, "cxr-demo")
    joblib_path = _download_by_alias(model_name, DEFAULT_ALIAS)
    obj = joblib.load(joblib_path)
    _model_cache.update(
        {"path": str(joblib_path), "obj": obj, "version": f"{model_name}@{DEFAULT_ALIAS}"}
    )
    return obj


def predict_batch(X):
    """Predict with cached model; loads if empty."""
    if _model_cache["obj"] is None:
        load_model()
    return _model_cache["obj"].predict(X)


def model_short_sha():
    """Return a short identifier for the loaded model (name@alias or 'unloaded')."""
    return _model_cache.get("version") or "unloaded"
