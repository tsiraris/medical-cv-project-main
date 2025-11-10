# src/utils/mlflow_ready.py
import os
import time
import urllib.parse
import requests
import mlflow
from mlflow.tracking import MlflowClient

READY_STATUS = {200, 401, 403}  # 200 OK; 401/403 mean "alive but protected"

def _is_http_uri(uri: str) -> bool:
    try:
        scheme = urllib.parse.urlparse(uri).scheme.lower()
        return scheme in {"http", "https"}
    except Exception:
        return False

def wait_for_mlflow(uri: str, timeout: float = 180.0, interval: float = 2.0) -> None:
    """
    Waits for an HTTP(S) MLflow tracking server to be reachable.
    - No-op for non-HTTP URIs (e.g., file:).
    - Accepts 404 on root (some servers route UI differently).
    - Final check uses MlflowClient.search_experiments (MLflow >= 2.x).
    Set MLFLOW_WAIT_DISABLE=1 to bypass.
    """
    if os.getenv("MLFLOW_WAIT_DISABLE") == "1":
        return

    if not _is_http_uri(uri):
        return  # file backend etc.

    base = uri.rstrip("/")
    endpoints = [
        base,  # may be 404 and still healthy
        f"{base}/health",
        f"{base}/api/2.0/mlflow/experiments/list",
    ]

    start = time.time()
    last_err = None

    while time.time() - start < timeout:
        for url in endpoints:
            try:
                resp = requests.get(url, timeout=5)
                if (resp.status_code in READY_STATUS) or (url == base and resp.status_code == 404):
                    # definitive check via tracking client
                    _old = mlflow.get_tracking_uri()
                    try:
                        mlflow.set_tracking_uri(uri)
                        # Works on MLflow 2.x
                        MlflowClient().search_experiments(max_results=1)
                        return
                    finally:
                        mlflow.set_tracking_uri(_old)
            except Exception as e:
                last_err = e
                # try next endpoint / retry
        time.sleep(interval)

    raise RuntimeError(
        f"MLflow not ready at {uri} within {timeout}s. Last error: {last_err}"
    )
