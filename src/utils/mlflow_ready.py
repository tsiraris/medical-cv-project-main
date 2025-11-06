"""
Waits for an MLflow tracking server *only when* we're using HTTP/HTTPS.
If using the file backend (MLFLOW_TRACKING_URI = 'file:./mlruns'), there is
no server to wait for, so we return immediately.

This makes local runs (with server) and CI runs (file backend) both happy.
"""
from __future__ import annotations

import time
from urllib.parse import urlparse

import requests


def wait_for_mlflow(uri: str, timeout_s: int = 180) -> None:
    """
    If uri is http(s), poll /api/2.0/mlflow/experiments/list until ready.
    If uri is 'file:...' (or empty), return immediately.
    """
    if not uri:
        return

    parsed = urlparse(uri)

    # MLflow "file" backend: nothing to wait for.
    # Note: valid MLflow file URI is 'file:./mlruns' (colon, no '//').
    if parsed.scheme == "file" or uri.startswith("file:"):
        return

    # If no recognizable scheme, don't wait.
    if parsed.scheme not in ("http", "https"):
        return

    url = uri.rstrip("/") + "/api/2.0/mlflow/experiments/list"
    deadline = time.time() + timeout_s
    last_err: object | None = None

    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=3)
            # 200: OK; 401/403 mean server is up but requires auth
            if r.status_code in (200, 401, 403):
                return
            last_err = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001 - we want to show whatever we got
            last_err = e
        time.sleep(2)

    raise RuntimeError(
        f"MLflow not ready at {uri} within {timeout_s}s. Last error: {last_err}"
    )
