import os
import time
import requests
from urllib.parse import urlparse, urlunparse

DEFAULT_TRACKING_URI = "http://127.0.0.1:5500"  # host → container 5000 via compose mapping

def _normalize_to_ipv4(uri: str) -> str:
    """Normalize localhost to 127.0.0.1 to avoid IPv6 (::1) issues on Windows."""
    try:
        p = urlparse(uri)
        host = "127.0.0.1" if p.hostname in (None, "", "localhost") else p.hostname
        port = p.port or 5500
        return urlunparse((p.scheme or "http", f"{host}:{port}", p.path or "", "", "", ""))
    except Exception:
        return "http://127.0.0.1:5500"

def get_tracking_uri() -> str:
    """Prefer env, else default. Always normalize to IPv4."""
    raw = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    return _normalize_to_ipv4(raw)

def wait_for_mlflow(uri: str, timeout_s: int = 180, interval_s: float = 1.5) -> None:
    """
    Poll the root page (/) until the server is ready or timeout.
    Any 2xx/3xx/4xx (non-5xx) means the app is up.
    """
    uri = _normalize_to_ipv4(uri)
    deadline = time.time() + timeout_s
    url = f"{uri.rstrip('/')}/"
    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code < 500:
                return
        except Exception as e:
            last_err = e
        time.sleep(interval_s)
    raise RuntimeError(f"MLflow not ready at {uri} within {timeout_s}s. Last error: {last_err}")
