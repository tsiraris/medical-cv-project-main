# scripts/smoke_canary.py
import os
import sys
import time
from typing import Any, Dict, List

import requests

BASE_URL = os.getenv("INFERENCE_BASE_URL", "http://demo.local")
TIMEOUT = float(os.getenv("SMOKE_TIMEOUT", "5.0"))


def _get(path: str) -> requests.Response:
    url = BASE_URL.rstrip("/") + path
    return requests.get(url, timeout=TIMEOUT)


def _post(path: str, json: Dict[str, Any]) -> requests.Response:
    url = BASE_URL.rstrip("/") + path
    return requests.post(url, json=json, timeout=TIMEOUT)


def check_health() -> None:
    print(f"[smoke] GET /health @ {BASE_URL}")
    r = _get("/health")
    r.raise_for_status()
    data = r.json()
    print(f"[smoke] /health -> {data}")
    if not data.get("status") == "ok":
        raise SystemExit("[smoke] FAIL: health.status != ok")


def check_model_info() -> None:
    print(f"[smoke] GET /model-info @ {BASE_URL}")
    r = _get("/model-info")
    r.raise_for_status()
    data = r.json()
    print(f"[smoke] /model-info -> {data}")
    if data.get("model_short_sha") in (None, "unloaded"):
        raise SystemExit("[smoke] FAIL: model_short_sha is 'unloaded'")


def check_predict() -> None:
    print(f"[smoke] POST /predict @ {BASE_URL}")
    payload = {
        "items": [
            {"f1": 0.1, "f2": 0.2, "f3": 0.3, "f4": 0.4},
            {"f1": 0.9, "f2": 0.8, "f3": 0.7, "f4": 0.6},
        ]
    }
    r = _post("/predict", payload)
    r.raise_for_status()
    data = r.json()
    print(f"[smoke] /predict -> {data}")
    preds: List[float] = data.get("predictions") or []
    if len(preds) != 2:
        raise SystemExit("[smoke] FAIL: expected 2 predictions")


def main() -> None:
    try:
        check_health()
        time.sleep(0.5)
        check_model_info()
        time.sleep(0.5)
        check_predict()
    except Exception as e:
        print(f"[smoke] ERROR: {e}")
        raise SystemExit(1)

    print("[smoke] OK: health, model-info and predict passed")


if __name__ == "__main__":
    main()
