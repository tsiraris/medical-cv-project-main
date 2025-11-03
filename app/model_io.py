import joblib
import hashlib
from typing import List

def load_model(path: str):
    return joblib.load(path)

def predict_batch(model, X: List[List[float]]):
    # supports sklearn linear/logistic models for baseline
    y = model.predict(X)
    try:
        p = model.predict_proba(X)[:, 1].tolist()
    except Exception:
        p = None
    return [{"y": int(yi), "p": (pi if p else None)} for yi, pi in zip(y.tolist(), p or [None]*len(y))]

def model_short_sha(path: str, n: int = 8) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:n]
