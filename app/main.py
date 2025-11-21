from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import os

# Registry-aware loader: pulls latest <MODEL_NAME>@<STAGE> from MLflow
# and caches it internally. No args required.
from app.model_io import load_model, predict_batch, model_short_sha

app = FastAPI(title="medical-cv-serve-baseline", version="0.0.1")

class Features(BaseModel):
    f1: float = Field(...)
    f2: float = Field(...)
    f3: float = Field(...)
    f4: float = Field(...)

class PredictRequest(BaseModel):
    items: List[Features]

BUILD_REV = os.getenv("BUILD_REV", "local")

@app.on_event("startup")
def _startup():
    try:
        load_model()
    except Exception as e:
        print(f"[startup] Model not loaded yet: {e}")

@app.get("/live")
def live():
    return {"status": "alive"}

@app.get("/ready")
def ready():
    # ok when model is loaded
    return {"ready": model_short_sha() != "unloaded"}

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "model": model_short_sha(),  # e.g., "cxr-demo@Production" or "local:<file>"
        "build_rev": BUILD_REV,
    }

@app.get("/version")
def version():
    return {"model": model_short_sha(), "build_rev": BUILD_REV}

@app.get("/model-info")
def model_info():
    return {
        "alias": os.getenv("MLFLOW_MODEL_ALIAS", "unknown"),
        "model_short_sha": model_short_sha(),
    }

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        X = [[it.f1, it.f2, it.f3, it.f4] for it in req.items]
        y = predict_batch(X)  # model is cached inside model_io
        # Cast to plain floats for JSON
        return {"predictions": [float(v) for v in y]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/reload")
def reload_model():
    # Pull the newest model from the registry (e.g., after promoting a version)
    load_model()
    return {"reloaded": True, "model": model_short_sha()}
