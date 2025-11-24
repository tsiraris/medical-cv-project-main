from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import os

from app.metrics import (
    metrics_middleware,
    metrics_endpoint,
    INFERENCE_LATENCY_SECONDS,
)
from app.model_io import load_model, predict_batch, model_short_sha

# Build / deployment metadata
BUILD_REV = os.getenv("BUILD_REV", "local")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "unknown")


# ------------------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------------------

app = FastAPI(title="medical-cv-serve-baseline", version="0.0.1")

# Wrap the app with the Prometheus metrics middleware
app.middleware("http")(metrics_middleware(app))


# ------------------------------------------------------------------------------
# Request / response models
# ------------------------------------------------------------------------------

class Features(BaseModel):
    f1: float = Field(..., description="Feature 1")
    f2: float = Field(..., description="Feature 2")
    f3: float = Field(..., description="Feature 3")
    f4: float = Field(..., description="Feature 4")


class PredictRequest(BaseModel):
    items: List[Features]


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "medical-cv-serve-baseline",
        "build_rev": BUILD_REV,
        "model_alias": MODEL_ALIAS,
        "model_short_sha": model_short_sha(),
    }


@app.get("/health")
def health():
    # Basic liveness endpoint
    return {"status": "ok", "model_loaded": model_short_sha() is not None}


@app.get("/model-info")
def model_info():
    return {
        "alias": MODEL_ALIAS,
        "model_short_sha": model_short_sha(),
        "build_rev": BUILD_REV,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Batch prediction endpoint.

    Also recording inference latency in the INFERENCE_LATENCY_SECONDS
    histogram, tagged by model_alias to compare Production vs Candidate.
    """
    try:
        # Convert list of Features → 2D list of floats
        X = [[it.f1, it.f2, it.f3, it.f4] for it in req.items]

        # Measure model inference latency
        with INFERENCE_LATENCY_SECONDS.labels(model_alias=MODEL_ALIAS).time():
            y = predict_batch(X)  # model_io handles caching / loading

        # Cast to plain floats for JSON
        return {"predictions": [float(v) for v in y]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/reload")
def reload_model():
    """
    Manually trigger a reload from the MLflow registry
    (e.g. after promoting a new version / alias in MLflow).
    """
    load_model()
    return {"reloaded": True, "model": model_short_sha()}


# Expose /metrics for Prometheus scraping (Step 3)
@app.get("/metrics")
async def metrics():
    return await metrics_endpoint()()
