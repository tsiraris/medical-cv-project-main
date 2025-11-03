from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import os
from .model_io import load_model, predict_batch, model_short_sha

app = FastAPI(title="medical-cv-serve-baseline", version="0.0.1")

class Features(BaseModel):
    f1: float = Field(...)
    f2: float = Field(...)
    f3: float = Field(...)
    f4: float = Field(...)

class PredictRequest(BaseModel):
    items: List[Features]

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
MODEL = load_model(MODEL_PATH)
MODEL_SHA = model_short_sha(MODEL_PATH)
BUILD_REV = os.getenv("BUILD_REV", "local")

@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "model_sha": MODEL_SHA,
        "build_rev": BUILD_REV,
    }

@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        X = [[it.f1, it.f2, it.f3, it.f4] for it in req.items]
        preds = predict_batch(MODEL, X)
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
