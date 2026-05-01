"""
Satellite Intelligence Platform — FastAPI prediction service.
Serves the trained Logistic Regression model for satellite damage detection.

Endpoints
---------
POST /predict          single-scene prediction
POST /predict/batch    batch predictions (up to 100)
GET  /health           liveness / readiness probe
GET  /model/info       model metadata
"""

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── model paths (overridable via env vars for Docker) ─────────────────────────
_BASE = Path(os.getenv("MODEL_DIR", Path(__file__).parent.parent / "satellite-damage-detection" / "models"))
_MODEL_PATH   = _BASE / "logistic_regression.pkl"
_SCALER_PATH  = _BASE / "scaler.pkl"
_FEATURES_PATH = _BASE / "feature_cols.json"

MODEL_VERSION = "1.0.0"
MODEL_NAME    = "satellite-damage-detector"

# ── globals populated at startup ──────────────────────────────────────────────
ml_model   = None
scaler     = None
features   = None
start_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model, scaler, features, start_time
    ml_model   = joblib.load(_MODEL_PATH)
    scaler     = joblib.load(_SCALER_PATH)
    features   = json.load(open(_FEATURES_PATH))
    start_time = time.time()
    print(f"[startup] model loaded — features: {features}")
    yield
    print("[shutdown] cleaning up")


app = FastAPI(
    title="Satellite Intelligence Platform API",
    description="Predicts infrastructure damage in satellite imagery using spectral indices.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class SceneFeatures(BaseModel):
    """Six spectral index statistics extracted from a Sentinel-2 scene."""
    ndvi_mean: float = Field(..., ge=-1.0, le=1.0,  description="Mean NDVI (-1 to 1)")
    ndvi_std:  float = Field(..., ge=0.0,  le=1.0,  description="Std-dev NDVI (≥0)")
    ndbi_mean: float = Field(..., ge=-1.0, le=1.0,  description="Mean NDBI (-1 to 1)")
    ndbi_std:  float = Field(..., ge=0.0,  le=1.0,  description="Std-dev NDBI (≥0)")
    bsi_mean:  float = Field(..., ge=-1.0, le=1.0,  description="Mean BSI (-1 to 1)")
    bsi_std:   float = Field(..., ge=0.0,  le=1.0,  description="Std-dev BSI (≥0)")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "ndvi_mean": 0.30,
                "ndvi_std":  0.17,
                "ndbi_mean": 0.19,
                "ndbi_std":  0.13,
                "bsi_mean":  0.20,
                "bsi_std":   0.12,
            }]
        }
    }


class PredictionResponse(BaseModel):
    prediction_id:     str
    damage_label:      int   # 0 = undamaged, 1 = damaged
    damage_label_text: str
    damage_probability: float
    model_version:     str


class BatchInput(BaseModel):
    records: List[SceneFeatures] = Field(..., max_length=100)


class BatchResponse(BaseModel):
    predictions:        List[PredictionResponse]
    count:              int
    processing_time_ms: float


# ── helpers ───────────────────────────────────────────────────────────────────

def _predict_one(scene: SceneFeatures) -> PredictionResponse:
    X = np.array([[
        scene.ndvi_mean, scene.ndvi_std,
        scene.ndbi_mean, scene.ndbi_std,
        scene.bsi_mean,  scene.bsi_std,
    ]])
    X_scaled = scaler.transform(X)
    label    = int(ml_model.predict(X_scaled)[0])
    prob     = round(float(ml_model.predict_proba(X_scaled)[0][1]), 4)
    return PredictionResponse(
        prediction_id=str(uuid.uuid4()),
        damage_label=label,
        damage_label_text="damaged" if label == 1 else "undamaged",
        damage_probability=prob,
        model_version=MODEL_VERSION,
    )


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Satellite Intelligence Platform API is running", "docs": "/docs"}


@app.post("/predict", response_model=PredictionResponse, summary="Single-scene damage prediction")
def predict(scene: SceneFeatures):
    return _predict_one(scene)


@app.post("/predict/batch", response_model=BatchResponse, summary="Batch damage prediction (≤100 scenes)")
def predict_batch(batch: BatchInput):
    t0 = time.time()
    results = [_predict_one(s) for s in batch.records]
    elapsed_ms = round((time.time() - t0) * 1000, 2)
    return BatchResponse(predictions=results, count=len(results), processing_time_ms=elapsed_ms)


@app.get("/health", summary="Health check")
def health():
    return {
        "status": "healthy",
        "model_loaded": ml_model is not None,
        "model_version": MODEL_VERSION,
        "uptime_seconds": round(time.time() - start_time, 1) if start_time else None,
    }


@app.get("/model/info", summary="Model metadata")
def model_info():
    return {
        "model_name":    MODEL_NAME,
        "model_type":    type(ml_model).__name__ if ml_model else None,
        "version":       MODEL_VERSION,
        "features":      features,
        "n_features":    len(features) if features else None,
        "classes":       [0, 1],
        "class_labels":  {0: "undamaged", 1: "damaged"},
        "metrics": {
            "test_accuracy":  0.9565,
            "test_f1":        0.975,
            "test_auc_roc":   0.9875,
        },
        "trained_on": "Sentinel-2 L2A — Mariupol, Kharkiv, Bakhmut (2021–2024)",
    }


# ── global error handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred."},
    )