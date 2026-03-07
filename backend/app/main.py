"""
Budget Risk Predictor - FastAPI Backend

Endpoints:
  POST /api/predict  — Run risk inference
  GET  /api/health   — Health check
  POST /api/train    — Trigger model (re)training
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import os
import logging

from .model import (
    predict,
    train_model,
    save_model,
    load_model,
    BudgetRiskNet,
    CATEGORIES,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Budget Risk Predictor API",
    description="Neural network inference for personal spending risk analysis",
    version="0.9.2",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global model state ---
MODEL_PATH = "model"
_model = None
_scaler = None


def get_model():
    global _model, _scaler
    if _model is None:
        if not os.path.exists(f"{MODEL_PATH}/risk_model.pt"):
            logger.info("No saved model found. Training from scratch...")
            _model, _scaler = train_model()
            save_model(_model, _scaler, MODEL_PATH)
        else:
            logger.info("Loading saved model...")
            _model, _scaler = load_model(MODEL_PATH)
    return _model, _scaler


@app.on_event("startup")
async def startup():
    get_model()
    logger.info("Risk model loaded and ready.")


# --- Schemas ---
class PredictRequest(BaseModel):
    last_month_spend: list[float] = Field(
        ...,
        min_items=5, max_items=5,
        description="Actual spending last month: [groceries, rent, utilities, entertainment, misc]",
        example=[480, 1200, 140, 320, 180],
    )
    next_budget: list[float] = Field(
        ...,
        min_items=5, max_items=5,
        description="Planned budget for next month",
        example=[420, 1200, 130, 200, 150],
    )

    @validator("last_month_spend", "next_budget", each_item=True)
    def must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError("Spending values must be non-negative")
        return v


class PredictResponse(BaseModel):
    predictions: dict[str, float]
    highest_risk: dict[str, object]
    safest: dict[str, object]
    model_version: str = "0.9.2"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    categories: list[str]


# --- Routes ---
@app.get("/api/health", response_model=HealthResponse)
def health():
    model, _ = get_model()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "categories": CATEGORIES,
    }


@app.post("/api/predict", response_model=PredictResponse)
def run_predict(req: PredictRequest):
    try:
        model, scaler = get_model()
        preds = predict(model, scaler, req.last_month_spend, req.next_budget)

        highest = max(preds.items(), key=lambda x: x[1])
        safest  = min(preds.items(), key=lambda x: x[1])

        return {
            "predictions": preds,
            "highest_risk": {"category": highest[0], "probability": highest[1]},
            "safest":        {"category": safest[0],  "probability": safest[1]},
        }
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train")
def retrain():
    """Retrain model on fresh synthetic data."""
    global _model, _scaler
    try:
        _model, _scaler = train_model()
        save_model(_model, _scaler, MODEL_PATH)
        return {"status": "training complete", "model_version": "0.9.2"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
