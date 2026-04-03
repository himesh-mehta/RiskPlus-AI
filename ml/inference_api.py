"""
inference_api.py
----------------
FastAPI REST API exposing both ML models.
This is the contract between Member 2 (ML) and Member 3 (Backend).

Endpoints:
  POST /predict/risk        ← Main endpoint: forecasts risk for next N days
  POST /classify/news       ← Classifies a single news item
  POST /classify/news/batch ← Classifies multiple news items
  POST /aggregate/news      ← Aggregates news into a risk signal
  GET  /health              ← Health check + model status
  GET  /models/info         ← Model versions and metadata

Run locally:
  uvicorn ml.inference_api:app --host 0.0.0.0 --port 8001 --reload

Member 3 calls this API at: http://ml-service:8001
"""

import os
import json
import logging
import time
from typing import Optional
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Add parent to path when running standalone
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.news_classifier import NewsClassifier, LABEL2ID
from models.risk_forecaster import RiskForecaster, FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Model Registry Paths ──────────────────────────────────────────────────────

NEWS_MODEL_PATH = os.getenv("NEWS_MODEL_PATH", "registry/news_model")
RISK_MODEL_PATH = os.getenv("RISK_MODEL_PATH", "registry/risk_model.pt")

# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Supply Chain ML Inference API",
    description="Electronics/Semiconductor sector disruption prediction models",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Member 3 will lock this down
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model singletons (loaded once at startup) ─────────────────────────────────

_news_classifier: Optional[NewsClassifier] = None
_risk_forecaster: Optional[RiskForecaster] = None
_startup_time = datetime.utcnow()


@app.on_event("startup")
async def load_models():
    global _news_classifier, _risk_forecaster, _risk_explainer, _news_explainer

    logger.info("Loading models...")

    # Load news classifier
    path = NEWS_MODEL_PATH if os.path.exists(NEWS_MODEL_PATH) else None
    _news_classifier = NewsClassifier(path)

    # Load risk forecaster
    path = RISK_MODEL_PATH if os.path.exists(RISK_MODEL_PATH) else None
    _risk_forecaster = RiskForecaster(path, horizon=30)

    logger.info("All models loaded and ready.")


def get_news_classifier() -> NewsClassifier:
    if _news_classifier is None:
        raise HTTPException(503, "News classifier not loaded")
    return _news_classifier


def get_risk_forecaster() -> RiskForecaster:
    if _risk_forecaster is None:
        raise HTTPException(503, "Risk forecaster not loaded")
    return _risk_forecaster


# ── Request / Response Schemas ────────────────────────────────────────────────

class NewsRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=2000,
                      example="TSMC halts fab operations due to Taiwan earthquake")
    include_explanation: bool = Field(True, description="Include SHAP-based explanation")

class NewsBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1, max_items=100)
    include_explanation: bool = False

class NewsAggregateRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1, max_items=500,
                              description="All news items for the time period")
    period_label: Optional[str] = Field(None, example="2024-W42")

class FeatureRow(BaseModel):
    """Single day of features. All values normalized to [0, 1] except price changes."""
    taiwan_port_congestion:    float = Field(..., ge=0, le=1)
    korea_port_congestion:     float = Field(..., ge=0, le=1)
    shanghai_port_congestion:  float = Field(..., ge=0, le=1)
    scfi_shipping_rate:        float = Field(..., ge=0, le=1)
    us_china_tension_score:    float = Field(..., ge=0, le=1)
    global_pmi_manufacturing:  float = Field(..., ge=0, le=1)
    dram_spot_price_change:    float = Field(..., ge=-1, le=1)
    nand_spot_price_change:    float = Field(..., ge=-1, le=1)
    news_risk_signal:          float = Field(..., ge=0, le=1)
    fab_utilization_rate:      float = Field(..., ge=0, le=1)
    date: Optional[str] = None

class RiskRequest(BaseModel):
    features: list[FeatureRow] = Field(
        ..., min_items=7,
        description="At least 7 days of features, ordered oldest → newest"
    )
    horizon: int = Field(7, ge=1, le=30, description="Days to forecast")
    include_explanation: bool = Field(True)

    @validator("features")
    def check_min_rows(cls, v):
        if len(v) < 7:
            raise ValueError("Need at least 7 days of features")
        return v


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — Member 3 calls this on startup."""
    return {
        "status": "healthy",
        "uptime_seconds": (datetime.utcnow() - _startup_time).total_seconds(),
        "models": {
            "news_classifier": _news_classifier is not None,
            "risk_forecaster": _risk_forecaster is not None,
        },
        "sector": "electronics_semiconductors",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/models/info")
def models_info():
    """Returns model metadata for the model registry dashboard."""
    news_meta, risk_meta = {}, {}

    news_eval_path = os.path.join(NEWS_MODEL_PATH, "eval_metrics.json")
    if os.path.exists(news_eval_path):
        with open(news_eval_path) as f:
            news_meta = json.load(f)

    risk_hist_path = RISK_MODEL_PATH.replace(".pt", "_history.json")
    if os.path.exists(risk_hist_path):
        with open(risk_hist_path) as f:
            hist = json.load(f)
            risk_meta["best_val_mae"] = min(hist.get("val_mae", [0]))
            risk_meta["epochs_trained"] = len(hist.get("val_mae", []))

    return {
        "news_classifier": {
            "base_model": "distilbert-base-uncased",
            "labels": list(LABEL2ID.keys()),
            "path": NEWS_MODEL_PATH,
            "eval": news_meta,
        },
        "risk_forecaster": {
            "architecture": "LSTM (2-layer, 128 hidden, attention)",
            "features": FEATURE_COLUMNS,
            "horizon_days": 30,
            "sequence_length": 30,
            "path": RISK_MODEL_PATH,
            "eval": risk_meta,
        },
    }


@app.post("/classify/news")
def classify_news(
    req: NewsRequest,
    clf: NewsClassifier = Depends(get_news_classifier),
):
    """
    Classify a single news headline or article.

    Member 3 calls this when a new article arrives from the news pipeline.
    """
    t0 = time.time()
    prediction = clf.predict(req.text)

    response = {
        "text": req.text[:200],
        "label": prediction["label"],
        "confidence": prediction["confidence"],
        "risk_weight": prediction["risk_weight"],
        "sector_relevant": prediction["sector_relevant"],
        "all_scores": prediction["all_scores"],
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }

    if req.include_explanation and _news_explainer:
        explanation = _news_explainer.explain(req.text)
        response["explanation"] = {
            "key_tokens": explanation["key_tokens"],
            "explanation_text": explanation["explanation_text"],
        }

    return response


@app.post("/classify/news/batch")
def classify_news_batch(
    req: NewsBatchRequest,
    clf: NewsClassifier = Depends(get_news_classifier),
):
    """
    Classify multiple news items at once. More efficient than individual calls.
    """
    t0 = time.time()
    predictions = clf.predict_batch(req.texts)
    results = []
    for text, pred in zip(req.texts, predictions):
        item = {
            "text": text[:200],
            "label": pred["label"],
            "confidence": pred["confidence"],
            "risk_weight": pred["risk_weight"],
            "sector_relevant": pred["sector_relevant"],
        }
        results.append(item)

    return {
        "results": results,
        "total": len(results),
        "sector_relevant_count": sum(1 for r in results if r["sector_relevant"]),
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }


@app.post("/aggregate/news")
def aggregate_news(
    req: NewsAggregateRequest,
    clf: NewsClassifier = Depends(get_news_classifier),
):
    """
    Aggregate a batch of news into a single risk signal (0–1).
    Member 3 calls this daily to produce the news_risk_signal feature.
    """
    t0 = time.time()
    agg = clf.aggregate_risk_signal(req.texts)
    return {
        **agg,
        "period": req.period_label,
        "sector": "semiconductors",
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }


@app.post("/predict/risk")
def predict_risk(
    req: RiskRequest,
    forecaster: RiskForecaster = Depends(get_risk_forecaster),
):
    """
    Forecast disruption risk scores for the next N days.

    This is the MAIN endpoint Member 3 calls to drive the dashboard.

    Input: last 30 days of feature store data
    Output: risk scores per day + explanation of top driving factors
    """
    t0 = time.time()

    # Convert to DataFrame
    features_df = pd.DataFrame([row.dict(exclude={"date"}) for row in req.features])

    try:
        prediction = forecaster.predict(features_df, horizon=req.horizon)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))

    response = {
        "sector": "semiconductors",
        "horizon_days": req.horizon,
        "risk_scores": prediction["risk_scores"],
        "risk_levels": prediction["risk_levels"],
        "current_risk": prediction["current_risk"],
        "peak_risk": prediction["peak_risk"],
        "peak_day": prediction["peak_day"],
        "trend": prediction["trend"],
        "summary": prediction["summary"],
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }
    return response


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "inference_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info",
    )