# ML Module — Electronics/Semiconductor Risk Predictor

**Owner:** Member 2 (ML Engineer)  
**Sector:** Electronics / Semiconductors

---

## Folder Structure

```
ml/
├── models/
│   ├── news_classifier.py    ← DistilBERT NLP classifier (6 event labels)
│   └── risk_forecaster.py    ← LSTM time-series risk scorer (0–100)
├── registry/                 ← Saved model checkpoints + eval metrics
│   ├── news_model/           ← Fine-tuned DistilBERT (after training)
│   └── risk_model.pt         ← LSTM checkpoint (after training)
├── training/
│   ├── train_news.py         ← Fine-tune NLP classifier
│   ├── train_risk.py         ← Train LSTM forecaster
│   ├── evaluate.py           ← Evaluate both models
│   └── explainability.py     ← SHAP explanations
├── inference_api.py          ← FastAPI server (what Member 3 calls)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
cd ml
pip install -r requirements.txt
```

### 2. Train models with synthetic data (no real data needed yet)
```bash
# Train news classifier
python training/train_news.py --synthetic --epochs 5

# Train risk forecaster
python training/train_risk.py --synthetic --epochs 30 --horizon 30
```

### 3. Evaluate
```bash
python training/evaluate.py --model both --synthetic
```

### 4. Start the inference API
```bash
uvicorn ml.inference_api:app --host 0.0.0.0 --port 8001 --reload
```

### 5. Test the API
```bash
# Health check
curl http://localhost:8001/health

# Classify news
curl -X POST http://localhost:8001/classify/news \
  -H "Content-Type: application/json" \
  -d '{"text": "TSMC halts Taiwan fab due to earthquake", "include_explanation": true}'

# Risk forecast (POST with feature payload)
# See /docs for interactive Swagger UI at http://localhost:8001/docs
```

---

## What Member 3 (Backend) Needs From You

### API Endpoints at `http://ml-service:8001`

| Endpoint | Method | When to Call |
|---|---|---|
| `/predict/risk` | POST | Every hour or on-demand for dashboard refresh |
| `/classify/news` | POST | When new article arrives from news pipeline |
| `/classify/news/batch` | POST | Batch process daily news digest |
| `/aggregate/news` | POST | Daily — produces `news_risk_signal` feature |
| `/health` | GET | On startup and monitoring |

### Feature Payload for `/predict/risk`
Member 3 sends the last 30 days of features (provided by Member 1):
```json
{
  "features": [
    {
      "date": "2024-01-01",
      "taiwan_port_congestion": 0.72,
      "korea_port_congestion": 0.45,
      "shanghai_port_congestion": 0.68,
      "scfi_shipping_rate": 0.55,
      "us_china_tension_score": 0.80,
      "global_pmi_manufacturing": 0.48,
      "dram_spot_price_change": 0.03,
      "nand_spot_price_change": -0.01,
      "news_risk_signal": 0.65,
      "fab_utilization_rate": 0.88
    }
  ],
  "horizon": 7,
  "include_explanation": true
}
```

### Risk Response
```json
{
  "sector": "semiconductors",
  "horizon_days": 7,
  "risk_scores": [62.1, 65.3, 68.7, 71.2, 73.8, 72.1, 69.4],
  "risk_levels": ["HIGH", "HIGH", "HIGH", "HIGH", "CRITICAL", "HIGH", "HIGH"],
  "current_risk": 62.1,
  "peak_risk": 73.8,
  "peak_day": 5,
  "trend": "RISING",
  "summary": "Risk is currently HIGH (62/100) and RISING. Expected to reach CRITICAL (74/100) by day 5.",
  "explanation": {
    "top_factors": [
      {"feature": "taiwan_port_congestion", "display_name": "Taiwan Port Congestion", "impact": 18.4, "direction": "increasing"},
      {"feature": "us_china_tension_score", "display_name": "US-China Trade Tensions",  "impact": 12.1, "direction": "stable"},
      {"feature": "news_risk_signal",        "display_name": "News Risk Signal (NLP)",   "impact": 9.3,  "direction": "increasing"}
    ],
    "explanation_text": "Risk is primarily driven by elevated and rising Taiwan Port Congestion, compounded by persistently high US-China Trade Tensions and elevated and rising News Risk Signal (NLP)."
  }
}
```

---

## What You Need From Member 1 (Data Engineer)

Feature store schema — daily Parquet file with these columns:

| Column | Type | Description |
|---|---|---|
| `date` | date | Row date |
| `taiwan_port_congestion` | float [0-1] | ICTSI/MPA congestion index |
| `korea_port_congestion` | float [0-1] | Busan/Incheon congestion |
| `shanghai_port_congestion` | float [0-1] | SIPG congestion index |
| `scfi_shipping_rate` | float [0-1] | SCFI index normalized |
| `us_china_tension_score` | float [0-1] | GDELT tension derived |
| `global_pmi_manufacturing` | float [0-1] | Global Manufacturing PMI normalized |
| `dram_spot_price_change` | float [-1,1] | Weekly % change in DRAM prices |
| `nand_spot_price_change` | float [-1,1] | Weekly % change in NAND prices |
| `news_risk_signal` | float [0-1] | **Provided by this module** (aggregate_news endpoint) |
| `fab_utilization_rate` | float [0-1] | SEMI utilization data |

---

## Risk Level Thresholds

| Score | Level | Meaning |
|---|---|---|
| 0–30 | LOW | Normal operations |
| 30–55 | MEDIUM | Monitor closely |
| 55–75 | HIGH | Alert: proactive action needed |
| 75–100 | CRITICAL | Severe disruption risk |

---

## Phase 2 Upgrade: TFT Model
Once sufficient real data is available (6+ months), upgrade `risk_forecaster.py` to Temporal Fusion Transformer:
```bash
pip install pytorch-forecasting pytorch-lightning
```
Then swap `LSTMRiskModel` for `TemporalFusionTransformer` in `risk_forecaster.py`. The API contract remains unchanged.