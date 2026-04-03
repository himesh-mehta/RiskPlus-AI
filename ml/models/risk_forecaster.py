"""
risk_forecaster.py
------------------
Time-series model that predicts semiconductor supply chain disruption
risk scores (0–100) for the next 7 / 14 / 30 days.

Architecture:
    Primary   → Temporal Fusion Transformer (pytorch-forecasting)
    Fallback  → LSTM (pure PyTorch, no extra deps needed)

Input features (from Member 1's feature store):
    - shipping_rate_index     : Freight cost index (e.g. SCFI)
    - port_congestion_score   : 0–100 congestion at key ports (Taiwan, Korea, Rotterdam)
    - news_event_count        : Weekly count of supply-chain-relevant news
    - export_restriction_flag : Binary — active export restriction this week
    - pmi_manufacturing       : Global Manufacturing PMI
    - dram_spot_price_change  : Week-over-week % change in DRAM spot price
    - taiwan_risk_index       : Composite geopolitical risk for Taiwan Strait

Output:
    {
        "sector": "semiconductors",
        "horizon_days": 7,
        "risk_scores": [45.2, 47.8, 52.1, 55.0, 58.3, 60.1, 63.4],
        "peak_risk": 63.4,
        "risk_level": "medium",
        "forecast_date": "2026-03-28"
    }
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "shipping_rate_index",
    "port_congestion_score",
    "news_event_count",
    "export_restriction_flag",
    "pmi_manufacturing",
    "dram_spot_price_change",
    "taiwan_risk_index",
]

RISK_THRESHOLDS = {
    "low":      (0,  35),
    "medium":   (35, 65),
    "high":     (65, 85),
    "critical": (85, 100),
}


# ─── LSTM Model (Fallback / Default) ─────────────────────────────────────────

class LSTMForecaster(nn.Module):
    """
    Lightweight LSTM that takes a window of historical features
    and predicts risk scores for each day in the forecast horizon.
    """

    def __init__(
        self,
        input_size:   int = len(FEATURE_COLUMNS),
        hidden_size:  int = 128,
        num_layers:   int = 2,
        dropout:      float = 0.2,
        horizon:      int = 30,
    ):
        super().__init__()
        self.horizon     = horizon
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout    = nn.Dropout(dropout)
        self.fc         = nn.Linear(hidden_size, horizon)
        self.activation = nn.Sigmoid()  # output in [0, 1] → scale to [0, 100]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            (batch_size, horizon)  — values in [0, 1]
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]           # take last timestep
        out         = self.dropout(last_hidden)
        out         = self.fc(out)
        return self.activation(out)


# ─── Main Forecaster Wrapper ──────────────────────────────────────────────────

class RiskForecaster:
    """
    Wraps the LSTM (or TFT) model with preprocessing, prediction, and
    risk-level classification logic.

    Designed to be called from inference_api.py with a feature DataFrame.
    """

    LOOKBACK_DAYS = 30   # how many past days the model sees
    MAX_HORIZON   = 30   # maximum forecast horizon

    def __init__(self, model_path: Optional[str] = None, horizon: int = 7):
        """
        Args:
            model_path: Path to saved .pt model weights (from registry/).
                        If None, initialises a fresh untrained model (dev mode).
            horizon:    Number of days to forecast (7, 14, or 30).
        """
        self.horizon    = min(horizon, self.MAX_HORIZON)
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        self.model = LSTMForecaster(
            input_size=len(FEATURE_COLUMNS),
            horizon=self.MAX_HORIZON,
        ).to(self.device)

        self._load_weights()
        self._load_scaler()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _load_weights(self):
        if self.model_path and os.path.exists(self.model_path):
            logger.info(f"Loading model weights from: {self.model_path}")
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            logger.info("Model weights loaded.")
        else:
            logger.warning(
                "No trained weights found — model outputs are random.\n"
                "Train with: python training/train_risk.py"
            )
            self.model.eval()

    def _load_scaler(self):
        """Load feature scaler stats (mean/std) saved during training."""
        scaler_path = (
            os.path.join(os.path.dirname(self.model_path), "scaler_stats.json")
            if self.model_path else None
        )
        if scaler_path and os.path.exists(scaler_path):
            import json
            with open(scaler_path) as f:
                stats = json.load(f)
            self.feature_mean = np.array(stats["mean"])
            self.feature_std  = np.array(stats["std"])
            logger.info("Scaler stats loaded.")
        else:
            # Default normalization — will be replaced after training
            self.feature_mean = np.zeros(len(FEATURE_COLUMNS))
            self.feature_std  = np.ones(len(FEATURE_COLUMNS))

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check all required columns are present, fill missing with 0."""
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            logger.warning(f"Missing feature columns, filling with 0: {missing}")
            for col in missing:
                df[col] = 0.0
        return df[FEATURE_COLUMNS].copy()

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Z-score normalization using training stats."""
        std = np.where(self.feature_std == 0, 1, self.feature_std)
        return (arr - self.feature_mean) / std

    def _prepare_tensor(self, df: pd.DataFrame) -> torch.Tensor:
        """Convert DataFrame window → (1, seq_len, features) tensor."""
        df    = self._validate_features(df)
        arr   = df.values.astype(np.float32)

        # Ensure we always have LOOKBACK_DAYS rows
        if len(arr) < self.LOOKBACK_DAYS:
            pad   = np.zeros((self.LOOKBACK_DAYS - len(arr), arr.shape[1]), dtype=np.float32)
            arr   = np.vstack([pad, arr])
        arr   = arr[-self.LOOKBACK_DAYS:]  # take most recent window

        arr   = self._normalize(arr)
        tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1, seq, feat)
        return tensor.to(self.device)

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, features: pd.DataFrame, horizon: Optional[int] = None) -> dict:
        """
        Predict disruption risk scores for the semiconductor sector.

        Args:
            features: DataFrame with LOOKBACK_DAYS rows and FEATURE_COLUMNS.
                      Typically the last 30 days from the feature store.
            horizon:  Override default forecast horizon (days).

        Returns:
            {
                "sector": "semiconductors",
                "horizon_days": 7,
                "risk_scores": [45.2, 47.8, ...],   # one per day
                "peak_risk": 63.4,
                "average_risk": 52.7,
                "risk_level": "medium",
                "forecast_dates": ["2026-03-29", ...],
                "forecast_date": "2026-03-28"
            }
        """
        horizon = min(horizon or self.horizon, self.MAX_HORIZON)

        x         = self._prepare_tensor(features)
        with torch.no_grad():
            raw_out = self.model(x)                    # (1, MAX_HORIZON) in [0,1]

        scores = (raw_out[0, :horizon].cpu().numpy() * 100).tolist()
        scores = [round(s, 2) for s in scores]

        today         = datetime.utcnow().date()
        forecast_dates = [
            str(today + timedelta(days=i + 1)) for i in range(horizon)
        ]

        peak    = round(max(scores), 2)
        average = round(sum(scores) / len(scores), 2)
        level   = self._risk_level(peak)

        return {
            "sector":         "semiconductors",
            "horizon_days":   horizon,
            "risk_scores":    scores,
            "peak_risk":      peak,
            "average_risk":   average,
            "risk_level":     level,
            "forecast_dates": forecast_dates,
            "forecast_date":  str(today),
        }

    @staticmethod
    def _risk_level(score: float) -> str:
        for level, (lo, hi) in RISK_THRESHOLDS.items():
            if lo <= score < hi:
                return level
        return "critical"

    # ── Utilities ─────────────────────────────────────────────────────────────

    def save(self, save_dir: str, version: str = "v1"):
        """Save model weights to registry/."""
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"risk_forecaster_{version}.pt")
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
        return path

    def save_scaler(self, save_dir: str, mean: np.ndarray, std: np.ndarray):
        """Save normalization stats alongside the model."""
        import json
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "scaler_stats.json")
        with open(path, "w") as f:
            json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)
        logger.info(f"Scaler stats saved to {path}")


# ─── Smoke Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate 30 days of feature data from the feature store
    np.random.seed(42)
    dummy_features = pd.DataFrame({
        "shipping_rate_index":      np.random.uniform(800, 1200, 30),
        "port_congestion_score":    np.random.uniform(40, 80, 30),
        "news_event_count":         np.random.randint(0, 15, 30),
        "export_restriction_flag":  np.random.randint(0, 2, 30),
        "pmi_manufacturing":        np.random.uniform(45, 55, 30),
        "dram_spot_price_change":   np.random.uniform(-5, 5, 30),
        "taiwan_risk_index":        np.random.uniform(20, 60, 30),
    })

    forecaster = RiskForecaster(horizon=7)

    for h in [7, 14, 30]:
        result = forecaster.predict(dummy_features, horizon=h)
        print(f"\n{'='*55}")
        print(f"  Forecast Horizon: {h} days")
        print(f"  Sector     : {result['sector']}")
        print(f"  Peak Risk  : {result['peak_risk']}  ({result['risk_level'].upper()})")
        print(f"  Avg Risk   : {result['average_risk']}")
        print(f"  Scores     : {result['risk_scores']}")