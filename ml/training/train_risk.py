"""
train_risk.py
-------------
Training pipeline for the semiconductor supply chain risk forecaster.

What this script does:
  1. Loads the feature store (Parquet) from Member 1
  2. Engineers the target variable (disruption_risk_score)
  3. Creates sliding window sequences for LSTM training
  4. Trains the LSTMForecaster model
  5. Saves weights + scaler stats to registry/risk_forecaster/

How to run:
    python training/train_risk.py --data_path data/feature_store.parquet --epochs 50

Feature store format expected:
    date | shipping_rate_index | port_congestion_score | news_event_count |
         | export_restriction_flag | pmi_manufacturing | dram_spot_price_change |
         | taiwan_risk_index | [disruption_risk_score (optional)]

If disruption_risk_score is not in the feature store, this script
auto-generates it as a composite of the available features.
"""

import os
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.risk_forecaster import (
    LSTMForecaster, FEATURE_COLUMNS, RiskForecaster
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "lookback":     30,    # days of history fed to LSTM
    "horizon":      30,    # days of future to predict
    "batch_size":   32,
    "epochs":       50,
    "lr":           1e-3,
    "hidden_size":  128,
    "num_layers":   2,
    "dropout":      0.2,
    "val_split":    0.2,
    "seed":         42,
    "save_dir":     "registry/risk_forecaster",
}


# ─── Dataset ──────────────────────────────────────────────────────────────────

class RiskDataset(Dataset):
    """
    Sliding window dataset.
    X: (lookback, n_features)  → features window
    y: (horizon,)              → risk scores for next N days
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─── Feature Engineering ─────────────────────────────────────────────────────

def engineer_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target variable: disruption_risk_score (0–100).

    Formula (heuristic until historical disruption labels are available):
        - High port congestion        → +weight
        - High shipping rates         → +weight
        - Export restriction active   → +big weight
        - High taiwan_risk_index      → +weight
        - Low PMI                     → +weight (< 50 = contraction)
        - High news event count       → +weight
        - DRAM price spike > 5%       → +weight

    Replace with actual labeled disruption events once available.
    """
    df = df.copy()

    score = np.zeros(len(df))

    if "port_congestion_score" in df.columns:
        score += df["port_congestion_score"].clip(0, 100) * 0.20

    if "shipping_rate_index" in df.columns:
        # Normalise to 0–100 range
        sr = df["shipping_rate_index"]
        sr_norm = ((sr - sr.min()) / (sr.max() - sr.min() + 1e-9)) * 100
        score += sr_norm * 0.15

    if "export_restriction_flag" in df.columns:
        score += df["export_restriction_flag"] * 25.0

    if "taiwan_risk_index" in df.columns:
        score += df["taiwan_risk_index"].clip(0, 100) * 0.20

    if "pmi_manufacturing" in df.columns:
        # PMI < 50 = contraction territory → higher risk
        pmi_risk = (50 - df["pmi_manufacturing"]).clip(0, 10) * 3
        score += pmi_risk

    if "news_event_count" in df.columns:
        ne_norm = (df["news_event_count"].clip(0, 20) / 20) * 15
        score += ne_norm

    if "dram_spot_price_change" in df.columns:
        dram_spike = df["dram_spot_price_change"].clip(-20, 20).abs() * 0.5
        score += dram_spike

    df["disruption_risk_score"] = score.clip(0, 100)
    return df


def create_windows(
    features: np.ndarray,
    targets:  np.ndarray,
    lookback: int,
    horizon:  int,
) -> tuple:
    """Create sliding window sequences."""
    X, y = [], []
    total = len(features)
    for i in range(lookback, total - horizon + 1):
        X.append(features[i - lookback : i])           # past window
        y.append(targets[i : i + horizon] / 100.0)     # next horizon, scaled [0,1]
    return np.array(X), np.array(y)


# ─── Synthetic Feature Store ──────────────────────────────────────────────────

def generate_synthetic_feature_store(n_days: int = 500) -> pd.DataFrame:
    """
    Generate synthetic historical feature data for testing.
    Replace with actual data from Member 1's feature store.
    """
    logger.warning("No feature store found — generating synthetic data.")
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq="D")

    # Simulate a known disruption event around day 300 (e.g., 2021-like chip shortage)
    disruption_bump = np.zeros(n_days)
    disruption_bump[280:340] = np.linspace(0, 40, 60)   # ramp up
    disruption_bump[340:380] = np.linspace(40, 10, 40)  # taper down

    df = pd.DataFrame({
        "date":                     dates,
        "shipping_rate_index":      np.random.uniform(600, 1400, n_days) + disruption_bump * 3,
        "port_congestion_score":    np.random.uniform(20, 70, n_days) + disruption_bump * 0.4,
        "news_event_count":         np.random.randint(0, 10, n_days) + (disruption_bump / 5).astype(int),
        "export_restriction_flag":  (disruption_bump > 20).astype(int),
        "pmi_manufacturing":        np.random.uniform(47, 56, n_days) - disruption_bump * 0.1,
        "dram_spot_price_change":   np.random.uniform(-3, 3, n_days) + disruption_bump * 0.15,
        "taiwan_risk_index":        np.random.uniform(15, 50, n_days) + disruption_bump * 0.5,
    })
    df.set_index("date", inplace=True)
    logger.info(f"Generated {n_days} days of synthetic feature data.")
    return df


# ─── Training Logic ───────────────────────────────────────────────────────────

def train(config: dict, data_path: str = None):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # 1. Load feature store
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading feature store from {data_path}")
        df = pd.read_parquet(data_path) if data_path.endswith(".parquet") \
             else pd.read_csv(data_path, parse_dates=["date"], index_col="date")
    else:
        df = generate_synthetic_feature_store()

    logger.info(f"Feature store shape: {df.shape}")

    # 2. Engineer target variable
    if "disruption_risk_score" not in df.columns:
        logger.info("Engineering disruption_risk_score from features...")
        df = engineer_risk_score(df)

    # Fill missing feature columns with 0
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    df = df.sort_index().fillna(method="ffill").fillna(0)

    # 3. Scale features
    scaler   = StandardScaler()
    feat_arr = scaler.fit_transform(df[FEATURE_COLUMNS].values)
    tgt_arr  = df["disruption_risk_score"].values

    # Save scaler stats for inference
    os.makedirs(config["save_dir"], exist_ok=True)
    scaler_path = os.path.join(config["save_dir"], "scaler_stats.json")
    with open(scaler_path, "w") as f:
        json.dump({"mean": scaler.mean_.tolist(), "std": scaler.scale_.tolist()}, f, indent=2)
    logger.info(f"Scaler stats saved to {scaler_path}")

    # 4. Create sliding windows
    X, y = create_windows(feat_arr, tgt_arr, config["lookback"], config["horizon"])
    logger.info(f"Sequences: X={X.shape}, y={y.shape}")

    # 5. Train/val split (time-aware — no shuffle)
    split_idx = int(len(X) * (1 - config["val_split"]))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    train_loader = DataLoader(RiskDataset(X_train, y_train),
                              batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(RiskDataset(X_val, y_val),
                              batch_size=config["batch_size"])

    # 6. Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    model = LSTMForecaster(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        horizon=config["horizon"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    # 7. Training loop
    best_val_loss = float("inf")
    best_model_path = None

    for epoch in range(config["epochs"]):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss, mae = 0.0, 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds    = model(X_batch)
                val_loss += criterion(preds, y_batch).item()
                mae      += (preds - y_batch).abs().mean().item() * 100  # scale back to 0-100

        val_loss /= len(val_loader)
        mae      /= len(val_loader)
        scheduler.step(val_loss)

        logger.info(
            f"Epoch {epoch+1:3d}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"MAE: {mae:.2f} risk pts"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            version         = f"v{epoch+1}_mae{mae:.1f}"
            best_model_path = os.path.join(config["save_dir"], f"risk_forecaster_{version}.pt")
            torch.save(model.state_dict(), best_model_path)

            meta = {
                "version":     version,
                "val_loss":    round(best_val_loss, 6),
                "val_mae":     round(mae, 2),
                "train_size":  len(X_train),
                "val_size":    len(X_val),
                "config":      config,
            }
            with open(os.path.join(config["save_dir"], "best_model_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            logger.info(f"✅ Best model saved: {best_model_path} (MAE={mae:.2f})")

    logger.info(f"\n🎉 Training complete. Best Val Loss: {best_val_loss:.6f}")
    logger.info(f"Best model: {best_model_path}")
    return best_model_path


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train semiconductor risk forecaster")
    parser.add_argument("--data_path",   type=str,   default=None)
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size",  type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",          type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--hidden_size", type=int,   default=DEFAULT_CONFIG["hidden_size"])
    parser.add_argument("--save_dir",    type=str,   default=DEFAULT_CONFIG["save_dir"])
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update({k: v for k, v in vars(args).items() if v is not None})

    train(config, data_path=args.data_path)