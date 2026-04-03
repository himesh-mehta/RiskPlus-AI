"""
train_news.py
-------------
Fine-tuning pipeline for the semiconductor news classifier.

What this script does:
  1. Loads labeled news data from the feature store (CSV or Parquet)
  2. Splits into train / validation sets
  3. Fine-tunes DistilBERT on 6 semiconductor supply-chain labels
  4. Evaluates on validation set (F1, precision, recall per label)
  5. Saves the best model to registry/news_classifier/

How to run:
    python training/train_news.py --data_path data/news_labeled.csv --epochs 5

Data format expected (CSV or Parquet):
    text    | label
    --------|---------------------
    "TSMC.."| fab_shutdown
    "Port.."| logistics_delay
    ...

If you don't have labeled data yet, this script generates a small
synthetic seed dataset so you can test the pipeline end-to-end.
"""

import os
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# ─── Add parent directory to path ────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.news_classifier import LABEL2ID, ID2LABEL, LABEL_DESCRIPTIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "base_model":   "distilbert-base-uncased",
    "max_length":   128,
    "batch_size":   16,
    "epochs":       5,
    "lr":           2e-5,
    "warmup_ratio": 0.1,
    "val_split":    0.15,
    "seed":         42,
    "save_dir":     "registry/news_classifier",
}


# ─── Dataset ──────────────────────────────────────────────────────────────────

class NewsDataset(Dataset):
    def __init__(self, texts: list, labels: list, tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ─── Synthetic Seed Data (used when no real data is available) ────────────────

def generate_seed_data() -> pd.DataFrame:
    """
    Generate a small synthetic dataset for pipeline testing.
    Replace this with real labeled data from NewsAPI / GDELT.
    """
    logger.warning("No data file found — generating synthetic seed data for testing.")

    samples = {
        "fab_shutdown": [
            "TSMC halts production at Fab 18 after major earthquake strikes Taiwan",
            "Samsung shuts down semiconductor plant due to fire at Pyeongtaek facility",
            "Intel forced to suspend chip manufacturing amid power grid failure",
            "TSMC confirms temporary suspension of advanced node production",
            "Micron Technology closes Boise fab for emergency maintenance",
            "SK Hynix pauses DRAM production following chemical leak at fab",
            "GlobalFoundries halts operations at Malta NY plant",
            "Taiwan fab shutdowns expected after typhoon makes landfall",
        ],
        "raw_material_shortage": [
            "Global neon gas shortage threatens chip production as Ukraine conflict continues",
            "Palladium supply disruption risks semiconductor manufacturing slowdown",
            "Silicon wafer shortage hits foundries as demand outpaces supply",
            "Shortage of specialty chemicals threatens advanced semiconductor packaging",
            "Rare earth supply crunch intensifies as China tightens export quotas",
            "Fluorine gas shortage affecting photolithography processes at major fabs",
            "Critical material supply gap seen for next-gen chip manufacturing",
            "Wafer substrate shortage compounds semiconductor supply chain stress",
        ],
        "logistics_delay": [
            "Port of Rotterdam faces record congestion delaying semiconductor equipment",
            "Shanghai port backlog causes 3-week delay for chip component shipments",
            "Suez Canal disruption reroutes semiconductor cargo adding 12 days transit",
            "Air freight capacity shortage delays urgent chip deliveries to automakers",
            "Logistics bottlenecks at Kaohsiung port impact Taiwan semiconductor exports",
            "West coast port strikes threaten semiconductor component supply chains",
            "DHL reports significant delays on Asia-Europe semiconductor freight lanes",
            "Container shortage worsens as shipping rates hit record highs for chip cargo",
        ],
        "export_restriction": [
            "US imposes sweeping export controls on advanced semiconductor equipment to China",
            "Netherlands restricts ASML lithography machine exports amid geopolitical pressure",
            "China announces retaliatory restrictions on gallium and germanium exports",
            "Biden administration tightens chip export rules targeting Chinese AI firms",
            "Japan joins US in restricting semiconductor manufacturing equipment exports",
            "New US rules bar advanced chip exports to 21 additional Chinese entities",
            "South Korea reviews semiconductor technology export controls following US pressure",
            "EU considers coordinated semiconductor export control framework",
        ],
        "demand_surge": [
            "AI boom drives unprecedented demand for Nvidia H100 GPU chips",
            "Automotive chip demand surges as EV production accelerates globally",
            "Apple increases chip orders sharply for next generation iPhone lineup",
            "Data center expansion creates record demand for high bandwidth memory",
            "5G rollout acceleration sparks shortage of RF semiconductor components",
            "Cloud hyperscalers dramatically increase custom ASIC chip orders",
            "Edge AI applications drive surge in demand for embedded processors",
            "Electric vehicle manufacturers triple semiconductor orders for 2026",
        ],
        "neutral": [
            "TSMC CEO delivers keynote at annual technology conference in San Francisco",
            "Semiconductor industry trade association releases annual membership report",
            "Intel and AMD announce new partnership for developer certification program",
            "Quarterly earnings season begins for major chip manufacturers",
            "Technology museum opens new exhibit on history of semiconductor industry",
            "University researchers publish paper on theoretical quantum computing",
            "Chip industry veterans gather for annual golf charity tournament",
            "New semiconductor engineering curriculum launched at MIT and Stanford",
        ],
    }

    rows = []
    for label, texts in samples.items():
        for text in texts:
            rows.append({"text": text, "label": label})

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(f"Generated {len(df)} synthetic training samples.")
    return df


# ─── Training Logic ───────────────────────────────────────────────────────────

def train(config: dict, data_path: str = None):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # 1. Load data
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading data from {data_path}")
        if data_path.endswith(".parquet"):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} rows.")
    else:
        df = generate_seed_data()

    # Validate columns
    assert "text" in df.columns and "label" in df.columns, \
        "Data must have 'text' and 'label' columns."

    # Encode labels
    df = df[df["label"].isin(LABEL2ID)].copy()
    df["label_id"] = df["label"].map(LABEL2ID)
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    # 2. Split
    train_df, val_df = train_test_split(
        df, test_size=config["val_split"],
        random_state=config["seed"], stratify=df["label_id"]
    )
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])

    train_dataset = NewsDataset(
        train_df["text"].tolist(), train_df["label_id"].tolist(),
        tokenizer, config["max_length"]
    )
    val_dataset = NewsDataset(
        val_df["text"].tolist(), val_df["label_id"].tolist(),
        tokenizer, config["max_length"]
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config["batch_size"])

    # 4. Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(
        config["base_model"],
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    # 5. Optimiser + scheduler
    total_steps   = len(train_loader) * config["epochs"]
    warmup_steps  = int(total_steps * config["warmup_ratio"])

    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 6. Training loop
    best_val_f1   = 0.0
    best_model_dir = None

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            batch     = {k: v.to(device) for k, v in batch.items()}
            outputs   = model(**batch)
            loss      = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if step % 10 == 0:
                logger.info(f"Epoch {epoch+1} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        # 7. Validation
        val_f1, report = evaluate_model(model, val_loader, device)
        logger.info(f"\nEpoch {epoch+1} | Avg Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}")
        logger.info(f"\n{report}")

        # 8. Save best model
        if val_f1 > best_val_f1:
            best_val_f1   = val_f1
            version       = f"v{epoch+1}_f1{val_f1:.2f}"
            best_model_dir = os.path.join(config["save_dir"], version)
            os.makedirs(best_model_dir, exist_ok=True)

            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)

            # Save config + metrics
            meta = {
                "version":    version,
                "val_f1":     round(best_val_f1, 4),
                "train_size": len(train_df),
                "val_size":   len(val_df),
                "config":     config,
            }
            with open(os.path.join(best_model_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            logger.info(f"✅ New best model saved to: {best_model_dir}")

    logger.info(f"\n🎉 Training complete. Best Val F1: {best_val_f1:.4f}")
    logger.info(f"Best model: {best_model_dir}")
    return best_model_dir


def evaluate_model(model, val_loader, device) -> tuple:
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch     = {k: v.to(device) for k, v in batch.items()}
            outputs   = model(**batch)
            preds     = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    label_names = [ID2LABEL[i] for i in range(len(ID2LABEL))]
    report      = classification_report(all_labels, all_preds, target_names=label_names, zero_division=0)
    macro_f1    = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return macro_f1, report


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train semiconductor news classifier")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to labeled CSV or Parquet file")
    parser.add_argument("--epochs",    type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size",type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",        type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--save_dir",  type=str, default=DEFAULT_CONFIG["save_dir"])
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update({
        "epochs":     args.epochs,
        "batch_size": args.batch_size,
        "lr":         args.lr,
        "save_dir":   args.save_dir,
    })

    train(config, data_path=args.data_path)