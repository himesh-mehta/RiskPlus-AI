"""
news_classifier.py
------------------
NLP model that classifies news headlines/articles into semiconductor
supply chain disruption categories.

Labels:
    - fab_shutdown          : Factory / fabrication plant closure
    - raw_material_shortage : Silicon, rare earth, chemical shortages
    - logistics_delay       : Shipping, port, freight disruptions
    - export_restriction    : Trade bans, tariffs, sanctions
    - demand_surge          : Sudden spike in chip demand
    - neutral               : Not supply-chain relevant

Usage:
    classifier = NewsClassifier()
    result = classifier.predict("TSMC halts production in Taiwan fab")
    # {"label": "fab_shutdown", "confidence": 0.94, ...}
"""

import os
import json
import logging
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Label Definitions ────────────────────────────────────────────────────────

LABEL2ID = {
    "fab_shutdown": 0,
    "raw_material_shortage": 1,
    "logistics_delay": 2,
    "export_restriction": 3,
    "demand_surge": 4,
    "neutral": 5,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

LABEL_DESCRIPTIONS = {
    "fab_shutdown":           "Manufacturing facility shutdown or production halt",
    "raw_material_shortage":  "Shortage of raw materials (silicon, rare earths, chemicals)",
    "logistics_delay":        "Shipping, port congestion, or freight disruption",
    "export_restriction":     "Export ban, tariff, sanction, or trade restriction",
    "demand_surge":           "Sudden spike in chip demand",
    "neutral":                "No supply chain relevance detected",
}

# Labels considered high-severity for alerting
HIGH_SEVERITY_LABELS = {"fab_shutdown", "export_restriction", "raw_material_shortage"}


# ─── Model Class ──────────────────────────────────────────────────────────────

class NewsClassifier:
    """
    Wraps a fine-tuned DistilBERT model for semiconductor news classification.

    Falls back to zero-shot classification (facebook/bart-large-mnli) if no
    fine-tuned model is available — so the API works even before training.
    """

    BASE_MODEL      = "distilbert-base-uncased"
    ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
    MAX_LENGTH      = 512

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Path to fine-tuned model directory saved in registry/.
                        Pass None to use zero-shot fallback during development.
        """
        self.device     = 0 if torch.cuda.is_available() else -1
        self.model_path = model_path
        self._load_model()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _load_model(self):
        if self.model_path and os.path.exists(self.model_path):
            logger.info(f"Loading fine-tuned model from: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model     = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=len(LABEL2ID),
                id2label=ID2LABEL,
                label2id=LABEL2ID,
            )
            self.pipe = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                top_k=None,
            )
            self.mode = "finetuned"
            logger.info("Fine-tuned model loaded.")
        else:
            logger.warning(
                "No fine-tuned model found — using zero-shot fallback.\n"
                "Train a proper model with: python training/train_news.py"
            )
            self.pipe = pipeline(
                "zero-shot-classification",
                model=self.ZERO_SHOT_MODEL,
                device=self.device,
            )
            self.mode = "zero_shot"

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Classify a single news headline or article snippet.

        Returns:
            {
                "label": "fab_shutdown",
                "confidence": 0.94,
                "description": "Manufacturing facility shutdown...",
                "severity": "high",
                "all_scores": {"fab_shutdown": 0.94, "neutral": 0.03, ...},
                "is_supply_chain_relevant": True
            }
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        text = text.strip()[:2000]  # safety cap

        if self.mode == "finetuned":
            return self._predict_finetuned(text)
        return self._predict_zero_shot(text)

    def predict_batch(self, texts: list) -> list:
        """Classify a list of news items. Returns list of prediction dicts."""
        return [self.predict(t) for t in texts]

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _predict_finetuned(self, text: str) -> dict:
        results    = self.pipe(text, truncation=True, max_length=self.MAX_LENGTH)
        all_scores = {r["label"]: round(r["score"], 4) for r in results[0]}
        top_label  = max(all_scores, key=all_scores.get)
        return self._format_output(top_label, all_scores[top_label], all_scores)

    def _predict_zero_shot(self, text: str) -> dict:
        candidate_labels = list(LABEL2ID.keys())
        result     = self.pipe(text, candidate_labels=candidate_labels)
        all_scores = {
            lbl: round(score, 4)
            for lbl, score in zip(result["labels"], result["scores"])
        }
        top_label  = result["labels"][0]
        return self._format_output(top_label, all_scores[top_label], all_scores)

    def _format_output(self, label: str, confidence: float, all_scores: dict) -> dict:
        severity = "high" if label in HIGH_SEVERITY_LABELS else \
                   "medium" if label != "neutral" else "low"
        return {
            "label":                    label,
            "confidence":               confidence,
            "description":              LABEL_DESCRIPTIONS.get(label, ""),
            "severity":                 severity,
            "all_scores":               all_scores,
            "is_supply_chain_relevant": label != "neutral",
        }

    def save_label_config(self, save_dir: str):
        """Save label mappings alongside model artifacts in registry/."""
        os.makedirs(save_dir, exist_ok=True)
        config = {
            "label2id":     LABEL2ID,
            "id2label":     ID2LABEL,
            "descriptions": LABEL_DESCRIPTIONS,
        }
        path = os.path.join(save_dir, "label_config.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Label config saved to {path}")


# ─── Smoke Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    clf = NewsClassifier()  # zero-shot until fine-tuned model exists

    test_headlines = [
        "TSMC suspends operations at Fab 18 following earthquake in Taiwan",
        "Samsung reports critical shortage of NAND flash raw materials",
        "Port of Rotterdam congestion causes 3-week delay in chip shipments",
        "US imposes new export controls on advanced semiconductor equipment to China",
        "AI boom drives unprecedented demand for Nvidia H100 GPUs",
        "Apple increases iPhone production forecast for Q4",
        "Local sports team wins regional championship",
    ]

    print("\n" + "=" * 65)
    print("  NEWS CLASSIFIER — SEMICONDUCTOR SUPPLY CHAIN")
    print("=" * 65)
    for headline in test_headlines:
        r    = clf.predict(headline)
        flag = "🚨" if r["severity"] == "high" else \
               "⚠️ " if r["severity"] == "medium" else "✅"
        print(f"\n{flag}  {headline[:70]}")
        print(f"    Label      : {r['label']}  ({r['confidence']:.0%} confidence)")
        print(f"    Severity   : {r['severity']}")
        print(f"    Description: {r['description']}")