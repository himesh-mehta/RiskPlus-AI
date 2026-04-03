"""
evaluate.py
-----------
Evaluation and benchmarking for both ML models:
  1. NewsClassifier   → F1, Precision, Recall per label + confusion matrix
  2. RiskForecaster   → MAE, RMSE, Disruption Detection Rate (DDR)

Disruption Detection Rate (DDR):
    Did the model flag HIGH risk (>65) within 7 days BEFORE a known disruption?
    This is the most business-critical metric — a low F1 matters less than
    catching real disruptions before they hit.

How to run:
    # Evaluate both models
    python training/evaluate.py --mode both

    # Evaluate only the news classifier
    python training/evaluate.py --mode news --model_path registry/news_classifier/v3_f1_0.89

    # Evaluate only the risk forecaster
    python training/evaluate.py --mode risk --model_path registry/risk_forecaster/risk_forecaster_v50.pt
"""

import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─── News Classifier Evaluation ───────────────────────────────────────────────

def evaluate_news_classifier(model_path: str = None, test_data_path: str = None):
    """
    Evaluate the news classifier on a labeled test set.
    Falls back to built-in test samples if no test file provided.
    """
    from models.news_classifier import NewsClassifier, LABEL2ID, ID2LABEL
    from sklearn.metrics import (
        classification_report, confusion_matrix, f1_score
    )

    logger.info("=" * 55)
    logger.info("  EVALUATING: News Classifier")
    logger.info("=" * 55)

    clf = NewsClassifier(model_path=model_path)

    # Load test data
    if test_data_path and os.path.exists(test_data_path):
        df = pd.read_csv(test_data_path)
    else:
        logger.warning("No test data provided — using built-in test samples.")
        df = _get_builtin_news_test_data()

    df = df[df["label"].isin(LABEL2ID)].copy()
    texts  = df["text"].tolist()
    labels = df["label"].tolist()

    # Predict
    preds       = clf.predict_batch(texts)
    pred_labels = [p["label"] for p in preds]

    # Metrics
    label_names = list(LABEL2ID.keys())
    macro_f1    = f1_score(labels, pred_labels, average="macro",
                           labels=label_names, zero_division=0)
    report      = classification_report(
        labels, pred_labels, labels=label_names, zero_division=0
    )
    cm          = confusion_matrix(labels, pred_labels, labels=label_names)

    print(f"\n{'─'*55}")
    print(f"  Macro F1 Score: {macro_f1:.4f}")
    print(f"{'─'*55}")
    print(report)
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(pd.DataFrame(cm, index=label_names, columns=label_names).to_string())

    results = {
        "macro_f1":   round(macro_f1, 4),
        "report":     report,
        "n_samples":  len(df),
        "evaluated_at": datetime.utcnow().isoformat(),
    }

    # Save evaluation results
    os.makedirs("registry/evaluations", exist_ok=True)
    out_path = "registry/evaluations/news_classifier_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    return results


def _get_builtin_news_test_data() -> pd.DataFrame:
    """Built-in test headlines not used in training."""
    return pd.DataFrame([
        {"text": "ASML reports earthquake damage at Veldhoven facility halts production", "label": "fab_shutdown"},
        {"text": "Critical argon gas shortage threatens chip fabs in South Korea",          "label": "raw_material_shortage"},
        {"text": "Major storm disrupts shipping lanes between Japan and US West Coast",      "label": "logistics_delay"},
        {"text": "Commerce Dept adds 14 Chinese chip companies to export control list",     "label": "export_restriction"},
        {"text": "Microsoft Azure capacity expansion drives GPU chip orders to record high", "label": "demand_surge"},
        {"text": "SIA hosts annual semiconductor policy forum in Washington DC",             "label": "neutral"},
        {"text": "Taiwan Strait exercises force rerouting of chip cargo freighters",        "label": "logistics_delay"},
        {"text": "Hydrogen fluoride ban in South Korea threatens photoresist supply chain", "label": "raw_material_shortage"},
        {"text": "Intel announces Fab 52 in Arizona is operational ahead of schedule",      "label": "neutral"},
        {"text": "US tightens HBM memory export rules targeting Chinese AI applications",   "label": "export_restriction"},
    ])


# ─── Risk Forecaster Evaluation ───────────────────────────────────────────────

def evaluate_risk_forecaster(model_path: str = None, data_path: str = None):
    """
    Evaluate the risk forecaster using MAE, RMSE, and Disruption Detection Rate.
    """
    from models.risk_forecaster import RiskForecaster, FEATURE_COLUMNS
    from training.train_risk import (
        generate_synthetic_feature_store, engineer_risk_score, create_windows
    )

    logger.info("=" * 55)
    logger.info("  EVALUATING: Risk Forecaster")
    logger.info("=" * 55)

    # Load data
    if data_path and os.path.exists(data_path):
        df = pd.read_parquet(data_path) if data_path.endswith(".parquet") \
             else pd.read_csv(data_path, parse_dates=["date"], index_col="date")
    else:
        df = generate_synthetic_feature_store(n_days=500)

    if "disruption_risk_score" not in df.columns:
        df = engineer_risk_score(df)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    df = df.sort_index().fillna(method="ffill").fillna(0)

    # Use last 20% as holdout test set
    n        = len(df)
    test_df  = df.iloc[int(n * 0.8):]

    forecaster = RiskForecaster(model_path=model_path, horizon=7)

    # Roll through test set with step=1
    errors, ddr_correct, ddr_total = [], 0, 0
    lookback = 30

    all_actuals, all_preds = [], []

    for i in range(lookback, len(test_df) - 7):
        window = test_df.iloc[i - lookback : i]
        actual = test_df["disruption_risk_score"].iloc[i : i + 7].values

        result    = forecaster.predict(window[FEATURE_COLUMNS], horizon=7)
        predicted = np.array(result["risk_scores"])

        mae = np.abs(predicted - actual).mean()
        errors.append(mae)

        all_actuals.extend(actual.tolist())
        all_preds.extend(predicted.tolist())

        # Disruption Detection Rate
        # "Disruption" = actual peak risk > 65 in next 7 days
        if actual.max() > 65:
            ddr_total += 1
            # Did model predict > 55 at least once in the window? (early warning)
            if predicted.max() > 55:
                ddr_correct += 1

    # Metrics
    all_actuals = np.array(all_actuals)
    all_preds   = np.array(all_preds)

    mae  = np.mean(errors)
    rmse = np.sqrt(np.mean((all_actuals - all_preds) ** 2))
    ddr  = (ddr_correct / ddr_total * 100) if ddr_total > 0 else 0.0

    print(f"\n{'─'*55}")
    print(f"  MAE  (risk points) : {mae:.2f}")
    print(f"  RMSE (risk points) : {rmse:.2f}")
    print(f"  Disruption Detection Rate: {ddr:.1f}%  ({ddr_correct}/{ddr_total} events caught)")
    print(f"{'─'*55}")

    results = {
        "mae":  round(mae, 2),
        "rmse": round(rmse, 2),
        "disruption_detection_rate": round(ddr, 2),
        "disruption_events_caught": ddr_correct,
        "disruption_events_total":  ddr_total,
        "test_samples":             len(errors),
        "evaluated_at":             datetime.utcnow().isoformat(),
    }

    os.makedirs("registry/evaluations", exist_ok=True)
    out_path = "registry/evaluations/risk_forecaster_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    return results


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ML models")
    parser.add_argument("--mode",       type=str, default="both",
                        choices=["both", "news", "risk"],
                        help="Which model to evaluate")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to saved model weights/directory")
    parser.add_argument("--data_path",  type=str, default=None,
                        help="Path to test data (CSV or Parquet)")
    args = parser.parse_args()

    if args.mode in ("both", "news"):
        evaluate_news_classifier(
            model_path=args.model_path if args.mode == "news" else None,
            test_data_path=args.data_path,
        )

    if args.mode in ("both", "risk"):
        evaluate_risk_forecaster(
            model_path=args.model_path if args.mode == "risk" else None,
            data_path=args.data_path,
        )