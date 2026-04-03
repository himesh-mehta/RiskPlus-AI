"""
explainability.py
-----------------
SHAP-based explainability for the risk forecaster.

Converts raw model outputs into human-readable reasons like:
  "Risk is HIGH because:
    1. Taiwan port congestion has spiked to 78/100 (+18 risk pts)
    2. Export restrictions are currently active (+15 risk pts)
    3. Shipping rates are 34% above average (+12 risk pts)"

These explanations are:
  - Stored in every /predict/risk API response
  - Rendered by the frontend as readable cards (Member 4 consumes this)
  - Sent in alerts by the backend (Member 3 consumes this)

Usage:
    explainer = RiskExplainer(forecaster.model)
    explanation = explainer.explain(feature_df)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.risk_forecaster import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Feature Display Config ───────────────────────────────────────────────────

FEATURE_DISPLAY = {
    "shipping_rate_index":      "Shipping rate index",
    "port_congestion_score":    "Port congestion level",
    "news_event_count":         "Supply-chain news events",
    "export_restriction_flag":  "Active export restrictions",
    "pmi_manufacturing":        "Manufacturing PMI",
    "dram_spot_price_change":   "DRAM spot price change",
    "taiwan_risk_index":        "Taiwan Strait risk index",
}

FEATURE_DIRECTION = {
    # positive value = higher risk
    "shipping_rate_index":      "higher",
    "port_congestion_score":    "higher",
    "news_event_count":         "higher",
    "export_restriction_flag":  "active",
    "pmi_manufacturing":        "lower",    # PMI < 50 = higher risk
    "dram_spot_price_change":   "higher_abs",
    "taiwan_risk_index":        "higher",
}


# ─── SHAP Explainer ───────────────────────────────────────────────────────────

class RiskExplainer:
    """
    Produces SHAP-style feature importance explanations for risk predictions.

    Uses DeepExplainer for the LSTM model. Falls back to a fast
    gradient-based approximation if SHAP is not installed.
    """

    TOP_N = 3  # number of top factors to surface

    def __init__(self, model, feature_columns: list = None):
        """
        Args:
            model: Trained LSTMForecaster (torch.nn.Module)
            feature_columns: List of feature names (default: FEATURE_COLUMNS)
        """
        self.model    = model
        self.features = feature_columns or FEATURE_COLUMNS
        self.device   = next(model.parameters()).device
        self._init_shap()

    def _init_shap(self):
        try:
            import shap
            self.shap = shap
            self._shap_available = True
            logger.info("SHAP library loaded.")
        except ImportError:
            logger.warning(
                "SHAP not installed — using gradient approximation.\n"
                "Install with: pip install shap"
            )
            self._shap_available = False

    # ── Public API ────────────────────────────────────────────────────────────

    def explain(
        self,
        features: pd.DataFrame,
        prediction: Optional[dict] = None,
        background_size: int = 20,
    ) -> dict:
        """
        Generate an explanation for the current risk prediction.

        Args:
            features:        DataFrame with LOOKBACK_DAYS rows and FEATURE_COLUMNS.
            prediction:      Output from RiskForecaster.predict() (optional, for context).
            background_size: Number of background samples for DeepExplainer.

        Returns:
            {
                "top_factors": [
                    {
                        "feature": "taiwan_risk_index",
                        "display_name": "Taiwan Strait risk index",
                        "current_value": 58.3,
                        "impact_score": 18.4,
                        "direction": "higher",
                        "reason": "Taiwan Strait risk index is elevated at 58.3 (+18.4 risk pts)"
                    },
                    ...
                ],
                "summary": "Risk is elevated primarily due to high Taiwan Strait risk ...",
                "method": "shap_deep"
            }
        """
        if self._shap_available:
            return self._explain_shap(features, prediction, background_size)
        return self._explain_gradient(features, prediction)

    # ── SHAP-based Explanation ────────────────────────────────────────────────

    def _explain_shap(
        self, features: pd.DataFrame, prediction: Optional[dict], bg_size: int
    ) -> dict:
        from models.risk_forecaster import LSTMForecaster

        self.model.eval()

        # Prepare input tensor (1, lookback, n_features)
        arr    = features[self.features].values.astype(np.float32)[-30:]
        if len(arr) < 30:
            pad = np.zeros((30 - len(arr), arr.shape[1]), dtype=np.float32)
            arr = np.vstack([pad, arr])
        x_tensor = torch.tensor(arr).unsqueeze(0).to(self.device)

        # Background: random noise baseline (shape: bg_size, lookback, n_features)
        background = torch.zeros((bg_size, 30, len(self.features))).to(self.device)

        # DeepExplainer
        explainer   = self.shap.DeepExplainer(self.model, background)
        shap_values = explainer.shap_values(x_tensor)

        # shap_values shape: (1, lookback, n_features) or list
        if isinstance(shap_values, list):
            sv = np.array(shap_values[0])
        else:
            sv = np.array(shap_values)

        # Average SHAP impact across the time dimension
        sv = sv.squeeze(0)           # (lookback, n_features)
        feature_impact = sv.mean(axis=0) * 100   # scale to risk points

        return self._build_output(feature_impact, features, method="shap_deep")

    # ── Gradient-based Approximation (no SHAP needed) ────────────────────────

    def _explain_gradient(
        self, features: pd.DataFrame, prediction: Optional[dict]
    ) -> dict:
        """
        Integrated gradients approximation — fast and dependency-free.
        Less accurate than SHAP but good enough for production explanations.
        """
        self.model.eval()

        arr = features[self.features].values.astype(np.float32)[-30:]
        if len(arr) < 30:
            pad = np.zeros((30 - len(arr), arr.shape[1]), dtype=np.float32)
            arr = np.vstack([pad, arr])

        x_tensor   = torch.tensor(arr).unsqueeze(0).to(self.device).requires_grad_(True)
        baseline   = torch.zeros_like(x_tensor)

        # Integrated gradients: average gradient over 50 alpha steps
        steps = 50
        grads = []
        for k in range(1, steps + 1):
            alpha      = k / steps
            interp     = baseline + alpha * (x_tensor - baseline)
            interp     = interp.detach().requires_grad_(True)
            output     = self.model(interp).sum()
            output.backward()
            grads.append(interp.grad.detach().cpu().numpy())

        avg_grad     = np.mean(grads, axis=0).squeeze(0)     # (lookback, n_features)
        delta        = (x_tensor - baseline).detach().cpu().numpy().squeeze(0)
        ig           = avg_grad * delta                        # (lookback, n_features)
        feature_impact = ig.mean(axis=0) * 100                # average over time

        return self._build_output(feature_impact, features, method="integrated_gradients")

    # ── Output Builder ────────────────────────────────────────────────────────

    def _build_output(
        self,
        feature_impact: np.ndarray,
        features: pd.DataFrame,
        method: str,
    ) -> dict:
        """Convert raw impact array → structured explanation dict."""
        current_values = features[self.features].iloc[-1].to_dict()

        # Sort by absolute impact, take top N
        ranked = sorted(
            zip(self.features, feature_impact),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:self.TOP_N]

        top_factors = []
        for feat, impact in ranked:
            display = FEATURE_DISPLAY.get(feat, feat)
            value   = current_values.get(feat, 0)
            sign    = "+" if impact > 0 else ""
            reason  = self._generate_reason(feat, value, impact)

            top_factors.append({
                "feature":      feat,
                "display_name": display,
                "current_value": round(float(value), 2),
                "impact_score":  round(float(impact), 2),
                "direction":     FEATURE_DIRECTION.get(feat, "higher"),
                "reason":        reason,
            })

        summary = self._generate_summary(top_factors)

        return {
            "top_factors": top_factors,
            "summary":     summary,
            "method":      method,
        }

    def _generate_reason(self, feature: str, value: float, impact: float) -> str:
        """Generate a single human-readable reason string."""
        display = FEATURE_DISPLAY.get(feature, feature)
        impact_str = f"{impact:+.1f} risk pts"

        if feature == "export_restriction_flag":
            state = "active" if value >= 0.5 else "inactive"
            return f"Export restrictions are currently {state} ({impact_str})"

        if feature == "pmi_manufacturing":
            zone = "contraction" if value < 50 else "expansion"
            return f"{display} is at {value:.1f} ({zone} territory, {impact_str})"

        if feature == "dram_spot_price_change":
            direction = "up" if value > 0 else "down"
            return f"DRAM spot price moved {direction} {abs(value):.1f}% this week ({impact_str})"

        return f"{display} is at {value:.1f} ({impact_str})"

    def _generate_summary(self, top_factors: list) -> str:
        """Generate a 1-sentence summary of the top driver."""
        if not top_factors:
            return "Insufficient data to generate explanation."
        top = top_factors[0]
        return (
            f"Risk is primarily driven by {top['display_name'].lower()} "
            f"({top['impact_score']:+.1f} pts), "
            f"followed by {top_factors[1]['display_name'].lower() if len(top_factors) > 1 else 'other factors'}."
        )


# ─── Smoke Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from models.risk_forecaster import LSTMForecaster, FEATURE_COLUMNS

    np.random.seed(0)
    dummy_features = pd.DataFrame(
        np.random.randn(30, len(FEATURE_COLUMNS)) * [200, 20, 5, 0.3, 3, 2, 15] +
                                                     [1000, 55, 7, 0.3, 50, 0, 35],
        columns=FEATURE_COLUMNS,
    )
    # Simulate a stressed scenario
    dummy_features.iloc[-5:, FEATURE_COLUMNS.index("taiwan_risk_index")]  = 78
    dummy_features.iloc[-5:, FEATURE_COLUMNS.index("export_restriction_flag")] = 1

    model    = LSTMForecaster()
    explainer = RiskExplainer(model)
    result   = explainer.explain(dummy_features)

    print("\n" + "=" * 60)
    print("  EXPLAINABILITY — SEMICONDUCTOR RISK FORECAST")
    print("=" * 60)
    print(f"\nSummary: {result['summary']}")
    print(f"Method : {result['method']}\n")
    for i, f in enumerate(result["top_factors"], 1):
        print(f"  {i}. {f['reason']}")