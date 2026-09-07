"""Explainability module: Gradient-based feature attribution for edge anomaly alerts.

Uses gradient magnitude as a proxy for SHAP values — no external SHAP library needed.
Every alert raised by the system includes a ranked feature attribution so fire
authorities understand exactly WHY the model flagged a sensor node.

Example output:
    Zone 7 alert — top drivers: humidity (42%), temperature (31%), pm25 (18%)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Optional

# Canonical sensor feature names — must match data/generator.py column order
FEATURE_NAMES = [
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
    "co2",
    "pm25",
    "soil_moisture",
]


class EdgeModelExplainer:
    """Gradient-based feature attribution for the LSTM Autoencoder.

    Computes how much each input feature contributed to the reconstruction
    error (anomaly score) for a given sensor window. Higher attribution =
    that feature drove the anomaly alert.

    Works by computing the gradient of the reconstruction loss with respect
    to the input tensor — features with larger gradients had more influence
    on the model's output. This is a well-established proxy for feature
    importance in sequence models.

    Args:
        model:   Trained LSTMAutoencoder in eval mode.
        device:  Torch device string — 'cpu' or 'cuda'.
    """

    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.feature_names = FEATURE_NAMES

    def _compute_gradients(self, windows: torch.Tensor) -> np.ndarray:
        """Compute input gradients for a batch of windows.

        Args:
            windows: Tensor of shape (batch, seq_len, input_dim).

        Returns:
            Gradient magnitudes of shape (batch, input_dim) — averaged over seq_len.
        """
        x = windows.to(self.device).float()
        x.requires_grad_(True)

        # Forward pass — compute reconstruction loss per sample
        reconstruction = self.model(x)
        # Per-sample MSE: shape (batch,)
        loss_per_sample = ((reconstruction - x) ** 2).mean(dim=(1, 2))
        # Sum losses so we get one gradient pass for the whole batch
        loss_per_sample.sum().backward()

        # Gradient magnitude: (batch, seq_len, input_dim) → average over time
        grad_magnitude = x.grad.abs().mean(dim=1)  # (batch, input_dim)
        return grad_magnitude.detach().cpu().numpy()

    def explain_timeseries_history(
        self,
        windows: torch.Tensor,
    ) -> pd.DataFrame:
        """Compute per-feature relative contribution for each window.

        This is the main method used by the dashboard Tab 4 (SHAP Explainability).
        Returns proportions (sum to 1.0 per row) so values are directly
        interpretable as percentage contributions.

        Args:
            windows: Sliding-window tensor of shape (num_windows, seq_len, input_dim).
                     input_dim must be 7 (matching FEATURE_NAMES).

        Returns:
            DataFrame of shape (num_windows, 7) with columns matching FEATURE_NAMES.
            Each row sums to 1.0 — values represent relative feature importance.
        """
        assert windows.shape[-1] == len(self.feature_names), (
            f"Expected {len(self.feature_names)} features, got {windows.shape[-1]}. "
            f"Feature order must be: {self.feature_names}"
        )

        all_attributions = []
        batch_size = 32

        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size]
            grads = self._compute_gradients(batch)  # (batch, input_dim)

            # Normalize to proportions so each row sums to 1.0
            row_sums = grads.sum(axis=1, keepdims=True)
            # Avoid division by zero for zero-gradient edge case
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            proportions = grads / row_sums

            all_attributions.append(proportions)

        attributions = np.concatenate(all_attributions, axis=0)
        return pd.DataFrame(attributions, columns=self.feature_names)

    def explain_single_alert(
        self,
        window: torch.Tensor,
        zone_id: Optional[str] = None,
        anomaly_score: Optional[float] = None,
    ) -> dict:
        """Explain a single alert in human-readable format for fire authorities.

        Args:
            window:        Single window tensor of shape (seq_len, input_dim).
            zone_id:       Optional zone identifier string e.g. 'Zone-7-Alpine'.
            anomaly_score: Optional reconstruction error score for context.

        Returns:
            Dict with keys: zone_id, anomaly_score, top_features, attribution,
            narrative — a plain-English explanation of what drove the alert.
        """
        # Add batch dimension
        batch = window.unsqueeze(0)
        df = self.explain_timeseries_history(batch)
        attribution = df.iloc[0].to_dict()

        # Rank features by contribution
        ranked = sorted(attribution.items(), key=lambda x: x[1], reverse=True)
        top_features = [(feat, round(score * 100, 1)) for feat, score in ranked[:3]]

        # Build plain-English narrative
        narrative = _build_narrative(top_features, zone_id, anomaly_score)

        return {
            "zone_id": zone_id or "Unknown",
            "anomaly_score": round(anomaly_score, 4) if anomaly_score else None,
            "top_features": top_features,
            "attribution": {k: round(v * 100, 2) for k, v in attribution.items()},
            "narrative": narrative,
        }

    def get_feature_importance_summary(
        self,
        windows: torch.Tensor,
        top_k: int = 3,
    ) -> pd.DataFrame:
        """Aggregate feature importance across all windows for a zone.

        Useful for understanding which features are most consistently
        important across an entire monitoring session — not just one alert.

        Args:
            windows: All windows for a zone (num_windows, seq_len, input_dim).
            top_k:   Number of top features to highlight.

        Returns:
            DataFrame with mean, std, and rank of each feature's contribution.
        """
        df = self.explain_timeseries_history(windows)

        summary = pd.DataFrame({
            "feature": self.feature_names,
            "mean_contribution_%": (df.mean() * 100).round(2).values,
            "std_%": (df.std() * 100).round(2).values,
            "rank": df.mean().rank(ascending=False).astype(int).values,
        }).sort_values("rank").reset_index(drop=True)

        summary["is_top_driver"] = summary["rank"] <= top_k
        return summary


def _build_narrative(
    top_features: List[tuple],
    zone_id: Optional[str],
    anomaly_score: Optional[float],
) -> str:
    """Build a plain-English alert explanation for fire authorities.

    Args:
        top_features:  List of (feature_name, percentage) tuples, ranked.
        zone_id:       Zone identifier string.
        anomaly_score: Reconstruction error score.

    Returns:
        Human-readable explanation string.
    """
    feature_descriptions = {
        "temperature":    "temperature spike",
        "humidity":       "humidity drop",
        "wind_speed":     "wind speed anomaly",
        "wind_direction": "wind direction shift",
        "co2":            "CO₂ concentration surge",
        "pm25":           "PM2.5 particulate surge",
        "soil_moisture":  "soil moisture deficit",
    }

    zone_str = f"in {zone_id}" if zone_id else ""
    score_str = f" (anomaly score: {anomaly_score:.3f})" if anomaly_score else ""

    drivers = ", ".join(
        f"{feature_descriptions.get(f, f)} ({pct}%)"
        for f, pct in top_features
    )

    return (
        f"Alert raised {zone_str}{score_str}. "
        f"Primary drivers: {drivers}. "
        f"Immediate sensor inspection recommended."
    )


def build_explainer(model: nn.Module, device: str = "cpu") -> EdgeModelExplainer:
    """Convenience factory used by dashboard and training scripts.

    Args:
        model:  Trained LSTMAutoencoder.
        device: 'cpu' or 'cuda'.

    Returns:
        Ready-to-use EdgeModelExplainer instance.
    """
    return EdgeModelExplainer(model=model, device=device)
