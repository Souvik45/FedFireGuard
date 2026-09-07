"""Spatially Adaptive Differential Privacy Scheduler for FedFireGuard.

Standard differential privacy applies a uniform privacy budget (epsilon) to
all nodes equally. This is wasteful — an open hillside node far from any
protected land doesn't need the same strict privacy guarantee as a node
sitting on the boundary of an indigenous forest reserve.

Spatially Adaptive DP fixes this by:
  1. Assigning each node a sensitivity score based on proximity to
     protected/sensitive land boundaries (GDPR / India DPDP Act 2023)
  2. Allocating tighter epsilon (stronger privacy) to high-sensitivity nodes
  3. Allowing relaxed epsilon (better utility) to low-sensitivity nodes
  4. Dynamically adjusting budgets each round based on GNN fire spread
     probability (higher fire risk = relax utility, maintain privacy)

This directly addresses the Privacy & Compliance gap identified in the
FedFireGuard architecture and is compliant with:
  - India Digital Personal Data Protection Act 2023 (DPDP)
  - EU General Data Protection Regulation (GDPR Article 25)

Reference: Dwork & Roth (2014) — The Algorithmic Foundations of
           Differential Privacy. Foundations and Trends in TCS.
           Our extension: spatially adaptive epsilon allocation.
"""

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Node sensitivity classification
# ---------------------------------------------------------------------------

@dataclass
class NodePrivacyProfile:
    """Privacy profile for a single IoT sensor node.

    Attributes:
        node_id:              Unique node identifier.
        zone_id:              Ecological zone membership.
        sensitivity_score:    Float in [0, 1]. 1.0 = maximum sensitivity
                              (e.g. node on indigenous land boundary).
        land_type:            Human-readable land classification.
        assigned_epsilon:     Current DP epsilon budget for this node.
        cumulative_epsilon:   Total privacy budget consumed so far.
        max_epsilon_budget:   Hard cap — node stops contributing if exceeded.
    """
    node_id: int
    zone_id: int
    sensitivity_score: float          # 0.0 (open land) → 1.0 (protected boundary)
    land_type: str = "open"           # 'protected', 'indigenous', 'conservation', 'open'
    assigned_epsilon: float = 1.0
    cumulative_epsilon: float = 0.0
    max_epsilon_budget: float = 10.0  # Lifetime privacy budget cap
    history: List[float] = field(default_factory=list)

    def is_budget_exhausted(self) -> bool:
        """True if this node has consumed its lifetime privacy budget."""
        return self.cumulative_epsilon >= self.max_epsilon_budget

    def consume(self, epsilon: float) -> None:
        """Record epsilon consumption for this round."""
        self.cumulative_epsilon += epsilon
        self.history.append(epsilon)


# ---------------------------------------------------------------------------
# Epsilon allocation strategies
# ---------------------------------------------------------------------------

def sensitivity_to_epsilon(
    sensitivity_score: float,
    epsilon_min: float = 0.1,
    epsilon_max: float = 5.0,
) -> float:
    """Map sensitivity score to epsilon using inverse linear scaling.

    High sensitivity (score → 1.0) → tight epsilon (→ epsilon_min)
    Low sensitivity  (score → 0.0) → relaxed epsilon (→ epsilon_max)

    Args:
        sensitivity_score: Float in [0, 1].
        epsilon_min:       Minimum epsilon for maximum-sensitivity nodes.
        epsilon_max:       Maximum epsilon for minimum-sensitivity nodes.

    Returns:
        Assigned epsilon value for this node.
    """
    sensitivity_score = float(np.clip(sensitivity_score, 0.0, 1.0))
    # Inverse linear: high sensitivity → low epsilon
    epsilon = epsilon_max - sensitivity_score * (epsilon_max - epsilon_min)
    return round(float(np.clip(epsilon, epsilon_min, epsilon_max)), 4)


def compute_sensitivity_from_distance(
    distance_to_protected_km: float,
    threshold_km: float = 5.0,
) -> float:
    """Compute sensitivity score from physical distance to protected land.

    Nodes within threshold_km of a protected boundary get high sensitivity.
    Nodes beyond threshold_km get linearly decreasing sensitivity.

    Args:
        distance_to_protected_km: Distance from node to nearest protected boundary.
        threshold_km:             Distance within which max sensitivity applies.

    Returns:
        Sensitivity score in [0, 1].
    """
    if distance_to_protected_km <= 0:
        return 1.0
    if distance_to_protected_km >= threshold_km * 3:
        return 0.0
    # Exponential decay with distance
    score = np.exp(-distance_to_protected_km / threshold_km)
    return round(float(np.clip(score, 0.0, 1.0)), 4)


# ---------------------------------------------------------------------------
# Main scheduler
# ---------------------------------------------------------------------------

class SpatiallyAdaptiveDPScheduler:
    """Per-round epsilon scheduler for spatially adaptive differential privacy.

    Manages the privacy budget for all nodes across FL training rounds.
    Each round it:
      1. Computes base epsilon from each node's sensitivity score
      2. Adjusts for fire spread probability (GNN output) if available
      3. Checks lifetime budget caps
      4. Returns per-node epsilon assignments for Opacus DP-SGD

    Args:
        node_profiles:    List of NodePrivacyProfile for all nodes.
        epsilon_min:      Global minimum epsilon (hardest privacy guarantee).
        epsilon_max:      Global maximum epsilon (most relaxed).
        noise_multiplier: Base Gaussian noise multiplier for DP-SGD.
        max_grad_norm:    Gradient clipping norm for Opacus.
    """

    def __init__(
        self,
        node_profiles: List[NodePrivacyProfile],
        epsilon_min: float = 0.1,
        epsilon_max: float = 5.0,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.profiles: Dict[int, NodePrivacyProfile] = {
            p.node_id: p for p in node_profiles
        }
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.round_num = 0
        self.round_log: List[dict] = []

    def step(
        self,
        fire_spread_probs: Optional[Dict[int, float]] = None,
    ) -> Dict[int, float]:
        """Compute per-node epsilon assignments for the current round.

        Args:
            fire_spread_probs: Optional {node_id: probability} from GNN.
                               High fire probability relaxes epsilon slightly
                               to improve detection utility in crisis zones.

        Returns:
            {node_id: epsilon} for all active (non-exhausted) nodes.
        """
        self.round_num += 1
        assignments: Dict[int, float] = {}
        round_record = {"round": self.round_num, "assignments": {}}

        for node_id, profile in self.profiles.items():
            if profile.is_budget_exhausted():
                assignments[node_id] = None  # Node excluded this round
                continue

            # Base epsilon from spatial sensitivity
            base_eps = sensitivity_to_epsilon(
                profile.sensitivity_score,
                self.epsilon_min,
                self.epsilon_max,
            )

            # Fire-adaptive adjustment: high fire risk → relax epsilon slightly
            # so detection utility is maintained during active fire events
            if fire_spread_probs and node_id in fire_spread_probs:
                fire_prob = float(np.clip(fire_spread_probs[node_id], 0.0, 1.0))
                # Max relaxation: +20% epsilon during high fire probability
                fire_adjustment = 1.0 + 0.2 * fire_prob
                adjusted_eps = base_eps * fire_adjustment
            else:
                adjusted_eps = base_eps

            # Enforce hard cap: don't exceed remaining lifetime budget
            remaining = profile.max_epsilon_budget - profile.cumulative_epsilon
            final_eps = float(min(adjusted_eps, remaining, self.epsilon_max))
            final_eps = round(final_eps, 4)

            # Record consumption
            profile.consume(final_eps)
            assignments[node_id] = final_eps

            round_record["assignments"][node_id] = {
                "epsilon": final_eps,
                "sensitivity": profile.sensitivity_score,
                "land_type": profile.land_type,
                "cumulative_epsilon": round(profile.cumulative_epsilon, 4),
                "budget_remaining": round(remaining - final_eps, 4),
            }

        self.round_log.append(round_record)
        return assignments

    def get_noise_multiplier(self, epsilon: float, delta: float = 1e-5) -> float:
        """Approximate noise multiplier for a target epsilon using Gaussian mechanism.

        For Opacus DP-SGD: higher epsilon → lower noise → better utility.
        This is an approximation; Opacus computes exact accountant internally.

        Args:
            epsilon: Target privacy budget.
            delta:   Privacy failure probability (typically 1/n where n=dataset size).

        Returns:
            Approximate Gaussian noise multiplier.
        """
        # Gaussian mechanism: sigma ≈ sqrt(2 * ln(1.25/delta)) / epsilon
        sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        return round(float(sigma), 4)

    def get_opacus_config(
        self,
        node_id: int,
        delta: float = 1e-5,
    ) -> dict:
        """Get Opacus DP-SGD configuration for a specific node this round.

        Plug these values directly into Opacus make_private():
            privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=loader,
                noise_multiplier=config['noise_multiplier'],
                max_grad_norm=config['max_grad_norm'],
            )

        Args:
            node_id: Node to get config for.
            delta:   Privacy failure probability.

        Returns:
            Dict with noise_multiplier, max_grad_norm, target_epsilon, delta.
        """
        profile = self.profiles.get(node_id)
        if profile is None:
            raise ValueError(f"Node {node_id} not found in scheduler.")

        eps = profile.assigned_epsilon if profile.history else self.epsilon_max
        sigma = self.get_noise_multiplier(eps, delta)

        return {
            "noise_multiplier": sigma,
            "max_grad_norm": self.max_grad_norm,
            "target_epsilon": eps,
            "delta": delta,
        }

    def get_privacy_summary(self) -> List[dict]:
        """Summary of all nodes' privacy status for logging and dashboard."""
        return [
            {
                "node_id": p.node_id,
                "zone_id": p.zone_id,
                "land_type": p.land_type,
                "sensitivity_score": p.sensitivity_score,
                "current_epsilon": p.history[-1] if p.history else None,
                "cumulative_epsilon": round(p.cumulative_epsilon, 4),
                "budget_remaining": round(
                    p.max_epsilon_budget - p.cumulative_epsilon, 4
                ),
                "budget_exhausted": p.is_budget_exhausted(),
                "rounds_participated": len(p.history),
            }
            for p in self.profiles.values()
        ]

    def save_privacy_utility_log(
        self,
        f1_score: float,
        log_path: str = "logs/privacy_utility_curve.csv",
    ) -> None:
        """Save privacy-utility data point for dashboard Tab 3 curve.

        Args:
            f1_score: Detection F1 score achieved this round.
            log_path: Path to CSV log file.
        """
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_exists = os.path.exists(log_path)

        # Mean epsilon across active nodes this round
        active_eps = [
            p.history[-1] for p in self.profiles.values()
            if p.history and not p.is_budget_exhausted()
        ]
        mean_eps = round(float(np.mean(active_eps)), 4) if active_eps else 0.0

        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["round", "target_epsilon", "anomaly_f1_score"]
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "round": self.round_num,
                "target_epsilon": mean_eps,
                "anomaly_f1_score": round(f1_score, 4),
            })


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_scheduler_from_metadata(
    node_metadata: dict,
    epsilon_min: float = 0.1,
    epsilon_max: float = 5.0,
    protected_zone_ids: Optional[List[int]] = None,
) -> "SpatiallyAdaptiveDPScheduler":
    """Build scheduler directly from WildfireDataGenerator node metadata.

    Args:
        node_metadata:       Output of WildfireDataGenerator.node_metadata.
        epsilon_min:         Minimum epsilon for protected nodes.
        epsilon_max:         Maximum epsilon for open nodes.
        protected_zone_ids:  Zone IDs considered environmentally sensitive.
                             If None, zone 0 is treated as protected by default.

    Returns:
        Configured SpatiallyAdaptiveDPScheduler ready for FL training.
    """
    if protected_zone_ids is None:
        protected_zone_ids = [0]

    profiles = []
    for node_id, meta in node_metadata.items():
        zone_id = meta.get("zone_id", 0)
        x_km = meta.get("x_km", 0.0)
        y_km = meta.get("y_km", 0.0)

        # Nodes in protected zones get high sensitivity
        if zone_id in protected_zone_ids:
            sensitivity = 0.9
            land_type = "protected"
        else:
            # Distance-based sensitivity for non-protected zones
            dist = np.sqrt(x_km**2 + y_km**2)
            sensitivity = compute_sensitivity_from_distance(dist, threshold_km=10.0)
            land_type = "open" if sensitivity < 0.3 else "conservation"

        profile = NodePrivacyProfile(
            node_id=node_id,
            zone_id=zone_id,
            sensitivity_score=sensitivity,
            land_type=land_type,
            assigned_epsilon=sensitivity_to_epsilon(sensitivity, epsilon_min, epsilon_max),
            max_epsilon_budget=50.0,
        )
        profiles.append(profile)

    return SpatiallyAdaptiveDPScheduler(
        profiles, epsilon_min=epsilon_min, epsilon_max=epsilon_max
    )
