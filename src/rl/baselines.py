"""Baseline 4: Static heuristic alert policy directly mapping GNN belief scores without RL."""
from typing import Dict, Any

class StaticPolicyBaseline:
    """Baseline 4: Static alert policy mapping GNN belief probability scores to fixed emergency levels.
    
    Serves as the ablation proving why PPO reinforcement learning is necessary to optimize
    the temporal trade-off between prompt early detection and minimizing false alarm fatigue.
    """

    def __init__(
        self,
        watch_threshold: float = 0.30,
        warning_threshold: float = 0.60,
        evacuate_threshold: float = 0.85
    ):
        self.watch_thresh = watch_threshold
        self.warning_thresh = warning_threshold
        self.evac_thresh = evacuate_threshold

    def predict_action(self, belief_map: Dict[int, float]) -> Dict[int, Dict[str, int]]:
        """Map per-zone belief probabilities to static alert directives."""
        directives = {}
        for zone_id, prob in belief_map.items():
            if prob >= self.evac_thresh:
                alert = 3  # Evacuate
                resource = 2  # Heavy Equipment
            elif prob >= self.warning_thresh:
                alert = 2  # Warning
                resource = 1  # Patrol
            elif prob >= self.watch_thresh:
                alert = 1  # Watch
                resource = 1  # Patrol
            else:
                alert = 0  # Clear
                resource = 0  # None
                
            directives[zone_id] = {
                "alert_level": alert,
                "resource_dispatch": resource
            }
        return directives
