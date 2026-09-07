"""Spatial epsilon scheduler for adaptive differential privacy across wildfire zones."""
import numpy as np
from typing import Dict, List, Any

class SpatialEpsilonScheduler:
    """Adaptive Differential Privacy budget scheduler based on spatial land proximity.
    
    In real-world wildfire IoT deployments, sensors located near private land boundaries,
    residential perimeters, or sensitive military/tribal structures demand stringent
    privacy protections (tight epsilon). Conversely, sensors isolated deep within public
    wilderness or national parks can tolerate looser privacy budgets (higher epsilon),
    preserving signal fidelity for rapid wildfire early warning without endangering privacy.
    """

    def __init__(
        self,
        eps_tight: float = 1.0,
        eps_loose: float = 15.0,
        d_min_km: float = 1.0,
        d_max_km: float = 10.0
    ):
        if eps_tight >= eps_loose:
            raise ValueError(f"eps_tight ({eps_tight}) must be smaller than eps_loose ({eps_loose})")
        self.eps_tight = eps_tight
        self.eps_loose = eps_loose
        self.d_min_km = d_min_km
        self.d_max_km = d_max_km

    def get_epsilon(self, boundary_dist_km: float) -> float:
        """Calculate spatial epsilon budget for an individual node based on boundary distance."""
        if boundary_dist_km <= self.d_min_km:
            return float(self.eps_tight)
        elif boundary_dist_km >= self.d_max_km:
            return float(self.eps_loose)
        
        # Linear interpolation across transition zone
        slope = (self.eps_loose - self.eps_tight) / (self.d_max_km - self.d_min_km)
        epsilon = self.eps_tight + slope * (boundary_dist_km - self.d_min_km)
        return float(round(epsilon, 3))

    def assign_node_budgets(self, node_metadata: Dict[int, Dict[str, Any]]) -> Dict[int, float]:
        """Map all participating IoT nodes to personalized spatial epsilon budgets."""
        budgets = {}
        for node_id, meta in node_metadata.items():
            dist = meta.get("boundary_dist_km", self.d_max_km)
            budgets[node_id] = self.get_epsilon(dist)
        return budgets
