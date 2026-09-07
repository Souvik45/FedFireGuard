"""Baseline 3: Threshold rule system evaluating raw sensor readings directly without ML/GNN."""
import pandas as pd
from typing import Dict, Any, List

class ThresholdRuleSystem:
    """Baseline 3: Legacy hardcoded heuristic rule system evaluating raw sensor thresholds.
    
    Represents currently-deployed traditional wildfire monitoring systems.
    Evaluates raw time series (temp, humidity, PM2.5) without edge AI anomaly scoring,
    federated consensus, or GNN predictive belief mapping.
    """

    def __init__(
        self,
        temp_evac: float = 36.0,
        hum_evac: float = 22.0,
        pm_evac: float = 50.0,
        temp_warn: float = 31.0,
        hum_warn: float = 32.0,
        pm_watch: float = 25.0
    ):
        self.temp_evac = temp_evac
        self.hum_evac = hum_evac
        self.pm_evac = pm_evac
        self.temp_warn = temp_warn
        self.hum_warn = hum_warn
        self.pm_watch = pm_watch

    def evaluate_node_step(self, readings: Dict[str, float]) -> int:
        """Evaluate raw sensor dictionary at a single timestamp and output alert level."""
        t = readings.get("temperature", 20.0)
        h = readings.get("humidity", 50.0)
        pm = readings.get("pm25", 10.0)

        # Severe multi-sensor threshold breach -> Evacuate
        if t >= self.temp_evac and h <= self.hum_evac and pm >= self.pm_evac:
            return 3
        # Thermal & atmospheric dry warning -> Warning
        elif t >= self.temp_warn and h <= self.hum_warn:
            return 2
        # Particulate spike -> Watch
        elif pm >= self.pm_watch or t >= 32.0:
            return 1
        else:
            return 0

    def evaluate_zone_directives(
        self,
        zone_to_nodes_readings: Dict[int, List[Dict[str, float]]]
    ) -> Dict[int, Dict[str, int]]:
        """Compute zone-level emergency directives by taking max alert severity across zone sensors."""
        directives = {}
        for zone_id, readings_list in zone_to_nodes_readings.items():
            max_alert = 0
            for r in readings_list:
                a = self.evaluate_node_step(r)
                if a > max_alert:
                    max_alert = a
                    
            res_dispatch = 2 if max_alert == 3 else (1 if max_alert >= 1 else 0)
            directives[zone_id] = {
                "alert_level": max_alert,
                "resource_dispatch": res_dispatch
            }
        return directives
