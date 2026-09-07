"""Synthetic sensor simulation and dataset generation with non-IID microclimates."""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

class WildfireDataGenerator:
    """Generates multivariate time-series IoT sensor data with non-IID zone microclimates
    and injectable wildfire pre-ignition signatures.
    """

    def __init__(
        self,
        num_nodes: int = 8,
        num_zones: int = 2,
        time_steps: int = 240,
        seed: int = 42,
        sampling_rate_mins: int = 60
    ):
        self.num_nodes = num_nodes
        self.num_zones = num_zones
        self.time_steps = time_steps
        self.seed = seed
        self.sampling_rate_mins = sampling_rate_mins
        self.rng = np.random.default_rng(seed)
        
        # Node metadata: zone mapping and 2D spatial coordinates (km)
        self.node_metadata: Dict[int, Dict[str, Any]] = {}
        self._init_node_geographies()

    def _init_node_geographies(self) -> None:
        """Assign nodes to distinct microclimate zones with coordinates."""
        # Create diverse zone baselines to guarantee non-IID behavior across clusters
        base_configs = [
            # Zone 0: High-elevation alpine forest (cooler, more humid, higher wind)
            {"temp_base": 12.0, "temp_amp": 5.0, "hum_base": 65.0, "hum_amp": 15.0, "wind_base": 18.0, "co2_base": 395.0, "pm_base": 8.0, "soil_base": 35.0},
            # Zone 1: Low-elevation valley / canyon (hotter, dry, stagnant air, higher baseline PM)
            {"temp_base": 26.0, "temp_amp": 9.0, "hum_base": 35.0, "hum_amp": 12.0, "wind_base": 6.0, "co2_base": 420.0, "pm_base": 16.0, "soil_base": 18.0},
            # Zone 2: Coastal / riparian brush (moderate temp, high humidity, breezy)
            {"temp_base": 18.0, "temp_amp": 4.0, "hum_base": 75.0, "hum_amp": 8.0, "wind_base": 14.0, "co2_base": 405.0, "pm_base": 10.0, "soil_base": 45.0},
            # Zone 3: Inland plateau (high thermal range, dry wind corridors)
            {"temp_base": 21.0, "temp_amp": 12.0, "hum_base": 42.0, "hum_amp": 18.0, "wind_base": 22.0, "co2_base": 410.0, "pm_base": 12.0, "soil_base": 22.0},
        ]

        for n in range(self.num_nodes):
            zone_id = n % self.num_zones
            config = base_configs[zone_id % len(base_configs)]
            
            # Place nodes spatially clustered by zone around zone centers
            center_x = (zone_id * 15.0)
            center_y = ((zone_id % 2) * 15.0)
            x_coord = float(center_x + self.rng.normal(0, 2.5))
            y_coord = float(center_y + self.rng.normal(0, 2.5))
            
            # Distance to simulated private land boundary (e.g., along vertical line at x=5.0)
            # Used later in Phase 4 for adaptive differential privacy epsilon scheduling
            sensitive_boundary_x = 5.0
            boundary_dist = abs(x_coord - sensitive_boundary_x)

            self.node_metadata[n] = {
                "node_id": n,
                "zone_id": zone_id,
                "x_km": x_coord,
                "y_km": y_coord,
                "boundary_dist_km": boundary_dist,
                "microclimate": config
            }

    def generate_node_timeseries(self, node_id: int) -> pd.DataFrame:
        """Generate baseline time series for a single IoT sensor node."""
        meta = self.node_metadata[node_id]["microclimate"]
        t = np.arange(self.time_steps)
        hours = (t * self.sampling_rate_mins / 60.0) % 24.0

        # Diurnal cycles with AR(1) autocorrelated noise
        temp_diurnal = meta["temp_base"] + meta["temp_amp"] * -np.cos(2 * np.pi * hours / 24.0)
        hum_diurnal = meta["hum_base"] + meta["hum_amp"] * np.cos(2 * np.pi * hours / 24.0)
        
        def generate_ar1_noise(size: int, alpha: float = 0.8, scale: float = 1.0) -> np.ndarray:
            noise = np.zeros(size)
            w = self.rng.normal(0, scale, size)
            for i in range(1, size):
                noise[i] = alpha * noise[i-1] + (1 - alpha) * w[i]
            return noise

        temp = temp_diurnal + generate_ar1_noise(self.time_steps, alpha=0.85, scale=2.5)
        hum = np.clip(hum_diurnal + generate_ar1_noise(self.time_steps, alpha=0.8, scale=5.0), 5.0, 100.0)
        
        wind_speed = np.clip(meta["wind_base"] + generate_ar1_noise(self.time_steps, alpha=0.7, scale=4.0), 0.0, 100.0)
        wind_direction = (180.0 + generate_ar1_noise(self.time_steps, alpha=0.9, scale=60.0)) % 360.0
        
        co2 = meta["co2_base"] + generate_ar1_noise(self.time_steps, alpha=0.9, scale=10.0)
        pm25 = np.clip(meta["pm_base"] + generate_ar1_noise(self.time_steps, alpha=0.7, scale=3.0), 0.5, 1000.0)
        soil_moisture = np.clip(meta["soil_base"] + generate_ar1_noise(self.time_steps, alpha=0.95, scale=2.0), 0.0, 100.0)

        df = pd.DataFrame({
            "timestamp": t,
            "node_id": node_id,
            "zone_id": self.node_metadata[node_id]["zone_id"],
            "temperature": temp,
            "humidity": hum,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction,
            "co2": co2,
            "pm25": pm25,
            "soil_moisture": soil_moisture,
            "fire_label": np.zeros(self.time_steps, dtype=int)
        })
        return df

    def inject_pre_ignition_signature(
        self,
        df: pd.DataFrame,
        start_step: int,
        duration: int,
        magnitude_scale: float = 1.0
    ) -> pd.DataFrame:
        """Inject known pre-ignition signature (temp spike + humidity drop + PM2.5 & CO2 rise)."""
        end_step = min(start_step + duration, len(df))
        if start_step >= end_step:
            return df
            
        steps = np.arange(end_step - start_step)
        ramp = (steps + 1) / (end_step - start_step)  # Progressive increase over the event window
        
        df.loc[start_step:end_step-1, "temperature"] += (15.0 * magnitude_scale) * ramp
        df.loc[start_step:end_step-1, "humidity"] = np.clip(
            df.loc[start_step:end_step-1, "humidity"] - (30.0 * magnitude_scale) * ramp, 5.0, 100.0
        )
        df.loc[start_step:end_step-1, "pm25"] += (120.0 * magnitude_scale) * (ramp ** 2)
        df.loc[start_step:end_step-1, "co2"] += (90.0 * magnitude_scale) * ramp
        df.loc[start_step:end_step-1, "fire_label"] = 1
        
        return df

    def generate_all_nodes(
        self,
        fire_scenarios: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[int, pd.DataFrame]:
        """Generate data for all nodes, optionally applying specified fire scenario injections."""
        node_dfs = {}
        for n in range(self.num_nodes):
            df = self.generate_node_timeseries(n)
            node_dfs[n] = df
            
        if fire_scenarios:
            for scen in fire_scenarios:
                target_node = scen["node_id"]
                start = scen["start_step"]
                duration = scen.get("duration", 12)
                scale = scen.get("magnitude_scale", 1.0)
                if target_node in node_dfs:
                    node_dfs[target_node] = self.inject_pre_ignition_signature(
                        node_dfs[target_node], start, duration, scale
                    )
        return node_dfs

    def save_to_parquet(self, output_dir: str = "data", node_dfs: Optional[Dict[int, pd.DataFrame]] = None) -> str:
        """Save generated time series as labeled Parquet files per node along with metadata."""
        os.makedirs(output_dir, exist_ok=True)
        if node_dfs is None:
            node_dfs = self.generate_all_nodes()
            
        for n, df in node_dfs.items():
            file_path = os.path.join(output_dir, f"node_{n}.parquet")
            df.to_parquet(file_path, index=False)
            
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.node_metadata, f, indent=2)
            
        return output_dir
