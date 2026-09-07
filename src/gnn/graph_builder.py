"""Directional sensor graph construction weighted by distance and wind vectors."""
import math
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from torch_geometric.data import Data

class WildfireGraphBuilder:
    """Constructs directed spatio-temporal sensor graphs for PyTorch Geometric GAT models.
    
    In wildfire dynamics, fire spread is fundamentally directional: flames ride prevailing
    winds. Therefore, graph edges are asymmetric: edge weight from Node A -> Node B is
    amplified if prevailing winds blow from A toward B, and suppressed if winds oppose B.
    """

    def __init__(self, max_distance_km: float = 30.0, wind_amplification_gamma: float = 2.0):
        self.max_distance_km = max_distance_km
        self.gamma = wind_amplification_gamma

    def build_graph(
        self,
        node_metadata: Dict[int, Dict[str, Any]],
        current_readings: Dict[int, Dict[str, float]],
        zone_targets: Optional[Dict[int, float]] = None
    ) -> Data:
        """Create PyG Data graph object from current IoT sensor state.
        
        Args:
            node_metadata: Mapping of node_id -> spatial coordinates ('x_km', 'y_km') & 'zone_id'.
            current_readings: Mapping of node_id -> sensor feature dictionary (including 'wind_speed', 'wind_direction', 'anomaly_score').
            zone_targets: Mapping of zone_id -> binary or probability fire ground-truth label for 6-hour horizon.
            
        Returns:
            torch_geometric.data.Data object with directed edge attributes.
        """
        node_ids = sorted(list(node_metadata.keys()))
        id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        num_nodes = len(node_ids)

        # Build node feature vectors (anomaly_score + raw meteorological features + operational mask flag)
        feature_list = []
        zone_mapping = []
        for nid in node_ids:
            reads = current_readings.get(nid, {})
            vec = [
                reads.get("anomaly_score", 0.0),
                reads.get("temperature", 20.0),
                reads.get("humidity", 50.0),
                reads.get("wind_speed", 10.0),
                reads.get("wind_direction", 180.0) / 360.0,
                reads.get("pm25", 10.0),
                reads.get("co2", 400.0),
                reads.get("soil_moisture", 30.0),
                reads.get("is_operational", 1.0)  # Binary mask flag (1.0 = active, 0.0 = offline/burned)
            ]
            feature_list.append(vec)
            zone_mapping.append(node_metadata[nid]["zone_id"])

        x_tensor = torch.tensor(feature_list, dtype=torch.float32)
        zone_tensor = torch.tensor(zone_mapping, dtype=torch.int64)

        # Build directional wind-weighted edge adjacency
        edge_src: List[int] = []
        edge_dst: List[int] = []
        edge_weights: List[float] = []

        for src_id in node_ids:
            src_idx = id_to_idx[src_id]
            x_s = node_metadata[src_id]["x_km"]
            y_s = node_metadata[src_id]["y_km"]
            
            reads_s = current_readings.get(src_id, {})
            wind_deg = reads_s.get("wind_direction", 0.0)
            wind_spd = reads_s.get("wind_speed", 10.0)
            
            # Convert meteorological wind degrees to Cartesian directional unit vector
            # Meteorological 0 is North (+Y), 90 is East (+X) blowing FROM that direction
            wind_rad = math.radians(wind_deg)
            wx = math.sin(wind_rad) * wind_spd
            wy = math.cos(wind_rad) * wind_spd
            w_norm = math.sqrt(wx*wx + wy*wy) + 1e-6

            for dst_id in node_ids:
                if src_id == dst_id:
                    # Self-loop for neighborhood GAT retention
                    edge_src.append(src_idx)
                    edge_dst.append(src_idx)
                    edge_weights.append(1.0)
                    continue
                    
                dst_idx = id_to_idx[dst_id]
                x_d = node_metadata[dst_id]["x_km"]
                y_d = node_metadata[dst_id]["y_km"]
                
                dx = x_d - x_s
                dy = y_d - y_s
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist <= self.max_distance_km:
                    # Calculate vector alignment between wind direction and neighbor destination
                    dot = (wx * dx + wy * dy)
                    cos_align = dot / (w_norm * dist)
                    # Positive alignment means wind blows directly from src toward dst
                    wind_factor = math.exp(self.gamma * max(0.0, float(cos_align)))
                    
                    # Directional edge weight combining spatial decay and wind push
                    weight = (1.0 / (1.0 + dist)) * wind_factor
                    
                    edge_src.append(src_idx)
                    edge_dst.append(dst_idx)
                    edge_weights.append(float(round(weight, 4)))

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(-1)

        # Format zone target probabilities if ground truth available
        y_tensor = None
        if zone_targets is not None:
            num_zones = max(zone_mapping) + 1
            targets = [zone_targets.get(z, 0.0) for z in range(num_zones)]
            y_tensor = torch.tensor(targets, dtype=torch.float32)

        data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)
        data.zone_id = zone_tensor
        return data
