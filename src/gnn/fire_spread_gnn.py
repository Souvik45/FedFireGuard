"""Graph Attention Network for 6-hour fire spread belief map generation.

Models the sensor network as a spatial graph where:
  - Nodes = IoT sensor stations (each with 7 environmental features)
  - Edges = physical proximity between stations, weighted by wind direction
             (downwind nodes get stronger edge weights — fire spreads with wind)

The GNN learns to propagate fire risk signals across the network topology,
generating a probability map of where fire will spread in the next 6 hours.
This output feeds directly into the PPO alerting agent as its observation space.

Architecture:
    Layer 1: Graph Attention (GAT) — attends to neighbour risk signals
    Layer 2: Graph Attention (GAT) — aggregates spatial context
    Layer 3: Linear readout — per-node 6-hour fire spread probability

Why GAT over GCN:
    Attention weights are learned per-edge, so the model naturally learns
    that downwind neighbours matter more than upwind ones — without hardcoding
    wind physics. The attention coefficients are also interpretable.

Reference: Kipf & Welling (2017) — Semi-Supervised Classification with GCNs.
           Veličković et al. (2018) — Graph Attention Networks. ICLR.
           Our extension: wind-aware edge weighting for wildfire propagation.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Graph construction utilities
# ---------------------------------------------------------------------------

def build_adjacency_matrix(
    node_positions: np.ndarray,
    wind_directions: Optional[np.ndarray] = None,
    distance_threshold_km: float = 15.0,
    wind_weight: float = 0.4,
) -> torch.Tensor:
    """Build a wind-aware weighted adjacency matrix for the sensor graph.

    Edge weight between nodes i and j:
        w_ij = proximity_weight * (1 + wind_weight * downwind_factor)

    Downwind factor is high when j is downwind from i — fire from i
    is more likely to spread toward j.

    Args:
        node_positions:        Array of shape (num_nodes, 2) — (x_km, y_km).
        wind_directions:       Array of shape (num_nodes,) — wind direction
                               in degrees (0=North, 90=East) per node.
                               If None, uniform adjacency used.
        distance_threshold_km: Maximum distance for edge creation.
        wind_weight:           How much wind direction amplifies edge weights.

    Returns:
        Weighted adjacency tensor of shape (num_nodes, num_nodes).
        Self-loops included (diagonal = 1).
    """
    num_nodes = len(node_positions)
    adj = torch.zeros(num_nodes, num_nodes)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                adj[i, j] = 1.0  # self-loop
                continue

            dx = node_positions[j, 0] - node_positions[i, 0]
            dy = node_positions[j, 1] - node_positions[i, 1]
            dist = np.sqrt(dx**2 + dy**2)

            if dist > distance_threshold_km:
                continue

            # Proximity weight: closer nodes get stronger edges
            proximity = 1.0 - (dist / distance_threshold_km)

            # Wind amplification: is j downwind from i?
            if wind_directions is not None:
                wind_deg = wind_directions[i]
                wind_rad = np.deg2rad(wind_deg)
                # Wind vector: direction wind is BLOWING TOWARD
                wind_vec = np.array([np.sin(wind_rad), np.cos(wind_rad)])
                # Direction vector from i to j
                if dist > 0:
                    dir_vec = np.array([dx, dy]) / dist
                    # Dot product: 1 = fully downwind, -1 = fully upwind
                    downwind = float(np.dot(wind_vec, dir_vec))
                    downwind_factor = (downwind + 1.0) / 2.0  # scale to [0,1]
                else:
                    downwind_factor = 0.5
            else:
                downwind_factor = 0.5

            adj[i, j] = proximity * (1.0 + wind_weight * downwind_factor)

    # Row-normalise for stable message passing
    row_sums = adj.sum(dim=1, keepdim=True).clamp(min=1e-6)
    adj = adj / row_sums

    return adj


def build_edge_index(adj: torch.Tensor) -> torch.Tensor:
    """Convert adjacency matrix to COO edge index for sparse message passing.

    Args:
        adj: Adjacency matrix (num_nodes, num_nodes).

    Returns:
        Edge index tensor of shape (2, num_edges) — [source_nodes, target_nodes].
    """
    edge_index = adj.nonzero(as_tuple=False).t().contiguous()
    return edge_index


# ---------------------------------------------------------------------------
# Graph Attention Layer (manual implementation — no torch_geometric required)
# ---------------------------------------------------------------------------

class GraphAttentionLayer(nn.Module):
    """Single graph attention layer (GAT) with multi-head attention.

    Computes attention-weighted message passing:
        h_i' = σ(Σ_j α_ij W h_j)
    where α_ij are learned attention coefficients.

    Args:
        in_features:   Input feature dimension per node.
        out_features:  Output feature dimension per node.
        num_heads:     Number of attention heads.
        dropout:       Attention dropout rate.
        concat:        If True, concatenate heads; if False, average them.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat

        # Linear transform for each head
        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)
        # Attention vector for each head
        self.a = nn.Parameter(torch.empty(num_heads, 2 * out_features))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with adjacency-masked attention.

        Args:
            x:   Node features (num_nodes, in_features).
            adj: Adjacency matrix (num_nodes, num_nodes) — used as attention mask.

        Returns:
            Updated node features (num_nodes, out_features * num_heads) if concat,
            else (num_nodes, out_features).
        """
        num_nodes = x.size(0)

        # Linear transform → (num_nodes, num_heads, out_features)
        Wh = self.W(x).view(num_nodes, self.num_heads, self.out_features)

        # Compute attention scores for all node pairs
        # Wh_i: (num_nodes, 1, num_heads, out_features)
        # Wh_j: (1, num_nodes, num_heads, out_features)
        Wh_i = Wh.unsqueeze(1)  # (N, 1, H, F)
        Wh_j = Wh.unsqueeze(0)  # (1, N, H, F)

        # Concatenate for attention: (N, N, H, 2F)
        Wh_cat = torch.cat([Wh_i.expand(num_nodes, num_nodes, -1, -1),
                            Wh_j.expand(num_nodes, num_nodes, -1, -1)], dim=-1)

        # Attention scores: (N, N, H)
        e = self.leaky_relu((Wh_cat * self.a).sum(dim=-1))

        # Mask non-edges with large negative value
        mask = (adj == 0).unsqueeze(-1).expand_as(e)
        e = e.masked_fill(mask, -1e9)

        # Softmax over neighbours for each head: (N, N, H)
        alpha = F.softmax(e, dim=1)
        alpha = self.dropout(alpha)

        # Weighted aggregation: (N, H, F)
        # alpha: (N, N, H) → (N, N, H, 1)
        # Wh_j: (N, N, H, F) → weighted sum over dim=1
        out = (alpha.unsqueeze(-1) * Wh_j.expand(num_nodes, num_nodes, -1, -1)).sum(dim=1)

        if self.concat:
            # Concatenate heads: (N, H*F)
            out = out.view(num_nodes, self.num_heads * self.out_features)
        else:
            # Average heads: (N, F)
            out = out.mean(dim=1)

        return F.elu(out)


# ---------------------------------------------------------------------------
# Full GNN model
# ---------------------------------------------------------------------------

class FireSpreadGNN(nn.Module):
    """Graph Attention Network for 6-hour fire spread probability prediction.

    Takes current sensor readings across all nodes and outputs a per-node
    probability that fire will reach or escalate at that node within 6 hours.

    Architecture:
        Input projection → GAT Layer 1 → GAT Layer 2 → Readout → Sigmoid

    Args:
        node_feature_dim:  Number of input features per node (default 7).
        hidden_dim:        Hidden dimension per attention head.
        num_heads:         Number of attention heads in each GAT layer.
        dropout:           Dropout rate.
    """

    def __init__(
        self,
        node_feature_dim: int = 7,
        hidden_dim: int = 16,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        # GAT Layer 1: hidden_dim → hidden_dim (concat heads)
        self.gat1 = GraphAttentionLayer(
            in_features=hidden_dim,
            out_features=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            concat=True,
        )

        # GAT Layer 2: hidden_dim*num_heads → hidden_dim (average heads)
        self.gat2 = GraphAttentionLayer(
            in_features=hidden_dim * num_heads,
            out_features=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            concat=False,
        )

        # Readout: per-node fire spread probability
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass — predict fire spread probabilities.

        Args:
            node_features: Tensor (num_nodes, node_feature_dim) — current
                           sensor readings for all nodes.
            adj:           Adjacency matrix (num_nodes, num_nodes).

        Returns:
            Fire spread probabilities (num_nodes,) — P(fire reaches node
            within 6 hours) for each node.
        """
        # Input projection
        x = F.relu(self.input_proj(node_features))
        x = self.dropout(x)

        # GAT layers
        x = self.gat1(x, adj)
        x = self.dropout(x)
        x = self.gat2(x, adj)

        # Per-node probability
        probs = self.readout(x).squeeze(-1)  # (num_nodes,)
        return probs

    def get_attention_weights(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get fire spread probabilities and attention weights for interpretability.

        Attention weights show which sensor-to-sensor connections the model
        considers most important for fire spread prediction.

        Returns:
            Tuple of (fire_probs, attention_matrix) where attention_matrix
            is (num_nodes, num_nodes) averaged over all heads.
        """
        x = F.relu(self.input_proj(node_features))

        # Extract attention from GAT layer 1
        num_nodes = x.size(0)
        Wh = self.gat1.W(x).view(num_nodes, self.gat1.num_heads, self.gat1.out_features)
        Wh_i = Wh.unsqueeze(1)
        Wh_j = Wh.unsqueeze(0)
        Wh_cat = torch.cat([Wh_i.expand(num_nodes, num_nodes, -1, -1),
                           Wh_j.expand(num_nodes, num_nodes, -1, -1)], dim=-1)
        e = self.gat1.leaky_relu((Wh_cat * self.gat1.a).sum(dim=-1))
        mask = (adj == 0).unsqueeze(-1).expand_as(e)
        e = e.masked_fill(mask, -1e9)
        alpha = F.softmax(e, dim=1).mean(dim=-1)  # avg over heads: (N, N)

        probs = self.forward(node_features, adj)
        return probs, alpha.detach()


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def generate_gnn_training_data(
    node_dfs: dict,
    adj: torch.Tensor,
    fire_node_ids: List[int],
    seq_len: int = 6,
    feature_cols: Optional[List[str]] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate (node_features, fire_labels) pairs for GNN training.

    For each timestep window, creates:
        - node_features: current mean sensor readings per node (num_nodes, 7)
        - fire_labels:   1.0 for fire nodes, 0.0 for normal nodes (num_nodes,)

    Args:
        node_dfs:      {node_id: DataFrame} from WildfireDataGenerator.
        adj:           Adjacency matrix.
        fire_node_ids: Nodes with fire injected.
        seq_len:       Window length to average over.
        feature_cols:  Sensor feature columns to use.

    Returns:
        List of (node_features_tensor, label_tensor) tuples.
    """
    if feature_cols is None:
        feature_cols = [
            "temperature", "humidity", "wind_speed", "wind_direction",
            "co2", "pm25", "soil_moisture"
        ]

    node_ids = sorted(node_dfs.keys())
    num_nodes = len(node_ids)
    min_len = min(len(df) for df in node_dfs.values())
    samples = []

    for t in range(seq_len, min_len):
        # Per-node mean features over the window
        features = []
        for nid in node_ids:
            df = node_dfs[nid]
            window = df[feature_cols].iloc[t - seq_len:t].values.astype(np.float32)
            mean_features = window.mean(axis=0)
            features.append(mean_features)

        node_feat = torch.tensor(np.array(features), dtype=torch.float32)

        # Normalise per feature across nodes
        feat_mean = node_feat.mean(dim=0)
        feat_std = node_feat.std(dim=0) + 1e-6
        node_feat = (node_feat - feat_mean) / feat_std

        # Labels: 1.0 for fire nodes (we use a soft label based on timestep)
        labels = torch.zeros(num_nodes, dtype=torch.float32)
        for i, nid in enumerate(node_ids):
            if nid in fire_node_ids:
                # Fire probability increases toward the end of the series
                progress = t / min_len
                labels[i] = float(np.clip(progress * 1.5, 0.0, 1.0))

        samples.append((node_feat, labels))

    return samples


def train_gnn(
    model: FireSpreadGNN,
    training_data: List[Tuple[torch.Tensor, torch.Tensor]],
    adj: torch.Tensor,
    num_epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
) -> List[float]:
    """Train the GNN on generated fire spread data.

    Args:
        model:         FireSpreadGNN to train.
        training_data: List of (node_features, labels) tuples.
        adj:           Adjacency matrix.
        num_epochs:    Training epochs.
        lr:            Learning rate.
        device:        Torch device.

    Returns:
        List of per-epoch mean losses.
    """
    model = model.to(device)
    adj = adj.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    epoch_losses = []

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        np.random.shuffle(training_data)

        for node_feat, labels in training_data:
            node_feat = node_feat.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            probs = model(node_feat, adj)
            loss = criterion(probs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        mean_loss = total_loss / max(len(training_data), 1)
        epoch_losses.append(mean_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  GNN Epoch {epoch+1}/{num_epochs} | Loss: {mean_loss:.4f}")

    return epoch_losses


# ---------------------------------------------------------------------------
# Inference & belief map generation
# ---------------------------------------------------------------------------

def generate_fire_spread_belief_map(
    model: FireSpreadGNN,
    node_features: torch.Tensor,
    adj: torch.Tensor,
    node_positions: np.ndarray,
    node_ids: List[int],
    device: str = "cpu",
) -> dict:
    """Generate 6-hour fire spread belief map for dashboard Tab 1.

    Args:
        model:          Trained FireSpreadGNN.
        node_features:  Current sensor readings (num_nodes, 7).
        adj:            Adjacency matrix.
        node_positions: Node (x_km, y_km) coordinates.
        node_ids:       List of node IDs.
        device:         Torch device.

    Returns:
        Dict with keys: node_id, fire_probability, alert_tier, x_km, y_km,
        attention_weights — ready for dashboard rendering.
    """
    model.eval()
    model = model.to(device)
    node_features = node_features.to(device)
    adj = adj.to(device)

    with torch.no_grad():
        probs, attention = model.get_attention_weights(node_features, adj)
        probs = probs.cpu().numpy()
        attention = attention.cpu().numpy()

    # Map probabilities to alert tiers
    def prob_to_tier(p: float) -> str:
        if p >= 0.75:
            return "EVACUATION"
        elif p >= 0.50:
            return "WARNING"
        elif p >= 0.25:
            return "ADVISORY"
        else:
            return "MONITORING"

    belief_map = {
        "nodes": [
            {
                "node_id": nid,
                "fire_probability": round(float(probs[i]), 4),
                "alert_tier": prob_to_tier(float(probs[i])),
                "x_km": float(node_positions[i, 0]),
                "y_km": float(node_positions[i, 1]),
            }
            for i, nid in enumerate(node_ids)
        ],
        "attention_matrix": attention.tolist(),
        "max_risk_node": int(node_ids[int(np.argmax(probs))]),
        "global_risk_level": prob_to_tier(float(probs.max())),
        "mean_fire_probability": round(float(probs.mean()), 4),
    }

    return belief_map


def save_belief_map(
    belief_map: dict,
    path: str = "logs/gnn_belief_map.json",
) -> None:
    """Save belief map to JSON for dashboard Tab 1 overlay."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(belief_map, f, indent=2)
    print(f"Belief map saved to {path}")


def build_gnn(
    node_feature_dim: int = 7,
    hidden_dim: int = 16,
    num_heads: int = 4,
) -> FireSpreadGNN:
    """Factory function to instantiate FireSpreadGNN."""
    return FireSpreadGNN(
        node_feature_dim=node_feature_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
    )
