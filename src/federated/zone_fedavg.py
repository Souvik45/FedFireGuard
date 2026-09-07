"""Zone-Clustered Federated Averaging for FedFireGuard.

Standard FedAvg (McMahan et al. 2017) fails when client data is non-IID —
an alpine forest node and a coastal valley node have completely different
'normal' microclimate distributions. Averaging them globally degrades both.

Zone-Clustered FedAvg fixes this by:
  1. Grouping nodes into ecologically similar zones (alpine, valley, coastal etc.)
  2. Averaging gradients WITHIN each zone first (intra-zone aggregation)
  3. Then doing a weighted global merge across zone representatives
  4. Distributing the global model back to all nodes

This preserves zone-specific microclimate knowledge while still enabling
cross-zone generalisation for rare fire signatures.

Reference: McMahan et al. (2017) — Communication-Efficient Learning of Deep
           Networks from Decentralized Data. AISTATS.
           Our extension: zone-aware clustering before aggregation.
"""

import copy
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
NodeID = int
ZoneID = int
StateDict = Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# Core aggregation utilities
# ---------------------------------------------------------------------------

def fedavg_aggregate(
    state_dicts: List[StateDict],
    weights: Optional[List[float]] = None,
) -> StateDict:
    """Weighted average of model state dicts (standard FedAvg core).

    Args:
        state_dicts: List of model state dicts from participating clients.
        weights:     Optional per-client weights (e.g. num_samples).
                     If None, uniform averaging is used.

    Returns:
        Aggregated state dict with averaged parameters.
    """
    if not state_dicts:
        raise ValueError("No state dicts provided for aggregation.")

    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    aggregated = copy.deepcopy(state_dicts[0])
    for key in aggregated:
        aggregated[key] = torch.zeros_like(aggregated[key], dtype=torch.float32)
        for sd, w in zip(state_dicts, weights):
            aggregated[key] += w * sd[key].float()

    return aggregated


def zone_clustered_fedavg(
    node_state_dicts: Dict[NodeID, StateDict],
    node_to_zone: Dict[NodeID, ZoneID],
    node_sample_counts: Optional[Dict[NodeID, int]] = None,
    intra_zone_weight: float = 0.7,
    inter_zone_weight: float = 0.3,
) -> Tuple[StateDict, Dict[ZoneID, StateDict]]:
    """Zone-Clustered FedAvg — two-stage aggregation.

    Stage 1 — Intra-zone: Average nodes within each ecological zone.
    Stage 2 — Inter-zone: Weighted merge of zone representatives into
               a global model, then blend back with zone models.

    Args:
        node_state_dicts:    {node_id: state_dict} for all participating nodes.
        node_to_zone:        {node_id: zone_id} mapping.
        node_sample_counts:  {node_id: num_samples} for weighted averaging.
                             If None, uniform weights used.
        intra_zone_weight:   How much to keep zone-specific knowledge (0–1).
        inter_zone_weight:   How much global knowledge to blend in (0–1).
                             intra + inter should sum to 1.0.

    Returns:
        Tuple of:
            global_model: Single global state dict (cross-zone knowledge).
            zone_models:  {zone_id: state_dict} zone-specific models
                          (blended intra+global — what nodes actually receive).
    """
    # --- Group nodes by zone ---
    zone_nodes: Dict[ZoneID, List[NodeID]] = defaultdict(list)
    for node_id, zone_id in node_to_zone.items():
        if node_id in node_state_dicts:
            zone_nodes[zone_id].append(node_id)

    # --- Stage 1: Intra-zone aggregation ---
    zone_state_dicts: Dict[ZoneID, StateDict] = {}
    zone_weights: List[float] = []

    for zone_id, nodes in zone_nodes.items():
        zone_sds = [node_state_dicts[n] for n in nodes]

        if node_sample_counts:
            w = [node_sample_counts.get(n, 1) for n in nodes]
        else:
            w = None

        zone_state_dicts[zone_id] = fedavg_aggregate(zone_sds, w)

        # Zone weight for inter-zone step = total samples in zone
        if node_sample_counts:
            zone_weights.append(sum(node_sample_counts.get(n, 1) for n in nodes))
        else:
            zone_weights.append(len(nodes))

    # --- Stage 2: Inter-zone aggregation → global model ---
    zone_sd_list = list(zone_state_dicts.values())
    global_model = fedavg_aggregate(zone_sd_list, zone_weights)

    # --- Blend: each zone model = intra_zone * zone_model + inter_zone * global ---
    blended_zone_models: Dict[ZoneID, StateDict] = {}
    for zone_id, zone_sd in zone_state_dicts.items():
        blended = {}
        for key in global_model:
            blended[key] = (
                intra_zone_weight * zone_sd[key].float()
                + inter_zone_weight * global_model[key].float()
            )
        blended_zone_models[zone_id] = blended

    return global_model, blended_zone_models


# ---------------------------------------------------------------------------
# Client — represents a single IoT sensor node
# ---------------------------------------------------------------------------

class FederatedClient:
    """Single federated learning client representing one IoT sensor node.

    Each client:
      - Holds its own local dataset (sensor windows)
      - Trains the local model for E epochs
      - Returns updated state dict (gradients implicitly via weight diff)
      - Never shares raw sensor data — only model parameters

    Args:
        node_id:    Unique node identifier.
        zone_id:    Ecological zone this node belongs to.
        model:      A fresh copy of the global model architecture.
        data:       Local training data tensor (num_windows, seq_len, input_dim).
        lr:         Local learning rate.
        local_epochs: Number of local training epochs per round.
        device:     Torch device string.
    """

    def __init__(
        self,
        node_id: NodeID,
        zone_id: ZoneID,
        model: nn.Module,
        data: torch.Tensor,
        lr: float = 1e-3,
        local_epochs: int = 3,
        device: str = "cpu",
    ) -> None:
        self.node_id = node_id
        self.zone_id = zone_id
        self.device = torch.device(device)
        self.data = data.to(self.device)
        self.local_epochs = local_epochs
        self.lr = lr
        self.model = copy.deepcopy(model).to(self.device)
        self.num_samples = len(data)

    def set_model(self, state_dict: StateDict) -> None:
        """Load global/zone model weights into local model."""
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in state_dict.items()}
        )

    def local_train(self) -> Tuple[StateDict, float, int]:
        """Run local training for E epochs on local sensor data.

        Returns:
            Tuple of (updated_state_dict, mean_loss, num_samples).
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        total_loss = 0.0
        steps = 0

        for _ in range(self.local_epochs):
            # Shuffle data each epoch
            idx = torch.randperm(len(self.data))
            shuffled = self.data[idx]

            # Mini-batch training
            batch_size = min(32, len(shuffled))
            for start in range(0, len(shuffled), batch_size):
                batch = shuffled[start : start + batch_size]
                optimizer.zero_grad()
                loss = self.model.reconstruction_loss(batch)
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                steps += 1

        mean_loss = total_loss / max(steps, 1)
        return copy.deepcopy(self.model.state_dict()), mean_loss, self.num_samples


# ---------------------------------------------------------------------------
# Divergence tracking (for dashboard Tab 3)
# ---------------------------------------------------------------------------

def compute_gradient_divergence(
    state_dicts: List[StateDict],
    reference_sd: StateDict,
) -> float:
    """Compute mean cosine divergence of client models from a reference.

    Used to measure non-IID gradient conflict — how much clients disagree.
    High divergence = strong non-IID data heterogeneity.

    Args:
        state_dicts:  List of client state dicts after local training.
        reference_sd: Reference model (e.g. previous global model).

    Returns:
        Mean cosine divergence across all layers and clients (0=aligned, 1=orthogonal).
    """
    divergences = []

    for sd in state_dicts:
        layer_divs = []
        for key in reference_sd:
            if "weight" not in key:
                continue
            ref_flat = reference_sd[key].float().flatten()
            client_flat = sd[key].float().flatten()

            # Cosine similarity → divergence
            cos_sim = torch.nn.functional.cosine_similarity(
                ref_flat.unsqueeze(0),
                client_flat.unsqueeze(0),
            ).item()
            layer_divs.append(1.0 - cos_sim)

        if layer_divs:
            divergences.append(np.mean(layer_divs))

    return float(np.mean(divergences)) if divergences else 0.0


def log_fl_round(
    round_num: int,
    strategy: str,
    mean_divergence: float,
    mean_loss: float,
    log_path: str = "logs/fl_divergence_history.json",
) -> None:
    """Append FL round metrics to JSON log file for dashboard Tab 3.

    Args:
        round_num:       Current communication round number.
        strategy:        Strategy name e.g. 'ZoneClustered' or 'StandardFedAvg'.
        mean_divergence: Cosine divergence metric for this round.
        mean_loss:       Mean reconstruction loss across clients.
        log_path:        Path to JSON log file.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    entry = {
        "round": round_num,
        "strategy": strategy,
        "mean_divergence": round(mean_divergence, 6),
        "mean_loss": round(mean_loss, 6),
    }

    existing = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []

    existing.append(entry)
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)
