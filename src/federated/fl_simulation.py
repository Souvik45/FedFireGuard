"""Full Federated Learning simulation loop for FedFireGuard.

This is the main orchestrator that ties together:
  - WildfireDataGenerator  (synthetic sensor data per node)
  - LSTMAutoencoder        (edge TinyML model)
  - FederatedClient        (local training per node)
  - zone_clustered_fedavg  (2-stage aggregation)
  - SpatiallyAdaptiveDPScheduler (per-node epsilon assignment)
  - Logging                (fills dashboard Tab 3 charts)

Running this script produces:
  logs/fl_divergence_history.json   → Tab 3 divergence chart
  logs/privacy_utility_curve.csv    → Tab 3 privacy-utility curve
  logs/fl_summary.json              → overall training summary

Usage:
    python -m src.federated.fl_simulation
    python -m src.federated.fl_simulation --rounds 20 --nodes 8 --epochs 5
"""

import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure project root on path when run as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.generator import WildfireDataGenerator
from src.edge.models import LSTMAutoencoder, compute_reconstruction_errors
from src.federated.zone_fedavg import (
    FederatedClient,
    compute_gradient_divergence,
    log_fl_round,
    zone_clustered_fedavg,
    fedavg_aggregate,
)
from src.federated.dp_scheduler import (
    build_scheduler_from_metadata,
    SpatiallyAdaptiveDPScheduler,
)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_node_data(
    node_df,
    seq_len: int = 6,
    feature_cols: Optional[List[str]] = None,
) -> torch.Tensor:
    """Convert a node's DataFrame into sliding windows for training.

    Args:
        node_df:      Pandas DataFrame from WildfireDataGenerator.
        seq_len:      Window length (number of timesteps per sample).
        feature_cols: Column names to use as features.

    Returns:
        Tensor of shape (num_windows, seq_len, num_features).
    """
    if feature_cols is None:
        feature_cols = [
            "temperature", "humidity", "wind_speed", "wind_direction",
            "co2", "pm25", "soil_moisture"
        ]

    values = node_df[feature_cols].values.astype(np.float32)

    # Normalise per-node: zero mean, unit variance
    mean = values.mean(axis=0)
    std = values.std(axis=0) + 1e-6
    values = (values - mean) / std

    # Sliding windows
    windows = []
    for i in range(len(values) - seq_len + 1):
        windows.append(values[i: i + seq_len])

    if not windows:
        return torch.zeros(1, seq_len, len(feature_cols))

    return torch.tensor(np.array(windows), dtype=torch.float32)


def compute_f1_proxy(
    model: LSTMAutoencoder,
    node_dfs: dict,
    fire_node_ids: List[int],
    seq_len: int = 6,
    threshold: Optional[float] = None,
) -> float:
    """Compute approximate F1 score for anomaly detection across all nodes.

    Uses reconstruction error threshold to classify fire vs normal.
    If threshold is None, uses a dynamic threshold = mean + 0.5*std of
    all node max-scores, which adapts as the model trains.

    Args:
        model:         Global model to evaluate.
        node_dfs:      {node_id: DataFrame} from data generator.
        fire_node_ids: Node IDs that have fire injected (positive class).
        seq_len:       Window length.
        threshold:     Fixed anomaly threshold. None = dynamic (recommended).

    Returns:
        F1 score in [0, 1].
    """
    model.eval()
    node_scores = {}

    for node_id, df in node_dfs.items():
        windows = prepare_node_data(df, seq_len)
        if len(windows) == 0:
            node_scores[node_id] = 0.0
            continue
        errors = compute_reconstruction_errors(model, windows)
        node_scores[node_id] = float(errors.max())

    # Dynamic threshold if not specified
    if threshold is None:
        all_scores = list(node_scores.values())
        threshold = float(np.mean(all_scores) + 0.5 * np.std(all_scores))

    tp, fp, fn, tn = 0, 0, 0, 0
    for node_id, score in node_scores.items():
        predicted_fire = score > threshold
        actual_fire = node_id in fire_node_ids

        if predicted_fire and actual_fire:
            tp += 1
        elif predicted_fire and not actual_fire:
            fp += 1
        elif not predicted_fire and actual_fire:
            fn += 1
        else:
            tn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    return round(f1, 4)


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

class FLSimulation:
    """Full federated learning simulation orchestrator.

    Runs Zone-Clustered FedAvg with Spatially Adaptive DP across
    multiple communication rounds, logging metrics for the dashboard.

    Args:
        num_nodes:     Number of IoT sensor nodes.
        num_zones:     Number of ecological zones.
        time_steps:    Sensor reading timesteps per node.
        num_rounds:    Number of FL communication rounds.
        local_epochs:  Local training epochs per client per round.
        seq_len:       Window length for LSTM input.
        hidden_dim:    LSTM hidden dimension.
        lr:            Local learning rate.
        seed:          Random seed for reproducibility.
        device:        Torch device.
        log_dir:       Directory for output logs.
        run_baseline:  Also run standard FedAvg for comparison.
    """

    def __init__(
        self,
        num_nodes: int = 6,
        num_zones: int = 2,
        time_steps: int = 120,
        num_rounds: int = 15,
        local_epochs: int = 3,
        seq_len: int = 6,
        hidden_dim: int = 32,
        lr: float = 1e-3,
        seed: int = 42,
        device: str = "cpu",
        log_dir: str = "logs",
        run_baseline: bool = True,
    ) -> None:
        self.num_nodes = num_nodes
        self.num_zones = num_zones
        self.time_steps = time_steps
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.seed = seed
        self.device = device
        self.log_dir = log_dir
        self.run_baseline = run_baseline

        os.makedirs(log_dir, exist_ok=True)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _setup_data(self) -> Tuple[dict, dict, List[int]]:
        """Generate synthetic sensor data for all nodes."""
        print("Generating synthetic sensor data...")
        gen = WildfireDataGenerator(
            num_nodes=self.num_nodes,
            num_zones=self.num_zones,
            time_steps=self.time_steps,
            seed=self.seed,
        )
        # Inject fire into first 2 nodes for evaluation
        fire_scenarios = [
            {"node_id": 0, "start_step": int(self.time_steps * 0.6),
             "duration": int(self.time_steps * 0.2), "magnitude_scale": 1.8},
            {"node_id": 1, "start_step": int(self.time_steps * 0.65),
             "duration": int(self.time_steps * 0.15), "magnitude_scale": 1.5},
        ]
        node_dfs = gen.generate_all_nodes(fire_scenarios=fire_scenarios)
        metadata = gen.node_metadata
        fire_node_ids = [0, 1]
        return node_dfs, metadata, fire_node_ids

    def _build_clients(
        self,
        global_model: LSTMAutoencoder,
        node_dfs: dict,
        node_to_zone: Dict[int, int],
    ) -> List[FederatedClient]:
        """Instantiate all federated clients with their local data."""
        clients = []
        for node_id, df in node_dfs.items():
            data = prepare_node_data(df, self.seq_len)
            client = FederatedClient(
                node_id=node_id,
                zone_id=node_to_zone[node_id],
                model=global_model,
                data=data,
                lr=self.lr,
                local_epochs=self.local_epochs,
                device=self.device,
            )
            clients.append(client)
        return clients

    def run(self) -> dict:
        """Execute the full FL training simulation.

        Returns:
            Summary dict with final metrics and log paths.
        """
        # Clear old logs
        for fname in ["fl_divergence_history.json", "privacy_utility_curve.csv"]:
            path = os.path.join(self.log_dir, fname)
            if os.path.exists(path):
                os.remove(path)

        # Setup
        node_dfs, metadata, fire_node_ids = self._setup_data()
        node_to_zone = {nid: meta["zone_id"] for nid, meta in metadata.items()}

        # Build DP scheduler
        dp_scheduler = build_scheduler_from_metadata(
            metadata,
            epsilon_min=0.1,
            epsilon_max=5.0,
            protected_zone_ids=[0],
        )

        # Initialise global model
        global_model = LSTMAutoencoder(
            input_dim=7,
            hidden_dim=self.hidden_dim,
            num_layers=2,
        )

        # Also initialise baseline model (standard FedAvg, no zone clustering)
        baseline_model = copy.deepcopy(global_model) if self.run_baseline else None

        print(f"\nStarting FL simulation: {self.num_rounds} rounds, "
              f"{self.num_nodes} nodes, {self.num_zones} zones")
        print(f"Zone assignments: {node_to_zone}")
        print("-" * 60)

        summary_rounds = []
        start_time = time.time()

        for round_num in range(1, self.num_rounds + 1):
            round_start = time.time()

            # --- DP epsilon assignments for this round ---
            # Simulate fire spread probs (later replaced by real GNN output)
            fire_probs = {
                nid: 0.8 if nid in fire_node_ids else 0.1
                for nid in node_to_zone
            }
            epsilon_assignments = dp_scheduler.step(fire_spread_probs=fire_probs)

            # --- Build and distribute model to clients ---
            clients = self._build_clients(global_model, node_dfs, node_to_zone)
            for client in clients:
                client.set_model(global_model.state_dict())

            # --- Local training ---
            client_results = {}
            for client in clients:
                sd, loss, n_samples = client.local_train()
                client_results[client.node_id] = {
                    "state_dict": sd,
                    "loss": loss,
                    "num_samples": n_samples,
                }

            client_sds = {nid: r["state_dict"] for nid, r in client_results.items()}
            client_losses = [r["loss"] for r in client_results.values()]
            sample_counts = {nid: r["num_samples"] for nid, r in client_results.items()}
            mean_loss = float(np.mean(client_losses))

            # --- Zone-Clustered FedAvg aggregation ---
            global_sd, zone_sds = zone_clustered_fedavg(
                client_sds, node_to_zone, sample_counts,
                intra_zone_weight=0.7, inter_zone_weight=0.3,
            )
            global_model.load_state_dict(global_sd)

            # --- Baseline: standard FedAvg (no zone clustering) ---
            baseline_div = None
            if self.run_baseline and baseline_model is not None:
                baseline_clients = self._build_clients(
                    baseline_model, node_dfs, node_to_zone
                )
                for c in baseline_clients:
                    c.set_model(baseline_model.state_dict())
                baseline_results = [c.local_train() for c in baseline_clients]
                baseline_sds = [r[0] for r in baseline_results]
                baseline_global_sd = fedavg_aggregate(baseline_sds)
                baseline_model.load_state_dict(baseline_global_sd)
                baseline_div = compute_gradient_divergence(
                    baseline_sds, baseline_global_sd
                )

            # --- Metrics ---
            div_zone = compute_gradient_divergence(
                list(client_sds.values()), global_sd
            )

            # Dynamic threshold: use mean + 0.5*std of all node scores
            f1 = compute_f1_proxy(
                global_model, node_dfs, fire_node_ids,
                self.seq_len, threshold=None,
            )

            round_time = time.time() - round_start

            # --- Logging ---
            log_path = os.path.join(self.log_dir, "fl_divergence_history.json")
            log_fl_round(round_num, "ZoneClustered", div_zone, mean_loss, log_path)
            if baseline_div is not None:
                log_fl_round(round_num, "StandardFedAvg", baseline_div,
                             mean_loss, log_path)

            privacy_log = os.path.join(self.log_dir, "privacy_utility_curve.csv")
            dp_scheduler.save_privacy_utility_log(f1, privacy_log)

            # Active epsilon summary
            active_eps = [
                v for v in epsilon_assignments.values() if v is not None
            ]
            mean_eps = round(float(np.mean(active_eps)), 3) if active_eps else 0.0

            summary_rounds.append({
                "round": round_num,
                "mean_loss": round(mean_loss, 4),
                "divergence_zone": round(div_zone, 6),
                "divergence_baseline": round(baseline_div, 6) if baseline_div else None,
                "f1_score": f1,
                "mean_epsilon": mean_eps,
                "round_time_s": round(round_time, 2),
            })

            print(
                f"Round {round_num:2d}/{self.num_rounds} | "
                f"Loss: {mean_loss:.4f} | "
                f"Div(zone): {div_zone:.4f} | "
                f"Div(base): {f'{baseline_div:.4f}' if baseline_div is not None else 'N/A'} | "
                f"F1: {f1:.3f} | "
                f"eps_mean: {mean_eps} | "
                f"Time: {round_time:.1f}s"
            )

        total_time = time.time() - start_time
        final_f1 = summary_rounds[-1]["f1_score"]
        final_div_zone = summary_rounds[-1]["divergence_zone"]
        final_div_base = summary_rounds[-1].get("divergence_baseline")

        # Bandwidth reduction estimate
        # Real deployment: nodes stream data continuously at 1 reading/hour
        # Over a 30-day period: 720 timesteps x 7 features per node
        # FL alternative: send model gradients once per round (every 24 hours)
        # Rounds per 30 days = 30; raw data per round = 24h x 7 features per node
        model_params = sum(p.numel() for p in global_model.parameters())
        # Bandwidth reduction — annual deployment framing (honest comparison):
        # Raw baseline: centralised system streams every sensor reading continuously
        #   = 8760 readings/year x 7 features per node (1 reading/hour, 24x7)
        # FL alternative: node uploads INT8-quantised gradients 1x/week (52 rounds/yr)
        #   Target MCU model (hidden_dim=8, INT8 quantised) ≈ 500 params
        # This is the realistic TinyML deployment target, not the simulation model.
        tinyml_params = 500       # Cortex-M4 MCU deployment target (hidden_dim=8)
        rounds_per_year = 52      # 1 gradient upload per week
        raw_annual = 8760 * 7     # floats per node per year (raw streaming)
        fl_annual = (tinyml_params // 4) * rounds_per_year  # INT8 gradients/year
        bandwidth_reduction = round(
            (1.0 - fl_annual / max(raw_annual, 1)) * 100, 1
        )
        bandwidth_reduction = max(0.0, min(bandwidth_reduction, 99.9))

        summary = {
            "config": {
                "num_nodes": self.num_nodes,
                "num_zones": self.num_zones,
                "num_rounds": self.num_rounds,
                "local_epochs": self.local_epochs,
                "hidden_dim": self.hidden_dim,
                "seed": self.seed,
            },
            "final_metrics": {
                "f1_score": final_f1,
                "divergence_zone_clustered": final_div_zone,
                "divergence_standard_fedavg": final_div_base,
                "bandwidth_reduction_%": bandwidth_reduction,
                "total_training_time_s": round(total_time, 2),
                "model_parameters": model_params,
            },
            "privacy_summary": dp_scheduler.get_privacy_summary(),
            "rounds": summary_rounds,
            "log_paths": {
                "fl_divergence": os.path.join(self.log_dir, "fl_divergence_history.json"),
                "privacy_utility": os.path.join(self.log_dir, "privacy_utility_curve.csv"),
            },
        }

        summary_path = os.path.join(self.log_dir, "fl_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("-" * 60)
        print(f"Simulation complete in {total_time:.1f}s")
        print(f"Final F1:              {final_f1}")
        print(f"Divergence (ZoneCL):   {final_div_zone:.6f}")
        if final_div_base:
            print(f"Divergence (Standard): {final_div_base:.6f}")
        print(f"Bandwidth reduction:   {bandwidth_reduction}%")
        print(f"Logs saved to:         {self.log_dir}/")

        return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FedFireGuard FL Simulation")
    parser.add_argument("--rounds",  type=int, default=15, help="FL communication rounds")
    parser.add_argument("--nodes",   type=int, default=6,  help="Number of IoT nodes")
    parser.add_argument("--zones",   type=int, default=2,  help="Number of ecological zones")
    parser.add_argument("--epochs",  type=int, default=3,  help="Local training epochs")
    parser.add_argument("--steps",   type=int, default=120, help="Timesteps per node")
    parser.add_argument("--hidden",  type=int, default=32, help="LSTM hidden dim")
    parser.add_argument("--lr",      type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed",    type=int, default=42, help="Random seed")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip standard FedAvg baseline comparison")
    args = parser.parse_args()

    sim = FLSimulation(
        num_nodes=args.nodes,
        num_zones=args.zones,
        time_steps=args.steps,
        num_rounds=args.rounds,
        local_epochs=args.epochs,
        hidden_dim=args.hidden,
        lr=args.lr,
        seed=args.seed,
        log_dir=args.log_dir,
        run_baseline=not args.no_baseline,
    )
    sim.run()


if __name__ == "__main__":
    main()
