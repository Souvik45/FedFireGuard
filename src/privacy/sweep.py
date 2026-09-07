"""Privacy-Utility sweep harness evaluating anomaly detection accuracy across epsilon ranges."""
import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.metrics import f1_score, roc_auc_score
from src.data.generator import WildfireDataGenerator
from src.edge.client import EdgeNodeClient
from src.federated.server import FLSimulationServer
from src.privacy.engine import apply_dp_sgd, calibrate_noise_multiplier, compute_empirical_privacy_spent
from src.privacy.scheduler import SpatialEpsilonScheduler

class PrivacyUtilitySweeper:
    """Harness executing systematic sweeps over spatial differential privacy budgets.
    
    Generates the required privacy-utility tradeoff curves across real simulated epsilon sweeps
    by perturbing federated gradient updates with calibrated noise and tracking resulting F1 / AUC scores.
    """

    def __init__(self, output_dir: str = "logs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_epsilon_sweep(
        self,
        target_epsilons: List[float] = [0.5, 1.0, 3.0, 5.0, 10.0, 20.0, 100.0],
        num_nodes: int = 4,
        rounds: int = 3,
        seed: int = 42
    ) -> pd.DataFrame:
        """Run simulated federated experiments at varying epsilon thresholds and measure utility."""
        results: List[Dict[str, Any]] = []
        
        # Initialize synthetic non-IID sensor stream with pre-ignition fire injection on Node 0 and Node 1
        gen = WildfireDataGenerator(num_nodes=num_nodes, num_zones=2, time_steps=100, seed=seed)
        scenarios = [
            {"node_id": 0, "start_step": 60, "duration": 20, "magnitude_scale": 1.5},
            {"node_id": 1, "start_step": 50, "duration": 20, "magnitude_scale": 1.5}
        ]
        dfs = gen.generate_all_nodes(fire_scenarios=scenarios)

        for target_eps in target_epsilons:
            sigma = calibrate_noise_multiplier(target_epsilon=target_eps, delta=1e-5, rounds=rounds)
            emp_eps, emp_delta = compute_empirical_privacy_spent(noise_multiplier=sigma, delta=1e-5, rounds=rounds)
            
            # Setup federated edge clients
            clients = [EdgeNodeClient(node_id=i, zone_id=dfs[i]["zone_id"].iloc[0], seq_len=6) for i in range(num_nodes)]
            for i, client in enumerate(clients):
                client.load_dataframe(dfs[i])
                
            server = FLSimulationServer(clients=clients, strategy_type="clustered", logger_dir=os.path.join(self.output_dir, f"fl_eps_{target_eps}"))
            
            # Custom DP federated training loop applying gradient perturbation before server aggregation
            for r in range(rounds):
                active_clients = server.clients
                client_gradients = []
                sample_counts = []
                
                for client in active_clients:
                    _ = client.train_local_epoch(epochs=1, batch_size=16)
                    raw_grads = client.extract_gradients()
                    
                    # Apply Differential Privacy per client update
                    dp_grads = apply_dp_sgd(raw_grads, max_grad_norm=1.0, noise_multiplier=sigma)
                    client_gradients.append(dp_grads)
                    sample_counts.append(len(client.local_data_tensor))
                    
                agg_updates = server.strategy.aggregate_gradients(client_gradients, sample_counts)
                for i, client in enumerate(active_clients):
                    new_state = {}
                    for p_name, old_val in client._initial_state_dict.items():
                        new_state[p_name] = old_val + agg_updates[i][p_name].to(client.device)
                    client.synchronize_with_server(new_state)

            # Measure final reconstruction anomaly detection utility across fire-injected nodes
            y_true_all = []
            y_scores_all = []
            for i in [0, 1]:
                scores = clients[i].compute_anomaly_scores()
                labels = clients[i].local_labels
                y_true_all.extend(labels)
                y_scores_all.extend(scores)
                
            y_true = np.array(y_true_all)
            y_scores = np.array(y_scores_all)
            
            # Choose median threshold on normal scores for binary classification evaluation
            threshold = np.percentile(y_scores, 70)
            y_pred = (y_scores > threshold).astype(int)
            
            f1 = f1_score(y_true, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_true, y_scores)
            except ValueError:
                auc = 0.5
                
            results.append({
                "target_epsilon": target_eps,
                "empirical_epsilon": emp_eps if np.isfinite(emp_eps) else 999.0,
                "empirical_delta": emp_delta,
                "noise_multiplier_sigma": sigma,
                "anomaly_f1_score": round(float(f1), 4),
                "anomaly_auc_roc": round(float(auc), 4)
            })

        df_out = pd.DataFrame(results)
        df_out.to_json(os.path.join(self.output_dir, "privacy_utility_curve.json"), orient="records", indent=2)
        df_out.to_csv(os.path.join(self.output_dir, "privacy_utility_curve.csv"), index=False)
        return df_out
