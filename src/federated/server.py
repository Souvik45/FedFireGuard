"""Federated learning server orchestrating distributed Edge AI rounds and divergence logging."""
import copy
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from src.edge.client import EdgeNodeClient
from src.federated.strategies import StandardFedAvgStrategy, ClusteredFedAvgStrategy
from src.federated.metrics import compute_mean_divergence, compute_cosine_divergence, FLDivergenceLogger

class FLSimulationServer:
    """Orchestrates federated communication rounds across simulated IoT edge clients.
    
    Coordinates broadcasting global models, triggering local client training epochs,
    aggregating gradient deltas via designated strategies, and explicitly logging
    gradient divergence metrics between Standard FedAvg and Clustered FedAvg.
    """

    def __init__(
        self,
        clients: List[EdgeNodeClient],
        strategy_type: str = "clustered",
        logger_dir: str = "logs",
        temperature: float = 5.0
    ):
        self.clients = clients
        self.strategy_type = strategy_type.lower()
        if self.strategy_type in ["clustered", "clustered_fedavg"]:
            self.strategy = ClusteredFedAvgStrategy(temperature=temperature)
        else:
            self.strategy = StandardFedAvgStrategy()
            
        self.logger = FLDivergenceLogger(log_dir=logger_dir)
        self.current_round: int = 0
        
        # Initialize a central reference model state from client 0 to synchronize all nodes initially
        if self.clients:
            self.reference_model_state: Dict[str, torch.Tensor] = copy.deepcopy(self.clients[0].model.state_dict())
            self._broadcast_initial_model()
        else:
            self.reference_model_state = {}

    def _broadcast_initial_model(self) -> None:
        """Ensure all edge clients start communication round 0 from uniform weights."""
        for client in self.clients:
            client.synchronize_with_server(self.reference_model_state)

    def run_communication_round(
        self,
        local_epochs: int = 1,
        batch_size: int = 16,
        active_client_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Execute one complete round of decentralized training, gradient aggregation, and broadcast."""
        self.current_round += 1
        
        if active_client_indices is None:
            active_client_indices = list(range(len(self.clients)))
            
        active_clients = [self.clients[idx] for idx in active_client_indices]
        
        # Step 1: Local edge training on normal sensor time series
        client_losses = []
        client_gradients = []
        sample_counts = []
        
        for client in active_clients:
            loss = client.train_local_epoch(batch_size=batch_size, epochs=local_epochs)
            client_losses.append(loss)
            grads = client.extract_gradients()
            client_gradients.append(grads)
            # Record dataset sample count for weighted averaging
            num_samples = len(client.local_data_tensor) if client.local_data_tensor is not None else 1
            sample_counts.append(num_samples)
            
        # Step 2: Server Strategy Aggregation
        aggregated_updates = self.strategy.aggregate_gradients(client_gradients, sample_counts)
        
        # Step 3: Explicitly compute and log non-IID gradient divergence
        mean_div = compute_mean_divergence(client_gradients, aggregated_updates)
        per_client_div = [
            compute_cosine_divergence(cg, ag) for cg, ag in zip(client_gradients, aggregated_updates)
        ]
        
        self.logger.log_round(
            round_num=self.current_round,
            strategy_name=self.strategy.strategy_name,
            mean_divergence=mean_div,
            per_client_divergence=per_client_div,
            additional_metrics={"mean_local_loss": float(np.mean(client_losses))}
        )
        
        # Step 4: Broadcast updated weights back to clients
        # For standard FedAvg, all clients get the identical new global consensus
        # For clustered FedAvg, each node gets its microclimate cluster's customized update
        for i, client in enumerate(active_clients):
            agg_grad = aggregated_updates[i]
            new_state: Dict[str, torch.Tensor] = {}
            for param_name, old_val in client._initial_state_dict.items():
                new_state[param_name] = old_val.to(client.device) + agg_grad[param_name].to(client.device)
            client.synchronize_with_server(new_state)
            if i == 0 and self.strategy.strategy_name == "standard_fedavg":
                self.reference_model_state = copy.deepcopy(new_state)

        return {
            "round": self.current_round,
            "strategy": self.strategy.strategy_name,
            "mean_divergence": round(mean_div, 4),
            "mean_loss": round(float(np.mean(client_losses)), 4),
            "num_participating_nodes": len(active_clients)
        }

    def train_federated_loop(self, total_rounds: int = 5, local_epochs: int = 1) -> List[Dict[str, Any]]:
        """Run multi-round federated training experiments and return metrics summary."""
        summary = []
        for r in range(total_rounds):
            metrics = self.run_communication_round(local_epochs=local_epochs)
            summary.append(metrics)
        return summary

    def get_all_node_anomaly_scores(self) -> Dict[int, np.ndarray]:
        """Collect local inference anomaly scores across all client nodes."""
        scores = {}
        for client in self.clients:
            scores[client.node_id] = client.compute_anomaly_scores()
        return scores
