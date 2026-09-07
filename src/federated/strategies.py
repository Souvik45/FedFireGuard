"""Federated learning strategies: Standard FedAvg (Baseline 2) and Non-IID Clustered FedAvg."""
import torch
import numpy as np
from typing import List, Dict, Union, Tuple, Optional, Any
from src.federated.metrics import flatten_gradients, compute_cosine_divergence

class StandardFedAvgStrategy:
    """Baseline 2: McMahan et al. 2017 standard Federated Averaging strategy.
    
    Aggregates all participating node gradients into a single global model consensus
    via sample-weighted mean, regardless of microclimate zone divergences.
    """
    def __init__(self):
        self.strategy_name = "standard_fedavg"

    def aggregate_gradients(
        self,
        client_gradients: List[Dict[str, torch.Tensor]],
        client_sample_counts: List[int]
    ) -> List[Dict[str, torch.Tensor]]:
        """Compute sample-weighted global average gradient and return to all clients.
        
        Returns:
            List containing identical global aggregated gradient dict for each client.
        """
        if not client_gradients:
            return []
            
        total_samples = sum(client_sample_counts)
        if total_samples <= 0:
            total_samples = len(client_sample_counts)
            weights = [1.0 / len(client_sample_counts)] * len(client_sample_counts)
        else:
            weights = [count / total_samples for count in client_sample_counts]

        global_grad: Dict[str, torch.Tensor] = {}
        for param_name in client_gradients[0].keys():
            accum = torch.zeros_like(client_gradients[0][param_name], dtype=torch.float32)
            for i, client_g in enumerate(client_gradients):
                accum += weights[i] * client_g[param_name].float()
            global_grad[param_name] = accum
            
        # Standard FedAvg broadcasts the exact same aggregated update to all N clients
        return [global_grad for _ in range(len(client_gradients))]


class ClusteredFedAvgStrategy:
    """Novelty 1: Zone-Clustered FedAvg for Non-IID wildfire microclimates.
    
    Calculates pairwise gradient-space cosine similarity across node updates and computes
    soft-clustered consensus updates. Nodes experiencing similar microclimate conditions
    (e.g., cool high altitude vs dry canyon) aggregate together, avoiding non-IID gradient cancelation
    and drastically reducing gradient divergence.
    """
    def __init__(self, temperature: float = 5.0, hard_clustering_threshold: Optional[float] = None):
        self.strategy_name = "clustered_fedavg"
        self.temperature = temperature
        self.hard_clustering_threshold = hard_clustering_threshold

    def compute_similarity_matrix(self, client_gradients: List[Dict[str, torch.Tensor]]) -> np.ndarray:
        """Compute pairwise cosine similarity matrix S[i,j] between client gradient updates."""
        num_clients = len(client_gradients)
        sim_matrix = np.eye(num_clients, dtype=np.float32)
        
        flat_vecs = [flatten_gradients(g) for g in client_gradients]
        norms = [torch.norm(v).item() for v in flat_vecs]
        
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                if norms[i] == 0.0 or norms[j] == 0.0:
                    sim = 0.0
                else:
                    dot = torch.dot(flat_vecs[i], flat_vecs[j]).item()
                    sim = dot / (norms[i] * norms[j])
                sim = np.clip(sim, -1.0, 1.0)
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
                
        return sim_matrix

    def aggregate_gradients(
        self,
        client_gradients: List[Dict[str, torch.Tensor]],
        client_sample_counts: List[int]
    ) -> List[Dict[str, torch.Tensor]]:
        """Perform cosine-weighted soft cluster aggregation for each client update."""
        num_clients = len(client_gradients)
        if num_clients == 0:
            return []
        if num_clients == 1:
            return [client_gradients[0]]

        sim_matrix = self.compute_similarity_matrix(client_gradients)
        updated_client_gradients: List[Dict[str, torch.Tensor]] = []
        
        for i in range(num_clients):
            # Calculate attention weights over peer client gradients for node i
            sim_row = sim_matrix[i]
            
            if self.hard_clustering_threshold is not None:
                # Hard clustering: only aggregate with peers exceeding similarity threshold
                mask = sim_row >= self.hard_clustering_threshold
                if not np.any(mask):
                    mask[i] = True  # Fallback to self-update if no cluster peers match
                weights = mask.astype(np.float32)
                weights = weights / np.sum(weights)
            else:
                # Soft-clustered softmax attention weighted by similarity and sample count
                logits = self.temperature * sim_row
                # Numerical stability shift
                exp_logits = np.exp(logits - np.max(logits)) * np.array(client_sample_counts)
                weights = exp_logits / np.sum(exp_logits)

            cluster_update: Dict[str, torch.Tensor] = {}
            for param_name in client_gradients[0].keys():
                accum = torch.zeros_like(client_gradients[i][param_name], dtype=torch.float32)
                for j in range(num_clients):
                    accum += float(weights[j]) * client_gradients[j][param_name].float()
                cluster_update[param_name] = accum
                
            updated_client_gradients.append(cluster_update)

        return updated_client_gradients
