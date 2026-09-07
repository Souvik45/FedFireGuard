"""Gradient divergence calculation and logging for non-IID federated experiments."""
import json
import os
import torch
import numpy as np
from typing import List, Dict, Union, Optional, Any

def flatten_gradients(grad_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a dictionary of layer gradient tensors into a single 1D vector."""
    vectors = []
    for param_name in sorted(grad_dict.keys()):
        tensor = grad_dict[param_name].float()
        vectors.append(tensor.view(-1))
    return torch.cat(vectors)

def compute_cosine_divergence(client_grads: Dict[str, torch.Tensor], target_grad: Dict[str, torch.Tensor]) -> float:
    """Compute cosine distance divergence: 1.0 - cosine_similarity(client_vec, target_vec).
    
    A value near 0.0 means gradients point in identical directions (homogeneous IID consensus).
    A high value indicates conflicting parameter updates due to non-IID zone heterogeneities.
    """
    vec_c = flatten_gradients(client_grads)
    vec_t = flatten_gradients(target_grad)
    
    norm_c = torch.norm(vec_c)
    norm_t = torch.norm(vec_t)
    
    if norm_c == 0.0 or norm_t == 0.0:
        return 0.0
        
    cos_sim = torch.dot(vec_c, vec_t) / (norm_c * norm_t)
    # Clamp to prevent numerical drift outside [-1, 1]
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    divergence = 1.0 - float(cos_sim.item())
    return max(0.0, divergence)

def compute_mean_divergence(
    client_grad_list: List[Dict[str, torch.Tensor]],
    target_grad_list: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]
) -> float:
    """Compute average gradient divergence across all participating IoT edge nodes.
    
    Args:
        client_grad_list: List of local client gradient dictionaries.
        target_grad_list: Either a single global gradient dictionary (standard FedAvg)
                          or a corresponding list of cluster-tailored gradient dictionaries (Clustered FedAvg).
                          
    Returns:
        Mean cosine divergence score across clients.
    """
    if not client_grad_list:
        return 0.0
        
    div_vals = []
    for idx, c_grads in enumerate(client_grad_list):
        if isinstance(target_grad_list, list):
            t_grads = target_grad_list[idx]
        else:
            t_grads = target_grad_list
        div_vals.append(compute_cosine_divergence(c_grads, t_grads))
        
    return float(np.mean(div_vals))

class FLDivergenceLogger:
    """Persistent logger tracking non-IID gradient divergence across communication rounds.
    
    Provides the core empirical evidence demonstrating why standard FedAvg fails on
    spatially diverse wildfire microclimates, validating Clustered FedAvg.
    """
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "fl_divergence_history.json")
        self.history: List[Dict[str, Any]] = []
        os.makedirs(log_dir, exist_ok=True)

    def log_round(
        self,
        round_num: int,
        strategy_name: str,
        mean_divergence: float,
        per_client_divergence: Optional[List[float]] = None,
        additional_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record divergence metrics for a communication round."""
        entry = {
            "round": round_num,
            "strategy": strategy_name,
            "mean_divergence": round(mean_divergence, 5),
            "per_client_divergence": [round(x, 5) for x in per_client_divergence] if per_client_divergence else [],
            "metrics": additional_metrics or {}
        }
        self.history.append(entry)
        self._save()

    def _save(self) -> None:
        """Export cumulative log history to JSON for downstream eval figures."""
        with open(self.log_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def get_strategy_divergence_series(self, strategy_name: str) -> List[float]:
        """Extract historical mean divergence series for a specific strategy."""
        return [entry["mean_divergence"] for entry in self.history if entry["strategy"] == strategy_name]
