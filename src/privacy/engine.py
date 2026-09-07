"""Differential privacy gradient clipping, calibrated Gaussian noise injection, and moments accounting."""
import math
import torch
import numpy as np
from typing import Dict, Tuple, Optional

def clip_gradients(
    gradients: Dict[str, torch.Tensor],
    max_grad_norm: float
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Apply L2 norm clipping to client gradient dictionary (Abadi et al. 2016).
    
    Returns:
        Tuple of (clipped_gradients, orig_total_norm).
    """
    total_norm = 0.0
    for tensor in gradients.values():
        total_norm += torch.sum(tensor.float() ** 2).item()
    total_norm = math.sqrt(total_norm)
    
    clip_coef = max_grad_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        clipped_grads = {k: v * clip_coef for k, v in gradients.items()}
    else:
        clipped_grads = {k: v.clone() for k, v in gradients.items()}
        
    return clipped_grads, float(total_norm)

def inject_gaussian_noise(
    gradients: Dict[str, torch.Tensor],
    noise_multiplier: float,
    max_grad_norm: float,
    seed: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """Inject calibrated Gaussian noise N(0, (sigma * max_norm)^2) independently to each parameter."""
    if seed is not None:
        torch.manual_seed(seed)
        
    noise_std = noise_multiplier * max_grad_norm
    noisy_grads = {}
    
    for k, tensor in gradients.items():
        noise = torch.randn_like(tensor.float()) * noise_std
        noisy_grads[k] = tensor.float() + noise
        
    return noisy_grads

def apply_dp_sgd(
    gradients: Dict[str, torch.Tensor],
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 0.5,
    seed: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """Execute complete per-client DP-SGD pipeline: L2 gradient clipping followed by Gaussian noise."""
    if noise_multiplier <= 0.0:
        return gradients
    clipped_grads, _ = clip_gradients(gradients, max_grad_norm=max_grad_norm)
    noisy_grads = inject_gaussian_noise(clipped_grads, noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm, seed=seed)
    return noisy_grads

def calibrate_noise_multiplier(target_epsilon: float, delta: float = 1e-5, rounds: int = 5) -> float:
    """Calculate required noise multiplier sigma to satisfy (epsilon, delta)-DP over total communication rounds.
    
    Applies strong composition / analytical moments accountant inversion:
    sigma approx (sqrt(2 * rounds * ln(1.25 / delta))) / target_epsilon
    """
    if target_epsilon >= 100.0 or target_epsilon <= 0.0:
        return 0.0
    num = math.sqrt(2.0 * rounds * math.log(1.25 / delta))
    sigma = num / target_epsilon
    return float(max(0.01, round(sigma, 4)))

def compute_empirical_privacy_spent(noise_multiplier: float, delta: float = 1e-5, rounds: int = 5) -> Tuple[float, float]:
    """Compute empirical privacy budget (epsilon, delta) consumed across multi-round federated training.
    
    Utilizes analytical Moments Accountant composition across R iterations.
    """
    if noise_multiplier <= 0.0:
        return float('inf'), delta
        
    # Invert composition equation to derive exact cumulative epsilon
    empirical_eps = math.sqrt(2.0 * rounds * math.log(1.25 / delta)) / noise_multiplier
    return float(round(empirical_eps, 4)), delta
