import os
import pytest
import torch
import numpy as np
from src.privacy.scheduler import SpatialEpsilonScheduler
from src.privacy.engine import clip_gradients, inject_gaussian_noise, apply_dp_sgd, calibrate_noise_multiplier, compute_empirical_privacy_spent
from src.privacy.sweep import PrivacyUtilitySweeper

def test_spatial_epsilon_scheduler():
    """Verify nodes near sensitive boundaries receive tighter privacy budgets (lower epsilon)."""
    scheduler = SpatialEpsilonScheduler(eps_tight=1.0, eps_loose=20.0, d_min_km=1.0, d_max_km=10.0)
    
    eps_near_boundary = scheduler.get_epsilon(0.5)
    eps_mid = scheduler.get_epsilon(5.5)
    eps_deep_forest = scheduler.get_epsilon(15.0)
    
    assert eps_near_boundary == 1.0
    assert eps_deep_forest == 20.0
    assert 1.0 < eps_mid < 20.0, f"Intermediate distance failed linear interpolation: got {eps_mid}"
    
    meta = {0: {"boundary_dist_km": 0.2}, 1: {"boundary_dist_km": 12.0}}
    budgets = scheduler.assign_node_budgets(meta)
    assert budgets[0] == 1.0
    assert budgets[1] == 20.0

def test_dp_clipping_and_noise():
    """Verify gradient clipping strictly enforces max L2 norm and calibrated noise perturbs tensors."""
    # Create large gradient vector with norm sqrt(100 + 100) = 14.14 > max_norm=5.0
    grads = {"w1": torch.tensor([10.0, 10.0])}
    clipped, orig_norm = clip_gradients(grads, max_grad_norm=5.0)
    
    assert orig_norm > 10.0
    new_norm = torch.norm(clipped["w1"]).item()
    assert np.isclose(new_norm, 5.0, atol=1e-4), f"Clipped norm {new_norm} does not match max_grad_norm 5.0"
    
    # Verify noise injection adds non-zero variance
    noisy = inject_gaussian_noise(clipped, noise_multiplier=1.0, max_grad_norm=5.0, seed=42)
    assert not torch.allclose(noisy["w1"], clipped["w1"])

def test_noise_calibration_and_empirical_spending():
    """Verify analytical noise calibration and RDP moments accountant formulas."""
    sigma = calibrate_noise_multiplier(target_epsilon=2.0, delta=1e-5, rounds=5)
    assert sigma > 0.0
    
    emp_eps, emp_delta = compute_empirical_privacy_spent(noise_multiplier=sigma, delta=1e-5, rounds=5)
    assert np.isclose(emp_eps, 2.0, atol=0.1), f"Empirical budget {emp_eps} diverged from target 2.0"
    
    # Infinite privacy budget (sigma = 0) returns unconstrained infinity
    emp_inf, _ = compute_empirical_privacy_spent(noise_multiplier=0.0)
    assert emp_inf == float("inf")

def test_privacy_utility_curve_generation(tmp_path):
    """CRITICAL TEST: Verify privacy-utility curve is produced across a real epsilon sweep."""
    out_dir = str(tmp_path / "privacy_test_logs")
    sweeper = PrivacyUtilitySweeper(output_dir=out_dir)
    
    # Sweep across very tight epsilon vs relaxed epsilon
    df_curve = sweeper.run_epsilon_sweep(target_epsilons=[1.0, 100.0], num_nodes=2, rounds=2, seed=42)
    
    assert len(df_curve) == 2
    assert os.path.exists(os.path.join(out_dir, "privacy_utility_curve.json"))
    assert os.path.exists(os.path.join(out_dir, "privacy_utility_curve.csv"))
    
    # Relaxed epsilon (100.0 / almost no noise) should achieve higher or equal F1 compared to high-noise epsilon 1.0
    f1_tight = df_curve[df_curve["target_epsilon"] == 1.0]["anomaly_f1_score"].values[0]
    f1_loose = df_curve[df_curve["target_epsilon"] == 100.0]["anomaly_f1_score"].values[0]
    
    print(f"Tight DP (eps=1.0) F1: {f1_tight} | Loose DP (eps=100.0) F1: {f1_loose}")
    assert f1_loose >= f1_tight, f"Loose DP F1 ({f1_loose}) should not degrade below tight DP F1 ({f1_tight})"
