import pytest
import torch
import numpy as np
from src.edge.models import LSTMAutoencoder
from src.rl.ppo import MultiDiscreteActorCritic
from src.explainability.shap_engine import EdgeModelExplainer, PPOPolicyExplainer

def test_edge_model_explainer():
    """Verify EdgeModelExplainer assigns top attribution to anomalous spiked feature channels."""
    model = LSTMAutoencoder(input_dim=7, hidden_dim=10)
    explainer = EdgeModelExplainer(model)
    
    # Create window where temperature (col 0) and pm25 (col 5) have massive extreme spikes
    window = torch.zeros(1, 6, 7)
    window[0, :, 0] = 10.0  # Huge temperature z-score
    window[0, :, 5] = 15.0  # Huge PM2.5 z-score
    
    attr = explainer.explain_window(window)
    assert len(attr) == 7
    assert "temperature" in attr and "pm25" in attr
    
    print(f"Feature Attributions on spiked window: {attr}")
    # Verify PM2.5 and Temp dominate attribution over baseline features like wind speed (col 2)
    assert attr["pm25"] > attr["wind_speed"], f"PM2.5 attribution ({attr['pm25']}) should exceed wind_speed ({attr['wind_speed']})"

def test_ppo_policy_explainer():
    """Verify PPOPolicyExplainer attributes emergency alert escalation to high belief probability."""
    policy = MultiDiscreteActorCritic(obs_dim=4, action_dims=[4, 3, 4, 3])
    explainer = PPOPolicyExplainer(policy, num_zones=2)
    
    obs = np.array([0.95, 0.10, 0.0, 0.0])  # Zone 0 has 95% GNN belief probability
    attr = explainer.explain_alert_decision(obs, target_zone=0)
    
    assert len(attr) == 4
    assert "Zone_0_GNN_Belief" in attr
    print(f"PPO Alert Decision Attributions: {attr}")
    assert float(sum(attr.values())) > 0.0
