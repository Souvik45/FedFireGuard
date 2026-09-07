import pytest
import numpy as np
from src.rl.env import WildfireAlertEnv
from src.rl.ppo import PPOAlertAgent
from src.rl.baselines import StaticPolicyBaseline

def test_alert_env_reward_shaping():
    """Verify delayed detection and false alarm fatigue incur severe negative rewards."""
    env = WildfireAlertEnv(num_zones=2, w_delay=10.0, w_false_alarm=5.0, w_resource=1.0)
    obs, info = env.reset(seed=42)
    
    # Force true fire in Zone 0, clear in Zone 1
    env.true_fire_state = np.array([1, 0])
    
    # Test Failure Case: Action = Clear (0,0) in Zone 0 (Delayed detection), Evacuate (3,2) in Zone 1 (False alarm!)
    bad_action = np.array([0, 0, 3, 2])
    _, bad_reward, _, _, bad_info = env.step(bad_action)
    
    assert bad_reward < -10.0, f"Bad alerting must incur heavy penalties! Got {bad_reward}"
    assert bad_info["mean_detection_delay"] > 0.0
    assert bad_info["total_false_alarms"] == 1
    
    # Test Success Case: Action = Warning (2,1) in Zone 0 (Accurate warning), Clear (0,0) in Zone 1 (No false alarm)
    env.reset(seed=42)
    env.true_fire_state = np.array([1, 0])
    good_action = np.array([2, 1, 0, 0])
    _, good_reward, _, _, good_info = env.step(good_action)
    
    print(f"Bad Alert Reward: {bad_reward} | Good Alert Reward: {good_reward}")
    assert good_reward > bad_reward + 15.0, "Accurate alerting must dramatically outperform false alarms/delays!"

def test_ppo_agent_curriculum():
    """Verify PPO agent executes curriculum progression across Tier 0 and Tier 1 without error."""
    env = WildfireAlertEnv(num_zones=2, max_steps=16)
    agent = PPOAlertAgent(num_zones=2, lr=0.01)
    
    history = agent.train_curriculum(env, epochs_tier0=2, epochs_tier1=2)
    assert len(history) == 4
    assert history[0]["curriculum_tier"] == 0
    assert history[2]["curriculum_tier"] == 1
    
    # Verify inference output matches required directive schema
    belief_map = {0: 0.90, 1: 0.10}
    directives = agent.predict_action(belief_map)
    assert 0 in directives and 1 in directives
    assert "alert_level" in directives[0]
    assert directives[0]["alert_level"] in [0, 1, 2, 3]

def test_static_policy_baseline():
    """Verify Baseline 4 maps belief score directly to alert levels."""
    static = StaticPolicyBaseline(watch_threshold=0.30, warning_threshold=0.60, evacuate_threshold=0.85)
    belief = {0: 0.92, 1: 0.65, 2: 0.40, 3: 0.10}
    
    out = static.predict_action(belief)
    assert out[0]["alert_level"] == 3  # Evacuate
    assert out[1]["alert_level"] == 2  # Warning
    assert out[2]["alert_level"] == 1  # Watch
    assert out[3]["alert_level"] == 0  # Clear
