"""Gymnasium environment wrapping GNN belief maps with dense reward shaping and curriculum scheduling."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Any, Optional

class WildfireAlertEnv(gym.Env):
    """Reinforcement Learning environment for autonomous wildfire alerting under partial observability.
    
    CRITICAL ARCHITECTURAL BOUNDARY: The RL agent's observation state consists ONLY of the GNN's
    probabilistic fire belief map and prior alert states. It never directly accesses raw time series.
    
    Curriculum Scheduling:
    - Tier 0: Single-zone distinct fire scenarios (dense deterministic supervision to bootstrap policy).
    - Tier 1: Multi-zone stochastic wind-driven spread with partial sensor dropouts.
    """

    def __init__(
        self,
        num_zones: int = 2,
        w_delay: float = 12.0,
        w_false_alarm: float = 5.0,
        w_resource: float = 2.0,
        curriculum_tier: int = 0,
        max_steps: int = 24
    ):
        super(WildfireAlertEnv, self).__init__()
        self.num_zones = num_zones
        self.w_delay = w_delay
        self.w_false_alarm = w_false_alarm
        self.w_resource = w_resource
        self.curriculum_tier = curriculum_tier
        self.max_steps = max_steps

        # Observation space: [gnn_belief_probability (N zones), previous_alert_level_normalized (N zones)]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_zones * 2,),
            dtype=np.float32
        )

        # Action space: per-zone Alert Level (0=Clear, 1=Watch, 2=Warning, 3=Evacuate) and Resource Pre-positioning (0=None, 1=Patrol, 2=Heavy Equipment)
        # Flattened MultiDiscrete array: [z0_alert, z0_res, z1_alert, z1_res, ...]
        n_vec = []
        for _ in range(num_zones):
            n_vec.extend([4, 3])
        self.action_space = spaces.MultiDiscrete(n_vec)

        self.current_step = 0
        self.true_fire_state = np.zeros(num_zones, dtype=int)
        self.prev_alerts = np.zeros(num_zones, dtype=int)
        self.detection_delays = np.zeros(num_zones, dtype=int)

    def set_curriculum_tier(self, tier: int) -> None:
        """Advance curriculum difficulty tier."""
        self.curriculum_tier = tier

    def _generate_belief_state(self) -> np.ndarray:
        """Simulate GNN belief inference over true wildfire ground state."""
        beliefs = np.zeros(self.num_zones, dtype=np.float32)
        for z in range(self.num_zones):
            if self.true_fire_state[z] == 1:
                # Active fire produces high GNN probability belief with slight observation noise
                beliefs[z] = np.clip(np.random.normal(0.88, 0.08), 0.5, 1.0)
            else:
                # Clear zone produces low GNN belief
                beliefs[z] = np.clip(np.random.normal(0.12, 0.08), 0.0, 0.4)
                
        obs = np.concatenate([beliefs, self.prev_alerts.astype(np.float32) / 3.0])
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.prev_alerts = np.zeros(self.num_zones, dtype=int)
        self.detection_delays = np.zeros(self.num_zones, dtype=int)
        
        # Initialize wildfire breakout scenarios according to curriculum tier
        self.true_fire_state = np.zeros(self.num_zones, dtype=int)
        if self.curriculum_tier == 0:
            # Tier 0: Exactly one zone breaks out into fire at initial episode step
            target_zone = np.random.randint(0, self.num_zones)
            self.true_fire_state[target_zone] = 1
        else:
            # Tier 1: Multi-zone stochastic probability emergence
            self.true_fire_state = (np.random.rand(self.num_zones) > 0.6).astype(int)
            if np.sum(self.true_fire_state) == 0:
                self.true_fire_state[0] = 1

        obs = self._generate_belief_state()
        return obs, {"true_fire_state": self.true_fire_state.copy()}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        
        # Unpack actions per zone
        alert_actions = np.zeros(self.num_zones, dtype=int)
        res_actions = np.zeros(self.num_zones, dtype=int)
        for z in range(self.num_zones):
            alert_actions[z] = int(action[z * 2])
            res_actions[z] = int(action[z * 2 + 1])

        reward = 0.0
        info_delays = 0
        info_false_alarms = 0
        
        for z in range(self.num_zones):
            fire = self.true_fire_state[z]
            alert = alert_actions[z]
            res = res_actions[z]

            # 1. Detection delay penalty: Fire is active but agent failed to warn (alert == 0)
            if fire == 1 and alert == 0:
                self.detection_delays[z] += 1
                penalty = -self.w_delay * float(self.detection_delays[z])
                reward += penalty
                info_delays += self.detection_delays[z]
            elif fire == 1 and alert > 0:
                # Successful detection reward scaled by promptness and alert adequacy
                self.detection_delays[z] = 0
                reward += 8.0 * float(alert) / 3.0
                
            # 2. False alarm fatigue penalty: No fire present but agent sounded warning
            if fire == 0 and alert > 0:
                penalty = -self.w_false_alarm * float(alert)
                reward += penalty
                info_false_alarms += 1
                
            # 3. Resource expenditure cost
            reward -= self.w_resource * float(res)

        # Stochastic wildfire spread dynamics across steps in Tier 1
        if self.curriculum_tier >= 1 and self.current_step < self.max_steps:
            for z in range(self.num_zones):
                if self.true_fire_state[z] == 0 and np.any(self.true_fire_state == 1):
                    if np.random.rand() < 0.25:
                        self.true_fire_state[z] = 1

        self.prev_alerts = alert_actions.copy()
        terminated = (self.current_step >= self.max_steps)
        truncated = False
        
        obs = self._generate_belief_state()
        info = {
            "true_fire_state": self.true_fire_state.copy(),
            "mean_detection_delay": float(np.mean(self.detection_delays)),
            "total_false_alarms": info_false_alarms,
            "resource_cost_expended": float(np.sum(res_actions))
        }
        
        return obs, float(round(reward, 4)), terminated, truncated, info
