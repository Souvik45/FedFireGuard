"""PPO-based Reinforcement Learning Alerting Agent for FedFireGuard.

The RL agent observes the GNN fire spread belief map and decides which
alert tier to issue for each zone. It learns to balance two competing costs:
  - Missed detection penalty: failing to alert when fire is spreading
  - False alarm fatigue: over-alerting erodes trust with fire authorities

Alert tiers (action space):
    0 = MONITORING   — normal operations, no action required
    1 = ADVISORY     — increased vigilance, prepare resources
    2 = WARNING      — deploy pre-positioned resources, notify communities
    3 = EVACUATION   — immediate evacuation order

The agent is trained using Proximal Policy Optimization (PPO) in a custom
wildfire environment built on the GNN belief map outputs.

Reference: Schulman et al. (2017) — Proximal Policy Optimization Algorithms.
           arXiv:1707.06347.
           Our extension: tiered reward shaping for wildfire alert decision-making.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Alert tier definitions
# ---------------------------------------------------------------------------

ALERT_TIERS = {
    0: "MONITORING",
    1: "ADVISORY",
    2: "WARNING",
    3: "EVACUATION",
}

# Reward shaping constants
MISSED_DETECTION_PENALTY = -10.0   # fire spread, no alert issued
FALSE_ALARM_PENALTY      = -2.0    # alert issued, no fire
CORRECT_ALERT_REWARD     = +5.0    # right tier at right time
PROPORTIONAL_BONUS       = +2.0    # extra reward for proportional response
EARLY_WARNING_BONUS      = +3.0    # reward for catching fire before it spreads


# ---------------------------------------------------------------------------
# Wildfire Alert Environment
# ---------------------------------------------------------------------------

class WildfireAlertEnv:
    """Custom RL environment for wildfire alert decision-making.

    State space:  GNN fire spread probabilities per node + recent history
                  Shape: (num_nodes * (1 + history_len),)
    Action space: 4 discrete alert tiers (0=MONITORING to 3=EVACUATION)
    Reward:       Shaped reward balancing detection sensitivity vs false alarms

    Args:
        num_nodes:    Number of sensor nodes in the network.
        history_len:  Number of previous timesteps included in state.
        fire_threshold_warning:   P(fire) above which WARNING is appropriate.
        fire_threshold_evacuation: P(fire) above which EVACUATION is appropriate.
        episode_len:  Maximum steps per episode.
    """

    def __init__(
        self,
        num_nodes: int = 6,
        history_len: int = 3,
        fire_threshold_advisory:   float = 0.25,
        fire_threshold_warning:    float = 0.50,
        fire_threshold_evacuation: float = 0.75,
        episode_len: int = 50,
        seed: int = 42,
    ) -> None:
        self.num_nodes = num_nodes
        self.history_len = history_len
        self.fire_threshold_advisory   = fire_threshold_advisory
        self.fire_threshold_warning    = fire_threshold_warning
        self.fire_threshold_evacuation = fire_threshold_evacuation
        self.episode_len = episode_len
        self.rng = np.random.default_rng(seed)

        # State: current probs + history for each node
        self.obs_dim = num_nodes * (1 + history_len)
        self.action_dim = 4

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.step_count = 0
        self.fire_probs = np.zeros(self.num_nodes)
        self.history = np.zeros((self.history_len, self.num_nodes))
        self.fire_active = False
        self.fire_start_step = self.rng.integers(
            self.episode_len // 3, 2 * self.episode_len // 3
        )
        self.fire_nodes = self.rng.choice(
            self.num_nodes,
            size=max(1, self.num_nodes // 3),
            replace=False,
        ).tolist()
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Build observation: current fire probs + flattened history."""
        return np.concatenate([self.fire_probs, self.history.flatten()])

    def _step_fire_dynamics(self) -> None:
        """Simulate fire spread dynamics for this timestep."""
        if self.step_count >= self.fire_start_step:
            self.fire_active = True
            progress = (self.step_count - self.fire_start_step) / max(
                self.episode_len - self.fire_start_step, 1
            )
            for nid in range(self.num_nodes):
                if nid in self.fire_nodes:
                    # Fire nodes ramp up probability
                    base = min(0.3 + progress * 0.7, 1.0)
                    self.fire_probs[nid] = base + self.rng.normal(0, 0.05)
                    self.fire_probs[nid] = np.clip(self.fire_probs[nid], 0.0, 1.0)
                else:
                    # Neighbour nodes get moderate spillover
                    self.fire_probs[nid] = min(progress * 0.3, 0.4)
                    self.fire_probs[nid] += self.rng.normal(0, 0.03)
                    self.fire_probs[nid] = np.clip(self.fire_probs[nid], 0.0, 1.0)
        else:
            # Pre-fire: low baseline noise
            self.fire_probs = np.clip(
                self.rng.normal(0.05, 0.03, self.num_nodes), 0.0, 0.3
            )

    def _compute_appropriate_tier(self) -> int:
        """Determine the objectively correct alert tier given fire probabilities."""
        max_prob = float(self.fire_probs.max())
        if max_prob >= self.fire_threshold_evacuation:
            return 3  # EVACUATION
        elif max_prob >= self.fire_threshold_warning:
            return 2  # WARNING
        elif max_prob >= self.fire_threshold_advisory:
            return 1  # ADVISORY
        else:
            return 0  # MONITORING

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one environment step.

        Args:
            action: Alert tier (0-3).

        Returns:
            Tuple of (next_obs, reward, done, info).
        """
        self._step_fire_dynamics()
        appropriate_tier = self._compute_appropriate_tier()
        max_prob = float(self.fire_probs.max())

        # --- Reward shaping ---
        reward = 0.0

        if action == appropriate_tier:
            reward += CORRECT_ALERT_REWARD
            # Bonus for proportional response (not over/under-alerting)
            reward += PROPORTIONAL_BONUS

        elif action < appropriate_tier:
            # Under-alerting: missed detection
            tier_gap = appropriate_tier - action
            reward += MISSED_DETECTION_PENALTY * tier_gap
            # Extra penalty for missing evacuation
            if appropriate_tier == 3 and action == 0:
                reward += MISSED_DETECTION_PENALTY * 2

        elif action > appropriate_tier:
            # Over-alerting: false alarm fatigue
            tier_gap = action - appropriate_tier
            reward += FALSE_ALARM_PENALTY * tier_gap

        # Early warning bonus: catch fire before max_prob > 0.5
        if self.fire_active and action >= 1 and max_prob < 0.5:
            reward += EARLY_WARNING_BONUS

        # Update history buffer
        self.history = np.roll(self.history, 1, axis=0)
        self.history[0] = self.fire_probs.copy()

        self.step_count += 1
        done = self.step_count >= self.episode_len

        info = {
            "appropriate_tier": appropriate_tier,
            "issued_tier": action,
            "max_fire_prob": round(max_prob, 4),
            "fire_active": self.fire_active,
            "correct": action == appropriate_tier,
        }

        return self._get_obs(), reward, done, info


# ---------------------------------------------------------------------------
# PPO Actor-Critic Network
# ---------------------------------------------------------------------------

class PPOActorCritic(nn.Module):
    """Shared actor-critic network for PPO.

    Actor:  Outputs action logits (alert tier probabilities)
    Critic: Outputs state value estimate V(s)

    Shared backbone extracts features from the observation, then
    splits into separate actor and critic heads.

    Args:
        obs_dim:    Observation dimension.
        action_dim: Number of discrete actions (alert tiers).
        hidden_dim: Hidden layer size.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 4,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head: policy logits
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Critic head: state value
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and value estimate.

        Args:
            obs: Observation tensor (batch, obs_dim).

        Returns:
            Tuple of (action_logits, value) — shapes (batch, action_dim), (batch, 1).
        """
        features = self.backbone(obs)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy and compute log probability.

        Args:
            obs: Observation tensor (batch, obs_dim) or (obs_dim,).

        Returns:
            Tuple of (action, log_prob, value).
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log probabilities and entropy of given actions.

        Args:
            obs:     Observation batch (batch, obs_dim).
            actions: Action batch (batch,).

        Returns:
            Tuple of (log_probs, values, entropy).
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, value.squeeze(-1), entropy


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """Proximal Policy Optimization trainer for the wildfire alert agent.

    Implements the clipped surrogate objective with entropy regularization
    and generalized advantage estimation (GAE).

    Args:
        env:          WildfireAlertEnv instance.
        model:        PPOActorCritic network.
        lr:           Learning rate.
        gamma:        Discount factor.
        gae_lambda:   GAE lambda for advantage estimation.
        clip_epsilon: PPO clip parameter.
        entropy_coef: Entropy bonus coefficient.
        value_coef:   Value loss coefficient.
        device:       Torch device.
    """

    def __init__(
        self,
        env: WildfireAlertEnv,
        model: PPOActorCritic,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.env = env
        self.model = model.to(device)
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
        self.episode_log: List[dict] = []

    def collect_rollout(
        self,
        rollout_len: int = 256,
    ) -> dict:
        """Collect experience by running the policy in the environment.

        Args:
            rollout_len: Number of environment steps to collect.

        Returns:
            Dict of tensors: obs, actions, log_probs, rewards, values, dones.
        """
        obs_list, action_list, logprob_list = [], [], []
        reward_list, value_list, done_list = [], [], []

        obs = self.env.reset()
        episode_reward = 0.0
        episode_correct = 0
        episode_steps = 0

        for _ in range(rollout_len):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, log_prob, value = self.model.get_action(obs_t)

            next_obs, reward, done, info = self.env.step(action.item())

            obs_list.append(obs)
            action_list.append(action.item())
            logprob_list.append(log_prob.item())
            reward_list.append(reward)
            value_list.append(value.item())
            done_list.append(float(done))

            episode_reward += reward
            episode_correct += int(info["correct"])
            episode_steps += 1

            obs = next_obs
            if done:
                self.episode_log.append({
                    "reward": round(episode_reward, 2),
                    "accuracy": round(episode_correct / max(episode_steps, 1), 3),
                })
                obs = self.env.reset()
                episode_reward = 0.0
                episode_correct = 0
                episode_steps = 0

        return {
            "obs":       torch.FloatTensor(np.array(obs_list)).to(self.device),
            "actions":   torch.LongTensor(action_list).to(self.device),
            "log_probs": torch.FloatTensor(logprob_list).to(self.device),
            "rewards":   torch.FloatTensor(reward_list).to(self.device),
            "values":    torch.FloatTensor(value_list).to(self.device),
            "dones":     torch.FloatTensor(done_list).to(self.device),
        }

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards, values, dones: Collected rollout tensors.
            last_value: Value estimate for the state after the rollout.

        Returns:
            Tuple of (advantages, returns) tensors.
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else values[t + 1].item()
            next_non_terminal = 1.0 - dones[t].item()
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(
        self,
        rollout: dict,
        num_epochs: int = 4,
        batch_size: int = 64,
    ) -> dict:
        """PPO policy update step.

        Args:
            rollout:    Collected experience dict.
            num_epochs: Number of optimization epochs per rollout.
            batch_size: Mini-batch size.

        Returns:
            Dict of mean losses for logging.
        """
        obs = rollout["obs"]
        actions = rollout["actions"]
        old_log_probs = rollout["log_probs"]
        rewards = rollout["rewards"]
        values = rollout["values"]
        dones = rollout["dones"]

        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(num_epochs):
            # Random mini-batch sampling
            idx = torch.randperm(len(obs))
            for start in range(0, len(obs), batch_size):
                batch_idx = idx[start: start + batch_size]

                b_obs       = obs[batch_idx]
                b_actions   = actions[batch_idx]
                b_old_lp    = old_log_probs[batch_idx]
                b_adv       = advantages[batch_idx]
                b_returns   = returns[batch_idx]

                new_log_probs, new_values, entropy = self.model.evaluate_actions(
                    b_obs, b_actions
                )

                # PPO clipped surrogate loss
                ratio = torch.exp(new_log_probs - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values, b_returns)

                # Total loss
                loss = (policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy.mean())

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.mean().item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss":  total_value_loss  / max(num_updates, 1),
            "entropy":     total_entropy     / max(num_updates, 1),
        }

    def train(
        self,
        total_steps: int = 10000,
        rollout_len: int = 256,
        log_interval: int = 5,
    ) -> List[dict]:
        """Full PPO training loop.

        Args:
            total_steps:   Total environment steps to train for.
            rollout_len:   Steps per rollout collection.
            log_interval:  Log every N updates.

        Returns:
            Training log — list of dicts with losses and episode metrics.
        """
        training_log = []
        steps_done = 0
        update_num = 0

        print(f"Training PPO agent for {total_steps} steps...")
        while steps_done < total_steps:
            rollout = self.collect_rollout(rollout_len)
            losses = self.update(rollout)
            steps_done += rollout_len
            update_num += 1

            if self.episode_log:
                recent = self.episode_log[-min(5, len(self.episode_log)):]
                mean_reward = np.mean([e["reward"] for e in recent])
                mean_acc = np.mean([e["accuracy"] for e in recent])
            else:
                mean_reward = 0.0
                mean_acc = 0.0

            log_entry = {
                "update": update_num,
                "steps": steps_done,
                "policy_loss": round(losses["policy_loss"], 4),
                "value_loss":  round(losses["value_loss"],  4),
                "entropy":     round(losses["entropy"],     4),
                "mean_reward": round(float(mean_reward),   2),
                "alert_accuracy": round(float(mean_acc),   3),
            }
            training_log.append(log_entry)

            if update_num % log_interval == 0:
                print(
                    f"  Step {steps_done:5d}/{total_steps} | "
                    f"Reward: {mean_reward:6.2f} | "
                    f"Accuracy: {mean_acc:.3f} | "
                    f"Entropy: {losses['entropy']:.3f}"
                )

        return training_log


# ---------------------------------------------------------------------------
# Inference & alert generation
# ---------------------------------------------------------------------------

def issue_alert(
    model: PPOActorCritic,
    fire_probs: np.ndarray,
    history: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> dict:
    """Issue an alert decision given current GNN fire spread probabilities.

    This is the inference function called by the dashboard Tab 1.

    Args:
        model:      Trained PPOActorCritic.
        fire_probs: Current GNN output (num_nodes,) probabilities.
        history:    Previous fire_probs arrays stacked (history_len, num_nodes).
                    If None, zeros used.
        device:     Torch device.

    Returns:
        Dict with: alert_tier, alert_name, confidence, node_probs, rationale.
    """
    model.eval()
    num_nodes = len(fire_probs)

    if history is None:
        history = np.zeros((3, num_nodes))

    obs = np.concatenate([fire_probs, history.flatten()])
    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, value = model(obs_t)
        probs_dist = F.softmax(logits, dim=-1)
        action = torch.argmax(probs_dist, dim=-1).item()
        confidence = float(probs_dist[0, action].item())

    max_fire_prob = float(fire_probs.max())
    high_risk_nodes = [i for i, p in enumerate(fire_probs) if p > 0.5]

    rationale = _build_alert_rationale(
        action, max_fire_prob, high_risk_nodes, confidence
    )

    return {
        "alert_tier": int(action),
        "alert_name": ALERT_TIERS[action],
        "confidence": round(confidence, 3),
        "node_probs": {i: round(float(p), 4) for i, p in enumerate(fire_probs)},
        "state_value": round(float(value.item()), 3),
        "rationale": rationale,
    }


def _build_alert_rationale(
    tier: int,
    max_prob: float,
    high_risk_nodes: List[int],
    confidence: float,
) -> str:
    """Build plain-English rationale for the alert decision."""
    tier_name = ALERT_TIERS[tier]
    node_str = (
        f"Nodes {high_risk_nodes} show elevated fire probability."
        if high_risk_nodes else
        "No nodes currently above 0.5 fire probability threshold."
    )
    return (
        f"[{tier_name}] issued with {confidence*100:.1f}% confidence. "
        f"Peak GNN fire spread probability: {max_prob:.1%}. "
        f"{node_str}"
    )


def save_rl_log(
    training_log: List[dict],
    log_path: str = "logs/rl_training_log.json",
) -> None:
    """Save RL training log for dashboard."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"RL training log saved to {log_path}")


def build_ppo_agent(obs_dim: int, hidden_dim: int = 64) -> PPOActorCritic:
    """Factory function to build PPO actor-critic network."""
    return PPOActorCritic(obs_dim=obs_dim, action_dim=4, hidden_dim=hidden_dim)


def build_alert_env(num_nodes: int = 6, seed: int = 42) -> WildfireAlertEnv:
    """Factory function to build wildfire alert environment."""
    return WildfireAlertEnv(num_nodes=num_nodes, episode_len=50, seed=seed)
