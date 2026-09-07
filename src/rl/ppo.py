"""Custom PPO actor-critic loop tailored for MultiDiscrete alerting under partial observability."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from src.rl.env import WildfireAlertEnv

class MultiDiscreteActorCritic(nn.Module):
    """Shared actor-critic neural network mapping belief embeddings to multi-zone alert logits."""

    def __init__(self, obs_dim: int, action_dims: List[int], hidden_dim: int = 64):
        super(MultiDiscreteActorCritic, self).__init__()
        self.action_dims = action_dims
        
        # Shared feature extractor over belief probability vectors
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Critic: predicts V(s) scalar baseline
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Actor heads for each separate zone alert level and resource decision
        self.actor_heads = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for dim in action_dims
        ])

    def forward(self, obs: torch.Tensor) -> Tuple[List[Categorical], torch.Tensor]:
        features = self.shared(obs)
        val = self.critic(features)
        
        dists = []
        for head in self.actor_heads:
            logits = head(features)
            dists.append(Categorical(logits=logits))
            
        return dists, val

    def get_action_and_val(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dists, val = self.forward(obs)
        
        if action is None:
            actions = [d.sample() for d in dists]
            action_tensor = torch.stack(actions, dim=-1)
        else:
            action_tensor = action
            
        log_prob = 0.0
        entropy = 0.0
        for idx, d in enumerate(dists):
            log_prob = log_prob + d.log_prob(action_tensor[:, idx])
            entropy = entropy + d.entropy()
            
        return action_tensor, log_prob, entropy, val.squeeze(-1)


class PPOAlertAgent:
    """Proximal Policy Optimization (PPO) agent for autonomous wildfire emergency alerting.
    
    Note on Framework Selection: Implements a concise, fully customizable PyTorch PPO loop
    over Stable-Baselines3. While SB3 supports simple MultiDiscrete MLPs, our capstone requires
    explicit curriculum tier transitions, multi-objective reward itemization (delay vs false alarm fatigue),
    and fast local testing without multiprocess IPC overhead.
    """

    def __init__(
        self,
        num_zones: int = 2,
        lr: float = 0.003,
        gamma: float = 0.98,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.02,
        vf_coef: float = 0.5,
        device: str = "cpu"
    ):
        self.num_zones = num_zones
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.device = device
        
        obs_dim = num_zones * 2
        action_dims = []
        for _ in range(num_zones):
            action_dims.extend([4, 3])
            
        self.policy = MultiDiscreteActorCritic(obs_dim=obs_dim, action_dims=action_dims).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def train_episode_batch(self, env: WildfireAlertEnv, batch_size_steps: int = 256, epochs: int = 4) -> Dict[str, float]:
        """Collect rollouts and execute PPO objective gradient optimizations."""
        self.policy.train()
        
        obs_list = []
        actions_list = []
        logprobs_list = []
        rewards_list = []
        values_list = []
        dones_list = []
        
        obs, _ = env.reset()
        for step in range(batch_size_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action_t, log_prob, _, val = self.policy.get_action_and_val(obs_tensor)
                
            act_np = action_t.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(act_np)
            done = terminated or truncated
            
            obs_list.append(obs)
            actions_list.append(act_np)
            logprobs_list.append(log_prob.item())
            rewards_list.append(reward)
            values_list.append(val.item())
            dones_list.append(done)
            
            obs = next_obs
            if done:
                obs, _ = env.reset()
                
        # Compute GAE (Generalized Advantage Estimation)
        advantages = np.zeros_like(rewards_list, dtype=np.float32)
        lastgaelam = 0.0
        with torch.no_grad():
            next_obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, _, _, next_value = self.policy.get_action_and_val(next_obs_t)
            next_value = next_value.item()
            
        for t in reversed(range(batch_size_steps)):
            if t == batch_size_steps - 1:
                nextnonterminal = 1.0 - int(dones_list[-1])
                nextval = next_value
            else:
                nextnonterminal = 1.0 - int(dones_list[t + 1])
                nextval = values_list[t + 1]
                
            delta = rewards_list[t] + self.gamma * nextval * nextnonterminal - values_list[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            
        returns = advantages + np.array(values_list, dtype=np.float32)
        
        # Optimize PPO clipped objective
        b_obs = torch.tensor(np.array(obs_list), dtype=torch.float32, device=self.device)
        b_actions = torch.tensor(np.array(actions_list), dtype=torch.long, device=self.device)
        b_logprobs = torch.tensor(np.array(logprobs_list), dtype=torch.float32, device=self.device)
        b_advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        b_returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        total_policy_loss = 0.0
        total_val_loss = 0.0
        
        for ep in range(epochs):
            _, new_logprob, entropy, new_value = self.policy.get_action_and_val(b_obs, action=b_actions)
            logratio = new_logprob - b_logprobs
            ratio = torch.exp(logratio)
            
            pg_loss1 = -b_advantages * ratio
            pg_loss2 = -b_advantages * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            
            v_loss = 0.5 * ((new_value - b_returns) ** 2).mean()
            entropy_loss = entropy.mean()
            
            loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_policy_loss += pg_loss.item()
            total_val_loss += v_loss.item()
            
        return {
            "mean_episode_reward": float(np.mean(rewards_list)),
            "policy_loss": round(total_policy_loss / max(1, epochs), 4),
            "value_loss": round(total_val_loss / max(1, epochs), 4)
        }

    def train_curriculum(self, env: WildfireAlertEnv, epochs_tier0: int = 5, epochs_tier1: int = 8) -> List[Dict[str, Any]]:
        """Run two-phase curriculum training: simple single-zone first, then multi-zone spread."""
        history = []
        
        # Tier 0: Single-zone simple fires
        env.set_curriculum_tier(0)
        for ep in range(epochs_tier0):
            stats = self.train_episode_batch(env, batch_size_steps=128, epochs=3)
            stats["curriculum_tier"] = 0
            stats["epoch"] = ep + 1
            history.append(stats)
            
        # Tier 1: Multi-zone stochastic wind-driven spread
        env.set_curriculum_tier(1)
        for ep in range(epochs_tier1):
            stats = self.train_episode_batch(env, batch_size_steps=256, epochs=4)
            stats["curriculum_tier"] = 1
            stats["epoch"] = epochs_tier0 + ep + 1
            history.append(stats)
            
        return history

    def predict_action(self, belief_map: Dict[int, float], prev_alerts: Optional[Dict[int, int]] = None) -> Dict[int, Dict[str, int]]:
        """Inference interface: convert GNN belief probability map directly into zone emergency directives."""
        self.policy.eval()
        if prev_alerts is None:
            prev_alerts = {z: 0 for z in range(self.num_zones)}
            
        probs_vec = [belief_map.get(z, 0.0) for z in range(self.num_zones)]
        prev_vec = [prev_alerts.get(z, 0) / 3.0 for z in range(self.num_zones)]
        obs = np.concatenate([probs_vec, prev_vec])
        
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_t, _, _, _ = self.policy.get_action_and_val(obs_tensor)
            
        act_np = action_t.squeeze(0).cpu().numpy()
        directives = {}
        for z in range(self.num_zones):
            directives[z] = {
                "alert_level": int(act_np[z * 2]),
                "resource_dispatch": int(act_np[z * 2 + 1])
            }
        return directives
