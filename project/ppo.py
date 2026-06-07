"""
ppo.py - Constrained Proximal Policy Optimisation with Lagrangian multipliers.

Features:
  • Clipped surrogate objective
  • Generalised Advantage Estimation (GAE)
  • Entropy regularisation
  • Lagrangian-multiplier equity & budget constraints
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from config import PPO_CFG


# ══════════════════════════════════════════════════════════════════════════════
# Rollout buffer
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Transition:
    """Single transition stored in the rollout buffer."""
    state: np.ndarray            # (H, W, C)
    action: int                  # node index
    reward: float
    log_prob: float
    value: float
    done: bool
    feasible_mask: np.ndarray    # (N,) bool - for the graph


class RolloutBuffer:
    """
    Stores one episode of transitions and computes GAE advantages.
    """

    def __init__(self):
        self.transitions: List[Transition] = []

    def add(self, t: Transition):
        self.transitions.append(t)

    def clear(self):
        self.transitions = []

    def __len__(self):
        return len(self.transitions)

    def compute_gae(self,
                    last_value: float,
                    gamma: float = None,
                    gae_lambda: float = None) -> tuple:
        """
        Compute Generalised Advantage Estimation.

        Returns
        -------
        advantages : ndarray (T,)
        returns : ndarray (T,)
        """
        gamma = gamma or PPO_CFG.gamma
        gae_lambda = gae_lambda or PPO_CFG.gae_lambda

        T = len(self.transitions)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)

        gae = 0.0
        next_value = last_value

        for t in reversed(range(T)):
            tr = self.transitions[t]
            if tr.done:
                delta = tr.reward - tr.value
                gae = delta
            else:
                delta = tr.reward + gamma * next_value - tr.value
                gae = delta + gamma * gae_lambda * gae

            advantages[t] = gae
            returns[t] = gae + tr.value
            next_value = tr.value

        return advantages, returns

    def get_batch(self):
        """Extract parallel arrays from the buffer."""
        states = [t.state for t in self.transitions]
        actions = np.array([t.action for t in self.transitions])
        log_probs = np.array([t.log_prob for t in self.transitions])
        values = np.array([t.value for t in self.transitions])
        rewards = np.array([t.reward for t in self.transitions])
        dones = np.array([t.done for t in self.transitions])
        masks = [t.feasible_mask for t in self.transitions]
        return states, actions, log_probs, values, rewards, dones, masks


# ══════════════════════════════════════════════════════════════════════════════
# Constrained PPO
# ══════════════════════════════════════════════════════════════════════════════

class ConstrainedPPO:
    """
    PPO with Lagrangian-multiplier constraints for:
      1. Equity: placements should favour high-population areas
      2. Budget: total placements ≤ budget (soft penalty)
    """

    def __init__(self, model, cfg=None, device: str = "cpu"):
        self.cfg = cfg or PPO_CFG
        self.model = model
        self.device = device

        self.optimizer = optim.Adam(model.parameters(), lr=self.cfg.lr)

        # Lagrangian multipliers (non-negative)
        self.lambda_equity = 0.1
        self.lambda_budget = 0.0  # budget is hard-limited by env; kept for flexibility

        self.buffer = RolloutBuffer()

        # Logging
        self.update_count = 0

    def store_transition(self, state, action, reward, log_prob, value, done,
                         feasible_mask):
        """Store a transition in the rollout buffer."""
        self.buffer.add(Transition(
            state=state,
            action=action,
            reward=reward,
            log_prob=log_prob if isinstance(log_prob, float) else log_prob.item(),
            value=value if isinstance(value, float) else value.item(),
            done=done,
            feasible_mask=feasible_mask,
        ))

    def update(self, graph_data, last_value: float = 0.0,
               equity_score: float = 0.0) -> Dict[str, float]:
        """
        Perform a PPO update using the current rollout buffer.

        Parameters
        ----------
        graph_data : torch_geometric.data.Data
            Graph structure (edge_index, node_rows, node_cols).
        last_value : float
            Bootstrap value for GAE if episode didn't terminate.
        equity_score : float
            Mean normalised population density of placed cells this episode.

        Returns
        -------
        metrics : dict with loss components
        """
        if len(self.buffer) == 0:
            return {}

        # ── Compute advantages ───────────────────────────────────────────
        advantages, returns = self.buffer.compute_gae(
            last_value, self.cfg.gamma, self.cfg.gae_lambda
        )

        # Normalise advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # ── Extract batch ────────────────────────────────────────────────
        (states, actions, old_log_probs,
         old_values, rewards, dones, masks) = self.buffer.get_batch()

        # Convert to tensors
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)

        # ── PPO update epochs ────────────────────────────────────────────
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        T = len(states)

        for epoch in range(self.cfg.update_epochs):
            # For simplicity, iterate over each timestep
            # (batch processing is complex with varying masks)
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy = 0.0

            for t in range(T):
                state_t = torch.tensor(states[t], dtype=torch.float32,
                                       device=self.device)
                mask_t = torch.tensor(masks[t], dtype=torch.bool,
                                      device=self.device)

                # Forward pass
                log_probs, value, entropy = self.model.evaluate_actions(
                    state_t, graph_data, actions_t[t:t+1], mask_t
                )

                # PPO clipped objective
                ratio = torch.exp(log_probs - old_log_probs_t[t])
                adv = advantages_t[t]

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio,
                                    1.0 + self.cfg.clip_ratio) * adv
                policy_loss = -torch.min(surr1, surr2)

                # Value loss
                value_loss = F.mse_loss(value, returns_t[t])

                # Entropy bonus
                entropy_bonus = entropy

                # Total loss (per-step)
                loss = (policy_loss +
                        self.cfg.value_coef * value_loss -
                        self.cfg.entropy_coef * entropy_bonus)

                # Add Lagrangian equity penalty
                equity_violation = max(0, self.cfg.equity_threshold - equity_score)
                loss = loss + self.lambda_equity * equity_violation

                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy += entropy_bonus.item()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.cfg.max_grad_norm)
                self.optimizer.step()

            total_policy_loss += epoch_policy_loss / T
            total_value_loss += epoch_value_loss / T
            total_entropy += epoch_entropy / T

        # ── Update Lagrangian multipliers ────────────────────────────────
        equity_violation = max(0, self.cfg.equity_threshold - equity_score)
        self.lambda_equity = max(0, self.lambda_equity +
                                 self.cfg.lagrangian_lr * equity_violation)

        # Clear buffer
        self.buffer.clear()
        self.update_count += 1

        n_epochs = self.cfg.update_epochs
        metrics = {
            "policy_loss": total_policy_loss / n_epochs,
            "value_loss": total_value_loss / n_epochs,
            "entropy": total_entropy / n_epochs,
            "lambda_equity": self.lambda_equity,
            "equity_violation": equity_violation,
            "mean_return": returns.mean(),
            "mean_advantage": advantages.mean(),
        }
        return metrics


# ── Needed by model.evaluate_actions ─────────────────────────────────────
F = torch.nn.functional


if __name__ == "__main__":
    print("[OK] PPO module loaded successfully")
    print(f"  Config: lr={PPO_CFG.lr}, clip={PPO_CFG.clip_ratio}, "
          f"entropy={PPO_CFG.entropy_coef}")
