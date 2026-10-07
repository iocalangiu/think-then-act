"""
think_then_act.policy.subgoal_policy

Small tanh-squashed Gaussian MLP policy for the low-level, subgoal-
conditioned controller (see memory: hierarchical_architecture.md). Trained
by training/low_level_grpo.py — a from-scratch GRPO trainer reusing the same
group-relative-advantage algorithm as training/grpo_trainer.py (the VLM's
trainer), not stable-baselines3 (tried and removed — SB3's torch/gymnasium
version requirements repeatedly conflicted with this project's pins; see
bugs_and_fixes memory, 2026-07-11).

Action squashing follows the standard SAC-style construction (Haarnoja et
al. 2018, appendix C): sample a raw Gaussian draw, squash through tanh to
land in (-1, 1)^action_dim (matching the env's actual action_space), and
correct log_prob for the tanh Jacobian. Entropy uses the closed-form
Gaussian differential entropy of the PRE-squash distribution — an
approximation that ignores the tanh Jacobian's effect on entropy, same
practice as SAC/PPO implementations generally use for the entropy bonus.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

ACTION_DIM = 4
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class SubgoalGaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = ACTION_DIM, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # build_subgoal_observation concatenates raw, unnormalized values —
        # absolute gripper/goal positions (~0.4-1.6m, not zero-centered),
        # euler angles, velocities, a one-hot, all on different scales. Fed
        # directly into a Tanh trunk, that risks saturating early units at
        # init (large inputs -> tanh output near +-1 -> ~0 gradient), which
        # would cap how much signal reaches mean_head as training
        # progresses. LayerNorm learns a per-feature scale/shift from data
        # instead of hardcoding one, and works per-sample (batch size 1
        # during rollout collection is fine, unlike BatchNorm).
        self.input_norm = nn.LayerNorm(obs_dim)

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        # PyTorch's default nn.Linear init (Kaiming-uniform, roughly
        # U(-1/sqrt(hidden_dim), 1/sqrt(hidden_dim)) per weight) gives no
        # guarantee the mean output starts near 0 — with hidden_dim=64
        # inputs summed through it, the pre-tanh mean can easily start with
        # enough magnitude to already sit partway into tanh's saturated
        # region (found 2026-08-13: a size-randomized close_gripper run
        # produced a state-INDEPENDENT, near-±1-saturated action at iter50
        # AND iter200, unchanged by a size curriculum or a 5x higher
        # entropy_coef — neither touches the MEAN's starting point, only
        # exploration/task-difficulty, so neither could pull an
        # already-saturated mean back through tanh's near-zero gradient
        # there). Small explicit init keeps the initial mean close to 0
        # (tanh's strongest-gradient region, derivative ~1.0) instead of
        # wherever default init happens to land — same small-final-layer-
        # init convention used in DDPG/TD3/SAC-style continuous-control
        # actors specifically to avoid this failure mode.
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.mean_head.bias)
        # State-independent learned log_std — standard for simple continuous
        # control (vanilla policy-gradient/PPO style); simpler than a
        # state-dependent std head and makes entropy a direct closed-form
        # function of log_std alone.
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> tuple:
        """obs: (B, obs_dim). Returns (mean, log_std), each (B, action_dim)."""
        h = self.trunk(self.input_norm(obs))
        mean = self.mean_head(h)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).expand_as(mean)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple:
        """
        Reparameterized sample.

        Returns (action, raw_sample, log_prob, entropy):
          action     : tanh-squashed, in (-1, 1)^action_dim, (B, action_dim)
          raw_sample : pre-tanh Gaussian draw, (B, action_dim) — STORE this
                       (not `action`) to exactly recompute log_prob later,
                       since tanh isn't invertible at its asymptotes
          log_prob   : (B,), tanh-corrected
          entropy    : (B,), closed-form pre-squash Gaussian entropy
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        raw_sample = mean + std * eps
        action = torch.tanh(raw_sample)

        log_prob = self._gaussian_log_prob(raw_sample, mean, log_std)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

        entropy = self._gaussian_entropy(log_std)

        return action, raw_sample, log_prob, entropy

    def recompute_log_prob(self, obs: torch.Tensor, raw_sample: torch.Tensor) -> tuple:
        """
        Differentiable (log_prob, entropy) of a STORED raw_sample under the
        CURRENT policy parameters — used at gradient time in
        low_level_grpo.py, mirroring grpo_trainer.py's
        _step_log_prob_and_entropy pattern: rollout collection runs under
        torch.no_grad(), so log_prob has to be recomputed post-hoc for the
        backward pass, not reused from collection.
        """
        mean, log_std = self.forward(obs)
        log_prob = self._gaussian_log_prob(raw_sample, mean, log_std)
        action = torch.tanh(raw_sample)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        entropy = self._gaussian_entropy(log_std)
        return log_prob, entropy

    @staticmethod
    def _gaussian_log_prob(x: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        """Sum of per-dimension diagonal Gaussian log-density, over the last dim."""
        std = log_std.exp()
        log_prob = -0.5 * (((x - mean) / std) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return log_prob.sum(dim=-1)

    @staticmethod
    def _gaussian_entropy(log_std: torch.Tensor) -> torch.Tensor:
        """Closed-form differential entropy of a diagonal Gaussian, summed over dims."""
        return (0.5 * (1.0 + np.log(2 * np.pi)) + log_std).sum(dim=-1)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Convenience: numpy obs -> numpy action, for eval/rollout use."""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                if deterministic:
                    mean, _ = self.forward(obs_t)
                    action = torch.tanh(mean)
                else:
                    action, _, _, _ = self.sample(obs_t)
            return action.squeeze(0).numpy()
        finally:
            if was_training:
                self.train()


class SubgoalValueNetwork(nn.Module):
    """
    Critic for training/low_level_ppo.py's GAE advantage estimation — same
    obs schema and trunk shape as SubgoalGaussianPolicy (LayerNorm then two
    Tanh layers) but its own separate weights, no sharing. Kept as a
    standalone module (not a second head bolted onto SubgoalGaussianPolicy)
    so GRPO's actor-only checkpoints and PPO's actor+critic checkpoints stay
    structurally independent — loading one never has to know about the
    other.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.input_norm = nn.LayerNorm(obs_dim)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, obs_dim). Returns V(s), shape (B,) — squeezed, not (B, 1)."""
        h = self.trunk(self.input_norm(obs))
        return self.value_head(h).squeeze(-1)
