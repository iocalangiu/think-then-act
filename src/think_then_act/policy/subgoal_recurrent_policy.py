"""
think_then_act.policy.subgoal_recurrent_policy

GRU-based recurrent counterpart to policy/subgoal_policy.py's
SubgoalGaussianPolicy, for the meta-RL experiment (see memory:
meta_rl_sim2real_direction, hierarchical_architecture). Implements the same
4-method contract (forward/sample/recompute_log_prob/act), each extended
with an optional hidden_state in / next_hidden_state out (None -> zeros),
so the recurrent hidden state can condition on the policy's own recent
(action, achieved-outcome) history within an episode — see
training/singularity_force_env.py for the discrepancy/force-onset
observation channels this is meant to key off of.

GRU chosen over LSTM: fewer parameters, and reasonable given the small
obs_dim (~40s) and short episodes (<=30 steps) this trains against.

_gaussian_log_prob/_gaussian_entropy and the action-squashing math are
copied (not imported) from subgoal_policy.py — same SAC-style tanh-squashed
Gaussian construction (Haarnoja et al. 2018, appendix C) — so this file has
no inheritance dependency on that module; a future change there can't
silently change this class's behavior.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

ACTION_DIM = 4
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _run_recurrent_trunk(input_norm, pre_gru, gru, obs: torch.Tensor, hidden_state, rnn_hidden_size: int):
    """
    Shared plumbing for both classes below: normalizes obs to (B, T, obs_dim)
    (accepting (B, obs_dim) single-step input too), defaults hidden_state to
    zeros, runs input_norm -> pre_gru -> gru, and un-normalizes the output
    rank to match the input. Returns (gru_out, next_hidden, squeeze_back) —
    callers apply their own head (mean_head / value_head) to gru_out and
    squeeze it themselves using the returned flag, since only they know
    their head's output width.

    Does NOT share weights between callers — each of SubgoalRecurrentPolicy
    and SubgoalRecurrentValueNetwork owns its own input_norm/pre_gru/gru
    modules (same standalone-no-sharing rationale as SubgoalValueNetwork
    vs. SubgoalGaussianPolicy in subgoal_policy.py) — this only factors out
    the rank-handling logic, not the parameters.
    """
    squeeze_back = obs.dim() == 2
    if squeeze_back:
        obs = obs.unsqueeze(1)   # (B, obs_dim) -> (B, 1, obs_dim)
    batch_size = obs.shape[0]
    if hidden_state is None:
        hidden_state = torch.zeros(1, batch_size, rnn_hidden_size, dtype=obs.dtype, device=obs.device)
    h = pre_gru(input_norm(obs))
    gru_out, next_hidden = gru(h, hidden_state)
    return gru_out, next_hidden, squeeze_back


class SubgoalRecurrentPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = ACTION_DIM, hidden_dim: int = 64, rnn_hidden_size: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.rnn_hidden_size = rnn_hidden_size

        self.input_norm = nn.LayerNorm(obs_dim)
        self.pre_gru = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh())
        self.gru = nn.GRU(hidden_dim, rnn_hidden_size, batch_first=True)
        self.mean_head = nn.Linear(rnn_hidden_size, action_dim)
        # Same small-final-layer-init rationale as SubgoalGaussianPolicy
        # (subgoal_policy.py) — keeps the initial mean near 0, tanh's
        # strongest-gradient region, instead of wherever default init lands.
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.mean_head.bias)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor, hidden_state=None) -> tuple:
        """
        obs: (B, obs_dim) single-step, or (B, T, obs_dim) chunk. Returns
        (mean, log_std, next_hidden_state) with mean/log_std matching obs's
        rank (squeezed back to (B, action_dim) if obs had no T dim).
        """
        gru_out, next_hidden, squeeze_back = _run_recurrent_trunk(
            self.input_norm, self.pre_gru, self.gru, obs, hidden_state, self.rnn_hidden_size
        )
        mean = self.mean_head(gru_out)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).expand_as(mean)
        if squeeze_back:
            mean = mean.squeeze(1)
            log_std = log_std.squeeze(1)
        return mean, log_std, next_hidden

    def sample(self, obs: torch.Tensor, hidden_state=None) -> tuple:
        """
        Reparameterized sample, single-step ((B, obs_dim)) rollout-time use.
        Returns (action, raw_sample, log_prob, entropy, next_hidden_state) —
        same shapes as SubgoalGaussianPolicy.sample, plus the trailing
        hidden state.
        """
        mean, log_std, next_hidden = self.forward(obs, hidden_state)
        std = log_std.exp()
        raw_sample = mean + std * torch.randn_like(mean)
        action = torch.tanh(raw_sample)

        log_prob = self._gaussian_log_prob(raw_sample, mean, log_std)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        entropy = self._gaussian_entropy(log_std)

        return action, raw_sample, log_prob, entropy, next_hidden

    def recompute_log_prob(self, obs: torch.Tensor, raw_sample: torch.Tensor, hidden_state=None) -> tuple:
        """
        Differentiable (log_prob, entropy, next_hidden_state) of a STORED
        raw_sample under the CURRENT policy params — training-time use.
        obs/raw_sample: (B, T, obs_dim)/(B, T, action_dim) chunk-shaped, or
        (B, obs_dim)/(B, action_dim) single-step. log_prob/entropy match
        obs's rank minus the last dim (e.g. (B, T) for a chunk).
        hidden_state: (1, B, rnn_hidden_size), the state CARRIED IN at the
        start of this chunk — callers are responsible for detaching it
        before passing it in if it came from a previous backward pass.
        """
        mean, log_std, next_hidden = self.forward(obs, hidden_state)
        log_prob = self._gaussian_log_prob(raw_sample, mean, log_std)
        action = torch.tanh(raw_sample)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        entropy = self._gaussian_entropy(log_std)
        return log_prob, entropy, next_hidden

    @staticmethod
    def _gaussian_log_prob(x: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        std = log_std.exp()
        log_prob = -0.5 * (((x - mean) / std) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return log_prob.sum(dim=-1)

    @staticmethod
    def _gaussian_entropy(log_std: torch.Tensor) -> torch.Tensor:
        return (0.5 * (1.0 + np.log(2 * np.pi)) + log_std).sum(dim=-1)

    def act(self, obs: np.ndarray, hidden_state=None, deterministic: bool = False) -> tuple:
        """
        Convenience: numpy obs -> (numpy action, next_hidden_state). Note
        the extra return value vs. SubgoalGaussianPolicy.act — a deliberate
        widened signature only this recurrent class has; callers must pass
        the returned hidden_state into the next act() call to actually get
        recurrent behavior, and reset it to None at the start of each
        episode.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                if deterministic:
                    mean, _, next_hidden = self.forward(obs_t, hidden_state)
                    action = torch.tanh(mean)
                else:
                    action, _, _, _, next_hidden = self.sample(obs_t, hidden_state)
            return action.squeeze(0).numpy(), next_hidden
        finally:
            if was_training:
                self.train()


class SubgoalRecurrentValueNetwork(nn.Module):
    """
    Recurrent critic, mirrors SubgoalValueNetwork (policy/subgoal_policy.py)
    with the same GRU trunk shape as SubgoalRecurrentPolicy, own separate
    weights — no sharing, same standalone-network rationale as the
    memoryless pair.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 64, rnn_hidden_size: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.input_norm = nn.LayerNorm(obs_dim)
        self.pre_gru = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh())
        self.gru = nn.GRU(hidden_dim, rnn_hidden_size, batch_first=True)
        self.value_head = nn.Linear(rnn_hidden_size, 1)

    def forward(self, obs: torch.Tensor, hidden_state=None) -> tuple:
        """
        Returns (value, next_hidden_state). value: (B,) for (B, obs_dim)
        input, (B, T) for (B, T, obs_dim) input.
        """
        gru_out, next_hidden, squeeze_back = _run_recurrent_trunk(
            self.input_norm, self.pre_gru, self.gru, obs, hidden_state, self.rnn_hidden_size
        )
        value = self.value_head(gru_out).squeeze(-1)
        if squeeze_back:
            value = value.squeeze(1)
        return value, next_hidden
