"""
think_then_act.policy.transformer_policy

Causal-self-attention counterpart to SubgoalRecurrentPolicy (the GRU-based
class every existing architecture in this project's BC comparison shares a
trunk with) -- built to test whether attention over the full in-episode
observation history helps vs. the GRU's single continuously-overwritten
hidden state. See memory: flat_policy_ppo_generalization's "architectural
limit, not just a data problem" finding -- a GRU structurally cannot do
genuine in-context learning (no explicit example channel, no content-
addressable memory); a transformer is the architecturally correct
alternative, which this class exists to actually test rather than just
argue about.

Deliberately lightweight to match this project's task scale: short
episodes (<=320 steps), 22-dim obs, 4-dim action -- 2 layers, d_model=64,
4 heads is already generous here, not a scaled-down "real" transformer.

Drop-in interface (forward/act) matching SubgoalRecurrentPolicy, with one
real semantic difference: forward(obs, hidden_state=None) IGNORES
hidden_state entirely. A transformer processes a whole (B, T, obs_dim)
chunk via self-attention in one shot -- there's no running state to thread
through during BC TRAINING, which always gets full padded episodes, so
MultiHeadBCTrainer's existing "mse"-style loss code (mean, _, _ =
self.actor.forward(obs_batch, hidden_state)) works UNCHANGED, simply
discarding the hidden_state it passes in. hidden_state only carries real
meaning in act() (single-step ROLLOUT use), where it's repurposed as the
growing buffer of this episode's past observations -- necessary because,
unlike a GRU, this architecture has no other mechanism for carrying
context between individual act() calls.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from think_then_act.policy.subgoal_recurrent_policy import LOG_STD_MIN, LOG_STD_MAX

ACTION_DIM = 4


class TransformerPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = ACTION_DIM, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, dim_feedforward: int = 128,
                 max_seq_len: int = 320):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.input_norm = nn.LayerNorm(obs_dim)
        self.obs_embed = nn.Linear(obs_dim, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        # dropout=0.0 is load-bearing, not just a regularization choice: a
        # nonzero dropout makes forward() STOCHASTIC depending on
        # self.training, and this project's PPO code calls it from two
        # different code paths that end up in different train/eval states
        # (rollout_workers.py explicitly .eval()s a separately-reconstructed
        # copy for sample(), but FlatTaskPPOTrainer's own self.actor is
        # never switched to eval() before recompute_log_prob() during
        # ppo_step()) -- confirmed directly (2026-10-05): with dropout
        # active, recompute_log_prob() on the SAME weights and SAME
        # (obs, raw_sample) pairs sample() had just produced disagreed by a
        # mean 0.24 / max 2.88 nats, nowhere near floating-point noise --
        # which alone explains the persistently inflated approx_kl and
        # saturated clip_frac seen in every PPO attempt so far. Every other
        # policy class in this project (GRU + Tanh + Linear, no dropout
        # anywhere) is deterministic given fixed weights regardless of
        # .training state; this keeps that same invariant here instead of
        # chasing eval()-call-site correctness forever.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.mean_head = nn.Linear(d_model, action_dim)
        # Same small-final-layer-init rationale as SubgoalRecurrentPolicy.
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.mean_head.bias)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

    def forward(self, obs: torch.Tensor, hidden_state=None) -> tuple:
        """
        obs: (B, obs_dim) single-step, or (B, T, obs_dim) chunk. Returns
        (mean, log_std, next_hidden_state=None) -- next_hidden_state is
        always None here; see module docstring for why act() below does
        NOT reuse this method's hidden_state handling at all.
        """
        squeeze_back = obs.dim() == 2
        if squeeze_back:
            obs = obs.unsqueeze(1)
        B, T, _ = obs.shape
        x = self.obs_embed(self.input_norm(obs))
        positions = torch.arange(T, device=obs.device).clamp(max=self.max_seq_len - 1)
        x = x + self.pos_embed(positions).unsqueeze(0)
        mask = self._causal_mask(T, obs.device)
        h = self.encoder(x, mask=mask)
        mean = self.mean_head(h)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).expand_as(mean)
        if squeeze_back:
            mean = mean.squeeze(1)
            log_std = log_std.squeeze(1)
        return mean, log_std, None

    def sample(self, obs: torch.Tensor, hidden_state=None) -> tuple:
        """
        Reparameterized sample, single-step ((B, obs_dim)) rollout-time use
        -- needed for PPO (rollout_workers.py's _run_episode_flat calls
        this, same as SubgoalRecurrentPolicy). hidden_state here follows
        THIS method's own contract, not act()'s buffer -- PPO's rollout
        loop carries whatever this returns as next_hidden straight back
        into the next sample() call, same as every other policy class, so
        using the (B, t, obs_dim) buffer convention here too keeps that
        loop unaware it's talking to a transformer at all.
        """
        squeeze_back = obs.dim() == 2
        obs_in = obs.unsqueeze(1) if squeeze_back else obs
        buf = obs_in if hidden_state is None else torch.cat([hidden_state, obs_in], dim=1)
        if buf.shape[1] > self.max_seq_len:
            buf = buf[:, -self.max_seq_len:, :]
        mean_seq, log_std_seq, _ = self.forward(buf)
        mean, log_std = mean_seq[:, -1, :], log_std_seq[:, -1, :]
        std = log_std.exp()
        raw_sample = mean + std * torch.randn_like(mean)
        action = torch.tanh(raw_sample)

        log_prob = self._gaussian_log_prob(raw_sample, mean, log_std)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        entropy = self._gaussian_entropy(log_std)

        return action, raw_sample, log_prob, entropy, buf

    def recompute_log_prob(self, obs: torch.Tensor, raw_sample: torch.Tensor, hidden_state=None) -> tuple:
        """
        Differentiable (log_prob, entropy, next_hidden_state) of a STORED
        raw_sample under the CURRENT policy params -- PPO training-time
        use (FlatTaskPPOTrainer._compute_masked_losses). obs/raw_sample:
        (B, T, obs_dim)/(B, T, action_dim) chunk-shaped -- PPO always
        passes a full minibatch chunk here, so this can call forward()
        directly on the whole chunk (hidden_state is accepted for
        interface symmetry but, like forward() itself, ignored -- a full
        chunk needs no running buffer, see module docstring).
        """
        mean, log_std, _ = self.forward(obs, hidden_state)
        log_prob = self._gaussian_log_prob(raw_sample, mean, log_std)
        action = torch.tanh(raw_sample)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        entropy = self._gaussian_entropy(log_std)
        return log_prob, entropy, None

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
        hidden_state here is the running (1, t, obs_dim) buffer of this
        episode's past observations -- NOT a GRU state, see module
        docstring. Appends the new obs, runs the transformer over the
        whole buffer (causally masked, so earlier positions are
        unaffected by the new one), and returns the LAST timestep's
        action. Callers must still pass the returned buffer into the next
        act() call and reset it to None at the start of each episode --
        same calling convention as every other recurrent-style policy in
        this project, just a different kind of state underneath.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).view(1, 1, -1)
                buf = obs_t if hidden_state is None else torch.cat([hidden_state, obs_t], dim=1)
                if buf.shape[1] > self.max_seq_len:
                    buf = buf[:, -self.max_seq_len:, :]
                mean, log_std, _ = self.forward(buf)
                last_mean, last_log_std = mean[:, -1, :], log_std[:, -1, :]
                if deterministic:
                    action = torch.tanh(last_mean)
                else:
                    std = last_log_std.exp()
                    raw = last_mean + std * torch.randn_like(last_mean)
                    action = torch.tanh(raw)
            return action.squeeze(0).numpy(), buf
        finally:
            if was_training:
                self.train()
