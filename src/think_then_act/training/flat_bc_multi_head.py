"""
think_then_act.training.flat_bc_multi_head

Generalizes training/behavioral_cloning.py's BehavioralCloningTrainer to
train any of the 4 BC architectures being compared in the scaling
experiment (completion_rate vs. dataset size, at 100/500/1000 genuine
demonstrations) — "mse" (SubgoalRecurrentPolicy, byte-for-byte the same as
BehavioralCloningTrainer's own actor), "mdn", "autoregressive", "cvae"
(policy/flat_bc_heads.py). Deliberately a SEPARATE class from
BehavioralCloningTrainer, not a refactor of it — that class is already
validated and used elsewhere (chained_teacher_rollout's BC->RL warm-start
recipe); this one exists only for the architecture comparison and keeping
them independent means neither can silently break the other.

The masked-episode-minibatch training loop (pad to T_max per minibatch,
multiply the per-step loss by a 0/1 mask, normalize by the real step count)
is copied from BehavioralCloningTrainer.fit() verbatim — only the per-step
LOSS computation differs per policy_type, via _compute_loss's dispatch.
Same reasoning as that module's own docstring for why this needs to be
sequence-based (masked per-episode minibatches), not a shuffled-transition
loop: every one of these 4 heads conditions on recent history through the
SAME GRU hidden state, carried across a real episode's timesteps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MultiHeadBCConfig:
    obs_dim: int
    action_dim: int = 4
    hidden_dim: int = 64
    rnn_hidden_size: int = 64
    lr: float = 1e-3
    max_grad_norm: float = 1.0
    n_epochs: int = 20
    episodes_per_minibatch: int = 16
    policy_type: str = "mse"   # "mse" | "mdn" | "autoregressive" | "factored" | "cvae"
    mdn_components: int = 5
    mdn_decode_mode: str = "argmax"   # "argmax" | "weighted_mean" — see
                                  # MDNPolicy's own docstring for the tradeoff
    mdn_smoothness_weight: float = 0.0   # 0 = off (original behavior). >0
                                  # adds beta*mean_t(||decoded_t - decoded_{t-1}||^2)
                                  # to the NLL loss, decoded_t = the pi-weighted
                                  # mean (sum_k pi_k*mean_k) at each step — this
                                  # directly penalizes the mixture's own readout
                                  # jumping between consecutive timesteps, rather
                                  # than trying to fix it at decode time (both
                                  # argmax and weighted_mean decode were tried
                                  # and ruled out — pi itself flips abruptly
                                  # between steps with nothing in the NLL loss
                                  # discouraging that, confirmed via direct
                                  # trace inspection, 2026-10-02).
    vae_latent_dim: int = 8
    vae_beta: float = 1.0

    def as_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}


_VALID_POLICY_TYPES = ("mse", "mdn", "autoregressive", "factored", "cvae")


class MultiHeadBCTrainer:
    def __init__(self, config: MultiHeadBCConfig) -> None:
        if config.policy_type not in _VALID_POLICY_TYPES:
            raise ValueError(f"policy_type={config.policy_type!r} must be one of {_VALID_POLICY_TYPES}")
        self.config = config
        self._setup_model()

    def _setup_model(self) -> None:
        import torch
        from think_then_act.policy.subgoal_recurrent_policy import (
            SubgoalRecurrentPolicy, SubgoalRecurrentValueNetwork,
        )
        from think_then_act.policy.flat_bc_heads import MDNPolicy, AutoregressiveDiscretePolicy, CVAEPolicy

        c = self.config
        if c.policy_type == "mse":
            self.actor = SubgoalRecurrentPolicy(
                obs_dim=c.obs_dim, action_dim=c.action_dim,
                hidden_dim=c.hidden_dim, rnn_hidden_size=c.rnn_hidden_size,
            )
        elif c.policy_type == "mdn":
            self.actor = MDNPolicy(
                obs_dim=c.obs_dim, action_dim=c.action_dim, hidden_dim=c.hidden_dim,
                rnn_hidden_size=c.rnn_hidden_size, n_components=c.mdn_components,
                decode_mode=c.mdn_decode_mode,
            )
        elif c.policy_type == "autoregressive":
            self.actor = AutoregressiveDiscretePolicy(
                obs_dim=c.obs_dim, action_dim=c.action_dim,
                hidden_dim=c.hidden_dim, rnn_hidden_size=c.rnn_hidden_size,
                autoregressive=True,
            )
        elif c.policy_type == "factored":
            self.actor = AutoregressiveDiscretePolicy(
                obs_dim=c.obs_dim, action_dim=c.action_dim,
                hidden_dim=c.hidden_dim, rnn_hidden_size=c.rnn_hidden_size,
                autoregressive=False,
            )
        else:  # cvae
            self.actor = CVAEPolicy(
                obs_dim=c.obs_dim, action_dim=c.action_dim, hidden_dim=c.hidden_dim,
                rnn_hidden_size=c.rnn_hidden_size, latent_dim=c.vae_latent_dim,
            )
        # Untrained — see behavioral_cloning.py's docstring for why a fresh
        # critic is still saved (PPO always relearns V(s) regardless of
        # actor warm-start; costs nothing to include).
        self.critic = SubgoalRecurrentValueNetwork(
            obs_dim=c.obs_dim, hidden_dim=c.hidden_dim, rnn_hidden_size=c.rnn_hidden_size,
        )
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=c.lr)

    def _compute_loss(self, obs_batch, action_batch, mask):
        import torch

        c = self.config
        B = obs_batch.shape[0]
        hidden_state = torch.zeros(1, B, c.rnn_hidden_size)

        extra_loss = obs_batch.new_zeros(())

        if c.policy_type == "mse":
            mean, _, _ = self.actor.forward(obs_batch, hidden_state)
            predicted = torch.tanh(mean)
            per_step = ((predicted - action_batch) ** 2).sum(dim=-1)
        elif c.policy_type == "mdn":
            per_step, _ = self.actor.nll_loss(obs_batch, action_batch, hidden_state)
            if c.mdn_smoothness_weight > 0:
                # The pi-weighted readout at every step — same formula
                # MDNPolicy.act()'s weighted_mean decode uses, but kept
                # differentiable here (no torch.no_grad()) since this IS the
                # training signal, not an inference-time convenience.
                pi_logits, means, _, _ = self.actor.forward(obs_batch, hidden_state)
                pi = torch.softmax(pi_logits, dim=-1)
                decoded = (pi.unsqueeze(-1) * means).sum(dim=-2)              # (B, T, action_dim)
                diff = ((decoded[:, 1:] - decoded[:, :-1]) ** 2).sum(dim=-1)  # (B, T-1)
                pair_mask = mask[:, 1:] * mask[:, :-1]   # both steps of the pair must be
                                                          # real (not padding) and not
                                                          # cross an episode boundary —
                                                          # minibatches pad per-episode, so
                                                          # a pair straddling two different
                                                          # episodes never occurs within one
                                                          # row's contiguous real region.
                pair_mask_sum = pair_mask.sum().clamp(min=1.0)
                extra_loss = c.mdn_smoothness_weight * (pair_mask * diff).sum() / pair_mask_sum
        elif c.policy_type in ("autoregressive", "factored"):
            per_step, _ = self.actor.ce_loss(obs_batch, action_batch, hidden_state)
        else:  # cvae
            per_step, _ = self.actor.elbo_loss(obs_batch, action_batch, hidden_state, beta=c.vae_beta)

        mask_sum = mask.sum().clamp(min=1.0)
        return (mask * per_step).sum() / mask_sum + extra_loss

    def fit(self, demonstrations: list) -> dict:
        """
        Same demonstrations shape/contract as BehavioralCloningTrainer.fit()
        — {"obs": (T, obs_dim), "teacher_action": (T, action_dim), "weight":
        float (optional)}. Returns {"epoch_losses": list[float]}.
        """
        import torch

        if not demonstrations:
            raise ValueError("fit() called with zero demonstrations — nothing to train on.")

        c = self.config
        episodes = [
            {"obs": np.asarray(d["obs"], dtype=np.float32),
             "action": np.asarray(d["teacher_action"], dtype=np.float32),
             "weight": float(d.get("weight", 1.0)),
             "T": len(d["obs"])}
            for d in demonstrations
        ]
        n_eps = len(episodes)
        epoch_losses = []

        for _ in range(c.n_epochs):
            perm = torch.randperm(n_eps)
            batch_losses = []
            for start in range(0, n_eps, c.episodes_per_minibatch):
                idx = perm[start:start + c.episodes_per_minibatch]
                batch = [episodes[i] for i in idx.tolist()]
                T_max = max(ep["T"] for ep in batch)
                B = len(batch)

                obs_batch = torch.zeros(B, T_max, c.obs_dim)
                action_batch = torch.zeros(B, T_max, c.action_dim)
                mask = torch.zeros(B, T_max)
                for b, ep in enumerate(batch):
                    T = ep["T"]
                    obs_batch[b, :T] = torch.from_numpy(ep["obs"])
                    action_batch[b, :T] = torch.from_numpy(ep["action"])
                    mask[b, :T] = ep["weight"]

                loss = self._compute_loss(obs_batch, action_batch, mask)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), c.max_grad_norm)
                self.optimizer.step()

                batch_losses.append(float(loss.item()))

            epoch_losses.append(float(np.mean(batch_losses)))

        return {"epoch_losses": epoch_losses}

    def save_checkpoint(self, path: str) -> None:
        import os
        import torch
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "policy_type": self.config.policy_type,   # safety: load_checkpoint
                                  # checks this matches before loading weights
                                  # into a potentially-different architecture.
        }, path)

    def load_checkpoint(self, path: str) -> None:
        import torch
        ckpt = torch.load(path, map_location="cpu")
        saved_type = ckpt.get("policy_type")
        if saved_type is not None and saved_type != self.config.policy_type:
            raise ValueError(
                f"checkpoint was saved with policy_type={saved_type!r}, "
                f"this trainer is configured for policy_type={self.config.policy_type!r}"
            )
        self.actor.load_state_dict(ckpt["actor"])
        if "critic" in ckpt:
            self.critic.load_state_dict(ckpt["critic"])
