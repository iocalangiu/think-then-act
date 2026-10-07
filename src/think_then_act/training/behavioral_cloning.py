"""
think_then_act.training.behavioral_cloning

Sequence-based supervised training for SubgoalRecurrentPolicy against
FROZEN teacher-labeled demonstrations (training/chained_teacher_rollout.py)
— see memory: meta_rl_sim2real_direction. Pure PyTorch, no mujoco/gymnasium
needed (demonstrations are already-collected numpy arrays by the time this
runs) — unlike collection itself, this module IS unit-testable locally.

Why sequence-based, not shuffled-transition BC: SubgoalRecurrentPolicy's
whole point is conditioning on recent history via its GRU hidden state. A
plain per-transition BC loop (shuffle all (obs, action) pairs across every
episode, train independently) would never expose the network to a REAL
temporal sequence during training, so its hidden-state dynamics would
never actually get trained — it would just learn a memoryless mapping with
extra unused parameters. Same masked, padded, per-episode-minibatch
pattern as training/low_level_ppo_recurrent.py's PPO update, reused here
for consistency (and because it already solves "episodes have different
lengths" correctly).

No value-network TRAINING here — BC has no reward/advantage signal. A
fresh, untrained SubgoalRecurrentValueNetwork is still included in saved
checkpoints purely so they load directly into
RecurrentLowLevelPPOTrainer.load_checkpoint (which expects both "actor" and
"critic" keys) as a warm start for later RL fine-tuning — PPO always needs
to (re)learn V(s) early in training regardless of where the actor started,
so an untrained critic here costs nothing.

Loss target is the teacher's tanh-squashed action directly (not the
pre-tanh mean) — MSE(tanh(student_mean), teacher_action) — the standard,
simplest continuous-control BC objective; no need for the teacher's own
raw pre-tanh internals since SubgoalGaussianPolicy.act() already returns
the bounded action.

Episode-level success filtering (only clone from demonstrations where the
teacher chain actually reached the safe regime) happens BEFORE this module,
in chained_teacher_rollout.collect_successful_demonstrations — fit() itself
does no filtering, so a caller that hands it a failed episode will happily
clone it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BCConfig:
    obs_dim: int
    action_dim: int = 4
    hidden_dim: int = 64
    rnn_hidden_size: int = 64
    lr: float = 1e-3
    max_grad_norm: float = 1.0
    n_epochs: int = 20
    episodes_per_minibatch: int = 16

    def as_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}


class BehavioralCloningTrainer:
    def __init__(self, config: BCConfig) -> None:
        self.config = config
        self._setup_model()

    def _setup_model(self) -> None:
        import torch
        from think_then_act.policy.subgoal_recurrent_policy import (
            SubgoalRecurrentPolicy, SubgoalRecurrentValueNetwork,
        )

        self.actor = SubgoalRecurrentPolicy(
            obs_dim=self.config.obs_dim, action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim, rnn_hidden_size=self.config.rnn_hidden_size,
        )
        # Untrained — see module docstring on why a fresh critic is still saved.
        self.critic = SubgoalRecurrentValueNetwork(
            obs_dim=self.config.obs_dim, hidden_dim=self.config.hidden_dim,
            rnn_hidden_size=self.config.rnn_hidden_size,
        )
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.lr)

    def _compute_masked_mse(self, obs_batch, action_batch, mask):
        import torch

        B = obs_batch.shape[0]
        hidden_state = torch.zeros(1, B, self.config.rnn_hidden_size)
        mean, _, _ = self.actor.forward(obs_batch, hidden_state)
        predicted_action = torch.tanh(mean)

        squared_error = ((predicted_action - action_batch) ** 2).sum(dim=-1)   # (B, T)
        mask_sum = mask.sum().clamp(min=1.0)
        return (mask * squared_error).sum() / mask_sum

    def fit(self, demonstrations: list) -> dict:
        """
        demonstrations: list of {"obs": (T, obs_dim), "teacher_action": (T, action_dim),
        "weight": float (optional, default 1.0)} — only SUCCESSFUL episodes
        (see chained_teacher_rollout.collect_successful_demonstrations,
        which also attaches "weight" — see that module's
        compute_episode_weight for why: the success filter itself skews
        toward easier scenes, so harder survivors are upweighted to
        compensate). This function does no filtering of its own, and
        treats a missing "weight" key as 1.0 (unweighted). Returns
        {"epoch_losses": list[float]} (mean masked, weighted MSE per epoch,
        across all minibatches).
        """
        import torch

        if not demonstrations:
            raise ValueError(
                "fit() called with zero demonstrations — nothing to train on. "
                "Check the teacher chain's success_rate during collection; "
                "0 successes usually means the safe-regime thresholds are too "
                "tight, or the align_xy/descend checkpoints don't chain well."
            )

        episodes = [
            {"obs": np.asarray(d["obs"], dtype=np.float32),
             "action": np.asarray(d["teacher_action"], dtype=np.float32),
             "weight": float(d.get("weight", 1.0)),
             "T": len(d["obs"])}
            for d in demonstrations
        ]
        n_eps = len(episodes)
        epoch_losses = []

        for _ in range(self.config.n_epochs):
            perm = torch.randperm(n_eps)
            batch_losses = []
            for start in range(0, n_eps, self.config.episodes_per_minibatch):
                idx = perm[start:start + self.config.episodes_per_minibatch]
                batch = [episodes[i] for i in idx.tolist()]
                T_max = max(ep["T"] for ep in batch)
                B = len(batch)

                obs_batch = torch.zeros(B, T_max, self.config.obs_dim)
                action_batch = torch.zeros(B, T_max, self.config.action_dim)
                # mask doubles as the per-episode WEIGHT: real (non-padded)
                # positions get the episode's weight instead of a flat 1.0,
                # padded positions stay 0 — _compute_masked_mse's existing
                # (mask * error).sum() / mask.sum() is already a weighted
                # average, so this needs no change there at all.
                mask = torch.zeros(B, T_max)
                for b, ep in enumerate(batch):
                    T = ep["T"]
                    obs_batch[b, :T] = torch.from_numpy(ep["obs"])
                    action_batch[b, :T] = torch.from_numpy(ep["action"])
                    mask[b, :T] = ep["weight"]

                loss = self._compute_masked_mse(obs_batch, action_batch, mask)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
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
            "critic": self.critic.state_dict(),   # untrained — see module docstring
        }, path)

    def load_checkpoint(self, path: str) -> None:
        import torch
        ckpt = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ckpt["actor"])
        if "critic" in ckpt:
            self.critic.load_state_dict(ckpt["critic"])
