"""
think_then_act.training.flat_task_ppo

PPO+GAE trainer for the flat full-task policy (SubgoalRecurrentPolicy at
obs_dim=FLAT_OBS_DIM=22, action_dim=4 — byte-for-byte the same class
training/flat_bc_multi_head.py's policy_type="mse" uses, so a BC
checkpoint from that trainer loads here directly) against
training/flat_task_env.py's FlatTaskEnv.

Standalone, NOT a subclass of RecurrentLowLevelPPOTrainer (training/
low_level_ppo_recurrent.py) — same rationale that class's own docstring
gives for not subclassing training/low_level_ppo.py: the rollout
collection, GAE/masked-loss math, and checkpoint I/O would all need
overriding anyway (different rollout_workers functions, different
per-episode diagnostics to aggregate), so inheriting would only share the
optimizer-construction boilerplate while still duplicating everything
that matters. Per-episode GAE math (compute_gae) and the masked
clipped-surrogate loss shape ARE copied from that file nearly verbatim —
this is the SAME established pattern, not a new one.
"""

from __future__ import annotations

import itertools

import numpy as np


class FlatTaskPPOConfig:
    def __init__(
        self,
        obs_dim: int = 22,   # FLAT_OBS_DIM
        action_dim: int = 4,
        hidden_dim: int = 64,
        rnn_hidden_size: int = 64,
        n_rollouts: int = 64,
        max_episode_steps: int = 100,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        lr: float = 3e-4,
        max_grad_norm: float = 1.0,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        clip_eps: float = 0.2,
        n_epochs: int = 4,
        episodes_per_minibatch: int = 16,
        n_workers: int = 8,
        weights_kwargs: dict = None,
        block_resting_z: float = 0.425,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.n_rollouts = n_rollouts
        self.max_episode_steps = max_episode_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.episodes_per_minibatch = episodes_per_minibatch
        self.n_workers = n_workers
        self.weights_kwargs = weights_kwargs or {}
        self.block_resting_z = block_resting_z

    def as_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}

    def env_kwargs(self) -> dict:
        return {
            "max_episode_steps": self.max_episode_steps,
            "weights_kwargs": self.weights_kwargs,
            "block_resting_z": self.block_resting_z,
        }


class FlatTaskPPOTrainer:
    def __init__(self, config: FlatTaskPPOConfig) -> None:
        self.config = config
        self._pool = None
        self._pool_env_kwargs = None
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
        self.critic = SubgoalRecurrentValueNetwork(
            obs_dim=self.config.obs_dim, hidden_dim=self.config.hidden_dim,
            rnn_hidden_size=self.config.rnn_hidden_size,
        )
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.actor.parameters(), self.critic.parameters()),
            lr=self.config.lr,
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.lr)

    # ------------------------------------------------------------------
    def _ensure_pool(self, env_kwargs: dict):
        from think_then_act.training import rollout_workers

        if self._pool is not None and self._pool_env_kwargs == env_kwargs:
            return self._pool
        self.close_pool()
        self._pool = rollout_workers.make_pool_flat(env_kwargs, self.config.n_workers)
        self._pool_env_kwargs = env_kwargs
        return self._pool

    def close_pool(self) -> None:
        if self._pool is not None:
            from think_then_act.training import rollout_workers
            rollout_workers.close_pool(self._pool)
            self._pool = None
            self._pool_env_kwargs = None

    def collect_rollouts(self, env_kwargs: dict, seeds: list) -> list:
        from think_then_act.training import rollout_workers

        actor_state  = {k: v.detach().cpu() for k, v in self.actor.state_dict().items()}
        critic_state = {k: v.detach().cpu() for k, v in self.critic.state_dict().items()}

        if self.config.n_workers <= 1:
            return rollout_workers.collect_serial_flat(
                actor_state, critic_state,
                self.config.obs_dim, self.config.action_dim, self.config.hidden_dim, self.config.rnn_hidden_size,
                seeds, env_kwargs,
            )

        pool = self._ensure_pool(env_kwargs)
        return rollout_workers.collect_with_pool_flat(
            pool, actor_state, critic_state,
            self.config.obs_dim, self.config.action_dim, self.config.hidden_dim, self.config.rnn_hidden_size,
            seeds,
        )

    # ------------------------------------------------------------------
    # Critic-only warmup — same rationale/necessity as
    # RecurrentLowLevelPPOTrainer.critic_warmup_step: a BC checkpoint's
    # critic is fresh/untrained (BC never touches it), and feeding the
    # first several PPO updates near-random GAE advantages from it can
    # erode an already-good warm-started actor before the critic catches
    # up (see that class's docstring for the exact regression this
    # prevents, found 2026-09-18 on the hierarchical align_xy policy).
    # ------------------------------------------------------------------
    def critic_warmup_step(self, rollouts: list) -> dict:
        import torch
        from think_then_act.training.low_level_ppo import LowLevelPPOTrainer

        episodes = []
        for r in rollouts:
            rewards = np.array([s["reward"] for s in r["steps"]], dtype=np.float64)
            values  = np.array([s["value"]  for s in r["steps"]], dtype=np.float64)
            _, returns = LowLevelPPOTrainer.compute_gae(
                rewards, values, r["bootstrap_value"], self.config.gamma, self.config.gae_lambda,
            )
            episodes.append({
                "obs": np.stack([s["obs"] for s in r["steps"]]).astype(np.float32),
                "ret": returns.astype(np.float32),
                "T": len(r["steps"]),
            })

        n_eps = len(episodes)
        value_losses = []
        for _ in range(self.config.n_epochs):
            perm = torch.randperm(n_eps)
            for start in range(0, n_eps, self.config.episodes_per_minibatch):
                idx = perm[start:start + self.config.episodes_per_minibatch]
                batch = [episodes[i] for i in idx.tolist()]
                T_max = max(ep["T"] for ep in batch)
                B = len(batch)

                obs_batch = torch.zeros(B, T_max, self.config.obs_dim)
                ret_batch = torch.zeros(B, T_max)
                mask = torch.zeros(B, T_max)
                for b, ep in enumerate(batch):
                    T = ep["T"]
                    obs_batch[b, :T] = torch.from_numpy(ep["obs"])
                    ret_batch[b, :T] = torch.from_numpy(ep["ret"])
                    mask[b, :T] = 1.0

                hidden_state = torch.zeros(1, B, self.config.rnn_hidden_size)
                values, _ = self.critic(obs_batch, hidden_state)
                mask_sum = mask.sum().clamp(min=1.0)
                value_loss = (mask * (values - ret_batch) ** 2).sum() / mask_sum

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()

                value_losses.append(float(value_loss.item()))

        return {"value_loss": float(np.mean(value_losses))}

    # ------------------------------------------------------------------
    def _compute_masked_losses(self, obs_batch, raw_batch, old_lp_batch, adv_batch, ret_batch, mask, clip_eps: float) -> dict:
        import torch

        B = obs_batch.shape[0]
        hidden_state = torch.zeros(1, B, self.config.rnn_hidden_size)
        log_probs, entropies, _ = self.actor.recompute_log_prob(obs_batch, raw_batch, hidden_state)
        values, _ = self.critic(obs_batch, hidden_state)

        ratio = torch.exp(log_probs - old_lp_batch)
        surr1 = ratio * adv_batch
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_batch
        mask_sum = mask.sum().clamp(min=1.0)
        policy_loss  = -(mask * torch.min(surr1, surr2)).sum() / mask_sum
        value_loss   = (mask * (values - ret_batch) ** 2).sum() / mask_sum
        entropy_mean = (mask * entropies).sum() / mask_sum

        with torch.no_grad():
            approx_kl = (((old_lp_batch - log_probs) * mask).sum() / mask_sum).item()
            clip_fraction = ((((torch.abs(ratio - 1.0) > clip_eps).float()) * mask).sum() / mask_sum).item()

        return {
            "policy_loss": policy_loss, "value_loss": value_loss, "entropy_mean": entropy_mean,
            "approx_kl": approx_kl, "clip_fraction": clip_fraction,
        }

    def ppo_step(self, rollouts: list) -> dict:
        import torch
        from think_then_act.training.low_level_ppo import LowLevelPPOTrainer

        episodes = []
        all_adv = []
        for r in rollouts:
            rewards = np.array([s["reward"] for s in r["steps"]], dtype=np.float64)
            values  = np.array([s["value"]  for s in r["steps"]], dtype=np.float64)
            advantages, returns = LowLevelPPOTrainer.compute_gae(
                rewards, values, r["bootstrap_value"], self.config.gamma, self.config.gae_lambda,
            )
            episodes.append({
                "obs"   : np.stack([s["obs"] for s in r["steps"]]).astype(np.float32),
                "raw"   : np.stack([s["raw_sample"] for s in r["steps"]]).astype(np.float32),
                "old_lp": np.array([s["old_log_prob"] for s in r["steps"]], dtype=np.float32),
                "adv"   : advantages.astype(np.float32),
                "ret"   : returns.astype(np.float32),
                "T"     : len(r["steps"]),
            })
            all_adv.append(advantages)

        flat_adv = np.concatenate(all_adv)
        adv_mean, adv_std = float(flat_adv.mean()), float(flat_adv.std()) + 1e-8
        for ep in episodes:
            ep["adv"] = (ep["adv"] - adv_mean) / adv_std

        clip_eps = self.config.clip_eps
        beta     = self.config.entropy_coef
        vf_coef  = self.config.vf_coef
        n_eps    = len(episodes)

        metrics_accum = {"policy_loss": [], "value_loss": [], "entropy": [],
                          "approx_kl": [], "clip_fraction": []}

        for _ in range(self.config.n_epochs):
            perm = torch.randperm(n_eps)
            for start in range(0, n_eps, self.config.episodes_per_minibatch):
                idx = perm[start:start + self.config.episodes_per_minibatch]
                batch = [episodes[i] for i in idx.tolist()]
                T_max = max(ep["T"] for ep in batch)
                B = len(batch)

                obs_batch    = torch.zeros(B, T_max, self.config.obs_dim)
                raw_batch    = torch.zeros(B, T_max, self.config.action_dim)
                old_lp_batch = torch.zeros(B, T_max)
                adv_batch    = torch.zeros(B, T_max)
                ret_batch    = torch.zeros(B, T_max)
                mask         = torch.zeros(B, T_max)
                for b, ep in enumerate(batch):
                    T = ep["T"]
                    obs_batch[b, :T]    = torch.from_numpy(ep["obs"])
                    raw_batch[b, :T]    = torch.from_numpy(ep["raw"])
                    old_lp_batch[b, :T] = torch.from_numpy(ep["old_lp"])
                    adv_batch[b, :T]    = torch.from_numpy(ep["adv"])
                    ret_batch[b, :T]    = torch.from_numpy(ep["ret"])
                    mask[b, :T] = 1.0

                losses = self._compute_masked_losses(
                    obs_batch, raw_batch, old_lp_batch, adv_batch, ret_batch, mask, clip_eps,
                )
                loss = losses["policy_loss"] + vf_coef * losses["value_loss"] - beta * losses["entropy_mean"]

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(self.actor.parameters(), self.critic.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                metrics_accum["policy_loss"].append(float(losses["policy_loss"].item()))
                metrics_accum["value_loss"].append(float(losses["value_loss"].item()))
                metrics_accum["entropy"].append(float(losses["entropy_mean"].item()))
                metrics_accum["approx_kl"].append(float(losses["approx_kl"]))
                metrics_accum["clip_fraction"].append(float(losses["clip_fraction"]))

        all_rewards = [r["total_reward"] for r in rollouts]

        def _mean_or_none(key):
            values = [r[key] for r in rollouts if r.get(key) is not None]
            return float(np.mean(values)) if values else None

        return {
            "policy_loss"       : float(np.mean(metrics_accum["policy_loss"])),
            "value_loss"        : float(np.mean(metrics_accum["value_loss"])),
            "mean_entropy"      : float(np.mean(metrics_accum["entropy"])),
            "approx_kl"         : float(np.mean(metrics_accum["approx_kl"])),
            "clip_fraction"     : float(np.mean(metrics_accum["clip_fraction"])),
            "mean_reward"       : float(np.mean(all_rewards)),
            "std_reward"        : float(np.std(all_rewards)),
            "genuine_success_rate": float(np.mean([1.0 if r.get("genuine_success") else 0.0 for r in rollouts])),
            "mean_discrepancy"  : _mean_or_none("mean_discrepancy"),
            "mean_drag_steps"   : _mean_or_none("drag_steps"),
        }

    def train_iteration(self, env_kwargs: dict, iteration: int) -> dict:
        import time

        seeds = list(range(
            self.config.n_rollouts * iteration,
            self.config.n_rollouts * (iteration + 1),
        ))

        t_collect = time.time()
        rollouts = self.collect_rollouts(env_kwargs, seeds)
        collect_s = time.time() - t_collect

        t_update = time.time()
        metrics = self.ppo_step(rollouts)
        update_s = time.time() - t_update

        metrics["iteration"] = iteration + 1
        metrics["collect_s"] = round(collect_s, 3)
        metrics["update_s"]  = round(update_s, 3)
        return metrics

    def save_checkpoint(self, path: str) -> None:
        import os
        import torch
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor" : self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load_checkpoint(self, path: str) -> None:
        import torch
        ckpt = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ckpt["actor"])
        if "critic" in ckpt:
            self.critic.load_state_dict(ckpt["critic"])
