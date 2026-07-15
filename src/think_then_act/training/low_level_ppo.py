"""
think_then_act.training.low_level_ppo

PPO+GAE trainer for the low-level, subgoal-conditioned continuous-control
policy — an alternative to training/low_level_grpo.py's GRPO trainer, not a
replacement (GRPO's code stays as-is, unused but intact; see bugs_and_fixes
memory, 2026-07-14, for why: GRPO broadcasts one whole-episode advantage
across all ~30 timesteps because it was designed for LLM RLHF, where a
"rollout" is one full generation scored ONCE and no finer-grained reward
exists. This task's reward IS dense per-timestep (reward/subgoal_reward.py),
so a real value-function baseline + GAE — the standard tool for exactly
this regime — is a better fit than GRPO's critic-free group-relative
baseline, which needs a fresh empirical mean/std per state every iteration
instead of a baseline that generalizes across states via a learned V(s)).

Unlike GRPO's collect_rollouts (rollouts grouped by shared start-state
seed, needed for its group-relative baseline), PPO rollouts here are each
an INDEPENDENT random reset — no grouping — since the baseline comes from
SubgoalValueNetwork instead of other rollouts sharing the same start.
Collection itself (needs a live env) is delegated to
training/rollout_workers.py, which parallelizes across a process pool —
collect_s dominates >99% of a GRPO iteration's wall time (see that
module's docstring), and the same is true here.

GAE / PPO math (this module) is pure tensor/numpy code, unit-testable
without mujoco/gymnasium — same split as low_level_grpo.py.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class LowLevelPPOConfig:
    obs_dim    : int = 38   # training.subgoal_features.SUBGOAL_OBS_DIM
    action_dim : int = 4
    hidden_dim : int = 64

    n_rollouts        : int = 64    # independent episodes collected per iteration
    max_episode_steps : int = 30

    gamma       : float = 0.99
    gae_lambda  : float = 0.95

    lr            : float = 3e-4
    max_grad_norm : float = 1.0
    entropy_coef  : float = 0.01   # beta, same convention as low_level_grpo.py
    vf_coef       : float = 0.5
    clip_eps      : float = 0.2
    n_epochs      : int = 4        # optimization passes over one collected batch
    minibatch_size: int = 256

    n_workers : int = 8   # rollout_workers.py process-pool size; 1 = serial, no pool

    def as_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}


class LowLevelPPOTrainer:
    def __init__(self, config: LowLevelPPOConfig) -> None:
        self.config = config
        self._pool = None          # lazily created — see _ensure_pool
        self._pool_env_kwargs = None
        self._setup_model()

    def _setup_model(self) -> None:
        import torch
        from think_then_act.policy.subgoal_policy import SubgoalGaussianPolicy, SubgoalValueNetwork

        self.actor = SubgoalGaussianPolicy(
            obs_dim=self.config.obs_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
        )
        self.critic = SubgoalValueNetwork(
            obs_dim=self.config.obs_dim,
            hidden_dim=self.config.hidden_dim,
        )
        # One optimizer over both modules' params — standard for small PPO
        # implementations (CleanRL etc.); no reason here to give the critic
        # its own learning rate given both are the same tiny MLP scale.
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.actor.parameters(), self.critic.parameters()),
            lr=self.config.lr,
        )

    # ------------------------------------------------------------------
    # GAE — pure numpy, no env/torch needed
    # ------------------------------------------------------------------
    @staticmethod
    def compute_gae(
        rewards: np.ndarray, values: np.ndarray, bootstrap_value: float,
        gamma: float, gae_lambda: float,
    ) -> tuple:
        """
        rewards, values: (T,) — values[t] is V(obs_t) at collection time.
        bootstrap_value: V(obs_T) if the rollout was cut off by the step
        limit (truncated, not a true terminal state) — 0.0 if it ended
        because the subgoal was actually achieved (terminated), since
        there's no real "continuation" to bootstrap from in that case.

        Returns (advantages, returns), each (T,): returns = advantages +
        values is the critic's regression target; advantages get
        batch-normalized by the caller (across ALL collected rollouts, not
        per-rollout — PPO has no grouping to normalize within, unlike
        GRPO).
        """
        T = len(rewards)
        advantages = np.empty(T, dtype=np.float64)
        next_value = bootstrap_value
        next_advantage = 0.0
        for t in range(T - 1, -1, -1):
            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = delta + gamma * gae_lambda * next_advantage
            next_value = values[t]
            next_advantage = advantages[t]
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # Rollout collection (needs a live env — delegated, parallelized)
    # ------------------------------------------------------------------
    def _ensure_pool(self, env_kwargs: dict):
        """
        Builds (once) a persistent process pool whose workers each own one
        long-lived MuJoCo env — see rollout_workers.py's docstring for why
        this must be built ONCE per subgoal, not once per iteration
        (env/GL-context construction is the expensive part; re-paying it
        every iteration would eat most of the parallelism's benefit).
        Rebuilds only if env_kwargs changes (i.e. a new subgoal).
        """
        from think_then_act.training import rollout_workers

        if self._pool is not None and self._pool_env_kwargs == env_kwargs:
            return self._pool
        self.close_pool()
        self._pool = rollout_workers.make_pool(env_kwargs, self.config.n_workers)
        self._pool_env_kwargs = env_kwargs
        return self._pool

    def close_pool(self) -> None:
        if self._pool is not None:
            from think_then_act.training import rollout_workers
            rollout_workers.close_pool(self._pool)
            self._pool = None
            self._pool_env_kwargs = None

    def collect_rollouts(self, env_kwargs: dict, seeds: list) -> list:
        """
        Runs len(seeds) INDEPENDENT episodes (each its own random reset —
        no shared-seed grouping, unlike GRPO) using the actor/critic's
        CURRENT weights, in parallel if config.n_workers > 1.
        """
        from think_then_act.training import rollout_workers

        actor_state  = {k: v.detach().cpu() for k, v in self.actor.state_dict().items()}
        critic_state = {k: v.detach().cpu() for k, v in self.critic.state_dict().items()}

        if self.config.n_workers <= 1:
            return rollout_workers.collect_serial(
                actor_state, critic_state,
                self.config.obs_dim, self.config.action_dim, self.config.hidden_dim,
                seeds, env_kwargs,
            )

        pool = self._ensure_pool(env_kwargs)
        return rollout_workers.collect_with_pool(
            pool, actor_state, critic_state,
            self.config.obs_dim, self.config.action_dim, self.config.hidden_dim,
            seeds,
        )

    # ------------------------------------------------------------------
    # Per-step log-prob / value / entropy (WITH gradient)
    # ------------------------------------------------------------------
    def _batch_log_prob_value_entropy(self, obs_batch, raw_batch) -> tuple:
        """obs_batch, raw_batch: (N, dim) tensors. Returns (log_probs (N,),
        values (N,), entropies (N,)), all differentiable."""
        log_probs, entropies = self.actor.recompute_log_prob(obs_batch, raw_batch)
        values = self.critic(obs_batch)
        return log_probs, values, entropies

    # ------------------------------------------------------------------
    # One PPO update: multiple epochs of clipped-surrogate minibatch SGD
    # over one collected batch of rollouts.
    # ------------------------------------------------------------------
    def ppo_step(self, rollouts: list) -> dict:
        import torch

        obs_list, raw_list, old_lp_list, adv_list, ret_list = [], [], [], [], []
        for r in rollouts:
            rewards = np.array([s["reward"] for s in r["steps"]], dtype=np.float64)
            values  = np.array([s["value"]  for s in r["steps"]], dtype=np.float64)
            advantages, returns = self.compute_gae(
                rewards, values, r["bootstrap_value"],
                self.config.gamma, self.config.gae_lambda,
            )
            for s, adv, ret in zip(r["steps"], advantages, returns):
                obs_list.append(s["obs"])
                raw_list.append(s["raw_sample"])
                old_lp_list.append(s["old_log_prob"])
                adv_list.append(adv)
                ret_list.append(ret)

        obs_all     = torch.from_numpy(np.stack(obs_list).astype(np.float32))
        raw_all     = torch.from_numpy(np.stack(raw_list).astype(np.float32))
        old_lp_all  = torch.tensor(old_lp_list, dtype=torch.float32)
        adv_all     = torch.tensor(adv_list, dtype=torch.float32)
        ret_all     = torch.tensor(ret_list, dtype=torch.float32)

        # Batch-normalized advantage — PPO has no group to normalize
        # within (unlike GRPO), so this is across every step of every
        # collected rollout this iteration.
        adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)

        N = obs_all.shape[0]
        minibatch_size = min(self.config.minibatch_size, N)
        clip_eps = self.config.clip_eps
        beta     = self.config.entropy_coef
        vf_coef  = self.config.vf_coef

        metrics_accum = {"policy_loss": [], "value_loss": [], "entropy": [],
                          "approx_kl": [], "clip_fraction": []}

        for _ in range(self.config.n_epochs):
            perm = torch.randperm(N)
            for start in range(0, N, minibatch_size):
                idx = perm[start:start + minibatch_size]

                log_probs, values, entropies = self._batch_log_prob_value_entropy(
                    obs_all[idx], raw_all[idx]
                )
                ratio = torch.exp(log_probs - old_lp_all[idx])
                surr1 = ratio * adv_all[idx]
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_all[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = ((values - ret_all[idx]) ** 2).mean()
                entropy_mean = entropies.mean()

                loss = policy_loss + vf_coef * value_loss - beta * entropy_mean

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(self.actor.parameters(), self.critic.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_lp_all[idx] - log_probs).mean().item()
                    clip_fraction = (torch.abs(ratio - 1.0) > clip_eps).float().mean().item()

                metrics_accum["policy_loss"].append(float(policy_loss.item()))
                metrics_accum["value_loss"].append(float(value_loss.item()))
                metrics_accum["entropy"].append(float(entropy_mean.item()))
                metrics_accum["approx_kl"].append(approx_kl)
                metrics_accum["clip_fraction"].append(clip_fraction)

        all_rewards = [r["total_reward"] for r in rollouts]
        return {
            "policy_loss"  : float(np.mean(metrics_accum["policy_loss"])),
            "value_loss"   : float(np.mean(metrics_accum["value_loss"])),
            "mean_entropy" : float(np.mean(metrics_accum["entropy"])),
            "approx_kl"    : float(np.mean(metrics_accum["approx_kl"])),
            "clip_fraction": float(np.mean(metrics_accum["clip_fraction"])),
            "mean_reward"  : float(np.mean(all_rewards)),
            "std_reward"   : float(np.std(all_rewards)),
        }

    # ------------------------------------------------------------------
    # One full training iteration
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Checkpoint I/O — actor + critic together, one file
    # ------------------------------------------------------------------
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
        self.critic.load_state_dict(ckpt["critic"])
