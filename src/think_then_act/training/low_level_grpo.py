"""
think_then_act.training.low_level_grpo

GRPO trainer for the low-level, subgoal-conditioned continuous-control
policy (SubgoalGaussianPolicy). Same group-relative-advantage algorithm as
training/grpo_trainer.py (the VLM's GRPO trainer) — group_size stochastic
rollouts per state, reward-normalized advantages, -mean(advantage*log_prob)
loss, no value function, no PPO-style clipping — just applied to a plain
Gaussian MLP instead of an autoregressive text-generating VLM.

Deliberately NOT stable-baselines3: SB3's own torch/gymnasium version
requirements repeatedly conflicted with this project's pins (torch==2.3.0
for the Qwen2-VL stack — see bugs_and_fixes memory, 2026-07-11). This task
is simple enough (dense reward, short horizon, small obs/action space) that
a from-scratch trainer is little extra code and removes the dependency
entirely — and it reuses an algorithm this project already implemented and
understands, rather than introducing a second one (PPO/SAC).

Unlike the VLM trainer, log_prob for an entire rollout can be recomputed in
ONE batched forward pass through the (tiny) MLP instead of per-step
backward+empty_cache — that memory-management dance in grpo_trainer.py
exists specifically for a multi-gigabyte VLM with long token sequences; it
doesn't apply here.

Advantage is PER-TIMESTEP return-to-go, not per-episode total reward. GRPO
was designed for LLM RLHF, where a "rollout" is one full completion scored
once — applying that same single scalar to every timestep's log-prob in a
~30-step control episode dilutes credit assignment across all steps equally,
regardless of which actions were actually good (observed in practice:
align_xy training stayed flat for 150+ iterations under the old whole-
episode version — see bugs_and_fixes memory, 2026-07-13). The baseline for
each timestep t is the mean/std of OTHER group members' return-to-go AT THE
SAME t (same start state, so directly comparable) rather than one pooled
group statistic — return-to-go shrinks toward 0 near an episode's end
regardless of action quality (fewer steps left to accumulate reward), so a
single pooled baseline across all t would systematically bias late-episode
steps to look better than early ones.

collect_rollouts is the only method needing a live env (mujoco/gymnasium) —
lazily imported inside the method, same pattern grpo_trainer.py already
uses, so the rest of this module (policy setup, grpo_step, entropy math) is
unit-testable without mujoco/gymnasium installed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class LowLevelGRPOConfig:
    obs_dim    : int = 38   # training.subgoal_features.SUBGOAL_OBS_DIM
    action_dim : int = 4
    hidden_dim : int = 64

    group_size        : int = 8    # G: rollouts per state
    n_states          : int = 4    # states per training iteration
    max_episode_steps : int = 30

    lr            : float = 3e-4
    max_grad_norm : float = 1.0
    entropy_coef  : float = 0.01   # beta — see grpo_trainer.py's GRPOConfig for the same field
    gamma         : float = 1.0    # return-to-go discount; 1.0 = undiscounted (episodes are short, <=30 steps)

    def as_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}


class LowLevelGRPOTrainer:
    def __init__(self, config: LowLevelGRPOConfig) -> None:
        self.config = config
        self._setup_model()

    def _setup_model(self) -> None:
        import torch
        from think_then_act.policy.subgoal_policy import SubgoalGaussianPolicy

        self.policy = SubgoalGaussianPolicy(
            obs_dim=self.config.obs_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.lr)

    # ------------------------------------------------------------------
    # Rollout collection (needs a live env — lazy import, no gradient)
    # ------------------------------------------------------------------
    def collect_rollouts(self, env, state_seed: int) -> list:
        """
        Runs group_size rollouts of `env` (a SubgoalConditionedEnv), each
        reset to the SAME randomized scene (fresh np.random.default_rng
        with the same seed every time, matching grpo_trainer.py's VLM
        collect_rollouts) so within-group reward variance reflects
        genuinely different sampled actions, not different starting
        conditions.
        """
        import torch

        rollouts = []
        self.policy.eval()
        try:
            for _ in range(self.config.group_size):
                rng = np.random.default_rng(state_seed)   # same seed every rollout in group
                obs, info = env.reset(rng=rng)

                steps = []
                for _ in range(self.config.max_episode_steps):
                    obs_arr = np.asarray(obs, dtype=np.float32)
                    obs_t = torch.from_numpy(obs_arr).unsqueeze(0)
                    with torch.no_grad():
                        action_t, raw_sample_t, _, _ = self.policy.sample(obs_t)
                    action = action_t.squeeze(0).numpy()

                    next_obs, reward, terminated, truncated, info = env.step(action)

                    steps.append({
                        "obs"       : obs_arr,
                        "raw_sample": raw_sample_t.squeeze(0).numpy(),
                        "reward"    : float(reward),
                    })
                    obs = next_obs
                    if terminated or truncated:
                        break

                rollouts.append({
                    "steps"       : steps,
                    "total_reward": sum(s["reward"] for s in steps),
                    "n_steps"     : len(steps),
                })
        finally:
            self.policy.train()

        return rollouts

    # ------------------------------------------------------------------
    # Log prob / entropy (WITH gradient) — pure tensor math, no env needed
    # ------------------------------------------------------------------
    def _rollout_log_prob_and_entropy(self, rollout: dict) -> tuple:
        """
        Batches a whole rollout's steps into one forward pass (cheap — a
        tiny MLP, unlike the VLM's per-step approach which exists to bound
        memory for a multi-gigabyte model). Returns (log_probs, mean_entropy):
        log_probs is (T,) — one PER-TIMESTEP log-prob, NOT summed, so the
        caller can weight each step by its own return-to-go advantage
        instead of one scalar for the whole episode. mean_entropy is a
        scalar. Both differentiable w.r.t. the CURRENT policy parameters.
        """
        import torch

        assert rollout["steps"], "rollout must have at least one step"
        obs_batch = torch.from_numpy(np.stack([s["obs"] for s in rollout["steps"]]))
        raw_batch = torch.from_numpy(np.stack([s["raw_sample"] for s in rollout["steps"]]))
        log_probs, entropies = self.policy.recompute_log_prob(obs_batch, raw_batch)
        return log_probs, entropies.mean()

    @staticmethod
    def _reward_to_go(rewards: list, gamma: float) -> np.ndarray:
        """Per-timestep return-to-go: returns[t] = sum_{k>=t} gamma**(k-t) * rewards[k]."""
        returns = np.empty(len(rewards), dtype=np.float64)
        running = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            running = rewards[t] + gamma * running
            returns[t] = running
        return returns

    @staticmethod
    def _group_relative_advantages(returns_to_go: list) -> list:
        """
        Per-timestep group-relative advantage: normalizes each rollout's
        return-to-go at step t against the mean/std of the OTHER rollouts'
        return-to-go at that SAME t (all rollouts in a group share the same
        start state/seed, so step t is directly comparable across them —
        this plays the role a value-function baseline would, without
        needing a critic network). Rollouts already terminated by step t
        are excluded from that t's baseline. See module docstring for why
        this is per-t rather than one pooled statistic.
        """
        max_t = max(len(rtg) for rtg in returns_to_go)
        means = np.empty(max_t)
        stds  = np.empty(max_t)
        for t in range(max_t):
            vals = np.array([rtg[t] for rtg in returns_to_go if len(rtg) > t])
            means[t] = vals.mean()
            stds[t]  = vals.std() + 1e-8
        return [(rtg - means[:len(rtg)]) / stds[:len(rtg)] for rtg in returns_to_go]

    # ------------------------------------------------------------------
    # GRPO gradient step — same math as grpo_trainer.py's grpo_step
    # ------------------------------------------------------------------
    def grpo_step(self, rollout_groups: list) -> dict:
        import torch

        self.optimizer.zero_grad()

        all_rewards, all_advantages = [], []
        within_group_stds = []
        entropy_terms = []
        total_loss        = torch.tensor(0.0)
        policy_loss_total = torch.tensor(0.0)
        total_rollouts = sum(len(g) for g in rollout_groups)
        beta = self.config.entropy_coef

        for group in rollout_groups:
            # Episode-total reward stats — logging only now (advantage below
            # is per-timestep return-to-go, not this).
            rewards = np.array([r["total_reward"] for r in group], dtype=np.float64)
            within_group_stds.append(float(rewards.std()))
            all_rewards.extend(rewards.tolist())

            returns_to_go = [
                self._reward_to_go([s["reward"] for s in r["steps"]], self.config.gamma)
                for r in group
            ]
            advantages = self._group_relative_advantages(returns_to_go)
            all_advantages.extend(np.concatenate(advantages).tolist())

            for rollout, adv in zip(group, advantages):
                log_probs, mean_entropy = self._rollout_log_prob_and_entropy(rollout)
                adv_t = torch.from_numpy(adv.astype(np.float32))

                policy_loss = (-adv_t * log_probs).sum() / total_rollouts
                # Subtracting the entropy bonus from a loss-to-minimize is
                # equivalent to adding beta*H to the maximized objective —
                # same convention as grpo_trainer.py's GRPOConfig.entropy_coef.
                loss = policy_loss - (beta * mean_entropy) / total_rollouts

                total_loss = total_loss + loss
                policy_loss_total = policy_loss_total + policy_loss
                entropy_terms.append(float(mean_entropy.item()))

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {
            "loss"          : float(total_loss.item()),
            "policy_loss"   : float(policy_loss_total.item()),
            "mean_entropy"  : float(np.mean(entropy_terms)) if entropy_terms else float("nan"),
            "mean_reward"   : float(np.mean(all_rewards)),
            "std_reward"    : float(np.std(all_rewards)),
            "mean_abs_adv"  : float(np.mean(np.abs(all_advantages))),
            "avg_within_std": float(np.mean(within_group_stds)),
        }

    # ------------------------------------------------------------------
    # One full training iteration
    # ------------------------------------------------------------------
    def train_iteration(self, env, iteration: int) -> dict:
        seeds = [self.config.n_states * iteration + i for i in range(self.config.n_states)]

        t_collect = time.time()
        rollout_groups = [self.collect_rollouts(env, seed) for seed in seeds]
        collect_s = time.time() - t_collect

        t_update = time.time()
        metrics = self.grpo_step(rollout_groups)
        update_s = time.time() - t_update

        metrics["iteration"] = iteration + 1
        metrics["collect_s"] = round(collect_s, 3)
        metrics["update_s"]  = round(update_s, 3)
        return metrics

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str) -> None:
        import os
        import torch
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        import torch
        self.policy.load_state_dict(torch.load(path, map_location="cpu"))
