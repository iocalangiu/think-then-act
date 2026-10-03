"""
think_then_act.training.low_level_ppo_recurrent

Standalone PPO+GAE trainer for the recurrent, meta-RL-style subgoal policy
(SubgoalRecurrentPolicy) — see policy/subgoal_recurrent_policy.py and
training/singularity_force_env.py for what this trains against (memory:
meta_rl_sim2real_direction).

Standalone, NOT a subclass of training/low_level_ppo.py's
LowLevelPPOTrainer: that class's _setup_model/collect_rollouts/ppo_step are
monolithic with no extension seams a recurrent override could hook into
partially — subclassing would mean overriding nearly everything anyway
while inheriting only the generic checkpoint save/load methods. A
standalone file keeps this fully self-contained and auditable, at the cost
of some duplicated boilerplate (optimizer construction, timing wrapper) —
the right trade given the "never edit shared paths" constraint.

Key difference from LowLevelPPOTrainer.ppo_step: NO flattening across
episodes. Minibatches are formed over EPISODES (not individual timesteps),
zero-padded to each minibatch's longest episode, with a mask so padded
steps contribute nothing to any loss term. Per-episode GAE math is
UNCHANGED — reused directly from LowLevelPPOTrainer.compute_gae (a
@staticmethod with no self dependency).

Scope note — hidden state resets every episode (whole-episode BPTT, zero
initial state), not carried across episodes. This is WITHIN-episode
adaptation (detect-and-compensate for a discrepancy that appears partway
through one episode), not classic multi-trial RL^2 (which carries hidden
state ACROSS several episodes sharing one perturbation draw). A deliberate
scope choice: max_episode_steps is capped at 30 today, so full BPTT through
one episode is cheap and needs no truncated-chunk windowing; if episode
length grows substantially later, a bptt_chunk_len option would need new
batching logic here (the per-method hidden_state in/out plumbing on
SubgoalRecurrentPolicy already supports it without further interface
changes).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class RecurrentLowLevelPPOConfig:
    obs_dim         : int = 43   # base obs_dim_for_subgoal(subgoal) + wrapper's extra
                                  # dims — see SingularityForceAugmentedEnv.obs_dim_for
    action_dim      : int = 4
    hidden_dim      : int = 64
    rnn_hidden_size : int = 64   # unvalidated — start here, sweep (memorized risk:
                                  # too much capacity + too little perturbation
                                  # diversity can memorize training-seed shortcuts
                                  # instead of learning genuine online adaptation)

    n_rollouts        : int = 64
    max_episode_steps : int = 30

    gamma      : float = 0.99
    gae_lambda : float = 0.95

    lr             : float = 3e-4
    max_grad_norm  : float = 1.0
    entropy_coef   : float = 0.01
    vf_coef        : float = 0.5
    clip_eps       : float = 0.2
    n_epochs       : int = 4
    episodes_per_minibatch: int = 16   # minibatches count EPISODES, not steps —
                                        # see module docstring

    n_workers : int = 8   # rollout_workers.py process-pool size; 1 = serial, no pool

    def as_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}


class RecurrentLowLevelPPOTrainer:
    def __init__(self, config: RecurrentLowLevelPPOConfig) -> None:
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
        # Separate from self.optimizer deliberately — critic_warmup_step must
        # never touch the actor, and using a dedicated optimizer (rather than
        # zeroing/freezing actor.requires_grad around a shared optimizer)
        # means that guarantee holds structurally, not by convention.
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.lr)

    # ------------------------------------------------------------------
    # Rollout collection (recurrent-policy pool — see rollout_workers.py's
    # dedicated _worker_init_recurrent/collect_*_recurrent functions).
    # ------------------------------------------------------------------
    def _ensure_pool(self, env_kwargs: dict):
        from think_then_act.training import rollout_workers

        if self._pool is not None and self._pool_env_kwargs == env_kwargs:
            return self._pool
        self.close_pool()
        self._pool = rollout_workers.make_pool_recurrent(env_kwargs, self.config.n_workers)
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
            return rollout_workers.collect_serial_recurrent(
                actor_state, critic_state,
                self.config.obs_dim, self.config.action_dim, self.config.hidden_dim, self.config.rnn_hidden_size,
                seeds, env_kwargs,
            )

        pool = self._ensure_pool(env_kwargs)
        return rollout_workers.collect_with_pool_recurrent(
            pool, actor_state, critic_state,
            self.config.obs_dim, self.config.action_dim, self.config.hidden_dim, self.config.rnn_hidden_size,
            seeds,
        )

    # ------------------------------------------------------------------
    # Critic-only warmup — no policy update at all. Intended for a handful
    # of iterations before the first real ppo_step call when warm-starting
    # from a checkpoint whose critic was never actually trained (e.g.
    # behavioral_cloning.py's saved critic, which is fresh/random by
    # design — BC has no value function of its own). An untrained critic
    # gives near-random GAE advantages for the first several iterations of
    # ORDINARY joint actor+critic training, and a noisy/misleading policy
    # gradient derived from those advantages can meaningfully erode an
    # already-good warm-started actor before the critic catches up — see
    # scripts/train_low_level_ppo_recurrent.py's --critic-warmup-iters
    # (added 2026-09-18 after exactly this: a BC-warm-started align_xy run
    # regressed from ~45% success to 0% completion_rate within 50
    # iterations, with entropy never moving off its std=1 initial value).
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
    # Masked clipped-surrogate PPO loss for one (B, T_max, ...) padded
    # minibatch — factored out of ppo_step so the masking behavior itself
    # (padded steps must contribute exactly nothing) is directly unit-
    # testable without needing to go through episode-dict/T-inference
    # plumbing. Every episode's GRU hidden state starts from zero — see
    # module docstring's whole-episode-BPTT scope note.
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

    # ------------------------------------------------------------------
    # One PPO update: per-episode GAE (unchanged math), then masked,
    # padded, per-episode-sequence minibatch SGD with whole-episode BPTT.
    # ------------------------------------------------------------------
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

        # Global advantage normalization — same semantics as
        # LowLevelPPOTrainer.ppo_step (across every step of every collected
        # rollout this iteration), applied per-episode afterward instead of
        # into one flat array (no cross-episode flattening here — see
        # module docstring).
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
            "policy_loss"  : float(np.mean(metrics_accum["policy_loss"])),
            "value_loss"   : float(np.mean(metrics_accum["value_loss"])),
            "mean_entropy" : float(np.mean(metrics_accum["entropy"])),
            "approx_kl"    : float(np.mean(metrics_accum["approx_kl"])),
            "clip_fraction": float(np.mean(metrics_accum["clip_fraction"])),
            "mean_reward"  : float(np.mean(all_rewards)),
            "std_reward"   : float(np.std(all_rewards)),
            "mean_initial_d_grip_block": _mean_or_none("initial_d_grip_block"),
            "mean_final_d_grip_block"  : _mean_or_none("final_d_grip_block"),
            "mean_final_closedness"       : _mean_or_none("final_closedness"),
            "mean_final_translation_norm" : _mean_or_none("final_translation_norm"),
            "mean_initial_d_xy": _mean_or_none("initial_d_xy"),
            "mean_final_d_xy"  : _mean_or_none("final_d_xy"),
            "mean_initial_d_z" : _mean_or_none("initial_d_z"),
            "mean_final_d_z"   : _mean_or_none("final_d_z"),
            # Meta-RL/perturbation-specific diagnostics (see rollout_workers.py's
            # _run_episode_recurrent and the project's ablation protocol) — None/0
            # for every rollout where perturbation was never enabled this run.
            "mean_discrepancy_norm": _mean_or_none("mean_discrepancy_norm"),
            "force_onset_rate"     : float(np.mean([1.0 if r.get("had_force_onset") else 0.0 for r in rollouts])),
            "mean_recovery_steps"  : _mean_or_none("recovery_steps"),
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
