"""
think_then_act.training.flat_task_env

Gym wrapper that turns plain FetchPickAndPlace-v3 into the flat full-task
PPO training/eval environment: build_flat_observation's 22-dim obs in,
reward/flat_task_reward.py's dense reward out, same reset(rng=...)->
(obs, info) / step(action)->(obs, reward, terminated, truncated, info)
shape as training/subgoal_env.py's SubgoalConditionedEnv (the hierarchical
work's analogous per-subgoal wrapper), so this slots into the SAME
rollout-worker/eval patterns (training/rollout_workers.py's recurrent
helpers, scripts/train_low_level_ppo_recurrent.py's run_eval) with no
further adaptation needed on the caller side.

Deliberately NOT a subclass or extension of SubgoalConditionedEnv — that
class's whole purpose is per-subgoal setup/reward dispatch for the
hierarchical pivot, which has nothing in common with scoring the WHOLE
task at once; forcing a shared base would couple two things this project
has kept deliberately separate since the hierarchical pivot (see memory:
hierarchical_architecture).

info["done"] mirrors the project's standing genuine-grasp criterion (two-
finger contact AND lifted >lift_threshold, confirmed at ANY point so far
this episode) — the SAME key name SubgoalConditionedEnv's reward functions
use for their own "done" gate, so eval code written against that
convention (e.g. run_eval's `if info.get("done", False): success = True`)
works here unchanged.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from think_then_act.env.setup import grip_contact_forces
from think_then_act.training.subgoal_features import build_flat_observation, FLAT_OBS_DIM
from think_then_act.reward.flat_task_reward import (
    FlatTaskWeights, DEFAULT_FLAT_TASK_WEIGHTS, compute_flat_task_reward, drag_penalty,
)


class FlatTaskEnv(gym.Wrapper):
    obs_dim = FLAT_OBS_DIM

    def __init__(self, env, weights: FlatTaskWeights = DEFAULT_FLAT_TASK_WEIGHTS,
                 block_resting_z: float = 0.425, randomize_pose_prob: float = 0.0,
                 pose_exclude_band: float = 0.35, pose_max_frac: float = 0.85):
        """
        randomize_pose_prob: fraction of episodes that start from a
        randomized arm joint configuration (env.setup.randomize_joint_
        angles) instead of the one fixed pose every episode used before
        2026-10-03. Defaults to 0.0 — PRESERVES old behavior for any
        existing caller. Added after finding that warm-starting PPO from
        a pose-randomized BC checkpoint isn't enough on its own: if PPO's
        own ROLLOUT COLLECTION never samples a randomized pose either, it
        has no training signal keeping that generalization and can erode
        it over many iterations, same as how a fixed-pose-only demo pool
        produced a fixed-pose-only policy in the first place.
        """
        super().__init__(env)
        self.weights = weights
        self.block_resting_z = block_resting_z
        self.randomize_pose_prob = randomize_pose_prob
        self.pose_exclude_band = pose_exclude_band
        self.pose_max_frac = pose_max_frac
        self._ever_genuinely_grasped = False
        self._prev_grip_pos = None
        self._prev_block_xy = None

    def reset(self, *, rng=None, seed=None, options=None):
        from think_then_act.env.setup import init_random_episode, randomize_joint_angles

        self.env.reset(seed=seed)
        if rng is None:
            rng = np.random.default_rng()
        if rng.uniform(0.0, 1.0) < self.randomize_pose_prob:
            obs, pose_ok = randomize_joint_angles(
                self.env, rng, exclude_band=self.pose_exclude_band, max_frac=self.pose_max_frac,
            )
            if not pose_ok:
                return self.reset(rng=rng, seed=seed, options=options)
        obs, setup_ok = init_random_episode(self.env, rng)
        self._ever_genuinely_grasped = False
        self._prev_grip_pos = np.asarray(obs["observation"][:3], dtype=np.float64).copy()
        self._prev_block_xy = np.asarray(obs["achieved_goal"][:2], dtype=np.float64).copy()
        flat_obs = build_flat_observation(obs["observation"], obs["achieved_goal"], obs["desired_goal"])
        return flat_obs, {"setup_ok": setup_ok}

    def step(self, action):
        grip_pos_before = self._prev_grip_pos
        block_xy_before = self._prev_block_xy

        obs, _env_reward, terminated, truncated, info = self.env.step(action)
        forces = grip_contact_forces(self.env)

        reward, breakdown = compute_flat_task_reward(
            obs["observation"], obs["achieved_goal"], obs["desired_goal"],
            action, grip_pos_before, forces, self._ever_genuinely_grasped,
            self.block_resting_z, self.weights,
        )
        if breakdown["genuine_grasp_now"]:
            self._ever_genuinely_grasped = True

        block_xy = np.asarray(obs["achieved_goal"][:2], dtype=np.float64)
        block_xy_delta = float(np.linalg.norm(block_xy - block_xy_before))
        reward += drag_penalty(block_xy_delta, breakdown["genuine_grasp_now"], self.weights)

        self._prev_grip_pos = np.asarray(obs["observation"][:3], dtype=np.float64).copy()
        self._prev_block_xy = block_xy.copy()

        flat_obs = build_flat_observation(obs["observation"], obs["achieved_goal"], obs["desired_goal"])
        info = dict(info)
        info.update(breakdown)
        info["block_xy_delta"] = round(block_xy_delta, 5)
        info["done"] = self._ever_genuinely_grasped
        return flat_obs, reward, terminated, truncated, info
