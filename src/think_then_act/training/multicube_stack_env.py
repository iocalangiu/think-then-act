"""
think_then_act.training.multicube_stack_env

PPO training environment for "learn to stack N cubes itself": unlike
scripts/collect_stack3_demos.py's SCRIPTED release-and-reposition
transition, THIS environment switches the active goal to the next cube
the instant the current one is genuinely placed -- part of the
environment's own dynamics, not a scripted behavior -- and the dense
reward (reused almost unchanged from reward/flat_task_reward.py, the
same terms that already fixed drag/idle-churn via PPO rather than
scripting) is left to teach release-and-continue through ordinary
reward-driven exploration, not imitation of hand-authored demos. See
memory: flat_policy_ppo_generalization, "multi-cube PPO env" section, for
why this replaces the demo-collection+BC-fine-tune approach tried first
(that one measurably broke the base skill without teaching the target
behavior; this one tries reward-driven discovery instead, the same
mechanism that already worked for drag/idle-churn on the single-cube
task, rather than more imitation data).

Cube count, which specific cubes are active, and their stacking order all
vary PER EPISODE (see reset()) so the policy can't shortcut via step-
counting or a memorized fixed sequence -- it has to recognize "the goal
just changed" from the pattern in its own observation stream, not from
"this is always the 3rd cube." This is ordinary trained generalization
(weights updated during training, frozen at deployment), NOT literal
in-context learning in the LLM/attention sense -- see the project history
around 2026-10-03 for why that distinction matters for this GRU
architecture specifically (no separate example-vs-live input channel, no
content-addressable memory).

Two NEW reward terms beyond flat_task_reward.py's existing ones (both
computed here in the wrapper, not added to the shared single-cube reward
function, since neither concept exists for a single object):
- precision_weight: extra reward for being CLOSER than move_to_target_
  threshold while at_target, not just within it -- added after directly
  observing (user watching the stacking videos, 2026-10-03) that cubes
  landing inside the 5cm success zone but not tightly enough caused the
  tower to topple. Zero outside at_target (reuses that same reward-phase
  gating as the existing stillness term).
- disturbance_weight: penalizes any ALREADY-PLACED cube moving from where
  it was frozen at placement time -- protects earlier layers from being
  knocked by the gripper while it works on a later cube, a distinct
  failure mode from "placed imprecisely to begin with."

Both new weights start at reasoned-but-unvalidated defaults, same as
every other reward term's history in this project -- check actual
training telemetry before trusting these numbers.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from think_then_act.env.setup import grip_contact_forces, randomize_joint_angles, teleport_block
from think_then_act.env.multicube import (
    make_multicube_env, object_observation, get_object_xyz, park_unused_cubes,
    DEFAULT_CUBE_HALF_SIZES, TABLE_TOP_Z,
)
from think_then_act.training.subgoal_features import build_flat_observation, FLAT_OBS_DIM
from think_then_act.reward.flat_task_reward import (
    FlatTaskWeights, DEFAULT_FLAT_TASK_WEIGHTS, compute_flat_task_reward, drag_penalty,
)

STACK_XY = (1.30, 0.75)
TABLE_CX, TABLE_CY, TABLE_R = 1.30, 0.75, 0.20


class MultiCubeStackEnv(gym.Wrapper):
    obs_dim = FLAT_OBS_DIM

    def __init__(
        self,
        weights: FlatTaskWeights = DEFAULT_FLAT_TASK_WEIGHTS,
        min_cubes: int = 1,
        max_cubes: int = 3,
        cube_half_sizes: list = None,
        max_episode_steps: int = 320,
        randomize_pose_prob: float = 1.0,
        pose_exclude_band: float = 0.35,
        pose_max_frac: float = 0.85,
        min_separation: float = 0.09,
        precision_weight: float = 10.0,
        disturbance_weight: float = 15.0,
        stack_xy_jitter: float = 0.03,
        render_mode: str = None,
    ) -> None:
        """
        max_cubes fixes the PHYSICAL model (max_cubes extra bodies +
        object0 = max_cubes+1 total) for this env instance's whole
        lifetime -- rebuilding the MJCF every episode to get a different
        cube count would mean recompiling the MuJoCo model on every
        reset(), too expensive for a PPO rollout worker. min_cubes/
        max_cubes instead bound how many of those bodies are ACTIVE each
        episode (randomized in reset(), see env/multicube.py's module
        docstring); the rest get parked off-table via park_unused_cubes.
        """
        self.weights = weights
        self.min_cubes = min_cubes
        self.max_cubes = max_cubes
        self.cube_half_sizes = cube_half_sizes or DEFAULT_CUBE_HALF_SIZES[:max_cubes + 1]
        assert len(self.cube_half_sizes) == max_cubes + 1
        self.max_episode_steps = max_episode_steps
        self.randomize_pose_prob = randomize_pose_prob
        self.pose_exclude_band = pose_exclude_band
        self.pose_max_frac = pose_max_frac
        self.min_separation = min_separation
        self.precision_weight = precision_weight
        self.disturbance_weight = disturbance_weight
        self.stack_xy_jitter = stack_xy_jitter

        env = make_multicube_env(max_cubes, self.cube_half_sizes, max_episode_steps, render_mode=render_mode)
        super().__init__(env)

        self._n_active = 0
        self._order = []                # stacking order: list of physical cube indices, e.g. [2, 0, 1]
        self._active_pos_in_order = 0   # which LAYER we're currently working on
        self._target_z = []             # per-layer target z, aligned with self._order
        self._resting_z = {}            # physical cube idx -> resting-center z on the bare table
        self._placed_xyz = {}           # physical cube idx -> frozen (x,y,z) once placed
        self._stack_xy = STACK_XY
        self._ever_grasped_this_cube = False
        self._all_done = False
        self._prev_grip_pos = None
        self._prev_block_xy = None
        self._step_count = 0

    # ------------------------------------------------------------------
    def _site_name(self, physical_idx: int) -> str:
        return f"object{physical_idx}"

    def _active_physical_idx(self) -> int:
        return self._order[self._active_pos_in_order]

    def reset(self, *, rng=None, seed=None, options=None):
        if rng is None:
            rng = np.random.default_rng()
        self.env.reset(seed=seed)

        if rng.uniform(0.0, 1.0) < self.randomize_pose_prob:
            _, pose_ok = randomize_joint_angles(
                self.env, rng, exclude_band=self.pose_exclude_band, max_frac=self.pose_max_frac,
            )
            if not pose_ok:
                return self.reset(rng=rng, seed=seed, options=options)

        self._n_active = int(rng.integers(self.min_cubes, self.max_cubes + 1))
        active_indices = [int(i) for i in rng.choice(self.max_cubes + 1, size=self._n_active, replace=False)]
        self._order = [int(i) for i in rng.permutation(active_indices)]
        park_unused_cubes(self.env, active_indices, self.max_cubes)

        stack_x = STACK_XY[0] + rng.uniform(-self.stack_xy_jitter, self.stack_xy_jitter)
        stack_y = STACK_XY[1] + rng.uniform(-self.stack_xy_jitter, self.stack_xy_jitter)
        self._stack_xy = (stack_x, stack_y)

        self._resting_z = {i: TABLE_TOP_Z + self.cube_half_sizes[i] for i in active_indices}
        full_heights = [2 * self.cube_half_sizes[i] for i in self._order]
        self._target_z = [
            TABLE_TOP_Z + sum(full_heights[:k]) + self.cube_half_sizes[self._order[k]]
            for k in range(self._n_active)
        ]

        # Non-overlapping XY start positions for the active cubes -- same
        # disk + pairwise-separation rejection loop as scripts/
        # eval_stack3_cubes.py / collect_stack3_demos.py.
        points = [np.array(self._stack_xy)]
        for _ in range(self._n_active):
            for _ in range(50):
                r = np.sqrt(rng.uniform(0.0, 1.0)) * TABLE_R
                theta = rng.uniform(0.0, 2.0 * np.pi)
                candidate = np.array([TABLE_CX + r * np.cos(theta), TABLE_CY + r * np.sin(theta)])
                if all(np.linalg.norm(candidate - p) > self.min_separation for p in points):
                    points.append(candidate)
                    break
            else:
                points.append(candidate)
        cube_xy = points[1:]
        for pos_in_order, physical_idx in enumerate(self._order):
            xyz = [cube_xy[pos_in_order][0], cube_xy[pos_in_order][1], self._resting_z[physical_idx]]
            teleport_block(self.env, xyz, joint_name=f"{self._site_name(physical_idx)}:joint")
        self.env.step([0.0, 0.0, 0.0, 0.0])

        self._active_pos_in_order = 0
        self._placed_xyz = {}
        self._ever_grasped_this_cube = False
        self._all_done = False
        self._step_count = 0

        active_idx = self._active_physical_idx()
        desired_goal = np.array([self._stack_xy[0], self._stack_xy[1], self._target_z[0]])
        self.env.unwrapped.goal = desired_goal.copy()
        observation, achieved_goal, desired = object_observation(self.env, self._site_name(active_idx), desired_goal)

        self._prev_grip_pos = np.asarray(observation[:3], dtype=np.float64).copy()
        self._prev_block_xy = np.asarray(achieved_goal[:2], dtype=np.float64).copy()

        flat_obs = build_flat_observation(observation, achieved_goal, desired)
        return flat_obs, {"setup_ok": True, "n_cubes": self._n_active}

    def step(self, action):
        self._step_count += 1
        grip_pos_before = self._prev_grip_pos
        block_xy_before = self._prev_block_xy

        active_idx = self._active_physical_idx()
        site_name = self._site_name(active_idx)
        desired_goal = np.array([self._stack_xy[0], self._stack_xy[1], self._target_z[self._active_pos_in_order]])

        _, _env_reward, terminated, truncated, info = self.env.step(action)
        observation, achieved_goal, desired = object_observation(self.env, site_name, desired_goal)
        forces = grip_contact_forces(self.env, block_body_name=site_name)

        reward, breakdown = compute_flat_task_reward(
            observation, achieved_goal, desired, action, grip_pos_before, forces,
            self._ever_grasped_this_cube, self._resting_z[active_idx], self.weights,
        )
        if breakdown["genuine_grasp_now"]:
            self._ever_grasped_this_cube = True

        block_xy = np.asarray(achieved_goal[:2], dtype=np.float64)
        block_xy_delta = float(np.linalg.norm(block_xy - block_xy_before))
        reward += drag_penalty(block_xy_delta, breakdown["genuine_grasp_now"], self.weights)

        if breakdown["at_target"]:
            precision_margin = max(0.0, self.weights.move_to_target_threshold - breakdown["d_block_target"])
            reward += self.precision_weight * precision_margin

        disturbance = 0.0
        for placed_idx, placed_xyz in self._placed_xyz.items():
            cur_xyz = get_object_xyz(self.env, self._site_name(placed_idx))
            disturbance += float(np.linalg.norm(cur_xyz - placed_xyz))
        if disturbance > 0.0:
            reward -= self.disturbance_weight * disturbance

        switched = False
        if breakdown["at_target"]:
            self._placed_xyz[active_idx] = achieved_goal.copy()
            self._active_pos_in_order += 1
            self._ever_grasped_this_cube = False
            switched = True
            if self._active_pos_in_order >= self._n_active:
                self._all_done = True
                terminated = True

        if switched and not self._all_done:
            next_idx = self._active_physical_idx()
            next_desired = np.array([self._stack_xy[0], self._stack_xy[1], self._target_z[self._active_pos_in_order]])
            self.env.unwrapped.goal = next_desired.copy()
            observation, achieved_goal, desired = object_observation(self.env, self._site_name(next_idx), next_desired)
            block_xy = np.asarray(achieved_goal[:2], dtype=np.float64)

        self._prev_grip_pos = np.asarray(observation[:3], dtype=np.float64).copy()
        self._prev_block_xy = block_xy.copy()

        if self._step_count >= self.max_episode_steps:
            truncated = True

        flat_obs = build_flat_observation(observation, achieved_goal, desired)
        info = dict(info)
        info.update(breakdown)
        info["block_xy_delta"] = round(block_xy_delta, 5)
        info["disturbance"] = round(disturbance, 5)
        info["n_cubes_placed"] = len(self._placed_xyz)
        info["n_cubes"] = self._n_active
        info["done"] = self._all_done
        return flat_obs, reward, terminated, truncated, info
