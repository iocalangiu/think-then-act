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

release-settle gating (added 2026-10-05): advancement to the next cube
(and the final done=True) used to fire off breakdown["at_target"] alone
-- a SINGLE-STEP check that only requires d_block_target <= threshold,
and "carrying" in flat_task_reward.py is a STICKY flag (stays True for
the rest of the episode once ever genuinely grasped), so this could
(and, confirmed via direct frame-by-frame video inspection against
multicube_stack_ppo_selfimitation_v1_best.pt, DID) fire while the cube
was still mid-air, held, just passing within 5cm of the target point --
not actually released or resting there. The episode then ends/advances
immediately, so nothing in training ever sees what happens when the
gripper actually opens: in the confirmed case, the cube was only
loosely within the 5cm sphere (not centered on the cube below it), and
rolled off to the side once released -- a real stack failure that
training/eval both silently counted as success. This directly
reproduces the gap eval_trustworthy_completion.py's grace-period check
was built to catch as a POST-HOC eval filter (see memory:
flat_policy_ppo_generalization) but which was never fed back into the
actual training signal -- user flagged this directly ("maybe thats why
ppo on 2 cube does no work, cause we stop it too early"). Fix: track
consecutive steps where the cube is UNASSISTED (gripper not holding,
via the same grip_contact_forces check used everywhere else in this
project) AND within move_to_target_threshold; only count a cube as
placed once that holds for release_settle_steps steps running (same
window eval_trustworthy_completion.py already used), matching training
and eval on the same definition of "done" for the first time.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from think_then_act.env.setup import grip_contact_forces, randomize_joint_angles, randomize_gripper_start_3d, teleport_block
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
        pose_scheme: str = "joint_angles",   # "joint_angles" (default) or "gripper_3d" --
                                  # see FlatTaskEnv's own docstring, 2026-10-04
        min_separation: float = 0.09,
        precision_weight: float = 10.0,
        disturbance_weight: float = 15.0,
        stack_xy_jitter: float = 0.03,
        release_settle_steps: int = 10,
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
        self.pose_scheme = pose_scheme
        self.min_separation = min_separation
        self.precision_weight = precision_weight
        self.disturbance_weight = disturbance_weight
        self.stack_xy_jitter = stack_xy_jitter
        self.release_settle_steps = release_settle_steps

        env = make_multicube_env(max_cubes, self.cube_half_sizes, max_episode_steps, render_mode=render_mode)
        super().__init__(env)

        self._n_active = 0
        self._order = []                # stacking order: list of physical cube indices, e.g. [2, 0, 1]
        self._active_pos_in_order = 0   # which LAYER we're currently working on
        self._resting_z = {}            # physical cube idx -> resting-center z on the bare table
        self._placed_xyz = {}           # physical cube idx -> frozen (x,y,z) once placed
        self._stack_xy = STACK_XY
        self._ever_grasped_this_cube = False
        self._all_done = False
        self._prev_grip_pos = None
        self._prev_block_xy = None
        self._step_count = 0
        self._settle_count = 0

    # ------------------------------------------------------------------
    def _site_name(self, physical_idx: int) -> str:
        return f"object{physical_idx}"

    def _active_physical_idx(self) -> int:
        return self._order[self._active_pos_in_order]

    def _dynamic_target(self, pos_in_order: int) -> np.ndarray:
        """
        Target for the cube at this position in the stacking order. Layer 0 targets the
        (jittered-once) stack point on the bare table -- nothing to stack ON yet, a fixed
        point is correct there. Layer k>0 targets directly on top of whatever the PREVIOUS
        cube's ACTUAL CURRENT position is, re-read fresh every call (this is called once per
        step() invocation, so it's naturally re-evaluated continuously, not just once at the
        goal switch) -- not a precomputed fixed height. So a previously-placed cube that gets
        nudged is correctly followed instead of the policy (or the reward computed against
        it) chasing a stale point it may no longer occupy.

        Added 2026-10-04 -- this env never had it until the user asked directly "are you
        still moving the second target if the first cube moves?" (it wasn't). The equivalent
        fix already existed in scripts/collect_stack2_scripted_trials_v2.py (the user's own
        idea, applied there first) but was never carried over here, meaning every PPO
        training/eval run and BC-alternation comparison done via THIS env (both the 100-demo
        and 400-demo rounds) was measured against a stale-target protocol -- those numbers
        need to be re-checked, not trusted as-is.
        """
        physical_idx = self._order[pos_in_order]
        if pos_in_order == 0:
            return np.array([self._stack_xy[0], self._stack_xy[1], self._resting_z[physical_idx]])
        prev_idx = self._order[pos_in_order - 1]
        prev_xyz = get_object_xyz(self.env, self._site_name(prev_idx))
        return np.array([
            prev_xyz[0], prev_xyz[1],
            prev_xyz[2] + self.cube_half_sizes[prev_idx] + self.cube_half_sizes[physical_idx],
        ])

    def reset(self, *, rng=None, seed=None, options=None):
        if rng is None:
            rng = np.random.default_rng()
        reset_obs, _ = self.env.reset(seed=seed)

        if rng.uniform(0.0, 1.0) < self.randomize_pose_prob:
            if self.pose_scheme == "gripper_3d":
                _, pose_ok, _ = randomize_gripper_start_3d(self.env, rng, reset_obs)
            else:
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
        self._settle_count = 0

        active_idx = self._active_physical_idx()
        desired_goal = self._dynamic_target(0)
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
        desired_goal = self._dynamic_target(self._active_pos_in_order)

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

        # Release-settle gating (see module docstring, 2026-10-05) -- a cube only
        # counts as "placed" once it has been UNASSISTED (gripper not holding it,
        # same contact check as everywhere else) and within threshold for
        # release_settle_steps CONSECUTIVE steps, not on the first instant
        # breakdown["at_target"] happens to be True (which "carrying" being sticky
        # let fire while still mid-air/held). Any step that breaks either
        # condition -- re-grasped, or drifted back out of tolerance -- resets the
        # counter, so a cube that gets bumped mid-settle has to genuinely re-settle.
        is_held_now = min(forces.get("left", 0.0), forces.get("right", 0.0)) > 0.0
        unassisted_in_tol = (not is_held_now) and breakdown["d_block_target"] <= self.weights.move_to_target_threshold
        self._settle_count = self._settle_count + 1 if unassisted_in_tol else 0
        settled = self._settle_count >= self.release_settle_steps

        switched = False
        if settled:
            self._placed_xyz[active_idx] = achieved_goal.copy()
            self._active_pos_in_order += 1
            self._ever_grasped_this_cube = False
            self._settle_count = 0
            switched = True
            if self._active_pos_in_order >= self._n_active:
                # STACK-INTEGRITY CHECK (added 2026-10-04, user caught the gap): the active
                # cube hitting its own target is NOT enough to call the whole stack done --
                # verify every PREVIOUSLY placed cube is still within tolerance of where it
                # was frozen too. Without this, a trial where cube N's own placement knocked
                # cube N-1 loose could still silently set done=True as long as cube N itself
                # landed correctly -- exactly the bug already found and fixed in
                # scripts/collect_stack2_scripted_trials_v2.py, which this env never had
                # (the `disturbance` term above is a REWARD penalty, it never gated success).
                all_still_placed = all(
                    float(np.linalg.norm(get_object_xyz(self.env, self._site_name(idx)) - placed_xyz))
                    <= self.weights.move_to_target_threshold
                    for idx, placed_xyz in self._placed_xyz.items()
                )
                if all_still_placed:
                    self._all_done = True
                    terminated = True
                else:
                    # Last cube placed, but an earlier one had drifted out of tolerance --
                    # not a success, and there's no next cube to advance to either
                    # (_active_pos_in_order is already == _n_active, out of range for
                    # _order). Nothing more productive can happen this episode;
                    # end it cleanly as a truncation rather than leaving _active_pos_in_order
                    # stuck out-of-range for every subsequent step (which crashed on the
                    # very first smoke test of this fix, 2026-10-04).
                    truncated = True

        # "switched" alone isn't enough to mean "advance to a next cube" -- when the LAST
        # cube was just placed, there's no next cube regardless of whether the stack-
        # integrity check above passed or failed.
        if switched and self._active_pos_in_order < self._n_active:
            next_idx = self._active_physical_idx()
            next_desired = self._dynamic_target(self._active_pos_in_order)
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
