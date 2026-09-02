"""
think_then_act.training.subgoal_features

Pure functions for assembling the low-level controller's input vector.
Deliberately has NO gymnasium/mujoco import — subgoal_env.py (the actual
gym.Wrapper) depends on this, not the other way around, so this half of the
logic is unit-testable without a live env or the mujoco/gymnasium stack
(neither is installed on the local dev machine — see env/wrapper.py's
docstring for why the project keeps this split).

Observation vector layout for every subgoal EXCEPT close_gripper and
RELATIVE_OBS_SUBGOALS:
    obs(25) + achieved_goal(3) + desired_goal(3) + subgoal one-hot(6) + collision_prob(1) = 38

RELATIVE_OBS_SUBGOALS ({"align_xy", "descend"} as of 2026-09-01 — see
RELATIVE_OBS_DIM's comment below) use a SMALLER, frame-relative 29-dim
layout instead: no absolute grip_pos/object_pos, achieved_goal/desired_goal
collapsed into one delta. align_xy was the first (retrained 2026-09-01
after a real-UR3e rollout exposed the coordinate-frame issue); descend
followed once align_xy's fix was validated — both subgoals' reward
functions (reward_align_xy, reward_descend) only ever read RELATIVE
geometry (_geometry()'s d_xy/d_z, never absolute grip_pos/object_pos), so
the same fix generalizes cleanly. Deliberately NOT extended to
close_gripper/lift/move_to_target/release yet — that's a separate,
not-yet-made decision, not an oversight.

close_gripper gets 3 extra floats appended — a perceived (noisy) reading of
the block's own [width, length, height] (env.block_randomization), added
2026-08-10 for block-size randomization (see hierarchical_architecture
memory, Stage 4). Deliberately NOT added to the other 5 subgoals: nothing
requires a uniform vector length across subgoals (each trains its own
separate policy network with its own obs_dim, and nothing stacks different
subgoals' observations together), so padding the other five with an unused
zero-filler would only force retraining their already-converged checkpoints
for no benefit. See close_gripper's own reward (reward_close_gripper) for
why it needs this and the others don't: it's the one subgoal whose correct
behavior (how far to close) actually depends on the object's shape, mirrors
how a real deployment would only have this from the perception module once
the gripper is actually approaching/holding the object.
"""

from __future__ import annotations
import numpy as np

from think_then_act.reward.subgoal_reward import SUBGOAL_LABELS

SUBGOAL_OBS_DIM        = 25 + 3 + 3 + len(SUBGOAL_LABELS) + 1
CLOSE_GRIPPER_OBS_DIM  = SUBGOAL_OBS_DIM + 3  # + perceived [width, length, height]

# RELATIVE_OBS_SUBGOALS, started 2026-09-01 with align_xy: every OTHER
# subgoal still uses SUBGOAL_OBS_DIM's absolute grip_pos(3)/object_pos(3) +
# separate achieved_goal(3)/desired_goal(3). align_xy's own reward
# (reward_align_xy) — and descend's (reward_descend), added to this set the
# same day once align_xy's fix was validated — never use absolute
# position, only relative geometry (block_pos - grip_pos, via the shared
# _geometry() helper in subgoal_reward.py) — so those absolute values were
# pure sim2real liability: the real UR3e's base_link-frame coordinates
# (~0.1-0.5m) are a completely different numeric range than Fetch's
# world-frame training data (~0.7-2.0m), and SubgoalGaussianPolicy.
# input_norm's per-SAMPLE LayerNorm has no way to correct for that (see
# memory ur3e_sim2real.md, 2026-09-01 real-robot rollout finding). Dropping
# grip_pos/object_pos and collapsing achieved_goal+desired_goal into one
# (desired-achieved) delta makes the observation invariant to the
# coordinate frame's origin by construction — translating everything by
# any constant vector leaves every fed feature unchanged. Deliberately NOT
# extended to close_gripper/lift/move_to_target/release yet — a separate,
# not-yet-made decision for each (close_gripper's reward is NOT purely
# relative — see reward_close_gripper's own grip_force/closedness terms —
# so it may not even generalize the same way without more thought).
#   layout: object_rel_pos(3) + gripper_state(2) + object_rot(3)
#         + object_velp(3) + object_velr(3) + grip_velp(3) + gripper_vel(2)
#         + goal_rel_pos(3) + subgoal one-hot(6) + collision_prob(1) = 29
RELATIVE_OBS_SUBGOALS = {"align_xy", "descend"}
RELATIVE_OBS_DIM = SUBGOAL_OBS_DIM - 9   # -grip_pos(3) -object_pos(3) -(achieved+desired -> delta)(3)
ALIGN_XY_OBS_DIM = RELATIVE_OBS_DIM      # kept as an alias — existing callers/tests reference this name


def obs_dim_for_subgoal(subgoal: str) -> int:
    """
    The one place that knows which obs dim a given subgoal's policy/critic
    network needs — every script that builds a SubgoalGaussianPolicy/
    SubgoalValueNetwork for a KNOWN specific subgoal should call this
    instead of hardcoding SUBGOAL_OBS_DIM, or a close_gripper/
    RELATIVE_OBS_SUBGOALS checkpoint (a different width) will mismatch the
    network's actual input layer shape.
    """
    if subgoal not in SUBGOAL_LABELS:
        raise ValueError(f"Unknown subgoal {subgoal!r}; must be one of {SUBGOAL_LABELS}")
    if subgoal == "close_gripper":
        return CLOSE_GRIPPER_OBS_DIM
    if subgoal in RELATIVE_OBS_SUBGOALS:
        return RELATIVE_OBS_DIM
    return SUBGOAL_OBS_DIM


def subgoal_to_onehot(subgoal: str) -> np.ndarray:
    if subgoal not in SUBGOAL_LABELS:
        raise ValueError(f"Unknown subgoal {subgoal!r}; must be one of {SUBGOAL_LABELS}")
    onehot = np.zeros(len(SUBGOAL_LABELS), dtype=np.float32)
    onehot[SUBGOAL_LABELS.index(subgoal)] = 1.0
    return onehot


def sanitize_observation_for_perception(observation, block_pos) -> np.ndarray:
    """
    Returns a COPY of `observation` (the raw 25-dim FetchPickAndPlace-v3
    vector) with the object-position-derived slices overwritten using
    `block_pos` (a perceived estimate) instead of whatever privileged values
    the underlying env baked in.

    FetchPickAndPlace-v3's own _get_obs() (gymnasium_robotics/envs/fetch/
    fetch_env.py, confirmed directly against the installed
    gymnasium-robotics==1.3.1 source, 2026-07-23) concatenates:
        grip_pos(3) + object_pos(3) + object_rel_pos(3) + gripper_state(2)
        + object_rot(3) + object_velp(3) + object_velr(3) + grip_velp(3)
        + gripper_vel(2)  =  25 floats
    `object_pos` (indices 3:6) and `object_rel_pos` (indices 6:9, =
    object_pos - grip_pos) are both read straight from live physics
    (sim.data.get_site_xpos) inside that call — completely independent of,
    and untouched by, subgoal_env.py's separate achieved_goal substitution.
    Swapping ONLY achieved_goal (as build_subgoal_observation alone did
    before this function existed) leaves TWO exact, uncorrupted copies of
    the true block position sitting in the observation the policy actually
    receives, making a perception-noise evaluation meaningless. Confirmed
    empirically 2026-07-23: an align_xy rollout barely changed despite
    ~6-8cm of achieved_goal error, because the policy could — and evidently
    did — still read the real position straight out of obs[6:9].

    Does NOT touch object_rot/object_velp/object_velr (indices 11:20) —
    those are also technically privileged (read from real physics too), but
    fixing them needs temporal differencing of the perceived position across
    steps (velocity can't be derived from a single frame the way position
    can) — a materially bigger change deliberately deferred until
    position-only perception is validated.
    """
    sanitized = np.array(observation, dtype=np.float32, copy=True)
    grip_pos  = sanitized[0:3]
    block_pos = np.asarray(block_pos, dtype=np.float32).reshape(3)
    sanitized[3:6] = block_pos
    sanitized[6:9] = block_pos - grip_pos
    return sanitized


def build_subgoal_observation(
    observation, achieved_goal, desired_goal,
    subgoal: str, collision_prob: float,
    block_dims=None,
) -> np.ndarray:
    """
    observation   : (25,) raw FetchPickAndPlace observation vector
    achieved_goal : (3,) block XYZ
    desired_goal  : (3,) target XYZ
    subgoal       : one of SUBGOAL_LABELS
    collision_prob: scalar in [0, 1] from perception.collision_predictor
    block_dims    : (3,) perceived [width, length, height] — ONLY appended
                    when subgoal=="close_gripper" (see module docstring for
                    why the other 5 subgoals don't get this field at all,
                    not even as a zero-filler). Ignored for every other
                    subgoal regardless of whether it's passed.

    Returns a (SUBGOAL_OBS_DIM,) float32 vector, (CLOSE_GRIPPER_OBS_DIM,)
    when subgoal=="close_gripper", or (RELATIVE_OBS_DIM,) when subgoal is
    in RELATIVE_OBS_SUBGOALS (see that constant's comment — frame-relative,
    no absolute position).
    """
    if subgoal in RELATIVE_OBS_SUBGOALS:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        achieved = np.asarray(achieved_goal, dtype=np.float32).reshape(-1)
        desired  = np.asarray(desired_goal,  dtype=np.float32).reshape(-1)
        grip_pos = obs[0:3]
        # Computed fresh from achieved_goal/grip_pos rather than reusing
        # obs[6:9] (gymnasium_robotics's own object_rel_pos) deliberately —
        # obs[6:9] is always ground-truth-computed by the sim engine
        # directly, independent of whatever caller-supplied achieved_goal
        # is passed in (e.g. a perception-noise estimate). Reusing it here
        # would silently leak true block position past a noisy achieved_goal
        # the exact bug sanitize_observation_for_perception's docstring
        # describes and fixes for the OTHER subgoals' obs[3:6]/[6:9].
        object_rel_pos = achieved - grip_pos
        goal_rel_pos    = desired - achieved
        parts = [
            object_rel_pos,
            obs[9:11],    # gripper_state
            obs[11:14],   # object_rot
            obs[14:17],   # object_velp
            obs[17:20],   # object_velr
            obs[20:23],   # grip_velp
            obs[23:25],   # gripper_vel
            goal_rel_pos,
            subgoal_to_onehot(subgoal),
            np.array([collision_prob], dtype=np.float32),
        ]
        return np.concatenate(parts)

    parts = [
        np.asarray(observation,   dtype=np.float32).reshape(-1),
        np.asarray(achieved_goal, dtype=np.float32).reshape(-1),
        np.asarray(desired_goal,  dtype=np.float32).reshape(-1),
        subgoal_to_onehot(subgoal),
        np.array([collision_prob], dtype=np.float32),
    ]
    if subgoal == "close_gripper":
        dims = np.zeros(3, dtype=np.float32) if block_dims is None else \
            np.asarray(block_dims, dtype=np.float32).reshape(3)
        parts.append(dims)
    return np.concatenate(parts)
