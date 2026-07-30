"""
think_then_act.training.subgoal_features

Pure functions for assembling the low-level controller's input vector.
Deliberately has NO gymnasium/mujoco import — subgoal_env.py (the actual
gym.Wrapper) depends on this, not the other way around, so this half of the
logic is unit-testable without a live env or the mujoco/gymnasium stack
(neither is installed on the local dev machine — see env/wrapper.py's
docstring for why the project keeps this split).

Observation vector layout, fixed size regardless of which subgoal is active:
    obs(25) + achieved_goal(3) + desired_goal(3) + subgoal one-hot(6) + collision_prob(1) = 38
"""

from __future__ import annotations
import numpy as np

from think_then_act.reward.subgoal_reward import SUBGOAL_LABELS

SUBGOAL_OBS_DIM = 25 + 3 + 3 + len(SUBGOAL_LABELS) + 1


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
) -> np.ndarray:
    """
    observation   : (25,) raw FetchPickAndPlace observation vector
    achieved_goal : (3,) block XYZ
    desired_goal  : (3,) target XYZ
    subgoal       : one of SUBGOAL_LABELS
    collision_prob: scalar in [0, 1] from perception.collision_predictor

    Returns a (SUBGOAL_OBS_DIM,) float32 vector.
    """
    return np.concatenate([
        np.asarray(observation,   dtype=np.float32).reshape(-1),
        np.asarray(achieved_goal, dtype=np.float32).reshape(-1),
        np.asarray(desired_goal,  dtype=np.float32).reshape(-1),
        subgoal_to_onehot(subgoal),
        np.array([collision_prob], dtype=np.float32),
    ])
