"""
reward.py

Dense reward shaping for FetchPickAndPlace-v3.

The default gymnasium-robotics reward is sparse: -1 every step until
the block reaches the target (reward 0).  A random or untrained policy
essentially never reaches the target, so the RL signal is zero for
thousands of steps — learning is impossible.

This module provides a dense alternative that gives useful gradient
information at every step, decomposed into four components:

  R1 (approach)  : -d(gripper, block)           — always active
  R2 (transport) : -2 * d(block, target)         — always active, higher weight
  R3 (grasp)     : +0.5 * closedness             — active when gripper near block
  R4 (success)   : +10.0                         — large one-time bonus on success

Observation vector layout for FetchPickAndPlace (25 floats):
  obs[ 0: 3] = grip_pos          (gripper end-effector XYZ)
  obs[ 3: 6] = object_pos        (block XYZ, same as achieved_goal)
  obs[ 6: 9] = object_rel_pos    (block − gripper)
  obs[ 9:11] = gripper_state     (finger widths, ~0=closed, ~0.05=open each)
  obs[11:14] = object_rot        (block rotation)
  obs[14:17] = object_velp       (block linear velocity)
  obs[17:20] = object_velr       (block angular velocity)
  obs[20:23] = grip_velp         (gripper velocity)
  obs[23:25] = gripper_vel       (finger velocity)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# Reward configuration — all weights in one place for easy tuning
# ---------------------------------------------------------------------------
@dataclass
class RewardWeights:
    """
    Tunable hyperparameters for the dense reward.
    These are reasonable starting values; you may need to adjust them
    during RL training (Milestone 5) if one component dominates too early.
    """
    w_approach   : float = 1.0    # multiplier for gripper→block distance penalty
    w_transport  : float = 2.0    # multiplier for block→target distance penalty
    w_grasp      : float = 0.5    # bonus for closed gripper when near block
    w_success    : float = 10.0   # large reward for task completion
    grasp_radius : float = 0.05   # metres: gripper must be within this to get grasp bonus
    finger_open  : float = 0.10   # sum of both finger widths when fully open

    def as_dict(self) -> dict:
        return {
            "w_approach"  : self.w_approach,
            "w_transport" : self.w_transport,
            "w_grasp"     : self.w_grasp,
            "w_success"   : self.w_success,
            "grasp_radius": self.grasp_radius,
            "finger_open" : self.finger_open,
        }


# Default weights used by compute_dense_reward when none are supplied
DEFAULT_WEIGHTS = RewardWeights()


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------
def compute_dense_reward(
    obs: list | np.ndarray,
    achieved_goal: list | np.ndarray,
    desired_goal: list | np.ndarray,
    info: dict,
    weights: RewardWeights = DEFAULT_WEIGHTS,
) -> tuple[float, dict]:
    """
    Compute one-step dense reward for FetchPickAndPlace.

    Args:
        obs           : (25,) observation vector from the harness log entry
        achieved_goal : (3,) block XYZ position
        desired_goal  : (3,) target XYZ position
        info          : step info dict (contains 'is_success')
        weights       : RewardWeights config; uses DEFAULT_WEIGHTS if None

    Returns:
        total_reward : scalar float
        breakdown    : dict with per-component values for logging/debugging
    """
    obs      = np.asarray(obs,           dtype=np.float64)
    achieved = np.asarray(achieved_goal, dtype=np.float64)
    desired  = np.asarray(desired_goal,  dtype=np.float64)

    # ------------------------------------------------------------------
    # Extract positions from observation vector
    # ------------------------------------------------------------------
    grip_pos      = obs[0:3]    # gripper end-effector XYZ
    block_pos     = obs[3:6]    # block XYZ (redundant with achieved_goal, but convenient)
    gripper_state = obs[9:11]   # finger widths: ~0 each = closed, ~0.05 each = open

    # ------------------------------------------------------------------
    # Distance metrics
    # ------------------------------------------------------------------
    d_grip_block  = float(np.linalg.norm(block_pos  - grip_pos))
    d_block_target = float(np.linalg.norm(desired   - achieved))

    # ------------------------------------------------------------------
    # R1: Approach reward
    # Negative distance — larger (less negative) as gripper nears block.
    # ------------------------------------------------------------------
    r_approach = -weights.w_approach * d_grip_block

    # ------------------------------------------------------------------
    # R2: Transport reward
    # Negative distance — larger (less negative) as block nears target.
    # Weighted higher than approach because transport is the task goal.
    # ------------------------------------------------------------------
    r_transport = -weights.w_transport * d_block_target

    # ------------------------------------------------------------------
    # R3: Grasp bonus
    # Incentivises closing the gripper once it's physically near the block.
    # gripper_closedness: 0 = fully open, 1 = fully closed.
    # We only award this bonus when the gripper is within grasp_radius.
    # ------------------------------------------------------------------
    total_finger_width  = float(np.sum(gripper_state))
    gripper_closedness  = 1.0 - np.clip(total_finger_width / weights.finger_open, 0.0, 1.0)
    is_near_block       = d_grip_block <= weights.grasp_radius
    r_grasp             = weights.w_grasp * float(is_near_block) * gripper_closedness

    # ------------------------------------------------------------------
    # R4: Success bonus
    # Large one-time reward for reaching the goal. Encourages the policy
    # to actually complete the task rather than just getting close.
    # ------------------------------------------------------------------
    is_success = bool(info.get("is_success", False))
    r_success  = weights.w_success * float(is_success)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    total = r_approach + r_transport + r_grasp + r_success

    breakdown = {
        "dense_total"      : round(total,              5),
        "r_approach"       : round(r_approach,         5),
        "r_transport"      : round(r_transport,        5),
        "r_grasp"          : round(r_grasp,            5),
        "r_success"        : round(r_success,          5),
        # Raw diagnostics — useful for debugging and plotting in M6
        "d_grip_block"     : round(d_grip_block,       5),
        "d_block_target"   : round(d_block_target,     5),
        "gripper_closedness": round(gripper_closedness, 5),
        "is_near_block"    : is_near_block,
        "is_success"       : is_success,
    }

    return total, breakdown


# ---------------------------------------------------------------------------
# Convenience: apply dense reward to a full episode log from ObservationHarness
# ---------------------------------------------------------------------------
def apply_to_episode(
    episode_log: list[dict],
    weights: RewardWeights = DEFAULT_WEIGHTS,
) -> list[dict]:
    """
    Recompute dense rewards for every step in a harness episode_log.
    Adds 'dense_reward' and 'reward_breakdown' keys to each entry.
    Does NOT modify the original list — returns a new list.

    Usage:
        log = env.metadata_only()
        enriched = apply_to_episode(log)
    """
    enriched = []
    for entry in episode_log:
        if entry["observation"] is None:
            enriched.append(dict(entry))
            continue
        dense, breakdown = compute_dense_reward(
            obs           = entry["observation"],
            achieved_goal = entry["achieved_goal"],
            desired_goal  = entry["desired_goal"],
            info          = {"is_success": entry["is_success"]},
            weights       = weights,
        )
        new_entry = dict(entry)
        new_entry["dense_reward"]      = dense
        new_entry["reward_breakdown"]  = breakdown
        enriched.append(new_entry)
    return enriched
