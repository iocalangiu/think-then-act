"""
think_then_act.reward.flat_task_reward

Dense reward for the FLAT (non-subgoal-conditioned) full pick-and-place
task — see memory: flat_policy_bc_scaling. Unlike reward/subgoal_reward.py
(one function per coarse subgoal, dispatched by a hierarchical selector),
this scores the WHOLE task with one function, for PPO fine-tuning the flat
MSE BC policy (training/flat_bc_multi_head.py, policy_type="mse" ==
SubgoalRecurrentPolicy, byte-for-byte reusable here).

Built 2026-10-02 after a direct telemetry investigation of two specific
behaviors found in BC rollouts of checkpoints/bc_scaling/mse_n2000_seed0_
smw0.0_k5.pt: the gripper/block gets dragged along the table instead of
lifted cleanly, and the arm visibly moves (joints working) while the
gripper's own net position barely changes. Two new terms target these
directly, on top of the standard two-phase distance shaping and the
project's standing genuine-grasp success bonus:

- discrepancy_weight: penalizes ||intended_displacement - realized_
  displacement|| for the GRIPPER itself, using action[:3]*pos_scale as
  "intent" (pos_scale=0.05, this project's own known per-step action scale
  — see env/setup.py's randomize_gripper_start docstring) and the observed
  grip_pos delta as "realized." This is the direct fix for "joints moving
  while the gripper doesn't": confirmed via scripts/diagnose_drag_
  mechanism.py that BC rollouts can command strongly negative dz for
  dozens of consecutive steps while grip_z changes by <0.001m each step
  (the policy has no signal telling it the command isn't working). An
  earlier attempt to fix this by having the ENV silently zero out a
  detected-blocked action was tested directly and made literally zero
  difference to outcomes — unsurprising in hindsight: a frozen policy
  re-evaluates the same observation/hidden state every step and has no
  way to learn from an action that was overridden out from under it.
  Only a REWARD signal (which PPO can actually learn from) can change
  this. Deliberately NOT an action-rate (||a_t - a_{t-1}||) term, which
  was the original plan before this investigation — that would also
  punish legitimate fast, sustained motion; this term is zero for any
  motion that actually succeeds, however large, and nonzero only when a
  commanded displacement fails to materialize.

- drag_weight: penalizes the block's own xy displacement on any step
  where it moves without the project's standing genuine-grasp condition
  (two-finger contact + something — see env/setup.py's grip_contact_
  forces/the genuine-grasp verification used everywhere else in this
  project) being true. Direct fix for "dragged along the table" — found
  via the same telemetry that the drag episodes' per-step traces all show
  the SAME signature: large sustained lateral (dx, dy) commands while
  contact is either absent or starkly asymmetric (one finger spiking past
  1000N, the other at exactly 0N) — the gripper shoving the block
  sideways with a misaligned, one-sided touch, not a controlled grasp.
  This is a DIFFERENT failure than the discrepancy one above (there the
  GRIPPER's own displacement fails to materialize; here the BLOCK's
  displacement DOES materialize, just via an illegitimate contact) so it
  needs its own term, not a side effect of the other.

Both weights start at a reasoned-but-unvalidated default, same as every
other reward term's history in this project (see subgoal_reward.py's own
extensive per-weight calibration comments) — check actual training
telemetry (mean drag distance, mean discrepancy, genuine completion_rate)
before trusting these numbers.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class FlatTaskWeights:
    # Standard two-phase pick-and-place shaping: pulls the gripper to the
    # block before a genuine grasp has ever happened this episode, pulls
    # the block to the target afterward. Unweighted (1.0) — same convention
    # as reward_align_xy/reward_move_to_target's own bare -distance terms.
    approach_weight: float = 1.0
    carry_weight: float = 1.0
    # See module docstring — the two new, specifically-motivated terms.
    discrepancy_weight: float = 2.0   # penalty per metre of ||intended - realized|| gripper displacement
    drag_weight: float = 20.0         # penalty per metre of block xy motion while not genuinely gripped —
                                       # same order of magnitude as close_gripper_distance_weight (20.0 in
                                       # subgoal_reward.py), which penalizes a conceptually similar
                                       # "ended up far from the block" quantity on a comparable (metres)
                                       # scale, not independently tuned yet.
    success_bonus: float = 5.0        # one-time-per-step bonus while genuine success holds
    lift_threshold: float = 0.02      # metres above resting height counted as "lifted" — same
                                       # value used everywhere else in this project
    pos_scale: float = 0.05           # metres per action unit — this project's own known per-step
                                       # action scale (env/setup.py's randomize_gripper_start docstring)

    def as_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}


DEFAULT_FLAT_TASK_WEIGHTS = FlatTaskWeights()


def compute_flat_task_reward(
    obs, achieved_goal, desired_goal,
    action,
    grip_pos_before,
    grip_force: dict,
    ever_genuinely_grasped: bool,
    block_resting_z: float = 0.425,
    weights: FlatTaskWeights = DEFAULT_FLAT_TASK_WEIGHTS,
) -> tuple:
    """
    One-step flat-task reward. Call AFTER env.step() — obs/achieved_goal/
    desired_goal/grip_force are the POST-step values; grip_pos_before is
    the gripper position BEFORE this step's action was applied (caller's
    responsibility to carry across steps); action is what was actually
    sent to env.step(); ever_genuinely_grasped is whether the project's
    standard genuine-grasp condition (two-finger contact AND lifted
    >lift_threshold) has been true at ANY point so far this episode
    (caller tracks this — same running-flag pattern train_flat_task_bc.py
    and run_bc_scaling_cell already use for eval).

    Returns (reward: float, breakdown: dict) — breakdown always has
    "genuine_grasp_now": bool (this step's instantaneous grasp state,
    distinct from the running ever_genuinely_grasped flag callers pass in)
    and "block_xy_delta": float, so callers can update their own running
    state (ever_genuinely_grasped, previous block xy) from one place.
    """
    obs = np.asarray(obs, dtype=np.float64)
    achieved = np.asarray(achieved_goal, dtype=np.float64)
    desired = np.asarray(desired_goal, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    grip_pos_before = np.asarray(grip_pos_before, dtype=np.float64)

    grip_pos = obs[0:3]
    d_grip_block = float(np.linalg.norm(achieved - grip_pos))
    d_block_target = float(np.linalg.norm(desired - achieved))

    left = grip_force.get("left", 0.0)
    right = grip_force.get("right", 0.0)
    genuine_grip_now = min(left, right) > 0.0
    height_above_resting = float(achieved[2]) - block_resting_z
    is_lifted = height_above_resting > weights.lift_threshold
    genuinely_grasped_now = genuine_grip_now and is_lifted

    # Shaping: once a genuine grasp has happened at ANY point this
    # episode, the task is "carry to target" from then on, even in a step
    # where contact has since loosened — matches the project's own
    # ever_lifted_and_gripped running-flag convention used throughout
    # eval code, rather than switching back to "approach" the instant
    # contact force reads zero for one step.
    carrying = ever_genuinely_grasped or genuinely_grasped_now
    reward = (-weights.carry_weight * d_block_target if carrying
              else -weights.approach_weight * d_grip_block)

    intended_delta = action[:3] * weights.pos_scale
    realized_delta = grip_pos - grip_pos_before
    discrepancy = float(np.linalg.norm(intended_delta - realized_delta))
    reward -= weights.discrepancy_weight * discrepancy

    if carrying:
        reward += weights.success_bonus

    return reward, {
        "d_grip_block": round(d_grip_block, 5), "d_block_target": round(d_block_target, 5),
        "genuine_grasp_now": genuinely_grasped_now, "discrepancy": round(discrepancy, 5),
        "height_above_resting": round(height_above_resting, 5), "carrying": carrying,
    }


def drag_penalty(block_xy_delta: float, genuine_grasp_now: bool,
                  weights: FlatTaskWeights = DEFAULT_FLAT_TASK_WEIGHTS) -> float:
    """
    Separate from compute_flat_task_reward's return (not folded in
    directly) because the caller needs the PREVIOUS step's block xy
    position to compute block_xy_delta, same "caller carries state across
    steps" pattern as grip_pos_before above — kept as its own function so
    that threading requirement is explicit at the call site instead of
    hidden inside one large reward function with many carried-state
    arguments.
    """
    if genuine_grasp_now:
        return 0.0
    return -weights.drag_weight * block_xy_delta
