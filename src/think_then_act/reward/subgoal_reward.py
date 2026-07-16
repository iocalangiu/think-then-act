"""
think_then_act.reward.subgoal_reward

Per-subgoal reward functions for the low-level controller in the hierarchical
architecture (see memory: hierarchical_architecture.md). Each function scores
progress toward ONE coarse subgoal the VLM high-level would choose, as
opposed to dense_reward.py's compute_dense_reward, which scores the whole
pick-and-place task at once.

Same observation-vector layout as dense_reward.py (25 floats) — see that
module's docstring for the full breakdown and the obs[3:6]-is-not-block-
position caveat (use achieved_goal instead, always).

Subgoal vocabulary, in the order a full pick-and-place attempt would use them:
    align_xy       — move gripper laterally above the block
    descend        — lower gripper to grasp height, without drifting off XY
    close_gripper  — close fingers around the block
    lift           — raise the block off the table
    move_to_target — carry the block to the target position
    release        — open the gripper once the block is at the target
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

SUBGOAL_LABELS = [
    "align_xy",
    "descend",
    "close_gripper",
    "lift",
    "move_to_target",
    "release",
]


@dataclass
class SubgoalWeights:
    """Tunable thresholds/weights, one dataclass shared by all subgoal reward fns."""
    align_xy_threshold      : float = 0.02   # metres, planar gripper-block distance
    descend_threshold       : float = 0.02   # metres, vertical gripper-block gap
    collision_penalty       : float = 5.0    # weight on collision_prob during descend
    # NOT 0.9: measured 2026-07-15 via scripts/measure_close_gripper_geometry.py
    # that genuinely gripping this project's cube (a real 0.05m box, per
    # that script's geom enumeration) caps closedness at ~0.52 (steady-state
    # finger_width ~0.048, consistent across 5 seeds, 0.515-0.523) — the
    # cube's own width physically stops the fingers closing further, so 0.9
    # was only reachable by closing on EMPTY AIR (missing the block
    # entirely), the opposite of a successful grasp. 0.45 gives margin below
    # the empirical ~0.52 ceiling while still requiring real closure.
    close_gripper_threshold : float = 0.45   # gripper_closedness, 0=open 1=closed
    finger_open             : float = 0.10   # sum of both finger widths when fully open
    # Closing the fingers physically shoves the block (contact dynamics),
    # and without a check the arm can then drift away entirely rather than
    # recovering — found 2026-07-14 via video, after the pre-subgoal-setup
    # fix (env/setup.py's init_episode_before_subgoal) already got it
    # starting right at the block. subgoal_env.py truncates the episode
    # once d_grip_block exceeds this rather than burning the rest of the
    # step budget on a rollout that's already failed.
    close_gripper_drift_limit: float = 0.05  # metres, gripper-block distance
    # Direct, every-step incentive to hold the arm still while closing —
    # the oracle (env/oracle.py's GRASP branch) achieves this by construction,
    # zeroing dx/dy/dz outright once close enough. The RL policy has no such
    # constraint, and `-0.1*d_grip_block` above only penalizes drift AFTER
    # contact has already shoved the block (confirmed 2026-07-16: 150-iter
    # training run showed entropy flat and completion_rate stuck at 0%,
    # d_grip_block(final) sitting right at the drift limit every iteration —
    # the indirect penalty alone wasn't enough signal to discover "don't
    # move"). This term penalizes ||dx,dy,dz|| directly, every step,
    # regardless of whether displacement has manifested yet.
    close_gripper_stillness_weight: float = 0.15  # penalty on translation-action L2 norm
    # Block's default resting-center height (init_random_episode's
    # teleport_block places it at z=0.425), used here as the "not lifted"
    # baseline — NOT the physical table surface height. That's actually
    # 0.400 (confirmed via validate_collision_labels.py's geom enumeration,
    # 2026-07-12: table0 body xpos_z=0.20 + geom size_z=0.20). The 0.025
    # gap is exactly the block's own half-thickness (its geom size_z), i.e.
    # table_z = table_top + block_half_size — correct for measuring lift
    # amount, just not literally "the table."
    table_z                 : float = 0.425
    lift_height             : float = 0.10   # metres above table_z counted as "lifted"
    move_to_target_threshold: float = 0.05   # metres, block-target distance (matches env success)
    release_open_threshold  : float = 0.08   # sum of both finger widths counted as "open"

    def as_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}


DEFAULT_WEIGHTS = SubgoalWeights()


# ---------------------------------------------------------------------------
# Shared geometry extraction — same quantities every subgoal fn needs
# ---------------------------------------------------------------------------
def _geometry(obs, achieved_goal, desired_goal) -> dict:
    obs      = np.asarray(obs,           dtype=np.float64)
    achieved = np.asarray(achieved_goal, dtype=np.float64)
    desired  = np.asarray(desired_goal,  dtype=np.float64)

    grip_pos      = obs[0:3]
    block_pos     = achieved   # NOT obs[3:6] — see module docstring
    gripper_state = obs[9:11]

    d_xy           = float(np.linalg.norm(block_pos[:2] - grip_pos[:2]))
    d_z            = float(grip_pos[2] - block_pos[2])
    d_grip_block   = float(np.linalg.norm(block_pos - grip_pos))
    d_block_target = float(np.linalg.norm(desired - achieved))

    total_finger_width = float(np.sum(gripper_state))
    return {
        "grip_pos": grip_pos, "block_pos": block_pos, "target_pos": desired,
        "d_xy": d_xy, "d_z": d_z,
        "d_grip_block": d_grip_block, "d_block_target": d_block_target,
        "total_finger_width": total_finger_width,
    }


# ---------------------------------------------------------------------------
# Per-subgoal reward functions
# All return (reward: float, breakdown: dict) — breakdown always has "done".
# ---------------------------------------------------------------------------
def reward_align_xy(obs, achieved_goal, desired_goal, weights: SubgoalWeights = DEFAULT_WEIGHTS):
    g = _geometry(obs, achieved_goal, desired_goal)
    reward = -g["d_xy"]
    done    = bool(g["d_xy"] <= weights.align_xy_threshold)
    return reward, {"d_xy": round(g["d_xy"], 5), "done": done}


def reward_descend(obs, achieved_goal, desired_goal, collision_prob: float = 0.0,
                    weights: SubgoalWeights = DEFAULT_WEIGHTS):
    """
    Penalizes vertical gap AND predicted collision risk — this is the subgoal
    where the project's premature-descent/table-collision bug actually lives,
    so collision_prob (from perception.collision_predictor) is a first-class
    input here, not an afterthought.
    """
    g = _geometry(obs, achieved_goal, desired_goal)
    reward = -g["d_z"] - weights.collision_penalty * float(collision_prob)
    done    = bool(g["d_z"] <= weights.descend_threshold)
    return reward, {
        "d_z": round(g["d_z"], 5), "d_xy": round(g["d_xy"], 5),
        "collision_prob": round(float(collision_prob), 5), "done": done,
    }


def reward_close_gripper(obs, achieved_goal, desired_goal, weights: SubgoalWeights = DEFAULT_WEIGHTS,
                          action=None):
    g = _geometry(obs, achieved_goal, desired_goal)
    closedness = float(1.0 - np.clip(g["total_finger_width"] / weights.finger_open, 0.0, 1.0))
    # Small penalty for drifting away from the block while closing — closing
    # around empty air shouldn't score as well as closing around the block.
    reward = closedness - 0.1 * g["d_grip_block"]
    # action is None at env.reset() (nothing has been taken yet) — only
    # step() has a real action to penalize, so this term is a no-op there.
    translation_norm = 0.0
    if action is not None:
        translation_norm = float(np.linalg.norm(np.asarray(action, dtype=np.float64)[:3]))
        reward -= weights.close_gripper_stillness_weight * translation_norm
    # closedness alone can't tell "closed around the block" apart from
    # "closed on empty air" — only the latter can ever approach 1.0 (see
    # close_gripper_threshold's comment for the measurement). Requiring
    # d_grip_block to also be small closes that loophole: success needs
    # BOTH real closure AND actually being at the block.
    done = bool(closedness >= weights.close_gripper_threshold
                and g["d_grip_block"] <= weights.close_gripper_drift_limit)
    # d_grip_block surfaced here (not just used internally above) so
    # subgoal_env.py can check it against close_gripper_drift_limit without
    # recomputing the geometry itself.
    return reward, {"closedness": round(closedness, 5),
                     "d_grip_block": round(g["d_grip_block"], 5),
                     "translation_norm": round(translation_norm, 5), "done": done}


def reward_lift(obs, achieved_goal, desired_goal, weights: SubgoalWeights = DEFAULT_WEIGHTS):
    g = _geometry(obs, achieved_goal, desired_goal)
    height_above_table = float(g["block_pos"][2] - weights.table_z)
    reward = height_above_table
    done    = bool(height_above_table >= weights.lift_height)
    return reward, {"height_above_table": round(height_above_table, 5), "done": done}


def reward_move_to_target(obs, achieved_goal, desired_goal, weights: SubgoalWeights = DEFAULT_WEIGHTS):
    g = _geometry(obs, achieved_goal, desired_goal)
    reward = -g["d_block_target"]
    done    = g["d_block_target"] <= weights.move_to_target_threshold
    return reward, {"d_block_target": round(g["d_block_target"], 5), "done": done}


def reward_release(obs, achieved_goal, desired_goal, weights: SubgoalWeights = DEFAULT_WEIGHTS):
    g = _geometry(obs, achieved_goal, desired_goal)
    openness = np.clip(g["total_finger_width"] / weights.finger_open, 0.0, 1.0)
    reward = float(openness)
    done    = g["total_finger_width"] >= weights.release_open_threshold
    return reward, {"total_finger_width": round(g["total_finger_width"], 5), "done": done}


_SUBGOAL_FN = {
    "align_xy"      : reward_align_xy,
    "descend"       : reward_descend,
    "close_gripper" : reward_close_gripper,
    "lift"          : reward_lift,
    "move_to_target": reward_move_to_target,
    "release"       : reward_release,
}


def compute_subgoal_reward(
    subgoal: str,
    obs, achieved_goal, desired_goal,
    collision_prob: float = 0.0,
    weights: SubgoalWeights = DEFAULT_WEIGHTS,
    action=None,
) -> tuple:
    """
    Dispatch to the reward function matching `subgoal` (must be one of
    SUBGOAL_LABELS). Only `descend` uses collision_prob and only
    `close_gripper` uses action (translation-stillness penalty); other
    subgoals ignore the one(s) that don't apply to them. Returns
    (reward: float, breakdown: dict with "done": bool).
    """
    if subgoal not in _SUBGOAL_FN:
        raise ValueError(f"Unknown subgoal {subgoal!r}; must be one of {SUBGOAL_LABELS}")
    if subgoal == "descend":
        return reward_descend(obs, achieved_goal, desired_goal, collision_prob, weights)
    if subgoal == "close_gripper":
        return reward_close_gripper(obs, achieved_goal, desired_goal, weights, action)
    return _SUBGOAL_FN[subgoal](obs, achieved_goal, desired_goal, weights)
