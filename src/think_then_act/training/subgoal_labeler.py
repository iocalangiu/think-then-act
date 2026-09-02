"""
think_then_act.training.subgoal_labeler

Rule-based state -> next-subgoal classifier, used to generate SFT labels
for the high-level VLM (see memory: hierarchical_architecture.md). Every
branch below is exactly one of reward/subgoal_reward.py's own `done`
checks, not a hand-rederived geometric boundary — a labeling rule with
independent thresholds would silently teach the VLM a different notion of
"aligned"/"grasped"/"lifted" than the low-level skill policies were
actually trained to satisfy.

Stateless, like every reward_* function it calls: reads only the current
(obs, achieved_goal, desired_goal), no episode history. That means it can
label any single frame in isolation — one from a full rollout, a
random-policy step, or a subgoal-specific reset
(env.setup.init_episode_before_subgoal) — with no notion of "phase" carried
in from outside.
"""

from __future__ import annotations

from think_then_act.reward.subgoal_reward import (
    DEFAULT_WEIGHTS,
    SubgoalWeights,
    reward_align_xy,
    reward_close_gripper,
    reward_descend,
    reward_lift,
    reward_move_to_target,
    reward_release,
)


def subgoal_diagnostics(obs, achieved_goal, desired_goal,
                         weights: SubgoalWeights = DEFAULT_WEIGHTS) -> dict:
    """
    The six boolean facts label_subgoal's state machine branches on, each
    read straight off the matching reward function's `done` field. Exposed
    separately (not just inlined into label_subgoal) so callers that want
    per-frame diagnostic logging — e.g. the SFT data generation script's
    verbose mode, matching generate_sft_data.py's existing per-step prints
    — don't have to recompute or re-derive any of this themselves.
    """
    return {
        "is_aligned_xy": reward_align_xy(obs, achieved_goal, desired_goal, weights)[1]["done"],
        # collision_prob doesn't affect reward_descend's `done` (only its
        # reward magnitude) — 0.0 is a safe placeholder here.
        "is_descended":  reward_descend(obs, achieved_goal, desired_goal, 0.0, weights)[1]["done"],
        "is_grasped":    reward_close_gripper(obs, achieved_goal, desired_goal, weights)[1]["done"],
        "is_lifted":     reward_lift(obs, achieved_goal, desired_goal, weights)[1]["done"],
        "is_at_target":  reward_move_to_target(obs, achieved_goal, desired_goal, weights)[1]["done"],
        "is_open":       reward_release(obs, achieved_goal, desired_goal, weights)[1]["done"],
    }


def is_block_grasped(obs, achieved_goal, desired_goal,
                      weights: SubgoalWeights = DEFAULT_WEIGHTS) -> bool:
    """
    Single-purpose wrapper around subgoal_diagnostics's "is_grasped" flag
    (itself reward_close_gripper's own `done` check) — for callers that
    only need this one fact, not the other five. Added 2026-07-17 so the
    high-level VLM's prompt (policy.subgoal_vlm_policy.USER_PROMPT_TEMPLATE)
    can state "is the block currently grasped" as a GIVEN fact rather than
    something the model has to infer purely from finger-width + geometry
    numbers: a held-out accuracy eval showed close_gripper/lift recall
    collapsing to ~4% (vs ~58% for align_xy) while the model over-predicted
    "release" as a generic fallback — the failing classes are exactly the
    ones whose correct label depends on already knowing whether the block
    is grasped, which nothing in the prompt stated explicitly.
    """
    return reward_close_gripper(obs, achieved_goal, desired_goal, weights)[1]["done"]


def label_subgoal(obs, achieved_goal, desired_goal,
                   weights: SubgoalWeights = DEFAULT_WEIGHTS) -> str | None:
    """
    Return the subgoal that should be selected next, or None if the task
    already looks complete (block at target, ungrasped, fingers open) —
    callers should skip frames labeled None rather than invent a label for
    a state with no next subgoal.

    Priority follows the fixed pick-and-place ordering. "is_grasped" (not
    episode history) is what disambiguates the approach sequence
    (align_xy/descend/close_gripper) from the carry sequence
    (lift/move_to_target/release) — e.g. d_xy is near-zero throughout
    carrying too, since the block moves with the gripper once grasped, so
    geometry alone can't tell "still approaching" from "already holding it"
    without that signal.
    """
    d = subgoal_diagnostics(obs, achieved_goal, desired_goal, weights)

    if d["is_grasped"] and d["is_at_target"]:
        return "release"
    if d["is_grasped"] and d["is_lifted"]:
        return "move_to_target"
    if d["is_grasped"]:
        return "lift"
    if d["is_at_target"] and d["is_open"]:
        return None  # delivered: block at target, released, nothing to do
    if not d["is_aligned_xy"]:
        return "align_xy"
    if not d["is_descended"]:
        return "descend"
    return "close_gripper"
