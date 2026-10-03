"""
think_then_act.training.fetch_skills

Adapter wiring hrl.skill_env.Skill to THIS project's existing subgoal
machinery (reward/subgoal_reward.py, training/subgoal_features.py) and
trained low-level policies. hrl/skill_env.py itself has no Fetch/MuJoCo-
specific code; this is where that wiring actually happens, so a different
project would swap this file for its own equivalent adapter instead of
touching skill_env.py.

No mujoco/gymnasium import here — same as subgoal_reward.py/
subgoal_features.py, this is pure Python/numpy glue and is unit-testable
locally (mujoco only shows up inside whatever live env object gets passed
in at actual runtime, e.g. via rollout_workers.py or a real training
script).
"""

from __future__ import annotations

from think_then_act.env.block_randomization import get_block_dims, get_perceived_block_dims
from think_then_act.env.setup import grip_contact_forces
from think_then_act.hrl.skill_env import Skill
from think_then_act.reward.subgoal_reward import SUBGOAL_LABELS, compute_subgoal_reward
from think_then_act.training.subgoal_features import (
    build_subgoal_observation, sanitize_observation_for_perception,
)


def _make_collision_prob_fn(collision_model):
    """
    Every subgoal's build_obs needs SOME collision_prob value (fixed
    38-dim schema — see subgoal_features.py), but only `descend`'s reward
    actually uses it. Memoized by the rendered frame's identity: within
    one SkillEnv.step() loop iteration, build_obs (next decision) and
    reward_and_done (this transition's outcome) are each called once
    against the state at that point in time, and the frame doesn't change
    until the next base_env.step() — so this avoids running the collision
    CNN twice per step for no reason, across all 6 subgoals, not just
    descend.
    """
    if collision_model is None:
        return lambda base_env: 0.0

    cache = {"frame_id": None, "prob": 0.0}

    def collision_prob(base_env) -> float:
        frame = base_env.last_frame()
        frame_id = id(frame)
        if frame_id != cache["frame_id"]:
            cache["frame_id"] = frame_id
            cache["prob"] = collision_model.predict_proba(frame)
        return cache["prob"]

    return collision_prob


def _make_pose_fn(pose_model):
    """
    Same per-frame memoization rationale as _make_collision_prob_fn — avoid
    running the pose CNN twice per step across the build_obs/reward_and_done
    pair when nothing about the frame changed. Falls back to
    obs["achieved_goal"] (ground truth) when pose_model is None — same
    None-means-unchanged-behavior convention as collision_model.
    """
    if pose_model is None:
        return lambda obs, base_env: obs["achieved_goal"]

    cache = {"frame_id": None, "pos": None}

    def block_pos(obs, base_env):
        frame = base_env.last_frame()
        frame_id = id(frame)
        if frame_id != cache["frame_id"]:
            cache["frame_id"] = frame_id
            cache["pos"] = pose_model.predict_position(frame)
        return cache["pos"]

    return block_pos


def build_fetch_skills(policies: dict, collision_model=None, pose_model=None, max_steps: int = 30,
                        randomize_block_size: bool = False) -> dict:
    """
    policies: {subgoal_name: policy} for some or all of SUBGOAL_LABELS —
    each policy needs `.act(obs_vector, deterministic=True) -> action`
    (e.g. policy.subgoal_policy.SubgoalGaussianPolicy, or
    training.low_level_ppo.LowLevelPPOTrainer(...).actor).
    collision_model: optional perception.collision_predictor.CollisionPredictor.
    pose_model: optional perception.block_pose_predictor.BlockPosePredictor —
    when set, build_obs uses its estimate of the block position instead of
    the privileged obs["achieved_goal"]. reward_and_done deliberately keeps
    using obs["achieved_goal"] (ground truth) regardless of pose_model — see
    block_pose_predictor.py's docstring for why reward/done stay privileged
    while only the observation switches.
    randomize_block_size: gates whether lift/release/close_gripper's reward
    reads block_half_height/block_width from the live model's CURRENT
    geometry — False (default) means they always get None, same as
    training/subgoal_env.py's SubgoalConditionedEnv(randomize_block_size=
    False). Needed because get_block_dims happily falls back to whatever
    geometry the model currently has REGARDLESS of whether size was ever
    actually randomized, and (for release/close_gripper, unlike lift) that
    derived value is numerically DIFFERENT from the flat fixed-cube
    constants (reward/subgoal_reward.py's release_open_threshold=0.08 vs.
    derived ~0.07 at the default 5cm width; close_gripper_force_scale=130
    vs. derived ~138) — without this gate, EVERY caller would silently
    switch onto the derived-constants path the moment size randomization
    existed in the codebase, whether or not it was ever requested here.

    Returns {subgoal_name: Skill}, ready to hand to hrl.skill_env.SkillEnv:
        skills = build_fetch_skills({"align_xy": actor_align_xy, ...}, collision_model)
        env = SkillEnv(base_env, skills)
    """
    unknown = set(policies) - set(SUBGOAL_LABELS)
    if unknown:
        raise ValueError(f"Unknown subgoal(s) {unknown}; must be a subset of {SUBGOAL_LABELS}")

    collision_prob_fn = _make_collision_prob_fn(collision_model)
    pose_fn = _make_pose_fn(pose_model)

    def make_build_obs(subgoal: str):
        def build_obs(obs, base_env):
            perceived_pos = pose_fn(obs, base_env)
            # obs["observation"] itself bakes in the TRUE block position
            # twice more (indices 3:6, 6:9 — see
            # sanitize_observation_for_perception's docstring); swapping
            # only the achieved_goal argument above would leave those
            # privileged copies fully intact. Only sanitize when pose_model
            # is actually active — a no-op otherwise (achieved_goal is
            # already ground truth in that case).
            observation = obs["observation"]
            if pose_model is not None:
                observation = sanitize_observation_for_perception(observation, perceived_pos)
            # get_perceived_block_dims needs LIVE mujoco model state, same
            # close_gripper-only gating as grip_contact_forces below (see
            # that comment) — build_subgoal_observation ignores block_dims
            # for every other subgoal anyway, so there's nothing to compute
            # for them.
            block_dims = get_perceived_block_dims(base_env.unwrapped.model) if subgoal == "close_gripper" else None
            return build_subgoal_observation(
                observation, perceived_pos, obs["desired_goal"],
                subgoal, collision_prob_fn(base_env), block_dims=block_dims,
            )
        return build_obs

    def make_reward_and_done(subgoal: str):
        def reward_and_done(obs, base_env):
            # grip_contact_forces needs LIVE mujoco contact data
            # (base_env.unwrapped.data.contact), unlike collision_prob_fn/
            # pose_fn above which only ever touch base_env.last_frame() —
            # gated to close_gripper only (the one subgoal that actually
            # uses it, same as compute_subgoal_reward's own dispatch) so
            # every other subgoal stays exercisable against a fake base_env
            # exposing nothing but last_frame(), preserving this module's
            # "no mujoco/gymnasium import here" property for them.
            grip_force = grip_contact_forces(base_env) if subgoal == "close_gripper" else None
            # Same live-mujoco gating as grip_force, PLUS the
            # randomize_block_size gate (see build_fetch_skills' docstring
            # for why that second gate is required, not just belt-and-
            # braces) — only lift/release/close_gripper consume these (see
            # compute_subgoal_reward's dispatch), so every other subgoal
            # (and any fake-base_env unit test exercising them) never
            # touches get_block_dims at all.
            block_half_height = block_width = None
            if randomize_block_size and subgoal in ("lift", "release", "close_gripper"):
                dims = get_block_dims(base_env.unwrapped.model)
                block_half_height, block_width = dims["height"] / 2.0, dims["width"]
            reward, breakdown = compute_subgoal_reward(
                subgoal, obs["observation"], obs["achieved_goal"], obs["desired_goal"],
                collision_prob=collision_prob_fn(base_env), grip_force=grip_force,
                block_half_height=block_half_height, block_width=block_width,
            )
            return reward, breakdown["done"]
        return reward_and_done

    return {
        subgoal: Skill(
            name=subgoal,
            policy=policy,
            build_obs=make_build_obs(subgoal),
            reward_and_done=make_reward_and_done(subgoal),
            max_steps=max_steps,
        )
        for subgoal, policy in policies.items()
    }
