"""
generate_subgoal_sft_data.py

Generates SFT examples for the HIGH-LEVEL VLM (see memory:
hierarchical_architecture.md) — the one that looks at a frame and picks
which of the 6 subgoals (SUBGOAL_LABELS) should run next, as opposed to
the old generate_sft_data.py/vlm_policy.py, which predate the SkillEnv
pivot and have the VLM output raw dx/dy/dz/grip numbers directly. This
script's target format is:
    <think>[reasoning grounded in gripper/block/target state]</think>
    <action>subgoal_label</action>
No coordinates in <action> — SkillEnv.step(label) only ever consumes the
label string, and Skill.build_obs already reads achieved_goal/desired_goal
straight from the env, not from the VLM's output (see fetch_skills.py).

Every row's label comes from training/subgoal_labeler.py's label_subgoal,
which is itself built directly on reward/subgoal_reward.py's `done`
checks — so a labeled example's category can't drift from what the
low-level skills were actually trained to consider "aligned"/"grasped"/
etc.

Three data sources, mixed together:
  chained — from a randomized reset, actually run the TRAINED align_xy ->
            descend -> close_gripper skills (verified per memory,
            2026-07-16) through their own policies, capturing a frame at
            every skill-call boundary. This is the realistic distribution:
            SkillEnv only ever queries the high-level agent at boundaries
            like these (reset, or right after a skill reports done/
            timeout), not at arbitrary mid-skill steps.
  seeded  — lift/move_to_target/release have no trained policy yet, so
            there's no skill to chain into. Instead, reuse the same
            oracle-driven pre-subgoal setup RL training itself uses
            (env.setup.init_episode_before_subgoal) to reach a state
            representative of each, then take a few small random actions
            to get some local variety around it.
  random  — small random actions from a fresh randomized reset, no
            structure at all. Secondary/OOD-robustness slice only, not the
            primary source — see the module docstring discussion for why
            random-policy states alone would under-represent the "clean"
            boundary states a live VLM is actually queried at.

Run with:
    modal run scripts/generate_subgoal_sft_data.py                    # defaults
    modal run scripts/generate_subgoal_sft_data.py --debug            # small + verbose
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


# ---------------------------------------------------------------------------
# Pure-numpy / stdlib helpers — deliberately no gymnasium/mujoco/torch
# imports at module scope, mirroring generate_sft_data.py and
# record_subgoal_video.py: those get imported LAZILY inside the
# @app.function body / per-call, since they aren't installed on the local
# machine that parses this file to register the Modal app.
# ---------------------------------------------------------------------------

def frame_to_b64(frame) -> str:
    import base64, io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# Multiple phrasings for the two labels found (2026-07-20) to leak into the
# <action> tag via lexical shortcut rather than genuine state discrimination
# — align_xy's old fixed "I need to move over the block first" collapsed
# 54% of true align_xy examples into "move_to_target" (the only other label
# starting with "move_"), and hard-coding the SAME label's own word ("align")
# into its own template just teaches the opposite-direction version of the
# identical shortcut (keyword-spot your own reasoning, copy it into the
# tag, instead of learning the actual geometric rule). Picking one of
# several phrasings per example (seeded, not module-level random — keeps
# this pipeline's existing seed-in/seed-out reproducibility) means no
# single fixed string is available to latch onto.
ALIGN_XY_PHRASINGS = [
    "I need to align the gripper with the block.",
    "I need to move the gripper to the block.",
    "I need to take the gripper to the block.",
]
MOVE_TO_TARGET_PHRASINGS = [
    "The block is grasped and lifted — I need to carry it there.",
    "The block is in the gripper and above the table, so I can move it to the target.",
    "I need to take the block to the target.",
]


def make_subgoal_think_text(obs_arr, achieved_goal, desired_goal, subgoal: str, rng) -> str:
    """
    Grounded reasoning text for <think>, in the same "repeat the real
    numbers, then explain the decision" style as generate_sft_data.py's
    make_think_text. Reuses subgoal_reward's own geometry helper rather
    than recomputing d_xy/d_z/d_block_target a second, possibly
    inconsistent way.

    Shows the WORKED arithmetic (per-axis differences -> combined distance
    -> explicit threshold comparison) leading to each conclusion, not just
    the conclusion itself. Confirmed 2026-07-16 via a live demo rollout that
    a model trained on conclusion-only targets (e.g. "xy gap=0.018m" stated
    with no derivation) learned to hallucinate a plausible-looking SMALL
    number regardless of the real, often much larger, distance between its
    own stated gripper/block positions — close_gripper was picked 8/10 times
    from states 0.5-0.9m away, never once succeeding. Chain-of-thought
    scratchpad steps (each one individually a much easier next-token
    prediction than a silent full 3D-distance computation) are the standard
    mitigation for this in small models — the goal isn't for a 2B model to
    do exact float arithmetic, it's a learnable path to the right
    CATEGORICAL decision, so the threshold comparison is stated explicitly
    too, not left for the model to infer from a bare magnitude.

    Also states "Block grasped: yes/no" as a GIVEN fact (via
    subgoal_labeler.is_block_grasped), not something left for the model to
    derive from finger-width + geometry — added 2026-07-17 after a held-out
    eval showed close_gripper/lift recall collapsing to ~4% while the model
    over-predicted "release" as a fallback guess. Those are exactly the
    subgoals whose correct choice hinges on already knowing whether the
    block is grasped, and nothing previously stated that outright.

    `rng`: used only to pick among ALIGN_XY_PHRASINGS/MOVE_TO_TARGET_PHRASINGS
    (see their module-level docstring) — required, not optional, so this
    stays reproducible per-seed like everything else in this pipeline.

    Two more fixes (2026-07-20), targeting a held-out eval failure where
    align_xy/move_to_target acted as attractors for descend/close_gripper/
    lift: raw think traces showed the model computing xy_distance correctly
    (e.g. 0.009m against a 0.02m tolerance) but then asserting align_xy's
    fixed conclusion regardless of the number, and skipping dz/height_above_
    table entirely rather than getting them wrong — i.e. not (only) a
    numeric-comparison failure, but every approach-phase template except
    align_xy's already required computing dz, and every carry-phase
    template except lift's already required computing height_above_table —
    the two "smallest" templates were the only ones that let the model
    reach a conclusion without ever running the disambiguating check.
    1. All small deltas/thresholds now ALSO show an integer-millimetre form
       (`{x}m = {x_mm}mm`) alongside the original metre value — millimetre
       integers don't have the tokenization/place-value fragility that
       multi-decimal metre comparisons do (e.g. "9mm > 20mm" vs
       "0.009m > 0.02m"), a known small-LLM weak point independent of
       whether the underlying arithmetic was computed correctly.
    2. align_xy's template now also computes dz (unused for its own
       decision, but present) and move_to_target/release now also compute
       height_above_table first — every template within a phase now has
       the SAME required computation steps, so no label has a structurally
       shorter/different-shaped path to its conclusion for the model to
       default to.
    """
    from think_then_act.reward.subgoal_reward import _geometry, DEFAULT_WEIGHTS
    from think_then_act.training.subgoal_labeler import is_block_grasped

    W = DEFAULT_WEIGHTS
    g = _geometry(obs_arr, achieved_goal, desired_goal)
    grip_pos  = [round(float(v), 3) for v in g["grip_pos"]]
    block_pos = [round(float(v), 3) for v in g["block_pos"]]
    target    = [round(float(v), 3) for v in g["target_pos"]]
    fw        = round(g["total_finger_width"], 3)
    finger_state = "open" if fw > 0.07 else "closed"
    grasped   = is_block_grasped(obs_arr, achieved_goal, desired_goal)

    prefix = (f"Gripper at {grip_pos}, block at {block_pos}, target at {target}. "
              f"Fingers are {finger_state} (width={fw}m). "
              f"Block grasped: {'yes' if grasped else 'no'}. "
              f"Table height: {W.table_z}m.")

    def mm(x: float) -> int:
        # Values passed in are already rounded to 3 decimal places (mm
        # precision), so this is an exact unit conversion, not a second
        # rounding — see docstring above for why mm form exists at all.
        return round(x * 1000)

    # block - gripper, matching env/oracle.py's object_rel_pos convention.
    dx, dy = round(block_pos[0] - grip_pos[0], 3), round(block_pos[1] - grip_pos[1], 3)
    d_xy   = round(g["d_xy"], 3)
    d_z    = round(g["d_z"], 3)   # grip_z - block_z, same sign as _geometry

    # target - block, matching _geometry's d_block_target = ||desired - achieved||.
    tdx = round(target[0] - block_pos[0], 3)
    tdy = round(target[1] - block_pos[1], 3)
    tdz = round(target[2] - block_pos[2], 3)
    d_bt = round(g["d_block_target"], 3)

    height_above_table = round(block_pos[2] - W.table_z, 3)

    align_xy_threshold_mm       = mm(W.align_xy_threshold)
    descend_threshold_mm        = mm(W.descend_threshold)
    close_gripper_dxy_limit_mm  = mm(W.close_gripper_dxy_limit)
    close_gripper_dz_limit_mm   = mm(W.close_gripper_dz_limit)
    lift_height_mm              = mm(W.lift_height)
    move_to_target_threshold_mm = mm(W.move_to_target_threshold)

    reasons = {
        "align_xy": (
            f"dx = block_x - gripper_x = {block_pos[0]} - {grip_pos[0]} = {dx}m = {mm(dx)}mm. "
            f"dy = block_y - gripper_y = {block_pos[1]} - {grip_pos[1]} = {dy}m = {mm(dy)}mm. "
            f"xy_distance = sqrt(dx^2 + dy^2) = {d_xy}m = {mm(d_xy)}mm, which is greater than "
            f"the {align_xy_threshold_mm}mm alignment tolerance. "
            f"dz = gripper_z - block_z = {grip_pos[2]} - {block_pos[2]} = {d_z}m = {mm(d_z)}mm. "
            f"xy isn't aligned yet, so grasp height doesn't matter until that's fixed. "
            f"{ALIGN_XY_PHRASINGS[rng.integers(len(ALIGN_XY_PHRASINGS))]}"
        ),
        "descend": (
            f"dx = block_x - gripper_x = {block_pos[0]} - {grip_pos[0]} = {dx}m = {mm(dx)}mm. "
            f"dy = block_y - gripper_y = {block_pos[1]} - {grip_pos[1]} = {dy}m = {mm(dy)}mm. "
            f"xy_distance = sqrt(dx^2 + dy^2) = {d_xy}m = {mm(d_xy)}mm, already within "
            f"the {align_xy_threshold_mm}mm alignment tolerance. "
            f"dz = gripper_z - block_z = {grip_pos[2]} - {block_pos[2]} = {d_z}m = {mm(d_z)}mm, "
            f"which is greater than the {descend_threshold_mm}mm grasp-height tolerance. "
            f"I need to descend."
        ),
        "close_gripper": (
            f"dx = block_x - gripper_x = {block_pos[0]} - {grip_pos[0]} = {dx}m = {mm(dx)}mm. "
            f"dy = block_y - gripper_y = {block_pos[1]} - {grip_pos[1]} = {dy}m = {mm(dy)}mm. "
            f"xy_distance = sqrt(dx^2 + dy^2) = {d_xy}m = {mm(d_xy)}mm "
            f"(<= {close_gripper_dxy_limit_mm}mm tolerance). "
            f"dz = gripper_z - block_z = {grip_pos[2]} - {block_pos[2]} = {d_z}m = {mm(d_z)}mm "
            f"(<= {close_gripper_dz_limit_mm}mm tolerance) — positioned at the "
            f"block, fingers still open ({fw}m). Time to close the gripper."
        ),
        "lift": (
            f"height_above_table = block_z - table_z = {block_pos[2]} - {W.table_z} = "
            f"{height_above_table}m = {mm(height_above_table)}mm, which is less than "
            f"the {lift_height_mm}mm lift target. "
            f"The block is grasped (fingers at {fw}m) but still near table height. "
            f"I need to lift it clear of the table."
        ),
        "move_to_target": (
            f"height_above_table = block_z - table_z = {block_pos[2]} - {W.table_z} = "
            f"{height_above_table}m = {mm(height_above_table)}mm, at or above "
            f"the {lift_height_mm}mm lift threshold — the block is lifted clear of the table. "
            f"dx = target_x - block_x = {target[0]} - {block_pos[0]} = {tdx}m = {mm(tdx)}mm, "
            f"dy = {tdy}m = {mm(tdy)}mm, dz = {tdz}m = {mm(tdz)}mm. "
            f"block_target_distance = sqrt(dx^2+dy^2+dz^2) = {d_bt}m = {mm(d_bt)}mm, "
            f"which is greater than the {move_to_target_threshold_mm}mm delivery tolerance. "
            f"{MOVE_TO_TARGET_PHRASINGS[rng.integers(len(MOVE_TO_TARGET_PHRASINGS))]}"
        ),
        "release": (
            f"height_above_table = block_z - table_z = {block_pos[2]} - {W.table_z} = "
            f"{height_above_table}m = {mm(height_above_table)}mm. "
            f"dx = target_x - block_x = {target[0]} - {block_pos[0]} = {tdx}m = {mm(tdx)}mm, "
            f"dy = {tdy}m = {mm(tdy)}mm, dz = {tdz}m = {mm(tdz)}mm. "
            f"block_target_distance = sqrt(dx^2+dy^2+dz^2) = {d_bt}m = {mm(d_bt)}mm, "
            f"which is within the {move_to_target_threshold_mm}mm delivery tolerance. "
            f"The block is grasped and close enough to the target. Time to release it."
        ),
    }
    return f"{prefix} {reasons[subgoal]}"


def _is_physically_plausible(obs_arr, achieved_goal, desired_goal, z_tolerance: float = 0.02) -> bool:
    """
    Rejects frames where the gripper has ended up meaningfully BELOW the
    block/table -- a state no legitimate approach/grasp/lift/carry
    trajectory ever visits, but one unconstrained random actions (the
    `random` source's whole episode, and `seeded`'s small post-handoff
    jitter) CAN produce, since nothing stops a random dz from driving the
    arm down through/past the table. label_subgoal still labels these
    "correctly" per its own rules (e.g. a laterally-misaligned, buried
    gripper still reads as align_xy, since that check only looks at xy)
    but the SCENE itself is unrealistic/garbage -- caught 2026-07-17 via
    manual review of sample_sft_quality_check.py's output (a frame showing
    the gripper below the table, labeled align_xy, correctly per the
    rules but not something a real robot would ever be shown to imitate
    recovering from). Reuses _geometry's d_z (grip_z - block_z) rather
    than a second, independently-derived height check; z_tolerance=0.02
    matches the same precision SubgoalWeights.close_gripper_dz_limit
    already uses elsewhere, so states genuinely AT grasp height (d_z
    slightly negative from settling/noise) aren't wrongly rejected.
    """
    from think_then_act.reward.subgoal_reward import _geometry
    g = _geometry(obs_arr, achieved_goal, desired_goal)
    return g["d_z"] >= -z_tolerance


def make_example(obs, frame, subgoal: str, source: str, episode_id: int, rng) -> dict:
    import numpy as np
    from think_then_act.training.subgoal_labeler import is_block_grasped

    obs_arr = np.asarray(obs["observation"])
    think = make_subgoal_think_text(obs_arr, obs["achieved_goal"], obs["desired_goal"], subgoal, rng)
    # Stored separately (not just inside `think`) so the PROMPT-building
    # side (subgoal_sft_train.py's compute_loss/quick_eval, eval_subgoal_vlm.py)
    # can state it as a given fact too, matching what
    # policy.subgoal_vlm_policy.USER_PROMPT_TEMPLATE now expects at live
    # inference time — see is_block_grasped's docstring for why.
    grasped = is_block_grasped(obs_arr, obs["achieved_goal"], obs["desired_goal"])
    return {
        "episode"      : episode_id,   # groups steps from the same episode/reset — for a
                                        # leakage-free train/val split, same convention as
                                        # generate_sft_data.py
        "source"       : source,       # "chained" | "seeded" | "random"
        "frame_b64"    : frame_to_b64(frame),
        "gripper_pos"  : [float(v) for v in obs_arr[0:3]],
        "achieved_goal": [float(v) for v in obs["achieved_goal"]],
        "desired_goal" : [float(v) for v in obs["desired_goal"]],
        "is_grasped"   : bool(grasped),
        "think"        : think,
        "subgoal"      : subgoal,
    }


def _run_skill(skill, base_env, obs) -> tuple:
    """
    Mirrors hrl.skill_env.SkillEnv.step()'s internal loop for ONE skill,
    but takes/returns obs explicitly instead of tracking it internally —
    SkillEnv only lets you seed its starting state via .reset(), which
    always does a fresh base_env.reset() with no chance to randomize
    block/target in between (env.setup.init_random_episode expects to run
    AFTER reset but BEFORE the first skill call). Duck-types against the
    same Skill fields SkillEnv itself uses, so this stays a thin
    reordering, not a second implementation of the stepping logic.
    """
    for _ in range(skill.max_steps):
        obs_vec = skill.build_obs(obs, base_env)
        action = skill.policy.act(obs_vec, deterministic=True)
        obs, _, terminated, truncated, info = base_env.step(action)
        _, done = skill.reward_and_done(obs, base_env)
        if done or terminated or truncated:
            return obs, terminated, truncated
    return obs, False, False


def run_chained_episode(skills: dict, base_env, rng, episode_id: int) -> list:
    from think_then_act.env.setup import init_random_episode, randomize_gripper_start
    from think_then_act.training.subgoal_labeler import label_subgoal

    base_env.reset()
    obs, ok = init_random_episode(base_env, rng)
    if not ok:
        return []
    # init_random_episode randomizes block/target but never the gripper's
    # own starting position -- confirmed 2026-07-17 via a position-
    # variability audit that every fresh-reset align_xy example otherwise
    # shares the IDENTICAL gripper number (0 variance across 150 examples).
    obs, ok, _info = randomize_gripper_start(base_env, rng, obs)
    if not ok:
        return []

    examples = []
    for subgoal in ("align_xy", "descend", "close_gripper"):
        frame = base_env.last_frame()
        label = label_subgoal(obs["observation"], obs["achieved_goal"], obs["desired_goal"])
        if label is not None and _is_physically_plausible(
            obs["observation"], obs["achieved_goal"], obs["desired_goal"]
        ):
            examples.append(make_example(obs, frame, label, "chained", episode_id, rng))

        obs, terminated, truncated = _run_skill(skills[subgoal], base_env, obs)
        if terminated or truncated:
            return examples

    frame = base_env.last_frame()
    label = label_subgoal(obs["observation"], obs["achieved_goal"], obs["desired_goal"])
    if label is not None and _is_physically_plausible(
        obs["observation"], obs["achieved_goal"], obs["desired_goal"]
    ):
        examples.append(make_example(obs, frame, label, "chained", episode_id, rng))
    return examples


def run_seeded_subgoal(base_env, subgoal: str, rng, episode_id: int, n_jitter_steps: int,
                        max_align_steps: int = 50) -> list:
    import numpy as np
    from think_then_act.env.oracle import oracle_action
    from think_then_act.env.setup import init_episode_before_subgoal
    from think_then_act.training.subgoal_labeler import label_subgoal

    base_env.reset()
    # randomize_gripper=True -- see run_chained_episode's comment; opt-in
    # here specifically so RL training's OWN calls to this function (via
    # SubgoalConditionedEnv.reset(), which never passes this) stay
    # byte-for-byte unaffected.
    obs, ok = init_episode_before_subgoal(base_env, rng, subgoal, randomize_gripper=True)
    if not ok:
        return []

    # init_episode_before_subgoal's handoff point is tuned for RL TRAINING
    # of `subgoal`'s own skill in isolation, not for matching
    # label_subgoal's thresholds -- confirmed 2026-07-16: move_to_target's
    # handoff (_subgoal_setup_reached's "block_z > 0.45", i.e. barely off
    # the table) lands well short of label_subgoal's is_lifted gate
    # (reward_lift's lift_height=0.10), so every seeded move_to_target
    # attempt was silently absorbed into "lift" labels instead. align_xy/
    # descend get no oracle fast-forward at all from init_episode_before_
    # subgoal (a fresh reset already IS align_xy's start state), so
    # "descend" specifically was never actually being targeted either.
    # Rather than re-tuning setup.py's thresholds to match (or vice versa,
    # risking desyncing them from what they're each individually tuned
    # for), keep driving the SAME scripted oracle init_episode_before_
    # subgoal already used internally, until label_subgoal itself agrees
    # this is a `subgoal` state -- that's self-consistent by construction,
    # not by coincidentally-matched constants.
    carrying = False
    for _ in range(max_align_steps):
        current_label = label_subgoal(obs["observation"], obs["achieved_goal"], obs["desired_goal"])
        if current_label == subgoal:
            break
        if current_label == "lift":
            # oracle_action's CARRY branch triggers the moment block_z>0.45
            # (its own "just cleared the table" heuristic) and immediately
            # beelines toward desired_goal -- which sits at table height for
            # most targets, so that straight line descends right away and
            # never climbs to reward_lift's stricter is_lifted gate (0.10m).
            # There's no trained lift skill yet to chain into for real (see
            # module docstring), so approximate one here: climb straight up
            # until label_subgoal agrees it's actually lifted, THEN let the
            # oracle carry from a state that matches what a real lift skill
            # handoff would look like.
            action = np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float32)
            carrying = False
        else:
            action, _phase, carrying = oracle_action(
                obs["observation"], obs["achieved_goal"], obs["desired_goal"], carrying)
        obs, _, terminated, truncated, _ = base_env.step(action)
        if terminated or truncated:
            return []
    else:
        return []  # never converged to `subgoal` within budget

    examples = []
    for _ in range(n_jitter_steps + 1):
        frame = base_env.last_frame()
        label = label_subgoal(obs["observation"], obs["achieved_goal"], obs["desired_goal"])
        if label is not None and _is_physically_plausible(
            obs["observation"], obs["achieved_goal"], obs["desired_goal"]
        ):
            examples.append(make_example(obs, frame, label, "seeded", episode_id, rng))
        # Small perturbation, not full [-1, 1] — the point is local variety
        # around the seeded state, not immediately leaving it.
        action = rng.uniform(-0.3, 0.3, size=4).astype(np.float32)
        obs, _, terminated, truncated, _ = base_env.step(action)
        if terminated or truncated:
            break
    return examples


def run_random_policy_episode(base_env, rng, episode_id: int, max_steps: int) -> list:
    import numpy as np
    from think_then_act.env.setup import init_random_episode, randomize_gripper_start
    from think_then_act.training.subgoal_labeler import label_subgoal

    base_env.reset()
    obs, ok = init_random_episode(base_env, rng)
    if not ok:
        return []
    obs, ok, _info = randomize_gripper_start(base_env, rng, obs)
    if not ok:
        return []

    examples = []
    for _ in range(max_steps):
        frame = base_env.last_frame()
        label = label_subgoal(obs["observation"], obs["achieved_goal"], obs["desired_goal"])
        if label is not None and _is_physically_plausible(
            obs["observation"], obs["achieved_goal"], obs["desired_goal"]
        ):
            examples.append(make_example(obs, frame, label, "random", episode_id, rng))
        action = rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
        obs, _, terminated, truncated, _ = base_env.step(action)
        if terminated or truncated:
            break
    return examples


def _load_actor(ckpt_path: str, obs_dim: int):
    import torch
    from think_then_act.policy.subgoal_policy import SubgoalGaussianPolicy
    actor = SubgoalGaussianPolicy(obs_dim=obs_dim)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # PPO checkpoints (train_low_level_ppo.py) wrap {"actor":..., "critic":...};
    # GRPO checkpoints are a bare actor state_dict — same distinction
    # record_subgoal_video.py handles.
    actor.load_state_dict(ckpt["actor"] if isinstance(ckpt, dict) and "actor" in ckpt else ckpt)
    actor.eval()
    return actor


# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------

@app.function(
    image=rl_image,
    gpu=None,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=3600,
)
def generate_subgoal_sft_data(
    n_chained_episodes: int = 150,
    # Seeded is the ONLY source covering lift/move_to_target/release, and
    # (per debug run 2026-07-16) gives roughly balanced per-label yield
    # across all 6 -- it's the main lever for fixing the align_xy-dominant
    # skew random-policy rollouts produce, so it gets by far the largest
    # budget, not chained/random.
    n_seeded_per_subgoal: int = 150,
    # Cut hard from 40 -- the debug run showed random-policy rollouts
    # (previously 90/160 = 56% of all examples) are almost entirely
    # align_xy, since a short random walk rarely drifts far enough to
    # register as anything else. Kept only as a small OOD-robustness slice.
    n_random_episodes: int = 10,
    # Also cut from 30 -- a long single random walk mostly produces
    # near-duplicate align_xy frames; more, shorter episodes (more distinct
    # resets) beat one long one for the same total step budget.
    max_random_steps: int = 15,
    seeded_jitter_steps: int = 2,
    algo: str = "ppo",
    use_best: bool = False,
    verbose_every: int = 0,
) -> dict:
    import os, json
    import numpy as np
    import torch

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.perception.collision_predictor import CollisionPredictor
    from think_then_act.reward.subgoal_reward import SUBGOAL_LABELS
    from think_then_act.training.checkpoints import resolve_subgoal_checkpoint
    from think_then_act.training.fetch_skills import build_fetch_skills
    from think_then_act.training.subgoal_features import SUBGOAL_OBS_DIM

    print(f"\n{'='*60}")
    print(f"  HIGH-LEVEL VLM SFT DATA GENERATION")
    print(f"  chained={n_chained_episodes}  seeded={n_seeded_per_subgoal}/subgoal"
          f"  random={n_random_episodes}")
    print(f"{'='*60}")

    ckpt_dir = os.path.join(MODEL_CACHE_DIR, "checkpoints")

    collision_model = None
    collision_ckpt = os.path.join(ckpt_dir, "collision_predictor.pt")
    if os.path.exists(collision_ckpt):
        collision_model = CollisionPredictor()
        collision_model.load_state_dict(torch.load(collision_ckpt, map_location="cpu"))
        collision_model.eval()
        print(f"  collision model   <- {collision_ckpt}")

    # Only align_xy/descend/close_gripper have trained, verified policies
    # (per hierarchical_architecture memory, 2026-07-16). lift/
    # move_to_target/release are covered by the seeded phase below instead
    # of chained skill execution — there's no policy yet to run them.
    trained_subgoals = ("align_xy", "descend", "close_gripper")
    policies = {}
    for subgoal in trained_subgoals:
        ckpt = resolve_subgoal_checkpoint(ckpt_dir, subgoal, algo=algo, use_best=use_best)
        policies[subgoal] = _load_actor(ckpt, obs_dim=SUBGOAL_OBS_DIM)
        print(f"  {subgoal:14s}    <- {ckpt}")

    skills = build_fetch_skills(policies, collision_model)

    base_env = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                  # +350, not +250: the base +250 (same rationale as
                  # rollout_workers.py/record_subgoal_video.py) only covered
                  # init_episode_before_subgoal's own up-to-200-step oracle
                  # setup. run_seeded_subgoal's WORST case (e.g. "release")
                  # now stacks randomize_gripper_start (<=40 steps, added
                  # 2026-07-17) + that 200-step setup + its own up-to-50-step
                  # align loop + jitter -- ~290 steps before max_random_steps
                  # is even counted, which would have silently exceeded the
                  # old +250 budget (a mid-setup truncation just shows up as
                  # an extra skipped episode, not a crash, so this is easy to
                  # under-notice) -- +350 leaves real margin above that.
                  max_episode_steps=max_random_steps + 350)
    )
    setup_env(base_env)

    all_examples  = []
    label_counts  = {label: 0 for label in SUBGOAL_LABELS}
    source_counts = {"chained": 0, "seeded": 0, "random": 0}
    skip_count    = 0
    episode_id    = 0

    def _record(examples: list, source: str):
        nonlocal skip_count
        if not examples:
            skip_count += 1
            return
        for ex in examples:
            label_counts[ex["subgoal"]] += 1
        source_counts[source] += len(examples)
        all_examples.extend(examples)

    print(f"\n  Chained rollouts ({n_chained_episodes})...")
    for i in range(n_chained_episodes):
        rng = np.random.default_rng(1_000_000 + i)
        examples = run_chained_episode(skills, base_env, rng, episode_id)
        episode_id += 1
        _record(examples, "chained")
        if verbose_every and i % verbose_every == 0:
            print(f"    ep={i:4d}  n_examples={len(examples)}")

    print(f"\n  Seeded resets ({n_seeded_per_subgoal} x {len(SUBGOAL_LABELS)} subgoals)...")
    for subgoal in SUBGOAL_LABELS:
        for i in range(n_seeded_per_subgoal):
            rng = np.random.default_rng(2_000_000 + SUBGOAL_LABELS.index(subgoal) * 100_000 + i)
            examples = run_seeded_subgoal(base_env, subgoal, rng, episode_id, seeded_jitter_steps)
            episode_id += 1
            _record(examples, "seeded")
        print(f"    {subgoal:14s}  label_count={label_counts[subgoal]}")

    print(f"\n  Random-policy rollouts ({n_random_episodes})...")
    for i in range(n_random_episodes):
        rng = np.random.default_rng(3_000_000 + i)
        examples = run_random_policy_episode(base_env, rng, episode_id, max_random_steps)
        episode_id += 1
        _record(examples, "random")

    base_env.close()

    out_path = os.path.join(MODEL_CACHE_DIR, "subgoal_sft_data.jsonl")
    with open(out_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    model_volume.commit()

    print(f"\n  Done. {len(all_examples)} examples -> {out_path}")
    print(f"  Label distribution : {label_counts}")
    print(f"  Source distribution: {source_counts}")
    print(f"  Skipped episodes   : {skip_count}")
    print(f"{'='*60}")

    return {
        "n_examples"   : len(all_examples),
        "label_counts" : label_counts,
        "source_counts": source_counts,
        "skip_count"   : skip_count,
        "output_path"  : out_path,
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(debug: bool = False):
    # .spawn(), not .remote() -- .remote() blocks on the CLI's connection to
    # receive the return value, so under `modal run --detach` (meant to let
    # the job outlive the CLI disconnecting) the still-in-flight call gets
    # cancelled the moment the CLI exits after dispatch. Confirmed
    # 2026-07-20: a full-run --detach invocation was killed by exactly this
    # ("Received a cancellation signal") right after printing the header,
    # before any real per-episode progress. .spawn() is fire-and-forget and
    # survives that, same as subgoal_sft_train.py/eval_subgoal_vlm.py/
    # record_subgoal_demo.py's local entrypoints already do.
    if debug:
        print("\nRunning DEBUG mode (small counts, verbose)...")
        # Scaled down but same ~15:15:1 chained:seeded:random ratio as the
        # real defaults, not an independently-picked small mix -- otherwise
        # debug's own label distribution (which is what past runs actually
        # used to catch the align_xy skew and the move_to_target bug) would
        # preview a different balance than the full run produces.
        handle = generate_subgoal_sft_data.spawn(
            n_chained_episodes=5, n_seeded_per_subgoal=5, n_random_episodes=1,
            verbose_every=1,
        )
    else:
        handle = generate_subgoal_sft_data.spawn()

    print(f"\nJob spawned. Function call ID: {handle.object_id}")
    print(f"Monitor at https://modal.com")
    print(f"Output will be saved to: {MODEL_CACHE_DIR}/subgoal_sft_data.jsonl")
    print(f"\nDownload when finished:")
    print(f"  modal volume get rl-harness-model-cache subgoal_sft_data.jsonl ./artifacts/")
    print(f"\nDownload with:")
    print(f"  modal volume get rl-harness-model-cache subgoal_sft_data.jsonl ./artifacts/subgoal_sft_data.jsonl")
