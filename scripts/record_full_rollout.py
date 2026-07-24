"""
record_full_rollout.py

End-to-end visual sanity check chaining a (configurable, ordered) subset of
the six subgoal controllers together on ONE live episode — defaults to all
six: align_xy -> descend -> close_gripper -> lift -> move_to_target ->
release. Unlike record_subgoal_video.py (one subgoal, in isolation, with an
oracle-constructed starting state), this runs each trained policy in
sequence on whatever state the PREVIOUS policy actually left the arm in —
no oracle involved anywhere — so it's the real test of whether the
controllers compose into a working sequence, not just whether each one
works in isolation.

--subgoals lets the chain stop early, e.g. "align_xy,descend,close_gripper,
lift" to demo just grasp+lift without needing move_to_target/release
trained yet. Always executes in the canonical SUBGOAL_LABELS order
regardless of the order given (this is a physical sequence, not a set) and
only loads checkpoints for the requested subset — a partial chain doesn't
require every subgoal to have a trained policy.

Reuses hrl/skill_env.py's Skill abstraction (via training/fetch_skills.py's
build_fetch_skills) for build_obs/reward_and_done per subgoal, but drives
the loop directly here instead of going through SkillEnv, since this needs
a frame captured after every individual base-env step (for the video),
which SkillEnv's one-transition-per-skill-call abstraction doesn't expose.

Each subgoal runs for up to --max-steps-per-subgoal env steps or until its
own reward_and_done reports done, whichever comes first, then the NEXT
subgoal's policy takes over from wherever that left the arm — even if the
previous one didn't actually finish (its skill_success is still reported
in the summary printed at the end, so a failure to complete is visible,
not silently masked).

Two distinct success signals are reported, since they answer different
questions:
  - task_success: the base env's OWN success condition (block within
    threshold of desired_goal) — only meaningful if the chain runs all the
    way through move_to_target/release.
  - grasp_lift_success: GROUND TRUTH check of "is the block currently
    grasped AND elevated" at the end of whatever chain was requested —
    height_above_table >= lift_height AND d_grip_block <=
    close_gripper_drift_limit (both existing thresholds from
    reward/subgoal_reward.py's SubgoalWeights, not new arbitrary numbers).
    This is the meaningful signal for a chain that stops at `lift` — did
    the arm actually grab and hold the block, not just satisfy lift's own
    proxy `done` condition at one instant before possibly dropping it.
    Same "don't trust the proxy alone" spirit as
    scripts/verify_close_gripper_grasp.py.

Output video has the requested subgoal names listed top-right, always
black; the one currently executing is rendered bold (larger +
stroke-outlined — the default bitmap font has no real bold variant) so the
highlighted label visibly changes as the video plays.

Run with:
    modal run scripts/record_full_rollout.py
    modal run scripts/record_full_rollout.py --seed 3
    modal run scripts/record_full_rollout.py --subgoals align_xy,descend,close_gripper,lift
    modal run scripts/record_full_rollout.py --max-steps-per-subgoal 40 --algo ppo --use-best
    modal run scripts/record_full_rollout.py --algo grpo --no-use-best   # latest GRPO checkpoints

Output (on the volume, under /model-cache/subgoal_videos/):
    full_rollout_seed{seed}{_ppo}{_to_<last_subgoal> if not the full chain}.mp4

Download with:
    python3 -m modal volume get rl-harness-model-cache subgoal_videos/ ./artifacts/subgoal_videos/
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(
    image=rl_image,
    gpu=None,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=900,
)
def record_full_rollout(
    seed: int = 0,
    subgoals: str = "align_xy,descend,close_gripper,lift,move_to_target,release",
    max_steps_per_subgoal: int = 30,
    algo: str = "ppo",       # "grpo" or "ppo" — determines checkpoint filename pattern
    use_best: bool = True,   # per-subgoal best-by-completion-rate checkpoint rather than
                              # latest/final — matters because training can regress after
                              # peaking (see training/checkpoints.py)
    use_pose_model: bool = True,  # False: force ground-truth achieved_goal even if a
                              # block_pose_predictor.pt checkpoint exists — lets the chain
                              # be A/B'd with vs. without perception noise on the same seed.
) -> dict:
    import os
    import json
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env, save_video, init_random_episode
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.perception.block_pose_predictor import BlockPosePredictor
    from think_then_act.perception.collision_predictor import CollisionPredictor
    from think_then_act.policy.subgoal_policy import SubgoalGaussianPolicy
    from think_then_act.reward.subgoal_reward import SUBGOAL_LABELS, DEFAULT_WEIGHTS
    from think_then_act.training.checkpoints import resolve_subgoal_checkpoint
    from think_then_act.training.fetch_skills import build_fetch_skills
    from think_then_act.training.subgoal_features import SUBGOAL_OBS_DIM

    requested = {s.strip() for s in subgoals.split(",") if s.strip()}
    unknown = requested - set(SUBGOAL_LABELS)
    if unknown:
        raise ValueError(f"Unknown subgoal(s) {unknown}; must be a subset of {SUBGOAL_LABELS}")
    # Canonical order always wins — this is a physical sequence, not a set,
    # so the requested subgoals run in SUBGOAL_LABELS' order regardless of
    # how they were listed on the command line.
    subgoal_list = [s for s in SUBGOAL_LABELS if s in requested]

    print("\n" + "=" * 60)
    print(f"  CHAINED ROLLOUT  seed={seed}  subgoals={subgoal_list}  algo={algo}  use_best={use_best}")
    print("=" * 60)

    ckpt_dir = os.path.join(MODEL_CACHE_DIR, "checkpoints")

    # ------------------------------------------------------------------
    # Load one trained actor per REQUESTED subgoal only — a partial chain
    # (e.g. stopping at lift) doesn't need move_to_target/release trained.
    # Fails fast (naming exactly which subgoal) rather than silently
    # substituting an untrained policy for a missing checkpoint.
    # ------------------------------------------------------------------
    policies = {}
    for subgoal in subgoal_list:
        ckpt_path = resolve_subgoal_checkpoint(ckpt_dir, subgoal, algo=algo, use_best=use_best)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        policy = SubgoalGaussianPolicy(obs_dim=SUBGOAL_OBS_DIM)
        # PPO checkpoints (low_level_ppo.py's save_checkpoint) are
        # {"actor": ..., "critic": ...}; GRPO checkpoints are a flat
        # state_dict, loaded as-is — same convention as record_subgoal_video.py.
        policy.load_state_dict(ckpt["actor"] if isinstance(ckpt, dict) and "actor" in ckpt else ckpt)
        policy.eval()
        policies[subgoal] = policy
        print(f"  {subgoal:15s} <- {os.path.basename(ckpt_path)}")

    collision_model = None
    collision_ckpt = os.path.join(ckpt_dir, "collision_predictor.pt")
    if os.path.exists(collision_ckpt):
        collision_model = CollisionPredictor()
        collision_model.load_state_dict(torch.load(collision_ckpt, map_location="cpu"))
        collision_model.eval()

    # Optional block pose predictor — same as record_subgoal_demo.py; every
    # skill's build_obs uses its estimate instead of the privileged
    # achieved_goal when present (reward_and_done stays ground truth
    # regardless — see block_pose_predictor.py's docstring).
    pose_model = None
    pose_ckpt = os.path.join(ckpt_dir, "block_pose_predictor.pt")
    if use_pose_model and os.path.exists(pose_ckpt):
        pose_model = BlockPosePredictor()
        pose_model.load_state_dict(torch.load(pose_ckpt, map_location="cpu"))
        pose_model.eval()
        print(f"  pose model        <- {pose_ckpt}")
    elif not use_pose_model:
        print(f"  pose model disabled (--use-pose-model=False) — using ground-truth achieved_goal")

    skills = build_fetch_skills(policies, collision_model=collision_model,
                                 pose_model=pose_model, max_steps=max_steps_per_subgoal)

    # ------------------------------------------------------------------
    # One live env for the whole chained episode — no per-subgoal reset,
    # no oracle. +50 headroom on top of the worst case (every subgoal
    # burning its full budget) so the base env's own TimeLimit never cuts
    # a subgoal off mid-attempt.
    # ------------------------------------------------------------------
    base = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                  max_episode_steps=max_steps_per_subgoal * len(subgoal_list) + 50)
    )
    setup_env(base)
    base.reset()
    rng = np.random.default_rng(seed)
    obs, ok = init_random_episode(base, rng)
    if not ok:
        raise RuntimeError(f"init_random_episode failed for seed={seed}; try a different seed")

    # ------------------------------------------------------------------
    # Overlay: requested subgoal names, top-right, always black; the one
    # currently executing rendered larger + stroke-outlined to fake bold
    # (the default bitmap font — no ttf files baked into rl_image — has no
    # real bold variant).
    # ------------------------------------------------------------------
    font_normal = ImageFont.load_default(size=15)
    font_bold   = ImageFont.load_default(size=18)

    def annotate(frame: np.ndarray, current: str) -> np.ndarray:
        img = Image.fromarray(frame).convert("RGB")
        draw = ImageDraw.Draw(img)
        margin, line_h, y = 10, 24, 10
        for name in subgoal_list:
            is_current = name == current
            font = font_bold if is_current else font_normal
            stroke = 1 if is_current else 0
            bbox = draw.textbbox((0, 0), name, font=font, stroke_width=stroke)
            x = img.width - margin - (bbox[2] - bbox[0])
            draw.text((x, y), name, font=font, fill=(0, 0, 0),
                      stroke_width=stroke, stroke_fill=(0, 0, 0))
            y += line_h
        return np.array(img)

    frames = [annotate(base.last_frame(), subgoal_list[0])]
    summary = []
    info = {}
    ended_early = False

    # Per-step block/gripper trajectory across the WHOLE chain, not just one
    # subgoal — lets a failure be traced to the exact step/subgoal transition
    # it started at, instead of only seeing the final aggregate d_grip_block.
    # Same motivation as record_subgoal_video.py's trajectory log (which
    # already found descend's earlier 0.5m-drift bug this way).
    def _grip_pos(o):  return np.asarray(o["observation"][0:3], dtype=np.float64)
    def _block_pos(o): return np.asarray(o["achieved_goal"], dtype=np.float64)

    trajectory = [{
        "step": 0, "subgoal": None, "action": None,
        "block_pos": _block_pos(obs).tolist(), "grip_pos": _grip_pos(obs).tolist(),
        "d_grip_block": float(np.linalg.norm(_block_pos(obs) - _grip_pos(obs))),
    }]

    for subgoal in subgoal_list:
        skill = skills[subgoal]
        skill_success = False
        terminated = truncated = False
        steps_run = 0
        start_d_grip_block = trajectory[-1]["d_grip_block"]

        for steps_run in range(1, skill.max_steps + 1):
            obs_vec = skill.build_obs(obs, base)
            action = skill.policy.act(obs_vec, deterministic=True)
            obs, reward, terminated, truncated, info = base.step(action)
            frames.append(annotate(base.last_frame(), subgoal))
            trajectory.append({
                "step": len(trajectory), "subgoal": subgoal,
                "action": np.asarray(action, dtype=np.float64).tolist(),
                "block_pos": _block_pos(obs).tolist(), "grip_pos": _grip_pos(obs).tolist(),
                "d_grip_block": float(np.linalg.norm(_block_pos(obs) - _grip_pos(obs))),
            })

            _, skill_done = skill.reward_and_done(obs, base)
            if skill_done:
                skill_success = True
                break
            if terminated or truncated:
                break

        end_d_grip_block = trajectory[-1]["d_grip_block"]
        summary.append({"subgoal": subgoal, "steps": steps_run, "skill_success": skill_success,
                         "start_d_grip_block": round(start_d_grip_block, 4),
                         "end_d_grip_block": round(end_d_grip_block, 4)})
        print(f"  {subgoal:15s} steps={steps_run:3d}  skill_done={skill_success}  "
              f"d_grip_block(start={start_d_grip_block:.4f}, end={end_d_grip_block:.4f})")

        if terminated or truncated:
            print(f"  base env ended during {subgoal} (terminated={terminated} truncated={truncated})")
            ended_early = True
            break

    base.close()

    traj_path = os.path.join(MODEL_CACHE_DIR, "subgoal_videos",
                              f"full_rollout_seed{seed}_trajectory.json")
    os.makedirs(os.path.dirname(traj_path), exist_ok=True)
    with open(traj_path, "w") as f:
        json.dump(trajectory, f, indent=2)

    # ------------------------------------------------------------------
    # GROUND TRUTH check: is the block currently grasped AND elevated, at
    # whatever point the chain ended? Independent of any subgoal's own
    # `done` proxy — reuses obs["achieved_goal"] and obs["observation"]
    # directly (both untouched by pose_model; see sanitize_observation_
    # for_perception's docstring), same "verify physically, don't trust
    # the proxy alone" spirit as verify_close_gripper_grasp.py. Meaningful
    # once close_gripper/lift have run; harmless (just reports height~0,
    # not grasped) if the chain never got that far.
    # ------------------------------------------------------------------
    grip_pos  = np.asarray(obs["observation"][0:3], dtype=np.float64)
    block_pos = np.asarray(obs["achieved_goal"], dtype=np.float64)
    height_above_table = float(block_pos[2] - DEFAULT_WEIGHTS.table_z)
    d_grip_block = float(np.linalg.norm(block_pos - grip_pos))
    grasp_lift_success = bool(
        height_above_table >= DEFAULT_WEIGHTS.lift_height
        and d_grip_block <= DEFAULT_WEIGHTS.close_gripper_drift_limit
    )

    out_dir = os.path.join(MODEL_CACHE_DIR, "subgoal_videos")
    os.makedirs(out_dir, exist_ok=True)
    suffix = "_ppo" if algo == "ppo" else ""
    pose_tag = "_perceived" if pose_model is not None else ""
    chain_tag = "" if subgoal_list == SUBGOAL_LABELS else f"_to_{subgoal_list[-1]}"
    out_path = os.path.join(out_dir, f"full_rollout_seed{seed}{suffix}{pose_tag}{chain_tag}.mp4")
    save_video(frames, out_path, fps=10)
    model_volume.commit()

    task_success = bool(info.get("is_success", False))
    print("\n" + "=" * 60)
    print(f"  task_success={task_success}  (base env's own — only meaningful if the "
          f"chain reached move_to_target/release)")
    print(f"  grasp_lift_success={grasp_lift_success}  "
          f"(height_above_table={height_above_table:.4f}m >= {DEFAULT_WEIGHTS.lift_height}m AND "
          f"d_grip_block={d_grip_block:.4f}m <= {DEFAULT_WEIGHTS.close_gripper_drift_limit}m)")
    print(f"  ended_early={ended_early}  n_frames={len(frames)}")
    print(f"  Saved -> {out_path}")
    print(f"  Saved -> {traj_path}")
    print("=" * 60)

    return {
        "status": "PASS", "seed": seed, "summary": summary,
        "task_success": task_success, "grasp_lift_success": grasp_lift_success,
        "height_above_table": round(height_above_table, 4), "d_grip_block": round(d_grip_block, 4),
        "ended_early": ended_early, "video_path": out_path, "n_frames": len(frames),
        "trajectory_path": traj_path,
    }


@app.local_entrypoint()
def main(
    seed: int = 0, subgoals: str = "align_xy,descend,close_gripper,lift,move_to_target,release",
    max_steps_per_subgoal: int = 30,
    algo: str = "ppo", use_best: bool = True, use_pose_model: bool = True,
):
    print(f"\nRecording chained rollout ({subgoals}), "
          f"seed={seed} algo={algo} use_best={use_best} use_pose_model={use_pose_model}...")
    result = record_full_rollout.remote(
        seed=seed, subgoals=subgoals, max_steps_per_subgoal=max_steps_per_subgoal,
        algo=algo, use_best=use_best, use_pose_model=use_pose_model,
    )
    print(f"\nDone. task_success={result['task_success']}  grasp_lift_success={result['grasp_lift_success']}")
    for s in result["summary"]:
        print(f"  {s['subgoal']:15s} steps={s['steps']:3d}  skill_done={s['skill_success']}")
    print(f"  -> {result['video_path']}")
    print(f"\nDownload with:")
    print(f"  python3 -m modal volume get --force rl-harness-model-cache "
          f"subgoal_videos/ ./artifacts/subgoal_videos/")
