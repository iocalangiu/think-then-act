"""
record_align_xy_multi_location_video.py

Runs the trained align_xy policy against the block placed at several
DIFFERENT, explicitly-chosen table locations in ONE continuous video: for
each location the arm is freshly re-initialized (env.reset), the block is
teleported to that location (env.setup.teleport_block), then the policy
runs until done/max_steps. Frames from every location are concatenated
back to back into a single mp4, so watching it end to end is a fast visual
generalization check across the workspace rather than trusting one lucky/
unlucky seed.

Written after the 2026-09-01 align_xy retrain (frame-relative observation +
z-penalty, see subgoal_features.py's ALIGN_XY_OBS_DIM comment) specifically
to see how the new checkpoint holds up away from wherever training happened
to sample most often — obs_dim_for_subgoal("align_xy") already returns the
new 29-dim size, so this picks up whichever checkpoint shape is actually on
disk automatically, no special-casing needed here.

Same dual-camera (default angled view + synthetic top-down) frame layout as
record_subgoal_video.py, for the same reason: the angled default camera
foreshortens lateral (xy) distances, top-down makes them directly legible.

Run with:
    modal run scripts/record_align_xy_multi_location_video.py
    modal run scripts/record_align_xy_multi_location_video.py --locations "1.30,0.75;1.45,0.85;1.15,0.65"
    modal run scripts/record_align_xy_multi_location_video.py --max-steps 40 --stochastic

Output (on the volume, under /model-cache/subgoal_videos/):
    align_xy_multi_location.mp4
    align_xy_multi_location_summary.json — per-location success/final d_xy/steps

Download with:
    modal volume get rl-harness-model-cache subgoal_videos/align_xy_multi_location.mp4 ./artifacts/subgoal_videos/
    modal volume get rl-harness-model-cache subgoal_videos/align_xy_multi_location_summary.json ./artifacts/subgoal_videos/
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR

# Table disk is centre [1.30, 0.75], r=0.20 (env/setup.py's
# init_random_episode) — these five points deliberately span it: center,
# then four points at r=0.15 in +x/-x/+y/-y, comfortably inside the disk
# (margin from the r=0.20 boundary for the gripper's own footprint) but
# spread across the whole reachable workspace rather than clustering near
# wherever training happened to sample most.
DEFAULT_LOCATIONS = "1.30,0.75;1.45,0.75;1.15,0.75;1.30,0.90;1.30,0.60"


@app.function(
    image=rl_image,
    gpu=None,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=600,
)
def record_align_xy_multi_location_video(
    locations: str = DEFAULT_LOCATIONS,   # "x1,y1;x2,y2;..."
    max_steps: int = 30,
    seed: int = 0,
    stochastic: bool = False,
    algo: str = "ppo",       # matches how align_xy was just retrained (train_low_level_ppo.py)
    use_best: bool = True,   # low_level_align_xy_ppo_best.pt — the highest-completion_rate
                              # checkpoint tracked during training, not just the final one
    ckpt_iter: int = 0,      # >0 with use_best=False: pin an explicit iter checkpoint instead
    use_pose_model: bool = True,
) -> dict:
    import os, json
    import numpy as np
    import torch
    import mujoco

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from PIL import Image, ImageDraw, ImageFont

    from think_then_act.env.setup import setup_env, save_video, teleport_block
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.perception.block_pose_predictor import BlockPosePredictor
    from think_then_act.perception.collision_predictor import CollisionPredictor
    from think_then_act.policy.subgoal_policy import SubgoalGaussianPolicy
    from think_then_act.training.checkpoints import resolve_subgoal_checkpoint
    from think_then_act.training.subgoal_env import SubgoalConditionedEnv
    from think_then_act.training.subgoal_features import obs_dim_for_subgoal

    subgoal = "align_xy"
    ckpt_dir = os.path.join(MODEL_CACHE_DIR, "checkpoints")
    suffix = "_ppo" if algo == "ppo" else ""

    if ckpt_iter > 0 and not use_best:
        ckpt_path = os.path.join(ckpt_dir, f"low_level_{subgoal}{suffix}_iter{ckpt_iter}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    else:
        ckpt_path = resolve_subgoal_checkpoint(ckpt_dir, subgoal, algo=algo, use_best=use_best)
    print(f"  checkpoint -> {ckpt_path}")

    policy = SubgoalGaussianPolicy(obs_dim=obs_dim_for_subgoal(subgoal))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # PPO checkpoints (low_level_ppo.py's save_checkpoint) are
    # {"actor": ..., "critic": ...}; GRPO checkpoints are a flat state_dict.
    policy.load_state_dict(ckpt["actor"] if isinstance(ckpt, dict) and "actor" in ckpt else ckpt)
    policy.eval()

    collision_model = None
    collision_ckpt = os.path.join(ckpt_dir, "collision_predictor.pt")
    if os.path.exists(collision_ckpt):
        collision_model = CollisionPredictor()
        collision_model.load_state_dict(torch.load(collision_ckpt, map_location="cpu"))
        collision_model.eval()

    pose_model = None
    pose_ckpt = os.path.join(ckpt_dir, "block_pose_predictor.pt")
    if use_pose_model and os.path.exists(pose_ckpt):
        pose_model = BlockPosePredictor()
        pose_model.load_state_dict(torch.load(pose_ckpt, map_location="cpu"))
        pose_model.eval()
        print(f"  pose model <- {pose_ckpt}")

    points = []
    for chunk in locations.split(";"):
        x_str, y_str = chunk.split(",")
        points.append((float(x_str), float(y_str)))
    if not points:
        raise ValueError(f"No locations parsed from {locations!r}")
    print(f"  {len(points)} locations: {points}")

    # +5, not +250 (unlike record_subgoal_video.py's make_env) — align_xy
    # has no oracle pre-subgoal setup to budget for (it's the first
    # subgoal), the only extra step here is this script's own one-time
    # zero-action refresh after teleporting the block each location.
    base = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                  max_episode_steps=max_steps + 5)
    )
    setup_env(base)
    env = SubgoalConditionedEnv(base, subgoal=subgoal, collision_model=collision_model,
                                 pose_model=pose_model, max_episode_steps=max_steps)

    # Second camera, straight down over the table — same construction as
    # record_subgoal_video.py's make_topdown_env/combined_frame (see that
    # script's comment for why this is a SEPARATE gym.make(...) env whose
    # physics are mirrored from the primary each frame, not a second
    # Renderer sharing the primary's OSMesa context — that corrupts both).
    cam_cfg = {"distance": 3.0, "azimuth": 90.0, "elevation": -90.0,
               "lookat": np.array([1.30, 0.75, 0.40])}
    topdown_env = gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                            default_camera_config=cam_cfg)
    # MUST match the primary env's setup_env() call — that shifts the
    # robot base body_pos to [0.85, 0.75, 0.0] on the MODEL, not just
    # per-step state. Without this, combined_frame()'s qpos/qvel copy from
    # primary to mirror computes forward kinematics against a DIFFERENT
    # base offset than the primary env used, so the rendered arm ends up
    # mispositioned/rotated relative to the (unaffected, correctly placed)
    # table/block/target — confirmed 2026-09-01 from an actual multi-
    # location video screenshot: table/block/target were correct in the
    # top-down view, only the robot was wrong. Same latent bug also exists
    # in record_subgoal_video.py's make_topdown_env(), not fixed there yet.
    setup_env(topdown_env)
    topdown_env.reset()

    font_normal = ImageFont.load_default(size=16)
    font_bold   = ImageFont.load_default(size=20)

    def annotate(frame, label, status=None):
        img = Image.fromarray(frame).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), label, font=font_bold, fill=(0, 0, 0),
                   stroke_width=1, stroke_fill=(0, 0, 0))
        if status is not None:
            color = (0, 140, 0) if status == "SUCCESS" else (0, 0, 0)
            draw.text((10, 36), status, font=font_normal, fill=color)
        return np.array(img)

    def combined_frame(label, status=None):
        primary = base.unwrapped
        mirror = topdown_env.unwrapped
        mirror.data.qpos[:] = primary.data.qpos[:]
        mirror.data.qvel[:] = primary.data.qvel[:]
        mujoco.mj_forward(mirror.model, mirror.data)
        combined = np.concatenate([base.last_frame(), topdown_env.render()], axis=1)
        return annotate(combined, label, status)

    rng = np.random.default_rng(seed)
    all_frames = []
    summary = []
    resting_z = 0.425   # same fixed-5cm-cube resting height env/setup.py's
                         # init_random_episode uses when block size isn't randomized

    for i, (x, y) in enumerate(points):
        label = f"Location {i+1}/{len(points)}: ({x:.2f}, {y:.2f})"
        print(f"\n[{i+1}/{len(points)}] {label}")

        obs, info = env.reset(rng=rng)
        # Overrides wherever reset's own init_random_episode randomly
        # placed the block — deliberate control over the test location
        # for a systematic sweep, same mechanism as
        # measure_grip_contact_force_by_size.py's use of teleport_block.
        teleport_block(base, np.array([x, y, resting_z]))
        obs, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        all_frames.append(combined_frame(label))

        success = False
        final_d_xy = None
        steps_run = 0
        for step in range(1, max_steps + 1):
            action = policy.act(obs, deterministic=not stochastic)
            obs, reward, terminated, truncated, info = env.step(action)
            steps_run = step
            done = info.get("done", False)
            if done:
                success = True
            all_frames.append(combined_frame(label, status="SUCCESS" if done else None))
            block_pos, grip_pos = info.get("block_pos"), info.get("grip_pos")
            if block_pos is not None and grip_pos is not None:
                final_d_xy = float(np.linalg.norm(
                    np.array(block_pos[:2]) - np.array(grip_pos[:2])
                ))
            if terminated or truncated:
                break

        d_xy_str = f"{final_d_xy:.4f}" if final_d_xy is not None else "n/a"
        print(f"  steps={steps_run}  final_d_xy={d_xy_str}  success={success}")
        summary.append({"location": [x, y], "steps": steps_run,
                         "final_d_xy": final_d_xy, "success": success})

    env.close()
    topdown_env.close()

    out_dir = os.path.join(MODEL_CACHE_DIR, "subgoal_videos")
    os.makedirs(out_dir, exist_ok=True)
    video_path = os.path.join(out_dir, "align_xy_multi_location.mp4")
    save_video(all_frames, video_path, fps=10)
    summary_path = os.path.join(out_dir, "align_xy_multi_location_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    model_volume.commit()

    n_success = sum(s["success"] for s in summary)
    print("\n" + "=" * 60)
    print(f"  {n_success}/{len(summary)} locations succeeded")
    for s in summary:
        print(f"  {s['location']}: success={s['success']}  final_d_xy={s['final_d_xy']}  steps={s['steps']}")
    print(f"  video   -> {video_path}")
    print(f"  summary -> {summary_path}")
    print("=" * 60)

    return {"status": "PASS", "video_path": video_path, "summary_path": summary_path, "summary": summary}


@app.local_entrypoint()
def main(
    locations: str = DEFAULT_LOCATIONS, max_steps: int = 30, seed: int = 0,
    stochastic: bool = False, algo: str = "ppo", use_best: bool = True,
    ckpt_iter: int = 0, use_pose_model: bool = True,
):
    print(f"\nRecording align_xy across locations={locations} algo={algo} use_best={use_best}...")
    result = record_align_xy_multi_location_video.remote(
        locations=locations, max_steps=max_steps, seed=seed, stochastic=stochastic,
        algo=algo, use_best=use_best, ckpt_iter=ckpt_iter, use_pose_model=use_pose_model,
    )
    n_success = sum(s["success"] for s in result["summary"])
    print(f"\nDone. {n_success}/{len(result['summary'])} locations succeeded.")
    print(f"\nDownload with:")
    print(f"  modal volume get rl-harness-model-cache subgoal_videos/align_xy_multi_location.mp4 ./artifacts/subgoal_videos/")
    print(f"  modal volume get rl-harness-model-cache subgoal_videos/align_xy_multi_location_summary.json ./artifacts/subgoal_videos/")
