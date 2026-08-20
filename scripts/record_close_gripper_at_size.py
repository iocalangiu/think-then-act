"""
record_close_gripper_at_size.py

Records a close_gripper rollout at an EXPLICIT, PINNED block size, rather
than a randomly-sampled one — verify_close_gripper_grasp.py/
record_subgoal_video.py don't offer this, they either use the fixed 5cm
default or sample randomly per episode. Built to directly compare a size
FROM the training distribution ([0.01, 0.08]) against one OUTSIDE it, on
the SAME checkpoint, to see whether the policy actually generalizes or
only memorized the trained range.

Defaults to a uniform cube (width=length=height=size), but --width/
--length/--height each independently override that one axis — e.g. taking
a 5cm cube and stretching just the height to 7cm, to test a genuine
non-cube shape rather than only isotropic scaling.

Mechanism: uses SubgoalConditionedEnv(randomize_block_size=True,
size_range=(size, size), width_range=..., ...) — a degenerate (x, x) range
collapses block_randomization.py's sampler to always pick exactly x on
that axis (see init_random_episode's size_range/width_range/etc. params),
so this reuses the SAME setup/spawn-height/oracle-handoff path every real
training episode goes through, not a hand-rolled shortcut.

--use-pose-model (default True, matching record_full_rollout.py's convention):
when set, close_gripper's observation uses block_pose_predictor.pt's learned
position estimate instead of ground-truth achieved_goal — same None-means-
unchanged wiring subgoal_env.py already supports, just never plumbed through
here before. Pass --no-use-pose-model to force ground truth (isolates the
policy's own capability from the pose model's, the same A/B this project
already ran for align_xy on 2026-08-19).

Run with:
    modal run scripts/record_close_gripper_at_size.py --size 0.05
    modal run scripts/record_close_gripper_at_size.py --size 0.005 --tag out_of_dist
    modal run scripts/record_close_gripper_at_size.py --size 0.05 --use-best
    modal run scripts/record_close_gripper_at_size.py --size 0.05 --height 0.07 --use-best --tag tall
    modal run scripts/record_close_gripper_at_size.py --size 0.01 --no-use-pose-model  # ground truth only
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(
    image=rl_image, gpu=None, volumes={MODEL_CACHE_DIR: model_volume}, timeout=300,
)
def record_close_gripper_at_size(
    size: float = 0.05,
    width: float = 0.0,    # 0.0 = use `size` for this axis (default cube)
    length: float = 0.0,   # 0.0 = use `size` for this axis
    height: float = 0.0,   # 0.0 = use `size` for this axis
    tag: str = "",
    seed: int = 0,
    max_steps: int = 30,
    lift_steps: int = 8,
    ckpt_iter: int = 0,
    use_best: bool = False,
    video_fps: int = 10,
    use_pose_model: bool = True,
) -> dict:
    import os
    import numpy as np
    import torch

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env, save_video as save_video_fn
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.env.block_randomization import get_block_dims
    from think_then_act.perception.block_pose_predictor import BlockPosePredictor
    from think_then_act.policy.subgoal_policy import SubgoalGaussianPolicy
    from think_then_act.training.checkpoints import resolve_subgoal_checkpoint
    from think_then_act.training.subgoal_env import SubgoalConditionedEnv
    from think_then_act.training.subgoal_features import CLOSE_GRIPPER_OBS_DIM

    subgoal = "close_gripper"
    ckpt_dir = os.path.join(MODEL_CACHE_DIR, "checkpoints")
    if ckpt_iter > 0 and not use_best:
        ckpt_path = os.path.join(ckpt_dir, f"low_level_{subgoal}_ppo_iter{ckpt_iter}.pt")
    else:
        ckpt_path = resolve_subgoal_checkpoint(ckpt_dir, subgoal, algo="ppo", use_best=use_best)

    width_val  = width  or size
    length_val = length or size
    height_val = height or size

    print("\n" + "=" * 60)
    print(f"  close_gripper @ width={width_val} length={length_val} height={height_val}  tag={tag!r}")
    print(f"  checkpoint -> {ckpt_path}")
    print("=" * 60)

    policy = SubgoalGaussianPolicy(obs_dim=CLOSE_GRIPPER_OBS_DIM)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    policy.load_state_dict(ckpt["actor"] if isinstance(ckpt, dict) and "actor" in ckpt else ckpt)
    policy.eval()

    pose_model = None
    pose_ckpt = os.path.join(ckpt_dir, "block_pose_predictor.pt")
    if use_pose_model and os.path.exists(pose_ckpt):
        pose_model = BlockPosePredictor()
        pose_model.load_state_dict(torch.load(pose_ckpt, map_location="cpu"))
        pose_model.eval()
        print(f"  pose model        <- {pose_ckpt}")
    elif not use_pose_model:
        print(f"  pose model disabled (--no-use-pose-model) — using ground-truth achieved_goal")

    base = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array", max_episode_steps=max_steps + 250)
    )
    setup_env(base)
    # Degenerate (x, x) ranges -> block_randomization.py's sampler always
    # picks exactly that value on each axis, going through the SAME
    # spawn-height/oracle-handoff path a real training episode uses.
    env = SubgoalConditionedEnv(
        base, subgoal=subgoal, max_episode_steps=max_steps,
        randomize_block_size=True, pose_model=pose_model,
        width_range=(width_val, width_val), length_range=(length_val, length_val),
        height_range=(height_val, height_val),
    )
    rng = np.random.default_rng(seed)
    obs, info = env.reset(rng=rng)
    actual_dims = get_block_dims(base.unwrapped.model)
    print(f"  actual sampled dims: {actual_dims}")

    frames = [base.last_frame()]
    proxy_done, proxy_step = False, None
    for step in range(max_steps):
        action = policy.act(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(base.last_frame())
        if info.get("done", False):
            proxy_done, proxy_step = True, step
            break
        if terminated or truncated:
            break

    verified = None
    if proxy_done:
        d_grip_block_at_claim = float(info["d_grip_block"])
        block_z_at_claim = float(info["block_pos"][2])
        lift_action = np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float32)
        for _ in range(lift_steps):
            obs, reward, terminated, truncated, info = env.step(lift_action)
            frames.append(base.last_frame())
        block_rise = float(info["block_pos"][2]) - block_z_at_claim
        d_grip_block_after_lift = float(info["d_grip_block"])
        verified = bool(d_grip_block_after_lift <= 0.05)
        print(f"  proxy_step={proxy_step}  d_grip_block(claim={d_grip_block_at_claim:.4f}, "
              f"after_lift={d_grip_block_after_lift:.4f})  block_rise={block_rise:.4f}  "
              f"verified={'YES' if verified else 'NO -- FALSE POSITIVE'}")
    else:
        print(f"  proxy_done=False (no claimed success across {max_steps} steps)")

    env.close()

    out_dir = os.path.join(MODEL_CACHE_DIR, "close_gripper_size_videos")
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    name = f"w{width_val:.4f}_l{length_val:.4f}_h{height_val:.4f}{suffix}.mp4"
    video_path = os.path.join(out_dir, name)
    save_video_fn(frames, video_path, fps=video_fps)
    model_volume.commit()
    print(f"  {len(frames)} frames -> {video_path}")

    return {
        "status": "PASS", "width": width_val, "length": length_val, "height": height_val,
        "actual_dims": actual_dims,
        "proxy_done": proxy_done, "proxy_step": proxy_step, "verified": verified,
        "video_path": video_path, "n_frames": len(frames),
    }


@app.local_entrypoint()
def main(
    size: float = 0.05, width: float = 0.0, length: float = 0.0, height: float = 0.0,
    tag: str = "", seed: int = 0, max_steps: int = 30,
    lift_steps: int = 8, ckpt_iter: int = 0, use_best: bool = False, video_fps: int = 10,
    use_pose_model: bool = True,
):
    result = record_close_gripper_at_size.remote(
        size=size, width=width, length=length, height=height,
        tag=tag, seed=seed, max_steps=max_steps, lift_steps=lift_steps,
        ckpt_iter=ckpt_iter, use_best=use_best, video_fps=video_fps,
        use_pose_model=use_pose_model,
    )
    print(f"\nDone: {result}")
    print(f"\nDownload with:")
    print(f"  python3 -m modal volume get --force rl-harness-model-cache "
          f"close_gripper_size_videos/ ./artifacts/close_gripper_size_videos/")
