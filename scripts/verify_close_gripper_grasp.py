"""
verify_close_gripper_grasp.py

Diagnostic-only (no training-loop changes): checks whether close_gripper's
`done` (closedness Gaussian + tight dxy/dz gate, reward/subgoal_reward.py)
actually corresponds to a genuine, physically stable grasp, or a false
positive the proxy accepted anyway. `done` has already been fooled twice in
this project's history — "closed on far empty air" and "closed while
hovering above the block" — each patched by tightening the proxy further.
Rather than trust the proxy a third time, this runs the trained policy
until it claims `done`, then takes control away and applies a fixed
SCRIPTED lift action (matching env/oracle.py's own lift branch: dz=+1.0,
grip=-1.0) for a few more steps, and checks whether the block actually
comes along (d_grip_block stays small) instead of being left behind/falling.

Run with:
    modal run scripts/verify_close_gripper_grasp.py --use-best
    modal run scripts/verify_close_gripper_grasp.py --n-episodes 20 --lift-steps 10
    modal run scripts/verify_close_gripper_grasp.py --ckpt-iter 150
    modal run scripts/verify_close_gripper_grasp.py --use-best --n-episodes 5 --lift-steps 1 --save-video

By default prints to stdout only, no video/volume writes. Pass --save-video
to additionally record one mp4 per episode (close_gripper policy rollout,
seamlessly continuing into the scripted lift steps on the same clip so the
handoff is visible), saved under
/model-cache/close_gripper_verify_videos/seed{N}.mp4 on the volume.

Download with:
    modal volume get --force rl-harness-model-cache \\
        close_gripper_verify_videos/ ./artifacts/close_gripper_verify_videos/
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(
    image=rl_image,
    gpu=None,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=600,
)
def verify_close_gripper_grasp(
    n_episodes: int = 10,
    seed_offset: int = 0,
    max_steps: int = 30,
    lift_steps: int = 8,
    lift_dz: float = 1.0,           # matches env/oracle.py's own lift branch (full-magnitude +z)
    verify_tolerance: float = 0.05, # matches close_gripper_drift_limit's current value
    ckpt_iter: int = 0,
    algo: str = "ppo",
    use_best: bool = False,
    save_video: bool = False,
    video_fps: int = 10,
    randomize_block_size: bool = False,  # opt-in — verify a size-randomized-trained
                                          # checkpoint against varying block sizes
                                          # instead of only the original fixed cube.
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
    from think_then_act.perception.collision_predictor import CollisionPredictor
    from think_then_act.policy.subgoal_policy import SubgoalGaussianPolicy
    from think_then_act.training.checkpoints import resolve_subgoal_checkpoint
    from think_then_act.training.subgoal_env import SubgoalConditionedEnv
    from think_then_act.training.subgoal_features import CLOSE_GRIPPER_OBS_DIM

    subgoal = "close_gripper"
    suffix = "_ppo" if algo == "ppo" else ""
    ckpt_dir = os.path.join(MODEL_CACHE_DIR, "checkpoints")

    if ckpt_iter > 0 and not use_best:
        ckpt_path = os.path.join(ckpt_dir, f"low_level_{subgoal}{suffix}_iter{ckpt_iter}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    else:
        ckpt_path = resolve_subgoal_checkpoint(ckpt_dir, subgoal, algo=algo, use_best=use_best)
    print("\n" + "=" * 60)
    print(f"  GRASP VERIFICATION (lift-test) — close_gripper")
    print(f"  checkpoint -> {ckpt_path}")
    print(f"  n_episodes={n_episodes}  lift_steps={lift_steps}  lift_dz={lift_dz}  "
          f"verify_tolerance={verify_tolerance}")
    print("=" * 60)

    collision_model = None
    collision_ckpt = os.path.join(ckpt_dir, "collision_predictor.pt")
    if os.path.exists(collision_ckpt):
        collision_model = CollisionPredictor()
        collision_model.load_state_dict(torch.load(collision_ckpt, map_location="cpu"))
        collision_model.eval()

    policy = SubgoalGaussianPolicy(obs_dim=CLOSE_GRIPPER_OBS_DIM)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # PPO checkpoints (low_level_ppo.py's save_checkpoint) are
    # {"actor": ..., "critic": ...}; GRPO checkpoints are a flat state_dict.
    policy.load_state_dict(ckpt["actor"] if isinstance(ckpt, dict) and "actor" in ckpt else ckpt)
    policy.eval()

    def make_env():
        base = ObservationHarness(
            gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                      max_episode_steps=max_steps + 250)
        )
        setup_env(base)
        wrapped = SubgoalConditionedEnv(
            base, subgoal=subgoal, collision_model=collision_model, max_episode_steps=max_steps,
            randomize_block_size=randomize_block_size,
        )
        return base, wrapped

    video_dir = os.path.join(MODEL_CACHE_DIR, "close_gripper_verify_videos")
    if save_video:
        os.makedirs(video_dir, exist_ok=True)

    # Fixed scripted probe — NOT the trained policy. Matches env/oracle.py's
    # own lift branch (direction=[0,0,1] scaled to full magnitude, grip=-1)
    # so this asks the same physical question the full oracle relies on:
    # does closing + lifting actually keep the block, using the SAME lift
    # mechanics the oracle itself depends on, not an arbitrary new probe.
    lift_action = np.array([0.0, 0.0, lift_dz, -1.0], dtype=np.float32)

    results = []
    for i in range(n_episodes):
        seed = seed_offset + i
        base, env = make_env()
        rng = np.random.default_rng(seed)
        obs, info = env.reset(rng=rng)
        frames = [base.last_frame()] if save_video else None

        proxy_done, proxy_step = False, None
        for step in range(max_steps):
            action = policy.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if save_video:
                frames.append(base.last_frame())
            if info.get("done", False):
                proxy_done, proxy_step = True, step
                break
            if terminated or truncated:
                break

        result = {"seed": seed, "proxy_done": proxy_done, "proxy_step": proxy_step}

        if not proxy_done:
            result["verified"] = None
            print(f"  seed={seed:>3}  proxy_done=False (no claimed success to verify)")
            if save_video:
                video_path = os.path.join(video_dir, f"seed{seed}_no_success.mp4")
                save_video_fn(frames, video_path, fps=video_fps)
                result["video_path"] = video_path
                print(f"    video -> {video_path}")
            env.close()
            results.append(result)
            continue

        d_grip_block_at_claim = float(info["d_grip_block"])
        block_z_at_claim = float(info["block_pos"][2])
        grip_z_at_claim  = float(info["grip_pos"][2])

        # NOT the trained policy from here on — a fixed scripted lift,
        # bypassing whatever the policy would have done next. This is the
        # actual verification: physically try to lift and see if the block
        # comes along, rather than trusting the proxy that just fired. Frame
        # capture continues seamlessly across this handoff (same `frames`
        # list) so a saved video shows the close_gripper policy AND the
        # scripted lift as one continuous clip, not two separate videos.
        for _ in range(lift_steps):
            obs, reward, terminated, truncated, info = env.step(lift_action)
            if save_video:
                frames.append(base.last_frame())

        d_grip_block_after_lift = float(info["d_grip_block"])
        block_z_after_lift = float(info["block_pos"][2])
        grip_z_after_lift  = float(info["grip_pos"][2])
        grip_rise  = grip_z_after_lift - grip_z_at_claim
        block_rise = block_z_after_lift - block_z_at_claim
        verified = bool(d_grip_block_after_lift <= verify_tolerance)

        result.update({
            "d_grip_block_at_claim"  : round(d_grip_block_at_claim, 5),
            "d_grip_block_after_lift": round(d_grip_block_after_lift, 5),
            "grip_rise" : round(grip_rise, 5),
            "block_rise": round(block_rise, 5),
            "verified"  : verified,
        })
        if save_video:
            tag = "verified" if verified else "FALSE_POSITIVE"
            video_path = os.path.join(video_dir, f"seed{seed}_{tag}.mp4")
            save_video_fn(frames, video_path, fps=video_fps)
            result["video_path"] = video_path
        print(f"  seed={seed:>3}  proxy_step={proxy_step:>2}  "
              f"d_grip_block(claim={d_grip_block_at_claim:.4f}, after_lift={d_grip_block_after_lift:.4f})  "
              f"grip_rise={grip_rise:.4f}  block_rise={block_rise:.4f}  "
              f"verified={'YES' if verified else 'NO -- FALSE POSITIVE'}"
              + (f"  video={result['video_path']}" if save_video else ""))
        env.close()
        results.append(result)

    if save_video:
        model_volume.commit()

    claimed  = [r for r in results if r["proxy_done"]]
    verified = [r for r in claimed if r["verified"]]
    print("\n" + "=" * 60)
    print(f"  proxy claimed success : {len(claimed)}/{n_episodes}")
    print(f"  verified by lift-test : {len(verified)}/{len(claimed) if claimed else 0}")
    if claimed and len(verified) < len(claimed):
        print(f"  FALSE POSITIVES: {len(claimed) - len(verified)} — proxy said done, "
              f"block did not stay with the gripper through the lift")
    print("=" * 60)

    return {
        "status": "PASS", "ckpt": ckpt_path,
        "n_episodes": n_episodes,
        "n_proxy_claimed": len(claimed),
        "n_verified": len(verified),
        "results": results,
    }


@app.local_entrypoint()
def main(
    n_episodes: int = 10, seed_offset: int = 0, max_steps: int = 30,
    lift_steps: int = 8, lift_dz: float = 1.0, verify_tolerance: float = 0.05,
    ckpt_iter: int = 0, algo: str = "ppo", use_best: bool = False,
    save_video: bool = False, video_fps: int = 10,
    randomize_block_size: bool = False,
):
    print(f"\nVerifying close_gripper grasp stability via scripted lift-test "
          f"(algo={algo} use_best={use_best} save_video={save_video} "
          f"randomize_block_size={randomize_block_size})...")
    result = verify_close_gripper_grasp.remote(
        n_episodes=n_episodes, seed_offset=seed_offset, max_steps=max_steps,
        lift_steps=lift_steps, lift_dz=lift_dz, verify_tolerance=verify_tolerance,
        ckpt_iter=ckpt_iter, algo=algo, use_best=use_best,
        save_video=save_video, video_fps=video_fps, randomize_block_size=randomize_block_size,
    )
    print(f"\nDone. {result['n_verified']}/{result['n_proxy_claimed']} claimed successes verified "
          f"(out of {result['n_episodes']} episodes total).")
    if save_video:
        print(f"\nDownload with:")
        print(f"  python3 -m modal volume get --force rl-harness-model-cache "
              f"close_gripper_verify_videos/ ./artifacts/close_gripper_verify_videos/")
