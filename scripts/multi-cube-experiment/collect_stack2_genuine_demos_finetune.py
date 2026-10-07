"""
collect_stack2_genuine_demos_finetune.py

Experience-replay harvest: checkpoints/bc_multicube_finetune_v1.pt (the
fine-tune-from-single-cube-PPO checkpoint, 46.7% fully-autonomous 2-cube
success -- see memory: flat_policy_ppo_generalization) is now good enough
to harvest MUCH more efficiently than either earlier source policy
(collect_stack2_demos_for_bc.py's original raw-PPO source ran at
19.3-25%). Reuses collect_multicube_genuine_demos.py's exact pattern
(MultiCubeStackEnv drives goal-switching itself, zero scripting, only
episodes where info["done"] fires genuinely are kept) but as its OWN
standalone script with its own checkpoint/demo-path/log paths, per this
project's standing rule (see memory: run_bc_scaling_cell's hardcoded-path
near-miss) that a new source checkpoint or demo-naming scheme always gets
its own script, never a param override on an existing one whose paths
are hardcoded elsewhere.

Two safety properties baked in from the start (both learned the hard way
earlier this session, see memory: feedback_overwrite_safety):
1. Incremental saving after every new success (survives the recurring
   Modal-preemption pattern without losing progress).
2. Resume-safe: loads any existing pool at out_path first and appends,
   never starts a fresh list that could silently overwrite one.

Run with:
    modal run --detach scripts/collect_stack2_genuine_demos_finetune.py --target-successes 200
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(image=rl_image, gpu=None, cpu=2.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=3600 * 3)
def collect_stack2_genuine_demos_finetune(
    ckpt_path: str = "checkpoints/bc_multicube_finetune_v1.pt",
    target_successes: int = 200,
    max_attempts: int = 2000,
    min_cubes: int = 2,
    max_cubes: int = 2,
    max_episode_steps: int = 320,
    seed_start: int = 0,
    out_path: str = "demonstrations/stack2_genuine_demos_finetune_v1.pkl",
    n_videos: int = 3,
    video_dir: str = "demonstrations/videos_stack2_genuine_finetune",
    report_every: int = 20,
) -> dict:
    import os
    import pickle
    import numpy as np

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import save_video
    from think_then_act.training.multicube_stack_env import MultiCubeStackEnv
    from think_then_act.training.flat_bc_multi_head import MultiHeadBCConfig, MultiHeadBCTrainer
    from think_then_act.training.subgoal_features import FLAT_OBS_DIM

    cfg = MultiHeadBCConfig(obs_dim=FLAT_OBS_DIM, policy_type="mse")
    trainer = MultiHeadBCTrainer(cfg)
    trainer.load_checkpoint(os.path.join(MODEL_CACHE_DIR, ckpt_path))
    print(f"loaded {ckpt_path}", flush=True)

    full_out_path = os.path.join(MODEL_CACHE_DIR, out_path)
    demonstrations = []
    if os.path.exists(full_out_path):
        with open(full_out_path, "rb") as f:
            existing = pickle.load(f)
        demonstrations = existing.get("demonstrations", [])
        print(f"resuming from existing pool at {out_path}: {len(demonstrations)} demos already present", flush=True)
    n_success = len(demonstrations)
    n_success_by_cubes = {}
    for d in demonstrations:
        nc = d.get("n_cubes")
        n_success_by_cubes[nc] = n_success_by_cubes.get(nc, 0) + 1

    def save_pool():
        os.makedirs(os.path.dirname(full_out_path), exist_ok=True)
        with open(full_out_path, "wb") as f:
            pickle.dump({
                "demonstrations": demonstrations, "ckpt_path": ckpt_path,
                "n_successful": len(demonstrations), "n_success_by_cubes": n_success_by_cubes,
                "min_cubes": min_cubes, "max_cubes": max_cubes,
            }, f)
        model_volume.commit()

    n_success_at_start = n_success
    attempt = 0
    while n_success < target_successes and attempt < max_attempts:
        seed = seed_start + attempt
        attempt += 1
        rng = np.random.default_rng(seed)
        record_video = (n_success - n_success_at_start) < n_videos
        env = MultiCubeStackEnv(
            min_cubes=min_cubes, max_cubes=max_cubes, max_episode_steps=max_episode_steps,
            randomize_pose_prob=1.0, pose_scheme="gripper_3d", render_mode="rgb_array" if record_video else None,
        )
        obs, reset_info = env.reset(rng=rng, seed=seed)
        n_cubes = reset_info["n_cubes"]

        frames = [env.render()] if record_video else None
        obs_list, action_list = [], []
        hidden_state = None
        info = {}
        for _ in range(max_episode_steps):
            action, hidden_state = trainer.actor.act(obs, hidden_state, deterministic=True)
            obs_list.append(np.asarray(obs, dtype=np.float32))
            action_list.append(np.asarray(action, dtype=np.float32))
            obs, reward, terminated, truncated, info = env.step(action)
            if frames is not None:
                frames.append(env.render())
            if terminated or truncated:
                break

        genuine_full_stack = info.get("done", False)
        if attempt % report_every == 0:
            rate_this_run = (n_success - n_success_at_start) / attempt
            print(f"  attempt={attempt}  n_success_total={n_success}  success_rate_this_run={rate_this_run:.1%}", flush=True)

        if genuine_full_stack:
            n_success += 1
            n_success_by_cubes[n_cubes] = n_success_by_cubes.get(n_cubes, 0) + 1
            demonstrations.append({
                "obs": np.stack(obs_list).astype(np.float32),
                "teacher_action": np.stack(action_list).astype(np.float32),
                "success": True, "n_steps": len(obs_list), "n_cubes": n_cubes,
            })
            save_pool()
            print(f"  NEW SUCCESS #{n_success} (attempt={attempt}, n_cubes={n_cubes}) -- saved", flush=True)
            if record_video and frames is not None:
                video_path = os.path.join(MODEL_CACHE_DIR, video_dir, f"stack2_genuine_finetune_{n_success-1}.mp4")
                os.makedirs(os.path.dirname(video_path), exist_ok=True)
                save_video(frames, video_path, fps=20)
                model_volume.commit()

        env.close()

    print(f"\n{n_success}/{target_successes} genuine demos in pool after {attempt} attempts this run", flush=True)
    print(f"  by cube count: {n_success_by_cubes}", flush=True)
    save_pool()
    print(f"Saved {n_success} total demonstrations -> {full_out_path}", flush=True)

    return {"n_successful": n_success, "n_attempted_this_run": attempt, "n_success_by_cubes": n_success_by_cubes}


@app.local_entrypoint()
def main(
    ckpt_path: str = "checkpoints/bc_multicube_finetune_v1.pt",
    target_successes: int = 200,
    max_attempts: int = 2000,
):
    result = collect_stack2_genuine_demos_finetune.remote(
        ckpt_path=ckpt_path, target_successes=target_successes, max_attempts=max_attempts,
    )
    print(f"\nDone: {result}")
