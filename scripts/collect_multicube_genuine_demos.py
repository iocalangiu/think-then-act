"""
collect_multicube_genuine_demos.py

Collects GENUINE (fully autonomous, zero scripting) multi-cube stacking
demo traces for a large-scale BC retrain -- the alternative path to
continued PPO after the lr=3e-4 continuation's real, visually-confirmed
regression (see memory: flat_policy_ppo_generalization). Unlike
collect_stack3_demos.py (which scripted the release+reposition
transition because the policy at the time never did it on its own), this
reuses training/multicube_stack_env.py's MultiCubeStackEnv directly:
the environment itself switches goals on genuine placement, and
checkpoints/multicube_stack_ppo_v1_cont_best.pt (86.7% base-skill /
30.0% 2-cube / 13.3% 3-cube, the pre-degradation PPO checkpoint)
SOMETIMES completes full multi-cube stacks entirely on its own now --
so every demo collected here is the policy's own genuine behavior being
distilled into more training data, not a hand-authored stand-in for a
behavior it never demonstrated. That's the main suspect for why the
earlier 10-demo scripted fine-tune regressed the base skill (long
constant-action scripted segments corrupting the GRU's learned
dynamics) -- this collection has no scripted segments to cause that.

min_cubes=2 (not 1): the existing demonstrations/sb3_teacher_full_task_
joint_randomized_v2.pkl pool already has 2000 genuine single-cube demos
-- collection effort here is spent specifically on the multi-cube
transition cases that pool has none of, per the user's own framing
("alternate basic skill with 2 and 3 cubes" -- the "basic skill" side of
that alternation is the EXISTING pool, this collects the other side).

Run with:
    modal run --detach scripts/collect_multicube_genuine_demos.py --target-successes 300
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(image=rl_image, gpu=None, cpu=2.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=3600 * 3)
def collect_multicube_genuine_demos(
    ckpt_path: str = "checkpoints/multicube_stack_ppo_v1_cont_best.pt",
    target_successes: int = 300,
    max_attempts: int = 3000,
    min_cubes: int = 2,
    max_cubes: int = 3,
    max_episode_steps: int = 320,
    seed_start: int = 0,
    out_path: str = "demonstrations/multicube_genuine_demos.pkl",
    n_videos: int = 3,
    video_dir: str = "demonstrations/videos_multicube_genuine_demos",
    report_every: int = 50,
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

    demonstrations = []
    n_success = 0
    n_success_by_cubes = {}
    attempt = 0
    while n_success < target_successes and attempt < max_attempts:
        seed = seed_start + attempt
        attempt += 1
        rng = np.random.default_rng(seed)
        record_video = n_success < n_videos
        env = MultiCubeStackEnv(
            min_cubes=min_cubes, max_cubes=max_cubes, max_episode_steps=max_episode_steps,
            randomize_pose_prob=1.0, render_mode="rgb_array" if record_video else None,
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
            print(f"  attempt={attempt}  n_success={n_success}  n_cubes_this_attempt={n_cubes}  "
                  f"genuine={genuine_full_stack}  success_rate_so_far={n_success/attempt:.1%}", flush=True)

        if genuine_full_stack:
            n_success += 1
            n_success_by_cubes[n_cubes] = n_success_by_cubes.get(n_cubes, 0) + 1
            demonstrations.append({
                "obs": np.stack(obs_list).astype(np.float32),
                "teacher_action": np.stack(action_list).astype(np.float32),
                "success": True, "n_steps": len(obs_list), "n_cubes": n_cubes,
            })
            if record_video and frames is not None:
                video_path = os.path.join(MODEL_CACHE_DIR, video_dir, f"multicube_genuine_{n_success-1}_ncubes{n_cubes}.mp4")
                os.makedirs(os.path.dirname(video_path), exist_ok=True)
                save_video(frames, video_path, fps=20)
                model_volume.commit()
                print(f"    saved video -> {video_path}", flush=True)

        env.close()

    print(f"\n{n_success}/{target_successes} genuine demos collected in {attempt} attempts "
          f"({n_success/attempt:.1%} success rate)", flush=True)
    print(f"  by cube count: {n_success_by_cubes}", flush=True)

    full_out_path = os.path.join(MODEL_CACHE_DIR, out_path)
    os.makedirs(os.path.dirname(full_out_path), exist_ok=True)
    with open(full_out_path, "wb") as f:
        pickle.dump({
            "demonstrations": demonstrations, "ckpt_path": ckpt_path,
            "n_attempted": attempt, "n_successful": n_success,
            "success_rate": n_success / attempt if attempt else 0.0,
            "n_success_by_cubes": n_success_by_cubes,
            "min_cubes": min_cubes, "max_cubes": max_cubes,
        }, f)
    model_volume.commit()
    print(f"Saved {n_success} demonstrations -> {full_out_path}", flush=True)

    return {"n_successful": n_success, "n_attempted": attempt, "n_success_by_cubes": n_success_by_cubes}


@app.local_entrypoint()
def main(
    ckpt_path: str = "checkpoints/multicube_stack_ppo_v1_cont_best.pt",
    target_successes: int = 300,
    max_attempts: int = 3000,
    min_cubes: int = 2,
    max_cubes: int = 3,
):
    result = collect_multicube_genuine_demos.remote(
        ckpt_path=ckpt_path, target_successes=target_successes, max_attempts=max_attempts,
        min_cubes=min_cubes, max_cubes=max_cubes,
    )
    print(f"\nDone: {result['n_successful']}/{result['n_attempted']} attempts, by cube count: {result['n_success_by_cubes']}")
