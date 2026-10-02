"""
render_architecture_success_videos.py

For each of the 4 architectures that actually scale (mse, autoregressive,
factored, cvae — MDN gets its own dedicated trace/video via
render_mdn_rollout_video.py, since it never achieves a genuine success),
trains one fresh checkpoint on n_demos demonstrations, then rolls across
eval seeds until it finds a GENUINE success (two-finger contact + block
lifted >2cm, not just raw is_success — same verification used throughout
this project) and renders that one episode as video.

One Modal function per architecture, run in parallel via .starmap() — each
trains independently so there's no reason to serialize them.

Run with:
    modal run --detach scripts/render_architecture_success_videos.py
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR

ARCHITECTURES = ["mse", "autoregressive", "factored", "cvae"]


@app.function(image=rl_image, gpu=None, cpu=2.0,
              volumes={MODEL_CACHE_DIR: model_volume}, timeout=900)
def render_architecture_success_video(
    policy_type: str,
    n_demos: int = 2000,
    train_seed: int = 0,
    n_epochs: int = 30,
    max_eval_seeds: int = 40,
    max_steps: int = 100,
    lift_threshold: float = 0.02,
    block_resting_z: float = 0.425,
    demos_path: str = "demonstrations/sb3_teacher_full_task.pkl",
) -> dict:
    import os
    import pickle
    import numpy as np
    import torch

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env, init_random_episode, save_video, grip_contact_forces
    from think_then_act.training.subgoal_features import build_flat_observation, FLAT_OBS_DIM
    from think_then_act.training.flat_bc_multi_head import MultiHeadBCConfig, MultiHeadBCTrainer

    torch.manual_seed(train_seed)
    rng_subsample = np.random.default_rng(train_seed)

    full_demos_path = os.path.join(MODEL_CACHE_DIR, demos_path)
    with open(full_demos_path, "rb") as f:
        demo_data = pickle.load(f)
    all_demos = demo_data["demonstrations"]
    subsample_idx = rng_subsample.choice(len(all_demos), size=n_demos, replace=False)
    demos = [all_demos[i] for i in subsample_idx]

    cfg = MultiHeadBCConfig(obs_dim=FLAT_OBS_DIM, policy_type=policy_type, n_epochs=n_epochs)
    trainer = MultiHeadBCTrainer(cfg)
    trainer.fit(demos)
    ckpt_path = f"checkpoints/flat_bc_{policy_type}_n{n_demos}_seed{train_seed}.pt"
    full_ckpt_path = os.path.join(MODEL_CACHE_DIR, ckpt_path)
    trainer.save_checkpoint(full_ckpt_path)
    model_volume.commit()
    print(f"[{policy_type}] trained on {n_demos} demos, checkpoint -> {full_ckpt_path}", flush=True)

    env = gym.make("FetchPickAndPlace-v3", max_episode_steps=max_steps, render_mode="rgb_array")
    setup_env(env)

    found_genuine = False
    out_path = f"demonstrations/videos/success_{policy_type}.mp4"

    for eval_ep in range(max_eval_seeds):
        seed = 100_000 + eval_ep   # same fixed held-out eval seed convention as run_bc_scaling_cell
        rng = np.random.default_rng(seed)
        env.reset(seed=seed)
        obs, setup_ok = init_random_episode(env, rng)
        if not setup_ok:
            continue

        frames = [env.render()]
        hidden_state = None
        success = False
        ever_lifted_and_gripped = False
        for _ in range(max_steps):
            flat_obs = build_flat_observation(obs["observation"], obs["achieved_goal"], obs["desired_goal"])
            action, hidden_state = trainer.actor.act(flat_obs, hidden_state, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            frames.append(env.render())

            height_above_resting = float(obs["achieved_goal"][2]) - block_resting_z
            forces = grip_contact_forces(env)
            if min(forces["left"], forces["right"]) > 0.0 and height_above_resting > lift_threshold:
                ever_lifted_and_gripped = True
            if info.get("is_success", False):
                success = True
            if terminated or truncated:
                break

        genuine = success and ever_lifted_and_gripped
        print(f"[{policy_type}] eval_seed={seed}  success={success}  genuine={genuine}", flush=True)

        if genuine:
            full_out_path = os.path.join(MODEL_CACHE_DIR, out_path)
            os.makedirs(os.path.dirname(full_out_path), exist_ok=True)
            save_video(frames, full_out_path, fps=20)
            model_volume.commit()
            found_genuine = True
            print(f"[{policy_type}] GENUINE SUCCESS at eval_seed={seed} -> {full_out_path}", flush=True)
            break

    env.close()
    return {"policy_type": policy_type, "found_genuine": found_genuine,
            "out_path": out_path if found_genuine else None, "ckpt_path": ckpt_path}


@app.local_entrypoint()
def main(n_demos: int = 2000, max_eval_seeds: int = 40):
    results = list(render_architecture_success_video.starmap(
        [(pt, n_demos, 0, 30, max_eval_seeds) for pt in ARCHITECTURES]
    ))
    print("\nSummary:")
    for r in results:
        print(f"  {r['policy_type']:16s}  found_genuine={r['found_genuine']}  "
              f"out_path={r['out_path']}  ckpt_path={r['ckpt_path']}")
