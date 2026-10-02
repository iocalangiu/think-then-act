"""
render_bc_scaling_success_videos.py

Renders one table-height genuine-success video per architecture, loading the
ALREADY-TRAINED bc_scaling checkpoints (n_demos=2000, seed=0 — the same
checkpoints the scaling chart's numbers and the elevated-target eval both
use) instead of retraining a separate one-off checkpoint the way
render_architecture_success_videos.py originally did. Written 2026-10-02
after the demo pool was redone to mix in elevated targets: keeps every video
in the "BC Architecture Scaling" artifact traceable to the SAME checkpoint
lineage, and lets MDN get a real table-height success video now that it
reaches 22% genuine completion at n=2000 on the new data (it never achieved
one on the old table-height-only data, so it was excluded before).

Run with:
    modal run --detach scripts/render_bc_scaling_success_videos.py
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR

ARCHITECTURES = ["mse", "mdn", "autoregressive", "factored", "cvae"]


@app.function(image=rl_image, gpu=None, cpu=2.0,
              volumes={MODEL_CACHE_DIR: model_volume}, timeout=600)
def render_success_video(
    policy_type: str,
    ckpt_path: str = "",
    max_eval_seeds: int = 40,
    max_steps: int = 100,
    lift_threshold: float = 0.02,
    block_resting_z: float = 0.425,
) -> dict:
    import os
    import numpy as np

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env, init_random_episode, save_video, grip_contact_forces
    from think_then_act.training.subgoal_features import build_flat_observation, FLAT_OBS_DIM
    from think_then_act.training.flat_bc_multi_head import MultiHeadBCConfig, MultiHeadBCTrainer

    if not ckpt_path:
        ckpt_path = f"checkpoints/bc_scaling/{policy_type}_n2000_seed0_smw0.0_k5.pt"

    cfg = MultiHeadBCConfig(obs_dim=FLAT_OBS_DIM, policy_type=policy_type)
    trainer = MultiHeadBCTrainer(cfg)
    trainer.load_checkpoint(os.path.join(MODEL_CACHE_DIR, ckpt_path))
    print(f"[{policy_type}] loaded {ckpt_path}", flush=True)

    env = gym.make("FetchPickAndPlace-v3", max_episode_steps=max_steps, render_mode="rgb_array")
    setup_env(env)

    found_genuine = False
    out_path = f"demonstrations/videos/success_{policy_type}_mixed.mp4"

    for eval_ep in range(max_eval_seeds):
        seed = 100_000 + eval_ep   # same held-out table-height eval seed convention as run_bc_scaling_cell
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
def main(max_eval_seeds: int = 40):
    results = list(render_success_video.starmap(
        [(pt, "", max_eval_seeds) for pt in ARCHITECTURES]
    ))
    print("\nSummary:")
    for r in results:
        print(f"  {r['policy_type']:16s}  found_genuine={r['found_genuine']}  out_path={r['out_path']}")
