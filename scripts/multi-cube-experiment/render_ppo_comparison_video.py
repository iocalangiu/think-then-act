"""
render_ppo_comparison_video.py

Renders the SAME fixed eval seed (100000, a genuine-success episode for
both checkpoints) against both the BC checkpoint and the PPO-fine-tuned
checkpoint, for a direct before/after video comparison — built 2026-10-02
after scripts/diagnose_drag_mechanism.py's trace showed the "joints churn
while the gripper stays frozen" pattern is still present in the PPO
checkpoint's genuine-success rollouts, essentially unchanged from BC
(confirmed via grip_z flatlining at 0.4156 for t=70-98 while dz stays
around -0.85 to -0.92 the whole time).

Run with:
    modal run --detach scripts/render_ppo_comparison_video.py
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(image=rl_image, gpu=None, cpu=2.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=300)
def render_one(
    ckpt_path: str,
    policy_type: str,
    out_path: str,
    seed: int = 100000,
    max_steps: int = 100,
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

    cfg = MultiHeadBCConfig(obs_dim=FLAT_OBS_DIM, policy_type=policy_type)
    trainer = MultiHeadBCTrainer(cfg)
    trainer.load_checkpoint(os.path.join(MODEL_CACHE_DIR, ckpt_path))

    env = gym.make("FetchPickAndPlace-v3", max_episode_steps=max_steps, render_mode="rgb_array")
    setup_env(env)

    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    obs, setup_ok = init_random_episode(env, rng)
    frames = [env.render()]

    hidden_state = None
    success = False
    ever_lifted_and_gripped = False
    for _ in range(max_steps):
        flat_obs = build_flat_observation(obs["observation"], obs["achieved_goal"], obs["desired_goal"])
        action, hidden_state = trainer.actor.act(flat_obs, hidden_state, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())

        height_above_resting = float(obs["achieved_goal"][2]) - 0.425
        forces = grip_contact_forces(env)
        if min(forces["left"], forces["right"]) > 0.0 and height_above_resting > 0.02:
            ever_lifted_and_gripped = True
        if info.get("is_success", False):
            success = True
        if terminated or truncated:
            break

    env.close()
    genuine = success and ever_lifted_and_gripped

    full_out_path = os.path.join(MODEL_CACHE_DIR, out_path)
    os.makedirs(os.path.dirname(full_out_path), exist_ok=True)
    save_video(frames, full_out_path, fps=20)
    model_volume.commit()
    return {"ckpt_path": ckpt_path, "out_path": out_path, "genuine": genuine}


@app.local_entrypoint()
def main():
    results = list(render_one.starmap([
        ("checkpoints/bc_scaling/mse_n2000_seed0_smw0.0_k5.pt", "mse",
         "demonstrations/videos/ppo_compare_bc_seed100000.mp4", 100000, 100),
        ("checkpoints/flat_task_ppo_best.pt", "mse",
         "demonstrations/videos/ppo_compare_ppo_seed100000.mp4", 100000, 100),
    ]))
    for r in results:
        print(r)
