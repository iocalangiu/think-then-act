"""
render_one_flat_bc_video.py

One-off: renders a single rollout of the current flat_task_bc.pt checkpoint
and saves it via env.setup.save_video (now with -movflags +faststart,
2026-10-02) — to test whether a properly-faststarted file fixes in-artifact
<video> playback, independent of delivery mechanism (files vs assets vs a
base64 data URI).

Run with:
    modal run scripts/multi-cube-experiment/render_one_flat_bc_video.py
    modal run scripts/multi-cube-experiment/render_one_flat_bc_video.py --seed 8   # a known-slide seed
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(image=rl_image, gpu=None, cpu=2.0,
              volumes={MODEL_CACHE_DIR: model_volume}, timeout=300)
def render_one_flat_bc_video(
    ckpt_path: str = "checkpoints/flat_task_bc.pt",
    seed: int = 0,
    max_steps: int = 100,
    out_path: str = "demonstrations/videos/flat_bc_faststart_test.mp4",
) -> dict:
    import os
    import numpy as np
    import torch

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env, init_random_episode, save_video
    from think_then_act.training.subgoal_features import build_flat_observation, FLAT_OBS_DIM
    from think_then_act.policy.subgoal_recurrent_policy import SubgoalRecurrentPolicy

    full_ckpt_path = os.path.join(MODEL_CACHE_DIR, ckpt_path)
    actor = SubgoalRecurrentPolicy(obs_dim=FLAT_OBS_DIM, action_dim=4)
    ckpt = torch.load(full_ckpt_path, map_location="cpu")
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    env = gym.make("FetchPickAndPlace-v3", max_episode_steps=max_steps, render_mode="rgb_array")
    setup_env(env)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    obs, setup_ok = init_random_episode(env, rng)

    frames = [env.render()]
    hidden_state = None
    success = False
    for _ in range(max_steps):
        flat_obs = build_flat_observation(obs["observation"], obs["achieved_goal"], obs["desired_goal"])
        action, hidden_state = actor.act(flat_obs, hidden_state, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        if info.get("is_success", False):
            success = True
        if terminated or truncated:
            break
    env.close()

    full_out_path = os.path.join(MODEL_CACHE_DIR, out_path)
    os.makedirs(os.path.dirname(full_out_path), exist_ok=True)
    save_video(frames, full_out_path, fps=20)
    model_volume.commit()
    print(f"Saved ({'SUCCESS' if success else 'fail'}) -> {full_out_path}")
    return {"success": success, "out_path": out_path}


@app.local_entrypoint()
def main(seed: int = 0):
    result = render_one_flat_bc_video.remote(seed=seed)
    print(result)
