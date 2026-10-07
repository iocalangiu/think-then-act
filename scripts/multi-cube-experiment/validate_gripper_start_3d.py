"""
validate_gripper_start_3d.py

Renders 10 frames of env.setup.randomize_gripper_start_3d's resulting
poses, same validation practice as randomize_joint_angles's own fix
(2026-10-03): render and actually look before trusting a randomization
scheme, since aggregate numeric ranges alone already fooled this project
once (see memory: flat_policy_ppo_generalization).

Run with:
    modal run scripts/multi-cube-experiment/validate_gripper_start_3d.py
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(image=rl_image, gpu=None, cpu=2.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=300)
def validate_gripper_start_3d(n_frames: int = 10, seed_start: int = 0) -> dict:
    import os
    import numpy as np

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env, init_random_episode, randomize_gripper_start_3d, save_video

    results = []
    frames = []
    for i in range(n_frames):
        seed = seed_start + i
        rng = np.random.default_rng(seed)
        env = gym.make("FetchPickAndPlace-v3", max_episode_steps=100, render_mode="rgb_array")
        setup_env(env)
        obs, _ = env.reset(seed=seed)
        obs, setup_ok = init_random_episode(env, rng)
        obs, pos_ok, info = randomize_gripper_start_3d(env, rng, obs)
        grip = obs["observation"][0:3]
        frame = env.render()
        frames.append(frame)
        results.append({"seed": seed, "ok": pos_ok, "grip_xyz": [round(float(x), 3) for x in grip],
                         "intended": [round(float(x), 3) for x in info["intended_start"]]})
        print(f"  seed={seed}  ok={pos_ok}  grip_xyz={results[-1]['grip_xyz']}  "
              f"intended={results[-1]['intended']}", flush=True)
        env.close()

    import numpy as np
    from PIL import Image
    grid_path = os.path.join(MODEL_CACHE_DIR, "demonstrations/videos/gripper_start_3d_grid.png")
    os.makedirs(os.path.dirname(grid_path), exist_ok=True)
    cols = 5
    rows = (n_frames + cols - 1) // cols
    h, w = frames[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        r, c = i // cols, i % cols
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = f[:, :, :3]
    Image.fromarray(grid).save(grid_path)
    model_volume.commit()
    print(f"saved frame grid -> {grid_path}", flush=True)

    return {"results": results}


@app.local_entrypoint()
def main(n_frames: int = 10, seed_start: int = 0):
    result = validate_gripper_start_3d.remote(n_frames=n_frames, seed_start=seed_start)
    print("\n", result)
