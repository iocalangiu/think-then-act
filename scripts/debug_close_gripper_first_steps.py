"""
debug_close_gripper_first_steps.py

One-off: load the close_gripper iter50 checkpoint (size-randomized run,
2026-08-11) and print the FULL observation vector and action taken at each
of the first few steps, across a few seeds. Training telemetry shows
d_grip_block(final) consistently ABOVE close_gripper_drift_limit (0.05)
and closedness EXACTLY 0.0000 every logged iteration — a verify-script
video confirmed episodes are ending after essentially ONE step (drift
truncation), meaning grip_strength/closedness necessarily reads zero at
that final step regardless of whether the force-scale fix itself works —
this is a DIFFERENT problem than the force-scale calibration. Printing raw
obs/action values directly to see whether something (e.g. the new
block_dims tail of the observation, on a very different numeric scale
than the rest of the 41-dim vector) is producing an anomalous first
action, rather than guessing further.

Run with:
    modal run scripts/debug_close_gripper_first_steps.py
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(image=rl_image, gpu=None, volumes={MODEL_CACHE_DIR: model_volume}, timeout=180)
def debug_close_gripper_first_steps(n_seeds: int = 3, n_steps: int = 3, ckpt_iter: int = 50) -> dict:
    import os
    import numpy as np
    import torch

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.policy.subgoal_policy import SubgoalGaussianPolicy
    from think_then_act.training.subgoal_env import SubgoalConditionedEnv
    from think_then_act.training.subgoal_features import CLOSE_GRIPPER_OBS_DIM

    ckpt_path = os.path.join(MODEL_CACHE_DIR, "checkpoints", f"low_level_close_gripper_ppo_iter{ckpt_iter}.pt")
    policy = SubgoalGaussianPolicy(obs_dim=CLOSE_GRIPPER_OBS_DIM)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    policy.load_state_dict(ckpt["actor"] if isinstance(ckpt, dict) and "actor" in ckpt else ckpt)
    policy.eval()

    print("\n" + "=" * 60)
    print(f"  FIRST-STEP DEBUG — {ckpt_path}")
    print("=" * 60)

    for seed in range(n_seeds):
        base = ObservationHarness(
            gym.make("FetchPickAndPlace-v3", render_mode="rgb_array", max_episode_steps=300)
        )
        setup_env(base)
        env = SubgoalConditionedEnv(
            base, subgoal="close_gripper", max_episode_steps=30, randomize_block_size=True,
        )
        rng = np.random.default_rng(seed)
        obs, info = env.reset(rng=rng)
        print(f"\n  seed={seed}")
        print(f"    obs shape={obs.shape}  block_pos={info.get('block_pos')}  "
              f"grip_pos={info.get('grip_pos')}  d_grip_block={info.get('d_grip_block')}")
        print(f"    obs[-3:] (perceived block_dims) = {obs[-3:]}")
        print(f"    obs any-nan={np.isnan(obs).any()}  obs min/max = {obs.min():.4f}/{obs.max():.4f}")

        for step in range(n_steps):
            action = policy.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"    step={step}  action={np.round(action, 4).tolist()}  "
                  f"reward={reward:.4f}  d_grip_block={info.get('d_grip_block')}  "
                  f"closedness={info.get('closedness')}  drifted_too_far={info.get('drifted_too_far')}  "
                  f"terminated={terminated}  truncated={truncated}")
            if terminated or truncated:
                break
        env.close()

    return {"status": "PASS"}


@app.local_entrypoint()
def main(n_seeds: int = 3, n_steps: int = 3, ckpt_iter: int = 50):
    result = debug_close_gripper_first_steps.remote(n_seeds=n_seeds, n_steps=n_steps, ckpt_iter=ckpt_iter)
    print(f"\nDone: {result}")
