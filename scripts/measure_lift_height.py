"""
measure_lift_height.py

Physical measurement: how high does the scripted oracle (env.oracle.oracle_action)
actually lift the block during its CARRY phase, across N episodes? Same
motivation as measure_close_gripper_geometry.py (2026-07-15): lift_height=0.10m
(reward/subgoal_reward.py's SubgoalWeights) was never empirically verified
against real physics, unlike close_gripper_threshold (which turned out to be
unreachable and needed recalibrating). Before trusting an RL training plateau
as evidence of a reward/threshold problem, confirm directly whether a
KNOWN-WORKING scripted lift can even reach the threshold at all.

Run with:
    modal run scripts/measure_lift_height.py --n-episodes 10
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(
    image=rl_image,
    gpu=None,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=300,
)
def measure_lift_height(n_episodes: int = 10, max_steps: int = 60) -> dict:
    import os
    import numpy as np

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env, init_random_episode
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.env.oracle import oracle_action
    from think_then_act.reward.subgoal_reward import DEFAULT_WEIGHTS

    print("\n" + "=" * 60)
    print(f"  LIFT HEIGHT MEASUREMENT (scripted oracle, {n_episodes} episodes)")
    print(f"  lift_height threshold = {DEFAULT_WEIGHTS.lift_height}m")
    print("=" * 60)

    env = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array", max_episode_steps=max_steps * 2)
    )
    setup_env(env)

    max_heights = []
    for ep in range(n_episodes):
        env.reset(seed=ep)
        rng = np.random.default_rng(ep)
        obs, ok = init_random_episode(env, rng)
        if not ok:
            print(f"  ep {ep}: init_random_episode failed, skipping")
            continue

        carrying = False
        max_height = -np.inf
        for step in range(max_steps):
            obs_arr, achieved, desired = obs["observation"], obs["achieved_goal"], obs["desired_goal"]
            action, phase, carrying = oracle_action(obs_arr, achieved, desired, carrying)
            height_above_table = float(achieved[2] - DEFAULT_WEIGHTS.table_z)
            max_height = max(max_height, height_above_table)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        max_heights.append(max_height)
        print(f"  ep {ep}: max height_above_table={max_height:.4f}m")

    env.close()

    arr = np.array(max_heights)
    frac_reaching = float((arr >= DEFAULT_WEIGHTS.lift_height).mean()) if len(arr) else float("nan")

    print(f"\nAcross {len(arr)} episodes:")
    print(f"  mean max height             : {arr.mean():.4f}m")
    print(f"  min max height              : {arr.min():.4f}m")
    print(f"  max max height              : {arr.max():.4f}m")
    print(f"  lift_height threshold       : {DEFAULT_WEIGHTS.lift_height}m")
    print(f"  fraction reaching threshold : {frac_reaching:.1%}")
    print("=" * 60)

    return {
        "n_episodes": len(arr),
        "mean_max_height": round(float(arr.mean()), 4) if len(arr) else None,
        "min_max_height": round(float(arr.min()), 4) if len(arr) else None,
        "max_max_height": round(float(arr.max()), 4) if len(arr) else None,
        "lift_height": DEFAULT_WEIGHTS.lift_height,
        "fraction_reaching_threshold": frac_reaching,
    }


@app.local_entrypoint()
def main(n_episodes: int = 10, max_steps: int = 60):
    result = measure_lift_height.remote(n_episodes=n_episodes, max_steps=max_steps)
    print(f"\nDone. {result}")
