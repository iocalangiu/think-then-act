"""
debug_random_exploration_grip_force.py

Does purely RANDOM (untrained, freshly-initialized) exploration ever
stumble into real grip force within close_gripper's step budget — for the
fixed 5cm cube vs. size-randomized blocks? Training telemetry only logs
final-step closedness, which can't distinguish "never got close" from "got
close then drifted away before the episode ended" — this runs a FRESH,
randomly-initialized SubgoalGaussianPolicy with STOCHASTIC sampling (what
actual PPO data collection uses, not the deterministic .act() used in
earlier debug scripts) for the full 30-step budget regardless of drift
(overriding close_gripper_drift_limit so early truncation doesn't cut
episodes short before finding out whether they'd have recovered), and
tracks the MAX closedness/grip_strength reached at ANY point in each
episode — not just wherever the episode happened to end.

Motivation: a live training run stalled at completion_rate=0% and
closedness pinned at 0.0000 for the full 300 iterations, and the
deterministic policy at iter50/iter200 showed a state-independent,
saturated action. Before trying an exploration/init fix, confirm the
actual root cause: does size randomization remove the "lucky accidental
success early in training" signal that a FIXED cube's identical-every-
episode geometry made possible, or is early success comparably rare in
both cases?

Run with:
    modal run scripts/debug_random_exploration_grip_force.py
"""

import modal
from think_then_act.modal_app import app, rl_image


@app.function(image=rl_image, gpu=None, timeout=300)
def debug_random_exploration_grip_force(n_episodes: int = 30, max_steps: int = 30) -> dict:
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

    print("\n" + "=" * 60)
    print("  RANDOM-INIT STOCHASTIC EXPLORATION vs. GRIP FORCE")
    print("=" * 60)

    results = {}
    for label, randomize in [("fixed_cube", False), ("randomized", True)]:
        torch.manual_seed(0)
        policy = SubgoalGaussianPolicy(obs_dim=CLOSE_GRIPPER_OBS_DIM)
        policy.eval()

        max_closedness_per_episode = []
        any_real_contact_per_episode = []  # grip_strength > 1N at any step

        for seed in range(n_episodes):
            base = ObservationHarness(
                gym.make("FetchPickAndPlace-v3", render_mode="rgb_array", max_episode_steps=300)
            )
            setup_env(base)
            env = SubgoalConditionedEnv(
                base, subgoal="close_gripper", max_episode_steps=max_steps,
                randomize_block_size=randomize,
            )
            rng = np.random.default_rng(seed)
            obs, info = env.reset(rng=rng)

            max_closedness = float(info.get("closedness", 0.0) or 0.0)
            max_grip_strength = float(info.get("grip_strength", 0.0) or 0.0)

            with torch.no_grad():
                for step in range(max_steps):
                    obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                    action_t, _raw, _logp, _ = policy.sample(obs_t)
                    action = action_t.squeeze(0).numpy()
                    obs, reward, terminated, truncated, info = env.step(action)
                    max_closedness = max(max_closedness, float(info.get("closedness", 0.0) or 0.0))
                    max_grip_strength = max(max_grip_strength, float(info.get("grip_strength", 0.0) or 0.0))
                    if terminated:
                        break
                    # Deliberately IGNORE truncated (drifted_too_far/timed_out)
                    # for this diagnostic — keep stepping to the full budget
                    # to see whether it ever would have found contact, rather
                    # than stopping the moment real training would have cut
                    # it off.

            max_closedness_per_episode.append(max_closedness)
            any_real_contact_per_episode.append(max_grip_strength > 1.0)
            env.close()

        n_any_contact = sum(any_real_contact_per_episode)
        avg_max_closedness = float(np.mean(max_closedness_per_episode))
        print(f"\n  {label}: episodes_with_any_real_contact = {n_any_contact}/{n_episodes}  "
              f"avg_max_closedness = {avg_max_closedness:.4f}")
        print(f"    max_closedness per episode: {[round(x, 3) for x in max_closedness_per_episode]}")
        results[label] = {
            "n_any_contact": n_any_contact, "n_episodes": n_episodes,
            "avg_max_closedness": avg_max_closedness,
            "max_closedness_per_episode": max_closedness_per_episode,
        }

    print("\n" + "=" * 60)
    print(f"  fixed_cube : {results['fixed_cube']['n_any_contact']}/{n_episodes} episodes found real contact")
    print(f"  randomized : {results['randomized']['n_any_contact']}/{n_episodes} episodes found real contact")
    print("=" * 60)

    return {"status": "PASS", "results": results}


@app.local_entrypoint()
def main(n_episodes: int = 30, max_steps: int = 30):
    result = debug_random_exploration_grip_force.remote(n_episodes=n_episodes, max_steps=max_steps)
    print(f"\nDone: {result}")
