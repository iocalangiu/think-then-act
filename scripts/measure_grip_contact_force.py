"""
measure_grip_contact_force.py

One-off diagnostic: what contact FORCE (Newtons) does the gripper actually
settle at once it's genuinely holding the block — not just approaching?
Needed to calibrate reward/subgoal_reward.py's close_gripper_force_scale,
the new contact-force-based replacement for close_gripper_target_width (a
finger-WIDTH proxy that only worked because there was one fixed 5cm cube —
see env/setup.py's grip_contact_forces docstring for why force generalizes
where width doesn't).

Same pattern as measure_close_gripper_geometry.py/measure_lift_height.py:
run the scripted oracle (a known-working policy) to actually grasp the
block on several seeds, and report the steady-state min(left, right)
contact force once solidly in CARRY phase — the "both fingers genuinely
engaged" bottleneck grip_contact_forces is designed around, not just
whichever single finger happens to be pressing hardest.

No GPU needed.

Run with:
    modal run scripts/measure_grip_contact_force.py
    modal run scripts/measure_grip_contact_force.py --n-seeds 10
"""

import modal
from think_then_act.modal_app import app, rl_image


@app.function(image=rl_image, gpu=None, timeout=300)
def measure_grip_contact_force(n_seeds: int = 5, max_steps: int = 100) -> dict:
    import os
    import numpy as np

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env, init_random_episode, grip_contact_forces
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.env.oracle import oracle_action

    print("\n" + "=" * 60)
    print("  close_gripper CONTACT-FORCE CHECK")
    print("=" * 60)

    env = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array", max_episode_steps=max_steps + 20)
    )
    setup_env(env)

    settled_grip_strengths = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        obs, info = env.reset(seed=seed)
        obs, ok = init_random_episode(env, rng)
        if not ok:
            print(f"  seed={seed}: init_random_episode failed, skipping")
            continue

        carrying = False
        carry_forces = []   # list of (left, right) while solidly in CARRY
        for _ in range(max_steps):
            obs_arr, achieved, desired = obs["observation"], obs["achieved_goal"], obs["desired_goal"]
            action, phase, carrying = oracle_action(obs_arr, achieved, desired, carrying)
            if carrying:
                forces = grip_contact_forces(env)
                carry_forces.append((forces["left"], forces["right"]))
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        if len(carry_forces) >= 5:
            last5 = np.array(carry_forces[-5:])
            steady_left  = float(np.mean(last5[:, 0]))
            steady_right = float(np.mean(last5[:, 1]))
            steady_grip_strength = float(np.mean(np.minimum(last5[:, 0], last5[:, 1])))
            settled_grip_strengths.append(steady_grip_strength)
            print(f"  seed={seed}: reached CARRY after {len(carry_forces)} carry-steps, "
                  f"steady-state left={steady_left:.4f}N  right={steady_right:.4f}N  "
                  f"grip_strength(min)={steady_grip_strength:.4f}N")
        else:
            print(f"  seed={seed}: never solidly reached CARRY phase (only {len(carry_forces)} carry-steps)")

    env.close()

    result = {
        "status": "PASS",
        "settled_grip_strengths": settled_grip_strengths,
    }

    if settled_grip_strengths:
        avg = float(np.mean(settled_grip_strengths))
        lo, hi = float(np.min(settled_grip_strengths)), float(np.max(settled_grip_strengths))
        print("\n" + "=" * 60)
        print(f"  ACROSS {len(settled_grip_strengths)}/{n_seeds} SUCCESSFUL GRASPS:")
        print(f"    avg steady-state grip_strength (min of L/R) = {avg:.4f}N")
        print(f"    range = [{lo:.4f}N, {hi:.4f}N]")
        print("=" * 60)
        result["avg_grip_strength"] = avg
        result["min_grip_strength"] = lo
        result["max_grip_strength"] = hi
    else:
        print("\n  No successful grasps across any seed — can't estimate steady-state grip force.")
        result["avg_grip_strength"] = None
        result["min_grip_strength"] = None
        result["max_grip_strength"] = None

    return result


@app.local_entrypoint()
def main(n_seeds: int = 5, max_steps: int = 100):
    result = measure_grip_contact_force.remote(n_seeds=n_seeds, max_steps=max_steps)
    print(f"\nDone: {result}")
