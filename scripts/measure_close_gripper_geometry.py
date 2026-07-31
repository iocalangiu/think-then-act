"""
measure_close_gripper_geometry.py

One-off diagnostic: how wide is the manipulated block (body "object0"), and
what total_finger_width/closedness does the gripper actually settle at
once it's genuinely holding it — not just approaching? Needed to calibrate
reward/subgoal_reward.py's close_gripper_threshold (currently 0.9, i.e.
near-total finger closure): found 2026-07-14 that this may be physically
unreachable while actually gripping a real, rigid object, since the
block's own width stops the fingers from closing further — closing all
the way to 0.9 would require nothing between the fingers, the OPPOSITE of
a successful grasp.

Enumerates the block's geom size the same way validate_collision_labels.py
already does for the table (by BODY name — object0 is real/explicitly
declared, unlike some geoms — see that script's own comment for why body
name, not geom name), then runs the scripted oracle (env.oracle.oracle_action)
to actually grasp the block on several seeds and reports the steady-state
finger_width once solidly in CARRY phase (averaged over the last few
grasp-holding steps, to smooth transient contact settling).

No GPU needed.

Run with:
    modal run scripts/measure_close_gripper_geometry.py
    modal run scripts/measure_close_gripper_geometry.py --n-seeds 10
"""

import modal
from think_then_act.modal_app import app, rl_image


@app.function(image=rl_image, gpu=None, timeout=300)
def measure_close_gripper_geometry(n_seeds: int = 5, max_steps: int = 100) -> dict:
    import os
    import numpy as np

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import mujoco
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env, init_random_episode
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.env.oracle import oracle_action

    print("\n" + "=" * 60)
    print("  close_gripper GEOMETRY CHECK")
    print("=" * 60)

    env = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array", max_episode_steps=max_steps + 20)
    )
    setup_env(env)

    # ------------------------------------------------------------------
    # Block geometry
    # ------------------------------------------------------------------
    raw = env.unwrapped
    block_size, block_gtype = None, None
    for gid in range(raw.model.ngeom):
        body_id   = raw.model.geom_bodyid[gid]
        body_name = mujoco.mj_id2name(raw.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body{body_id}"
        if body_name == "object0":
            block_size  = raw.model.geom_size[gid].copy()
            block_gtype = int(raw.model.geom_type[gid])
            break
    if block_size is None:
        raise RuntimeError("No geom found with body name 'object0'")

    print(f"\n  object0 geom: type={block_gtype}  size (half-extents)={block_size}")
    print(f"    full width  x={2*block_size[0]:.4f}m  y={2*block_size[1]:.4f}m  z={2*block_size[2]:.4f}m")

    # ------------------------------------------------------------------
    # Grasp via the scripted oracle, report steady-state finger_width
    # once solidly in CARRY (grasped and being lifted).
    # ------------------------------------------------------------------
    settled_widths = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        obs, info = env.reset(seed=seed)
        obs, ok = init_random_episode(env, rng)
        if not ok:
            print(f"  seed={seed}: init_random_episode failed, skipping")
            continue

        carrying = False
        carry_widths = []
        for _ in range(max_steps):
            obs_arr, achieved, desired = obs["observation"], obs["achieved_goal"], obs["desired_goal"]
            action, phase, carrying = oracle_action(obs_arr, achieved, desired, carrying)
            finger_width = float(np.sum(obs_arr[9:11]))
            if carrying:
                carry_widths.append(finger_width)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        if len(carry_widths) >= 5:
            steady = float(np.mean(carry_widths[-5:]))
            closedness = 1.0 - min(max(steady / 0.10, 0.0), 1.0)
            settled_widths.append(steady)
            print(f"  seed={seed}: reached CARRY after {len(carry_widths)} carry-steps, "
                  f"steady-state finger_width={steady:.4f}  closedness={closedness:.4f}")
        else:
            print(f"  seed={seed}: never solidly reached CARRY phase (only {len(carry_widths)} carry-steps)")

    env.close()

    result = {
        "status": "PASS",
        "block_geom_size": block_size.tolist(),
        "block_geom_type": block_gtype,
        "settled_widths": settled_widths,
    }

    if settled_widths:
        avg_width = float(np.mean(settled_widths))
        avg_closedness = 1.0 - min(max(avg_width / 0.10, 0.0), 1.0)
        print("\n" + "=" * 60)
        print(f"  ACROSS {len(settled_widths)}/{n_seeds} SUCCESSFUL GRASPS:")
        print(f"    avg steady-state finger_width = {avg_width:.4f}")
        print(f"    avg steady-state closedness   = {avg_closedness:.4f}")
        print(f"    current close_gripper_threshold = 0.9")
        print("=" * 60)
        result["avg_steady_state_finger_width"] = avg_width
        result["avg_steady_state_closedness"]   = avg_closedness
    else:
        print("\n  No successful grasps across any seed — can't estimate steady-state closedness.")
        result["avg_steady_state_finger_width"] = None
        result["avg_steady_state_closedness"]   = None

    return result


@app.local_entrypoint()
def main(n_seeds: int = 5, max_steps: int = 100):
    result = measure_close_gripper_geometry.remote(n_seeds=n_seeds, max_steps=max_steps)
    print(f"\nDone: {result}")
