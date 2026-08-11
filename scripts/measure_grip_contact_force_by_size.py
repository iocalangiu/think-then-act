"""
measure_grip_contact_force_by_size.py

Diagnostic follow-up to measure_grip_contact_force.py: does steady-state
grip contact force actually stay in the same ballpark across the
randomized size range, or does close_gripper_force_scale=130N (calibrated
ONLY on the original 5cm cube) silently break for other sizes?

Motivated by a live training run (close_gripper + randomize_block_size)
stalling at closedness=0.0000 and completion_rate=0% for 190+ iterations —
before touching the reward code, measure whether the physical premise
(contact force lands in a similar range regardless of size) actually
holds. Mass/inertia are NOT scaled with size (user's call, 2026-08-09), so
CARRY-phase (weight-bearing) force should be roughly size-invariant — but
close_gripper's own contact happens BEFORE lift, while the block still
rests on the table, where force comes from the finger position-controller
squeezing against the object's resistance, not from supporting its weight.
That's a geometry-dependent mechanism, not obviously size-invariant, and
worth measuring rather than assuming either way.

Explicitly sets specific (not randomly sampled) width/length/height combos
via block_randomization.sample_and_apply_block_size's underlying geom_size
write, spanning the extremes of the [0.01, 0.08] range, then runs the
SAME scripted-oracle grasp+carry measure_grip_contact_force.py already
used (oracle physically closes and holds the block — a working grasp,
policy-independent, so this measures the PHYSICS, not a trained policy's
competence).

No GPU needed.

Run with:
    modal run scripts/measure_grip_contact_force_by_size.py
"""

import modal
from think_then_act.modal_app import app, rl_image


@app.function(image=rl_image, gpu=None, timeout=300)
def measure_grip_contact_force_by_size(n_seeds: int = 3, max_steps: int = 100) -> dict:
    import os
    import numpy as np

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import mujoco
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env, grip_contact_forces, TABLE_TOP_Z, BLOCK_BODY_NAME
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.env.oracle import oracle_action

    def _block_geom_id(model) -> int:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BLOCK_BODY_NAME)
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == body_id:
                return gid
        raise RuntimeError(f"No geom found for body {BLOCK_BODY_NAME!r}")

    print("\n" + "=" * 60)
    print("  GRIP CONTACT FORCE vs. BLOCK SIZE")
    print("=" * 60)

    # (width_y, length_x, height_z) — full extents. "baseline" reproduces
    # the original fixed cube exactly, as a sanity check against the
    # already-measured 184.5-274.9N range.
    size_cases = {
        "baseline_5cm_cube": (0.05, 0.05, 0.05),
        "thin_width_1cm":    (0.01, 0.05, 0.05),
        "wide_width_8cm":    (0.08, 0.05, 0.05),
        "short_height_1cm":  (0.05, 0.05, 0.01),
        "tall_height_8cm":   (0.05, 0.05, 0.08),
        "small_cube_2cm":    (0.02, 0.02, 0.02),
        "large_cube_7cm":    (0.07, 0.07, 0.07),
    }

    results = {}
    for case_name, (width, length, height) in size_cases.items():
        env = ObservationHarness(
            gym.make("FetchPickAndPlace-v3", render_mode="rgb_array", max_episode_steps=max_steps + 20)
        )
        setup_env(env)
        raw = env.unwrapped
        gid = _block_geom_id(raw.model)
        raw.model.geom_size[gid] = [length / 2.0, width / 2.0, height / 2.0]
        resting_z = TABLE_TOP_Z + height / 2.0

        grip_strengths = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            obs, info = env.reset(seed=seed)
            # Manual mini version of init_random_episode's teleport, at
            # THIS case's own resting height (not the general randomizer —
            # this script pins an exact size per case, not a random draw).
            from think_then_act.env.setup import teleport_block
            table_cx, table_cy = 1.30, 0.75
            teleport_block(env, np.array([table_cx, table_cy, resting_z]))
            for fname in ("robot0:r_gripper_finger_joint", "robot0:l_gripper_finger_joint"):
                fid = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_JOINT, fname)
                raw.data.qpos[raw.model.jnt_qposadr[fid]] = 0.05
            mujoco.mj_forward(raw.model, raw.data)
            obs, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))

            carrying = False
            carry_forces = []
            for _ in range(max_steps):
                obs_arr, achieved, desired = obs["observation"], obs["achieved_goal"], obs["desired_goal"]
                action, phase, carrying = oracle_action(obs_arr, achieved, desired, carrying,
                                                          block_resting_z=resting_z)
                if carrying:
                    forces = grip_contact_forces(env)
                    carry_forces.append((forces["left"], forces["right"]))
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

            if len(carry_forces) >= 5:
                last5 = np.array(carry_forces[-5:])
                grip_strengths.append(float(np.mean(np.minimum(last5[:, 0], last5[:, 1]))))

        env.close()
        if grip_strengths:
            avg = float(np.mean(grip_strengths))
            print(f"  {case_name:20s} dims(w,l,h)=({width:.2f},{length:.2f},{height:.2f})  "
                  f"grip_strength={grip_strengths}  avg={avg:.4f}N")
            results[case_name] = {"dims": [width, length, height], "grip_strengths": grip_strengths, "avg": avg}
        else:
            print(f"  {case_name:20s} dims(w,l,h)=({width:.2f},{length:.2f},{height:.2f})  "
                  f"NEVER reached solid CARRY across {n_seeds} seeds")
            results[case_name] = {"dims": [width, length, height], "grip_strengths": [], "avg": None}

    print("\n" + "=" * 60)
    avgs = {k: v["avg"] for k, v in results.items() if v["avg"] is not None}
    if avgs:
        print(f"  Range across cases: {min(avgs.values()):.2f}N - {max(avgs.values()):.2f}N")
        print(f"  Current close_gripper_force_scale=130N, threshold-crossing ~=142.8N")
    print("=" * 60)

    return {"status": "PASS", "results": results}


@app.local_entrypoint()
def main(n_seeds: int = 3, max_steps: int = 100):
    result = measure_grip_contact_force_by_size.remote(n_seeds=n_seeds, max_steps=max_steps)
    print(f"\nDone: {result}")
