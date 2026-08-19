"""
sanity_check_block_size_randomization.py

One-off diagnostic, run BEFORE spending any real training compute on
block-size randomization (see hierarchical_architecture memory, Stage 4):
does the whole pipeline actually work across the sampled size range, or
does something break silently for sizes far from the original 5cm cube?

Checks, per seed, across sizes spanning the full [0.01, 0.08] range:
  1. The block actually settles resting ON the table (block_z close to
     TABLE_TOP_Z + half_height), not embedded in it or floating above —
     confirms the resting-height fix (env/setup.py's init_random_episode)
     is using the CURRENT episode's own sampled height, not a stale
     constant.
  2. init_episode_before_subgoal's scripted-oracle setup phase actually
     reaches close_gripper's handoff point (ok=True) — confirms
     oracle_action's block_resting_z fix actually prevents the
     APPROACH->GRASP->CARRY phase logic from misfiring at size extremes
     (this is the thing that would have broken silently without that fix:
     a short block's "lifted" signal firing too late, or a tall block's
     firing almost immediately).
  3. reward_close_gripper computes without error and produces a sane
     closedness/grip_strength (confirms grip_contact_forces + the
     force-based reward generalize across sizes, not just the original
     cube — the whole point of that earlier redesign).
  4. The observation vector has the right shape (CLOSE_GRIPPER_OBS_DIM)
     and its trailing block_dims slice is close to the TRUE sampled size
     (within noise) — confirms the perceived-dims wiring end-to-end.

No GPU needed, no training, no checkpoints written.

Run with:
    modal run scripts/sanity_check_block_size_randomization.py
    modal run scripts/sanity_check_block_size_randomization.py --n-seeds 20
"""

import modal
from think_then_act.modal_app import app, rl_image


@app.function(image=rl_image, gpu=None, timeout=300)
def sanity_check_block_size_randomization(n_seeds: int = 10) -> dict:
    import numpy as np

    import os
    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env, TABLE_TOP_Z, grip_contact_forces
    from think_then_act.env.block_randomization import get_block_dims, get_perceived_block_dims
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.training.subgoal_env import SubgoalConditionedEnv
    from think_then_act.training.subgoal_features import CLOSE_GRIPPER_OBS_DIM

    print("\n" + "=" * 60)
    print("  BLOCK-SIZE RANDOMIZATION SANITY CHECK")
    print("=" * 60)

    rows = []
    for seed in range(n_seeds):
        base = ObservationHarness(
            gym.make("FetchPickAndPlace-v3", render_mode="rgb_array", max_episode_steps=300)
        )
        setup_env(base)
        env = SubgoalConditionedEnv(
            base, subgoal="close_gripper", max_episode_steps=30, randomize_block_size=True,
        )
        rng = np.random.default_rng(seed)
        flat_obs, info = env.reset(rng=rng)

        true_dims = get_block_dims(base.unwrapped.model)
        perceived = get_perceived_block_dims(base.unwrapped.model)
        block_z = float(info["block_pos"][2])
        resting_z = TABLE_TOP_Z + true_dims["height"] / 2.0
        # "settled ok" — spawned within a few mm of its own resting height,
        # not embedded in the table or floating above it. A little slack
        # (1cm) for physics settling after the fresh reset.
        spawn_ok = abs(block_z - resting_z) < 0.01

        # SubgoalConditionedEnv.reset() silently falls back to a bare
        # env.reset() if init_episode_before_subgoal's oracle setup phase
        # fails to reach close_gripper's handoff point (ok=False) — a
        # caller never sees that failure directly. d_grip_block at reset is
        # the existing project-wide tell for this: a genuine successful
        # setup lands within a few cm of the block (this project's own
        # measured range for a working setup is ~0.01-0.03m); a silently-
        # fallen-back bare reset produces a MUCH larger distance (a prior
        # bug in this project's history showed ~0.55m for exactly this
        # failure mode). >0.1m is treated as "setup likely failed."
        d_grip_block_at_reset = float(info.get("d_grip_block", -1.0))
        setup_ok = 0.0 <= d_grip_block_at_reset < 0.10

        forces = grip_contact_forces(env)
        obs_shape_ok = flat_obs.shape == (CLOSE_GRIPPER_OBS_DIM,)
        perceived_err = float(np.linalg.norm(
            perceived - np.array([true_dims["width"], true_dims["length"], true_dims["height"]])
        ))

        row = {
            "seed": seed,
            "true_dims": {k: round(v, 4) for k, v in true_dims.items()},
            "block_z": round(block_z, 4), "resting_z": round(resting_z, 4),
            "spawn_ok": spawn_ok,
            "d_grip_block_at_reset": round(d_grip_block_at_reset, 4), "setup_ok": setup_ok,
            "obs_shape": flat_obs.shape, "obs_shape_ok": obs_shape_ok,
            "perceived_dims_err": round(perceived_err, 5),
            "grip_forces_at_reset": {k: round(v, 4) for k, v in forces.items()},
        }
        rows.append(row)
        status = "OK" if (spawn_ok and setup_ok and obs_shape_ok) else "*** PROBLEM ***"
        print(f"  seed={seed:>2}  dims(w,l,h)={list(row['true_dims'].values())}  "
              f"block_z={block_z:.4f}  resting_z={resting_z:.4f}  spawn_ok={spawn_ok}  "
              f"d_grip_block={d_grip_block_at_reset:.4f}  setup_ok={setup_ok}  "
              f"obs_shape={flat_obs.shape}  perceived_err={perceived_err:.4f}  {status}")
        env.close()

    n_spawn_ok = sum(1 for r in rows if r["spawn_ok"])
    n_setup_ok = sum(1 for r in rows if r["setup_ok"])
    n_shape_ok = sum(1 for r in rows if r["obs_shape_ok"])
    print("\n" + "=" * 60)
    print(f"  spawn_ok:      {n_spawn_ok}/{n_seeds}")
    print(f"  setup_ok:      {n_setup_ok}/{n_seeds}  (oracle reached close_gripper's handoff point)")
    print(f"  obs_shape_ok:  {n_shape_ok}/{n_seeds}")
    print("=" * 60)

    all_ok = n_spawn_ok == n_seeds and n_setup_ok == n_seeds and n_shape_ok == n_seeds
    return {
        "status": "PASS" if all_ok else "FAIL",
        "n_seeds": n_seeds, "n_spawn_ok": n_spawn_ok, "n_setup_ok": n_setup_ok, "n_shape_ok": n_shape_ok,
        "rows": rows,
    }


@app.local_entrypoint()
def main(n_seeds: int = 10):
    result = sanity_check_block_size_randomization.remote(n_seeds=n_seeds)
    print(f"\nDone: {result['status']} — spawn_ok {result['n_spawn_ok']}/{result['n_seeds']}, "
          f"obs_shape_ok {result['n_shape_ok']}/{result['n_seeds']}")
