"""
diagnose_move_to_target_setup.py

One-off diagnostic (NOT a training run) to check one specific hypothesis
about why move_to_target's block-size-randomized retrain (2026-08-17) is
regressing instead of converging (completion_rate 10%->30%->10%->0% across
iter50-200, value_loss never settling — see hierarchical_architecture
memory, Stage 4): is init_episode_before_subgoal's scripted-oracle setup
chain (align_xy->descend->close_gripper->lift, all the way to "just
entered CARRY") silently failing more often now that block size AND mass
vary per episode, handing move_to_target a bad/ungrasped starting state
more often than before?

Same "read the d_grip_block(init=...) tell" diagnostic already used to
catch this exact failure mode for close_gripper (2026-07-15/16, see
bugs_and_fixes memory): a genuine successful setup lands d_grip_block
within a few cm; a silently-fallen-back bare env.reset() produces a much
larger value (historically ~0.55m). Stratifies by block size bucket (not
just an aggregate rate) to check whether failures concentrate at size
extremes, the same pattern close_gripper's own geometry measurement found.

No GPU, no checkpoints needed — move_to_target's setup chain uses the
SCRIPTED oracle (env/oracle.py), not any trained policy (align_xy_policy
is only used for descend's own setup, per init_episode_before_subgoal's
signature) — so this needs nothing from the model volume.

Run with:
    modal run scripts/diagnose_move_to_target_setup.py
    modal run scripts/diagnose_move_to_target_setup.py --n-seeds 40
"""

import modal
from think_then_act.modal_app import app, rl_image


@app.function(image=rl_image, gpu=None, timeout=300)
def diagnose_move_to_target_setup(n_seeds: int = 30) -> dict:
    import numpy as np

    import os
    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env, TABLE_TOP_Z
    from think_then_act.env.block_randomization import get_block_dims
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.training.subgoal_env import SubgoalConditionedEnv

    print("\n" + "=" * 70)
    print("  move_to_target SETUP-CHAIN DIAGNOSTIC (block-size randomized)")
    print("=" * 70)

    rows = []
    for seed in range(n_seeds):
        base = ObservationHarness(
            gym.make("FetchPickAndPlace-v3", render_mode="rgb_array", max_episode_steps=300)
        )
        setup_env(base)
        env = SubgoalConditionedEnv(
            base, subgoal="move_to_target", max_episode_steps=30, randomize_block_size=True,
        )
        rng = np.random.default_rng(seed)
        flat_obs, info = env.reset(rng=rng)

        true_dims = get_block_dims(base.unwrapped.model)
        block_z = float(info["block_pos"][2])
        resting_z = TABLE_TOP_Z + true_dims["height"] / 2.0
        height_above_table = block_z - resting_z

        # reward_move_to_target's own breakdown does NOT report d_grip_block
        # (that's a close_gripper-only field) — env/setup.py's
        # _subgoal_setup_reached("move_to_target", ...) predicate IS
        # `carrying and block_z > block_resting_z + 0.025`, so
        # height_above_table is the directly-meaningful setup-success tell
        # for THIS subgoal: a genuine successful chained setup hands off
        # right at/above that +0.025 margin (still holding, just lifted); a
        # silent fallback to a bare env.reset() leaves the block resting ON
        # the table (height_above_table ~ 0), same class of tell as
        # close_gripper's d_grip_block(init=...) blowup, just the field
        # that's actually populated for this subgoal.
        setup_ok = height_above_table > 0.02
        d_block_target_at_reset = float(info.get("d_block_target", -1.0))

        max_dim = max(true_dims.values())
        min_dim = min(true_dims.values())
        size_bucket = "small(<3cm)" if max_dim < 0.03 else ("large(>6cm)" if min_dim > 0.06 else "mid")

        row = {
            "seed": seed,
            "true_dims": {k: round(v, 4) for k, v in true_dims.items()},
            "size_bucket": size_bucket,
            "block_z": round(block_z, 4), "resting_z": round(resting_z, 4),
            "height_above_table": round(height_above_table, 4),
            "setup_ok": setup_ok,
            "d_block_target_at_reset": round(d_block_target_at_reset, 4),
        }
        rows.append(row)
        status = "OK" if setup_ok else "*** SETUP FAILED (fell back) ***"
        print(f"  seed={seed:>2}  bucket={size_bucket:<11}  dims(w,l,h)={list(row['true_dims'].values())}  "
              f"height_above_table={height_above_table:.4f}  "
              f"d_block_target={d_block_target_at_reset:.4f}  {status}")
        env.close()

    n_seeds_total = len(rows)
    n_setup_ok = sum(1 for r in rows if r["setup_ok"])
    buckets = {}
    for r in rows:
        b = buckets.setdefault(r["size_bucket"], {"n": 0, "ok": 0})
        b["n"] += 1
        b["ok"] += int(r["setup_ok"])

    print("\n" + "=" * 70)
    print(f"  overall setup_ok: {n_setup_ok}/{n_seeds_total} ({100*n_setup_ok/n_seeds_total:.0f}%)")
    for bucket, counts in buckets.items():
        print(f"  {bucket:<11}: {counts['ok']}/{counts['n']} setup_ok "
              f"({100*counts['ok']/counts['n']:.0f}%)")
    ok_rows  = [r for r in rows if r["setup_ok"]]
    bad_rows = [r for r in rows if not r["setup_ok"]]
    if ok_rows:
        print(f"  d_block_target_at_reset (setup_ok rows): "
              f"mean={np.mean([r['d_block_target_at_reset'] for r in ok_rows]):.4f}  "
              f"min={min(r['d_block_target_at_reset'] for r in ok_rows):.4f}  "
              f"max={max(r['d_block_target_at_reset'] for r in ok_rows):.4f}")
    if bad_rows:
        print(f"  height_above_table (FAILED rows): "
              f"{[round(r['height_above_table'], 3) for r in bad_rows]}")
    print("=" * 70)

    return {
        "n_seeds": n_seeds_total, "n_setup_ok": n_setup_ok,
        "buckets": buckets, "rows": rows,
    }


@app.local_entrypoint()
def main(n_seeds: int = 30):
    result = diagnose_move_to_target_setup.remote(n_seeds=n_seeds)
    print(f"\nDone: setup_ok {result['n_setup_ok']}/{result['n_seeds']}")
