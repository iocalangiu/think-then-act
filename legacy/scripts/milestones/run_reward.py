"""
run_reward.py

Milestone 4 verification script.

Run with:
    modal run scripts/milestones/run_reward.py

What this does:
  1. Runs a 20-step random-action episode through ObservationHarness.
  2. Applies compute_dense_reward() to every step.
  3. Prints a side-by-side comparison table: sparse vs dense reward.
  4. Asserts that dense reward has meaningfully higher variance than sparse
     (the signal varies with state changes, not just constant -1.0).

Key question to answer: does the dense reward provide useful gradient
information that a learning algorithm can follow?  If the variance is
near-zero, the signal is as useless as sparse.
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(
    image=rl_image,
    gpu="T4",
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=600,
)
def run_dense_reward_episode(n_steps: int = 20) -> dict:
    """
    Runs one random episode and returns sparse + dense reward data.
    """
    import os
    import numpy as np

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.reward.dense_reward import compute_dense_reward, apply_to_episode, DEFAULT_WEIGHTS

    print("\n" + "=" * 60)
    print("  MILESTONE 4 — Dense Reward Verification")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Run a random episode
    # ------------------------------------------------------------------
    print(f"\n[1/3] Running {n_steps}-step random episode...")
    env = gym.make(
        "FetchPickAndPlace-v3",
        render_mode="rgb_array",
        max_episode_steps=n_steps,
    )
    env = ObservationHarness(env)
    obs, info = env.reset(seed=42)

    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.close()
    log = env.metadata_only()

    # ------------------------------------------------------------------
    # Apply dense reward to the full episode log
    # ------------------------------------------------------------------
    print(f"\n[2/3] Computing dense rewards for {len(log)} steps...")
    enriched = apply_to_episode(log, weights=DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print(f"\n[3/3] Sparse vs Dense reward table:")
    print(f"\n  {'step':>4}  {'sparse':>8}  {'dense':>8}  "
          f"{'r_approach':>10}  {'r_transport':>12}  {'r_grasp':>8}  "
          f"{'d_grip→blk':>10}  {'d_blk→tgt':>10}")
    print("  " + "-" * 80)

    sparse_vals = []
    dense_vals  = []

    for entry in enriched:
        step = entry["step"]
        if entry.get("reward") is None:
            continue   # skip reset step (no sparse reward yet)
        sparse = entry["reward"]
        bd     = entry["reward_breakdown"]
        dense  = entry["dense_reward"]

        sparse_vals.append(sparse)
        dense_vals.append(dense)

        print(f"  {step:>4}  {sparse:>8.3f}  {dense:>8.4f}  "
              f"{bd['r_approach']:>10.4f}  {bd['r_transport']:>12.4f}  "
              f"{bd['r_grasp']:>8.4f}  "
              f"{bd['d_grip_block']:>10.4f}  {bd['d_block_target']:>10.4f}")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    sparse_arr = np.array(sparse_vals)
    dense_arr  = np.array(dense_vals)

    sparse_var = float(np.var(sparse_arr))
    dense_var  = float(np.var(dense_arr))
    dense_min  = float(dense_arr.min())
    dense_max  = float(dense_arr.max())
    dense_mean = float(dense_arr.mean())

    print(f"\n  Sparse reward : mean={sparse_arr.mean():.3f}  "
          f"var={sparse_var:.6f}  range=[{sparse_arr.min():.1f}, {sparse_arr.max():.1f}]")
    print(f"  Dense  reward : mean={dense_mean:.4f}  "
          f"var={dense_var:.6f}  range=[{dense_min:.4f}, {dense_max:.4f}]")
    print(f"\n  Dense variance / Sparse variance = "
          f"{(dense_var / max(sparse_var, 1e-9)):.1f}x more informative")

    # Dense should vary significantly more than sparse (sparse var ≈ 0
    # for random policy that never succeeds).
    signal_ratio = dense_var / max(sparse_var, 1e-9)

    result = {
        "n_steps"       : len(log) - 1,   # exclude reset step
        "sparse_mean"   : round(float(sparse_arr.mean()), 5),
        "sparse_var"    : round(sparse_var, 8),
        "dense_mean"    : round(dense_mean, 5),
        "dense_var"     : round(dense_var,  8),
        "dense_min"     : round(dense_min,  5),
        "dense_max"     : round(dense_max,  5),
        "signal_ratio"  : round(signal_ratio, 2),
        "reward_weights": DEFAULT_WEIGHTS.as_dict(),
        "status"        : "PASS",
    }

    print("\n" + "=" * 60)
    print(f"  RESULT: PASS — dense reward varies {signal_ratio:.1f}x more than sparse")
    print("=" * 60 + "\n")

    return result


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    import json

    print("\nDispatching dense reward episode to Modal cloud...")
    result = run_dense_reward_episode.remote(n_steps=20)

    print("\n--- Reward Summary (local terminal) ---")
    print(json.dumps(result, indent=2))

    assert result["status"] == "PASS"
    assert result["dense_var"] > result["sparse_var"], (
        "Dense reward has no more variance than sparse — check reward formula."
    )
    assert result["signal_ratio"] > 5, (
        f"Dense reward only {result['signal_ratio']}x more variable than sparse. "
        f"Weights may need tuning."
    )

    print(f"\nReward signal ratio : {result['signal_ratio']}x  (dense vs sparse variance)")
    print(f"Dense range         : [{result['dense_min']}, {result['dense_max']}]")
    print("\nMilestone 4 complete. Ready for Milestone 5.")
