"""
run_harness.py

Milestone 2 verification script.

Run with:
    modal run run_harness.py

What this does:
  1. Creates FetchPickAndPlace-v3 with render_mode="rgb_array" so every
     call to env.render() returns a (H, W, 3) uint8 numpy array.
  2. Wraps the env with ObservationHarness to record frames + state.
  3. Runs N random-action steps and asserts the harness captured them all.
  4. Returns a JSON-safe summary dict to your local terminal.

Key thing we're verifying: the osmesa headless renderer produces real
pixel data (non-zero frames) — this is the hard part on a GPU server.
"""

import modal
from modal_config import app, rl_image


@app.function(
    image=rl_image,
    gpu="T4",
    timeout=600,
)
def run_episode_with_harness(n_steps: int = 10) -> dict:
    """
    Runs one FetchPickAndPlace episode under the ObservationHarness and
    returns a JSON-safe summary.  All imports are deferred so they execute
    in the cloud container, not on your local machine.
    """
    import os
    import sys
    import time

    # Headless renderer — must be set before any mujoco/gymnasium import.
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    print("\n" + "=" * 60)
    print("  MILESTONE 2 — Observation Harness Verification")
    print("=" * 60)

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401 — registers Fetch* envs
    from obs_wrapper import ObservationHarness

    # ------------------------------------------------------------------
    # Step 1: Create env with rgb_array render mode
    # ------------------------------------------------------------------
    print(f"\n[1/4] Creating FetchPickAndPlace-v3 (render_mode='rgb_array')...")
    t0 = time.time()
    env = gym.make(
        "FetchPickAndPlace-v3",
        render_mode="rgb_array",   # tells MuJoCo to return pixel arrays
        max_episode_steps=n_steps,
    )
    env = ObservationHarness(env)
    print(f"      Done in {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # Step 2: Reset — captures the initial frame (step 0)
    # ------------------------------------------------------------------
    print(f"\n[2/4] Resetting environment (captures frame 0)...")
    obs, info = env.reset(seed=42)
    initial_frame = env.last_frame()
    print(f"      Frame 0 shape : {initial_frame.shape}")
    print(f"      Frame 0 dtype : {initial_frame.dtype}")
    print(f"      Pixel range   : [{initial_frame.min()}, {initial_frame.max()}]")

    # A non-zero max is the key check: osmesa rendered real pixels.
    if initial_frame.max() == 0:
        raise RuntimeError(
            "Frame is all zeros — osmesa renderer produced a black frame. "
            "Check MUJOCO_GL=osmesa and libosmesa6 installation."
        )

    # ------------------------------------------------------------------
    # Step 3: Run N random steps
    # ------------------------------------------------------------------
    print(f"\n[3/4] Running {n_steps} random-action steps...")
    for step_i in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"      step {step_i + 1:>2}: "
              f"reward={reward:+.1f}  "
              f"success={info.get('is_success', False)}  "
              f"frame_max={env.last_frame().max()}")
        if terminated or truncated:
            break

    env.close()

    # ------------------------------------------------------------------
    # Step 4: Build and validate the episode summary
    # ------------------------------------------------------------------
    print(f"\n[4/4] Validating episode log...")
    log = env.metadata_only()   # JSON-safe — no numpy arrays
    summary = env.episode_summary()

    print(f"      Steps logged     : {summary['total_steps']}")
    print(f"      Frame shape      : {summary['frame_shape']}  (H x W x RGB)")
    print(f"      Pixel range      : [{summary['pixel_min']}, {summary['pixel_max']}]")
    print(f"      Total reward     : {summary['total_reward']:.1f}")
    print(f"      Any success      : {summary['any_success']}")

    # Sample the first logged entry to show state vector layout
    first_entry = log[0]   # step=0 (after reset)
    print(f"\n      Sample step-0 state:")
    print(f"        achieved_goal (block XYZ) : {first_entry['achieved_goal']}")
    print(f"        desired_goal  (target XYZ): {first_entry['desired_goal']}")
    print(f"        observation   (len={len(first_entry['observation'])}): "
          f"{[round(v, 3) for v in first_entry['observation'][:6]]} ...")

    result = {
        "steps_logged"        : len(log),
        "frame_shape"         : summary["frame_shape"],
        "frame_dtype"         : summary["frame_dtype"],
        "pixel_min"           : summary["pixel_min"],
        "pixel_max"           : summary["pixel_max"],
        "total_reward"        : summary["total_reward"],
        "any_success"         : summary["any_success"],
        "sample_achieved_goal": first_entry["achieved_goal"],
        "sample_desired_goal" : first_entry["desired_goal"],
        "obs_vector_length"   : len(first_entry["observation"]),
        "status"              : "PASS",
    }

    print("\n" + "=" * 60)
    print("  RESULT: PASS — harness captured frames and state correctly")
    print("=" * 60 + "\n")

    return result


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    import json

    print("\nDispatching harness episode to Modal cloud...")
    result = run_episode_with_harness.remote(n_steps=10)

    print("\n--- Harness Summary (local terminal) ---")
    print(json.dumps(result, indent=2))

    # Hard assertions — these must hold before we proceed to M3
    assert result["status"] == "PASS"
    assert result["frame_shape"][-1] == 3,    "Expected 3-channel RGB frames"
    assert result["pixel_max"] > 0,           "Renderer returned all-black frames"
    assert result["obs_vector_length"] == 25, "Unexpected observation vector length"
    assert result["steps_logged"] >= 2,       "Episode log too short"

    print(f"\nFrame shape  : {result['frame_shape']}  (osmesa rendered real pixels)")
    print(f"Pixel range  : [{result['pixel_min']}, {result['pixel_max']}]")
    print(f"Steps logged : {result['steps_logged']}  (includes step-0 reset frame)")
    print(f"Obs vector   : length {result['obs_vector_length']}  (25 floats as expected)")
    print("\nMilestone 2 complete. Ready for Milestone 3.")
