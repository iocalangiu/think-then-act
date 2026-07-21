"""
verify_mujoco.py

Milestone 1 verification script.

Run with:
    modal run scripts/milestones/verify_mujoco.py

What this does:
  1. Boots our Modal cloud container (the `rl_image` we defined in modal_config.py).
  2. Sets the MUJOCO_GL env var to 'osmesa' so MuJoCo uses the headless
     software renderer instead of looking for a physical display.
  3. Imports mujoco and gymnasium_robotics — any missing system lib will
     explode here with a clear error, which is what we want to catch now.
  4. Creates the FetchPickAndPlace-v3 environment (a 7-DOF robot arm that
     must pick up a block and place it at a target position).
  5. Runs 5 random-action steps and prints structured logs.
  6. Confirms the observation space and action space shapes so we know
     exactly what tensors we'll be feeding the vision model later.
"""

import modal
from think_then_act.modal_app import app, rl_image

# ---------------------------------------------------------------------------
# The @app.function decorator tells Modal:
#   - Run this function inside `rl_image` (our custom container).
#     modal_config is already bundled into rl_image via add_local_python_source.
#   - Give it a GPU (T4 is fine for simulation; no training yet).
#   - Allow up to 10 minutes (plenty for a quick smoke-test).
# ---------------------------------------------------------------------------
@app.function(
    image=rl_image,
    gpu="T4",           # cheapest GPU tier — just to confirm GPU path works
    timeout=600,
)
def verify_headless_mujoco() -> dict:
    """
    Runs a 5-step headless MuJoCo episode and returns a summary dict.
    All heavy imports happen *inside* this function so they execute in
    the cloud container, not on your local machine.
    """
    import os
    import sys
    import time

    # Tell MuJoCo to use the OSMesa software renderer.
    # Must be set BEFORE importing mujoco or gymnasium.
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    print("\n" + "=" * 60)
    print("  MILESTONE 1 — Headless MuJoCo Verification")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Step 1: Confirm Python + system context
    # -----------------------------------------------------------------------
    print(f"\n[1/5] Runtime info")
    print(f"      Python  : {sys.version.split()[0]}")
    print(f"      Platform: {sys.platform}")
    print(f"      MUJOCO_GL: {os.environ['MUJOCO_GL']}")

    # -----------------------------------------------------------------------
    # Step 2: Import MuJoCo and print its version
    # -----------------------------------------------------------------------
    print(f"\n[2/5] Importing mujoco...")
    import mujoco  # noqa: E402
    print(f"      mujoco version : {mujoco.__version__}")

    # -----------------------------------------------------------------------
    # Step 3: Import gymnasium + gymnasium_robotics
    # -----------------------------------------------------------------------
    print(f"\n[3/5] Importing gymnasium & gymnasium_robotics...")
    import gymnasium as gym
    import gymnasium_robotics  # registers the Fetch* envs
    print(f"      gymnasium version           : {gym.__version__}")
    print(f"      gymnasium_robotics version  : {gymnasium_robotics.__version__}")

    # -----------------------------------------------------------------------
    # Step 4: Create the FetchPickAndPlace environment
    #
    # Target task: a 7-DOF Fetch robot arm must grasp a block and move it
    # to a randomly placed 3-D target position.
    #   Observation: dict with 'observation' (25,), 'achieved_goal' (3,),
    #                'desired_goal' (3,)
    #   Action     : 4-dim continuous [dx, dy, dz, gripper_cmd] in [-1, 1]
    #   Reward     : sparse (-1 / 0) by default; replaced in Milestone 4.
    #
    # gymnasium-robotics 1.3.x registers v3 envs (mujoco 3.x physics).
    # We probe v3 first and fall back to v2 so the script is version-agnostic.
    # -----------------------------------------------------------------------
    ENV_CANDIDATES = ["FetchPickAndPlace-v3", "FetchPickAndPlace-v2"]
    env_id = None
    for candidate in ENV_CANDIDATES:
        try:
            gym.make(candidate, render_mode=None)
            env_id = candidate
            break
        except Exception:
            continue
    if env_id is None:
        raise RuntimeError(f"None of {ENV_CANDIDATES} could be created. "
                           "Check gymnasium-robotics version.")
    print(f"\n[4/5] Creating {env_id} environment...")
    t0 = time.time()
    env = gym.make(
        env_id,
        render_mode=None,   # None = no rendering window (headless)
        max_episode_steps=50,
    )
    obs, info = env.reset(seed=42)
    elapsed = time.time() - t0
    print(f"      Environment created in {elapsed:.2f}s")

    # Print observation and action shapes — we'll need these numbers in M2/M3
    obs_shape = obs["observation"].shape
    ag_shape  = obs["achieved_goal"].shape
    dg_shape  = obs["desired_goal"].shape
    act_shape = env.action_space.shape

    print(f"\n      Observation space breakdown:")
    print(f"        obs['observation']    shape: {obs_shape}")
    print(f"        obs['achieved_goal']  shape: {ag_shape}")
    print(f"        obs['desired_goal']   shape: {dg_shape}")
    print(f"      Action space shape           : {act_shape}")
    print(f"      Action space bounds   low={env.action_space.low[0]:.1f}  "
          f"high={env.action_space.high[0]:.1f}")

    # -----------------------------------------------------------------------
    # Step 5: Run 5 random-action steps
    # -----------------------------------------------------------------------
    print(f"\n[5/5] Running 5 random-action steps...")
    step_log = []
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        entry = {
            "step"      : step,
            "reward"    : float(reward),
            "terminated": terminated,
            "truncated" : truncated,
            "is_success": bool(info.get("is_success", False)),
        }
        step_log.append(entry)
        print(f"      step {step}: reward={reward:+.1f}  "
              f"success={entry['is_success']}  "
              f"done={terminated or truncated}")

    env.close()

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    summary = {
        "mujoco_version"            : mujoco.__version__,
        "gymnasium_version"         : gym.__version__,
        "gymnasium_robotics_version": gymnasium_robotics.__version__,
        "env_id"                    : env_id,
        "obs_shape"                 : list(obs_shape),
        "action_shape"              : list(act_shape),
        "steps_completed"           : len(step_log),
        "step_log"                  : step_log,
        "status"                    : "PASS",
    }

    print("\n" + "=" * 60)
    print("  RESULT: PASS — headless MuJoCo is fully operational")
    print("=" * 60 + "\n")

    return summary


# ---------------------------------------------------------------------------
# Local entrypoint: called when you run `modal run scripts/milestones/verify_mujoco.py`
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    """
    Triggers the remote function and pretty-prints the returned summary
    so you get a clear green/red signal on your local terminal.
    """
    import json

    print("\nDispatching job to Modal cloud...")
    result = verify_headless_mujoco.remote()

    print("\n--- Summary (returned to local terminal) ---")
    print(json.dumps(result, indent=2))

    assert result["status"] == "PASS", "Verification FAILED — check logs above."
    print("\nMilestone 1 complete. Ready for Milestone 2.")
