"""
diagnose_flat_bc_success_quality.py

One-off diagnostic (NOT a training run, writes nothing): checks whether
train_flat_task_bc.py's eval "SUCCESS" episodes are genuine grasp-carry-
place completions or just the block sliding/getting knocked near the goal
by incidental contact — is_success in this env only checks "is achieved_goal
within tolerance of desired_goal RIGHT NOW," with no requirement the block
was ever actually lifted/held. Same class of proxy-gaming issue as the
close_gripper reward's own history (see hierarchical_architecture memory) —
verify_close_gripper_grasp.py exists for exactly this reason, applied here
to the full-task success signal instead.

For every episode, tracks per-step: block height above its resting surface
(achieved_goal[2] - block_resting_z, block_resting_z=0.425 for the
project's fixed 5cm cube — same constant env/setup.py's init_random_episode
uses) and real per-finger contact force (env.setup.grip_contact_forces).
A genuine grasp-carry needs BOTH fingers in contact (min(left,right) > 0)
AND the block meaningfully lifted (> 0.02m, matching the project's existing
lift/height thresholds) at some point before is_success fires — sliding
keeps the block near table height with no sustained two-finger contact.

Run with:
    modal run scripts/multi-cube-experiment/diagnose_flat_bc_success_quality.py
    modal run scripts/multi-cube-experiment/diagnose_flat_bc_success_quality.py --n-seeds 40
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(image=rl_image, gpu=None, cpu=2.0,
              volumes={MODEL_CACHE_DIR: model_volume}, timeout=600)
def diagnose_flat_bc_success_quality(
    ckpt_path: str = "checkpoints/flat_task_bc.pt",
    n_seeds: int = 20,
    max_steps: int = 100,
    lift_threshold: float = 0.02,
    block_resting_z: float = 0.425,
) -> dict:
    import os
    import numpy as np
    import torch

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env, init_random_episode, grip_contact_forces
    from think_then_act.training.subgoal_features import build_flat_observation, FLAT_OBS_DIM
    from think_then_act.policy.subgoal_recurrent_policy import SubgoalRecurrentPolicy

    print("\n" + "=" * 70)
    print("  FLAT BC SUCCESS-QUALITY CHECK (is 'success' a real grasp-carry?)")
    print("=" * 70)

    full_ckpt_path = os.path.join(MODEL_CACHE_DIR, ckpt_path)
    actor = SubgoalRecurrentPolicy(obs_dim=FLAT_OBS_DIM, action_dim=4)
    ckpt = torch.load(full_ckpt_path, map_location="cpu")
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    print(f"  Loaded {full_ckpt_path}")

    env = gym.make("FetchPickAndPlace-v3", max_episode_steps=max_steps)
    setup_env(env)

    results = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        env.reset(seed=seed)
        obs, setup_ok = init_random_episode(env, rng)
        if not setup_ok:
            continue

        hidden_state = None
        success_step = None
        max_height = -999.0
        ever_two_finger_contact_while_lifted = False

        for step in range(max_steps):
            flat_obs = build_flat_observation(obs["observation"], obs["achieved_goal"], obs["desired_goal"])
            action, hidden_state = actor.act(flat_obs, hidden_state, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            block_z = float(obs["achieved_goal"][2])
            height_above_resting = block_z - block_resting_z
            max_height = max(max_height, height_above_resting)

            forces = grip_contact_forces(env)
            two_finger_contact = min(forces["left"], forces["right"]) > 0.0
            if two_finger_contact and height_above_resting > lift_threshold:
                ever_two_finger_contact_while_lifted = True

            if info.get("is_success", False) and success_step is None:
                success_step = step

            if terminated or truncated:
                break

        if success_step is not None:
            verdict = "GENUINE grasp-carry" if ever_two_finger_contact_while_lifted else \
                      "SLIDE/NUDGE — never genuinely lifted+gripped"
            print(f"  seed={seed:>3}  SUCCESS @ step {success_step:>3}  "
                  f"max_height_above_resting={max_height:+.4f}m  "
                  f"ever_lifted+gripped={ever_two_finger_contact_while_lifted}  -> {verdict}")
            results.append({"seed": seed, "success_step": success_step,
                             "max_height_above_resting": max_height,
                             "genuine_grasp_carry": ever_two_finger_contact_while_lifted})

    env.close()

    n_success = len(results)
    n_genuine = sum(1 for r in results if r["genuine_grasp_carry"])
    print("\n" + "=" * 70)
    if n_success == 0:
        print("  No successes in this sample — can't assess success quality here.")
    else:
        print(f"  {n_genuine}/{n_success} 'successes' were genuine grasp-carries "
              f"({100*n_genuine/n_success:.0f}%); {n_success - n_genuine}/{n_success} "
              f"were slide/nudge — never a lifted+gripped state.")
    print("=" * 70)

    return {"n_seeds": n_seeds, "n_success": n_success, "n_genuine_grasp_carry": n_genuine,
            "results": results}


@app.local_entrypoint()
def main(ckpt_path: str = "checkpoints/flat_task_bc.pt", n_seeds: int = 20):
    result = diagnose_flat_bc_success_quality.remote(ckpt_path=ckpt_path, n_seeds=n_seeds)
    print(f"\nDone: {result['n_genuine_grasp_carry']}/{result['n_success']} genuine "
          f"(of {result['n_seeds']} seeds attempted)")
