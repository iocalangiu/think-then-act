"""
diagnose_close_gripper_small_width.py

close_gripper fails at width=0.01 (the trained WIDTH_RANGE's own floor —
should be in-distribution) under BOTH ground truth AND the pose model
(record_close_gripper_at_size.py, 2026-08-19), which rules out perception
as the cause. The obvious next suspect — a degenerate (near-zero/negative)
force_scale from extrapolating the force-vs-width linear fit down to small
widths — was checked analytically and ruled out too: force_scale=35.4N at
width=0.01, nowhere near degenerate (see reward_close_gripper's
close_gripper_force_intercept/_per_width). So the failure has to be in the
ACTUAL simulated contact force, not the reward's calibration curve — this
traces closedness/grip_strength/force_scale/d_xy/d_z step-by-step to see
which one it actually is: never-contacts (grip_strength stays ~0 the whole
episode), contacts-but-under-threshold (grip_strength rises but caps below
what's needed), or loses-track (d_xy/d_z drift instead).

Ground truth only (no pose model) — isolates the physical/reward question
from the perception question already answered separately.

Run with:
    modal run scripts/diagnose_close_gripper_small_width.py --width 0.01
    modal run scripts/diagnose_close_gripper_small_width.py --width 0.01 --n-seeds 5
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(image=rl_image, gpu=None, volumes={MODEL_CACHE_DIR: model_volume}, timeout=300)
def diagnose_close_gripper_small_width(width: float = 0.01, n_seeds: int = 3, max_steps: int = 30) -> dict:
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
    from think_then_act.training.checkpoints import resolve_subgoal_checkpoint
    from think_then_act.training.subgoal_env import SubgoalConditionedEnv
    from think_then_act.training.subgoal_features import CLOSE_GRIPPER_OBS_DIM

    subgoal = "close_gripper"
    ckpt_dir = os.path.join(MODEL_CACHE_DIR, "checkpoints")
    ckpt_path = resolve_subgoal_checkpoint(ckpt_dir, subgoal, algo="ppo", use_best=True)
    policy = SubgoalGaussianPolicy(obs_dim=CLOSE_GRIPPER_OBS_DIM)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    policy.load_state_dict(ckpt["actor"] if isinstance(ckpt, dict) and "actor" in ckpt else ckpt)
    policy.eval()
    print(f"checkpoint <- {ckpt_path}\nwidth={width}  n_seeds={n_seeds}\n")

    all_rows = []
    for seed in range(n_seeds):
        base = ObservationHarness(
            gym.make("FetchPickAndPlace-v3", render_mode="rgb_array", max_episode_steps=max_steps + 250)
        )
        setup_env(base)
        env = SubgoalConditionedEnv(
            base, subgoal=subgoal, max_episode_steps=max_steps, randomize_block_size=True,
            width_range=(width, width), length_range=(width, width), height_range=(width, width),
        )
        rng = np.random.default_rng(seed)
        obs, info = env.reset(rng=rng)
        print(f"--- seed={seed} ---  init d_grip_block={info.get('d_grip_block'):.4f}")
        print(f"{'step':>4} {'closedness':>10} {'grip_str':>9} {'force_scale':>11} "
              f"{'left_F':>7} {'right_F':>7} {'d_xy':>7} {'d_z':>7} {'transl':>7}")

        for step in range(max_steps):
            action = policy.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            row = {
                "seed": seed, "step": step,
                "closedness": info.get("closedness"), "grip_strength": info.get("grip_strength"),
                "force_scale": info.get("force_scale"),
                "left_force": info.get("left_force"), "right_force": info.get("right_force"),
                "d_xy": info.get("d_xy"), "d_z": info.get("d_z"),
                "translation_norm": info.get("translation_norm"), "done": info.get("done"),
            }
            all_rows.append(row)
            print(f"{step:>4} {row['closedness']:>10.4f} {row['grip_strength']:>9.4f} "
                  f"{row['force_scale']:>11.4f} {row['left_force']:>7.4f} {row['right_force']:>7.4f} "
                  f"{row['d_xy']:>7.4f} {row['d_z']:>7.4f} {row['translation_norm']:>7.4f}")
            if info.get("done") or terminated or truncated:
                print(f"  -> done={info.get('done')} terminated={terminated} truncated={truncated}")
                break
        env.close()

    max_closedness = max(r["closedness"] for r in all_rows)
    max_grip_strength = max(r["grip_strength"] for r in all_rows)
    min_d_xy_at_max_closedness = min(
        r["d_xy"] for r in all_rows if r["closedness"] == max_closedness
    )
    print(f"\nmax closedness reached across all seeds: {max_closedness:.4f} "
          f"(threshold=0.8)  max grip_strength: {max_grip_strength:.4f}N")

    return {
        "status": "PASS", "width": width, "n_seeds": n_seeds,
        "max_closedness": max_closedness, "max_grip_strength": max_grip_strength,
        "rows": all_rows,
    }


@app.local_entrypoint()
def main(width: float = 0.01, n_seeds: int = 3, max_steps: int = 30):
    result = diagnose_close_gripper_small_width.remote(width=width, n_seeds=n_seeds, max_steps=max_steps)
    print(f"\nDone. max_closedness={result['max_closedness']:.4f}  "
          f"max_grip_strength={result['max_grip_strength']:.4f}N")
