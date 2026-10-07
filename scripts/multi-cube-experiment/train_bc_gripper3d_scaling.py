"""
train_bc_gripper3d_scaling.py

BC data-scaling sweep (completion_rate vs. n_demos) on the gripper3d pose
scheme's demo pool, for BOTH policy_type="mse" and "cvae" (user's own
call, 2026-10-04) -- same methodology as train_bc_pose_randomized.py
(itself modeled on the original run_bc_scaling_experiment.py sweep), just
on demonstrations/sb3_teacher_full_task_gripper3d.pkl and sized by
POLICY_TYPE too, not just n_demos.

Own isolated checkpoint/log paths throughout (checkpoints/gripper3d_v1/
{policy_type}_n{n}_seed{s}.pt -- note seed0 of n=2000 for each policy_type
already exists from train_bc_gripper3d.py's own earlier run; this sweep
reuses that exact path convention so seed0/n2000 cells just overwrite
with the SAME result, not a collision with anything else).

Run with:
    modal run --detach scripts/train_bc_gripper3d_scaling.py
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR

DATASET_SIZES = [100, 500, 1000, 2000]


@app.function(image=rl_image, gpu=None, cpu=2.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=1800)
def train_one_cell(
    seed: int,
    n_demos: int,
    policy_type: str,
    n_epochs: int = 20,
    n_eval_episodes: int = 30,
    max_steps: int = 100,
    lr: float = 1e-3,
    lift_threshold: float = 0.02,
    block_resting_z: float = 0.425,
    demos_path: str = "demonstrations/sb3_teacher_full_task_gripper3d.pkl",
) -> dict:
    import os
    import pickle
    import numpy as np
    import torch

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env, init_random_episode, grip_contact_forces, randomize_gripper_start_3d
    from think_then_act.training.subgoal_features import build_flat_observation, FLAT_OBS_DIM
    from think_then_act.training.flat_bc_multi_head import MultiHeadBCConfig, MultiHeadBCTrainer

    torch.manual_seed(seed)
    rng_subsample = np.random.default_rng(seed)

    full_demos_path = os.path.join(MODEL_CACHE_DIR, demos_path)
    with open(full_demos_path, "rb") as f:
        demo_data = pickle.load(f)
    all_demos = demo_data["demonstrations"]
    if n_demos > len(all_demos):
        raise ValueError(f"n_demos={n_demos} > available {len(all_demos)} demonstrations")
    subsample_idx = rng_subsample.choice(len(all_demos), size=n_demos, replace=False)
    demos = [all_demos[i] for i in subsample_idx]

    cfg = MultiHeadBCConfig(obs_dim=FLAT_OBS_DIM, policy_type=policy_type, lr=lr, n_epochs=n_epochs)
    trainer = MultiHeadBCTrainer(cfg)
    fit_result = trainer.fit(demos)
    final_loss = fit_result["epoch_losses"][-1]

    ckpt_path = f"checkpoints/gripper3d_v1/{policy_type}_n{n_demos}_seed{seed}.pt"
    full_ckpt_path = os.path.join(MODEL_CACHE_DIR, ckpt_path)
    trainer.save_checkpoint(full_ckpt_path)
    model_volume.commit()

    # Pose-randomized eval ONLY (gripper3d scheme) -- a fixed-pose number
    # isn't a meaningful comparison point here the way it was for the
    # joint_angles sweep (that comparison existed because fixed-pose was
    # the OLD, pre-randomization baseline; there's no equivalent "old
    # baseline" for this brand-new scheme worth reporting alongside).
    env = gym.make("FetchPickAndPlace-v3", max_episode_steps=max_steps)
    setup_env(env)
    n_genuine = 0
    n_is_success_raw = 0
    n_pose_setup_failed = 0
    for ep in range(n_eval_episodes):
        eval_rng = np.random.default_rng(200_000 + ep)
        reset_obs, _ = env.reset(seed=200_000 + ep)
        obs, pose_ok, _ = randomize_gripper_start_3d(env, eval_rng, reset_obs)
        if not pose_ok:
            n_pose_setup_failed += 1
            continue
        obs, setup_ok = init_random_episode(env, eval_rng)
        if not setup_ok:
            continue

        hidden_state = None
        success = False
        ever_lifted_and_gripped = False
        for _ in range(max_steps):
            flat_obs = build_flat_observation(obs["observation"], obs["achieved_goal"], obs["desired_goal"])
            action, hidden_state = trainer.actor.act(flat_obs, hidden_state, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            height_above_resting = float(obs["achieved_goal"][2]) - block_resting_z
            forces = grip_contact_forces(env)
            if min(forces["left"], forces["right"]) > 0.0 and height_above_resting > lift_threshold:
                ever_lifted_and_gripped = True
            if info.get("is_success", False):
                success = True
            if terminated or truncated:
                break
        if success:
            n_is_success_raw += 1
        if success and ever_lifted_and_gripped:
            n_genuine += 1
    env.close()
    completion_rate = n_genuine / n_eval_episodes if n_eval_episodes else 0.0

    print(f"[gripper3d {policy_type} n_demos={n_demos} seed={seed}] final_loss={final_loss:.4f}  "
          f"completion_rate={n_genuine}/{n_eval_episodes} ({completion_rate:.1%})", flush=True)

    return {
        "policy_type": policy_type, "n_demos": n_demos, "seed": seed,
        "final_loss": final_loss, "n_eval_episodes": n_eval_episodes,
        "n_genuine": n_genuine, "n_is_success_raw": n_is_success_raw,
        "completion_rate": completion_rate, "ckpt_path": ckpt_path,
        "n_pose_setup_failed": n_pose_setup_failed,
    }


@app.function(image=rl_image, gpu=None, volumes={MODEL_CACHE_DIR: model_volume}, timeout=120)
def _save_results(results: list, out_path: str) -> None:
    import json
    import os
    full_path = os.path.join(MODEL_CACHE_DIR, out_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2)
    model_volume.commit()
    print(f"  Saved results -> {full_path}")


@app.local_entrypoint()
def main(n_seeds: int = 2, seed_start: int = 0, n_epochs: int = 20, n_eval_episodes: int = 30,
          dataset_sizes: str = "100,500,1000,2000", policy_types: str = "mse,cvae",
          out_path: str = "logs/bc_gripper3d_scaling.json"):
    import json
    import os
    import statistics

    sizes = [int(s.strip()) for s in dataset_sizes.split(",") if s.strip()]
    types = [t.strip() for t in policy_types.split(",") if t.strip()]
    cells = [(seed, n_demos, pt) for pt in types for n_demos in sizes
             for seed in range(seed_start, seed_start + n_seeds)]
    print(f"Running {len(cells)} cells ({len(types)} policy types x {len(sizes)} sizes x {n_seeds} seeds)...")

    results = list(train_one_cell.starmap(
        [(seed, n_demos, pt, n_epochs, n_eval_episodes) for seed, n_demos, pt in cells]
    ))

    local_out = os.path.join(os.path.dirname(__file__), "..", "artifacts", "bc_gripper3d_scaling.json")
    existing = []
    if os.path.exists(local_out):
        with open(local_out) as f:
            existing = json.load(f)
    new_keys = {(r["policy_type"], r["n_demos"], r["seed"]) for r in results}
    merged = [r for r in existing if (r["policy_type"], r["n_demos"], r["seed"]) not in new_keys] + results

    os.makedirs(os.path.dirname(local_out), exist_ok=True)
    with open(local_out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nSaved {len(merged)} total results ({len(results)} new/updated) -> {local_out}")
    _save_results.remote(merged, out_path)

    print("\nSummary (completion_rate mean +/- stdev):")
    for pt in types:
        for n_demos in sizes:
            cell_results = [r for r in merged if r["policy_type"] == pt and r["n_demos"] == n_demos]
            if not cell_results:
                continue
            rates = [r["completion_rate"] for r in cell_results]
            mean = statistics.mean(rates)
            stdev = statistics.stdev(rates) if len(rates) > 1 else 0.0
            print(f"  {pt:5s} n_demos={n_demos:4d}: mean={mean:.1%}  stdev={stdev:.1%}  (n_seeds={len(cell_results)})")
