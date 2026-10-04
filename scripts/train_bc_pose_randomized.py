"""
train_bc_pose_randomized.py

Retrains the flat MSE BC policy on the joint-angle-randomized demo pool
(demonstrations/sb3_teacher_full_task_joint_randomized_v2.pkl — 2000
genuine demos, every episode starting from a perturbed arm pose instead
of the single fixed one every prior BC/PPO checkpoint in this project has
used; see env.setup.randomize_joint_angles and
collect_sb3_teacher_demos.py's randomize_joint_angles_prob). Motivated by
scripts/eval_stack3_cubes.py: the existing policy generalizes well to a
new TARGET height but fails hard the moment it has to pick something up
starting from wherever a prior sub-task left the arm — exactly the
state-space gap this new demo pool targets.

_v2, not the original _joint_randomized pool: that first pool's own
randomization sampled joint angles symmetrically AROUND the stretched
default pose, which (confirmed via direct video review, "i feel like
they all start with the arm stretched out") produced starting poses that
all still looked basically fully-extended, just redirected to a
different compass direction — not genuinely varied joint bending. Fixed
in env.setup.randomize_joint_angles (2026-10-03) by excluding a central
band around the default instead of just widening a symmetric scale; see
that function's own docstring and the "Pose Generalization & Stacking"
artifact's 10-frame diversity comparison for the before/after.

Deliberately a STANDALONE script, not a call into run_bc_scaling_cell
(training/run_bc_scaling_experiment.py) — that function's checkpoint path
is hardcoded purely from (policy_type, n_demos, seed), which would
silently collide with and overwrite the ALREADY-EXISTING mse_n2000_seed*
checkpoints every PPO run in this project warm-starts from (confirmed
close call 2026-10-03 — a run_bc_scaling_experiment.py invocation against
this new demo pool was stopped mid-flight specifically to avoid that).
Every path this script touches is new and distinct from anything else in
the project: checkpoints/pose_randomized_v2/, logs/bc_pose_randomized_v2.json
— also distinct from THIS script's own first run (checkpoints/
pose_randomized/, trained on the flawed v1 demo pool, kept as-is rather
than overwritten, so the two remain directly comparable).

Same MultiHeadBCTrainer/MultiHeadBCConfig (policy_type="mse") and the
same held-out eval convention (seed=100_000+ep, two-finger-contact +
lift>2cm genuine-grasp verification) as run_bc_scaling_cell, so results
are directly comparable to the existing scaling-sweep numbers already in
the BC Architecture Scaling artifact — just saved somewhere that can
never collide with them.

Run with:
    modal run --detach scripts/train_bc_pose_randomized.py
    modal run --detach scripts/train_bc_pose_randomized.py --n-seeds 3
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR

DATASET_SIZES = [100, 500, 1000, 2000]   # 2026-10-03: extended from a single
                                  # n_demos=2000-only script to a full size sweep
                                  # (same sizes as the original run_bc_scaling_
                                  # experiment.py comparison), still entirely via
                                  # this script's own safe, distinct checkpoint
                                  # path prefix — see module docstring.


@app.function(image=rl_image, gpu=None, cpu=2.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=1800)
def train_one_seed(
    seed: int,
    n_demos: int = 2000,
    n_epochs: int = 30,
    n_eval_episodes: int = 30,
    max_steps: int = 100,
    lr: float = 1e-3,
    lift_threshold: float = 0.02,
    block_resting_z: float = 0.425,
    demos_path: str = "demonstrations/sb3_teacher_full_task_joint_randomized_v2.pkl",
    pose_exclude_band: float = 0.35,
    pose_max_frac: float = 0.85,
) -> dict:
    import os
    import pickle
    import numpy as np
    import torch

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env, init_random_episode, grip_contact_forces, randomize_joint_angles
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

    cfg = MultiHeadBCConfig(obs_dim=FLAT_OBS_DIM, policy_type="mse", lr=lr, n_epochs=n_epochs)
    trainer = MultiHeadBCTrainer(cfg)
    fit_result = trainer.fit(demos)
    final_loss = fit_result["epoch_losses"][-1]

    # New, distinct path — see module docstring for why this is never
    # allowed to collide with checkpoints/bc_scaling/*.
    ckpt_path = f"checkpoints/pose_randomized_v2/mse_n{n_demos}_seed{seed}.pt"
    full_ckpt_path = os.path.join(MODEL_CACHE_DIR, ckpt_path)
    trainer.save_checkpoint(full_ckpt_path)
    model_volume.commit()

    env = gym.make("FetchPickAndPlace-v3", max_episode_steps=max_steps)
    setup_env(env)

    n_genuine = 0
    n_is_success_raw = 0
    for ep in range(n_eval_episodes):
        eval_rng = np.random.default_rng(100_000 + ep)   # same fixed held-out eval
                                  # seeds as run_bc_scaling_cell — comparable
                                  # against the existing scaling-sweep numbers.
        env.reset(seed=100_000 + ep)
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

    # SECOND eval, pose-randomized (genuinely varied starting arm pose) —
    # added 2026-10-03 after the fixed-pose number above turned out
    # seriously misleading on its own for these checkpoints: the n=2000
    # pilot run scored only 6.7-16.7% here (looked like a regression vs.
    # the original fixed-pose-trained checkpoint's 46.3%) but 0.0% on
    # THIS pose-randomized eval for the original checkpoint vs. 23.3-36.7%
    # for the new ones — the fixed-pose number alone was actively
    # misleading about which checkpoint actually generalizes. Reporting
    # both from the start for the rest of this sweep rather than risking
    # the same confusion at 4x the scale.
    env2 = gym.make("FetchPickAndPlace-v3", max_episode_steps=max_steps)
    setup_env(env2)
    n_genuine_pr = 0
    n_is_success_raw_pr = 0
    n_pose_setup_failed = 0
    for ep in range(n_eval_episodes):
        eval_rng = np.random.default_rng(200_000 + ep)   # distinct range from the
                                  # fixed-pose eval above — genuinely different episodes.
        env2.reset(seed=200_000 + ep)
        obs, pose_ok = randomize_joint_angles(env2, eval_rng, exclude_band=pose_exclude_band, max_frac=pose_max_frac)
        if not pose_ok:
            n_pose_setup_failed += 1
            continue
        obs, setup_ok = init_random_episode(env2, eval_rng)
        if not setup_ok:
            continue

        hidden_state = None
        success = False
        ever_lifted_and_gripped = False
        for _ in range(max_steps):
            flat_obs = build_flat_observation(obs["observation"], obs["achieved_goal"], obs["desired_goal"])
            action, hidden_state = trainer.actor.act(flat_obs, hidden_state, deterministic=True)
            obs, reward, terminated, truncated, info = env2.step(action)

            height_above_resting = float(obs["achieved_goal"][2]) - block_resting_z
            forces = grip_contact_forces(env2)
            if min(forces["left"], forces["right"]) > 0.0 and height_above_resting > lift_threshold:
                ever_lifted_and_gripped = True
            if info.get("is_success", False):
                success = True
            if terminated or truncated:
                break

        if success:
            n_is_success_raw_pr += 1
        if success and ever_lifted_and_gripped:
            n_genuine_pr += 1
    env2.close()
    completion_rate_pose_randomized = n_genuine_pr / n_eval_episodes if n_eval_episodes else 0.0

    print(f"[pose_randomized_v2 mse n_demos={n_demos} seed={seed}] final_loss={final_loss:.4f}  "
          f"completion_rate(fixed-pose)={n_genuine}/{n_eval_episodes} ({completion_rate:.1%})  "
          f"completion_rate(pose-randomized)={n_genuine_pr}/{n_eval_episodes} ({completion_rate_pose_randomized:.1%})  "
          f"raw_is_success={n_is_success_raw}/{n_eval_episodes}", flush=True)

    return {
        "policy_type": "mse", "n_demos": n_demos, "seed": seed,
        "final_loss": final_loss, "n_eval_episodes": n_eval_episodes,
        "n_genuine": n_genuine, "n_is_success_raw": n_is_success_raw,
        "completion_rate": completion_rate, "ckpt_path": ckpt_path,
        "n_genuine_pose_randomized": n_genuine_pr,
        "n_is_success_raw_pose_randomized": n_is_success_raw_pr,
        "completion_rate_pose_randomized": completion_rate_pose_randomized,
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
def main(n_seeds: int = 3, seed_start: int = 0, n_epochs: int = 30, n_eval_episodes: int = 30,
          dataset_sizes: str = "100,500,1000,2000",
          out_path: str = "logs/bc_pose_randomized_v2.json"):
    """
    seed_start lets this be split into rounds under a container-concurrency
    cap, same convention as run_bc_scaling_experiment.py's own seed_start —
    results merge by (n_demos, seed) key below, so running this twice with
    disjoint seed ranges produces the same final merged set as one big run.
    """
    import json
    import os
    import statistics

    sizes = [int(s.strip()) for s in dataset_sizes.split(",") if s.strip()]
    cells = [(seed, n_demos) for n_demos in sizes for seed in range(seed_start, seed_start + n_seeds)]
    print(f"Running {len(cells)} cells ({len(sizes)} dataset sizes x {n_seeds} seeds)...")

    results = list(train_one_seed.starmap(
        [(seed, n_demos, n_epochs, n_eval_episodes) for seed, n_demos in cells]
    ))

    local_out = os.path.join(os.path.dirname(__file__), "..", "artifacts", "bc_pose_randomized_v2.json")
    existing = []
    if os.path.exists(local_out):
        with open(local_out) as f:
            existing = json.load(f)
    new_keys = {(r["n_demos"], r["seed"]) for r in results}
    merged = [r for r in existing if (r["n_demos"], r["seed"]) not in new_keys] + results

    os.makedirs(os.path.dirname(local_out), exist_ok=True)
    with open(local_out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nSaved {len(merged)} total results ({len(results)} new/updated) -> {local_out}")
    _save_results.remote(merged, out_path)

    print("\nSummary (fixed-pose completion_rate / pose-randomized completion_rate):")
    for n_demos in sizes:
        cell_results = [r for r in merged if r["n_demos"] == n_demos]
        fixed_rates = [r["completion_rate"] for r in cell_results]
        pr_rates = [r["completion_rate_pose_randomized"] for r in cell_results]
        if not cell_results:
            continue
        print(f"  n_demos={n_demos:4d}: fixed-pose mean={statistics.mean(fixed_rates):.1%}  "
              f"pose-randomized mean={statistics.mean(pr_rates):.1%}  "
              f"(n_seeds={len(cell_results)})")
