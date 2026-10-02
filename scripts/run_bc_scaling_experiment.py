"""
run_bc_scaling_experiment.py

BC architecture scaling comparison: genuine-grasp completion_rate vs.
dataset size (100/500/1000 demonstrations), for 4 action-distribution
heads (mse, mdn, autoregressive, cvae — training/flat_bc_multi_head.py +
policy/flat_bc_heads.py), x3 seeds each for error bars (36 cells total).

Each cell: subsamples n_demos from the full 1000-genuine-demo pool
(demonstrations/sb3_teacher_full_task.pkl) using `seed` (same seed drives
BOTH which demos get subsampled AND model init/training stochasticity —
deliberate: real-world variance from "collect N demos and train" includes
which N you happened to get, not just training noise alone), trains via
MultiHeadBCTrainer, then evaluates genuine completion_rate with the SAME
verification train_flat_task_bc.py's eval uses (two-finger contact + block
lifted >2cm, not just raw is_success — root-caused 2026-10-02, see that
script's docstring).

Runs all 36 cells in parallel via Modal's .starmap() — each cell is
independent (own subsample, own fresh model), so this is embarrassingly
parallel; no reason to run them serially.

Run with:
    modal run --detach scripts/run_bc_scaling_experiment.py
    modal run --detach scripts/run_bc_scaling_experiment.py --n-seeds 3 --n-epochs 30
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR

POLICY_TYPES = ["mse", "mdn", "autoregressive", "factored", "cvae"]
DATASET_SIZES = [100, 500, 1000, 2000]


@app.function(image=rl_image, gpu=None, cpu=2.0,
              volumes={MODEL_CACHE_DIR: model_volume}, timeout=1800)
def run_bc_scaling_cell(
    policy_type: str,
    n_demos: int,
    seed: int,
    n_epochs: int = 30,
    n_eval_episodes: int = 30,
    max_steps: int = 100,
    lr: float = 1e-3,
    lift_threshold: float = 0.02,
    block_resting_z: float = 0.425,
    demos_path: str = "demonstrations/sb3_teacher_full_task.pkl",
    mdn_decode_mode: str = "argmax",   # "argmax" | "weighted_mean" — only
                                  # read when policy_type=="mdn", see
                                  # MDNPolicy's own docstring.
    mdn_smoothness_weight: float = 0.0,
    mdn_components: int = 5,
) -> dict:
    import os
    import pickle
    import numpy as np
    import torch

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env, init_random_episode, grip_contact_forces
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

    cfg = MultiHeadBCConfig(obs_dim=FLAT_OBS_DIM, policy_type=policy_type, lr=lr, n_epochs=n_epochs,
                             mdn_decode_mode=mdn_decode_mode, mdn_smoothness_weight=mdn_smoothness_weight,
                             mdn_components=mdn_components)
    trainer = MultiHeadBCTrainer(cfg)
    fit_result = trainer.fit(demos)
    final_loss = fit_result["epoch_losses"][-1]

    # Checkpointed so any cell can be inspected/videoed later without
    # retraining (added 2026-10-02 — every earlier MDN investigation needed
    # a fresh ad-hoc retrain first since nothing was saved before this).
    ckpt_path = (f"checkpoints/bc_scaling/{policy_type}_n{n_demos}_seed{seed}"
                 f"_smw{mdn_smoothness_weight}_k{mdn_components}.pt")
    full_ckpt_path = os.path.join(MODEL_CACHE_DIR, ckpt_path)
    trainer.save_checkpoint(full_ckpt_path)
    model_volume.commit()

    # Real eval rollout, same init_random_episode setup + genuine-grasp
    # verification as train_flat_task_bc.py.
    env = gym.make("FetchPickAndPlace-v3", max_episode_steps=max_steps)
    setup_env(env)

    n_genuine = 0
    n_is_success_raw = 0
    for ep in range(n_eval_episodes):
        eval_rng = np.random.default_rng(100_000 + ep)   # fixed held-out eval
                                  # seeds, same across every cell — comparable
                                  # across policy_type/n_demos/seed, not
                                  # confounded by which eval episodes ran.
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

    print(f"[{policy_type:14s} n_demos={n_demos:4d} seed={seed} "
          f"smooth_w={mdn_smoothness_weight} K={mdn_components}] "
          f"final_loss={final_loss:.4f}  completion_rate={n_genuine}/{n_eval_episodes} "
          f"({completion_rate:.1%})  raw_is_success={n_is_success_raw}/{n_eval_episodes}", flush=True)

    return {
        "policy_type": policy_type, "n_demos": n_demos, "seed": seed,
        "final_loss": final_loss, "epoch_losses": fit_result["epoch_losses"],
        "n_eval_episodes": n_eval_episodes, "n_genuine": n_genuine,
        "n_is_success_raw": n_is_success_raw, "completion_rate": completion_rate,
        "mdn_decode_mode": mdn_decode_mode if policy_type == "mdn" else None,
        "mdn_smoothness_weight": mdn_smoothness_weight if policy_type == "mdn" else None,
        "mdn_components": mdn_components if policy_type == "mdn" else None,
        "ckpt_path": ckpt_path,
    }


@app.function(image=rl_image, gpu=None, volumes={MODEL_CACHE_DIR: model_volume}, timeout=120)
def _save_results_to_volume(results: list, out_path: str) -> None:
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
         out_path: str = "logs/bc_scaling_experiment.json",
         policy_types: str = "mse,mdn,autoregressive,factored,cvae",   # subset
                                  # for a targeted rerun, e.g. --policy-types mdn
                                  # after a bug fix, without re-running the
                                  # other architectures' already-good cells
         demos_path: str = "demonstrations/sb3_teacher_full_task.pkl",
         mdn_decode_mode: str = "argmax"):
    import json
    import os

    # seed_start lets a run be split into multiple rounds against a fixed
    # container-concurrency cap (e.g. 2 rounds of 100 cells instead of one
    # 200-cell submission) — results merge by (policy_type, n_demos, seed)
    # below regardless of which round produced them, so running this twice
    # with disjoint seed ranges (e.g. --n-seeds 5 then --seed-start 5
    # --n-seeds 5) produces the same final merged set as one big run would.
    selected_policy_types = [p.strip() for p in policy_types.split(",") if p.strip()]
    cells = [
        (policy_type, n_demos, seed)
        for policy_type in selected_policy_types
        for n_demos in DATASET_SIZES
        for seed in range(seed_start, seed_start + n_seeds)
    ]
    print(f"Running {len(cells)} cells ({len(selected_policy_types)} policy types x "
          f"{len(DATASET_SIZES)} dataset sizes x {n_seeds} seeds) in parallel, "
          f"demos_path={demos_path}, mdn_decode_mode={mdn_decode_mode}...")

    results = list(run_bc_scaling_cell.starmap(
        [(pt, nd, s, n_epochs, n_eval_episodes, 100, 1e-3, 0.02, 0.425, demos_path, mdn_decode_mode)
         for pt, nd, s in cells]
    ))

    # Merge into any existing results rather than overwrite — a targeted
    # rerun (e.g. --policy-types mdn after a bug fix) must not discard the
    # other architectures' already-good cells. A cell is identified by
    # (policy_type, n_demos, seed); a rerun replaces matching cells, leaves
    # the rest untouched.
    local_out = os.path.join(os.path.dirname(__file__), "..", "artifacts", "bc_scaling_experiment.json")
    existing = []
    if os.path.exists(local_out):
        with open(local_out) as f:
            existing = json.load(f)
    new_keys = {(r["policy_type"], r["n_demos"], r["seed"]) for r in results}
    merged = [r for r in existing if (r["policy_type"], r["n_demos"], r["seed"]) not in new_keys] + results

    os.makedirs(os.path.dirname(local_out), exist_ok=True)
    with open(local_out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nSaved {len(merged)} total cell results ({len(results)} new/updated) -> {local_out}")

    _save_results_to_volume.remote(merged, out_path)

    results = merged   # summary below reports the full merged set
    print("\nSummary (mean completion_rate across seeds):")
    for policy_type in POLICY_TYPES:
        for n_demos in DATASET_SIZES:
            cell_results = [r for r in results if r["policy_type"] == policy_type and r["n_demos"] == n_demos]
            rates = [r["completion_rate"] for r in cell_results]
            mean_rate = sum(rates) / len(rates) if rates else float("nan")
            print(f"  {policy_type:14s} n_demos={n_demos:4d}: mean completion_rate={mean_rate:.1%} "
                  f"(seeds: {[f'{r:.0%}' for r in rates]})")
