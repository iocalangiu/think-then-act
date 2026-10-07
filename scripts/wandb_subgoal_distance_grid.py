"""
wandb_subgoal_distance_grid.py

Per-step W&B telemetry for each trained low-level subgoal controller, swept
across a block-WIDTH grid (length/height held fixed — a controlled 1-D
sweep, not a full 3-axis grid) x several seeds. Built to visualize how
distance-to-block (align_xy/descend/close_gripper/lift) or
distance-to-target (move_to_target) and per-step reward behave as block
size moves away from the original 5cm cube each policy was originally
calibrated against.

use_pose_model (default True): loads block_pose_predictor.pt/
collision_predictor.pt the same way record_full_rollout.py does and wires
them into SubgoalConditionedEnv, so the policy's observation gets the
learned position estimate instead of privileged achieved_goal — the real
deployment condition, not just a physics/policy-behavior diagnostic.
Block-size noise is already active regardless (see
subgoal_env.py's _perceived_block_dims, injected unconditionally every
episode) — this flag only controls the position-perception half. Pass
--no-use-pose-model to fall back to the original ground-truth-only sweep
(reward/done always stays ground truth either way — see
block_pose_predictor.py's docstring for that distinction).

When the pose model is active, also logs `{subgoal}/pose_error` — the
per-step norm between true and perceived block position (from
SubgoalConditionedEnv's info["block_pos"]/info["perceived_block_pos"]) — so
a width-correlated localization failure (e.g. the pose model, trained
before block-size randomization existed, losing track of small/thin
blocks) shows up directly on the dashboard rather than only as a
downstream reward/distance symptom.

No rendering, no video: `render_mode="rgb_array"` is still set at env
construction (ObservationHarness requires it — see its docstring), but
rendering in Gymnasium only happens on an explicit `.render()`/`.last_frame()`
call, never automatically inside `step()` — this script never calls either,
so no frames are produced or saved, not just "produced but discarded".

Distance definitions (computed directly from subgoal_env.py's `info` dict,
not any one subgoal's own reward breakdown, so they're uniform across all
five subgoals):
    align_xy / descend / close_gripper / lift:
        d_xy, d_xyz  = gripper -> BLOCK  (info["grip_pos"] vs info["block_pos"])
    move_to_target:
        d_xy, d_xyz  = BLOCK -> target   (info["block_pos"] vs info["target_pos"])
        matches reward_move_to_target's own d_block_target exactly (its d_xyz
        IS that reward's -reward each step).

W&B layout: one run, metrics logged under a `{subgoal}/...` prefix each step
with a monotonically increasing global step — W&B's workspace groups panels
by prefix automatically, giving one panel group (reward/d_xy/d_xyz) per
subgoal. `width`/`seed`/`episode_step` are also logged each point so you can
filter/color by them in the UI (e.g. group by `width` to see the grid-search
effect directly, or by `seed` to see variance at a fixed size).

Primary output path (no account/cost needed): every run, regardless of
--wandb-project, also returns its full per-step records to the local
entrypoint, which dumps them to a local JSON file (--out-path, default
artifacts/subgoal_distance_grid.json). That JSON is what
artifacts/subgoal_distance_grid_dashboard.html (a Claude-published,
redeployable-in-place artifact — see project memory) reads to render charts.
--wandb-project is now optional extra logging, not required — leave it
unset (the default) to skip W&B entirely; useful if that account's on a
trial/paid tier you don't want to touch (the wandb-secret account's trial
expired 2026-08-24). W&B, if used, still requires:
    modal secret create wandb-secret WANDB_API_KEY=<your-key>

Run with:
    modal run scripts/wandb_subgoal_distance_grid.py   # writes the local JSON, no W&B, no cost
    modal run scripts/wandb_subgoal_distance_grid.py --n-seeds 3   # quick smoke test
    modal run scripts/wandb_subgoal_distance_grid.py --no-use-pose-model   # ground-truth-only sweep
    modal run scripts/wandb_subgoal_distance_grid.py --out-path artifacts/my_run.json
    modal run scripts/wandb_subgoal_distance_grid.py --wandb-project rl-harness-robotics   # ALSO log to W&B (needs a live account)
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(
    image=rl_image,
    gpu=None,
    cpu=8.0,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=3600 * 2,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def wandb_subgoal_distance_grid(
    subgoals: str = "align_xy,descend,close_gripper,lift,move_to_target",
    width_min: float = 0.01, width_max: float = 0.07, width_step: float = 0.01,
    n_seeds: int = 10,
    max_steps: int = 30,
    length: float = 0.05, height: float = 0.05,   # pinned — only width varies across the grid
    algo: str = "ppo", use_best: bool = True,
    use_pose_model: bool = True,   # False: force ground-truth achieved_goal even if a
                                # block_pose_predictor.pt checkpoint exists — see this
                                # function's own docstring for the A/B use case.
    wandb_project: str = "",   # "" (default) = no W&B, dry run with stdout summary only.
                                # Requires a `wandb-secret` Modal secret with WANDB_API_KEY set.
) -> dict:
    import os
    import time
    import numpy as np
    import torch

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.perception.block_pose_predictor import BlockPosePredictor
    from think_then_act.perception.collision_predictor import CollisionPredictor
    from think_then_act.policy.subgoal_policy import SubgoalGaussianPolicy
    from think_then_act.reward.subgoal_reward import SUBGOAL_LABELS
    from think_then_act.training.checkpoints import resolve_subgoal_checkpoint
    from think_then_act.training.subgoal_env import SubgoalConditionedEnv
    from think_then_act.training.subgoal_features import obs_dim_for_subgoal

    requested = [s.strip() for s in subgoals.split(",") if s.strip()]
    unknown = set(requested) - set(SUBGOAL_LABELS)
    if unknown:
        raise ValueError(f"Unknown subgoal(s) {unknown}; must be a subset of {SUBGOAL_LABELS}")

    widths = [round(w, 4) for w in np.arange(width_min, width_max + 1e-9, width_step).tolist()]

    print("\n" + "=" * 60)
    print("  SUBGOAL DISTANCE/REWARD GRID  (W&B telemetry, no recordings)")
    print(f"  subgoals={requested}  widths={widths}  n_seeds={n_seeds}  "
          f"length={length}  height={height}  use_pose_model={use_pose_model}")
    print("=" * 60)

    ckpt_dir = os.path.join(MODEL_CACHE_DIR, "checkpoints")

    # Same load pattern as record_full_rollout.py — None means "no checkpoint
    # found" (collision_model) or "disabled/missing" (pose_model), and
    # SubgoalConditionedEnv already treats None as ground-truth/no-op, so
    # this degrades gracefully rather than erroring if a checkpoint is
    # absent.
    collision_model = None
    collision_ckpt = os.path.join(ckpt_dir, "collision_predictor.pt")
    if os.path.exists(collision_ckpt):
        collision_model = CollisionPredictor()
        collision_model.load_state_dict(torch.load(collision_ckpt, map_location="cpu"))
        collision_model.eval()

    pose_model = None
    pose_ckpt = os.path.join(ckpt_dir, "block_pose_predictor.pt")
    if use_pose_model and os.path.exists(pose_ckpt):
        pose_model = BlockPosePredictor()
        pose_model.load_state_dict(torch.load(pose_ckpt, map_location="cpu"))
        pose_model.eval()
        print(f"  pose model        <- {pose_ckpt}")
    elif not use_pose_model:
        print("  pose model disabled (--no-use-pose-model) — using ground-truth achieved_goal")
    else:
        print(f"  no pose model checkpoint found at {pose_ckpt} — using ground-truth achieved_goal")

    wandb_enabled = bool(os.environ.get("WANDB_API_KEY") and wandb_project)
    if wandb_enabled:
        import wandb
        wandb.init(project=wandb_project, name=f"subgoal-distance-grid-{int(time.time())}", config={
            "subgoals": requested, "widths": widths, "n_seeds": n_seeds,
            "max_steps": max_steps, "length": length, "height": height,
            "algo": algo, "use_best": use_best,
            "use_pose_model": pose_model is not None,
            "use_collision_model": collision_model is not None,
        })
        print(f"\nW&B logging -> project: {wandb_project}")
    else:
        print("\nW&B not configured (pass --wandb-project + a wandb-secret) — "
              "running dry, stdout summary only.")

    global_step = 0
    per_subgoal_summary = {}
    # Flat per-step records, independent of whether W&B is configured — this
    # is what gets returned to the local entrypoint so it can be dumped to a
    # local JSON file and charted without needing a W&B account at all (see
    # this module's docstring: the wandb-secret account's trial expired
    # 2026-08-24, so this is now the primary output path, not a fallback).
    all_records = []

    for subgoal in requested:
        print(f"\n--- {subgoal} ---")
        ckpt_path = resolve_subgoal_checkpoint(ckpt_dir, subgoal, algo=algo, use_best=use_best)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        policy = SubgoalGaussianPolicy(obs_dim=obs_dim_for_subgoal(subgoal))
        policy.load_state_dict(ckpt["actor"] if isinstance(ckpt, dict) and "actor" in ckpt else ckpt)
        policy.eval()
        print(f"  checkpoint <- {os.path.basename(ckpt_path)}")

        episode_rewards, final_d_xyz, final_pose_errors = [], [], []
        for width in widths:
            for seed in range(n_seeds):
                base = ObservationHarness(
                    gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                              max_episode_steps=max_steps + 250)
                )
                setup_env(base)
                env = SubgoalConditionedEnv(
                    base, subgoal=subgoal, max_episode_steps=max_steps,
                    collision_model=collision_model, pose_model=pose_model,
                    randomize_block_size=True,
                    width_range=(width, width), length_range=(length, length),
                    height_range=(height, height),
                )
                rng = np.random.default_rng(seed)
                obs, info = env.reset(rng=rng)

                ep_reward = 0.0
                d_xyz = None
                ep_pose_errors = []
                episode_start_idx = len(all_records)
                episode_success = False
                for step in range(max_steps):
                    action = policy.act(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    ep_reward += float(reward)

                    grip_pos  = np.asarray(info["grip_pos"],  dtype=np.float64)
                    block_pos = np.asarray(info["block_pos"], dtype=np.float64)
                    if subgoal == "move_to_target":
                        target_pos = np.asarray(info["target_pos"], dtype=np.float64)
                        delta = block_pos - target_pos
                    else:
                        delta = grip_pos - block_pos
                    d_xy  = float(np.linalg.norm(delta[:2]))
                    d_xyz = float(np.linalg.norm(delta))

                    log_payload = {
                        f"{subgoal}/reward": float(reward),
                        f"{subgoal}/d_xy": d_xy,
                        f"{subgoal}/d_xyz": d_xyz,
                        f"{subgoal}/width": width,
                        f"{subgoal}/seed": seed,
                        f"{subgoal}/episode_step": step,
                    }
                    pose_error = None
                    if pose_model is not None:
                        # info["block_pos"] is ground truth, info["perceived_block_pos"]
                        # is what the policy's own observation actually contains (see
                        # SubgoalConditionedEnv._perceived_block_pos) — their gap is the
                        # pose model's real localization error THIS step, at THIS width.
                        perceived_pos = np.asarray(info["perceived_block_pos"], dtype=np.float64)
                        pose_error = float(np.linalg.norm(block_pos - perceived_pos))
                        ep_pose_errors.append(pose_error)
                        log_payload[f"{subgoal}/pose_error"] = pose_error

                    if wandb_enabled:
                        wandb.log(log_payload, step=global_step)
                    all_records.append({
                        "subgoal": subgoal, "width": width, "seed": seed, "episode_step": step,
                        "reward": float(reward), "d_xy": d_xy, "d_xyz": d_xyz,
                        "pose_error": pose_error,
                    })
                    global_step += 1

                    if terminated or truncated:
                        # `terminated` (not `truncated`) is SubgoalConditionedEnv.step's
                        # own `subgoal_done or terminated` — the base FetchPickAndPlace
                        # env essentially never sets terminated=True on its own (success
                        # there is communicated via info["is_success"]/reward, not this
                        # flag), so in practice this IS "did this subgoal's own `done`
                        # condition fire" — a real completion, not a timeout/truncation
                        # (drifted_too_far, step budget exhausted, etc.).
                        episode_success = bool(terminated)
                        break

                # Backfill success onto every record this episode already
                # appended (episode-level fact, but the per-step aggregation
                # downstream reads one field per row rather than needing a
                # separate episode-keyed table).
                for rec in all_records[episode_start_idx:]:
                    rec["success"] = episode_success

                env.close()
                episode_rewards.append(ep_reward)
                final_d_xyz.append(d_xyz)
                if ep_pose_errors:
                    final_pose_errors.append(float(np.mean(ep_pose_errors)))

        per_subgoal_summary[subgoal] = {
            "mean_episode_reward": round(float(np.mean(episode_rewards)), 4),
            "mean_final_d_xyz": round(float(np.mean(final_d_xyz)), 4),
            "n_episodes": len(episode_rewards),
        }
        if final_pose_errors:
            per_subgoal_summary[subgoal]["mean_pose_error"] = round(float(np.mean(final_pose_errors)), 4)
        print(f"  mean_episode_reward={per_subgoal_summary[subgoal]['mean_episode_reward']}  "
              f"mean_final_d_xyz={per_subgoal_summary[subgoal]['mean_final_d_xyz']}  "
              + (f"mean_pose_error={per_subgoal_summary[subgoal].get('mean_pose_error')}  "
                 if final_pose_errors else "")
              + f"over {per_subgoal_summary[subgoal]['n_episodes']} episodes")

    if wandb_enabled:
        wandb.finish()

    print("\n" + "=" * 60)
    print(f"  Done. total_steps_logged={global_step}")
    print("=" * 60)

    return {
        "status": "PASS", "subgoals": requested, "widths": widths, "n_seeds": n_seeds,
        "total_steps_logged": global_step, "per_subgoal_summary": per_subgoal_summary,
        "wandb_enabled": wandb_enabled, "use_pose_model": pose_model is not None,
        "records": all_records,
    }


@app.local_entrypoint()
def main(
    subgoals: str = "align_xy,descend,close_gripper,lift,move_to_target",
    width_min: float = 0.01, width_max: float = 0.07, width_step: float = 0.01,
    n_seeds: int = 10, max_steps: int = 30,
    length: float = 0.05, height: float = 0.05,
    algo: str = "ppo", use_best: bool = True, use_pose_model: bool = True,
    wandb_project: str = "",
    out_path: str = "artifacts/subgoal_distance_grid.json",   # local file — no W&B
                     # account needed; this is what the dashboard artifact reads.
):
    import json
    import os

    print(f"\nRunning subgoal distance/reward grid ({subgoals}), "
          f"width=[{width_min},{width_max}] step={width_step}, n_seeds={n_seeds}, "
          f"use_pose_model={use_pose_model}, "
          f"wandb_project={wandb_project or 'none (dry run)'}...")
    result = wandb_subgoal_distance_grid.remote(
        subgoals=subgoals, width_min=width_min, width_max=width_max, width_step=width_step,
        n_seeds=n_seeds, max_steps=max_steps, length=length, height=height,
        algo=algo, use_best=use_best, use_pose_model=use_pose_model, wandb_project=wandb_project,
    )
    print(f"\nDone. total_steps_logged={result['total_steps_logged']}")
    for subgoal, s in result["per_subgoal_summary"].items():
        pose_str = f"mean_pose_error={s['mean_pose_error']:>8}  " if "mean_pose_error" in s else ""
        print(f"  {subgoal:15s} mean_episode_reward={s['mean_episode_reward']:>10}  "
              f"mean_final_d_xyz={s['mean_final_d_xyz']:>8}  {pose_str}n_episodes={s['n_episodes']}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "subgoals": result["subgoals"], "widths": result["widths"], "n_seeds": result["n_seeds"],
            "use_pose_model": result["use_pose_model"], "per_subgoal_summary": result["per_subgoal_summary"],
            "records": result["records"],
        }, f)
    print(f"\nWrote {len(result['records'])} records -> {out_path}")
