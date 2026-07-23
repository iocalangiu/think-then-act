"""
record_subgoal_video.py

Visual sanity check for one low-level subgoal controller (training/low_level_grpo.py):
records a "before" video (freshly-initialized, untrained policy) and an
"after" video (a checkpoint saved by train_low_level.py) on the SAME seeded
scene, so you can eyeball what the policy is actually doing instead of only
reading loss/reward/entropy numbers — same motivation as run_episode.py's
sanity check for the VLM policy.

train_low_level.py checkpoints every `checkpoint_every` iterations (default
50) to checkpoints/low_level_{subgoal}_iter{N}.pt, plus a final
checkpoints/low_level_{subgoal}.pt when the whole run for that subgoal
finishes. If neither exists yet for the requested subgoal, this errors out
with the checkpoints dir listing rather than silently doing nothing.

Actions are deterministic (tanh(mean), no sampling noise) by default — the
question this answers is "what has the policy learned as its best guess",
not "what does its current exploration noise look like". Pass --stochastic
to sample instead.

No GPU needed — same as train_low_level.py.

Run with:
    modal run scripts/record_subgoal_video.py --subgoal align_xy
    modal run scripts/record_subgoal_video.py --subgoal align_xy --ckpt-iter 50
    modal run scripts/record_subgoal_video.py --subgoal descend --seed 3 --stochastic
    modal run scripts/record_subgoal_video.py --subgoal align_xy --algo ppo --use-best
                                                              # PPO's best-by-completion-rate
                                                              # checkpoint (low_level_{subgoal}
                                                              # _ppo_best.pt) rather than the
                                                              # latest/final one — matters
                                                              # because training can regress
                                                              # after peaking (seen with GRPO)

Outputs (on the volume, under /model-cache/subgoal_videos/):
    {subgoal}_before.mp4   — random-init policy
    {subgoal}_after_iter{N}.mp4 — trained checkpoint
    {subgoal}_after_iter{N}_trajectory.json — per-step block_pos/grip_pos/
        d_grip_block/action for the AFTER rollout only, so you can see
        exactly which step the block starts moving and whether it
        correlates with the action's dx/dy/dz (also printed to stdout,
        with a "<- block moved" marker on steps where block_pos changed).

Download with:
    modal volume get rl-harness-model-cache subgoal_videos/ ./artifacts/subgoal_videos/
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(
    image=rl_image,
    gpu=None,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=600,
)
def record_subgoal_video(
    subgoal: str = "align_xy",
    seed: int = 0,
    max_steps: int = 30,
    ckpt_iter: int = 0,     # 0 = auto-pick latest available checkpoint for this subgoal
    stochastic: bool = False,
    algo: str = "grpo",     # "grpo" or "ppo" — determines checkpoint filename pattern
                            # (low_level_{subgoal}.pt vs low_level_{subgoal}_ppo.pt etc.)
    use_best: bool = False, # True: use low_level_{subgoal}[_ppo]_best.pt directly (the
                            # highest-completion_rate checkpoint tracked during training),
                            # bypassing ckpt_iter/latest-iter entirely. Matters because
                            # training can regress after peaking (observed with GRPO).
) -> dict:
    import os, re, json
    import numpy as np
    import torch

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env, save_video
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.perception.collision_predictor import CollisionPredictor
    from think_then_act.policy.subgoal_policy import SubgoalGaussianPolicy
    from think_then_act.reward.subgoal_reward import SUBGOAL_LABELS
    from think_then_act.training.checkpoints import resolve_subgoal_checkpoint
    from think_then_act.training.subgoal_env import SubgoalConditionedEnv
    from think_then_act.training.subgoal_features import SUBGOAL_OBS_DIM

    if subgoal not in SUBGOAL_LABELS:
        raise ValueError(f"Unknown subgoal {subgoal!r}; must be one of {SUBGOAL_LABELS}")

    print("\n" + "=" * 60)
    print(f"  SUBGOAL VIDEO — {subgoal}  seed={seed}  "
          f"{'stochastic' if stochastic else 'deterministic'}")
    print("=" * 60)

    if algo not in ("grpo", "ppo"):
        raise ValueError(f"algo must be 'grpo' or 'ppo', got {algo!r}")
    suffix = "_ppo" if algo == "ppo" else ""   # PPO checkpoints (train_low_level_ppo.py)
                                                # use a _ppo suffix so they never collide
                                                # with GRPO's on the same volume.

    ckpt_dir = os.path.join(MODEL_CACHE_DIR, "checkpoints")

    # ------------------------------------------------------------------
    # Resolve the "after" checkpoint: best (if requested), else explicit
    # iter, else final, else highest available iter checkpoint (the last
    # three via the shared helper — see training/checkpoints.py).
    # ------------------------------------------------------------------
    if ckpt_iter > 0 and not use_best:
        after_ckpt = os.path.join(ckpt_dir, f"low_level_{subgoal}{suffix}_iter{ckpt_iter}.pt")
        if not os.path.exists(after_ckpt):
            raise FileNotFoundError(f"No checkpoint at {after_ckpt}")
    else:
        after_ckpt = resolve_subgoal_checkpoint(ckpt_dir, subgoal, algo=algo, use_best=use_best)

    print(f"  after checkpoint  -> {after_ckpt}")

    # ------------------------------------------------------------------
    # Optional collision predictor — same as train_low_level.py, so the
    # `descend` observation/reward matches what training actually saw.
    # ------------------------------------------------------------------
    collision_model = None
    collision_ckpt = os.path.join(ckpt_dir, "collision_predictor.pt")
    if os.path.exists(collision_ckpt):
        collision_model = CollisionPredictor()
        collision_model.load_state_dict(torch.load(collision_ckpt, map_location="cpu"))
        collision_model.eval()

    def make_env():
        # +250, not *2: init_episode_before_subgoal's oracle pre-subgoal
        # setup (close_gripper/lift/move_to_target/release) needs headroom
        # on top of the actual episode — see rollout_workers.py's
        # _worker_init for the full rationale.
        base = ObservationHarness(
            gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                      max_episode_steps=max_steps + 250)
        )
        setup_env(base)
        wrapped = SubgoalConditionedEnv(
            base, subgoal=subgoal, collision_model=collision_model,
            max_episode_steps=max_steps,
        )
        return base, wrapped

    def rollout(policy) -> tuple:
        base, env = make_env()
        rng = np.random.default_rng(seed)
        obs, info = env.reset(rng=rng)
        frames = [base.last_frame()]
        total_reward = 0.0
        success = False
        # Per-step block/gripper trajectory — lets a caller pinpoint exactly
        # WHEN the block starts moving and correlate it with the action
        # taken that step, rather than just eyeballing the video (added
        # 2026-07-16 while diagnosing close_gripper's block-drift-on-contact
        # issue; info["block_pos"]/["grip_pos"] come from subgoal_env.py).
        trajectory = [{
            "step": 0, "action": None,
            "block_pos": info.get("block_pos"), "grip_pos": info.get("grip_pos"),
            "d_grip_block": info.get("d_grip_block"),
        }]

        for step in range(1, max_steps + 1):
            action = policy.act(obs, deterministic=not stochastic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            frames.append(base.last_frame())
            trajectory.append({
                "step": step, "action": np.asarray(action).tolist(),
                "block_pos": info.get("block_pos"), "grip_pos": info.get("grip_pos"),
                "d_grip_block": info.get("d_grip_block"),
            })
            if info.get("done", False):
                success = True
            if terminated or truncated:
                break

        env.close()
        return frames, total_reward, success, trajectory

    out_dir = os.path.join(MODEL_CACHE_DIR, "subgoal_videos")
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    # ------------------------------------------------------------------
    # "before" — freshly-initialized policy, untrained
    # ------------------------------------------------------------------
    print(f"\n[1/2] Recording BEFORE (random init)...")
    torch.manual_seed(seed)
    before_policy = SubgoalGaussianPolicy(obs_dim=SUBGOAL_OBS_DIM)
    frames, total_reward, success, before_trajectory = rollout(before_policy)
    before_path = os.path.join(out_dir, f"{subgoal}_before.mp4")
    save_video(frames, before_path, fps=10)
    before_traj_path = os.path.join(out_dir, f"{subgoal}_before_trajectory.json")
    with open(before_traj_path, "w") as f:
        json.dump(before_trajectory, f, indent=2)
    model_volume.commit()   # commit immediately — if the "after" step below throws
                             # (e.g. bad checkpoint), this video must not be lost too
    print(f"  {len(frames)} frames  total_reward={total_reward:.3f}  success={success}")
    print(f"  Saved -> {before_traj_path}")
    # step-0 comes straight from init_episode_before_subgoal's scripted-oracle
    # setup, which runs BEFORE either policy (random-init or trained) takes a
    # single action — so it should be numerically identical to the AFTER
    # rollout's step-0 below (same seed). Printed explicitly so "does the
    # gripper really start higher in one video?" can be checked against
    # numbers instead of two videos with possibly different camera framing.
    step0 = before_trajectory[0]
    print(f"  step0 block_pos={step0['block_pos']}  grip_pos={step0['grip_pos']}")
    print(f"  Saved -> {before_path}")
    results["before"] = {"video_path": before_path, "total_reward": round(total_reward, 4),
                          "success": success, "n_frames": len(frames)}

    # ------------------------------------------------------------------
    # "after" — trained checkpoint
    # ------------------------------------------------------------------
    print(f"\n[2/2] Recording AFTER ({os.path.basename(after_ckpt)})...")
    after_policy = SubgoalGaussianPolicy(obs_dim=SUBGOAL_OBS_DIM)
    ckpt = torch.load(after_ckpt, map_location="cpu")
    # PPO checkpoints (low_level_ppo.py's save_checkpoint) are
    # {"actor": ..., "critic": ...} — only the actor is needed for a
    # rollout. GRPO checkpoints are a flat state_dict, loaded as-is.
    after_policy.load_state_dict(ckpt["actor"] if isinstance(ckpt, dict) and "actor" in ckpt else ckpt)
    after_policy.eval()
    frames, total_reward, success, trajectory = rollout(after_policy)
    iter_match = re.search(r"_iter(\d+)\.pt$", after_ckpt)
    if iter_match:
        tag = iter_match.group(1)
    elif after_ckpt.endswith("_best.pt"):
        tag = "best"
    else:
        tag = "final"
    after_path = os.path.join(out_dir, f"{subgoal}{suffix}_after_{tag}.mp4")
    save_video(frames, after_path, fps=10)

    # Per-step block/gripper trajectory for the trained rollout — pinpoints
    # exactly which step the block starts moving (as opposed to eyeballing
    # the video), and whether it correlates with a nonzero dx/dy/dz action
    # (wrist translating) or happens even while the action's translation
    # is ~0 (fingers-closing-on-off-center-block instead). Saved as JSON
    # next to the video; also printed so it's visible straight in the logs.
    traj_path = os.path.join(out_dir, f"{subgoal}{suffix}_after_{tag}_trajectory.json")
    with open(traj_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    model_volume.commit()

    print(f"  {len(frames)} frames  total_reward={total_reward:.3f}  success={success}")
    print(f"  Saved -> {after_path}")
    print(f"  Saved -> {traj_path}")
    print(f"\n  {'step':>4} {'block_pos':>28} {'grip_pos':>28} {'d_grip_block':>13} {'action':>28}")
    prev_block_pos = None
    for row in trajectory:
        block_pos = row["block_pos"]
        moved = (
            prev_block_pos is not None and block_pos is not None
            and float(np.linalg.norm(np.array(block_pos) - np.array(prev_block_pos))) > 1e-4
        )
        marker = "  <- block moved" if moved else ""
        block_str = "[" + ", ".join(f"{v:.4f}" for v in block_pos) + "]" if block_pos else "None"
        grip_str  = "[" + ", ".join(f"{v:.4f}" for v in row["grip_pos"]) + "]" if row["grip_pos"] else "None"
        action_str = ("[" + ", ".join(f"{v:.3f}" for v in row["action"]) + "]") if row["action"] else "None"
        d_gb = row["d_grip_block"]
        d_gb_str = f"{d_gb:.4f}" if d_gb is not None else "n/a"
        print(f"  {row['step']:>4} {block_str:>28} {grip_str:>28} {d_gb_str:>13} {action_str:>28}{marker}")
        if block_pos is not None:
            prev_block_pos = block_pos

    results["after"] = {"video_path": after_path, "total_reward": round(total_reward, 4),
                         "success": success, "n_frames": len(frames), "ckpt": after_ckpt,
                         "trajectory_path": traj_path}

    print("\n" + "=" * 60)
    print(f"  before: total_reward={results['before']['total_reward']}  success={results['before']['success']}")
    print(f"  after : total_reward={results['after']['total_reward']}  success={results['after']['success']}")
    # Direct numeric check for "does the AFTER rollout genuinely start in a
    # different position than BEFORE?" — both use the same seed and the
    # same scripted-oracle setup (init_episode_before_subgoal), which runs
    # before either policy acts, so step0 should be identical. A nonzero
    # diff here would mean something ELSE differs between the two calls
    # (not policy behavior), worth its own investigation.
    before_step0, after_step0 = before_trajectory[0], trajectory[0]
    grip_diff = float(np.linalg.norm(
        np.array(before_step0["grip_pos"]) - np.array(after_step0["grip_pos"])
    ))
    print(f"  step0 grip_pos  before={before_step0['grip_pos']}")
    print(f"                  after ={after_step0['grip_pos']}")
    print(f"                  diff  ={grip_diff:.6f}  "
          f"({'IDENTICAL as expected' if grip_diff < 1e-6 else 'DIFFERENT -- investigate'})")
    print("=" * 60)

    return {"status": "PASS", "subgoal": subgoal, "results": results}


@app.local_entrypoint()
def main(
    subgoal: str = "align_xy", seed: int = 0, max_steps: int = 30,
    ckpt_iter: int = 0, stochastic: bool = False,
    algo: str = "grpo", use_best: bool = False,
):
    print(f"\nRecording before/after videos for subgoal={subgoal} algo={algo} "
          f"use_best={use_best}...")
    result = record_subgoal_video.remote(
        subgoal=subgoal, seed=seed, max_steps=max_steps,
        ckpt_iter=ckpt_iter, stochastic=stochastic, algo=algo, use_best=use_best,
    )
    b, a = result["results"]["before"], result["results"]["after"]
    print(f"\nDone.")
    print(f"  before: total_reward={b['total_reward']}  success={b['success']}  -> {b['video_path']}")
    print(f"  after : total_reward={a['total_reward']}  success={a['success']}  -> {a['video_path']}")
    print(f"\nDownload with:")
    print(f"  python3 -m modal volume get --force rl-harness-model-cache "
          f"subgoal_videos/ ./artifacts/subgoal_videos/")
