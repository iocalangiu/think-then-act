"""
train_flat_task_bc.py

BC pretraining for the flat (non-subgoal-conditioned) full-task policy, on
the plain FLAT_OBS_DIM=22 observation (training.subgoal_features.
build_flat_observation — relative geometry only, no subgoal one-hot, no
discrepancy/force/slip channels yet, per the user's call to start narrow).
Demonstrations are whatever collect_sb3_teacher_demos.py already produced
(demonstrations/sb3_teacher_full_task.pkl — 11 successful episodes from the
external SB3 TQC+HER teacher, post init_random_episode fix). No new teacher-
collection code needed: that file's "obs"/"teacher_action" keys already
match training/behavioral_cloning.py's BehavioralCloningTrainer.fit()
input shape exactly.

Uses rl_image (think_then_act.modal_app), NOT a separate image — BC itself
needs no mujoco/gymnasium (training/behavioral_cloning.py is pure PyTorch),
and the post-training eval rollout uses this project's OWN env stack
(gymnasium-robotics + env/setup.py), already present in rl_image. No SB3
dependency anywhere in this script.

Writes logs/flat_bc_metrics.jsonl (one line per epoch: {"epoch", "loss"})
and, after training, runs a real eval rollout (same init_random_episode
setup as collect_sb3_teacher_demos.py) to get a genuine completion_rate —
BC epoch loss alone doesn't tell you whether the policy actually does the
task. The eval's "success" requires the same grasp-carry verification
collect_sb3_teacher_demos.py's collection filter uses (two-finger contact +
block lifted >2cm at some point, not just final distance-to-goal) — plain
is_success alone accepts slide/nudge episodes that never actually lifted
the block (root-caused 2026-10-02). Saves one success + one fail video
(whichever appears first in eval, not just episode 0/1 — see the "first N
regardless of outcome" miss from collect_sb3_teacher_demos.py's early
videos) to demonstrations/videos/ for the Fetch Policy Telemetry dashboard.

Run with:
    modal run scripts/multi-cube-experiment/train_flat_task_bc.py
    modal run scripts/multi-cube-experiment/train_flat_task_bc.py --n-epochs 40 --n-eval-episodes 30
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(image=rl_image, gpu=None, cpu=4.0,
              volumes={MODEL_CACHE_DIR: model_volume}, timeout=1800)
def train_flat_task_bc(
    demos_path: str = "demonstrations/sb3_teacher_full_task.pkl",
    n_epochs: int = 20,
    lr: float = 1e-3,
    n_eval_episodes: int = 20,
    max_steps: int = 100,
    ckpt_path: str = "checkpoints/flat_task_bc.pt",
    metrics_path: str = "logs/flat_bc_metrics.jsonl",
    video_dir: str = "demonstrations/videos",
    require_genuine_grasp: bool = True,   # is_success alone accepts slide/nudge
                                  # episodes that never actually lifted the block
                                  # (root-caused 2026-10-02 via
                                  # diagnose_flat_bc_success_quality.py — same
                                  # fix already applied to
                                  # collect_sb3_teacher_demos.py's collection
                                  # filter, applied here too so this eval's
                                  # completion_rate can't be inflated the same way).
    lift_threshold: float = 0.02,
    block_resting_z: float = 0.425,
) -> dict:
    import os
    import json
    import pickle
    import numpy as np
    import torch

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    # torch before gymnasium/gymnasium_robotics — see
    # collect_sb3_teacher_demos.py's docstring for why this order matters
    # when torch and mujoco share a process (not actually needed for
    # rl_image specifically, which is already proven in this combination
    # elsewhere in the project, but keeping the convention consistent).
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    # save_video, not imageio.mimsave directly — this project already found
    # imageio's default MP4 path unreliable (PyAV backend has no libx264;
    # see that function's own docstring) and shells out to the system
    # ffmpeg binary with explicit libx264/yuv420p instead — the codec/pixel
    # format browsers actually need, which is also the likely root cause of
    # collect_sb3_teacher_demos.py's videos failing to play once embedded in
    # the dashboard artifact (confirmed 2026-10-02).
    from think_then_act.env.setup import setup_env, init_random_episode, save_video, grip_contact_forces
    from think_then_act.training.subgoal_features import build_flat_observation, FLAT_OBS_DIM
    from think_then_act.training.behavioral_cloning import BCConfig, BehavioralCloningTrainer

    print("\n" + "=" * 70)
    print("  FLAT-TASK BC TRAINING")
    print("=" * 70)

    full_demos_path = os.path.join(MODEL_CACHE_DIR, demos_path)
    with open(full_demos_path, "rb") as f:
        demo_data = pickle.load(f)
    demonstrations = demo_data["demonstrations"]
    print(f"  Loaded {len(demonstrations)} demonstrations from {full_demos_path} "
          f"(collected at success_rate={demo_data.get('success_rate', float('nan')):.1%})")
    if len(demonstrations) < 20:
        print(f"  NOTE: only {len(demonstrations)} demonstrations — thin for a GRU "
              f"policy, this run is a first-pass sanity check, not a final result. "
              f"Consider a bigger --n-episodes collection run before trusting this.")

    config = BCConfig(obs_dim=FLAT_OBS_DIM, action_dim=4, lr=lr, n_epochs=n_epochs)
    trainer = BehavioralCloningTrainer(config)
    fit_result = trainer.fit(demonstrations)
    epoch_losses = fit_result["epoch_losses"]

    full_metrics_path = os.path.join(MODEL_CACHE_DIR, metrics_path)
    os.makedirs(os.path.dirname(full_metrics_path), exist_ok=True)
    with open(full_metrics_path, "w") as f:
        for epoch, loss in enumerate(epoch_losses):
            f.write(json.dumps({"epoch": epoch, "loss": loss}) + "\n")
    print(f"  Wrote {len(epoch_losses)} epoch losses -> {full_metrics_path}")
    print(f"  final loss: {epoch_losses[-1]:.4f}  (first: {epoch_losses[0]:.4f})")

    full_ckpt_path = os.path.join(MODEL_CACHE_DIR, ckpt_path)
    trainer.save_checkpoint(full_ckpt_path)
    print(f"  Saved checkpoint -> {full_ckpt_path}")

    # ------------------------------------------------------------------
    # Real eval rollout — BC loss alone doesn't say whether the policy
    # actually does the task. Same env setup as collect_sb3_teacher_demos.py
    # (init_random_episode, not the native reset — see that script's
    # docstring for why the native placement breaks once the robot base is
    # moved).
    # ------------------------------------------------------------------
    print("\n  Running eval rollout...")
    env = gym.make("FetchPickAndPlace-v3", max_episode_steps=max_steps, render_mode="rgb_array")
    setup_env(env)

    n_success = 0
    n_is_success_raw = 0   # is_success==True regardless of genuine — tracked
                            # separately so the eval summary shows the
                            # contamination rate, same convention as
                            # collect_sb3_teacher_demos.py
    saved_success_video = False
    saved_fail_video = False

    for ep in range(n_eval_episodes):
        rng = np.random.default_rng(ep)
        env.reset(seed=ep)
        obs, setup_ok = init_random_episode(env, rng)
        if not setup_ok:
            continue

        want_video = (not saved_success_video) or (not saved_fail_video)
        frames = [env.render()] if want_video else None

        hidden_state = None
        success = False
        ever_lifted_and_gripped = False
        for _ in range(max_steps):
            flat_obs = build_flat_observation(obs["observation"], obs["achieved_goal"], obs["desired_goal"])
            action, hidden_state = trainer.actor.act(flat_obs, hidden_state, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if want_video:
                frames.append(env.render())

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
        genuine = success and (ever_lifted_and_gripped or not require_genuine_grasp)
        status = "fail" if not success else ("SUCCESS" if genuine else "SUCCESS(slide, rejected)")
        print(f"    eval ep={ep:>3}  {status}")
        if genuine:
            n_success += 1

        if want_video and ((genuine and not saved_success_video) or (not genuine and not saved_fail_video)):
            tag = "success" if genuine else "fail"
            video_path = os.path.join(MODEL_CACHE_DIR, video_dir, f"flat_bc_eval_{tag}.mp4")
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            save_video(frames, video_path, fps=20)
            print(f"      saved {tag} video -> {video_path}")
            if genuine:
                saved_success_video = True
            else:
                saved_fail_video = True

    env.close()
    completion_rate = n_success / n_eval_episodes if n_eval_episodes else 0.0
    raw_completion_rate = n_is_success_raw / n_eval_episodes if n_eval_episodes else 0.0
    print(f"\n  genuine completion_rate: {n_success}/{n_eval_episodes} ({completion_rate:.1%})")
    if require_genuine_grasp:
        print(f"  (raw is_success fired {n_is_success_raw}/{n_eval_episodes} = {raw_completion_rate:.1%} — "
              f"{n_is_success_raw - n_success} of those were slide/nudge, not counted)")

    eval_summary_path = os.path.join(MODEL_CACHE_DIR, "logs/flat_bc_eval.json")
    with open(eval_summary_path, "w") as f:
        json.dump({"n_eval_episodes": n_eval_episodes, "n_success": n_success,
                    "n_is_success_raw": n_is_success_raw,
                    "completion_rate": completion_rate, "final_bc_loss": epoch_losses[-1]}, f)
    model_volume.commit()
    print(f"  Saved eval summary -> {eval_summary_path}")
    print("=" * 70)

    return {"epoch_losses": epoch_losses, "completion_rate": completion_rate,
            "n_demonstrations": len(demonstrations)}


@app.local_entrypoint()
def main(
    demos_path: str = "demonstrations/sb3_teacher_full_task.pkl",
    n_epochs: int = 20,
    lr: float = 1e-3,
    n_eval_episodes: int = 20,
):
    result = train_flat_task_bc.remote(
        demos_path=demos_path, n_epochs=n_epochs, lr=lr, n_eval_episodes=n_eval_episodes,
    )
    print(f"\nDone: {result['n_demonstrations']} demos, "
          f"final_loss={result['epoch_losses'][-1]:.4f}, "
          f"eval completion_rate={result['completion_rate']:.1%}")
