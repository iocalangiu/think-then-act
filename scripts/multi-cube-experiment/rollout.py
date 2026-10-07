"""
think_then_act multi-cube-experiment rollout.py

General-purpose "roll out a checkpoint and look at it" tool -- supersedes
the pile of one-off eval_trustworthy_completion.py / find_better_clips.py /
diagnose_*.py / build_grid_*.py scripts built ad hoc across this project's
sessions, each of which re-decided from scratch which env/trainer class a
checkpoint needs. Resolve that ONCE via model_registry.py instead.

IMPORTANT, read before trusting a video's success/failure label:
Recording video (render_mode="rgb_array" + calling env.render() every step)
measurably perturbs this project's MuJoCo physics simulation -- confirmed
directly (2026-10-07): across 6 held-out seeds, the SAME checkpoint/seed
pair gave identical genuine/failure classifications in 5/6 cases between a
rendered and non-rendered rollout, but step counts differed by 1-3 steps in
EVERY case (tiny floating-point-level perturbation, most likely from the
OSMesa software rendering context), and the 6th seed (900021) flipped from
a clean success (74 steps, no rendering) to a failure (69 steps, rendered)
-- it happened to sit right on this policy's decision boundary. The
completion-rate NUMBERS in models.yaml were all measured WITHOUT rendering
and are unaffected. A specific "this seed is a failure" claim drawn only
from a recorded video is not fully trustworthy for borderline trials --
use --also-clean-pass (on by default when --record-video is set) to get
the non-rendered classification for the same seeds and treat THAT as the
authoritative one; the video is illustrative, not evidence.

Usage:
    modal run scripts/multi-cube-experiment/rollout.py --model multicube_stack_ppo_selfimitation_v2 --n-trials 10
    modal run scripts/multi-cube-experiment/rollout.py --model multicube_stack_ppo_selfimitation_v2 --n-trials 6 --record-video --out-dir rollouts/selfimit_v2
    modal run scripts/multi-cube-experiment/rollout.py --ckpt-path checkpoints/some_new.pt --architecture mse --task stack2 --n-trials 5
"""

import os

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR

# NOTE: model_registry is intentionally NOT imported at module scope. Modal
# only mounts this one file for the remote function below, not its sibling
# files in this folder -- importing model_registry up here makes the
# REMOTE container (which re-executes this whole module to find
# run_rollouts) crash-loop on ModuleNotFoundError, since model_registry.py
# was never shipped to it. Deferred into main() instead, which only ever
# runs locally.


@app.function(image=rl_image, gpu=None, cpu=2.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=1800)
def run_rollouts(
    ckpt_path: str,
    architecture: str,
    task: str,
    n_trials: int,
    seed_start: int,
    record_video: bool,
    also_clean_pass: bool,
    out_dir: str,
    max_episode_steps: int,
):
    import os
    import json
    import numpy as np

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    import gymnasium_robotics  # noqa: F401
    from think_then_act.env.setup import (
        save_video, setup_env, init_random_episode, grip_contact_forces, randomize_gripper_start_3d,
    )
    from think_then_act.training.subgoal_features import FLAT_OBS_DIM, build_flat_observation
    from think_then_act.training.flat_task_ppo import FlatTaskPPOConfig, FlatTaskPPOTrainer
    from think_then_act.training.flat_bc_multi_head import MultiHeadBCConfig, MultiHeadBCTrainer

    assert task in ("basic_skill", "stack2"), f"unknown task {task!r}"

    full_ckpt = os.path.join(MODEL_CACHE_DIR, ckpt_path)

    def load_trainer():
        """
        Try PPO-trained first (architecture=... dispatch handles mse/
        transformer), fall back to BC-only (MultiHeadBCTrainer handles all
        4 architectures) -- same "PPO checkpoints don't carry a
        policy_type key, BC ones do" duck-typing this project has used
        since train_bc_multicube_finetune.py.
        """
        try:
            cfg = FlatTaskPPOConfig(obs_dim=FLAT_OBS_DIM, architecture=architecture)
            trainer = FlatTaskPPOTrainer(cfg)
            trainer.load_checkpoint(full_ckpt)
            return trainer.actor
        except Exception:
            cfg = MultiHeadBCConfig(obs_dim=FLAT_OBS_DIM, policy_type=architecture)
            trainer = MultiHeadBCTrainer(cfg)
            trainer.load_checkpoint(full_ckpt)
            return trainer.actor

    actor = load_trainer()
    actor.eval()
    print(f"loaded {ckpt_path}  (task={task} architecture={architecture})", flush=True)

    def run_one_stack2(seed: int, render: bool):
        from think_then_act.training.multicube_stack_env import MultiCubeStackEnv
        rng = np.random.default_rng(seed)
        env = MultiCubeStackEnv(min_cubes=2, max_cubes=2, max_episode_steps=max_episode_steps,
                                 randomize_pose_prob=1.0, pose_scheme="gripper_3d",
                                 render_mode="rgb_array" if render else None)
        obs, reset_info = env.reset(rng=rng, seed=seed)
        frames = [env.render()] if render else None
        hidden_state = None
        info = {}
        n_steps = 0
        for _ in range(max_episode_steps):
            action, hidden_state = actor.act(obs, hidden_state, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            n_steps += 1
            if render:
                frames.append(env.render())
            if terminated or truncated:
                break
        env.close()
        return info.get("done", False), info.get("n_cubes_placed"), n_steps, frames

    def run_one_basic_skill(seed: int, render: bool):
        import gymnasium as gym
        rng = np.random.default_rng(seed)
        env = gym.make("FetchPickAndPlace-v3", max_episode_steps=max_episode_steps,
                        render_mode="rgb_array" if render else None)
        setup_env(env)
        reset_obs, _ = env.reset(seed=seed)
        obs, pose_ok, _ = randomize_gripper_start_3d(env, rng, reset_obs)
        if not pose_ok:
            env.close()
            return None, None, 0, None
        obs, setup_ok = init_random_episode(env, rng)
        if not setup_ok:
            env.close()
            return None, None, 0, None
        frames = [env.render()] if render else None
        hidden_state = None
        success = False
        ever_lifted_and_gripped = False
        n_steps = 0
        for _ in range(max_episode_steps):
            flat_obs = build_flat_observation(obs["observation"], obs["achieved_goal"], obs["desired_goal"])
            action, hidden_state = actor.act(flat_obs, hidden_state, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            n_steps += 1
            if render:
                frames.append(env.render())
            height_above_resting = float(obs["achieved_goal"][2]) - 0.425
            forces = grip_contact_forces(env)
            if min(forces["left"], forces["right"]) > 0.0 and height_above_resting > 0.02:
                ever_lifted_and_gripped = True
            if info.get("is_success", False):
                success = True
            if terminated or truncated:
                break
        env.close()
        genuine = success and ever_lifted_and_gripped
        return genuine, None, n_steps, frames

    run_one = run_one_stack2 if task == "stack2" else run_one_basic_skill

    results = []
    out_full = os.path.join(MODEL_CACHE_DIR, out_dir) if out_dir else None
    if out_full:
        os.makedirs(out_full, exist_ok=True)

    for i in range(n_trials):
        seed = seed_start + i
        row = {"seed": seed}
        if record_video:
            genuine_r, n_placed_r, steps_r, frames = run_one(seed, render=True)
            row["genuine_rendered"] = genuine_r
            row["n_cubes_placed"] = n_placed_r
            row["n_steps_rendered"] = steps_r
            if out_full:
                video_path = os.path.join(out_full, f"seed{seed}.mp4")
                save_video(frames, video_path, fps=20)
                row["video"] = os.path.join(out_dir, f"seed{seed}.mp4")
            if also_clean_pass:
                genuine_c, _, steps_c, _ = run_one(seed, render=False)
                row["genuine_clean"] = genuine_c
                row["n_steps_clean"] = steps_c
                row["rendering_changed_outcome"] = (genuine_c != genuine_r)
        else:
            genuine_c, n_placed_c, steps_c, _ = run_one(seed, render=False)
            row["genuine_clean"] = genuine_c
            row["n_cubes_placed"] = n_placed_c
            row["n_steps_clean"] = steps_c
        print(f"  {row}", flush=True)
        results.append(row)

    n_flipped = sum(1 for r in results if r.get("rendering_changed_outcome"))
    authoritative_key = "genuine_clean" if (not record_video or also_clean_pass) else "genuine_rendered"
    n_genuine = sum(1 for r in results if r.get(authoritative_key))
    summary = {
        "ckpt_path": ckpt_path, "architecture": architecture, "task": task,
        "n_trials": n_trials, "n_genuine": n_genuine,
        "completion_rate": n_genuine / n_trials if n_trials else 0.0,
        "n_rendering_flipped_outcome": n_flipped,
        "results": results,
    }
    if out_full:
        with open(os.path.join(out_full, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=float)
        model_volume.commit()
    print(f"\ncompletion_rate={summary['completion_rate']:.1%} over {n_trials} trials "
          f"(authoritative={authoritative_key}); rendering flipped the outcome on "
          f"{n_flipped} trial(s)", flush=True)
    return summary


@app.local_entrypoint()
def main(
    model: str = "",
    ckpt_path: str = "",
    architecture: str = "",
    task: str = "",
    n_trials: int = 10,
    seed_start: int = 900100,
    record_video: bool = False,
    also_clean_pass: bool = True,
    out_dir: str = "",
    max_episode_steps: int = 320,
):
    if model:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from model_registry import load as load_model
        entry = load_model(model)
        ckpt_path = ckpt_path or entry.path
        architecture = architecture or entry.architecture
        task = task or entry.task
        if entry.superseded:
            print(f"NOTE: {model!r} is marked superseded in models.yaml "
                  f"({entry.notes.strip()})")
    if not (ckpt_path and architecture and task):
        raise SystemExit(
            "Need either --model <name from models.yaml> or all three of "
            "--ckpt-path/--architecture/--task."
        )
    out_dir = out_dir or (f"rollouts/{model or os.path.basename(ckpt_path)}" if record_video else "")
    result = run_rollouts.remote(
        ckpt_path=ckpt_path, architecture=architecture, task=task,
        n_trials=n_trials, seed_start=seed_start, record_video=record_video,
        also_clean_pass=also_clean_pass, out_dir=out_dir, max_episode_steps=max_episode_steps,
    )
    print("\n", result)
