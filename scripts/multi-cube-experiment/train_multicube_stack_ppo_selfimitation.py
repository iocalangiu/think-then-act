"""
train_multicube_stack_ppo_selfimitation.py

PPO on MultiCubeStackEnv, warm-started from the BEST available checkpoint
(bc_multicube_finetune_v1.pt, 46.7% fully-autonomous 2-cube success — see
memory: flat_policy_ppo_generalization) instead of the raw single-cube PPO
checkpoint the earlier, abandoned PPO-for-2-cubes attempt used. That
earlier attempt showed a real, sustained collapse even at lr=1e-4 — the
user's hypothesis for a retry: a much stronger starting point should be
far more stable, AND augment PPO with an experience-replay / self-
imitation auxiliary update (FlatTaskPPOTrainer.bc_replay_step, added
alongside this script) that directly re-emphasizes genuine successes via
a supervised regression step, independent of the PPO batch's own GAE
math. Motivation: this task's success rate is nowhere near 100% (19-47%
range across every checkpoint tried so far), so a sparse-success on-
policy batch's few successes can get outweighed by its many failures in
the clipped-surrogate gradient even when the successes themselves are
exactly the behavior worth reinforcing.

Replay buffer: seeded from the existing combined 2-cube demo pool
(demonstrations/stack2_demos_combined_v2.pkl, 600 genuine demos — already
on disk, no need to wait for PPO's own successes to populate it from
scratch), then grown every iteration with whatever genuine on-policy
successes this run's own rollouts produce (info["done"], the same
stack-integrity-checked flag collect_rollouts already reports as
"genuine_success" per rollout — see rollout_workers._run_episode_flat).
Capped at replay_buffer_max via FIFO eviction of the OLDEST entries (seed
demos included) — intentional: as training progresses, the buffer should
increasingly reflect the CURRENT policy's own successful behavior
distribution, not stay anchored to the original scripted-assisted demos.

Fallback not automated here, left as a manual next step if this still
doesn't improve: scripts/collect_stack2_genuine_demos_finetune.py can be
rerun against whatever the current best checkpoint is (bootstrapping a
fresh, larger genuine-demo batch), merged into a new combined pool, and
passed as --replay-seed-demos-path on a continued run — same resume-safe/
incremental-save discipline as that script already has. Deliberately NOT
wired to auto-trigger mid-training: that would mean spawning a second
Modal function from inside this one, a much larger change to review than
starting with a clean, cheap first attempt.

Run with:
    modal run --detach scripts/train_multicube_stack_ppo_selfimitation.py
    modal run --detach scripts/train_multicube_stack_ppo_selfimitation.py \\
        --n-workers 1 --n-iterations 3 --n-rollouts 8   # quick sanity check, no process pool
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(image=rl_image, gpu=None, cpu=8.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=3600 * 6)
def train_multicube_stack_ppo_selfimitation(
    n_iterations: int = 300,
    max_episode_steps: int = 320,
    eval_episodes: int = 30,
    checkpoint_every: int = 20,
    seed: int = 0,
    n_rollouts: int = 64,
    n_workers: int = 8,
    rnn_hidden_size: int = 64,
    episodes_per_minibatch: int = 16,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    lr: float = 1e-4,
    bc_replay_lr: float = 1e-4,
    entropy_coef: float = 0.01,
    clip_eps: float = 0.2,
    n_epochs: int = 4,
    early_stop_patience: int = 3,
    early_stop_threshold: float = 0.6,
    warm_start_ckpt: str = "checkpoints/bc_multicube_finetune_v1.pt",
    warm_start_log_std_init: float = -2.0,
    critic_warmup_iters: int = 5,
    discrepancy_weight: float = 2.0,
    drag_weight: float = 20.0,
    success_bonus: float = 5.0,
    stillness_weight: float = 3.0,
    precision_weight: float = 10.0,
    disturbance_weight: float = 15.0,
    randomize_pose_prob: float = 1.0,
    pose_scheme: str = "gripper_3d",
    min_cubes: int = 2,
    max_cubes: int = 2,
    replay_seed_demos_path: str = "demonstrations/stack2_demos_combined_v2.pkl",
    replay_buffer_max: int = 2000,
    replay_batch_episodes: int = 64,
    ckpt_name: str = "multicube_stack_ppo_selfimitation_v1",
) -> dict:
    import os
    import math
    import pickle
    import numpy as np
    import torch

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium_robotics  # noqa: F401

    from think_then_act.training.subgoal_features import FLAT_OBS_DIM
    from think_then_act.training.flat_task_ppo import FlatTaskPPOConfig, FlatTaskPPOTrainer
    from think_then_act.training.multicube_stack_env import MultiCubeStackEnv

    print("\n" + "=" * 60)
    print("  MULTI-CUBE STACK PPO + SELF-IMITATION REPLAY")
    print("  (warm-started from the fine-tuned BC checkpoint, not raw single-cube PPO)")
    print("=" * 60)

    torch.manual_seed(seed)
    rng_buf = np.random.default_rng(seed)

    replay_buffer = []
    if replay_seed_demos_path:
        with open(os.path.join(MODEL_CACHE_DIR, replay_seed_demos_path), "rb") as f:
            seed_data = pickle.load(f)
        for d in seed_data["demonstrations"]:
            replay_buffer.append({"obs": d["obs"], "teacher_action": d["teacher_action"]})
        print(f"  [replay] seeded buffer with {len(replay_buffer)} demos from {replay_seed_demos_path}")

    weights_kwargs = dict(discrepancy_weight=discrepancy_weight, drag_weight=drag_weight,
                           success_bonus=success_bonus, stillness_weight=stillness_weight)
    config = FlatTaskPPOConfig(
        obs_dim=FLAT_OBS_DIM, max_episode_steps=max_episode_steps, n_rollouts=n_rollouts,
        n_workers=n_workers, rnn_hidden_size=rnn_hidden_size, episodes_per_minibatch=episodes_per_minibatch,
        gamma=gamma, gae_lambda=gae_lambda, lr=lr, entropy_coef=entropy_coef, clip_eps=clip_eps,
        n_epochs=n_epochs, weights_kwargs=weights_kwargs,
        randomize_pose_prob=randomize_pose_prob, pose_scheme=pose_scheme,
        env_variant="multicube", min_cubes=min_cubes, max_cubes=max_cubes,
        precision_weight=precision_weight, disturbance_weight=disturbance_weight,
        bc_replay_lr=bc_replay_lr,
    )
    trainer = FlatTaskPPOTrainer(config)

    if warm_start_ckpt:
        warm_start_path = os.path.join(MODEL_CACHE_DIR, warm_start_ckpt)
        trainer.load_checkpoint(warm_start_path)
        print(f"  [warm-start] loaded {warm_start_path}")
        if warm_start_log_std_init != 0.0:
            with torch.no_grad():
                trainer.actor.log_std.fill_(warm_start_log_std_init)
            print(f"  [warm-start] log_std initialized to {warm_start_log_std_init} "
                  f"(std={math.exp(warm_start_log_std_init):.3f})")

    env_kwargs = config.env_kwargs()
    print(f"  min_cubes={min_cubes}  max_cubes={max_cubes}  randomize_pose_prob={randomize_pose_prob}  "
          f"pose_scheme={pose_scheme}")

    if warm_start_ckpt and critic_warmup_iters > 0:
        print(f"  [warm-start] running {critic_warmup_iters} critic-only warmup iteration(s)...")
        WARMUP_SEED_BASE = 1_000_000_000
        for wi in range(critic_warmup_iters):
            warmup_seeds = list(range(
                WARMUP_SEED_BASE + config.n_rollouts * wi,
                WARMUP_SEED_BASE + config.n_rollouts * (wi + 1),
            ))
            warmup_rollouts = trainer.collect_rollouts(env_kwargs, warmup_seeds)
            warmup_metrics = trainer.critic_warmup_step(warmup_rollouts)
            print(f"    critic warmup {wi+1}/{critic_warmup_iters}: value_loss={warmup_metrics['value_loss']:.4f}")

    def run_eval(actor=None, n_eval_episodes: int = eval_episodes) -> dict:
        actor = actor if actor is not None else trainer.actor
        eval_env = MultiCubeStackEnv(
            min_cubes=min_cubes, max_cubes=max_cubes, max_episode_steps=max_episode_steps,
            randomize_pose_prob=1.0, pose_scheme=pose_scheme,
            precision_weight=precision_weight, disturbance_weight=disturbance_weight,
        )
        n_full_stack = 0
        total_placed = 0
        total_cubes = 0
        for ep in range(n_eval_episodes):
            seed_ep = 300_000 + ep
            rng = np.random.default_rng(seed_ep)
            obs, reset_info = eval_env.reset(rng=rng, seed=seed_ep)
            hidden_state = None
            info = {}
            for _ in range(max_episode_steps):
                action, hidden_state = actor.act(obs, hidden_state, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                if terminated or truncated:
                    break
            if info.get("done", False):
                n_full_stack += 1
            total_placed += info.get("n_cubes_placed", 0)
            total_cubes += info.get("n_cubes", reset_info["n_cubes"])
        eval_env.close()
        return {
            "completion_rate": n_full_stack / n_eval_episodes if n_eval_episodes else 0.0,
            "cube_placement_rate": total_placed / total_cubes if total_cubes else 0.0,
        }

    def harvest_genuine_rollouts(rollouts: list) -> int:
        """Append this iteration's own genuine on-policy successes to the
        replay buffer (bounded-action target = tanh(raw_sample), the same
        convention bc_replay_step's loss uses), then FIFO-evict down to
        replay_buffer_max. Returns how many were added."""
        n_added = 0
        for r in rollouts:
            if not r.get("genuine_success"):
                continue
            obs = np.stack([s["obs"] for s in r["steps"]]).astype(np.float32)
            action = np.tanh(np.stack([s["raw_sample"] for s in r["steps"]])).astype(np.float32)
            replay_buffer.append({"obs": obs, "teacher_action": action})
            n_added += 1
        if len(replay_buffer) > replay_buffer_max:
            del replay_buffer[: len(replay_buffer) - replay_buffer_max]
        return n_added

    def sample_replay_batch() -> list:
        if not replay_buffer:
            return []
        n = min(replay_batch_episodes, len(replay_buffer))
        idx = rng_buf.choice(len(replay_buffer), size=n, replace=False)
        return [replay_buffer[i] for i in idx]

    ckpt_dir = os.path.join(MODEL_CACHE_DIR, "checkpoints")
    best_ckpt_path = os.path.join(ckpt_dir, f"{ckpt_name}_best.pt")
    initial_eval = run_eval()
    print(f"  [eval @ iter 0 / warm-start] full_stack={initial_eval['completion_rate']:.1%}  "
          f"cube_placement_rate={initial_eval['cube_placement_rate']:.1%}")
    best_completion_rate = initial_eval["completion_rate"]
    trainer.save_checkpoint(best_ckpt_path)
    model_volume.commit()

    def maybe_save_best(completion_rate: float) -> None:
        nonlocal best_completion_rate
        if completion_rate > best_completion_rate:
            best_completion_rate = completion_rate
            trainer.save_checkpoint(best_ckpt_path)
            model_volume.commit()
            print(f"  [best] completion_rate={completion_rate:.1%} -> {best_ckpt_path}")

    latest_ckpt_path = os.path.join(ckpt_dir, f"{ckpt_name}_latest.pt")

    history = []
    eval_history = [{"iteration": 0, **initial_eval}]
    consecutive_at_threshold = 0
    stopped_early_at = None
    try:
        for i in range(n_iterations):
            seeds = list(range(config.n_rollouts * i, config.n_rollouts * (i + 1)))
            rollouts = trainer.collect_rollouts(env_kwargs, seeds)
            ppo_metrics = trainer.ppo_step(rollouts)

            n_harvested = harvest_genuine_rollouts(rollouts)
            replay_batch = sample_replay_batch()
            replay_metrics = trainer.bc_replay_step(replay_batch)

            metrics = dict(ppo_metrics)
            metrics["iteration"] = i + 1
            metrics["n_harvested_this_iter"] = n_harvested
            metrics["replay_buffer_size"] = len(replay_buffer)
            metrics["bc_replay_loss"] = replay_metrics["bc_replay_loss"]
            history.append(metrics)

            trainer.save_checkpoint(latest_ckpt_path)
            model_volume.commit()

            history_path = os.path.join(ckpt_dir, f"{ckpt_name}_history.json")
            with open(history_path, "w") as f:
                import json
                json.dump({"history": history, "eval_history": eval_history,
                           "lr": lr, "ckpt_name": ckpt_name}, f, default=float)

            if (i + 1) % 5 == 0:
                bc_loss_str = f"{metrics['bc_replay_loss']:.4f}" if metrics["bc_replay_loss"] is not None else "n/a"
                print(f"  iter {i+1}/{n_iterations}  policy_loss={metrics['policy_loss']:.4f}  "
                      f"value_loss={metrics['value_loss']:.4f}  mean_reward={metrics['mean_reward']:.4f}  "
                      f"entropy={metrics['mean_entropy']:.4f}  approx_kl={metrics['approx_kl']:.4f}  "
                      f"train_genuine_rate={metrics['genuine_success_rate']:.2f}  "
                      f"bc_replay_loss={bc_loss_str}  replay_buf={metrics['replay_buffer_size']}  "
                      f"harvested={metrics['n_harvested_this_iter']}")

            if (i + 1) % checkpoint_every == 0:
                ckpt = os.path.join(ckpt_dir, f"{ckpt_name}_iter{i+1}.pt")
                trainer.save_checkpoint(ckpt)
                model_volume.commit()

                eval_result = run_eval()
                maybe_save_best(eval_result["completion_rate"])
                eval_history.append({"iteration": i + 1, **eval_result})
                with open(history_path, "w") as f:
                    import json
                    json.dump({"history": history, "eval_history": eval_history,
                               "lr": lr, "ckpt_name": ckpt_name}, f, default=float)
                model_volume.commit()
                print(f"  [eval @ iter {i+1}] full_stack={eval_result['completion_rate']:.1%}  "
                      f"cube_placement_rate={eval_result['cube_placement_rate']:.1%} over {eval_episodes} episodes")

                if early_stop_patience > 0:
                    if eval_result["completion_rate"] >= early_stop_threshold:
                        consecutive_at_threshold += 1
                    else:
                        consecutive_at_threshold = 0
                    if consecutive_at_threshold >= early_stop_patience:
                        stopped_early_at = i + 1
                        print(f"  [early-stop] completion_rate>={early_stop_threshold:.0%} held for "
                              f"{consecutive_at_threshold} consecutive eval checkpoints — stopping at "
                              f"iter {stopped_early_at}.")
                        break

        final_iteration = stopped_early_at if stopped_early_at is not None else n_iterations
        ckpt_out = os.path.join(ckpt_dir, f"{ckpt_name}.pt")
        trainer.save_checkpoint(ckpt_out)
        model_volume.commit()
        print(f"  Saved -> {ckpt_out}")

        if eval_history[-1]["iteration"] == final_iteration:
            final_eval = eval_history[-1]
        else:
            final_eval = run_eval()
            maybe_save_best(final_eval["completion_rate"])
            eval_history.append({"iteration": final_iteration, **final_eval})

        print(f"  [eval] final: full_stack={final_eval['completion_rate']:.1%}  "
              f"cube_placement_rate={final_eval['cube_placement_rate']:.1%}")
        print(f"  best completion_rate={best_completion_rate:.1%} -> {best_ckpt_path}")
        return {
            "status": "PASS", "ckpt_path": ckpt_out, "best_ckpt_path": best_ckpt_path,
            "completion_rate": round(final_eval["completion_rate"], 4),
            "cube_placement_rate": round(final_eval["cube_placement_rate"], 4),
            "best_completion_rate": round(best_completion_rate, 4),
            "eval_history": eval_history, "stopped_early_at": stopped_early_at,
            "final_replay_buffer_size": len(replay_buffer),
        }
    finally:
        trainer.close_pool()


@app.local_entrypoint()
def main(
    n_iterations: int = 300,
    max_episode_steps: int = 320,
    eval_episodes: int = 30,
    checkpoint_every: int = 20,
    seed: int = 0,
    n_rollouts: int = 64,
    n_workers: int = 8,
    lr: float = 1e-4,
    bc_replay_lr: float = 1e-4,
    critic_warmup_iters: int = 5,
    warm_start_log_std_init: float = -2.0,
    early_stop_patience: int = 3,
    early_stop_threshold: float = 0.6,
    warm_start_ckpt: str = "checkpoints/bc_multicube_finetune_v1.pt",
    min_cubes: int = 2,
    max_cubes: int = 2,
    replay_seed_demos_path: str = "demonstrations/stack2_demos_combined_v2.pkl",
    replay_buffer_max: int = 2000,
    replay_batch_episodes: int = 64,
    ckpt_name: str = "multicube_stack_ppo_selfimitation_v1",
):
    result = train_multicube_stack_ppo_selfimitation.remote(
        n_iterations=n_iterations, max_episode_steps=max_episode_steps, eval_episodes=eval_episodes,
        checkpoint_every=checkpoint_every, seed=seed, n_rollouts=n_rollouts, n_workers=n_workers,
        lr=lr, bc_replay_lr=bc_replay_lr, critic_warmup_iters=critic_warmup_iters,
        warm_start_log_std_init=warm_start_log_std_init, early_stop_patience=early_stop_patience,
        early_stop_threshold=early_stop_threshold, warm_start_ckpt=warm_start_ckpt,
        min_cubes=min_cubes, max_cubes=max_cubes, replay_seed_demos_path=replay_seed_demos_path,
        replay_buffer_max=replay_buffer_max, replay_batch_episodes=replay_batch_episodes,
        ckpt_name=ckpt_name,
    )
    print("\n", result)
