"""
train_multicube_stack_ppo.py

PPO on training/multicube_stack_env.py's MultiCubeStackEnv -- the
reward-driven alternative to the demo-collection+BC-fine-tune approach
tried first for "release the cube and continue to the next one" (that
approach broke the base single-cube skill without teaching the target
behavior, see memory: flat_policy_ppo_generalization). The environment
itself switches the active goal to the next cube the instant the current
one is genuinely placed -- not a scripted transition -- so the policy has
to discover release-and-continue through ordinary PPO exploration against
the reward, the same mechanism that already fixed drag/idle-churn on the
single-cube task without anyone hand-authoring those fixes.

Warm-starts from the pose-randomized single-cube PPO checkpoint (already
knows grasp/carry/place) rather than training from scratch -- same
warm-start-from-BC recipe as train_flat_task_ppo.py (critic-only warmup +
low initial log_std), since a from-scratch actor on a sparse multi-cube
task would have to rediscover the base skill AND the transition behavior
simultaneously, a much harder exploration problem.

Cube count/identity/order are randomized every episode (see
MultiCubeStackEnv.reset()) so the policy can't shortcut via step-counting
-- see that class's own docstring and the project history around
2026-10-03 for why this is "trained generalization," not literal
in-context learning (this GRU has no mechanism for the latter).

Run with:
    modal run --detach scripts/train_multicube_stack_ppo.py \\
        --warm-start-ckpt checkpoints/flat_task_ppo_poserand_v2_cont_best.pt
    modal run --detach scripts/train_multicube_stack_ppo.py --n-workers 1 --n-iterations 3 --n-rollouts 8
                                                              # quick sanity check, no process pool
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(image=rl_image, gpu=None, cpu=8.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=3600 * 6)
def train_multicube_stack_ppo(
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
    lr: float = 3e-4,
    entropy_coef: float = 0.01,
    clip_eps: float = 0.2,
    n_epochs: int = 4,
    early_stop_patience: int = 3,
    early_stop_threshold: float = 0.6,
    warm_start_ckpt: str = "checkpoints/flat_task_ppo_poserand_v2_cont_best.pt",
    warm_start_log_std_init: float = -2.0,
    critic_warmup_iters: int = 5,
    discrepancy_weight: float = 2.0,
    drag_weight: float = 20.0,
    success_bonus: float = 5.0,
    stillness_weight: float = 3.0,
    precision_weight: float = 10.0,
    disturbance_weight: float = 15.0,
    randomize_pose_prob: float = 1.0,
    pose_exclude_band: float = 0.35,
    pose_max_frac: float = 0.85,
    min_cubes: int = 1,
    max_cubes: int = 3,
    ckpt_name: str = "multicube_stack_ppo",
) -> dict:
    import os
    import math
    import numpy as np
    import torch

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium_robotics  # noqa: F401

    from think_then_act.training.subgoal_features import FLAT_OBS_DIM
    from think_then_act.training.flat_task_ppo import FlatTaskPPOConfig, FlatTaskPPOTrainer
    from think_then_act.training.multicube_stack_env import MultiCubeStackEnv

    print("\n" + "=" * 60)
    print("  MULTI-CUBE STACK PPO (reward-driven, warm-started from single-cube PPO)")
    print("=" * 60)

    torch.manual_seed(seed)

    weights_kwargs = dict(discrepancy_weight=discrepancy_weight, drag_weight=drag_weight,
                           success_bonus=success_bonus, stillness_weight=stillness_weight)
    config = FlatTaskPPOConfig(
        obs_dim=FLAT_OBS_DIM, max_episode_steps=max_episode_steps, n_rollouts=n_rollouts,
        n_workers=n_workers, rnn_hidden_size=rnn_hidden_size, episodes_per_minibatch=episodes_per_minibatch,
        gamma=gamma, gae_lambda=gae_lambda, lr=lr, entropy_coef=entropy_coef, clip_eps=clip_eps,
        n_epochs=n_epochs, weights_kwargs=weights_kwargs,
        randomize_pose_prob=randomize_pose_prob, pose_exclude_band=pose_exclude_band, pose_max_frac=pose_max_frac,
        env_variant="multicube", min_cubes=min_cubes, max_cubes=max_cubes,
        precision_weight=precision_weight, disturbance_weight=disturbance_weight,
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
          f"precision_weight={precision_weight}  disturbance_weight={disturbance_weight}")

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
        """
        Fixed held-out seeds (300_000+ep, a range distinct from every
        other eval convention in this project) against MultiCubeStackEnv
        directly. Reports BOTH the strict full-stack completion_rate
        (every active cube in the episode genuinely placed) and a
        graduated cube_placement_rate (total cubes placed / total cubes
        across all episodes) -- the strict metric will likely be noisy
        and near-zero early in training given episodes can have up to
        max_cubes cubes each needing a genuine success in sequence; the
        graduated one is the more informative early-progress signal.
        """
        actor = actor if actor is not None else trainer.actor
        eval_env = MultiCubeStackEnv(
            min_cubes=min_cubes, max_cubes=max_cubes, max_episode_steps=max_episode_steps,
            randomize_pose_prob=1.0, pose_exclude_band=pose_exclude_band, pose_max_frac=pose_max_frac,
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
            metrics = trainer.train_iteration(env_kwargs, i)
            history.append(metrics)

            # Cheap (no eval) checkpoint EVERY iteration, decoupled from the
            # costly eval+named-checkpoint cadence below -- running on
            # preemptible Modal compute by deliberate choice (cheaper, but
            # can be reclaimed at any point with no warning -- confirmed via
            # modal.App.function's own nonpreemptible=False default, see
            # memory: flat_policy_ppo_generalization's recurring-
            # cancellation section), so minimizing how much training this
            # loses per preemption matters more than it would on reserved
            # compute. torch.save of this small GRU (hidden_dim=64) is fast;
            # the volume commit is the real cost, worth paying every
            # iteration specifically BECAUSE that's the only way a
            # checkpoint survives the container actually being reclaimed.
            trainer.save_checkpoint(latest_ckpt_path)
            model_volume.commit()

            # Written to the volume every iteration -- NOT just printed to
            # stdout -- after two separate occasions this session where the
            # background-task log capture silently lost most of a run's
            # printed progress (only the tail/traceback survived), leaving
            # no way to reconstruct the loss/eval curve after the fact. A
            # JSON file on the volume survives exactly the same failure
            # modes the checkpoints already do (preemption, connection
            # loss) since it's committed on the same cadence.
            history_path = os.path.join(ckpt_dir, f"{ckpt_name}_history.json")
            with open(history_path, "w") as f:
                import json
                json.dump({"history": history, "eval_history": eval_history,
                           "lr": lr, "ckpt_name": ckpt_name}, f, default=float)

            if (i + 1) % 5 == 0:
                print(f"  iter {i+1}/{n_iterations}  policy_loss={metrics['policy_loss']:.4f}  "
                      f"value_loss={metrics['value_loss']:.4f}  mean_reward={metrics['mean_reward']:.4f}  "
                      f"entropy={metrics['mean_entropy']:.4f}  approx_kl={metrics['approx_kl']:.4f}  "
                      f"clip_frac={metrics['clip_fraction']:.3f}  "
                      f"train_genuine_rate={metrics['genuine_success_rate']:.2f}  "
                      f"collect_s={metrics['collect_s']:.2f}  update_s={metrics['update_s']:.2f}")

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
    rnn_hidden_size: int = 64,
    episodes_per_minibatch: int = 16,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    lr: float = 3e-4,
    entropy_coef: float = 0.01,
    clip_eps: float = 0.2,
    n_epochs: int = 4,
    early_stop_patience: int = 3,
    early_stop_threshold: float = 0.6,
    warm_start_ckpt: str = "checkpoints/flat_task_ppo_poserand_v2_cont_best.pt",
    warm_start_log_std_init: float = -2.0,
    critic_warmup_iters: int = 5,
    discrepancy_weight: float = 2.0,
    drag_weight: float = 20.0,
    success_bonus: float = 5.0,
    stillness_weight: float = 3.0,
    precision_weight: float = 10.0,
    disturbance_weight: float = 15.0,
    randomize_pose_prob: float = 1.0,
    pose_exclude_band: float = 0.35,
    pose_max_frac: float = 0.85,
    min_cubes: int = 1,
    max_cubes: int = 3,
    ckpt_name: str = "multicube_stack_ppo",
):
    result = train_multicube_stack_ppo.remote(
        n_iterations=n_iterations, max_episode_steps=max_episode_steps, eval_episodes=eval_episodes,
        checkpoint_every=checkpoint_every, seed=seed, n_rollouts=n_rollouts, n_workers=n_workers,
        rnn_hidden_size=rnn_hidden_size, episodes_per_minibatch=episodes_per_minibatch, gamma=gamma,
        gae_lambda=gae_lambda, lr=lr, entropy_coef=entropy_coef, clip_eps=clip_eps, n_epochs=n_epochs,
        early_stop_patience=early_stop_patience, early_stop_threshold=early_stop_threshold,
        warm_start_ckpt=warm_start_ckpt, warm_start_log_std_init=warm_start_log_std_init,
        critic_warmup_iters=critic_warmup_iters, discrepancy_weight=discrepancy_weight,
        drag_weight=drag_weight, success_bonus=success_bonus, stillness_weight=stillness_weight,
        precision_weight=precision_weight, disturbance_weight=disturbance_weight,
        randomize_pose_prob=randomize_pose_prob, pose_exclude_band=pose_exclude_band, pose_max_frac=pose_max_frac,
        min_cubes=min_cubes, max_cubes=max_cubes, ckpt_name=ckpt_name,
    )
    print("\n", result)
