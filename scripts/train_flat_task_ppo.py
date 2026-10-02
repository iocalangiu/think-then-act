"""
train_flat_task_ppo.py

PPO fine-tuning of the flat full-task MSE BC policy (training/flat_task_
ppo.py's FlatTaskPPOTrainer, against training/flat_task_env.py's
FlatTaskEnv) — built 2026-10-02 specifically to correct two behaviors
found in BC rollouts via scripts/diagnose_drag_mechanism.py: the block
getting dragged along the table instead of lifted cleanly, and the arm
visibly moving while the gripper's own net position barely changes. See
reward/flat_task_reward.py's docstring for exactly how each is targeted
(a discrepancy penalty for the second, a drag penalty gated on genuine-
grasp state for the first) and why an earlier attempt to fix the first one
by having the ENV silently override a detected-blocked action was tested
directly and ruled out (zero effect on outcomes — a frozen policy can't
learn from an action it never sees was changed).

Same warm-start recipe as train_low_level_ppo_recurrent.py (critic-only
warmup + low initial log_std when loading a BC checkpoint) — that
recipe's own rationale (a BC checkpoint's critic is untrained, and full-
noise exploration on an already-precise warm-started actor erodes it
before training improves anything) applies identically here, since the
warm-start source is exactly the same kind of checkpoint
(flat_bc_multi_head.py's policy_type="mse" BC trainer).

Eval uses the SAME genuine-grasp verification and fixed held-out seed
convention (100_000+ep) as run_bc_scaling_cell/eval_elevated_target, for
direct comparability against those numbers — NOT the shaped training
reward, so a reward hack can't inflate the reported completion_rate.

Run with:
    modal run --detach scripts/train_flat_task_ppo.py \\
        --warm-start-ckpt checkpoints/bc_scaling/mse_n2000_seed0_smw0.0_k5.pt
    modal run --detach scripts/train_flat_task_ppo.py --n-workers 1 --n-iterations 3 --n-rollouts 8
                                                              # quick sanity check, no process pool
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(image=rl_image, gpu=None, cpu=8.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=3600 * 4)
def train_flat_task_ppo(
    n_iterations: int = 300,
    max_episode_steps: int = 100,
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
    early_stop_threshold: float = 0.7,
    warm_start_ckpt: str = "checkpoints/bc_scaling/mse_n2000_seed0_smw0.0_k5.pt",
    warm_start_log_std_init: float = -2.0,
    critic_warmup_iters: int = 5,
    discrepancy_weight: float = 2.0,
    drag_weight: float = 20.0,
    success_bonus: float = 5.0,
    ckpt_name: str = "flat_task_ppo",
) -> dict:
    import os
    import glob
    import math
    import re
    import numpy as np
    import torch

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env, init_random_episode, grip_contact_forces
    from think_then_act.training.subgoal_features import build_flat_observation, FLAT_OBS_DIM
    from think_then_act.training.flat_task_ppo import FlatTaskPPOConfig, FlatTaskPPOTrainer

    print("\n" + "=" * 60)
    print("  FLAT FULL-TASK PPO (warm-started from BC)")
    print("=" * 60)

    torch.manual_seed(seed)

    weights_kwargs = dict(discrepancy_weight=discrepancy_weight, drag_weight=drag_weight,
                           success_bonus=success_bonus)
    config = FlatTaskPPOConfig(
        obs_dim=FLAT_OBS_DIM, max_episode_steps=max_episode_steps, n_rollouts=n_rollouts,
        n_workers=n_workers, rnn_hidden_size=rnn_hidden_size, episodes_per_minibatch=episodes_per_minibatch,
        gamma=gamma, gae_lambda=gae_lambda, lr=lr, entropy_coef=entropy_coef, clip_eps=clip_eps,
        n_epochs=n_epochs, weights_kwargs=weights_kwargs,
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
        Genuine-grasp completion_rate on a FRESH plain FetchPickAndPlace-v3
        (not FlatTaskEnv's shaped reward) — same verification and fixed
        held-out seed convention (100_000+ep) as run_bc_scaling_cell/
        eval_elevated_target, for direct comparability.
        """
        actor = actor if actor is not None else trainer.actor
        eval_env = gym.make("FetchPickAndPlace-v3", max_episode_steps=max_episode_steps)
        setup_env(eval_env)
        n_genuine = 0
        n_raw = 0
        for ep in range(n_eval_episodes):
            rng = np.random.default_rng(100_000 + ep)
            eval_env.reset(seed=100_000 + ep)
            obs, setup_ok = init_random_episode(eval_env, rng)
            if not setup_ok:
                continue
            hidden_state = None
            success = False
            ever_lifted_and_gripped = False
            for _ in range(max_episode_steps):
                flat_obs = build_flat_observation(obs["observation"], obs["achieved_goal"], obs["desired_goal"])
                action, hidden_state = actor.act(flat_obs, hidden_state, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                height_above_resting = float(obs["achieved_goal"][2]) - 0.425
                forces = grip_contact_forces(eval_env)
                if min(forces["left"], forces["right"]) > 0.0 and height_above_resting > 0.02:
                    ever_lifted_and_gripped = True
                if info.get("is_success", False):
                    success = True
                if terminated or truncated:
                    break
            if success:
                n_raw += 1
            if success and ever_lifted_and_gripped:
                n_genuine += 1
        eval_env.close()
        return {"completion_rate": n_genuine / n_eval_episodes if n_eval_episodes else 0.0,
                "raw_success_rate": n_raw / n_eval_episodes if n_eval_episodes else 0.0}

    ckpt_dir = os.path.join(MODEL_CACHE_DIR, "checkpoints")
    best_ckpt_path = os.path.join(ckpt_dir, f"{ckpt_name}_best.pt")
    initial_eval = run_eval()
    print(f"  [eval @ iter 0 / warm-start] completion_rate={initial_eval['completion_rate']:.1%} "
          f"raw_success_rate={initial_eval['raw_success_rate']:.1%}")
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

    history = []
    eval_history = [{"iteration": 0, **initial_eval}]
    consecutive_at_threshold = 0
    stopped_early_at = None
    try:
        for i in range(n_iterations):
            metrics = trainer.train_iteration(env_kwargs, i)
            history.append(metrics)

            if (i + 1) % 5 == 0:
                print(f"  iter {i+1}/{n_iterations}  policy_loss={metrics['policy_loss']:.4f}  "
                      f"value_loss={metrics['value_loss']:.4f}  mean_reward={metrics['mean_reward']:.4f}  "
                      f"entropy={metrics['mean_entropy']:.4f}  approx_kl={metrics['approx_kl']:.4f}  "
                      f"clip_frac={metrics['clip_fraction']:.3f}  "
                      f"train_genuine_rate={metrics['genuine_success_rate']:.2f}  "
                      f"mean_discrepancy={metrics['mean_discrepancy']:.4f}  "
                      f"mean_drag_steps={metrics['mean_drag_steps']:.2f}  "
                      f"collect_s={metrics['collect_s']:.2f}  update_s={metrics['update_s']:.2f}")

            if (i + 1) % checkpoint_every == 0:
                ckpt = os.path.join(ckpt_dir, f"{ckpt_name}_iter{i+1}.pt")
                trainer.save_checkpoint(ckpt)
                model_volume.commit()

                eval_result = run_eval()
                maybe_save_best(eval_result["completion_rate"])
                eval_history.append({"iteration": i + 1, **eval_result})
                print(f"  [eval @ iter {i+1}] completion_rate={eval_result['completion_rate']:.1%} "
                      f"raw_success_rate={eval_result['raw_success_rate']:.1%} over {eval_episodes} episodes")

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

        print(f"  [eval] final completion_rate={final_eval['completion_rate']:.1%}")
        print(f"  best  completion_rate={best_completion_rate:.1%} -> {best_ckpt_path}")
        return {
            "status": "PASS", "ckpt_path": ckpt_out, "best_ckpt_path": best_ckpt_path,
            "completion_rate": round(final_eval["completion_rate"], 4),
            "best_completion_rate": round(best_completion_rate, 4),
            "eval_history": eval_history, "stopped_early_at": stopped_early_at,
        }
    finally:
        trainer.close_pool()


@app.local_entrypoint()
def main(
    n_iterations: int = 300,
    max_episode_steps: int = 100,
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
    early_stop_threshold: float = 0.7,
    warm_start_ckpt: str = "checkpoints/bc_scaling/mse_n2000_seed0_smw0.0_k5.pt",
    warm_start_log_std_init: float = -2.0,
    critic_warmup_iters: int = 5,
    discrepancy_weight: float = 2.0,
    drag_weight: float = 20.0,
    success_bonus: float = 5.0,
    ckpt_name: str = "flat_task_ppo",
):
    result = train_flat_task_ppo.remote(
        n_iterations=n_iterations, max_episode_steps=max_episode_steps, eval_episodes=eval_episodes,
        checkpoint_every=checkpoint_every, seed=seed, n_rollouts=n_rollouts, n_workers=n_workers,
        rnn_hidden_size=rnn_hidden_size, episodes_per_minibatch=episodes_per_minibatch, gamma=gamma,
        gae_lambda=gae_lambda, lr=lr, entropy_coef=entropy_coef, clip_eps=clip_eps, n_epochs=n_epochs,
        early_stop_patience=early_stop_patience, early_stop_threshold=early_stop_threshold,
        warm_start_ckpt=warm_start_ckpt, warm_start_log_std_init=warm_start_log_std_init,
        critic_warmup_iters=critic_warmup_iters, discrepancy_weight=discrepancy_weight,
        drag_weight=drag_weight, success_bonus=success_bonus, ckpt_name=ckpt_name,
    )
    print("\n", result)
