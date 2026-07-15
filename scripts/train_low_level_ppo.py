"""
train_low_level_ppo.py

PPO+GAE variant of train_low_level.py — trains one controller per subgoal
using training/low_level_ppo.py instead of the GRPO trainer (which stays
in place, unused; see bugs_and_fixes memory, 2026-07-14, for why GRPO's
whole-episode advantage wasn't a good fit for this dense-per-timestep-
reward, multi-step task, and why a real critic + GAE is the standard tool
for this regime instead).

Checkpoints use a `_ppo` suffix (low_level_{subgoal}_ppo.pt etc.) so they
never collide with train_low_level.py's GRPO checkpoints on the same
volume — both can be run/compared side by side.

Rollout collection is parallelized across a persistent worker-process pool
(training/rollout_workers.py) — collection dominates >99% of a training
iteration's wall time (observed under GRPO, same physics/rendering cost
here), so this is requested with `cpu=8.0` to actually have that many
cores available. --n-workers should not exceed the `cpu=` value below
unless that's also raised.

Run with:
    modal run --detach scripts/train_low_level_ppo.py
    modal run --detach scripts/train_low_level_ppo.py --subgoals align_xy --n-workers 8
    modal run --detach scripts/train_low_level_ppo.py --subgoals align_xy --n-iterations 100 --n-workers 1
                                                              # --n-workers 1 = serial,
                                                              # no process pool — useful
                                                              # for a quick sanity check
                                                              # before a long parallel run
    modal run --detach scripts/train_low_level_ppo.py --resume
    modal run --detach scripts/train_low_level_ppo.py --subgoals close_gripper,lift,move_to_target,release
                                                              # skip subgoals already
                                                              # finished in an earlier run

Early stopping: a subgoal stops (and moves to the next) once completion_rate
holds >= --early-stop-threshold (default 1.0) for --early-stop-patience
(default 2) CONSECUTIVE eval checkpoints — not just one, since a single
checkpoint's completion_rate is an eval over only `eval_episodes` (default
10) seeds and can be a fluke (seen with GRPO: a lone 10% blip that reverted
next checkpoint). Pass --early-stop-patience 0 to disable and always run
the full --n-iterations.

Download checkpoints with:
    python3 -m modal volume get rl-harness-model-cache checkpoints/ ./artifacts/checkpoints/
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(
    image=rl_image,
    gpu=None,
    cpu=8.0,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=3600 * 4,
)
def train_low_level_ppo(
    subgoals: str = "align_xy,descend,close_gripper,lift,move_to_target,release",
    n_iterations: int = 300,
    max_episode_steps: int = 30,
    eval_episodes: int = 10,
    checkpoint_every: int = 50,
    seed: int = 0,
    resume: bool = False,
    resume_ckpt_iter: int = 0,
    n_rollouts: int = 0,       # 0 = LowLevelPPOConfig default (64)
    n_workers: int = 8,        # 1 = serial (no process pool)
    gamma: float = 0.0,        # 0 = default (0.99); a real gamma=0 override isn't
                               # meaningful for this reward, so 0 means "unset"
    gae_lambda: float = -1.0,  # -1 = default (0.95); 0 is a legitimate override
    lr: float = 0.0,           # 0 = default (3e-4)
    entropy_coef: float = -1.0,     # -1 = default (0.01); 0 is a legitimate override
    clip_eps: float = 0.0,     # 0 = default (0.2)
    n_epochs: int = 0,         # 0 = default (4)
    minibatch_size: int = 0,   # 0 = default (256)
    early_stop_patience: int = 2,     # consecutive eval checkpoints at/above threshold
                                      # before stopping early; 0 disables early stopping
    early_stop_threshold: float = 1.0,
) -> dict:
    import os
    import glob
    import re
    import numpy as np
    import torch

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.perception.collision_predictor import CollisionPredictor
    from think_then_act.reward.subgoal_reward import SUBGOAL_LABELS
    from think_then_act.training.subgoal_env import SubgoalConditionedEnv
    from think_then_act.training.subgoal_features import SUBGOAL_OBS_DIM
    from think_then_act.training.low_level_ppo import LowLevelPPOConfig, LowLevelPPOTrainer

    print("\n" + "=" * 60)
    print("  LOW-LEVEL CONTROLLER TRAINING (PPO+GAE, per-subgoal)")
    print("=" * 60)

    torch.manual_seed(seed)

    subgoal_list = [s.strip() for s in subgoals.split(",") if s.strip()]
    for s in subgoal_list:
        if s not in SUBGOAL_LABELS:
            raise ValueError(f"Unknown subgoal {s!r}; must be one of {SUBGOAL_LABELS}")

    # ------------------------------------------------------------------
    # Collision predictor (optional — only affects `descend`'s reward, and
    # only its checkpoint PATH is passed to worker processes — each worker
    # loads its own copy in rollout_workers._worker_init).
    # ------------------------------------------------------------------
    collision_ckpt = os.path.join(MODEL_CACHE_DIR, "checkpoints", "collision_predictor.pt")
    if os.path.exists(collision_ckpt):
        print(f"  Collision predictor checkpoint found <- {collision_ckpt}")
    else:
        collision_ckpt = None
        print(f"  No collision predictor checkpoint — "
              f"'descend' will train with collision_prob=0.0 for every step "
              f"(run train_collision_predictor.py first for the real signal).")

    def find_resume_checkpoint(subgoal: str, trainer: LowLevelPPOTrainer) -> int:
        """Same newest-compatible-checkpoint search as train_low_level.py's
        GRPO version, adapted for PPO's {"actor":..., "critic":...} checkpoint
        dict — see that script's docstring for the full rationale."""
        ckpt_dir = os.path.join(MODEL_CACHE_DIR, "checkpoints")

        if resume_ckpt_iter > 0:
            path = os.path.join(ckpt_dir, f"low_level_{subgoal}_ppo_iter{resume_ckpt_iter}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"--resume-ckpt-iter {resume_ckpt_iter}: no such checkpoint {path}")
            trainer.load_checkpoint(path)   # let this raise on mismatch
            print(f"  [resume] {subgoal}: forced load of {path} (iteration {resume_ckpt_iter})")
            return min(resume_ckpt_iter, n_iterations)

        candidates = []
        final_ckpt = os.path.join(ckpt_dir, f"low_level_{subgoal}_ppo.pt")
        if os.path.exists(final_ckpt):
            candidates.append((n_iterations, final_ckpt))
        for p in glob.glob(os.path.join(ckpt_dir, f"low_level_{subgoal}_ppo_iter*.pt")):
            m = re.search(r"_iter(\d+)\.pt$", p)
            if m:
                candidates.append((int(m.group(1)), p))
        candidates.sort(key=lambda t: -t[0])

        for iteration, path in candidates:
            try:
                trainer.load_checkpoint(path)
            except RuntimeError as e:
                print(f"  [resume] {path} incompatible with current actor/critic "
                      f"architecture ({e}) — skipping.")
                continue
            print(f"  [resume] {subgoal}: loaded {path} (iteration {iteration})")
            return min(iteration, n_iterations)

        if candidates:
            print(f"  [resume] {subgoal}: no checkpoint compatible with the current "
                  f"architecture — starting fresh from iteration 0.")
        return 0

    def make_eval_env(subgoal: str, collision_model):
        # +250, not *2: init_episode_before_subgoal's oracle pre-subgoal
        # setup (close_gripper/lift/move_to_target/release) needs headroom
        # on top of the actual episode — see rollout_workers.py's
        # _worker_init for the full rationale.
        base = ObservationHarness(
            gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                      max_episode_steps=max_episode_steps + 250)
        )
        setup_env(base)
        return SubgoalConditionedEnv(
            base, subgoal=subgoal, collision_model=collision_model,
            max_episode_steps=max_episode_steps,
        )

    results = {}

    for subgoal in subgoal_list:
        print(f"\n--- Training subgoal: {subgoal} (PPO) ---")

        config_kwargs = dict(obs_dim=SUBGOAL_OBS_DIM, max_episode_steps=max_episode_steps,
                              n_workers=n_workers)
        if n_rollouts > 0:
            config_kwargs["n_rollouts"] = n_rollouts
        if gamma > 0:
            config_kwargs["gamma"] = gamma
        if gae_lambda >= 0:
            config_kwargs["gae_lambda"] = gae_lambda
        if lr > 0:
            config_kwargs["lr"] = lr
        if entropy_coef >= 0:
            config_kwargs["entropy_coef"] = entropy_coef
        if clip_eps > 0:
            config_kwargs["clip_eps"] = clip_eps
        if n_epochs > 0:
            config_kwargs["n_epochs"] = n_epochs
        if minibatch_size > 0:
            config_kwargs["minibatch_size"] = minibatch_size
        config = LowLevelPPOConfig(**config_kwargs)
        trainer = LowLevelPPOTrainer(config)

        env_kwargs = dict(subgoal=subgoal, max_episode_steps=max_episode_steps,
                           collision_ckpt=collision_ckpt)

        # Eval uses its own plain (non-pooled) env — 10 episodes is cheap
        # sequentially, no need to spin up the worker pool for it.
        eval_collision_model = None
        if collision_ckpt is not None:
            eval_collision_model = CollisionPredictor()
            eval_collision_model.load_state_dict(torch.load(collision_ckpt, map_location="cpu"))
            eval_collision_model.eval()
        eval_env = make_eval_env(subgoal, eval_collision_model)

        start_iteration = 0
        if resume:
            start_iteration = find_resume_checkpoint(subgoal, trainer)

        # Same fixed-seed held-out eval convention as train_low_level.py —
        # see that script's comment for why FIXED (not just disjoint) matters.
        def run_eval() -> float:
            completions = []
            for ep in range(eval_episodes):
                rng = np.random.default_rng(90_000 + ep)
                obs, info = eval_env.reset(rng=rng)
                success = False
                for _ in range(max_episode_steps):
                    action = trainer.actor.act(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    if info.get("done", False):
                        success = True
                    if terminated or truncated:
                        break
                completions.append(float(success))
            return float(np.mean(completions))

        best_ckpt_path = os.path.join(MODEL_CACHE_DIR, "checkpoints", f"low_level_{subgoal}_ppo_best.pt")
        best_completion_rate = -1.0

        def maybe_save_best(completion_rate: float) -> None:
            nonlocal best_completion_rate
            if completion_rate > best_completion_rate:
                best_completion_rate = completion_rate
                trainer.save_checkpoint(best_ckpt_path)
                model_volume.commit()
                print(f"  [best] {subgoal}: completion_rate={completion_rate:.1%} -> {best_ckpt_path}")

        try:
            if start_iteration >= n_iterations:
                completion_rate = run_eval()
                maybe_save_best(completion_rate)
                print(f"  {subgoal}: already complete at iteration {start_iteration} "
                      f"(target {n_iterations}) — skipping training, eval only. "
                      f"completion_rate={completion_rate:.1%}")
                results[subgoal] = {
                    "ckpt_path"           : os.path.join(MODEL_CACHE_DIR, "checkpoints", f"low_level_{subgoal}_ppo.pt"),
                    "completion_rate"     : round(completion_rate, 4),
                    "best_ckpt_path"      : best_ckpt_path,
                    "best_completion_rate": round(best_completion_rate, 4),
                    "eval_history"        : [{"iteration": start_iteration, "completion_rate": round(completion_rate, 4)}],
                    "final_policy_loss"   : None,
                    "final_reward"        : None,
                }
                continue

            history = []
            eval_history = []
            consecutive_at_threshold = 0
            stopped_early_at = None
            for i in range(start_iteration, n_iterations):
                metrics = trainer.train_iteration(env_kwargs, i)
                history.append(metrics)

                if (i + 1) % 10 == 0:
                    print(f"  iter {i+1}/{n_iterations}  policy_loss={metrics['policy_loss']:.4f}  "
                          f"value_loss={metrics['value_loss']:.4f}  mean_reward={metrics['mean_reward']:.4f}  "
                          f"entropy={metrics['mean_entropy']:.4f}  approx_kl={metrics['approx_kl']:.4f}  "
                          f"clip_frac={metrics['clip_fraction']:.3f}  "
                          f"collect_s={metrics['collect_s']:.2f}  update_s={metrics['update_s']:.2f}")

                if (i + 1) % checkpoint_every == 0:
                    ckpt = os.path.join(MODEL_CACHE_DIR, "checkpoints", f"low_level_{subgoal}_ppo_iter{i+1}.pt")
                    trainer.save_checkpoint(ckpt)
                    model_volume.commit()

                    completion_rate = run_eval()
                    maybe_save_best(completion_rate)
                    eval_history.append({"iteration": i + 1, "completion_rate": round(completion_rate, 4)})
                    print(f"  [eval @ iter {i+1}] {subgoal}: completion_rate={completion_rate:.1%} "
                          f"over {eval_episodes} fixed-seed episodes")

                    if early_stop_patience > 0:
                        if completion_rate >= early_stop_threshold:
                            consecutive_at_threshold += 1
                        else:
                            consecutive_at_threshold = 0
                        if consecutive_at_threshold >= early_stop_patience:
                            stopped_early_at = i + 1
                            streak_start = stopped_early_at - checkpoint_every * (consecutive_at_threshold - 1)
                            print(f"  [early-stop] {subgoal}: completion_rate>={early_stop_threshold:.0%} "
                                  f"held for {consecutive_at_threshold} consecutive eval checkpoints "
                                  f"(since iter {streak_start}) — stopping at iter {stopped_early_at}, "
                                  f"skipping remaining {n_iterations - stopped_early_at} iterations.")
                            break

            final_iteration = stopped_early_at if stopped_early_at is not None else n_iterations

            ckpt_out = os.path.join(MODEL_CACHE_DIR, "checkpoints", f"low_level_{subgoal}_ppo.pt")
            trainer.save_checkpoint(ckpt_out)
            model_volume.commit()
            print(f"  Saved -> {ckpt_out}")

            if final_iteration % checkpoint_every == 0 and eval_history and eval_history[-1]["iteration"] == final_iteration:
                completion_rate = eval_history[-1]["completion_rate"]
            else:
                completion_rate = run_eval()
                maybe_save_best(completion_rate)
                eval_history.append({"iteration": final_iteration, "completion_rate": round(completion_rate, 4)})

            print(f"  [eval] {subgoal}: completion_rate={completion_rate:.1%} over {eval_episodes} episodes")
            print(f"  best  : completion_rate={best_completion_rate:.1%} -> {best_ckpt_path}")
            print(f"  eval_history: {eval_history}")
            results[subgoal] = {
                "ckpt_path"           : ckpt_out,
                "completion_rate"     : round(completion_rate, 4),
                "best_ckpt_path"      : best_ckpt_path,
                "best_completion_rate": round(best_completion_rate, 4),
                "eval_history"        : eval_history,
                "final_policy_loss"   : history[-1]["policy_loss"] if history else None,
                "final_reward"        : history[-1]["mean_reward"] if history else None,
                "stopped_early_at"    : stopped_early_at,
            }
        finally:
            # Tears down this subgoal's persistent worker pool before the
            # next subgoal builds its own (each pool's workers are pinned
            # to one subgoal's env — see rollout_workers.py).
            trainer.close_pool()
            eval_env.close()

    print("\n" + "=" * 60)
    print("Summary:")
    for subgoal, r in results.items():
        print(f"  {subgoal}: final completion_rate={r['completion_rate']:.1%}  -> {r['ckpt_path']}")
        print(f"    best  completion_rate={r['best_completion_rate']:.1%}  -> {r['best_ckpt_path']}")
    print("=" * 60)

    return {"status": "PASS", "results": results}


@app.local_entrypoint()
def main(
    subgoals: str = "align_xy,descend,close_gripper,lift,move_to_target,release",
    n_iterations: int = 300,
    resume: bool = False,
    resume_ckpt_iter: int = 0,
    n_rollouts: int = 0,
    n_workers: int = 8,
    gamma: float = 0.0,
    gae_lambda: float = -1.0,
    lr: float = 0.0,
    entropy_coef: float = -1.0,
    clip_eps: float = 0.0,
    n_epochs: int = 0,
    minibatch_size: int = 0,
    early_stop_patience: int = 2,
    early_stop_threshold: float = 1.0,
):
    print(f"\nDispatching PPO+GAE low-level controller training to Modal (CPU)...")
    print(f"  subgoals={subgoals}  n_iterations={n_iterations}  resume={resume}  "
          f"resume_ckpt_iter={resume_ckpt_iter or 'auto'}  "
          f"n_rollouts={n_rollouts or 'default'}  n_workers={n_workers}  "
          f"gamma={gamma or 'default'}  gae_lambda={'default' if gae_lambda < 0 else gae_lambda}  "
          f"lr={lr or 'default'}  entropy_coef={'default' if entropy_coef < 0 else entropy_coef}  "
          f"clip_eps={clip_eps or 'default'}  n_epochs={n_epochs or 'default'}  "
          f"minibatch_size={minibatch_size or 'default'}  "
          f"early_stop_patience={early_stop_patience}  early_stop_threshold={early_stop_threshold:.0%}\n")
    handle = train_low_level_ppo.spawn(
        subgoals=subgoals, n_iterations=n_iterations, resume=resume, resume_ckpt_iter=resume_ckpt_iter,
        n_rollouts=n_rollouts, n_workers=n_workers, gamma=gamma, gae_lambda=gae_lambda,
        lr=lr, entropy_coef=entropy_coef, clip_eps=clip_eps, n_epochs=n_epochs, minibatch_size=minibatch_size,
        early_stop_patience=early_stop_patience, early_stop_threshold=early_stop_threshold,
    )
    print(f"Job spawned. Function call ID: {handle.object_id}")
    print(f"Monitor at https://modal.com")
