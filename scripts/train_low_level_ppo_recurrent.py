"""
train_low_level_ppo_recurrent.py

Meta-RL-style recurrent variant of train_low_level_ppo.py — trains
SubgoalRecurrentPolicy (policy/subgoal_recurrent_policy.py) via
RecurrentLowLevelPPOTrainer (training/low_level_ppo_recurrent.py) against
SingularityForceAugmentedEnv (training/singularity_force_env.py) instead of
the plain SubgoalConditionedEnv. See memory: meta_rl_sim2real_direction.

Zero import dependency on train_low_level_ppo.py — every helper here
(curriculum staging, early-stopping, best-checkpoint seeding, resume) is
copied, not imported, so nothing in that script's default path can be
regressed by this one, and vice versa. train_low_level_ppo.py's existing
checkpoints, training runs, and behavior are completely untouched by this
file's existence.

Checkpoints use a `_ppo_rnn` suffix (low_level_{subgoal}_ppo_rnn.pt etc.)
so they never collide with either the GRPO (`_grpo`, if that ever wrote
here) or plain-PPO (`_ppo`) checkpoints on the same volume — all three can
be run/compared side by side.

Run with:
    modal run --detach scripts/train_low_level_ppo_recurrent.py --subgoals align_xy
    modal run --detach scripts/train_low_level_ppo_recurrent.py --subgoals align_xy --n-workers 1 --n-iterations 3 --n-rollouts 8
                                                              # quick sanity check, no process pool
    modal run --detach scripts/train_low_level_ppo_recurrent.py --subgoals align_xy \\
        --enable-singularity-perturbation --enable-force-perturbation
                                                              # full proposed system (ablation cell F)
    modal run --detach scripts/train_low_level_ppo_recurrent.py --subgoals align_xy \\
        --enable-singularity-perturbation --enable-force-perturbation \\
        --no-include-discrepancy-obs --no-include-force-obs
                                                              # recurrence alone, no augmented obs (cell C)

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
def train_low_level_ppo_recurrent(
    subgoals: str = "align_xy,descend,close_gripper,lift,move_to_target,release",
    n_iterations: int = 300,
    max_episode_steps: int = 30,
    eval_episodes: int = 10,
    checkpoint_every: int = 50,
    seed: int = 0,
    resume: bool = False,
    resume_ckpt_iter: int = 0,
    n_rollouts: int = 0,       # 0 = RecurrentLowLevelPPOConfig default (64)
    n_workers: int = 8,        # 1 = serial (no process pool)
    rnn_hidden_size: int = 0,  # 0 = default (64)
    episodes_per_minibatch: int = 0,   # 0 = default (16)
    gamma: float = 0.0,
    gae_lambda: float = -1.0,
    lr: float = 0.0,
    entropy_coef: float = -1.0,
    clip_eps: float = 0.0,
    n_epochs: int = 0,
    early_stop_patience: int = 2,
    early_stop_threshold: float = 1.0,
    randomize_block_size: bool = False,
    warm_start_ckpt: str = "",
    warm_start_log_std_init: float = -2.0,   # applied ONLY when warm_start_ckpt is set —
                                              # a warm-started actor's log_std is whatever the
                                              # checkpoint saved (BC checkpoints save it
                                              # UNTRAINED at 0, i.e. std=1, since BC's loss
                                              # never touches log_std at all — see
                                              # behavioral_cloning.py). std=1 exploration
                                              # noise layered on an already-precise
                                              # warm-started actor can erode it before
                                              # training ever improves anything (found
                                              # 2026-09-18: a BC-warm-started align_xy run
                                              # regressed from ~45% to 0% completion_rate
                                              # within 50 iterations with entropy never
                                              # moving off its std=1 initial value). Pass
                                              # 0.0 to keep whatever the checkpoint has.
    critic_warmup_iters: int = 5,   # ALSO only applied when warm_start_ckpt is set — runs
                                     # this many critic-ONLY update iterations (actor
                                     # completely untouched, see
                                     # RecurrentLowLevelPPOTrainer.critic_warmup_step)
                                     # before the first real ppo_step, since a warm-started
                                     # actor's checkpoint critic is fresh/untrained (BC has
                                     # no value function of its own) and would otherwise
                                     # feed the first several PPO updates near-random GAE
                                     # advantages. 0 disables.
    use_pose_model: bool = True,
    pos_noise_std: float = 0.0,
    # -- observation augmentation / perturbation, all default to the
    # underlying wrapper's own "off/unchanged" values --
    include_discrepancy_obs: bool = True,
    include_force_obs: bool = True,
    action_scale_m: float = 0.05,
    enable_singularity_perturbation: bool = False,
    enable_force_perturbation: bool = False,
    attenuation_range_low: float = 0.3, attenuation_range_high: float = 1.0,
    leak_range_low: float = -0.3, leak_range_high: float = 0.3,
    onset_step_range_low: int = 0, onset_step_range_high: int = 15,
    duration_range_low: int = 5, duration_range_high: int = 30,
    force_scale_range_low: float = 0.5, force_scale_range_high: float = 2.0,
    force_noise_std_range_low: float = 0.0, force_noise_std_range_high: float = 0.5,
    force_offset_range_low: float = -0.2, force_offset_range_high: float = 0.2,
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
    from think_then_act.perception.block_pose_predictor import BlockPosePredictor
    from think_then_act.perception.collision_predictor import CollisionPredictor
    from think_then_act.policy.subgoal_policy import SubgoalGaussianPolicy
    from think_then_act.policy.subgoal_recurrent_policy import SubgoalRecurrentPolicy
    from think_then_act.reward.subgoal_reward import SUBGOAL_LABELS
    from think_then_act.training.subgoal_env import SubgoalConditionedEnv
    from think_then_act.training.subgoal_features import obs_dim_for_subgoal
    from think_then_act.training.singularity_force_env import SingularityForceAugmentedEnv
    from think_then_act.training.low_level_ppo_recurrent import (
        RecurrentLowLevelPPOConfig, RecurrentLowLevelPPOTrainer,
    )

    print("\n" + "=" * 60)
    print("  LOW-LEVEL CONTROLLER TRAINING (recurrent PPO+GAE, meta-RL, per-subgoal)")
    print("=" * 60)

    torch.manual_seed(seed)

    subgoal_list = [s.strip() for s in subgoals.split(",") if s.strip()]
    for s in subgoal_list:
        if s not in SUBGOAL_LABELS:
            raise ValueError(f"Unknown subgoal {s!r}; must be one of {SUBGOAL_LABELS}")

    singularity_kwargs = dict(
        attenuation_range=(attenuation_range_low, attenuation_range_high),
        leak_range=(leak_range_low, leak_range_high),
        onset_step_range=(onset_step_range_low, onset_step_range_high),
        duration_range=(duration_range_low, duration_range_high),
    )
    force_kwargs = dict(
        scale_range=(force_scale_range_low, force_scale_range_high),
        noise_std_range=(force_noise_std_range_low, force_noise_std_range_high),
        offset_range=(force_offset_range_low, force_offset_range_high),
    )
    sf_env_kwargs = dict(
        include_discrepancy_obs=include_discrepancy_obs, include_force_obs=include_force_obs,
        action_scale_m=action_scale_m,
        enable_singularity_perturbation=enable_singularity_perturbation,
        enable_force_perturbation=enable_force_perturbation,
        singularity_kwargs=singularity_kwargs, force_kwargs=force_kwargs,
    )

    collision_ckpt = os.path.join(MODEL_CACHE_DIR, "checkpoints", "collision_predictor.pt")
    if os.path.exists(collision_ckpt):
        print(f"  Collision predictor checkpoint found <- {collision_ckpt}")
    else:
        collision_ckpt = None
        print(f"  No collision predictor checkpoint — 'descend' will train with collision_prob=0.0.")

    pose_ckpt = os.path.join(MODEL_CACHE_DIR, "checkpoints", "block_pose_predictor.pt")
    if not use_pose_model:
        pose_ckpt = None
        print(f"  --use-pose-model=False — ground-truth achieved_goal"
              + (f" + pos_noise_std={pos_noise_std}" if pos_noise_std > 0 else "") + " for every subgoal.")
    elif os.path.exists(pose_ckpt):
        print(f"  Block pose predictor checkpoint found <- {pose_ckpt}")
    else:
        pose_ckpt = None
        print(f"  No block pose predictor checkpoint — privileged ground-truth achieved_goal.")

    align_xy_ckpt = os.path.join(MODEL_CACHE_DIR, "checkpoints", "low_level_align_xy_ppo_best.pt")
    if os.path.exists(align_xy_ckpt):
        print(f"  Trained align_xy checkpoint found <- {align_xy_ckpt}")
    else:
        align_xy_ckpt = None
        print(f"  No trained align_xy checkpoint — 'descend' will train from a fresh scattered reset.")

    def find_resume_checkpoint(subgoal: str, trainer: RecurrentLowLevelPPOTrainer) -> int:
        ckpt_dir = os.path.join(MODEL_CACHE_DIR, "checkpoints")

        if resume_ckpt_iter > 0:
            path = os.path.join(ckpt_dir, f"low_level_{subgoal}_ppo_rnn_iter{resume_ckpt_iter}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"--resume-ckpt-iter {resume_ckpt_iter}: no such checkpoint {path}")
            trainer.load_checkpoint(path)
            print(f"  [resume] {subgoal}: forced load of {path} (iteration {resume_ckpt_iter})")
            return min(resume_ckpt_iter, n_iterations)

        candidates = []
        final_ckpt = os.path.join(ckpt_dir, f"low_level_{subgoal}_ppo_rnn.pt")
        if os.path.exists(final_ckpt):
            candidates.append((n_iterations, final_ckpt))
        for p in glob.glob(os.path.join(ckpt_dir, f"low_level_{subgoal}_ppo_rnn_iter*.pt")):
            m = re.search(r"_iter(\d+)\.pt$", p)
            if m:
                candidates.append((int(m.group(1)), p))
        candidates.sort(key=lambda t: -t[0])

        for iteration, path in candidates:
            try:
                trainer.load_checkpoint(path)
            except RuntimeError as e:
                print(f"  [resume] {path} incompatible with current actor/critic architecture ({e}) — skipping.")
                continue
            print(f"  [resume] {subgoal}: loaded {path} (iteration {iteration})")
            return min(iteration, n_iterations)

        if candidates:
            print(f"  [resume] {subgoal}: no compatible checkpoint — starting fresh from iteration 0.")
        return 0

    def make_eval_env(subgoal: str, collision_model, pose_model, align_xy_policy=None):
        base = ObservationHarness(
            gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                      max_episode_steps=max_episode_steps + 250)
        )
        setup_env(base)
        env = SubgoalConditionedEnv(
            base, subgoal=subgoal, collision_model=collision_model,
            pose_model=pose_model, align_xy_policy=align_xy_policy,
            max_episode_steps=max_episode_steps, randomize_block_size=randomize_block_size,
            pos_noise_std=pos_noise_std,
        )
        return SingularityForceAugmentedEnv(env, **sf_env_kwargs)

    # Same size/done-streak curricula as train_low_level_ppo.py — copied,
    # not imported (see module docstring). eval_env deliberately stays on
    # the FULL range/streak throughout for the same reason as that script:
    # completion_rate/early-stopping must always reflect the real target
    # robustness bar, never an eased curriculum stage.
    SIZE_CURRICULUM_STAGES = [
        (0.0,     (0.04, 0.06)),
        (1 / 3.0, (0.025, 0.07)),
        (2 / 3.0, None),
    ]

    def size_range_for_iteration(i: int):
        if not randomize_block_size:
            return None
        progress = i / max(n_iterations, 1)
        current = SIZE_CURRICULUM_STAGES[0][1]
        for start_frac, stage_range in SIZE_CURRICULUM_STAGES:
            if progress >= start_frac:
                current = stage_range
        return current

    DONE_STREAK_CURRICULUM_STAGES = [(0.0, 1), (1 / 3.0, 2), (2 / 3.0, 3)]

    def done_streak_for_iteration(i: int) -> int:
        if not randomize_block_size:
            return 3
        progress = i / max(n_iterations, 1)
        current = DONE_STREAK_CURRICULUM_STAGES[0][1]
        for start_frac, stage_streak in DONE_STREAK_CURRICULUM_STAGES:
            if progress >= start_frac:
                current = stage_streak
        return current

    results = {}

    for subgoal in subgoal_list:
        print(f"\n--- Training subgoal: {subgoal} (recurrent PPO) ---")

        obs_dim = SingularityForceAugmentedEnv.obs_dim_for(subgoal, include_discrepancy_obs, include_force_obs)
        config_kwargs = dict(obs_dim=obs_dim, max_episode_steps=max_episode_steps, n_workers=n_workers)
        if n_rollouts > 0:
            config_kwargs["n_rollouts"] = n_rollouts
        if rnn_hidden_size > 0:
            config_kwargs["rnn_hidden_size"] = rnn_hidden_size
        if episodes_per_minibatch > 0:
            config_kwargs["episodes_per_minibatch"] = episodes_per_minibatch
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
        config = RecurrentLowLevelPPOConfig(**config_kwargs)
        trainer = RecurrentLowLevelPPOTrainer(config)

        if warm_start_ckpt:
            warm_start_path = os.path.join(MODEL_CACHE_DIR, "checkpoints", warm_start_ckpt)
            trainer.load_checkpoint(warm_start_path)
            print(f"  [warm-start] {subgoal}: loaded initial weights from {warm_start_path} "
                  f"(training still starts counting at iteration 0)")

            if warm_start_log_std_init != 0.0:
                import math
                with torch.no_grad():
                    trainer.actor.log_std.fill_(warm_start_log_std_init)
                print(f"  [warm-start] {subgoal}: log_std initialized to {warm_start_log_std_init} "
                      f"(std={math.exp(warm_start_log_std_init):.3f}) instead of the checkpoint's own "
                      f"value, to avoid full-noise exploration eroding warm-started precision from "
                      f"iteration 0 (pass --warm-start-log-std-init 0 to keep the checkpoint's value).")

        env_kwargs = dict(subgoal=subgoal, max_episode_steps=max_episode_steps,
                           collision_ckpt=collision_ckpt, pose_ckpt=pose_ckpt,
                           align_xy_ckpt=align_xy_ckpt, randomize_block_size=randomize_block_size,
                           pos_noise_std=pos_noise_std, **sf_env_kwargs)

        if warm_start_ckpt and critic_warmup_iters > 0:
            print(f"  [warm-start] {subgoal}: running {critic_warmup_iters} critic-only warmup "
                  f"iteration(s) (actor untouched) before the first real PPO update...")
            # Large positive offset, disjoint from the main loop's seeds
            # (n_rollouts*iteration .. n_rollouts*(iteration+1), always well
            # under this) — np.random.default_rng requires a non-negative
            # int, so a negative-range scheme (an earlier version of this
            # code) crashed every warmup rollout with "expected non-negative
            # integer". No correctness dependency on the exact value, just
            # keeps the two phases' rollouts visibly distinct if inspected.
            WARMUP_SEED_BASE = 1_000_000_000
            for wi in range(critic_warmup_iters):
                warmup_seeds = list(range(
                    WARMUP_SEED_BASE + config.n_rollouts * wi,
                    WARMUP_SEED_BASE + config.n_rollouts * (wi + 1),
                ))
                warmup_rollouts = trainer.collect_rollouts(env_kwargs, warmup_seeds)
                warmup_metrics = trainer.critic_warmup_step(warmup_rollouts)
                print(f"    critic warmup {wi+1}/{critic_warmup_iters}: "
                      f"value_loss={warmup_metrics['value_loss']:.4f}")

        eval_collision_model = None
        if collision_ckpt is not None:
            eval_collision_model = CollisionPredictor()
            eval_collision_model.load_state_dict(torch.load(collision_ckpt, map_location="cpu"))
            eval_collision_model.eval()
        eval_pose_model = None
        if pose_ckpt is not None:
            eval_pose_model = BlockPosePredictor()
            eval_pose_model.load_state_dict(torch.load(pose_ckpt, map_location="cpu"))
            eval_pose_model.eval()
        eval_align_xy_policy = None
        if align_xy_ckpt is not None and subgoal == "descend":
            eval_align_xy_policy = SubgoalGaussianPolicy(obs_dim=obs_dim_for_subgoal("align_xy"))
            align_xy_ckpt_data = torch.load(align_xy_ckpt, map_location="cpu")
            eval_align_xy_policy.load_state_dict(
                align_xy_ckpt_data["actor"] if isinstance(align_xy_ckpt_data, dict) and "actor" in align_xy_ckpt_data
                else align_xy_ckpt_data
            )
            eval_align_xy_policy.eval()
        eval_env = make_eval_env(subgoal, eval_collision_model, eval_pose_model, eval_align_xy_policy)

        start_iteration = 0
        if resume:
            start_iteration = find_resume_checkpoint(subgoal, trainer)

        def run_eval(actor=None) -> float:
            # Recurrent-specific: actor.act returns (action, next_hidden_state),
            # and hidden_state resets to None at the start of EVERY eval
            # episode — same whole-episode-BPTT scope as training.
            actor = actor if actor is not None else trainer.actor
            completions = []
            for ep in range(eval_episodes):
                rng = np.random.default_rng(90_000 + ep)
                obs, info = eval_env.reset(rng=rng)
                hidden_state = None
                success = False
                for _ in range(max_episode_steps):
                    action, hidden_state = actor.act(obs, hidden_state, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    if info.get("done", False):
                        success = True
                    if terminated or truncated:
                        break
                completions.append(float(success))
            return float(np.mean(completions))

        best_ckpt_path = os.path.join(MODEL_CACHE_DIR, "checkpoints", f"low_level_{subgoal}_ppo_rnn_best.pt")
        best_completion_rate = -1.0
        if os.path.exists(best_ckpt_path):
            try:
                probe_actor = SubgoalRecurrentPolicy(obs_dim=obs_dim, rnn_hidden_size=config.rnn_hidden_size)
                probe_ckpt = torch.load(best_ckpt_path, map_location="cpu")
                probe_actor.load_state_dict(probe_ckpt["actor"] if isinstance(probe_ckpt, dict) and "actor" in probe_ckpt else probe_ckpt)
                probe_actor.eval()
                best_completion_rate = run_eval(probe_actor)
                print(f"  [resume] {subgoal}: existing best checkpoint completion_rate={best_completion_rate:.1%} "
                      f"(seeded from {best_ckpt_path})")
            except RuntimeError as e:
                print(f"  [resume] {subgoal}: existing {best_ckpt_path} incompatible with the "
                      f"current actor architecture ({e}) — not seeding, starting from -1.0.")

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
                    "ckpt_path"           : os.path.join(MODEL_CACHE_DIR, "checkpoints", f"low_level_{subgoal}_ppo_rnn.pt"),
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
                new_size_range = size_range_for_iteration(i)
                new_done_streak = done_streak_for_iteration(i)
                if (new_size_range != env_kwargs.get("size_range")
                        or new_done_streak != env_kwargs.get("done_streak")):
                    env_kwargs = dict(env_kwargs, size_range=new_size_range, done_streak=new_done_streak)
                    print(f"  [curriculum] {subgoal}: size_range -> "
                          f"{new_size_range or 'full [0.01, 0.08]'}  done_streak -> {new_done_streak} at iter {i}")
                metrics = trainer.train_iteration(env_kwargs, i)
                history.append(metrics)

                if (i + 1) % 10 == 0:
                    def _fmt(key):
                        v = metrics.get(key)
                        return f"{v:.4f}" if v is not None else "n/a"
                    print(f"  iter {i+1}/{n_iterations}  policy_loss={metrics['policy_loss']:.4f}  "
                          f"value_loss={metrics['value_loss']:.4f}  mean_reward={metrics['mean_reward']:.4f}  "
                          f"entropy={metrics['mean_entropy']:.4f}  approx_kl={metrics['approx_kl']:.4f}  "
                          f"clip_frac={metrics['clip_fraction']:.3f}  "
                          f"discrepancy_norm={_fmt('mean_discrepancy_norm')}  "
                          f"force_onset_rate={metrics['force_onset_rate']:.2f}  "
                          f"recovery_steps={_fmt('mean_recovery_steps')}  "
                          f"collect_s={metrics['collect_s']:.2f}  update_s={metrics['update_s']:.2f}")

                if (i + 1) % checkpoint_every == 0:
                    ckpt = os.path.join(MODEL_CACHE_DIR, "checkpoints", f"low_level_{subgoal}_ppo_rnn_iter{i+1}.pt")
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

            ckpt_out = os.path.join(MODEL_CACHE_DIR, "checkpoints", f"low_level_{subgoal}_ppo_rnn.pt")
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
    max_episode_steps: int = 30,
    resume: bool = False,
    resume_ckpt_iter: int = 0,
    n_rollouts: int = 0,
    n_workers: int = 8,
    rnn_hidden_size: int = 0,
    episodes_per_minibatch: int = 0,
    gamma: float = 0.0,
    gae_lambda: float = -1.0,
    lr: float = 0.0,
    entropy_coef: float = -1.0,
    clip_eps: float = 0.0,
    n_epochs: int = 0,
    early_stop_patience: int = 2,
    early_stop_threshold: float = 1.0,
    randomize_block_size: bool = False,
    warm_start_ckpt: str = "",
    warm_start_log_std_init: float = -2.0,
    critic_warmup_iters: int = 5,
    use_pose_model: bool = True,
    pos_noise_std: float = 0.0,
    include_discrepancy_obs: bool = True,
    include_force_obs: bool = True,
    action_scale_m: float = 0.05,
    enable_singularity_perturbation: bool = False,
    enable_force_perturbation: bool = False,
    attenuation_range_low: float = 0.3, attenuation_range_high: float = 1.0,
    leak_range_low: float = -0.3, leak_range_high: float = 0.3,
    onset_step_range_low: int = 0, onset_step_range_high: int = 15,
    duration_range_low: int = 5, duration_range_high: int = 30,
    force_scale_range_low: float = 0.5, force_scale_range_high: float = 2.0,
    force_noise_std_range_low: float = 0.0, force_noise_std_range_high: float = 0.5,
    force_offset_range_low: float = -0.2, force_offset_range_high: float = 0.2,
):
    print(f"\nDispatching recurrent meta-RL PPO+GAE low-level controller training to Modal (CPU)...")
    print(f"  subgoals={subgoals}  n_iterations={n_iterations}  n_workers={n_workers}  "
          f"include_discrepancy_obs={include_discrepancy_obs}  include_force_obs={include_force_obs}  "
          f"enable_singularity_perturbation={enable_singularity_perturbation}  "
          f"enable_force_perturbation={enable_force_perturbation}\n")
    handle = train_low_level_ppo_recurrent.spawn(
        subgoals=subgoals, n_iterations=n_iterations, max_episode_steps=max_episode_steps,
        resume=resume, resume_ckpt_iter=resume_ckpt_iter,
        n_rollouts=n_rollouts, n_workers=n_workers, rnn_hidden_size=rnn_hidden_size,
        episodes_per_minibatch=episodes_per_minibatch,
        gamma=gamma, gae_lambda=gae_lambda, lr=lr, entropy_coef=entropy_coef,
        clip_eps=clip_eps, n_epochs=n_epochs,
        early_stop_patience=early_stop_patience, early_stop_threshold=early_stop_threshold,
        randomize_block_size=randomize_block_size, warm_start_ckpt=warm_start_ckpt,
        warm_start_log_std_init=warm_start_log_std_init, critic_warmup_iters=critic_warmup_iters,
        use_pose_model=use_pose_model, pos_noise_std=pos_noise_std,
        include_discrepancy_obs=include_discrepancy_obs, include_force_obs=include_force_obs,
        action_scale_m=action_scale_m,
        enable_singularity_perturbation=enable_singularity_perturbation,
        enable_force_perturbation=enable_force_perturbation,
        attenuation_range_low=attenuation_range_low, attenuation_range_high=attenuation_range_high,
        leak_range_low=leak_range_low, leak_range_high=leak_range_high,
        onset_step_range_low=onset_step_range_low, onset_step_range_high=onset_step_range_high,
        duration_range_low=duration_range_low, duration_range_high=duration_range_high,
        force_scale_range_low=force_scale_range_low, force_scale_range_high=force_scale_range_high,
        force_noise_std_range_low=force_noise_std_range_low, force_noise_std_range_high=force_noise_std_range_high,
        force_offset_range_low=force_offset_range_low, force_offset_range_high=force_offset_range_high,
    )
    print(f"Job spawned. Function call ID: {handle.object_id}")
    print(f"Monitor at https://modal.com")
