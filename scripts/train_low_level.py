"""
train_low_level.py

Trains one GRPO controller per subgoal (align_xy, descend, close_gripper,
lift, move_to_target, release) on SubgoalConditionedEnv's dense reward,
using training/low_level_grpo.py — a from-scratch GRPO trainer, NOT
stable-baselines3 (tried and removed; see bugs_and_fixes memory,
2026-07-11 — SB3's torch/gymnasium version requirements repeatedly
conflicted with this project's pins). Same group-relative-advantage
algorithm as the VLM's GRPO trainer (training/grpo_trainer.py), applied to a
small Gaussian MLP instead of an autoregressive text-generating policy.

Loads the collision predictor checkpoint if train_collision_predictor.py has
already been run — otherwise `descend` trains with collision_prob fixed at
0.0 (still a valid run, just without the learned collision penalty).

No GPU needed — small MLP policy, CPU is fine and cheaper.

Run with:
    modal run --detach scripts/train_low_level.py
    modal run --detach scripts/train_low_level.py --subgoals align_xy,descend
    modal run --detach scripts/train_low_level.py --n-iterations 500
    modal run --detach scripts/train_low_level.py --resume   # continue from the latest
                                                              # checkpoint per subgoal,
                                                              # rather than from scratch
    modal run --detach scripts/train_low_level.py --resume --group-size 16 --n-states 8
                                                              # resume with a larger,
                                                              # less noisy GRPO batch
                                                              # (group_size/n_states/lr/
                                                              # entropy_coef all default
                                                              # to LowLevelGRPOConfig's
                                                              # values unless passed)

--resume picks, per subgoal, the newest checkpoint (low_level_{subgoal}.pt if the
subgoal already fully finished, else the highest low_level_{subgoal}_iterN.pt) and
tries to load it; if that checkpoint's architecture doesn't match the current
SubgoalGaussianPolicy (e.g. a stale checkpoint from before a policy code change —
this bit us 2026-07-13, an old pre-LayerNorm checkpoint sat alongside newer ones and
had a higher iteration number), it's skipped with a warning and the next-newest
compatible one is tried instead. A subgoal with no checkpoint at all just starts
fresh from iteration 0, same as without --resume.

Download checkpoints with:
    python3 -m modal volume get rl-harness-model-cache checkpoints/ ./artifacts/checkpoints/
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(
    image=rl_image,
    gpu=None,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=3600 * 4,
)
def train_low_level(
    subgoals: str = "align_xy,descend,close_gripper,lift,move_to_target,release",
    n_iterations: int = 300,
    max_episode_steps: int = 30,
    eval_episodes: int = 10,
    checkpoint_every: int = 50,
    seed: int = 0,
    resume: bool = False,
    resume_ckpt_iter: int = 0,   # 0 = auto-pick newest compatible checkpoint;
                                 # >0 = force resuming from low_level_{subgoal}_iterN.pt
                                 # specifically (e.g. when the newest checkpoint
                                 # regressed and an earlier one — or _best.pt — is
                                 # actually preferable to continue from)
    group_size: int = 0,     # 0 = LowLevelGRPOConfig's default (8)
    n_states: int = 0,       # 0 = default (4)
    lr: float = 0.0,         # 0 = default (3e-4)
    entropy_coef: float = -1.0,   # -1 = default (0.01); 0 is a legitimate override, so -1 means "unset"
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
    from think_then_act.training.low_level_grpo import LowLevelGRPOConfig, LowLevelGRPOTrainer

    print("\n" + "=" * 60)
    print("  LOW-LEVEL CONTROLLER TRAINING (from-scratch GRPO, per-subgoal)")
    print("=" * 60)

    torch.manual_seed(seed)

    subgoal_list = [s.strip() for s in subgoals.split(",") if s.strip()]
    for s in subgoal_list:
        if s not in SUBGOAL_LABELS:
            raise ValueError(f"Unknown subgoal {s!r}; must be one of {SUBGOAL_LABELS}")

    # ------------------------------------------------------------------
    # Collision predictor (optional — only affects `descend`'s reward)
    # ------------------------------------------------------------------
    collision_model = None
    collision_ckpt = os.path.join(MODEL_CACHE_DIR, "checkpoints", "collision_predictor.pt")
    if os.path.exists(collision_ckpt):
        collision_model = CollisionPredictor()
        collision_model.load_state_dict(torch.load(collision_ckpt, map_location="cpu"))
        collision_model.eval()
        print(f"  Loaded collision predictor <- {collision_ckpt}")
    else:
        print(f"  No collision predictor checkpoint at {collision_ckpt} — "
              f"'descend' will train with collision_prob=0.0 for every step "
              f"(run train_collision_predictor.py first for the real signal).")

    def find_resume_checkpoint(subgoal: str, policy) -> int:
        """
        Tries checkpoints for `subgoal`, newest iteration first, loading the
        first one whose keys match the current policy architecture (in
        place). Returns the iteration to resume from (0 if nothing usable
        was found). Skips, rather than crashes on, an architecture
        mismatch — otherwise a stale checkpoint from a since-changed
        SubgoalGaussianPolicy (e.g. a pre-LayerNorm checkpoint left over
        from an earlier run, with a HIGHER iteration number than the
        current run's latest) would either crash --resume outright or,
        worse, silently pick the wrong one.

        If resume_ckpt_iter > 0, "newest" is overridden entirely: only
        low_level_{subgoal}_iter{resume_ckpt_iter}.pt is tried, and a
        missing/incompatible file raises rather than silently falling back
        — this path is an explicit user request (e.g. "the latest checkpoint
        regressed, continue from an earlier one instead"), so silently
        substituting a different iteration would defeat the point.
        """
        ckpt_dir = os.path.join(MODEL_CACHE_DIR, "checkpoints")

        if resume_ckpt_iter > 0:
            path = os.path.join(ckpt_dir, f"low_level_{subgoal}_iter{resume_ckpt_iter}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"--resume-ckpt-iter {resume_ckpt_iter}: no such checkpoint {path}")
            policy.load_state_dict(torch.load(path, map_location="cpu"))   # let this raise on mismatch
            print(f"  [resume] {subgoal}: forced load of {path} (iteration {resume_ckpt_iter})")
            return min(resume_ckpt_iter, n_iterations)

        candidates = []
        final_ckpt = os.path.join(ckpt_dir, f"low_level_{subgoal}.pt")
        if os.path.exists(final_ckpt):
            candidates.append((n_iterations, final_ckpt))
        for p in glob.glob(os.path.join(ckpt_dir, f"low_level_{subgoal}_iter*.pt")):
            m = re.search(r"_iter(\d+)\.pt$", p)
            if m:
                candidates.append((int(m.group(1)), p))
        candidates.sort(key=lambda t: -t[0])

        for iteration, path in candidates:
            try:
                policy.load_state_dict(torch.load(path, map_location="cpu"))
            except RuntimeError as e:
                print(f"  [resume] {path} incompatible with current policy "
                      f"architecture ({e}) — skipping.")
                continue
            print(f"  [resume] {subgoal}: loaded {path} (iteration {iteration})")
            return min(iteration, n_iterations)

        if candidates:
            print(f"  [resume] {subgoal}: no checkpoint compatible with the current "
                  f"policy architecture — starting fresh from iteration 0.")
        return 0

    def make_env(subgoal: str):
        base = ObservationHarness(
            gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                      max_episode_steps=max_episode_steps * 2)
        )
        setup_env(base)
        return SubgoalConditionedEnv(
            base, subgoal=subgoal, collision_model=collision_model,
            max_episode_steps=max_episode_steps,
        )

    results = {}

    for subgoal in subgoal_list:
        print(f"\n--- Training subgoal: {subgoal} ---")
        env = make_env(subgoal)

        config_kwargs = dict(obs_dim=SUBGOAL_OBS_DIM, max_episode_steps=max_episode_steps)
        if group_size > 0:
            config_kwargs["group_size"] = group_size
        if n_states > 0:
            config_kwargs["n_states"] = n_states
        if lr > 0:
            config_kwargs["lr"] = lr
        if entropy_coef >= 0:
            config_kwargs["entropy_coef"] = entropy_coef
        config = LowLevelGRPOConfig(**config_kwargs)
        trainer = LowLevelGRPOTrainer(config)

        start_iteration = 0
        if resume:
            start_iteration = find_resume_checkpoint(subgoal, trainer.policy)

        # ------------------------------------------------------------------
        # Held-out eval: subgoal-completion rate on a FIXED set of seeds
        # (90_000+), disjoint from training (train_iteration's seeds are
        # n_states*iteration+i, always < n_states*n_iterations << 90_000).
        # Fixed, not just disjoint, matters here: train_iteration samples a
        # NEW, deterministic-but-different set of random states every
        # iteration, so raw per-iteration mean_reward is confounded by
        # which states that iteration happened to draw (harder start
        # position != worse policy) — found empirically 2026-07-13, two
        # separate runs both peaking/dipping at the same iterations. Calling
        # this on the SAME seeds every time removes that confound and gives
        # an actual learning curve.
        # ------------------------------------------------------------------
        def run_eval() -> float:
            completions = []
            for ep in range(eval_episodes):
                rng = np.random.default_rng(90_000 + ep)
                obs, info = env.reset(rng=rng)
                success = False
                for _ in range(max_episode_steps):
                    action = trainer.policy.act(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    if info.get("done", False):
                        success = True
                    if terminated or truncated:
                        break
                completions.append(float(success))
            return float(np.mean(completions))

        # Track BEST (by completion_rate), not just latest — same convention
        # as train_collision_predictor.py. Found necessary in practice
        # 2026-07-13: align_xy's own run regressed 10% -> 0% completion_rate
        # in its last 50 iterations (a late bad gradient step, visible as a
        # sharp late-training loss spike), so the unconditionally-overwritten
        # "final" checkpoint ended up WORSE than an earlier one still sitting
        # on disk under its _iterN name.
        best_ckpt_path = os.path.join(MODEL_CACHE_DIR, "checkpoints", f"low_level_{subgoal}_best.pt")
        best_completion_rate = -1.0

        def maybe_save_best(completion_rate: float) -> None:
            nonlocal best_completion_rate
            if completion_rate > best_completion_rate:
                best_completion_rate = completion_rate
                trainer.save_checkpoint(best_ckpt_path)
                model_volume.commit()
                print(f"  [best] {subgoal}: completion_rate={completion_rate:.1%} -> {best_ckpt_path}")

        if start_iteration >= n_iterations:
            completion_rate = run_eval()
            maybe_save_best(completion_rate)
            print(f"  {subgoal}: already complete at iteration {start_iteration} "
                  f"(target {n_iterations}) — skipping training, eval only. "
                  f"completion_rate={completion_rate:.1%}")
            results[subgoal] = {
                "ckpt_path"          : os.path.join(MODEL_CACHE_DIR, "checkpoints", f"low_level_{subgoal}.pt"),
                "completion_rate"    : round(completion_rate, 4),
                "best_ckpt_path"     : best_ckpt_path,
                "best_completion_rate": round(best_completion_rate, 4),
                "eval_history"       : [{"iteration": start_iteration, "completion_rate": round(completion_rate, 4)}],
                "final_loss"         : None,
                "final_reward"       : None,
            }
            env.close()
            continue

        history = []
        eval_history = []
        for i in range(start_iteration, n_iterations):
            metrics = trainer.train_iteration(env, i)
            history.append(metrics)

            if (i + 1) % 10 == 0:
                print(f"  iter {i+1}/{n_iterations}  loss={metrics['loss']:.4f}  "
                      f"mean_reward={metrics['mean_reward']:.4f}  "
                      f"avg_within_std={metrics['avg_within_std']:.4f}  "
                      f"entropy={metrics['mean_entropy']:.4f}  "
                      f"collect_s={metrics['collect_s']:.2f}  update_s={metrics['update_s']:.2f}")

            if (i + 1) % checkpoint_every == 0:
                ckpt = os.path.join(MODEL_CACHE_DIR, "checkpoints", f"low_level_{subgoal}_iter{i+1}.pt")
                trainer.save_checkpoint(ckpt)
                model_volume.commit()

                completion_rate = run_eval()
                maybe_save_best(completion_rate)
                eval_history.append({"iteration": i + 1, "completion_rate": round(completion_rate, 4)})
                print(f"  [eval @ iter {i+1}] {subgoal}: completion_rate={completion_rate:.1%} "
                      f"over {eval_episodes} fixed-seed episodes")

        ckpt_out = os.path.join(MODEL_CACHE_DIR, "checkpoints", f"low_level_{subgoal}.pt")
        trainer.save_checkpoint(ckpt_out)
        model_volume.commit()
        print(f"  Saved -> {ckpt_out}")

        # n_iterations already landed on a checkpoint_every boundary above —
        # don't re-run eval on the identical final policy (was a redundant
        # duplicate entry in eval_history until 2026-07-13).
        if n_iterations % checkpoint_every == 0 and eval_history and eval_history[-1]["iteration"] == n_iterations:
            completion_rate = eval_history[-1]["completion_rate"]
        else:
            completion_rate = run_eval()
            maybe_save_best(completion_rate)
            eval_history.append({"iteration": n_iterations, "completion_rate": round(completion_rate, 4)})

        print(f"  [eval] {subgoal}: completion_rate={completion_rate:.1%} over {eval_episodes} episodes")
        print(f"  best  : completion_rate={best_completion_rate:.1%} -> {best_ckpt_path}")
        print(f"  eval_history: {eval_history}")
        results[subgoal] = {
            "ckpt_path"           : ckpt_out,
            "completion_rate"     : round(completion_rate, 4),
            "best_ckpt_path"      : best_ckpt_path,
            "best_completion_rate": round(best_completion_rate, 4),
            "eval_history"        : eval_history,
            "final_loss"          : history[-1]["loss"] if history else None,
            "final_reward"        : history[-1]["mean_reward"] if history else None,
        }

        env.close()

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
    group_size: int = 0,
    n_states: int = 0,
    lr: float = 0.0,
    entropy_coef: float = -1.0,
):
    print(f"\nDispatching low-level controller training to Modal (CPU)...")
    print(f"  subgoals={subgoals}  n_iterations={n_iterations}  resume={resume}  "
          f"resume_ckpt_iter={resume_ckpt_iter or 'auto'}  "
          f"group_size={group_size or 'default'}  n_states={n_states or 'default'}  "
          f"lr={lr or 'default'}  entropy_coef={'default' if entropy_coef < 0 else entropy_coef}\n")
    handle = train_low_level.spawn(
        subgoals=subgoals, n_iterations=n_iterations, resume=resume, resume_ckpt_iter=resume_ckpt_iter,
        group_size=group_size, n_states=n_states, lr=lr, entropy_coef=entropy_coef,
    )
    print(f"Job spawned. Function call ID: {handle.object_id}")
    print(f"Monitor at https://modal.com")
