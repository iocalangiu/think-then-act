"""
eval_align_descend_convergence.py

Read-only sanity check for scripts/train_align_descend_bc.py: before
trusting align_xy's and descend's `_ppo_best.pt` checkpoints as frozen BC
teachers, measure (1) each one's OWN standalone completion_rate, using the
SAME methodology their own training already used (fixed eval seeds,
deterministic actions, descend evaluated with the trained align_xy actor
handing it off — not a fresh scattered reset, matching
env/setup.py's _run_align_xy_until_done), and (2) the CHAINED success rate
the BC script itself would see (reusing
training/chained_teacher_rollout.collect_successful_demonstrations
directly, so this number is exactly what BC collection would report).

Does not train, modify, or resave either checkpoint.

Run with:
    modal run scripts/eval_align_descend_convergence.py
    modal run scripts/eval_align_descend_convergence.py --eval-episodes 50
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(
    image=rl_image,
    gpu=None,
    cpu=2.0,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=1200,
)
def eval_align_descend_convergence(
    eval_episodes: int = 20,
    max_episode_steps: int = 30,
    max_chained_steps: int = 60,
    d_xy_limit: float = 0.01,
    d_z_limit: float = 0.02,
    safe_regime_streak: int = 3,
) -> dict:
    import os
    import numpy as np
    import torch

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.policy.subgoal_policy import SubgoalGaussianPolicy
    from think_then_act.training.subgoal_env import SubgoalConditionedEnv
    from think_then_act.training.subgoal_features import obs_dim_for_subgoal
    from think_then_act.training.singularity_force_env import SingularityForceAugmentedEnv
    from think_then_act.training.chained_teacher_rollout import run_chained_align_descend_episode

    def load_actor(ckpt_path: str) -> SubgoalGaussianPolicy:
        actor = SubgoalGaussianPolicy(obs_dim=obs_dim_for_subgoal("align_xy"))
        ckpt = torch.load(ckpt_path, map_location="cpu")
        actor.load_state_dict(ckpt["actor"] if isinstance(ckpt, dict) and "actor" in ckpt else ckpt)
        actor.eval()
        return actor

    ckpt_dir = os.path.join(MODEL_CACHE_DIR, "checkpoints")
    align_xy_ckpt = os.path.join(ckpt_dir, "low_level_align_xy_ppo_best.pt")
    descend_ckpt = os.path.join(ckpt_dir, "low_level_descend_ppo_best.pt")
    for p, name in [(align_xy_ckpt, "align_xy"), (descend_ckpt, "descend")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"No {name} checkpoint at {p}")

    align_xy_actor = load_actor(align_xy_ckpt)
    descend_actor = load_actor(descend_ckpt)

    def make_env(subgoal: str, max_steps: int, align_xy_policy=None) -> SubgoalConditionedEnv:
        base = ObservationHarness(
            gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                      max_episode_steps=max_steps + 250)
        )
        setup_env(base)
        return SubgoalConditionedEnv(base, subgoal=subgoal, max_episode_steps=max_steps,
                                      align_xy_policy=align_xy_policy)

    def run_eval(env, actor, n_episodes: int, max_steps: int, seed_offset: int = 90_000) -> dict:
        completions = []
        final_d_xy, final_d_z = [], []
        for ep in range(n_episodes):
            rng = np.random.default_rng(seed_offset + ep)
            obs, info = env.reset(rng=rng)
            success = False
            last_info = info
            for _ in range(max_steps):
                action = actor.act(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                last_info = info
                if info.get("done", False):
                    success = True
                if terminated or truncated:
                    break
            completions.append(float(success))
            if "d_xy" in last_info:
                final_d_xy.append(last_info["d_xy"])
            if "d_z" in last_info:
                final_d_z.append(last_info["d_z"])
        return {
            "completion_rate": float(np.mean(completions)),
            "mean_final_d_xy": float(np.mean(final_d_xy)) if final_d_xy else None,
            "mean_final_d_z": float(np.mean(final_d_z)) if final_d_z else None,
        }

    print("\n" + "=" * 60)
    print("  ALIGN_XY / DESCEND CONVERGENCE CHECK (read-only)")
    print("=" * 60)

    # Same seed offset (90_000+ep) used for ALL THREE evals below — a fair,
    # apples-to-apples comparison needs identical scenes, not just identical
    # episode COUNTS. (An earlier version of this script used seeds 0..N-1
    # for the chained eval and 90_000..90_000+N-1 for the standalone ones —
    # different scenes entirely, which would have confounded any gap between
    # them with plain scene-difficulty variance.)
    SEED_OFFSET = 90_000

    align_env = make_env("align_xy", max_episode_steps)
    align_result = run_eval(align_env, align_xy_actor, eval_episodes, max_episode_steps, SEED_OFFSET)
    align_env.close()
    print(f"  align_xy  standalone: completion_rate={align_result['completion_rate']:.1%}  "
          f"mean_final_d_xy={align_result['mean_final_d_xy']}")

    # descend evaluated with the SAME align_xy-actor handoff its own training/
    # eval used (env/setup.py's _run_align_xy_until_done) — not a fresh
    # scattered reset, since that's a training/deployment distribution
    # mismatch this codebase has already hit and fixed once before.
    descend_env = make_env("descend", max_episode_steps, align_xy_policy=align_xy_actor)
    descend_result = run_eval(descend_env, descend_actor, eval_episodes, max_episode_steps, SEED_OFFSET)
    descend_env.close()
    print(f"  descend   standalone: completion_rate={descend_result['completion_rate']:.1%}  "
          f"mean_final_d_z={descend_result['mean_final_d_z']}  mean_final_d_xy={descend_result['mean_final_d_xy']}")

    # Chained: exactly what train_align_descend_bc.py's collection step will
    # see — align_xy MLP acts until ITS OWN done fires, hand off to descend
    # MLP, stop once the new safe-regime condition holds. Called directly
    # (not via collect_successful_demonstrations) so failure diagnostics
    # aren't thrown away — that function only keeps the successes.
    base = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                  max_episode_steps=max_chained_steps + 250)
    )
    setup_env(base)
    chained_inner = SubgoalConditionedEnv(base, subgoal="align_xy", max_episode_steps=max_chained_steps)
    chained_env = SingularityForceAugmentedEnv(
        chained_inner, enable_singularity_perturbation=False, enable_force_perturbation=False,
    )
    chained_episodes = [
        run_chained_align_descend_episode(
            chained_env, align_xy_actor, descend_actor, SEED_OFFSET + ep, max_steps=max_chained_steps,
            d_xy_limit=d_xy_limit, d_z_limit=d_z_limit, safe_regime_streak=safe_regime_streak,
        )
        for ep in range(eval_episodes)
    ]
    chained_env.close()

    n_successful = sum(ep["success"] for ep in chained_episodes)
    chained_success_rate = n_successful / eval_episodes
    print(f"  chained align_xy->descend (safe-regime): success_rate={chained_success_rate:.1%} "
          f"({n_successful}/{eval_episodes})")

    failures = [ep for ep in chained_episodes if not ep["success"]]
    if failures:
        print(f"\n  --- {len(failures)} failed chained episode(s), per-episode breakdown ---")
        never_reached_descend = sum(1 for ep in failures if ep["switched_at_step"] is None)
        hit_ceiling = sum(1 for ep in failures if ep["hit_step_ceiling"])
        had_onset = sum(1 for ep in failures if ep["any_force_onset_during_descend"])
        print(f"  never reached descend (align_xy itself didn't finish in {max_chained_steps} steps): "
              f"{never_reached_descend}/{len(failures)}")
        print(f"  ran out of steps (hit_step_ceiling)                                    : "
              f"{hit_ceiling}/{len(failures)}")
        print(f"  had at least one force-onset event during descend                      : "
              f"{had_onset}/{len(failures)}")
        for i, ep in enumerate(failures):
            print(f"    [{i}] switched_at_step={ep['switched_at_step']}  n_steps={ep['n_steps']}  "
                  f"hit_step_ceiling={ep['hit_step_ceiling']}  "
                  f"best_streak_during_descend={ep['best_streak_during_descend']}/{safe_regime_streak}  "
                  f"any_force_onset={ep['any_force_onset_during_descend']}  "
                  f"final_d_xy={ep['final_d_xy']}  final_d_z={ep['final_d_z']}")
    print("=" * 60)

    return {
        "align_xy": align_result,
        "descend": descend_result,
        "chained_success_rate": chained_success_rate,
        "chained_n_successful": n_successful,
        "chained_n_attempted": eval_episodes,
        "chained_failures": [
            {k: v for k, v in ep.items() if k not in ("obs", "teacher_action")}
            for ep in failures
        ],
    }


@app.local_entrypoint()
def main(
    eval_episodes: int = 20,
    max_episode_steps: int = 30,
    max_chained_steps: int = 60,
    d_xy_limit: float = 0.01,
    d_z_limit: float = 0.02,
    safe_regime_streak: int = 3,
):
    print(f"\nDispatching align_xy/descend convergence check to Modal (CPU)...")
    print(f"  eval_episodes={eval_episodes}  max_episode_steps={max_episode_steps}  "
          f"max_chained_steps={max_chained_steps}\n")
    handle = eval_align_descend_convergence.spawn(
        eval_episodes=eval_episodes, max_episode_steps=max_episode_steps,
        max_chained_steps=max_chained_steps, d_xy_limit=d_xy_limit, d_z_limit=d_z_limit,
        safe_regime_streak=safe_regime_streak,
    )
    print(f"Job spawned. Function call ID: {handle.object_id}")
    print(f"Monitor at https://modal.com")
