"""
debug_success_filter_bias.py

Tests a specific hypothesis raised about scripts/train_align_descend_bc.py:
is the episode-level success filter (training/chained_teacher_rollout.
collect_successful_demonstrations) systematically REMOVING scenes that need
a large lateral (action dim 1) correction — because those scenes take
longer to converge and are more likely to run out of the chained step
budget — leaving the BC training set biased toward small/near-zero dim-1
targets? That would directly explain the observed collapse (debug_
align_descend_bc_rollout.py): the student never seeing enough large-dim-1
examples to learn the scene-dependence, rather than simply being
undertrained.

Runs the chained teacher rollout directly (not the filtering wrapper) over
a batch of seeds, splits into successful vs. failed episodes, and compares
the distribution of (a) the align_xy teacher's FIRST action's dim-1 value
(the initial correction magnitude/direction before any convergence has
happened) and (b) the mean |dim-1| action over the whole align_xy phase,
between the two groups. Does not train or modify anything.

Run with:
    modal run scripts/debug_success_filter_bias.py
    modal run scripts/debug_success_filter_bias.py --n-episodes 200
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(
    image=rl_image,
    gpu=None,
    cpu=4.0,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=1800,
)
def debug_success_filter_bias(
    n_episodes: int = 200,
    max_chained_steps: int = 60,
    d_xy_limit: float = 0.01,
    d_z_limit: float = 0.02,
    safe_regime_streak: int = 3,
    seed: int = 0,
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
    align_xy_actor = load_actor(os.path.join(ckpt_dir, "low_level_align_xy_ppo_best.pt"))
    descend_actor = load_actor(os.path.join(ckpt_dir, "low_level_descend_ppo_best.pt"))

    base = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                  max_episode_steps=max_chained_steps + 250)
    )
    setup_env(base)
    inner_env = SubgoalConditionedEnv(base, subgoal="align_xy", max_episode_steps=max_chained_steps)
    env = SingularityForceAugmentedEnv(
        inner_env, enable_singularity_perturbation=False, enable_force_perturbation=False,
    )

    print("\n" + "=" * 70)
    print(f"  SUCCESS-FILTER BIAS CHECK — {n_episodes} chained episodes")
    print("=" * 70)

    successes, failures = [], []
    for ep_idx in range(n_episodes):
        episode = run_chained_align_descend_episode(
            env, align_xy_actor, descend_actor, seed + ep_idx, max_steps=max_chained_steps,
            d_xy_limit=d_xy_limit, d_z_limit=d_z_limit, safe_regime_streak=safe_regime_streak,
        )
        align_xy_steps = episode["switched_at_step"] or episode["n_steps"]
        # Absolute value deliberately — the SIGN of dim-1's correction depends
        # on which side of the target the block happens to be on (arbitrary
        # per scene), so averaging signed values would cancel toward zero
        # regardless of whether an episode was actually easy or hard,
        # hiding exactly the effect being tested for here.
        first_dim1_abs = abs(float(episode["teacher_action"][0, 1]))
        mean_abs_dim1 = float(np.mean(np.abs(episode["teacher_action"][:align_xy_steps, 1]))) \
            if align_xy_steps > 0 else float("nan")
        record = {"first_dim1_abs": first_dim1_abs, "mean_abs_dim1": mean_abs_dim1,
                  "align_xy_steps": align_xy_steps}
        (successes if episode["success"] else failures).append(record)

    def summarize(records: list, key: str) -> str:
        values = [r[key] for r in records]
        if not values:
            return "n=0"
        return (f"n={len(values)}  mean={np.mean(values):.4f}  std={np.std(values):.4f}  "
                f"min={np.min(values):.4f}  max={np.max(values):.4f}")

    print(f"\n  successes: {len(successes)}   failures: {len(failures)}\n")
    print("  |first align_xy action's dim-1| (initial lateral-correction magnitude):")
    print(f"    successes: {summarize(successes, 'first_dim1_abs')}")
    print(f"    failures : {summarize(failures, 'first_dim1_abs')}")
    print("\n  mean |dim-1| action over the whole align_xy phase:")
    print(f"    successes: {summarize(successes, 'mean_abs_dim1')}")
    print(f"    failures : {summarize(failures, 'mean_abs_dim1')}")
    print("\n  align_xy phase length (steps until handoff, or n_steps if never handed off):")
    print(f"    successes: {summarize(successes, 'align_xy_steps')}")
    print(f"    failures : {summarize(failures, 'align_xy_steps')}")
    print("=" * 70)

    env.close()

    return {
        "n_successes": len(successes), "n_failures": len(failures),
        "successes": successes, "failures": failures,
    }


@app.local_entrypoint()
def main(
    n_episodes: int = 200, max_chained_steps: int = 60,
    d_xy_limit: float = 0.01, d_z_limit: float = 0.02, safe_regime_streak: int = 3, seed: int = 0,
):
    print(f"\nDispatching success-filter bias check to Modal (CPU)...")
    handle = debug_success_filter_bias.spawn(
        n_episodes=n_episodes, max_chained_steps=max_chained_steps,
        d_xy_limit=d_xy_limit, d_z_limit=d_z_limit, safe_regime_streak=safe_regime_streak, seed=seed,
    )
    print(f"Job spawned. Function call ID: {handle.object_id}")
    print(f"Monitor at https://modal.com")
