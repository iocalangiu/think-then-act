"""
train_align_descend_bc.py

Behavioral-cloning pretraining for ONE recurrent policy spanning
align_xy -> descend as a single continuous episode (hidden state carried
across the handoff) — see memory: meta_rl_sim2real_direction. Two already-
trained, UNTOUCHED SubgoalGaussianPolicy checkpoints (align_xy's and
descend's own `_ppo_best.pt`) act as frozen teachers; only successful
chained rollouts (teacher chain actually reaches the new "safe to hand off
to the scripted gripper close" condition) are kept as demonstrations — see
training/safe_regime.py and training/chained_teacher_rollout.py.

Does NOT retrain, modify, or even load-and-resave either teacher
checkpoint — they're read-only inputs. Does NOT train close_gripper/lift/
move_to_target — closing the real gripper is a single scripted ROS2 action
call (`/robotiq_gripper_controller/gripper_cmd`), not a learned skill, and
lift/move_to_target are explicitly postponed.

Output checkpoint (`low_level_align_descend_bc.pt`) is a drop-in
--warm-start-ckpt for scripts/train_low_level_ppo_recurrent.py's existing
RL fine-tuning path (RecurrentLowLevelPPOTrainer.load_checkpoint expects
exactly the {"actor":..., "critic":...} shape this script saves — see
training/behavioral_cloning.py's docstring for why an untrained critic is
included). Reconciling RL fine-tuning across a still-chained two-phase
episode (vs. today's per-subgoal-only recurrent PPO trainer) is an open
follow-up, not attempted here — this script only produces the BC-pretrained
weights.

Run with:
    modal run --detach scripts/train_align_descend_bc.py
    modal run --detach scripts/train_align_descend_bc.py --n-demo-episodes 50 --n-epochs 5
                                                          # quick sanity check
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(
    image=rl_image,
    gpu=None,
    cpu=4.0,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=3600 * 2,
)
def train_align_descend_bc(
    n_demo_episodes: int = 400,   # bumped from 200 (2026-09-18): at ~70% chained
                                   # survival, 200 attempts gave only 146 successes,
                                   # and the harder (large lateral-correction) ones
                                   # among them were under-represented — more
                                   # attempts means more absolute hard-example count
                                   # even at the same relative skew (see
                                   # debug_success_filter_bias.py).
    max_chained_steps: int = 60,
    d_xy_limit: float = 0.01,
    d_z_limit: float = 0.02,
    safe_regime_streak: int = 3,
    include_discrepancy_obs: bool = True,
    include_force_obs: bool = True,
    action_scale_m: float = 0.05,
    weight_dim: int = 1,   # action index used for compute_episode_weight's
                            # importance-weighting (see chained_teacher_rollout.py) —
                            # None disables weighting (every demo gets weight=1.0).
    rnn_hidden_size: int = 64,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    n_epochs: int = 100,   # bumped from 30 (2026-09-18): the first BC run's loss
                            # (1.73 -> 0.057) was still clearly decreasing, and 146
                            # demonstrations can support far more epochs before
                            # overfitting than 30.
    episodes_per_minibatch: int = 16,
    seed: int = 0,
    eval_episodes: int = 20,
) -> dict:
    import os
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
    from think_then_act.training.chained_teacher_rollout import (
        collect_successful_demonstrations, run_recurrent_student_episode,
    )
    from think_then_act.training.behavioral_cloning import BCConfig, BehavioralCloningTrainer

    print("\n" + "=" * 60)
    print("  ALIGN_XY -> DESCEND CHAINED BEHAVIORAL CLONING")
    print("=" * 60)

    torch.manual_seed(seed)

    def load_frozen_teacher(ckpt_path: str) -> SubgoalGaussianPolicy:
        actor = SubgoalGaussianPolicy(obs_dim=obs_dim_for_subgoal("align_xy"))
        ckpt = torch.load(ckpt_path, map_location="cpu")
        actor.load_state_dict(ckpt["actor"] if isinstance(ckpt, dict) and "actor" in ckpt else ckpt)
        actor.eval()
        for p in actor.parameters():
            p.requires_grad_(False)
        return actor

    align_xy_ckpt = os.path.join(MODEL_CACHE_DIR, "checkpoints", "low_level_align_xy_ppo_best.pt")
    descend_ckpt = os.path.join(MODEL_CACHE_DIR, "checkpoints", "low_level_descend_ppo_best.pt")
    for path, name in [(align_xy_ckpt, "align_xy"), (descend_ckpt, "descend")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No trained {name} checkpoint at {path} — this script needs both "
                f"align_xy and descend's already-trained _ppo_best.pt as frozen teachers. "
                f"Train them first via scripts/train_low_level_ppo.py."
            )
    print(f"  align_xy teacher <- {align_xy_ckpt}")
    print(f"  descend  teacher <- {descend_ckpt}")
    align_xy_actor = load_frozen_teacher(align_xy_ckpt)
    descend_actor = load_frozen_teacher(descend_ckpt)

    # +250: same TimeLimit headroom convention as rollout_workers.py's
    # _worker_init (unused here since align_xy/descend need no oracle
    # setup phase, but harmless, and keeps this consistent with every
    # other env-construction site in the codebase).
    base = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                  max_episode_steps=max_chained_steps + 250)
    )
    setup_env(base)
    inner_env = SubgoalConditionedEnv(base, subgoal="align_xy", max_episode_steps=max_chained_steps)
    env = SingularityForceAugmentedEnv(
        inner_env,
        include_discrepancy_obs=include_discrepancy_obs, include_force_obs=include_force_obs,
        action_scale_m=action_scale_m,
        # Perturbation OFF for BC — a teacher never trained under perturbation
        # is not a valid label source once things are actually going wrong
        # (see training/chained_teacher_rollout.py's module docstring).
        enable_singularity_perturbation=False, enable_force_perturbation=False,
    )

    print(f"\n  Collecting up to {n_demo_episodes} chained demonstrations "
          f"(d_xy_limit={d_xy_limit}, d_z_limit={d_z_limit}, safe_regime_streak={safe_regime_streak})...")
    seeds = list(range(seed, seed + n_demo_episodes))
    collection = collect_successful_demonstrations(
        env, align_xy_actor, descend_actor, seeds, max_steps=max_chained_steps,
        d_xy_limit=d_xy_limit, d_z_limit=d_z_limit, safe_regime_streak=safe_regime_streak,
        weight_dim=weight_dim,
    )
    print(f"  Teacher chain success_rate={collection['success_rate']:.1%} "
          f"({collection['n_successful']}/{collection['n_attempted']} episodes)")
    if weight_dim is not None and collection["demonstrations"]:
        weights = [d["weight"] for d in collection["demonstrations"]]
        import numpy as np
        print(f"  importance weights (dim={weight_dim}): mean={np.mean(weights):.3f}  "
              f"min={np.min(weights):.3f}  max={np.max(weights):.3f}")

    if collection["n_successful"] == 0:
        print("  ABORTING: zero successful teacher demonstrations — nothing to clone. "
              "Check whether align_xy/descend's checkpoints actually chain (e.g. run "
              "them separately first), or loosen d_xy_limit/d_z_limit/safe_regime_streak.")
        env.close()
        return {
            "status": "FAIL", "reason": "no_successful_demonstrations",
            "n_attempted": collection["n_attempted"],
        }

    obs_dim = env.obs_dim()
    bc_config = BCConfig(
        obs_dim=obs_dim, hidden_dim=hidden_dim, rnn_hidden_size=rnn_hidden_size,
        lr=lr, n_epochs=n_epochs, episodes_per_minibatch=episodes_per_minibatch,
    )
    bc_trainer = BehavioralCloningTrainer(bc_config)

    print(f"\n  Training recurrent student (obs_dim={obs_dim}) for {n_epochs} epochs "
          f"on {collection['n_successful']} demonstrations...")
    fit_result = bc_trainer.fit(collection["demonstrations"])
    losses = fit_result["epoch_losses"]
    print(f"  loss: epoch 1={losses[0]:.5f}  epoch {len(losses)}={losses[-1]:.5f}")

    ckpt_out = os.path.join(MODEL_CACHE_DIR, "checkpoints", "low_level_align_descend_bc.pt")
    bc_trainer.save_checkpoint(ckpt_out)
    model_volume.commit()
    print(f"  Saved -> {ckpt_out}")

    print(f"\n  Evaluating the trained student alone (no teacher) over {eval_episodes} episodes...")
    eval_seeds = list(range(90_000, 90_000 + eval_episodes))
    student_successes = 0
    switched_count = 0
    for s in eval_seeds:
        ep = run_recurrent_student_episode(
            env, bc_trainer.actor, s, max_steps=max_chained_steps,
            d_xy_limit=d_xy_limit, d_z_limit=d_z_limit, safe_regime_streak=safe_regime_streak,
        )
        student_successes += int(ep["success"])
        switched_count += int(ep["switched_at_step"] is not None)
    student_success_rate = student_successes / len(eval_seeds)
    switch_rate = switched_count / len(eval_seeds)
    print(f"  student success_rate={student_success_rate:.1%}  "
          f"(reached descend phase at all: {switch_rate:.1%})")

    env.close()

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  teacher chain success_rate : {collection['success_rate']:.1%} "
          f"({collection['n_successful']}/{collection['n_attempted']})")
    print(f"  BC final loss              : {losses[-1]:.5f}")
    print(f"  student success_rate (eval): {student_success_rate:.1%}")
    print(f"  student reached descend    : {switch_rate:.1%}")
    print(f"  checkpoint                 : {ckpt_out}")
    print("=" * 60)

    return {
        "status": "PASS",
        "teacher_success_rate": collection["success_rate"],
        "n_demonstrations": collection["n_successful"],
        "epoch_losses": losses,
        "student_success_rate": student_success_rate,
        "student_switch_rate": switch_rate,
        "ckpt_path": ckpt_out,
    }


@app.local_entrypoint()
def main(
    n_demo_episodes: int = 400,
    max_chained_steps: int = 60,
    d_xy_limit: float = 0.01,
    d_z_limit: float = 0.02,
    safe_regime_streak: int = 3,
    include_discrepancy_obs: bool = True,
    include_force_obs: bool = True,
    action_scale_m: float = 0.05,
    weight_dim: int = 1,
    rnn_hidden_size: int = 64,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    n_epochs: int = 100,
    episodes_per_minibatch: int = 16,
    seed: int = 0,
    eval_episodes: int = 20,
):
    print(f"\nDispatching align_xy->descend chained BC training to Modal (CPU)...")
    print(f"  n_demo_episodes={n_demo_episodes}  max_chained_steps={max_chained_steps}  "
          f"d_xy_limit={d_xy_limit}  d_z_limit={d_z_limit}  n_epochs={n_epochs}  weight_dim={weight_dim}\n")
    handle = train_align_descend_bc.spawn(
        n_demo_episodes=n_demo_episodes, max_chained_steps=max_chained_steps,
        d_xy_limit=d_xy_limit, d_z_limit=d_z_limit, safe_regime_streak=safe_regime_streak,
        include_discrepancy_obs=include_discrepancy_obs, include_force_obs=include_force_obs,
        action_scale_m=action_scale_m, weight_dim=weight_dim,
        rnn_hidden_size=rnn_hidden_size, hidden_dim=hidden_dim,
        lr=lr, n_epochs=n_epochs, episodes_per_minibatch=episodes_per_minibatch,
        seed=seed, eval_episodes=eval_episodes,
    )
    print(f"Job spawned. Function call ID: {handle.object_id}")
    print(f"Monitor at https://modal.com")
