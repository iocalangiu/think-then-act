"""
debug_align_descend_bc_rollout.py

Read-only diagnostic pulling RAW per-step trajectories instead of just an
aggregate completion_rate — to tell apart "not precise enough yet" (d_xy
should trend down, even if it doesn't fully converge) from "something is
actually broken" (action is degenerate/near-constant, or d_xy diverges
immediately). Originally written for train_align_descend_bc.py's BC
checkpoint; `--student-ckpt` makes it reusable for ANY SubgoalRecurrentPolicy
checkpoint sharing align_xy's obs schema — e.g. an in-progress
train_low_level_ppo_recurrent.py RL fine-tuning run's periodic
`low_level_align_xy_ppo_rnn_iter{N}.pt` checkpoints, to see what the actor
is actually doing at a given iteration instead of reasoning from aggregate
metrics alone.

Also directly compares the trained student's action against the FROZEN
align_xy teacher's action at the SAME starting observation (before any
rollout divergence has had a chance to happen) — isolates whether the
divergence starts immediately (bug/undertrained) or builds up over many
steps (classic compounding error).

Does not train or modify anything.

Run with:
    modal run scripts/debug_align_descend_bc_rollout.py
    modal run scripts/debug_align_descend_bc_rollout.py --student-ckpt low_level_align_xy_ppo_rnn_iter50.pt
    modal run scripts/debug_align_descend_bc_rollout.py --n-episodes 3 --print_every 5
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(
    image=rl_image,
    gpu=None,
    cpu=8.0,   # matches train_low_level_ppo_recurrent.py's own cpu=8.0 — deliberately, not
               # a tuning choice: a mismatch here was found (2026-09-18) to potentially
               # explain a real train/eval discrepancy (a seed that succeeds here at
               # cpu=2.0 but was reported as a failure by that script's own run_eval at
               # cpu=8.0, same weights/seed/deterministic=True) — if MuJoCo's contact
               # solver has any thread-count-dependent floating-point non-associativity,
               # a task this sensitive to a tight distance threshold could plausibly flip
               # outcomes across core counts even with otherwise identical code.
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=600,
)
def debug_align_descend_bc_rollout(
    n_episodes: int = 3,
    max_chained_steps: int = 30,   # matches train_low_level_ppo_recurrent.py's own
                                    # max_episode_steps default for a SINGLE-subgoal
                                    # align_xy run — pass 60 explicitly when inspecting
                                    # the chained align_xy->descend BC checkpoint instead
                                    # (train_align_descend_bc.py's own budget). Using the
                                    # wrong one gives episodes extra runway a real eval
                                    # never had, making a failure look like a near-success.
    print_every: int = 5,
    seed: int = 90_000,
    student_ckpt: str = "low_level_align_descend_bc.pt",
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
    from think_then_act.policy.subgoal_recurrent_policy import SubgoalRecurrentPolicy
    from think_then_act.training.subgoal_env import SubgoalConditionedEnv
    from think_then_act.training.subgoal_features import obs_dim_for_subgoal
    from think_then_act.training.singularity_force_env import SingularityForceAugmentedEnv

    ckpt_dir = os.path.join(MODEL_CACHE_DIR, "checkpoints")
    align_xy_ckpt = os.path.join(ckpt_dir, "low_level_align_xy_ppo_best.pt")
    student_ckpt_path = os.path.join(ckpt_dir, student_ckpt)
    for p in (align_xy_ckpt, student_ckpt_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing {p}")

    align_xy_actor = SubgoalGaussianPolicy(obs_dim=obs_dim_for_subgoal("align_xy"))
    ckpt = torch.load(align_xy_ckpt, map_location="cpu")
    align_xy_actor.load_state_dict(ckpt["actor"] if isinstance(ckpt, dict) and "actor" in ckpt else ckpt)
    align_xy_actor.eval()

    student_ckpt_data = torch.load(student_ckpt_path, map_location="cpu")
    student_obs_dim = student_ckpt_data["actor"]["input_norm.weight"].shape[0]
    student = SubgoalRecurrentPolicy(obs_dim=student_obs_dim)
    student.load_state_dict(student_ckpt_data["actor"])
    student.eval()
    print(f"  student checkpoint: {student_ckpt}  obs_dim={student_obs_dim}")

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
    print("  STEP 1: student vs. teacher action at the SAME starting observation")
    print("  (isolates: does divergence start immediately, or build up over steps?)")
    print("=" * 70)
    base_dim = obs_dim_for_subgoal("align_xy")
    for ep in range(n_episodes):
        rng = np.random.default_rng(seed + ep)
        obs, info = env.reset(rng=rng)
        teacher_action = align_xy_actor.act(obs[:base_dim], deterministic=True)
        student_action, _ = student.act(obs, None, deterministic=True)
        action_diff = float(np.linalg.norm(teacher_action - student_action))
        print(f"  ep{ep}: teacher={np.round(teacher_action, 3)}  student={np.round(student_action, 3)}  "
              f"||diff||={action_diff:.4f}  initial_d_xy={info.get('d_xy')}")

    print("\n" + "=" * 70)
    print("  STEP 2: full student rollout trajectories (align_xy phase)")
    print("=" * 70)
    for ep in range(n_episodes):
        rng = np.random.default_rng(seed + ep)
        obs, info = env.reset(rng=rng)
        env.env.set_subgoal("align_xy")
        hidden_state = None
        print(f"\n  --- episode {ep} (seed={seed + ep}) ---")
        print(f"    step   0: d_xy={info.get('d_xy')}")
        for step_idx in range(max_chained_steps):
            action, hidden_state = student.act(obs, hidden_state, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if (step_idx + 1) % print_every == 0 or info.get("done", False) or truncated:
                print(f"    step {step_idx+1:3d}: action={np.round(action, 3)}  "
                      f"d_xy={info.get('d_xy')}  done={info.get('done', False)}")
            if info.get("done", False):
                print(f"    -> align_xy done at step {step_idx+1}")
                break
            if truncated:
                print(f"    -> truncated at step {step_idx+1} without align_xy done")
                break

    env.close()
    print("=" * 70)
    return {"status": "PASS"}


@app.local_entrypoint()
def main(n_episodes: int = 3, max_chained_steps: int = 30, print_every: int = 5, seed: int = 90_000,
          student_ckpt: str = "low_level_align_descend_bc.pt"):
    print(f"\nDispatching align_xy/descend BC rollout debug to Modal (CPU)...")
    print(f"  student_ckpt={student_ckpt}\n")
    handle = debug_align_descend_bc_rollout.spawn(
        n_episodes=n_episodes, max_chained_steps=max_chained_steps,
        print_every=print_every, seed=seed, student_ckpt=student_ckpt,
    )
    print(f"Job spawned. Function call ID: {handle.object_id}")
    print(f"Monitor at https://modal.com")
