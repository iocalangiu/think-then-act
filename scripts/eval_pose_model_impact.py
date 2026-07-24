"""
eval_pose_model_impact.py

Quantifies how much the block pose predictor's inference noise actually
degrades low-level subgoal completion, instead of trusting a single demo
seed (per project history — "a single demo rollout per seed is not
reproducible evidence," see memory: hierarchical_architecture.md). Runs the
SAME fixed-seed completion_rate eval train_low_level_ppo.py already uses
during training (rng = np.random.default_rng(90_000 + ep)), twice per
subgoal against the SAME trained low-level checkpoint: once with the
trained pose model wired in (matches real deployment — no privileged
achieved_goal), once with ground truth (today's baseline). Only the
observation's block-position source differs between the two runs; reward/
done are always ground truth regardless (see block_pose_predictor.py).

Skips a subgoal cleanly (reports None, doesn't error the whole run) if it
has no trained low-level checkpoint yet.

Run with:
    modal run scripts/eval_pose_model_impact.py
    modal run scripts/eval_pose_model_impact.py --subgoals align_xy,descend,close_gripper --n-eval-episodes 20
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(
    image=rl_image,
    gpu=None,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=1800,
)
def eval_pose_model_impact(
    subgoals: str = "align_xy,descend,close_gripper,lift,move_to_target,release",
    n_eval_episodes: int = 20,
    max_episode_steps: int = 30,
    algo: str = "ppo",
    use_best: bool = True,
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
    from think_then_act.perception.block_pose_predictor import BlockPosePredictor
    from think_then_act.perception.collision_predictor import CollisionPredictor
    from think_then_act.policy.subgoal_policy import SubgoalGaussianPolicy
    from think_then_act.reward.subgoal_reward import SUBGOAL_LABELS
    from think_then_act.training.checkpoints import resolve_subgoal_checkpoint
    from think_then_act.training.subgoal_env import SubgoalConditionedEnv
    from think_then_act.training.subgoal_features import SUBGOAL_OBS_DIM

    subgoal_list = [s.strip() for s in subgoals.split(",") if s.strip()]
    for s in subgoal_list:
        if s not in SUBGOAL_LABELS:
            raise ValueError(f"Unknown subgoal {s!r}; must be one of {SUBGOAL_LABELS}")

    ckpt_dir = os.path.join(MODEL_CACHE_DIR, "checkpoints")

    collision_model = None
    collision_ckpt = os.path.join(ckpt_dir, "collision_predictor.pt")
    if os.path.exists(collision_ckpt):
        collision_model = CollisionPredictor()
        collision_model.load_state_dict(torch.load(collision_ckpt, map_location="cpu"))
        collision_model.eval()

    pose_model = None
    pose_ckpt = os.path.join(ckpt_dir, "block_pose_predictor.pt")
    if os.path.exists(pose_ckpt):
        pose_model = BlockPosePredictor()
        pose_model.load_state_dict(torch.load(pose_ckpt, map_location="cpu"))
        pose_model.eval()
    else:
        print("  WARNING: no block_pose_predictor.pt checkpoint — nothing to compare against ground truth.")

    def make_env(subgoal: str, use_pose: bool):
        base = ObservationHarness(
            gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                      max_episode_steps=max_episode_steps + 250)
        )
        setup_env(base)
        return SubgoalConditionedEnv(
            base, subgoal=subgoal, collision_model=collision_model,
            pose_model=(pose_model if use_pose else None),
            max_episode_steps=max_episode_steps,
        )

    def run_eval(subgoal: str, actor, use_pose: bool) -> dict:
        env = make_env(subgoal, use_pose)
        completions = []
        pose_errs_cm = []
        for ep in range(n_eval_episodes):
            # SAME fixed-seed convention as train_low_level_ppo.py's run_eval
            # — comparable numbers, not a different sample of scenes.
            rng = np.random.default_rng(90_000 + ep)
            obs, info = env.reset(rng=rng)
            success = False
            for _ in range(max_episode_steps):
                action = actor.act(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                if use_pose and info.get("block_pos") is not None and info.get("perceived_block_pos") is not None:
                    pose_errs_cm.append(100.0 * float(np.linalg.norm(
                        np.array(info["block_pos"]) - np.array(info["perceived_block_pos"])
                    )))
                if info.get("done", False):
                    success = True
                if terminated or truncated:
                    break
            completions.append(float(success))
        env.close()
        return {
            "completion_rate": float(np.mean(completions)),
            "mean_pose_error_cm": float(np.mean(pose_errs_cm)) if pose_errs_cm else None,
        }

    print("\n" + "=" * 70)
    print("  POSE MODEL IMPACT — completion_rate WITH vs. WITHOUT perception")
    print(f"  {n_eval_episodes} fixed-seed episodes per subgoal (seeds 90000-{90000 + n_eval_episodes - 1})")
    print("=" * 70)

    results = {}
    for subgoal in subgoal_list:
        try:
            ckpt = resolve_subgoal_checkpoint(ckpt_dir, subgoal, algo=algo, use_best=use_best)
        except FileNotFoundError:
            print(f"\n  {subgoal:15s}  SKIPPED — no trained low-level checkpoint yet")
            results[subgoal] = None
            continue

        actor = SubgoalGaussianPolicy(obs_dim=SUBGOAL_OBS_DIM)
        ckpt_data = torch.load(ckpt, map_location="cpu")
        # PPO checkpoints wrap {"actor":..., "critic":...}; GRPO checkpoints
        # are a bare state_dict — same distinction record_subgoal_video.py handles.
        actor.load_state_dict(ckpt_data["actor"] if isinstance(ckpt_data, dict) and "actor" in ckpt_data else ckpt_data)
        actor.eval()

        with_pose = run_eval(subgoal, actor, use_pose=True) if pose_model is not None else None
        without_pose = run_eval(subgoal, actor, use_pose=False)

        print(f"\n  {subgoal}  <- {os.path.basename(ckpt)}")
        print(f"    without perception (ground truth): completion_rate={without_pose['completion_rate']:.1%}")
        if with_pose is not None:
            delta = with_pose["completion_rate"] - without_pose["completion_rate"]
            print(f"    with perception                  : completion_rate={with_pose['completion_rate']:.1%}  "
                  f"mean_pose_error={with_pose['mean_pose_error_cm']:.2f}cm  "
                  f"delta={delta:+.1%}")
        else:
            print(f"    with perception                  : skipped (no pose checkpoint)")

        results[subgoal] = {"ckpt": ckpt, "without_pose": without_pose, "with_pose": with_pose}

    print("\n" + "=" * 70)
    print("  SUMMARY")
    for subgoal, r in results.items():
        if r is None:
            print(f"    {subgoal:15s}  no checkpoint")
            continue
        wp = r["with_pose"]
        wo = r["without_pose"]
        if wp is None:
            print(f"    {subgoal:15s}  without={wo['completion_rate']:.1%}  with=n/a")
        else:
            delta = wp["completion_rate"] - wo["completion_rate"]
            flag = "OK" if delta > -0.10 else "DEGRADED"
            print(f"    {subgoal:15s}  without={wo['completion_rate']:.1%}  with={wp['completion_rate']:.1%}  "
                  f"delta={delta:+.1%}  pose_err={wp['mean_pose_error_cm']:.2f}cm  [{flag}]")
    print("=" * 70)

    return {"status": "PASS", "results": results}


@app.local_entrypoint()
def main(
    subgoals: str = "align_xy,descend,close_gripper,lift,move_to_target,release",
    n_eval_episodes: int = 20,
    max_episode_steps: int = 30,
    algo: str = "ppo",
    use_best: bool = True,
):
    print(f"\nEvaluating pose model impact: subgoals={subgoals}  n_eval_episodes={n_eval_episodes}...")
    result = eval_pose_model_impact.remote(
        subgoals=subgoals, n_eval_episodes=n_eval_episodes,
        max_episode_steps=max_episode_steps, algo=algo, use_best=use_best,
    )
    print("\nDone.")
    for subgoal, r in result["results"].items():
        if r is None:
            print(f"  {subgoal}: no checkpoint")
