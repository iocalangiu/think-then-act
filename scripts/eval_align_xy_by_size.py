"""
eval_align_xy_by_size.py

Diagnostic: is align_xy's iter150->iter300 completion_rate regression
(50%->30%, seen 2026-09-01/02 on the z-penalty retrain) a real policy
regression under the harder full-size-range curriculum, or a pose-model
localization artifact at small block widths (the same shape of bug found
2026-08-19/20 for the PRE-z-penalty checkpoint — see hierarchical_architecture
memory)? Isolates by running the SAME deterministic completion_rate eval
(10 fixed seeds, same convention as train_low_level_ppo.py's run_eval) at
several pinned widths, both WITH the pose model (what training's own eval
used) and against ground-truth achieved_goal (isolates the policy alone).

Run with:
    modal run scripts/eval_align_xy_by_size.py
    modal run scripts/eval_align_xy_by_size.py --ckpts low_level_align_xy_ppo_best.pt,low_level_align_xy_ppo.pt
    modal run scripts/eval_align_xy_by_size.py --widths 0.01,0.02,0.05,0.08
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(
    image=rl_image,
    gpu=None,
    cpu=2.0,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=1800,
)
def eval_align_xy_by_size(
    ckpts: str = "low_level_align_xy_ppo_best.pt,low_level_align_xy_ppo.pt",
    widths: str = "0.05,0.03,0.02,0.01",   # 0.05 ~ near-original cube (stage 1),
                                            # 0.03/0.02 mid full-range, 0.01 the
                                            # extreme small end that broke the
                                            # pose model last time (memory,
                                            # 2026-08-19/20)
    eval_episodes: int = 10,
    max_episode_steps: int = 30,
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
    from think_then_act.policy.subgoal_policy import SubgoalGaussianPolicy
    from think_then_act.training.subgoal_env import SubgoalConditionedEnv
    from think_then_act.training.subgoal_features import obs_dim_for_subgoal

    ckpt_dir = os.path.join(MODEL_CACHE_DIR, "checkpoints")
    ckpt_list = [c.strip() for c in ckpts.split(",") if c.strip()]
    width_list = [float(w.strip()) for w in widths.split(",") if w.strip()]

    pose_ckpt = os.path.join(ckpt_dir, "block_pose_predictor.pt")
    pose_model = None
    if os.path.exists(pose_ckpt):
        pose_model = BlockPosePredictor()
        pose_model.load_state_dict(torch.load(pose_ckpt, map_location="cpu"))
        pose_model.eval()
        print(f"Pose model found <- {pose_ckpt}")
    else:
        print("No pose model found — perceived-eval rows will be skipped.")

    def make_env(width: float, use_pose_model: bool):
        base = ObservationHarness(
            gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                      max_episode_steps=max_episode_steps + 250)
        )
        setup_env(base)
        return SubgoalConditionedEnv(
            base, subgoal="align_xy",
            pose_model=pose_model if use_pose_model else None,
            max_episode_steps=max_episode_steps,
            randomize_block_size=True,
            width_range=(width, width),
        )

    def run_eval(actor, width: float, use_pose_model: bool) -> dict:
        env = make_env(width, use_pose_model)
        completions, d_xy_finals, d_z_finals = [], [], []
        for ep in range(eval_episodes):
            rng = np.random.default_rng(90_000 + ep)
            obs, info = env.reset(rng=rng)
            success = False
            last_d_xy, last_d_z = None, None
            for _ in range(max_episode_steps):
                action = actor.act(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                last_d_xy = info.get("d_xy", last_d_xy)
                last_d_z = info.get("d_z", last_d_z)
                if info.get("done", False):
                    success = True
                if terminated or truncated:
                    break
            completions.append(float(success))
            if last_d_xy is not None:
                d_xy_finals.append(last_d_xy)
            if last_d_z is not None:
                d_z_finals.append(last_d_z)
        env.close()
        return {
            "completion_rate": round(float(np.mean(completions)), 4),
            "mean_d_xy_final": round(float(np.mean(d_xy_finals)), 5) if d_xy_finals else None,
            "mean_d_z_final": round(float(np.mean(d_z_finals)), 5) if d_z_finals else None,
        }

    results = {}
    for ckpt_name in ckpt_list:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {ckpt_name}: not found at {ckpt_path}")
            continue
        actor = SubgoalGaussianPolicy(obs_dim=obs_dim_for_subgoal("align_xy"))
        ckpt_data = torch.load(ckpt_path, map_location="cpu")
        actor.load_state_dict(ckpt_data["actor"] if isinstance(ckpt_data, dict) and "actor" in ckpt_data else ckpt_data)
        actor.eval()

        print(f"\n=== {ckpt_name} ===")
        results[ckpt_name] = {}
        for width in width_list:
            for use_pose_model in ([True, False] if pose_model is not None else [False]):
                tag = "perceived" if use_pose_model else "ground_truth"
                r = run_eval(actor, width, use_pose_model)
                results[ckpt_name][f"width={width}_{tag}"] = r
                print(f"  width={width:.3f}  {tag:12s}  completion_rate={r['completion_rate']:.1%}  "
                      f"d_xy(final)={r['mean_d_xy_final']}  d_z(final)={r['mean_d_z_final']}")

    return results


@app.local_entrypoint()
def main(
    ckpts: str = "low_level_align_xy_ppo_best.pt,low_level_align_xy_ppo.pt",
    widths: str = "0.05,0.03,0.02,0.01",
    eval_episodes: int = 10,
):
    result = eval_align_xy_by_size.remote(ckpts=ckpts, widths=widths, eval_episodes=eval_episodes)
    print("\n" + "=" * 60)
    print("Result:", result)
