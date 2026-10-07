"""
eval_multicube_by_cube_count.py

Breaks out MultiCubeStackEnv eval by EXACT cube count instead of the
random min_cubes..max_cubes mix train_multicube_stack_ppo.py's own
run_eval() uses -- built specifically to answer "is the base single-cube
skill being preserved or eroded while training on the multi-cube task,"
which the aggregate completion_rate/cube_placement_rate metrics can't
answer on their own (they conflate "failed cube 2 of a 3-stack" with
"forgot the base skill entirely").

n_cubes=1 here is physically the SAME task the original single-cube PPO
checkpoint (checkpoints/flat_task_ppo_poserand_v2_cont_best.pt, 76.7%
pose-randomized completion) was evaluated on -- same pose randomization,
same genuine-grasp criterion -- just run inside MultiCubeStackEnv's
machinery instead of eval_stack_ppo's plain-env one, so this number is
directly comparable to that 76.7% baseline.

Run with:
    modal run scripts/multi-cube-experiment/eval_multicube_by_cube_count.py --ckpt-path checkpoints/multicube_stack_ppo_v1_best.pt
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(image=rl_image, gpu=None, cpu=2.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=900)
def eval_multicube_by_cube_count(
    ckpt_path: str,
    max_cubes: int = 3,
    n_episodes_per_count: int = 30,
    max_episode_steps: int = 320,
    seed_start: int = 500_000,   # a fresh range, distinct from every other eval convention in this project
    pose_exclude_band: float = 0.35,
    pose_max_frac: float = 0.85,
) -> dict:
    import os
    import numpy as np

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium_robotics  # noqa: F401

    from think_then_act.training.multicube_stack_env import MultiCubeStackEnv
    from think_then_act.training.flat_bc_multi_head import MultiHeadBCConfig, MultiHeadBCTrainer
    from think_then_act.training.subgoal_features import FLAT_OBS_DIM

    cfg = MultiHeadBCConfig(obs_dim=FLAT_OBS_DIM, policy_type="mse")
    trainer = MultiHeadBCTrainer(cfg)
    trainer.load_checkpoint(os.path.join(MODEL_CACHE_DIR, ckpt_path))
    print(f"loaded {ckpt_path}", flush=True)

    results = {}
    for n_cubes in range(1, max_cubes + 1):
        env = MultiCubeStackEnv(
            min_cubes=n_cubes, max_cubes=n_cubes,   # pin to EXACTLY n_cubes this whole sweep,
                                  # not the random mix -- that's the entire point of this script
            max_episode_steps=max_episode_steps, randomize_pose_prob=1.0,
            pose_exclude_band=pose_exclude_band, pose_max_frac=pose_max_frac,
        )
        n_full_stack = 0
        per_layer_genuine = [0] * n_cubes
        for ep in range(n_episodes_per_count):
            seed = seed_start + ep
            rng = np.random.default_rng(seed)
            obs, reset_info = env.reset(rng=rng, seed=seed)
            hidden_state = None
            info = {}
            n_placed_before = 0
            for _ in range(max_episode_steps):
                action, hidden_state = trainer.actor.act(obs, hidden_state, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                if info["n_cubes_placed"] > n_placed_before:
                    for layer in range(n_placed_before, info["n_cubes_placed"]):
                        per_layer_genuine[layer] += 1
                    n_placed_before = info["n_cubes_placed"]
                if terminated or truncated:
                    break
            if info.get("done", False):
                n_full_stack += 1
        env.close()
        completion_rate = n_full_stack / n_episodes_per_count
        per_layer_rate = [c / n_episodes_per_count for c in per_layer_genuine]
        results[n_cubes] = {"completion_rate": completion_rate, "per_layer_genuine_rate": per_layer_rate}
        print(f"  n_cubes={n_cubes}  full_stack={completion_rate:.1%}  per_layer={['%.1f%%' % (r*100) for r in per_layer_rate]}",
              flush=True)

    print(f"\n=== {ckpt_path} ===", flush=True)
    for n_cubes, r in results.items():
        print(f"  n_cubes={n_cubes}: {r['completion_rate']:.1%} full-stack", flush=True)
    return {"ckpt_path": ckpt_path, "results": results}


@app.local_entrypoint()
def main(ckpt_path: str, max_cubes: int = 3, n_episodes_per_count: int = 30, max_episode_steps: int = 320):
    result = eval_multicube_by_cube_count.remote(
        ckpt_path=ckpt_path, max_cubes=max_cubes, n_episodes_per_count=n_episodes_per_count,
        max_episode_steps=max_episode_steps,
    )
    print("\n", result)
