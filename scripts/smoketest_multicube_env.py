"""
smoketest_multicube_env.py

Cheap sanity check for training/multicube_stack_env.py's MultiCubeStackEnv
BEFORE trusting it with a real PPO run: drives it with the EXISTING
trained single-cube policy (not random actions, not a scripted sequence)
so the resulting video shows realistic behavior, and prints the
reward/info breakdown every step so the goal-switching, parking,
precision/disturbance terms can be checked against what's actually
happening physically -- same "verify before trusting" practice used for
every other new env/script this session (randomize_joint_angles, the
3-cube XML patcher, etc. were all frame-checked before being relied on).

Run with:
    modal run scripts/smoketest_multicube_env.py
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(image=rl_image, gpu=None, cpu=2.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=300)
def smoketest_multicube_env(
    ckpt_path: str = "checkpoints/flat_task_ppo_poserand_v2_cont_best.pt",
    n_episodes: int = 5,
    min_cubes: int = 1,
    max_cubes: int = 3,
    seed_start: int = 0,
    video_out_name: str = "multicube_smoketest",
) -> dict:
    import os
    import numpy as np

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import save_video
    from think_then_act.training.multicube_stack_env import MultiCubeStackEnv
    from think_then_act.training.flat_bc_multi_head import MultiHeadBCConfig, MultiHeadBCTrainer
    from think_then_act.training.subgoal_features import FLAT_OBS_DIM

    cfg = MultiHeadBCConfig(obs_dim=FLAT_OBS_DIM, policy_type="mse")
    trainer = MultiHeadBCTrainer(cfg)
    trainer.load_checkpoint(os.path.join(MODEL_CACHE_DIR, ckpt_path))
    print(f"loaded {ckpt_path}", flush=True)

    env = MultiCubeStackEnv(min_cubes=min_cubes, max_cubes=max_cubes, max_episode_steps=320,
                              randomize_pose_prob=1.0, render_mode="rgb_array")

    for ep in range(n_episodes):
        seed = seed_start + ep
        rng = np.random.default_rng(seed)
        obs, reset_info = env.reset(rng=rng, seed=seed)
        print(f"\n=== episode {ep} (seed={seed}) n_cubes={reset_info['n_cubes']} order={env._order} "
              f"target_z={[round(z,3) for z in env._target_z]} ===", flush=True)

        frames = [env.render()]
        hidden_state = None
        total_reward = 0.0
        n_switches = 0
        last_n_placed = 0
        for t in range(320):
            action, hidden_state = trainer.actor.act(obs, hidden_state, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            frames.append(env.render())
            total_reward += reward
            if info["n_cubes_placed"] > last_n_placed:
                n_switches += 1
                last_n_placed = info["n_cubes_placed"]
                print(f"  t={t}  CUBE PLACED (n_placed={info['n_cubes_placed']}/{info['n_cubes']})  "
                      f"d_block_target={info['d_block_target']}  disturbance={info['disturbance']}", flush=True)
            if t % 40 == 0:
                print(f"  t={t}  reward={reward:.3f}  carrying={info['carrying']}  "
                      f"d_block_target={info['d_block_target']}  d_grip_block={info['d_grip_block']}  "
                      f"disturbance={info['disturbance']}", flush=True)
            if terminated or truncated:
                print(f"  episode ended at t={t}  terminated={terminated}  truncated={truncated}  "
                      f"done={info['done']}  n_placed={info['n_cubes_placed']}/{info['n_cubes']}", flush=True)
                break

        print(f"  total_reward={total_reward:.2f}  n_switches_observed={n_switches}", flush=True)

        out_path = os.path.join(MODEL_CACHE_DIR, f"demonstrations/videos/{video_out_name}_ep{ep}.mp4")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_video(frames, out_path, fps=20)
        model_volume.commit()
        print(f"  saved video -> {out_path}", flush=True)

    env.close()
    return {"status": "done"}


@app.local_entrypoint()
def main(n_episodes: int = 5, min_cubes: int = 1, max_cubes: int = 3,
          ckpt_path: str = "checkpoints/flat_task_ppo_poserand_v2_cont_best.pt",
          video_out_name: str = "multicube_smoketest", seed_start: int = 0):
    result = smoketest_multicube_env.remote(n_episodes=n_episodes, min_cubes=min_cubes, max_cubes=max_cubes,
                                              ckpt_path=ckpt_path, video_out_name=video_out_name, seed_start=seed_start)
    print("\n", result)
