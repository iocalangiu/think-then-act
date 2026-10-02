"""
eval_elevated_target.py

Generalization check: evaluates already-trained BC checkpoints against a
target placed IN THE AIR above the table, not at table height like every
other eval in this project. init_random_episode() always sets the goal's
z to resting_z (table height) — deliberately NOT modified here (it's used
everywhere else and changing its behavior would be a much bigger, riskier
change than this needs). Instead: call it normally (block placement,
finger reset, table-height target all unchanged), then raise
env.unwrapped.goal's z afterward and take one more zero-action step so the
returned observation actually reflects the new target before the real
rollout starts.

This is a meaningfully harder/different test than every other completion
rate in this project's charts: reaching an elevated target requires the
block to actually be LIFTED to that height (is_success is a 3D distance
check, z included) — a slide/nudge along the table can never satisfy it,
unlike the table-height target where that was the whole contamination
problem this session found and fixed.

Run with:
    modal run scripts/eval_elevated_target.py
    modal run scripts/eval_elevated_target.py --elevation-min 0.15 --elevation-max 0.35
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR

ARCHITECTURES = ["mse", "mdn", "autoregressive", "factored", "cvae"]   # mdn
                                  # re-included 2026-10-02: on the new mixed
                                  # table-height/elevated-target training
                                  # data it reaches 22% genuine completion at
                                  # n_demos=2000 (was stuck at 0% on the old
                                  # table-height-only data), so its
                                  # generalization is now meaningful to test.


@app.function(image=rl_image, gpu=None, cpu=2.0,
              volumes={MODEL_CACHE_DIR: model_volume}, timeout=600)
def eval_elevated_target(
    policy_type: str,
    ckpt_path: str = "",
    n_eval_episodes: int = 30,
    max_steps: int = 100,
    elevation_min: float = 0.10,
    elevation_max: float = 0.30,
    block_resting_z: float = 0.425,
    lift_threshold: float = 0.02,
    render_one_video: bool = True,
) -> dict:
    import os
    import numpy as np
    import torch

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env, init_random_episode, save_video, grip_contact_forces
    from think_then_act.training.subgoal_features import build_flat_observation, FLAT_OBS_DIM
    from think_then_act.training.flat_bc_multi_head import MultiHeadBCConfig, MultiHeadBCTrainer

    if not ckpt_path:
        # bc_scaling checkpoints are trained on the new mixed table-height/
        # elevated-target demo pool (2026-10-02 redo) — the older
        # flat_bc_{policy_type}_n2000_seed0.pt checkpoints were trained on
        # table-height-only data and are now stale for this eval's purpose.
        ckpt_path = f"checkpoints/bc_scaling/{policy_type}_n2000_seed0_smw0.0_k5.pt"

    cfg = MultiHeadBCConfig(obs_dim=FLAT_OBS_DIM, policy_type=policy_type)
    trainer = MultiHeadBCTrainer(cfg)
    trainer.load_checkpoint(os.path.join(MODEL_CACHE_DIR, ckpt_path))
    print(f"[{policy_type}] loaded {ckpt_path}", flush=True)

    env = gym.make("FetchPickAndPlace-v3", max_episode_steps=max_steps, render_mode="rgb_array")
    setup_env(env)

    n_genuine = 0
    n_is_success_raw = 0
    video_saved = False
    out_path = f"demonstrations/videos/elevated_target_{policy_type}.mp4"

    for ep in range(n_eval_episodes):
        seed = 200_000 + ep   # distinct seed range from the table-height evals,
                                # so these are genuinely new episodes, not a
                                # re-read of ones already seen table-height-only
        rng = np.random.default_rng(seed)
        env.reset(seed=seed)
        obs, setup_ok = init_random_episode(env, rng)
        if not setup_ok:
            continue

        # Raise the target above the table — init_random_episode itself is
        # untouched; this just overrides its result afterward.
        elevation = float(rng.uniform(elevation_min, elevation_max))
        env.unwrapped.goal[2] += elevation
        obs, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        if terminated or truncated:
            continue

        want_video = render_one_video and not video_saved
        frames = [env.render()] if want_video else None

        hidden_state = None
        success = False
        ever_lifted_and_gripped = False
        for _ in range(max_steps):
            flat_obs = build_flat_observation(obs["observation"], obs["achieved_goal"], obs["desired_goal"])
            action, hidden_state = trainer.actor.act(flat_obs, hidden_state, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if want_video:
                frames.append(env.render())

            height_above_resting = float(obs["achieved_goal"][2]) - block_resting_z
            forces = grip_contact_forces(env)
            if min(forces["left"], forces["right"]) > 0.0 and height_above_resting > lift_threshold:
                ever_lifted_and_gripped = True
            if info.get("is_success", False):
                success = True
            if terminated or truncated:
                break

        genuine = success and ever_lifted_and_gripped
        if success:
            n_is_success_raw += 1
        if genuine:
            n_genuine += 1

        print(f"[{policy_type}] ep={ep:>3} elevation={elevation:.3f}  "
              f"success={success}  genuine={genuine}", flush=True)

        if want_video:
            full_out_path = os.path.join(MODEL_CACHE_DIR, out_path)
            os.makedirs(os.path.dirname(full_out_path), exist_ok=True)
            save_video(frames, full_out_path, fps=20)
            model_volume.commit()
            video_saved = True
            print(f"[{policy_type}] saved video -> {full_out_path}", flush=True)

    env.close()
    completion_rate = n_genuine / n_eval_episodes if n_eval_episodes else 0.0
    raw_rate = n_is_success_raw / n_eval_episodes if n_eval_episodes else 0.0
    print(f"[{policy_type}] ELEVATED-TARGET completion_rate: {n_genuine}/{n_eval_episodes} "
          f"({completion_rate:.1%})  raw_is_success={n_is_success_raw}/{n_eval_episodes} ({raw_rate:.1%})",
          flush=True)

    return {"policy_type": policy_type, "n_eval_episodes": n_eval_episodes, "n_genuine": n_genuine,
            "n_is_success_raw": n_is_success_raw, "completion_rate": completion_rate,
            "video_path": out_path if video_saved else None}


@app.local_entrypoint()
def main(n_eval_episodes: int = 30, elevation_min: float = 0.10, elevation_max: float = 0.30):
    results = list(eval_elevated_target.starmap(
        [(pt, "", n_eval_episodes, 100, elevation_min, elevation_max) for pt in ARCHITECTURES]
    ))
    print("\nSummary (elevated-target completion_rate):")
    for r in results:
        print(f"  {r['policy_type']:16s}  {r['n_genuine']}/{r['n_eval_episodes']} "
              f"({r['completion_rate']:.1%})  video={r['video_path']}")
