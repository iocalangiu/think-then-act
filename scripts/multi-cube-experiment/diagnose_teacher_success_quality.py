"""
diagnose_teacher_success_quality.py

One-off diagnostic (NOT a collection run, writes nothing): checks whether
the SB3 TQC+HER teacher's OWN "successful" episodes (the ones
collect_sb3_teacher_demos.py keeps as BC demonstrations) are genuine
grasp-carry-place completions, or slide/nudge artifacts that happen to end
near the goal — is_success only checks final block-to-goal distance, no
requirement the block was ever lifted/held. Built because
diagnose_flat_bc_success_quality.py found the BC-trained flat policy's own
"successes" were 0% genuine (both were slide/nudge) — if the TEACHER's
demonstrations have the same problem, the BC student would be learning to
slide because that's what a chunk of its training data actually shows, not
because of under-training. Same verification logic as that script (height
above resting + two-finger contact), applied to the teacher instead.

Image/loading identical to collect_sb3_teacher_demos.py (same TQC-load
fixes already solved there: legacy-gym custom_objects, shimmy space
conversion, dropped HerReplayBuffer kwargs, CPU-only torch, import order) —
duplicated rather than cross-imported, since scripts/ has no __init__.py
and isn't set up for cross-script imports under `modal run`.

Run with:
    modal run scripts/multi-cube-experiment/diagnose_teacher_success_quality.py --n-seeds 30
"""

import modal
from think_then_act.modal_app import app, model_volume, MODEL_CACHE_DIR

sb3_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1-mesa-glx", "libgl1-mesa-dev", "libglfw3", "libglfw3-dev",
        "libgles2-mesa-dev", "libegl1-mesa-dev", "libosmesa6", "libosmesa6-dev",
        "libglew-dev", "patchelf", "ffmpeg",
    )
    .pip_install(
        "mujoco==3.1.6",
        "gymnasium==1.0.0",
        "gymnasium-robotics==1.3.1",
        "numpy==1.26.4",
        "imageio==2.34.1",
    )
    .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "stable-baselines3==2.4.0",
        "sb3-contrib==2.4.0",
        "huggingface_sb3==3.0",
        "gym",
        "shimmy>=0.2.1",
    )
    .add_local_python_source("think_then_act", copy=True)
)


@app.function(image=sb3_image, gpu=None, cpu=4.0, memory=8192,
              volumes={MODEL_CACHE_DIR: model_volume}, timeout=1800)
def diagnose_teacher_success_quality(
    repo_id: str = "sb3/tqc-FetchPickAndPlace-v1",
    filename: str = "tqc-FetchPickAndPlace-v1.zip",
    n_seeds: int = 30,
    max_steps: int = 100,
    apply_setup_env: bool = True,
    lift_threshold: float = 0.02,
    block_resting_z: float = 0.425,
) -> dict:
    import os
    import numpy as np

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    import torch
    torch.set_num_threads(1)
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    from huggingface_sb3 import load_from_hub
    from sb3_contrib import TQC
    from sb3_contrib.common.wrappers import TimeFeatureWrapper

    from think_then_act.env.setup import setup_env, init_random_episode, grip_contact_forces

    print("\n" + "=" * 70)
    print("  TEACHER SUCCESS-QUALITY CHECK (is its 'success' a real grasp-carry?)")
    print("=" * 70)

    env = gym.make("FetchPickAndPlace-v3", max_episode_steps=max_steps)
    if apply_setup_env:
        setup_env(env)
    env = TimeFeatureWrapper(env, max_steps=max_steps)

    ckpt_path = load_from_hub(repo_id=repo_id, filename=filename)
    custom_objects = {
        "learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0,
        "observation_space": env.observation_space, "action_space": env.action_space,
        "replay_buffer_class": None, "replay_buffer_kwargs": {},
    }
    model = TQC.load(ckpt_path, env=env, device="cpu", custom_objects=custom_objects)
    print(f"  Loaded {repo_id}/{filename}")

    results = []
    for seed in range(n_seeds):
        env.reset(seed=seed)
        # Same init_random_episode table-centered placement
        # collect_sb3_teacher_demos.py actually uses — matching it exactly so
        # this checks the SAME episode distribution that produced our real
        # demonstrations, not the native env's own (different) placement.
        rng = np.random.default_rng(seed)
        obs, setup_ok = init_random_episode(env, rng)
        if not setup_ok:
            continue
        success_step = None
        max_height = -999.0
        ever_two_finger_contact_while_lifted = False

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            block_z = float(obs["achieved_goal"][2])
            height_above_resting = block_z - block_resting_z
            max_height = max(max_height, height_above_resting)

            forces = grip_contact_forces(env)
            two_finger_contact = min(forces["left"], forces["right"]) > 0.0
            if two_finger_contact and height_above_resting > lift_threshold:
                ever_two_finger_contact_while_lifted = True

            if info.get("is_success", False) and success_step is None:
                success_step = step
            if terminated or truncated:
                break

        if success_step is not None:
            verdict = "GENUINE grasp-carry" if ever_two_finger_contact_while_lifted else \
                      "SLIDE/NUDGE — never genuinely lifted+gripped"
            print(f"  seed={seed:>3}  SUCCESS @ step {success_step:>3}  "
                  f"max_height_above_resting={max_height:+.4f}m  "
                  f"ever_lifted+gripped={ever_two_finger_contact_while_lifted}  -> {verdict}")
            results.append({"seed": seed, "success_step": success_step,
                             "max_height_above_resting": max_height,
                             "genuine_grasp_carry": ever_two_finger_contact_while_lifted})

    env.close()

    n_success = len(results)
    n_genuine = sum(1 for r in results if r["genuine_grasp_carry"])
    print("\n" + "=" * 70)
    if n_success == 0:
        print("  No successes in this sample.")
    else:
        print(f"  {n_genuine}/{n_success} teacher 'successes' were genuine grasp-carries "
              f"({100*n_genuine/n_success:.0f}%); {n_success - n_genuine}/{n_success} "
              f"were slide/nudge.")
    print("=" * 70)

    return {"n_seeds": n_seeds, "n_success": n_success, "n_genuine_grasp_carry": n_genuine,
            "results": results}


@app.local_entrypoint()
def main(n_seeds: int = 30, apply_setup_env: bool = True):
    result = diagnose_teacher_success_quality.remote(n_seeds=n_seeds, apply_setup_env=apply_setup_env)
    print(f"\nDone: {result['n_genuine_grasp_carry']}/{result['n_success']} genuine "
          f"(of {result['n_seeds']} seeds attempted)")
