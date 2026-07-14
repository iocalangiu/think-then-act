"""
run_episode.py

Sanity-check tool: run ONE episode on a real training seed with the current
policy, save the video and the full per-step VLM output (think/action/raw
response) to a JSON file — so you can eyeball whether the policy is doing
anything unsafe/nonsensical before committing more training compute to it.

Uses the same env setup as training (setup_env + init_random_episode), unlike
run_policy.py's Milestone-3 verification env, so this reflects actual
training conditions.

Run with:
    modal run scripts/run_episode.py --seed 0                    # base model (no LoRA)
    modal run scripts/run_episode.py --seed 0 --lora sft_warmstart
    modal run scripts/run_episode.py --seed 0 --lora grpo_m6c_iter_15

Outputs (on the volume, under /model-cache/episode_checks/):
    ep_seed{N}_{lora}.mp4    — video
    ep_seed{N}_{lora}.json   — per-step think/action/raw_response + state

Download with:
    modal volume get rl-harness-model-cache episode_checks/ ./artifacts/episode_checks/
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(
    image=rl_image,
    gpu="T4",
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=1800,
)
def run_episode(seed: int = 0, lora: str = "none", max_steps: int = 45) -> dict:
    import os, time, json
    import numpy as np

    os.environ["MUJOCO_GL"]          = "osmesa"
    os.environ["PYOPENGL_PLATFORM"]  = "osmesa"
    os.environ["HF_HOME"]            = MODEL_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_DIR

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.policy.vlm_policy import VLMPolicy
    from think_then_act.reward.dense_reward import compute_dense_reward
    from think_then_act.env.setup import setup_env, init_random_episode, save_video

    print("\n" + "=" * 60)
    print(f"  Episode sanity check — seed={seed}  lora={lora}")
    print("=" * 60)

    lora_path = None if lora == "none" else f"{MODEL_CACHE_DIR}/checkpoints/{lora}"
    print(f"\n[1/3] Loading policy ({'base, no LoRA' if lora_path is None else lora_path})...")
    t0 = time.time()
    policy = VLMPolicy(cache_dir=MODEL_CACHE_DIR, lora_path=lora_path)
    print(f"      Ready in {time.time() - t0:.1f}s")

    print(f"\n[2/3] Setting up env (seed={seed}, training config)...")
    env = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array", max_episode_steps=max_steps)
    )
    setup_env(env)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    obs, ok = init_random_episode(env, rng)
    if not ok:
        print("      init_random_episode failed — falling back to plain reset")
        obs, _ = env.reset(seed=seed)

    print(f"\n[3/3] Running up to {max_steps} steps...")
    frames = [env.last_frame()]
    steps  = []

    for i in range(max_steps):
        state_entry = env.episode_log[-1]
        frame        = env.last_frame()

        t_inf = time.time()
        raw_response, action, think_text, think_found, action_found = policy.act(
            frame, state_entry
        )
        inf_ms = (time.time() - t_inf) * 1000

        obs2, _, terminated, truncated, info = env.step(action)
        step_reward, breakdown = compute_dense_reward(
            obs=obs2["observation"], achieved_goal=obs2["achieved_goal"],
            desired_goal=obs2["desired_goal"], info=info,
        )
        frames.append(env.last_frame())

        print(f"  step {i:>2}: <think>={'Y' if think_found else 'N'} "
              f"<action>={'Y' if action_found else 'N'}  action={action.tolist()}  "
              f"reward={step_reward:+.2f}  d_grip_block={breakdown['d_grip_block']:.3f}  "
              f"success={info.get('is_success', False)}  inf={inf_ms:.0f}ms")

        steps.append({
            "step"          : i,
            "gripper_pos"   : [round(v, 4) for v in state_entry["observation"][0:3]],
            "achieved_goal" : [round(v, 4) for v in state_entry["achieved_goal"]],
            "desired_goal"  : [round(v, 4) for v in state_entry["desired_goal"]],
            "raw_response"  : raw_response,
            "think_text"    : think_text,
            "think_found"   : think_found,
            "action"        : action.tolist(),
            "action_found"  : action_found,
            "reward"        : round(step_reward, 4),
            "d_grip_block"  : round(breakdown["d_grip_block"], 4),
            "d_block_target": round(breakdown["d_block_target"], 4),
            "gripper_closedness": round(breakdown["gripper_closedness"], 4),
            "is_success"    : bool(info.get("is_success", False)),
            "inf_ms"        : round(inf_ms, 1),
        })

        if terminated or truncated:
            print(f"  Episode ended at step {i}.")
            break

    env.close()

    out_dir = os.path.join(MODEL_CACHE_DIR, "episode_checks")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"seed{seed}_{lora}"

    video_path = os.path.join(out_dir, f"ep_{tag}.mp4")
    save_video(frames, video_path, fps=10)

    json_path = os.path.join(out_dir, f"ep_{tag}.json")
    with open(json_path, "w") as f:
        json.dump({
            "seed": seed, "lora": lora, "max_steps": max_steps,
            "n_steps_run": len(steps),
            "final_success": steps[-1]["is_success"] if steps else False,
            "steps": steps,
        }, f, indent=2)

    model_volume.commit()
    print(f"\nSaved video → {video_path}")
    print(f"Saved log   → {json_path}")

    return {
        "video_path": video_path, "json_path": json_path,
        "n_steps": len(steps),
        "final_success": steps[-1]["is_success"] if steps else False,
    }


@app.local_entrypoint()
def main(seed: int = 0, lora: str = "none", max_steps: int = 45):
    print(f"\nDispatching episode check — seed={seed}, lora={lora}...")
    result = run_episode.remote(seed=seed, lora=lora, max_steps=max_steps)
    print(f"\nDone. n_steps={result['n_steps']}  final_success={result['final_success']}")
    print(f"\nDownload with:")
    print(f"  python3 -m modal volume get --force rl-harness-model-cache "
          f"episode_checks/ ./artifacts/episode_checks/")
