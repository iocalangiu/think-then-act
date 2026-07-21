"""
eval.py

Milestone 6B — Policy evaluation harness.

Run with:
    modal run scripts/eval.py                                                   # base model
    modal run scripts/eval.py --checkpoint-path /model-cache/checkpoints/grpo_iter_5

What this does:
  1. Loads Qwen2-VL-2B-Instruct + optional LoRA checkpoint.
  2. Runs N=20 deterministic (greedy) episodes on FetchPickAndPlace-v3.
  3. Tracks per-episode: total reward, steps taken, success, action parse rate.
  4. Saves the first episode as an MP4 to the Modal Volume.
  5. Returns aggregated metrics dict.

Use this before and after training to compare base vs trained policy.
Download the rollout video with:
    modal volume get rl-harness-model-cache eval_rollout.mp4 ./artifacts/eval_rollout.mp4
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


# ---------------------------------------------------------------------------
# Remote function
# ---------------------------------------------------------------------------

@app.function(
    image=rl_image,
    gpu="T4",
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=3600,
)
def run_evaluation(
    checkpoint_path: str | None = None,
    n_episodes: int = 10,
    max_episode_steps: int = 20,
) -> dict:
    import os, time
    import numpy as np
    from think_then_act.env.setup import save_video

    os.environ["MUJOCO_GL"]          = "osmesa"
    os.environ["PYOPENGL_PLATFORM"]  = "osmesa"
    os.environ["HF_HOME"]            = MODEL_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_DIR

    import torch
    from PIL import Image as PILImage
    from qwen_vl_utils import process_vision_info
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.reward.dense_reward import compute_dense_reward
    from think_then_act.policy.vlm_policy import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, VLMPolicy
    from think_then_act.policy.model_loader import MODEL_ID, load_base_model, load_lora_checkpoint

    print("\n" + "=" * 60)
    label = f"LoRA:{checkpoint_path.split('/')[-1]}" if checkpoint_path else "base (no LoRA)"
    print(f"  MILESTONE 6B — Policy Evaluation [{label}]")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print("\n[1/4] Loading model...")
    base_model, processor = load_base_model(MODEL_ID, cache_dir=MODEL_CACHE_DIR)

    if checkpoint_path:
        print(f"  Applying LoRA from {checkpoint_path}...")
        model = load_lora_checkpoint(base_model, checkpoint_path)
    else:
        model = base_model

    model.eval()
    print(f"  Ready: {label}")

    # ------------------------------------------------------------------
    # 2. Create environment
    # ------------------------------------------------------------------
    print("\n[2/4] Creating FetchPickAndPlace-v3...")
    env = gym.make(
        "FetchPickAndPlace-v3",
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
    )
    env = ObservationHarness(env)

    # ------------------------------------------------------------------
    # 3. Episode runner
    # ------------------------------------------------------------------
    def run_episode(seed: int) -> dict:
        current_obs, _ = env.reset(seed=seed)

        episode_reward    = 0.0
        n_steps           = 0
        n_parsed          = 0
        success           = False
        steps_to_success  = None

        for _ in range(max_episode_steps):
            frame = env.last_frame()

            obs_arr     = np.array(current_obs["observation"])
            gripper_pos = [round(v, 4) for v in obs_arr[0:3]]
            achieved    = [round(v, 4) for v in current_obs["achieved_goal"]]
            desired     = [round(v, 4) for v in current_obs["desired_goal"]]
            user_text   = USER_PROMPT_TEMPLATE.format(
                gripper_pos=gripper_pos, achieved_goal=achieved, desired_goal=desired,
            )
            pil_image = PILImage.fromarray(frame).resize((224, 224), PILImage.LANCZOS)
            messages  = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text",  "text": user_text},
                ]},
            ]

            text_input     = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            img_inputs, vid_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text_input], images=img_inputs, videos=vid_inputs,
                return_tensors="pt", padding=True,
            ).to("cuda")

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.3,   # near-deterministic; greedy (do_sample=False)
                    top_p=0.95,        # collapses format compliance to ~10% with this prompt
                )

            gen_ids  = [o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)]
            response = processor.batch_decode(
                gen_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            action, action_found = VLMPolicy._parse_action(response)
            if action_found:
                n_parsed += 1

            next_obs, _, terminated, truncated, info = env.step(action)
            step_reward, _ = compute_dense_reward(
                obs           = next_obs["observation"],
                achieved_goal = next_obs["achieved_goal"],
                desired_goal  = next_obs["desired_goal"],
                info          = info,
            )
            episode_reward += step_reward
            n_steps        += 1

            if info.get("is_success", False) and steps_to_success is None:
                success          = True
                steps_to_success = n_steps

            current_obs = next_obs
            if terminated or truncated:
                break

        return {
            "seed"             : seed,
            "total_reward"     : episode_reward,
            "n_steps"          : n_steps,
            "n_parsed"         : n_parsed,
            "success"          : success,
            "steps_to_success" : steps_to_success,
            "frames"           : env.frames_as_array(),   # (T, H, W, 3) uint8
        }

    # ------------------------------------------------------------------
    # 4. Run evaluation episodes
    # ------------------------------------------------------------------
    print(f"\n[3/4] Running {n_episodes} episodes (temp=0.3, seed=1000–{999+n_episodes})...")
    print(f"  {'ep':>3}  {'steps':>5}  {'parsed':>8}  {'reward':>9}  status")
    print("  " + "-" * 44)

    results    = []
    video_frames = None

    for ep in range(n_episodes):
        t0     = time.time()
        result = run_episode(seed=1000 + ep)
        elapsed = time.time() - t0

        parse_pct = result["n_parsed"] / max(result["n_steps"], 1)
        status    = "SUCCESS" if result["success"] else "      "
        print(f"  {ep:>3}  {result['n_steps']:>5}  "
              f"{result['n_parsed']}/{result['n_steps']} ({parse_pct:.0%})  "
              f"{result['total_reward']:>9.3f}  {status}  ({elapsed:.0f}s)")

        if ep == 0:
            video_frames = result["frames"]

        results.append({k: v for k, v in result.items() if k != "frames"})

    env.close()

    # ------------------------------------------------------------------
    # 5. Save rollout video (first episode)
    # ------------------------------------------------------------------
    print("\n[4/4] Saving rollout video...")
    video_path = os.path.join(MODEL_CACHE_DIR, "eval_rollout.mp4")
    save_video(video_frames, video_path, fps=10)
    model_volume.commit()
    print(f"  Saved → {video_path}")
    print(f"  Download: modal volume get rl-harness-model-cache eval_rollout.mp4 ./artifacts/eval_rollout.mp4")

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    n_success    = sum(1 for r in results if r["success"])
    success_rate = n_success / n_episodes
    mean_return  = float(np.mean([r["total_reward"]  for r in results]))
    mean_steps   = float(np.mean([r["n_steps"]        for r in results]))
    parse_rate   = float(np.mean(
        [r["n_parsed"] / max(r["n_steps"], 1) for r in results]
    ))
    suc_steps    = [r["steps_to_success"] for r in results if r["steps_to_success"] is not None]
    mean_steps_to_success = float(np.mean(suc_steps)) if suc_steps else None

    metrics = {
        "model"                : label,
        "n_episodes"           : n_episodes,
        "success_rate"         : round(success_rate, 4),
        "n_successes"          : n_success,
        "mean_return"          : round(mean_return,  4),
        "mean_episode_steps"   : round(mean_steps,   2),
        "action_parse_rate"    : round(parse_rate,   4),
        "mean_steps_to_success": (round(mean_steps_to_success, 2)
                                  if mean_steps_to_success else None),
        "video_path"           : video_path,
        "status"               : "PASS",
    }

    print("\n" + "=" * 60)
    print(f"  Model           : {label}")
    print(f"  Success rate    : {n_success}/{n_episodes}  ({success_rate:.1%})")
    print(f"  Mean return     : {mean_return:.4f}")
    print(f"  Action parse    : {parse_rate:.1%}")
    if mean_steps_to_success:
        print(f"  Steps to success: {mean_steps_to_success:.1f}")
    print(f"  Video           : {video_path}")
    print("=" * 60 + "\n")

    return metrics


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(checkpoint_path: str = ""):
    import json

    ckpt   = checkpoint_path or None
    label  = f"LoRA checkpoint: {ckpt}" if ckpt else "base model (no training)"

    print(f"\nEvaluating {label}...")
    result = run_evaluation.remote(checkpoint_path=ckpt)

    print("\n--- Evaluation Summary ---")
    display = {k: v for k, v in result.items() if k != "video_path"}
    print(json.dumps(display, indent=2))

    assert result["status"] == "PASS"
    assert result["action_parse_rate"] > 0.5, (
        f"Action parse rate {result['action_parse_rate']:.1%} < 50% — "
        "model is not producing valid <action> tags reliably."
    )

    print(f"\nSuccess rate : {result['success_rate']:.1%}  ({result['n_successes']}/{result['n_episodes']})")
    print(f"Mean return  : {result['mean_return']:.4f}")
    print(f"Parse rate   : {result['action_parse_rate']:.1%}")
    print(f"\nDownload video:")
    print(f"  modal volume get rl-harness-model-cache eval_rollout.mp4 ./artifacts/eval_rollout.mp4")
    print(f"\nMilestone 6B complete.")
