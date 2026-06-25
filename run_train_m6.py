"""
run_train_m6.py

Milestone 6C — Full GRPO training run (50-100 iterations).

  • Checkpoints every N iterations → Modal Volume
  • Interleaved eval every N iterations (success_rate, mean_return, parse_rate)
  • Optional W&B logging

Run with:
    modal run --detach run_train_m6.py
    modal run --detach run_train_m6.py --n-iterations 100

W&B (optional):
    modal secret create wandb-secret WANDB_API_KEY=<your-key>
    modal run --detach run_train_m6.py --wandb-project rl-harness-robotics

Compare final metrics against M6B baseline:
    success_rate=0%  mean_return=-35.86  parse_rate=61%

Checkpoints land at:
    /model-cache/checkpoints/grpo_m6c_iter_{N}   (every checkpoint_every iters)
    /model-cache/checkpoints/grpo_m6c_final       (end of run)

Download final checkpoint:
    modal volume get rl-harness-model-cache checkpoints/grpo_m6c_final ./grpo_m6c_final

Run full eval against it:
    modal run eval.py --checkpoint-path /model-cache/checkpoints/grpo_m6c_final
"""

import modal
from modal_config import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(
    image=rl_image,
    gpu="A10G",
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=3600 * 10,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_training(
    n_iterations: int = 50,
    checkpoint_every: int = 10,
    eval_every: int = 10,
    n_eval_episodes: int = 5,
    wandb_project: str = "",
) -> dict:
    import os, time
    import numpy as np
    import torch

    os.environ["MUJOCO_GL"]          = "osmesa"
    os.environ["PYOPENGL_PLATFORM"]  = "osmesa"
    os.environ["HF_HOME"]            = MODEL_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_DIR

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    from obs_wrapper import ObservationHarness
    from trainer import GRPOTrainer, GRPOConfig
    from policy import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, VLMPolicy
    from reward import compute_dense_reward

    print("\n" + "=" * 60)
    print("  MILESTONE 6C — GRPO Full Training Run")
    print("=" * 60)

    # ------------------------------------------------------------------
    # W&B setup (optional — only runs if WANDB_API_KEY is set)
    # ------------------------------------------------------------------
    wandb_enabled = bool(os.environ.get("WANDB_API_KEY") and wandb_project)
    if wandb_enabled:
        import wandb
        print(f"\nW&B logging → project: {wandb_project}")
    else:
        print("\nW&B not configured — metrics logged to stdout only.")

    # ------------------------------------------------------------------
    # Model + trainer
    # ------------------------------------------------------------------
    config = GRPOConfig(
        max_episode_steps=10,   # more steps per rollout than M5's 5 for richer signal
        reward_noise_std=0.02,  # reduced from M5's 0.05; small safety margin
    )
    trainer = GRPOTrainer(config)

    # ------------------------------------------------------------------
    # Environments
    # ------------------------------------------------------------------
    train_env = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array", max_episode_steps=50)
    )
    eval_env = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array", max_episode_steps=20)
    )

    # ------------------------------------------------------------------
    # Interleaved eval helper (seeds 9000+ to avoid training seed collision)
    # ------------------------------------------------------------------
    def quick_eval(n_episodes: int) -> dict:
        from PIL import Image as PILImage
        from qwen_vl_utils import process_vision_info

        trainer.model.eval()
        rewards, parse_rates, successes = [], [], []

        for ep in range(n_episodes):
            obs, _ = eval_env.reset(seed=9000 + ep)
            ep_reward, n_parsed, n_steps, success = 0.0, 0, 0, False

            for _ in range(20):
                frame    = eval_env.last_frame()
                achieved = [round(v, 4) for v in obs["achieved_goal"]]
                desired  = [round(v, 4) for v in obs["desired_goal"]]
                distance = float(np.linalg.norm(
                    np.array(obs["desired_goal"]) - np.array(obs["achieved_goal"])
                ))
                user_text = USER_PROMPT_TEMPLATE.format(
                    achieved_goal=achieved, desired_goal=desired, distance=distance
                )
                pil_image = PILImage.fromarray(frame).resize((224, 224), PILImage.LANCZOS)
                messages  = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text",  "text": user_text},
                    ]},
                ]
                text_input     = trainer.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                img_inputs, vid_inputs = process_vision_info(messages)
                inputs = trainer.processor(
                    text=[text_input], images=img_inputs, videos=vid_inputs,
                    return_tensors="pt", padding=True,
                ).to("cuda")

                with torch.no_grad():
                    out_ids = trainer.model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.95,
                    )
                gen_ids  = [o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)]
                response = trainer.processor.batch_decode(
                    gen_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                action, action_found = VLMPolicy._parse_action(response)
                if action_found:
                    n_parsed += 1

                next_obs, _, terminated, truncated, info = eval_env.step(action)
                step_reward, _ = compute_dense_reward(
                    obs           = next_obs["observation"],
                    achieved_goal = next_obs["achieved_goal"],
                    desired_goal  = next_obs["desired_goal"],
                    info          = info,
                )
                ep_reward += step_reward
                n_steps   += 1
                if info.get("is_success", False):
                    success = True
                obs = next_obs
                if terminated or truncated:
                    break

            rewards.append(ep_reward)
            parse_rates.append(n_parsed / max(n_steps, 1))
            successes.append(float(success))

        trainer.model.train()

        return {
            "success_rate"     : round(float(np.mean(successes)),   4),
            "mean_return"      : round(float(np.mean(rewards)),     4),
            "action_parse_rate": round(float(np.mean(parse_rates)), 4),
        }

    # ------------------------------------------------------------------
    # Baseline eval before any training (step 0)
    # ------------------------------------------------------------------
    print(f"\n[Pre-train eval] {n_eval_episodes} episodes (seeds 9000–{8999+n_eval_episodes})...")
    baseline = quick_eval(n_eval_episodes)
    print(f"  Baseline → success={baseline['success_rate']:.1%}  "
          f"return={baseline['mean_return']:.4f}  "
          f"parse={baseline['action_parse_rate']:.1%}")

    if wandb_enabled:
        wandb.init(
            project=wandb_project,
            name=f"grpo-m6c-{n_iterations}iters",
            config=config.as_dict(),
        )
        wandb.log({"eval/success_rate" : baseline["success_rate"],
                   "eval/mean_return"  : baseline["mean_return"],
                   "eval/parse_rate"   : baseline["action_parse_rate"]}, step=0)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\nStarting {n_iterations} GRPO iterations "
          f"(ckpt every {checkpoint_every}, eval every {eval_every})...\n")

    history = []

    for i in range(n_iterations):
        print(f"--- Iteration {i + 1} / {n_iterations} ---")
        t0      = time.time()
        metrics = trainer.train_iteration(train_env, i)
        elapsed = time.time() - t0
        metrics["elapsed_s"] = round(elapsed, 1)
        history.append(metrics)

        print(f"  loss={metrics['loss']:.6f}  "
              f"mean_reward={metrics['mean_reward']:.4f}  "
              f"std={metrics['std_reward']:.4f}  "
              f"elapsed={elapsed:.1f}s")

        if wandb_enabled:
            wandb.log({
                "train/loss"        : metrics["loss"],
                "train/mean_reward" : metrics["mean_reward"],
                "train/std_reward"  : metrics["std_reward"],
                "train/mean_abs_adv": metrics["mean_abs_adv"],
            }, step=i + 1)

        # Checkpoint
        if (i + 1) % checkpoint_every == 0:
            ckpt = os.path.join(MODEL_CACHE_DIR, "checkpoints", f"grpo_m6c_iter_{i+1}")
            trainer.save_checkpoint(ckpt)
            model_volume.commit()
            print(f"  [ckpt] Saved → {ckpt}")

        # Interleaved eval
        if (i + 1) % eval_every == 0:
            print(f"  [eval] Running {n_eval_episodes} episodes...")
            em = quick_eval(n_eval_episodes)
            print(f"  [eval] success={em['success_rate']:.1%}  "
                  f"return={em['mean_return']:.4f}  "
                  f"parse={em['action_parse_rate']:.1%}")
            if wandb_enabled:
                wandb.log({
                    "eval/success_rate" : em["success_rate"],
                    "eval/mean_return"  : em["mean_return"],
                    "eval/parse_rate"   : em["action_parse_rate"],
                }, step=i + 1)

        print()

    # ------------------------------------------------------------------
    # Final checkpoint + eval
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Training complete. Saving final checkpoint...")
    final_ckpt = os.path.join(MODEL_CACHE_DIR, "checkpoints", "grpo_m6c_final")
    trainer.save_checkpoint(final_ckpt)

    print(f"[Final eval] Running {n_eval_episodes} episodes...")
    final_eval = quick_eval(n_eval_episodes)
    model_volume.commit()

    print(f"  Final → success={final_eval['success_rate']:.1%}  "
          f"return={final_eval['mean_return']:.4f}  "
          f"parse={final_eval['action_parse_rate']:.1%}")

    if wandb_enabled:
        wandb.log({
            "eval/success_rate" : final_eval["success_rate"],
            "eval/mean_return"  : final_eval["mean_return"],
            "eval/parse_rate"   : final_eval["action_parse_rate"],
        }, step=n_iterations)
        wandb.finish()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    loss_trend   = [round(h["loss"],        6) for h in history]
    reward_trend = [round(h["mean_reward"], 4) for h in history]

    print("\n" + "=" * 60)
    print(f"  Baseline → success={baseline['success_rate']:.1%}  "
          f"return={baseline['mean_return']:.4f}  "
          f"parse={baseline['action_parse_rate']:.1%}")
    print(f"  Final    → success={final_eval['success_rate']:.1%}  "
          f"return={final_eval['mean_return']:.4f}  "
          f"parse={final_eval['action_parse_rate']:.1%}")
    print(f"  Checkpoint: {final_ckpt}")
    print("=" * 60 + "\n")

    return {
        "status"      : "PASS",
        "n_iterations": n_iterations,
        "loss_trend"  : loss_trend,
        "reward_trend": reward_trend,
        "baseline"    : baseline,
        "final_eval"  : final_eval,
        "final_ckpt"  : final_ckpt,
        "config"      : config.as_dict(),
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    n_iterations: int = 50,
    checkpoint_every: int = 10,
    eval_every: int = 10,
    n_eval_episodes: int = 5,
    wandb_project: str = "",
):
    import json

    print(f"\nDispatching M6C to Modal (A10G, {n_iterations} iterations)...")
    if wandb_project:
        print(f"W&B project: {wandb_project}")
    print("Use --detach to run without keeping terminal open. Monitor at https://modal.com\n")

    result = run_training.remote(
        n_iterations    = n_iterations,
        checkpoint_every = checkpoint_every,
        eval_every      = eval_every,
        n_eval_episodes = n_eval_episodes,
        wandb_project   = wandb_project,
    )

    print("\n--- M6C Summary ---")
    display = {k: v for k, v in result.items() if k not in ("loss_trend", "reward_trend", "config")}
    print(json.dumps(display, indent=2))

    print(f"\nLoss trend   (last 10): {result['loss_trend'][-10:]}")
    print(f"Reward trend (last 10): {result['reward_trend'][-10:]}")

    delta_return = result["final_eval"]["mean_return"] - result["baseline"]["mean_return"]
    print(f"\nReturn delta vs baseline: {delta_return:+.4f}")

    print(f"\nDownload checkpoint:")
    print(f"  modal volume get rl-harness-model-cache checkpoints/grpo_m6c_final ./grpo_m6c_final")
    print(f"\nRun full 10-episode eval against trained policy:")
    print(f"  modal run eval.py --checkpoint-path /model-cache/checkpoints/grpo_m6c_final")
    print(f"\nMilestone 6C complete.")
