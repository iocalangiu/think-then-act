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
    checkpoint_every: int = 5,
    eval_every: int = 10,
    n_eval_episodes: int = 5,
    wandb_project: str = "",
    resume_from_iter: int = 0,
    sft_warmstart: bool = False,
) -> dict:
    import os, time, json
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
    from env_utils import setup_env, init_random_episode, save_video

    print("\n" + "=" * 60)
    print("  MILESTONE 6C — GRPO Full Training Run")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Structured diagnostic log — persisted to the volume independent of
    # W&B, so every run leaves an offline-replayable record. Appends one
    # JSON line per event (train/canary/eval/baseline/final) and commits
    # immediately: the file is tiny, so the extra commit is cheap relative
    # to the checkpoint commits, and it means a spot-instance preemption
    # loses at most the in-flight iteration, not the whole log.
    # ------------------------------------------------------------------
    log_dir = os.path.join(MODEL_CACHE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    metrics_log_path = os.path.join(log_dir, "grpo_m6c_metrics_v2.jsonl")

    def log_jsonl(record: dict) -> None:
        record = {"ts": round(time.time(), 3), **record}
        with open(metrics_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        model_volume.commit()

    print(f"\nDiagnostic log → {metrics_log_path}")

    # ------------------------------------------------------------------
    # W&B setup (optional — only runs if WANDB_API_KEY is set)
    # ------------------------------------------------------------------
    wandb_enabled = bool(os.environ.get("WANDB_API_KEY") and wandb_project)
    if wandb_enabled:
        import wandb
        print(f"\nW&B logging → project: {wandb_project}")
    else:
        print("\nW&B not configured — metrics logged to stdout + JSONL only.")

    # ------------------------------------------------------------------
    # Model + trainer
    #
    # MAX_EPISODE_STEPS: the oracle needs up to ~50 steps for full
    # APPROACH+GRASP+CARRY+success on the harder training seeds (per
    # analyze_seeds.py). The old cap of 25 (train) / 20 (eval) meant success
    # was structurally unreachable even for a perfect policy — every episode
    # got cut off mid-CARRY. Bumped to 45 for both, so success_rate is
    # actually a meaningful signal. This roughly doubles per-iteration
    # rollout-collection time.
    # ------------------------------------------------------------------
    MAX_EPISODE_STEPS = 45

    config = GRPOConfig(
        max_episode_steps=MAX_EPISODE_STEPS,
        n_states=2,            # 2 seeds × 4 rollouts = 8 rollouts per iter
        reward_noise_std=0.0,  # SFT model generates diverse actions; no artificial noise needed
        lr=5e-6,               # 10× lower than default — SFT warmstart needs gentle updates
        max_grad_norm=0.5,     # tighter clip to prevent format-breaking gradient spikes
    )
    trainer = GRPOTrainer(config)

    if sft_warmstart:
        sft_path = os.path.join(MODEL_CACHE_DIR, "checkpoints", "sft_warmstart")
        print(f"\n[SFT Warmstart] Loading SFT checkpoint: {sft_path}")
        trainer.load_checkpoint(sft_path)
    elif resume_from_iter > 0:
        ckpt = os.path.join(MODEL_CACHE_DIR, "checkpoints", f"grpo_m6c_iter_{resume_from_iter}")
        print(f"\n[Resume] Loading checkpoint from iter {resume_from_iter}: {ckpt}")
        trainer.load_checkpoint(ckpt)

    # ------------------------------------------------------------------
    # Environments
    # ------------------------------------------------------------------
    train_env = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                 max_episode_steps=MAX_EPISODE_STEPS)
    )
    eval_env = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                 max_episode_steps=MAX_EPISODE_STEPS)
    )
    setup_env(train_env)
    setup_env(eval_env)

    # ------------------------------------------------------------------
    # Interleaved eval helper (seeds 9000+ to avoid training seed collision)
    # ------------------------------------------------------------------
    def quick_eval(n_episodes: int, grpo_iter: int = 0, save_video_flag: bool = True,
                   seeds: list = None) -> dict:
        from PIL import Image as PILImage
        from qwen_vl_utils import process_vision_info

        trainer.model.eval()
        rewards, parse_rates, successes = [], [], []

        video_dir = os.path.join(MODEL_CACHE_DIR, "videos")
        os.makedirs(video_dir, exist_ok=True)

        eval_seeds = seeds if seeds is not None else [9000 + ep for ep in range(n_episodes)]

        for ep, seed in enumerate(eval_seeds):
            obs, _ = eval_env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            obs, _ = init_random_episode(eval_env, rng)
            ep_reward, n_parsed, n_steps, success = 0.0, 0, 0, False
            frames = []

            for _ in range(MAX_EPISODE_STEPS):
                frame       = eval_env.last_frame()
                frames.append(frame)
                obs_arr     = np.array(obs["observation"])
                gripper_pos = [round(v, 4) for v in obs_arr[0:3]]
                achieved    = [round(v, 4) for v in obs["achieved_goal"]]
                desired     = [round(v, 4) for v in obs["desired_goal"]]
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
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.95,
                        stop_strings=["</action>"],
                        tokenizer=trainer.processor.tokenizer,
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

            if save_video_flag and frames:
                outcome = "success" if success else "fail"
                vid_path = os.path.join(video_dir, f"iter{grpo_iter:03d}_ep{ep:02d}_{outcome}.mp4")
                save_video(frames, vid_path, fps=10)
                print(f"  [video] saved → {os.path.basename(vid_path)}")

        if save_video_flag:
            model_volume.commit()

        trainer.model.train()

        return {
            "success_rate"     : round(float(np.mean(successes)),   4),
            "mean_return"      : round(float(np.mean(rewards)),     4),
            "action_parse_rate": round(float(np.mean(parse_rates)), 4),
        }

    # ------------------------------------------------------------------
    # Baseline eval (skipped when resuming)
    # ------------------------------------------------------------------
    if resume_from_iter == 0:
        print(f"\n[Pre-train eval] {n_eval_episodes} episodes (seeds 9000–{8999+n_eval_episodes})...")
        baseline = quick_eval(n_eval_episodes, grpo_iter=0)
        print(f"  Baseline → success={baseline['success_rate']:.1%}  "
              f"return={baseline['mean_return']:.4f}  "
              f"parse={baseline['action_parse_rate']:.1%}")
        log_jsonl({"type": "baseline", "iteration": 0, **baseline})
    else:
        baseline = {"success_rate": None, "mean_return": None, "action_parse_rate": None}
        print(f"\n[Resume from iter {resume_from_iter}] Skipping baseline eval.")

    if wandb_enabled:
        run_name = (f"grpo-m6c-{n_iterations}iters-from{resume_from_iter}"
                    if resume_from_iter > 0 else f"grpo-m6c-{n_iterations}iters")
        wandb.init(project=wandb_project, name=run_name, config=config.as_dict())
        if resume_from_iter == 0:
            wandb.log({"eval/success_rate" : baseline["success_rate"],
                       "eval/mean_return"  : baseline["mean_return"],
                       "eval/parse_rate"   : baseline["action_parse_rate"]}, step=0)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\nStarting {n_iterations} GRPO iterations "
          f"(ckpt every {checkpoint_every}, eval every {eval_every})...\n")

    history = []

    for i in range(resume_from_iter, n_iterations):
        print(f"--- Iteration {i + 1} / {n_iterations} ---")
        t0      = time.time()
        metrics = trainer.train_iteration(train_env, i)
        elapsed = time.time() - t0
        metrics["elapsed_s"] = round(elapsed, 1)
        history.append(metrics)

        parse_rate = metrics.get("parse_rate", float("nan"))
        avg_within_std = metrics.get("avg_within_std", float("nan"))
        print(f"  loss={metrics['loss']:.4f}  "
              f"mean_reward={metrics['mean_reward']:.4f}  "
              f"std={metrics['std_reward']:.4f}  "
              f"avg_within_std={avg_within_std:.4f}  "
              f"mean_abs_adv={metrics['mean_abs_adv']:.4f}  "
              f"parse={parse_rate:.1%}  "
              f"elapsed={elapsed:.1f}s")

        log_jsonl({"type": "train", **metrics})

        if wandb_enabled:
            wandb.log({
                "train/loss"          : metrics["loss"],
                "train/mean_reward"   : metrics["mean_reward"],
                "train/std_reward"    : metrics["std_reward"],
                "train/avg_within_std": avg_within_std,
                "train/mean_abs_adv"  : metrics["mean_abs_adv"],
                "train/parse_rate"    : parse_rate,
            }, step=i + 1)

        # Canary: 1 greedy episode on a fixed seed every 5 iters — trend signal between evals.
        # Seed 7777 is independent of the periodic eval's seeds (9000+), and never rotates,
        # so it isolates real task-learning drift from seed-difficulty noise.
        if (i + 1) % 5 == 0 and (i + 1) % eval_every != 0:
            canary = quick_eval(1, grpo_iter=-(i + 1), save_video_flag=False, seeds=[7777])
            print(f"  [canary] return={canary['mean_return']:.4f}  "
                  f"parse={canary['action_parse_rate']:.1%}  (seed 7777, greedy)")
            log_jsonl({"type": "canary", "iteration": i + 1, **canary})
            if wandb_enabled:
                wandb.log({"canary/return": canary["mean_return"],
                           "canary/parse" : canary["action_parse_rate"]}, step=i + 1)

        # Checkpoint
        if (i + 1) % checkpoint_every == 0:
            ckpt = os.path.join(MODEL_CACHE_DIR, "checkpoints", f"grpo_m6c_iter_{i+1}")
            trainer.save_checkpoint(ckpt)
            model_volume.commit()
            print(f"  [ckpt] Saved → {ckpt}")

        # Interleaved eval
        if (i + 1) % eval_every == 0:
            print(f"  [eval] Running {n_eval_episodes} episodes...")
            em = quick_eval(n_eval_episodes, grpo_iter=i + 1)
            print(f"  [eval] success={em['success_rate']:.1%}  "
                  f"return={em['mean_return']:.4f}  "
                  f"parse={em['action_parse_rate']:.1%}")
            log_jsonl({"type": "eval", "iteration": i + 1, **em})
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
    final_eval = quick_eval(n_eval_episodes, grpo_iter=n_iterations)
    model_volume.commit()

    print(f"  Final → success={final_eval['success_rate']:.1%}  "
          f"return={final_eval['mean_return']:.4f}  "
          f"parse={final_eval['action_parse_rate']:.1%}")
    log_jsonl({"type": "final", "iteration": n_iterations, **final_eval})

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
    checkpoint_every: int = 5,
    eval_every: int = 10,
    n_eval_episodes: int = 5,
    wandb_project: str = "",
    resume_from_iter: int = 0,
    sft_warmstart: bool = False,
):
    if resume_from_iter > 0:
        print(f"\nResuming M6C from iter {resume_from_iter} → {n_iterations}...")
    else:
        print(f"\nDispatching M6C to Modal (A10G, {n_iterations} iterations)...")
    if wandb_project:
        print(f"W&B project: {wandb_project}")
    print("Spawning job — terminal will return immediately. Monitor at https://modal.com\n")

    handle = run_training.spawn(
        n_iterations     = n_iterations,
        checkpoint_every = checkpoint_every,
        eval_every       = eval_every,
        n_eval_episodes  = n_eval_episodes,
        wandb_project    = wandb_project,
        resume_from_iter = resume_from_iter,
        sft_warmstart    = sft_warmstart,
    )

    print(f"Job spawned. Function call ID: {handle.object_id}")
    print(f"\nCheckpoint will be saved to:")
    print(f"  /model-cache/checkpoints/grpo_m6c_final")
    print(f"\nDownload after training completes:")
    print(f"  modal volume get rl-harness-model-cache checkpoints/grpo_m6c_final ./grpo_m6c_final")
