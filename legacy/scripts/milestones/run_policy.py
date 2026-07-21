"""
run_policy.py

Milestone 3 verification script.

Run with:
    modal run scripts/milestones/run_policy.py

What this does:
  1. Boots the cloud container (rl_image with torch + transformers).
  2. Downloads Qwen2-VL-2B-Instruct from HuggingFace (~4 GB, cached in
     the Modal Volume so subsequent runs are instant).
  3. Creates FetchPickAndPlace-v3 and wraps it with ObservationHarness.
  4. Runs 3 VLM-driven steps:
       frame + state → VLMPolicy.act() → <think>/<action> → env.step()
  5. Validates that <think> text was produced and <action> was parsed
     into a valid 4D numpy array.

After this milestone we have the full perception → reasoning → action
pipeline. Milestone 4 adds the dense reward; Milestone 5 trains it.
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(
    image=rl_image,
    gpu="T4",                              # 16 GB VRAM; fits Qwen2-VL-2B in fp16
    volumes={MODEL_CACHE_DIR: model_volume},  # persist downloaded weights
    timeout=1800,                          # first run downloads ~4 GB
)
def run_vlm_policy_steps(n_steps: int = 3, use_sft: bool = False) -> dict:
    """
    Runs n_steps of VLM-driven control and returns a JSON-safe summary.
    """
    import os, time
    import numpy as np

    # Headless renderer + point HuggingFace at our cached volume.
    os.environ["MUJOCO_GL"]          = "osmesa"
    os.environ["PYOPENGL_PLATFORM"]  = "osmesa"
    os.environ["HF_HOME"]            = MODEL_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_DIR

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.policy.vlm_policy import VLMPolicy

    print("\n" + "=" * 60)
    print("  MILESTONE 3 — VLM Policy Actor Verification")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load VLM policy (downloads model on first run)
    # ------------------------------------------------------------------
    print(f"\n[1/4] Loading VLM policy{'  [SFT LoRA]' if use_sft else '  [base]'}...")
    t0 = time.time()
    lora_path = f"{MODEL_CACHE_DIR}/checkpoints/sft_warmstart" if use_sft else None
    policy = VLMPolicy(cache_dir=MODEL_CACHE_DIR, lora_path=lora_path)
    print(f"      Model ready in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 2: Create environment with harness
    # ------------------------------------------------------------------
    print(f"\n[2/4] Creating environment + harness...")
    env = gym.make(
        "FetchPickAndPlace-v3",
        render_mode="rgb_array",
        max_episode_steps=50,
    )
    env = ObservationHarness(env)
    obs, info = env.reset(seed=7)
    print(f"      Environment ready. Initial frame: {env.last_frame().shape}")

    # ------------------------------------------------------------------
    # Step 3: Run n_steps VLM-driven steps
    # ------------------------------------------------------------------
    print(f"\n[3/4] Running {n_steps} VLM-driven steps...")
    step_results = []

    for i in range(n_steps):
        state_entry = env.episode_log[-1]   # most recent logged state
        frame       = env.last_frame()

        print(f"\n  --- Step {i + 1} ---")
        print(f"  achieved_goal : {[round(v, 3) for v in state_entry['achieved_goal']]}")
        print(f"  desired_goal  : {[round(v, 3) for v in state_entry['desired_goal']]}")

        t_inf = time.time()
        raw_response, action, think_text, think_found, action_found = policy.act(
            frame, state_entry
        )
        inf_ms = (time.time() - t_inf) * 1000

        # Always print the first 500 chars of the raw response so we can
        # debug format issues without having to re-run.
        print(f"\n  [RAW (first 500 chars)]\n  {raw_response[:500]}")
        print(f"\n  [THINK tag found: {think_found}]")
        if think_text:
            print(f"  {think_text[:300]}{'...' if len(think_text) > 300 else ''}")
        print(f"\n  [ACTION tag found: {action_found}]  {action.tolist()}")
        print(f"  Inference : {inf_ms:.0f} ms")

        obs, reward, terminated, truncated, info = env.step(action)

        step_results.append({
            "step"         : i + 1,
            "think_text"   : think_text,
            "raw_response" : raw_response,
            "action"       : action.tolist(),
            "reward"       : float(reward),
            "is_success"   : bool(info.get("is_success", False)),
            # Use tag_found, not non-zero, to detect real parsing vs. fallback zeros
            "think_found"  : think_found,
            "action_found" : action_found,
            "inf_ms"       : round(inf_ms, 1),
        })

        if terminated or truncated:
            print(f"  Episode ended at step {i + 1}.")
            break

    env.close()

    # ------------------------------------------------------------------
    # Step 4: Validation summary
    # ------------------------------------------------------------------
    print(f"\n[4/4] Validating outputs...")
    # Use tag_found (not non-zero) to measure real parsing.
    # Require at least 2/3 steps have each tag — one miss is acceptable
    # for a pre-training model; consistent failure would indicate a real issue.
    n = len(step_results)
    think_found_count  = sum(s["think_found"]  for s in step_results)
    action_found_count = sum(s["action_found"] for s in step_results)
    majority = n // 2 + 1   # e.g. 2 out of 3

    for s in step_results:
        ok = s["think_found"] and s["action_found"]
        print(f"  step {s['step']}: [{'OK  ' if ok else 'WARN'}]  "
              f"<think>={'yes' if s['think_found'] else 'NO '}  "
              f"<action>={'yes' if s['action_found'] else 'NO '}  "
              f"values={s['action']}  "
              f"reward={s['reward']:+.1f}  "
              f"inf={s['inf_ms']} ms")

    passed = (think_found_count >= majority) and (action_found_count >= majority)
    result = {
        "n_steps"            : n,
        "think_found_count"  : think_found_count,
        "action_found_count" : action_found_count,
        "majority_threshold" : majority,
        "avg_inf_ms"         : round(sum(s["inf_ms"] for s in step_results) / n, 1),
        "step_results"       : step_results,
        "status"             : "PASS" if passed else "WARN",
    }

    print("\n" + "=" * 60)
    print(f"  RESULT: {'PASS' if passed else 'WARN — check raw output above'}")
    print(f"  <think> found  : {think_found_count}/{n}")
    print(f"  <action> found : {action_found_count}/{n}")
    print(f"  avg inference  : {result['avg_inf_ms']} ms / step")
    print("=" * 60 + "\n")

    return result


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(n_steps: int = 15, sft: bool = False):
    import json

    print("\nDispatching VLM policy run to Modal cloud...")
    print("(First run downloads Qwen2-VL-2B-Instruct ~4 GB — cached after that)\n")
    result = run_vlm_policy_steps.remote(n_steps=n_steps, use_sft=sft)

    print("\n--- Policy Summary (local terminal) ---")
    # Print without the verbose step_results
    compact = {k: v for k, v in result.items() if k != "step_results"}
    print(json.dumps(compact, indent=2))

    # Show each step's think + action for inspection
    for s in result["step_results"]:
        print(f"\nStep {s['step']} <think> [found={s['think_found']}]:")
        print(f"  {s['think_text'][:400]}")
        print(f"Step {s['step']} <action> [found={s['action_found']}]: {s['action']}")

    assert result["n_steps"] > 0, "No steps completed"
    majority = result["majority_threshold"]
    assert result["think_found_count"] >= majority, (
        f"<think> tag found in only {result['think_found_count']}/{result['n_steps']} steps "
        f"(need >= {majority}). Check raw output above and adjust SYSTEM_PROMPT."
    )
    assert result["action_found_count"] >= majority, (
        f"<action> tag found in only {result['action_found_count']}/{result['n_steps']} steps "
        f"(need >= {majority}). Check raw output above and adjust SYSTEM_PROMPT."
    )

    print(f"\nPipeline verified:")
    print(f"  <think> found  : {result['think_found_count']}/{result['n_steps']} steps")
    print(f"  <action> found : {result['action_found_count']}/{result['n_steps']} steps")
    print(f"  Avg inference  : {result['avg_inf_ms']} ms/step on T4")
    print("\nMilestone 3 complete. Ready for Milestone 4.")
