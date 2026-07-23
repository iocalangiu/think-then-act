"""
record_subgoal_demo.py

Records ONE full high-level rollout for a demo: at each decision point the
trained high-level VLM (policy.subgoal_vlm_policy.SubgoalVLMPolicy) looks at
the current frame + state, picks a subgoal, and a trained low-level skill
(training/fetch_skills.py) executes it — the same loop hrl.skill_env.
SkillEnv implements, but driven manually here so every frame AND the VLM's
think/action text at each decision boundary can be captured together, for a
synced video+transcript demo (see memory: hierarchical_architecture.md).

ONLY covers the subgoals with a trained low-level policy so far
(align_xy/descend/close_gripper, per memory 2026-07-16) — lift/
move_to_target/release have no skill to execute yet, so if the VLM picks one
of those the rollout stops there rather than pretending to run it.

Saves (both on the model volume, under /model-cache/demo/), once per seed:
    subgoal_demo_{seed}.mp4         — the full rollout, every base-env frame
    subgoal_demo_{seed}_transcript.json
        — one entry per VLM decision: {call_index, frame_index, think,
          subgoal, raw_response, skill_success, stop_reason}. frame_index
          indexes into the saved mp4 (at the `fps` recorded alongside it) so
          a demo page can sync a chat-style transcript to video playback time.

Accepts multiple seeds in ONE call (loads the VLM + low-level checkpoints
once, loops the rollout per seed) rather than one `modal run` per seed —
the low-level PPO policies run deterministic, so the VLM's own do_sample
sampling is the only source of run-to-run variance; a single seed's demo
is one sampled trajectory, not a characterization of that seed (see memory,
2026-07-20). Getting real evidence means many seeds, and paying a fresh
GPU container + ~4GB model load per seed would make that expensive — this
amortizes that cost across the whole batch. Also writes one
subgoal_demo_batch_summary.json indexing every seed's outcome, so you can
triage which seeds are worth opening in detail instead of reading every
transcript by hand.

Run with:
    modal run --detach scripts/record_subgoal_demo.py --seeds 0,1,2,3,4,5,6,7,8,9
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


def _load_actor(ckpt_path: str, obs_dim: int):
    import torch
    from think_then_act.policy.subgoal_policy import SubgoalGaussianPolicy
    actor = SubgoalGaussianPolicy(obs_dim=obs_dim)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # PPO checkpoints wrap {"actor":..., "critic":...}; GRPO checkpoints are
    # a bare state_dict — same distinction record_subgoal_video.py handles.
    actor.load_state_dict(ckpt["actor"] if isinstance(ckpt, dict) and "actor" in ckpt else ckpt)
    actor.eval()
    return actor


def _run_skill_recording(skill, base_env, obs, frames: list) -> tuple:
    """
    Mirrors hrl.skill_env.SkillEnv.step()'s internal loop for ONE skill (same
    as generate_subgoal_sft_data.py's _run_skill), but also appends every
    intermediate frame to `frames` so the saved video shows continuous
    motion, not just before/after snapshots, and returns the last `info` so
    the caller can check task success.
    """
    info = {}
    for _ in range(skill.max_steps):
        obs_vec = skill.build_obs(obs, base_env)
        action = skill.policy.act(obs_vec, deterministic=True)
        obs, _, terminated, truncated, info = base_env.step(action)
        frames.append(base_env.last_frame())
        _, done = skill.reward_and_done(obs, base_env)
        if done or terminated or truncated:
            return obs, terminated, truncated, done, info
    return obs, False, False, False, info


def run_demo_rollout(vlm_policy, skills: dict, base_env, seed: int, max_skill_calls: int) -> tuple:
    import numpy as np
    from think_then_act.env.setup import init_random_episode

    rng = np.random.default_rng(seed)
    base_env.reset()
    obs, ok = init_random_episode(base_env, rng)
    if not ok:
        raise RuntimeError(f"init_random_episode failed for seed={seed}")

    frames    = [base_env.last_frame()]
    transcript = []

    for call_index in range(max_skill_calls):
        state_entry = {
            "observation"  : obs["observation"],
            "achieved_goal": obs["achieved_goal"],
            "desired_goal" : obs["desired_goal"],
        }
        raw_response, subgoal, think, think_found, action_found = vlm_policy.act(
            frames[-1], state_entry
        )

        entry = {
            "call_index" : call_index,
            "frame_index": len(frames) - 1,   # the frame the VLM actually looked at
            "think"      : think,
            "subgoal"    : subgoal,
            "raw_response": raw_response,
        }

        if subgoal is None:
            entry["stop_reason"] = "unparseable_or_unknown_action"
            transcript.append(entry)
            break
        if subgoal not in skills:
            entry["stop_reason"] = f"chose {subgoal!r}, which has no trained low-level policy yet"
            transcript.append(entry)
            break

        transcript.append(entry)

        obs, terminated, truncated, skill_success, info = _run_skill_recording(
            skills[subgoal], base_env, obs, frames
        )
        entry["skill_success"] = skill_success

        if info.get("is_success"):
            entry["stop_reason"] = "task_success"
            break
        if terminated or truncated:
            entry["stop_reason"] = "env_terminated_or_truncated"
            break

    return frames, transcript


# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------

def _record_one_seed(seed, vlm_policy, skills, max_skill_calls, max_steps_per_skill,
                      fps, out_dir, gym, ObservationHarness, setup_env, save_video, Image):
    import os, json

    base_env = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                  max_episode_steps=max_skill_calls * max_steps_per_skill + 50)
    )
    setup_env(base_env)

    frames, transcript = run_demo_rollout(vlm_policy, skills, base_env, seed, max_skill_calls)
    base_env.close()

    frames_dir = os.path.join(out_dir, f"subgoal_demo_{seed}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    video_path       = os.path.join(out_dir, f"subgoal_demo_{seed}.mp4")
    transcript_path  = os.path.join(out_dir, f"subgoal_demo_{seed}_transcript.json")
    readable_path    = os.path.join(out_dir, f"subgoal_demo_{seed}_transcript.txt")

    save_video(frames, video_path, fps=fps)

    # One PNG per DECISION (not per frame) — scrubbing frame_index inside a
    # continuous mp4 to find which frame a given <think>/<action> corresponds
    # to is slow and error-prone; a standalone image per decision, paired
    # with plain-text reasoning right next to it, is much faster to eyeball
    # for verification.
    frame_paths = []
    for entry in transcript:
        frame_path = os.path.join(
            frames_dir, f"decision_{entry['call_index']:02d}_{entry['subgoal'] or 'none'}.png"
        )
        Image.fromarray(frames[entry["frame_index"]]).save(frame_path)
        frame_paths.append(frame_path)
        entry["frame_path"] = frame_path

    with open(transcript_path, "w") as f:
        json.dump({"fps": fps, "seed": seed, "transcript": transcript}, f, indent=2)

    with open(readable_path, "w") as f:
        for entry in transcript:
            f.write(f"=== call {entry['call_index']}  frame={os.path.basename(entry['frame_path'])} ===\n")
            f.write(f"think : {entry['think']}\n")
            f.write(f"action: {entry['subgoal']}\n")
            if entry.get("skill_success") is not None:
                f.write(f"skill_success: {entry['skill_success']}\n")
            if entry.get("stop_reason"):
                f.write(f"stop_reason: {entry['stop_reason']}\n")
            f.write("\n")

    print(f"\n  seed={seed}: {len(frames)} frames  {len(transcript)} VLM decisions")
    for entry in transcript:
        print(f"    call={entry['call_index']}  subgoal={entry['subgoal']!r}"
              f"  skill_success={entry.get('skill_success')}"
              f"  stop_reason={entry.get('stop_reason')}")
    print(f"  Saved -> {video_path}")
    print(f"  Saved -> {transcript_path}")
    print(f"  Saved -> {readable_path}")
    print(f"  Saved -> {frames_dir}/ ({len(frame_paths)} decision frames)")

    return {
        "n_frames"       : len(frames),
        "n_decisions"    : len(transcript),
        "video_path"     : video_path,
        "transcript_path": transcript_path,
        "readable_path"  : readable_path,
        "frames_dir"     : frames_dir,
        "transcript"     : transcript,
    }


@app.function(
    image=rl_image,
    gpu="A10G",
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=900,
)
def record_subgoal_demo(
    seeds: str = "0",
    max_skill_calls: int = 10,
    max_steps_per_skill: int = 30,
    fps: int = 10,
    algo: str = "ppo",
    use_best: bool = False,
) -> dict:
    import os, json
    import torch

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    from PIL import Image

    from think_then_act.env.setup import setup_env, save_video
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.perception.collision_predictor import CollisionPredictor
    from think_then_act.policy.subgoal_vlm_policy import SubgoalVLMPolicy
    from think_then_act.training.checkpoints import resolve_subgoal_checkpoint
    from think_then_act.training.fetch_skills import build_fetch_skills
    from think_then_act.training.subgoal_features import SUBGOAL_OBS_DIM

    seed_list = [int(s) for s in seeds.split(",") if s.strip() != ""]

    print("\n" + "=" * 60)
    print(f"  SUBGOAL DEMO RECORDING  seeds={seed_list}")
    print("=" * 60)

    ckpt_dir = os.path.join(MODEL_CACHE_DIR, "checkpoints")

    collision_model = None
    collision_ckpt = os.path.join(ckpt_dir, "collision_predictor.pt")
    if os.path.exists(collision_ckpt):
        collision_model = CollisionPredictor()
        collision_model.load_state_dict(torch.load(collision_ckpt, map_location="cpu"))
        collision_model.eval()
        print(f"  collision model   <- {collision_ckpt}")

    # Only the subgoals with a trained low-level policy so far (per
    # hierarchical_architecture memory, 2026-07-16) — lift/move_to_target/
    # release have no skill to execute yet, so the rollout stops if the VLM
    # ever picks one of those (see run_demo_rollout's "not in skills" check).
    trained_subgoals = ("align_xy", "descend", "close_gripper")
    policies = {}
    for subgoal in trained_subgoals:
        ckpt = resolve_subgoal_checkpoint(ckpt_dir, subgoal, algo=algo, use_best=use_best)
        policies[subgoal] = _load_actor(ckpt, obs_dim=SUBGOAL_OBS_DIM)
        print(f"  {subgoal:14s}    <- {ckpt}")

    skills = build_fetch_skills(policies, collision_model, max_steps=max_steps_per_skill)

    print(f"\n  Loading high-level VLM (checkpoints/subgoal_sft_warmstart)...")
    vlm_policy = SubgoalVLMPolicy(
        cache_dir=MODEL_CACHE_DIR,
        lora_path=os.path.join(ckpt_dir, "subgoal_sft_warmstart"),
        device="cuda",
    )

    out_dir = os.path.join(MODEL_CACHE_DIR, "demo")
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for seed in seed_list:
        results[seed] = _record_one_seed(
            seed, vlm_policy, skills, max_skill_calls, max_steps_per_skill,
            fps, out_dir, gym, ObservationHarness, setup_env, save_video, Image,
        )

    # Index across the whole batch — which seeds are worth opening in
    # detail, without reading every transcript by hand (per diagnostic-rigor
    # memory: pull raw examples, but triage first via grounded telemetry).
    summary = {
        "seeds": seed_list,
        "algo": algo,
        "per_seed": {
            str(seed): {
                "n_decisions": r["n_decisions"],
                "subgoal_sequence": [e["subgoal"] for e in r["transcript"]],
                "skill_success_sequence": [e.get("skill_success") for e in r["transcript"]],
                "stop_reason": r["transcript"][-1].get("stop_reason") if r["transcript"] else None,
            }
            for seed, r in results.items()
        },
    }
    summary_path = os.path.join(out_dir, "subgoal_demo_batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    model_volume.commit()

    print("\n" + "=" * 60)
    print("  BATCH SUMMARY")
    for seed, entry in summary["per_seed"].items():
        print(f"    seed={seed}  subgoals={entry['subgoal_sequence']}  "
              f"success={entry['skill_success_sequence']}  stop={entry['stop_reason']}")
    print(f"\n  Saved -> {summary_path}")
    print("=" * 60)

    return {"summary_path": summary_path, "summary": summary,
            "per_seed": {seed: {k: v for k, v in r.items() if k != "transcript"}
                         for seed, r in results.items()}}


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(seeds: str = "0", max_skill_calls: int = 10):
    # .spawn(), not .remote() -- see eval_subgoal_vlm.py's local entrypoint
    # comment: .remote() blocks on the CLI's connection, so under
    # `modal run --detach` the call gets cancelled the moment the CLI exits
    # after dispatch. .spawn() is fire-and-forget and survives that.
    handle = record_subgoal_demo.spawn(seeds=seeds, max_skill_calls=max_skill_calls)
    print(f"\nJob spawned. Function call ID: {handle.object_id}")
    print(f"Monitor at https://modal.com")
    print(f"\nDownload when finished (batch summary first, to triage which seeds to look at):")
    print(f"  modal volume get rl-harness-model-cache demo/subgoal_demo_batch_summary.json ./artifacts/")
    for seed in [s for s in seeds.split(",") if s.strip() != ""]:
        print(f"  modal volume get rl-harness-model-cache demo/subgoal_demo_{seed}.mp4 ./artifacts/")
        print(f"  modal volume get rl-harness-model-cache demo/subgoal_demo_{seed}_transcript.json ./artifacts/")
        print(f"  modal volume get rl-harness-model-cache demo/subgoal_demo_{seed}_transcript.txt ./artifacts/")
        print(f"  modal volume get rl-harness-model-cache demo/subgoal_demo_{seed}_frames/ ./artifacts/subgoal_demo_{seed}_frames/")
