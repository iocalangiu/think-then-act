"""
record_subgoal_demo.py

Records ONE full high-level rollout for a demo: at each decision point the
trained high-level VLM (policy.subgoal_vlm_policy.SubgoalVLMPolicy) looks at
the current frame + state, picks a subgoal, and a trained low-level skill
(training/fetch_skills.py) executes it — the same loop hrl.skill_env.
SkillEnv implements, but driven manually here so every frame AND the VLM's
think/action text at each decision boundary can be captured together, for a
synced video+transcript demo (see memory: hierarchical_architecture.md).

Covers whichever subgoals are passed via `subgoals` (default
align_xy/descend/close_gripper, per memory 2026-07-16) — if the VLM picks
one not in that set, the rollout stops there rather than pretending to run
it. Pass `use_perception=False` to drive skills off ground-truth
achieved_goal instead of the pose/collision models (isolates the demo from
the separate, still-open pose-model-noise question — see bugs_and_fixes
memory). Pass `scripted_close_lift_tail=True` to append a deterministic,
non-VLM close+lift proof (env/oracle.py's own GRASP->CARRY logic) right
after the VLM hands off to a subgoal NOT in `subgoals` (e.g. `subgoals=
"align_xy"` — the VLM picks close_gripper next, which isn't loaded, and
that handoff triggers the scripted proof) — instead of relying on
close_gripper's own trained policy.

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


def _run_scripted_close_lift(base_env, obs, frames: list, max_steps: int) -> tuple:
    """
    Deterministic, non-VLM, non-RL proof that the preceding skill call (e.g.
    align_xy) actually left the gripper positioned to grasp — NOT a demo of
    close_gripper's own trained policy (which has known unresolved issues
    at small finger widths, see bugs_and_fixes memory) and not a new
    scripted sequence invented for this script. Reuses env/oracle.py's own
    GRASP branch verbatim (hold position + close fingers, then — once
    fingers are actually closed — lift) — the same proven heuristic this
    project already trusts for SFT data generation and per-subgoal episode
    setup — rather than hand-rolling a second close+lift sequence that
    could quietly drift out of sync with it.

    Returns (obs, summary_entry). summary_entry["lift_start_frame_index"]
    marks the frame where the scripted motion switches from closing to
    lifting — oracle_action's `phase` string alone can't tell these apart
    (both are "GRASP"; it only flips to "CARRY" once the block has already
    risen partway through the lift), so this is inferred from the action's
    own dz component instead, for callers (e.g. a demo UI's label track)
    that want to highlight "close_gripper" vs "lift" as separate segments.
    """
    from think_then_act.env.oracle import oracle_action

    block_z_start = float(obs["achieved_goal"][2])
    grip_z_start  = float(obs["observation"][2])
    carrying = False
    phase = None
    lift_start_frame_index = None
    for _ in range(max_steps):
        action, phase, carrying = oracle_action(
            obs["observation"], obs["achieved_goal"], obs["desired_goal"], carrying=carrying,
        )
        # GRASP's "close" sub-branch commands zero xyz motion; its "lift"
        # sub-branch commands full +dz — a clean signal to split on, well
        # before `phase` itself would ever say so.
        is_lift_step = float(action[2]) > 0.3
        obs, _, terminated, truncated, info = base_env.step(action)
        frames.append(base_env.last_frame())
        if is_lift_step and lift_start_frame_index is None:
            lift_start_frame_index = len(frames) - 1
        if terminated or truncated or phase == "CARRY":
            break

    block_rise = float(obs["achieved_goal"][2]) - block_z_start
    grip_rise  = float(obs["observation"][2]) - grip_z_start
    entry = {
        "subgoal"       : "scripted_close_lift",
        "scripted"      : True,
        "think"         : (
            "(scripted, not the VLM or a trained low-level policy) closed the "
            "fingers and lifted using env/oracle.py's own GRASP->CARRY logic, "
            "to verify the preceding skill call actually left the gripper "
            "positioned to grasp."
        ),
        "phase_reached"         : phase,
        "lift_start_frame_index": lift_start_frame_index,
        "block_rise_m"          : round(block_rise, 5),
        "grip_rise_m"           : round(grip_rise, 5),
        # CARRY requires oracle.py's own is_grasped check (block lifted AND
        # close to the gripper) -- a real physical grasp, not just closed
        # fingers near the block. See oracle_action's docstring.
        "grasp_verified": bool(phase == "CARRY" and block_rise > 0.01),
    }
    return obs, entry


def run_demo_rollout(vlm_policy, skills: dict, base_env, seed: int, max_skill_calls: int,
                      scripted_close_lift_tail: bool = False,
                      scripted_close_lift_max_steps: int = 25,
                      init_subgoal: str | None = None, align_xy_policy=None) -> tuple:
    import time
    import numpy as np
    from think_then_act.env.setup import (
        init_random_episode, init_episode_before_subgoal, randomize_gripper_start,
    )

    rng = np.random.default_rng(seed)
    base_env.reset()
    if init_subgoal == "hover_above_block":
        # A genuine "xy-aligned, still elevated" frame -- NOT the same as
        # init_subgoal="descend" below, which runs align_xy's OWN trained
        # policy to ITS done condition and (per its reward's z-penalty
        # side effect) ends up already near grasp height, so it can never
        # actually produce this state. Here the gripper is driven laterally
        # (bounded action steps, dz held at 0 throughout -- the only proven
        # way to relocate it, see randomize_gripper_start's docstring)
        # directly above the block's xy, leaving z untouched at its fresh-
        # reset height. Lets a demo show what the VLM says when descend is
        # actually the right call, not structurally pre-empted by align_xy.
        obs, ok = init_random_episode(base_env, rng)
        if ok:
            obs, ok, _info = randomize_gripper_start(
                base_env, rng, obs, target_xy=obs["achieved_goal"][:2], xy_bias_strength=1.0,
            )
    elif init_subgoal:
        # Starts the episode already in the canonical state that subgoal's
        # OWN low-level training/eval uses (env/setup.py's own setup logic,
        # not a second implementation here) -- e.g. init_subgoal="descend"
        # runs align_xy_policy (scripted, not the VLM) to align_xy's own
        # done condition, so the demo starts with the gripper already
        # above the block, to see what the VLM says from THAT state rather
        # than always starting from a fresh reset.
        obs, ok = init_episode_before_subgoal(
            base_env, rng, subgoal=init_subgoal, align_xy_policy=align_xy_policy
        )
    else:
        obs, ok = init_random_episode(base_env, rng)
    if not ok:
        raise RuntimeError(f"init_{init_subgoal or 'random'}_episode failed for seed={seed}")

    frames    = [base_env.last_frame()]
    transcript = []

    for call_index in range(max_skill_calls):
        state_entry = {
            "observation"  : obs["observation"],
            "achieved_goal": obs["achieved_goal"],
            "desired_goal" : obs["desired_goal"],
        }
        # Wall-clock time for this ONE VLM decision (prompt build + Qwen2-VL
        # generate() + parse), on the actual A10G this project trains on —
        # not a synthetic/estimated number. First call in a batch includes
        # any lazy CUDA-kernel warmup, so it isn't representative of steady
        # state; callers/consumers should treat call_index==0 accordingly.
        t0 = time.perf_counter()
        raw_response, subgoal, think, think_found, action_found = vlm_policy.act(
            frames[-1], state_entry
        )
        decision_latency_s = time.perf_counter() - t0

        entry = {
            "call_index" : call_index,
            "frame_index": len(frames) - 1,   # the frame the VLM actually looked at
            "think"      : think,
            "subgoal"    : subgoal,
            "raw_response": raw_response,
            "decision_latency_s": round(decision_latency_s, 3),
        }

        if subgoal is None:
            entry["stop_reason"] = "unparseable_or_unknown_action"
            transcript.append(entry)
            break
        if subgoal not in skills:
            entry["stop_reason"] = f"chose {subgoal!r}, which has no trained low-level policy yet"
            transcript.append(entry)
            # The VLM's decision to move on IS the real "handoff" moment --
            # append it to the transcript/think-trace like any other
            # decision (a demo UI wants to show this reasoning), then prove
            # the PRECEDING skill call actually left the gripper ready,
            # scripted -- but only if that preceding call actually
            # succeeded; a failed skill has nothing real to verify. EXCEPT
            # under init_subgoal="hover_above_block": there IS no preceding
            # skill call when the VLM's very first decision is already
            # "not in skills" -- run the scripted probe anyway so a WRONG
            # first call (e.g. "close_gripper" while still 30cm up) shows
            # its real physical consequence rather than just stopping
            # silently. This is deliberately a "what actually happens"
            # demonstration here, not a success-precondition check.
            precondition_ok = len(transcript) >= 2 and transcript[-2].get("skill_success")
            show_consequence_anyway = len(transcript) == 1 and init_subgoal == "hover_above_block"
            if scripted_close_lift_tail and (precondition_ok or show_consequence_anyway):
                obs, scripted_entry = _run_scripted_close_lift(
                    base_env, obs, frames, scripted_close_lift_max_steps
                )
                scripted_entry["call_index"]  = len(transcript)
                scripted_entry["frame_index"] = len(frames) - 1
                transcript.append(scripted_entry)
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
                      fps, out_dir, gym, ObservationHarness, setup_env, save_video, Image,
                      scripted_close_lift_tail=False, scripted_close_lift_max_steps=25,
                      init_subgoal=None, align_xy_policy=None):
    import os, json

    base_env = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                  max_episode_steps=max_skill_calls * max_steps_per_skill + scripted_close_lift_max_steps + 50)
    )
    setup_env(base_env)

    frames, transcript = run_demo_rollout(
        vlm_policy, skills, base_env, seed, max_skill_calls,
        scripted_close_lift_tail=scripted_close_lift_tail,
        scripted_close_lift_max_steps=scripted_close_lift_max_steps,
        init_subgoal=init_subgoal, align_xy_policy=align_xy_policy,
    )
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
    subgoals: str = "align_xy,descend,close_gripper",
    use_perception: bool = True,
    scripted_close_lift_tail: bool = False,
    scripted_close_lift_max_steps: int = 25,
    init_subgoal: str = "",
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
    from think_then_act.perception.block_pose_predictor import BlockPosePredictor
    from think_then_act.perception.collision_predictor import CollisionPredictor
    from think_then_act.policy.subgoal_vlm_policy import SubgoalVLMPolicy
    from think_then_act.training.checkpoints import resolve_subgoal_checkpoint
    from think_then_act.training.fetch_skills import build_fetch_skills
    from think_then_act.training.subgoal_features import obs_dim_for_subgoal

    seed_list = [int(s) for s in seeds.split(",") if s.strip() != ""]

    print("\n" + "=" * 60)
    print(f"  SUBGOAL DEMO RECORDING  seeds={seed_list}")
    print("=" * 60)

    ckpt_dir = os.path.join(MODEL_CACHE_DIR, "checkpoints")

    # use_perception=False drives the low-level skills off ground-truth
    # achieved_goal (no pose/collision model) -- matches the exact config
    # eval_align_descend_convergence.py used to confirm align_xy=90%/
    # descend=100% standalone completion, isolating the demo from the
    # separate, still-open pose-model-noise question (see hierarchical_
    # architecture / bugs_and_fixes memory: an align_xy precision retrain
    # aimed at that had an unconfirmed final outcome).
    collision_model = None
    pose_model = None
    if use_perception:
        collision_ckpt = os.path.join(ckpt_dir, "collision_predictor.pt")
        if os.path.exists(collision_ckpt):
            collision_model = CollisionPredictor()
            collision_model.load_state_dict(torch.load(collision_ckpt, map_location="cpu"))
            collision_model.eval()
            print(f"  collision model   <- {collision_ckpt}")

        pose_ckpt = os.path.join(ckpt_dir, "block_pose_predictor.pt")
        if os.path.exists(pose_ckpt):
            pose_model = BlockPosePredictor()
            pose_model.load_state_dict(torch.load(pose_ckpt, map_location="cpu"))
            pose_model.eval()
            print(f"  pose model        <- {pose_ckpt}")
    else:
        print(f"  use_perception=False -- skills driven by ground-truth achieved_goal, "
              f"no pose/collision model loaded")

    # Which subgoals the VLM is offered / has a trained low-level policy
    # loaded for -- caller-controlled so a demo can be deliberately narrowed
    # (e.g. "align_xy,descend" to exclude close_gripper's own known small-
    # width issues and instead prove descend worked via a scripted tail;
    # see run_demo_rollout's scripted_close_lift_tail). If the VLM picks
    # a subgoal NOT in this set, the rollout stops there rather than
    # pretending to run it (see run_demo_rollout's "not in skills" check).
    trained_subgoals = tuple(s.strip() for s in subgoals.split(",") if s.strip())
    policies = {}
    for subgoal in trained_subgoals:
        ckpt = resolve_subgoal_checkpoint(ckpt_dir, subgoal, algo=algo, use_best=use_best)
        policies[subgoal] = _load_actor(ckpt, obs_dim=obs_dim_for_subgoal(subgoal))
        print(f"  {subgoal:14s}    <- {ckpt}")

    skills = build_fetch_skills(policies, collision_model, pose_model, max_steps=max_steps_per_skill)

    print(f"\n  Loading high-level VLM (checkpoints/subgoal_sft_warmstart)...")
    vlm_policy = SubgoalVLMPolicy(
        cache_dir=MODEL_CACHE_DIR,
        lora_path=os.path.join(ckpt_dir, "subgoal_sft_warmstart"),
        device="cuda",
    )

    out_dir = os.path.join(MODEL_CACHE_DIR, "demo")
    os.makedirs(out_dir, exist_ok=True)

    init_subgoal = init_subgoal.strip() or None
    # Per init_episode_before_subgoal: "descend" setup runs align_xy's own
    # trained actor (scripted, not the VLM) to align_xy's done condition --
    # every other subgoal's setup uses env/oracle.py's scripted heuristic
    # directly and needs no actor at all.
    if init_subgoal == "descend" and "align_xy" not in policies:
        raise ValueError(
            "init_subgoal='descend' needs align_xy's actor loaded for setup "
            f"(add align_xy to --subgoals) -- currently loaded: {list(policies)}"
        )

    results = {}
    for seed in seed_list:
        results[seed] = _record_one_seed(
            seed, vlm_policy, skills, max_skill_calls, max_steps_per_skill,
            fps, out_dir, gym, ObservationHarness, setup_env, save_video, Image,
            scripted_close_lift_tail=scripted_close_lift_tail,
            scripted_close_lift_max_steps=scripted_close_lift_max_steps,
            init_subgoal=init_subgoal, align_xy_policy=policies.get("align_xy"),
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
def main(seeds: str = "0", max_skill_calls: int = 10, algo: str = "ppo", use_best: bool = False,
         subgoals: str = "align_xy,descend,close_gripper", use_perception: bool = True,
         scripted_close_lift_tail: bool = False, scripted_close_lift_max_steps: int = 25,
         init_subgoal: str = "", max_steps_per_skill: int = 30):
    # .spawn(), not .remote() -- see eval_subgoal_vlm.py's local entrypoint
    # comment: .remote() blocks on the CLI's connection, so under
    # `modal run --detach` the call gets cancelled the moment the CLI exits
    # after dispatch. .spawn() is fire-and-forget and survives that.
    handle = record_subgoal_demo.spawn(
        seeds=seeds, max_skill_calls=max_skill_calls, algo=algo, use_best=use_best,
        subgoals=subgoals, use_perception=use_perception,
        scripted_close_lift_tail=scripted_close_lift_tail,
        scripted_close_lift_max_steps=scripted_close_lift_max_steps,
        init_subgoal=init_subgoal, max_steps_per_skill=max_steps_per_skill,
    )
    print(f"\nJob spawned. Function call ID: {handle.object_id}")
    print(f"Monitor at https://modal.com")
    print(f"\nDownload when finished (batch summary first, to triage which seeds to look at):")
    print(f"  modal volume get rl-harness-model-cache demo/subgoal_demo_batch_summary.json ./artifacts/")
    for seed in [s for s in seeds.split(",") if s.strip() != ""]:
        print(f"  modal volume get rl-harness-model-cache demo/subgoal_demo_{seed}.mp4 ./artifacts/")
        print(f"  modal volume get rl-harness-model-cache demo/subgoal_demo_{seed}_transcript.json ./artifacts/")
        print(f"  modal volume get rl-harness-model-cache demo/subgoal_demo_{seed}_transcript.txt ./artifacts/")
        print(f"  modal volume get rl-harness-model-cache demo/subgoal_demo_{seed}_frames/ ./artifacts/subgoal_demo_{seed}_frames/")
