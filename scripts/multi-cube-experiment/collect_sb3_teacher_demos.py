"""
collect_sb3_teacher_demos.py

Rolls out a PRETRAINED, externally-sourced policy (default:
sb3/tqc-FetchPickAndPlace-v1, a TQC+HER agent from the Hugging Face RL
Baselines3 Zoo) against this project's OWN FetchPickAndPlace-v3 env, and
records successful full pick-and-place episodes as (flat_obs, action) pairs
in this project's own observation format (training.subgoal_features.
build_flat_observation) — not the checkpoint's original dict-obs format, and
not whatever env version it was originally trained on. See
meta_rl_sim2real_direction / hierarchical_architecture memory: the point is
only "what would this policy do, given what OUR env is showing it" — the
algorithm/version that produced the checkpoint is irrelevant to a BC
student, only the resulting (obs, action) trajectories matter.

Deliberately a SEPARATE Modal image, not rl_image (think_then_act.modal_app)
— stable-baselines3/sb3-contrib/huggingface_sb3 pull their own torch and
gymnasium requirements, and this project already tried + removed SB3 once
after its version pins repeatedly conflicted with rl_image's VLM-stack pins
(torch==2.3.0 for Qwen2-VL — see bugs_and_fixes memory, 2026-07-11). Keeping
this entirely separate means that conflict can't recur: this image shares
nothing with rl_image except the `app` object and the mujoco/gymnasium-
robotics simulation layer (pinned to the SAME versions as rl_image's, so the
physics/task match exactly what the rest of the project trains/evaluates
against).

apply_setup_env matters and is left as a flag, not hardcoded: setup_env()
(env/setup.py) shifts the robot base to [0.85, 0.75, 0.0], a change the
external checkpoint was never trained under. Its ABSOLUTE-position features
(grip_pos/object_pos, the first part of the raw 25-dim obs) could plausibly
go out of distribution under that shift even though build_flat_observation
itself only keeps RELATIVE geometry (frame-invariant by construction — see
that function's docstring) — the policy's own DECISIONS, not the recorded
features, are what might degrade. Run with --n-episodes small first and
read the printed success rate before committing to a big collection run;
if it's poor with apply_setup_env=True, rerun with --no-apply-setup-env to
check whether the shift specifically is the cause.

Run with:
    modal run scripts/multi-cube-experiment/collect_sb3_teacher_demos.py --n-episodes 20
    modal run scripts/multi-cube-experiment/collect_sb3_teacher_demos.py --n-episodes 300 --max-steps 100
    modal run scripts/multi-cube-experiment/collect_sb3_teacher_demos.py --no-apply-setup-env --n-episodes 20
    modal run scripts/multi-cube-experiment/collect_sb3_teacher_demos.py --target-successes 100 --n-episodes 400
                                                              # keep attempting (seed
                                                              # 0,1,2,...) until 100
                                                              # SUCCESSFUL demos are
                                                              # collected or 400 total
                                                              # attempts are exhausted
                                                              # (~55% success_rate observed
                                                              # so far -> expect ~180-200
                                                              # attempts needed for 100)

Download the result with:
    python3 -m modal volume get rl-harness-model-cache demonstrations/sb3_teacher_full_task.pkl ./artifacts/

Download the videos (saved for the first --n-videos episodes, success or
not — pass --n-videos 0 to skip rendering entirely and run faster) with:
    python3 -m modal volume get rl-harness-model-cache demonstrations/videos/ ./artifacts/videos/
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
    # Same simulation-layer pins as rl_image (think_then_act.modal_app) —
    # same physics/task version as the rest of the project. Deliberately NOT
    # importing rl_image itself: this image's torch/gymnasium resolution is
    # left to stable-baselines3's own requirements, not rl_image's VLM-stack
    # pins (torch==2.3.0) — that pin conflict is exactly why SB3 was removed
    # from rl_image before. See module docstring.
    .pip_install(
        "mujoco==3.1.6",
        "gymnasium==1.0.0",
        "gymnasium-robotics==1.3.1",
        "numpy==1.26.4",
        "imageio==2.34.1",
        # imageio's core package has no actual mp4 codec of its own — the
        # FFMPEG backend is a separate plugin package, confirmed from the
        # actual runtime error ("Could not find a backend... FFMPEG: pip
        # install imageio[ffmpeg]", 2026-10-02). apt's `ffmpeg` CLI binary
        # (installed above) is not sufficient by itself.
        "imageio-ffmpeg",
    )
    # CPU-only torch, pinned, from PyTorch's own CPU wheel index — installed
    # BEFORE stable-baselines3 so its own "torch>=1.13" dependency is
    # already satisfied and pip never reaches for a default (CUDA-bundled)
    # PyPI wheel instead. Root-caused 2026-10-02 via a standalone diagnostic
    # (diagnose_torch_mujoco_segfault.py): the unpinned install resolved
    # torch==2.14.1+cu130 (a CUDA 13 build) into this gpu=None container
    # with no NVIDIA driver at all — even core SB3's plain SAC construction
    # (nothing TQC/checkpoint-specific) segfaulted on it immediately.
    .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "stable-baselines3==2.4.0",   # first version with Gymnasium v1.0 support
        "sb3-contrib==2.4.0",         # TQC lives here, not in core stable-baselines3
        "huggingface_sb3==3.0",
        # This checkpoint predates the gym->gymnasium rename: its data.pkl
        # has at least one cloudpickled object (observation_space/
        # action_space, not just the schedule callables — custom_objects=
        # {"learning_rate":..., "lr_schedule":..., "clip_range":...} alone
        # did NOT fix this, confirmed 2026-10-01) whose class reference
        # resolves through the legacy `gym` package. We never call into
        # gym's actual functionality — this is here purely so
        # cloudpickle.loads() can resolve the module path during TQC.load().
        "gym",
        # SB3's own load() path then auto-converts those deserialized
        # gym.spaces objects to gymnasium.spaces via shimmy (base_class.py's
        # _convert_space) — confirmed by the actual runtime error asking for
        # exactly this, 2026-10-01: "Missing shimmy installation. You
        # provided an OpenAI Gym space."
        "shimmy>=0.2.1",
    )
    .add_local_python_source("think_then_act", copy=True)
)


@app.function(image=sb3_image, gpu=None, cpu=4.0, memory=8192,
              volumes={MODEL_CACHE_DIR: model_volume}, timeout=3600 * 6)   # bumped for
                                  # large target_successes collection runs (e.g. 1000
                                  # genuine demos at ~35% genuine rate needs ~2850
                                  # attempts) — 286 attempts took well under 1hr, but
                                  # ~10x that risked the old 3600s cap
def collect_sb3_teacher_demos(
    repo_id: str = "sb3/tqc-FetchPickAndPlace-v1",
    filename: str = "tqc-FetchPickAndPlace-v1.zip",
    n_episodes: int = 20,     # safety cap on total ATTEMPTS, always enforced —
                               # also the exact episode count when target_successes=0
    target_successes: int = 0,   # 0 = old behavior (run exactly n_episodes, keep
                                  # whichever succeed). >0 = keep attempting
                                  # (incrementing seed) until this many SUCCESSFUL
                                  # demonstrations are collected or n_episodes
                                  # attempts are exhausted, whichever first — the
                                  # quantity that actually matters for BC is
                                  # successful demos, not raw attempts, and at
                                  # ~55% success_rate guessing a raw count to hit a
                                  # target is just extra arithmetic for the caller.
    max_steps: int = 100,
    apply_setup_env: bool = True,
    seed_offset: int = 0,
    out_path: str = "demonstrations/sb3_teacher_full_task.pkl",
    n_videos: int = 2,   # save this many of the FIRST episodes as mp4s,
                          # regardless of success/failure — for visually
                          # inspecting what the policy actually does, not
                          # just the aggregate success_rate number.
    video_dir: str = "demonstrations/videos",
    require_genuine_grasp: bool = True,   # is_success alone is NOT sufficient —
                                  # it only checks final block-to-goal distance,
                                  # with no requirement the block was ever
                                  # lifted/held. Root-caused 2026-10-02
                                  # (diagnose_teacher_success_quality.py): only
                                  # 45% of this teacher's own "successes" were
                                  # genuine grasp-carries, the rest were
                                  # slide/nudge artifacts — BC trained on the
                                  # unfiltered set learned to slide, not grasp.
                                  # True (default): also require >=1 step with
                                  # both fingers in contact AND the block lifted
                                  # > lift_threshold above its resting height.
    lift_threshold: float = 0.02,
    block_resting_z: float = 0.425,
    elevate_target_prob: float = 0.5,   # fraction of episodes whose target
                                  # gets raised into the air instead of
                                  # staying at table height. init_random_episode
                                  # (unchanged) always places it at table
                                  # height — this project's own simplification
                                  # of the native FetchPickAndPlace task, which
                                  # natively samples an elevated target roughly
                                  # half the time. Added 2026-10-02 after
                                  # finding every BC architecture trained on
                                  # table-height-only demos collapses to ~0%
                                  # completion the moment the target is
                                  # elevated — the teacher itself was trained
                                  # on the native (mixed-height) distribution,
                                  # so this isn't asking it to do something new,
                                  # just not throwing away part of what it
                                  # already knows via this project's own
                                  # restricted episode-init.
    elevation_min: float = 0.10,
    elevation_max: float = 0.30,
    randomize_joint_angles_prob: float = 0.0,   # fraction of episodes that start from a
                                  # PERTURBED arm joint configuration instead of the
                                  # one fixed pose every episode in this project has
                                  # always started from (confirmed 2026-10-02: all 4
                                  # of env.setup._ARM_POSE_JOINTS read exactly 0.0 at
                                  # every reset). 0.0 (default) preserves the old
                                  # fixed-start behavior byte-for-byte for every
                                  # EXISTING caller; this collection run itself should
                                  # pass 1.0 (randomize every episode) since the whole
                                  # point is a demo pool with genuine starting-pose
                                  # diversity, not a probabilistic mix of the two.
    joint_angle_exclude_band: float = 0.35,   # see env.setup.randomize_joint_angles's
                                  # own docstring for why these two params (not a bare
                                  # scale) and why 0.35/0.85 specifically.
    joint_angle_max_frac: float = 0.85,
    pose_scheme: str = "joint_angles",   # "joint_angles" (default, randomize_joint_angles --
                                  # byte-for-byte unchanged behavior for every existing
                                  # caller) or "gripper_3d" (env.setup.randomize_gripper_
                                  # start_3d, 2026-10-04 -- drives the gripper via real
                                  # bounded actions toward a randomized 3D point instead of
                                  # sampling joint angles independently; produces visually
                                  # natural, non-"torturous" poses -- see that function's own
                                  # docstring and the "Pose Generalization & Stacking"
                                  # artifact's skeleton-overlay comparison for why)
) -> dict:
    import os
    import pickle
    import numpy as np
    import imageio

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    # The segfault (SIGSEGV, no Python traceback) happening right after
    # model/env setup and before any prediction, reproducibly across
    # retries (confirmed 2026-10-01), matches a known crash class: PyTorch
    # and MuJoCo's native extensions both bundling their own OpenMP
    # runtime in the same process, which can corrupt state silently at the
    # C level instead of raising a catchable Python exception. Must be set
    # before torch (pulled in by stable_baselines3) is imported at all.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # torch imported BEFORE gymnasium/gymnasium_robotics, deliberately —
    # matches the working import order already established elsewhere in
    # this project (train_low_level_ppo.py etc., which also combine torch +
    # mujoco in one process). This script originally imported gymnasium
    # first and pulled torch in later (transitively, via sb3_contrib) —
    # that reversed order segfaulted (SIGSEGV, no Python traceback)
    # reproducibly inside TQC.load(), confirmed across several other ruled-
    # out hypotheses (OpenMP env vars, explicit device="cpu", more memory —
    # none of those fixed it, 2026-10-01). Two native C-extensions (torch's
    # and mujoco's) loaded in the wrong order in one process is a known
    # class of ABI/symbol-version conflict; this project's own working
    # scripts already demonstrate the order that avoids it.
    import torch
    torch.set_num_threads(1)
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    from huggingface_sb3 import load_from_hub
    from sb3_contrib import TQC
    from sb3_contrib.common.wrappers import TimeFeatureWrapper

    from think_then_act.env.setup import (
        setup_env, init_random_episode, grip_contact_forces, randomize_joint_angles, randomize_gripper_start_3d,
    )
    from think_then_act.training.subgoal_features import build_flat_observation

    print("\n" + "=" * 70)
    print("  SB3 TEACHER DEMO COLLECTION (external pretrained policy)")
    print(f"  repo_id={repo_id}  apply_setup_env={apply_setup_env}")
    print("=" * 70)

    # Built BEFORE loading the model: this checkpoint's replay_buffer_class
    # is HerReplayBuffer, which needs a live env at load time (its own
    # compute_reward, for relabeling) — SB3._setup_model() asserts on this
    # even though we only ever call .predict(), never .learn() here.
    env = gym.make("FetchPickAndPlace-v3", max_episode_steps=max_steps,
                    render_mode="rgb_array" if n_videos > 0 else None)
    if apply_setup_env:
        setup_env(env)
    # Required at inference, not just training — this checkpoint's config.yml
    # (sb3/tqc-FetchPickAndPlace-v1) specifies env_wrapper:
    # sb3_contrib.common.wrappers.TimeFeatureWrapper; the policy was trained
    # conditioned on a remaining-time scalar appended to obs["observation"]
    # (25->26 dims) and its decisions depend on it (see that wrapper's own
    # docstring — this isn't a cosmetic preprocessing step). max_steps=100
    # matches the checkpoint's own training config (replay_buffer_kwargs:
    # max_episode_length=100), confirmed via its args.yml on Hugging Face —
    # not guessed. build_flat_observation() only ever reads obs[0:25], so
    # the extra trailing time feature is harmless for OUR recorded data;
    # it only matters for what gets fed into model.predict() below.
    env = TimeFeatureWrapper(env, max_steps=max_steps)

    ckpt_path = load_from_hub(repo_id=repo_id, filename=filename)
    # This checkpoint predates the gym->gymnasium rename; its saved
    # lr_schedule/clip_range callables were cloudpickled referencing the
    # now-uninstalled legacy `gym` package and fail to unpickle even though
    # we never use them (inference only, never resuming training). SB3's
    # documented fix: substitute harmless stand-ins via custom_objects so
    # those specific keys are never actually deserialized from the zip.
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
        # The saved observation/action spaces are IDENTICAL to this env's in
        # shape and bounds — only the dtype declaration differs (float32 at
        # save time vs float64 from this gymnasium-robotics version's native
        # Fetch env, confirmed from the actual mismatch error, 2026-10-01).
        # Overriding with the live env's own spaces here makes
        # check_for_correct_spaces compare an object against itself, so it
        # can't fail on this cosmetic difference.
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        # The saved replay_buffer_kwargs (online_sampling=True,
        # goal_selection_strategy=..., ...) target an older HerReplayBuffer
        # signature that this SB3 version no longer accepts (online sampling
        # is now the only mode — the kwarg was dropped, confirmed from the
        # actual TypeError, 2026-10-01). We never call .learn()/sample from
        # this buffer, only .predict() — forcing it to a plain buffer with
        # no special kwargs sidesteps the version-skew entirely since
        # nothing downstream depends on it being the real HER buffer.
        "replay_buffer_class": None,
        "replay_buffer_kwargs": {},
    }
    print(f"  about to TQC.load (ckpt_path={ckpt_path})", flush=True)
    # device="cpu" explicit, not "auto" — this container has gpu=None, and
    # letting SB3/torch auto-probe for CUDA in a driver-less container is a
    # plausible segfault source (the previous two runs crashed with SIGSEGV
    # and no Python traceback at exactly this call, 2026-10-01).
    model = TQC.load(ckpt_path, env=env, device="cpu", custom_objects=custom_objects)
    print(f"  Loaded {repo_id}/{filename} -> {ckpt_path}", flush=True)

    demonstrations = []
    n_success = 0          # genuine grasp-carries kept as demonstrations
    n_is_success_raw = 0   # is_success==True regardless of genuine — tracked
                            # separately so the summary shows the contamination
                            # rate (slide/nudge vs genuine), not just the final count
    episode_lengths = []
    n_attempted = 0

    ep = 0
    while True:
        if target_successes > 0:
            if len(demonstrations) >= target_successes or ep >= n_episodes:
                break
        else:
            if ep >= n_episodes:
                break
        n_attempted = ep + 1

        record_video = ep < n_videos
        frames = []

        obs, _ = env.reset(seed=seed_offset + ep)
        # init_random_episode teleports the block/goal onto a fixed,
        # table-centered disk (center [1.30, 0.75], r=0.20) and re-settles
        # the gripper fingers — this project's OWN established episode-init
        # path (used by every other script that touches this env), and
        # deliberately NOT the bare env's native reset placement (which
        # samples relative to wherever the gripper happens to rest). Root-
        # caused 2026-10-02 from a recorded video: with apply_setup_env=True
        # the robot base matched the checkpoint's own demo video, but the
        # cube was missing entirely — FetchEnv's native placement was never
        # designed to cope with a relocated base, which is exactly why this
        # project's own code already overrides it everywhere else instead
        # of trusting it.
        rng = np.random.default_rng(seed_offset + ep)
        randomized_pose_this_ep = bool(rng.uniform(0.0, 1.0) < randomize_joint_angles_prob)
        if randomized_pose_this_ep:
            # Before init_random_episode places the block/target, not after —
            # avoids the arm's own settling step (inside the pose-randomize
            # call) interacting with a block that's already sitting where the
            # arm is about to move through.
            if pose_scheme == "gripper_3d":
                obs, pose_ok, _ = randomize_gripper_start_3d(env, rng, obs)
            else:
                obs, pose_ok = randomize_joint_angles(env, rng, exclude_band=joint_angle_exclude_band,
                                                       max_frac=joint_angle_max_frac)
            if not pose_ok:
                print(f"  ep={ep:>4}  {pose_scheme} pose randomization exhausted its retries — skipping", flush=True)
                ep += 1
                continue
        obs, setup_ok = init_random_episode(env, rng)
        if not setup_ok:
            print(f"  ep={ep:>4}  init_random_episode setup itself ended the "
                  f"episode (done/truncated) — skipping", flush=True)
            ep += 1
            continue
        elevated_this_ep = bool(rng.uniform(0.0, 1.0) < elevate_target_prob)
        if elevated_this_ep:
            elevation = float(rng.uniform(elevation_min, elevation_max))
            env.unwrapped.goal[2] += elevation   # .unwrapped resolves through
                                  # TimeFeatureWrapper + whatever gym.make()
                                  # itself auto-wraps with (TimeLimit etc.) to
                                  # the base FetchPickAndPlace env, regardless
                                  # of wrapping depth — same pattern already
                                  # verified working in eval_elevated_target.py.
            obs, _, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
            if terminated or truncated:
                ep += 1
                continue

        if record_video:
            frames.append(env.render())
        flat_obs_list, action_list = [], []
        success = False
        ever_lifted_and_gripped = False

        for _ in range(max_steps):
            flat_obs_list.append(
                build_flat_observation(obs["observation"], obs["achieved_goal"], obs["desired_goal"])
            )
            action, _ = model.predict(obs, deterministic=True)
            action_list.append(np.asarray(action, dtype=np.float32))

            obs, reward, terminated, truncated, info = env.step(action)
            if record_video:
                frames.append(env.render())

            height_above_resting = float(obs["achieved_goal"][2]) - block_resting_z
            forces = grip_contact_forces(env)
            if min(forces["left"], forces["right"]) > 0.0 and height_above_resting > lift_threshold:
                ever_lifted_and_gripped = True

            if info.get("is_success", False):
                success = True
            if terminated or truncated:
                break

        if success:
            n_is_success_raw += 1
        genuine = success and (ever_lifted_and_gripped or not require_genuine_grasp)
        episode_lengths.append(len(flat_obs_list))
        if success and not genuine:
            status = "SUCCESS(slide, rejected)"
        elif success:
            status = "SUCCESS"
        else:
            status = "fail"
        elev_tag = "elevated" if elevated_this_ep else "table   "
        pose_tag = "pose-rand" if randomized_pose_this_ep else "pose-fixed"
        print(f"  ep={ep:>4}  {elev_tag}  {pose_tag}  steps={len(flat_obs_list):>3}  {status}")

        if record_video:
            status_tag = status.split("(")[0].strip().lower()
            video_path = os.path.join(MODEL_CACHE_DIR, video_dir, f"sb3_teacher_ep{ep}_{status_tag}.mp4")
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            imageio.mimsave(video_path, frames, fps=20)
            print(f"    saved video -> {video_path}")

        if genuine:
            n_success += 1
            demonstrations.append({
                "obs": np.stack(flat_obs_list).astype(np.float32),
                "teacher_action": np.stack(action_list).astype(np.float32),
                "success": True,
                "n_steps": len(flat_obs_list),
                "elevated_target": elevated_this_ep,
                "randomized_pose": randomized_pose_this_ep,
            })

        ep += 1

    env.close()

    if target_successes > 0 and len(demonstrations) < target_successes:
        print(f"\n  WARNING: hit the n_episodes={n_episodes} attempt cap with only "
              f"{len(demonstrations)}/{target_successes} successes collected — "
              f"raise --n-episodes to actually reach the target.")

    success_rate = n_success / n_attempted if n_attempted else 0.0
    raw_success_rate = n_is_success_raw / n_attempted if n_attempted else 0.0
    print("\n" + "=" * 70)
    print(f"  genuine grasp-carry rate: {n_success}/{n_attempted} attempts ({success_rate:.1%})")
    if require_genuine_grasp:
        print(f"  (is_success alone fired {n_is_success_raw}/{n_attempted} = {raw_success_rate:.1%} — "
              f"{n_is_success_raw - n_success} of those were slide/nudge, rejected as demos)")
    print(f"  mean episode length (all episodes): {np.mean(episode_lengths):.1f}")
    n_elevated_kept = sum(1 for d in demonstrations if d["elevated_target"])
    print(f"  kept demos: {n_elevated_kept} elevated-target, "
          f"{len(demonstrations) - n_elevated_kept} table-height (target elevate_target_prob={elevate_target_prob})")
    n_pose_randomized_kept = sum(1 for d in demonstrations if d["randomized_pose"])
    print(f"  kept demos: {n_pose_randomized_kept} randomized-start-pose, "
          f"{len(demonstrations) - n_pose_randomized_kept} fixed-start-pose "
          f"(target randomize_joint_angles_prob={randomize_joint_angles_prob})")
    print("=" * 70)

    full_out_path = os.path.join(MODEL_CACHE_DIR, out_path)
    os.makedirs(os.path.dirname(full_out_path), exist_ok=True)
    with open(full_out_path, "wb") as f:
        pickle.dump({
            "demonstrations": demonstrations,
            "repo_id": repo_id, "filename": filename,
            "apply_setup_env": apply_setup_env,
            "require_genuine_grasp": require_genuine_grasp,
            "elevate_target_prob": elevate_target_prob,
            "elevation_min": elevation_min, "elevation_max": elevation_max,
            "randomize_joint_angles_prob": randomize_joint_angles_prob,
            "joint_angle_exclude_band": joint_angle_exclude_band,
            "joint_angle_max_frac": joint_angle_max_frac,
            "pose_scheme": pose_scheme,
            "n_attempted": n_attempted, "n_successful": n_success,
            "n_is_success_raw": n_is_success_raw,
            "success_rate": success_rate,
        }, f)
    model_volume.commit()
    print(f"  Saved {n_success} genuine demonstrations -> {full_out_path}")

    return {"n_attempted": n_attempted, "n_successful": n_success, "success_rate": success_rate}


@app.local_entrypoint()
def main(
    repo_id: str = "sb3/tqc-FetchPickAndPlace-v1",
    filename: str = "tqc-FetchPickAndPlace-v1.zip",
    n_episodes: int = 20,
    target_successes: int = 0,
    max_steps: int = 100,
    apply_setup_env: bool = True,
    n_videos: int = 2,
    elevate_target_prob: float = 0.5,
    randomize_joint_angles_prob: float = 0.0,
    joint_angle_exclude_band: float = 0.35,
    joint_angle_max_frac: float = 0.85,
    pose_scheme: str = "joint_angles",
    out_path: str = "demonstrations/sb3_teacher_full_task.pkl",
    video_dir: str = "demonstrations/videos",
):
    result = collect_sb3_teacher_demos.remote(
        repo_id=repo_id, filename=filename, n_episodes=n_episodes,
        target_successes=target_successes, max_steps=max_steps,
        apply_setup_env=apply_setup_env, n_videos=n_videos,
        elevate_target_prob=elevate_target_prob,
        randomize_joint_angles_prob=randomize_joint_angles_prob,
        joint_angle_exclude_band=joint_angle_exclude_band,
        joint_angle_max_frac=joint_angle_max_frac,
        pose_scheme=pose_scheme,
        out_path=out_path, video_dir=video_dir,
    )
    print(f"\nDone: success_rate={result['success_rate']:.1%} "
          f"({result['n_successful']}/{result['n_attempted']})")
