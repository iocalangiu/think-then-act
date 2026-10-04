"""
collect_stack3_demos.py

Generates full 3-cube-stacking demonstration traces for the in-context
fine-tune: each demo is ONE CONTINUOUS episode (single GRU hidden state,
never reset mid-trace) covering all 3 cubes, so training on these traces
can teach the policy to infer "the goal just changed, release and move
on" from the pattern in its own observation stream — nothing else marks
that transition; there's no explicit phase/subtask channel in
build_flat_observation.

Per cube: the ALREADY-TRAINED policy (checkpoints/flat_task_ppo_poserand_
v2_cont_best.pt) runs autonomously until genuine success (two-finger
contact + lifted + at target — this project's standing verification),
then a SCRIPTED (not live-teleoperated) sequence takes over: open the
gripper, lift clear of the stack, hold briefly — the one behavior nothing
in this project has ever demonstrated. Then the goal switches to the next
cube and control returns to the trained policy. The scripted segment's
own actions are recorded as the "teacher" label for those steps, exactly
like every other demo in this project.

Uses the TIERED cube sizes (big/medium/small, see eval_stack3_cubes.py's
TIERED_HALF_SIZES) specifically so collection doesn't need placement
precision to succeed reliably, and randomized starting poses (env.setup.
randomize_joint_angles) — confirmed 2026-10-03 to be essential: the SAME
checkpoint goes from 3/10 to 10/10 on cube 1 alone once poses are
randomized instead of fixed-stretched (see the "Pose Generalization &
Stacking" artifact). Keeps attempting (new seed each time) until
n_target successes are collected or the attempt budget runs out — same
target_successes convention as collect_sb3_teacher_demos.py.

Run with:
    modal run --detach scripts/collect_stack3_demos.py
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR

STACK_XY = (1.30, 0.75)
TABLE_CX, TABLE_CY, TABLE_R = 1.30, 0.75, 0.20
TABLE_TOP_Z = 0.400
TIERED_HALF_SIZES = [0.025, 0.020, 0.015]
# "Much bigger" tier, 2026-10-04: current (5/4/3cm) sizes judged too small/imprecise-
# looking in the scripted demos. Gripper's physical max opening measured directly at
# ~0.10m (scripts/measure_gripper_axis.py); this project's own prior validated-safe
# ceiling for the grasping (width) axis is 0.08m full-width (env/block_randomization.py's
# WIDTH_RANGE upper bound, used for close_gripper's curriculum training) -- biggest tier
# here (0.040 half-size = 8cm full) sits right at that validated ceiling, not guessed.
BIG_TIERED_HALF_SIZES = [0.040, 0.035, 0.030]
CUBE_COLORS = [None, "0.15 0.75 0.15 1", "0.15 0.35 0.85 1"]

OPEN_GRIPPER_STEPS = 10   # steps of action=[0,0,0,+1] (max open) after genuine success
LIFT_CLEAR_STEPS = 8      # steps of action=[0,0,+1,+1] (lift, stay open) afterward
HOLD_STEPS = 3            # steps of zero action before handing back to the policy


def _write_patched_xml(cube_half_sizes) -> str:
    import os
    import re
    import gymnasium_robotics

    assets_dir = os.path.join(os.path.dirname(gymnasium_robotics.__file__), "envs", "assets", "fetch")
    src_path = os.path.join(assets_dir, "pick_and_place.xml")
    with open(src_path) as f:
        xml = f.read()
    match = re.search(r'<body name="object0"[^>]*>.*?</body>', xml, re.DOTALL)
    assert match, "pick_and_place.xml has no \"object0\">...</body> block"
    object0_block_orig = match.group(0)

    # Resize object0's OWN geom too (previously left at its native 5cm always) --
    # needed for a "much bigger, all 3 cubes" tier, not just the 2 duplicated extras.
    # Native line: <geom size="0.025 0.025 0.025" type="box" condim="3" name="object0"
    # material="block_mat" mass="2"></geom> -- only the size="..." attribute changes,
    # the site (just a small fixed-size visual marker, not the cube) is untouched.
    hs0 = cube_half_sizes[0]
    object0_block, n_resized = re.subn(
        r'(<geom size=")[\d. ]+(" type="box" condim="3" name="object0")',
        rf'\g<1>{hs0} {hs0} {hs0}\g<2>', object0_block_orig, count=1,
    )
    assert n_resized == 1, "failed to resize object0's own geom -- pick_and_place.xml format changed?"

    extra_blocks = []
    for i in (1, 2):
        hs = cube_half_sizes[i]
        rgba = CUBE_COLORS[i]
        site_size = min(0.02, hs * 0.8)
        extra_blocks.append(f'''<body name="object{i}" pos="{0.025 + 0.1 * i} 0.125 0.025">
                        <joint name="object{i}:joint" type="free" damping="0.01"></joint>
                        <geom size="{hs} {hs} {hs}" type="box" condim="3" name="object{i}" rgba="{rgba}" mass="2"></geom>
                        <site name="object{i}" pos="0 0 0" size="{site_size} {site_size} {site_size}" rgba="{rgba}" type="sphere"></site>
                </body>''')
    patched = xml.replace(object0_block_orig, object0_block + "\n                " + "\n                ".join(extra_blocks), 1)
    out_path = os.path.join(assets_dir, "pick_and_place_stack3.xml")
    with open(out_path, "w") as f:
        f.write(patched)
    return os.path.join("fetch", "pick_and_place_stack3.xml")


def _make_stack_env(max_episode_steps: int, render_mode: str = None, cube_half_sizes=None):
    import gymnasium as gym
    import gymnasium_robotics.envs.fetch.pick_and_place as pap_module
    from think_then_act.env.setup import setup_env

    pap_module.MODEL_XML_PATH = _write_patched_xml(cube_half_sizes or TIERED_HALF_SIZES)
    kwargs = dict(max_episode_steps=max_episode_steps)
    if render_mode:
        kwargs["render_mode"] = render_mode
    env = gym.make("FetchPickAndPlace-v3", **kwargs)
    setup_env(env)
    return env


def _object_observation(env, site_name: str, desired_goal):
    import numpy as np
    from gymnasium_robotics.utils import rotations

    raw = env.unwrapped
    utils = raw._utils
    dt = raw.n_substeps * raw.model.opt.timestep
    grip_pos = utils.get_site_xpos(raw.model, raw.data, "robot0:grip")
    grip_velp = utils.get_site_xvelp(raw.model, raw.data, "robot0:grip") * dt
    robot_qpos, robot_qvel = utils.robot_get_obs(raw.model, raw.data, raw._model_names.joint_names)
    gripper_state = robot_qpos[-2:]
    gripper_vel = robot_qvel[-2:] * dt
    achieved_goal = utils.get_site_xpos(raw.model, raw.data, site_name).copy()
    object_rot = rotations.mat2euler(utils.get_site_xmat(raw.model, raw.data, site_name))
    object_velp = utils.get_site_xvelp(raw.model, raw.data, site_name) * dt - grip_velp
    object_velr = utils.get_site_xvelr(raw.model, raw.data, site_name) * dt
    observation = np.concatenate([
        grip_pos, np.zeros(3), np.zeros(3), gripper_state,
        object_rot, object_velp, object_velr, grip_velp, gripper_vel,
    ]).astype(np.float32)
    return observation, achieved_goal, np.asarray(desired_goal, dtype=np.float32)


@app.function(image=rl_image, gpu=None, cpu=2.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=3600)
def collect_stack3_demos(
    ckpt_path: str = "checkpoints/flat_task_ppo_poserand_v2_cont_best.pt",
    policy_type: str = "mse",
    target_successes: int = 10,
    max_attempts: int = 200,
    max_steps_per_cube: int = 100,
    seed_start: int = 0,
    lift_threshold: float = 0.02,
    distance_threshold: float = 0.05,
    min_separation: float = 0.09,
    pose_exclude_band: float = 0.35,
    pose_max_frac: float = 0.85,
    size_variant: str = "tiered",   # "tiered" (original 5/4/3cm) or "big" (new 8/7/6cm,
                                  # the validated-safe ceiling for the gripper's ~10cm
                                  # measured max opening -- see BIG_TIERED_HALF_SIZES)
    out_path: str = "demonstrations/stack3_incontext_demos_big.pkl",
    n_videos: int = 2,
    video_dir: str = "demonstrations/videos_stack3_demos_big",
) -> dict:
    import os
    import pickle
    import numpy as np

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import (
        teleport_block, grip_contact_forces, save_video, randomize_joint_angles,
    )
    from think_then_act.training.subgoal_features import build_flat_observation, FLAT_OBS_DIM
    from think_then_act.training.flat_bc_multi_head import MultiHeadBCConfig, MultiHeadBCTrainer

    cube_half_sizes = BIG_TIERED_HALF_SIZES if size_variant == "big" else TIERED_HALF_SIZES
    cube_resting_z = [TABLE_TOP_Z + hs for hs in cube_half_sizes]
    cube_full_heights = [2 * hs for hs in cube_half_sizes]
    cube_target_z = [
        TABLE_TOP_Z + sum(cube_full_heights[:i]) + cube_half_sizes[i] for i in range(3)
    ]
    print(f"cube_half_sizes={cube_half_sizes}  resting_z={cube_resting_z}  target_z={cube_target_z}", flush=True)

    cfg = MultiHeadBCConfig(obs_dim=FLAT_OBS_DIM, policy_type=policy_type)
    trainer = MultiHeadBCTrainer(cfg)
    trainer.load_checkpoint(os.path.join(MODEL_CACHE_DIR, ckpt_path))
    print(f"loaded {ckpt_path}", flush=True)

    demonstrations = []
    n_success = 0
    attempt = 0
    while n_success < target_successes and attempt < max_attempts:
        seed = seed_start + attempt
        attempt += 1
        rng = np.random.default_rng(seed)
        record_video = n_success < n_videos
        env = _make_stack_env(max_steps_per_cube * 3 + 80, render_mode="rgb_array" if record_video else None,
                               cube_half_sizes=cube_half_sizes)
        env.reset(seed=seed)

        _, pose_ok = randomize_joint_angles(env, rng, exclude_band=pose_exclude_band, max_frac=pose_max_frac)
        if not pose_ok:
            env.close()
            continue

        points = [np.array(STACK_XY)]
        ok_placement = True
        for _ in range(3):
            placed = False
            for _ in range(50):
                r = np.sqrt(rng.uniform(0.0, 1.0)) * TABLE_R
                theta = rng.uniform(0.0, 2.0 * np.pi)
                candidate = np.array([TABLE_CX + r * np.cos(theta), TABLE_CY + r * np.sin(theta)])
                if all(np.linalg.norm(candidate - p) > min_separation for p in points):
                    points.append(candidate)
                    placed = True
                    break
            if not placed:
                ok_placement = False
                break
        if not ok_placement:
            env.close()
            continue
        cube_xy = points[1:]

        for i in range(3):
            teleport_block(env, [cube_xy[i][0], cube_xy[i][1], cube_resting_z[i]], joint_name=f"object{i}:joint")
        env.step([0.0, 0.0, 0.0, 0.0])

        frames = [env.render()] if record_video else None
        flat_obs_list, action_list = [], []
        hidden_state = None   # ONE hidden state, carried across all 3 cubes + scripted segments
        all_genuine = True

        for i in range(3):
            site_name = f"object{i}"
            desired_goal = np.array([STACK_XY[0], STACK_XY[1], cube_target_z[i]])
            env.unwrapped.goal = desired_goal.copy()

            observation, achieved_goal, desired = _object_observation(env, site_name, desired_goal)
            genuine_this_cube = False
            for _ in range(max_steps_per_cube):
                flat_obs = build_flat_observation(observation, achieved_goal, desired)
                action, hidden_state = trainer.actor.act(flat_obs, hidden_state, deterministic=True)
                flat_obs_list.append(flat_obs)
                action_list.append(np.asarray(action, dtype=np.float32))

                env.step(action)
                if frames is not None:
                    frames.append(env.render())
                observation, achieved_goal, desired = _object_observation(env, site_name, desired_goal)

                forces = grip_contact_forces(env, block_body_name=f"object{i}")
                height_above_resting = float(achieved_goal[2]) - cube_resting_z[i]
                lifted_and_gripped = min(forces["left"], forces["right"]) > 0.0 and height_above_resting > lift_threshold
                d = float(np.linalg.norm(desired - achieved_goal))
                if lifted_and_gripped and d <= distance_threshold:
                    genuine_this_cube = True
                    break   # stop the autonomous phase the INSTANT genuine success is reached

            if not genuine_this_cube:
                all_genuine = False
                break   # this attempt failed -- discard, don't bother scripting the release

            # ---- scripted release + reposition (the behavior nothing else has demonstrated) ----
            for _ in range(OPEN_GRIPPER_STEPS):
                action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)   # open gripper
                flat_obs = build_flat_observation(observation, achieved_goal, desired)
                flat_obs_list.append(flat_obs)
                action_list.append(action)
                _, hidden_state = trainer.actor.act(flat_obs, hidden_state, deterministic=True)   # advance hidden
                                  # state through the SCRIPTED action too (act() is only called
                                  # here to propagate the GRU's hidden state consistently with how
                                  # BC training will replay it -- the returned action is discarded,
                                  # the SCRIPTED action is what's recorded and stepped).
                env.step(action)
                if frames is not None:
                    frames.append(env.render())
                observation, achieved_goal, desired = _object_observation(env, site_name, desired)

            for _ in range(LIFT_CLEAR_STEPS):
                action = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)   # lift clear, stay open
                flat_obs = build_flat_observation(observation, achieved_goal, desired)
                flat_obs_list.append(flat_obs)
                action_list.append(action)
                _, hidden_state = trainer.actor.act(flat_obs, hidden_state, deterministic=True)
                env.step(action)
                if frames is not None:
                    frames.append(env.render())
                observation, achieved_goal, desired = _object_observation(env, site_name, desired)

            for _ in range(HOLD_STEPS):
                action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                flat_obs = build_flat_observation(observation, achieved_goal, desired)
                flat_obs_list.append(flat_obs)
                action_list.append(action)
                _, hidden_state = trainer.actor.act(flat_obs, hidden_state, deterministic=True)
                env.step(action)
                if frames is not None:
                    frames.append(env.render())
                observation, achieved_goal, desired = _object_observation(env, site_name, desired)

        status = "SUCCESS" if all_genuine else "fail"
        print(f"  attempt={attempt-1}  seed={seed}  {status}  n_steps={len(flat_obs_list)}", flush=True)

        if all_genuine:
            n_success += 1
            demonstrations.append({
                "obs": np.stack(flat_obs_list).astype(np.float32),
                "teacher_action": np.stack(action_list).astype(np.float32),
                "success": True,
                "n_steps": len(flat_obs_list),
            })
            if record_video and frames is not None:
                video_path = os.path.join(MODEL_CACHE_DIR, video_dir, f"stack3_demo_{n_success-1}.mp4")
                os.makedirs(os.path.dirname(video_path), exist_ok=True)
                save_video(frames, video_path, fps=20)
                model_volume.commit()
                print(f"    saved video -> {video_path}", flush=True)

        env.close()

    print(f"\n{n_success}/{target_successes} full-stack demos collected in {attempt} attempts "
          f"({n_success/attempt:.1%} success rate)", flush=True)

    full_out_path = os.path.join(MODEL_CACHE_DIR, out_path)
    os.makedirs(os.path.dirname(full_out_path), exist_ok=True)
    with open(full_out_path, "wb") as f:
        pickle.dump({
            "demonstrations": demonstrations, "ckpt_path": ckpt_path,
            "n_attempted": attempt, "n_successful": n_success,
            "success_rate": n_success / attempt if attempt else 0.0,
            "cube_half_sizes": cube_half_sizes,
        }, f)
    model_volume.commit()
    print(f"Saved {n_success} demonstrations -> {full_out_path}", flush=True)

    return {"n_successful": n_success, "n_attempted": attempt}


@app.local_entrypoint()
def main(
    ckpt_path: str = "checkpoints/flat_task_ppo_poserand_v2_cont_best.pt",
    target_successes: int = 10,
    max_attempts: int = 200,
    size_variant: str = "tiered",
    out_path: str = "demonstrations/stack3_incontext_demos_big.pkl",
    video_dir: str = "demonstrations/videos_stack3_demos_big",
):
    result = collect_stack3_demos.remote(ckpt_path=ckpt_path, target_successes=target_successes,
                                          max_attempts=max_attempts, size_variant=size_variant,
                                          out_path=out_path, video_dir=video_dir)
    print(f"\nDone: {result['n_successful']}/{result['n_attempted']} attempts")
