"""
collect_stack2_scripted_trials.py

Runs exactly 10 FIXED trials (not "retry until N successes" like
collect_stack3_demos.py) of the scripted 2-cube approach at NATIVE 5cm
cube size for BOTH cubes (abandoning the "bigger cubes" idea from
scripts/collect_stack3_demos.py's BIG_TIERED_HALF_SIZES -- directly
tested and found to HURT grasping reliability, 25.6%->7.1% success,
since checkpoints/flat_task_ppo_poserand_v2_cont_best.pt was only ever
trained on native-size cubes, no size randomization in this line of
work -- bigger cubes are genuinely out-of-distribution for grasping,
whatever they do for landing-margin precision).

Per trial: the trained policy runs autonomously on cube 1 until genuine
success or the step budget runs out; if genuine, a SCRIPTED (not live-
teleoperated) open-gripper/lift-clear/hold sequence takes over, then the
goal switches to cube 2 and the policy runs again, autonomously, until
genuine success or budget-out. ALL 10 trials get a video saved --
successes AND failures -- specifically so the real distribution of
outcomes is visible, not a cherry-picked sample (same reasoning behind
every "run N times, save N videos" eval script in this project, as
opposed to collect_stack3_demos.py's own success-only collection, which
served a different purpose -- generating clean BC training data, not
showing real reliability).

Also builds the MATCHED single-cube comparison: for each of the 10
trials, the EXACT starting arm pose (the 4 _ARM_POSE_JOINTS values) and
cube 1's EXACT starting XY are captured right after setup, then directly
REPLAYED (not re-sampled via the same seed/rng, which would NOT
guarantee an exact match since the 2-cube env's rng call sequence
diverges from a 1-cube env's partway through reset()) into a plain
single-object FetchPickAndPlace-v3 env, so the "basic skill" video is a
true apples-to-apples comparison at the identical starting condition the
2-cube trial actually used, not just a statistically similar one.

Run with:
    modal run --detach scripts/collect_stack2_scripted_trials.py
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR

STACK_XY = (1.30, 0.75)
TABLE_CX, TABLE_CY, TABLE_R = 1.30, 0.75, 0.20
TABLE_TOP_Z = 0.400
CUBE_HALF_SIZE = 0.025   # native size, both cubes -- see module docstring for why "bigger" was dropped

OPEN_GRIPPER_STEPS = 10
LIFT_CLEAR_STEPS = 8
HOLD_STEPS = 3

_ARM_POSE_JOINTS = [
    "robot0:shoulder_pan_joint", "robot0:shoulder_lift_joint",
    "robot0:elbow_flex_joint", "robot0:wrist_flex_joint",
]


def _get_arm_joint_angles(env) -> list:
    import mujoco
    raw = env.unwrapped
    values = []
    for name in _ARM_POSE_JOINTS:
        jid = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_adr = raw.model.jnt_qposadr[jid]
        values.append(float(raw.data.qpos[qpos_adr]))
    return values


def _set_arm_joint_angles(env, values: list) -> None:
    import mujoco
    raw = env.unwrapped
    for name, v in zip(_ARM_POSE_JOINTS, values):
        jid = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_adr = raw.model.jnt_qposadr[jid]
        raw.data.qpos[qpos_adr] = v
    mujoco.mj_forward(raw.model, raw.data)


@app.function(image=rl_image, gpu=None, cpu=2.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=3600)
def collect_stack2_scripted_trials(
    ckpt_path: str = "checkpoints/flat_task_ppo_poserand_v2_cont_best.pt",
    policy_type: str = "mse",
    n_trials: int = 10,
    max_steps_per_cube: int = 100,
    seed_start: int = 0,
    lift_threshold: float = 0.02,
    distance_threshold: float = 0.05,
    min_separation: float = 0.09,
    pose_exclude_band: float = 0.35,
    pose_max_frac: float = 0.85,
    video_dir: str = "demonstrations/videos_stack2_scripted_trials",
) -> dict:
    import os
    import numpy as np

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    import gymnasium_robotics.envs.fetch.pick_and_place as pap_module

    from think_then_act.env.setup import (
        teleport_block, grip_contact_forces, save_video, randomize_joint_angles, setup_env,
    )
    from think_then_act.env.multicube import write_patched_xml, object_observation
    from think_then_act.training.subgoal_features import build_flat_observation, FLAT_OBS_DIM
    from think_then_act.training.flat_bc_multi_head import MultiHeadBCConfig, MultiHeadBCTrainer

    cfg = MultiHeadBCConfig(obs_dim=FLAT_OBS_DIM, policy_type=policy_type)
    trainer = MultiHeadBCTrainer(cfg)
    trainer.load_checkpoint(os.path.join(MODEL_CACHE_DIR, ckpt_path))
    print(f"loaded {ckpt_path}", flush=True)

    cube_half_sizes = [CUBE_HALF_SIZE, CUBE_HALF_SIZE]  # [object0, object1], both native 5cm
    cube_resting_z = [TABLE_TOP_Z + hs for hs in cube_half_sizes]
    cube_full_heights = [2 * hs for hs in cube_half_sizes]
    cube_target_z = [TABLE_TOP_Z + sum(cube_full_heights[:i]) + cube_half_sizes[i] for i in range(2)]

    def make_2cube_env(render_mode=None):
        pap_module.MODEL_XML_PATH = write_patched_xml(1, cube_half_sizes)
        env = gym.make("FetchPickAndPlace-v3", max_episode_steps=max_steps_per_cube * 2 + 40,
                        **({"render_mode": render_mode} if render_mode else {}))
        setup_env(env)
        return env

    trial_results = []
    snapshots = []   # (trial, seed, arm_angles, cube1_xy) for the matched single-cube replay

    for trial in range(n_trials):
        seed = seed_start + trial
        rng = np.random.default_rng(seed)
        env = make_2cube_env(render_mode="rgb_array")
        env.reset(seed=seed)
        _, pose_ok = randomize_joint_angles(env, rng, exclude_band=pose_exclude_band, max_frac=pose_max_frac)
        if not pose_ok:
            print(f"  trial={trial}  pose setup failed, skipping", flush=True)
            env.close()
            continue
        arm_angles = _get_arm_joint_angles(env)

        points = [np.array(STACK_XY)]
        for _ in range(2):
            for _ in range(50):
                r = np.sqrt(rng.uniform(0.0, 1.0)) * TABLE_R
                theta = rng.uniform(0.0, 2.0 * np.pi)
                candidate = np.array([TABLE_CX + r * np.cos(theta), TABLE_CY + r * np.sin(theta)])
                if all(np.linalg.norm(candidate - p) > min_separation for p in points):
                    points.append(candidate)
                    break
            else:
                points.append(candidate)
        cube_xy = points[1:]
        for i in range(2):
            teleport_block(env, [cube_xy[i][0], cube_xy[i][1], cube_resting_z[i]], joint_name=f"object{i}:joint")
        env.step([0.0, 0.0, 0.0, 0.0])
        cube1_xy = (float(cube_xy[0][0]), float(cube_xy[0][1]))
        snapshots.append({"trial": trial, "seed": seed, "arm_angles": arm_angles, "cube1_xy": cube1_xy})

        frames = [env.render()]
        hidden_state = None
        per_cube_genuine = []
        for i in range(2):
            site_name = f"object{i}"
            desired_goal = np.array([STACK_XY[0], STACK_XY[1], cube_target_z[i]])
            env.unwrapped.goal = desired_goal.copy()
            observation, achieved_goal, desired = object_observation(env, site_name, desired_goal)
            genuine = False
            for _ in range(max_steps_per_cube):
                flat_obs = build_flat_observation(observation, achieved_goal, desired)
                action, hidden_state = trainer.actor.act(flat_obs, hidden_state, deterministic=True)
                env.step(action)
                frames.append(env.render())
                observation, achieved_goal, desired = object_observation(env, site_name, desired)
                forces = grip_contact_forces(env, block_body_name=site_name)
                height_above_resting = float(achieved_goal[2]) - cube_resting_z[i]
                lifted_and_gripped = min(forces["left"], forces["right"]) > 0.0 and height_above_resting > lift_threshold
                d = float(np.linalg.norm(desired - achieved_goal))
                if lifted_and_gripped and d <= distance_threshold:
                    genuine = True
                    break
            per_cube_genuine.append(genuine)
            if not genuine:
                break

            # scripted release + reposition
            for action in (
                [np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)] * OPEN_GRIPPER_STEPS
                + [np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)] * LIFT_CLEAR_STEPS
                + [np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)] * HOLD_STEPS
            ):
                _, hidden_state = trainer.actor.act(build_flat_observation(observation, achieved_goal, desired),
                                                      hidden_state, deterministic=True)
                env.step(action)
                frames.append(env.render())
                observation, achieved_goal, desired = object_observation(env, site_name, desired)

        all_genuine = len(per_cube_genuine) == 2 and all(per_cube_genuine)
        trial_results.append({"trial": trial, "seed": seed, "per_cube_genuine": per_cube_genuine, "all_genuine": all_genuine})
        print(f"  trial={trial}  seed={seed}  per_cube_genuine={per_cube_genuine}  all_genuine={all_genuine}", flush=True)

        out_path = os.path.join(MODEL_CACHE_DIR, video_dir, f"stack2_trial{trial}.mp4")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_video(frames, out_path, fps=20)
        model_volume.commit()
        env.close()

    n_full = sum(1 for r in trial_results if r["all_genuine"])
    print(f"\n{n_full}/{len(trial_results)} trials fully genuine (both cubes placed)", flush=True)

    # ---- matched single-cube comparison, exact state replay ----
    print("\nRunning matched single-cube comparison (same pose + cube XY as each trial above)...", flush=True)
    for snap in snapshots:
        env = gym.make("FetchPickAndPlace-v3", max_episode_steps=max_steps_per_cube + 20, render_mode="rgb_array")
        setup_env(env)
        env.reset(seed=snap["seed"])
        _set_arm_joint_angles(env, snap["arm_angles"])
        teleport_block(env, [snap["cube1_xy"][0], snap["cube1_xy"][1], cube_resting_z[0]], joint_name="object0:joint")
        env.step([0.0, 0.0, 0.0, 0.0])

        desired_goal = np.array([STACK_XY[0], STACK_XY[1], cube_target_z[0]])
        env.unwrapped.goal = desired_goal.copy()
        observation, achieved_goal, desired = object_observation(env, "object0", desired_goal)
        frames = [env.render()]
        hidden_state = None
        genuine = False
        for _ in range(max_steps_per_cube):
            flat_obs = build_flat_observation(observation, achieved_goal, desired)
            action, hidden_state = trainer.actor.act(flat_obs, hidden_state, deterministic=True)
            env.step(action)
            frames.append(env.render())
            observation, achieved_goal, desired = object_observation(env, "object0", desired)
            forces = grip_contact_forces(env, block_body_name="object0")
            height_above_resting = float(achieved_goal[2]) - cube_resting_z[0]
            lifted_and_gripped = min(forces["left"], forces["right"]) > 0.0 and height_above_resting > lift_threshold
            d = float(np.linalg.norm(desired - achieved_goal))
            if lifted_and_gripped and d <= distance_threshold:
                genuine = True
                break
        print(f"  basic-skill trial={snap['trial']}  genuine={genuine}", flush=True)

        out_path = os.path.join(MODEL_CACHE_DIR, video_dir, f"basicskill_trial{snap['trial']}.mp4")
        save_video(frames, out_path, fps=20)
        model_volume.commit()
        env.close()

    return {"n_trials": len(trial_results), "n_full_genuine": n_full, "trial_results": trial_results}


@app.local_entrypoint()
def main(
    ckpt_path: str = "checkpoints/flat_task_ppo_poserand_v2_cont_best.pt",
    n_trials: int = 10,
    seed_start: int = 0,
):
    result = collect_stack2_scripted_trials.remote(ckpt_path=ckpt_path, n_trials=n_trials, seed_start=seed_start)
    print("\n", result)
