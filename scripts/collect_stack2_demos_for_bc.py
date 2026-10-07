"""
collect_stack2_demos_for_bc.py

Collects genuine, complete 2-cube stacking demo traces for BC, using the
SAME pipeline as collect_stack2_scripted_trials_v2.py (policy does cube
0's whole pick-and-place and cube 1's grasp autonomously; only cube 1's
final PLACEMENT -- lift->align-XY->descend -- is scripted, via a smooth
goal-directed proportional controller, not crude constant-action
segments) -- but run in a retry-until-N-successes loop, recording the
FULL (obs, action) trace of every attempt that fully succeeds, instead of
a fixed N trials just for video review.

Why this instead of bootstrapping a multi-cube-capable policy via PPO
first (the originally-proposed, more expensive path): the critical
missing behavior -- continuing to approach/grasp cube 1 after releasing
cube 0, instead of drifting -- is NEVER scripted here. It's exactly what
the CURRENT single-cube checkpoint sometimes does right already (we saw
3/10 complete stacks in collect_stack2_scripted_trials_v2.py's own test
run) -- just unreliably. Harvesting only the attempts where it already
worked, at larger scale, gives real demonstrations of that exact
behavior without needing a separate, expensive bootstrapping stage.
User's own idea, 2026-10-04.

Trade-off worth tracking: unlike the fully-autonomous 300-demo pool that
worked for the 3-cube case, these demos DO include a scripted segment
(cube 1's placement descent). That's a smooth, goal-directed controller,
not the abrupt constant-action segments that corrupted the earlier
10-demo fine-tune -- but it's not identical to what worked last time, so
watch the retrained checkpoint's base single-cube performance carefully,
same as always.

Run with:
    modal run --detach scripts/collect_stack2_demos_for_bc.py --target-successes 100
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR

STACK_XY = (1.30, 0.75)
TABLE_CX, TABLE_CY, TABLE_R = 1.30, 0.75, 0.20
TABLE_TOP_Z = 0.400
CUBE_HALF_SIZE = 0.025

CLEARANCE_Z = 0.15
ALIGN_XY_TOL = 0.01
DESCEND_TOL = 0.005
MAX_ALIGN_STEPS = 40
MAX_DESCEND_STEPS = 40
OPEN_GRIPPER_STEPS = 10
LIFT_CLEAR_STEPS = 10
POS_SCALE = 0.05


def _proportional_action(current_xyz, target_xyz, axes_mask, grip_value):
    import numpy as np
    direction = (target_xyz - current_xyz) * np.asarray(axes_mask)
    norm = float(np.linalg.norm(direction)) + 1e-8
    scale = min(1.0, norm / POS_SCALE)
    action = np.zeros(4, dtype=np.float32)
    action[:3] = (direction / norm) * scale
    action[3] = grip_value
    return np.clip(action, -1.0, 1.0).astype(np.float32)


@app.function(image=rl_image, gpu=None, cpu=2.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=3600 * 3)
def collect_stack2_demos_for_bc(
    ckpt_path: str = "checkpoints/flat_task_ppo_gripper3d_v1_cont_best.pt",
    policy_type: str = "mse",
    target_successes: int = 100,
    max_attempts: int = 1500,
    max_steps_per_cube: int = 100,
    seed_start: int = 0,
    lift_threshold: float = 0.02,
    distance_threshold: float = 0.05,
    min_separation: float = 0.09,
    out_path: str = "demonstrations/stack2_demos_for_bc_gripper3d.pkl",
    n_videos: int = 3,
    video_dir: str = "demonstrations/videos_stack2_demos_for_bc",
    report_every: int = 25,
) -> dict:
    import os
    import pickle
    import numpy as np

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    import gymnasium_robotics.envs.fetch.pick_and_place as pap_module

    from think_then_act.env.setup import (
        teleport_block, grip_contact_forces, save_video, randomize_gripper_start_3d, setup_env,
        init_random_episode,
    )
    from think_then_act.env.multicube import write_patched_xml, object_observation, get_object_xyz
    from think_then_act.training.subgoal_features import build_flat_observation, FLAT_OBS_DIM
    from think_then_act.training.flat_bc_multi_head import MultiHeadBCConfig, MultiHeadBCTrainer

    cfg = MultiHeadBCConfig(obs_dim=FLAT_OBS_DIM, policy_type=policy_type)
    trainer = MultiHeadBCTrainer(cfg)
    trainer.load_checkpoint(os.path.join(MODEL_CACHE_DIR, ckpt_path))
    print(f"loaded {ckpt_path}", flush=True)

    cube_half_sizes = [CUBE_HALF_SIZE, CUBE_HALF_SIZE]
    cube_resting_z = [TABLE_TOP_Z + hs for hs in cube_half_sizes]
    cube_full_heights = [2 * hs for hs in cube_half_sizes]
    cube_target_z = [TABLE_TOP_Z + sum(cube_full_heights[:i]) + cube_half_sizes[i] for i in range(2)]

    def make_2cube_env(render_mode=None):
        pap_module.MODEL_XML_PATH = write_patched_xml(1, cube_half_sizes)
        env = gym.make("FetchPickAndPlace-v3", max_episode_steps=max_steps_per_cube * 2 + 120,
                        **({"render_mode": render_mode} if render_mode else {}))
        setup_env(env)
        return env

    # Resume-safe: if out_path already has a demo pool (e.g. from an earlier run at this
    # same path), load and KEEP it, appending new demos on top -- never start fresh and
    # silently overwrite. Added 2026-10-04 after exactly that happened: relaunching this
    # script at the default out_path (meant to scale up the pool, not replace it) destroyed
    # the original 100 demos the moment the first new success landed, since the incremental
    # save always wrote a freshly-built `demonstrations` list. The trained checkpoint from
    # that original pool was already safe (saved separately, unaffected) -- only the raw
    # traces were lost, recoverable here going forward but should never happen again.
    full_out_path_check = os.path.join(MODEL_CACHE_DIR, out_path)
    demonstrations = []
    if os.path.exists(full_out_path_check):
        with open(full_out_path_check, "rb") as f:
            existing = pickle.load(f)
        demonstrations = existing.get("demonstrations", [])
        print(f"resuming from existing pool at {out_path}: {len(demonstrations)} demos already present", flush=True)
    n_success = len(demonstrations)
    attempt = 0
    while n_success < target_successes and attempt < max_attempts:
        seed = seed_start + attempt
        attempt += 1
        rng = np.random.default_rng(seed)
        record_video = n_success < n_videos
        env = make_2cube_env(render_mode="rgb_array" if record_video else None)
        reset_obs, _ = env.reset(seed=seed)
        reset_obs, setup_ok = init_random_episode(env, rng)
        if not setup_ok:
            env.close()
            continue
        reset_obs, pose_ok, _ = randomize_gripper_start_3d(env, rng, reset_obs)
        if not pose_ok:
            env.close()
            continue

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

        frames = [env.render()] if record_video else None
        hidden_state = None
        obs_list, action_list = [], []

        def record(flat_obs, action):
            obs_list.append(np.asarray(flat_obs, dtype=np.float32))
            action_list.append(np.asarray(action, dtype=np.float32))

        # ---- cube 0: fully autonomous ----
        site_name = "object0"
        desired_goal = np.array([STACK_XY[0], STACK_XY[1], cube_target_z[0]])
        env.unwrapped.goal = desired_goal.copy()
        observation, achieved_goal, desired = object_observation(env, site_name, desired_goal)
        cube0_genuine = False
        for _ in range(max_steps_per_cube):
            flat_obs = build_flat_observation(observation, achieved_goal, desired)
            action, hidden_state = trainer.actor.act(flat_obs, hidden_state, deterministic=True)
            record(flat_obs, action)
            env.step(action)
            if frames is not None:
                frames.append(env.render())
            observation, achieved_goal, desired = object_observation(env, site_name, desired)
            forces = grip_contact_forces(env, block_body_name=site_name)
            height_above_resting = float(achieved_goal[2]) - cube_resting_z[0]
            lifted_and_gripped = min(forces["left"], forces["right"]) > 0.0 and height_above_resting > lift_threshold
            d = float(np.linalg.norm(desired - achieved_goal))
            if lifted_and_gripped and d <= distance_threshold:
                cube0_genuine = True
                break

        cube1_genuine = False
        if cube0_genuine:
            # scripted release + reposition -- recorded as teacher actions too
            for action in (
                [np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)] * 10
                + [np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)] * 8
                + [np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)] * 3
            ):
                flat_obs = build_flat_observation(observation, achieved_goal, desired)
                record(flat_obs, action)
                _, hidden_state = trainer.actor.act(flat_obs, hidden_state, deterministic=True)
                env.step(action)
                if frames is not None:
                    frames.append(env.render())
                observation, achieved_goal, desired = object_observation(env, "object0", desired)

            def cube0_top_target():
                cube0_xyz = get_object_xyz(env, "object0")
                return np.array([cube0_xyz[0], cube0_xyz[1], cube0_xyz[2] + cube_full_heights[0]])

            # ---- cube 1: policy grasps (autonomous, the critical behavior), THEN scripted placement ----
            site_name = "object1"
            desired_goal = cube0_top_target()
            env.unwrapped.goal = desired_goal.copy()
            observation, achieved_goal, desired = object_observation(env, site_name, desired_goal)
            genuinely_grasped = False
            for _ in range(max_steps_per_cube):
                desired_goal = cube0_top_target()
                flat_obs = build_flat_observation(observation, achieved_goal, desired_goal)
                action, hidden_state = trainer.actor.act(flat_obs, hidden_state, deterministic=True)
                record(flat_obs, action)
                env.step(action)
                if frames is not None:
                    frames.append(env.render())
                observation, achieved_goal, desired = object_observation(env, site_name, desired_goal)
                forces = grip_contact_forces(env, block_body_name=site_name)
                height_above_resting = float(achieved_goal[2]) - cube_resting_z[1]
                if min(forces["left"], forces["right"]) > 0.0 and height_above_resting > lift_threshold:
                    genuinely_grasped = True
                    break

            if genuinely_grasped:
                grip_xyz = observation[0:3]
                target = cube0_top_target()
                clear_target = np.array([grip_xyz[0], grip_xyz[1], target[2] + CLEARANCE_Z])
                for _ in range(20):
                    flat_obs = build_flat_observation(observation, achieved_goal, desired)
                    action = _proportional_action(grip_xyz, clear_target, [1, 1, 1], -1.0)
                    record(flat_obs, action)
                    env.step(action)
                    if frames is not None:
                        frames.append(env.render())
                    observation, achieved_goal, desired = object_observation(env, site_name, desired)
                    grip_xyz = observation[0:3]
                    if abs(grip_xyz[2] - clear_target[2]) < 0.01:
                        break

                target = cube0_top_target()
                align_target = np.array([target[0], target[1], grip_xyz[2]])
                for _ in range(MAX_ALIGN_STEPS):
                    flat_obs = build_flat_observation(observation, achieved_goal, desired)
                    action = _proportional_action(grip_xyz, align_target, [1, 1, 0], -1.0)
                    record(flat_obs, action)
                    env.step(action)
                    if frames is not None:
                        frames.append(env.render())
                    observation, achieved_goal, desired = object_observation(env, site_name, desired)
                    grip_xyz = observation[0:3]
                    if np.linalg.norm(grip_xyz[:2] - align_target[:2]) < ALIGN_XY_TOL:
                        break

                target = cube0_top_target()
                desired_goal = target
                descend_target = np.array([target[0], target[1], target[2] + (achieved_goal[2] - grip_xyz[2])])
                for _ in range(MAX_DESCEND_STEPS):
                    flat_obs = build_flat_observation(observation, achieved_goal, desired_goal)
                    action = _proportional_action(grip_xyz, descend_target, [0, 0, 1], -1.0)
                    record(flat_obs, action)
                    env.step(action)
                    if frames is not None:
                        frames.append(env.render())
                    observation, achieved_goal, desired = object_observation(env, site_name, desired_goal)
                    grip_xyz = observation[0:3]
                    d = float(np.linalg.norm(desired_goal - achieved_goal))
                    if d <= distance_threshold:
                        break

                for action in (
                    [np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)] * OPEN_GRIPPER_STEPS
                    + [np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)] * LIFT_CLEAR_STEPS
                ):
                    flat_obs = build_flat_observation(observation, achieved_goal, desired_goal)
                    record(flat_obs, action)
                    env.step(action)
                    if frames is not None:
                        frames.append(env.render())
                    observation, achieved_goal, desired = object_observation(env, site_name, desired_goal)

                final_target = cube0_top_target()
                d = float(np.linalg.norm(final_target[:2] - achieved_goal[:2]))
                cube1_genuine = d <= distance_threshold

        all_genuine = cube0_genuine and cube1_genuine
        if attempt % report_every == 0:
            print(f"  attempt={attempt}  n_success={n_success}  this_attempt_genuine={all_genuine}  "
                  f"success_rate_so_far={n_success/attempt:.1%}", flush=True)

        if all_genuine:
            n_success += 1
            demonstrations.append({
                "obs": np.stack(obs_list).astype(np.float32),
                "teacher_action": np.stack(action_list).astype(np.float32),
                "success": True, "n_steps": len(obs_list),
            })
            if record_video and frames is not None:
                video_path = os.path.join(MODEL_CACHE_DIR, video_dir, f"stack2_demo_{n_success-1}.mp4")
                os.makedirs(os.path.dirname(video_path), exist_ok=True)
                save_video(frames, video_path, fps=20)
                model_volume.commit()
                print(f"    saved video -> {video_path}", flush=True)

            # Saved after EVERY new success, not just at the end -- this whole collection
            # runs on preemptible compute (see memory: flat_policy_ppo_generalization's
            # recurring-cancellation section) and this script previously had NO incremental
            # save at all, so a prior run was cancelled mid-flight and lost every single demo
            # collected up to that point. Fixed 2026-10-04, same lesson already applied to
            # the PPO training scripts earlier this session.
            full_out_path = os.path.join(MODEL_CACHE_DIR, out_path)
            os.makedirs(os.path.dirname(full_out_path), exist_ok=True)
            with open(full_out_path, "wb") as f:
                pickle.dump({
                    "demonstrations": demonstrations, "ckpt_path": ckpt_path,
                    "n_attempted": attempt, "n_successful": n_success,
                    "success_rate": n_success / attempt if attempt else 0.0,
                }, f)
            model_volume.commit()

        env.close()

    print(f"\n{n_success}/{target_successes} genuine 2-cube demos collected in {attempt} attempts "
          f"({n_success/attempt:.1%} success rate)", flush=True)

    full_out_path = os.path.join(MODEL_CACHE_DIR, out_path)
    os.makedirs(os.path.dirname(full_out_path), exist_ok=True)
    with open(full_out_path, "wb") as f:
        pickle.dump({
            "demonstrations": demonstrations, "ckpt_path": ckpt_path,
            "n_attempted": attempt, "n_successful": n_success,
            "success_rate": n_success / attempt if attempt else 0.0,
        }, f)
    model_volume.commit()
    print(f"Saved {n_success} demonstrations -> {full_out_path}", flush=True)

    return {"n_successful": n_success, "n_attempted": attempt}


@app.local_entrypoint()
def main(
    ckpt_path: str = "checkpoints/flat_task_ppo_gripper3d_v1_cont_best.pt",
    policy_type: str = "mse",
    target_successes: int = 100,
    max_attempts: int = 1500,
    out_path: str = "demonstrations/stack2_demos_for_bc_gripper3d.pkl",
    video_dir: str = "demonstrations/videos_stack2_demos_for_bc",
):
    result = collect_stack2_demos_for_bc.remote(
        ckpt_path=ckpt_path, policy_type=policy_type, target_successes=target_successes,
        max_attempts=max_attempts, out_path=out_path, video_dir=video_dir,
    )
    print(f"\nDone: {result['n_successful']}/{result['n_attempted']} attempts")
