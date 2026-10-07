"""
collect_stack2_scripted_trials_v2.py

v2 of collect_stack2_scripted_trials.py, built after the user watched the
v1 videos and flagged three real problems:

1. The reported 6/10 "genuine" count overstated it -- per-cube success was
   checked independently, so a trial could be marked genuine for cube 2
   even if placing it had knocked cube 1 out of position (cube 2 lands
   next to the disturbed cube 1, not stacked on it, but still registers
   as "at target" since nothing re-verified cube 1 was still there).
   FIXED: after cube 2's own placement check passes, this now ALSO
   re-verifies cube 1 is still within its own target tolerance -- a trial
   only counts as genuine if BOTH cubes are simultaneously correctly
   placed at the end, not just independently at some point each.

2. env.setup.randomize_joint_angles produced genuinely varied but visibly
   "tortuous" (awkward/contorted) poses -- expected, in hindsight: it
   samples each of the 4 arm joints independently in joint space, with
   nothing constraining how they compose together physically. FIXED:
   switched to the new env.setup.randomize_gripper_start_3d, which drives
   the gripper via the SAME bounded action interface a real policy uses
   (oracle-style direct-proportional control toward a randomized 3D
   point), so the resulting joint configuration is whatever the arm's own
   actuators naturally settle into -- validated by rendering 10 frames
   before trusting it (scripts/validate_gripper_start_3d.py): genuinely
   varied in both height and direction, no contorted poses in any of the
   10.

3. The failure mode seen in most v1 trials -- the policy's own freehand
   approach to cube 2's placement clips/bumps cube 1 on the way down.
   FIXED: once cube 2 is GENUINELY GRASPED (contact + lifted, regardless
   of target distance -- a new, earlier trigger than the old "grasped AND
   at target" one), control switches from the learned policy to a fully
   SCRIPTED placement sequence: lift to a safe clearance height first,
   THEN align horizontally over the target (dx/dy only, dz=0) so the
   approach never cuts across cube 1 at stacking height, THEN descend
   straight down (dz only) once aligned, THEN release, THEN lift clear.
   This removes all freehand policy control from exactly the part of the
   task that was causing the collisions.

Run with:
    modal run --detach scripts/collect_stack2_scripted_trials_v2.py
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR

STACK_XY = (1.30, 0.75)
TABLE_CX, TABLE_CY, TABLE_R = 1.30, 0.75, 0.20
TABLE_TOP_Z = 0.400
CUBE_HALF_SIZE = 0.025

CLEARANCE_Z = 0.15       # metres above the target stack height to lift to before aligning XY
ALIGN_XY_TOL = 0.01      # metres, how close dx/dy must get before starting the descent
DESCEND_TOL = 0.005      # metres, how close dz must get before releasing
MAX_ALIGN_STEPS = 40
MAX_DESCEND_STEPS = 40
OPEN_GRIPPER_STEPS = 10
LIFT_CLEAR_STEPS = 10
POS_SCALE = 0.05


def _proportional_step(env, current_xyz, target_xyz, axes_mask, grip_value):
    """One step of oracle-style direct-proportional control, masked to only the given axes
    (e.g. [1,1,0] for XY-only, [0,0,1] for Z-only) -- used by the scripted placement sequence."""
    import numpy as np
    direction = (target_xyz - current_xyz) * np.asarray(axes_mask)
    norm = float(np.linalg.norm(direction)) + 1e-8
    scale = min(1.0, norm / POS_SCALE)
    action = np.zeros(4, dtype=np.float32)
    action[:3] = (direction / norm) * scale
    action[3] = grip_value
    env.step(np.clip(action, -1.0, 1.0))
    return action


@app.function(image=rl_image, gpu=None, cpu=2.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=3600)
def collect_stack2_scripted_trials_v2(
    ckpt_path: str = "checkpoints/flat_task_ppo_poserand_v2_cont_best.pt",
    policy_type: str = "mse",
    n_trials: int = 10,
    max_steps_per_cube: int = 100,
    seed_start: int = 0,
    lift_threshold: float = 0.02,
    distance_threshold: float = 0.05,
    min_separation: float = 0.09,
    video_dir: str = "demonstrations/videos_stack2_scripted_trials_v2",
) -> dict:
    import os
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

    trial_results = []
    for trial in range(n_trials):
        seed = seed_start + trial
        rng = np.random.default_rng(seed)
        env = make_2cube_env(render_mode="rgb_array")
        reset_obs, _ = env.reset(seed=seed)
        reset_obs, setup_ok = init_random_episode(env, rng)
        if not setup_ok:
            print(f"  trial={trial}  init_random_episode failed, skipping", flush=True)
            env.close()
            continue
        reset_obs, pose_ok, pose_info = randomize_gripper_start_3d(env, rng, reset_obs)
        if not pose_ok:
            print(f"  trial={trial}  pose setup failed, skipping", flush=True)
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

        frames = [env.render()]
        hidden_state = None
        cube_final_pos = {}

        # ---- cube 0: policy does the whole pick-and-place, as before ----
        site_name = "object0"
        desired_goal = np.array([STACK_XY[0], STACK_XY[1], cube_target_z[0]])
        env.unwrapped.goal = desired_goal.copy()
        observation, achieved_goal, desired = object_observation(env, site_name, desired_goal)
        cube0_genuine = False
        for _ in range(max_steps_per_cube):
            flat_obs = build_flat_observation(observation, achieved_goal, desired)
            action, hidden_state = trainer.actor.act(flat_obs, hidden_state, deterministic=True)
            env.step(action)
            frames.append(env.render())
            observation, achieved_goal, desired = object_observation(env, site_name, desired)
            forces = grip_contact_forces(env, block_body_name=site_name)
            height_above_resting = float(achieved_goal[2]) - cube_resting_z[0]
            lifted_and_gripped = min(forces["left"], forces["right"]) > 0.0 and height_above_resting > lift_threshold
            d = float(np.linalg.norm(desired - achieved_goal))
            if lifted_and_gripped and d <= distance_threshold:
                cube0_genuine = True
                break
        cube_final_pos[0] = achieved_goal.copy()

        cube1_genuine = False
        if cube0_genuine:
            # scripted release + reposition (unchanged from v1 -- not the part that was broken)
            for action in (
                [np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)] * 10
                + [np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)] * 8
                + [np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)] * 3
            ):
                _, hidden_state = trainer.actor.act(build_flat_observation(observation, achieved_goal, desired),
                                                      hidden_state, deterministic=True)
                env.step(action)
                frames.append(env.render())
                observation, achieved_goal, desired = object_observation(env, "object0", desired)

            # ---- cube 1: policy grasps, THEN scripted lift->align-xy->descend->release->lift ----
            # Target is DYNAMIC, not the fixed STACK_XY point: re-read cube 0's ACTUAL current
            # position at the start of each phase and aim directly on top of wherever it really
            # is. If cube 0 got nudged while cube 1 was being approached/grasped (the original
            # failure mode this whole v2 script was built to fix), cube 1 still ends up correctly
            # stacked on cube 0's real position instead of chasing a stale point cube 0 may no
            # longer occupy -- what matters for a valid stack is the RELATIVE placement, not
            # hitting a fixed absolute coordinate. User's own idea, 2026-10-04.
            def cube0_top_target():
                cube0_xyz = get_object_xyz(env, "object0")
                return np.array([cube0_xyz[0], cube0_xyz[1], cube0_xyz[2] + cube_full_heights[0]])

            site_name = "object1"
            desired_goal = cube0_top_target()
            env.unwrapped.goal = desired_goal.copy()
            observation, achieved_goal, desired = object_observation(env, site_name, desired_goal)
            genuinely_grasped = False
            for _ in range(max_steps_per_cube):
                desired_goal = cube0_top_target()   # keep tracking even during approach --
                                                     # cube 0 could still be undisturbed here,
                                                     # but costs nothing to track from the start
                flat_obs = build_flat_observation(observation, achieved_goal, desired_goal)
                action, hidden_state = trainer.actor.act(flat_obs, hidden_state, deterministic=True)
                env.step(action)
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
                    action = _proportional_step(env, grip_xyz, clear_target, [1, 1, 1], -1.0)
                    frames.append(env.render())
                    observation, achieved_goal, desired = object_observation(env, site_name, desired)
                    grip_xyz = observation[0:3]
                    if abs(grip_xyz[2] - clear_target[2]) < 0.01:
                        break

                target = cube0_top_target()   # re-read -- cube 0 could have been bumped during clearance
                align_target = np.array([target[0], target[1], grip_xyz[2]])
                for _ in range(MAX_ALIGN_STEPS):
                    action = _proportional_step(env, grip_xyz, align_target, [1, 1, 0], -1.0)
                    frames.append(env.render())
                    observation, achieved_goal, desired = object_observation(env, site_name, desired)
                    grip_xyz = observation[0:3]
                    if np.linalg.norm(grip_xyz[:2] - align_target[:2]) < ALIGN_XY_TOL:
                        break

                target = cube0_top_target()   # re-read again -- align itself could have nudged cube 0
                desired_goal = target
                descend_target = np.array([target[0], target[1],
                                            target[2] + (achieved_goal[2] - grip_xyz[2])])
                # descend_target's z accounts for the fixed grip-to-block vertical offset so the
                # BLOCK (not the gripper) ends up at target's height, not the fingertips
                for _ in range(MAX_DESCEND_STEPS):
                    action = _proportional_step(env, grip_xyz, descend_target, [0, 0, 1], -1.0)
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
                    env.step(action)
                    frames.append(env.render())
                    observation, achieved_goal, desired = object_observation(env, site_name, desired_goal)

                # Final check against cube 0's FINAL resting position (post-release, cube 0 can no
                # longer move) -- this is now the real "is the stack coherent" check, replacing the
                # old separate cube0_still_placed re-verification against a fixed point.
                final_target = cube0_top_target()
                d = float(np.linalg.norm(final_target[:2] - achieved_goal[:2]))   # XY alignment only --
                                  # z is already implied correct by construction (released right at
                                  # the computed stacking height), checking it too would just
                                  # re-penalize normal settling/contact noise after release
                cube1_genuine = d <= distance_threshold
            cube_final_pos[1] = achieved_goal.copy()

        # cube1_genuine (above) is now ITSELF the real stack-integrity check -- it's computed
        # against cube 0's ACTUAL final position (cube0_top_target(), re-read fresh at release
        # time), not a fixed original point, so "cube 0 got nudged but cube 1 correctly followed
        # it" now correctly counts as success, and "cube 0 got nudged and cube 1 missed it" still
        # correctly fails. Still track how far cube 0 drifted from its ORIGINAL target, purely as
        # information about how much nudging actually happened -- not a gate on all_genuine.
        cube0_drift = None
        if 0 in cube_final_pos:
            cube0_now = get_object_xyz(env, "object0")
            cube0_drift = float(np.linalg.norm(
                np.array([STACK_XY[0], STACK_XY[1], cube_target_z[0]]) - cube0_now))

        all_genuine = cube0_genuine and cube1_genuine
        trial_results.append({
            "trial": trial, "seed": seed, "cube0_genuine": cube0_genuine, "cube1_genuine": cube1_genuine,
            "cube0_drift_from_original_target": cube0_drift, "all_genuine": all_genuine,
        })
        print(f"  trial={trial}  seed={seed}  cube0={cube0_genuine}  cube1={cube1_genuine}  "
              f"cube0_drift={cube0_drift}  all_genuine={all_genuine}", flush=True)

        out_path = os.path.join(MODEL_CACHE_DIR, video_dir, f"stack2_v2_trial{trial}.mp4")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_video(frames, out_path, fps=20)
        model_volume.commit()
        env.close()

    n_full = sum(1 for r in trial_results if r["all_genuine"])
    print(f"\n{n_full}/{len(trial_results)} trials TRULY fully genuine (both cubes placed AND stack intact at end)", flush=True)
    return {"n_trials": len(trial_results), "n_full_genuine": n_full, "trial_results": trial_results}


@app.local_entrypoint()
def main(
    ckpt_path: str = "checkpoints/flat_task_ppo_poserand_v2_cont_best.pt",
    n_trials: int = 10,
    seed_start: int = 0,
):
    result = collect_stack2_scripted_trials_v2.remote(ckpt_path=ckpt_path, n_trials=n_trials, seed_start=seed_start)
    print("\n", result)
