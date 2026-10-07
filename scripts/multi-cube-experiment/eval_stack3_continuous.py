"""
eval_stack3_continuous.py

Tests whether train_stack3_incontext_finetune.py's checkpoint actually
learned the in-context release-and-continue behavior, as opposed to just
lowering its training loss. Critical difference from eval_stack3_cubes.py:
this NEVER resets hidden_state between cubes within a trial -- one GRU
hidden state carries continuously across all 3 cubes, and there is NO
scripted release here (unlike collect_stack3_demos.py) -- the moment
genuine success fires for cube i, the goal switches straight to cube i+1
and the SAME policy has to decide on its own, from nothing but the
observation stream, to let go and reposition. eval_stack3_cubes.py's own
per-cube hidden_state=None reset makes it structurally incapable of
testing this -- a network can't exhibit a behavior that depends on
remembering what just happened if its memory is wiped at exactly the
moment that "what just happened" occurred.

Runs the SAME protocol against both the original (pre-fine-tune) and the
fine-tuned checkpoint, so any difference is attributable to the fine-tune
and not to some property of the continuous-episode protocol itself.

Run with:
    modal run scripts/multi-cube-experiment/eval_stack3_continuous.py
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR

TABLE_TOP_Z = 0.400
TIERED_HALF_SIZES = [0.025, 0.020, 0.015]
STACK_XY = (1.30, 0.75)
TABLE_CX, TABLE_CY, TABLE_R = 1.30, 0.75, 0.20
CUBE_COLORS = [None, "0.15 0.75 0.15 1", "0.15 0.35 0.85 1"]


def _write_patched_xml(cube_half_sizes) -> str:
    import os
    import re
    import gymnasium_robotics

    assets_dir = os.path.join(os.path.dirname(gymnasium_robotics.__file__), "envs", "assets", "fetch")
    src_path = os.path.join(assets_dir, "pick_and_place.xml")
    with open(src_path) as f:
        xml = f.read()
    match = re.search(r'<body name="object0"[^>]*>.*?</body>', xml, re.DOTALL)
    assert match, "pick_and_place.xml has no <body name=\"object0\">...</body> block"
    object0_block = match.group(0)
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
    patched = xml.replace(object0_block, object0_block + "\n                " + "\n                ".join(extra_blocks), 1)
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


@app.function(image=rl_image, gpu=None, cpu=2.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=900)
def eval_stack3_continuous(
    ckpt_path: str,
    policy_type: str = "mse",
    max_steps_per_cube: int = 100,
    n_trials: int = 10,
    seed_start: int = 0,
    lift_threshold: float = 0.02,
    distance_threshold: float = 0.05,
    min_separation: float = 0.09,
    pose_exclude_band: float = 0.35,
    pose_max_frac: float = 0.85,
    render_first_trial: bool = True,
    video_out_name: str = "stack3_continuous_trial0",
) -> dict:
    import os
    import numpy as np

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import teleport_block, grip_contact_forces, save_video, randomize_joint_angles
    from think_then_act.training.subgoal_features import build_flat_observation, FLAT_OBS_DIM
    from think_then_act.training.flat_bc_multi_head import MultiHeadBCConfig, MultiHeadBCTrainer

    cube_half_sizes = TIERED_HALF_SIZES
    cube_resting_z = [TABLE_TOP_Z + hs for hs in cube_half_sizes]
    cube_full_heights = [2 * hs for hs in cube_half_sizes]
    cube_target_z = [TABLE_TOP_Z + sum(cube_full_heights[:i]) + cube_half_sizes[i] for i in range(3)]

    cfg = MultiHeadBCConfig(obs_dim=FLAT_OBS_DIM, policy_type=policy_type)
    trainer = MultiHeadBCTrainer(cfg)
    trainer.load_checkpoint(os.path.join(MODEL_CACHE_DIR, ckpt_path))
    print(f"loaded {ckpt_path}", flush=True)

    trial_results = []
    for trial in range(n_trials):
        seed = seed_start + trial
        rng = np.random.default_rng(seed)
        render_this_trial = render_first_trial and trial == 0
        env = _make_stack_env(max_steps_per_cube * 3 + 20, render_mode="rgb_array" if render_this_trial else None,
                               cube_half_sizes=cube_half_sizes)
        env.reset(seed=seed)
        _, pose_ok = randomize_joint_angles(env, rng, exclude_band=pose_exclude_band, max_frac=pose_max_frac)
        if not pose_ok:
            env.close()
            continue

        points = [np.array(STACK_XY)]
        for _ in range(3):
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

        for i in range(3):
            teleport_block(env, [cube_xy[i][0], cube_xy[i][1], cube_resting_z[i]], joint_name=f"object{i}:joint")
        env.step([0.0, 0.0, 0.0, 0.0])

        frames = [env.render()] if render_this_trial else None
        hidden_state = None   # ONE hidden state across all 3 cubes -- the entire point of this eval
        per_cube = []

        for i in range(3):
            site_name = f"object{i}"
            desired_goal = np.array([STACK_XY[0], STACK_XY[1], cube_target_z[i]])
            env.unwrapped.goal = desired_goal.copy()

            observation, achieved_goal, desired = _object_observation(env, site_name, desired_goal)
            genuine = False
            for t in range(max_steps_per_cube):
                flat_obs = build_flat_observation(observation, achieved_goal, desired)
                action, hidden_state = trainer.actor.act(flat_obs, hidden_state, deterministic=True)
                env.step(action)
                if frames is not None:
                    frames.append(env.render())
                observation, achieved_goal, desired = _object_observation(env, site_name, desired)

                forces = grip_contact_forces(env, block_body_name=f"object{i}")
                height_above_resting = float(achieved_goal[2]) - cube_resting_z[i]
                lifted_and_gripped = min(forces["left"], forces["right"]) > 0.0 and height_above_resting > lift_threshold
                d = float(np.linalg.norm(desired - achieved_goal))
                if lifted_and_gripped and d <= distance_threshold:
                    genuine = True
                    break   # NO scripted takeover -- advance straight to the next cube's goal and
                            # let the SAME policy (and hidden state) decide what to do next, on its own

            per_cube.append({"cube": i, "genuine": genuine,
                              "final_dist_to_target": round(float(np.linalg.norm(desired - achieved_goal)), 4)})
            print(f"  trial={trial} cube={i} genuine={genuine} dist={per_cube[-1]['final_dist_to_target']}", flush=True)
            if not genuine:
                break   # cube i never succeeded -- no point in continuing to i+1 with a block never placed

        all_three = len(per_cube) == 3 and all(c["genuine"] for c in per_cube)
        trial_results.append({"trial": trial, "seed": seed, "per_cube": per_cube, "all_three_stacked": all_three})
        print(f"trial={trial}  all_three_stacked={all_three}", flush=True)

        if frames is not None:
            out_path = os.path.join(MODEL_CACHE_DIR, f"demonstrations/videos/{video_out_name}.mp4")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            save_video(frames, out_path, fps=20)
            model_volume.commit()
            print(f"  saved video -> {out_path}", flush=True)

        env.close()

    n_all_three = sum(1 for r in trial_results if r["all_three_stacked"])
    per_layer_genuine = [
        sum(1 for r in trial_results if len(r["per_cube"]) > i and r["per_cube"][i]["genuine"])
        for i in range(3)
    ]
    print(f"\n=== Summary over {len(trial_results)} trials ({ckpt_path}) ===", flush=True)
    print(f"  all 3 genuinely stacked: {n_all_three}/{len(trial_results)}", flush=True)
    print(f"  per-layer genuine rate: layer0={per_layer_genuine[0]}  layer1={per_layer_genuine[1]}  layer2={per_layer_genuine[2]}", flush=True)

    return {"ckpt_path": ckpt_path, "n_trials": len(trial_results), "n_all_three_stacked": n_all_three,
            "per_layer_genuine": per_layer_genuine, "trial_results": trial_results}


@app.local_entrypoint()
def main(
    ckpt_path: str = "checkpoints/stack3_incontext_v1.pt",
    n_trials: int = 10,
    video_out_name: str = "stack3_continuous_trial0",
):
    result = eval_stack3_continuous.remote(ckpt_path=ckpt_path, n_trials=n_trials, video_out_name=video_out_name)
    print("\n", result)
