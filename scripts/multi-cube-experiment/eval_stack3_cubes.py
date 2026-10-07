"""
eval_stack3_cubes.py

Evaluates the current flat single-object pick-and-place policy on a task
it was never trained for: stacking 3 cubes at one FIXED location. Built
2026-10-03 to directly test how an already-trained single-object policy
generalizes when (a) extra, otherwise-irrelevant objects are physically
present in the scene and (b) it has to be invoked three times in a row,
each time treating a different cube as "the" object, with the target's
z increasing by one cube-height each time to represent stacking.

The policy's own observation (training/subgoal_features.py's
build_flat_observation, 22-dim) only ever perceives ONE object and ONE
target at a time — nothing about this eval changes that. What it DOES
need is an environment that physically contains 3 free-floating cubes
simultaneously (FetchPickAndPlace-v3 natively has exactly one, "object0"),
and a way to correctly read/track whichever cube is "active" at each of
the 3 sequential sub-episodes.

How the extra cubes are added: a patched copy of gymnasium_robotics' own
pick_and_place.xml, written into the SAME directory as the original (so
its <include file="shared.xml"> and compiler meshdir/texturedir paths
still resolve), with object0's whole <body> element duplicated twice as
"object1"/"object2" (own joint, geom, site — distinct rgba so a rendered
video can tell them apart). gymnasium_robotics.envs.fetch.pick_and_place's
module-level MODEL_XML_PATH constant is monkeypatched to this new file
before gym.make() — MujocoFetchPickAndPlaceEnv.__init__ doesn't expose
model_path as a constructor kwarg, it's read from that module global at
call time, so this is the minimal way to redirect it without copying that
whole __init__ body into a subclass.

Why NOT the native obs/achieved_goal/info["is_success"] for cubes 1/2:
confirmed directly from gymnasium_robotics' own fetch_env.py source
(generate_mujoco_observations) that achieved_goal and 9 of the raw 25-dim
observation's entries (object_rot/object_velp/object_velr) are always
computed from the SITE NAMED "object0" specifically — they would silently
report object0's state while this script is trying to manipulate object1
or object2. Fixed by recomputing those 3 blocks directly via the SAME
mujoco_utils calls fetch_env.py itself uses (get_site_xmat/xvelp/xvelr),
just parameterized on whichever site name is "active" this sub-episode —
not by touching the native obs machinery at all.

Run with:
    modal run --detach scripts/eval_stack3_cubes.py
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR

BLOCK_HALF_SIZE = 0.025   # matches object0's own geom size in pick_and_place.xml
BLOCK_FULL_HEIGHT = 2 * BLOCK_HALF_SIZE
BLOCK_RESTING_Z = 0.425
STACK_XY = (1.30, 0.75)   # table center, same constant init_random_episode uses
TABLE_CX, TABLE_CY, TABLE_R = 1.30, 0.75, 0.20
TABLE_TOP_Z = 0.400   # see env/setup.py's own TABLE_TOP_Z — table surface, not block-resting-CENTER height
CUBE_COLORS = [None, "0.15 0.75 0.15 1", "0.15 0.35 0.85 1"]   # object0 keeps its
                                  # native block_mat color; object1 green, object2 blue
EQUAL_HALF_SIZES = [0.025, 0.025, 0.025]        # original variant: all 3 cubes identical
TIERED_HALF_SIZES = [0.025, 0.020, 0.015]       # 2026-10-03: biggest-first pyramid, added to
                                  # test whether the compounding failures in the equal-size
                                  # version are partly a PRECISION problem (hitting a small
                                  # target margin) rather than purely the missing release-on-
                                  # target behavior — object0 kept at the native 0.025 (no XML
                                  # change needed there), object1/object2 shrunk so each cube
                                  # lands on a progressively bigger margin of exposed surface.


def _write_patched_xml(cube_half_sizes: list = None) -> str:
    """
    Duplicates object0's <body> twice (object1 green, object2 blue) into a
    sibling copy of pick_and_place.xml, written in the SAME directory as
    the original so its relative <include>/compiler paths keep resolving.
    cube_half_sizes (default EQUAL_HALF_SIZES): the [object0, object1,
    object2] half-extent for each cube's geom — object0 itself is never
    rewritten (stays whatever pick_and_place.xml's own native block is,
    0.025), only used here to size object1/object2 relative to it; pass
    TIERED_HALF_SIZES for the biggest-first pyramid variant. Returns the
    relative path (relative to the assets root, matching MODEL_XML_PATH's
    own os.path.join("fetch", ...) convention) to pass into the monkeypatch.
    """
    import os
    import gymnasium_robotics

    cube_half_sizes = cube_half_sizes or EQUAL_HALF_SIZES

    assets_dir = os.path.join(os.path.dirname(gymnasium_robotics.__file__), "envs", "assets", "fetch")
    src_path = os.path.join(assets_dir, "pick_and_place.xml")
    with open(src_path) as f:
        xml = f.read()

    import re
    match = re.search(r'<body name="object0"[^>]*>.*?</body>', xml, re.DOTALL)
    assert match, "pick_and_place.xml has no <body name=\"object0\">...</body> block — gymnasium_robotics version changed?"
    object0_block = match.group(0)

    extra_blocks = []
    for i in (1, 2):
        rgba = CUBE_COLORS[i]
        hs = cube_half_sizes[i]
        site_size = min(0.02, hs * 0.8)   # keep the little marker sphere inside the (possibly smaller) cube
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


def _make_stack_env(max_episode_steps: int, render_mode: str = None, cube_half_sizes: list = None):
    import gymnasium as gym
    import gymnasium_robotics.envs.fetch.pick_and_place as pap_module
    from think_then_act.env.setup import setup_env

    patched_rel_path = _write_patched_xml(cube_half_sizes)
    pap_module.MODEL_XML_PATH = patched_rel_path

    kwargs = dict(max_episode_steps=max_episode_steps)
    if render_mode:
        kwargs["render_mode"] = render_mode
    env = gym.make("FetchPickAndPlace-v3", **kwargs)
    setup_env(env)
    return env


def _object_observation(env, site_name: str, desired_goal):
    """
    Builds the SAME 25-dim raw observation vector shape fetch_env.py's own
    _get_obs returns, but with object_rot/object_velp/object_velr (and
    achieved_goal) computed from `site_name` instead of hardcoded
    "object0" — see module docstring for why the native obs can't be used
    as-is for cubes 1/2. object_pos/object_rel_pos (obs[3:9]) are left
    zero: build_flat_observation never reads them (it recomputes the
    equivalent relative features fresh from achieved_goal/grip_pos
    instead — see subgoal_features.py's _relative_geometry_features).
    """
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


@app.function(image=rl_image, gpu=None, cpu=2.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=600)
def eval_stack3(
    ckpt_path: str = "checkpoints/flat_task_ppo_v3_best.pt",
    policy_type: str = "mse",
    max_steps_per_cube: int = 100,
    n_trials: int = 10,
    seed_start: int = 0,
    lift_threshold: float = 0.02,
    distance_threshold: float = 0.05,
    min_separation: float = 0.09,
    render_first_trial: bool = True,
    size_variant: str = "equal",   # "equal" (original, all 3 cubes identical) or
                                  # "tiered" (biggest-first pyramid — see TIERED_HALF_SIZES'
                                  # own comment for why this was added)
    randomize_pose: bool = False,   # 2026-10-03: this eval NEVER randomized the starting
                                  # arm pose in any run before this flag was added — every
                                  # trial always started from the native fixed (fully-
                                  # extended/"stretched") default, confirmed via direct code
                                  # inspection after a user caught it from watching a video
                                  # ("the behavior is worse before it touches any cube").
                                  # That means every "cube 0" number reported so far was
                                  # never actually comparable to this checkpoint's 76.7%
                                  # POSE-RANDOMIZED eval number — the right baseline was
                                  # always the ~36.7% FIXED-pose number. Defaults to False
                                  # (preserves exact prior behavior for reproducibility);
                                  # pass True to test whether a randomized start (matching
                                  # what this checkpoint was actually mostly trained on)
                                  # does better in this multi-cube scene than the fixed
                                  # stretched one, which also has the longest possible
                                  # traverse across a now-more-cluttered table.
    pose_exclude_band: float = 0.35,
    pose_max_frac: float = 0.85,
    video_out_name: str = "stack3_trial0",
) -> dict:
    import os
    import numpy as np

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import teleport_block, grip_contact_forces, save_video, randomize_joint_angles
    from think_then_act.training.subgoal_features import build_flat_observation
    from think_then_act.training.flat_bc_multi_head import MultiHeadBCConfig, MultiHeadBCTrainer
    from think_then_act.training.subgoal_features import FLAT_OBS_DIM

    cube_half_sizes = TIERED_HALF_SIZES if size_variant == "tiered" else EQUAL_HALF_SIZES
    cube_resting_z = [TABLE_TOP_Z + hs for hs in cube_half_sizes]   # each cube's own
                                  # resting-CENTER height on the bare table, same
                                  # table_top + half_height convention as BLOCK_RESTING_Z
                                  # (which only matches cube_resting_z[0] when hs=0.025)
    cube_full_heights = [2 * hs for hs in cube_half_sizes]
    # Target z for cube i stacked on top of cubes 0..i-1: table top + the stack's
    # accumulated height so far + this cube's own half-height (centered on top).
    cube_target_z = [
        TABLE_TOP_Z + sum(cube_full_heights[:i]) + cube_half_sizes[i]
        for i in range(3)
    ]
    print(f"size_variant={size_variant}  half_sizes={cube_half_sizes}  "
          f"resting_z={cube_resting_z}  target_z={cube_target_z}", flush=True)

    cfg = MultiHeadBCConfig(obs_dim=FLAT_OBS_DIM, policy_type=policy_type)
    trainer = MultiHeadBCTrainer(cfg)
    trainer.load_checkpoint(os.path.join(MODEL_CACHE_DIR, ckpt_path))
    print(f"loaded {ckpt_path}", flush=True)

    trial_results = []
    for trial in range(n_trials):
        seed = seed_start + trial
        rng = np.random.default_rng(seed)
        render_this_trial = render_first_trial and trial == 0
        env = _make_stack_env(max_steps_per_cube * 3 + 10, render_mode="rgb_array" if render_this_trial else None,
                               cube_half_sizes=cube_half_sizes)
        env.reset(seed=seed)
        if randomize_pose:
            _, pose_ok = randomize_joint_angles(env, rng, exclude_band=pose_exclude_band, max_frac=pose_max_frac)
            if not pose_ok:
                print(f"  trial={trial}  randomize_joint_angles exhausted its retries — skipping", flush=True)
                env.close()
                continue

        # Sample 3 cube start XY positions, pairwise-separated AND separated
        # from the fixed stack point — same disk sample_disk/init_random_
        # episode uses, with a rejection loop for minimum separation (same
        # bounded-retry convention used throughout this project).
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
                points.append(candidate)   # best-effort if 50 retries all failed
        cube_xy = points[1:]

        for i in range(3):
            teleport_block(env, [cube_xy[i][0], cube_xy[i][1], cube_resting_z[i]], joint_name=f"object{i}:joint")
        obs_dict, _, terminated, truncated, _ = env.step([0.0, 0.0, 0.0, 0.0])

        frames = [env.render()] if render_this_trial else None
        per_cube = []
        for i in range(3):
            site_name = f"object{i}"
            target_z = cube_target_z[i]
            desired_goal = np.array([STACK_XY[0], STACK_XY[1], target_z])
            env.unwrapped.goal = desired_goal.copy()   # purely cosmetic: repositions the red
                                  # target-site marker for rendering; success is computed
                                  # manually below, not via the native is_success.

            observation, achieved_goal, desired = _object_observation(env, site_name, desired_goal)
            hidden_state = None
            success = False
            ever_lifted_and_gripped = False
            for t in range(max_steps_per_cube):
                flat_obs = build_flat_observation(observation, achieved_goal, desired)
                action, hidden_state = trainer.actor.act(flat_obs, hidden_state, deterministic=True)
                env.step(action)
                if frames is not None:
                    frames.append(env.render())

                observation, achieved_goal, desired = _object_observation(env, site_name, desired_goal)
                forces = grip_contact_forces(env, block_body_name=f"object{i}")
                height_above_resting = float(achieved_goal[2]) - cube_resting_z[i]
                if min(forces["left"], forces["right"]) > 0.0 and height_above_resting > lift_threshold:
                    ever_lifted_and_gripped = True
                d = float(np.linalg.norm(desired - achieved_goal))
                if d <= distance_threshold:
                    success = True

            genuine = success and ever_lifted_and_gripped
            final_pos = achieved_goal.copy()
            per_cube.append({
                "cube": i, "genuine": genuine, "success": success,
                "ever_lifted_and_gripped": ever_lifted_and_gripped,
                "final_pos": [round(float(x), 4) for x in final_pos],
                "target": [round(float(x), 4) for x in desired_goal],
                "final_dist_to_target": round(float(np.linalg.norm(desired_goal - final_pos)), 4),
            })
            print(f"  trial={trial} cube={i} target_z={target_z:.3f} genuine={genuine} "
                  f"final_pos={per_cube[-1]['final_pos']} dist={per_cube[-1]['final_dist_to_target']}", flush=True)

        all_three = all(c["genuine"] for c in per_cube)
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
    per_layer_genuine = [sum(1 for r in trial_results if r["per_cube"][i]["genuine"]) for i in range(3)]
    print(f"\n=== Summary over {n_trials} trials ===", flush=True)
    print(f"  all 3 genuinely stacked: {n_all_three}/{n_trials}", flush=True)
    print(f"  per-layer genuine rate: layer0={per_layer_genuine[0]}/{n_trials}  "
          f"layer1={per_layer_genuine[1]}/{n_trials}  layer2={per_layer_genuine[2]}/{n_trials}", flush=True)

    return {"n_trials": n_trials, "n_all_three_stacked": n_all_three,
            "per_layer_genuine": per_layer_genuine, "trial_results": trial_results}


@app.local_entrypoint()
def main(
    ckpt_path: str = "checkpoints/flat_task_ppo_v3_best.pt",
    n_trials: int = 10,
    size_variant: str = "equal",
    randomize_pose: bool = False,
    video_out_name: str = "stack3_trial0",
):
    result = eval_stack3.remote(ckpt_path=ckpt_path, n_trials=n_trials,
                                 size_variant=size_variant, randomize_pose=randomize_pose,
                                 video_out_name=video_out_name)
    print("\n", result)
