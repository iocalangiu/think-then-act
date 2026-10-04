"""
think_then_act.env.multicube

Shared infrastructure for running FetchPickAndPlace-v3 with N>=1 EXTRA
movable cubes physically present in the scene simultaneously, tracking
each in turn as "the" active object. Factored out of scripts/eval_stack3_
cubes.py / collect_stack3_demos.py (which each independently built this
for a FIXED 2-extra-cubes/3-total case) once a THIRD consumer --
training/multicube_stack_env.py's PPO environment -- needed the SAME
machinery generalized to a variable cube count. See those two scripts'
own docstrings for the original discovery/validation history (byte-
identical observation reconstruction confirmed against the native
single-cube env's own _get_obs, 2026-10-03).

Why the cube COUNT is fixed at env-construction time, not randomized per
episode: the physical MuJoCo model (number of bodies/geoms) is baked into
the MJCF XML at gym.make() time -- rebuilding it every episode would mean
recompiling the MuJoCo model every reset(), expensive for a PPO rollout
worker that resets constantly. Instead, build the env ONCE with max_cubes
extra bodies, and let each episode choose how many are ACTIVE (teleport
the rest to a dormant parking spot, see park_unused_cubes) -- the
training/multicube_stack_env.py env is what actually does this
per-episode choice; this module just gives it the building blocks.
"""

from __future__ import annotations

import os
import re

import numpy as np

TABLE_TOP_Z = 0.400
# object0 (index 0) keeps its native XML size always -- never rewritten,
# this entry exists purely so every other list in this module stays
# index-aligned with "cube 0, 1, 2, ...".
DEFAULT_CUBE_HALF_SIZES = [0.025, 0.025, 0.021, 0.018, 0.015]
CUBE_COLORS = [None, "0.15 0.75 0.15 1", "0.15 0.35 0.85 1", "0.85 0.70 0.15 1", "0.65 0.15 0.75 1"]
PARK_XY = (2.00, 2.00)   # far outside the table/workspace -- see park_unused_cubes


def write_patched_xml(n_extra_cubes: int, cube_half_sizes: list) -> str:
    """
    cube_half_sizes: length n_extra_cubes+1, index 0 is object0's own
    (native, informational only -- its geom is never rewritten here).
    Returns the relative path (fetch/...) to pass into the
    MODEL_XML_PATH monkeypatch, matching that constant's own convention.
    """
    import gymnasium_robotics

    assert n_extra_cubes >= 0
    assert len(cube_half_sizes) == n_extra_cubes + 1
    assert n_extra_cubes <= len(CUBE_COLORS) - 1, f"only {len(CUBE_COLORS) - 1} extra-cube colors defined"

    assets_dir = os.path.join(os.path.dirname(gymnasium_robotics.__file__), "envs", "assets", "fetch")
    src_path = os.path.join(assets_dir, "pick_and_place.xml")
    with open(src_path) as f:
        xml = f.read()
    match = re.search(r'<body name="object0"[^>]*>.*?</body>', xml, re.DOTALL)
    assert match, 'pick_and_place.xml has no <body name="object0">...</body> block -- gymnasium_robotics version changed?'
    object0_block = match.group(0)

    extra_blocks = []
    for i in range(1, n_extra_cubes + 1):
        hs = cube_half_sizes[i]
        rgba = CUBE_COLORS[i]
        site_size = min(0.02, hs * 0.8)
        extra_blocks.append(f'''<body name="object{i}" pos="{0.025 + 0.1 * i} 0.125 0.025">
                        <joint name="object{i}:joint" type="free" damping="0.01"></joint>
                        <geom size="{hs} {hs} {hs}" type="box" condim="3" name="object{i}" rgba="{rgba}" mass="2"></geom>
                        <site name="object{i}" pos="0 0 0" size="{site_size} {site_size} {site_size}" rgba="{rgba}" type="sphere"></site>
                </body>''')
    if extra_blocks:
        patched = xml.replace(
            object0_block, object0_block + "\n                " + "\n                ".join(extra_blocks), 1,
        )
    else:
        patched = xml
    out_path = os.path.join(assets_dir, f"pick_and_place_multicube_{n_extra_cubes}.xml")
    with open(out_path, "w") as f:
        f.write(patched)
    return os.path.join("fetch", f"pick_and_place_multicube_{n_extra_cubes}.xml")


def make_multicube_env(n_extra_cubes: int, cube_half_sizes: list, max_episode_steps: int, render_mode: str = None):
    import gymnasium as gym
    import gymnasium_robotics.envs.fetch.pick_and_place as pap_module
    from think_then_act.env.setup import setup_env

    pap_module.MODEL_XML_PATH = write_patched_xml(n_extra_cubes, cube_half_sizes)
    kwargs = dict(max_episode_steps=max_episode_steps)
    if render_mode:
        kwargs["render_mode"] = render_mode
    env = gym.make("FetchPickAndPlace-v3", **kwargs)
    setup_env(env)
    return env


def object_observation(env, site_name: str, desired_goal):
    """
    Same 25-dim-shaped raw observation fetch_env.py's own _get_obs
    returns, but object_rot/object_velp/object_velr/achieved_goal
    computed from `site_name` instead of hardcoded "object0" -- see
    module docstring. object_pos/object_rel_pos (obs[3:9]) left zero --
    build_flat_observation never reads them.
    """
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


def get_object_xyz(env, site_name: str) -> np.ndarray:
    """Just the position -- cheaper than object_observation when that's all that's needed
    (e.g. checking whether an already-placed cube has been disturbed)."""
    raw = env.unwrapped
    return raw._utils.get_site_xpos(raw.model, raw.data, site_name).copy()


def teleport_cube(env, site_name: str, xyz, joint_name: str = None) -> None:
    from think_then_act.env.setup import teleport_block
    teleport_block(env, xyz, joint_name=joint_name or f"{site_name}:joint")


def park_unused_cubes(env, active_indices: list, max_cubes: int) -> None:
    """
    Teleports every cube index in range(0, max_cubes+1) NOT in
    active_indices to a far-away, out-of-workspace spot (PARK_XY, well
    clear of the table and the robot's reach), spaced out along x so
    multiple parked cubes don't overlap each other. Index 0 (object0) IS
    included here -- it's always physically present in the MJCF (can't be
    excluded from the model the way extras can), but it still needs
    parking like any other cube when this episode's random active_indices
    doesn't happen to include it, or it'd sit on the table as a stray
    physical obstacle from wherever its previous episode left it.
    """
    object_names = [f"object{i}" for i in range(0, max_cubes + 1)]
    parked_i = 0
    for i in range(0, max_cubes + 1):
        if i in active_indices:
            continue
        teleport_cube(env, object_names[i], [PARK_XY[0] + 0.1 * parked_i, PARK_XY[1], TABLE_TOP_Z + 0.025])
        parked_i += 1
