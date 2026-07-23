"""
think_then_act.env.domain_randomization

Staged visual domain randomization for the FetchPickAndPlace-v3 MjModel,
introduced incrementally (see the perception-module plan / project memory)
rather than all at once — each level composes on top of the previous one,
gated behind `level` so a data-collection run can enable exactly as much
randomization as has already been validated not to break pose-estimation
accuracy.

Mutates the already-loaded MjModel's writable arrays directly — no new
XML/texture assets needed, MuJoCo exposes color/lighting/camera as plain
numpy arrays on the model. Only VISUAL properties are touched (geom_rgba,
light_*, cam_*) — nothing here affects physics, so mj_forward() isn't
needed after calling this.

Levels (each includes all lower levels):
    0 — off (default): today's fixed appearance, unchanged.
    1 — color jitter: block/table/robot geom_rgba.
    2 — + lighting: light_ambient/light_diffuse/light_pos.
    3 — + camera pose: cam_pos/cam_quat, small jitter around the default
        camera — the one most relevant to real-camera transfer later (a
        real mount is never pixel-identical to sim's), so it's staged
        last/most cautiously.

Randomizes FROM each model's ORIGINAL (first-seen) values every call, not
from whatever the previous call left behind — the same live MjModel object
is reused across many episodes within one collection run (see
scripts/collect_pose_data.py), so re-perturbing an already-perturbed value
every episode would let the appearance drift unboundedly over a long run
instead of sampling the same fixed distribution each time.

Needs mujoco — not installed locally, so (like env/setup.py) this module is
integration-tested only, via `modal run`.
"""

from __future__ import annotations

import numpy as np

_ORIGINAL_STATE: dict = {}  # id(model) -> {"geom_rgba": ..., "light_ambient": ..., ...}

TABLE_BODY_NAME    = "table0"
BLOCK_BODY_NAME     = "object0"
ROBOT_BODY_PREFIX  = "robot0:"


def _snapshot(model) -> dict:
    key = id(model)
    if key not in _ORIGINAL_STATE:
        _ORIGINAL_STATE[key] = {
            "geom_rgba"    : model.geom_rgba.copy(),
            "light_ambient": model.light_ambient.copy() if model.nlight > 0 else None,
            "light_diffuse": model.light_diffuse.copy() if model.nlight > 0 else None,
            "light_pos"    : model.light_pos.copy() if model.nlight > 0 else None,
            "cam_pos"      : model.cam_pos.copy() if model.ncam > 0 else None,
            "cam_quat"     : model.cam_quat.copy() if model.ncam > 0 else None,
        }
    return _ORIGINAL_STATE[key]


def _geom_ids_for_body(model, body_id: int) -> list:
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]


def _randomized_body_geom_ids(model) -> list:
    """Table + block + every robot0:* body's geoms — the visual elements a
    pose estimator actually needs to tell apart, not the world/floor."""
    import mujoco
    ids = []
    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TABLE_BODY_NAME)
    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BLOCK_BODY_NAME)
    ids += _geom_ids_for_body(model, table_id)
    ids += _geom_ids_for_body(model, block_id)
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if name.startswith(ROBOT_BODY_PREFIX):
            ids += _geom_ids_for_body(model, body_id)
    return ids


def _randomize_colors(model, rng, original: dict, strength: float = 0.25) -> None:
    base_rgba = original["geom_rgba"]
    for g in _randomized_body_geom_ids(model):
        jitter = rng.uniform(-strength, strength, size=3)
        model.geom_rgba[g, :3] = np.clip(base_rgba[g, :3] + jitter, 0.0, 1.0)
        # alpha (geom_rgba[g, 3]) left untouched — randomizing transparency
        # risks making the block/table partially see-through, a much bigger
        # visual shift than intended for this level.


def _randomize_lighting(model, rng, original: dict, strength: float = 0.3) -> None:
    if original["light_ambient"] is None:
        return
    n = model.nlight
    ambient_jitter = 1.0 + rng.uniform(-strength, strength, size=(n, 3))
    diffuse_jitter = 1.0 + rng.uniform(-strength, strength, size=(n, 3))
    model.light_ambient[:] = np.clip(original["light_ambient"] * ambient_jitter, 0.0, 1.0)
    model.light_diffuse[:] = np.clip(original["light_diffuse"] * diffuse_jitter, 0.0, 1.0)
    model.light_pos[:] = original["light_pos"] + rng.uniform(-0.15, 0.15, size=(n, 3))


def _randomize_camera(model, rng, original: dict, pos_strength: float = 0.03,
                       angle_strength_deg: float = 3.0) -> None:
    import mujoco
    if original["cam_pos"] is None:
        return
    n = model.ncam
    model.cam_pos[:] = original["cam_pos"] + rng.uniform(-pos_strength, pos_strength, size=(n, 3))

    for c in range(n):
        axis = rng.normal(size=3)
        norm = np.linalg.norm(axis)
        axis = axis / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0])
        angle = np.deg2rad(rng.uniform(-angle_strength_deg, angle_strength_deg))
        jitter_quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(jitter_quat, axis, angle)
        out_quat = np.zeros(4)
        mujoco.mju_mulQuat(out_quat, jitter_quat, original["cam_quat"][c])
        model.cam_quat[c] = out_quat


def randomize_appearance(model, rng, level: int = 0) -> None:
    """
    Apply staged visual domain randomization to `model` (an MjModel, e.g.
    env.unwrapped.model) in place. `level` composes: 0=off, 1=+color,
    2=+lighting, 3=+camera pose. rng: np.random.Generator — same seeded-rng
    convention as env/setup.py's init_random_episode, so a fixed seed
    reproduces the same appearance.
    """
    if level <= 0:
        return
    original = _snapshot(model)
    if level >= 1:
        _randomize_colors(model, rng, original)
    if level >= 2:
        _randomize_lighting(model, rng, original)
    if level >= 3:
        _randomize_camera(model, rng, original)
