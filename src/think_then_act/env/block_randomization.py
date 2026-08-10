"""
think_then_act.env.block_randomization

Per-episode block SIZE randomization for eventual sim2real transfer to a
UR3e handling an object of unknown/varying size (see hierarchical_
architecture memory, Stage 4). Unlike env/domain_randomization.py (visual
only — geom_rgba/light/cam, deliberately doesn't touch physics), this
mutates the block's actual geom_size, a real physics change — reward/spawn
code that used to assume ONE fixed 5cm cube (the old table_z constant,
release_open_threshold, and close_gripper's original width-based target —
already replaced by env/setup.py's grip_contact_forces for exactly this
reason) has to read the CURRENT episode's sampled size instead.

MuJoCo box geoms are analytic primitives, not baked meshes — model.geom_size
is a plain writable half-extents array, no MJCF recompilation needed to
resize one, same "mutate the already-loaded MjModel directly" approach
domain_randomization.py already uses for color/lighting/camera.

Axis mapping (MEASURED, not assumed — scripts/measure_gripper_axis.py,
2026-08-10): the gripper's fingers close purely along the LOCAL Y axis (x
and z deltas were exactly zero at every finger opening tested). The block
is always spawned axis-aligned to world (teleport_block sets an identity
qpos quaternion), so this is a fixed WORLD axis too. geom_size layout:
    index 0 (x) = length — unconstrained by the gripper
    index 1 (y) = width  — the axis finger_open's ~0.10m ceiling constrains
    index 2 (z) = height

Ordering requirement: sample_and_apply_block_size() must be called BEFORE
env/setup.py's teleport_block() each episode — it only mutates geom_size,
it does NOT call mj_forward itself (teleport_block already does, right
after, so physics picks up the new geometry for free as long as this runs
first; calling mj_forward twice per reset would be wasted work).

Mass/inertia deliberately NOT touched when resizing (user's explicit call,
2026-08-09) — a bigger block ends up unrealistically light relative to a
real object of that size. A deliberate simplification for this pass, not
an oversight.
"""

from __future__ import annotations
import numpy as np

from think_then_act.env.setup import BLOCK_BODY_NAME

WIDTH_RANGE  = (0.01, 0.08)  # metres, full extent along the gripper's closing (y) axis
LENGTH_RANGE = (0.01, 0.08)  # metres, full extent along x (unconstrained by the gripper)
HEIGHT_RANGE = (0.01, 0.08)  # metres, full extent along z

# Stands in for a real perception module's measurement noise (user's call,
# 2026-08-09: "+/- 5mm") — no trained shape estimator exists yet (unlike
# block POSITION, which already has perception/block_pose_predictor.py), so
# this is synthetic noise added directly to the ground-truth dims, not a
# learned model's output. Revisit once/if a real shape estimator is built,
# same "swap the synthetic stand-in for a trained model" progression
# position perception already went through.
PERCEIVED_DIMS_NOISE_STD = 0.005  # metres

_EPISODE_STATE: dict = {}  # id(model) -> {"width","length","height","perceived": np.ndarray(3,)}


def _block_geom_id(model) -> int:
    import mujoco
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BLOCK_BODY_NAME)
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] == body_id:
            return gid
    raise RuntimeError(f"No geom found for body {BLOCK_BODY_NAME!r}")


def sample_and_apply_block_size(
    model, rng,
    width_range: tuple = WIDTH_RANGE, length_range: tuple = LENGTH_RANGE,
    height_range: tuple = HEIGHT_RANGE, noise_std: float = PERCEIVED_DIMS_NOISE_STD,
) -> dict:
    """
    Call once per episode reset (see module docstring for the required
    ordering relative to teleport_block). Samples true width/length/height
    independently uniform at random — NOT a cube, each dimension varies on
    its own (user's call, 2026-08-09) — writes them into model.geom_size,
    and also samples a FIXED perceived reading (true + Gaussian noise) for
    this episode. The perceived reading is sampled ONCE here, not resampled
    per step — a real camera looking at an object that isn't moving
    relative to it wouldn't produce a fresh independent reading every
    control step.

    Returns {"width","length","height","perceived"} (also cached under
    id(model) — see get_block_dims/get_perceived_block_dims for reading it
    back later in the episode without re-deriving or accidentally
    re-sampling a different value).
    """
    gid = _block_geom_id(model)

    width  = float(rng.uniform(*width_range))
    length = float(rng.uniform(*length_range))
    height = float(rng.uniform(*height_range))
    model.geom_size[gid] = [length / 2.0, width / 2.0, height / 2.0]

    true_dims = np.array([width, length, height], dtype=np.float64)
    # Clipped at a small positive floor — noise on a true dim near the 0.01m
    # floor can otherwise land at/below zero, a nonsensical "perceived"
    # reading no real sensor would report.
    perceived = np.clip(true_dims + rng.normal(0.0, noise_std, size=3), 0.001, None)

    state = {"width": width, "length": length, "height": height, "perceived": perceived}
    _EPISODE_STATE[id(model)] = state
    return dict(state)


def get_block_dims(model) -> dict:
    """
    True {"width","length","height"} (metres) for the CURRENT episode —
    reads back the last sample_and_apply_block_size() call for this model.
    Falls back to reading model.geom_size directly if size randomization
    was never enabled for this model (e.g. the original fixed-cube
    geometry, or a caller that predates this module) rather than erroring
    — same None-means-unchanged convention used throughout this project.
    """
    cached = _EPISODE_STATE.get(id(model))
    if cached is not None:
        return {"width": cached["width"], "length": cached["length"], "height": cached["height"]}
    gid = _block_geom_id(model)
    length2, width2, height2 = model.geom_size[gid]
    return {"width": float(2 * width2), "length": float(2 * length2), "height": float(2 * height2)}


def get_perceived_block_dims(model) -> np.ndarray:
    """
    (3,) [width, length, height] perceived reading for the CURRENT episode
    — true dims + the SAME per-episode noise sampled once in
    sample_and_apply_block_size, not resampled here. Falls back to the
    ground-truth dims (zero noise) if no episode has been sampled yet for
    this model, matching get_block_dims' fallback.
    """
    cached = _EPISODE_STATE.get(id(model))
    if cached is not None:
        return np.asarray(cached["perceived"], dtype=np.float32)
    dims = get_block_dims(model)
    return np.array([dims["width"], dims["length"], dims["height"]], dtype=np.float32)
