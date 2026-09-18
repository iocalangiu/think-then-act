"""
think_then_act.env.action_force_perturbation

Per-episode, opt-in perturbation of (a) commanded Cartesian displacement and
(b) contact force, standing in for two real-UR3e symptoms the 7-DOF Fetch
sim can't otherwise reproduce:

  1. Near a kinematic singularity, a commanded Cartesian delta doesn't land
     as commanded — small steps get attenuated and/or leak into other axes
     (near-singular Jacobian cross-talk). The 7-DOF sim's own kinematics
     have nothing to do with the UR3e's, so this fakes the SYMPTOM (commanded
     != achieved) directly rather than trying to reproduce the UR3e's actual
     singularity geometry in an unrelated arm.
  2. The MuJoCo Fetch gripper and whatever gripper is physically on the UR3e
     are unrelated geometries, so raw contact-force magnitude won't transfer
     — this randomizes force scale/noise/offset per episode so a policy
     conditioned on force learns to key off onset/relative change, not a
     magnitude that's meaningless outside this one gripper model.

Same pattern as env/block_randomization.py: plain functions taking
(model, rng, **range kwargs), called once per episode, opt-in via
enable=False (old/unperturbed behavior — see the RNG-contamination note
below), per-episode state cached in a module-level dict keyed by id(model).

RNG-contamination note: enable=False is a genuine early return BEFORE any
rng draw. If it drew from rng and then discarded the draw, toggling
perturbation on/off would silently change the random sequence everything
ELSE that episode consumes (block/target position, pose noise, ...),
breaking reproducibility between ablation cells that are supposed to differ
ONLY in the perturbation flag. See tests/unit/test_action_force_perturbation.py.
"""

from __future__ import annotations
import numpy as np

_EPISODE_STATE: dict = {}  # id(model) -> {"singularity": {...}, "force": {...}}


def sample_singularity_perturbation(
    model, rng,
    enable: bool = False,
    attenuation_range: tuple = (0.3, 1.0),
    leak_range: tuple = (-0.3, 0.3),
    onset_step_range: tuple = (0, 15),
    duration_range: tuple = (5, 30),
) -> dict:
    """
    Call once per episode, before the first apply_singularity_perturbation()
    call this episode. Samples a 3x3 "leak matrix" active for a randomized
    step window within the episode: diagonal entries are per-axis
    attenuation (a near-singular Jacobian shrinking the achievable motion
    along that axis), off-diagonal entries are cross-axis coupling (motion
    commanded on one axis leaking into another, the way a near-singular
    Jacobian couples joint velocities across Cartesian axes).

    enable=False (default) returns {"enable": False} without touching rng
    at all — see module docstring's RNG-contamination note.

    Returns the sampled state dict (also cached under id(model)).
    """
    if not enable:
        state = {"enable": False}
        _EPISODE_STATE.setdefault(id(model), {})["singularity"] = state
        return dict(state)

    scale = rng.uniform(*attenuation_range, size=3)
    leak = rng.uniform(*leak_range, size=(3, 3))
    np.fill_diagonal(leak, scale)
    onset_step = int(rng.integers(*onset_step_range))
    duration = int(rng.integers(*duration_range))

    state = {
        "enable": True,
        "leak_matrix": leak,
        "onset_step": onset_step,
        "end_step": onset_step + duration,
    }
    _EPISODE_STATE.setdefault(id(model), {})["singularity"] = state
    return dict(state)


def apply_singularity_perturbation(model, step_idx: int, commanded_delta_m: np.ndarray) -> np.ndarray:
    """
    Returns `leak_matrix @ commanded_delta_m` while the sampled perturbation
    window is active this episode, otherwise `commanded_delta_m` unchanged
    (including when no perturbation was ever sampled for this model, or
    enable=False was used — same None/False-means-unchanged convention as
    block_randomization.py).
    """
    state = _EPISODE_STATE.get(id(model), {}).get("singularity")
    if state is None or not state.get("enable", False):
        return commanded_delta_m
    if not (state["onset_step"] <= step_idx < state["end_step"]):
        return commanded_delta_m
    return state["leak_matrix"] @ np.asarray(commanded_delta_m, dtype=np.float64)


def sample_force_perturbation(
    model, rng,
    enable: bool = False,
    scale_range: tuple = (0.5, 2.0),
    noise_std_range: tuple = (0.0, 0.5),
    offset_range: tuple = (-0.2, 0.2),
) -> dict:
    """
    Call once per episode. Samples a per-episode (scale, noise_std, offset)
    applied to grip_contact_forces()'s raw Newton readings — standing in for
    the real UR3e gripper's unmodeled, geometrically-unrelated force
    response (see module docstring). Same enable=False early-return-before-
    any-rng-draw contract as sample_singularity_perturbation.
    """
    if not enable:
        state = {"enable": False}
        _EPISODE_STATE.setdefault(id(model), {})["force"] = state
        return dict(state)

    scale = float(rng.uniform(*scale_range))
    noise_std = float(rng.uniform(*noise_std_range))
    offset = float(rng.uniform(*offset_range))

    state = {"enable": True, "scale": scale, "noise_std": noise_std, "offset": offset}
    _EPISODE_STATE.setdefault(id(model), {})["force"] = state
    return dict(state)


def apply_force_perturbation(model, rng, raw_forces: dict) -> dict:
    """
    raw_forces: {"left": float, "right": float} from grip_contact_forces().
    Returns a NEW dict (never mutates raw_forces) with this episode's
    scale/offset/noise applied: `max(0.0, raw * scale + offset + noise)`
    per finger, floored at 0.0 since a contact-normal force reading below
    zero is nonsensical. rng may be None — an episode's scale/noise_std/
    offset are already fixed at sample time; noise is still drawn fresh per
    call using rng if given, otherwise np.random's global state (only
    relevant if noise_std > 0 was sampled). Passthrough (raw_forces
    unchanged, but still copied) if perturbation was never enabled this
    episode.
    """
    state = _EPISODE_STATE.get(id(model), {}).get("force")
    if state is None or not state.get("enable", False):
        return dict(raw_forces)

    scale, noise_std, offset = state["scale"], state["noise_std"], state["offset"]
    draw = rng.normal if rng is not None else np.random.normal
    out = {}
    for side, value in raw_forces.items():
        noise = draw(0.0, noise_std) if noise_std > 0 else 0.0
        out[side] = max(0.0, value * scale + offset + noise)
    return out
