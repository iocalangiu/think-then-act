"""
Unit tests for training.chained_teacher_rollout.compute_episode_weight —
pure function over already-collected episode dicts, no mujoco/gymnasium
needed (the module itself has no such import at load time; only
run_chained_align_descend_episode/run_recurrent_student_episode need a
live env, at CALL time, not exercised here).
"""
import numpy as np
import pytest

from think_then_act.training.chained_teacher_rollout import compute_episode_weight


def _episode(teacher_action: np.ndarray, switched_at_step) -> dict:
    return {
        "teacher_action": teacher_action.astype(np.float32),
        "switched_at_step": switched_at_step,
        "n_steps": len(teacher_action),
    }


def test_weight_scales_with_dim1_magnitude_during_align_xy_phase():
    small = _episode(np.array([[0.0, 0.05, 0.0, 0.0]] * 10), switched_at_step=10)
    large = _episode(np.array([[0.0, 0.6, 0.0, 0.0]] * 10), switched_at_step=10)
    assert compute_episode_weight(large) > compute_episode_weight(small)


def test_weight_only_looks_at_the_align_xy_portion_not_descend():
    # dim-1 is huge AFTER the handoff (descend phase) but small during
    # align_xy — weight should reflect only the align_xy portion.
    action = np.zeros((10, 4), dtype=np.float32)
    action[:4, 1] = 0.02      # align_xy phase: steps 0-3
    action[4:, 1] = 0.9       # descend phase: steps 4-9 — must be ignored
    episode = _episode(action, switched_at_step=4)
    weight = compute_episode_weight(episode)
    assert weight < 0.1   # reflects the small align_xy-phase values, not the large descend ones


def test_weight_floored_for_near_zero_correction_episodes():
    tiny = _episode(np.array([[0.0, 0.0001, 0.0, 0.0]] * 5), switched_at_step=5)
    weight = compute_episode_weight(tiny, floor=0.05)
    assert weight == 0.05


def test_weight_uses_full_trajectory_when_never_switched():
    # switched_at_step=None (align_xy itself never finished) -> weight
    # should still be computable from the whole trajectory, not crash.
    action = np.array([[0.0, 0.3, 0.0, 0.0]] * 6, dtype=np.float32)
    episode = _episode(action, switched_at_step=None)
    weight = compute_episode_weight(episode)
    assert weight == pytest.approx(0.3, abs=1e-4)


def test_weight_respects_custom_weight_dim():
    action = np.zeros((5, 4), dtype=np.float32)
    action[:, 0] = 0.9   # dim 0, not the default weight_dim=1
    episode = _episode(action, switched_at_step=5)
    assert compute_episode_weight(episode, weight_dim=0) > compute_episode_weight(episode, weight_dim=1)
