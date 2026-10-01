"""
Unit tests for training.safe_regime — pure functions, no mujoco needed.
"""
from think_then_act.training.safe_regime import is_safe_regime, SafeRegimeGate


def test_is_safe_regime_true_when_close_and_no_force_onset():
    assert is_safe_regime(d_xy=0.005, d_z=0.01, force_onset_flag=0.0, d_xy_limit=0.01, d_z_limit=0.02)


def test_is_safe_regime_false_when_too_far_laterally():
    assert not is_safe_regime(d_xy=0.02, d_z=0.01, force_onset_flag=0.0, d_xy_limit=0.01, d_z_limit=0.02)


def test_is_safe_regime_false_when_too_far_vertically():
    assert not is_safe_regime(d_xy=0.005, d_z=0.05, force_onset_flag=0.0, d_xy_limit=0.01, d_z_limit=0.02)


def test_is_safe_regime_false_when_force_onset_fired():
    assert not is_safe_regime(d_xy=0.005, d_z=0.01, force_onset_flag=1.0, d_xy_limit=0.01, d_z_limit=0.02)


def test_is_safe_regime_uses_absolute_value_for_d_z():
    # d_z is a signed gripper-to-block gap (subgoal_reward.py: grip_pos[2] - block_pos[2]);
    # a negative value that's small in magnitude should still count as close.
    assert is_safe_regime(d_xy=0.005, d_z=-0.01, force_onset_flag=0.0, d_xy_limit=0.01, d_z_limit=0.02)


def test_gate_requires_consecutive_steps_not_just_one():
    gate = SafeRegimeGate(streak_required=3, d_xy_limit=0.01, d_z_limit=0.02)
    assert gate.update(0.005, 0.01, 0.0) is False   # streak=1
    assert gate.update(0.005, 0.01, 0.0) is False   # streak=2
    assert gate.update(0.005, 0.01, 0.0) is True    # streak=3 -> fires


def test_gate_resets_streak_on_any_failing_step():
    gate = SafeRegimeGate(streak_required=3, d_xy_limit=0.01, d_z_limit=0.02)
    assert gate.update(0.005, 0.01, 0.0) is False
    assert gate.update(0.005, 0.01, 0.0) is False
    assert gate.update(0.02, 0.01, 0.0) is False    # fails -> streak resets to 0
    assert gate.update(0.005, 0.01, 0.0) is False   # streak=1 again, not 3
    assert gate.update(0.005, 0.01, 0.0) is False   # streak=2
    assert gate.update(0.005, 0.01, 0.0) is True    # streak=3


def test_gate_reset_method_clears_streak():
    gate = SafeRegimeGate(streak_required=2)
    gate.update(0.001, 0.001, 0.0)
    gate.reset()
    assert gate.update(0.001, 0.001, 0.0) is False   # streak=1, not 2 — reset() worked


def test_gate_streak_required_one_fires_immediately():
    gate = SafeRegimeGate(streak_required=1, d_xy_limit=0.01, d_z_limit=0.02)
    assert gate.update(0.005, 0.01, 0.0) is True
