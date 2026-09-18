"""
Unit tests for env.action_force_perturbation — pure numpy, no mujoco needed.
"""
import numpy as np
import pytest

from think_then_act.env.action_force_perturbation import (
    sample_singularity_perturbation, apply_singularity_perturbation,
    sample_force_perturbation, apply_force_perturbation,
    get_singularity_perturbation_state,
)


class FakeModel:
    """Standalone object so id(model) gives a stable, distinct key without
    needing a real MuJoCo model — the pattern under test (mirroring
    env/block_randomization.py) only ever keys off id(model), never its
    contents."""
    pass


def test_singularity_disabled_is_exact_passthrough_and_consumes_no_rng():
    model = FakeModel()
    rng = np.random.default_rng(0)
    state_before = rng.bit_generator.state

    result = sample_singularity_perturbation(model, rng, enable=False)
    assert result == {"enable": False}
    assert rng.bit_generator.state == state_before

    delta = np.array([0.01, -0.02, 0.03])
    out = apply_singularity_perturbation(model, step_idx=5, commanded_delta_m=delta)
    np.testing.assert_array_equal(out, delta)


def test_force_disabled_is_exact_passthrough_and_consumes_no_rng():
    model = FakeModel()
    rng = np.random.default_rng(0)
    state_before = rng.bit_generator.state

    result = sample_force_perturbation(model, rng, enable=False)
    assert result == {"enable": False}
    assert rng.bit_generator.state == state_before

    raw = {"left": 1.5, "right": 0.0}
    out = apply_force_perturbation(model, rng, raw)
    assert out == raw
    assert out is not raw   # copied, never the same dict object


def test_singularity_perturbation_only_active_within_sampled_window():
    model = FakeModel()
    rng = np.random.default_rng(1)
    state = sample_singularity_perturbation(
        model, rng, enable=True, attenuation_range=(0.1, 0.2),
        onset_step_range=(3, 4), duration_range=(2, 3),
    )
    assert state["onset_step"] == 3
    assert state["end_step"] == 5   # onset + duration

    delta = np.array([0.05, 0.0, 0.0])
    before = apply_singularity_perturbation(model, step_idx=2, commanded_delta_m=delta)
    np.testing.assert_array_equal(before, delta)   # before window: unchanged

    during = apply_singularity_perturbation(model, step_idx=3, commanded_delta_m=delta)
    assert not np.allclose(during, delta)   # inside window: attenuation_range=(0.1,0.2)
                                             # guarantees a detectable change regardless
                                             # of the off-diagonal leak draw

    after = apply_singularity_perturbation(model, step_idx=5, commanded_delta_m=delta)
    np.testing.assert_array_equal(after, delta)   # after window: unchanged


def test_singularity_perturbation_same_seed_is_deterministic():
    model_a, model_b = FakeModel(), FakeModel()
    state_a = sample_singularity_perturbation(model_a, np.random.default_rng(42), enable=True)
    state_b = sample_singularity_perturbation(model_b, np.random.default_rng(42), enable=True)
    np.testing.assert_array_equal(state_a["leak_matrix"], state_b["leak_matrix"])
    assert state_a["onset_step"] == state_b["onset_step"]
    assert state_a["end_step"] == state_b["end_step"]


def test_per_model_state_is_isolated():
    model_a, model_b = FakeModel(), FakeModel()
    sample_singularity_perturbation(model_a, np.random.default_rng(0), enable=True,
                                     attenuation_range=(0.1, 0.2),
                                     onset_step_range=(0, 1), duration_range=(100, 101))
    sample_singularity_perturbation(model_b, np.random.default_rng(0), enable=False)

    delta = np.array([1.0, 0.0, 0.0])
    out_a = apply_singularity_perturbation(model_a, step_idx=0, commanded_delta_m=delta)
    out_b = apply_singularity_perturbation(model_b, step_idx=0, commanded_delta_m=delta)
    assert not np.allclose(out_a, delta)          # model_a: perturbation active
    np.testing.assert_array_equal(out_b, delta)   # model_b: disabled, unaffected by model_a's state


def test_get_singularity_perturbation_state_reads_back_sampled_state():
    model = FakeModel()
    assert get_singularity_perturbation_state(model) is None   # never sampled yet

    sample_singularity_perturbation(model, np.random.default_rng(0), enable=True)
    state = get_singularity_perturbation_state(model)
    assert state is not None
    assert state["enable"] is True
    assert "onset_step" in state and "end_step" in state


def test_force_perturbation_scales_offsets_and_floors_at_zero():
    model = FakeModel()
    rng = np.random.default_rng(0)
    sample_force_perturbation(model, rng, enable=True, scale_range=(2.0, 2.0),
                               noise_std_range=(0.0, 0.0), offset_range=(-100.0, -100.0))
    raw = {"left": 1.0, "right": 5.0}
    out = apply_force_perturbation(model, rng, raw)
    # left: 1.0*2.0 - 100.0 = -98 -> floored at 0.0; right: 5.0*2.0 - 100.0 = -90 -> floored at 0.0
    assert out["left"] == 0.0
    assert out["right"] == 0.0


def test_force_perturbation_no_noise_is_deterministic():
    model = FakeModel()
    rng = np.random.default_rng(0)
    sample_force_perturbation(model, rng, enable=True, scale_range=(1.5, 1.5),
                               noise_std_range=(0.0, 0.0), offset_range=(0.5, 0.5))
    raw = {"left": 2.0, "right": 0.0}
    out = apply_force_perturbation(model, rng, raw)
    assert out["left"] == pytest.approx(2.0 * 1.5 + 0.5)
    assert out["right"] == pytest.approx(0.0 * 1.5 + 0.5)
