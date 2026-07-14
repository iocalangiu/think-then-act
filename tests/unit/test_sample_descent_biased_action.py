"""
Unit tests for env.setup.sample_descent_biased_action — pure numpy, no
mujoco needed.
"""

import numpy as np

from think_then_act.env.setup import sample_descent_biased_action


def test_shape_and_dtype():
    rng = np.random.default_rng(0)
    action = sample_descent_biased_action(rng)
    assert action.shape == (4,)
    assert action.dtype == np.float32


def test_dx_dy_grip_stay_within_full_range():
    rng = np.random.default_rng(0)
    actions = np.stack([sample_descent_biased_action(rng) for _ in range(500)])
    for dim in (0, 1, 3):
        assert actions[:, dim].min() >= -1.0
        assert actions[:, dim].max() <= 1.0


def test_dz_is_bounded_by_dz_upper():
    rng = np.random.default_rng(0)
    dz_upper = 0.1
    actions = np.stack([sample_descent_biased_action(rng, dz_upper=dz_upper) for _ in range(500)])
    assert actions[:, 2].min() >= -1.0
    assert actions[:, 2].max() <= dz_upper


def test_dz_mean_is_more_negative_than_dx_dy_mean():
    """The whole point: dz should be measurably biased toward descent
    relative to the unbiased dimensions, not just technically bounded."""
    rng = np.random.default_rng(0)
    actions = np.stack([sample_descent_biased_action(rng) for _ in range(2000)])
    assert actions[:, 2].mean() < actions[:, 0].mean()
    assert actions[:, 2].mean() < actions[:, 1].mean()


def test_reproducible_given_same_rng_state():
    action_a = sample_descent_biased_action(np.random.default_rng(42))
    action_b = sample_descent_biased_action(np.random.default_rng(42))
    np.testing.assert_array_equal(action_a, action_b)


def test_without_grip_xy_target_xy_dxdy_is_unbiased_toward_any_direction():
    """Default (no grip_xy/target_xy) must stay exactly the old pure-random
    behavior — this is the no-args call site used before target-biasing existed."""
    rng = np.random.default_rng(0)
    actions = np.stack([sample_descent_biased_action(rng) for _ in range(2000)])
    # Mean dx, dy should each be close to 0 (unbiased uniform), not skewed
    # toward any particular direction.
    assert abs(actions[:, 0].mean()) < 0.1
    assert abs(actions[:, 1].mean()) < 0.1


def test_with_grip_xy_target_xy_dxdy_biases_toward_target_direction():
    # Explicit xy_bias_strength, decoupled from the module default (which is
    # deliberately weak, 0.15 as of 2026-07-13 — see the function's
    # docstring) — this test checks the MECHANISM biases correctly, not
    # what today's default value happens to be.
    rng = np.random.default_rng(0)
    grip_xy = np.array([1.98, 0.77])
    target_xy = np.array([1.30, 0.75])   # target is in -x direction from grip

    actions = np.stack([
        sample_descent_biased_action(rng, grip_xy=grip_xy, target_xy=target_xy, xy_bias_strength=0.6)
        for _ in range(500)
    ])
    # Biased dx should trend clearly negative (toward target), not average to ~0.
    assert actions[:, 0].mean() < -0.3


def test_xy_bias_strength_one_is_deterministic_unit_direction():
    rng = np.random.default_rng(0)
    grip_xy = np.array([0.0, 0.0])
    target_xy = np.array([1.0, 0.0])   # pure +x direction, unit distance

    action = sample_descent_biased_action(
        rng, grip_xy=grip_xy, target_xy=target_xy, xy_bias_strength=1.0
    )
    np.testing.assert_allclose(action[0], 1.0, atol=1e-6)
    np.testing.assert_allclose(action[1], 0.0, atol=1e-6)


def test_dxdy_stays_bounded_even_with_bias():
    rng = np.random.default_rng(0)
    grip_xy = np.array([1.98, 0.77])
    target_xy = np.array([1.30, 0.75])

    actions = np.stack([
        sample_descent_biased_action(rng, grip_xy=grip_xy, target_xy=target_xy)
        for _ in range(500)
    ])
    assert actions[:, 0].min() >= -1.0 and actions[:, 0].max() <= 1.0
    assert actions[:, 1].min() >= -1.0 and actions[:, 1].max() <= 1.0


def test_zero_distance_to_target_falls_back_to_zero_direction_without_error():
    rng = np.random.default_rng(0)
    same_xy = np.array([1.3, 0.75])
    action = sample_descent_biased_action(rng, grip_xy=same_xy, target_xy=same_xy)
    assert np.isfinite(action).all()
