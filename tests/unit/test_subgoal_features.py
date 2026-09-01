"""
Unit tests for training.subgoal_features — pure numpy, no gymnasium needed.
"""

import numpy as np
import pytest

from think_then_act.reward.subgoal_reward import SUBGOAL_LABELS
from think_then_act.training.subgoal_features import (
    CLOSE_GRIPPER_OBS_DIM,
    RELATIVE_OBS_DIM,
    RELATIVE_OBS_SUBGOALS,
    SUBGOAL_OBS_DIM,
    build_subgoal_observation,
    sanitize_observation_for_perception,
    subgoal_to_onehot,
)


def test_subgoal_obs_dim_matches_the_actual_concatenation():
    # "lift" here (not "align_xy"/"descend") — both of those are now in
    # RELATIVE_OBS_SUBGOALS with their own SMALLER layout, see the
    # RELATIVE_OBS_SUBGOALS-specific tests below.
    obs   = np.zeros(25)
    block = [1.3, 0.75, 0.425]
    target = [1.5, 0.75, 0.425]
    flat = build_subgoal_observation(obs, block, target, "lift", collision_prob=0.0)
    assert flat.shape == (SUBGOAL_OBS_DIM,)
    assert flat.dtype == np.float32


@pytest.mark.parametrize("subgoal", sorted(RELATIVE_OBS_SUBGOALS))
def test_relative_obs_subgoal_observation_is_the_smaller_relative_layout(subgoal):
    obs   = np.arange(25, dtype=np.float32)
    block = [1.3, 0.75, 0.425]
    target = [1.5, 0.75, 0.425]

    flat = build_subgoal_observation(obs, block, target, subgoal, collision_prob=0.0)

    assert flat.shape == (RELATIVE_OBS_DIM,)
    assert RELATIVE_OBS_DIM == SUBGOAL_OBS_DIM - 9
    assert flat.dtype == np.float32


@pytest.mark.parametrize("subgoal", sorted(RELATIVE_OBS_SUBGOALS))
def test_relative_obs_subgoal_observation_is_invariant_to_a_global_translation(subgoal):
    # The whole point of this layout: translate grip_pos, block_pos, and
    # target by the SAME arbitrary vector (simulating a different
    # coordinate frame's origin, e.g. sim world-frame vs real base_link
    # frame) and the fed observation must come out byte-identical — see
    # RELATIVE_OBS_SUBGOALS's comment in subgoal_features.py for why.
    rng = np.random.default_rng(0)
    obs = rng.normal(size=25).astype(np.float32)
    block  = np.array([1.3, 0.75, 0.425], dtype=np.float32)
    target = np.array([1.5, 0.75, 0.425], dtype=np.float32)
    shift = np.array([-1.9, 0.4, -0.6], dtype=np.float32)   # arbitrary, nonzero on every axis

    obs_shifted = obs.copy()
    obs_shifted[0:3] += shift   # grip_pos
    obs_shifted[3:6] += shift   # object_pos (unused by this layout, but shift it anyway)

    flat          = build_subgoal_observation(obs,         block,         target,         subgoal, collision_prob=0.3)
    flat_shifted  = build_subgoal_observation(obs_shifted,  block + shift, target + shift, subgoal, collision_prob=0.3)

    np.testing.assert_allclose(flat, flat_shifted, atol=1e-5)


@pytest.mark.parametrize("subgoal", sorted(RELATIVE_OBS_SUBGOALS))
def test_relative_obs_subgoal_observation_has_no_close_gripper_block_dims_leak(subgoal):
    # block_dims is close_gripper-only — passing it here must be a silent
    # no-op, same guarantee test_other_subgoals_ignore_block_dims_even_if_
    # passed already gives the other non-RELATIVE_OBS_SUBGOALS subgoals.
    obs   = np.zeros(25)
    block = [1.3, 0.75, 0.425]
    target = [1.5, 0.75, 0.425]

    without = build_subgoal_observation(obs, block, target, subgoal, collision_prob=0.0)
    with_dims = build_subgoal_observation(obs, block, target, subgoal, collision_prob=0.0,
                                           block_dims=[0.03, 0.05, 0.04])

    assert without.shape == (RELATIVE_OBS_DIM,)
    np.testing.assert_array_equal(without, with_dims)


def test_close_gripper_observation_is_wider_and_appends_block_dims():
    obs   = np.zeros(25)
    block = [1.3, 0.75, 0.425]
    target = [1.5, 0.75, 0.425]
    dims = [0.03, 0.05, 0.04]   # [width, length, height]

    flat = build_subgoal_observation(obs, block, target, "close_gripper", collision_prob=0.0,
                                      block_dims=dims)

    assert flat.shape == (CLOSE_GRIPPER_OBS_DIM,)
    assert CLOSE_GRIPPER_OBS_DIM == SUBGOAL_OBS_DIM + 3
    np.testing.assert_allclose(flat[-3:], dims)


def test_close_gripper_observation_defaults_block_dims_to_zero():
    obs   = np.zeros(25)
    block = [1.3, 0.75, 0.425]
    target = [1.5, 0.75, 0.425]

    flat = build_subgoal_observation(obs, block, target, "close_gripper", collision_prob=0.0)

    assert flat.shape == (CLOSE_GRIPPER_OBS_DIM,)
    np.testing.assert_allclose(flat[-3:], [0.0, 0.0, 0.0])


def test_other_subgoals_ignore_block_dims_even_if_passed():
    # Passing block_dims for a subgoal that isn't close_gripper must be a
    # silent no-op — the vector length/content must be byte-identical to
    # not passing it at all, so the other 5 subgoals' existing checkpoints
    # never see an unexpected shape change.
    obs   = np.zeros(25)
    block = [1.3, 0.75, 0.425]
    target = [1.5, 0.75, 0.425]

    without = build_subgoal_observation(obs, block, target, "lift", collision_prob=0.0)
    with_dims = build_subgoal_observation(obs, block, target, "lift", collision_prob=0.0,
                                           block_dims=[0.03, 0.05, 0.04])

    assert without.shape == (SUBGOAL_OBS_DIM,)
    np.testing.assert_array_equal(without, with_dims)


def test_onehot_is_one_hot_and_matches_label_index():
    for i, label in enumerate(SUBGOAL_LABELS):
        onehot = subgoal_to_onehot(label)
        assert onehot.shape == (len(SUBGOAL_LABELS),)
        assert onehot.sum() == 1.0
        assert onehot[i] == 1.0


def test_onehot_rejects_unknown_subgoal():
    with pytest.raises(ValueError):
        subgoal_to_onehot("not_a_real_subgoal")


def test_build_observation_rejects_unknown_subgoal():
    obs = np.zeros(25)
    with pytest.raises(ValueError):
        build_subgoal_observation(obs, [0, 0, 0], [0, 0, 0], "not_a_real_subgoal", 0.0)


def test_build_observation_encodes_collision_prob_and_subgoal_correctly():
    # "lift" here, not "descend" — descend is now in RELATIVE_OBS_SUBGOALS.
    obs = np.arange(25, dtype=np.float32)
    block = [1.0, 2.0, 3.0]
    target = [4.0, 5.0, 6.0]
    flat = build_subgoal_observation(obs, block, target, "lift", collision_prob=0.42)

    # Layout: obs(25) | achieved(3) | desired(3) | onehot(6) | collision_prob(1)
    np.testing.assert_allclose(flat[0:25], obs)
    np.testing.assert_allclose(flat[25:28], block)
    np.testing.assert_allclose(flat[28:31], target)
    onehot = flat[31:31 + len(SUBGOAL_LABELS)]
    np.testing.assert_allclose(onehot, subgoal_to_onehot("lift"))
    assert flat[-1] == pytest.approx(0.42)


def test_build_observation_is_deterministic():
    obs = np.random.default_rng(0).normal(size=25)
    block = [1.3, 0.75, 0.425]
    target = [1.5, 0.75, 0.425]
    a = build_subgoal_observation(obs, block, target, "lift", collision_prob=0.1)
    b = build_subgoal_observation(obs, block, target, "lift", collision_prob=0.1)
    np.testing.assert_array_equal(a, b)


def test_sanitize_overwrites_object_pos_and_object_rel_pos_only():
    # FetchPickAndPlace-v3 layout: grip_pos(3) object_pos(3) object_rel_pos(3)
    # gripper_state(2) ... — see sanitize_observation_for_perception's
    # docstring for how this was confirmed against the actual env source.
    obs = np.arange(25, dtype=np.float32)   # obs[0:3]=[0,1,2] grip_pos
    perceived = np.array([9.0, 9.0, 9.0], dtype=np.float32)

    sanitized = sanitize_observation_for_perception(obs, perceived)

    np.testing.assert_allclose(sanitized[0:3], obs[0:3])            # grip_pos untouched
    np.testing.assert_allclose(sanitized[3:6], perceived)           # object_pos -> perceived
    np.testing.assert_allclose(sanitized[6:9], perceived - obs[0:3])  # object_rel_pos recomputed
    np.testing.assert_allclose(sanitized[9:], obs[9:])              # everything past index 9 untouched


def test_sanitize_does_not_mutate_the_input_observation():
    obs = np.arange(25, dtype=np.float32)
    original = obs.copy()
    sanitize_observation_for_perception(obs, [9.0, 9.0, 9.0])
    np.testing.assert_array_equal(obs, original)


def test_sanitize_with_true_block_pos_matches_the_original_object_rel_pos_formula():
    """Sanity check against the real formula (object_rel_pos = object_pos -
    grip_pos): sanitizing with the TRUE block position should reproduce
    exactly what a genuine (unsanitized) FetchPickAndPlace-v3 observation
    would already contain at those indices."""
    grip_pos = np.array([1.0, 0.5, 0.8], dtype=np.float32)
    true_block_pos = np.array([1.3, 0.75, 0.425], dtype=np.float32)
    obs = np.zeros(25, dtype=np.float32)
    obs[0:3] = grip_pos
    obs[3:6] = true_block_pos
    obs[6:9] = true_block_pos - grip_pos

    sanitized = sanitize_observation_for_perception(obs, true_block_pos)
    np.testing.assert_allclose(sanitized, obs)
