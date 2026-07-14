"""
Unit tests for training.subgoal_features — pure numpy, no gymnasium needed.
"""

import numpy as np
import pytest

from think_then_act.reward.subgoal_reward import SUBGOAL_LABELS
from think_then_act.training.subgoal_features import (
    SUBGOAL_OBS_DIM,
    build_subgoal_observation,
    subgoal_to_onehot,
)


def test_subgoal_obs_dim_matches_the_actual_concatenation():
    obs   = np.zeros(25)
    block = [1.3, 0.75, 0.425]
    target = [1.5, 0.75, 0.425]
    flat = build_subgoal_observation(obs, block, target, "align_xy", collision_prob=0.0)
    assert flat.shape == (SUBGOAL_OBS_DIM,)
    assert flat.dtype == np.float32


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
    obs = np.arange(25, dtype=np.float32)
    block = [1.0, 2.0, 3.0]
    target = [4.0, 5.0, 6.0]
    flat = build_subgoal_observation(obs, block, target, "descend", collision_prob=0.42)

    # Layout: obs(25) | achieved(3) | desired(3) | onehot(6) | collision_prob(1)
    np.testing.assert_allclose(flat[0:25], obs)
    np.testing.assert_allclose(flat[25:28], block)
    np.testing.assert_allclose(flat[28:31], target)
    onehot = flat[31:31 + len(SUBGOAL_LABELS)]
    np.testing.assert_allclose(onehot, subgoal_to_onehot("descend"))
    assert flat[-1] == pytest.approx(0.42)


def test_build_observation_is_deterministic():
    obs = np.random.default_rng(0).normal(size=25)
    block = [1.3, 0.75, 0.425]
    target = [1.5, 0.75, 0.425]
    a = build_subgoal_observation(obs, block, target, "lift", collision_prob=0.1)
    b = build_subgoal_observation(obs, block, target, "lift", collision_prob=0.1)
    np.testing.assert_array_equal(a, b)
