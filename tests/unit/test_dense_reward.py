"""Unit tests for think_then_act.reward.dense_reward — pure numpy logic, no mujoco/torch."""

import numpy as np
import pytest

from think_then_act.reward.dense_reward import (
    DEFAULT_WEIGHTS,
    RewardWeights,
    apply_to_episode,
    compute_dense_reward,
)

# 25-length observation vector: only obs[0:3] (grip_pos) and obs[9:11]
# (gripper_state) are read by compute_dense_reward — everything else is padding.
def _make_obs(grip_pos, gripper_state):
    obs = np.zeros(25)
    obs[0:3] = grip_pos
    obs[9:11] = gripper_state
    return obs


def test_approach_reward_grows_more_negative_with_distance():
    near_obs = _make_obs(grip_pos=[0.0, 0.0, 0.0], gripper_state=[0.05, 0.05])
    far_obs = _make_obs(grip_pos=[1.0, 0.0, 0.0], gripper_state=[0.05, 0.05])
    block = [0.0, 0.0, 0.0]
    target = [5.0, 5.0, 5.0]  # far from block so transport/success terms don't dominate

    _, near = compute_dense_reward(near_obs, block, target, {"is_success": False})
    _, far = compute_dense_reward(far_obs, block, target, {"is_success": False})

    assert near["r_approach"] == 0.0
    assert far["r_approach"] == pytest.approx(-DEFAULT_WEIGHTS.w_approach * 1.0)
    assert far["r_approach"] < near["r_approach"]


def test_transport_weighted_higher_than_approach():
    # Equal-magnitude distances for approach vs. transport; transport's
    # weight (2.0) must dominate approach's (1.0) in the total.
    obs = _make_obs(grip_pos=[1.0, 0.0, 0.0], gripper_state=[0.05, 0.05])
    block = [0.0, 0.0, 0.0]
    target = [1.0, 0.0, 0.0]

    _, breakdown = compute_dense_reward(obs, block, target, {"is_success": False})

    assert breakdown["r_approach"] == pytest.approx(-1.0)
    assert breakdown["r_transport"] == pytest.approx(-2.0)


def test_grasp_bonus_requires_both_proximity_and_closed_gripper():
    block = [0.0, 0.0, 0.0]
    target = [0.0, 0.0, 0.0]

    near_closed = _make_obs(grip_pos=[0.0, 0.0, 0.0], gripper_state=[0.0, 0.0])
    near_open = _make_obs(grip_pos=[0.0, 0.0, 0.0], gripper_state=[0.05, 0.05])
    far_closed = _make_obs(grip_pos=[1.0, 0.0, 0.0], gripper_state=[0.0, 0.0])

    _, near_closed_bd = compute_dense_reward(near_closed, block, target, {"is_success": False})
    _, near_open_bd = compute_dense_reward(near_open, block, target, {"is_success": False})
    _, far_closed_bd = compute_dense_reward(far_closed, block, target, {"is_success": False})

    assert near_closed_bd["r_grasp"] == pytest.approx(DEFAULT_WEIGHTS.w_grasp)
    assert near_open_bd["r_grasp"] == pytest.approx(0.0)
    assert far_closed_bd["r_grasp"] == pytest.approx(0.0)


def test_success_bonus_only_applied_when_flagged():
    obs = _make_obs(grip_pos=[0.0, 0.0, 0.0], gripper_state=[0.05, 0.05])
    block = target = [0.0, 0.0, 0.0]

    total_success, bd_success = compute_dense_reward(obs, block, target, {"is_success": True})
    total_fail, bd_fail = compute_dense_reward(obs, block, target, {"is_success": False})

    assert bd_success["r_success"] == pytest.approx(DEFAULT_WEIGHTS.w_success)
    assert bd_fail["r_success"] == 0.0
    assert total_success - total_fail == pytest.approx(DEFAULT_WEIGHTS.w_success)


def test_custom_weights_override_defaults():
    obs = _make_obs(grip_pos=[1.0, 0.0, 0.0], gripper_state=[0.05, 0.05])
    block = [0.0, 0.0, 0.0]
    target = [0.0, 0.0, 0.0]
    weights = RewardWeights(w_approach=10.0)

    _, breakdown = compute_dense_reward(obs, block, target, {"is_success": False}, weights=weights)

    assert breakdown["r_approach"] == pytest.approx(-10.0)


def test_apply_to_episode_adds_reward_fields_without_mutating_input():
    episode_log = [
        {
            "observation": _make_obs([0.0, 0.0, 0.0], [0.05, 0.05]).tolist(),
            "achieved_goal": [0.0, 0.0, 0.0],
            "desired_goal": [1.0, 0.0, 0.0],
            "is_success": False,
        },
        {
            "observation": None,  # e.g. a pre-reset placeholder entry
            "achieved_goal": None,
            "desired_goal": None,
            "is_success": False,
        },
    ]

    enriched = apply_to_episode(episode_log)

    assert len(enriched) == 2
    assert "dense_reward" in enriched[0]
    assert "reward_breakdown" in enriched[0]
    assert "dense_reward" not in enriched[1]  # observation-less entries pass through unchanged
    assert "dense_reward" not in episode_log[0]  # original list untouched
