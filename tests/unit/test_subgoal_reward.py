"""
Unit tests for reward.subgoal_reward — pure numpy, synthetic states, no mujoco.
"""

import numpy as np
import pytest

from think_then_act.reward.subgoal_reward import (
    SUBGOAL_LABELS,
    SubgoalWeights,
    compute_subgoal_reward,
    reward_align_xy,
    reward_close_gripper,
    reward_descend,
    reward_lift,
    reward_move_to_target,
    reward_release,
)

W = SubgoalWeights()


def _make_obs(grip_pos, finger_widths=(0.05, 0.05)):
    """25-float obs vector with only grip_pos (0:3) and gripper_state (9:11) set."""
    obs = np.zeros(25)
    obs[0:3] = grip_pos
    obs[9:11] = finger_widths
    return obs


# ---------------------------------------------------------------------------
# align_xy
# ---------------------------------------------------------------------------
def test_align_xy_improves_as_gripper_approaches_block_xy():
    block = [1.3, 0.75, 0.5]
    far_obs   = _make_obs([1.0, 0.75, 0.6])
    close_obs = _make_obs([1.29, 0.751, 0.6])

    r_far,   b_far   = reward_align_xy(far_obs,   block, [0, 0, 0])
    r_close, b_close  = reward_align_xy(close_obs, block, [0, 0, 0])

    assert r_close > r_far
    assert b_far["done"] is False
    assert b_close["done"] is True


# ---------------------------------------------------------------------------
# descend
# ---------------------------------------------------------------------------
def test_descend_rewards_closing_vertical_gap():
    block = [1.3, 0.75, 0.425]
    high_obs = _make_obs([1.3, 0.75, 0.55])
    low_obs  = _make_obs([1.3, 0.75, 0.43])

    r_high, b_high = reward_descend(high_obs, block, [0, 0, 0])
    r_low,  b_low  = reward_descend(low_obs,  block, [0, 0, 0])

    assert r_low > r_high
    assert b_high["done"] is False
    assert b_low["done"] is True


def test_descend_penalizes_collision_probability():
    block = [1.3, 0.75, 0.425]
    obs = _make_obs([1.3, 0.75, 0.55])

    r_safe, _ = reward_descend(obs, block, [0, 0, 0], collision_prob=0.0)
    r_risky, _ = reward_descend(obs, block, [0, 0, 0], collision_prob=0.9)

    assert r_risky < r_safe


# ---------------------------------------------------------------------------
# close_gripper
# ---------------------------------------------------------------------------
def test_close_gripper_rewards_closing_near_the_block():
    block = [1.3, 0.75, 0.425]
    open_obs   = _make_obs([1.3, 0.75, 0.425], finger_widths=(0.05, 0.05))
    closed_obs = _make_obs([1.3, 0.75, 0.425], finger_widths=(0.0, 0.0))

    r_open,   b_open   = reward_close_gripper(open_obs,   block, [0, 0, 0])
    r_closed, b_closed = reward_close_gripper(closed_obs, block, [0, 0, 0])

    assert r_closed > r_open
    assert b_open["done"] is False
    assert b_closed["done"] is True


def test_close_gripper_penalizes_closing_away_from_block():
    block = [1.3, 0.75, 0.425]
    closed_near_obs = _make_obs([1.3, 0.75, 0.425],  finger_widths=(0.0, 0.0))
    closed_far_obs  = _make_obs([1.5, 0.75, 0.425],  finger_widths=(0.0, 0.0))

    r_near, _ = reward_close_gripper(closed_near_obs, block, [0, 0, 0])
    r_far,  _ = reward_close_gripper(closed_far_obs,  block, [0, 0, 0])

    assert r_near > r_far


# ---------------------------------------------------------------------------
# lift
# ---------------------------------------------------------------------------
def test_lift_rewards_height_above_table():
    low_block  = [1.3, 0.75, W.table_z + 0.01]
    high_block = [1.3, 0.75, W.table_z + 0.15]
    obs = _make_obs([1.3, 0.75, 0.5])

    r_low,  b_low  = reward_lift(obs, low_block,  [0, 0, 0])
    r_high, b_high = reward_lift(obs, high_block, [0, 0, 0])

    assert r_high > r_low
    assert b_low["done"] is False
    assert b_high["done"] is True


# ---------------------------------------------------------------------------
# move_to_target
# ---------------------------------------------------------------------------
def test_move_to_target_rewards_block_target_proximity():
    target = [1.5, 0.75, 0.425]
    far_block   = [1.0, 0.75, 0.425]
    close_block = [1.49, 0.751, 0.425]
    obs = _make_obs([1.3, 0.75, 0.5])

    r_far,   b_far   = reward_move_to_target(obs, far_block,   target)
    r_close, b_close = reward_move_to_target(obs, close_block, target)

    assert r_close > r_far
    assert b_far["done"] is False
    assert b_close["done"] is True


# ---------------------------------------------------------------------------
# release
# ---------------------------------------------------------------------------
def test_release_rewards_opening_gripper():
    block = [1.3, 0.75, 0.425]
    closed_obs = _make_obs([1.3, 0.75, 0.425], finger_widths=(0.0, 0.0))
    open_obs   = _make_obs([1.3, 0.75, 0.425], finger_widths=(0.05, 0.05))

    r_closed, b_closed = reward_release(closed_obs, block, [0, 0, 0])
    r_open,   b_open   = reward_release(open_obs,   block, [0, 0, 0])

    assert r_open > r_closed
    assert b_closed["done"] is False
    assert b_open["done"] is True


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def test_compute_subgoal_reward_dispatches_to_matching_function():
    block = [1.3, 0.75, 0.425]
    obs = _make_obs([1.3, 0.75, 0.55])

    direct_reward, direct_breakdown = reward_descend(obs, block, [0, 0, 0], collision_prob=0.3)
    dispatch_reward, dispatch_breakdown = compute_subgoal_reward(
        "descend", obs, block, [0, 0, 0], collision_prob=0.3
    )

    assert dispatch_reward == direct_reward
    assert dispatch_breakdown == direct_breakdown


def test_compute_subgoal_reward_covers_every_label():
    block = [1.3, 0.75, 0.425]
    obs = _make_obs([1.3, 0.75, 0.5])
    for label in SUBGOAL_LABELS:
        reward, breakdown = compute_subgoal_reward(label, obs, block, [1.5, 0.75, 0.425])
        assert np.isfinite(reward)
        assert isinstance(breakdown["done"], bool)


def test_compute_subgoal_reward_rejects_unknown_label():
    block = [1.3, 0.75, 0.425]
    obs = _make_obs([1.3, 0.75, 0.5])
    with pytest.raises(ValueError):
        compute_subgoal_reward("not_a_real_subgoal", obs, block, [0, 0, 0])
