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
def test_close_gripper_rewards_a_real_grasp_over_staying_open():
    # closedness is a Gaussian peaked at the measured real-grasp width
    # (0.048m) — a real grasp should score higher AND register done, while
    # staying fully open should do neither.
    block = [1.3, 0.75, 0.425]
    open_obs  = _make_obs([1.3, 0.75, 0.425], finger_widths=(0.05, 0.05))    # total 0.10, fully open
    grasp_obs = _make_obs([1.3, 0.75, 0.425], finger_widths=(0.024, 0.024)) # total 0.048, real-grasp width

    r_open,  b_open  = reward_close_gripper(open_obs,  block, [0, 0, 0])
    r_grasp, b_grasp = reward_close_gripper(grasp_obs, block, [0, 0, 0])

    assert r_grasp > r_open
    assert b_open["done"] is False
    assert b_grasp["done"] is True


def test_close_gripper_closing_on_nothing_scores_worse_than_a_real_grasp():
    # Closing all the way to width=0 (nothing between the fingers) used to
    # be the linear formula's MAXIMUM reward (closedness=1.0) — confirmed
    # exploitable 2026-07-16 (see close_gripper_target_width's comment): a
    # trained checkpoint retreated above the block specifically to close on
    # empty air instead of gripping it. The Gaussian must score this WORSE
    # than a real grasp and refuse to mark it done.
    block = [1.3, 0.75, 0.425]
    empty_close_obs = _make_obs([1.3, 0.75, 0.425], finger_widths=(0.0, 0.0))   # total 0, closed on nothing
    grasp_obs       = _make_obs([1.3, 0.75, 0.425], finger_widths=(0.024, 0.024)) # total 0.048, real grasp

    r_empty, b_empty = reward_close_gripper(empty_close_obs, block, [0, 0, 0])
    r_grasp, _        = reward_close_gripper(grasp_obs,       block, [0, 0, 0])

    assert r_empty < r_grasp
    assert b_empty["done"] is False


def test_close_gripper_penalizes_ending_up_far_from_block():
    # The `-0.1*d_grip_block` per-step distance penalty was removed
    # 2026-07-16 (conflicted with the stillness penalty, same "don't drift"
    # idea via position instead of action) then re-added the same day at a
    # higher weight (0.5, matching close_gripper_stillness_weight) — removing
    # it entirely turned out to drop the only thing penalizing ENDING UP far
    # from the block as a STATE; the stillness penalty only covers the
    # action magnitude of the current step, so small drift could still
    # accumulate across an episode with nothing discouraging it continuously
    # (see close_gripper_distance_weight's comment). Confirm it's back.
    block = [1.3, 0.75, 0.425]
    closed_near_obs = _make_obs([1.3, 0.75, 0.425],  finger_widths=(0.0, 0.0))
    closed_far_obs  = _make_obs([1.5, 0.75, 0.425],  finger_widths=(0.0, 0.0))

    r_near, b_near = reward_close_gripper(closed_near_obs, block, [0, 0, 0])
    r_far,  b_far  = reward_close_gripper(closed_far_obs,  block, [0, 0, 0])

    assert r_near > r_far
    assert b_near["done"] is False  # closedness alone (closed on nothing) still fails
    assert b_far["done"] is False   # ...and distance still fails done regardless of reward


def test_close_gripper_closing_far_from_block_is_not_done():
    # Closed on nothing (width=0) AND 0.2m from the block — fails both the
    # closedness peak (see close_gripper_target_width) and the dxy/dz gates.
    block = [1.3, 0.75, 0.425]
    closed_far_obs = _make_obs([1.5, 0.75, 0.425], finger_widths=(0.0, 0.0))

    _, breakdown = reward_close_gripper(closed_far_obs, block, [0, 0, 0])

    assert breakdown["done"] is False


def test_close_gripper_reached_at_measured_real_grasp_closedness():
    # width=0.048 (this project's measured real-grasp steady-state, per
    # scripts/measure_close_gripper_geometry.py) is now the PEAK of the
    # Gaussian (closedness=1.0), not a ~0.52 ceiling under the old linear
    # formula — confirm the new threshold (0.8) accepts a real grasp
    # exactly at the measured value.
    block = [1.3, 0.75, 0.425]
    real_grasp_obs = _make_obs([1.3, 0.75, 0.425], finger_widths=(0.024, 0.024))

    _, breakdown = reward_close_gripper(real_grasp_obs, block, [0, 0, 0])

    assert breakdown["closedness"] == pytest.approx(1.0, abs=0.01)
    assert breakdown["done"] is True


def test_close_gripper_hovering_above_block_is_not_done():
    # Small xy offset, small aggregate 3D distance, but a ~4.5cm VERTICAL
    # gap — exactly the exploit found 2026-07-16 (record_subgoal_video.py's
    # trajectory log showed a checkpoint retreating upward and closing
    # there instead of descending). The old single-aggregate-distance check
    # let this pass; the split dxy/dz gate must reject it via d_z alone.
    block = [1.3, 0.75, 0.425]
    hovering_obs = _make_obs([1.3, 0.75, 0.425 + 0.045], finger_widths=(0.024, 0.024))

    _, breakdown = reward_close_gripper(hovering_obs, block, [0, 0, 0])

    assert breakdown["closedness"] == pytest.approx(1.0, abs=0.01)  # real grasp width
    assert breakdown["d_z"] == pytest.approx(0.045, abs=1e-6)
    assert breakdown["done"] is False


def test_close_gripper_breakdown_surfaces_d_grip_block():
    # subgoal_env.py's drift-interrupt reads breakdown["d_grip_block"]
    # directly rather than recomputing geometry — lock in that it's there
    # and correct.
    block = [1.3, 0.75, 0.425]
    obs = _make_obs([1.4, 0.75, 0.425])   # 0.10m away in x

    _, breakdown = reward_close_gripper(obs, block, [0, 0, 0])

    assert breakdown["d_grip_block"] == pytest.approx(0.10, abs=1e-6)


def test_close_gripper_penalizes_translation_action():
    # Same state, only the action's dx/dy/dz differ — a policy moving the
    # arm while closing (found 2026-07-16: real training telemetry showed
    # the indirect d_grip_block-after-the-fact penalty alone wasn't enough
    # signal, entropy stayed flat/completion_rate stuck at 0%) should score
    # strictly worse than one holding still, even with identical geometry.
    block = [1.3, 0.75, 0.425]
    obs = _make_obs([1.3, 0.75, 0.425], finger_widths=(0.024, 0.024))

    r_still, b_still = reward_close_gripper(obs, block, [0, 0, 0], action=[0.0, 0.0, 0.0, -1.0])
    r_moving, b_moving = reward_close_gripper(obs, block, [0, 0, 0], action=[1.0, 1.0, 1.0, -1.0])

    assert r_moving < r_still
    assert b_still["translation_norm"] == pytest.approx(0.0)
    assert b_moving["translation_norm"] == pytest.approx(np.sqrt(3), abs=1e-6)


def test_close_gripper_no_action_defaults_to_no_translation_penalty():
    # action=None (e.g. reward computed at env.reset(), before any action
    # exists) must not penalize or error — backward-compatible default.
    block = [1.3, 0.75, 0.425]
    obs = _make_obs([1.3, 0.75, 0.425], finger_widths=(0.024, 0.024))

    reward, breakdown = reward_close_gripper(obs, block, [0, 0, 0])

    assert breakdown["translation_norm"] == 0.0
    assert reward == pytest.approx(1.0, abs=0.01)


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
