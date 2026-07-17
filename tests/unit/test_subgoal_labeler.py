"""
Unit tests for training.subgoal_labeler — pure numpy, synthetic states, no mujoco.
"""

import numpy as np

from think_then_act.reward.subgoal_reward import SUBGOAL_LABELS, SubgoalWeights
from think_then_act.training.subgoal_labeler import label_subgoal, subgoal_diagnostics

W = SubgoalWeights()


def _make_obs(grip_pos, finger_widths=(0.05, 0.05)):
    """25-float obs vector with only grip_pos (0:3) and gripper_state (9:11) set."""
    obs = np.zeros(25)
    obs[0:3] = grip_pos
    obs[9:11] = finger_widths
    return obs


TARGET = [1.5, 0.75, 0.425]


def test_gripper_far_from_block_labels_align_xy():
    block = [1.3, 0.75, 0.425]
    obs = _make_obs([1.0, 0.75, 0.55])

    assert label_subgoal(obs, block, TARGET) == "align_xy"


def test_gripper_aligned_but_high_labels_descend():
    block = [1.3, 0.75, 0.425]
    obs = _make_obs([1.3, 0.75, 0.55])

    assert label_subgoal(obs, block, TARGET) == "descend"


def test_gripper_aligned_and_descended_open_labels_close_gripper():
    block = [1.3, 0.75, 0.425]
    obs = _make_obs([1.3, 0.75, 0.425], finger_widths=(0.05, 0.05))

    assert label_subgoal(obs, block, TARGET) == "close_gripper"


def test_grasped_low_labels_lift():
    # Real-grasp finger width, gripper co-located with a still-resting block.
    block = [1.3, 0.75, 0.425]
    obs = _make_obs([1.3, 0.75, 0.425], finger_widths=(0.024, 0.024))

    assert label_subgoal(obs, block, TARGET) == "lift"


def test_grasped_and_lifted_but_far_from_target_labels_move_to_target():
    block = [1.3, 0.75, 0.425 + 0.15]  # lifted well above lift_height
    obs = _make_obs(block, finger_widths=(0.024, 0.024))

    assert label_subgoal(obs, block, TARGET) == "move_to_target"


def test_grasped_and_at_target_labels_release_even_if_not_lifted():
    # Target sits at table height — carrying it there never needs to clear
    # lift_height, so "grasped + at target" must win over "grasped + not
    # lifted" (lift would be pointless if you're already there).
    target = [1.3, 0.75, 0.425]
    block = [1.3, 0.75, 0.425]
    obs = _make_obs(block, finger_widths=(0.024, 0.024))

    assert label_subgoal(obs, block, target) == "release"


def test_delivered_and_released_labels_none():
    block = [1.5, 0.75, 0.425]  # at target already
    obs = _make_obs([1.0, 0.75, 0.6], finger_widths=(0.05, 0.05))  # gripper elsewhere, open

    assert label_subgoal(obs, block, TARGET) is None


def test_diagnostics_exposes_all_six_flags():
    block = [1.3, 0.75, 0.425]
    obs = _make_obs([1.3, 0.75, 0.55])

    d = subgoal_diagnostics(obs, block, TARGET)

    assert set(d.keys()) == {
        "is_aligned_xy", "is_descended", "is_grasped",
        "is_lifted", "is_at_target", "is_open",
    }
    assert all(isinstance(v, bool) for v in d.values())


def test_label_subgoal_only_returns_known_labels_or_none():
    rng = np.random.default_rng(0)
    block = [1.3, 0.75, 0.425]
    for _ in range(50):
        grip = rng.uniform([1.0, 0.5, 0.4], [1.6, 1.0, 0.6])
        widths = rng.uniform(0.0, 0.05, size=2)
        obs = _make_obs(grip, finger_widths=widths)
        label = label_subgoal(obs, block, TARGET)
        assert label is None or label in SUBGOAL_LABELS
