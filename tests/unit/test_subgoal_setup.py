"""
Unit tests for env.oracle.oracle_action (moved 2026-07-14 from
generate_sft_data.py — these lock in that the move didn't change behavior)
and env.setup._subgoal_setup_reached (the new per-subgoal handoff
predicate init_episode_before_subgoal uses). Both are pure numpy, no
mujoco/gymnasium needed.
"""

import numpy as np
import pytest

from think_then_act.env.oracle import oracle_action
from think_then_act.env.setup import _subgoal_setup_reached


def _obs(rel, finger_width_each=0.05) -> np.ndarray:
    obs_arr = np.zeros(25, dtype=np.float32)
    obs_arr[6:9] = rel
    obs_arr[9] = finger_width_each
    obs_arr[10] = finger_width_each
    return obs_arr


# ---------------------------------------------------------------------------
# oracle_action — regression guard for the generate_sft_data.py -> env/oracle.py move
# ---------------------------------------------------------------------------

def test_oracle_far_from_block_approaches_laterally():
    # rel[2] = -0.3 -> grip_z = block_z + 0.3, well above at_block_zone's
    # block_z+0.10 band, so this is unambiguously still APPROACH (both the
    # d_3d<0.10 and at_block_zone GRASP triggers need this to be false).
    obs_arr = _obs(rel=[0.3, 0.0, -0.3])
    achieved_goal, desired_goal = [1.0, 0.75, 0.425], [1.2, 0.9, 0.425]

    action, phase, carrying = oracle_action(obs_arr, achieved_goal, desired_goal)

    assert phase == "APPROACH"
    assert carrying is False
    assert action[0] > 0        # moves toward the block in +x
    assert action[2] == pytest.approx(0.0)   # lateral only, no vertical motion yet
    assert action[3] == pytest.approx(1.0)   # OPEN


def test_oracle_grasp_zone_still_descending_moves_toward_block():
    # gripper above the block (grip_z = block_z - rel[2] = block_z + 0.05)
    obs_arr = _obs(rel=[0.02, 0.0, -0.05])
    achieved_goal, desired_goal = [1.0, 0.75, 0.425], [1.2, 0.9, 0.425]

    action, phase, carrying = oracle_action(obs_arr, achieved_goal, desired_goal)

    assert phase == "GRASP"
    assert action[3] == pytest.approx(1.0)   # still OPEN while descending


def test_oracle_grasp_zone_aligned_closes_fingers():
    obs_arr = _obs(rel=[0.0, 0.0, 0.0], finger_width_each=0.05)   # total 0.10 > 0.07
    achieved_goal, desired_goal = [1.0, 0.75, 0.425], [1.2, 0.9, 0.425]

    action, phase, carrying = oracle_action(obs_arr, achieved_goal, desired_goal)

    assert phase == "GRASP"
    assert action[3] == pytest.approx(-1.0)   # CLOSE
    np.testing.assert_allclose(action[:3], [0.0, 0.0, 0.0])


def test_oracle_grasp_zone_fingers_closed_lifts():
    obs_arr = _obs(rel=[0.0, 0.0, 0.0], finger_width_each=0.03)   # total 0.06 <= 0.07
    achieved_goal, desired_goal = [1.0, 0.75, 0.425], [1.2, 0.9, 0.425]

    action, phase, carrying = oracle_action(obs_arr, achieved_goal, desired_goal)

    assert phase == "GRASP"
    assert action[2] > 0        # lift
    assert action[3] == pytest.approx(-1.0)   # CLOSE — maintain grasp


def test_oracle_lifted_and_grasped_enters_carry_toward_target():
    obs_arr = _obs(rel=[0.0, 0.0, 0.0], finger_width_each=0.03)
    achieved_goal, desired_goal = [1.0, 0.75, 0.46], [1.2, 0.9, 0.425]

    action, phase, carrying = oracle_action(obs_arr, achieved_goal, desired_goal, carrying=False)

    assert phase == "CARRY"
    assert carrying is True
    assert action[0] > 0    # target is +x, +y of achieved_goal
    assert action[1] > 0
    assert action[3] == pytest.approx(-1.0)   # CLOSE — keep holding


# ---------------------------------------------------------------------------
# _subgoal_setup_reached — the new per-subgoal handoff predicate
# ---------------------------------------------------------------------------

def test_close_gripper_reached_when_aligned_and_at_grasp_height_fingers_open():
    # d_xy = 0.02 <= 0.03, d_z = grip_z-block_z = 0.02 <= 0.03, fingers open.
    obs_arr = _obs(rel=[0.02, 0.0, -0.02], finger_width_each=0.05)   # total 0.10 > 0.09
    achieved_goal, desired_goal = [1.0, 0.75, 0.425], [1.2, 0.9, 0.425]

    assert _subgoal_setup_reached("close_gripper", obs_arr, achieved_goal, desired_goal,
                                   carrying=False) is True


def test_close_gripper_not_reached_while_still_far_laterally():
    # d_xy = 0.3, well outside the 0.03 band, even though at grasp height.
    obs_arr = _obs(rel=[0.3, 0.0, 0.0])
    achieved_goal, desired_goal = [1.0, 0.75, 0.425], [1.2, 0.9, 0.425]

    assert _subgoal_setup_reached("close_gripper", obs_arr, achieved_goal, desired_goal,
                                   carrying=False) is False


def test_close_gripper_not_reached_while_still_high_above():
    # d_xy tight, but d_z = 0.3 -> still descending, not "right above the block" yet.
    obs_arr = _obs(rel=[0.0, 0.0, -0.3])
    achieved_goal, desired_goal = [1.0, 0.75, 0.425], [1.2, 0.9, 0.425]

    assert _subgoal_setup_reached("close_gripper", obs_arr, achieved_goal, desired_goal,
                                   carrying=False) is False


def test_lift_reached_when_grasped_but_not_yet_lifted():
    obs_arr = _obs(rel=[0.0, 0.0, 0.0], finger_width_each=0.03)   # total 0.06 <= 0.07
    achieved_goal, desired_goal = [1.0, 0.75, 0.425], [1.2, 0.9, 0.425]   # block_z = 0.425 <= 0.45

    assert _subgoal_setup_reached("lift", obs_arr, achieved_goal, desired_goal,
                                   carrying=False) is True


def test_lift_not_reached_while_fingers_still_open():
    obs_arr = _obs(rel=[0.0, 0.0, 0.0], finger_width_each=0.05)   # total 0.10 > 0.07
    achieved_goal, desired_goal = [1.0, 0.75, 0.425], [1.2, 0.9, 0.425]

    assert _subgoal_setup_reached("lift", obs_arr, achieved_goal, desired_goal,
                                   carrying=False) is False


def test_move_to_target_reached_when_carrying_and_lifted():
    obs_arr = _obs(rel=[0.0, 0.0, 0.0], finger_width_each=0.03)
    achieved_goal, desired_goal = [1.0, 0.75, 0.46], [1.2, 0.9, 0.425]   # block_z = 0.46 > 0.45

    assert _subgoal_setup_reached("move_to_target", obs_arr, achieved_goal, desired_goal,
                                   carrying=True) is True


def test_move_to_target_not_reached_before_lift():
    obs_arr = _obs(rel=[0.0, 0.0, 0.0], finger_width_each=0.03)
    achieved_goal, desired_goal = [1.0, 0.75, 0.425], [1.2, 0.9, 0.425]   # still at table height

    assert _subgoal_setup_reached("move_to_target", obs_arr, achieved_goal, desired_goal,
                                   carrying=False) is False


def test_release_reached_when_carrying_near_target():
    obs_arr = _obs(rel=[0.0, 0.0, 0.0], finger_width_each=0.03)
    achieved_goal, desired_goal = [1.2, 0.9, 0.46], [1.22, 0.91, 0.46]   # within 0.05

    assert _subgoal_setup_reached("release", obs_arr, achieved_goal, desired_goal,
                                   carrying=True) is True


def test_release_not_reached_while_far_from_target():
    obs_arr = _obs(rel=[0.0, 0.0, 0.0], finger_width_each=0.03)
    achieved_goal, desired_goal = [1.0, 0.75, 0.46], [1.2, 0.9, 0.425]   # far from target

    assert _subgoal_setup_reached("release", obs_arr, achieved_goal, desired_goal,
                                   carrying=True) is False


def test_unknown_subgoal_raises():
    obs_arr = _obs(rel=[0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        _subgoal_setup_reached("align_xy", obs_arr, [1.0, 0.75, 0.425], [1.2, 0.9, 0.425],
                                carrying=False)
