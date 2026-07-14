"""
Unit tests for env.setup.is_meaningful_table_collision — pure function over
string pairs, no mujoco needed. Uses the actual body names found empirically
via validate_collision_labels.py's full geom enumeration on 2026-07-12
(table's body is "table0"; geom itself has no explicit name in the MJCF).
"""

from think_then_act.env.setup import TABLE_BODY_NAME, is_meaningful_table_collision


def test_no_contacts_is_not_a_collision():
    assert is_meaningful_table_collision([]) is False


def test_benign_structural_contacts_are_not_collisions():
    pairs = [
        ("floor0", "robot0:base_link"),
        (TABLE_BODY_NAME, "robot0:base_link"),
        (TABLE_BODY_NAME, "robot0:laser_link"),
        (TABLE_BODY_NAME, "object0"),
    ]
    assert is_meaningful_table_collision(pairs) is False


def test_gripper_touching_table_is_benign():
    """
    2026-07-13: the gripper/fingers reaching down onto the table to grab a
    block resting there is normal grasping behavior, not the collision this
    predictor should flag — only added to the benign set explicitly.
    """
    pairs = [
        (TABLE_BODY_NAME, "robot0:gripper_link"),
        (TABLE_BODY_NAME, "robot0:r_gripper_finger_link"),
        (TABLE_BODY_NAME, "robot0:l_gripper_finger_link"),
    ]
    assert is_meaningful_table_collision(pairs) is False


def test_arm_touching_table_is_a_collision():
    """The actual failure mode this predictor targets: the ARM (not the
    gripper) hitting the table from premature descent while still
    laterally misaligned."""
    pairs = [
        ("floor0", "robot0:base_link"),                # benign, still present
        (TABLE_BODY_NAME, "robot0:forearm_roll_link"),  # the actual thing we care about
    ]
    assert is_meaningful_table_collision(pairs) is True


def test_order_of_bodies_in_pair_does_not_matter():
    assert is_meaningful_table_collision([("robot0:forearm_roll_link", TABLE_BODY_NAME)]) is True
    assert is_meaningful_table_collision([(TABLE_BODY_NAME, "robot0:forearm_roll_link")]) is True


def test_non_table_contacts_are_ignored():
    pairs = [("robot0:r_gripper_finger_link", "robot0:l_gripper_finger_link")]
    assert is_meaningful_table_collision(pairs) is False


def test_custom_table_body_name():
    pairs = [("custom_table", "robot0:forearm_roll_link")]
    assert is_meaningful_table_collision(pairs, table_body="custom_table") is True
    assert is_meaningful_table_collision(pairs, table_body=TABLE_BODY_NAME) is False
