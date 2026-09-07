#!/usr/bin/env python3
"""Moves the arm back to whatever joint positions were last saved to
~/ros2_ws/saved_pose.json (via the record_position one-liner). Plain
joint-space move via FollowJointTrajectory -- same mechanism as
ros2_smoke_test_ur3e.py, no Cartesian planning needed for a return-home."""
import json
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
ACTION_NAME = "/scaled_joint_trajectory_controller/follow_joint_trajectory"
MOVE_TIME_SEC = 4.0


def main():
    with open("/home/user/ros2_ws/saved_pose.json") as f:
        saved = json.load(f)
    target = [saved[j] for j in JOINT_NAMES]
    print("Returning to saved pose:", target)

    rclpy.init()
    node = Node("return_to_saved_pose")
    client = ActionClient(node, FollowJointTrajectory, ACTION_NAME)
    if not client.wait_for_server(timeout_sec=5.0):
        raise RuntimeError(f"Action server {ACTION_NAME} not available")

    answer = input("Execute return-to-saved-pose on the real robot? [y/N] ")
    if answer.strip().lower() != "y":
        print("Aborted.")
        rclpy.shutdown()
        return

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = JOINT_NAMES
    point = JointTrajectoryPoint()
    point.positions = target
    point.time_from_start.sec = int(MOVE_TIME_SEC)
    goal.trajectory.points = [point]

    future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, future)
    goal_handle = future.result()
    if not goal_handle.accepted:
        raise RuntimeError("Goal rejected")
    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    print("Done.")
    rclpy.shutdown()


if __name__ == "__main__":
    main()