#!/usr/bin/env python3
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
ACTION_NAME = "/scaled_joint_trajectory_controller/follow_joint_trajectory"
NUDGE_JOINT = "wrist_2_joint"
NUDGE_RAD = 0.34
MOVE_TIME_SEC = 3.0


class SmokeTest(Node):
    def __init__(self):
        super().__init__("ur3e_smoke_test")
        self._client = ActionClient(self, FollowJointTrajectory, ACTION_NAME)
        self._current_positions = None
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)

    def _on_joint_state(self, msg: JointState) -> None:
        if self._current_positions is None:
            name_to_pos = dict(zip(msg.name, msg.position))
            self._current_positions = [name_to_pos[j] for j in JOINT_NAMES]

    def wait_for_joint_state(self, timeout_sec: float = 5.0) -> None:
        deadline = time.time() + timeout_sec
        while self._current_positions is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._current_positions is None:
            raise RuntimeError("No /joint_states received — is the driver running?")

    def send_trajectory(self, positions: list, time_from_start_sec: float):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(time_from_start_sec)
        point.time_from_start.nanosec = int((time_from_start_sec % 1) * 1e9)
        goal.trajectory.points = [point]

        self.get_logger().info(f"Sending target: {positions}")
        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            raise RuntimeError("Goal rejected by controller")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return result_future.result()


def main():
    rclpy.init()
    node = SmokeTest()

    if not node._client.wait_for_server(timeout_sec=5.0):
        node.get_logger().error(f"Action server {ACTION_NAME} not available")
        rclpy.shutdown()
        return

    node.wait_for_joint_state()
    start = list(node._current_positions)
    node.get_logger().info(f"Start joint positions: {start}")

    idx = JOINT_NAMES.index(NUDGE_JOINT)
    nudged = list(start)
    nudged[idx] += NUDGE_RAD

    node.send_trajectory(nudged, MOVE_TIME_SEC)
    time.sleep(1.0)
    node.send_trajectory(start, MOVE_TIME_SEC)

    node.get_logger().info("Smoke test complete — returned to start position.")
    rclpy.shutdown()


if __name__ == "__main__":
    main()