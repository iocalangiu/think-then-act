#!/usr/bin/env python3
"""
Cartesian move test: /compute_cartesian_path (MoveIt2) plans, then executes
through the already-validated FollowJointTrajectory action.

move_group's own planning_scene_monitor was found NOT subscribed to
/joint_states (confirmed via `ros2 node info /move_group` -- its current
state was likely a stale URDF-default, not the real robot), so this script
explicitly fetches /joint_states itself and populates start_state, instead
of relying on move_group's internal (broken) current-state tracking.
"""
import argparse
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetCartesianPath
from control_msgs.action import FollowJointTrajectory

GROUP_NAME = "ur_manipulator"
LINK_NAME = "tool0"
BASE_FRAME = "base_link"
ARM_ACTION_NAME = "/scaled_joint_trajectory_controller/follow_joint_trajectory"
MIN_FRACTION = 0.9
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


class CartesianMover(Node):
    def __init__(self):
        super().__init__("cartesian_move_test")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cartesian_client = self.create_client(GetCartesianPath, "/compute_cartesian_path")
        self.arm_client = ActionClient(self, FollowJointTrajectory, ARM_ACTION_NAME)
        self._joint_positions = None
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)

    def _on_joint_state(self, msg: JointState) -> None:
        if self._joint_positions is None:
            name_to_pos = dict(zip(msg.name, msg.position))
            if all(j in name_to_pos for j in JOINT_NAMES):
                self._joint_positions = [name_to_pos[j] for j in JOINT_NAMES]

    def wait_for_joint_state(self, timeout_sec: float = 5.0) -> list:
        deadline = time.time() + timeout_sec
        while self._joint_positions is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._joint_positions is None:
            raise RuntimeError("No /joint_states received with all expected arm joints")
        return self._joint_positions

    def get_current_pose(self) -> Pose:
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                tf = self.tf_buffer.lookup_transform(BASE_FRAME, LINK_NAME, Time())
                pose = Pose()
                pose.position.x = tf.transform.translation.x
                pose.position.y = tf.transform.translation.y
                pose.position.z = tf.transform.translation.z
                pose.orientation = tf.transform.rotation
                return pose
            except Exception:
                rclpy.spin_once(self, timeout_sec=0.2)
        raise RuntimeError(f"Could not look up {BASE_FRAME} -> {LINK_NAME} transform")

    def plan_cartesian(self, dx: float, dy: float, dz: float, avoid_collisions: bool):
        current_positions = self.wait_for_joint_state()
        current_pose = self.get_current_pose()
        self.get_logger().info(
            f"Current {LINK_NAME} pose in {BASE_FRAME}: "
            f"x={current_pose.position.x:.4f} y={current_pose.position.y:.4f} z={current_pose.position.z:.4f}"
        )
        self.get_logger().info(f"Current joint positions: {current_positions}")

        target = Pose()
        target.position.x = current_pose.position.x + dx
        target.position.y = current_pose.position.y + dy
        target.position.z = current_pose.position.z + dz
        target.orientation = current_pose.orientation

        req = GetCartesianPath.Request()
        req.header.frame_id = BASE_FRAME
        req.start_state.joint_state.name = JOINT_NAMES
        req.start_state.joint_state.position = current_positions
        req.start_state.is_diff = False
        req.group_name = GROUP_NAME
        req.link_name = LINK_NAME
        req.waypoints = [target]
        req.max_step = 0.01
        req.jump_threshold = 0.0
        req.avoid_collisions = avoid_collisions

        if not self.cartesian_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("/compute_cartesian_path service not available")

        future = self.cartesian_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        self.get_logger().info(
            f"avoid_collisions={avoid_collisions} -> fraction achieved: {response.fraction:.3f}"
        )
        if response.fraction < MIN_FRACTION:
            raise RuntimeError(
                f"Only {response.fraction:.1%} of the path was plannable "
                f"(need >= {MIN_FRACTION:.0%}) -- aborting, not executing a partial path"
            )
        return response.solution.joint_trajectory

    def execute(self, joint_trajectory) -> None:
        if not self.arm_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f"Action server {ARM_ACTION_NAME} not available")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_trajectory
        self.get_logger().info(f"Executing trajectory with {len(joint_trajectory.points)} points")
        future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            raise RuntimeError("Trajectory goal rejected")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info("Execution complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx", type=float, default=0.0)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--dz", type=float, default=0.04)
    parser.add_argument("--no-avoid-collisions", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    node = CartesianMover()
    try:
        joint_trajectory = node.plan_cartesian(
            args.dx, args.dy, args.dz, avoid_collisions=not args.no_avoid_collisions
        )
        answer = input("Plan succeeded. Execute on the real robot? [y/N] ")
        if answer.strip().lower() == "y":
            node.execute(joint_trajectory)
        else:
            node.get_logger().info("Not executing (user declined).")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()