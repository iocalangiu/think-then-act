#!/usr/bin/env python3
"""
Diagnostic: test /compute_ik for an OFFSET pose (current + dx,dy,dz),
not the trivial current pose. Isolates whether compute_cartesian_path's
0.000 fraction is a real reachability failure or specific to its own
interpolation logic.
"""
import argparse
import time

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK

GROUP_NAME = "ur_manipulator"
LINK_NAME = "tool0"
BASE_FRAME = "base_link"
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

ERROR_CODES = {
    1: "SUCCESS", 99999: "FAILURE", -1: "PLANNING_FAILED",
    -2: "INVALID_MOTION_PLAN", -3: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
    -4: "CONTROL_FAILED", -5: "UNABLE_TO_AQUIRE_SENSOR_DATA", -6: "TIMED_OUT",
    -7: "PREEMPTED", -10: "START_STATE_IN_COLLISION",
    -11: "START_STATE_VIOLATES_PATH_CONSTRAINTS", -26: "START_STATE_INVALID",
    -12: "GOAL_IN_COLLISION", -13: "GOAL_VIOLATES_PATH_CONSTRAINTS",
    -14: "GOAL_CONSTRAINTS_VIOLATED", -27: "GOAL_STATE_INVALID",
    -28: "UNRECOGNIZED_GOAL_TYPE", -15: "INVALID_GROUP_NAME",
    -16: "INVALID_GOAL_CONSTRAINTS", -17: "INVALID_ROBOT_STATE",
    -18: "INVALID_LINK_NAME", -19: "INVALID_OBJECT_NAME",
    -21: "FRAME_TRANSFORM_FAILURE", -22: "COLLISION_CHECKING_UNAVAILABLE",
    -23: "ROBOT_STATE_STALE", -24: "SENSOR_INFO_STALE",
    -25: "COMMUNICATION_FAILURE", -29: "CRASH", -30: "ABORT",
    -31: "NO_IK_SOLUTION",
}


class IkTest(Node):
    def __init__(self):
        super().__init__("ik_test")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        self._joint_positions = None
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)

    def _on_joint_state(self, msg: JointState) -> None:
        if self._joint_positions is None:
            d = dict(zip(msg.name, msg.position))
            if all(j in d for j in JOINT_NAMES):
                self._joint_positions = [d[j] for j in JOINT_NAMES]

    def wait_for_joint_state(self, timeout_sec: float = 5.0) -> list:
        deadline = time.time() + timeout_sec
        while self._joint_positions is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._joint_positions is None:
            raise RuntimeError("No /joint_states received")
        return self._joint_positions

    def get_current_pose(self) -> PoseStamped:
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                tf = self.tf_buffer.lookup_transform(BASE_FRAME, LINK_NAME, Time())
                ps = PoseStamped()
                ps.header.frame_id = BASE_FRAME
                ps.pose.position.x = tf.transform.translation.x
                ps.pose.position.y = tf.transform.translation.y
                ps.pose.position.z = tf.transform.translation.z
                ps.pose.orientation = tf.transform.rotation
                return ps
            except Exception:
                rclpy.spin_once(self, timeout_sec=0.2)
        raise RuntimeError(f"Could not look up {BASE_FRAME} -> {LINK_NAME} transform")

    def run(self, dx: float, dy: float, dz: float):
        current_positions = self.wait_for_joint_state()
        current_pose = self.get_current_pose()
        target_pose = PoseStamped()
        target_pose.header.frame_id = BASE_FRAME
        target_pose.pose.position.x = current_pose.pose.position.x + dx
        target_pose.pose.position.y = current_pose.pose.position.y + dy
        target_pose.pose.position.z = current_pose.pose.position.z + dz
        target_pose.pose.orientation = current_pose.pose.orientation
        self.get_logger().info(
            f"Target pose: x={target_pose.pose.position.x:.4f} "
            f"y={target_pose.pose.position.y:.4f} z={target_pose.pose.position.z:.4f}"
        )

        req = GetPositionIK.Request()
        req.ik_request.group_name = GROUP_NAME
        req.ik_request.robot_state.joint_state.name = JOINT_NAMES
        req.ik_request.robot_state.joint_state.position = current_positions
        req.ik_request.robot_state.is_diff = False
        req.ik_request.avoid_collisions = False
        req.ik_request.ik_link_name = LINK_NAME
        req.ik_request.pose_stamped = target_pose
        req.ik_request.timeout.sec = 2

        if not self.ik_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("/compute_ik service not available")

        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        code = response.error_code.val
        self.get_logger().info(f"error_code: {code} ({ERROR_CODES.get(code, 'UNKNOWN')})")
        if code == 1:
            self.get_logger().info(f"Solution joints: {list(response.solution.joint_state.position)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx", type=float, default=0.0)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--dz", type=float, default=0.04)
    args = parser.parse_args()

    rclpy.init()
    node = IkTest()
    try:
        node.run(args.dx, args.dy, args.dz)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()