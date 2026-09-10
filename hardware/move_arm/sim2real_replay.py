#!/usr/bin/env python3
"""
Sim2real displacement comparison: replays the first N per-step Cartesian
deltas from a REAL recorded MuJoCo align_xy trajectory
(artifacts/subgoal_videos/subgoal_videos/align_xy_ppo_after_best_trajectory.json)
on the real UR3e, logging intended (sim) vs actual (real) displacement per
step. Uses the same validated /compute_cartesian_path + FollowJointTrajectory
pipeline as cartesian_move_test.py -- NOT RTDE, NOT moveit_py.

Deltas are the ACTUAL achieved grip_pos differences between consecutive sim
steps (real physics-driven displacement, not the raw normalized policy
action) -- this is what "the same movement" means here.

Full recorded trajectory has 16 steps totaling ~0.55m net displacement --
too large for a UR3e's ~0.5m reach, so only the first --num-steps are
replayed by default (5 steps, ~15-17cm cumulative, well within the range
already validated safe).
"""
import argparse
import math
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

# Per-step (dx,dy,dz) deltas -- ACTUAL achieved grip_pos differences from
# the recorded sim trajectory (align_xy_ppo_after_best_trajectory.json),
# consecutive steps 0->15.
SIM_DELTAS = [
    [-0.026606, -0.006229, 0.00865], [-0.031323, -0.00302, 0.012281],
    [-0.033595, -0.001959, 0.012479], [-0.035731, -0.002273, 0.012563],
    [-0.036929, -0.002576, 0.012277], [-0.037458, -0.002799, 0.011719],
    [-0.037501, -0.002966, 0.010945], [-0.0372, -0.002935, 0.010022],
    [-0.036634, -0.002882, 0.008996], [-0.035847, -0.003722, 0.007888],
    [-0.034809, -0.002545, 0.00688], [-0.03348, -0.001812, 0.005905],
    [-0.031818, -0.002032, 0.004985], [-0.029694, -0.002099, 0.003582],
    [-0.028387, -0.002061, 0.001854], [-0.026577, -0.001978, -0.000412],
]


class Sim2RealReplay(Node):
    def __init__(self):
        super().__init__("sim2real_replay")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cartesian_client = self.create_client(GetCartesianPath, "/compute_cartesian_path")
        self.arm_client = ActionClient(self, FollowJointTrajectory, ARM_ACTION_NAME)
        self._joint_positions = None
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)

    def _on_joint_state(self, msg: JointState) -> None:
        d = dict(zip(msg.name, msg.position))
        if all(j in d for j in JOINT_NAMES):
            self._joint_positions = [d[j] for j in JOINT_NAMES]

    def wait_for_joint_state(self, timeout_sec: float = 5.0) -> list:
        self._joint_positions = None
        deadline = time.time() + timeout_sec
        while self._joint_positions is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._joint_positions is None:
            raise RuntimeError("No /joint_states received")
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

    def plan_step(self, dx: float, dy: float, dz: float):
        current_positions = self.wait_for_joint_state()
        current_pose = self.get_current_pose()
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
        req.avoid_collisions = True

        future = self.cartesian_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        return response, current_pose

    def execute(self, joint_trajectory) -> None:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_trajectory
        future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            raise RuntimeError("Trajectory goal rejected")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", type=int, default=5,
                         help="how many of the 16 recorded sim steps to replay (default 5)")
    args = parser.parse_args()

    steps = SIM_DELTAS[:args.num_steps]
    net = [sum(d[i] for d in steps) for i in range(3)]
    net_mag = math.sqrt(sum(x * x for x in net))
    print(f"Replaying {len(steps)} steps. Net displacement: "
          f"dx={net[0]:.4f} dy={net[1]:.4f} dz={net[2]:.4f} (mag={net_mag:.4f}m)")

    rclpy.init()
    node = Sim2RealReplay()
    node.cartesian_client.wait_for_service(timeout_sec=5.0)
    node.arm_client.wait_for_server(timeout_sec=5.0)

    answer = input("Execute this sequence on the real robot? [y/N] ")
    if answer.strip().lower() != "y":
        print("Aborted.")
        rclpy.shutdown()
        return

    try:
        for i, (dx, dy, dz) in enumerate(steps):
            response, pose_before = node.plan_step(dx, dy, dz)
            node.get_logger().info(
                f"Step {i}: sim_delta=({dx:.4f},{dy:.4f},{dz:.4f}) "
                f"fraction={response.fraction:.3f}"
            )
            if response.fraction < MIN_FRACTION:
                node.get_logger().error(
                    f"Step {i}: only {response.fraction:.1%} plannable -- stopping here, "
                    f"not continuing the sequence"
                )
                break
            node.execute(response.solution.joint_trajectory)
            pose_after = node.get_current_pose()
            actual = (
                pose_after.position.x - pose_before.position.x,
                pose_after.position.y - pose_before.position.y,
                pose_after.position.z - pose_before.position.z,
            )
            node.get_logger().info(
                f"Step {i}: sim_delta=({dx:.4f},{dy:.4f},{dz:.4f}) "
                f"real_delta=({actual[0]:.4f},{actual[1]:.4f},{actual[2]:.4f})"
            )
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()