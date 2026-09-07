#!/usr/bin/env python3
"""
Interactive move-toward-blocks menu -- Cartesian moves (f/d), gripper
rotation (r/R, about tool0's own approach axis, confirmed correct via
camera), and now joint-space elbow jogs (e/E) since a pure Cartesian
orientation change naturally routes through the wrist joints, not the
elbow -- to reposture the elbow specifically, bypass Cartesian IK
entirely and nudge wrist_2_joint directly, same mechanism as
ros2_smoke_test_ur3e.py's original joint-space nudge.

Requires scaled_joint_trajectory_controller active.
"""
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
from trajectory_msgs.msg import JointTrajectoryPoint

import json

GROUP_NAME = "ur_manipulator"
LINK_NAME = "tool0"
BASE_FRAME = "base_link"
ARM_ACTION_NAME = "/scaled_joint_trajectory_controller/follow_joint_trajectory"
MIN_FRACTION = 0.9
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
FORWARD_STEP = +0.02
SIDE_STEP = 0.02
DESCEND_STEP = -0.02
ROTATE_STEP_DEG = 15.0
ELBOW_STEP_DEG = 20.0
JOINT_MOVE_TIME_SEC = 3.0


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def quat_from_z_angle(angle_rad):
    return (0.0, 0.0, math.sin(angle_rad / 2), math.cos(angle_rad / 2))


class MoveToBlocks(Node):
    def __init__(self):
        super().__init__("move_to_blocks")
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
        raise RuntimeError("Could not look up base_link -> tool0 transform")

    def save_current_pose(self, path="goal_pose.json"):
        """Records joint positions AND the Cartesian tool0 pose -- the
        Cartesian x,y,z is what run_align_xy_real.py reads as the
        perception-bypass 'block position'."""
        joint_positions = self.wait_for_joint_state()
        pose = self.get_current_pose()
        data = {
            "joints": dict(zip(JOINT_NAMES, joint_positions)),
            "cartesian": {
                "x": pose.position.x,
                "y": pose.position.y,
                "z": pose.position.z,
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved current pose to {path}: "
              f"x={pose.position.x:.4f} y={pose.position.y:.4f} z={pose.position.z:.4f}")

    def go_to_joint_positions(self, positions: list, move_time_sec: float = JOINT_MOVE_TIME_SEC):
        """Direct joint-space move to an absolute target -- same mechanism
        as ros2_smoke_test_ur3e.py's send_trajectory, NOT the Cartesian
        planner, so this can't fail to a singularity/reachability issue."""
        from trajectory_msgs.msg import JointTrajectoryPoint
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(move_time_sec)
        goal.trajectory.points = [point]

        future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            print("Trajectory goal rejected")
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return True

    def go_home(self, path="saved_pose.json"):
        with open(path) as f:
            saved = json.load(f)
        positions = [saved[j] for j in JOINT_NAMES]
        print(f"Returning to {path}...")
        return self.go_to_joint_positions(positions)

    def go_to_target(self, path="goal_pose.json"):
        with open(path) as f:
            saved = json.load(f)
        positions = [saved["joints"][j] for j in JOINT_NAMES]
        print(f"Go to target location {path}...")
        return self.go_to_joint_positions(positions)

    def plan_and_execute(self, dx, dy, dz, d_yaw_rad=0.0):
        current_positions = self.wait_for_joint_state()
        current_pose = self.get_current_pose()
        print(f"Current pose: x={current_pose.position.x:.4f} y={current_pose.position.y:.4f} "
              f"z={current_pose.position.z:.4f}")

        target = Pose()
        target.position.x = current_pose.position.x + dx
        target.position.y = current_pose.position.y + dy
        target.position.z = current_pose.position.z + dz

        cur_q = (current_pose.orientation.x, current_pose.orientation.y,
                  current_pose.orientation.z, current_pose.orientation.w)
        if d_yaw_rad != 0.0:
            new_q = quat_multiply(cur_q, quat_from_z_angle(d_yaw_rad))
        else:
            new_q = cur_q
        target.orientation.x, target.orientation.y, target.orientation.z, target.orientation.w = new_q

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
        print(f"fraction={response.fraction:.3f}")
        if response.fraction < MIN_FRACTION:
            print(f"Only {response.fraction:.1%} plannable -- not executing")
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = response.solution.joint_trajectory
        future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            print("Trajectory goal rejected")
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return True

    def jog_joint(self, joint_name, delta_rad):
        """Direct joint-space nudge, bypassing Cartesian IK entirely."""
        current_positions = self.wait_for_joint_state()
        idx = JOINT_NAMES.index(joint_name)
        target = list(current_positions)
        target[idx] += delta_rad
        print(f"Jogging {joint_name}: {current_positions[idx]:.4f} -> {target[idx]:.4f} rad")

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = target
        point.time_from_start.sec = int(JOINT_MOVE_TIME_SEC)
        goal.trajectory.points = [point]

        future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            print("Trajectory goal rejected")
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return True


def main():
    rclpy.init()
    node = MoveToBlocks()
    node.cartesian_client.wait_for_service(timeout_sec=5.0)
    node.arm_client.wait_for_server(timeout_sec=5.0)

    print(f"""
Commands (watch the camera between each one):
  f = move forward {FORWARD_STEP*100:.0f}cm toward the blocks (+x)
  s = side {SIDE_STEP*100:.0f}cm align with the blocks (+y)
  d = descend {abs(DESCEND_STEP)*100:.0f}cm
  r = rotate gripper +{ROTATE_STEP_DEG:.0f} deg about its own approach axis
  R = rotate gripper -{ROTATE_STEP_DEG:.0f} deg
  e = jog wrist_2_joint +{ELBOW_STEP_DEG:.0f} deg (joint-space, bypasses Cartesian IK)
  E = jog wrist_2_joint -{ELBOW_STEP_DEG:.0f} deg
  q = quit
  g = save current pose (joints + Cartesian x,y,z) to goal_pose.json
  h = return home (saved_pose.json)
  t = target (goal_pose.json)
""")

    while True:
        cmd = input("Command [f/s/d/r/R/e/E/q/g/t]: ").strip()
        if cmd == "q":
            break
        elif cmd == "f":
            ok = node.plan_and_execute(0.0, FORWARD_STEP, 0.0)
        elif cmd == "s":
            ok = node.plan_and_execute(-SIDE_STEP, 0.0, 0.0)
        elif cmd == "d":
            ok = node.plan_and_execute(0.0, 0.0, DESCEND_STEP)
        elif cmd == "r":
            ok = node.plan_and_execute(0.0, 0.0, 0.0, d_yaw_rad=math.radians(ROTATE_STEP_DEG))
        elif cmd == "R":
            ok = node.plan_and_execute(0.0, 0.0, 0.0, d_yaw_rad=-math.radians(ROTATE_STEP_DEG))
        elif cmd == "e":
            ok = node.jog_joint("wrist_2_joint", math.radians(ELBOW_STEP_DEG))
        elif cmd == "E":
            ok = node.jog_joint("wrist_2_joint", -math.radians(ELBOW_STEP_DEG))
        elif cmd == "g":
            node.save_current_pose()
            continue
        elif cmd == "h":
            ok = node.go_home()
        elif cmd == "t":
            ok = node.go_to_target()
        else:
            print("Unknown command.")
            continue
        if not ok:
            print("Step failed to plan/execute.")

    rclpy.shutdown()


if __name__ == "__main__":
    main()