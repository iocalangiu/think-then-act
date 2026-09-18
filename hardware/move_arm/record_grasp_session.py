#!/usr/bin/env python3
"""
record_grasp_session.py  (RUNS ON THE REMOTE CONSTRUCT ROS2 SESSION, not
here -- see run_align_xy_real.py's docstring for why real-robot scripts live
there instead of locally.)

Step 2 of the camera-guided fine-control policy pivot (see memory
ur3e_sim2real.md). Deliberately does NOT move the arm or command the
gripper itself -- it's a passive recorder, meant to run in one terminal
while you jog in another with move_to_blocks.py and trigger the actual
gripper close/open with the already-confirmed `ros2 action send_goal
/robotiq_gripper_controller/gripper_cmd control_msgs/action/GripperCommand
"{command: {position: 1, max_effort: 20.0}}"` (see run_pick_sequence_real.py's
GRIPPER INTERFACE note) in a third terminal, or however you're driving it
tonight. No ActionClients here at all, so this script can never contend
with either of those for a goal.

Just raw data collection, no live labeling or segmentation -- that's
deferred to an offline pass over whatever gets saved here. Continuously,
from start until Ctrl+C:
  - saves a point-cloud snapshot every --cloud-period seconds to
    <session_dir>/clouds/<timestamp>.npz  ({xyz, rgb})
  - appends every wrench reading (any forces) to
    <session_dir>/wrench_log.jsonl, flushed immediately after every write
  - appends full arm joint state (shoulder/elbow/wrist -- whatever
    /joint_states reports, not filtered to a fixed name list) AND gripper
    joint state (/gripper/joint_states, same treatment) every
    --cloud-period seconds to <session_dir>/pose_log.jsonl, also flushed
    immediately, plus a best-effort TF base_link->tool0 grip_pos alongside
    them as a convenience derived quantity

Needs record_camera_snapshot.py copied to the same directory (reuses
decode_points).

Usage:
  python3 record_grasp_session.py [--session-dir grasp_sessions/session_1]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState, PointCloud2
from geometry_msgs.msg import WrenchStamped
from tf2_ros import Buffer, TransformListener

from record_camera_snapshot import decode_points

BASE_FRAME = "base_link"
LINK_NAME = "tool0"
CLOUD_TOPIC = "/camera/depth/color/points"
WRENCH_TOPIC = "/force_torque_sensor_broadcaster/wrench"
ARM_JOINT_STATE_TOPIC = "/joint_states"
GRIPPER_JOINT_STATE_TOPIC = "/gripper/joint_states"  # confirmed to exist (memory, 2026-08-25),
                                                       # exact field names not yet confirmed --
                                                       # logged raw/whole, not assumed


def force_magnitude(fx, fy, fz):
    return float((fx * fx + fy * fy + fz * fz) ** 0.5)


class SessionRecorder(Node):
    def __init__(self, cloud_topic, wrench_topic, arm_topic, gripper_topic, session_dir, cloud_period):
        super().__init__("record_grasp_session")
        self.session_dir = session_dir
        self.cloud_period = cloud_period
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._arm_joint_state = None      # (names list, positions list)
        self._gripper_joint_state = None  # (names list, positions list)
        self._latest_wrench = None        # (t, fx, fy, fz, tx, ty, tz)
        self._n_clouds_saved = 0
        self._n_pose_logged = 0
        self._last_cloud_save = 0.0
        self._last_pose_log = 0.0

        (self.session_dir / "clouds").mkdir(parents=True, exist_ok=True)
        self.wrench_log_f = open(self.session_dir / "wrench_log.jsonl", "a", encoding="utf-8")
        self.pose_log_f = open(self.session_dir / "pose_log.jsonl", "a", encoding="utf-8")

        self.create_subscription(JointState, arm_topic, self._on_arm_joint_state, 10)
        self.create_subscription(JointState, gripper_topic, self._on_gripper_joint_state, 10)
        self.create_subscription(PointCloud2, cloud_topic, self._on_cloud, 1)
        self.create_subscription(WrenchStamped, wrench_topic, self._on_wrench, 50)

    def _on_arm_joint_state(self, msg) -> None:
        self._arm_joint_state = (list(msg.name), list(msg.position))

    def _on_gripper_joint_state(self, msg) -> None:
        self._gripper_joint_state = (list(msg.name), list(msg.position))

    def _on_cloud(self, msg) -> None:
        now = time.time()
        if now - self._last_cloud_save >= self.cloud_period:
            self._last_cloud_save = now
            self._save_cloud_snapshot(msg, now)

    def _save_cloud_snapshot(self, msg, t) -> None:
        xyz, rgb, valid, _organized = decode_points(msg)
        xyz, rgb = xyz[valid], rgb[valid]
        np.savez_compressed(self.session_dir / "clouds" / f"{t:.3f}.npz", xyz=xyz, rgb=rgb)
        self._n_clouds_saved += 1

    def _on_wrench(self, msg) -> None:
        f, t_ = msg.wrench.force, msg.wrench.torque
        now = time.time()
        self._latest_wrench = (now, f.x, f.y, f.z, t_.x, t_.y, t_.z)
        self.wrench_log_f.write(json.dumps(self._latest_wrench) + "\n")
        self.wrench_log_f.flush()

    def maybe_log_pose(self) -> None:
        now = time.time()
        if now - self._last_pose_log < self.cloud_period:
            return
        self._last_pose_log = now
        try:
            tf = self.tf_buffer.lookup_transform(BASE_FRAME, LINK_NAME, Time())
            gp = tf.transform.translation
            grip_pos = [gp.x, gp.y, gp.z]
        except Exception:
            grip_pos = None
        arm_names, arm_positions = self._arm_joint_state if self._arm_joint_state else (None, None)
        grip_names, grip_positions = self._gripper_joint_state if self._gripper_joint_state else (None, None)
        row = {
            "t": now,
            "arm_joint_names": arm_names,
            "arm_joint_positions": arm_positions,
            "gripper_joint_names": grip_names,
            "gripper_joint_positions": grip_positions,
            "grip_pos": grip_pos,
        }
        self.pose_log_f.write(json.dumps(row) + "\n")
        self.pose_log_f.flush()
        self._n_pose_logged += 1

    def close(self) -> None:
        self.wrench_log_f.close()
        self.pose_log_f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud-topic", default=CLOUD_TOPIC)
    parser.add_argument("--wrench-topic", default=WRENCH_TOPIC)
    parser.add_argument("--arm-joint-topic", default=ARM_JOINT_STATE_TOPIC)
    parser.add_argument("--gripper-joint-topic", default=GRIPPER_JOINT_STATE_TOPIC)
    parser.add_argument("--session-dir", default=None,
                         help="Default: grasp_sessions/session_<unix time> (new each run).")
    parser.add_argument("--cloud-period", type=float, default=1.0,
                         help="Seconds between continuous point-cloud/pose snapshots saved to disk.")
    args = parser.parse_args()

    session_dir = Path(args.session_dir) if args.session_dir else Path("grasp_sessions") / f"session_{int(time.time())}"
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"Recording to {session_dir}")

    rclpy.init()
    node = SessionRecorder(args.cloud_topic, args.wrench_topic, args.arm_joint_topic,
                            args.gripper_joint_topic, session_dir, args.cloud_period)

    print("""
Recording points + full arm state (shoulder/elbow/wrist) + gripper joint
state + wrench (forces) continuously. This script does NOT move the arm or
the gripper -- jog with move_to_blocks.py (or equivalent) in another
terminal, and close/open the gripper however you're driving it tonight.

Ctrl+C to stop.
""")

    last_heartbeat = 0.0
    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.05)
            node.maybe_log_pose()

            now = time.time()
            if now - last_heartbeat > 5.0:
                last_heartbeat = now
                if node._latest_wrench is not None:
                    age = now - node._latest_wrench[0]
                    force_str = f"|F|={force_magnitude(*node._latest_wrench[1:4]):.2f}N (age {age:.1f}s)"
                else:
                    force_str = "no wrench data yet"
                gripper_str = "gripper: seen" if node._gripper_joint_state else "gripper: NOT seen yet"
                print(f"[recording] clouds={node._n_clouds_saved} poses={node._n_pose_logged} "
                      f"{force_str} {gripper_str}")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        node.close()
        try:
            rclpy.shutdown()
        except Exception as e:
            # Ctrl+C's SIGINT handling can already tear down the rclpy
            # context before this explicit call runs -- harmless, all data
            # was already flushed to disk write-by-write above, not
            # buffered until shutdown.
            print(f"(rclpy.shutdown() raised {e!r} -- harmless, data was already flushed to disk)")
        print(f"\nSession saved to {session_dir} "
              f"({node._n_clouds_saved} clouds, {node._n_pose_logged} pose samples).")


if __name__ == "__main__":
    main()
