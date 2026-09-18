#!/usr/bin/env python3
"""
wrench_topic_sanity_check.py  (RUNS ON THE REMOTE CONSTRUCT ROS2 SESSION,
not here -- see run_align_xy_real.py's docstring for why real-robot scripts
live there instead of locally.)

Read-only gate for the meta-RL force-wiring experiment (see memory:
ur3e_sim2real, meta_rl_sim2real_direction). Before anything downstream
(robot/wrench_force_bridge.py, or any real-robot rollout script consuming
force) is trusted, THIS script must report a PASS with a steady, nonzero
message rate on /force_torque_sensor_broadcaster/wrench.

Deliberately has NO ActionClient, no rtde_control, no joint-trajectory code
at all -- structurally incapable of moving the arm or the gripper, unlike
every other script in this directory. Just subscribes, prints a heartbeat,
and exits with a clear PASS/FAIL.

Modeled on record_grasp_session.py's WRENCH_TOPIC / _on_wrench /
force_magnitude pattern and ros2_smoke_test_ur3e.py's bare
rclpy.init()/spin_once/shutdown() loop shape.

Usage:
  python3 wrench_topic_sanity_check.py [--duration 5.0] [--wrench-topic /force_torque_sensor_broadcaster/wrench]
"""
import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped

WRENCH_TOPIC = "/force_torque_sensor_broadcaster/wrench"  # same constant as record_grasp_session.py


def force_magnitude(fx, fy, fz) -> float:
    return float((fx * fx + fy * fy + fz * fz) ** 0.5)


class WrenchSanityCheck(Node):
    def __init__(self, wrench_topic: str):
        super().__init__("wrench_topic_sanity_check")
        self._latest_wrench = None   # (t, fx, fy, fz, tx, ty, tz)
        self._n_received = 0
        self.create_subscription(WrenchStamped, wrench_topic, self._on_wrench, 50)

    def _on_wrench(self, msg) -> None:
        f, t_ = msg.wrench.force, msg.wrench.torque
        self._latest_wrench = (time.time(), f.x, f.y, f.z, t_.x, t_.y, t_.z)
        self._n_received += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrench-topic", default=WRENCH_TOPIC)
    parser.add_argument("--duration", type=float, default=5.0,
                         help="Seconds to listen before reporting PASS/FAIL.")
    args = parser.parse_args()

    rclpy.init()
    node = WrenchSanityCheck(args.wrench_topic)

    print(f"Listening on {args.wrench_topic} for {args.duration:.1f}s "
          f"(no motion, no ActionClient -- this script cannot move the arm)...")

    start = time.time()
    last_heartbeat = 0.0
    try:
        while time.time() - start < args.duration:
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.time()
            if now - last_heartbeat >= 0.5:
                last_heartbeat = now
                if node._latest_wrench is not None:
                    t, fx, fy, fz, tx, ty, tz = node._latest_wrench
                    elapsed = max(now - start, 1e-6)
                    hz = node._n_received / elapsed
                    print(f"  [{elapsed:5.1f}s] Fx={fx:+7.3f} Fy={fy:+7.3f} Fz={fz:+7.3f}  "
                          f"|F|={force_magnitude(fx, fy, fz):6.3f}N  age={now - t:5.3f}s  "
                          f"n={node._n_received}  ~{hz:.1f}Hz")
                else:
                    print(f"  [{now - start:5.1f}s] no message received yet...")
    finally:
        rclpy.shutdown()

    if node._n_received == 0:
        print(f"\nFAIL: 0 messages received on {args.wrench_topic} in {args.duration:.1f}s -- "
              f"check `ros2 topic list` / `ros2 topic hz {args.wrench_topic}` before trusting "
              f"anything downstream of this topic (robot/wrench_force_bridge.py, any real-robot "
              f"force-conditioned rollout).")
        sys.exit(1)

    hz = node._n_received / args.duration
    print(f"\nPASS: {node._n_received} messages in {args.duration:.1f}s (~{hz:.1f} Hz).")


if __name__ == "__main__":
    main()
