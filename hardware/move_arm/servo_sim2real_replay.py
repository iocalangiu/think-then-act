#!/usr/bin/env python3
"""
Sim2real replay via moveit_servo -- v2, fixes a bug found in v1: doing a
TF lookup (get_position) between each step's streaming loop introduced a
gap in the command stream, which likely tripped Servo's stale-command
watchdog and silently killed motion (confirmed: same-magnitude velocity
via servo_test.py's uninterrupted loop DID move the robot). This version
streams the WHOLE sequence as one continuous, uninterrupted publish loop
-- velocity value changes at step boundaries, nothing ever blocks in
between -- and only reads position via TF once at the start and once at
the end.
"""
import argparse
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_srvs.srv import Trigger
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Int8
from tf2_ros import Buffer, TransformListener

RATE_HZ = 50.0
STEP_DURATION = 1.0
FRAME_ID = "base_link"
LINK_NAME = "tool0"
BASE_FRAME = "base_link"

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


class ServoReplay(Node):
    def __init__(self):
        super().__init__("servo_sim2real_replay")
        self.twist_pub = self.create_publisher(TwistStamped, "/servo_node/delta_twist_cmds", 10)
        self.start_client = self.create_client(Trigger, "/servo_node/start_servo")
        self.stop_client = self.create_client(Trigger, "/servo_node/stop_servo")
        self.last_status = None
        self.create_subscription(Int8, "/servo_node/status", self._on_status, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _on_status(self, msg):
        self.last_status = msg.data

    def call_trigger(self, client, name):
        client.wait_for_service(timeout_sec=5.0)
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        self.get_logger().info(f"{name}: success={result.success} message={result.message}")

    def get_position(self):
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                tf = self.tf_buffer.lookup_transform(BASE_FRAME, LINK_NAME, Time())
                t = tf.transform.translation
                return (t.x, t.y, t.z)
            except Exception:
                rclpy.spin_once(self, timeout_sec=0.2)
        raise RuntimeError("Could not look up base_link -> tool0 transform")

    def stream_all(self, steps, step_duration):
        dt = 1.0 / RATE_HZ
        ticks_per_step = int(step_duration * RATE_HZ)
        for i, (dx, dy, dz) in enumerate(steps):
            vx, vy, vz = dx / step_duration, dy / step_duration, dz / step_duration
            for _ in range(ticks_per_step):
                msg = TwistStamped()
                msg.header.frame_id = FRAME_ID
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = vx, vy, vz
                self.twist_pub.publish(msg)
                rclpy.spin_once(self, timeout_sec=0.0)
                if self.last_status not in (None, 0):
                    self.get_logger().warn(f"servo status code={self.last_status} (nonzero = warning)")
                time.sleep(dt)
            self.get_logger().info(f"Step {i} done: sim_delta=({dx:.4f},{dy:.4f},{dz:.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", type=int, default=5)
    args = parser.parse_args()

    steps = SIM_DELTAS[:args.num_steps]
    net = [sum(d[i] for d in steps) for i in range(3)]
    net_mag = math.sqrt(sum(x * x for x in net))
    print(f"Replaying {len(steps)} steps via moveit_servo (continuous stream). "
          f"Net intended displacement: dx={net[0]:.4f} dy={net[1]:.4f} dz={net[2]:.4f} (mag={net_mag:.4f}m)")

    rclpy.init()
    node = ServoReplay()

    answer = input("Execute this sequence via moveit_servo on the real robot? [y/N] ")
    if answer.strip().lower() != "y":
        print("Aborted.")
        rclpy.shutdown()
        return

    pos_before = node.get_position()
    node.call_trigger(node.start_client, "start_servo")
    time.sleep(0.5)
    node.stream_all(steps, STEP_DURATION)
    node.call_trigger(node.stop_client, "stop_servo")
    pos_after = node.get_position()

    total_actual = tuple(pos_after[j] - pos_before[j] for j in range(3))
    node.get_logger().info(
        f"TOTAL: sim_net=({net[0]:.4f},{net[1]:.4f},{net[2]:.4f}) "
        f"real_net=({total_actual[0]:.4f},{total_actual[1]:.4f},{total_actual[2]:.4f})"
    )
    rclpy.shutdown()


if __name__ == "__main__":
    main()