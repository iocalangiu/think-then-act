#!/usr/bin/env python3
"""
Minimal moveit_servo test: streams a small constant Cartesian velocity via
/servo_node/delta_twist_cmds for a short fixed duration, then stops.
Requires forward_position_controller active (not scaled_joint_trajectory_controller).
"""
import time
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Int8

RATE_HZ = 50.0
DURATION_SEC = 1.0
LINEAR_VELOCITY = [0.2, 0.0, 0.0]  # m/s -- small, +z
FRAME_ID = "base_link"


class ServoTest(Node):
    def __init__(self):
        super().__init__("servo_test")
        self.twist_pub = self.create_publisher(TwistStamped, "/servo_node/delta_twist_cmds", 10)
        self.start_client = self.create_client(Trigger, "/servo_node/start_servo")
        self.stop_client = self.create_client(Trigger, "/servo_node/stop_servo")
        self.last_status = None
        self.create_subscription(Int8, "/servo_node/status", self._on_status, 10)

    def _on_status(self, msg):
        self.last_status = msg.data

    def call_trigger(self, client, name):
        client.wait_for_service(timeout_sec=5.0)
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        self.get_logger().info(f"{name}: success={result.success} message={result.message}")

    def stream(self, duration_sec, linear):
        dt = 1.0 / RATE_HZ
        steps = int(duration_sec * RATE_HZ)
        for _ in range(steps):
            msg = TwistStamped()
            msg.header.frame_id = FRAME_ID
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = linear
            self.twist_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            if self.last_status not in (None, 0):
                # Nonzero = some kind of warning (deceleration/halt for
                # singularity or collision) -- exact code meanings not
                # verified against this specific moveit_servo build, but
                # 0 always means nominal, no warning.
                self.get_logger().warn(f"servo status code={self.last_status} (nonzero = warning)")
            time.sleep(dt)


def main():
    rclpy.init()
    node = ServoTest()
    answer = input(f"Stream {LINEAR_VELOCITY} m/s for {DURATION_SEC}s on the real robot? [y/N] ")
    if answer.strip().lower() != "y":
        print("Aborted.")
        rclpy.shutdown()
        return
    node.call_trigger(node.start_client, "start_servo")
    time.sleep(0.5)
    node.stream(DURATION_SEC, LINEAR_VELOCITY)
    node.call_trigger(node.stop_client, "stop_servo")
    rclpy.shutdown()


if __name__ == "__main__":
    main()