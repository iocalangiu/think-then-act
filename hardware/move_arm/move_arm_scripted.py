#!/usr/bin/env python3

import argparse

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import GripperCommand

ARM_VELOCITY = 0.05       # m/s — slow on purpose for a first run
ARM_ACCELERATION = 0.05   # m/s^2
GRIPPER_ACTION_NAME = "/robotiq_gripper_controller/gripper_cmd"
GRIPPER_CLOSED_POSITION = 0.8  # UNVERIFIED — see prerequisites above
GRIPPER_MAX_EFFORT = 10.0

def move_arm(robot_ip: str, dx: float, dy: float, dz: float) -> None:
    import rtde_control
    import rtde_receive

    rtde_c = rtde_control.RTDEControlInterface(robot_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
    try: 
        current_pose = np.array(rtde_r.getActualTCPPose())
        target_pose = current_pose.copy()
        target_pose[:3] += np.array([dx,dy,dz])
        print(f'Current TCP xyz: {current_pose[:3]}')
        print(f'Target TCP xyz: {target_pose[:3]}')
        rtde_c.moveL(target_pose.tolist(), ARM_VELOCITY, ARM_ACCELERATION)
    finally:
        rtde_c.stopScript()
        rtde_c.disconnect()
        rtde_r.disconnect()



class GripperCloser(Node):
    def __init__(self):
        super().__init__('ur3e_gripper_closer')
        self._client = ActionClient(self, GripperCommand, GRIPPER_ACTION_NAME)

    def close(self, position: float, max_effort: float) -> None:
        if not self._client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(
                f'Gripper action server {GRIPPER_ACTION_NAME} not available')
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = max_effort
        self.get_logger().info(f'Closing gripper: position = {position}')
        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            raise RuntimeError('Gripper goal rejected')

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info('Gripper close complete.')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_ip', required=True)
    parser.add_argument('--dx', type=float, default=0.0)
    parser.add_argument('--dy', type=float, default=0.0)
    parser.add_argument('--dz', type=float, default=0.04)
    args = parser.parse_args()

    move_arm(args.robot_ip, args.dx, args.dy, args.dz)

    rclpy.init()
    node = GripperCloser()
    try:
        node.close(GRIPPER_CLOSED_POSITION, GRIPPER_MAX_EFFORT)
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
