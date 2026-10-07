#!/usr/bin/env bash
# fix_moveit_urdf.sh  (RUNS ON THE REMOTE CONSTRUCT ROS2 SESSION, not here)
#
# Redoes the combined UR3e+Robotiq85 URDF fix from scratch -- needed after
# EVERY reconnect, since Construct spins up a fresh container that wipes
# both apt-installed packages and any previously captured URDF file (see
# memory ur3e_sim2real.md's "Operational gotcha", 2026-08-25). Without
# this, `ros2 launch ur_moveit_config ur_moveit.launch.py` fails with a
# joint-mismatch error citing robotiq_85_* joints not being in the loaded
# 'ur' model -- the combined URDF isn't built from anything in the
# filesystem, it only exists live on the ROS2 graph via /robot_description
# (published by a driver container this session can't see the process
# list of), so it has to be captured fresh each time.
#
# Run this once after each reconnect, BEFORE trying to launch MoveIt.
#
# Usage:
#   chmod +x fix_moveit_urdf.sh   # first time only
#   ./fix_moveit_urdf.sh

set -euo pipefail

# Running as ./fix_moveit_urdf.sh spawns a non-interactive bash process,
# which does NOT source ~/.bashrc -- so the ROS2 environment (what makes
# `rclpy` importable, among other things) never loads here even though it
# works fine typed directly at the interactive prompt. Source it
# explicitly rather than relying on it being inherited.
source /opt/ros/humble/setup.bash

echo "[1/4] Re-installing packages wiped by the fresh container..."
sudo apt install -y ros-humble-ur-description ros-humble-ur-moveit-config

echo "[2/4] Capturing live /robot_description -> /tmp/combined.urdf (waiting for it to publish)..."
python3 -c "
import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String

qos = QoSProfile(depth=1)
qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
qos.reliability = ReliabilityPolicy.RELIABLE

class Dump(Node):
    def __init__(self):
        super().__init__('dump_urdf')
        self.create_subscription(String, '/robot_description', self.cb, qos)
    def cb(self, msg):
        with open('/tmp/combined.urdf', 'w') as f:
            f.write(msg.data)
        self.get_logger().info('Saved.')
        rclpy.shutdown()

rclpy.init()
node = Dump()
try:
    rclpy.spin_until_future_complete(node, rclpy.Future(), timeout_sec=15.0) if False else None
    # spin with a manual timeout since spin() alone blocks forever if the
    # publisher never shows up (e.g. driver container not up yet)
    import time
    deadline = time.time() + 15.0
    while rclpy.ok() and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.5)
except KeyboardInterrupt:
    pass
finally:
    if rclpy.ok():
        rclpy.shutdown()
"

if [ ! -s /tmp/combined.urdf ]; then
    echo "ERROR: /tmp/combined.urdf is empty or missing -- /robot_description never published."
    echo "Check that the robot driver container is actually up before retrying."
    exit 1
fi

echo "[3/4] Verifying it's actually the combined model (expect ~74 robotiq matches)..."
count=$(grep -c robotiq /tmp/combined.urdf || true)
echo "  found $count robotiq references"
if [ "$count" -lt 1 ]; then
    echo "ERROR: captured URDF has no robotiq joints -- got the arm-only model, not the combined one."
    exit 1
fi

echo "[4/4] Installing into ur_description's share dir..."
target_dir="$(ros2 pkg prefix ur_description)/share/ur_description/urdf"
sudo cp /tmp/combined.urdf "$target_dir/combined_ur3e_robotiq.urdf.xacro"
echo "  -> $target_dir/combined_ur3e_robotiq.urdf.xacro"

echo ""
echo "Done. Now launch MoveIt with:"
echo "  ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur3e launch_rviz:=false description_file:=combined_ur3e_robotiq.urdf.xacro"
