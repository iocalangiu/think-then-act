#!/usr/bin/env python3
"""
record_camera_snapshot.py  (RUNS ON THE REMOTE CONSTRUCT ROS2 SESSION, not
here -- see run_align_xy_real.py's docstring for why real-robot scripts live
there instead of locally.)

Quick visual + numeric sanity check for the camera on the real UR3e rig (see
memory ur3e_sim2real.md -- this rig had NO camera topic at all until a Zenoh
bridge was set up to pipe it in). Grabs one message from
/camera/depth/color/points, decodes XYZ+RGB per point, prints range stats,
and writes an image so you can eyeball what the camera sees without a GUI.

There is no separate 2D image topic on this rig (only
/camera/depth/camera_info + /camera/depth/color/points), so the only way to
get a picture is from the point cloud. Two cases:
  - Organized cloud (height > 1, one point per pixel, row-major): reshape the
    per-point RGB straight into (height, width, 3) -- exact, lossless image.
  - Unorganized cloud (height == 1, flat list -- e.g. what the Zenoh bridge
    delivers): no pixel grid exists anymore, so this falls back to
    rasterizing an approximate top-down image from the points' own x/y
    spatial extent. This is a sanity-check projection, not the literal
    camera image -- orientation/aspect are not guaranteed to match reality.

Usage:
  python3 record_camera_snapshot.py [output.png] [--topic TOPIC]

Writes a PNG if Pillow is installed on the Construct session, otherwise
falls back to a .ppm (zero extra dependencies -- open with `xdg-open`,
`eog`, or `convert snap.ppm snap.png`).
"""
import argparse
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

DEFAULT_TOPIC = "/camera/depth/color/points"
DEFAULT_OUTPUT = "camera_snapshot.png"


def _field_offset(fields, name: str) -> int:
    for f in fields:
        if f.name == name:
            return f.offset
    raise RuntimeError(
        f"No '{name}' field in point cloud fields: {[f.name for f in fields]}"
    )


def decode_points(msg: PointCloud2):
    """Returns (xyz float32 Nx3, rgb uint8 Nx3, valid bool mask, organized bool)."""
    x_off = _field_offset(msg.fields, "x")
    rgb_off = _field_offset(msg.fields, "rgb")
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)

    xyz = raw[:, x_off:x_off + 12].copy().view(np.float32).reshape(-1, 3)

    # rgb is packed as a float32 whose bytes are actually [B, G, R, pad] (or
    # [R, G, B, pad] depending on driver) -- read as uint32 and unpack bytes
    # directly instead of round-tripping through float to avoid NaN issues.
    packed = raw[:, rgb_off:rgb_off + 4].copy().view(np.uint32).reshape(-1)
    r = ((packed >> 16) & 0xFF).astype(np.uint8)
    g = ((packed >> 8) & 0xFF).astype(np.uint8)
    b = (packed & 0xFF).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)

    valid = np.isfinite(xyz).all(axis=1)
    organized = msg.height > 1
    return xyz, rgb, valid, organized


def rasterize_top_down(xyz: np.ndarray, rgb: np.ndarray, out_width=640, out_height=480) -> np.ndarray:
    """Approximate image from unorganized points' x/y spatial extent.

    Not the literal camera image (no pixel grid survives an unorganized
    cloud) -- just enough to eyeball whether recognizable shapes/colors show
    up where you'd expect the workspace to be.
    """
    x, y = xyz[:, 0], xyz[:, 1]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)
    col = ((x - x_min) / x_range * (out_width - 1)).astype(np.int32)
    row = ((y - y_min) / y_range * (out_height - 1)).astype(np.int32)
    image = np.full((out_height, out_width, 3), 40, dtype=np.uint8)
    image[row, col] = rgb
    return image


def write_image(image: np.ndarray, path: str) -> str:
    try:
        from PIL import Image
        Image.fromarray(image, mode="RGB").save(path)
        return path
    except ImportError:
        ppm_path = path.rsplit(".", 1)[0] + ".ppm"
        height, width, _ = image.shape
        with open(ppm_path, "wb") as f:
            f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            f.write(image.tobytes())
        return ppm_path


class SnapshotGrabber(Node):
    def __init__(self, topic: str):
        super().__init__("camera_snapshot_grabber")
        self.msg = None
        self.create_subscription(PointCloud2, topic, self._on_cloud, 1)

    def _on_cloud(self, msg: PointCloud2) -> None:
        self.msg = msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    args = parser.parse_args()

    rclpy.init()
    node = SnapshotGrabber(args.topic)

    node.get_logger().info(f"Waiting for one message on {args.topic} ...")
    deadline = node.get_clock().now().nanoseconds + int(args.timeout_sec * 1e9)
    while node.msg is None and node.get_clock().now().nanoseconds < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)

    if node.msg is None:
        node.get_logger().error(
            f"No message received on {args.topic} within {args.timeout_sec}s -- "
            "is the camera driver running? (`ros2 topic hz " + args.topic + "`)"
        )
        rclpy.shutdown()
        sys.exit(1)

    msg = node.msg
    xyz, rgb, valid, organized = decode_points(msg)
    n_total = xyz.shape[0]
    n_valid = int(valid.sum())
    node.get_logger().info(
        f"{n_total} points total, {n_valid} valid (non-NaN). "
        f"organized={organized} (height={msg.height}, width={msg.width})"
    )

    if n_valid == 0:
        node.get_logger().error(
            "All points are NaN -- camera may be out of range, occluded, or the "
            "bridge is forwarding an empty cloud."
        )
        rclpy.shutdown()
        sys.exit(1)

    vxyz = xyz[valid]
    node.get_logger().info(
        f"x range [{vxyz[:, 0].min():.3f}, {vxyz[:, 0].max():.3f}]  "
        f"y range [{vxyz[:, 1].min():.3f}, {vxyz[:, 1].max():.3f}]  "
        f"z (depth) range [{vxyz[:, 2].min():.3f}, {vxyz[:, 2].max():.3f}] meters"
    )

    if organized:
        image = rgb.reshape(msg.height, msg.width, 3)
    else:
        node.get_logger().warning(
            "Cloud is unorganized -- rasterizing an approximate top-down "
            "projection from point positions, not the literal camera image."
        )
        image = rasterize_top_down(vxyz, rgb[valid])

    out_path = write_image(image, args.output)
    node.get_logger().info(f"Wrote {image.shape[1]}x{image.shape[0]} snapshot to {out_path}")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
