#!/usr/bin/env python3
"""
cluster_blocks_by_color.py  (RUNS ON THE REMOTE CONSTRUCT ROS2 SESSION, not
here -- see run_align_xy_real.py's docstring for why real-robot scripts live
there instead of locally.)

Grabs one message from the camera point cloud, drops low-saturation points
(table/wood/wall are all fairly gray/tan), buckets the remaining "colorful"
points into red/yellow/blue/green by dominant channel, clusters each color's
points by 3D proximity, and prints a centroid + point count per cluster.

Exists because record_camera_snapshot.py's raster (2026-09-04) showed the
cloud from this rig's Zenoh bridge is sparse -- an eyeballed picture doesn't
reliably show small objects like a single brick. This sidesteps needing a
"picture" at all: for the policy's object_pos/block_dims fields, a
color-filtered 3D centroid is exactly what's needed, whether or not the raw
cloud is dense enough to look like an image.

Usage:
  python3 cluster_blocks_by_color.py [--topic TOPIC] [--min-points N]

Tune SATURATION_THRESHOLD / CLUSTER_EPS / the per-color rules below if the
lighting or block colors on this rig don't match the defaults.
"""
import argparse
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

from record_camera_snapshot import decode_points

DEFAULT_TOPIC = "/camera/depth/color/points"
SATURATION_THRESHOLD = 40   # max(r,g,b) - min(r,g,b); wood/wall points are low-saturation
CLUSTER_EPS = 0.02          # meters; points within this distance join the same cluster
MIN_CLUSTER_POINTS = 5


def classify_color(rgb: np.ndarray) -> np.ndarray:
    """Per-point label in {"red","yellow","blue","green",""} by dominant channel.
    Heuristic, not a real color model -- tune the margins if blocks are
    misclassified or fall through as unlabeled."""
    r, g, b = rgb[:, 0].astype(int), rgb[:, 1].astype(int), rgb[:, 2].astype(int)
    labels = np.full(rgb.shape[0], "", dtype=object)
    red = (r > g + 30) & (r > b + 30)
    blue = (b > r + 30) & (b > g + 30)
    green = (g > r + 30) & (g > b + 30)
    yellow = (r > b + 30) & (g > b + 30) & (np.abs(r - g) < 40)
    labels[red] = "red"
    labels[blue] = "blue"
    labels[green] = "green"
    labels[yellow] = "yellow"
    return labels


def cluster_points(xyz: np.ndarray, eps: float) -> list:
    """Single-linkage clustering via union-find over pairwise distance < eps.
    O(n^2) in blocks of 500 rows -- fine for a few thousand already
    color-filtered points, not meant for a full raw cloud."""
    n = xyz.shape[0]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    block = 500
    for start in range(0, n, block):
        end = min(start + block, n)
        d = np.linalg.norm(xyz[start:end, None, :] - xyz[None, :, :], axis=-1)
        close_i, close_j = np.where(d < eps)
        for i, j in zip(close_i, close_j):
            union(start + i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


class CloudGrabber(Node):
    def __init__(self, topic: str):
        super().__init__("cluster_blocks_by_color")
        self.msg = None
        self.create_subscription(PointCloud2, topic, self._on_cloud, 1)

    def _on_cloud(self, msg: PointCloud2) -> None:
        self.msg = msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--min-points", type=int, default=MIN_CLUSTER_POINTS)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    args = parser.parse_args()

    rclpy.init()
    node = CloudGrabber(args.topic)
    node.get_logger().info(f"Waiting for one message on {args.topic} ...")
    deadline = node.get_clock().now().nanoseconds + int(args.timeout_sec * 1e9)
    while node.msg is None and node.get_clock().now().nanoseconds < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)

    if node.msg is None:
        node.get_logger().error(f"No message received on {args.topic} within {args.timeout_sec}s.")
        rclpy.shutdown()
        sys.exit(1)

    xyz, rgb, valid, _organized = decode_points(node.msg)
    xyz, rgb = xyz[valid], rgb[valid]
    node.get_logger().info(f"{xyz.shape[0]} valid points total.")

    saturation = rgb.max(axis=1).astype(int) - rgb.min(axis=1).astype(int)
    colorful = saturation > SATURATION_THRESHOLD
    node.get_logger().info(
        f"{int(colorful.sum())} points pass saturation > {SATURATION_THRESHOLD} "
        "(dropping low-saturation table/wall/wood points)."
    )

    if colorful.sum() == 0:
        node.get_logger().error(
            "No colorful points found -- either nothing colored is in view, or "
            "SATURATION_THRESHOLD is too strict for this lighting."
        )
        rclpy.shutdown()
        sys.exit(1)

    cxyz, crgb = xyz[colorful], rgb[colorful]
    labels = classify_color(crgb)

    for color in ("red", "yellow", "blue", "green"):
        mask = labels == color
        n_pts = int(mask.sum())
        if n_pts == 0:
            print(f"{color:>6}: 0 points")
            continue
        clusters = cluster_points(cxyz[mask], CLUSTER_EPS)
        clusters = [c for c in clusters if len(c) >= args.min_points]
        print(f"{color:>6}: {n_pts} colorful points -> {len(clusters)} cluster(s) "
              f"with >= {args.min_points} points")
        for c in sorted(clusters, key=len, reverse=True):
            centroid = cxyz[mask][c].mean(axis=0)
            print(f"    centroid=({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})  "
                  f"n_points={len(c)}")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
