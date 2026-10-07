"""Lightweight review video of JUST the width-axis projection panel across
a whole session -- no 3D skeleton rendering, no real-camera footage, so
it's much faster than the full synced/skeleton video and useful for
checking the projection panel's behavior on its own before committing to
a full re-render.

One output frame per recorded cloud (not real-time-matched to any video),
at a fixed review-pace fps.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import cv2  # noqa: E402

from extract_gripper_brick_geometry import load_cloud, BrickTracker  # noqa: E402
from make_geometry_videos import (  # noqa: E402
    compute_off_plane_clusters, render_width_projection_panel, load_pose_log, angles_for_cloud,
    FIG_SIZE_PX, PROJECTION_PANEL_WIDTH_PX,
)
import numpy as np

REVIEW_FPS = 2.0


def make_projection_video(session_dir: Path):
    clouds = sorted(session_dir.glob("clouds/*.npz"), key=lambda p: float(p.stem))
    if not clouds:
        print(f"  {session_dir}: no clouds found, skipping")
        return
    pose_rows = load_pose_log(session_dir)

    out_path = session_dir / "projections_only.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), REVIEW_FPS,
                              (PROJECTION_PANEL_WIDTH_PX, FIG_SIZE_PX[1]))

    brick_tracker = BrickTracker()
    n_written, n_skipped_no_pose = 0, 0
    for c in clouds:
        xyz = load_cloud(c)
        off_plane, cluster_info = compute_off_plane_clusters(xyz)
        tracked_brick, brick_mode = brick_tracker.update(cluster_info or [])

        angles = angles_for_cloud(pose_rows, float(c.stem))
        if angles is None:
            n_skipped_no_pose += 1
            continue

        is_brick = np.zeros(off_plane.shape[0], dtype=bool) if off_plane is not None else np.zeros(0, dtype=bool)
        if tracked_brick is not None and len(tracked_brick["idx"]) > 0:
            is_brick[tracked_brick["idx"]] = True

        frame = render_width_projection_panel(off_plane, is_brick, angles,
                                                PROJECTION_PANEL_WIDTH_PX, FIG_SIZE_PX[1])
        writer.write(frame)
        n_written += 1

    writer.release()
    print(f"  wrote {out_path} ({n_written}/{len(clouds)} frames written, "
          f"{n_skipped_no_pose} skipped for missing pose, {REVIEW_FPS:.0f}fps)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="A session directory (must contain clouds/ and pose_log.jsonl).")
    args = parser.parse_args()
    make_projection_video(Path(args.path))


if __name__ == "__main__":
    main()
