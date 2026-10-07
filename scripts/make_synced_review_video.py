"""Three-panel review video for one grasp session, temporally aligned to a
real screen recording of the physical setup: [real camera footage] |
[skeleton/point-cloud panel] | [width-axis projection panel].

The screen recording (a .mov, e.g. from QuickTime) and the recorded point
clouds are on two INDEPENDENT clocks -- the recording has no timestamp of
its own beyond "when the file was created", and the clouds are saved with
Unix-epoch filenames (record_grasp_session.py). Alignment works by reading
the .mov's own filesystem creation date (macOS `mdls`, kMDItemContent-
CreationDate) as the wall-clock moment recording started, then for each
cloud's own epoch timestamp, seeking the video to (cloud_epoch -
video_start_epoch) seconds in. This is only accurate to +-1s (mdls'
creation-date resolution is whole seconds, and note this measures when the
FILE was created, which may lag the physical "hit record" moment by up to
their own overhead) -- fine for a human-review video, not frame-exact.

Only cloud frames that fall within the recording's own [0, duration]
window have real footage to show, so the output is clipped to that
overlap (reported at the end) rather than padding with blank video for
the clouds recorded before/after the screen recording ran.

OUTPUT FRAME RATE (added 2026-09-15, user reported the first version felt
slow/choppy next to the raw screen recording): the point clouds only
arrive at ~0.75Hz, so the skeleton/projection panels can only ever UPDATE
that often -- but the real camera footage has full ~59fps motion available
between those updates that was previously being thrown away (one video
frame sampled per cloud, held via the same slow per-cloud loop). Now the
video is written at OUTPUT_FPS with the real-camera panel re-sampled fresh
every output frame (smooth, matches the raw recording's own motion) while
the skeleton/projection panels are computed once per cloud sample and
HELD (reused) across however many output frames span that cloud's real
time gap -- still only as many matplotlib renders as before, just spread
across a properly smooth-playing container.
"""
import argparse
import bisect
import datetime
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from extract_gripper_brick_geometry import load_cloud, BrickTracker  # noqa: E402
from make_geometry_videos import (  # noqa: E402
    compute_bounds, compute_off_plane_clusters, render_skeleton_frame,
    render_width_projection_panel, hstack_frames, load_pose_log, angles_for_cloud,
    FIG_SIZE_PX, PROJECTION_PANEL_WIDTH_PX,
)

VIDEO_PANEL_HEIGHT_PX = FIG_SIZE_PX[1]  # match the other two panels' height exactly
OUTPUT_FPS = 24.0  # smooth, standard container frame rate -- see module docstring
SPEED_MULTIPLIER = 1.5  # user-specified 2026-09-15: same frames, played back 1.5x faster --
                         # applied only to the VideoWriter's declared fps (below), not to
                         # sampling/content, so this doesn't change WHAT was recorded, just
                         # how quickly the existing frame sequence plays back.

# Crop the real-camera footage to this fraction of its ORIGINAL width, KEEPING the left part
# (dropping everything to the right) -- user-specified 2026-09-15 via a hand-drawn line on a
# screenshot, roughly at the post/gap behind the arm's base. Keeps the gripper/arm/brick
# region (the actual manipulation), drops the color-block-wall/poster area on the right --
# corrected 2026-09-15 after an initial attempt kept the wrong side. Also makes the panel
# itself narrower in the final 3-panel composite ("make the camera view a bit smaller").
CROP_KEEP_FRAC = 0.45

# Where to draw the "depth camera" box: fractions of the CROPPED (not original) frame, so
# they hold regardless of VIDEO_PANEL_HEIGHT_PX/crop width. Right at the edge of the kept
# (left) region -- not a precise calibration, "give an idea where it is".
CAMERA_BOX_X_FRAC = 0.975
CAMERA_BOX_Y_FRAC = 0.58
CAMERA_BOX_SIZE_FRAC = 0.05  # box side length, as a fraction of the (now narrower) frame width

# Single horizontal arrow from the camera box toward the gripper -- replaces the earlier
# two-line FOV wedge (2026-09-15 per user request), then straightened to horizontal (was
# angled up toward a fixed target point) same day per follow-up request. Length as a
# fraction of frame width -- same "approximate, give an idea" spirit as everything else here.
ARROW_LENGTH_FRAC = 0.32


def get_video_start_epoch(mov_path: Path) -> float:
    """macOS-only: reads the .mov's own filesystem creation date via `mdls` as a proxy
    for "when recording started" (see module docstring for the accuracy caveat)."""
    out = subprocess.check_output(
        ["mdls", "-raw", "-name", "kMDItemContentCreationDate", str(mov_path)],
        text=True,
    ).strip()
    if out in ("(null)", ""):
        raise RuntimeError(f"no creation date metadata found on {mov_path}")
    dt = datetime.datetime.strptime(out, "%Y-%m-%d %H:%M:%S %z")
    return dt.timestamp()


def draw_camera_box(frame):
    h, w = frame.shape[:2]
    box = int(w * CAMERA_BOX_SIZE_FRAC)
    cx, cy = int(w * CAMERA_BOX_X_FRAC), int(h * CAMERA_BOX_Y_FRAC)
    x0, y0 = cx - box // 2, cy - box // 2

    # Single horizontal arrow from the box's left-center edge (roughly where the camera
    # itself sits) pointing left, toward the gripper -- replaces the earlier two-line FOV
    # wedge. Drawn BEFORE the box so the box sits cleanly on top of the base.
    apex = (x0, cy)
    target = (int(x0 - w * ARROW_LENGTH_FRAC), cy)
    cv2.arrowedLine(frame, apex, target, (0, 0, 0), 2, cv2.LINE_AA, tipLength=0.06)

    cv2.rectangle(frame, (x0, y0), (x0 + box, y0 + box), (0, 0, 0), thickness=-1)
    cv2.rectangle(frame, (x0, y0), (x0 + box, y0 + box), (255, 255, 255), thickness=1)
    # Right-align the label so its right edge sits just left of the box, BELOW the box now
    # (moved down 2026-09-15 per user request).
    label = "depth camera (approx)"
    (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    text_x = min(x0 - 8 - text_w, w - text_w - 4)
    text_y = y0 + box + 24
    cv2.putText(frame, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def parse_time(s):
    """Accepts plain seconds ("14", "14.5") or mm:ss ("0:14") -- for --start/--end."""
    if ":" in s:
        mm, ss = s.split(":")
        return int(mm) * 60 + float(ss)
    return float(s)


def make_synced_video(session_dir: Path, start_s=None, end_s=None, video_start_offset_s=0.0):
    mov_paths = sorted(session_dir.glob("*.mov"))
    if not mov_paths:
        print(f"  {session_dir}: no .mov screen recording found, skipping")
        return
    mov_path = mov_paths[0]

    # video_start_offset_s (added 2026-09-15): manual correction on top of the mdls-derived
    # estimate, for when the user has directly observed a lag between the two panels (e.g. "the
    # gripper starts rising at 0:36 in the video but 0:37 in the point clouds" -> the video is
    # being sought too far ahead for a given elapsed time -> push video_start LATER by the
    # observed gap, +1.0s here, so the same elapsed time seeks an EARLIER video frame). This is
    # on top of, not instead of, the inherent up-to-~1.3s hold-lag from the point clouds' own
    # ~0.75Hz sampling rate (see module docstring) -- that part isn't fixable by shifting a
    # single offset, since it's not a constant lag.
    video_start = get_video_start_epoch(mov_path) + video_start_offset_s
    cap = cv2.VideoCapture(str(mov_path))
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps_native = cap.get(cv2.CAP_PROP_FPS)
    video_duration = n_frames / fps_native if fps_native else 0.0
    orig_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    orig_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    crop_x1 = int(orig_w * CROP_KEEP_FRAC)
    cropped_w = crop_x1
    resized_w = int(cropped_w * VIDEO_PANEL_HEIGHT_PX / orig_h)

    clouds = sorted(session_dir.glob("clouds/*.npz"), key=lambda p: float(p.stem))
    cloud_ts = [float(c.stem) for c in clouds]
    in_window = [i for i, t in enumerate(cloud_ts) if video_start <= t <= video_start + video_duration]
    if not in_window:
        print(f"  {session_dir}: no cloud frames overlap the recording's time window, skipping")
        return

    pose_rows = load_pose_log(session_dir)
    out_clouds = [clouds[i] for i in in_window]
    bounds = compute_bounds(out_clouds, pose_rows=pose_rows)

    # PASS 1: run BrickTracker chronologically over ALL clouds (including pre-window ones,
    # for continuity) and render the skeleton+projection panels ONCE per in-window cloud --
    # these are the "stills" that get held across however many output frames their real
    # time gap spans.
    brick_tracker = BrickTracker()
    still_times, still_frames = [], []
    n_skipped_no_pose = 0
    for i, c in enumerate(clouds):
        xyz = load_cloud(c)
        off_plane, cluster_info = compute_off_plane_clusters(xyz)
        tracked_brick, brick_mode = brick_tracker.update(cluster_info or [])

        if i not in in_window:
            continue
        angles = angles_for_cloud(pose_rows, cloud_ts[i])
        if angles is None:
            n_skipped_no_pose += 1
            continue

        geom = {"bricks": []}
        if tracked_brick is not None:
            geom["bricks"] = [{
                "n_points": len(tracked_brick["idx"]), "angle_deg": round(tracked_brick["angle_deg"], 1),
                "centroid": tracked_brick["centroid"], "direction": tracked_brick["direction"],
                "points": tracked_brick["pts"], "mode": brick_mode,
            }]

        elapsed = cloud_ts[i] - cloud_ts[in_window[0]]
        title = f"{session_dir.name}  t={c.stem}  (+{elapsed:.1f}s)  cloud {len(still_times)+1}/{len(in_window)}"
        skel_frame = render_skeleton_frame(xyz, geom, bounds, title, angles)

        is_brick = np.zeros(off_plane.shape[0], dtype=bool) if off_plane is not None else np.zeros(0, dtype=bool)
        if tracked_brick is not None and len(tracked_brick["idx"]) > 0:
            is_brick[tracked_brick["idx"]] = True
        proj_frame = render_width_projection_panel(off_plane, is_brick, angles,
                                                     PROJECTION_PANEL_WIDTH_PX, FIG_SIZE_PX[1])

        still_times.append(cloud_ts[i])
        still_frames.append(hstack_frames(skel_frame, proj_frame))

    if not still_times:
        print(f"  {session_dir}: no in-window cloud had usable pose, skipping")
        return

    # PASS 2: composite at OUTPUT_FPS -- the real-camera panel is re-sampled fresh every
    # output frame (smooth), the skeleton/projection still is whichever cloud sample is
    # most recent at that output time (held/reused, per the module docstring).
    # start_s/end_s (added 2026-09-15): trim to a sub-range of the OUTPUT video's own elapsed
    # time (0 == still_times[0], same "+Xs" the skeleton panel's title shows) -- writes to a
    # SEPARATE file, doesn't touch/overwrite the full synced_review.mp4.
    span = still_times[-1] - still_times[0]
    trim = start_s is not None or end_s is not None
    clip_start = max(0.0, start_s if start_s is not None else 0.0)
    clip_end = min(span, end_s if end_s is not None else span)
    out_name = f"synced_review_{clip_start:.0f}s-{clip_end:.0f}s.mp4" if trim else "synced_review.mp4"
    out_path = session_dir / out_name
    frame_size = (resized_w + FIG_SIZE_PX[0] + PROJECTION_PANEL_WIDTH_PX, VIDEO_PANEL_HEIGHT_PX)
    # SPEED_MULTIPLIER applied ONLY to the declared container fps -- same frame sequence,
    # written faster, so it plays back in 1/SPEED_MULTIPLIER the real time.
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"),
                              OUTPUT_FPS * SPEED_MULTIPLIER, frame_size)

    n_out = max(1, int(span * OUTPUT_FPS) + 1)
    n_written = 0
    for k in range(n_out):
        elapsed = k / OUTPUT_FPS
        if trim and not (clip_start <= elapsed <= clip_end):
            continue
        abs_t = still_times[0] + elapsed
        video_t = abs_t - video_start
        if video_t < 0 or video_t > video_duration:
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, video_t * 1000)
        ok, vframe = cap.read()
        if not ok:
            vframe = np.full((int(orig_h), int(cropped_w), 3), 40, dtype=np.uint8)
        else:
            vframe = vframe[:, :crop_x1]
        vframe = cv2.resize(vframe, (resized_w, VIDEO_PANEL_HEIGHT_PX))
        vframe = draw_camera_box(vframe)

        # Most recent still whose sample time is <= abs_t (bisect on the sorted still_times).
        idx = bisect.bisect_right(still_times, abs_t) - 1
        idx = max(0, idx)

        frame = hstack_frames(vframe, still_frames[idx])
        writer.write(frame)
        n_written += 1

    writer.release()
    cap.release()
    print(f"  wrote {out_path} ({len(still_times)}/{len(clouds)} cloud samples used -- "
          f"{len(clouds) - len(in_window)} fell outside the recording's time window, "
          f"{n_skipped_no_pose} skipped for missing pose -- {n_written} output frames, written at "
          f"{OUTPUT_FPS * SPEED_MULTIPLIER:.0f}fps ({SPEED_MULTIPLIER}x speed) so it plays back "
          f"in {n_written / (OUTPUT_FPS * SPEED_MULTIPLIER):.1f}s, "
          f"video_start_epoch={video_start:.1f}, video_duration={video_duration:.1f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="A session directory (must contain clouds/, pose_log.jsonl, and a .mov).")
    parser.add_argument("--start", type=parse_time, default=None,
                         help="Trim start, in seconds or mm:ss, relative to the output video's own "
                              "elapsed time (0 = first in-window cloud sample). Writes a separate "
                              "file, doesn't overwrite synced_review.mp4.")
    parser.add_argument("--end", type=parse_time, default=None, help="Trim end, same units as --start.")
    parser.add_argument("--video-start-offset", type=float, default=0.0,
                         help="Manual correction (seconds) added to the mdls-derived video start "
                              "time, for when a real observed lag between the two panels calls for "
                              "it (see make_synced_video's docstring for sign convention).")
    args = parser.parse_args()
    make_synced_video(Path(args.path), start_s=args.start, end_s=args.end,
                       video_start_offset_s=args.video_start_offset)


if __name__ == "__main__":
    main()
