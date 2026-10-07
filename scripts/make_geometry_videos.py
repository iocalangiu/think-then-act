#!/usr/bin/env python3
"""
make_geometry_videos.py  (runs locally, no rclpy dependency.)

Renders one review video per grasp-recording session, saved into that
session's own folder: every saved cloud frame, in order, with the detected
brick drawn as a thick line across its actual extent (see
extract_gripper_brick_geometry.extract_geometry's brick candidate, the
smallest-angle one -- the one validated 2026-09-11 as reliably tracking the
real static brick across a whole session) and the points belonging to the
detected gripper region rendered as larger/thicker dots, so the geometry
extraction can be eyeballed frame-by-frame instead of trusting the printed
numbers alone.

Uses cv2.VideoWriter (mp4v) directly on matplotlib-rendered frame buffers --
no ffmpeg binary needed (checked 2026-09-11: not installed here, and
matplotlib's own animation writers only had 'pillow'/'html' registered
without it).

Usage:
    python3 scripts/make_geometry_videos.py                     # all sessions under artifacts/grasp_sessions
    python3 scripts/make_geometry_videos.py artifacts/grasp_sessions/session_X
    python3 scripts/make_geometry_videos.py --fps 6 --every 2    # every 2nd frame, at 6fps
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from extract_gripper_brick_geometry import (  # noqa: E402
    extract_geometry, load_cloud, fit_table_plane, PLANE_INLIER_DIST_M, FingerTracker, BrickTracker,
    planar_contour, hull_edges, hull_outer_points, remove_dominant_planes, fast_cluster, pca_axes,
    CLUSTER_EPS, MIN_CLUSTER_POINTS, VERTICAL_ANGLE_TOLERANCE_DEG, TABLE_CONTACT_TOLERANCE_M,
)
from ur3e_fk import (  # noqa: E402
    tool0_fingertip_position, tool0_approach_axis, tool0_width_axis, tool0_position,
    ur3e_fk_chain, base_point_to_camera, base_direction_to_camera, extract_arm_angles,
)

AXIS_LINE_LENGTH_M = 0.15    # length of the standalone width-axis/approach-axis lines drawn in
                              # render_skeleton_frame (added 2026-09-15, lengthened from an initial
                              # 0.08 same day per user request) -- picked by eye to be clearly
                              # visible/labelable, not tied to any measured quantity
APPROACH_AXIS_ANCHOR_DIST_M = 0.06  # user-specified 2026-09-15 (corrected from an initial
                              # 0.60): how far along the approach axis from tool0 the approach
                              # axis indicator LINE is drawn from -- roughly the fingertip area.
WIDTH_AXIS_ANCHOR_DIST_M = 0.24  # user-specified 2026-09-15 (split out from a single shared
                              # anchor at 0.06, then moved to 0.12, now 0.24): how far along the
                              # approach axis from tool0 the width axis indicator line is
                              # CENTERED. Also used as render_width_projection_panel's own
                              # projection origin, so the projection panel's "0mm" always
                              # matches where this line sits.

FIG_SIZE_PX = (800, 800)
ELEV, AZIM = 12, 140  # fixed camera angle -- consistent framing across the whole video.
                       # Changed 2026-09-15 (was 30, 110, before that 15, -60) per user
                       # request + a hand-drawn reference sketch: a low, close viewpoint
                       # near the table/brick, with the robot's base/shoulder receding up
                       # and away in the background instead of dominating the frame.
JOINT_CHAIN_LABELS = ["base_link", "shoulder_pan", "shoulder_lift", "elbow",
                       "wrist_1", "wrist_2", "tool0"]  # ur3e_fk_chain's 7 points, in order --
                                                        # last one is tool0 itself (the wrist_3
                                                        # joint's own output), not a separate point
PROJ_FOCAL_LENGTH = 0.15  # render_skeleton_frame's ax.set_proj_type('persp', focal_length=...) --
                           # TRUE perspective (near things bigger, far things smaller, straight
                           # lines converge), not mplot3d's default near-orthographic look. Added
                           # together with the ELEV/AZIM change above for the same request -- the
                           # "robot recedes into the back" read needs actual foreshortening, a
                           # camera angle change alone doesn't produce it. Smaller = more dramatic
                           # (fisheye-like); 0.15 was picked by eye against the reference sketch.


def to_display_frame(pts):
    """Fixed display-only rotation, no translation -- extract_geometry's own
    math stays in the original camera frame, only rendering uses this.

    camera_depth_optical_frame convention: x=right, y=down, z=forward
    (depth). Confirmed 2026-09-11: the RANSAC-fitted table normal comes
    out as ~[-0.07, -0.99, -0.11] -- dominated by -Y, not Z -- so Y is the
    real vertical axis in this data, not the raw Z a naive plot would
    otherwise show as "up" (matplotlib's 3D z-axis is always the one drawn
    vertical). Remap x->x, z->y (depth becomes the horizontal "into the
    scene" axis), -y->z (up becomes positive display-z). This is a proper
    orthonormal rotation (determinant +1), so it works unchanged on
    direction vectors too (no translation involved), not just positions."""
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    return np.stack([x, z, -y], axis=-1)


def depth_colors(z):
    """Near=red, far=blue -- same convention as record_camera_snapshot.py's
    depth_to_color, just float [0,1] RGB instead of uint8, for matplotlib's
    scatter `c=`. Colors the ORIGINAL camera-frame z (true optical depth),
    independent of how to_display_frame rearranges axes for plotting."""
    z_min, z_max = float(np.nanmin(z)), float(np.nanmax(z))
    z_norm = (z - z_min) / max(z_max - z_min, 1e-6)
    colors = np.zeros((z.shape[0], 3))
    colors[:, 0] = 1 - z_norm  # near -> red
    colors[:, 2] = z_norm      # far -> blue
    return colors


# Same red(near)->blue(far) mapping as depth_colors, as an actual Colormap object -- for
# drawing a colorbar legend next to the point cloud (added 2026-09-15 per user request) that
# matches the scatter's own coloring exactly, not an approximation from a stock cmap.
DEPTH_CMAP = LinearSegmentedColormap.from_list("depth_near_far", [(1, 0, 0), (0, 0, 1)])


def table_inlier_mask(xyz):
    """Recomputes JUST the first RANSAC plane pass (same function, same
    default seed, so it's the identical fit extract_geometry's own
    remove_dominant_planes used internally) and returns a boolean mask
    over xyz for which points are table inliers. Deliberately the FIRST
    plane only, not any second "wall" plane also removed inside
    remove_dominant_planes -- this is specifically "the table", per the
    2026-09-11 request, not all RANSAC-removed clutter."""
    normal, point = fit_table_plane(xyz)
    dist = np.abs((xyz - point) @ normal)
    return dist < PLANE_INLIER_DIST_M


def load_pose_log(session_dir):
    """Returns a time-sorted list of (t, arm_joint_names, arm_joint_positions)
    for rows that have joint names at all -- used for nearest-timestamp
    lookup against a cloud filename's own timestamp (clouds and pose
    samples are saved on independent ~1s timers, so they're close but not
    identical -- see record_grasp_session.py)."""
    path = session_dir / "pose_log.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("arm_joint_names"):
                rows.append((r["t"], r["arm_joint_names"], r["arm_joint_positions"]))
    rows.sort(key=lambda r: r[0])
    return rows


def nearest_pose(pose_rows, t, max_dt=2.0):
    """Nearest pose_log row to timestamp t, or None if none within max_dt
    seconds (rather than silently using a stale/irrelevant match)."""
    if not pose_rows:
        return None
    times = [r[0] for r in pose_rows]
    idx = int(np.searchsorted(times, t))
    candidates = [i for i in (idx - 1, idx) if 0 <= i < len(pose_rows)]
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(pose_rows[i][0] - t))
    if abs(pose_rows[best][0] - t) > max_dt:
        return None
    return pose_rows[best]


def fk_prior_for_cloud(pose_rows, t):
    """Nearest-timestamp joint angles -> FK -> (approach_axis, fingertip_
    position, width_axis), all transformed into camera frame.
    fingertip_position uses tool0_fingertip_position (tool0 nudged
    forward by the gripper's assumed length), NOT raw tool0_position --
    tool0 itself is the wrist flange, not the fingers (see the TCP
    discussion, 2026-09-11). width_axis is the wrist_3-informed
    finger-separation direction, used both to pick the gripper cluster
    and (by FingerTracker) to judge whether a frame's viewing geometry
    could even show two separate fingers. Returns (None, None, None) if
    no usable pose sample is nearby or the arm joints weren't all present
    in it."""
    row = nearest_pose(pose_rows, t)
    if row is None:
        return None, None, None
    angles = extract_arm_angles(row[1], row[2])
    if angles is None:
        return None, None, None
    axis_base = tool0_approach_axis(angles)
    pos_base = tool0_fingertip_position(angles)
    width_base = tool0_width_axis(angles)
    return (base_direction_to_camera(axis_base), base_point_to_camera(pos_base),
            base_direction_to_camera(width_base))


def angles_for_cloud(pose_rows, t):
    """Nearest-timestamp raw joint angles (radians, JOINT_NAMES order), or
    None -- for callers that need the angles themselves (e.g. the robot
    skeleton), not the derived axis/position quantities fk_prior_for_cloud
    returns."""
    row = nearest_pose(pose_rows, t)
    if row is None:
        return None
    return extract_arm_angles(row[1], row[2])


def compute_bounds(cloud_paths, sample=10, pose_rows=None):
    """pose_rows: when given (skeleton mode), also folds in the robot
    skeleton's own extent (base_link through tool0) across the same
    sampled frames -- the arm reaches back much farther from the camera
    than the point cloud's own visible region does, so bounds computed
    from the cloud alone clip the skeleton's near-base segments (confirmed
    2026-09-15 on a real frame: a jagged, cut-off zigzag at the plot
    edge)."""
    idx = np.linspace(0, len(cloud_paths) - 1, min(sample, len(cloud_paths))).astype(int)
    all_pts = [to_display_frame(load_cloud(cloud_paths[i])) for i in idx]
    if pose_rows:
        for i in idx:
            angles = angles_for_cloud(pose_rows, float(cloud_paths[i].stem))
            if angles is not None:
                chain_cam = np.array([base_point_to_camera(p) for p in ur3e_fk_chain(angles)])
                all_pts.append(to_display_frame(chain_cam))
        # Camera sits at the camera frame's own origin (see render_skeleton_frame's camera-dot
        # comment) -- fold it in too, since it's usually well outside the cloud/skeleton's own
        # span (it's behind/beside everything it's looking at) and would otherwise get clipped
        # off the plot entirely.
        all_pts.append(to_display_frame(np.zeros((1, 3))))
        # Same for the width/approach axis indicator LINES' full extent (added 2026-09-15) --
        # they point roughly toward/through the table, so tens of cm out can land well below
        # the cloud's own z-range and get clipped at the bottom of the plot otherwise. Folds in
        # both anchors AND each line's far end(s), not just the anchor points, since
        # AXIS_LINE_LENGTH_M is itself large enough to matter.
        for i in idx:
            angles = angles_for_cloud(pose_rows, float(cloud_paths[i].stem))
            if angles is not None:
                t0 = tool0_position(angles)
                approach_dir = tool0_approach_axis(angles)
                width_dir = tool0_width_axis(angles)
                approach_anchor = t0 + approach_dir * APPROACH_AXIS_ANCHOR_DIST_M
                width_anchor = t0 + approach_dir * WIDTH_AXIS_ANCHOR_DIST_M
                extra_pts = np.array([
                    approach_anchor + approach_dir * AXIS_LINE_LENGTH_M,
                    width_anchor + width_dir * (AXIS_LINE_LENGTH_M / 2),
                    width_anchor - width_dir * (AXIS_LINE_LENGTH_M / 2),
                ])
                all_pts.append(to_display_frame(np.array([base_point_to_camera(p) for p in extra_pts])))
    pts = np.vstack(all_pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = (hi - lo) * 0.05
    return lo - pad, hi + pad


def render_finger_lines(ax, finger_result):
    """Draws the FingerTracker result: solid lines for a this-frame-
    confident detection, dashed for a temporally-propagated estimate
    (fell back to the last confident frame's separation), a single solid
    line when neither applies (only one finger visible / no usable
    history yet)."""
    r = finger_result
    axis, half = r["length_axis"], r["length_half"]
    style = "--" if r["mode"] == "propagated" else "-"
    res = r["resolvable"]
    res_tag = "" if res is None else (", resolvable" if res else ", UNRESOLVABLE")

    c1 = r["finger1_centroid"]
    seg1 = to_display_frame(np.array([c1 - axis * half, c1 + axis * half]))
    ax.plot(seg1[:, 0], seg1[:, 1], seg1[:, 2], linewidth=4, color="darkorange", linestyle=style,
             label=f"finger1 ({r['mode']}{res_tag}, n={r['finger1_n_points']})")

    c2 = r["finger2_centroid"]
    if c2 is not None:
        seg2 = to_display_frame(np.array([c2 - axis * half, c2 + axis * half]))
        ax.plot(seg2[:, 0], seg2[:, 1], seg2[:, 2], linewidth=4, color="deepskyblue", linestyle=style,
                 label=f"finger2 ({r['mode']}{res_tag}, n={r['finger2_n_points']}) sep={r['separation_m']*1000:.1f}mm")


def render_fk_prior(ax, fk_axis, fk_position, half=0.06):
    """Dashed gray reference line: the UR3e's own FK-predicted tool0
    approach axis/position (already in camera frame) -- drawn regardless
    of what the point-cloud clusters say, so a frame where the FK line
    DOESN'T line up with the actual gripper structure is immediately
    visible (extrinsics/DH/axis-convention error), not just trusted
    blindly."""
    seg = to_display_frame(np.array([fk_position - fk_axis * half, fk_position + fk_axis * half]))
    ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], linewidth=3, color="gray", linestyle=":",
             label="FK prior")
    p = to_display_frame(fk_position[None, :])[0]
    ax.scatter([p[0]], [p[1]], [p[2]], s=25, c="gray", marker="x")


def render_skeleton_frame(xyz, geom, bounds, title, angles):
    """Dedicated render mode (2026-09-15 request): table contour + brick
    contour (from the point cloud, via RANSAC/clustering) plus a robot
    skeleton + kinematic grasp-plane quad (from the arm's OWN joint
    angles/FK, not the point cloud at all -- the grasp plane in particular
    is deliberately independent of the fragile point-cloud finger
    detection). `angles` is this frame's 6 arm joint angles (radians,
    JOINT_NAMES order) from the nearest pose_log sample."""
    fig = plt.figure(figsize=(FIG_SIZE_PX[0] / 100, FIG_SIZE_PX[1] / 100), dpi=100)
    fig.patch.set_facecolor("white")
    # Explicit rect (not add_subplot(111)) so there's a reserved strip on the right for the
    # depth colorbar (added 2026-09-15) -- negative left/bottom + width/height >1 crops most
    # of mplot3d's large inherent margin (the "cut the white space" request), while stopping
    # short of x=1 leaves room for cbar_ax below instead of the 3D axes eating the whole figure.
    # left nudged in from -0.08 to 0.05 (2026-09-15 per user report the video/skeleton panels
    # looked like they were overlapping when hstacked -- right edge (left+width) kept at the
    # same 0.80 as before, so this only opens a left margin, it doesn't touch the colorbar
    # strip on the right (which starts at x=0.90, see cbar_ax below).
    ax = fig.add_axes([0.05, -0.06, 0.75, 1.35], projection="3d")
    ax.set_proj_type("persp", focal_length=PROJ_FOCAL_LENGTH)
    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_alpha(1.0)

    xyz_disp = to_display_frame(xyz)
    # Full saturation (was blended 50% toward white, which washed out the near/far distinction
    # the coloring exists to show -- fixed 2026-09-15). Alpha alone now carries the "background
    # context, not the main signal" de-emphasis instead of desaturating the color itself.
    z_min, z_max = float(np.nanmin(xyz[:, 2])), float(np.nanmax(xyz[:, 2]))
    colors = depth_colors(xyz[:, 2])
    ax.scatter(xyz_disp[:, 0], xyz_disp[:, 1], xyz_disp[:, 2], s=1.5, c=colors, alpha=0.45)

    # 1) table contour: RANSAC inliers -> 2D convex hull on the plane -> lifted back to 3D
    # Legend entry drawn unconditionally (2026-09-15 fix) -- an empty placeholder in the rare
    # frame with too few table points, so the legend doesn't visibly appear/disappear frame to
    # frame across the video.
    table_normal, table_point = fit_table_plane(xyz)
    table_mask = np.abs((xyz - table_point) @ table_normal) < PLANE_INLIER_DIST_M
    if table_mask.sum() >= 3:
        contour = planar_contour(xyz[table_mask], table_normal)
        c_disp = to_display_frame(contour)
        ax.plot(c_disp[:, 0], c_disp[:, 1], c_disp[:, 2], color="darkorange", linewidth=2,
                 linestyle="--", label="table (RANSAC contour)")
        # Anchor to the NEAREST-TO-CAMERA edge of the table (smallest camera-frame z/depth),
        # not the full-inlier mean -- moved 2026-09-15 per user request to put it at the
        # table's own lowest/closest corner instead of its middle, which reads better with
        # the new low, close viewpoint (see ELEV/AZIM/PROJ_FOCAL_LENGTH). Averaged over the
        # nearest 2% of inlier points (not a single closest point) for the same jitter-
        # stability reason as before -- a single extremal point still moves frame to frame,
        # an average of many doesn't.
        table_pts = xyz[table_mask]
        near_thresh = np.percentile(table_pts[:, 2], 2)
        label_pt = to_display_frame(table_pts[table_pts[:, 2] <= near_thresh].mean(axis=0))
        ax.text(label_pt[0], label_pt[1], label_pt[2], "table (RANSAC)", color="darkorange",
                 fontsize=9, fontweight="bold", zorder=100,
                 path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    else:
        ax.plot([], [], color="darkorange", linewidth=2, linestyle="--", label="table (not found)")

    # 2) brick outline: the TRACKED brick cluster's (BrickTracker, see make_session_video)
    # own OUTER points (hull_outer_points), scattered directly rather than connected into
    # hull edges -- changed 2026-09-15 per user report that the connected-edge contour
    # looked unreliable/broken from the new close, steep perspective view (hull edges can
    # visually criss-cross at extreme viewing angles; the outer points themselves always
    # read correctly regardless of viewpoint). Bigger markers (s=35 vs the base cloud's
    # s=1.5) so they read as a clear highlight, not just more scatter. Legend entry ALWAYS
    # drawn (2026-09-15 fix), since "brick not currently tracked" is common (BrickTracker's
    # own "lost" mode) and the legend disappearing every time that happens made the video
    # visually jump around.
    if geom["bricks"]:
        b = geom["bricks"][0]
        pts = b["points"]
        mode_tag = f", {b['mode']}" if "mode" in b else ""
        if pts.shape[0] >= 1:
            outer = hull_outer_points(pts)
            outer_disp = to_display_frame(outer)
            ax.scatter(outer_disp[:, 0], outer_disp[:, 1], outer_disp[:, 2], color="green",
                       edgecolors="darkgreen", linewidth=0.6, s=35, alpha=0.95, zorder=99,
                       label=f"brick outline ({b['angle_deg']}deg{mode_tag})")
            # Label BELOW the brick (under its lowest point in display-height), not at its
            # centroid -- moved 2026-09-15 per user request, the centroid placement sat the
            # text right on top of/inside the highlighted points, harder to read.
            pts_disp = to_display_frame(pts)
            label_pt = pts_disp.mean(axis=0)
            label_pt[2] = pts_disp[:, 2].min() - 0.05
            ax.text(label_pt[0], label_pt[1], label_pt[2], "brick (PCA)", color="green",
                     fontsize=9, fontweight="bold", ha="center", zorder=100,
                     path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
        else:
            ax.scatter([], [], color="green", s=35, label=f"brick outline (too few points{mode_tag})")
    else:
        ax.scatter([], [], color="green", s=35, label="brick outline (not tracked)")

    # 3) robot skeleton: every intermediate joint position, base_link -> tool0, no gripper fingers.
    # Each of the 7 chain points labeled by name (added 2026-09-15, replaces the legend's single
    # generic "robot skeleton" entry now that the legend itself is gone) -- ur3e_fk_chain's own
    # docstring gives this exact order: base_link origin, then after each of the 6 joints in turn
    # (JOINT_NAMES order), the last of which IS tool0 (not a separate "wrist_3" point).
    chain_base = ur3e_fk_chain(angles)
    chain_cam = np.array([base_point_to_camera(p) for p in chain_base])
    chain_disp = to_display_frame(chain_cam)
    ax.plot(chain_disp[:, 0], chain_disp[:, 1], chain_disp[:, 2], color="steelblue", linewidth=3,
             marker="o", markersize=4)
    # tool0 (the last chain point) gets no label of its own -- it's already the origin the
    # width/approach axis lines are anchored near, labeling it too just overlapped those (user
    # report 2026-09-15, "wrist_tool0" reading as one fused word). The remaining 6 get small
    # alternating x/z offsets, growing per joint -- wrist_1/wrist_2 in particular tend to sit
    # close together on screen when the wrist is folded down toward the table, and shoulder_pan/
    # elbow likewise (both fixed 2026-09-15) -- a plain per-point offset wasn't enough to keep
    # them apart, and same-side offsets on adjacent joints didn't help; opposite sides (negative
    # vs positive dx) does. shoulder_pan additionally right-aligned (2026-09-15) -- a plain
    # negative dx with the default left alignment still let the tail end of the word reach back
    # toward the point since text grows rightward from its anchor; right-aligning it means the
    # word ends where the offset point is instead, keeping the whole label on elbow's other side.
    label_offsets = [(0.0, 0.0, "left"), (-0.015, 0.0, "right"), (0.02, 0.02, "left"),
                      (0.02, -0.02, "left"), (0.04, 0.05, "left"), (0.04, -0.05, "left")]
    for joint_pt, joint_label, (dx, dz, ha) in zip(chain_disp[:-1], JOINT_CHAIN_LABELS[:-1], label_offsets):
        ax.text(joint_pt[0] + dx, joint_pt[1], joint_pt[2] + dz, joint_label, color="steelblue",
                 fontsize=7, zorder=100, ha=ha,
                 path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # 3.5) camera position: the point cloud (and everything scattered above) is already IN the
    # camera's own frame (camera_depth_optical_frame), so the camera itself sits at that frame's
    # origin -- no transform needed, unlike the skeleton/grasp-plane which come from base_link.
    cam_disp = to_display_frame(np.zeros(3))
    ax.scatter([cam_disp[0]], [cam_disp[1]], [cam_disp[2]], color="black", s=40, marker="o", zorder=100)
    ax.text(cam_disp[0], cam_disp[1], cam_disp[2], "  camera", color="black", fontsize=8,
             fontweight="bold", zorder=100,
             path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # 4) width axis + approach axis: the robot's own pose (approach + width axes from FK), no
    # point-cloud involvement. width_axis is tool0's local X (switched 2026-09-15 from Y after
    # direct visual confirmation against real fingers -- see ur3e_fk.tool0_width_axis's
    # docstring for the still-unresolved tension with an earlier, different piece of evidence
    # that said Y). The kinematic grasp-plane trapezoid that used to be drawn here was removed
    # 2026-09-15 per user request (its "made up" dimensions/label were adding clutter without
    # much value once the axis lines below covered the same information more directly).
    tool0_base = tool0_position(angles)
    approach_base = tool0_approach_axis(angles)
    width_base = tool0_width_axis(angles)

    # width axis + approach axis: the same two FK vectors above, drawn as their own labeled
    # lines (added 2026-09-15 per user request) -- explicit rather than implied by the (now
    # removed) trapezoid's shape alone. Each anchored its OWN distance further out along the
    # approach axis from tool0 (split into two separate anchors 2026-09-15 -- originally one
    # shared 6cm anchor, then the width axis specifically moved out further to 12cm), not at
    # tool0 itself -- moves the lines/labels clear of the crowded wrist area into open space,
    # purely for legibility (the axis DIRECTIONS are the same everywhere along the line, so
    # this is just a drawing offset, not a different quantity). Approach axis one-directional
    # from its anchor (continues the same forward sense, anchor toward the fingers), width axis
    # symmetric about ITS OWN (farther-out) anchor (a spread direction, no inherent forward/back).
    approach_anchor_base = tool0_base + approach_base * APPROACH_AXIS_ANCHOR_DIST_M
    approach_end_base = approach_anchor_base + approach_base * AXIS_LINE_LENGTH_M
    approach_seg = to_display_frame(np.array([base_point_to_camera(approach_anchor_base),
                                                base_point_to_camera(approach_end_base)]))
    ax.plot(approach_seg[:, 0], approach_seg[:, 1], approach_seg[:, 2], color="crimson", linewidth=2)
    ax.text(approach_seg[1, 0], approach_seg[1, 1], approach_seg[1, 2], "  approach axis",
             color="crimson", fontsize=9, fontweight="bold", zorder=100,
             path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    width_anchor_base = tool0_base + approach_base * WIDTH_AXIS_ANCHOR_DIST_M
    width_end_base = width_anchor_base + width_base * (AXIS_LINE_LENGTH_M / 2)
    width_start_base = width_anchor_base - width_base * (AXIS_LINE_LENGTH_M / 2)
    width_seg = to_display_frame(np.array([base_point_to_camera(width_start_base),
                                             base_point_to_camera(width_end_base)]))
    ax.plot(width_seg[:, 0], width_seg[:, 1], width_seg[:, 2], color="darkmagenta", linewidth=2)
    ax.text(width_seg[1, 0], width_seg[1, 1], width_seg[1, 2], "  width axis",
             color="darkmagenta", fontsize=9, fontweight="bold", zorder=100,
             path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    lo, hi = bounds
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.view_init(elev=ELEV, azim=AZIM)
    # No title (removed 2026-09-15 per user request, reclaims the whitespace it left above the
    # plot) and no legend (removed same day) -- everything now labeled directly next
    # to what it refers to (table/brick/joints/axes/camera/grasp plane), which also reclaims
    # the whitespace the legend box used to occupy.

    # Depth colorbar, in the strip reserved by ax's rect above -- matches the scatter's own
    # per-frame near/far normalization exactly (same z_min/z_max), added 2026-09-15 per user
    # request ("red close blue far").
    cbar_ax = fig.add_axes([0.90, 0.15, 0.035, 0.5])
    mappable = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=z_min, vmax=z_max), cmap=DEPTH_CMAP)
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label("depth (m)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    # vmin (near, red) sits at the BOTTOM of a standard vertical colorbar, vmax (far, blue) at
    # the TOP -- label placement matches that, not the other way around (caught 2026-09-15 by
    # actually checking the rendered output against the tick values).
    cbar.ax.text(0.5, 1.03, "far", ha="center", va="bottom", fontsize=7, color="blue",
                  transform=cbar_ax.transAxes)
    cbar.ax.text(0.5, -0.03, "near", ha="center", va="top", fontsize=7, color="red",
                  transform=cbar_ax.transAxes)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    # Crop the blank margin above the plot content and rescale back to FIG_SIZE_PX -- added
    # 2026-09-15 per user request ("cut the white space above") after enlarging the axes
    # rect's own height stopped having any visible effect (mplot3d's internal padding doesn't
    # scale with the rect the way a 2D axes' would). Post-processing the rendered image
    # directly is the reliable fix: find the first row (top down) with any non-white pixel,
    # crop everything above it (minus a small margin), then resize back up to the original
    # panel size so it still hstacks cleanly with the other two panels.
    frame_bgr = crop_top_whitespace(frame_bgr, FIG_SIZE_PX)
    return frame_bgr


def crop_top_whitespace(frame_bgr, target_size, margin_px=8):
    non_white_rows = np.where(frame_bgr.min(axis=2).min(axis=1) < 250)[0]
    if non_white_rows.size == 0:
        return frame_bgr
    top = max(0, int(non_white_rows[0]) - margin_px)
    if top == 0:
        return frame_bgr
    return cv2.resize(frame_bgr[top:], target_size, interpolation=cv2.INTER_AREA)


def compute_off_plane_clusters(xyz):
    """Same table-removal + per-cluster PCA/angle pass extract_geometry
    uses internally, but returns the FULL cluster_info (every off-plane
    cluster, vertical or not -- idx/pts/centroid/direction/angle_deg),
    not a pre-filtered brick mask. Callers that need brick/not-brick
    decide that themselves -- as of 2026-09-15 that's BrickTracker, so a
    tracked, temporally-consistent identity can be used instead of each
    frame's independent "smallest angle wins" (see BrickTracker's
    docstring for why that mattered). Returns (off_plane_xyz,
    cluster_info) or (None, []) if no table plane / no off-plane residual
    was found."""
    off_plane, table_normal, table_point = remove_dominant_planes(xyz)
    if table_normal is None or off_plane.shape[0] == 0:
        return None, []
    clusters = fast_cluster(off_plane, CLUSTER_EPS, MIN_CLUSTER_POINTS)
    cluster_info = []
    for idx in clusters:
        pts = off_plane[idx]
        centroid, eigvecs, _eigvals = pca_axes(pts)
        direction = eigvecs[:, 0]
        angle_deg = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(direction, table_normal)))))))
        # Same "resting on the table" check as extract_geometry's own -- a real standing
        # brick's base should come within TABLE_CONTACT_TOLERANCE_M of the table surface;
        # added 2026-09-15 because the angle test alone kept accepting the gripper/wrist too.
        dist_to_table = np.abs((pts - table_point) @ table_normal)
        touches_table = bool(dist_to_table.min() < TABLE_CONTACT_TOLERANCE_M)
        cluster_info.append({"idx": idx, "pts": pts, "centroid": centroid, "direction": direction,
                              "angle_deg": angle_deg, "touches_table": touches_table})
    return off_plane, cluster_info


PROJECTION_XLIM_MM = (-200, 200)  # fixed, not auto-scaled per frame -- checked against real data
                                   # across all 4 sessions (2026-09-15): actual range varies wildly
                                   # frame to frame (as far as -259 to 419mm at the extremes), so
                                   # auto-scaling made the video's x-axis visibly jump around. This
                                   # fixed window covers the 1st-99th percentile of real data in
                                   # every session; a few outlier points get clipped off-screen.


PERP_DIST_VMIN_MM = 40   # colorbar bounds for perpendicular distance-to-width-axis-line --
PERP_DIST_VMAX_MM = 220  # data-driven (2026-09-15, like PROJECTION_XLIM_MM): checked the actual
                          # distribution across the whole session_1789476296 (all 76 frames, ~68k
                          # off-plane points total, using the 24cm width-axis anchor) -- p1=60mm,
                          # p50=123mm, p99=222mm, max=278mm. The earlier 0-80 (then 40-80) range
                          # was set from a single unusually-close preview frame and left almost
                          # everything in the session saturated at one end; 40-220 spans roughly
                          # the 1st-99th percentile of real data instead.


def render_width_projection_panel(off_plane, is_brick, angles, width_px, height_px):
    """2D (not 3D -- no perspective ambiguity) strip plot: every non-table
    point projected onto tool0's width axis (x position), row-separated by
    whether it was classified as the brick or not (y position), and
    colored by its PERPENDICULAR distance to the width-axis LINE itself --
    added 2026-09-15 as a first step toward a usable gripper-proximity
    signal: points that actually sit close to the axis line the fingers
    would close along are the ones a real "how close is the gripper"
    metric should care about, as opposed to material that's merely
    somewhere in the general vicinity along the axis. Does NOT yet
    separately distinguish "gripper" from other non-brick clutter -- that
    classification is exactly what's still unresolved elsewhere in this
    investigation. X-axis is FIXED (PROJECTION_XLIM_MM), not auto-scaled
    per frame -- see that constant's comment.

    off_plane/is_brick are PRECOMPUTED by the caller (compute_off_plane_
    clusters + BrickTracker), not recomputed here -- so this panel's
    brick/not-brick split is always the SAME tracked identity the left
    (skeleton) panel is drawing, not an independent per-frame guess that
    could disagree between the two panels.

    Always draws the SAME legend entries and colorbar, empty or not --
    fixed 2026-09-15 after the colorbar/legend previously only appeared in
    frames that happened to have off-plane points, which made the video
    visually jump between "has a colorbar" and "doesn't" frame to frame.
    An unconditional ScalarMappable (not tied to any actual scatter call)
    keeps the colorbar identical in every frame regardless of content."""
    fig, ax = plt.subplots(figsize=(width_px / 100, height_px / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    has_data = off_plane is not None and off_plane.shape[0] > 0
    if has_data:
        # Origin moved 2026-09-15 from tool0 itself to the same WIDTH_AXIS_ANCHOR_DIST_M-along-
        # approach-axis point the skeleton panel's own width axis line is now anchored/centered
        # at (per user request, so both panels agree on what "0mm" means) -- NOT tool0.
        anchor_base = tool0_position(angles) + tool0_approach_axis(angles) * WIDTH_AXIS_ANCHOR_DIST_M
        anchor_cam = base_point_to_camera(anchor_base)
        width_cam = base_direction_to_camera(tool0_width_axis(angles))
        vec = off_plane - anchor_cam
        proj_scalar = vec @ width_cam            # meters, signed, along the axis
        perp_vec = vec - np.outer(proj_scalar, width_cam)
        perp_dist_mm = np.linalg.norm(perp_vec, axis=1) * 1000  # perpendicular distance, unsigned
        proj_mm = proj_scalar * 1000
        rng = np.random.default_rng(0)
        y = np.where(is_brick, 1.0, 0.0) + rng.uniform(-0.35, 0.35, size=off_plane.shape[0])
        n_total, n_brick = off_plane.shape[0], int(is_brick.sum())
    else:
        proj_mm, y, perp_dist_mm = np.array([]), np.array([]), np.array([])
        n_total, n_brick = 0, 0

    ax.scatter(proj_mm, y, s=12, c=perp_dist_mm, cmap="RdYlGn", vmin=PERP_DIST_VMIN_MM, vmax=PERP_DIST_VMAX_MM,
               alpha=0.8, edgecolors="none")
    # Unconditional colorbar -- a ScalarMappable with no associated data still draws a full,
    # correctly-ranged colorbar, so this looks identical whether or not this frame has points.
    mappable = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=PERP_DIST_VMIN_MM, vmax=PERP_DIST_VMAX_MM), cmap="RdYlGn")
    cbar = fig.colorbar(mappable, ax=ax, pad=0.02, fraction=0.06)
    cbar.set_label("distance to width axis (mm)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.scatter([], [], c="black", s=10, label=f"n={n_total} (brick={n_brick})")

    ax.axvline(0, color="dimgray", linewidth=1.5, linestyle="--",
                label=f"anchor (0mm, {WIDTH_AXIS_ANCHOR_DIST_M*100:.0f}cm from tool0)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["non-brick\n(gripper proxy)", "brick"])
    ax.set_ylim(-0.8, 1.8)
    ax.set_xlim(*PROJECTION_XLIM_MM)
    ax.set_xlabel(f"position along width axis (mm from anchor, {WIDTH_AXIS_ANCHOR_DIST_M*100:.0f}cm "
                   f"from tool0 along approach axis)")
    ax.set_title("non-table points projected onto width axis", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return frame_bgr


def hstack_frames(left, right):
    """Side-by-side combine, padding the shorter one with white so heights
    match exactly (cv2.hconcat requires identical height)."""
    h = max(left.shape[0], right.shape[0])
    def pad(img):
        if img.shape[0] == h:
            return img
        pad_amt = h - img.shape[0]
        return cv2.copyMakeBorder(img, 0, pad_amt, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return cv2.hconcat([pad(left), pad(right)])


def render_frame(xyz, geom, bounds, title, highlight_table=False, finger_result=None,
                  fk_axis=None, fk_position=None):
    fig = plt.figure(figsize=(FIG_SIZE_PX[0] / 100, FIG_SIZE_PX[1] / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_alpha(1.0)

    xyz_disp = to_display_frame(xyz)
    colors = depth_colors(xyz[:, 2])
    if highlight_table:
        # Table (RANSAC's first-plane inliers) forced to plain gray;
        # everything else (wall/clutter/brick/gripper's own raw points)
        # keeps the depth gradient, so the table reads as "known/explained"
        # against whatever's left as still-colored residual.
        colors[table_inlier_mask(xyz)] = np.array([0.6, 0.6, 0.6])
    ax.scatter(xyz_disp[:, 0], xyz_disp[:, 1], xyz_disp[:, 2], s=1.5, c=colors, alpha=0.4)

    if geom["bricks"]:
        b = geom["bricks"][0]  # smallest angle = most-vertical = the validated real brick
        pts = b["points"]
        proj = (pts - b["centroid"]) @ b["direction"]
        half = max(float(proj.max()), -float(proj.min()), 0.01)
        seg = to_display_frame(np.array([b["centroid"] - b["direction"] * half,
                                          b["centroid"] + b["direction"] * half]))
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], linewidth=5, color="black",
                label=f"brick ({b['angle_deg']}deg, n={b['n_points']})")

    if fk_axis is not None and fk_position is not None:
        render_fk_prior(ax, fk_axis, fk_position)

    gripper_pts = geom.get("gripper_region_points")
    if gripper_pts is not None:
        gp_disp = to_display_frame(gripper_pts)
        if finger_result is not None:
            # Finger lines are the main signal here -- raw region points stay as small
            # dim context underneath rather than the big highlighted blob.
            ax.scatter(gp_disp[:, 0], gp_disp[:, 1], gp_disp[:, 2], s=6, c="lime", alpha=0.35)
            render_finger_lines(ax, finger_result)
        else:
            fk_match = geom.get("gripper_fk_match")
            label = f"gripper (n={gripper_pts.shape[0]}"
            label += f", fk_angle={fk_match['angle_to_fk_deg']}deg, fk_dist={fk_match['pos_dist_m']*1000:.0f}mm)" \
                if fk_match else ")"
            ax.scatter(gp_disp[:, 0], gp_disp[:, 1], gp_disp[:, 2],
                        s=40, c="lime", edgecolors="darkgreen", linewidths=0.3, alpha=0.9,
                        label=label)

    lo, hi = bounds
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper left")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return frame_bgr


PROJECTION_PANEL_WIDTH_PX = 500  # panel 2's width when --projection is on; height always
                                  # matches panel 1's FIG_SIZE_PX height so cv2.hconcat lines up


def make_session_video(session_dir: Path, fps: float, every: int, highlight_table: bool = False,
                        out_name: str = "geometry.mp4", track_fingers: bool = False,
                        fk_prior: bool = False, skeleton: bool = False, projection: bool = False):
    needs_pose = fk_prior or skeleton or projection  # projection's panel 1 is the skeleton render,
                                                       # and panel 2 needs angles for tool0/width axis
    clouds = sorted((session_dir / "clouds").glob("*.npz"))
    if every > 1:
        clouds = clouds[::every]
    if not clouds:
        print(f"  {session_dir}: no clouds found, skipping")
        return

    # skeleton mode needs raw joint angles every frame (for ur3e_fk_chain), same as
    # fk_prior needs the derived axis/position quantities -- both read the same pose_log.
    # Loaded BEFORE bounds so skeleton mode can fold the arm's own reach into them.
    pose_rows = load_pose_log(session_dir) if needs_pose else []

    print(f"  {session_dir}: {len(clouds)} frames, computing display bounds...")
    bounds = compute_bounds(clouds, pose_rows=pose_rows if (skeleton or projection) else None)
    # ONE tracker per session, fed frames in chronological order -- it's a temporal
    # filter (persists a separation/axis estimate across frames), so a fresh session
    # must start with no history and frames must never be processed out of order.
    tracker = FingerTracker() if track_fingers else None
    brick_tracker = BrickTracker() if (skeleton or projection) else None
    brick_mode_counts = {"continuity": 0, "detected": 0, "moved": 0, "occluded": 0, "lost": 0}
    if needs_pose and not pose_rows:
        print(f"  {session_dir}: no pose_log.jsonl rows found -- "
              f"{'falling back to point-cloud-only gripper selection' if fk_prior and not (skeleton or projection) else 'skipping (needs joint angles)'}")
        if skeleton or projection:
            return

    out_path = session_dir / out_name
    # avc1 (H.264), not mp4v -- mp4v's color-matrix tagging is ambiguous enough that some
    # players (confirmed: macOS QuickTime/Preview) misrender red as green/cyan even though
    # the encoded pixel data itself is correct (verified 2026-09-11 by reading raw frame
    # bytes back out of an mp4v file -- BGR=[0,0,255], genuinely red). This OpenCV build has
    # FFMPEG support compiled in even without a standalone `ffmpeg` binary on PATH, so avc1
    # works without any extra install.
    frame_size = (FIG_SIZE_PX[0] + PROJECTION_PANEL_WIDTH_PX, FIG_SIZE_PX[1]) if projection else FIG_SIZE_PX
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), fps, frame_size)

    t0 = time.time()
    n_brick, n_gripper, n_fk_used = 0, 0, 0
    n_skipped_no_pose = 0
    mode_counts = {"detected": 0, "propagated": 0, "single": 0}
    resolvable_counts = {"resolvable": 0, "unresolvable": 0, "unknown": 0}
    for i, c in enumerate(clouds):
        xyz = load_cloud(c)
        title = f"{session_dir.name}  t={c.stem}  frame {i+1}/{len(clouds)}"

        if skeleton or projection:
            angles = angles_for_cloud(pose_rows, float(c.stem))
            if angles is None:
                n_skipped_no_pose += 1
                continue

            # ONE clustering pass, shared by both the skeleton panel's brick contour and the
            # projection panel's brick/other coloring, so they always agree with each other
            # (see compute_off_plane_clusters/BrickTracker docstrings) -- no more independent
            # per-panel re-classification that could disagree.
            off_plane, cluster_info = compute_off_plane_clusters(xyz)
            tracked_brick, brick_mode = brick_tracker.update(cluster_info)
            brick_mode_counts[brick_mode] += 1
            geom = {"bricks": []}
            is_brick = np.zeros(off_plane.shape[0], dtype=bool) if off_plane is not None else np.zeros(0, dtype=bool)
            if tracked_brick is not None:
                n_brick += 1
                geom["bricks"] = [{
                    "n_points": len(tracked_brick["idx"]), "angle_deg": round(tracked_brick["angle_deg"], 1),
                    "centroid": tracked_brick["centroid"], "direction": tracked_brick["direction"],
                    "points": tracked_brick["pts"], "mode": brick_mode,
                }]
                is_brick[tracked_brick["idx"]] = True

            frame = render_skeleton_frame(xyz, geom, bounds, title, angles)
            if projection:
                panel2 = render_width_projection_panel(off_plane, is_brick, angles,
                                                         PROJECTION_PANEL_WIDTH_PX, FIG_SIZE_PX[1])
                frame = hstack_frames(frame, panel2)
            writer.write(frame)
            if (i + 1) % 20 == 0 or i + 1 == len(clouds):
                print(f"    {i+1}/{len(clouds)} frames ({time.time()-t0:.1f}s elapsed)")
            continue

        fk_axis, fk_position, fk_width_axis = (None, None, None)
        if pose_rows:
            fk_axis, fk_position, fk_width_axis = fk_prior_for_cloud(pose_rows, float(c.stem))
            if fk_axis is not None:
                n_fk_used += 1
        geom = extract_geometry(xyz, include_points=True, fk_axis=fk_axis, fk_position=fk_position,
                                 fk_width_axis=fk_width_axis)
        if geom["bricks"]:
            n_brick += 1
        gripper_pts = geom.get("gripper_region_points")
        finger_result = None
        if gripper_pts is not None:
            n_gripper += 1
            if tracker is not None:
                finger_result = tracker.update(gripper_pts, width_axis_prior=fk_width_axis)
                mode_counts[finger_result["mode"]] += 1
                r = finger_result["resolvable"]
                resolvable_counts["unknown" if r is None else ("resolvable" if r else "unresolvable")] += 1
        frame = render_frame(xyz, geom, bounds, title, highlight_table=highlight_table,
                              finger_result=finger_result, fk_axis=fk_axis, fk_position=fk_position)
        writer.write(frame)
        if (i + 1) % 20 == 0 or i + 1 == len(clouds):
            elapsed = time.time() - t0
            print(f"    {i+1}/{len(clouds)} frames ({elapsed:.1f}s elapsed)")
    writer.release()
    extra = f", finger modes={mode_counts}, resolvable={resolvable_counts}" if track_fingers else ""
    fk_extra = f", fk_prior used in {n_fk_used} frames" if fk_prior else ""
    skel_extra = f", skipped (no pose match)={n_skipped_no_pose}" if skeleton else ""
    brick_extra = f", brick_modes={brick_mode_counts}" if (skeleton or projection) else ""
    print(f"  wrote {out_path} ({len(clouds)} frames, brick found in {n_brick}, "
          f"gripper found in {n_gripper}{extra}{fk_extra}{skel_extra}{brick_extra}, {time.time()-t0:.1f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="artifacts/grasp_sessions",
                         help="A session directory, or a parent directory of session_* dirs (default).")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--every", type=int, default=1, help="Use every Nth saved cloud frame.")
    parser.add_argument("--table-gray", action="store_true",
                         help="Color RANSAC's table-plane inliers plain gray instead of the depth "
                              "gradient (everything else keeps depth coloring). Writes to "
                              "geometry_table_gray.mp4, NOT geometry.mp4 -- doesn't overwrite it.")
    parser.add_argument("--fingers", action="store_true",
                         help="Draw two parallel finger lines (FingerTracker, temporally smoothed "
                              "across the session) instead of the raw gripper-region blob. Writes to "
                              "geometry_fingers.mp4, NOT geometry.mp4 -- doesn't overwrite it. "
                              "Combinable with --table-gray/--fk-prior.")
    parser.add_argument("--fk-prior", action="store_true",
                         help="Use the UR3e's own FK-predicted gripper axis/position (from each "
                              "frame's nearest pose_log.jsonl sample, see ur3e_fk.py) to pick which "
                              "off-plane cluster is the gripper, instead of the point-cloud-only "
                              "heuristic -- also draws the FK prediction as a dashed gray reference "
                              "line. Writes to geometry_fk_prior.mp4, NOT geometry.mp4.")
    parser.add_argument("--skeleton", action="store_true",
                         help="Dedicated render: table contour + brick contour (from the point "
                              "cloud) plus a robot skeleton (base_link->tool0, no gripper fingers) "
                              "and a kinematic grasp-plane quad from tool0 (both from the arm's own "
                              "FK/joint angles, NOT the point cloud). Overrides --table-gray/"
                              "--fingers/--fk-prior. Writes to geometry_skeleton.mp4.")
    parser.add_argument("--projection", action="store_true",
                         help="Side-by-side video: left panel is the --skeleton render, right panel "
                              "is every non-table point projected onto tool0's width axis, colored "
                              "brick (firebrick) vs other (gray). Implies --skeleton for the left "
                              "panel. Writes to geometry_skeleton_projection.mp4.")
    args = parser.parse_args()

    p = Path(args.path)
    if (p / "clouds").is_dir():
        sessions = [p]
    else:
        sessions = sorted(d for d in p.glob("session_*") if d.is_dir())

    if args.projection:
        out_name = "geometry_skeleton_projection.mp4"
    elif args.skeleton:
        out_name = "geometry_skeleton.mp4"
    else:
        name_parts = []
        if args.table_gray:
            name_parts.append("table_gray")
        if args.fingers:
            name_parts.append("fingers")
        if args.fk_prior:
            name_parts.append("fk_prior")
        out_name = "geometry_" + "_".join(name_parts) + ".mp4" if name_parts else "geometry.mp4"

    print(f"Rendering {len(sessions)} session video(s)...")
    for s in sessions:
        make_session_video(s, args.fps, args.every, highlight_table=args.table_gray,
                            out_name=out_name, track_fingers=args.fingers, fk_prior=args.fk_prior,
                            skeleton=args.skeleton, projection=args.projection)


if __name__ == "__main__":
    main()
