#!/usr/bin/env python3
"""
extract_gripper_brick_geometry.py  (runs locally -- processes already-saved
.npz point clouds, e.g. from artifacts/grasp_sessions/*/clouds/*.npz, no
rclpy/ROS dependency.)

Step 3 of the camera-guided fine-control policy pivot (see memory
ur3e_sim2real.md): per-frame extraction of the standing brick(s) and the
gripper's two fingers from a single point cloud, geometrically -- no color,
no camera<->base_link extrinsics, no forward kinematics needed. Everything
below is O(n) or O(n log n) (a couple of small eigendecompositions plus a
KDTree radius query), not RANSAC's random-iteration search -- "fast" was
an explicit requirement, and once the input is a clean, already-isolated
region, plain PCA gives the same dominant-axis answer as line-RANSAC would,
without the iteration overhead.

Pipeline per cloud:
  1. DOMINANT PLANE REMOVAL, cascading RANSAC (see remove_dominant_planes):
     fit+remove the table plane, then check whether a SECOND RANSAC plane
     explains a large chunk of what's left -- confirmed necessary against
     real data (2026-09-11): this rig's view also has a background wall
     (a second flat surface ~73deg off the table normal) that a single
     plane fit leaves behind, and without removing it too, the whole wall
     chunk gets misclassified as "gripper" downstream. A whole-cloud PCA
     fit (least-variance eigenvector = table normal) was tried FIRST and
     is WRONG here for the same reason -- the table doesn't dominate the
     cloud's variance once the wall/other bricks/tripod pole are also in
     frame, confirmed by checking the actual residual distribution (median
     25mm off a "flat" table is not sensor noise, it's a wrong plane).
     RANSAC's random-sample-and-count-inliers search finds the true
     largest consistent flat surface regardless of what else is in frame;
     each pass is vectorized per-iteration (~35ms for ~14k points,
     measured), so two passes is still "fast."
  2. FAST CLUSTERING of the off-plane residual via scipy.spatial.cKDTree's
     query_pairs (radius-neighbor graph) + union-find -- O(n log n), not
     cluster_blocks_by_color.py's O(n^2) blocked pairwise-distance approach
     (fine there for a few hundred already color-filtered points; here the
     off-plane residual can be much larger, so the scaling matters more).
  3. PER-CLUSTER PCA LINE FIT (dominant axis = largest eigenvector). A
     cluster whose axis is close to PARALLEL to the table normal (i.e. the
     cluster stands close to PERPENDICULAR to the table) is a brick
     candidate -- a brick standing upright has its long axis vertical.
     A brick LYING FLAT does not produce this signature (its axis would be
     roughly IN-PLANE, i.e. close to perpendicular to the normal, not
     parallel to it) -- by design, this algorithm cannot find a lying-flat
     brick, same limitation flagged when this approach was proposed
     (2026-09-11 conversation), not a bug to fix here.
  4. GRIPPER FINGERS from whatever off-plane clusters aren't brick
     candidates. Fingers can come out as ONE merged blob (common with a
     sparse/unorganized cloud where each finger only gets a handful of
     points) or as TWO separate connectivity clusters (when there's more
     point density and separation) -- both cases are handled the same way:
     find the best CANDIDATE PAIR of non-brick clusters (most-parallel
     directions, centroids within GRIPPER_MAX_SPAN_M of each other) if one
     exists, else fall back to the single largest non-brick cluster; pool
     those points, then PCA that pool and split it along its SECOND
     principal axis (the "width"/separation direction) at the largest gap
     in the 1D projection. This gives two finger point sets -- their own
     PCA lines, and the distance between their centroids as the
     closeness/opening-width proxy -- even when they were never separate
     connectivity clusters to begin with. A near-zero gap (closed gripper,
     fingers touching) is a valid reading, not a failure.

Usage (single cloud):
    python3 scripts/extract_gripper_brick_geometry.py path/to/cloud.npz
Usage (sample N clouds across a session directory, for a quick sanity pass):
    python3 scripts/extract_gripper_brick_geometry.py \\
        artifacts/grasp_sessions/session_XXXX --sample 8
Add --plot to also render a 3D scatter (table=gray, brick=colored per
candidate, gripper fingers=two more colors) via matplotlib, one PNG per
cloud next to it (<cloud>_geom.png).
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree, ConvexHull

PLANE_RANSAC_ITERS = 150           # vectorized per-iteration inlier count -- ~35ms for ~14k points,
                                    # confirmed against real data (see 2026-09-11 conversation)
PLANE_INLIER_DIST_M = 0.01         # calibrated against real data: RANSAC's own inlier count barely
                                    # changes between 8mm and 20mm on a real cloud, so 10mm sits in
                                    # the flat part of that curve rather than right at a knee
CLUSTER_EPS = 0.02                 # matches cluster_blocks_by_color.py's CLUSTER_EPS
MIN_CLUSTER_POINTS = 8
VERTICAL_ANGLE_TOLERANCE_DEG = 40.0  # max angle between a cluster's PCA axis and the table
                                      # normal to still call it a standing brick. Originally
                                      # tightened to 10deg (2026-09-11) to keep the GRIPPER (whose
                                      # own axis also reads fairly close to vertical on this rig) out
                                      # of the "bricks" list, back when angle was the ONLY brick
                                      # criterion. Loosened back up to 40deg (2026-09-15) once
                                      # touches_table became a second, independent criterion: on real
                                      # session_1789457462 data, the brick's own PCA axis reads
                                      # 14-30deg off vertical in most frames (small/partially-occluded
                                      # clusters, 23-100 points, give a noisy long-axis fit) even
                                      # though it visibly never moves -- the 10deg gate was silently
                                      # dropping ~55% of true-positive frames. Verified safe: sampling
                                      # every 3rd frame of the whole session, only 4 clusters anywhere
                                      # touch the table >5cm from the known brick location, and all 4
                                      # are within the last ~40 frames (213, 237, 246, 252) -- i.e.
                                      # they're the real topple/move near the end, not the gripper
                                      # touching down elsewhere. touches_table now does the
                                      # discrimination that angle alone used to have to do.
TABLE_CONTACT_TOLERANCE_M = 0.02   # max distance from a cluster's CLOSEST point to the table
                                    # plane to still call it "resting on the table" -- second,
                                    # independent brick criterion added 2026-09-15 (see
                                    # extract_geometry's inline comment): the vertical-angle test
                                    # alone kept accepting the gripper/wrist (elevated above the
                                    # table, but sometimes reading as near-vertical too).
MIN_GRIPPER_POINTS = 10
GRIPPER_MAX_SPAN_M = 0.15          # max centroid-to-centroid distance between two non-brick
                                    # clusters to still consider them a "two fingers" pair --
                                    # roughly a generous multiple of a plausible gripper opening
GRIPPER_CANDIDATE_MIN_POINTS = 30  # FK-prior path only: minimum points for a cluster to even be
                                    # considered a gripper/finger candidate -- excludes small noise
                                    # fragments that can otherwise win the FK-position match by
                                    # chance (confirmed 2026-09-11, see extract_geometry's comment)
MAX_PLANES_TO_REMOVE = 2           # table + (if present) one more large flat structure (the
                                    # background wall, confirmed present on this rig)
PLANE_MIN_FRACTION = 0.15          # a 2nd/3rd plane must explain at least this fraction of the
                                    # CURRENT residual to be worth removing -- stops the cascade
                                    # from eating into real gripper/brick points via a small
                                    # coincidental planar patch in leftover clutter
PLANE_MIN_EXTENT_M = 0.15          # a 2nd/3rd plane's own inliers must ALSO span at least this
                                    # bounding-box diagonal to be worth removing -- fixes a real bug
                                    # found 2026-09-15: when the off-table residual is SMALL and
                                    # dominated by just the brick (no gripper/wall also in view),
                                    # the brick's own ~250 points can be well-approximated by SOME
                                    # plane too (satisfying PLANE_MIN_FRACTION easily, since it's
                                    # then ~100% of the residual), and gets wrongly swallowed as if
                                    # it were the wall. Confirmed on real data: the buggy fits were
                                    # 7-10cm across (brick-sized) vs. 20cm for a genuine wall
                                    # removal on a different session. The first plane (table, i==0)
                                    # is exempt -- it's always removed regardless of size.


def fit_table_plane(xyz, iters=PLANE_RANSAC_ITERS, inlier_dist=PLANE_INLIER_DIST_M, seed=0):
    """RANSAC plane fit, vectorized per-iteration (each iteration's inlier
    count is one numpy pass over all n points, not a Python loop over
    points) -- ~35ms for ~14k points, measured against real data.

    A whole-cloud PCA fit (least-variance eigenvector = table normal) was
    tried first and is WRONG on real data: this scene also has a
    background wall, other bricks, and the tripod mount pole in view (see
    memory ur3e_sim2real.md), so the table does NOT dominate the cloud's
    variance the way it would in an idealized single-plane scene --
    confirmed 2026-09-11 by checking the actual residual distribution
    (median 25mm off a "flat" table is not table noise, it's a wrong
    plane). RANSAC's random-sample-and-count-inliers search finds the
    actual largest consistent flat surface regardless of what else is in
    frame, which PCA's single global fit cannot do. Returns (unit normal,
    a point on the plane)."""
    n = xyz.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iters, 3))
    p0, p1, p2 = xyz[idx[:, 0]], xyz[idx[:, 1]], xyz[idx[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-9
    normals = np.divide(normals, norms, out=np.zeros_like(normals), where=norms > 1e-9)

    best_count, best_normal, best_point = -1, None, None
    for k in range(iters):
        if not valid[k]:
            continue
        dist = np.abs((xyz - p0[k]) @ normals[k])
        count = int((dist < inlier_dist).sum())
        if count > best_count:
            best_count, best_normal, best_point = count, normals[k], p0[k]
    return best_normal, best_point


def remove_dominant_planes(xyz, max_planes=MAX_PLANES_TO_REMOVE, min_fraction=PLANE_MIN_FRACTION,
                            iters=PLANE_RANSAC_ITERS, inlier_dist=PLANE_INLIER_DIST_M):
    """Iteratively fits+removes RANSAC planes: the first is always removed
    (the table); later ones only if they explain at least `min_fraction`
    of the CURRENT residual, so a small stray coincidental plane in
    leftover clutter doesn't eat real gripper/brick points. See module
    docstring for why a second pass is needed on this rig (the background
    wall). Returns (final_residual, table_normal, table_point) --
    table_normal/table_point are specifically the FIRST plane's fit (a
    normal + a point that lies ON the table plane, so
    abs((p - table_point) @ table_normal) gives any point's distance to
    the table surface -- used for the brick's "resting on the table"
    check, added 2026-09-15); any later planes' fits are only used for the
    removal itself, not returned."""
    residual = xyz
    table_normal, table_point = None, None
    for i in range(max_planes):
        if residual.shape[0] < 20:
            break
        normal, point = fit_table_plane(residual, iters=iters, inlier_dist=inlier_dist)
        dist = np.abs((residual - point) @ normal)
        inlier_mask = dist < inlier_dist
        if i == 0:
            table_normal, table_point = normal, point
        else:
            if inlier_mask.mean() < min_fraction:
                break
            inliers = residual[inlier_mask]
            extent = inliers.max(axis=0) - inliers.min(axis=0)
            diag = float(np.linalg.norm(extent))
            if diag < PLANE_MIN_EXTENT_M:
                break
        residual = residual[~inlier_mask]
    return residual, table_normal, table_point


def fast_cluster(xyz, eps, min_points):
    """KDTree radius-neighbor graph + union-find -- O(n log n). Returns a
    list of index arrays into xyz, one per cluster with >= min_points."""
    n = xyz.shape[0]
    if n == 0:
        return []
    tree = cKDTree(xyz)
    pairs = tree.query_pairs(r=eps, output_type="ndarray")
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

    for i, j in pairs:
        union(int(i), int(j))

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [np.array(idx) for idx in groups.values() if len(idx) >= min_points]


def planar_contour(points, normal):
    """Boundary of a (roughly flat) point set, as an ordered polyline in
    3D -- project onto the plane's own 2D coordinates (an arbitrary but
    consistent in-plane basis), take the 2D convex hull, then lift the
    hull vertices back to 3D. A convex hull, not a concave alpha-shape --
    simpler and robust, at the cost of "rounding off" any concave notches
    in the true boundary (fine for a table, which has none). Returns an
    (N+1, 3) array with the first point repeated at the end, ready to plot
    as a closed loop."""
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    origin = points.mean(axis=0)
    pts2d = np.column_stack([(points - origin) @ u, (points - origin) @ v])
    hull = ConvexHull(pts2d)
    contour = points[hull.vertices]
    return np.vstack([contour, contour[:1]])


def hull_edges(points):
    """Edges of the 3D convex hull of a point set -- for a small,
    non-planar cluster (a brick) where a single flat projection (see
    planar_contour) wouldn't capture its actual 3D shape. Returns a list
    of (p0, p1) segment pairs, deduplicated (each triangle shares edges
    with its neighbors). Superseded for brick rendering by
    hull_outer_points (2026-09-15) -- kept since other callers may still
    want actual edges, not just the vertex set."""
    hull = ConvexHull(points)
    seen = set()
    edges = []
    for simplex in hull.simplices:
        for a, b in ((simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])):
            key = (min(a, b), max(a, b))
            if key not in seen:
                seen.add(key)
                edges.append((points[a], points[b]))
    return edges


def hull_outer_points(points):
    """Just the vertex points of the 3D convex hull -- the 'outer' points
    of a cluster -- for scattering as a highlighted outline instead of
    connecting them into edges (hull_edges). Added 2026-09-15: connected
    3D hull edges can visually criss-cross or look broken/unreliable from
    steep or close viewing angles (confirmed after switching to a real
    perspective projection for render_skeleton_frame), while plotting just
    the outer points themselves reads correctly from any viewpoint. Falls
    back to returning all the input points unchanged if there are too few
    to form a hull (<4)."""
    if points.shape[0] < 4:
        return points
    hull = ConvexHull(points)
    return points[hull.vertices]


def pca_axes(xyz):
    """Returns (centroid, eigvecs[:, ::-1], eigvals[::-1]) -- eigenvectors/
    values sorted DESCENDING, so column 0 is the dominant axis, column 1
    the second ("width"), column 2 the minor axis."""
    centroid = xyz.mean(axis=0)
    centered = xyz - centroid
    cov = (centered.T @ centered) / len(xyz)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return centroid, eigvecs[:, order], eigvals[order]


def split_at_largest_gap(xyz, axis):
    """Projects points onto `axis`, splits at the largest gap in the
    sorted 1D projection. Returns (idx_a, idx_b) into xyz, or None if the
    points can't be split into two nonempty groups."""
    proj = xyz @ axis
    order = np.argsort(proj)
    gaps = np.diff(proj[order])
    if len(gaps) == 0:
        return None
    split = int(np.argmax(gaps))
    idx_a, idx_b = order[: split + 1], order[split + 1 :]
    if len(idx_a) == 0 or len(idx_b) == 0:
        return None
    return idx_a, idx_b


MIN_GAP_FRACTION = 0.25   # the largest gap along the gripper region's width axis must be at
                           # least this fraction of the region's own total width-axis extent to
                           # count as a REAL separation between two fingers, not just ordinary
                           # point-spacing noise within one finger's own spread
MIN_GAP_ABS_M = 0.004      # AND at least this many meters -- guards against a tiny total extent
                           # (a handful of points on one finger) where even a 1mm gap could be a
                           # large FRACTION of that tiny span and falsely look "confidently bimodal"
STALE_AFTER_FRAMES = 8     # how many consecutive frames without a confident two-finger detection
                           # before the persisted separation/axis estimate is dropped, rather than
                           # kept propagating a possibly-outdated guess indefinitely
CAMERA_DEPTH_AXIS = np.array([0.0, 0.0, 1.0])  # camera_depth_optical_frame's own viewing/depth
                           # direction is trivially +Z of ITSELF, by definition -- no transform
                           # needed since everything here is already worked in camera frame
RESOLVABLE_ANGLE_THRESHOLD_DEG = 60.0  # min angle between the (wrist_3-informed) predicted
                           # finger-separation axis and the camera's own depth axis to call a
                           # frame "resolvable" -- a depth camera resolves LATERAL separation
                           # (across the image) far better than separation along its own depth
                           # axis, so if the predicted axis points mostly AT/AWAY from the camera
                           # (small angle to the depth axis), failing to find a gap tells us
                           # NOTHING about whether the gripper is actually open or closed -- it
                           # may just be optically unresolvable from this viewing angle this
                           # frame. Starting guess, not yet tuned against real data.
MIN_SIDE_POINTS = 10       # each side of a split must have at least this many points --
MIN_SIDE_FRACTION = 0.15   # AND at least this fraction of the region's total points. Added after
                           # visually checking a "confident" split on real data (2026-09-11) that
                           # turned out to be 234-vs-13 points: the gap-significance test alone
                           # accepted it (a real gap existed), but the "second finger" was really a
                           # handful of stray/outlier points peeling off the main blob, not a
                           # second comparably-sized finger. This rejects that shape of split while
                           # still accepting genuinely-balanced ones (e.g. 47-vs-70 passed fine).


BRICK_POSITION_TOLERANCE_M = 0.03  # max centroid movement (across frames) to still count as
                                    # "the same brick" for continuity -- generous relative to the
                                    # few-mm jitter a genuinely static brick showed in earlier
                                    # sessions, tight enough to reject a clearly different object
                                    # (the gripper, another brick) several cm away
BRICK_STALE_AFTER_FRAMES = 8       # consecutive "nothing vertical anywhere" frames before giving
                                    # up the persisted position -- matches FingerTracker's own
                                    # staleness window for consistency


class BrickTracker:
    """Temporal continuity for brick identification, ONE FRAME AT A TIME IN
    CHRONOLOGICAL ORDER (same requirement as FingerTracker) -- added
    2026-09-15 to fix a real instability in the stateless per-frame
    classifier: with no memory, a frame where a SECOND cluster elsewhere
    also happens to read as near-vertical (another brick, or the gripper
    posed vertically) has no way to prefer "the one already being
    tracked" -- it just reports whichever cluster currently has the
    smallest angle, which can make the reported brick identity flicker
    between different physical things frame to frame.

    Each call to update(cluster_info) -- cluster_info: list of dicts with
    idx/pts/centroid/direction/angle_deg for EVERY off-plane cluster this
    frame, vertical or not (NOT pre-filtered to brick candidates, unlike
    extract_geometry's own result["bricks"] -- continuity needs to see a
    cluster that's still nearby but NO LONGER vertical, since that's
    exactly what "it toppled in place" looks like):

      1. CONTINUITY FIRST: if a persisted position exists, look for the
         nearest cluster within BRICK_POSITION_TOLERANCE_M of it -- if one
         exists, keep tracking it (mode "continuity"), ignore whatever
         other clusters exist elsewhere this frame no matter how vertical
         they look. Position proximity ALONE is enough here (changed
         2026-09-15, see the inline comment on update() for the real-data
         reasoning) -- unlike a fresh search, continuity does not re-check
         verticality/touches_table, since a partially-occluded brick
         often fails both while still being the same physical brick.
      2. If continuity fails (nothing nearby at all) -- do NOT wait out a
         staleness window before reacting. Immediately run a fresh
         whole-frame search for the most-vertical, table-touching cluster
         anywhere. If one exists, this is either a cold start ("detected")
         or a real relocation ("moved" -- distinguishable by whether a
         position was already being tracked). Registering this promptly,
         rather than lagging behind a staleness counter, is the whole
         point -- a real topple/move should show up right away, not
         several frames
         late.
      3. If nothing vertical exists anywhere this frame either, increment
         a staleness counter and report the FROZEN last-known brick (mode
         "occluded") -- same centroid/direction/points as last confirmed,
         so the rendered label/contour holds steady through a brief
         occlusion (the gripper passing in front) instead of flickering
         off every frame it happens (changed 2026-09-15, see update()'s
         inline comment). Only after BRICK_STALE_AFTER_FRAMES consecutive
         unmatched frames is the persisted position actually forgotten and
         "lost" (empty) reported -- a sustained absence (the brick is
         genuinely gone -- toppled off the table, out of frame) should
         stop biasing future continuity checks toward a now-irrelevant old
         position, and should read in the render as the brick having
         disappeared rather than still being tracked.

    Note (updated 2026-09-15): since continuity no longer re-checks
    verticality, a topple THAT LANDS WITHIN BRICK_POSITION_TOLERANCE_M of
    the last tracked position (a brick tipping over roughly in place,
    rather than sliding/falling away) would be silently confirmed as
    "continuity" instead of surfacing as a mode change -- this tracker
    cannot geometrically distinguish that case from occlusion, by design.
    On the real session this was tuned against, the actual topple relocates
    the centroid well past that tolerance (>25cm), so it still shows up
    correctly as "lost", but a topple-in-place on a different session
    would not be caught here. See the F/T+visual-recheck labeling
    discussion (project memory) for how a real topple/success signal was
    meant to be derived instead of relying on this geometric proxy alone.
    """

    def __init__(self):
        self.last_centroid = None
        self.last_direction = None
        self.last_pts = None
        self.last_angle_deg = None
        self.stale_count = 0

    def update(self, cluster_info, vertical_tol_deg=VERTICAL_ANGLE_TOLERANCE_DEG):
        # "touches_table" (added 2026-09-15, see extract_geometry's matching comment) is
        # required alongside the vertical-angle test for a FRESH detection (below) -- a real
        # brick rests ON the table; the angle test alone kept accepting the gripper/wrist too,
        # which is near-vertical sometimes but never actually touching the table surface.
        #
        # CONTINUITY, by contrast, deliberately does NOT re-require either gate (changed
        # 2026-09-15 after checking real session_1789457462 data): during active
        # gripper-brick interaction the gripper occludes most of the brick, so the only
        # visible fragment is often a small, partial, non-vertical-reading sliver that fails
        # both tests even though it's still physically the same brick sitting in the same
        # spot -- with the gate applied here too, continuity fell through to "lost" for most
        # of the interaction phase (58/253 frames), which is exactly the "recluster from
        # scratch every frame" behavior continuity exists to avoid. Position proximity alone
        # (BRICK_POSITION_TOLERANCE_M) is the only requirement to keep tracking; a real
        # topple/move is still caught because it also relocates the centroid past that
        # tolerance (confirmed: the actual end-of-session topple lands a cluster >25cm away,
        # far past the 3cm window), so continuity correctly fails there instead of confirming
        # a lying brick as "still standing".
        continuity = None
        if self.last_centroid is not None and cluster_info:
            near = [c for c in cluster_info
                    if float(np.linalg.norm(c["centroid"] - self.last_centroid)) <= BRICK_POSITION_TOLERANCE_M]
            if near:
                continuity = min(near, key=lambda c: float(np.linalg.norm(c["centroid"] - self.last_centroid)))

        if continuity is not None:
            self.last_centroid = continuity["centroid"]
            self.last_direction = continuity["direction"]
            self.last_pts = continuity["pts"]
            self.last_angle_deg = continuity["angle_deg"]
            self.stale_count = 0
            return continuity, "continuity"

        vertical = [c for c in cluster_info
                    if c["angle_deg"] <= vertical_tol_deg and c.get("touches_table", True)]
        if vertical:
            best = min(vertical, key=lambda c: c["angle_deg"])
            was_tracking = self.last_centroid is not None
            moved = was_tracking and float(np.linalg.norm(best["centroid"] - self.last_centroid)) > BRICK_POSITION_TOLERANCE_M
            self.last_centroid = best["centroid"]
            self.last_direction = best["direction"]
            self.last_pts = best["pts"]
            self.last_angle_deg = best["angle_deg"]
            self.stale_count = 0
            return best, ("moved" if moved else "detected")

        # Nothing matched this frame at all -- neither a nearby cluster (continuity) nor a
        # fresh vertical+table-touching one anywhere. Rather than blank the brick out
        # immediately (which is what made the rendered video flicker the brick label on/off
        # constantly during ordinary gripper occlusion -- 36/253 frames on the real session
        # this was checked against, spread through the WHOLE session, not clustered at the
        # actual end-of-session topple), keep showing the FROZEN last-known brick (mode
        # "occluded") for up to BRICK_STALE_AFTER_FRAMES frames -- only once that grace period
        # is exceeded do we actually forget the position and report "lost" (empty), which is
        # what should read as "toppled/gone", added 2026-09-15 per user request.
        self.stale_count += 1
        if self.last_centroid is None or self.stale_count > BRICK_STALE_AFTER_FRAMES:
            self.last_centroid, self.last_direction, self.last_pts, self.last_angle_deg = None, None, None, None
            return None, "lost"
        frozen = {"centroid": self.last_centroid, "direction": self.last_direction,
                  "pts": self.last_pts, "angle_deg": self.last_angle_deg, "idx": np.array([], dtype=int)}
        return frozen, "occluded"


class FingerTracker:
    """Extracts two parallel finger lines from a gripper region's raw
    point set, ONE FRAME AT A TIME IN CHRONOLOGICAL ORDER (unlike
    extract_geometry, which is stateless/per-frame) -- added 2026-09-11
    after a simple single-frame largest-gap split proved unreliable
    (lopsided splits like 632-vs-1 points) because the gripper region
    often only contains ONE visible finger, or two fingers merged/
    touching with no real gap, not because the split logic itself was
    wrong.

    Each call to update():
      1. PCAs the region -> length_axis (finger's long direction) and
         width_axis (separation direction).
      2. Tests whether there's a REAL gap along width_axis (see
         MIN_GAP_FRACTION/MIN_GAP_ABS_M) -- if so, this frame confidently
         shows two separate fingers: split, record both centroids, and
         PERSIST this separation distance + width_axis as the current
         best estimate.
      3. If not confident this frame (single finger visible, or fingers
         merged with no resolvable gap), FALL BACK to the persisted
         estimate (if it's not stale) rather than forcing a bad split or
         reporting nothing -- draws two predicted finger positions
         centered on this frame's own (still reliable) region centroid,
         offset by +/- half the persisted separation along the persisted
         width_axis.
      4. If the persisted estimate has gone stale (no confident detection
         for STALE_AFTER_FRAMES frames) or never existed, reports a
         single line only -- doesn't invent a second finger with no
         recent evidence at all.

    width_axis's sign is arbitrary per PCA call (eigenvectors have no
    canonical sign) -- flipped each frame to stay aligned with the
    previous frame's, so the persisted separation doesn't randomly
    "swap sides" between frames for no physical reason.
    """

    def __init__(self):
        self.last_separation = None
        self.last_width_axis = None
        self.stale_count = 0

    def update(self, gripper_pts, width_axis_prior=None):
        """width_axis_prior: the UR3e's own FK-predicted, wrist_3-informed
        finger-separation axis for THIS frame, already in camera frame
        (see ur3e_fk.tool0_width_axis) -- optional. When given, every
        returned dict also carries "resolvable": whether this frame's
        viewing geometry could plausibly show two separate fingers at all
        (see RESOLVABLE_ANGLE_THRESHOLD_DEG's comment) -- None if no prior
        was given, so a frame with no FK data doesn't silently look like a
        confirmed "resolvable" reading."""
        resolvable = None
        if width_axis_prior is not None:
            angle_to_depth = math.degrees(math.acos(
                min(1.0, abs(float(np.dot(width_axis_prior, CAMERA_DEPTH_AXIS))))))
            resolvable = angle_to_depth >= RESOLVABLE_ANGLE_THRESHOLD_DEG

        centroid, eigvecs, _eigvals = pca_axes(gripper_pts)
        length_axis, width_axis = eigvecs[:, 0], eigvecs[:, 1]
        if self.last_width_axis is not None and np.dot(width_axis, self.last_width_axis) < 0:
            width_axis = -width_axis

        proj = gripper_pts @ width_axis
        order = np.argsort(proj)
        sorted_proj = proj[order]
        total_extent = float(sorted_proj[-1] - sorted_proj[0]) if len(sorted_proj) > 1 else 0.0
        gaps = np.diff(sorted_proj)

        confident, split = False, None
        if len(gaps) > 0 and total_extent > 0:
            split = int(np.argmax(gaps))
            max_gap = float(gaps[split])
            n_total = len(order)
            min_side = min(split + 1, n_total - split - 1)
            confident = (max_gap > MIN_GAP_ABS_M and max_gap > MIN_GAP_FRACTION * total_extent
                         and min_side >= MIN_SIDE_POINTS and min_side >= MIN_SIDE_FRACTION * n_total)

        length_half = float(max((gripper_pts - centroid) @ length_axis, key=abs, default=0.0))
        length_half = max(abs(length_half), 0.01)

        if confident:
            idx_a, idx_b = order[: split + 1], order[split + 1 :]
            pts_a, pts_b = gripper_pts[idx_a], gripper_pts[idx_b]
            centroid_a, centroid_b = pts_a.mean(axis=0), pts_b.mean(axis=0)
            separation = float(np.linalg.norm(centroid_a - centroid_b))
            self.last_separation, self.last_width_axis, self.stale_count = separation, width_axis, 0
            return {
                "mode": "detected", "length_axis": length_axis, "length_half": length_half,
                "separation_m": separation, "resolvable": resolvable,
                "finger1_centroid": centroid_a, "finger1_n_points": len(idx_a),
                "finger2_centroid": centroid_b, "finger2_n_points": len(idx_b),
            }

        self.stale_count += 1
        if self.last_separation is not None and self.stale_count <= STALE_AFTER_FRAMES:
            half = self.last_separation / 2
            return {
                "mode": "propagated", "length_axis": length_axis, "length_half": length_half,
                "separation_m": self.last_separation, "resolvable": resolvable,
                "finger1_centroid": centroid - self.last_width_axis * half, "finger1_n_points": None,
                "finger2_centroid": centroid + self.last_width_axis * half, "finger2_n_points": None,
            }

        return {
            # "single" + resolvable=True is strong evidence of a genuinely closed/touching
            # gripper (viewing geometry favored seeing two fingers, and didn't); "single" +
            # resolvable=False means this frame's geometry couldn't have shown two fingers
            # regardless of the gripper's true state -- no claim either way.
            "mode": "single", "length_axis": length_axis, "length_half": length_half,
            "separation_m": None, "resolvable": resolvable,
            "finger1_centroid": centroid, "finger1_n_points": int(gripper_pts.shape[0]),
            "finger2_centroid": None, "finger2_n_points": None,
        }


FK_POSITION_SCALE_M = 0.05   # normalizes the FK-prior position-distance term against a
                              # roughly-gripper-scale distance -- starting guess, not yet tuned
FK_ANGLE_SCALE_DEG = 30.0    # normalizes the FK-prior axis-angle term -- same scale as
                              # VERTICAL_ANGLE_TOLERANCE_DEG's own starting-guess magnitude


def extract_geometry(xyz, include_points=False, fk_axis=None, fk_position=None, fk_width_axis=None):
    """Returns a dict: {bricks: [...], gripper: {...} | None, reason: str?}.
    Every brick candidate and the gripper (if found) carry enough to
    reconstruct a line: centroid + direction, plus n_points and (for
    bricks) angle_deg from the table normal. Pass include_points=True to
    also attach the raw point coordinates for each ("points" key) -- off
    by default since nothing needs it but visualization (make_geometry_
    videos.py), and it roughly doubles the returned payload size.

    Pass fk_axis/fk_position (the UR3e's own FK-predicted tool0 approach
    axis and fingertip position, BOTH ALREADY IN CAMERA FRAME -- see
    ur3e_fk.py) to
    use them as a PRIOR for gripper cluster selection: added 2026-09-11
    after the point-cloud-only heuristic (best near-parallel/nearby PAIR
    of non-brick clusters, else the largest one) proved too unreliable on
    its own. When given, every off-plane cluster (not just ones that
    failed the brick's vertical-angle test) is scored by axis alignment +
    position proximity to the FK prediction, and the best match is taken
    directly as the gripper -- a real kinematic signal beats guessing from
    point-cloud shape alone. The winning cluster is excluded from the
    brick candidate list even if its own angle would have qualified it,
    since a validated FK match is stronger evidence than the angle
    heuristic. Also pass fk_width_axis (tool0's wrist_3-informed local Y,
    also camera frame -- see ur3e_fk.tool0_width_axis) to score candidate
    PAIRS of clusters by whether their OWN separation vector matches the
    predicted finger-separation direction -- confirmed 2026-09-11 to be a
    far sharper discriminator than position matching alone (see this
    function's inline comment for the numbers). Falls back to the old
    point-cloud-only selection when fk_axis/fk_position aren't given (e.g.
    single-cloud CLI usage with no known robot pose)."""
    result = {"bricks": [], "gripper": None}
    if xyz.shape[0] < 50:
        result["reason"] = "too few points"
        return result

    off_plane, normal, table_point = remove_dominant_planes(xyz)
    if normal is None or off_plane.shape[0] < MIN_CLUSTER_POINTS:
        result["reason"] = "no off-plane points"
        return result

    clusters = fast_cluster(off_plane, CLUSTER_EPS, MIN_CLUSTER_POINTS)
    if not clusters:
        result["reason"] = "no cluster >= min_points"
        return result

    cluster_info = []
    for idx in clusters:
        pts = off_plane[idx]
        centroid, eigvecs, _eigvals = pca_axes(pts)
        direction = eigvecs[:, 0]
        angle_deg = math.degrees(math.acos(min(1.0, abs(float(np.dot(direction, normal))))))
        # A real standing brick rests ON the table -- its base should come within a couple cm
        # of the table surface. Added 2026-09-15 as a second, independent brick criterion
        # (alongside the vertical-angle test) after the angle test alone proved too permissive
        # in practice (the gripper/wrist, elevated well above the table, kept passing it too).
        dist_to_table = np.abs((pts - table_point) @ normal)
        touches_table = bool(dist_to_table.min() < TABLE_CONTACT_TOLERANCE_M)
        cluster_info.append({"idx": idx, "pts": pts, "centroid": centroid, "direction": direction,
                              "angle_deg": angle_deg, "touches_table": touches_table})

    gripper_pts, gripper_fk_match = None, None
    if fk_axis is not None and fk_position is not None:
        # FK-PRIOR PATH: a UNIFIED search over every SINGLE cluster (>=
        # GRIPPER_CANDIDATE_MIN_POINTS, to exclude noise fragments -- see
        # below) and every PAIR of such clusters within GRIPPER_MAX_SPAN_M.
        #
        # PAIRS are scored PRIMARILY by whether their OWN separation vector
        # (centroid_b - centroid_a) aligns with fk_width_axis, the wrist_3-
        # informed predicted finger-separation direction -- confirmed
        # 2026-09-11 to be a FAR sharper discriminator than matching
        # position/approach-axis alone: on a known-good frame the true pair
        # scored 8.9deg off the predicted width axis vs. 41.7deg for the
        # next-best candidate (~5x cleaner separation), whereas position-
        # based scoring put the true pair only 3rd, within 2% of two wrong
        # candidates (GRIPPER_LENGTH_M is a guess, so absolute position
        # matching has real error; a direction comparison has no such
        # error). Falls back to the old position+approach-axis pair scoring
        # if fk_width_axis isn't available.
        #
        # SINGLE clusters (no separation vector exists) still use
        # position+approach-axis matching -- unchanged from before.
        #
        # GRIPPER_CANDIDATE_MIN_POINTS excludes small clusters from being
        # candidates at all: also confirmed 2026-09-11 necessary -- a
        # 16-point noise fragment happening to sit close to the (uncertain)
        # predicted position otherwise won the single-cluster search
        # outright, then dragged in only one real finger as its "best
        # partner," losing the second finger entirely.
        best_group, best_score = None, None
        n = len(cluster_info)
        for i in range(n):
            a = cluster_info[i]
            if len(a["idx"]) >= GRIPPER_CANDIDATE_MIN_POINTS:
                angle_to_fk = math.degrees(math.acos(min(1.0, abs(float(np.dot(a["direction"], fk_axis))))))
                pos_dist = float(np.linalg.norm(a["centroid"] - fk_position))
                single_score = pos_dist / FK_POSITION_SCALE_M + angle_to_fk / FK_ANGLE_SCALE_DEG
                if best_score is None or single_score < best_score:
                    best_score, best_group = single_score, (i,)
            if len(a["idx"]) < GRIPPER_CANDIDATE_MIN_POINTS:
                continue
            for j in range(i + 1, n):
                b = cluster_info[j]
                if len(b["idx"]) < GRIPPER_CANDIDATE_MIN_POINTS:
                    continue
                span = float(np.linalg.norm(a["centroid"] - b["centroid"]))
                if span > GRIPPER_MAX_SPAN_M:
                    continue
                sep_vec = b["centroid"] - a["centroid"]
                sep_norm = float(np.linalg.norm(sep_vec))
                if fk_width_axis is not None and sep_norm > 1e-9:
                    angle_to_width = math.degrees(math.acos(min(1.0, abs(float(
                        np.dot(sep_vec / sep_norm, fk_width_axis))))))
                    pair_score = angle_to_width / FK_ANGLE_SCALE_DEG + (span / GRIPPER_MAX_SPAN_M) * 0.2
                else:
                    na, nb = len(a["idx"]), len(b["idx"])
                    combined_centroid = (a["centroid"] * na + b["centroid"] * nb) / (na + nb)
                    parallelism = abs(float(np.dot(a["direction"], b["direction"])))
                    angle_to_fk_pair = math.degrees(math.acos(min(1.0, abs(float(np.dot(a["direction"], fk_axis))))))
                    pos_dist_pair = float(np.linalg.norm(combined_centroid - fk_position))
                    pair_score = (pos_dist_pair / FK_POSITION_SCALE_M + angle_to_fk_pair / FK_ANGLE_SCALE_DEG
                                  + (1.0 - parallelism))
                if best_score is None or pair_score < best_score:
                    best_score, best_group = pair_score, (i, j)

        if best_group is not None:
            gripper_pts = np.vstack([cluster_info[i]["pts"] for i in best_group])
            anchor = cluster_info[best_group[0]]
            gripper_fk_match = {
                "angle_to_fk_deg": round(math.degrees(math.acos(min(1.0, abs(float(
                    np.dot(anchor["direction"], fk_axis)))))), 1),
                "pos_dist_m": round(float(np.linalg.norm(
                    gripper_pts.mean(axis=0) - fk_position)), 4),
                "score": round(best_score, 3),
                "n_clusters_merged": len(best_group),
            }
        brick_clusters = [c for i, c in enumerate(cluster_info)
                           if (best_group is None or i not in best_group)
                           and c["angle_deg"] <= VERTICAL_ANGLE_TOLERANCE_DEG and c["touches_table"]]
    else:
        # POINT-CLOUD-ONLY FALLBACK (original heuristic, no known robot pose):
        # brick = close to vertical AND resting on the table; gripper = the best
        # near-parallel, nearby PAIR of the REMAINING clusters if one exists (handles
        # fingers as two separate connectivity clusters), else the single largest
        # remaining cluster (handles fingers merged into one blob).
        brick_clusters, non_brick_clusters = [], []
        for c in cluster_info:
            is_brick = c["angle_deg"] <= VERTICAL_ANGLE_TOLERANCE_DEG and c["touches_table"]
            (brick_clusters if is_brick else non_brick_clusters).append(c)

        if len(non_brick_clusters) >= 2:
            best_pair, best_pair_score = None, None
            for i in range(len(non_brick_clusters)):
                for j in range(i + 1, len(non_brick_clusters)):
                    a, b = non_brick_clusters[i], non_brick_clusters[j]
                    span = float(np.linalg.norm(a["centroid"] - b["centroid"]))
                    if span > GRIPPER_MAX_SPAN_M:
                        continue
                    parallelism = abs(float(np.dot(a["direction"], b["direction"])))  # 1.0 = parallel
                    pair_score = parallelism - span  # prefer parallel AND close
                    if best_pair_score is None or pair_score > best_pair_score:
                        best_pair_score, best_pair = pair_score, (a, b)
            if best_pair is not None:
                gripper_pts = np.vstack([best_pair[0]["pts"], best_pair[1]["pts"]])
        if gripper_pts is None and non_brick_clusters:
            largest = max(non_brick_clusters, key=lambda c: len(c["idx"]))
            gripper_pts = largest["pts"]

    result["bricks"] = [
        {"n_points": len(c["idx"]), "angle_deg": round(c["angle_deg"], 1),
         "centroid": c["centroid"], "direction": c["direction"],
         **({"points": c["pts"]} if include_points else {})}
        for c in sorted(brick_clusters, key=lambda c: c["angle_deg"])
    ]

    if gripper_pts is None:
        return result
    if gripper_fk_match is not None:
        result["gripper_fk_match"] = gripper_fk_match

    if gripper_pts.shape[0] < MIN_GRIPPER_POINTS:
        result["reason"] = result.get("reason", "") + "; gripper region too small"
        return result

    # The selected gripper REGION (before any attempt to split it into two
    # fingers) is exposed independently of split success -- a video/plot
    # that just wants "the gripper's points, highlighted" shouldn't lose
    # them because the two-finger split heuristic below happened to fail
    # on this particular frame.
    if include_points:
        result["gripper_region_points"] = gripper_pts

    g_centroid, g_eigvecs, _g_eigvals = pca_axes(gripper_pts)
    length_axis, width_axis = g_eigvecs[:, 0], g_eigvecs[:, 1]
    split = split_at_largest_gap(gripper_pts, width_axis)
    if split is None:
        result["reason"] = result.get("reason", "") + "; gripper region couldn't be split"
        return result

    idx_a, idx_b = split
    pts_a, pts_b = gripper_pts[idx_a], gripper_pts[idx_b]
    dir_a = pca_axes(pts_a)[1][:, 0] if len(pts_a) >= 2 else length_axis
    dir_b = pca_axes(pts_b)[1][:, 0] if len(pts_b) >= 2 else length_axis
    centroid_a, centroid_b = pts_a.mean(axis=0), pts_b.mean(axis=0)

    result["gripper"] = {
        "n_points": int(gripper_pts.shape[0]),
        "finger1_centroid": centroid_a, "finger1_direction": dir_a, "finger1_n_points": len(idx_a),
        "finger2_centroid": centroid_b, "finger2_direction": dir_b, "finger2_n_points": len(idx_b),
        "separation_m": float(np.linalg.norm(centroid_a - centroid_b)),
        **({"finger1_points": pts_a, "finger2_points": pts_b} if include_points else {}),
    }
    return result


def load_cloud(path):
    data = np.load(path)
    return data["xyz"].astype(np.float64)


def plot_geometry(xyz, geom, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1, c="lightgray", alpha=0.3, label="cloud")

    for i, b in enumerate(geom["bricks"]):
        c, d = b["centroid"], b["direction"]
        seg = np.array([c - d * 0.03, c + d * 0.03])
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], linewidth=3, label=f"brick{i} ({b['angle_deg']}deg)")

    g = geom["gripper"]
    if g:
        for name, color in (("finger1", "red"), ("finger2", "blue")):
            c, d = g[f"{name}_centroid"], g[f"{name}_direction"]
            seg = np.array([c - d * 0.02, c + d * 0.02])
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], linewidth=3, color=color, label=name)
        ax.plot(*zip(g["finger1_centroid"], g["finger2_centroid"]), linestyle="--", color="green",
                label=f"separation={g['separation_m']*1000:.1f}mm")

    ax.legend(fontsize=7)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def report(path, plot):
    xyz = load_cloud(path)
    t0 = time.perf_counter()
    geom = extract_geometry(xyz)
    dt_ms = (time.perf_counter() - t0) * 1000
    print(f"\n=== {path} ({xyz.shape[0]} points, {dt_ms:.1f}ms) ===")
    if "reason" in geom:
        print(f"  note: {geom['reason']}")
    print(f"  bricks: {len(geom['bricks'])}")
    for i, b in enumerate(geom["bricks"]):
        print(f"    [{i}] n={b['n_points']} angle={b['angle_deg']}deg centroid={b['centroid'].round(4)}")
    if geom["gripper"]:
        g = geom["gripper"]
        print(f"  gripper: n={g['n_points']} separation={g['separation_m']*1000:.1f}mm "
              f"(finger1 n={g['finger1_n_points']}, finger2 n={g['finger2_n_points']})")
    else:
        print("  gripper: not found")
    if plot:
        out = Path(path).with_suffix("").with_name(Path(path).stem + "_geom.png")
        plot_geometry(xyz, geom, out)
        print(f"  wrote {out}")
    return geom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help=".npz cloud file, or a session directory (with --sample)")
    parser.add_argument("--sample", type=int, default=0,
                         help="If `path` is a session directory, sample this many clouds evenly across it.")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    p = Path(args.path)
    if p.is_dir():
        clouds = sorted((p / "clouds").glob("*.npz"))
        if args.sample > 0 and len(clouds) > args.sample:
            idx = np.linspace(0, len(clouds) - 1, args.sample).astype(int)
            clouds = [clouds[i] for i in idx]
        for c in clouds:
            report(c, args.plot)
    else:
        report(p, args.plot)


if __name__ == "__main__":
    main()
