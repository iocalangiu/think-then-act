#!/usr/bin/env python3
"""
ur3e_fk.py  (runs locally, no rclpy dependency.)

Forward kinematics for the UR3e -- added 2026-09-11 after the point-cloud-
only gripper/finger extraction (extract_gripper_brick_geometry.py) proved
too unreliable on its own (min-side-fraction fix dropped confident
detections to 4/257 frames). Idea: use the ARM's own known joint angles as
a PRIOR for which off-plane cluster is the gripper, instead of guessing
from point-cloud geometry alone.

Uses NOMINAL (Universal Robots' published standard) DH parameters, not
this specific physical robot's own calibrated parameters -- calibration
differences between individual UR3e units are sub-degree/sub-mm scale,
far smaller than the 10-25deg ambiguity this is meant to resolve, and
pulling this rig's actual calibration would mean reconnecting to the
Construct session, which isn't available for already-recorded data.
Standard DH convention (not "Modified" DH):
    T_i = Rot_z(theta_i) . Trans_z(d_i) . Trans_x(a_i) . Rot_x(alpha_i)

VALIDATION: every recorded session's pose_log.jsonl has BOTH the joint
angles AND the TF-measured base_link->tool0 grip_pos (position only) at
the same timestamps -- run this module directly against a real
pose_log.jsonl to check the FK-predicted TCP position against that
ground truth before trusting the orientation output for anything. See
`if __name__ == "__main__"` below.

KNOWN GOTCHA, checked for during validation: Universal Robots' ROS driver
defines "base_link" as the DH chain's own "base" frame rotated 180deg
about Z (a legacy right-hand-rule compatibility convention) -- BASE_Z_
ROTATION_DEG below exists to correct for this if the raw DH chain doesn't
match recorded grip_pos directly.
"""
import math
from pathlib import Path

import numpy as np

# Nominal UR3e DH parameters (Standard DH), joint order matches JOINT_NAMES
# used throughout hardware/move_arm/*.py: shoulder_pan, shoulder_lift,
# elbow, wrist_1, wrist_2, wrist_3.
D1, A2, A3, D4, D5, D6 = 0.15185, -0.24355, -0.2132, 0.13105, 0.08535, 0.0921
A = [0.0, A2, A3, 0.0, 0.0, 0.0]
D = [D1, 0.0, 0.0, D4, D5, D6]
ALPHA = [math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0]

# See module docstring's KNOWN GOTCHA. CONFIRMED 2026-09-11 by validating
# against session_1789118350's recorded grip_pos: with 0.0 here, predicted
# x/y came out exactly NEGATED relative to the real (TF-measured) grip_pos
# (z matched exactly) -- the textbook signature of the "base" (DH root)
# vs "base_link" (TF root) 180deg-about-Z convention difference. Applied
# as an extra Rot_z BEFORE joint 1 in the chain, i.e. on the "base" side.
BASE_Z_ROTATION_DEG = 180.0


def dh_transform(theta, d, a, alpha):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def rot_z(angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def ur3e_fk(joint_angles):
    """joint_angles: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2,
    wrist_3] in radians (same order as JOINT_NAMES elsewhere in this repo).
    Returns the 4x4 homogeneous transform base_link -> tool0."""
    assert len(joint_angles) == 6, f"expected 6 joint angles, got {len(joint_angles)}"
    T = rot_z(math.radians(BASE_Z_ROTATION_DEG))
    for i in range(6):
        T = T @ dh_transform(joint_angles[i], D[i], A[i], ALPHA[i])
    return T


def ur3e_fk_chain(joint_angles):
    """Same chain as ur3e_fk, but returns EVERY intermediate joint origin
    (not just the final tool0 transform) -- for drawing a robot skeleton.
    Returns a list of 7 (3,) positions in base_link frame: [base_link
    origin, after joint 1, after joint 2, ..., after joint 6 (== tool0)].
    Consecutive pairs are exactly the line segments a stick-figure skeleton
    should draw between joints."""
    assert len(joint_angles) == 6, f"expected 6 joint angles, got {len(joint_angles)}"
    T = rot_z(math.radians(BASE_Z_ROTATION_DEG))
    positions = [T[:3, 3].copy()]
    for i in range(6):
        T = T @ dh_transform(joint_angles[i], D[i], A[i], ALPHA[i])
        positions.append(T[:3, 3].copy())
    return positions


def tool0_position(joint_angles):
    return ur3e_fk(joint_angles)[:3, 3]


GRIPPER_LENGTH_M = 0.10  # approximate flange(tool0)-to-fingertip distance for the mounted
                          # Robotiq gripper -- starting guess (roughly matches the ~79mm fk_dist
                          # observed matching a real cluster on 2026-09-11), NOT independently
                          # measured against this specific mount. tool0 itself is the wrist
                          # flange, not the fingers (see the TCP discussion, same date) -- this
                          # offset is what turns "predicted wrist position" into "predicted
                          # fingertip-area position" for cluster-matching purposes.


def tool0_fingertip_position(joint_angles, gripper_length=GRIPPER_LENGTH_M):
    """tool0 position nudged forward along the approach axis by the
    gripper's assumed length -- use THIS, not tool0_position, when
    matching against where the visible gripper/finger point-cloud
    structure actually is."""
    T = ur3e_fk(joint_angles)
    return T[:3, 3] + T[:3, 2] * gripper_length


def tool0_approach_axis(joint_angles):
    """tool0's local +Z axis in base_link frame -- the UR/ROS convention
    for a tool's approach/pointing direction. ASSUMPTION, not yet
    independently confirmed against this rig -- worth a visual sanity
    check once used."""
    return ur3e_fk(joint_angles)[:3, 2]


def tool0_width_axis(joint_angles):
    """tool0's local +X axis in base_link frame -- the Robotiq mount's
    finger-separation direction.

    UNRESOLVED TENSION, flagged rather than hidden: this has flipped once
    already and the two pieces of evidence genuinely disagree.
      - 2026-09-11 (session_1789118350, frame t=1789118368.294): local Y
        matched that frame's own real PCA-computed separation axis (from a
        confirmed 47-vs-70-point, 24.7mm split) within 12.5deg, while
        local X was 9.9deg from the camera's depth axis (predicting
        "unresolvable" on a frame that was visibly NOT).
      - 2026-09-15 (session_1789457462, frame t=1789457671.357): local X
        was visually confirmed against real fingers directly visible in
        that frame ("looks correct" -- local Y did not).
    Switched to X on the strength of the more recent, direct visual
    confirmation, but this contradicts the earlier Y evidence -- for a
    fixed physical mounting, the true answer can't depend on which
    session you look at. Not yet root-caused (candidates: the earlier
    "confirmed" 47/70 split may not have been two real fingers either,
    given how often this investigation found ambiguous/non-finger
    clusters; or the coupler mounting isn't at a clean 0/90deg offset from
    either X or Y). Re-validate against more frames before trusting
    either answer fully. Local Z was also checked and ruled out -- that's
    tool0_approach_axis, a different, already-used quantity.

    Unlike the approach axis (set by joints 1-5), this one's SPATIAL
    orientation is directly sensitive to wrist_3 -- wrist_3 rolls the tool
    about its own Z axis, which is exactly the rotation that swings this
    X axis around. This is the whole point of separating it from
    tool0_approach_axis: it's what lets wrist_3's value predict which way
    the two fingers are actually splayed in space, not just which way the
    gripper points."""
    return ur3e_fk(joint_angles)[:3, 0]


# base_link -> camera_depth_optical_frame extrinsics, confirmed 2026-09-08
# (see memory ur3e_sim2real.md). p_base = CAM_TO_BASE_R @ p_cam + CAM_TO_BASE_T.
CAM_TO_BASE_R = np.array([
    [-1.000, -0.000, 0.001],
    [-0.001, 0.174, -0.985],
    [0.000, -0.985, -0.174],
])
CAM_TO_BASE_T = np.array([0.500, 0.605, 0.100])


def base_point_to_camera(p_base):
    """Inverse of p_base = R @ p_cam + t (R is orthogonal, so R^-1 = R^T)."""
    return CAM_TO_BASE_R.T @ (np.asarray(p_base) - CAM_TO_BASE_T)


def base_direction_to_camera(d_base):
    """Same rotation, no translation -- for direction vectors."""
    return CAM_TO_BASE_R.T @ np.asarray(d_base)


ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


def extract_arm_angles(names, positions):
    """record_grasp_session.py's pose_log.jsonl saves /joint_states RAW
    and unfiltered -- on this rig that topic carries BOTH the 6 arm
    joints AND the 6 Robotiq gripper joints together in one message (12
    entries total, confirmed 2026-09-11), not just the arm. Filters down
    to the 6 arm joints, in ARM_JOINT_NAMES order. Returns None if any are
    missing."""
    d = dict(zip(names, positions))
    if not all(j in d for j in ARM_JOINT_NAMES):
        return None
    return [d[j] for j in ARM_JOINT_NAMES]


def _validate(pose_log_path):
    import json
    rows = [json.loads(line) for line in open(pose_log_path, encoding="utf-8")]
    rows = [r for r in rows if r.get("arm_joint_names") and r.get("grip_pos")]
    print(f"{pose_log_path}: {len(rows)} rows with joint names and grip_pos")
    if not rows:
        return
    errors = []
    example = None
    for r in rows:
        angles = extract_arm_angles(r["arm_joint_names"], r["arm_joint_positions"])
        if angles is None:
            continue
        predicted = tool0_position(angles)
        actual = np.array(r["grip_pos"])
        errors.append(np.linalg.norm(predicted - actual))
        example = (angles, predicted, actual)
    errors = np.array(errors)
    print(f"  {len(errors)} rows had all 6 arm joints present")
    print(f"  position error vs recorded grip_pos: mean={errors.mean()*1000:.1f}mm "
          f"median={np.median(errors)*1000:.1f}mm max={errors.max()*1000:.1f}mm")
    if example:
        angles, predicted, actual = example
        print(f"  example: joints={np.round(angles, 3)}")
        print(f"           predicted={predicted.round(4)}  actual={actual.round(4)}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/grasp_sessions/session_1789118350/pose_log.jsonl"
    _validate(Path(path))
