#!/usr/bin/env python3
"""
random_walk_move_to_blocks.py  (RUNS ON THE REMOTE CONSTRUCT ROS2 SESSION,
not here -- see run_align_xy_real.py's docstring for why real-robot scripts
live there instead of locally.)

Automates move_to_blocks.py's interactive jog commands (f/s/d/r/R/e/E) --
same step primitives, same magnitudes -- so camera+pose data can be
collected unattended once you've driven the arm close to the ROI by hand.
Meant to run in one terminal while record_grasp_session.py records in
another, same pairing move_to_blocks.py itself already has with that
script, just with the human jogging replaced by random sampling here.

Needs move_to_blocks.py copied to the same directory (reuses MoveToBlocks
and its step constants FORWARD_STEP/SIDE_STEP/DESCEND_STEP/
ROTATE_STEP_DEG/ELBOW_STEP_DEG so the walk's step size always matches
whatever's set there).

Bounded, not an unconstrained random walk: f/s/d are one-directional in
move_to_blocks.py (there's no "back away"/"ascend" command), so bare
uniform sampling would just monotonically drift the gripper toward the
blocks and into the table. Each candidate step is checked against a box
around the START pose (recorded at launch -- wherever you manually jogged
to) before executing; a step that would leave the box is skipped and
another is drawn instead, so the arm dithers around the ROI instead of
wandering off. xy/z bounds are checked against the real TF pose each
iteration; yaw/wrist_2 bounds are running offsets this script maintains
itself (no cheap way to read "yaw since start" back out of a quaternion),
which assumes nothing else moves the arm while this is running.

Usage:
  python3 random_walk_move_to_blocks.py [--steps 200] [--pause 1.0] [--seed 0]
"""
import argparse
import math
import random
import time

import rclpy

from move_to_blocks import (
    MoveToBlocks, FORWARD_STEP, SIDE_STEP, DESCEND_STEP,
    ROTATE_STEP_DEG, ELBOW_STEP_DEG,
)

# Box around the start pose the walk is kept inside. Not empirically
# tuned -- wide enough for meaningful camera-pose diversity, tight enough
# to keep the ROI in frame and the gripper off the table. If the walk
# keeps printing "all actions blocked", widen the relevant bound.
XY_RADIUS_M = 0.05
Z_DOWN_LIMIT_M = 0.05   # how far below the start height 'd' may descend
Z_UP_LIMIT_M = 0.01     # no primitive ascends; kept as a symmetric guard
YAW_RANGE_DEG = 30.0
WRIST2_RANGE_DEG = 40.0
MAX_CONSECUTIVE_SKIPS = 50


def within_translation_bounds(cur_pos, start_pos, dx, dy, dz):
    nx, ny, nz = cur_pos[0] + dx, cur_pos[1] + dy, cur_pos[2] + dz
    if abs(nx - start_pos[0]) > XY_RADIUS_M or abs(ny - start_pos[1]) > XY_RADIUS_M:
        return False
    if nz < start_pos[2] - Z_DOWN_LIMIT_M or nz > start_pos[2] + Z_UP_LIMIT_M:
        return False
    return True


def build_actions(node, offsets):
    """Each action is (label, predicted_pose_delta, execute_fn, apply_offset_fn)."""
    yaw_step = math.radians(ROTATE_STEP_DEG)
    wrist2_step = math.radians(ELBOW_STEP_DEG)

    def bump(key, delta):
        offsets[key] += delta

    return [
        ("f", (0.0, FORWARD_STEP, 0.0),
         lambda: node.plan_and_execute(0.0, FORWARD_STEP, 0.0), lambda: None),
        ("s", (-SIDE_STEP, 0.0, 0.0),
         lambda: node.plan_and_execute(-SIDE_STEP, 0.0, 0.0), lambda: None),
        ("d", (0.0, 0.0, DESCEND_STEP),
         lambda: node.plan_and_execute(0.0, 0.0, DESCEND_STEP), lambda: None),
        ("r", (0.0, 0.0, 0.0),
         lambda: node.plan_and_execute(0.0, 0.0, 0.0, d_yaw_rad=yaw_step),
         lambda: bump("yaw_deg", ROTATE_STEP_DEG)),
        ("R", (0.0, 0.0, 0.0),
         lambda: node.plan_and_execute(0.0, 0.0, 0.0, d_yaw_rad=-yaw_step),
         lambda: bump("yaw_deg", -ROTATE_STEP_DEG)),
        ("e", (0.0, 0.0, 0.0),
         lambda: node.jog_joint("wrist_2_joint", wrist2_step),
         lambda: bump("wrist2_deg", ELBOW_STEP_DEG)),
        ("E", (0.0, 0.0, 0.0),
         lambda: node.jog_joint("wrist_2_joint", -wrist2_step),
         lambda: bump("wrist2_deg", -ELBOW_STEP_DEG)),
    ]


def action_allowed(label, delta, cur_pos, start_pos, offsets):
    if label in ("f", "s", "d"):
        return within_translation_bounds(cur_pos, start_pos, *delta)
    if label == "r":
        return offsets["yaw_deg"] + ROTATE_STEP_DEG <= YAW_RANGE_DEG
    if label == "R":
        return offsets["yaw_deg"] - ROTATE_STEP_DEG >= -YAW_RANGE_DEG
    if label == "e":
        return offsets["wrist2_deg"] + ELBOW_STEP_DEG <= WRIST2_RANGE_DEG
    if label == "E":
        return offsets["wrist2_deg"] - ELBOW_STEP_DEG >= -WRIST2_RANGE_DEG
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--pause", type=float, default=1.0,
                         help="Seconds to sit still after each step, so the "
                              "camera/pose recorder captures a settled frame.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    rclpy.init()
    node = MoveToBlocks()
    node.cartesian_client.wait_for_service(timeout_sec=5.0)
    node.arm_client.wait_for_server(timeout_sec=5.0)

    start_pose = node.get_current_pose()
    start_pos = (start_pose.position.x, start_pose.position.y, start_pose.position.z)
    print(f"Start pose: x={start_pos[0]:.4f} y={start_pos[1]:.4f} z={start_pos[2]:.4f}")
    print(f"Bounds: xy=+/-{XY_RADIUS_M}m  z=[-{Z_DOWN_LIMIT_M},+{Z_UP_LIMIT_M}]m  "
          f"yaw=+/-{YAW_RANGE_DEG}deg  wrist2=+/-{WRIST2_RANGE_DEG}deg")
    print(f"Running {args.steps} steps, {args.pause}s pause between each. "
          "Start record_grasp_session.py (or equivalent) in another terminal first.\n"
          "Ctrl+C to stop early.\n")

    offsets = {"yaw_deg": 0.0, "wrist2_deg": 0.0}
    actions = build_actions(node, offsets)

    n_executed = 0
    n_failed = 0
    try:
        for step in range(args.steps):
            cur_pose = node.get_current_pose()
            cur_pos = (cur_pose.position.x, cur_pose.position.y, cur_pose.position.z)

            candidates = list(actions)
            random.shuffle(candidates)
            chosen = None
            for label, delta, execute_fn, apply_offset_fn in candidates:
                if action_allowed(label, delta, cur_pos, start_pos, offsets):
                    chosen = (label, execute_fn, apply_offset_fn)
                    break

            if chosen is None:
                print(f"Step {step}: all actions blocked by bounds, skipping.")
                if step > 0 and step % MAX_CONSECUTIVE_SKIPS == 0:
                    print("Repeatedly blocked -- bounds may be too tight. Stopping.")
                    break
                time.sleep(args.pause)
                continue

            label, execute_fn, apply_offset_fn = chosen
            ok = execute_fn()
            if ok:
                apply_offset_fn()
                n_executed += 1
            else:
                n_failed += 1
            print(f"Step {step}: action={label} ok={ok} "
                  f"(executed={n_executed} failed={n_failed})")
            time.sleep(args.pause)
    except KeyboardInterrupt:
        print("\nStopped by user.")

    print(f"\nDone: {n_executed} steps executed, {n_failed} failed to plan/execute.")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
