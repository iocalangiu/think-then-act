#!/usr/bin/env python3
"""
run_pick_sequence_real.py  (RUNS ON THE REMOTE CONSTRUCT ROS2 SESSION, not
here -- see run_subgoal_real.py's docstring for why.)

Chains run_subgoal_real.py's per-step-confirm rollout loop across
subgoals into one sequence: align_xy until done -> descend until done ->
close the gripper -> return to saved_pose.json (the SAME flat
{joint_name: position} file move_to_blocks.py's 'h' command
(go_home/go_to_joint_positions) reads -- NOT the closed-loop script's
"wherever the arm started" convention, and NOT goal_pose.json, which is a
different file with a different {"joints": {...}, "cartesian": {...}}
shape used for the perception-bypass block position).

Deliberately keeps run_subgoal_real.py's per-step
"Execute this step? [Enter=yes, q=abort]" confirm during align_xy and
descend, unchanged, since that's the version already validated on this
rig -- the closed-loop (no-confirm) script exists separately for once
you trust a given goal_pose.json/setup. What's "continuous" here is
CHAINING the phases: once align_xy reaches its criterion, descend starts
automatically (no extra prompt to advance phases), and once descend
reaches its criterion, the gripper close and the return home also run
automatically with no further confirms. Answering 'q' at any per-step
prompt aborts that phase (and the rest of the sequence) the same way
run_subgoal_real.py's 'q' does, and Ctrl+C works at any point too --
either way, home is always attempted in a `finally`.

Gating: descend only runs if align_xy actually reached its criterion
(not MAX_STEPS/a failed plan/'q'), and the gripper only closes if
descend actually reached its criterion -- so a partial/aborted run never
proceeds to closing on the wrong position.

Needs BOTH weight files present next to this script:
  align_xy_policy_weights.npz, descend_policy_weights.npz
...and saved_pose.json already present (move_to_blocks.py only READS it
via 'h', it doesn't write it -- if it doesn't exist yet on this rig,
create it once by jogging to a safe retracted pose and writing
{joint_name: position} for all six JOINT_NAMES by hand, or add a
save-to-saved_pose.json command to move_to_blocks.py first).

GRIPPER INTERFACE -- confirmed 2026-09-01 via a direct
`ros2 action send_goal /robotiq_gripper_controller/gripper_cmd
control_msgs/action/GripperCommand "{command: {position: 1, max_effort:
20.0}}"` on this rig: it IS the GripperCommand action (the earlier
worry, based on /gripper/cmd/joint_states/stat looking like a separate
topic-based interface, was unfounded -- those are just status topics
alongside it). position=1/max_effort=20.0 confirmed as the values that
actually close it, not the 0.8/10.0 placeholder ur3e_cartesian_move_and_
grip.py guessed.

Usage:
  python3 scripts/run_pick_sequence_real.py
"""
import json
import time

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetCartesianPath
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectoryPoint

GROUP_NAME = "ur_manipulator"
LINK_NAME = "tool0"
BASE_FRAME = "base_link"
ARM_ACTION_NAME = "/scaled_joint_trajectory_controller/follow_joint_trajectory"
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
MIN_FRACTION = 0.9
GOAL_PATH = "goal_pose.json"      # perception-bypass block position, see run_subgoal_real.py
HOME_PATH = "saved_pose.json"     # flat {joint_name: position}, see move_to_blocks.py's go_home
HOME_MOVE_TIME_SEC = 3.0          # matches move_to_blocks.py's JOINT_MOVE_TIME_SEC
PHASE_SWITCH_PAUSE_SEC = 5.0      # pause between align_xy finishing and descend starting

# Confirmed 2026-09-01 via a direct `ros2 action send_goal` on this rig
# -- see module docstring's GRIPPER INTERFACE note.
GRIPPER_ACTION_NAME = "/robotiq_gripper_controller/gripper_cmd"
GRIPPER_CLOSED_POSITION = 1.0
GRIPPER_MAX_EFFORT = 20.0

ACTION_SCALE = 0.05        # confirmed for align_xy, see run_subgoal_real.py's comment
TAPER_RADIUS = 0.10        # align_xy-only overshoot taper
MAX_STEPS = 30
ALIGN_XY_Z_CLEARANCE = 0.025

SUBGOAL_LABELS = ["align_xy", "descend", "close_gripper", "lift", "move_to_target", "release"]

SUBGOAL_PARAMS = {
    "align_xy": dict(threshold=0.005),               # ALIGN_XY_THRESHOLD
    "descend": dict(threshold=0.02, dxy_limit=0.03),  # DESCEND_THRESHOLD / DESCEND_DXY_LIMIT
}


def load_policy(path):
    npz = np.load(path)
    return {k: npz[k] for k in npz.files}


def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean()
    var = x.var()
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def policy_act(obs, w):
    """Deterministic action: tanh(mean), matches SubgoalGaussianPolicy.act(deterministic=True)."""
    x = layer_norm(obs.astype(np.float64), w["input_norm.weight"], w["input_norm.bias"])
    h = np.tanh(w["trunk.0.weight"] @ x + w["trunk.0.bias"])
    h = np.tanh(w["trunk.2.weight"] @ h + w["trunk.2.bias"])
    mean = w["mean_head.weight"] @ h + w["mean_head.bias"]
    return np.tanh(mean)


def is_done(subgoal: str, d_xy: float, d_z: float) -> bool:
    p = SUBGOAL_PARAMS[subgoal]
    if subgoal == "align_xy":
        return d_xy <= p["threshold"]
    if subgoal == "descend":
        return d_z <= p["threshold"] and d_xy <= p["dxy_limit"]
    raise ValueError(f"No real-hardware done-criterion defined for subgoal={subgoal!r}")


class PickSequenceReal(Node):
    def __init__(self):
        super().__init__("pick_sequence_real")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cartesian_client = self.create_client(GetCartesianPath, "/compute_cartesian_path")
        self.arm_client = ActionClient(self, FollowJointTrajectory, ARM_ACTION_NAME)
        self.gripper_client = ActionClient(self, GripperCommand, GRIPPER_ACTION_NAME)
        self._joint_positions = None
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)

    def _on_joint_state(self, msg: JointState) -> None:
        d = dict(zip(msg.name, msg.position))
        if all(j in d for j in JOINT_NAMES):
            self._joint_positions = [d[j] for j in JOINT_NAMES]

    def wait_for_joint_state(self, timeout_sec: float = 5.0) -> list:
        self._joint_positions = None
        deadline = time.time() + timeout_sec
        while self._joint_positions is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._joint_positions is None:
            raise RuntimeError("No /joint_states received")
        return self._joint_positions

    def get_grip_pos(self) -> np.ndarray:
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                tf = self.tf_buffer.lookup_transform(BASE_FRAME, LINK_NAME, Time())
                t = tf.transform.translation
                return np.array([t.x, t.y, t.z], dtype=np.float64)
            except Exception:
                rclpy.spin_once(self, timeout_sec=0.2)
        raise RuntimeError("Could not look up base_link -> tool0 transform")

    def build_obs(self, subgoal: str, grip_pos, grip_velp, goal_xyz):
        """29-dim, frame-relative layout -- see subgoal_features.py's
        RELATIVE_OBS_DIM comment in the main repo for the source of truth."""
        onehot = np.zeros(len(SUBGOAL_LABELS), dtype=np.float32)
        onehot[SUBGOAL_LABELS.index(subgoal)] = 1.0

        object_pos = np.asarray(goal_xyz, dtype=np.float64)
        object_rel_pos = object_pos - grip_pos
        goal_rel_pos = np.zeros(3, dtype=np.float64)
        gripper_state = np.zeros(2, dtype=np.float64)
        object_rot = np.zeros(3, dtype=np.float64)
        object_velp = np.zeros(3, dtype=np.float64)
        object_velr = np.zeros(3, dtype=np.float64)
        gripper_vel = np.zeros(2, dtype=np.float64)

        obs = np.concatenate([
            object_rel_pos, gripper_state, object_rot, object_velp,
            object_velr, grip_velp, gripper_vel, goal_rel_pos,
            onehot, [0.0],
        ]).astype(np.float64)
        assert obs.shape == (29,), obs.shape
        return obs

    def plan_and_execute_delta(self, dx, dy, dz):
        current_positions = self.wait_for_joint_state()
        cur_pos = self.get_grip_pos()

        target = Pose()
        target.position.x = cur_pos[0] + dx
        target.position.y = cur_pos[1] + dy
        target.position.z = cur_pos[2] + dz
        tf = self.tf_buffer.lookup_transform(BASE_FRAME, LINK_NAME, Time())
        target.orientation = tf.transform.rotation  # hold orientation fixed

        req = GetCartesianPath.Request()
        req.header.frame_id = BASE_FRAME
        req.start_state.joint_state.name = JOINT_NAMES
        req.start_state.joint_state.position = current_positions
        req.start_state.is_diff = False
        req.group_name = GROUP_NAME
        req.link_name = LINK_NAME
        req.waypoints = [target]
        req.max_step = 0.01
        req.jump_threshold = 0.0
        req.avoid_collisions = True

        future = self.cartesian_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        print(f"    fraction={response.fraction:.3f}")
        if response.fraction < MIN_FRACTION:
            print(f"    Only {response.fraction:.1%} plannable -- not executing")
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = response.solution.joint_trajectory
        future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            print("    Trajectory goal rejected")
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return True

    def go_to_joint_positions(self, positions: list, move_time_sec: float) -> bool:
        """Direct joint-space move to an absolute target -- same mechanism
        as move_to_blocks.py's go_to_joint_positions / go_home, NOT the
        Cartesian planner, so this can't fail to a singularity/reachability
        issue the way plan_and_execute_delta can."""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(move_time_sec)
        point.time_from_start.nanosec = int((move_time_sec % 1) * 1e9)
        goal.trajectory.points = [point]

        future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            print("Home trajectory goal rejected")
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return True

    def go_home(self, path: str = HOME_PATH) -> bool:
        with open(path) as f:
            saved = json.load(f)
        positions = [saved[j] for j in JOINT_NAMES]
        print(f"Returning home ({path})...")
        return self.go_to_joint_positions(positions, HOME_MOVE_TIME_SEC)

    def close_gripper(self) -> bool:
        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            print(f"  Gripper action server {GRIPPER_ACTION_NAME} not available -- "
                  f"see this script's GRIPPER INTERFACE note. NOT closing gripper.")
            return False
        goal = GripperCommand.Goal()
        goal.command.position = GRIPPER_CLOSED_POSITION
        goal.command.max_effort = GRIPPER_MAX_EFFORT
        future = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            print("  Gripper goal rejected")
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        print("  Gripper close complete.")
        return True


def run_phase(node: PickSequenceReal, subgoal: str, weights: dict, goal_xyz: np.ndarray) -> bool:
    """run_subgoal_real.py's rollout loop, generalized over subgoal and
    kept exactly as validated -- same per-step confirm prompt, same
    align_xy taper/z-floor safety. Returns True iff the subgoal's own done
    criterion was actually reached (not MAX_STEPS, a failed plan, or 'q'),
    so callers can gate the next phase on it."""
    print(f"\n=== {subgoal} ===")
    prev_grip_pos = node.get_grip_pos()
    prev_t = time.time()

    for step in range(MAX_STEPS):
        grip_pos = node.get_grip_pos()
        now = time.time()
        dt = max(now - prev_t, 1e-3)
        grip_velp = (grip_pos - prev_grip_pos) / dt
        prev_grip_pos, prev_t = grip_pos, now

        d_xy = float(np.linalg.norm(goal_xyz[:2] - grip_pos[:2]))
        d_z = float(grip_pos[2] - goal_xyz[2])
        print(f"\n[{subgoal}] Step {step}: grip_pos={grip_pos.round(4)} d_xy={d_xy:.4f} d_z={d_z:.4f}")
        if is_done(subgoal, d_xy, d_z):
            print(f"Reached {subgoal} criterion -- done.")
            return True

        obs = node.build_obs(subgoal, grip_pos, grip_velp, goal_xyz)
        action = policy_act(obs, weights)

        if subgoal == "align_xy":
            taper = min(1.0, d_xy / TAPER_RADIUS)
            dx, dy = (action[:2] * ACTION_SCALE * taper).tolist()
            dz = float(action[2] * ACTION_SCALE)

            z_floor = goal_xyz[2] + ALIGN_XY_Z_CLEARANCE
            if grip_pos[2] <= z_floor:
                print(f"  SAFETY: grip z={grip_pos[2]:.4f} at/below align_xy floor "
                      f"{z_floor:.4f} -- refusing to descend further. Aborting phase.")
                return False
            if grip_pos[2] + dz < z_floor:
                clamped = z_floor - grip_pos[2]
                print(f"  [SAFETY] z delta clamped: {dz:.4f} -> {clamped:.4f} (floor={z_floor:.4f})")
                dz = clamped

            print(f"  action={action.round(3)}  taper={taper:.3f} -> delta=({dx:.4f}, {dy:.4f}, {dz:.4f})")
        else:
            dx, dy, dz = (action[:3] * ACTION_SCALE).tolist()
            print(f"  action={action.round(3)} -> delta=({dx:.4f}, {dy:.4f}, {dz:.4f})")

        resp = input("  Execute this step? [Enter=yes, q=abort]: ").strip()
        if resp == "q":
            print(f"{subgoal}: aborted by user.")
            return False

        ok = node.plan_and_execute_delta(dx, dy, dz)
        if not ok:
            print(f"{subgoal}: step failed to plan/execute -- aborting phase.")
            return False

    print(f"\n{subgoal}: hit MAX_STEPS={MAX_STEPS} without reaching done criterion.")
    return False


def main():
    with open(GOAL_PATH) as f:
        goal = json.load(f)["cartesian"]
    goal_xyz = np.array([goal["x"], goal["y"], goal["z"]], dtype=np.float64)
    print(f"Loaded goal from {GOAL_PATH}: {goal_xyz}")

    align_xy_weights = load_policy("align_xy_policy_weights.npz")
    descend_weights = load_policy("descend_policy_weights.npz")

    rclpy.init()
    node = PickSequenceReal()
    node.cartesian_client.wait_for_service(timeout_sec=5.0)
    node.arm_client.wait_for_server(timeout_sec=5.0)

    try:
        if run_phase(node, "align_xy", align_xy_weights, goal_xyz):
            print(f"\nPausing {PHASE_SWITCH_PAUSE_SEC:.0f}s before descend...")
            time.sleep(PHASE_SWITCH_PAUSE_SEC)
            if run_phase(node, "descend", descend_weights, goal_xyz):
                print("\n=== close_gripper ===")
                node.close_gripper()
            else:
                print("\ndescend did not complete -- skipping gripper close.")
        else:
            print("\nalign_xy did not complete -- skipping descend and gripper close.")
    except KeyboardInterrupt:
        print("\nInterrupted by user -- stopping and returning home.")
    finally:
        print("\n=== returning home ===")
        node.go_home()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
