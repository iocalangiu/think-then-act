#!/usr/bin/env python3
"""
run_align_xy_real.py  (RUNS ON THE REMOTE CONSTRUCT ROS2 SESSION, not here)

Rolls out the trained align_xy policy on the real UR3e, bypassing perception
entirely: the "block position" the policy needs is not detected by a camera
(no camera topic exists on this rig, see memory ur3e_sim2real.md) -- instead
it's read from goal_pose.json, written by the interactive move-to-blocks
script's 's' (save current pose) command after a human manually jogs the
gripper next to the real brick.

Pure numpy, no torch dependency -- see export_align_xy_policy_numpy.py
(run locally, where torch + the checkpoint exist) for how
align_xy_policy_weights.npz was produced. Copy both that file and this
script to the same directory on the Construct session before running.

UPDATED 2026-09-01 for the retrained, frame-relative align_xy checkpoint
(29-dim, not the original 38) -- see subgoal_features.py's
ALIGN_XY_OBS_DIM comment in the main repo for the source of truth. This is
exactly the fix for the coordinate-frame/scale mismatch this script
originally hit on its first real rollout (Fetch world-frame positions
~0.7-2.0m vs. the UR3e's base_link-frame positions ~0.1-0.5m) -- dropping
absolute grip_pos/object_pos and using only relative geometry means the
real robot's absolute coordinate scale no longer matters at all:
    obs29 = object_rel_pos(3) + gripper_state(2) + object_rot(3)
          + object_velp(3) + object_velr(3) + grip_velp(3) + gripper_vel(2)
          + goal_rel_pos(3) + subgoal_onehot(6) + collision_prob(1)

What's REAL vs APPROXIMATED here, since there's no perception or gripper
instrumentation wired into this script yet:
  - object_rel_pos: REAL -- goal_xyz (the human-recorded perception bypass)
                    minus grip_pos (REAL, TF base_link -> tool0). This IS
                    the one geometric quantity align_xy's reward actually
                    uses (reward_align_xy only ever reads achieved_goal,
                    i.e. block_pos, relative to grip_pos), and it's now
                    the ONLY position-derived feature that carries any
                    real information -- everything else below is either a
                    placeholder or algebraically forced to zero.
  - grip_velp     : REAL, finite-differenced from two consecutive TF reads
  - goal_rel_pos  : desired_goal - achieved_goal -- both are set to the
                    SAME recorded goal_xyz (no reason to invent a
                    different desired_goal), so this is always [0,0,0].
                    Kept as an explicit zero vector, not omitted, to match
                    the trained network's exact 29-dim input ordering.
  - gripper_state, gripper_vel, object_rot, object_velp, object_velr:
                    PLACEHOLDER ZEROS -- not measured. object_rot/velp/velr
                    being zero is arguably correct for a static real block
                    (zero true velocity), but gripper_state/vel are a real
                    approximation since no Robotiq joint-state subscriber
                    is wired in here. align_xy's own reward function never
                    reads these fields, but the POLICY saw real nonzero
                    values for them during sim training, so this is a
                    genuine (probably small) distribution-shift source --
                    worth remembering as a candidate cause if behavior
                    looks visibly wrong.
  - collision_prob: hardcoded 0.0 (no collision_predictor deployed here)

Motion execution reuses the SAME /compute_cartesian_path +
FollowJointTrajectory pattern as move_to_blocks.py's plan_and_execute
(proven working on this rig) -- not moveit_servo velocity streaming, so a
single step is a single planned Cartesian move, and a low planning
`fraction` (e.g. near a singularity) is caught and stops the rollout rather
than executing a partial/garbage trajectory.

Requires scaled_joint_trajectory_controller active (see move_to_blocks.py's
docstring -- if you switched to forward_position_controller for
moveit_servo testing, switch back first).
"""
import json
import math
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
from control_msgs.action import FollowJointTrajectory

GROUP_NAME = "ur_manipulator"
LINK_NAME = "tool0"
BASE_FRAME = "base_link"
ARM_ACTION_NAME = "/scaled_joint_trajectory_controller/follow_joint_trajectory"
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
MIN_FRACTION = 0.9

WEIGHTS_PATH = "align_xy_policy_weights.npz"
GOAL_PATH = "goal_pose.json"

# Fetch's fixed sim per-step Cartesian displacement (gymnasium_robotics
# hardcodes ~0.05m/step). CONFIRMED empirically 2026-09-01: commanded vs.
# TF-measured actual movement matched to the mm across multiple steps of a
# real rollout -- this scale is correct for this /compute_cartesian_path
# execution path, no longer an open calibration question.
ACTION_SCALE = 0.05
ALIGN_XY_THRESHOLD = 0.02  # matches SubgoalWeights.align_xy_threshold in sim
# Added 2026-09-01: a FIXED 0.05m step overshoots once d_xy gets down near
# that same magnitude -- confirmed on real hardware as a clean limit-cycle
# oscillation (dx/dy/dz flipping sign ~every step, near-identical
# magnitude, d_xy stuck bouncing in a band instead of continuing to shrink
# toward ALIGN_XY_THRESHOLD). Taper the commanded xy step size down as
# d_xy shrinks below this radius so steps naturally get smaller approaching
# the target, instead of staying at a constant 5cm and bouncing past it.
# Starting value (2x the align_xy_threshold's own natural step-count budget
# would need to shrink over), not empirically tuned yet -- watch whether
# d_xy actually keeps shrinking below ~0.05 next run before trusting this
# number, same "starting value, check telemetry" spirit as the sim-side
# weights in subgoal_reward.py's SubgoalWeights.
TAPER_RADIUS = 0.10
MAX_STEPS = 30

SUBGOAL_LABELS = ["align_xy", "descend", "close_gripper", "lift", "move_to_target", "release"]
ALIGN_XY_ONEHOT = np.zeros(len(SUBGOAL_LABELS), dtype=np.float32)
ALIGN_XY_ONEHOT[SUBGOAL_LABELS.index("align_xy")] = 1.0


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


class AlignXYReal(Node):
    def __init__(self):
        super().__init__("align_xy_real")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cartesian_client = self.create_client(GetCartesianPath, "/compute_cartesian_path")
        self.arm_client = ActionClient(self, FollowJointTrajectory, ARM_ACTION_NAME)
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

    def build_obs(self, grip_pos, grip_velp, goal_xyz):
        """29-dim, frame-relative layout -- see module docstring. No
        absolute grip_pos/object_pos anywhere in the returned vector, by
        design: that's the actual fix for the coordinate-frame mismatch."""
        object_pos = np.asarray(goal_xyz, dtype=np.float64)
        object_rel_pos = object_pos - grip_pos          # the one real geometric signal
        goal_rel_pos = np.zeros(3, dtype=np.float64)     # desired_goal - achieved_goal, both == goal_xyz
        gripper_state = np.zeros(2, dtype=np.float64)    # placeholder, see module docstring
        object_rot = np.zeros(3, dtype=np.float64)
        object_velp = np.zeros(3, dtype=np.float64)
        object_velr = np.zeros(3, dtype=np.float64)
        gripper_vel = np.zeros(2, dtype=np.float64)

        obs = np.concatenate([
            object_rel_pos, gripper_state, object_rot, object_velp,
            object_velr, grip_velp, gripper_vel, goal_rel_pos,
            ALIGN_XY_ONEHOT, [0.0],
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


def main():
    with open(GOAL_PATH) as f:
        goal = json.load(f)["cartesian"]
    goal_xyz = np.array([goal["x"], goal["y"], goal["z"]], dtype=np.float64)
    print(f"Loaded goal from {GOAL_PATH}: {goal_xyz}")

    weights = load_policy(WEIGHTS_PATH)

    rclpy.init()
    node = AlignXYReal()
    node.cartesian_client.wait_for_service(timeout_sec=5.0)
    node.arm_client.wait_for_server(timeout_sec=5.0)

    prev_grip_pos = node.get_grip_pos()
    prev_t = time.time()

    for step in range(MAX_STEPS):
        grip_pos = node.get_grip_pos()
        now = time.time()
        dt = max(now - prev_t, 1e-3)
        grip_velp = (grip_pos - prev_grip_pos) / dt
        prev_grip_pos, prev_t = grip_pos, now

        d_xy = float(np.linalg.norm(goal_xyz[:2] - grip_pos[:2]))
        print(f"\nStep {step}: grip_pos={grip_pos.round(4)} d_xy={d_xy:.4f}")
        if d_xy <= ALIGN_XY_THRESHOLD:
            print("Reached align_xy threshold -- done.")
            break

        obs = node.build_obs(grip_pos, grip_velp, goal_xyz)
        action = policy_act(obs, weights)
        # Taper: full 0.05m step while far away, shrinking proportionally
        # once inside TAPER_RADIUS -- fixes the overshoot oscillation (see
        # TAPER_RADIUS's comment above). Only applied to xy, not z (z isn't
        # gated by d_xy in any way, so tapering it against that distance
        # has no clear rationale).
        taper = min(1.0, d_xy / TAPER_RADIUS)
        dx, dy = (action[:2] * ACTION_SCALE * taper).tolist()
        # dz left as the policy's raw output (per user request 2026-09-01)
        # despite the earlier observation that z climbed steadily on real
        # hardware -- NOT clamped/tapered here. If it's still a problem,
        # the fix belongs in training (stronger align_xy_z_penalty) rather
        # than a deploy-time override.
        dz = float(action[2] * ACTION_SCALE)
        print(f"  action={action.round(3)}  taper={taper:.3f} -> delta=({dx:.4f}, {dy:.4f}, {dz:.4f})")

        resp = input("  Execute this step? [Enter=yes, q=abort]: ").strip()
        if resp == "q":
            print("Aborted by user.")
            break

        ok = node.plan_and_execute_delta(dx, dy, dz)
        if not ok:
            print("Step failed to plan/execute -- stopping rollout.")
            break
    else:
        print(f"\nHit MAX_STEPS={MAX_STEPS} without reaching threshold.")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
