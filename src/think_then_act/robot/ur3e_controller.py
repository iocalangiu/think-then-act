"""
think_then_act.robot.ur3e_controller

Real-time Cartesian-displacement controller for a physical UR3e over RTDE.
Consumes the (dx, dy, dz) actions produced by the MuJoCo-trained subgoal
policies (see policy/subgoal_policy.py) and streams them to the robot via
servoL, holding end-effector orientation fixed.

Deliberately NOT a custom Jacobian/analytic IK solver: the UR controller's
own firmware computes joint targets from a streamed TCP pose (servoL), which
is a strictly better fit here than reimplementing IK ourselves — dx,dy,dz
are position-only deltas with no target orientation, so an analytic 6-DOF
solve would need a synthesized orientation and branch selection it has no
information to make, and a from-scratch Jacobian solve would just be
duplicating logic already running on the robot's real-time controller.

Needs `pip install ur_rtde` (rtde_control / rtde_receive) and RTDE enabled
on the controller.
"""

from __future__ import annotations

import numpy as np


class UR3eCartesianController:
    """
    Turns per-step (dx, dy, dz) Cartesian displacements into streamed servoL
    commands, with orientation locked to whatever pose the TCP had at
    connect time (or an explicit override). Callers are responsible for
    scaling policy output into meters before calling step() — this class
    takes raw metric deltas, not normalized actions.

    Not thread-safe; drive step() from a single control loop, ideally at the
    same rate as `dt` — servoL is designed to be re-issued every control
    cycle, not fired once and left to run.
    """

    def __init__(
        self,
        robot_ip: str,
        fixed_orientation: np.ndarray | None = None,
        workspace_bounds: tuple[np.ndarray, np.ndarray] | None = None,
        velocity: float = 0.5,
        acceleration: float = 0.5,
        dt: float = 1.0 / 20.0,
        lookahead_time: float = 0.1,
        gain: int = 300,
    ):
        import rtde_control
        import rtde_receive

        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

        self.velocity = velocity
        self.acceleration = acceleration
        self.dt = dt
        self.lookahead_time = lookahead_time
        self.gain = gain
        # (low_xyz, high_xyz) clipped against every target before sending —
        # real hardware, so worth a floor even though the sim side has no
        # equivalent boundary check. None (default) disables clipping.
        self.workspace_bounds = workspace_bounds

        current_pose = self.rtde_r.getActualTCPPose()
        self.orientation = (
            np.array(fixed_orientation, dtype=np.float64)
            if fixed_orientation is not None
            else np.array(current_pose[3:6], dtype=np.float64)
        )

    def get_position(self) -> np.ndarray:
        """Current TCP xyz, meters, base frame."""
        return np.array(self.rtde_r.getActualTCPPose()[:3], dtype=np.float64)

    def step(self, dx: float, dy: float, dz: float) -> np.ndarray:
        """
        Advance the TCP by (dx, dy, dz) meters from its CURRENT actual
        position (read fresh via getActualTCPPose each call, not the last
        commanded target) so tracking error doesn't compound across steps.
        Orientation is always self.orientation, never touched by the delta.

        Returns the target pose actually sent (xyz + rotvec), for logging.
        """
        current_xyz = self.get_position()
        target_xyz = current_xyz + np.array([dx, dy, dz], dtype=np.float64)

        if self.workspace_bounds is not None:
            low, high = self.workspace_bounds
            target_xyz = np.clip(target_xyz, low, high)

        target_pose = np.concatenate([target_xyz, self.orientation])

        self.rtde_c.servoL(
            target_pose.tolist(),
            self.velocity, self.acceleration, self.dt,
            self.lookahead_time, self.gain,
        )
        return target_pose

    def stop(self) -> None:
        self.rtde_c.servoStop()

    def disconnect(self) -> None:
        self.rtde_c.servoStop()
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()

    def __enter__(self) -> "UR3eCartesianController":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()
