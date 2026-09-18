"""
think_then_act.robot.wrench_force_bridge

Bridges the real UR3e's ROS2 wrench topic (/force_torque_sensor_broadcaster/
wrench) into a background-threaded, pollable force summary — the real-robot
counterpart of training/singularity_force_env.py's force-onset observation
channel (see memory: ur3e_sim2real, meta_rl_sim2real_direction).

Composes, never edits, robot/ur3e_controller.py's UR3eCartesianController:
that class's step() is a synchronous, blocking RTDE call inside a plain
`while: controller.step(...)` loop, with no rclpy/asyncio anywhere in it —
existing sim2real replay scripts depend on that exact behavior unchanged.
So the wrench subscription runs on its OWN background thread with its own
executor, and this class only ever HOLDS a controller instance as a plain
attribute; it never subclasses or mutates it.

Deployment note (unresolved, out of scope for this bridge itself):
hardware/move_arm/record_grasp_session.py's own docstring says ROS2 scripts
in that directory run on the remote Construct session, not locally, while
UR3eCartesianController's only dependency is rtde_control/rtde_receive
(no rclpy) — suggesting it may currently run locally, connecting straight
to the robot controller's IP. If that's the case, this bridge (which needs
BOTH rclpy and a UR3eCartesianController in the same process) may need to
run wherever the ROS2 graph is actually reachable, not necessarily wherever
RTDE alone would work. Confirm this on your actual network topology before
wiring the bridge into a real rollout script — not something this file can
determine on its own.

Gate: do not trust this bridge's output until
hardware/move_arm/wrench_topic_sanity_check.py reports PASS with a steady,
nonzero message rate on the wrench topic on your actual rig.
"""

from __future__ import annotations
import threading
import time

WRENCH_TOPIC = "/force_torque_sensor_broadcaster/wrench"  # same constant as
                                                            # record_grasp_session.py /
                                                            # wrench_topic_sanity_check.py.
                                                            # Re-declared here (not
                                                            # imported from either) since
                                                            # both live under hardware/
                                                            # move_arm/, outside this
                                                            # installed package.


class _WrenchListenerNode:
    """rclpy.node.Node subclass, built lazily inside start() so importing
    this module never requires rclpy/ROS2 to be installed (mirrors how
    robot/ur3e_controller.py imports rtde_control/rtde_receive lazily
    inside __init__, not at module scope)."""

    def __new__(cls, wrench_topic: str, on_wrench):
        import rclpy.node
        from geometry_msgs.msg import WrenchStamped

        class _Impl(rclpy.node.Node):
            def __init__(self):
                super().__init__("wrench_force_bridge_listener")
                self._on_wrench_cb = on_wrench
                self.create_subscription(WrenchStamped, wrench_topic, self._handle, 50)

            def _handle(self, msg) -> None:
                f, t_ = msg.wrench.force, msg.wrench.torque
                self._on_wrench_cb(time.time(), f.x, f.y, f.z, t_.x, t_.y, t_.z)

        return _Impl()


class WrenchForceBridge:
    """
    Wraps (composes) a UR3eCartesianController with a background-threaded
    ROS2 wrench subscription. Held as self.controller, a plain attribute —
    never a base class — so UR3eCartesianController's own step()/
    get_position()/stop() are untouched and fully usable through
    bridge.controller directly.

    Usage:
        controller = UR3eCartesianController(robot_ip)
        with WrenchForceBridge(controller) as bridge:
            while True:
                action = policy.act(obs, hidden_state)
                controller.step(*action[:3] * ACTION_SCALE)
                force_summary = bridge.get_force_summary_since_last_call()
                obs, hidden_state = build_real_obs(..., force_summary)
    """

    def __init__(
        self,
        controller,
        wrench_topic: str = WRENCH_TOPIC,
        onset_force_threshold_n: float = 2.0,
        ema_alpha: float = 0.3,
    ) -> None:
        self.controller = controller
        self._wrench_topic = wrench_topic
        self._onset_threshold = onset_force_threshold_n
        self._ema_alpha = ema_alpha

        self._lock = threading.Lock()
        self._latest_wrench = None       # (t, fx, fy, fz, tx, ty, tz)
        self._max_force_since_read = 0.0
        self._onset_ts_since_read = None
        self._force_ema = 0.0

        self._rclpy_owns_init = False
        self._node = None
        self._executor = None
        self._thread = None

    def start(self) -> None:
        import rclpy
        import rclpy.executors

        if not rclpy.ok():
            rclpy.init()
            self._rclpy_owns_init = True

        self._node = _WrenchListenerNode(self._wrench_topic, self._on_wrench)
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def _on_wrench(self, t, fx, fy, fz, tx, ty, tz) -> None:
        mag = (fx * fx + fy * fy + fz * fz) ** 0.5
        with self._lock:
            self._latest_wrench = (t, fx, fy, fz, tx, ty, tz)
            self._force_ema = self._ema_alpha * mag + (1 - self._ema_alpha) * self._force_ema
            self._max_force_since_read = max(self._max_force_since_read, mag)
            if mag >= self._onset_threshold and self._onset_ts_since_read is None:
                self._onset_ts_since_read = t

    def get_force_summary_since_last_call(self) -> dict:
        """
        Latched summary, decoupled from and pollable at any rate relative to
        whatever cadence the caller's own control loop runs at (e.g.
        UR3eCartesianController's 20Hz servoL loop). Resets max_force_n/
        onset_ts on read (they describe "since the last read"); ema_force_n
        is a continuous running estimate and is NOT reset.
        """
        with self._lock:
            summary = {
                "max_force_n": self._max_force_since_read,
                "onset_ts": self._onset_ts_since_read,
                "latest_wrench": self._latest_wrench,
                "ema_force_n": self._force_ema,
            }
            self._max_force_since_read = 0.0
            self._onset_ts_since_read = None
            return summary

    def stop(self) -> None:
        import rclpy

        if self._executor is not None:
            self._executor.shutdown()
        if self._node is not None:
            self._node.destroy_node()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._rclpy_owns_init:
            rclpy.shutdown()

    def __enter__(self) -> "WrenchForceBridge":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
