"""
think_then_act.training.safe_regime

Pure functions for the "safe to hand off to the scripted gripper close"
condition used by the chained align_xy->descend BC pipeline (see
training/chained_teacher_rollout.py, memory: meta_rl_sim2real_direction).

On the real UR3e, closing the gripper is NOT a trained skill — it's one
scripted ROS2 action call to the Robotiq controller
(`/robotiq_gripper_controller/gripper_cmd`), which already force-limits and
generalizes across block widths mechanically. So the recurrent align_xy+
descend policy's job isn't "grasp the block," it's "get the gripper close
enough, without triggering a force spike, that firing that scripted close
is safe" — this module defines exactly that condition.

Deliberately separate from reward/subgoal_reward.py (not touched — see
memory: this project's MLP training pipeline stays as-is) — this is a NEW
condition for a NEW use (chained-episode handoff point), not a replacement
for descend's own reward/done logic, which keeps using its own threshold
constants unchanged.

d_xy_limit/d_z_limit default to reward/subgoal_reward.py's existing
descend_dxy_limit (0.01m) / descend_threshold (0.02m) — same physical
meaning (lateral / vertical gripper-to-block distance), reused rather than
reinvented so "safe regime" and "descend done" agree on what "close enough"
means.
"""

from __future__ import annotations


def is_safe_regime(
    d_xy: float, d_z: float, force_onset_flag: float,
    d_xy_limit: float = 0.01, d_z_limit: float = 0.02,
) -> bool:
    """
    True if the gripper is close enough to the block (both lateral and
    vertical distance under their limits) AND no contact-force spike has
    been detected yet this episode (force_onset_flag == 0 — see
    training/singularity_force_env.py's onset-flag semantics: it latches to
    1.0 once fired and stays 1.0 via the decayed trace, so checking the RAW
    flag, not the decayed trace, is what actually means "no spike happened
    AT this step" vs. "a spike happened recently").
    """
    return bool(d_xy <= d_xy_limit and abs(d_z) <= d_z_limit and force_onset_flag == 0.0)


class SafeRegimeGate:
    """
    Streak-gated version of is_safe_regime — same "must hold for N
    consecutive steps, not just one instant" rationale as
    SubgoalConditionedEnv's close_gripper_done_streak/align_xy_done_streak
    (a single-step MuJoCo settling transient or measurement blip shouldn't
    be mistaken for a genuinely stable safe position). Stateful: call
    update() once per step; reset() at the start of each episode/phase.
    """

    def __init__(self, streak_required: int = 3, d_xy_limit: float = 0.01, d_z_limit: float = 0.02):
        self.streak_required = streak_required
        self.d_xy_limit = d_xy_limit
        self.d_z_limit = d_z_limit
        self._streak = 0

    @property
    def streak(self) -> int:
        return self._streak

    def reset(self) -> None:
        self._streak = 0

    def update(self, d_xy: float, d_z: float, force_onset_flag: float) -> bool:
        """Returns True once the safe-regime condition has held for
        streak_required CONSECUTIVE calls (including this one)."""
        raw = is_safe_regime(d_xy, d_z, force_onset_flag, self.d_xy_limit, self.d_z_limit)
        self._streak = self._streak + 1 if raw else 0
        return self._streak >= self.streak_required
