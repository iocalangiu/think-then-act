"""
Unit tests for reward.subgoal_reward — pure numpy, synthetic states, no mujoco.
"""

import numpy as np
import pytest

from think_then_act.reward.subgoal_reward import (
    SUBGOAL_LABELS,
    SubgoalWeights,
    compute_subgoal_reward,
    reward_align_xy,
    reward_close_gripper,
    reward_descend,
    reward_lift,
    reward_move_to_target,
    reward_release,
)

W = SubgoalWeights()


def _make_obs(grip_pos, finger_widths=(0.05, 0.05)):
    """25-float obs vector with only grip_pos (0:3) and gripper_state (9:11) set."""
    obs = np.zeros(25)
    obs[0:3] = grip_pos
    obs[9:11] = finger_widths
    return obs


# ---------------------------------------------------------------------------
# align_xy
# ---------------------------------------------------------------------------
def test_align_xy_improves_as_gripper_approaches_block_xy():
    # close_obs is within align_xy_threshold=0.02 (d_xy ~0.0028).
    block = [1.3, 0.75, 0.5]
    far_obs   = _make_obs([1.0, 0.75, 0.6])
    close_obs = _make_obs([1.2975, 0.7505, 0.6])

    r_far,   b_far   = reward_align_xy(far_obs,   block, [0, 0, 0])
    r_close, b_close  = reward_align_xy(close_obs, block, [0, 0, 0])

    assert r_close > r_far
    assert b_far["done"] is False
    assert b_close["done"] is True


def test_align_xy_penalizes_z_drift_at_the_same_xy_distance():
    # Added 2026-09-01 after the real-UR3e rollout showed the arm
    # descending during what should be an XY-only move — confirmed even in
    # sim (a recorded align_xy trajectory's z rose ~0.13m, unconstrained by
    # reward). Same d_xy, different d_z: the drifted-in-z state must score
    # strictly worse now.
    block = [1.3, 0.75, 0.5]
    level_obs    = _make_obs([1.29, 0.751, 0.5])    # d_z = 0.0
    drifted_obs  = _make_obs([1.29, 0.751, 0.7])    # same d_xy, d_z = 0.2

    r_level,   b_level   = reward_align_xy(level_obs,   block, [0, 0, 0])
    r_drifted, b_drifted = reward_align_xy(drifted_obs, block, [0, 0, 0])

    assert b_level["d_xy"] == pytest.approx(b_drifted["d_xy"], abs=1e-6)
    assert r_level > r_drifted
    assert b_level["d_z"] == pytest.approx(0.0, abs=1e-6)
    assert b_drifted["d_z"] == pytest.approx(0.2, abs=1e-6)   # d_z = grip_pos.z - block_pos.z


def test_align_xy_stillness_penalty_discourages_large_actions_near_target():
    # Added 2026-09-01 alongside align_xy_done_streak (subgoal_env.py) after
    # a real-UR3e rollout of the (at the time) stillness-free checkpoint
    # oscillated in a stable limit cycle near the target instead of
    # settling. Same state (same d_xy), only the action differs: a large
    # xy-translation action must score strictly worse than a near-zero one.
    block = [1.3, 0.75, 0.5]
    obs = _make_obs([1.29, 0.751, 0.5])

    r_still, b_still = reward_align_xy(obs, block, [0, 0, 0], action=[0.0, 0.0, 0.0, 0.0])
    r_moving, b_moving = reward_align_xy(obs, block, [0, 0, 0], action=[1.0, 1.0, 0.0, 0.0])

    assert b_still["d_xy"] == pytest.approx(b_moving["d_xy"], abs=1e-6)
    assert r_still > r_moving


def test_align_xy_stillness_penalty_ignores_z_action_component():
    # Only action[:2] (xy) should count -- align_xy's own scope, matching
    # align_xy_z_penalty's same xy-only framing. A large z-only action must
    # score the SAME as no action at all.
    block = [1.3, 0.75, 0.5]
    obs = _make_obs([1.29, 0.751, 0.5])

    r_still, _ = reward_align_xy(obs, block, [0, 0, 0], action=[0.0, 0.0, 0.0, 0.0])
    r_z_only, _ = reward_align_xy(obs, block, [0, 0, 0], action=[0.0, 0.0, 1.0, 0.0])

    assert r_still == pytest.approx(r_z_only, abs=1e-9)


def test_align_xy_no_action_matches_zero_action():
    # action=None (e.g. at env.reset(), nothing taken yet) must be a true
    # no-op, same convention as reward_close_gripper's.
    block = [1.3, 0.75, 0.5]
    obs = _make_obs([1.29, 0.751, 0.5])

    r_none, _ = reward_align_xy(obs, block, [0, 0, 0], action=None)
    r_zero, _ = reward_align_xy(obs, block, [0, 0, 0], action=[0.0, 0.0, 0.0, 0.0])

    assert r_none == pytest.approx(r_zero, abs=1e-9)


# ---------------------------------------------------------------------------
# descend
# ---------------------------------------------------------------------------
def test_descend_rewards_closing_vertical_gap():
    block = [1.3, 0.75, 0.425]
    high_obs = _make_obs([1.3, 0.75, 0.55])
    low_obs  = _make_obs([1.3, 0.75, 0.43])

    r_high, b_high = reward_descend(high_obs, block, [0, 0, 0])
    r_low,  b_low  = reward_descend(low_obs,  block, [0, 0, 0])

    assert r_low > r_high
    assert b_high["done"] is False
    assert b_low["done"] is True


def test_descend_penalizes_collision_probability():
    block = [1.3, 0.75, 0.425]
    obs = _make_obs([1.3, 0.75, 0.55])

    r_safe, _ = reward_descend(obs, block, [0, 0, 0], collision_prob=0.0)
    r_risky, _ = reward_descend(obs, block, [0, 0, 0], collision_prob=0.9)

    assert r_risky < r_safe


def test_descend_done_requires_xy_alignment_not_just_height():
    # Regression guard for the 2026-07-20 bug: a live demo rollout showed a
    # trained policy reaching grasp height while having drifted ~0.5m
    # laterally, and the OLD done condition (d_z alone) still fired. This
    # reproduces that exact shape: height is satisfied but xy has drifted
    # well past descend_dxy_limit.
    block = [1.3, 0.75, 0.425]
    drifted_obs = _make_obs([1.3 + 0.5, 0.75, 0.43])  # d_z fine, d_xy=0.5m

    _, breakdown = reward_descend(drifted_obs, block, [0, 0, 0])

    assert breakdown["d_z"] <= W.descend_threshold
    assert breakdown["done"] is False


def test_descend_penalizes_xy_drift():
    block = [1.3, 0.75, 0.55]
    aligned_obs = _make_obs([1.3, 0.75, 0.6])
    drifted_obs = _make_obs([1.3 + 0.2, 0.75, 0.6])  # same d_z, worse d_xy

    r_aligned, _ = reward_descend(aligned_obs, block, [0, 0, 0])
    r_drifted, _ = reward_descend(drifted_obs, block, [0, 0, 0])

    assert r_aligned > r_drifted


# ---------------------------------------------------------------------------
# close_gripper
# ---------------------------------------------------------------------------
# Real steady-state grasp forces measured via scripts/measure_grip_contact_
# force.py (2026-08-09, 5/5 seeds reached CARRY): min(left,right) ranged
# 184.5-274.9N, mean 231.8N. Used directly below (not a round arbitrary
# number) so these tests double as a regression guard against the actual
# calibration data behind close_gripper_force_scale, not a hypothetical.
MEASURED_GRIP_STRENGTH = 231.8
WEAKEST_MEASURED_GRIP_STRENGTH = 184.5


def test_close_gripper_rewards_real_contact_over_no_contact():
    # closedness is now tanh(min(left,right)/close_gripper_force_scale) — a
    # real grasp (measured contact force) should score higher AND register
    # done, while no contact at all should do neither, regardless of
    # finger-JOINT width (that's exactly the proxy this replaced — see
    # close_gripper_force_scale's comment in SubgoalWeights).
    block = [1.3, 0.75, 0.425]
    no_contact_obs = _make_obs([1.3, 0.75, 0.425])
    grasp_obs      = _make_obs([1.3, 0.75, 0.425])

    r_open,  b_open  = reward_close_gripper(no_contact_obs, block, [0, 0, 0],
                                             grip_force={"left": 0.0, "right": 0.0})
    r_grasp, b_grasp = reward_close_gripper(grasp_obs, block, [0, 0, 0],
                                             grip_force={"left": MEASURED_GRIP_STRENGTH,
                                                         "right": MEASURED_GRIP_STRENGTH})

    assert r_grasp > r_open
    assert b_open["done"] is False
    assert b_grasp["done"] is True


def test_close_gripper_one_sided_contact_is_not_a_grasp():
    # Bottlenecked by min(left, right), not summed — one finger pressing
    # hard while the other floats free (e.g. the block got knocked to one
    # side) must not register as a real pinch grasp. This is the force-
    # based analogue of the old "closing on nothing scores worse than a
    # real grasp" regression guard: a symmetric no-object closure can't
    # happen at all anymore (no object, no force on either side), so the
    # failure mode worth guarding is lopsided contact instead.
    block = [1.3, 0.75, 0.425]
    lopsided_obs = _make_obs([1.3, 0.75, 0.425])
    grasp_obs    = _make_obs([1.3, 0.75, 0.425])

    r_lopsided, b_lopsided = reward_close_gripper(
        lopsided_obs, block, [0, 0, 0],
        grip_force={"left": MEASURED_GRIP_STRENGTH, "right": 0.0},
    )
    r_grasp, _ = reward_close_gripper(
        grasp_obs, block, [0, 0, 0],
        grip_force={"left": MEASURED_GRIP_STRENGTH, "right": MEASURED_GRIP_STRENGTH},
    )

    assert r_lopsided < r_grasp
    assert b_lopsided["done"] is False


def test_close_gripper_no_grip_force_defaults_to_no_contact():
    # grip_force=None (e.g. computed at env.reset() before any contact
    # exists, or a caller that hasn't wired grip_contact_forces through)
    # must default to zero force, not error — same None-means-unchanged
    # convention as action.
    block = [1.3, 0.75, 0.425]
    obs = _make_obs([1.3, 0.75, 0.425])

    reward, breakdown = reward_close_gripper(obs, block, [0, 0, 0])

    assert breakdown["grip_strength"] == 0.0
    assert breakdown["closedness"] == 0.0
    assert breakdown["done"] is False


def test_close_gripper_penalizes_ending_up_far_from_block():
    # Distance penalty is independent of grip force — no contact either way
    # here, but ending up far from the block must still score worse (see
    # close_gripper_distance_weight's comment for why this term exists
    # alongside the action-only stillness penalty).
    block = [1.3, 0.75, 0.425]
    closed_near_obs = _make_obs([1.3, 0.75, 0.425])
    closed_far_obs  = _make_obs([1.5, 0.75, 0.425])

    r_near, b_near = reward_close_gripper(closed_near_obs, block, [0, 0, 0])
    r_far,  b_far  = reward_close_gripper(closed_far_obs,  block, [0, 0, 0])

    assert r_near > r_far
    assert b_near["done"] is False  # no contact force at all
    assert b_far["done"] is False


def test_close_gripper_closing_far_from_block_is_not_done():
    # Real grasp-strength contact force from 0.2m away — the closedness
    # signal alone isn't enough, the dxy/dz gates must still reject this.
    block = [1.3, 0.75, 0.425]
    closed_far_obs = _make_obs([1.5, 0.75, 0.425])

    _, breakdown = reward_close_gripper(
        closed_far_obs, block, [0, 0, 0],
        grip_force={"left": MEASURED_GRIP_STRENGTH, "right": MEASURED_GRIP_STRENGTH},
    )

    assert breakdown["done"] is False


def test_close_gripper_reached_at_measured_real_grasp_force():
    # The WEAKEST of the 5 measured seeds (184.5N) must still clear
    # close_gripper_threshold=0.8 with margin — confirms the calibration
    # documented in close_gripper_force_scale's comment against the actual
    # low end observed, not just the mean.
    block = [1.3, 0.75, 0.425]
    real_grasp_obs = _make_obs([1.3, 0.75, 0.425])

    _, breakdown = reward_close_gripper(
        real_grasp_obs, block, [0, 0, 0],
        grip_force={"left": WEAKEST_MEASURED_GRIP_STRENGTH, "right": WEAKEST_MEASURED_GRIP_STRENGTH},
    )

    assert breakdown["closedness"] >= W.close_gripper_threshold
    assert breakdown["done"] is True


def test_close_gripper_hovering_above_block_is_not_done():
    # Small xy offset, real grasp-strength contact force, but a ~4.5cm
    # VERTICAL gap — exactly the exploit found 2026-07-16
    # (record_subgoal_video.py's trajectory log showed a checkpoint
    # retreating upward and closing there instead of descending). The split
    # dxy/dz gate must reject this via d_z alone, independent of the
    # (now force-based) closedness signal.
    block = [1.3, 0.75, 0.425]
    hovering_obs = _make_obs([1.3, 0.75, 0.425 + 0.045])

    _, breakdown = reward_close_gripper(
        hovering_obs, block, [0, 0, 0],
        grip_force={"left": MEASURED_GRIP_STRENGTH, "right": MEASURED_GRIP_STRENGTH},
    )

    assert breakdown["closedness"] >= W.close_gripper_threshold  # real grasp-strength contact
    assert breakdown["d_z"] == pytest.approx(0.045, abs=1e-6)
    assert breakdown["done"] is False


def test_close_gripper_breakdown_surfaces_d_grip_block():
    # subgoal_env.py's drift-interrupt reads breakdown["d_grip_block"]
    # directly rather than recomputing geometry — lock in that it's there
    # and correct.
    block = [1.3, 0.75, 0.425]
    obs = _make_obs([1.4, 0.75, 0.425])   # 0.10m away in x

    _, breakdown = reward_close_gripper(obs, block, [0, 0, 0])

    assert breakdown["d_grip_block"] == pytest.approx(0.10, abs=1e-6)


def test_close_gripper_penalizes_translation_action():
    # Same state and grip force, only the action's dx/dy/dz differ — a
    # policy moving the arm while closing (found 2026-07-16: real training
    # telemetry showed the indirect d_grip_block-after-the-fact penalty
    # alone wasn't enough signal, entropy stayed flat/completion_rate stuck
    # at 0%) should score strictly worse than one holding still.
    block = [1.3, 0.75, 0.425]
    obs = _make_obs([1.3, 0.75, 0.425])
    grasp = {"left": MEASURED_GRIP_STRENGTH, "right": MEASURED_GRIP_STRENGTH}

    r_still, b_still = reward_close_gripper(obs, block, [0, 0, 0], action=[0.0, 0.0, 0.0, -1.0], grip_force=grasp)
    r_moving, b_moving = reward_close_gripper(obs, block, [0, 0, 0], action=[1.0, 1.0, 1.0, -1.0], grip_force=grasp)

    assert r_moving < r_still
    assert b_still["translation_norm"] == pytest.approx(0.0)
    assert b_moving["translation_norm"] == pytest.approx(np.sqrt(3), abs=1e-6)


def test_close_gripper_no_action_defaults_to_no_translation_penalty():
    # action=None (e.g. reward computed at env.reset(), before any action
    # exists) must not penalize or error — backward-compatible default.
    block = [1.3, 0.75, 0.425]
    obs = _make_obs([1.3, 0.75, 0.425])

    reward, breakdown = reward_close_gripper(
        obs, block, [0, 0, 0],
        grip_force={"left": MEASURED_GRIP_STRENGTH, "right": MEASURED_GRIP_STRENGTH},
    )

    assert breakdown["translation_norm"] == 0.0
    assert reward == pytest.approx(breakdown["closedness"], abs=1e-4)  # same pos -> no distance penalty


def test_close_gripper_force_scale_falls_back_to_flat_constant_when_block_width_not_given():
    # block_width=None (the default) must reproduce the ORIGINAL flat
    # close_gripper_force_scale exactly — no caller that hasn't opted into
    # block-size randomization should see any change.
    block = [1.3, 0.75, 0.425]
    obs = _make_obs([1.3, 0.75, 0.425])

    _, breakdown = reward_close_gripper(
        obs, block, [0, 0, 0],
        grip_force={"left": MEASURED_GRIP_STRENGTH, "right": MEASURED_GRIP_STRENGTH},
    )

    assert breakdown["force_scale"] == pytest.approx(W.close_gripper_force_scale)


def test_close_gripper_force_scale_derived_from_block_width_when_given():
    # A THIN block's real achievable grip force (measured ~59N via
    # scripts/measure_grip_contact_force_by_size.py, 2026-08-10 — steady-
    # state force scales with width, not just a miscalibrated constant)
    # never crosses the OLD flat close_gripper_force_scale=130 threshold no
    # matter how well it's gripped — this is the actual bug that stalled a
    # live size-randomized training run at completion_rate=0% for 190+
    # iterations. Confirm the width-derived scale fixes it: the SAME 59N
    # force must now register as done for a block this thin.
    thin_width = 0.01
    thin_measured_force = 59.0
    block = [1.3, 0.75, 0.425]
    obs = _make_obs([1.3, 0.75, 0.425])

    _, flat_breakdown = reward_close_gripper(
        obs, block, [0, 0, 0],
        grip_force={"left": thin_measured_force, "right": thin_measured_force},
    )
    assert flat_breakdown["done"] is False   # old flat scale: unreachable for this width

    _, derived_breakdown = reward_close_gripper(
        obs, block, [0, 0, 0],
        grip_force={"left": thin_measured_force, "right": thin_measured_force},
        block_width=thin_width,
    )
    assert derived_breakdown["force_scale"] < W.close_gripper_force_scale
    assert derived_breakdown["done"] is True


def test_close_gripper_force_scale_scales_up_for_wider_blocks():
    # The same weak (thin-block-appropriate) force must NOT trivially pass
    # as a solid grip for a much wider block (measured ceiling ~312N at
    # 7cm) — otherwise the signal would stop distinguishing weak from
    # strong grips at the wide end of the range.
    block = [1.3, 0.75, 0.425]
    obs = _make_obs([1.3, 0.75, 0.425])
    weak_force = 59.0

    _, breakdown = reward_close_gripper(
        obs, block, [0, 0, 0],
        grip_force={"left": weak_force, "right": weak_force},
        block_width=0.07,
    )

    assert breakdown["force_scale"] > W.close_gripper_force_scale
    assert breakdown["done"] is False


# ---------------------------------------------------------------------------
# lift
# ---------------------------------------------------------------------------
def test_lift_rewards_height_above_table():
    low_block  = [1.3, 0.75, W.table_z + 0.01]
    high_block = [1.3, 0.75, W.table_z + 0.15]
    obs = _make_obs([1.3, 0.75, 0.5])

    r_low,  b_low  = reward_lift(obs, low_block,  [0, 0, 0])
    r_high, b_high = reward_lift(obs, high_block, [0, 0, 0])

    assert r_high > r_low
    assert b_low["done"] is False
    assert b_high["done"] is True


def test_lift_falls_back_to_weights_table_z_when_block_half_height_not_given():
    # block_half_height=None (the default) must reproduce the ORIGINAL
    # fixed-cube behavior exactly — no caller that hasn't opted into
    # block-size randomization should see any change.
    block = [1.3, 0.75, W.table_z + 0.05]
    obs = _make_obs([1.3, 0.75, 0.5])

    reward, breakdown = reward_lift(obs, block, [0, 0, 0])

    assert reward == pytest.approx(0.05)
    assert breakdown["done"] is True


def test_lift_uses_block_half_height_instead_of_table_z_when_given():
    # A TALLER block (half_height=0.06, vs the fixed cube's implicit 0.025)
    # rests higher off TABLE_TOP_Z — height_above_table must be measured
    # from ITS OWN resting height, not the old fixed-cube constant, or a
    # tall block would falsely appear already "lifted" the moment it spawns.
    from think_then_act.env.setup import TABLE_TOP_Z
    block_half_height = 0.06
    resting_pos = [1.3, 0.75, TABLE_TOP_Z + block_half_height]  # freshly spawned, not lifted at all
    obs = _make_obs([1.3, 0.75, 0.5])

    reward, breakdown = reward_lift(obs, resting_pos, [0, 0, 0], block_half_height=block_half_height)

    assert reward == pytest.approx(0.0, abs=1e-6)
    assert breakdown["done"] is False

    lifted_pos = [1.3, 0.75, TABLE_TOP_Z + block_half_height + 0.05]
    reward_lifted, breakdown_lifted = reward_lift(
        obs, lifted_pos, [0, 0, 0], block_half_height=block_half_height,
    )
    assert reward_lifted == pytest.approx(0.05)
    assert breakdown_lifted["done"] is True


# ---------------------------------------------------------------------------
# move_to_target
# ---------------------------------------------------------------------------
def test_move_to_target_rewards_block_target_proximity():
    target = [1.5, 0.75, 0.425]
    far_block   = [1.0, 0.75, 0.425]
    close_block = [1.49, 0.751, 0.425]
    obs = _make_obs([1.3, 0.75, 0.5])

    r_far,   b_far   = reward_move_to_target(obs, far_block,   target)
    r_close, b_close = reward_move_to_target(obs, close_block, target)

    assert r_close > r_far
    assert b_far["done"] is False
    assert b_close["done"] is True


# ---------------------------------------------------------------------------
# release
# ---------------------------------------------------------------------------
def test_release_rewards_opening_gripper():
    block = [1.3, 0.75, 0.425]
    closed_obs = _make_obs([1.3, 0.75, 0.425], finger_widths=(0.0, 0.0))
    open_obs   = _make_obs([1.3, 0.75, 0.425], finger_widths=(0.05, 0.05))

    r_closed, b_closed = reward_release(closed_obs, block, [0, 0, 0])
    r_open,   b_open   = reward_release(open_obs,   block, [0, 0, 0])

    assert r_open > r_closed
    assert b_closed["done"] is False
    assert b_open["done"] is True


def test_release_falls_back_to_fixed_threshold_when_block_width_not_given():
    # block_width=None (the default) must reproduce the ORIGINAL
    # fixed-cube-calibrated threshold exactly.
    block = [1.3, 0.75, 0.425]
    just_under_obs = _make_obs([1.3, 0.75, 0.425], finger_widths=(0.039, 0.039))  # total 0.078
    just_over_obs  = _make_obs([1.3, 0.75, 0.425], finger_widths=(0.041, 0.041))  # total 0.082

    _, b_under = reward_release(just_under_obs, block, [0, 0, 0])
    _, b_over  = reward_release(just_over_obs,  block, [0, 0, 0])

    assert b_under["done"] is False   # 0.078 < release_open_threshold (0.08)
    assert b_over["done"] is True     # 0.082 >= 0.08


def test_release_threshold_derived_from_block_width_when_given():
    # A NARROW block (0.02m) needs the fingers open only a little past its
    # own width + margin to count as "released" — the fixed 0.08 threshold
    # would be needlessly strict (and, for an even wider block, physically
    # unreachable), so this must scale with the actual episode's width.
    block = [1.3, 0.75, 0.425]
    narrow_width = 0.02   # threshold -> min(0.02 + 0.02, finger_open) = 0.04
    open_enough_obs  = _make_obs([1.3, 0.75, 0.425], finger_widths=(0.021, 0.021))  # total 0.042
    not_open_enough  = _make_obs([1.3, 0.75, 0.425], finger_widths=(0.019, 0.019))  # total 0.038

    _, b_open = reward_release(open_enough_obs, block, [0, 0, 0], block_width=narrow_width)
    _, b_not  = reward_release(not_open_enough, block, [0, 0, 0], block_width=narrow_width)

    assert b_open["done"] is True
    assert b_not["done"] is False


def test_release_threshold_clamped_to_finger_open_for_wide_block():
    # A wide enough block (0.095m — beyond the sampler's own 0.08m ceiling,
    # used here purely as a synthetic edge case) plus the margin (0.02m)
    # would naively require opening to 0.115m — physically IMPOSSIBLE, since
    # total_finger_width can never exceed finger_open (0.10m). Without
    # clamping, `done` would never fire even at maximum physical openness.
    # With clamping, the threshold caps at finger_open exactly.
    block = [1.3, 0.75, 0.425]
    wide_width = 0.095   # naive width+margin = 0.115 > finger_open (0.10)
    fully_open_obs = _make_obs([1.3, 0.75, 0.425], finger_widths=(0.05, 0.05))    # total 0.10, max possible
    almost_open_obs = _make_obs([1.3, 0.75, 0.425], finger_widths=(0.049, 0.049))  # total 0.098

    _, b_fully_open   = reward_release(fully_open_obs,   block, [0, 0, 0], block_width=wide_width)
    _, b_almost_open  = reward_release(almost_open_obs,  block, [0, 0, 0], block_width=wide_width)

    assert b_fully_open["done"] is True     # clamped threshold (0.10) reached exactly
    assert b_almost_open["done"] is False   # still short of the clamped threshold


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def test_compute_subgoal_reward_dispatches_to_matching_function():
    block = [1.3, 0.75, 0.425]
    obs = _make_obs([1.3, 0.75, 0.55])

    direct_reward, direct_breakdown = reward_descend(obs, block, [0, 0, 0], collision_prob=0.3)
    dispatch_reward, dispatch_breakdown = compute_subgoal_reward(
        "descend", obs, block, [0, 0, 0], collision_prob=0.3
    )

    assert dispatch_reward == direct_reward
    assert dispatch_breakdown == direct_breakdown


def test_compute_subgoal_reward_covers_every_label():
    block = [1.3, 0.75, 0.425]
    obs = _make_obs([1.3, 0.75, 0.5])
    for label in SUBGOAL_LABELS:
        reward, breakdown = compute_subgoal_reward(label, obs, block, [1.5, 0.75, 0.425])
        assert np.isfinite(reward)
        assert isinstance(breakdown["done"], bool)


def test_compute_subgoal_reward_rejects_unknown_label():
    block = [1.3, 0.75, 0.425]
    obs = _make_obs([1.3, 0.75, 0.5])
    with pytest.raises(ValueError):
        compute_subgoal_reward("not_a_real_subgoal", obs, block, [0, 0, 0])
