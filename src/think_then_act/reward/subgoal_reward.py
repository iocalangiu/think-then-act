"""
think_then_act.reward.subgoal_reward

Per-subgoal reward functions for the low-level controller in the hierarchical
architecture (see memory: hierarchical_architecture.md). Each function scores
progress toward ONE coarse subgoal the VLM high-level would choose, as
opposed to dense_reward.py's compute_dense_reward, which scores the whole
pick-and-place task at once.

Same observation-vector layout as dense_reward.py (25 floats) — see that
module's docstring for the full breakdown and the obs[3:6]-is-not-block-
position caveat (use achieved_goal instead, always).

Subgoal vocabulary, in the order a full pick-and-place attempt would use them:
    align_xy       — move gripper laterally above the block
    descend        — lower gripper to grasp height, without drifting off XY
    close_gripper  — close fingers around the block
    lift           — raise the block off the table
    move_to_target — carry the block to the target position
    release        — open the gripper once the block is at the target
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from think_then_act.env.setup import TABLE_TOP_Z

SUBGOAL_LABELS = [
    "align_xy",
    "descend",
    "close_gripper",
    "lift",
    "move_to_target",
    "release",
]


@dataclass
class SubgoalWeights:
    """Tunable thresholds/weights, one dataclass shared by all subgoal reward fns."""
    # 2026-09-01: tried retargeting 0.02 -> 0.005 (to match the real UR3e
    # brick's ~1.5-2cm width, narrower than the old 2cm threshold itself —
    # see git history for the full measurement rationale) but a full
    # 300-iter run came in at completion_rate=0.0% (vs 10.0% for the
    # pre-existing best under the SAME stricter eval threshold) — a 4x
    # tightening in one step was too aggressive for the curriculum to close
    # in 300 iterations, confounded further by block_pose_predictor.pt
    # (mis-localizes small blocks, see hierarchical_architecture memory
    # 2026-08-19/20) noising the observation on top of that.
    #
    # 2026-09-03: retrained against GROUND TRUTH + generic per-step
    # pos_noise_std (SubgoalConditionedEnv, use_pose_model=False) instead of
    # the pose model, isolating the actual policy from that CNN's small-
    # block localization failure — completion_rate went 60%->100%->100%
    # (iter50/100/150, early-stopped), with d_xy(final) settling to
    # ~0.008-0.011m at the OLD 0.02 threshold, i.e. it was already
    # converging well inside the never-reached 0.005 target. Tightened to
    # 0.01 (half of the prior attempt's 4x jump, not a repeat of it) —
    # completion_rate 100%->80%->100%->100% (iter50/100/150/200,
    # early-stopped; the single iter100 dip recovered next checkpoint, not
    # a real regression), d_xy(final) settled to ~0.005-0.008m — already in
    # the original brick-width target's range. Now tightening the last
    # step, 0.01 -> 0.005 (the real UR3e brick's own measured target from
    # 2026-09-01), warm-started from this run's _best.pt.
    align_xy_threshold      : float = 0.005   # metres, planar gripper-block distance
    # Added 2026-09-01: reward_align_xy used to score ONLY d_xy — nothing
    # penalized z drift, confirmed even in sim (a recorded rollout showed
    # grip z rising ~0.13m over an align_xy episode) and observed much more
    # visibly on the real UR3e (arm descending during what should be a
    # lateral-only move) — see memory ur3e_sim2real.md, 2026-09-01.
    # Started at 0.2 (same calibration STYLE as descend_xy_weight below,
    # but not scaled aggressively enough — d_xy is unbounded/unnormalized,
    # so 0.2 only cost -0.026 against the observed 0.13m drift, negligible
    # next to typical d_xy terms). Raised to 5.0 (0.13m drift now costs
    # -0.65, on par with a moderate d_xy error) specifically to make z
    # movement expensive rather than merely discouraged — check d_z(final)
    # AND that d_xy(final)/completion_rate don't regress from the policy
    # over-freezing z at the cost of actually closing xy distance, before
    # trusting this number either.
    align_xy_z_penalty      : float = 5.0
    # Added 2026-09-01, alongside the align_xy_done_streak gate in
    # subgoal_env.py (see its own comment) -- same motivation as
    # close_gripper_stillness_weight: without SOME penalty on action
    # magnitude, nothing discourages large actions once already close,
    # which is exactly what produced the observed real-hardware limit-
    # cycle oscillation (dx/dy flipping sign ~every step near the target,
    # never settling within align_xy_threshold). Deliberately SMALLER than
    # close_gripper_stillness_weight (0.5) -- unlike close_gripper (fine
    # manipulation the whole episode), align_xy also needs to cover
    # potentially large initial distances fast, so an equally strong
    # penalty risked discouraging appropriate large early actions too.
    # Starting value, not empirically tuned -- check whether d_xy(final)
    # actually settles near/below align_xy_threshold (not just gets close
    # and bounces) on the next training run before trusting this number.
    align_xy_stillness_weight: float = 0.1
    descend_threshold       : float = 0.02   # metres, vertical gripper-block gap
    # Added 2026-07-20: reward_descend's `done` used to gate on d_z ALONE —
    # confirmed via a live demo rollout (record_subgoal_demo.py, seed 1) that
    # the trained policy can satisfy that while drifting ~0.5m laterally
    # (started descend at d_xy=0.011m, ended at d_xy=0.508m) — nothing in
    # the reward or done condition ever penalized xy movement, so nothing
    # discouraged it. training-time completion_rate is computed from this
    # SAME d_z-only done check, so a drifting-but-fast-descending policy
    # would still report 100% completion, fully masking the problem.
    # Reuses close_gripper_dxy_limit's value (0.03), not a new independently
    # -picked number — descend's whole purpose is hand off to close_gripper
    # already within ITS dxy precondition, so the natural boundary is the
    # same one close_gripper's own `done` already requires.
    #
    # Tightened 0.03 -> 0.01 (2026-09-03): align_xy's own precision retrain
    # (see hierarchical_architecture memory) now converges to ~0.005-0.01m
    # d_xy, but this limit stayed at 0.03 -- descend's `done` would happily
    # accept letting that precision drift back out by 2-3x during the
    # vertical move and still call itself successful, undoing align_xy's
    # work before close_gripper ever sees it. Tightening incrementally
    # (0.03 -> 0.01, not straight to align_xy's own ~0.005 target) --same
    # lesson as align_xy_threshold's failed 4x jump on 2026-09-01.
    descend_dxy_limit       : float = 0.01   # metres, lateral gripper-block distance
    # Per-step penalty weight on d_xy during descent — a done-gate alone
    # only rejects the FINAL state, it doesn't discourage drifting-then-
    # recovering, and gives no gradient signal while d_xy is still small.
    # Calibrated against reward_descend's own -d_z term (which ranges
    # roughly 0 to -0.5 over a typical descent): at the drift actually
    # observed in the demo (0.5m), weight=2.0 costs -1.0 — as much as the
    # ENTIRE vertical task is worth — while a few cm of incidental drift
    # costs very little against the primary z-progress signal. Starting
    # value, not empirically re-tuned yet (unlike close_gripper's
    # multi-round calibration) — check completion_rate AND d_xy(final) on
    # the next training run before trusting this number.
    descend_xy_weight       : float = 2.0
    collision_penalty       : float = 5.0    # weight on collision_prob during descend
    # Closedness used to be LINEAR (1 - width/finger_open), then a Gaussian
    # peaked at an empirically-measured target finger WIDTH (0.048m,
    # calibrated to this project's one fixed 5cm cube via
    # measure_close_gripper_geometry.py) — both were proxies for "is a real
    # grasp happening," inferred indirectly from finger-JOINT position. The
    # width version doesn't generalize: it needs re-measuring for every new
    # block size, which directly blocks size-randomized training (see
    # hierarchical_architecture memory, Stage 4). Replaced 2026-08-09 with a
    # signal built on MuJoCo's own contact solver instead
    # (env/setup.py's grip_contact_forces) — min(left, right) contact-normal
    # FORCE in Newtons, a direct physical read of "are both fingers actually
    # pressing on something," which needs no block-size calibration at all
    # (contact force between two solids doesn't depend on how wide the
    # object is). Also closes the width-Gaussian's original reason for
    # existing more directly: closing on empty air used to score BETTER
    # than a real grasp under the old linear formula, and now simply
    # produces zero contact force rather than just a de-prioritized one.
    #
    # close_gripper_force_scale (130N) was calibrated ONLY against this
    # project's one 5cm cube via scripts/measure_grip_contact_force.py
    # (2026-08-09, 5/5 seeds: 184.5-274.9N, mean 231.8N) — kept below as the
    # FALLBACK for callers that don't pass block_width (byte-identical old
    # behavior). Once size randomization started (2026-08-10), a live
    # training run stalled at completion_rate=0% for 190+ iterations,
    # closedness pinned near 0.0000 — root-caused via a new per-size
    # measurement (scripts/measure_grip_contact_force_by_size.py, 3 seeds/
    # case) to a REAL physical effect, not just miscalibration: steady-state
    # grip force scales strongly with WIDTH (the gripper's closing axis) —
    # thin (1cm) ~59N, baseline (5cm) ~230N, large (7cm) ~312N — and is
    # essentially width-INDEPENDENT of height (tall-height-8cm, still 5cm
    # wide, measured ~231N, matching baseline). Mechanism: the Fetch
    # gripper's fingers are POSITION- not force-controlled, so the force
    # from a "close fully" command against a rigid obstacle is set by the
    # position servo's stiffness and how much travel the object's geometry
    # blocks — not by the object's weight (mass was never varied in this
    # measurement; see close_gripper_force_scale's use below for why that's
    # NOT what's driving this). A single 130N scale meant thin blocks could
    # never cross close_gripper_threshold=0.8 (needing ~143N) no matter how
    # well the policy gripped — reward, not just `done`, stayed flat since
    # tanh(59/130) never leaves its near-zero, near-flat region either.
    #
    # Fix: when block_width is given, predict this EPISODE's achievable
    # force via a linear width fit (from the two width-isolated
    # measurements sharing length=height=0.05: (0.01m, 58.94N) and
    # (0.05m, 229.98N) — cross-checked against large_cube_7cm's measured
    # 312.08N: predicted 315.5N, ~1% off, so the linear-in-width model
    # holds reasonably well outside its two fit points too) and scale the
    # tanh normalizer to a FRACTION of that prediction, same "comfortable
    # margin below the achievable ceiling" style as every other threshold
    # in this file. Only 3 seeds/case — treat as a first-pass calibration,
    # not final; short blocks additionally showed high force VARIANCE
    # (short_height_1cm: 203/46/203N across 3 seeds) that no width formula
    # can smooth over, a separate real instability at short heights.
    #
    # width=0.08 measured as UNGRASPABLE OUTRIGHT (the scripted oracle never
    # reached CARRY across 3 seeds — 0.10m finger_open leaves only 2cm
    # clearance) — no force-scale fix helps there; this remains a known,
    # unresolved limit of the current [0.01, 0.08] width range.
    close_gripper_force_intercept   : float = 16.2    # Newtons, linear-in-width fit intercept
    close_gripper_force_per_width   : float = 4276.0  # Newtons per metre of width, fit slope
    close_gripper_force_scale_fraction: float = 0.6   # fraction of predicted force the tanh
                                                       # normalizer is set to (margin below ceiling)
    close_gripper_force_scale: float = 130.0  # Newtons, tanh normalizer on grip_strength —
                                               # FALLBACK ONLY when block_width is None
    close_gripper_threshold : float = 0.8    # grip closedness (tanh-normalized force), done gate
    finger_open             : float = 0.10   # sum of both finger widths when fully open
    # `done` used to gate on a single aggregate 3D distance, which can't
    # tell "off to the side" apart from "hovering above at the wrong
    # height" — confirmed exploited 2026-07-16 alongside the closedness
    # issue above: the same retreat-and-close-on-air checkpoint stayed
    # under the old combined 0.05m bound almost entirely via a ~4.5cm
    # VERTICAL gap, never actually enclosing the block. Split into
    # separate xy/z bounds so a bad z-offset can't hide behind a small
    # xy-offset (or vice versa). dxy matches the setup handoff's own
    # achievable precision (see env/setup.py's _subgoal_setup_reached);
    # dz is deliberately tighter since z was the axis actually exploited,
    # and the block is only 0.05m thick so a real grasp shouldn't need
    # more than ~2cm of vertical slack.
    close_gripper_dxy_limit  : float = 0.03  # metres, lateral gripper-block distance
    close_gripper_dz_limit   : float = 0.02  # metres, vertical gripper-block gap
    # Closing the fingers physically shoves the block (contact dynamics),
    # and without a check the arm can then drift away entirely rather than
    # recovering — found 2026-07-14 via video, after the pre-subgoal-setup
    # fix (env/setup.py's init_episode_before_subgoal) already got it
    # starting right at the block. subgoal_env.py truncates the episode
    # once d_grip_block exceeds this rather than burning the rest of the
    # step budget on a rollout that's already failed. This stays a coarse
    # AGGREGATE safety net for truncation only — done's actual success
    # gating uses the tighter dxy/dz split above, not this.
    close_gripper_drift_limit: float = 0.05  # metres, gripper-block distance
    # Re-added 2026-07-16 (was `-0.1*d_grip_block`, removed earlier the same
    # day for conflicting with the stillness penalty). Removing it entirely
    # turned out to drop the only thing penalizing ENDING UP far from the
    # block as a state — the stillness penalty only penalizes the action
    # magnitude of the CURRENT step, so several individually-cheap small
    # translations could still accumulate real drift across an episode with
    # nothing in the continuous reward discouraging it (only the hard
    # truncation cutoff at close_gripper_drift_limit, which is a cliff, not
    # a gradient) — d_grip_block(final) kept climbing over training even
    # with the stillness penalty active.
    #
    # First re-add used 0.5 (matching close_gripper_stillness_weight
    # numerically) — but "equal weight" isn't "equal influence" when the two
    # quantities live on totally different scales: d_grip_block is in
    # metres (~0.05-0.065 in the 0.5-weight run's telemetry) while
    # translation_norm is a tanh-bounded action norm (~1.1-1.3, near its max
    # of sqrt(3)~=1.73). At weight=0.5 the stillness term (~0.5*1.2~=0.6) was
    # ~20x LARGER than the distance term (~0.5*0.06~=0.03) despite the equal
    # nominal weight — confirmed by that run's translation_norm NOT
    # decreasing (1.13->1.29) while d_grip_block(final) kept climbing; the
    # distance penalty was too small to matter against the stillness one.
    # Rescaled so both terms matter comparably at their own "clearly bad"
    # reference point: at d_grip_block=close_gripper_drift_limit (0.05), the
    # distance penalty (~1.0) now lands near a fully-saturated stillness
    # penalty (0.5*sqrt(3)~=0.87) and the max closedness reward (1.0) —
    # weight = 1.0/0.05 = 20.
    close_gripper_distance_weight: float = 20.0  # penalty on d_grip_block (state, not action)
    # Direct, every-step incentive to hold the arm still while closing —
    # the oracle (env/oracle.py's GRASP branch) achieves this by construction,
    # zeroing dx/dy/dz outright once close enough. The RL policy has no such
    # constraint, and `-0.1*d_grip_block` above only penalizes drift AFTER
    # contact has already shoved the block (confirmed 2026-07-16: 150-iter
    # training run showed entropy flat and completion_rate stuck at 0%,
    # d_grip_block(final) sitting right at the drift limit every iteration —
    # the indirect penalty alone wasn't enough signal to discover "don't
    # move"). This term penalizes ||dx,dy,dz|| directly, every step,
    # regardless of whether displacement has manifested yet.
    #
    # 0.15 (2026-07-16 first attempt) turned out too weak: a 300-iter run's
    # rollout video (record_subgoal_video.py + the block/gripper trajectory
    # log) showed a checkpoint still outputting near-max-magnitude actions
    # every step (e.g. dx=-1.0, dy=-0.57, dz=-0.88 -> ||.||~=1.45), dragging
    # the block along with the wrist (block_pos tracked grip_pos almost in
    # lockstep on the axis being translated) — at weight=0.15 that only
    # cost ~0.22, cheap against closedness's ~0.52 ceiling, so "rush the
    # close and drag the block" could still net positive. A follow-up
    # experiment shortening max_episode_steps 30->5 (to test whether a
    # tighter horizon alone would help) instead made d_grip_block(final)
    # WORSE (0.05-0.056 -> 0.065-0.072) while entropy fell faster —
    # evidence the policy was converging confidently toward a MORE
    # aggressive rush-and-drag strategy, not less, under time pressure.
    # Reverted to max_episode_steps=30 (isolate one variable) and raised
    # this weight instead: 0.5 makes a full-magnitude translation
    # (||.||~=1.73) cost up to ~0.87 — more than the entire closedness
    # ceiling, so no plausible amount of "rushing" can pay for itself.
    close_gripper_stillness_weight: float = 0.5  # penalty on translation-action L2 norm
    # Block's default resting-center height (init_random_episode's
    # teleport_block places it at z=0.425), used here as the "not lifted"
    # baseline — NOT the physical table surface height. That's actually
    # 0.400 (confirmed via validate_collision_labels.py's geom enumeration,
    # 2026-07-12: table0 body xpos_z=0.20 + geom size_z=0.20). The 0.025
    # gap is exactly the block's own half-thickness (its geom size_z), i.e.
    # table_z = table_top + block_half_size — correct for measuring lift
    # amount, just not literally "the table."
    table_z                 : float = 0.425
    # 0.10 (previous value) was never empirically verified against real
    # physics and turned out to be unreachable — same class of bug as
    # close_gripper_threshold's original 0.9. Confirmed 2026-07-24 via
    # scripts/measure_lift_height.py: the scripted oracle (env/oracle.py,
    # a known-working policy that completes full pick-and-place) never
    # exceeds height_above_table=0.0307m across 10 episodes, remarkably
    # tight/consistent (0.0278-0.0307m range). This isn't an oracle
    # limitation — the delivery target's z is always table_z (init_random_
    # episode sets target z=0.425, same as the block's resting height), so
    # there's no task reason to ever lift the block more than the minimal
    # clearance needed to slide it across the table without contact. An RL
    # policy chasing 0.10 was chasing a target 3-3.5x higher than anything
    # physically sensible for this task can ever produce. Recalibrated to
    # 0.02m: comfortable margin below even the tightest observed episode
    # (0.0278m, ~28% below), while still requiring genuine lift off the
    # resting height (not trivially satisfied near height=0).
    lift_height             : float = 0.02   # metres above table_z counted as "lifted"
    move_to_target_threshold: float = 0.05   # metres, block-target distance (matches env success)
    release_open_threshold  : float = 0.08   # sum of both finger widths counted as "open" —
                                              # calibrated to the ORIGINAL fixed 5cm cube; still
                                              # used as-is whenever reward_release isn't given a
                                              # per-episode block_width (see release_open_margin).
    # Margin added to THIS EPISODE's actual grasp-axis (y) width once
    # block-size randomization is active — release_open_threshold alone
    # can't stay correct once width varies (0.08 is barely "open" for a
    # thin 0.01m block, and physically unreachable to clear a block already
    # 0.08m wide). 0.02 chosen as the max headroom available at the TOP of
    # the sampled width range (0.01-0.08m): for the widest block (0.08m),
    # width+margin=0.10=finger_open exactly, i.e. "open enough to release"
    # correctly collapses to "fully open" at the point where nothing less
    # would physically clear the block — see reward_release's min(...,
    # finger_open) clamp.
    release_open_margin     : float = 0.02   # metres, added to this episode's block width

    def as_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}


DEFAULT_WEIGHTS = SubgoalWeights()


# ---------------------------------------------------------------------------
# Shared geometry extraction — same quantities every subgoal fn needs
# ---------------------------------------------------------------------------
def _geometry(obs, achieved_goal, desired_goal) -> dict:
    obs      = np.asarray(obs,           dtype=np.float64)
    achieved = np.asarray(achieved_goal, dtype=np.float64)
    desired  = np.asarray(desired_goal,  dtype=np.float64)

    grip_pos      = obs[0:3]
    block_pos     = achieved   # NOT obs[3:6] — see module docstring
    gripper_state = obs[9:11]

    d_xy           = float(np.linalg.norm(block_pos[:2] - grip_pos[:2]))
    d_z            = float(grip_pos[2] - block_pos[2])
    d_grip_block   = float(np.linalg.norm(block_pos - grip_pos))
    d_block_target = float(np.linalg.norm(desired - achieved))

    total_finger_width = float(np.sum(gripper_state))
    return {
        "grip_pos": grip_pos, "block_pos": block_pos, "target_pos": desired,
        "d_xy": d_xy, "d_z": d_z,
        "d_grip_block": d_grip_block, "d_block_target": d_block_target,
        "total_finger_width": total_finger_width,
    }


# ---------------------------------------------------------------------------
# Per-subgoal reward functions
# All return (reward: float, breakdown: dict) — breakdown always has "done".
# ---------------------------------------------------------------------------
def reward_align_xy(obs, achieved_goal, desired_goal, weights: SubgoalWeights = DEFAULT_WEIGHTS,
                     action=None):
    g = _geometry(obs, achieved_goal, desired_goal)
    reward = -g["d_xy"] - weights.align_xy_z_penalty * abs(g["d_z"])
    # Added 2026-09-01, same pattern as close_gripper_stillness_weight:
    # a real UR3e rollout of the (at the time) stillness-free checkpoint
    # oscillated in a stable limit cycle once close to the target (dx/dy
    # flipping sign nearly every step, same near-identical magnitude,
    # never settling within align_xy_threshold) — nothing in training ever
    # penalized LARGE actions once already close, since -d_xy alone gives
    # no incentive to ease off. Only xy (action[:2]), not z -- this
    # subgoal's job is xy-only, matching align_xy_z_penalty's same scoping
    # logic above. action is None at env.reset() (nothing taken yet) —
    # same None-means-no-penalty convention as close_gripper's.
    if action is not None:
        xy_translation_norm = float(np.linalg.norm(np.asarray(action, dtype=np.float64)[:2]))
        reward -= weights.align_xy_stillness_weight * xy_translation_norm
    done    = bool(g["d_xy"] <= weights.align_xy_threshold)
    return reward, {"d_xy": round(g["d_xy"], 5), "d_z": round(g["d_z"], 5), "done": done}


def reward_descend(obs, achieved_goal, desired_goal, collision_prob: float = 0.0,
                    weights: SubgoalWeights = DEFAULT_WEIGHTS):
    """
    Penalizes vertical gap, LATERAL drift, AND predicted collision risk —
    this is the subgoal where the project's premature-descent/table-collision
    bug actually lives, so collision_prob (from
    perception.collision_predictor) is a first-class input here, not an
    afterthought.

    The d_xy term (added 2026-07-20, see descend_dxy_limit/descend_xy_weight)
    is NOT optional polish — without it, a policy that only ever minimizes
    d_z has no reason not to drift laterally while descending, and nothing
    in `done` would catch it either (confirmed via a live rollout drifting
    ~0.5m in xy while still satisfying the old z-only done condition).
    """
    g = _geometry(obs, achieved_goal, desired_goal)
    reward = (-g["d_z"]
              - weights.descend_xy_weight * g["d_xy"]
              - weights.collision_penalty * float(collision_prob))
    done    = bool(g["d_z"] <= weights.descend_threshold
                   and g["d_xy"] <= weights.descend_dxy_limit)
    return reward, {
        "d_z": round(g["d_z"], 5), "d_xy": round(g["d_xy"], 5),
        "collision_prob": round(float(collision_prob), 5), "done": done,
    }


def reward_close_gripper(obs, achieved_goal, desired_goal, weights: SubgoalWeights = DEFAULT_WEIGHTS,
                          action=None, grip_force: dict = None, block_width: float = None):
    g = _geometry(obs, achieved_goal, desired_goal)
    # grip_force is None at env.reset() the same way action is (nothing has
    # made contact yet) and whenever a caller doesn't wire it through —
    # defaults to "no contact," not an error, same None-means-unchanged
    # convention as action.
    grip_force = grip_force or {"left": 0.0, "right": 0.0}
    # Bottlenecked by the WEAKER side, not summed — one finger pressing hard
    # while the other floats free isn't a real pinch grasp (see
    # env/setup.py's grip_contact_forces docstring).
    grip_strength = min(grip_force.get("left", 0.0), grip_force.get("right", 0.0))
    # block_width is None for every caller that hasn't opted into block-size
    # randomization — falls back to the flat close_gripper_force_scale (the
    # ORIGINAL fixed-cube calibration), byte-identical to before. Once
    # given, the achievable force for THIS episode's width is predicted via
    # the linear fit (close_gripper_force_intercept/_per_width) and the
    # tanh normalizer is set to a FRACTION of that prediction — see the
    # long comment on close_gripper_force_scale for why a flat scale
    # stalled training the moment width started varying.
    if block_width is None:
        force_scale = weights.close_gripper_force_scale
    else:
        predicted_force = weights.close_gripper_force_intercept + weights.close_gripper_force_per_width * block_width
        force_scale = weights.close_gripper_force_scale_fraction * predicted_force
    # tanh, not raw force — bounds the signal to [0,1) so an occasional
    # contact-solver force spike (stiff-contact settling transient) can't
    # blow the reward outside the scale the other terms (distance,
    # stillness) already live on.
    closedness = float(np.tanh(grip_strength / force_scale))
    # Distance penalty (see close_gripper_distance_weight's comment for why
    # removing it entirely wasn't right) — penalizes ENDING UP far from the
    # block (a state), which the action-only stillness penalty below can't
    # cover on its own.
    reward = closedness - weights.close_gripper_distance_weight * g["d_grip_block"]
    # action is None at env.reset() (nothing has been taken yet) — only
    # step() has a real action to penalize, so this term is a no-op there.
    translation_norm = 0.0
    if action is not None:
        translation_norm = float(np.linalg.norm(np.asarray(action, dtype=np.float64)[:3]))
        reward -= weights.close_gripper_stillness_weight * translation_norm
    # Split xy/z gating (see close_gripper_dxy_limit/close_gripper_dz_limit's
    # comment) — a single aggregate distance let "hovering above the block"
    # (small xy, large z) pass just as easily as "beside it at the right
    # height" (large xy, small z). Both axes now have to be genuinely tight,
    # on top of the force-based closedness already excluding "no contact."
    done = bool(closedness >= weights.close_gripper_threshold
                and g["d_xy"] <= weights.close_gripper_dxy_limit
                and abs(g["d_z"]) <= weights.close_gripper_dz_limit)
    # d_grip_block surfaced here (not just used internally above) so
    # subgoal_env.py can check it against close_gripper_drift_limit without
    # recomputing the geometry itself.
    return reward, {"closedness": round(closedness, 5),
                     "grip_strength": round(grip_strength, 5),
                     "force_scale": round(force_scale, 5),
                     "left_force": round(float(grip_force.get("left", 0.0)), 5),
                     "right_force": round(float(grip_force.get("right", 0.0)), 5),
                     "d_grip_block": round(g["d_grip_block"], 5),
                     "d_xy": round(g["d_xy"], 5), "d_z": round(g["d_z"], 5),
                     "translation_norm": round(translation_norm, 5), "done": done}


def reward_lift(obs, achieved_goal, desired_goal, weights: SubgoalWeights = DEFAULT_WEIGHTS,
                 block_half_height: float = None):
    g = _geometry(obs, achieved_goal, desired_goal)
    # block_half_height is None for every caller that hasn't opted into
    # block-size randomization (env/block_randomization.py) — falls back to
    # weights.table_z, the ORIGINAL fixed-cube resting height, byte-
    # identical to before. Once given, it overrides table_z with THIS
    # episode's own resting height (TABLE_TOP_Z + half its actual sampled
    # height), so "how much has it risen from resting" stays correct across
    # different block heights instead of drifting with a stale constant.
    resting_z = weights.table_z if block_half_height is None else TABLE_TOP_Z + block_half_height
    height_above_table = float(g["block_pos"][2] - resting_z)
    reward = height_above_table
    done    = bool(height_above_table >= weights.lift_height)
    return reward, {"height_above_table": round(height_above_table, 5), "done": done}


def reward_move_to_target(obs, achieved_goal, desired_goal, weights: SubgoalWeights = DEFAULT_WEIGHTS):
    g = _geometry(obs, achieved_goal, desired_goal)
    reward = -g["d_block_target"]
    done    = g["d_block_target"] <= weights.move_to_target_threshold
    return reward, {"d_block_target": round(g["d_block_target"], 5), "done": done}


def reward_release(obs, achieved_goal, desired_goal, weights: SubgoalWeights = DEFAULT_WEIGHTS,
                    block_width: float = None):
    g = _geometry(obs, achieved_goal, desired_goal)
    openness = np.clip(g["total_finger_width"] / weights.finger_open, 0.0, 1.0)
    reward = float(openness)
    # block_width is None for every caller that hasn't opted into block-size
    # randomization — falls back to weights.release_open_threshold exactly
    # (the ORIGINAL fixed-cube-calibrated constant), byte-identical to
    # before. Once given, the open-enough bar derives from THIS episode's
    # actual width instead (see release_open_margin's comment), clamped so
    # it never exceeds finger_open — can't require opening wider than
    # physically possible.
    threshold = (weights.release_open_threshold if block_width is None
                 else min(block_width + weights.release_open_margin, weights.finger_open))
    done = g["total_finger_width"] >= threshold
    return reward, {"total_finger_width": round(g["total_finger_width"], 5), "done": done}


_SUBGOAL_FN = {
    "align_xy"      : reward_align_xy,
    "descend"       : reward_descend,
    "close_gripper" : reward_close_gripper,
    "lift"          : reward_lift,
    "move_to_target": reward_move_to_target,
    "release"       : reward_release,
}


def compute_subgoal_reward(
    subgoal: str,
    obs, achieved_goal, desired_goal,
    collision_prob: float = 0.0,
    weights: SubgoalWeights = DEFAULT_WEIGHTS,
    action=None,
    grip_force: dict = None,
    block_half_height: float = None,
    block_width: float = None,
) -> tuple:
    """
    Dispatch to the reward function matching `subgoal` (must be one of
    SUBGOAL_LABELS). Only `descend` uses collision_prob; `close_gripper`
    and `align_xy` (added 2026-09-01, its own smaller xy-only stillness
    penalty — see align_xy_stillness_weight) use action for a translation-
    stillness penalty; `close_gripper` also uses grip_force
    (env.setup.grip_contact_forces' {"left","right"} dict, the contact-
    force-based closedness signal), and block_width (predicts this
    episode's achievable force, see close_gripper_force_scale's comment);
    `lift` uses block_half_height and `release` also uses block_width (all
    three env.block_randomization-sourced, None unless a caller has opted
    into block-size randomization — see each reward function's own
    docstring for the None-means-unchanged fallback). Other subgoals ignore
    whichever of these don't apply to them. Returns (reward: float,
    breakdown: dict with "done": bool).
    """
    if subgoal not in _SUBGOAL_FN:
        raise ValueError(f"Unknown subgoal {subgoal!r}; must be one of {SUBGOAL_LABELS}")
    if subgoal == "align_xy":
        return reward_align_xy(obs, achieved_goal, desired_goal, weights, action)
    if subgoal == "descend":
        return reward_descend(obs, achieved_goal, desired_goal, collision_prob, weights)
    if subgoal == "close_gripper":
        return reward_close_gripper(obs, achieved_goal, desired_goal, weights, action, grip_force, block_width)
    if subgoal == "lift":
        return reward_lift(obs, achieved_goal, desired_goal, weights, block_half_height)
    if subgoal == "release":
        return reward_release(obs, achieved_goal, desired_goal, weights, block_width)
    return _SUBGOAL_FN[subgoal](obs, achieved_goal, desired_goal, weights)
