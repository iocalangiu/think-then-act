"""
think_then_act.env.setup

Shared environment setup helpers used across data generation, training, and eval scripts.

All functions assume FetchPickAndPlace-v3 wrapped in ObservationHarness.
"""

from typing import Optional, List
import numpy as np


def save_video(frames: List[np.ndarray], path: str, fps: int = 10,
               scale: Optional[str] = None) -> None:
    """
    Write RGB frames to an MP4 at `path`.

    PyAV (imageio's default MP4 backend) doesn't ship libx264, so
    imageio.mimsave(...) on a .mp4 path raises
    `TypeError: expected bytes, NoneType found` from avcodec_find_encoder_by_name.
    Workaround: write frames as PNGs, then encode with the apt-installed
    system ffmpeg binary (which does have libx264).

    `scale`, if given, is an ffmpeg scale filter arg, e.g. "320:-2".
    """
    import os, subprocess, tempfile
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(os.path.join(tmp_dir, f"{i:04d}.png"))
        cmd = ["ffmpeg", "-y", "-framerate", str(fps),
               "-i", os.path.join(tmp_dir, "%04d.png")]
        if scale:
            cmd += ["-vf", f"scale={scale}"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", path]
        subprocess.run(cmd, check=True, capture_output=True)


def setup_env(env) -> None:
    """
    Shift robot base to [0.85, 0.75, 0.0] so the arm is centred on the table.
    Model-level change — persists across env.reset() calls. Call once after
    env creation.
    """
    import mujoco
    raw     = env.unwrapped
    base_id = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_BODY, "robot0:base_link")
    raw.model.body_pos[base_id] = [0.85, 0.75, 0.0]
    mujoco.mj_forward(raw.model, raw.data)


def teleport_block(env, target_xyz) -> None:
    """Move block to target_xyz by setting its free-joint qpos directly."""
    import mujoco
    raw      = env.unwrapped
    joint_id = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_JOINT, "object0:joint")
    qpos_adr = raw.model.jnt_qposadr[joint_id]
    dof_adr  = raw.model.jnt_dofadr[joint_id]
    raw.data.qpos[qpos_adr:qpos_adr + 3]     = np.array(target_xyz, dtype=np.float64)
    raw.data.qpos[qpos_adr + 3:qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    raw.data.qvel[dof_adr:dof_adr + 6]       = 0.0
    mujoco.mj_forward(raw.model, raw.data)


def get_contact_geoms(env) -> list:
    """
    Return (body1_name, body2_name) for every currently active MuJoCo
    contact, identified by BODY name, not raw geom name.

    Ground truth from the physics engine — free every step, no extra sensing.
    Used both as the collision-predictor's training label and by any reward
    term that cares about unwanted contact (see reward/subgoal_reward.py).

    Why body name, not geom name (fixed 2026-07-12): the table's own geom
    has no explicit name in FetchPickAndPlace-v3's MJCF, so identifying it
    required an id-based fallback ("geom22") — fragile (silently wrong if
    the model ever gains/loses a geom earlier in compile order) and, worse,
    NOT a valid mj_name2id lookup key despite looking like one: a diagnostic
    script that tried `mj_name2id(model, GEOM, "geom22")` got -1 (not
    found), and numpy's negative indexing then silently returned the LAST
    geom's data instead of erroring. Bodies are reliably, explicitly named
    in this MJCF (table0, object0, robot0:*, world), so this sidesteps both
    problems at the root.
    """
    import mujoco
    raw   = env.unwrapped
    pairs = []
    for i in range(raw.data.ncon):
        c        = raw.data.contact[i]
        body1_id = raw.model.geom_bodyid[c.geom1]
        body2_id = raw.model.geom_bodyid[c.geom2]
        name1    = mujoco.mj_id2name(raw.model, mujoco.mjtObj.mjOBJ_BODY, body1_id) or f"body{body1_id}"
        name2    = mujoco.mj_id2name(raw.model, mujoco.mjtObj.mjOBJ_BODY, body2_id) or f"body{body2_id}"
        pairs.append((name1, name2))
    return pairs


def has_contact(env, body_substring: str = None) -> bool:
    """
    True if any contact is currently active. If body_substring is given,
    only counts contacts where at least one BODY name contains it
    (case-insensitive) — e.g. has_contact(env, "table0") for table contacts.
    """
    pairs = get_contact_geoms(env)
    if body_substring is None:
        return len(pairs) > 0
    needle = body_substring.lower()
    return any(needle in n1.lower() or needle in n2.lower() for n1, n2 in pairs)


# The table's body name in FetchPickAndPlace-v3's MJCF — explicitly declared
# and stable (unlike its geom, which has no name — see get_contact_geoms's
# docstring). Confirmed via validate_collision_labels.py's full geom
# enumeration (2026-07-12): body=table0, xpos=[1.30,0.75,0.20],
# size=[0.25,0.35,0.2] -> real table top z = 0.20+0.2 = 0.400.
TABLE_BODY_NAME = "table0"

# Contacts that are benign, i.e. NOT the failure mode this predictor should
# flag:
#   - robot0:base_link, robot0:laser_link, object0: structurally always-on
#     (robot base/sensor mount placed at/near the table by setup_env, block
#     resting on the table by default) — true every step regardless of
#     policy, uninformative as a signal either way.
#   - robot0:gripper_link + both finger links: the gripper reaching down
#     near/onto the table surface to grab a block that's resting ON the
#     table is normal, expected grasping behavior, not the collision to
#     avoid (2026-07-13). The failure this predictor targets is the ARM
#     (wrist/forearm/elbow/upperarm) hitting the table from premature
#     descent while still laterally misaligned — those bodies are
#     deliberately NOT in this set, so contact involving them still counts.
_BENIGN_TABLE_CONTACT_BODIES = frozenset({
    "robot0:base_link", "robot0:laser_link", "object0",
    "robot0:gripper_link", "robot0:r_gripper_finger_link", "robot0:l_gripper_finger_link",
})


def sample_descent_biased_action(
    rng, dz_upper: float = 0.1,
    grip_xy=None, target_xy=None, xy_bias_strength: float = 0.15,
) -> np.ndarray:
    """
    Random action with dz skewed toward descent (U[-1, dz_upper] instead of
    the full U[-1, 1]) — used by collect_collision_data.py and
    validate_collision_labels.py so both sample from the same distribution.
    This is a data-collection exploration strategy, not a policy heuristic:
    a pure uniform-random dz essentially never produces the sustained
    downward motion needed to reach table height within a short episode, so
    without this bias the collision label would have ~0 positive examples
    regardless of how the label itself is defined.

    If grip_xy and target_xy are both given (current gripper XY, block XY),
    dx/dy are ALSO biased toward the direction from grip_xy to target_xy —
    found empirically (2026-07-12, validate_collision_labels.py) that dz
    bias alone isn't enough: pure random dx/dy is a local random walk with no
    net drift, so the gripper can easily never reach the table region at all
    within a short episode regardless of how biased dz is. This is still a
    noisy walk (xy_bias_strength=1.0 would be a straight line), not a
    scripted path.

    xy_bias_strength default lowered 0.5 -> 0.15 (2026-07-13): once the
    gripper/fingers touching the table became a BENIGN contact (see
    _BENIGN_TABLE_CONTACT_BODIES — normal grasping behavior, not the failure
    to detect), a strong homing bias became actively counterproductive. It
    steers the gripper toward good alignment WHILE it descends, so by the
    time it's low it's usually well-aligned — producing gripper-table
    contact (now benign), not the ARM-table contact this predictor actually
    targets. The failure mode needing detection is specifically "still
    misaligned AND already descending," which needs weak, not strong,
    homing — enough pull to reach the table region at all over an episode,
    not enough to resolve misalignment before descending into it.
    """
    if grip_xy is not None and target_xy is not None:
        direction = np.asarray(target_xy, dtype=np.float64) - np.asarray(grip_xy, dtype=np.float64)
        norm = np.linalg.norm(direction)
        unit_direction = direction / norm if norm > 1e-6 else np.zeros(2)
        noise = rng.uniform(-1.0, 1.0, size=2)
        dxdy = np.clip(xy_bias_strength * unit_direction + (1.0 - xy_bias_strength) * noise, -1.0, 1.0)
    else:
        dxdy = rng.uniform(-1.0, 1.0, size=2)

    return np.array([
        dxdy[0], dxdy[1],
        rng.uniform(-1.0, dz_upper),   # dz — biased toward descending
        rng.uniform(-1.0, 1.0),        # grip
    ], dtype=np.float32)


def _is_meaningful_table_pair(name1: str, name2: str, table_body: str) -> bool:
    """
    Shared predicate: True if (name1, name2) involves `table_body` and a
    body that ISN'T one of the known always-on structural/benign contacts.
    The one place this filtering logic lives — is_meaningful_table_collision
    and get_meaningful_table_collision_positions both call this instead of
    each reimplementing it (2026-07-13, they'd drifted into duplicating it).
    """
    pair = {name1, name2}
    if table_body not in pair:
        return False
    other_body = next(iter(pair - {table_body}), table_body)
    return other_body not in _BENIGN_TABLE_CONTACT_BODIES


def is_meaningful_table_collision(contact_pairs: list, table_body: str = TABLE_BODY_NAME) -> bool:
    """
    True if any (body1, body2) pair involves `table_body` and a body that
    ISN'T one of the known always-on structural contacts — i.e., something
    that looks like the arm/gripper actually hitting the table, not just the
    robot's fixed mounting geometry or the block resting where it should.

    Pure function over string pairs — no mujoco/env needed, so this can be
    called on already-collected data (e.g. train_collision_predictor.py
    relabeling from a jsonl's stored "contact_pairs" field) without
    re-running physics.
    """
    return any(_is_meaningful_table_pair(n1, n2, table_body) for n1, n2 in contact_pairs)


def get_meaningful_table_collision_positions(env, table_body: str = TABLE_BODY_NAME) -> list:
    """
    Return the world XYZ position of every currently active contact POINT
    that qualifies as a meaningful table collision (arm hitting the table —
    same filter as is_meaningful_table_collision, not the benign structural/
    gripper contacts).

    Deliberately the literal MuJoCo contact position (data.contact[i].pos),
    NOT a proxy like the gripper's own XY (2026-07-13: the gripper can be
    positioned well away from where an arm segment like the wrist or
    forearm is actually touching the table — using gripper position for
    coverage tracking would show where the gripper ended up, not where the
    collision actually happened).

    Needs mujoco directly (unlike is_meaningful_table_collision, which
    works on already-extracted name pairs) since contact positions aren't
    part of what get_contact_geoms returns.
    """
    import mujoco
    raw = env.unwrapped
    positions = []
    for i in range(raw.data.ncon):
        c = raw.data.contact[i]
        body1_id = raw.model.geom_bodyid[c.geom1]
        body2_id = raw.model.geom_bodyid[c.geom2]
        name1 = mujoco.mj_id2name(raw.model, mujoco.mjtObj.mjOBJ_BODY, body1_id) or f"body{body1_id}"
        name2 = mujoco.mj_id2name(raw.model, mujoco.mjtObj.mjOBJ_BODY, body2_id) or f"body{body2_id}"
        if _is_meaningful_table_pair(name1, name2, table_body):
            positions.append(c.pos.copy())
    return positions


def init_random_episode(env, rng) -> tuple:
    """
    Randomise block and target on the table disk (centre [1.30, 0.75], r=0.20m).
    Call after env.reset(). Returns (obs, ok).
    Same rng seed across rollouts in a group → same positions, different actions.
    """
    import mujoco
    raw = env.unwrapped

    table_cx, table_cy, r_max = 1.30, 0.75, 0.20

    def sample_disk():
        r     = np.sqrt(rng.uniform(0.0, 1.0)) * r_max
        theta = rng.uniform(0.0, 2.0 * np.pi)
        return np.array([table_cx + r * np.cos(theta),
                         table_cy + r * np.sin(theta)])

    block_xy = sample_disk()
    for _ in range(20):
        target_xy = sample_disk()
        if np.linalg.norm(target_xy - block_xy) > 0.10:
            break

    teleport_block(env, np.array([block_xy[0], block_xy[1], 0.425]))
    raw.goal = np.array([target_xy[0], target_xy[1], 0.425])

    for fname in ("robot0:r_gripper_finger_joint", "robot0:l_gripper_finger_joint"):
        fid = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_JOINT, fname)
        raw.data.qpos[raw.model.jnt_qposadr[fid]] = 0.05
    mujoco.mj_forward(raw.model, raw.data)

    obs, _, done, trunc, _ = env.step(np.zeros(4, dtype=np.float32))
    return obs, not (done or trunc)


def _subgoal_setup_reached(subgoal: str, obs_arr, achieved_goal, desired_goal,
                            carrying: bool) -> bool:
    """
    Pure predicate: has the scripted oracle (env.oracle.oracle_action)
    reached the state that should hand off to `subgoal`'s own policy?
    Used by init_episode_before_subgoal to fast-forward past whichever
    earlier subgoals `subgoal` implicitly depends on already having
    succeeded — close_gripper/lift/move_to_target/release cannot learn
    anything from a fresh, ungrasped reset, since their rewards
    (reward/subgoal_reward.py) assume the block is already in a state
    consistent with the PRIOR subgoal's completion (found 2026-07-14:
    `lift` had a completely CONSTANT reward all run — block never held,
    so its height literally couldn't change; `close_gripper` learned to
    crash the arm into the table chasing its distance-to-block term,
    since nothing discourages that on the way down from a fresh, far-away
    start). align_xy/descend need no setup — a fresh reset already IS
    their correct starting state — so this is only ever called for the
    other four.
    """
    finger_width = float(np.sum(obs_arr[9:11]))
    block_z = float(achieved_goal[2])
    rel = obs_arr[6:9]
    d_xy = float(np.linalg.norm(rel[:2]))
    grip_z = block_z - float(rel[2])
    d_z = grip_z - block_z
    d_block_target = float(np.linalg.norm(
        np.asarray(desired_goal, dtype=np.float64) - np.asarray(achieved_goal, dtype=np.float64)
    ))

    if subgoal == "close_gripper":
        # Actually well-aligned laterally AND at grasp height, fingers
        # still open — NOT just "oracle phase == GRASP" (that can trigger
        # on Z-proximity alone, at up to at_block_zone's 0.10m Z band,
        # before XY is tight). Found 2026-07-14 via video: the arm was
        # drifting off after closing, and starting genuinely "right above
        # the block" (not just Z-close) is part of the fix, alongside the
        # drift-interrupt in subgoal_env.py.
        #
        # d_z bound is 0.03, NOT tighter — this is NOT arbitrary slack,
        # it's what the oracle CAN deliver: oracle_action's GRASP-phase
        # logic stops descending entirely once d_z<=0.025 (switches to
        # "hold position, close fingers" — see env/oracle.py) and only
        # resumes moving once it starts LIFTING (away from the block).
        # There is no oracle behavior that continues descending toward a
        # tighter d_z once it crosses that internal 0.025 threshold, so
        # requiring less than roughly that (tried 0.01 on 2026-07-15) makes
        # the condition next-to-unreachable — init_episode_before_subgoal
        # keeps timing out and silently falling back to a bare, unpositioned
        # env.reset() (confirmed via the new d_grip_block(init=...) logging:
        # readings around 0.55m, an order of magnitude beyond anything a
        # real setup failure should produce). d_xy isn't gated by the same
        # freeze (it keeps shrinking in step with d_z right up to that
        # point) so it may achieve tighter than 0.03 in practice, but kept
        # at the same bound for simplicity absent evidence it needs to
        # differ.
        return d_xy <= 0.03 and d_z <= 0.03 and finger_width > 0.09
    if subgoal == "lift":
        # Fingers closed around the block, still resting at table height —
        # close_gripper's job is done, lift's hasn't started yet.
        return finger_width <= 0.07 and block_z <= 0.45
    if subgoal == "move_to_target":
        # Just lifted, oracle entering CARRY — block off the table, held.
        return carrying and block_z > 0.45
    if subgoal == "release":
        # Carried to (near) the target — move_to_target's job is done.
        return carrying and d_block_target <= 0.05
    raise ValueError(f"No pre-subgoal setup defined for {subgoal!r}")


def _run_align_xy_until_done(env, obs, align_xy_policy, max_steps: int) -> tuple:
    """
    Runs a TRAINED align_xy policy (not the scripted oracle) forward until
    align_xy's own done condition fires (reused directly from
    reward/subgoal_reward.py's compute_subgoal_reward, not a hand-copied
    threshold), so descend's training/eval starts from wherever a real
    trained align_xy checkpoint actually leaves the arm — its genuine
    deployment context — rather than an idealized oracle state or a fresh
    scattered reset.

    Found 2026-07-24 via scripts/record_full_rollout.py's chained-rollout
    trajectory logging: descend, deployed right after a trained align_xy
    (its real intended context), barely moved at all across its full step
    budget (d_grip_block: 0.5657 -> 0.5587 over 30 steps), then everything
    downstream (close_gripper, lift) inherited and compounded the failure.
    descend's OWN training reset (a fresh, unaligned init_random_episode)
    never once resembled "already xy-aligned, ready to descend" — a state
    its training distribution essentially never covered. This closes that
    gap the same way close_gripper/lift/move_to_target/release's own setup
    already does, just with a trained policy standing in for the oracle
    (align_xy has no oracle-scripted equivalent phase worth reusing here —
    the whole point is matching what the ACTUAL trained checkpoint does).

    Uses ground-truth achieved_goal for align_xy_policy's own observation
    during this pre-roll (not a pose_model estimate) — a deliberate
    simplification: this is SETUP, not the descend episode itself, and
    align_xy's behavior was already shown to be nearly identical with or
    without perception noise. Revisit only if that stops holding.

    Returns (obs, ok) — ok is False if align_xy's done never fires within
    max_steps or the env terminates/truncates first, same convention as
    every other setup path in this module.
    """
    from think_then_act.reward.subgoal_reward import compute_subgoal_reward
    from think_then_act.training.subgoal_features import build_subgoal_observation

    for _ in range(max_steps):
        obs_vec = build_subgoal_observation(
            obs["observation"], obs["achieved_goal"], obs["desired_goal"],
            "align_xy", collision_prob=0.0,
        )
        action = align_xy_policy.act(obs_vec, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)

        _, breakdown = compute_subgoal_reward(
            "align_xy", obs["observation"], obs["achieved_goal"], obs["desired_goal"],
        )
        if breakdown["done"]:
            return obs, True
        if terminated or truncated:
            return obs, False

    return obs, False


def init_episode_before_subgoal(env, rng, subgoal: str, max_setup_steps: int = 200,
                                 randomize_gripper: bool = False, align_xy_policy=None) -> tuple:
    """
    Constructs a starting state appropriate for training `subgoal` in
    isolation, instead of always the same fresh/ungrasped reset
    init_random_episode gives. For align_xy, that fresh reset already IS
    the right starting state — no change. For close_gripper/lift/
    move_to_target/release, each of which only makes sense once an earlier
    subgoal has already succeeded, this runs the scripted oracle
    (env.oracle.oracle_action — the same one scripts/generate_sft_data.py
    uses for VLM SFT data, reused here rather than re-deriving a second
    scripted policy) from that fresh reset up to the matching handoff
    point, then returns control WITHOUT taking that next oracle step —
    see _subgoal_setup_reached for exactly what each subgoal waits for.

    For descend specifically: if align_xy_policy is given (a trained
    policy.subgoal_policy.SubgoalGaussianPolicy-like object with
    .act(obs_vector, deterministic=True)), runs IT forward until align_xy's
    own done condition fires — see _run_align_xy_until_done's docstring for
    why this needed fixing (2026-07-24: descend, never trained on an
    already-aligned start, essentially froze when actually deployed right
    after align_xy). align_xy_policy=None (the default — no checkpoint
    threaded through, e.g. before one exists yet) falls back to the OLD
    behavior (fresh reset, unchanged) — same None-means-unchanged
    convention as collision_model/pose_model elsewhere in this project.

    randomize_gripper: if True, also drives the gripper to a random point
    around the table (randomize_gripper_start) right after init_random_episode,
    before anything else runs. Defaults to False so every EXISTING caller
    (SubgoalConditionedEnv.reset(), i.e. the actual RL training of the
    low-level skills) is byte-for-byte unaffected — only
    scripts/generate_subgoal_sft_data.py opts in. Added 2026-07-17 after a
    position-variability audit of subgoal_sft_data.jsonl found gripper
    start position had EXACTLY ZERO variance (1 unique value across 150
    fresh-reset examples) — init_random_episode randomizes block/target
    but never the gripper, so the high-level VLM was seeing an identical
    gripper number on a meaningful fraction of its align_xy training data.

    Returns (obs, ok) — ok is False if either init_random_episode's own
    reset failed, the (opt-in) gripper repositioning failed, or the
    oracle/align_xy_policy didn't reach the target state within
    max_setup_steps (the arm's reach is finite and randomized block/target
    positions occasionally aren't reachable in budget) — same convention as
    init_random_episode, so callers (subgoal_env.py) handle it the same way
    (fall back to a fresh env.reset()). max_setup_steps=200 is generous —
    generate_sft_data.py's oracle completes the FULL APPROACH->GRASP->CARRY
    sequence in ~50 steps typically, and this only ever needs to reach an
    earlier point than that.
    """
    from think_then_act.env.oracle import oracle_action

    obs, ok = init_random_episode(env, rng)
    if not ok:
        return obs, False

    if randomize_gripper:
        obs, ok, _info = randomize_gripper_start(env, rng, obs)
        if not ok:
            return obs, False

    if subgoal == "align_xy":
        return obs, True

    if subgoal == "descend":
        if align_xy_policy is None:
            return obs, True
        return _run_align_xy_until_done(env, obs, align_xy_policy, max_setup_steps)

    carrying = False
    for _ in range(max_setup_steps):
        obs_arr, achieved, desired = obs["observation"], obs["achieved_goal"], obs["desired_goal"]
        action, _phase, carrying = oracle_action(obs_arr, achieved, desired, carrying)

        if _subgoal_setup_reached(subgoal, obs_arr, achieved, desired, carrying):
            return obs, True

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            return obs, False

    return obs, False


def randomize_gripper_start(
    env, rng, obs, margin: float = 0.15, n_position_steps: int = 40,
    xy_bias_strength: float = 0.9, theta: float = None,
) -> tuple:
    """
    Drive the gripper toward a random point around the table's perimeter,
    using the SAME bounded action interface a real policy uses (each action
    moves at most 0.05m — see bugs_and_fixes, 2026-07-13). dz is held at
    exactly 0 throughout so this only relocates laterally, not descends.
    Call after init_random_episode, before starting the recorded/labeled
    portion of an episode.

    theta: angle around the table in radians (0=+x/"east", pi/2=+y/"north",
    etc.). If None (default), drawn uniformly from rng — pass an explicit
    value to test a specific side deterministically.

    Why this exists (2026-07-13): without it, every episode starts from the
    same fixed default arm pose regardless of seed — only the block's
    position varies (within a small disk) — so a descent-biased random walk
    always approaches the table from the same side.

    Why it's implemented THIS way and not by teleporting (2026-07-13,
    corrects the first version of this function): directly writing
    data.mocap_pos and stepping with a zero action does NOT work.
    gymnasium_robotics' mocap_set_action calls reset_mocap2body_xpos on
    EVERY step, which re-syncs mocap_pos to wherever the gripper body
    CURRENTLY physically is, before adding the action's delta — so a
    direct mocap_pos write is silently discarded the instant the next
    env.step() runs (confirmed against the actual gymnasium_robotics
    source, not assumed). There's no teleport shortcut for an articulated
    arm the way there is for a free-floating object (teleport_block) — the
    gripper has to actually be walked there through the bounded action
    interface, which is what this does.

    Returns (obs, ok, info) — ok is False if repositioning caused an
    unexpected termination/truncation; info has "theta", "intended_start_xy",
    and "actual_start_xy" so callers can check how close it actually got
    (there's no guarantee of exact convergence in a fixed step budget,
    especially for a large jump or an awkward angle for this arm's reach).
    """
    import mujoco
    raw = env.unwrapped

    table_body_id = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_BODY, TABLE_BODY_NAME)
    table_xy = raw.data.xpos[table_body_id][:2].copy()
    # Table's own half-extents (not the smaller block/target sampling disk)
    # so the start point clears the table's actual footprint on every side.
    table_geom_ids = [g for g in range(raw.model.ngeom) if raw.model.geom_bodyid[g] == table_body_id]
    half_extent = raw.model.geom_size[table_geom_ids[0]][:2].copy()

    if theta is None:
        theta = rng.uniform(0.0, 2.0 * np.pi)
    radius_xy = half_extent + margin
    start_xy = table_xy + radius_xy * np.array([np.cos(theta), np.sin(theta)])

    done, trunc = False, False
    for _ in range(n_position_steps):
        current_xy = obs["observation"][0:2]
        action = sample_descent_biased_action(
            rng, dz_upper=0.0, xy_bias_strength=xy_bias_strength,
            grip_xy=current_xy, target_xy=start_xy,
        )
        action[2] = 0.0   # no vertical motion while relocating laterally
        obs, _, done, trunc, _ = env.step(action)
        if done or trunc:
            break

    actual_xy = np.array(obs["observation"][0:2])
    info = {"theta": float(theta), "intended_start_xy": start_xy.copy(), "actual_start_xy": actual_xy}
    return obs, not (done or trunc), info
