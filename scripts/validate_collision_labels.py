"""
validate_collision_labels.py

Debugging aid for the collision-label pipeline (collect_collision_data.py +
env.setup.is_meaningful_table_collision): runs ONE episode with the same
descent-biased random policy used for data collection, and plots:
    - left panel: top-down (XY) schematic of the gripper's trajectory
    - right panel: gripper height over the episode, with a reference line at
      table_z — the XY view alone can't tell you whether the gripper was
      ever actually LOW ENOUGH to touch the table while over it, which
      matters since "the trajectory looks like it reaches the table region"
      and "the gripper touched the table" are different claims.
Both panels mark:
    - a red 'x' at every step where ANY MuJoCo contact was active
    - a green 'X' (drawn on top) at every step where table_collision (the
      refined, meaningful label) was active
Full per-step (grip_xy, grip_z, distance-to-disk, contact, table_collision,
contact_pairs) is also printed to stdout, so if the plot still doesn't make
the cause obvious, the exact numbers are already in the Modal run output.

Also saves the actual rendered episode as an .mp4 alongside the .png, so the
real footage and the schematic/height plots can be reviewed side by side.

This is deliberately a schematic plot, not markers overlaid on the actual
rendered camera frame — that would need MuJoCo's camera projection matrix to
place pixels correctly, which isn't worth the risk of getting subtly wrong
sight-unseen (no mujoco installed locally to verify against). The schematic
directly answers the question that actually matters here: is the gripper
over the table when these labels fire, not what the scene looks like.

Since "contact" was found to be ~100% (structural contacts, not the arm) and
"table_collision" excludes those, expect red to cover almost the entire
trajectory and green to be a sparse subset — that's confirmation the labels
are behaving as diagnosed, not a bug in this plot.

Per-step/per-geom stdout dumps are off by default (too much text across a
sweep of several angles) — add --verbose to bring them back when actually
debugging one specific episode.

Run with:
    modal run scripts/validate_collision_labels.py --seed 0
    modal run scripts/validate_collision_labels.py --seed 3 --max-steps 90
    modal run scripts/validate_collision_labels.py --xy-bias-strength 0.0   # no homing at all
    modal run scripts/validate_collision_labels.py --verbose               # full per-step/per-geom dump

Testing gripper-start randomization (2026-07-13, EXPERIMENTAL — see
env.setup.randomize_gripper_start's docstring):
    modal run scripts/validate_collision_labels.py --start-angle-deg 0     # east side of table
    modal run scripts/validate_collision_labels.py --start-angle-deg 90    # north side
    modal run scripts/validate_collision_labels.py --start-angle-deg 180   # west side
    modal run scripts/validate_collision_labels.py --start-angle-deg 270   # south side
    modal run scripts/validate_collision_labels.py --no-randomize-start    # old fixed-start behavior
Check the printed "gripper start: ... diff=..." line — a large diff means
the arm's controller didn't converge to the intended side within
n_settle_steps, before trusting this for a full data collection run.

Output files are named collision_labels_seed{N}_angle{deg}.{png,mp4} (or
_fixedstart if --no-randomize-start) — distinct per run, so different
--start-angle-deg values don't overwrite each other.

Download everything in the validation/ dir with:
    python3 -m modal volume get --force rl-harness-model-cache validation/ ./artifacts/
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR

plot_image = rl_image.pip_install("matplotlib==3.9.0")


@app.function(
    image=plot_image,
    gpu=None,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=600,
)
def validate_collision_labels(
    seed: int = 0, max_steps: int = 60, dz_upper: float = 0.1, xy_bias_strength: float = 0.15,
    randomize_start: bool = True, start_angle_deg: float = None, verbose: bool = False,
) -> dict:
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import (
        setup_env, init_random_episode, get_contact_geoms, randomize_gripper_start,
        is_meaningful_table_collision, sample_descent_biased_action, save_video,
    )
    from think_then_act.env.wrapper import ObservationHarness

    print(f"\nRunning 1 episode (seed={seed}, max_steps={max_steps}) for label validation...")

    env = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                  max_episode_steps=max_steps * 2)
    )
    setup_env(env)

    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    obs, ok = init_random_episode(env, rng)
    if not ok:
        raise RuntimeError(f"init_random_episode failed for seed={seed}")

    block_xy = np.array(obs["achieved_goal"][:2])
    target_xy = np.array(obs["desired_goal"][:2])
    print(f"  block_xy={block_xy}  target_xy={target_xy}")

    if randomize_start:
        theta = np.radians(start_angle_deg) if start_angle_deg is not None else None
        obs, ok, start_info = randomize_gripper_start(env, rng, obs, theta=theta)
        if not ok:
            raise RuntimeError(f"randomize_gripper_start failed for seed={seed}")
        angle_deg_used = round(float(np.degrees(start_info["theta"])))
        print(f"  gripper start: theta={angle_deg_used}deg  "
              f"intended_xy={start_info['intended_start_xy']}  actual_xy={start_info['actual_start_xy']}  "
              f"(diff={np.linalg.norm(start_info['actual_start_xy'] - start_info['intended_start_xy']):.4f} -- "
              f"large means it didn't fully reach the target within n_position_steps)")
        run_tag = f"seed{seed}_angle{angle_deg_used}"
    else:
        run_tag = f"seed{seed}_fixedstart"

    # Ground-truth: enumerate EVERY geom in the model (name, body, type, size,
    # world position), and capture the table BODY's own data while doing so
    # — NOT via mj_name2id on a synthetic "geomN" label (that was the actual
    # bug found 2026-07-12: the table geom has no real registered name, so
    # that lookup returned -1, and numpy's negative indexing silently gave
    # back the LAST geom's data — object0 — instead of erroring). Bodies
    # (table0, object0, robot0:*) ARE real, explicitly-declared names, so
    # matching on body_name here is both correct and mj_name2id-safe.
    import mujoco
    from think_then_act.env.setup import TABLE_BODY_NAME
    raw = env.unwrapped
    _GEOM_TYPE_NAMES = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule",
                        4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh"}
    if verbose:
        print(f"\n  All geoms in the model:")
    table_xpos, table_size, table_gtype = None, None, None
    for gid in range(raw.model.ngeom):
        body_id = raw.model.geom_bodyid[gid]
        body_name = mujoco.mj_id2name(raw.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body{body_id}"
        gtype = _GEOM_TYPE_NAMES.get(int(raw.model.geom_type[gid]), str(int(raw.model.geom_type[gid])))
        xpos = raw.data.geom_xpos[gid]
        size = raw.model.geom_size[gid]
        if verbose:
            name = mujoco.mj_id2name(raw.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}"
            print(f"    {name:20s} body={body_name:25s} type={gtype:8s} "
                  f"xpos=[{xpos[0]:6.3f},{xpos[1]:6.3f},{xpos[2]:6.3f}] size={size}")
        if body_name == TABLE_BODY_NAME:
            table_xpos, table_size, table_gtype = xpos.copy(), size.copy(), gtype

    if table_xpos is None:
        raise RuntimeError(f"No geom found with body name {TABLE_BODY_NAME!r} — check TABLE_BODY_NAME")

    real_table_top_z = float(table_xpos[2] + table_size[2])  # valid since table_gtype == "box"
    print(f"\n  {TABLE_BODY_NAME} geom: type={table_gtype} xpos={table_xpos} size={table_size}")
    print(f"  real_table_top_z (xpos_z + size_z) = {real_table_top_z:.4f}")
    print(f"  (compare to the table_z=0.425 proxy used in reward/subgoal_reward.py — that's the "
          f"BLOCK's resting-center height, table_top + block_half_size, not the table surface itself)")

    table_cx, table_cy, table_r = 1.30, 0.75, 0.20
    table_z = 0.425   # see reward/subgoal_reward.py's SubgoalWeights.table_z

    grip_xy_history, grip_z_history, contact_history, table_collision_history = [], [], [], []
    frames = [env.last_frame()]   # initial frame, before any action — same convention as run_episode.py

    if verbose:
        print(f"  {'step':>4} {'grip_xy':>18} {'grip_z':>8} {'dist_to_disk':>13} "
              f"{'contact':>8} {'table_col':>10}  contact_pairs")
    for step in range(max_steps):
        action = sample_descent_biased_action(
            rng, dz_upper=dz_upper, xy_bias_strength=xy_bias_strength,
            grip_xy=obs["observation"][0:2], target_xy=obs["achieved_goal"][:2],
        )
        obs, _, terminated, truncated, info = env.step(action)
        frames.append(env.last_frame())

        grip_xy = np.array(obs["observation"][0:2])
        grip_z  = float(obs["observation"][2])
        contact_pairs = get_contact_geoms(env)
        contact = len(contact_pairs) > 0
        table_collision = is_meaningful_table_collision(contact_pairs)
        dist_to_disk = float(np.linalg.norm(grip_xy - [table_cx, table_cy]) - table_r)

        # Full per-step diagnostic to stdout — off by default (too much text
        # across a run of many angles), pass verbose=True / --verbose when
        # actually debugging a specific episode.
        if verbose:
            print(f"  {step:>4} [{grip_xy[0]:6.3f},{grip_xy[1]:6.3f}] {grip_z:8.3f} "
                  f"{dist_to_disk:13.3f} {str(contact):>8} {str(table_collision):>10}  {contact_pairs}")

        grip_xy_history.append(grip_xy)
        grip_z_history.append(grip_z)
        contact_history.append(contact)
        table_collision_history.append(table_collision)

        if terminated or truncated:
            break

    env.close()

    grip_xy_history = np.stack(grip_xy_history)
    grip_z_history = np.array(grip_z_history)
    contact_history = np.array(contact_history)
    table_collision_history = np.array(table_collision_history)

    # ------------------------------------------------------------------
    # Plot — left: top-down XY schematic, right: height over time. The XY
    # view alone can't tell you if the gripper was ever actually LOW ENOUGH
    # to touch the table while over it — that's exactly the ambiguity this
    # second panel exists to resolve.
    # ------------------------------------------------------------------
    fig, (ax, ax_z) = plt.subplots(1, 2, figsize=(14, 7))

    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(table_cx + table_r * np.cos(theta), table_cy + table_r * np.sin(theta),
            "k--", alpha=0.4, label="block/target sampling disk (not full table extent)")

    ax.plot(grip_xy_history[:, 0], grip_xy_history[:, 1], "-", color="steelblue",
            alpha=0.6, linewidth=1.5, label="gripper XY trajectory", zorder=1)
    ax.plot(grip_xy_history[0, 0], grip_xy_history[0, 1], "o", color="steelblue",
            markersize=8, label="start", zorder=2)

    ax.plot(*block_xy, "ks", markersize=10, label="block", zorder=2)
    ax.plot(*target_xy, "k*", markersize=14, label="target", zorder=2)

    if contact_history.any():
        ax.plot(grip_xy_history[contact_history, 0], grip_xy_history[contact_history, 1],
                "x", color="red", markersize=6, alpha=0.6,
                label=f"any contact ({contact_history.sum()}/{len(contact_history)} steps)", zorder=3)

    if table_collision_history.any():
        ax.plot(grip_xy_history[table_collision_history, 0], grip_xy_history[table_collision_history, 1],
                "X", color="limegreen", markersize=14, markeredgecolor="black",
                label=f"table_collision ({table_collision_history.sum()}/{len(contact_history)} steps)", zorder=4)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"XY trajectory — {run_tag}, {len(contact_history)} steps")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")

    steps = np.arange(len(grip_z_history))
    ax_z.axhline(table_z, color="gray", linestyle="--", alpha=0.6, label=f"table_z proxy={table_z}")
    ax_z.axhline(real_table_top_z, color="darkorange", linestyle="--", alpha=0.8,
                 label=f"real table top (MuJoCo)={real_table_top_z:.3f}")
    ax_z.plot(steps, grip_z_history, "-", color="steelblue", alpha=0.6, linewidth=1.5,
              label="gripper height", zorder=1)
    if contact_history.any():
        ax_z.plot(steps[contact_history], grip_z_history[contact_history],
                  "x", color="red", markersize=6, alpha=0.6, label="any contact", zorder=2)
    if table_collision_history.any():
        ax_z.plot(steps[table_collision_history], grip_z_history[table_collision_history],
                  "X", color="limegreen", markersize=14, markeredgecolor="black",
                  label="table_collision", zorder=3)
    ax_z.set_xlabel("step")
    ax_z.set_ylabel("gripper z")
    ax_z.set_title("Gripper height over the episode")
    ax_z.legend(loc="upper right", fontsize=8)

    out_dir = os.path.join(MODEL_CACHE_DIR, "validation")
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"collision_labels_{run_tag}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    video_path = os.path.join(out_dir, f"collision_labels_{run_tag}.mp4")
    save_video(frames, video_path, fps=10)

    model_volume.commit()

    print(f"  any_contact: {contact_history.sum()}/{len(contact_history)} steps")
    print(f"  table_collision: {table_collision_history.sum()}/{len(contact_history)} steps")
    print(f"  min grip_z reached: {grip_z_history.min():.4f}  "
          f"(table_z proxy={table_z}, real_table_top_z={real_table_top_z:.4f})")
    print(f"  Saved -> {plot_path}")
    print(f"  Saved -> {video_path}")

    return {
        "status"        : "PASS",
        "n_steps"        : int(len(contact_history)),
        "n_contact"      : int(contact_history.sum()),
        "n_table_collision": int(table_collision_history.sum()),
        "real_table_top_z": round(real_table_top_z, 4),
        "min_grip_z"     : round(float(grip_z_history.min()), 4),
        "plot_path"      : plot_path,
        "video_path"     : video_path,
    }


@app.local_entrypoint()
def main(
    seed: int = 0, max_steps: int = 60, dz_upper: float = 0.1, xy_bias_strength: float = 0.15,
    randomize_start: bool = True, start_angle_deg: float = None, verbose: bool = False,
):
    result = validate_collision_labels.remote(
        seed=seed, max_steps=max_steps, dz_upper=dz_upper, xy_bias_strength=xy_bias_strength,
        randomize_start=randomize_start, start_angle_deg=start_angle_deg, verbose=verbose,
    )
    import os
    tag = os.path.basename(result["plot_path"])[len("collision_labels_"):-len(".png")]
    print(f"\nDone. n_steps={result['n_steps']}  any_contact={result['n_contact']}  "
          f"table_collision={result['n_table_collision']}")
    print(f"python3 -m modal volume get --force rl-harness-model-cache validation/ ./artifacts/  "
          f"# pulls everything; look for collision_labels_{tag}.{{png,mp4}}")
