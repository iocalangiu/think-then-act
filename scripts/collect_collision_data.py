"""
collect_collision_data.py

Collects (frame, action, contact_ground_truth) tuples for self-supervised
training of the collision predictor (perception/collision_predictor.py).

Deliberately not the oracle, not any trained policy — the label distribution
should reflect genuine physical contact/no-contact outcomes, not a scripted
trajectory's biases. contact_ground_truth comes from MuJoCo's own contact
state (env.setup.get_contact_geoms) — free, exact, no extra sensing. It's
used only as a training LABEL; the predictor itself only ever sees pixels
(see perception/collision_predictor.py's docstring for why that split
matters).

Two labels are stored per step:
  - "contact": ANY active MuJoCo contact. Found empirically (2026-07-12,
    first run of this script) to be ~100% at every checkpoint — dominated
    entirely by structural contacts that are always true regardless of
    policy (the robot base resting against the floor/table, the block
    resting on the table by default). Useless as a training signal on its
    own.
  - "table_collision": derived from env.setup.get_meaningful_table_collision_positions
    (len(positions) > 0) — same benign-body exclusion as
    is_meaningful_table_collision, just computed once alongside the actual
    contact-point positions instead of twice, so it actually reflects "did
    the arm hit the table" (gripper/finger touching it is excluded — normal
    grasping, not the failure to detect).

Action sampling (env.setup.sample_descent_biased_action) is biased two ways
— NOT a policy heuristic, a data-collection strategy: pure uniform random
actions (resampled independently every step) essentially never produce the
sustained motion needed to reach and touch the table within max_steps, so
an unbiased random policy generates ~zero real "arm hit the table" examples
even after fixing the label above. Biasing which region of state/action
space gets sampled more densely is standard practice for training a
classifier on a rare event; it doesn't tell the arm what a good outcome is
the way the oracle's action choices would.
  - dz skewed toward descent (not uniform [-1,1]).
  - dx/dy ALSO biased toward the direction from the gripper's current XY to
    the block's XY — found empirically (2026-07-12, validate_collision_
    labels.py) that dz bias alone isn't enough: pure random dx/dy is a local
    random walk with no net drift, so the gripper can wander the whole
    episode without ever reaching the table region at all if it starts far
    from it. Still substantially noisy, not a straight-line path — the goal
    is passing through "near the table but not yet aligned" states, not
    reaching the block precisely.

No GPU needed — this is pure physics stepping, same as generate_sft_data.py.

Gripper start position is also randomized around the table's perimeter per
episode (env.setup.randomize_gripper_start) — otherwise every episode starts
from the same fixed default arm pose, so collisions always occur from the
same side of the table regardless of seed.

Run with:
    modal run scripts/collect_collision_data.py                     # 200 episodes x 60 steps
    modal run scripts/collect_collision_data.py --n-episodes 500
    modal run scripts/collect_collision_data.py --xy-bias-strength 0.0     # no homing toward the block at all
    modal run scripts/collect_collision_data.py --no-randomize-start       # old fixed-start behavior

Also saves, at the end of the run:
    table_collision_coords.json  — [x, y] of every genuine table-collision
        CONTACT POINT (env.setup.get_meaningful_table_collision_positions —
        the actual MuJoCo contact position, not the gripper's own XY, since
        an arm segment like the wrist/forearm can be touching the table
        somewhere other than where the gripper currently is)
    table_collision_coverage.png — scatter of the above over the table's
        real footprint, so you can see at a glance whether collisions are
        spread around all sides or still clustered on one

Download everything with:
    python3 -m modal volume get --force rl-harness-model-cache collision_data.jsonl ./artifacts/
    python3 -m modal volume get --force rl-harness-model-cache table_collision_coords.json ./artifacts/
    python3 -m modal volume get --force rl-harness-model-cache table_collision_coverage.png ./artifacts/
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR
from think_then_act.env.setup import (
    setup_env, init_random_episode, get_contact_geoms,
    sample_descent_biased_action, randomize_gripper_start, get_meaningful_table_collision_positions,
)

plot_image = rl_image.pip_install("matplotlib==3.9.0")


def frame_to_b64(frame) -> str:
    import base64, io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.function(
    image=plot_image,
    gpu=None,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=1800,
)
def collect_collision_data(
    n_episodes: int = 200,
    max_steps: int = 60,   # bumped from 30 — confirmed 2026-07-13 as the actual fix for reachability
    seed_offset: int = 0,
    dz_upper: float = 0.1,           # dz sampled uniform in [-1, dz_upper] — biases toward descent
    xy_bias_strength: float = 0.15,  # see sample_descent_biased_action's docstring for why this is weak
    randomize_start: bool = True,    # see randomize_gripper_start's docstring
) -> dict:
    import os, json
    import numpy as np

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    from think_then_act.env.wrapper import ObservationHarness

    print("\n" + "=" * 60)
    print(f"  COLLISION DATA COLLECTION — {n_episodes} episodes x {max_steps} steps")
    print(f"  Random policy biased toward descent (dz ~ U[-1, {dz_upper}], "
          f"xy_bias_strength={xy_bias_strength}), MuJoCo contact as ground truth")
    print(f"  Gripper start randomized around table perimeter: {randomize_start}")
    print("=" * 60)

    env = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                 max_episode_steps=max_steps * 2)
    )
    setup_env(env)

    rows = []
    n_contact_steps = 0
    n_table_collision_steps = 0
    n_total_steps   = 0
    body_name_counts: dict = {}
    table_collision_xy: list = []   # actual collision contact-point XY —
                                     # for checking coverage is spread around the table, not one side

    for ep in range(n_episodes):
        seed = seed_offset + ep
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        obs, ok = init_random_episode(env, rng)
        if not ok:
            continue

        if randomize_start:
            obs, ok, _ = randomize_gripper_start(env, rng, obs)
            if not ok:
                continue

        for step in range(max_steps):
            frame = env.last_frame()
            action = sample_descent_biased_action(
                rng, dz_upper=dz_upper, xy_bias_strength=xy_bias_strength,
                grip_xy=obs["observation"][0:2], target_xy=obs["achieved_goal"][:2],
            )

            next_obs, _, terminated, truncated, info = env.step(action)

            contact_pairs = get_contact_geoms(env)
            contact_gt = len(contact_pairs) > 0
            # One pass, not two: get_meaningful_table_collision_positions already
            # applies the identical filter is_meaningful_table_collision does
            # (same benign-body exclusion), just also capturing WHERE — so the
            # boolean is derived from it instead of re-running the filter
            # separately over contact_pairs.
            table_collision_positions = get_meaningful_table_collision_positions(env)
            table_collision_gt = len(table_collision_positions) > 0
            for n1, n2 in contact_pairs:
                key = tuple(sorted((n1, n2)))
                body_name_counts[str(key)] = body_name_counts.get(str(key), 0) + 1

            rows.append({
                "episode"        : seed,
                "step"           : step,
                "frame_b64"      : frame_to_b64(frame),
                "action"         : [float(a) for a in action],
                "contact"        : contact_gt,          # raw — dominated by benign structural contact
                "table_collision": table_collision_gt,  # refined — the actual training label
                # Raw geom pairs too — lets any future relabeling rule be
                # applied to already-collected data without re-running physics.
                "contact_pairs"  : [[n1, n2] for n1, n2 in contact_pairs],
                "achieved_goal"  : [round(v, 4) for v in obs["achieved_goal"]],
                "desired_goal"   : [round(v, 4) for v in obs["desired_goal"]],
            })

            # Actual contact-point positions (arm hitting the table), NOT the
            # gripper's own XY — the gripper can be well away from where a
            # wrist/forearm segment is actually touching. Already computed
            # above (table_collision_positions), not re-derived here.
            for pos in table_collision_positions:
                table_collision_xy.append([round(float(pos[0]), 4), round(float(pos[1]), 4)])

            n_total_steps   += 1
            n_contact_steps += int(contact_gt)
            n_table_collision_steps += int(table_collision_gt)
            obs = next_obs

            if terminated or truncated:
                break

        if (ep + 1) % 20 == 0:
            rate = n_contact_steps / max(n_total_steps, 1)
            table_rate = n_table_collision_steps / max(n_total_steps, 1)
            print(f"  {ep+1}/{n_episodes}  total_steps={n_total_steps}  "
                  f"contact_rate={rate:.1%}  table_collision_rate={table_rate:.1%}")

    env.close()

    out_path = os.path.join(MODEL_CACHE_DIR, "collision_data.jsonl")
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    coords_path = os.path.join(MODEL_CACHE_DIR, "table_collision_coords.json")
    with open(coords_path, "w") as f:
        json.dump(table_collision_xy, f)

    # Coverage plot: where around the table did genuine collisions actually
    # happen — one glance answers "is this spread around all sides, or
    # still clustered on one" instead of having to eyeball collision_data.jsonl.
    coverage_plot_path = os.path.join(MODEL_CACHE_DIR, "table_collision_coverage.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    table_cx, table_cy = 1.30, 0.75          # env.setup.TABLE_BODY_NAME's xpos[:2]
    table_half_x, table_half_y = 0.25, 0.35  # env.setup.TABLE_BODY_NAME's geom size[:2]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(Rectangle(
        (table_cx - table_half_x, table_cy - table_half_y), 2 * table_half_x, 2 * table_half_y,
        fill=False, edgecolor="black", linewidth=2, label="table footprint",
    ))
    if table_collision_xy:
        xy = np.array(table_collision_xy)
        ax.scatter(xy[:, 0], xy[:, 1], s=10, alpha=0.35, color="limegreen", label="table_collision")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Table collision coverage — {len(table_collision_xy)} points, {n_episodes} episodes")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    fig.savefig(coverage_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    model_volume.commit()

    contact_rate = n_contact_steps / max(n_total_steps, 1)
    table_collision_rate = n_table_collision_steps / max(n_total_steps, 1)
    top_body_pairs = sorted(body_name_counts.items(), key=lambda kv: -kv[1])[:10]

    print(f"\n  Total steps          : {n_total_steps}")
    print(f"  Contact steps (raw)  : {n_contact_steps} ({contact_rate:.1%})")
    print(f"  Table collision steps: {n_table_collision_steps} ({table_collision_rate:.1%})")
    print(f"  Top contact body pairs: {top_body_pairs}")
    print(f"  Saved -> {out_path}")
    print(f"  Saved -> {coords_path}")
    print(f"  Saved -> {coverage_plot_path}")
    print("=" * 60)

    return {
        "status"              : "PASS",
        "n_steps"             : n_total_steps,
        "contact_rate"        : round(contact_rate, 4),
        "table_collision_rate": round(table_collision_rate, 4),
        "top_body_pairs"      : top_body_pairs,
        "out_path"            : out_path,
        "coords_path"         : coords_path,
        "coverage_plot_path"  : coverage_plot_path,
    }


@app.local_entrypoint()
def main(
    n_episodes: int = 200, max_steps: int = 60, seed_offset: int = 0,
    dz_upper: float = 0.1, xy_bias_strength: float = 0.15, randomize_start: bool = True,
):
    print(f"\nCollecting collision data: {n_episodes} episodes x {max_steps} steps (no GPU)...")
    result = collect_collision_data.remote(
        n_episodes=n_episodes, max_steps=max_steps, seed_offset=seed_offset,
        dz_upper=dz_upper, xy_bias_strength=xy_bias_strength, randomize_start=randomize_start,
    )
    print(f"\nDone. n_steps={result['n_steps']}  "
          f"contact_rate={result['contact_rate']:.1%}  "
          f"table_collision_rate={result['table_collision_rate']:.1%}")
    print(f"Top contact body pairs:")
    for pair, count in result["top_body_pairs"]:
        print(f"  {pair}: {count}")
    print(f"\nDownload with:")
    print(f"  python3 -m modal volume get --force rl-harness-model-cache collision_data.jsonl ./artifacts/")
    print(f"  python3 -m modal volume get --force rl-harness-model-cache table_collision_coords.json ./artifacts/")
    print(f"  python3 -m modal volume get --force rl-harness-model-cache table_collision_coverage.png ./artifacts/")
