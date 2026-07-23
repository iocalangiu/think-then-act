"""
collect_pose_data.py

Collects (frame, achieved_goal) pairs for self-supervised training of the
block pose predictor (perception/block_pose_predictor.py) — same recipe as
collect_collision_data.py: MuJoCo's own free state is the training LABEL,
never fed to the model itself.

Unlike collision data, block position is present at EVERY step (no rare-
event bias needed) — the thing that matters here is STATE-DISTRIBUTION
COVERAGE, not event-frequency biasing. Cycles through all 6 SUBGOAL_LABELS
via env.setup.init_episode_before_subgoal so the dataset spans the full
range of states the model will actually be queried against at inference
time: resting on the table (align_xy/descend's setup, which is exactly
init_random_episode), and held/elevated at various heights (close_gripper/
lift/move_to_target/release's setups, via the scripted-oracle fast-forward)
— a model trained only on "resting on table" frames would generalize
poorly to "held in the gripper, mid-air," which lift/move_to_target/release
critically depend on.

Gripper start is also randomized per episode (init_episode_before_subgoal's
randomize_gripper=True, reusing env.setup.randomize_gripper_start) for
occlusion/viewing-angle diversity, and a handful of small random actions
are taken per episode after setup to vary the exact camera view slightly
without leaving the setup's state regime.

Optional --dr-level (default 0 = off) applies env.domain_randomization's
staged visual randomization before each episode's frames are captured —
see that module's docstring for what each level adds. Left off by default:
collect a clean baseline dataset first, establish baseline accuracy, THEN
turn randomization on incrementally (isolate "does the model work at all"
from "does randomization break it").

No GPU needed — pure physics + rendering, same as collect_collision_data.py.

Run with:
    modal run scripts/collect_pose_data.py                       # 200 episodes x 10 steps
    modal run scripts/collect_pose_data.py --n-episodes 500
    modal run scripts/collect_pose_data.py --dr-level 1           # + color jitter

Download with:
    python3 -m modal volume get --force rl-harness-model-cache pose_data.jsonl ./artifacts/
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


def frame_to_b64(frame) -> str:
    import base64, io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.function(
    image=rl_image,
    gpu=None,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=1800,
)
def collect_pose_data(
    n_episodes: int = 200,
    steps_per_episode: int = 10,
    seed_offset: int = 0,
    action_scale: float = 0.3,   # small random actions per step, to vary the
                                  # camera view without leaving the setup's state regime
    dr_level: int = 0,           # 0 = off; see env.domain_randomization
) -> dict:
    import os, json
    import numpy as np

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    from think_then_act.env.setup import setup_env, init_episode_before_subgoal
    from think_then_act.env.domain_randomization import randomize_appearance
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.reward.subgoal_reward import SUBGOAL_LABELS

    print("\n" + "=" * 60)
    print(f"  POSE DATA COLLECTION — {n_episodes} episodes x {steps_per_episode} steps")
    print(f"  Cycling setup across all {len(SUBGOAL_LABELS)} subgoal stages, dr_level={dr_level}")
    print("=" * 60)

    # +250, not *2: init_episode_before_subgoal's oracle pre-subgoal setup
    # (close_gripper/lift/move_to_target/release) needs headroom on top of
    # the actual per-episode capture window — same rationale as
    # rollout_workers.py's _worker_init.
    env = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                 max_episode_steps=steps_per_episode + 250)
    )
    setup_env(env)

    rows = []
    n_total_steps = 0
    stage_counts: dict = {s: 0 for s in SUBGOAL_LABELS}

    for ep in range(n_episodes):
        seed = seed_offset + ep
        subgoal_stage = SUBGOAL_LABELS[ep % len(SUBGOAL_LABELS)]
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)

        if dr_level > 0:
            randomize_appearance(env.unwrapped.model, rng, dr_level)

        obs, ok = init_episode_before_subgoal(env, rng, subgoal_stage, randomize_gripper=True)
        if not ok:
            continue

        for step in range(steps_per_episode):
            frame = env.last_frame()
            rows.append({
                "episode"       : seed,
                "step"          : step,
                "subgoal_stage" : subgoal_stage,
                "frame_b64"     : frame_to_b64(frame),
                "achieved_goal" : [round(v, 5) for v in obs["achieved_goal"]],
            })
            stage_counts[subgoal_stage] += 1
            n_total_steps += 1

            action = rng.uniform(-action_scale, action_scale, size=4).astype(np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        if (ep + 1) % 20 == 0:
            print(f"  {ep+1}/{n_episodes}  total_steps={n_total_steps}  stage_counts={stage_counts}")

    env.close()

    out_path = os.path.join(MODEL_CACHE_DIR, "pose_data.jsonl")
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    model_volume.commit()

    print(f"\n  Total steps: {n_total_steps}")
    print(f"  Per-stage counts: {stage_counts}")
    print(f"  Saved -> {out_path}")
    print("=" * 60)

    return {
        "status"      : "PASS",
        "n_steps"     : n_total_steps,
        "stage_counts": stage_counts,
        "out_path"    : out_path,
    }


@app.local_entrypoint()
def main(
    n_episodes: int = 200, steps_per_episode: int = 10, seed_offset: int = 0,
    action_scale: float = 0.3, dr_level: int = 0,
):
    print(f"\nCollecting pose data: {n_episodes} episodes x {steps_per_episode} steps "
          f"(no GPU), dr_level={dr_level}...")
    result = collect_pose_data.remote(
        n_episodes=n_episodes, steps_per_episode=steps_per_episode, seed_offset=seed_offset,
        action_scale=action_scale, dr_level=dr_level,
    )
    print(f"\nDone. n_steps={result['n_steps']}")
    print(f"Per-stage counts: {result['stage_counts']}")
    print(f"\nDownload with:")
    print(f"  python3 -m modal volume get --force rl-harness-model-cache pose_data.jsonl ./artifacts/")
