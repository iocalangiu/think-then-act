"""
generate_sft_data.py

Generates SFT examples for warm-starting GRPO training.

A scripted oracle runs FetchPickAndPlace-v3 episodes. Each episode covers
all three phases (APPROACH → GRASP → CARRY) in one continuous run. Block
and target positions are randomised uniformly within a disk on the table
surface so the VLM learns to read actual state numbers, not pattern-match
a fixed scene.

Each row in sft_data.jsonl is one training example:
    frame_b64   — PNG of the current scene
    achieved_goal / desired_goal — state numbers
    think       — oracle reasoning grounded in those numbers
    action      — [dx, dy, dz, grip]  (grip: +1.0=OPEN, -1.0=CLOSE)
    phase       — APPROACH | GRASP | CARRY

Run with:
    modal run generate_sft_data.py                          # 50 episodes
    modal run generate_sft_data.py --n-episodes 100
    modal run generate_sft_data.py --debug                  # 1 episode, verbose
"""

import modal
from modal_config import app, rl_image, model_volume, MODEL_CACHE_DIR
from env_utils import setup_env, init_random_episode


# ---------------------------------------------------------------------------
# Oracle helpers
# ---------------------------------------------------------------------------

def oracle_action(obs_arr, achieved_goal, desired_goal):
    """Scripted heuristic for FetchPickAndPlace. Returns (action, phase)."""
    import numpy as np

    rel          = obs_arr[6:9]              # block - grip (object_rel_pos)
    finger_width = float(np.sum(obs_arr[9:11]))

    d_3d = float(np.linalg.norm(rel))
    d_xy = float(np.linalg.norm(rel[:2]))

    block_z = float(achieved_goal[2])
    grip_z  = block_z - float(rel[2])

    # "lifted" = block genuinely above table surface (table ~0.4m, resting z ~0.425m)
    block_lifted  = block_z > 0.45
    is_grasped    = block_lifted and d_3d < 0.10
    # Hysteresis: stay in GRASP while gripper is within 10 cm above block.
    # Prevents the open+descend ↔ close+lift 2-step oscillation.
    at_block_zone = grip_z <= block_z + 0.10

    if is_grasped:
        phase     = "CARRY"
        direction = np.array(desired_goal) - np.array(achieved_goal)
        grip      = -1.0                              # CLOSE — keep block grasped
    elif d_3d < 0.10 or at_block_zone:
        phase = "GRASP"
        if grip_z > block_z + 0.025:
            direction = np.array(rel)                 # move toward block (XY+Z)
            grip      = 1.0                           # OPEN during descent
        elif finger_width > 0.07:
            direction = np.zeros(3)                   # stay still, close fingers
            grip      = -1.0                          # CLOSE
        else:
            direction = np.array([0.0, 0.0, 1.0])    # lift
            grip      = -1.0                          # CLOSE — maintain grasp
    elif d_xy > 0.1:
        phase     = "APPROACH"
        direction = np.array([rel[0], rel[1], 0.0])   # lateral only
        grip      = 1.0                               # OPEN
    else:
        phase     = "APPROACH"
        direction = np.array([rel[0], rel[1], rel[2]])  # full 3D descent
        grip      = 1.0                               # OPEN

    norm  = float(np.linalg.norm(direction)) + 1e-8
    scale = min(1.0, float(np.linalg.norm(direction)) / 0.05)
    dx, dy, dz = (direction / norm) * scale

    return np.clip([dx, dy, dz, grip], -1.0, 1.0).astype(np.float32), phase


def make_think_text(obs_arr, achieved_goal, desired_goal, action, phase):
    """
    Generate oracle reasoning grounded in the actual state numbers.
    Grip convention: +1.0 = OPEN fingers, -1.0 = CLOSE fingers.
    """
    import numpy as np

    block_pos_arr = np.array(achieved_goal)
    grip_pos_arr  = block_pos_arr - obs_arr[6:9]

    grip_pos  = [round(float(v), 3) for v in grip_pos_arr]
    block_pos = [round(float(v), 3) for v in block_pos_arr]
    target    = [round(float(v), 3) for v in desired_goal]
    fw        = round(float(np.sum(obs_arr[9:11])), 3)

    d_gb = round(float(np.linalg.norm(block_pos_arr - grip_pos_arr)), 3)
    d_bt = round(float(np.linalg.norm(
        np.array(desired_goal) - np.array(achieved_goal)
    )), 3)

    dx, dy, dz, grip = [round(float(v), 2) for v in action]

    if phase == "APPROACH":
        return (
            f"The gripper is at {grip_pos} and the block is at {block_pos}. "
            f"Distance gripper→block: {d_gb}m. Target is at {target}. "
            f"I am in the APPROACH phase — I need to move the gripper to the block "
            f"before I can pick it up. I keep the gripper open (grip=+1.0) "
            f"and move toward the block: dx={dx}, dy={dy}, dz={dz}, grip={grip}."
        )
    elif phase == "GRASP":
        if grip > 0:  # descending, fingers open
            return (
                f"The gripper is at {grip_pos}, {d_gb}m from the block at {block_pos}. "
                f"I am in the GRASP phase, descending toward the block. "
                f"I keep the fingers open (grip=+1.0) during descent: "
                f"dx={dx}, dy={dy}, dz={dz}, grip={grip}."
            )
        elif fw > 0.07:  # at block level, closing fingers
            return (
                f"The gripper is at {grip_pos}, touching the block at {block_pos} "
                f"(distance: {d_gb}m). Finger width is {fw}m — fingers are not yet closed. "
                f"I stay still and close the gripper (grip=-1.0): "
                f"dx={dx}, dy={dy}, dz={dz}, grip={grip}."
            )
        else:  # fingers closed, lift
            return (
                f"The gripper is at {grip_pos}. Finger width is {fw}m — block is grasped. "
                f"I lift the block off the table: dz={dz}, grip={grip} (keep closed)."
            )
    else:  # CARRY
        return (
            f"The block is at {block_pos} (grasped, lifted). Target is at {target}, "
            f"{d_bt}m away. I am in the CARRY phase. "
            f"I move the block toward the target while keeping the gripper closed (grip=-1.0): "
            f"dx={dx}, dy={dy}, dz={dz}, grip={grip}."
        )


def frame_to_b64(frame) -> str:
    import base64, io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def save_mp4(frames: list, path: str, fps: int = 15) -> None:
    import subprocess, tempfile, os
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(tmpdir, f"{i:04d}.png"))
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(fps),
             "-i", os.path.join(tmpdir, "%04d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", path],
            check=True, capture_output=True,
        )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, seed: int, max_steps: int,
                verbose: bool = False) -> tuple[list[dict], list]:
    """
    Run one oracle episode with randomised block/target positions.
    Each episode covers APPROACH → GRASP → CARRY in sequence.
    Returns (examples, frames).
    """
    import numpy as np

    rng      = np.random.default_rng(seed)
    obs, _   = env.reset(seed=seed)
    obs, ok  = init_random_episode(env, rng)
    if not ok:
        return [], []

    if verbose:
        obs_arr = obs["observation"]
        grip  = [round(float(v), 3) for v in obs_arr[0:3]]
        block = [round(float(v), 3) for v in obs["achieved_goal"]]
        tgt   = [round(float(v), 3) for v in obs["desired_goal"]]
        print(f"  ep={seed}  grip={grip}  block={block}  target={tgt}")

    examples = []
    frames   = []

    for _ in range(max_steps):
        obs_arr       = obs["observation"]
        achieved_goal = obs["achieved_goal"]
        desired_goal  = obs["desired_goal"]
        frame         = env.last_frame()

        action, phase = oracle_action(obs_arr, achieved_goal, desired_goal)
        think         = make_think_text(obs_arr, achieved_goal, desired_goal, action, phase)

        if verbose:
            rel   = obs_arr[6:9]
            fw    = float(np.sum(obs_arr[9:11]))
            print(f"    step={len(examples):3d} phase={phase:8s} "
                  f"d3={np.linalg.norm(rel):.3f}m "
                  f"block_z={float(achieved_goal[2]):.3f}m "
                  f"grip_z={float(achieved_goal[2]) - float(rel[2]):.3f}m "
                  f"fw={fw:.3f}m")

        examples.append({
            "frame"        : frame,
            "achieved_goal": achieved_goal.tolist(),
            "desired_goal" : desired_goal.tolist(),
            "think"        : think,
            "action"       : action.tolist(),
            "phase"        : phase,
        })
        frames.append(frame)

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    return examples, frames


# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------

@app.function(
    image=rl_image,
    gpu=None,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=1800,
)
def generate_sft_data(n_episodes: int = 50, max_steps: int = 50,
                      verbose_every: int = 0) -> dict:
    import os, json

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    from obs_wrapper import ObservationHarness

    print(f"\n{'='*60}")
    print(f"  SFT DATA GENERATION — {n_episodes} episodes × {max_steps} steps")
    print(f"  Randomised block/target on table disk r=0.20m")
    print(f"{'='*60}")

    env = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                 max_episode_steps=max_steps * 2)
    )
    setup_env(env)   # shift robot base once — persists across resets

    all_examples = []
    phase_counts = {"APPROACH": 0, "GRASP": 0, "CARRY": 0}
    skip_count   = 0
    sample_frames = None

    for i in range(n_episodes):
        verbose = (verbose_every > 0 and i % verbose_every == 0)
        episode_examples, frames = run_episode(
            env, seed=i, max_steps=max_steps, verbose=verbose
        )

        if not episode_examples:
            skip_count += 1
            continue

        if sample_frames is None:
            sample_frames = frames

        for ex in episode_examples:
            phase_counts[ex["phase"]] += 1
        all_examples.extend(episode_examples)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_episodes}  total={len(all_examples)}"
                  f"  phases={phase_counts}  skipped={skip_count}")

    env.close()

    if sample_frames:
        mp4_path = os.path.join(MODEL_CACHE_DIR, "sft_sample.mp4")
        save_mp4(sample_frames, mp4_path)
        print(f"  Sample video → {mp4_path}  ({len(sample_frames)} frames)")

    out_path = os.path.join(MODEL_CACHE_DIR, "sft_data.jsonl")
    with open(out_path, "w") as f:
        for ex in all_examples:
            row = {k: v for k, v in ex.items() if k != "frame"}
            row["frame_b64"] = frame_to_b64(ex["frame"])
            f.write(json.dumps(row) + "\n")

    model_volume.commit()

    print(f"\n  Done. {len(all_examples)} examples → {out_path}")
    print(f"  Phase distribution : {phase_counts}")
    print(f"  Skipped episodes   : {skip_count}")
    print(f"{'='*60}")

    return {
        "n_examples"  : len(all_examples),
        "phase_counts": phase_counts,
        "skip_count"  : skip_count,
        "output_path" : out_path,
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(n_episodes: int = 50, debug: bool = False):
    if debug:
        print("\nRunning DEBUG mode (1 episode, all steps printed)...")
        result = generate_sft_data.remote(n_episodes=1, max_steps=50, verbose_every=1)
        print(f"\nDone.  Phases: {result['phase_counts']}")
        print("Download with:")
        print("  modal volume get rl-harness-model-cache sft_sample.mp4 ./sft_sample.mp4")
        return

    print(f"\nGenerating SFT data: {n_episodes} episodes × 50 steps (no GPU)...")
    result = generate_sft_data.remote(n_episodes=n_episodes)
    print(f"\nDone.")
    print(f"  Examples : {result['n_examples']}")
    print(f"  Phases   : {result['phase_counts']}")
    print(f"  Skipped  : {result['skip_count']}")
    print(f"  Output   : {result['output_path']}")
    print(f"\nDownload with:")
    print(f"  modal volume get rl-harness-model-cache sft_data.jsonl ./sft_data.jsonl")
    print(f"  modal volume get rl-harness-model-cache sft_sample.mp4 ./sft_sample.mp4")
