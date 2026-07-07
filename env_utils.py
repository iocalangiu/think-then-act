"""
env_utils.py

Shared environment setup helpers used by both generate_sft_data.py and trainer.py.

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
    import mujoco, numpy as np
    raw      = env.unwrapped
    joint_id = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_JOINT, "object0:joint")
    qpos_adr = raw.model.jnt_qposadr[joint_id]
    dof_adr  = raw.model.jnt_dofadr[joint_id]
    raw.data.qpos[qpos_adr:qpos_adr + 3]     = np.array(target_xyz, dtype=np.float64)
    raw.data.qpos[qpos_adr + 3:qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    raw.data.qvel[dof_adr:dof_adr + 6]       = 0.0
    mujoco.mj_forward(raw.model, raw.data)


def init_random_episode(env, rng) -> tuple:
    """
    Randomise block and target on the table disk (centre [1.30, 0.75], r=0.20m).
    Call after env.reset(). Returns (obs, ok).
    Same rng seed across rollouts in a group → same positions, different actions.
    """
    import mujoco, numpy as np
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
