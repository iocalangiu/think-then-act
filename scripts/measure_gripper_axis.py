"""
measure_gripper_axis.py

One-off diagnostic for block-size randomization (see hierarchical_architecture
memory, Stage 4): which WORLD axis do the gripper's two fingers actually
close along? Block-size randomization needs to know this — the "width"
constrained by finger_open (0.10m ceiling) is whichever of the block's three
half-extents lines up with that axis; the other two horizontal/vertical
dimensions are unconstrained by the gripper and can vary more freely.

Never assumed from the MJCF source before (the block was always a cube, so
it never mattered which axis was "width" — all three half-extents were
identical). Measured directly here instead of guessed: reads each finger
BODY's world position (raw.data.xpos) at a few different finger-joint
openings and reports which axis the displacement between them is
concentrated on.

No GPU needed.

Run with:
    modal run scripts/measure_gripper_axis.py
"""

import modal
from think_then_act.modal_app import app, rl_image


@app.function(image=rl_image, gpu=None, timeout=120)
def measure_gripper_axis() -> dict:
    import numpy as np
    import mujoco
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env

    env = gym.make("FetchPickAndPlace-v3", render_mode="rgb_array")
    setup_env(env)
    raw = env.unwrapped

    left_id  = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_BODY, "robot0:l_gripper_finger_link")
    right_id = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_BODY, "robot0:r_gripper_finger_link")
    gripper_id = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_BODY, "robot0:gripper_link")

    print("\n" + "=" * 60)
    print("  GRIPPER CLOSING-AXIS CHECK")
    print("=" * 60)

    readings = []
    for finger_qpos in (0.0, 0.025, 0.05):
        env.reset(seed=0)
        for fname in ("robot0:r_gripper_finger_joint", "robot0:l_gripper_finger_joint"):
            fid = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_JOINT, fname)
            raw.data.qpos[raw.model.jnt_qposadr[fid]] = finger_qpos
        mujoco.mj_forward(raw.model, raw.data)

        left_pos  = raw.data.xpos[left_id].copy()
        right_pos = raw.data.xpos[right_id].copy()
        gripper_pos = raw.data.xpos[gripper_id].copy()
        delta = left_pos - right_pos
        print(f"\n  finger_qpos={finger_qpos:.3f}")
        print(f"    gripper_link xpos = {gripper_pos}")
        print(f"    l_finger xpos     = {left_pos}")
        print(f"    r_finger xpos     = {right_pos}")
        print(f"    l - r delta       = {delta}  (|x|={abs(delta[0]):.5f} |y|={abs(delta[1]):.5f} |z|={abs(delta[2]):.5f})")
        readings.append(delta.tolist())

    env.close()

    axis_names = ["x", "y", "z"]
    abs_deltas = np.abs(np.array(readings))
    dominant_axis_idx = int(np.argmax(abs_deltas.mean(axis=0)))
    dominant_axis = axis_names[dominant_axis_idx]

    print("\n" + "=" * 60)
    print(f"  DOMINANT CLOSING AXIS: {dominant_axis} (index {dominant_axis_idx})")
    print(f"  mean |delta| per axis: {abs_deltas.mean(axis=0)}")
    print("=" * 60)

    return {
        "status": "PASS",
        "readings": readings,
        "dominant_axis": dominant_axis,
        "dominant_axis_idx": dominant_axis_idx,
    }


@app.local_entrypoint()
def main():
    result = measure_gripper_axis.remote()
    print(f"\nDone: dominant closing axis = {result['dominant_axis']}")
