"""
visualize_pose_spread.py

Overlays N sampled arm "skeletons" (torso -> shoulder_pan -> shoulder_lift
-> elbow_flex -> wrist_flex -> gripper, the chain through all 4
_ARM_POSE_JOINTS plus the fixed torso base and the end-effector) for
randomize_joint_angles vs randomize_gripper_start_3d, projected onto the
X-Z (reach vs height) plane -- the plane the "folded vs extended, high vs
low" spread already measured in this project's own history
(reach-from-base/height ranges) lives in. Each of the 5 segments gets its
OWN fixed color, consistent across every overlaid skeleton and across
BOTH images, and both images share the same axis scale -- built
specifically so "which scheme has more spread" is a fair visual
comparison, not an artifact of different autoscaling.

Run with:
    modal run scripts/visualize_pose_spread.py
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR

CHAIN_BODIES = [
    "robot0:torso_lift_link", "robot0:shoulder_pan_link", "robot0:shoulder_lift_link",
    "robot0:elbow_flex_link", "robot0:wrist_flex_link", "robot0:gripper_link",
]
SEGMENT_COLORS = [
    (230, 57, 70),    # torso -> shoulder_pan   (red)
    (241, 143, 1),    # shoulder_pan -> shoulder_lift (orange)
    (42, 157, 143),   # shoulder_lift -> elbow_flex   (teal)
    (38, 70, 170),    # elbow_flex -> wrist_flex      (blue)
    (106, 27, 154),   # wrist_flex -> gripper         (purple)
]


def _sample_poses(scheme: str, n_samples: int, seed_start: int = 0):
    import numpy as np
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    from think_then_act.env.setup import (
        setup_env, init_random_episode, randomize_joint_angles, randomize_gripper_start_3d,
    )

    poses = []
    for i in range(n_samples):
        seed = seed_start + i
        rng = np.random.default_rng(seed)
        env = gym.make("FetchPickAndPlace-v3", max_episode_steps=100)
        setup_env(env)
        obs, _ = env.reset(seed=seed)
        obs, ok = init_random_episode(env, rng)
        if scheme == "joint_angles":
            obs, pose_ok = randomize_joint_angles(env, rng)
        else:
            obs, pose_ok, _ = randomize_gripper_start_3d(env, rng, obs)
        if pose_ok:
            raw = env.unwrapped
            xz = [[float(raw.data.body(name).xpos[0]), float(raw.data.body(name).xpos[2])] for name in CHAIN_BODIES]
            poses.append(xz)
        env.close()
    return poses


def _draw_skeletons(poses: list, x_range: tuple, z_range: tuple, title: str, out_path: str):
    from PIL import Image, ImageDraw

    PLOT_W, H, PAD = 620, 620, 55
    LEGEND_W = 230
    W = PLOT_W + LEGEND_W
    img = Image.new("RGB", (W, H), (252, 252, 251))
    draw = ImageDraw.Draw(img, "RGBA")

    def to_px(x, z):
        px = PAD + (x - x_range[0]) / (x_range[1] - x_range[0]) * (PLOT_W - 2 * PAD)
        py = H - PAD - (z - z_range[0]) / (z_range[1] - z_range[0]) * (H - 2 * PAD)
        return px, py

    for gz in [z_range[0] + k * (z_range[1] - z_range[0]) / 4 for k in range(5)]:
        y = to_px(x_range[0], gz)[1]
        draw.line([(PAD, y), (PLOT_W - PAD, y)], fill=(225, 224, 218), width=1)
    for gx in [x_range[0] + k * (x_range[1] - x_range[0]) / 4 for k in range(5)]:
        x = to_px(gx, z_range[0])[0]
        draw.line([(x, PAD), (x, H - PAD)], fill=(225, 224, 218), width=1)

    alpha = max(55, int(220 / max(1, len(poses))))
    for xz in poses:
        pts = [to_px(x, z) for x, z in xz]
        for seg_i in range(len(pts) - 1):
            r, g, b = SEGMENT_COLORS[seg_i]
            draw.line([pts[seg_i], pts[seg_i + 1]], fill=(r, g, b, alpha), width=3)
        for (px, py) in pts:
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(50, 50, 50, min(255, alpha + 40)))

    draw.text((PAD, 18), title, fill=(20, 20, 20))

    legend_x = PLOT_W + 20
    legend_labels = ["torso -> shoulder_pan", "shoulder_pan -> shoulder_lift", "shoulder_lift -> elbow_flex",
                      "elbow_flex -> wrist_flex", "wrist_flex -> gripper"]
    for i, (label, color) in enumerate(zip(legend_labels, SEGMENT_COLORS)):
        ly = 60 + i * 34
        draw.line([(legend_x, ly + 6), (legend_x + 22, ly + 6)], fill=color, width=5)
        draw.text((legend_x, ly + 14), label, fill=(60, 60, 60))

    img.save(out_path)


@app.function(image=rl_image, gpu=None, cpu=2.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=600)
def visualize_pose_spread(n_samples: int = 30) -> dict:
    import os

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    print("sampling randomize_joint_angles...", flush=True)
    poses_joint = _sample_poses("joint_angles", n_samples)
    print(f"  {len(poses_joint)}/{n_samples} valid", flush=True)

    print("sampling randomize_gripper_start_3d...", flush=True)
    poses_3d = _sample_poses("gripper_3d", n_samples)
    print(f"  {len(poses_3d)}/{n_samples} valid", flush=True)

    all_pts = [pt for poses in (poses_joint, poses_3d) for xz in poses for pt in xz]
    xs = [p[0] for p in all_pts]
    zs = [p[1] for p in all_pts]
    pad_x = (max(xs) - min(xs)) * 0.08
    pad_z = (max(zs) - min(zs)) * 0.08
    x_range = (min(xs) - pad_x, max(xs) + pad_x)
    z_range = (min(zs) - pad_z, max(zs) + pad_z)
    print(f"shared scale: x_range={x_range}  z_range={z_range}", flush=True)

    out1 = os.path.join(MODEL_CACHE_DIR, "demonstrations/videos/pose_spread_joint_angles.png")
    out2 = os.path.join(MODEL_CACHE_DIR, "demonstrations/videos/pose_spread_gripper_3d.png")
    os.makedirs(os.path.dirname(out1), exist_ok=True)
    _draw_skeletons(poses_joint, x_range, z_range, f"randomize_joint_angles (n={len(poses_joint)})", out1)
    _draw_skeletons(poses_3d, x_range, z_range, f"randomize_gripper_start_3d (n={len(poses_3d)})", out2)
    model_volume.commit()
    print(f"saved -> {out1}\nsaved -> {out2}", flush=True)

    return {"n_joint_angles": len(poses_joint), "n_gripper_3d": len(poses_3d)}


@app.local_entrypoint()
def main(n_samples: int = 30):
    result = visualize_pose_spread.remote(n_samples=n_samples)
    print("\n", result)
