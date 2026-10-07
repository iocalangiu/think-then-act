"""
plot_brick_xy_distribution.py

Diagnostic scatter plot of the block's spawn distribution: x-y position as
points, z (height above table) encoded as color — same randomized-reset
distribution training/BC data collection already samples from (env/setup.py's
init_random_episode via SubgoalConditionedEnv(subgoal="align_xy")). Useful
for sanity-checking the scene distribution itself (workspace bounds, any
unexpected clustering/gaps) independent of any specific policy's behavior.

Reads : nothing (samples fresh resets directly)
Saves : /model-cache/brick_xy_z_distribution.png

Color choice: a single-hue sequential colormap (viridis) for z, since z is
a continuous magnitude, not a category — never a rainbow colormap for this
kind of data (see dataviz skill: "Sequential = one hue, light->dark").

Run with:
    modal run scripts/plot_brick_xy_distribution.py
    modal run scripts/plot_brick_xy_distribution.py --n-samples 1000

Download with:
    python3 -m modal volume get --force rl-harness-model-cache brick_xy_z_distribution.png ./artifacts/
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
def plot_brick_xy_distribution(n_samples: int = 300, seed: int = 0) -> dict:
    import os
    import numpy as np

    os.environ["MUJOCO_GL"]         = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401

    from think_then_act.env.setup import setup_env
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.training.subgoal_env import SubgoalConditionedEnv

    print("\n" + "=" * 60)
    print(f"  BRICK XY/Z SPAWN DISTRIBUTION — {n_samples} fresh resets")
    print("=" * 60)

    base = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array", max_episode_steps=30)
    )
    setup_env(base)
    env = SubgoalConditionedEnv(base, subgoal="align_xy", max_episode_steps=30)

    positions = np.zeros((n_samples, 3), dtype=np.float64)
    for i in range(n_samples):
        rng = np.random.default_rng(seed + i)
        _, info = env.reset(rng=rng)
        positions[i] = info["block_pos"]
    env.close()

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    print(f"  x: min={x.min():.4f}  max={x.max():.4f}  mean={x.mean():.4f}")
    print(f"  y: min={y.min():.4f}  max={y.max():.4f}  mean={y.mean():.4f}")
    print(f"  z: min={z.min():.4f}  max={z.max():.4f}  mean={z.mean():.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 6))
    scatter = ax.scatter(x, y, c=z, cmap="viridis", s=14, alpha=0.8, edgecolors="none")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("z (height, m)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Block spawn distribution — {n_samples} resets (align_xy)")
    ax.set_aspect("equal")

    out_path = os.path.join(MODEL_CACHE_DIR, "brick_xy_z_distribution.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    model_volume.commit()

    print(f"\n  Saved -> {out_path}")
    print("=" * 60)

    return {
        "n_samples": n_samples,
        "x_range": [float(x.min()), float(x.max())],
        "y_range": [float(y.min()), float(y.max())],
        "z_range": [float(z.min()), float(z.max())],
        "plot_path": out_path,
    }


@app.local_entrypoint()
def main(n_samples: int = 300, seed: int = 0):
    print(f"\nDispatching brick xy/z distribution plot to Modal (CPU)...")
    handle = plot_brick_xy_distribution.spawn(n_samples=n_samples, seed=seed)
    print(f"Job spawned. Function call ID: {handle.object_id}")
    print(f"Monitor at https://modal.com")
