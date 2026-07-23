"""
validate_pose_predictor.py

Ground-truth validation for perception/block_pose_predictor.py — checks the
trained checkpoint's actual prediction error against held-out data, broken
down by subgoal_stage (resting-on-table vs. held/lifted), rather than
trusting train_pose_estimator.py's aggregate val_loss alone. Mirrors
validate_collision_labels.py's role in the collision-predictor pipeline:
a diagnostic tool that can catch one state regime failing while another
pulls the average down, instead of trusting an aggregate number.

Reads : /model-cache/pose_data.jsonl, /model-cache/checkpoints/block_pose_predictor.pt
Saves : /model-cache/pose_validation_scatter.png (true vs. predicted XY,
        colored by subgoal_stage)
        /model-cache/pose_validation_error_hist.png (Euclidean error
        histogram, one series per subgoal_stage)

Run with:
    modal run scripts/validate_pose_predictor.py

Download with:
    python3 -m modal volume get --force rl-harness-model-cache pose_validation_scatter.png ./artifacts/
    python3 -m modal volume get --force rl-harness-model-cache pose_validation_error_hist.png ./artifacts/
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR

plot_image = rl_image.pip_install("matplotlib==3.9.0")


@app.function(
    image=plot_image,
    gpu=None,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=900,
)
def validate_pose_predictor(val_frac: float = 0.1, max_examples: int = 0) -> dict:
    import os, json, random, base64, io
    import numpy as np
    import torch
    from PIL import Image

    from think_then_act.perception.block_pose_predictor import BlockPosePredictor
    from think_then_act.reward.subgoal_reward import SUBGOAL_LABELS

    print("\n" + "=" * 60)
    print("  BLOCK POSE PREDICTOR — GROUND-TRUTH VALIDATION")
    print("=" * 60)

    data_path = os.path.join(MODEL_CACHE_DIR, "pose_data.jsonl")
    ckpt_path = os.path.join(MODEL_CACHE_DIR, "checkpoints", "block_pose_predictor.pt")

    examples = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line))
    if max_examples > 0:
        examples = examples[:max_examples]

    # Same episode-level split convention AND seed (0) as
    # train_pose_estimator.py — reproduces the exact same held-out set
    # training never saw, not an independent random sample that could overlap.
    episode_ids = sorted({ex["episode"] for ex in examples})
    rng_split = random.Random(0)
    rng_split.shuffle(episode_ids)
    n_val_eps = max(1, int(len(episode_ids) * val_frac))
    val_ep_ids = set(episode_ids[:n_val_eps])
    val_examples = [ex for ex in examples if ex["episode"] in val_ep_ids]
    print(f"  {len(val_examples)} held-out examples ({n_val_eps} episodes)")

    model = BlockPosePredictor()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    per_stage_errors: dict = {s: [] for s in SUBGOAL_LABELS}
    true_xy, pred_xy, stage_labels = [], [], []

    with torch.no_grad():
        for ex in val_examples:
            frame = np.array(Image.open(io.BytesIO(base64.b64decode(ex["frame_b64"]))))
            pred = model.predict_position(frame)
            true = np.array(ex["achieved_goal"], dtype=np.float32)
            err = float(np.linalg.norm(pred - true))
            stage = ex.get("subgoal_stage", "unknown")
            per_stage_errors.setdefault(stage, []).append(err)
            true_xy.append(true[:2])
            pred_xy.append(pred[:2])
            stage_labels.append(stage)

    print("\n  Mean Euclidean error by subgoal_stage:")
    summary = {}
    for stage, errs in per_stage_errors.items():
        if not errs:
            continue
        mean_cm = float(np.mean(errs) * 100.0)
        median_cm = float(np.median(errs) * 100.0)
        summary[stage] = {"n": len(errs), "mean_cm": round(mean_cm, 3), "median_cm": round(median_cm, 3)}
        print(f"    {stage:15s}  n={len(errs):5d}  mean={mean_cm:6.2f}cm  median={median_cm:6.2f}cm")

    all_errs = [e for errs in per_stage_errors.values() for e in errs]
    overall_mean_cm = float(np.mean(all_errs) * 100.0) if all_errs else float("nan")
    print(f"\n  Overall mean Euclidean error: {overall_mean_cm:.2f}cm  (n={len(all_errs)})")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true_xy = np.array(true_xy)
    pred_xy = np.array(pred_xy)
    stages = sorted(set(stage_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(stages), 1)))
    stage_color = dict(zip(stages, colors))

    fig, ax = plt.subplots(figsize=(6, 6))
    for stage in stages:
        mask = np.array([s == stage for s in stage_labels])
        ax.scatter(true_xy[mask, 0], true_xy[mask, 1], s=8, alpha=0.5,
                   color=stage_color[stage], label=f"true ({stage})", marker="o")
        ax.scatter(pred_xy[mask, 0], pred_xy[mask, 1], s=8, alpha=0.5,
                   color=stage_color[stage], label=f"pred ({stage})", marker="x")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"True vs. predicted block XY — {len(val_examples)} held-out examples")
    ax.legend(loc="upper left", fontsize=6, ncol=2)
    ax.set_aspect("equal")
    scatter_path = os.path.join(MODEL_CACHE_DIR, "pose_validation_scatter.png")
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for stage in stages:
        errs_cm = [e * 100.0 for e in per_stage_errors.get(stage, [])]
        if errs_cm:
            ax.hist(errs_cm, bins=20, alpha=0.5, label=stage, color=stage_color[stage])
    ax.set_xlabel("Euclidean error (cm)")
    ax.set_ylabel("count")
    ax.set_title("Pose prediction error by subgoal_stage")
    ax.legend(fontsize=7)
    hist_path = os.path.join(MODEL_CACHE_DIR, "pose_validation_error_hist.png")
    fig.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    model_volume.commit()

    print(f"\n  Saved -> {scatter_path}")
    print(f"  Saved -> {hist_path}")
    print("=" * 60)

    return {
        "status"         : "PASS",
        "overall_mean_cm": round(overall_mean_cm, 3),
        "per_stage"      : summary,
        "scatter_path"   : scatter_path,
        "hist_path"      : hist_path,
    }


@app.local_entrypoint()
def main(val_frac: float = 0.1, max_examples: int = 0):
    print(f"\nValidating block pose predictor...")
    result = validate_pose_predictor.remote(val_frac=val_frac, max_examples=max_examples)
    print(f"\nDone. overall_mean_cm={result['overall_mean_cm']}")
    for stage, s in result["per_stage"].items():
        print(f"  {stage}: n={s['n']}  mean={s['mean_cm']}cm  median={s['median_cm']}cm")
    print(f"\nDownload with:")
    print(f"  python3 -m modal volume get --force rl-harness-model-cache pose_validation_scatter.png ./artifacts/")
    print(f"  python3 -m modal volume get --force rl-harness-model-cache pose_validation_error_hist.png ./artifacts/")
