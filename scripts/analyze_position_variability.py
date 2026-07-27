"""
analyze_position_variability.py

Answers a specific question raised while diagnosing the high-level VLM's SFT
instability: is there enough variability in gripper/block/target positions
across subgoal_sft_data.jsonl for the model to actually have to READ the
numbers, rather than partially memorize a near-constant one?

Motivated by a real, confirmed gap: env.setup.init_random_episode
randomizes block and target position on the table disk, but never touches
the gripper's own starting position — that always comes from gym's fixed
reset pose. Every "align_xy" example captured at a fresh reset (most of the
`chained` source's first decision, and every `random`-source episode's
first step) therefore shares the IDENTICAL gripper_pos, down to the same
decimal digits — visible directly in training's quick_eval logs, which kept
showing "Gripper at [1.983, 0.75, 0.778]" across many different episodes.

Reads : /model-cache/subgoal_sft_data.jsonl (already generated — this does
        NOT regenerate anything, just analyzes what's there)
Saves : /model-cache/eval/position_variability.json — gripper_pos/
        achieved_goal/desired_goal/subgoal/source ONLY (frame_b64 dropped
        per row before anything is held in memory or written out) — small
        enough to download and drive a 3D plot from, unlike the full
        dataset (which is dominated by base64-encoded frames).

Run with:
    modal run --detach scripts/analyze_position_variability.py
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


def _axis_stats(values: list) -> dict:
    import numpy as np
    arr = np.asarray(values, dtype=np.float64)
    # Unique count at 3-decimal rounding -- distinguishes "genuinely varies"
    # from "technically float-unique but visually/semantically constant"
    # (e.g. floating-point noise around one fixed value).
    n_unique = len(set(round(float(v), 3) for v in arr))
    return {
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "range": round(float(arr.max() - arr.min()), 4),
        "std": round(float(arr.std()), 4),
        "mean": round(float(arr.mean()), 4),
        "n_unique_at_3dp": n_unique,
        "n": len(arr),
    }


def _point_type_stats(rows: list, key: str) -> dict:
    """key: 'gripper_pos' | 'achieved_goal' | 'desired_goal'."""
    xs = [r[key][0] for r in rows]
    ys = [r[key][1] for r in rows]
    zs = [r[key][2] for r in rows]
    return {"x": _axis_stats(xs), "y": _axis_stats(ys), "z": _axis_stats(zs)}


@app.function(
    image=rl_image,
    gpu=None,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=600,
)
def analyze_position_variability() -> dict:
    import os, json

    data_path = os.path.join(MODEL_CACHE_DIR, "subgoal_sft_data.jsonl")
    print(f"\nReading {data_path}...")

    rows = []
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line)
            # Drop frame_b64 immediately -- only the position/label fields
            # are needed, and the full file is dominated by base64 images.
            rows.append({
                "gripper_pos"  : ex["gripper_pos"],
                "achieved_goal": ex["achieved_goal"],
                "desired_goal" : ex["desired_goal"],
                "subgoal"      : ex["subgoal"],
                "source"       : ex.get("source"),
            })
    print(f"  {len(rows)} rows loaded (positions only, frames dropped)")

    overall = {
        "gripper_pos"  : _point_type_stats(rows, "gripper_pos"),
        "achieved_goal": _point_type_stats(rows, "achieved_goal"),
        "desired_goal" : _point_type_stats(rows, "desired_goal"),
    }

    # Same breakdown restricted to the "chained" source's FIRST decision per
    # episode (call_index==0, i.e. the fresh-reset align_xy example) --
    # exactly where the suspiciously-constant gripper_pos was observed. The
    # trimmed `rows` list above doesn't carry episode id, so re-read for
    # this specific breakdown: the FIRST row written per episode during
    # generation IS call_index 0 (see generate_subgoal_sft_data.py's
    # run_chained_episode -- align_xy's example is appended before the
    # skill runs, i.e. first), so first-occurrence order per episode id
    # reconstructs it without needing to store call_index explicitly.
    chained_first = []
    seen = set()
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line)
            if ex.get("source") != "chained":
                continue
            ep = ex["episode"]
            if ep in seen:
                continue
            seen.add(ep)
            chained_first.append({
                "gripper_pos": ex["gripper_pos"], "achieved_goal": ex["achieved_goal"],
                "desired_goal": ex["desired_goal"], "subgoal": ex["subgoal"],
            })

    chained_first_decision = {
        "gripper_pos"  : _point_type_stats(chained_first, "gripper_pos"),
        "achieved_goal": _point_type_stats(chained_first, "achieved_goal"),
        "desired_goal" : _point_type_stats(chained_first, "desired_goal"),
    } if chained_first else None

    # Per-label breakdown -- e.g. does close_gripper (which needs the
    # gripper co-located with the block) have less gripper_pos variability
    # than align_xy (which starts far away)?
    from think_then_act.reward.subgoal_reward import SUBGOAL_LABELS
    per_label = {}
    for label in SUBGOAL_LABELS:
        label_rows = [r for r in rows if r["subgoal"] == label]
        if label_rows:
            per_label[label] = {
                "n": len(label_rows),
                "gripper_pos_std_xyz": [
                    _point_type_stats(label_rows, "gripper_pos")[ax]["std"] for ax in "xyz"
                ],
            }

    print("\n" + "=" * 60)
    print("  POSITION VARIABILITY -- gripper_pos (all rows)")
    print("=" * 60)
    for ax in "xyz":
        s = overall["gripper_pos"][ax]
        print(f"  {ax}: range={s['range']:.3f}  std={s['std']:.4f}  "
              f"unique(3dp)={s['n_unique_at_3dp']}/{s['n']}")

    print("\n  achieved_goal (block):")
    for ax in "xyz":
        s = overall["achieved_goal"][ax]
        print(f"  {ax}: range={s['range']:.3f}  std={s['std']:.4f}  "
              f"unique(3dp)={s['n_unique_at_3dp']}/{s['n']}")

    print("\n  desired_goal (target):")
    for ax in "xyz":
        s = overall["desired_goal"][ax]
        print(f"  {ax}: range={s['range']:.3f}  std={s['std']:.4f}  "
              f"unique(3dp)={s['n_unique_at_3dp']}/{s['n']}")

    if chained_first_decision:
        print(f"\n  chained-source FIRST decision only (n={len(chained_first)}, "
              f"fresh-reset states -- the suspected near-constant subset):")
        for ax in "xyz":
            s = chained_first_decision["gripper_pos"][ax]
            print(f"    gripper {ax}: range={s['range']:.4f}  std={s['std']:.4f}  "
                  f"unique(3dp)={s['n_unique_at_3dp']}/{s['n']}")
    print("=" * 60)

    out_dir = os.path.join(MODEL_CACHE_DIR, "eval")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "position_variability.json")
    with open(out_path, "w") as f:
        json.dump({
            "overall": overall,
            "chained_first_decision": chained_first_decision,
            "per_label": per_label,
            "rows": rows,   # gripper_pos/achieved_goal/desired_goal/subgoal/source only
        }, f)
    model_volume.commit()
    print(f"\n  Saved -> {out_path}  ({len(rows)} rows, positions only)")

    return {
        "n_rows": len(rows),
        "overall": overall,
        "chained_first_decision": chained_first_decision,
        "per_label": per_label,
        "output_path": out_path,
    }


@app.local_entrypoint()
def main():
    handle = analyze_position_variability.spawn()
    print(f"\nJob spawned. Function call ID: {handle.object_id}")
    print(f"Monitor at https://modal.com")
    print(f"\nDownload the compact positions-only file when finished with:")
    print(f"  modal volume get rl-harness-model-cache eval/position_variability.json ./artifacts/")
