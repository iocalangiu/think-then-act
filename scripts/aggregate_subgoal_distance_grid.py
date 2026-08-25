"""
aggregate_subgoal_distance_grid.py

Local (no Modal, no mujoco/gymnasium needed) post-processing step for
wandb_subgoal_distance_grid.py's output. That script writes raw PER-STEP
records (~1-2MB, one row per env step across every subgoal x width x seed)
to a local JSON file. This script collapses those into per-(subgoal, width)
summary rows — mean/std reward, final distance-to-goal, final pose error,
and completion rate (fraction of episodes whose OWN `done` condition fired,
not a timeout/truncation) — small enough (~10KB) to embed directly in the
dashboard HTML template (see build_subgoal_dashboard.py).

Completion rate is computed from each episode's LAST record's "success"
field, which wandb_subgoal_distance_grid.py sets from SubgoalConditionedEnv.
step()'s own `terminated` flag — the base FetchPickAndPlace env essentially
never sets terminated=True on its own (success there is communicated via
info["is_success"]/reward, not this flag), so in practice this IS "did the
subgoal's own done condition fire", not "did the episode end for any
reason." See wandb_subgoal_distance_grid.py's own comment at the point it
sets this field.

Run with:
    python3 scripts/aggregate_subgoal_distance_grid.py
    python3 scripts/aggregate_subgoal_distance_grid.py --in-path artifacts/subgoal_distance_grid.json \
        --out-path artifacts/subgoal_distance_grid_agg.json
"""

import argparse
import json
import statistics as st
from collections import defaultdict


def aggregate(raw: dict) -> dict:
    records = raw["records"]

    # (subgoal, width, seed) -> ordered list of that episode's per-step records
    episodes = defaultdict(list)
    for r in records:
        episodes[(r["subgoal"], r["width"], r["seed"])].append(r)

    per_cell = defaultdict(list)  # (subgoal, width) -> list of per-episode summaries
    for (subgoal, width, seed), steps in episodes.items():
        steps.sort(key=lambda s: s["episode_step"])
        final = steps[-1]
        per_cell[(subgoal, width)].append({
            "seed": seed,
            "total_reward": sum(s["reward"] for s in steps),
            "final_d_xyz": final["d_xyz"],
            "final_pose_error": final.get("pose_error"),
            "n_steps": len(steps),
            "success": bool(final.get("success", False)),
        })

    out = {
        "subgoals": raw["subgoals"], "widths": raw["widths"], "n_seeds": raw["n_seeds"],
        "use_pose_model": raw["use_pose_model"], "per_subgoal_summary": raw["per_subgoal_summary"],
        "grid": [],
    }

    for subgoal in raw["subgoals"]:
        for width in raw["widths"]:
            eps = per_cell.get((subgoal, width), [])
            if not eps:
                continue
            rewards = [e["total_reward"] for e in eps]
            d_xyz = [e["final_d_xyz"] for e in eps]
            pose_errs = [e["final_pose_error"] for e in eps if e["final_pose_error"] is not None]
            n_steps = [e["n_steps"] for e in eps]
            n_success = sum(1 for e in eps if e["success"])
            out["grid"].append({
                "subgoal": subgoal, "width": width, "n": len(eps),
                "reward_mean": round(st.mean(rewards), 4), "reward_std": round(st.pstdev(rewards), 4),
                "d_xyz_mean": round(st.mean(d_xyz), 4), "d_xyz_std": round(st.pstdev(d_xyz), 4),
                "pose_error_mean": round(st.mean(pose_errs), 4) if pose_errs else None,
                "pose_error_std": round(st.pstdev(pose_errs), 4) if len(pose_errs) > 1 else 0.0,
                "mean_steps": round(st.mean(n_steps), 2),
                "completion_rate": round(100.0 * n_success / len(eps), 1),
            })

    # Overall (across all widths) completion rate per subgoal, added to the
    # existing per_subgoal_summary dict alongside mean_episode_reward/
    # mean_final_d_xyz/mean_pose_error (already written by
    # wandb_subgoal_distance_grid.py itself).
    for subgoal in raw["subgoals"]:
        eps = [e for w in raw["widths"] for e in per_cell.get((subgoal, w), [])]
        if eps:
            out["per_subgoal_summary"][subgoal]["completion_rate"] = round(
                100.0 * sum(1 for e in eps if e["success"]) / len(eps), 1)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-path", default="artifacts/subgoal_distance_grid.json")
    parser.add_argument("--out-path", default="artifacts/subgoal_distance_grid_agg.json")
    args = parser.parse_args()

    raw = json.load(open(args.in_path))
    out = aggregate(raw)

    with open(args.out_path, "w") as f:
        json.dump(out, f, indent=1)

    print(f"Aggregated {len(raw['records'])} records -> {len(out['grid'])} grid rows")
    print(f"Wrote -> {args.out_path}")


if __name__ == "__main__":
    main()
