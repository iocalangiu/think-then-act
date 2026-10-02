"""
sweep_mdn_fix.py

MDN fix sweep: mdn_smoothness_weight x mdn_components, on a moderate demo
count (n_demos=1000, not the full 2000 — this is a search over whether ANY
setting helps at all, not a final comparison, so kept smaller/faster) with
5 seeds per cell. Both decode-time fixes (argmax tanh bug, weighted_mean
smoothing) were tried and ruled out — see run_bc_scaling_experiment.py's
docstring and the BC Architecture Scaling artifact's MDN callouts for that
history. This sweeps the two TRAINING-side knobs instead: the smoothness
penalty (directly discourages the mixture's own readout from jumping
between timesteps, the confirmed root cause) and K (fewer components =
fewer possible switch points, never tuned before — K=5 was an arbitrary
early default).

Reuses run_bc_scaling_cell directly (same image/function, just called with
a 2D grid of mdn-specific params instead of the architecture-comparison's
grid) — no new Modal function needed.

Run with:
    modal run --detach scripts/sweep_mdn_fix.py
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR
from run_bc_scaling_experiment import run_bc_scaling_cell

SMOOTHNESS_WEIGHTS = [0.0, 1.0, 10.0, 50.0]
N_COMPONENTS = [1, 2, 5]
N_SEEDS = 5
N_DEMOS = 1000


@app.function(image=rl_image, gpu=None,   # NOT a bare modal.Image.debian_slim()
              # — that has no think_then_act package baked in, so importing
              # this very module inside the container crash-loops the moment
              # Modal tries to start it (confirmed 2026-10-02: 59/60 real
              # training cells succeeded fine via the imported
              # run_bc_scaling_cell, which correctly uses rl_image — only
              # THIS function's own image was wrong).
              volumes={MODEL_CACHE_DIR: model_volume}, timeout=120)
def _save_sweep_results(results: list, out_path: str) -> None:
    import json
    import os
    full_path = os.path.join(MODEL_CACHE_DIR, out_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2)
    model_volume.commit()
    print(f"  Saved {len(results)} sweep results -> {full_path}")


@app.local_entrypoint()
def push_local_results(local_path: str = "artifacts/mdn_fix_sweep.json",
                        out_path: str = "logs/mdn_fix_sweep.json"):
    """Persists an already-computed local results file to the volume — for
    recovering from exactly the image bug fixed above, without re-running
    the (expensive) sweep itself. `modal run scripts/sweep_mdn_fix.py::push_local_results`"""
    import json
    with open(local_path) as f:
        results = json.load(f)
    _save_sweep_results.remote(results, out_path)
    print(f"Pushed {len(results)} results from {local_path} -> {out_path}")


@app.local_entrypoint()
def run_sweep(out_path: str = "logs/mdn_fix_sweep.json"):
    import json
    import os
    import statistics

    cells = [
        (sw, k, seed)
        for sw in SMOOTHNESS_WEIGHTS
        for k in N_COMPONENTS
        for seed in range(N_SEEDS)
    ]
    print(f"Running {len(cells)} cells ({len(SMOOTHNESS_WEIGHTS)} smoothness weights x "
          f"{len(N_COMPONENTS)} component counts x {N_SEEDS} seeds), n_demos={N_DEMOS}...")

    results = list(run_bc_scaling_cell.starmap(
        [("mdn", N_DEMOS, seed, 30, 30, 100, 1e-3, 0.02, 0.425,
          "demonstrations/sb3_teacher_full_task.pkl", "argmax", sw, k)
         for sw, k, seed in cells]
    ))

    local_out = os.path.join(os.path.dirname(__file__), "..", "artifacts", "mdn_fix_sweep.json")
    os.makedirs(os.path.dirname(local_out), exist_ok=True)
    with open(local_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} cell results -> {local_out}")
    _save_sweep_results.remote(results, out_path)

    print("\nSummary (mean completion_rate across 5 seeds):")
    for sw in SMOOTHNESS_WEIGHTS:
        for k in N_COMPONENTS:
            cell_results = [r for r in results if r["mdn_smoothness_weight"] == sw and r["mdn_components"] == k]
            rates = [r["completion_rate"] for r in cell_results]
            mean_rate = statistics.mean(rates) if rates else float("nan")
            print(f"  smooth_w={sw:>5}  K={k}: mean completion_rate={mean_rate:.1%}  "
                  f"(seeds: {[f'{r:.0%}' for r in rates]})")
