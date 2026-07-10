"""
analyze_seeds.py

Run the oracle on N seeds and classify each as GOOD (oracle succeeds) or
HARD (oracle fails).  Saves a short MP4 per seed + an HTML report you can
open locally in your browser to scroll / filter / sort through everything.

Usage:
    modal run scripts/analyze_seeds.py                 # 100 seeds
    modal run scripts/analyze_seeds.py --n-seeds 200

Download + view:
    modal volume get rl-harness-model-cache seed_analysis/ ./artifacts/seed_analysis/
    open seed_analysis/report.html

Keyboard navigation in the report:
    ← → (or J/K)  step through seeds
    Enter          play / pause
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(
    image=rl_image,
    gpu=None,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=3600 * 2,
)
def analyze_seeds(n_seeds: int = 100, fps: int = 10, max_steps: int = 50) -> dict:
    import os, json
    import numpy as np

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    import gymnasium as gym
    import gymnasium_robotics  # noqa
    from think_then_act.env.wrapper import ObservationHarness
    from think_then_act.env.setup import setup_env, init_random_episode, save_video

    out_dir = os.path.join(MODEL_CACHE_DIR, "seed_analysis")
    os.makedirs(out_dir, exist_ok=True)

    env = ObservationHarness(
        gym.make("FetchPickAndPlace-v3", render_mode="rgb_array",
                 max_episode_steps=max_steps + 10)
    )
    setup_env(env)

    # ------------------------------------------------------------------
    # Oracle (identical to generate_sft_data.py)
    # ------------------------------------------------------------------
    def oracle_action(obs_arr, achieved_goal, desired_goal, carrying=False):
        rel          = obs_arr[6:9]
        finger_width = float(np.sum(obs_arr[9:11]))
        d_3d = float(np.linalg.norm(rel))
        d_xy = float(np.linalg.norm(rel[:2]))
        block_z = float(achieved_goal[2])
        grip_z  = block_z - float(rel[2])
        block_lifted  = block_z > 0.45
        is_grasped    = (block_lifted and d_3d < 0.10) or (carrying and d_3d < 0.12)
        at_block_zone = grip_z <= block_z + 0.10

        if is_grasped:
            phase = "CARRY"; carrying = True
            direction = np.array(desired_goal) - np.array(achieved_goal)
            grip = -1.0
        elif d_3d < 0.10 or at_block_zone:
            phase = "GRASP"; carrying = False
            if grip_z > block_z + 0.025:
                direction = np.array(rel); grip = 1.0
            elif finger_width > 0.07:
                direction = np.zeros(3);   grip = -1.0
            else:
                direction = np.array([0.0, 0.0, 1.0]); grip = -1.0
        elif d_xy > 0.1:
            phase = "APPROACH"; carrying = False
            direction = np.array([rel[0], rel[1], 0.0]); grip = 1.0
        else:
            phase = "APPROACH"; carrying = False
            direction = np.array(rel); grip = 1.0

        norm  = float(np.linalg.norm(direction)) + 1e-8
        scale = min(1.0, float(np.linalg.norm(direction)) / 0.05)
        dx, dy, dz = (direction / norm) * scale
        return np.clip([dx, dy, dz, grip], -1.0, 1.0).astype(np.float32), phase, carrying

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    results = []

    for seed in range(n_seeds):
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        obs, ok = init_random_episode(env, rng)
        if not ok:
            print(f"  seed={seed:04d} [SKIP]")
            continue

        obs_arr     = np.array(obs["observation"])
        block_pos   = np.array(obs["achieved_goal"])
        target_pos  = np.array(obs["desired_goal"])
        gripper_pos = obs_arr[0:3].copy()

        d_grip_block   = float(np.linalg.norm(gripper_pos - block_pos))
        d_block_target = float(np.linalg.norm(block_pos - target_pos))

        frames   = [env.last_frame()]
        carrying = False
        success  = False
        phases   = []

        for _ in range(max_steps):
            obs_arr = np.array(obs["observation"])
            ag = np.array(obs["achieved_goal"])
            dg = np.array(obs["desired_goal"])

            action, phase, carrying = oracle_action(obs_arr, ag, dg, carrying)
            obs, _, terminated, truncated, info = env.step(action)
            frames.append(env.last_frame())
            phases.append(phase)

            if info.get("is_success", False):
                success = True
                break
            if terminated or truncated:
                break

        label = "GOOD" if success else "HARD"
        vid_name = f"seed_{seed:04d}_{label.lower()}.mp4"
        save_video(frames, os.path.join(out_dir, vid_name), fps=fps, scale="320:-2")

        seen, unique_phases = set(), []
        for p in phases:
            if p not in seen:
                seen.add(p); unique_phases.append(p)

        result = {
            "seed"          : seed,
            "label"         : label,
            "success"       : success,
            "d_grip_block"  : round(d_grip_block,   4),
            "d_block_target": round(d_block_target,  4),
            "block_z"       : round(float(block_pos[2]),  4),
            "target_z"      : round(float(target_pos[2]), 4),
            "block_pos"     : [round(v, 4) for v in block_pos.tolist()],
            "target_pos"    : [round(v, 4) for v in target_pos.tolist()],
            "gripper_pos"   : [round(v, 4) for v in gripper_pos.tolist()],
            "n_steps"       : len(phases),
            "phases"        : unique_phases,
            "video"         : vid_name,
        }
        results.append(result)

        mark = "✓" if success else "✗"
        print(f"  seed={seed:04d} [{label}] {mark}  "
              f"d_grip={d_grip_block:.3f}  d_tgt={d_block_target:.3f}  "
              f"phases={unique_phases}  steps={len(phases)}")

    env.close()

    n_good = sum(1 for r in results if r["label"] == "GOOD")
    n_hard = len(results) - n_good
    print(f"\nSummary: {n_good}/{len(results)} GOOD   {n_hard}/{len(results)} HARD")

    with open(os.path.join(out_dir, "summary.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    _write_html(out_dir, results, n_good, n_hard)

    model_volume.commit()
    print(f"\nSaved → /model-cache/seed_analysis/")
    print(f"Download: modal volume get rl-harness-model-cache seed_analysis/ ./artifacts/seed_analysis/")
    print(f"View:     open seed_analysis/report.html")

    return {"n_good": n_good, "n_hard": n_hard, "n_total": len(results)}


def _write_html(out_dir, results, n_good, n_hard):
    import json, os

    n_total = len(results)

    # Build table rows
    rows_html = ""
    for i, r in enumerate(results):
        cls        = "good" if r["label"] == "GOOD" else "hard"
        phases_str = " → ".join(r["phases"]) if r["phases"] else "-"
        rows_html += (
            f'<tr class="{cls}" data-idx="{i}" data-label="{r["label"]}" onclick="selectRow(this)">'
            f'<td>{r["seed"]}</td>'
            f'<td><b>{r["label"]}</b></td>'
            f'<td>{r["d_grip_block"]:.3f}</td>'
            f'<td>{r["d_block_target"]:.3f}</td>'
            f'<td>{r["n_steps"]}</td>'
            f'<td>{phases_str}</td>'
            f'<td>{r["block_z"]:.3f}</td>'
            f'<td>{r["target_z"]:.3f}</td>'
            f'</tr>\n'
        )

    # Slim JSON for the JS data array (only what the info panel needs)
    js_data = json.dumps([{
        "seed"          : r["seed"],
        "label"         : r["label"],
        "d_grip_block"  : r["d_grip_block"],
        "d_block_target": r["d_block_target"],
        "phases"        : r["phases"],
        "n_steps"       : r["n_steps"],
        "block_z"       : r["block_z"],
        "target_z"      : r["target_z"],
        "video"         : r["video"],
    } for r in results])

    # The HTML uses __PLACEHOLDER__ tokens to avoid f-string brace escaping
    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Seed Analysis</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: monospace; font-size: 13px; background: #111; color: #ddd; }

  #top {
    position: sticky; top: 0; z-index: 10;
    display: flex; gap: 16px; padding: 12px 16px;
    background: #111; border-bottom: 2px solid #333;
  }
  #player { width: 320px; height: 320px; background: #000; object-fit: contain; }
  #info { flex: 1; display: flex; flex-direction: column; gap: 8px; }
  #sel-info { font-size: 13px; color: #aef; line-height: 1.5; }
  #hint { font-size: 11px; color: #666; }

  .filters { display: flex; gap: 6px; }
  button {
    background: #222; color: #ccc; border: 1px solid #444;
    padding: 4px 12px; cursor: pointer; font-size: 12px; font-family: monospace;
    border-radius: 3px;
  }
  button:hover { background: #333; }
  button.active { background: #2a6; color: #000; border-color: #2a6; }

  #summary { font-size: 12px; color: #888; padding: 8px 16px; }

  table { width: 100%; border-collapse: collapse; }
  thead th {
    position: sticky; top: 0; background: #222; color: #aaa;
    padding: 6px 10px; text-align: left; cursor: pointer;
    white-space: nowrap; border-bottom: 1px solid #444;
    user-select: none;
  }
  thead th:hover { background: #333; color: #fff; }
  td { padding: 5px 10px; border-bottom: 1px solid #1e1e1e; white-space: nowrap; }
  tr.good td { background: #0d1f0d; }
  tr.hard td { background: #1f0d0d; }
  tr.selected td { background: #1a3a1a !important; outline: 2px solid #2a6; outline-offset: -2px; }
  tr:hover td { filter: brightness(1.4); cursor: pointer; }
</style>
</head>
<body>

<div id="top">
  <video id="player" controls autoplay loop muted></video>
  <div id="info">
    <div class="filters">
      <button id="btn-all"  class="active" onclick="setFilter('ALL')">All (__N_TOTAL__)</button>
      <button id="btn-good"              onclick="setFilter('GOOD')">GOOD (__N_GOOD__)</button>
      <button id="btn-hard"              onclick="setFilter('HARD')">HARD (__N_HARD__)</button>
    </div>
    <div id="sel-info">Click a row to load its video.</div>
    <div id="hint">← → arrow keys · J/K to step through · Enter to play/pause</div>
  </div>
</div>

<div id="summary">
  __N_TOTAL__ seeds &nbsp;·&nbsp; <span style="color:#4a9">__N_GOOD__ GOOD</span>
  (oracle success) &nbsp;·&nbsp; <span style="color:#c55">__N_HARD__ HARD</span> (oracle fail)
</div>

<table id="tbl">
  <thead>
    <tr>
      <th onclick="sortTable(0)">Seed ↕</th>
      <th onclick="sortTable(1)">Label</th>
      <th onclick="sortTable(2)">d grip→block</th>
      <th onclick="sortTable(3)">d block→target</th>
      <th onclick="sortTable(4)">Steps</th>
      <th>Phases</th>
      <th onclick="sortTable(6)">Block Z</th>
      <th onclick="sortTable(7)">Target Z</th>
    </tr>
  </thead>
  <tbody id="tbody">
__ROWS__
  </tbody>
</table>

<script>
const DATA = __JS_DATA__;
let selectedIdx = -1;
let visibleRows = [];

function allRows() {
  return Array.from(document.querySelectorAll('#tbody tr[data-idx]'));
}

function setFilter(f) {
  ['all','good','hard'].forEach(x => {
    const active = (f === 'ALL' && x === 'all') || f === x.toUpperCase();
    document.getElementById('btn-' + x).classList.toggle('active', active);
  });
  allRows().forEach(tr => {
    tr.style.display = (f === 'ALL' || tr.dataset.label === f) ? '' : 'none';
  });
  visibleRows = allRows().filter(tr => tr.style.display !== 'none');
  selectedIdx = -1;
}

function selectRow(tr) {
  allRows().forEach(r => r.classList.remove('selected'));
  tr.classList.add('selected');
  const d = DATA[parseInt(tr.dataset.idx)];
  const player = document.getElementById('player');
  player.src = d.video;
  player.play();
  document.getElementById('sel-info').innerHTML =
    '<b>Seed ' + d.seed + '</b> [' + d.label + '] &nbsp;·&nbsp; ' +
    'd_grip=' + d.d_grip_block + ' &nbsp;·&nbsp; ' +
    'd_tgt=' + d.d_block_target + ' &nbsp;·&nbsp; ' +
    'block_z=' + d.block_z + ' &nbsp;·&nbsp; ' +
    'target_z=' + d.target_z + '<br>' +
    d.phases.join(' → ') + ' &nbsp;·&nbsp; ' + d.n_steps + ' steps';
  tr.scrollIntoView({ block: 'nearest' });
  visibleRows = allRows().filter(r => r.style.display !== 'none');
  selectedIdx = visibleRows.indexOf(tr);
}

document.addEventListener('keydown', e => {
  visibleRows = allRows().filter(r => r.style.display !== 'none');
  if (e.key === 'ArrowRight' || e.key === 'l' || e.key === 'j') {
    e.preventDefault();
    selectedIdx = Math.min(selectedIdx + 1, visibleRows.length - 1);
    if (visibleRows[selectedIdx]) selectRow(visibleRows[selectedIdx]);
  } else if (e.key === 'ArrowLeft' || e.key === 'h' || e.key === 'k') {
    e.preventDefault();
    selectedIdx = Math.max(selectedIdx - 1, 0);
    if (visibleRows[selectedIdx]) selectRow(visibleRows[selectedIdx]);
  } else if (e.key === 'Enter') {
    const p = document.getElementById('player');
    p.paused ? p.play() : p.pause();
  }
});

setFilter('ALL');
if (visibleRows.length > 0) selectRow(visibleRows[0]);
</script>
</body>
</html>"""

    html = (html
        .replace("__N_TOTAL__", str(n_total))
        .replace("__N_GOOD__",  str(n_good))
        .replace("__N_HARD__",  str(n_hard))
        .replace("__ROWS__",    rows_html)
        .replace("__JS_DATA__", js_data))

    with open(os.path.join(out_dir, "report.html"), "w") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(n_seeds: int = 100):
    print(f"\nAnalyzing {n_seeds} seeds (no GPU, oracle only)...")
    result = analyze_seeds.remote(n_seeds=n_seeds)
    print(f"\nDone: {result['n_good']}/{result['n_total']} GOOD  "
          f"{result['n_hard']}/{result['n_total']} HARD")
    print(f"\nDownload:")
    print(f"  modal volume get rl-harness-model-cache seed_analysis/ ./artifacts/seed_analysis/")
    print(f"View:")
    print(f"  open seed_analysis/report.html")
