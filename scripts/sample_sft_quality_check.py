"""
sample_sft_quality_check.py

Pulls a stratified sample (a few rows per SUBGOAL_LABELS) of
subgoal_sft_data.jsonl and packages each one as a (frame, exact prompt
text, think, action) unit — for actually EYEBALLING whether the frame
matches its claimed state, whether the reasoning is sensible given what's
visible, and whether the label is right. Complements
analyze_position_variability.py (numeric spread) and
eval_subgoal_vlm.py (held-out accuracy) — neither of those looks at
whether an individual (frame, label) PAIR is actually correct.

The "prompt text" saved here is built from the EXACT SAME
policy.subgoal_vlm_policy.USER_PROMPT_TEMPLATE + rounding convention
subgoal_sft_train.py's compute_loss uses — not a hand-rederived
approximation — so what you're checking is byte-identical to what the
model actually sees.

Reads : /model-cache/subgoal_sft_data.jsonl (already generated)
Saves : /model-cache/eval/sft_quality_sample.json — the sampled rows only
          (frame_b64 + prompt fields + think + subgoal), small regardless
          of how big the full dataset is (this project's dataset ranges
          from a 116-row debug run to a 3244-row full run, but the sample
          stays n_per_label * 6 rows either way)
        /model-cache/eval/sft_quality_sample_frames/{subgoal}_{i}.png
          — decoded images, for browsing directly instead of via the JSON
        /model-cache/eval/sft_quality_sample.txt — plain-text pairing of
          each image filename with its exact prompt + think + subgoal

Run with:
    modal run --detach scripts/sample_sft_quality_check.py
    modal run --detach scripts/sample_sft_quality_check.py --n-per-label 8
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(
    image=rl_image,
    gpu=None,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=600,
)
def sample_sft_quality_check(n_per_label: int = 5, seed: int = 0) -> dict:
    import os, json, random, base64, io
    from PIL import Image
    from think_then_act.policy.subgoal_vlm_policy import USER_PROMPT_TEMPLATE
    from think_then_act.reward.subgoal_reward import SUBGOAL_LABELS

    data_path = os.path.join(MODEL_CACHE_DIR, "subgoal_sft_data.jsonl")
    print(f"\nReading {data_path}...")

    by_label = {label: [] for label in SUBGOAL_LABELS}
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line)
            if ex["subgoal"] in by_label:
                by_label[ex["subgoal"]].append(ex)
    print("  rows per label: " + ", ".join(f"{k}={len(v)}" for k, v in by_label.items()))

    rng = random.Random(seed)
    sampled = []
    for label, rows in by_label.items():
        k = min(n_per_label, len(rows))
        if k < n_per_label:
            print(f"  [warn] only {k}/{n_per_label} available for {label!r}")
        sampled.extend(rng.sample(rows, k))
    print(f"  sampled {len(sampled)} rows total")

    out_dir = os.path.join(MODEL_CACHE_DIR, "eval")
    frames_dir = os.path.join(out_dir, "sft_quality_sample_frames")
    os.makedirs(frames_dir, exist_ok=True)

    entries = []
    for i, ex in enumerate(sampled):
        gripper_pos = [round(v, 4) for v in ex["gripper_pos"]]
        achieved    = [round(v, 4) for v in ex["achieved_goal"]]
        desired     = [round(v, 4) for v in ex["desired_goal"]]
        # Exact prompt text the model is actually trained/queried on.
        prompt = USER_PROMPT_TEMPLATE.format(
            gripper_pos=gripper_pos, achieved_goal=achieved, desired_goal=desired,
            is_grasped="yes" if ex["is_grasped"] else "no",
        )

        frame_filename = f"{ex['subgoal']}_{i:03d}.png"
        frame_path = os.path.join(frames_dir, frame_filename)
        Image.open(io.BytesIO(base64.b64decode(ex["frame_b64"]))).save(frame_path)

        entries.append({
            "frame_filename": frame_filename,
            "frame_b64"     : ex["frame_b64"],   # kept for the HTML gallery build
            "prompt"        : prompt,
            "think"         : ex["think"],
            "subgoal"       : ex["subgoal"],
            "source"        : ex.get("source"),
            "episode"       : ex.get("episode"),
        })

    sample_json_path = os.path.join(out_dir, "sft_quality_sample.json")
    with open(sample_json_path, "w") as f:
        json.dump(entries, f)

    txt_path = os.path.join(out_dir, "sft_quality_sample.txt")
    with open(txt_path, "w") as f:
        for e in entries:
            f.write(f"=== {e['frame_filename']}  (source={e['source']}  episode={e['episode']}) ===\n")
            f.write(f"{e['prompt']}\n\n")
            f.write(f"think : {e['think']}\n")
            f.write(f"action: {e['subgoal']}\n\n")

    # Zipped, not just a raw directory -- `modal volume get` on a directory
    # has repeatedly (2026-07-17, twice: this script's own frames dir and
    # record_subgoal_demo.py's) silently flattened it into a single file
    # client-side instead of recreating the directory. A single zip file
    # sidesteps that CLI quirk entirely: it downloads exactly the same way
    # every other single-file artifact in this project already does. The
    # raw PNGs are ALSO kept in frames_dir (not replaced) for anyone who'd
    # rather `modal volume ls`/pull individual images directly.
    import shutil
    zip_path = shutil.make_archive(frames_dir, "zip", frames_dir)

    model_volume.commit()
    print(f"\n  Saved -> {sample_json_path}")
    print(f"  Saved -> {txt_path}")
    print(f"  Saved -> {zip_path}")
    print(f"  Saved -> {frames_dir}/ ({len(entries)} images)")

    return {
        "n_sampled"       : len(entries),
        "labels_covered"  : {label: sum(1 for e in entries if e["subgoal"] == label) for label in SUBGOAL_LABELS},
        "sample_json_path": sample_json_path,
        "txt_path"        : txt_path,
        "frames_dir"      : frames_dir,
        "zip_path"        : zip_path,
    }


@app.local_entrypoint()
def main(n_per_label: int = 5, seed: int = 0):
    handle = sample_sft_quality_check.spawn(n_per_label=n_per_label, seed=seed)
    print(f"\nJob spawned. Function call ID: {handle.object_id}")
    print(f"Monitor at https://modal.com")
    print(f"\nDownload when finished:")
    print(f"  modal volume get rl-harness-model-cache eval/sft_quality_sample.json ./artifacts/")
    print(f"  modal volume get rl-harness-model-cache eval/sft_quality_sample.txt ./artifacts/")
    print(f"  modal volume get rl-harness-model-cache eval/sft_quality_sample_frames.zip ./artifacts/")
    print(f"  (then: unzip -o ./artifacts/sft_quality_sample_frames.zip -d ./artifacts/sft_quality_sample_frames/)")
