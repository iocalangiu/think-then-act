"""
eval_subgoal_vlm.py

Held-out accuracy check for the high-level VLM
(policy.subgoal_vlm_policy.SubgoalVLMPolicy) — how often does
<action>{predicted}</action> match the true label, broken down per
SUBGOAL_LABELS, over the SAME held-out episodes subgoal_sft_train.py never
trained on. Complements watching a single frozen training-set example
during training's quick_eval (see hierarchical_architecture memory,
2026-07-17): that one example can lock onto a stable wrong answer for a
specific hard case while the model genuinely improves everywhere else — the
aggregate val_loss said as much, but val_loss alone doesn't say WHICH
subgoals are confused with which.

Reads : /model-cache/subgoal_sft_data.jsonl (same file training reads)
        /model-cache/checkpoints/subgoal_sft_warmstart (best-val-loss LoRA,
          the same pointer training maintains — always the CURRENT best,
          not a fixed epoch)
Saves : /model-cache/eval/subgoal_vlm_eval.json (per-example predictions)
        /model-cache/eval/subgoal_vlm_eval_summary.txt (confusion matrix)

Run with:
    modal run --detach scripts/eval_subgoal_vlm.py
    modal run --detach scripts/eval_subgoal_vlm.py --max-examples 50   # quick subset
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


def _load_val_examples(data_path: str, val_frac: float, max_examples: int) -> list:
    """
    Exactly reproduces subgoal_sft_train.py's episode-level val split
    (same random.Random(0), same shuffle-then-take-first-n_val_eps logic) —
    otherwise "held-out" wouldn't actually mean held out from what this
    checkpoint trained on.
    """
    import json, random

    examples = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line))

    episode_ids = sorted({ex["episode"] for ex in examples if "episode" in ex})
    rng = random.Random(0)
    rng.shuffle(episode_ids)
    n_val_eps = max(1, int(len(episode_ids) * val_frac))
    val_ep_ids = set(episode_ids[:n_val_eps])
    val_examples = [ex for ex in examples if ex["episode"] in val_ep_ids]

    if max_examples > 0:
        val_examples = val_examples[:max_examples]
    return val_examples


@app.function(
    image=rl_image,
    gpu="A10G",
    volumes={MODEL_CACHE_DIR: model_volume},
    # Bumped 3600->14400 (2026-07-22): a full 685-example run was killed by
    # the OLD 3600s timeout at 300/685 (~44%) -- observed rate is ~12s/
    # example, so the full set needs ~8200s. 14400 leaves real margin above
    # that instead of just barely covering the observed rate.
    timeout=14400,
)
def eval_subgoal_vlm(
    val_frac: float = 0.1,
    max_examples: int = 0,   # 0 = full held-out set
    # Every SUBGOAL_LABELS template now shows full worked arithmetic
    # (2026-07-17 fix) — the target text alone runs ~140 tokens for the
    # longest template (close_gripper), and an early-training model prone
    # to rambling/repetition (seen directly in training logs) needs real
    # headroom above that, not the ~60 tokens 200 left. A tight cap here
    # silently shows up as a PARSE_FAIL (no closing </action> tag reached),
    # not as a wrong-but-parseable answer — confirmed 2026-07-17: a 200-cap
    # eval run hit 98% parse failures on an epoch-2 checkpoint.
    max_new_tokens: int = 400,
) -> dict:
    import os, json, base64, io
    from collections import Counter

    os.environ["HF_HOME"]            = MODEL_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_DIR

    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from think_then_act.policy.subgoal_vlm_policy import (
        SubgoalVLMPolicy, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
    )
    from think_then_act.policy.model_loader import MODEL_ID, load_base_model, load_lora_checkpoint
    from think_then_act.reward.subgoal_reward import SUBGOAL_LABELS, DEFAULT_WEIGHTS

    print("\n" + "=" * 60)
    print("  HIGH-LEVEL VLM — HELD-OUT ACCURACY EVAL")
    print("=" * 60)

    data_path = os.path.join(MODEL_CACHE_DIR, "subgoal_sft_data.jsonl")
    val_examples = _load_val_examples(data_path, val_frac, max_examples)
    print(f"\n[1/3] {len(val_examples)} held-out examples "
          f"(val_frac={val_frac}, same split subgoal_sft_train.py used)")

    ckpt_dir = os.path.join(MODEL_CACHE_DIR, "checkpoints", "subgoal_sft_warmstart")
    print(f"\n[2/3] Loading model + LoRA <- {ckpt_dir}...")
    base_model, processor = load_base_model(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    model = load_lora_checkpoint(base_model, ckpt_dir, trainable=False)
    model.eval()

    def predict(ex: dict) -> tuple:
        """Returns (predicted_subgoal_or_None, think_text, raw_response)."""
        pil_image = Image.open(
            io.BytesIO(base64.b64decode(ex["frame_b64"]))
        ).resize((224, 224), Image.LANCZOS)   # MUST match training's resize exactly

        gripper_pos = [round(v, 4) for v in ex["gripper_pos"]]
        achieved    = [round(v, 4) for v in ex["achieved_goal"]]
        desired     = [round(v, 4) for v in ex["desired_goal"]]
        user_text   = USER_PROMPT_TEMPLATE.format(
            gripper_pos=gripper_pos, achieved_goal=achieved, desired_goal=desired,
            is_grasped="yes" if ex["is_grasped"] else "no",
            table_height=DEFAULT_WEIGHTS.table_z,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": pil_image},
                {"type": "text",  "text": user_text},
            ]},
        ]
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        img_inputs, vid_inputs = process_vision_info(messages)
        enc = processor(
            text=[text_input], images=img_inputs, videos=vid_inputs,
            return_tensors="pt", padding=True,
        ).to("cuda")

        with torch.no_grad():
            # stop_strings=["</action>"]: cuts generation short the moment the
            # action tag closes instead of always running to max_new_tokens —
            # same trick subgoal_sft_train.py's quick_eval uses, matters a lot
            # here since this runs over hundreds of examples, not one.
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                stop_strings=["</action>"], tokenizer=processor.tokenizer,
            )
        new_tokens = out[0][enc["input_ids"].shape[1]:]
        text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

        think, _ = SubgoalVLMPolicy._extract_think(text)
        subgoal, _ = SubgoalVLMPolicy._parse_action(text)
        return subgoal, think, text

    def _compute_and_save(results: list, confusion: Counter, n_correct: int,
                           n_parse_fail: int, n_total: int, partial: bool) -> dict:
        """
        Shared by the periodic mid-loop save (so a timeout/preemption like
        the one that killed the first 3600s run doesn't lose the whole
        hour of GPU work again -- only whatever ran since the LAST periodic
        save) and the final save. `partial`/`n_examples_so_far` are stamped
        into the output so a reader can tell an interrupted run's numbers
        from a completed one's.
        """
        overall_acc = n_correct / n_total if n_total else float("nan")
        parse_fail_rate = n_parse_fail / n_total if n_total else float("nan")

        per_label_acc = {}
        for label in SUBGOAL_LABELS:
            label_results = [r for r in results if r["true_subgoal"] == label]
            if label_results:
                per_label_acc[label] = sum(r["correct"] for r in label_results) / len(label_results)
            else:
                per_label_acc[label] = None

        out_dir = os.path.join(MODEL_CACHE_DIR, "eval")
        os.makedirs(out_dir, exist_ok=True)
        results_path = os.path.join(out_dir, "subgoal_vlm_eval.json")
        summary_path = os.path.join(out_dir, "subgoal_vlm_eval_summary.txt")

        all_pred_labels = sorted({k[1] for k in confusion})
        true_pred_label = "true \\ pred"   # kept out of the f-string below -- Modal's
                                            # container runs Python 3.11, which (unlike
                                            # 3.12) disallows a backslash inside an
                                            # f-string expression part.
        header = f"    {true_pred_label:16s}" + "".join(f"{p:>16s}" for p in all_pred_labels)

        with open(results_path, "w") as f:
            json.dump({
                "checkpoint": ckpt_dir,
                "partial": partial,
                "n_examples_so_far": n_total,
                "n_examples_total": len(val_examples),
                "overall_accuracy": round(overall_acc, 4),
                "parse_fail_rate": round(parse_fail_rate, 4),
                "per_label_accuracy": {k: (round(v, 4) if v is not None else None)
                                        for k, v in per_label_acc.items()},
                "confusion_matrix": {f"{t}->{p}": c for (t, p), c in confusion.items()},
                "n_examples": n_total,
                "results": results,
            }, f, indent=2)

        with open(summary_path, "w") as f:
            f.write(f"checkpoint: {ckpt_dir}\n")
            if partial:
                f.write(f"PARTIAL RESULTS -- {n_total}/{len(val_examples)} examples "
                        f"(run still in progress or was interrupted)\n")
            f.write(f"overall_accuracy: {overall_acc:.3f} ({n_correct}/{n_total})\n")
            f.write(f"parse_fail_rate: {parse_fail_rate:.3f} ({n_parse_fail}/{n_total})\n\n")
            f.write("per-label accuracy:\n")
            for label in SUBGOAL_LABELS:
                acc = per_label_acc[label]
                n = sum(1 for r in results if r["true_subgoal"] == label)
                f.write(f"  {label:16s}  {('%.3f' % acc) if acc is not None else 'n/a':>6s}  (n={n})\n")
            f.write("\nconfusion matrix (true -> predicted, count):\n")
            f.write(header + "\n")
            for true_label in SUBGOAL_LABELS:
                row = f"    {true_label:16s}"
                for p in all_pred_labels:
                    row += f"{confusion.get((true_label, p), 0):>16d}"
                f.write(row + "\n")
            f.write("\nmisclassified examples:\n")
            for r in results:
                if not r["correct"]:
                    f.write(f"  ep={r['episode']} source={r['source']} "
                            f"true={r['true_subgoal']} pred={r['predicted_subgoal']}\n")
                    f.write(f"    think: {r['think'][:300]}\n")

        model_volume.commit()
        return {
            "overall_accuracy"  : round(overall_acc, 4),
            "parse_fail_rate"   : round(parse_fail_rate, 4),
            "per_label_accuracy": per_label_acc,
            "n_examples"        : n_total,
            "results_path"      : results_path,
            "summary_path"      : summary_path,
        }

    print(f"\n[3/3] Running inference on {len(val_examples)} examples...")
    results = []
    confusion = Counter()   # (true_label, pred_label_or_'PARSE_FAIL') -> count
    n_correct = 0
    n_parse_fail = 0

    for i, ex in enumerate(val_examples):
        pred, think, raw = predict(ex)
        true = ex["subgoal"]
        correct = (pred == true)
        n_correct += int(correct)
        pred_key = pred if pred is not None else "PARSE_FAIL"
        if pred is None:
            n_parse_fail += 1
        confusion[(true, pred_key)] += 1

        results.append({
            "episode": ex["episode"], "source": ex.get("source"),
            "true_subgoal": true, "predicted_subgoal": pred, "correct": correct,
            "think": think, "raw_response": raw,
        })

        if (i + 1) % 25 == 0:
            running_acc = n_correct / (i + 1)
            print(f"  {i+1}/{len(val_examples)}  running_accuracy={running_acc:.3f}")

        # Periodic partial save -- so a timeout/preemption (like the one
        # that killed a full run at 300/685 under the old 3600s cap) loses
        # at most 100 examples' worth of inference, not the whole run.
        if (i + 1) % 100 == 0:
            _compute_and_save(results, confusion, n_correct, n_parse_fail, i + 1, partial=True)
            print(f"  [partial save] {i+1}/{len(val_examples)} -> subgoal_vlm_eval.json")

    final = _compute_and_save(results, confusion, n_correct, n_parse_fail,
                               len(val_examples), partial=False)

    # ------------------------------------------------------------------
    # Summary (console only -- the files themselves are already written)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  OVERALL ACCURACY: {final['overall_accuracy']:.3f}  ({n_correct}/{len(val_examples)})")
    print(f"  PARSE FAILURES  : {final['parse_fail_rate']:.3f}  ({n_parse_fail}/{len(val_examples)})")
    print(f"\n  Per-label accuracy (of examples with this TRUE label):")
    for label in SUBGOAL_LABELS:
        acc = final["per_label_accuracy"][label]
        n = sum(1 for r in results if r["true_subgoal"] == label)
        print(f"    {label:16s}  {('%.3f' % acc) if acc is not None else 'n/a':>6s}  (n={n})")

    print(f"\n  Confusion matrix (true -> predicted, count):")
    all_pred_labels = sorted({k[1] for k in confusion})
    true_pred_label = "true \\ pred"   # kept out of the f-string below -- Modal's
                                        # container runs Python 3.11, which (unlike
                                        # 3.12) disallows a backslash inside an
                                        # f-string expression part.
    header = f"    {true_pred_label:16s}" + "".join(f"{p:>16s}" for p in all_pred_labels)
    print(header)
    for true_label in SUBGOAL_LABELS:
        row = f"    {true_label:16s}"
        for p in all_pred_labels:
            row += f"{confusion.get((true_label, p), 0):>16d}"
        print(row)
    print("=" * 60)
    print(f"\n  Saved -> {final['results_path']}")
    print(f"  Saved -> {final['summary_path']}")

    return final


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(max_examples: int = 0):
    # .spawn(), not .remote() -- .remote() blocks on the CLI's connection
    # to receive the return value, so under `modal run --detach` (meant to
    # let the job outlive the CLI disconnecting) the still-in-flight call
    # gets cancelled the moment the CLI exits after dispatch. .spawn() is
    # fire-and-forget and survives that, same as subgoal_sft_train.py's
    # local entrypoint already does.
    handle = eval_subgoal_vlm.spawn(max_examples=max_examples)
    print(f"\nJob spawned. Function call ID: {handle.object_id}")
    print(f"Monitor at https://modal.com")
    print(f"\nDownload when finished:")
    print(f"  modal volume get rl-harness-model-cache eval/subgoal_vlm_eval_summary.txt ./artifacts/")
    print(f"  modal volume get rl-harness-model-cache eval/subgoal_vlm_eval.json ./artifacts/")
