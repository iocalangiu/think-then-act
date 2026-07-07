"""
compare_runs.py

Compares two grpo_m6c_metrics*.jsonl logs side by side — e.g. the killed
run (buggy block_pos reward, 25/20-step episodes) vs the current run
(fixed reward.py, 45-step episodes).

mean_reward/mean_return are NOT directly comparable across runs with
different max_episode_steps (a longer episode accumulates more per-step
penalty even if the policy is identical) — this script normalizes those
to per-step reward for a fair comparison. parse_rate/success_rate are
already normalized (0-1 outcome), no adjustment needed, though longer
episodes do give more chances to succeed, which isn't controlled for.

Usage:
    python3 compare_runs.py grpo_m6c_metrics.jsonl grpo_m6c_metrics_v2.jsonl
"""
import json
import sys
from collections import defaultdict


def load(path: str) -> dict:
    by_type = defaultdict(dict)  # type -> {iteration: record}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_type[r["type"]][r["iteration"]] = r
    return by_type


def per_step_reward(r: dict, max_steps: int) -> float:
    key = "mean_reward" if "mean_reward" in r else "mean_return"
    return r[key] / max_steps


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python3 {sys.argv[0]} <old.jsonl> <new.jsonl>")
        sys.exit(1)

    old_path, new_path = sys.argv[1], sys.argv[2]
    old, new = load(old_path), load(new_path)

    # Old run: max_episode_steps=25 (train), 20 (eval/baseline/canary).
    # New run: 45 for everything.
    OLD_TRAIN_STEPS, OLD_EVAL_STEPS, NEW_STEPS = 25, 20, 45

    print(f"\n{'='*78}\nBASELINE\n{'='*78}")
    for label, log, steps in [("OLD", old, OLD_EVAL_STEPS), ("NEW", new, NEW_STEPS)]:
        b = log["baseline"].get(0)
        if b:
            print(f"  {label}: success={b['success_rate']:.0%}  "
                  f"return={b['mean_return']:.2f}  "
                  f"return/step={per_step_reward(b, steps):.3f}  "
                  f"parse={b['action_parse_rate']:.1%}")
        else:
            print(f"  {label}: (no baseline yet)")

    print(f"\n{'='*78}\nTRAIN ITERATIONS  (reward normalized to per-step)\n{'='*78}")
    print(f"  {'iter':>4}  {'old rwd/step':>13}  {'new rwd/step':>13}  "
          f"{'old parse':>10}  {'new parse':>10}  "
          f"{'old within_std':>15}  {'new within_std':>15}")
    all_iters = sorted(set(old["train"].keys()) | set(new["train"].keys()))
    for i in all_iters:
        o = old["train"].get(i)
        n = new["train"].get(i)
        o_rwd = f"{per_step_reward(o, OLD_TRAIN_STEPS):.3f}" if o else "  -  "
        n_rwd = f"{per_step_reward(n, NEW_STEPS):.3f}" if n else "  -  "
        o_parse = f"{o['parse_rate']:.0%}" if o else "  -  "
        n_parse = f"{n['parse_rate']:.0%}" if n else "  -  "
        o_std = f"{o['avg_within_std']:.3f}" if o else "  -  "
        n_std = f"{n['avg_within_std']:.3f}" if n else "  -  "
        print(f"  {i:>4}  {o_rwd:>13}  {n_rwd:>13}  {o_parse:>10}  {n_parse:>10}  "
              f"{o_std:>15}  {n_std:>15}")

    for kind, steps_old, steps_new in [("canary", OLD_EVAL_STEPS, NEW_STEPS),
                                        ("eval", OLD_EVAL_STEPS, NEW_STEPS)]:
        if not old[kind] and not new[kind]:
            continue
        print(f"\n{'='*78}\n{kind.upper()}\n{'='*78}")
        all_iters = sorted(set(old[kind].keys()) | set(new[kind].keys()))
        for i in all_iters:
            o = old[kind].get(i)
            n = new[kind].get(i)
            line = f"  iter {i:>3}:"
            if o:
                line += (f"  OLD success={o['success_rate']:.0%} "
                         f"return/step={per_step_reward(o, steps_old):.3f} "
                         f"parse={o['action_parse_rate']:.0%}")
            if n:
                line += (f"   |  NEW success={n['success_rate']:.0%} "
                         f"return/step={per_step_reward(n, steps_new):.3f} "
                         f"parse={n['action_parse_rate']:.0%}")
            print(line)
    print()


if __name__ == "__main__":
    main()
