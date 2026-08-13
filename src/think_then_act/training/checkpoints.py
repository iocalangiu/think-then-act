"""
think_then_act.training.checkpoints

Shared low-level-subgoal checkpoint resolution, used by any script that
loads a trained SubgoalGaussianPolicy off the model volume:
scripts/record_subgoal_video.py (per-subgoal before/after clips) and
scripts/record_full_rollout.py (chained align_xy->...->release demo). Kept
in one place so the "best checkpoint can be older than the latest, since
training can regress after peaking" resolution order (observed with GRPO —
see bugs_and_fixes memory) isn't re-derived per script.
"""

from __future__ import annotations

import glob
import os
import re


def resolve_subgoal_checkpoint(ckpt_dir: str, subgoal: str, algo: str = "ppo",
                                use_best: bool = False) -> str:
    """
    Resolution order: best (if use_best) -> final -> highest available
    _iter{N}.pt. Raises FileNotFoundError with the checkpoint dir's actual
    contents listed if nothing matches, rather than silently falling back
    to an untrained policy.
    """
    if algo not in ("grpo", "ppo"):
        raise ValueError(f"algo must be 'grpo' or 'ppo', got {algo!r}")
    suffix = "_ppo" if algo == "ppo" else ""   # PPO checkpoints (train_low_level_ppo.py)
                                                # use a _ppo suffix so they never collide
                                                # with GRPO's on the same volume.

    if use_best:
        best_ckpt = os.path.join(ckpt_dir, f"low_level_{subgoal}{suffix}_best.pt")
        if not os.path.exists(best_ckpt):
            raise FileNotFoundError(f"No best checkpoint at {best_ckpt}")
        return best_ckpt

    final_ckpt = os.path.join(ckpt_dir, f"low_level_{subgoal}{suffix}.pt")
    if os.path.exists(final_ckpt):
        return final_ckpt

    candidates = glob.glob(os.path.join(ckpt_dir, f"low_level_{subgoal}{suffix}_iter*.pt"))
    if not candidates:
        available = sorted(os.listdir(ckpt_dir)) if os.path.isdir(ckpt_dir) else []
        raise FileNotFoundError(
            f"No checkpoint found for subgoal={subgoal!r} algo={algo!r} in {ckpt_dir}. "
            f"Currently in {ckpt_dir}: {available}"
        )

    def iter_num(p):
        return int(re.search(r"_iter(\d+)\.pt$", p).group(1))
    return max(candidates, key=iter_num)
