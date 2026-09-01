#!/usr/bin/env python3
"""
export_subgoal_policy_numpy.py

Dumps a trained low-level subgoal policy's weights to a plain .npz so a
real-UR3e rollout script (run_subgoal_real.py, on the remote Construct
ROS2 session) can do a pure-numpy forward pass -- no torch install needed
there. Generalized from export_align_xy_policy_numpy.py (2026-09-01) once
descend also moved to the same RELATIVE_OBS_SUBGOALS layout -- both
checkpoints are the same tiny architecture (LayerNorm + 2x Linear/Tanh +
mean_head), just different obs_dim, so one script now covers both.

Run locally (where torch + the checkpoint both already exist):
    python3 scripts/export_subgoal_policy_numpy.py align_xy
    python3 scripts/export_subgoal_policy_numpy.py descend
Then copy the resulting artifacts/{subgoal}_policy_weights.npz to the
Construct session alongside run_subgoal_real.py.
"""
import sys

import numpy as np
import torch

CKPT_TEMPLATE = "artifacts/checkpoints/low_level_{subgoal}_ppo_best.pt"
OUT_TEMPLATE = "artifacts/{subgoal}_policy_weights.npz"


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("align_xy", "descend"):
        print("Usage: python3 scripts/export_subgoal_policy_numpy.py <align_xy|descend>")
        sys.exit(1)
    subgoal = sys.argv[1]

    ckpt_path = CKPT_TEMPLATE.format(subgoal=subgoal)
    out_path = OUT_TEMPLATE.format(subgoal=subgoal)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["actor"] if isinstance(ckpt, dict) and "actor" in ckpt else ckpt

    weights = {k: v.numpy() for k, v in sd.items()}
    np.savez(out_path, **weights)
    print(f"Wrote {out_path} (from {ckpt_path}):")
    for k, v in weights.items():
        print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    main()
