#!/usr/bin/env python3
"""
export_align_xy_policy_numpy.py

Dumps low_level_align_xy_ppo_best.pt's weights to a plain .npz so the real
UR3e rollout script (run_align_xy_real.py, on the remote Construct ROS2
session) can do a pure-numpy forward pass -- no torch install needed there.
The policy is tiny (LayerNorm + 2x Linear/Tanh + a linear mean_head), so
re-implementing forward() in numpy is a handful of lines, see
run_align_xy_real.py for the matching consumer code.

Run locally (where torch + the checkpoint both already exist):
    python3 scripts/export_align_xy_policy_numpy.py
Then copy the resulting artifacts/align_xy_policy_weights.npz to the
Construct session alongside run_align_xy_real.py.
"""
import numpy as np
import torch

CKPT_PATH = "artifacts/checkpoints/low_level_align_xy_ppo_best.pt"
OUT_PATH = "artifacts/align_xy_policy_weights.npz"


def main():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    sd = ckpt["actor"] if isinstance(ckpt, dict) and "actor" in ckpt else ckpt

    weights = {k: v.numpy() for k, v in sd.items()}
    np.savez(OUT_PATH, **weights)
    print(f"Wrote {OUT_PATH}:")
    for k, v in weights.items():
        print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    main()
