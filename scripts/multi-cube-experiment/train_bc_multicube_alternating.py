"""
train_bc_multicube_alternating.py

Fresh BC training (not a fine-tune of an existing checkpoint) on a
COMBINED pool: demonstrations/sb3_teacher_full_task_joint_randomized_v2.pkl
(2000 genuine single-cube demos, the existing "basic skill" pool) +
demonstrations/multicube_genuine_demos.pkl (300 genuine, fully autonomous
2-3-cube stacking traces from collect_multicube_genuine_demos.py -- no
scripted segments, collected directly from the pre-degradation PPO
checkpoint's own successful rollouts). This is the "alternate basic skill
with 2 and 3 cubes" approach -- the user's own framing -- built as a
direct alternative to continued PPO after the lr=3e-4 continuation's
real, video-confirmed regression and the earlier 10-demo scripted BC
fine-tune's own (smaller-scale) regression. Two things are different
this time vs. that failed fine-tune: scale (300 genuine multi-cube demos,
not 10) and no scripted segments at all (every multi-cube demo is the
policy's own successful behavior, removing the main suspect for why the
GRU's dynamics got corrupted last time).

Trained FROM SCRATCH on the combined pool, not continuing any existing
checkpoint -- a clean, standard supervised-learning setup, same
methodology as the original single-cube BC scaling sweep (see memory:
flat_policy_bc_scaling), just on a combined dataset this time.

multicube_oversample: multi-cube demos are much longer episodes (up to
320 steps vs ~30-130 for single-cube), so they already get proportionally
more weight in the per-STEP-averaged loss than their raw episode count
suggests -- oversampling on top of that is a smaller nudge than it looks,
not the dominant lever the way it was in the failed 10-demo fine-tune.

Run with:
    modal run --detach scripts/train_bc_multicube_alternating.py
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(image=rl_image, gpu=None, cpu=4.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=3600 * 2)
def train_bc_multicube_alternating(
    single_cube_demos_path: str = "demonstrations/sb3_teacher_full_task_joint_randomized_v2.pkl",
    multicube_demos_path: str = "demonstrations/multicube_genuine_demos.pkl",
    multicube_oversample: int = 2,
    n_epochs: int = 20,
    lr: float = 1e-3,
    episodes_per_minibatch: int = 16,
    seed: int = 0,
    out_ckpt_path: str = "checkpoints/bc_multicube_alternating_v1.pt",
    out_log_path: str = "logs/bc_multicube_alternating_v1.json",
) -> dict:
    import os
    import json
    import pickle
    import numpy as np
    import torch

    from think_then_act.training.flat_bc_multi_head import MultiHeadBCConfig, MultiHeadBCTrainer
    from think_then_act.training.subgoal_features import FLAT_OBS_DIM

    torch.manual_seed(seed)

    with open(os.path.join(MODEL_CACHE_DIR, single_cube_demos_path), "rb") as f:
        single_cube_data = pickle.load(f)
    single_cube_demos = single_cube_data["demonstrations"]
    print(f"loaded {len(single_cube_demos)} single-cube demos from {single_cube_demos_path}", flush=True)

    with open(os.path.join(MODEL_CACHE_DIR, multicube_demos_path), "rb") as f:
        multicube_data = pickle.load(f)
    multicube_demos = multicube_data["demonstrations"]
    print(f"loaded {len(multicube_demos)} multi-cube demos from {multicube_demos_path} "
          f"(by cube count: {multicube_data.get('n_success_by_cubes')})", flush=True)

    combined = single_cube_demos + multicube_demos * multicube_oversample
    print(f"combined training pool: {len(single_cube_demos)} single-cube + "
          f"{len(multicube_demos)}*{multicube_oversample}={len(multicube_demos)*multicube_oversample} multi-cube "
          f"= {len(combined)} total episodes", flush=True)

    cfg = MultiHeadBCConfig(obs_dim=FLAT_OBS_DIM, policy_type="mse", lr=lr, n_epochs=1,
                              episodes_per_minibatch=episodes_per_minibatch)
    trainer = MultiHeadBCTrainer(cfg)

    rng = np.random.default_rng(seed)
    epoch_losses = []
    for epoch in range(n_epochs):
        perm = rng.permutation(len(combined))
        shuffled = [combined[i] for i in perm]
        result = trainer.fit(shuffled)
        loss = result["epoch_losses"][0]
        epoch_losses.append(loss)
        print(f"  epoch={epoch}  loss={loss:.5f}", flush=True)

    full_ckpt_path = os.path.join(MODEL_CACHE_DIR, out_ckpt_path)
    trainer.save_checkpoint(full_ckpt_path)
    model_volume.commit()
    print(f"saved checkpoint -> {full_ckpt_path}", flush=True)

    log = {
        "single_cube_demos_path": single_cube_demos_path, "multicube_demos_path": multicube_demos_path,
        "n_single_cube": len(single_cube_demos), "n_multicube": len(multicube_demos),
        "multicube_oversample": multicube_oversample, "lr": lr, "n_epochs": n_epochs,
        "epoch_losses": epoch_losses,
    }
    out_log_full = os.path.join(MODEL_CACHE_DIR, out_log_path)
    os.makedirs(os.path.dirname(out_log_full), exist_ok=True)
    with open(out_log_full, "w") as f:
        json.dump(log, f, indent=2, default=float)
    model_volume.commit()

    return {"final_loss": epoch_losses[-1], "out_ckpt_path": out_ckpt_path}


@app.local_entrypoint()
def main(
    single_cube_demos_path: str = "demonstrations/sb3_teacher_full_task_joint_randomized_v2.pkl",
    multicube_demos_path: str = "demonstrations/multicube_genuine_demos.pkl",
    multicube_oversample: int = 2,
    n_epochs: int = 20,
    lr: float = 1e-3,
    out_ckpt_path: str = "checkpoints/bc_multicube_alternating_v1.pt",
):
    result = train_bc_multicube_alternating.remote(
        single_cube_demos_path=single_cube_demos_path, multicube_demos_path=multicube_demos_path,
        multicube_oversample=multicube_oversample, n_epochs=n_epochs, lr=lr, out_ckpt_path=out_ckpt_path,
    )
    print("\n", result)
