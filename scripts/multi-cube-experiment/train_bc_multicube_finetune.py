"""
train_bc_multicube_finetune.py

Alternative to train_bc_multicube_alternating.py's from-scratch combined-pool
training. That approach trains a FRESH policy on single-cube + multi-cube
demos mixed together; after fixing MultiCubeStackEnv's dynamic-retargeting and
stack-integrity gaps, the corrected eval showed it getting WORSE with more
multi-cube data (100-demo: 6.7%, 400-demo: 0.0%), and the corrected-environment
videos showed the policy getting confused about what to do after placing cube
0 -- consistent with a fresh mixed-pool train diluting the already-strong
single-cube skill rather than extending it.

This script instead starts from the single-cube PPO checkpoint
(flat_task_ppo_gripper3d_v1_cont_best.pt, 70.0% pose-randomized single-cube
completion -- the checkpoint from BEFORE any multi-cube alternation) and
fine-tunes it directly on ONLY the 2-cube demo pool. SubgoalRecurrentPolicy
is byte-for-byte the same actor class PPO and the "mse" BC head both use
(obs_dim=22, action_dim=4, hidden_dim=64, rnn_hidden_size=64), so the PPO
checkpoint's actor weights load directly into MultiHeadBCTrainer via its
existing load_checkpoint (the policy_type-mismatch guard is skipped since
PPO checkpoints don't carry a policy_type key).

lr defaults lower than the from-scratch run (3e-4 vs 1e-3) since this is a
fine-tune of an already-good policy, not a cold start -- we want to extend
the cube-0-done -> go get cube-1 behavior without catastrophically
overwriting the base reach/grasp/place skill a from-scratch low-lr run would
have to re-learn anyway.

Held-out val split added 2026-10-05: every BC run in this project before now
only ever reported TRAINING loss (MultiHeadBCTrainer.fit() has no val split
at all) -- the 600-demo combined-pool run's training loss was LOWER than the
400-demo run's (0.055 vs 0.083) while its rollout success rate was much
WORSE (16.7% vs 46.7%), the classic overfit/noisy-data signature a held-out
loss exists to catch, cheaply, before paying for a full rollout eval.
val_frac of the demos (held out BEFORE any oversampling/shuffling, fixed
across epochs by `seed`) are never trained on; MultiHeadBCTrainer.evaluate()
(no_grad, no optimizer step) scores them after every epoch.

Run with:
    modal run --detach scripts/train_bc_multicube_finetune.py
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(image=rl_image, gpu=None, cpu=4.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=3600 * 2)
def train_bc_multicube_finetune(
    init_ckpt_path: str = "checkpoints/flat_task_ppo_gripper3d_v1_cont_best.pt",
    multicube_demos_path: str = "demonstrations/stack2_demos_for_bc_gripper3d.pkl",
    n_epochs: int = 20,
    lr: float = 3e-4,
    episodes_per_minibatch: int = 16,
    val_frac: float = 0.15,
    seed: int = 0,
    out_ckpt_path: str = "checkpoints/bc_multicube_finetune_v1.pt",
    out_log_path: str = "logs/bc_multicube_finetune_v1.json",
) -> dict:
    import os
    import json
    import pickle
    import numpy as np
    import torch

    from think_then_act.training.flat_bc_multi_head import MultiHeadBCConfig, MultiHeadBCTrainer
    from think_then_act.training.subgoal_features import FLAT_OBS_DIM

    torch.manual_seed(seed)

    with open(os.path.join(MODEL_CACHE_DIR, multicube_demos_path), "rb") as f:
        multicube_data = pickle.load(f)
    all_demos = multicube_data["demonstrations"]
    print(f"loaded {len(all_demos)} multi-cube demos from {multicube_demos_path} "
          f"(by cube count: {multicube_data.get('n_success_by_cubes')})", flush=True)

    # Held-out val split, fixed once up front (not re-shuffled across epochs)
    # so val numbers are comparable epoch-to-epoch -- see module docstring.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_demos))
    n_val = max(1, int(round(len(all_demos) * val_frac)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    val_demos = [all_demos[i] for i in val_idx]
    multicube_demos = [all_demos[i] for i in train_idx]
    print(f"split: {len(multicube_demos)} train / {len(val_demos)} val (val_frac={val_frac})", flush=True)

    cfg = MultiHeadBCConfig(obs_dim=FLAT_OBS_DIM, policy_type="mse", lr=lr, n_epochs=1,
                              episodes_per_minibatch=episodes_per_minibatch)
    trainer = MultiHeadBCTrainer(cfg)

    init_full_path = os.path.join(MODEL_CACHE_DIR, init_ckpt_path)
    trainer.load_checkpoint(init_full_path)
    print(f"warm-started actor/critic from {init_ckpt_path}", flush=True)

    epoch_losses = []
    val_losses = []
    for epoch in range(n_epochs):
        shuf_perm = rng.permutation(len(multicube_demos))
        shuffled = [multicube_demos[i] for i in shuf_perm]
        result = trainer.fit(shuffled)
        loss = result["epoch_losses"][0]
        val_loss = trainer.evaluate(val_demos)
        epoch_losses.append(loss)
        val_losses.append(val_loss)
        print(f"  epoch={epoch}  train_loss={loss:.5f}  val_loss={val_loss:.5f}", flush=True)

    full_ckpt_path = os.path.join(MODEL_CACHE_DIR, out_ckpt_path)
    trainer.save_checkpoint(full_ckpt_path)
    model_volume.commit()
    print(f"saved checkpoint -> {full_ckpt_path}", flush=True)

    log = {
        "init_ckpt_path": init_ckpt_path, "multicube_demos_path": multicube_demos_path,
        "n_multicube_train": len(multicube_demos), "n_multicube_val": len(val_demos),
        "val_frac": val_frac, "lr": lr, "n_epochs": n_epochs,
        "epoch_losses": epoch_losses, "val_losses": val_losses,
    }
    out_log_full = os.path.join(MODEL_CACHE_DIR, out_log_path)
    os.makedirs(os.path.dirname(out_log_full), exist_ok=True)
    with open(out_log_full, "w") as f:
        json.dump(log, f, indent=2, default=float)
    model_volume.commit()

    return {"final_loss": epoch_losses[-1], "final_val_loss": val_losses[-1], "out_ckpt_path": out_ckpt_path}


@app.local_entrypoint()
def main(
    init_ckpt_path: str = "checkpoints/flat_task_ppo_gripper3d_v1_cont_best.pt",
    multicube_demos_path: str = "demonstrations/stack2_demos_for_bc_gripper3d.pkl",
    n_epochs: int = 20,
    lr: float = 3e-4,
    val_frac: float = 0.15,
    out_ckpt_path: str = "checkpoints/bc_multicube_finetune_v1.pt",
):
    result = train_bc_multicube_finetune.remote(
        init_ckpt_path=init_ckpt_path, multicube_demos_path=multicube_demos_path,
        n_epochs=n_epochs, lr=lr, val_frac=val_frac, out_ckpt_path=out_ckpt_path,
    )
    print("\n", result)
