"""
train_bc_gripper3d.py

BC training on the new demonstrations/sb3_teacher_full_task_gripper3d.pkl
pool (collect_sb3_teacher_demos.py, pose_scheme="gripper_3d" --
env.setup.randomize_gripper_start_3d, the natural/non-"torturous" pose
scheme validated via the skeleton-overlay comparison in the "Pose
Generalization & Stacking" artifact, 2026-10-04). Trains BOTH
policy_type="mse" and policy_type="cvae" (user's own call, having two
architectures to compare again like the original 5-way BC scaling sweep
-- see memory: flat_policy_bc_scaling) on the SAME demo pool, standard
fresh-training BC methodology (not a fine-tune).

Own isolated checkpoint/log paths throughout -- never touches
checkpoints/pose_randomized_v2/* or checkpoints/bc_scaling/*, the
existing randomize_joint_angles-trained checkpoints everything else in
this project still depends on.

Run with:
    modal run --detach scripts/train_bc_gripper3d.py --policy-type mse
    modal run --detach scripts/train_bc_gripper3d.py --policy-type cvae
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(image=rl_image, gpu=None, cpu=4.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=3600 * 2)
def train_bc_gripper3d(
    demos_path: str = "demonstrations/sb3_teacher_full_task_gripper3d.pkl",
    policy_type: str = "mse",
    n_demos: int = 2000,
    n_epochs: int = 20,
    lr: float = 1e-3,
    episodes_per_minibatch: int = 16,
    seed: int = 0,
    vae_latent_dim: int = 8,
    vae_beta: float = 1.0,
    out_ckpt_path: str = None,
    out_log_path: str = None,
) -> dict:
    import os
    import json
    import pickle
    import numpy as np
    import torch

    from think_then_act.training.flat_bc_multi_head import MultiHeadBCConfig, MultiHeadBCTrainer
    from think_then_act.training.subgoal_features import FLAT_OBS_DIM

    torch.manual_seed(seed)

    out_ckpt_path = out_ckpt_path or f"checkpoints/gripper3d_v1/{policy_type}_n{n_demos}_seed{seed}.pt"
    out_log_path = out_log_path or f"logs/bc_gripper3d_{policy_type}_n{n_demos}_seed{seed}.json"

    with open(os.path.join(MODEL_CACHE_DIR, demos_path), "rb") as f:
        demo_data = pickle.load(f)
    all_demos = demo_data["demonstrations"]
    print(f"loaded {len(all_demos)} demos from {demos_path} (pose_scheme={demo_data.get('pose_scheme')})", flush=True)
    if n_demos > len(all_demos):
        raise ValueError(f"n_demos={n_demos} > available {len(all_demos)}")
    rng_subsample = np.random.default_rng(seed)
    idx = rng_subsample.choice(len(all_demos), size=n_demos, replace=False)
    demos = [all_demos[i] for i in idx]

    cfg = MultiHeadBCConfig(obs_dim=FLAT_OBS_DIM, policy_type=policy_type, lr=lr, n_epochs=1,
                              episodes_per_minibatch=episodes_per_minibatch,
                              vae_latent_dim=vae_latent_dim, vae_beta=vae_beta)
    trainer = MultiHeadBCTrainer(cfg)

    epoch_losses = []
    for epoch in range(n_epochs):
        perm = rng_subsample.permutation(len(demos))
        shuffled = [demos[i] for i in perm]
        result = trainer.fit(shuffled)
        loss = result["epoch_losses"][0]
        epoch_losses.append(loss)
        print(f"  epoch={epoch}  loss={loss:.5f}", flush=True)

    full_ckpt_path = os.path.join(MODEL_CACHE_DIR, out_ckpt_path)
    trainer.save_checkpoint(full_ckpt_path)
    model_volume.commit()
    print(f"saved checkpoint -> {full_ckpt_path}", flush=True)

    log = {"demos_path": demos_path, "policy_type": policy_type, "n_demos": n_demos,
           "lr": lr, "n_epochs": n_epochs, "seed": seed, "epoch_losses": epoch_losses}
    out_log_full = os.path.join(MODEL_CACHE_DIR, out_log_path)
    os.makedirs(os.path.dirname(out_log_full), exist_ok=True)
    with open(out_log_full, "w") as f:
        json.dump(log, f, indent=2, default=float)
    model_volume.commit()

    return {"final_loss": epoch_losses[-1], "out_ckpt_path": out_ckpt_path}


@app.local_entrypoint()
def main(
    policy_type: str = "mse",
    n_demos: int = 2000,
    n_epochs: int = 20,
    lr: float = 1e-3,
    seed: int = 0,
):
    result = train_bc_gripper3d.remote(policy_type=policy_type, n_demos=n_demos, n_epochs=n_epochs, lr=lr, seed=seed)
    print("\n", result)
