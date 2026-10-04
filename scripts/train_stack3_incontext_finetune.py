"""
train_stack3_incontext_finetune.py

Fine-tunes the pose-randomized PPO checkpoint
(flat_task_ppo_poserand_v2_cont_best.pt) on the 10 continuous-episode
3-cube stacking traces collected by collect_stack3_demos.py
(demonstrations/stack3_incontext_demos.pkl), so the GRU's own recurrent
weights learn to recognize "the goal just changed mid-episode -> release
and reposition" purely from the pattern in the observation stream, with
no explicit phase/subtask channel (see flat_policy_ppo_generalization
memory's "Stage 3, clarified" section for the full design rationale —
this is meant to produce GENERALIZED in-context recognition from
training, not to memorize these 10 exact trajectories, which is why only
10 demos + a low LR + few epochs + an anti-forgetting mix-in, not a
from-scratch retrain).

Catastrophic-forgetting mitigation: fine-tuning on only 10 long episodes
with a policy that otherwise already works reasonably well (76.7%
pose-randomized completion) risks overwriting the base single-object
pick-and-place skill entirely, since MultiHeadBCTrainer.fit()'s loss has
no explicit regularization toward the starting weights. Mitigated by
mixing in a random sample of the ORIGINAL pose-randomized demo pool
(demonstrations/sb3_teacher_full_task_joint_randomized_v2.pkl) alongside
the 10 new traces every epoch — enough that the base skill keeps getting
reinforced, not so much that the 10 stacking traces are drowned out
(n_base_demos_per_epoch, default 60, vs. 10 stacking traces = ~14% of
each epoch's episodes are the new behavior, resampled to a fixed 60 each
epoch rather than including this whole 2000-demo pool every epoch, to
keep epochs cheap and to vary which base episodes are seen).

Saves to its own isolated path (checkpoints/stack3_incontext_v1.pt) --
never overwrites flat_task_ppo_poserand_v2_cont_best.pt or anything else
load-bearing, per this project's standing "don't overwrite anything"
rule (see flat_policy_ppo_generalization memory's near-miss section).

Run with:
    modal run scripts/train_stack3_incontext_finetune.py
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(image=rl_image, gpu=None, cpu=2.0, volumes={MODEL_CACHE_DIR: model_volume}, timeout=1800)
def train_stack3_incontext_finetune(
    base_ckpt_path: str = "checkpoints/flat_task_ppo_poserand_v2_cont_best.pt",
    stack3_demos_path: str = "demonstrations/stack3_incontext_demos.pkl",
    base_demos_path: str = "demonstrations/sb3_teacher_full_task_joint_randomized_v2.pkl",
    n_base_demos_per_epoch: int = 60,
    stack3_oversample: int = 3,   # repeat each of the 10 stacking traces this many times per
                                  # epoch's episode list, so they're a meaningful fraction of
                                  # every minibatch without literally dominating it (10*3=30
                                  # stacking "episodes" vs. 60 base episodes per epoch = 1/3)
    lr: float = 1e-4,             # 10x lower than the original BC default (1e-3) -- a small
                                  # nudge toward the new behavior, not a fresh training signal
                                  # strong enough to overwrite the existing skill
    n_epochs: int = 40,
    episodes_per_minibatch: int = 16,
    seed: int = 0,
    out_ckpt_path: str = "checkpoints/stack3_incontext_v1.pt",
    out_log_path: str = "logs/stack3_incontext_finetune.json",
) -> dict:
    import os
    import json
    import pickle
    import numpy as np

    from think_then_act.training.flat_bc_multi_head import MultiHeadBCConfig, MultiHeadBCTrainer
    from think_then_act.training.subgoal_features import FLAT_OBS_DIM

    rng = np.random.default_rng(seed)

    with open(os.path.join(MODEL_CACHE_DIR, stack3_demos_path), "rb") as f:
        stack3_data = pickle.load(f)
    stack3_demos = stack3_data["demonstrations"]
    print(f"loaded {len(stack3_demos)} stack3 in-context demos "
          f"(collected from {stack3_data['n_attempted']} attempts, "
          f"{stack3_data['success_rate']:.1%} success rate)", flush=True)
    assert len(stack3_demos) >= 1, "need at least 1 stack3 demo to fine-tune on"

    with open(os.path.join(MODEL_CACHE_DIR, base_demos_path), "rb") as f:
        base_demos_all = pickle.load(f)
    if isinstance(base_demos_all, dict):
        base_demos_all = base_demos_all["demonstrations"]
    print(f"loaded {len(base_demos_all)} base single-cube demos for the anti-forgetting mix-in", flush=True)

    cfg = MultiHeadBCConfig(
        obs_dim=FLAT_OBS_DIM, policy_type="mse", lr=lr, n_epochs=1,
        episodes_per_minibatch=episodes_per_minibatch,
    )
    trainer = MultiHeadBCTrainer(cfg)
    trainer.load_checkpoint(os.path.join(MODEL_CACHE_DIR, base_ckpt_path))
    print(f"loaded base checkpoint {base_ckpt_path}", flush=True)

    stack3_oversampled = stack3_demos * stack3_oversample

    epoch_losses = []
    for epoch in range(n_epochs):
        base_idx = rng.choice(len(base_demos_all), size=min(n_base_demos_per_epoch, len(base_demos_all)), replace=False)
        base_sample = [base_demos_all[i] for i in base_idx]
        epoch_demos = stack3_oversampled + base_sample
        result = trainer.fit(epoch_demos)   # n_epochs=1 in cfg -- fit() shuffles+minibatches once
        loss = result["epoch_losses"][0]
        epoch_losses.append(loss)
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"  epoch={epoch}  loss={loss:.5f}  (n_episodes_this_epoch={len(epoch_demos)})", flush=True)

    out_ckpt_full = os.path.join(MODEL_CACHE_DIR, out_ckpt_path)
    trainer.save_checkpoint(out_ckpt_full)
    model_volume.commit()
    print(f"saved fine-tuned checkpoint -> {out_ckpt_full}", flush=True)

    log = {
        "base_ckpt_path": base_ckpt_path, "stack3_demos_path": stack3_demos_path,
        "base_demos_path": base_demos_path, "n_stack3_demos": len(stack3_demos),
        "stack3_oversample": stack3_oversample, "n_base_demos_per_epoch": n_base_demos_per_epoch,
        "lr": lr, "n_epochs": n_epochs, "epoch_losses": epoch_losses,
    }
    out_log_full = os.path.join(MODEL_CACHE_DIR, out_log_path)
    os.makedirs(os.path.dirname(out_log_full), exist_ok=True)
    with open(out_log_full, "w") as f:
        json.dump(log, f, indent=2)
    model_volume.commit()

    return {"final_loss": epoch_losses[-1], "out_ckpt_path": out_ckpt_path}


@app.local_entrypoint()
def main(
    base_ckpt_path: str = "checkpoints/flat_task_ppo_poserand_v2_cont_best.pt",
    n_epochs: int = 40,
    lr: float = 1e-4,
    out_ckpt_path: str = "checkpoints/stack3_incontext_v1.pt",
):
    result = train_stack3_incontext_finetune.remote(
        base_ckpt_path=base_ckpt_path, n_epochs=n_epochs, lr=lr, out_ckpt_path=out_ckpt_path,
    )
    print("\n", result)
