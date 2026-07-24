"""
train_pose_estimator.py

Supervised training of perception/block_pose_predictor.py on the dataset
collected by collect_pose_data.py: (frame, achieved_goal) pairs, MSE loss.
Small model + small (64x64) images, so this runs fine on CPU — no A10G
needed, same as train_collision_predictor.py.

Reads : /model-cache/pose_data.jsonl
Saves : /model-cache/checkpoints/block_pose_predictor.pt (best val loss
        only, same "track best, not latest" convention as
        train_collision_predictor.py/sft_train.py)

Run with:
    modal run scripts/train_pose_estimator.py
    modal run scripts/train_pose_estimator.py --n-epochs 30 --lr 5e-4
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(
    image=rl_image,
    gpu=None,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=1800,
)
def train_pose_estimator(
    n_epochs     : int   = 20,
    lr           : float = 1e-3,
    batch_size   : int   = 32,
    val_frac     : float = 0.1,   # fraction of EPISODES held out, same leakage
                                  # rationale as train_collision_predictor.py's split
    patience     : int   = 3,
    min_delta    : float = 0.0001,
    max_examples : int   = 0,     # 0 = use all
) -> dict:
    import os, json, random, base64, io
    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image

    from think_then_act.perception.block_pose_predictor import BlockPosePredictor

    print("\n" + "=" * 60)
    print("  BLOCK POSE PREDICTOR TRAINING")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load + split data (by episode, not example — same leakage rationale
    #    as train_collision_predictor.py: adjacent steps in one episode are
    #    correlated).
    # ------------------------------------------------------------------
    data_path = os.path.join(MODEL_CACHE_DIR, "pose_data.jsonl")
    print(f"\n[1/3] Loading pose data from {data_path}...")

    examples = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line))
    if max_examples > 0:
        examples = examples[:max_examples]
    print(f"  {len(examples)} examples")

    episode_ids = sorted({ex["episode"] for ex in examples})
    rng = random.Random(0)
    rng.shuffle(episode_ids)
    n_val_eps = max(1, int(len(episode_ids) * val_frac))
    val_ep_ids = set(episode_ids[:n_val_eps])

    train_examples = [ex for ex in examples if ex["episode"] not in val_ep_ids]
    val_examples   = [ex for ex in examples if ex["episode"] in val_ep_ids]
    print(f"  train={len(train_examples)} examples ({len(episode_ids) - n_val_eps} episodes)"
          f"  val={len(val_examples)} examples ({n_val_eps} episodes)")

    def to_tensors(exs: list) -> tuple:
        frames = []
        for ex in exs:
            frame = np.array(Image.open(io.BytesIO(base64.b64decode(ex["frame_b64"]))))
            frames.append(BlockPosePredictor.preprocess(frame))
        x = torch.stack(frames)
        y = torch.tensor([ex["achieved_goal"] for ex in exs], dtype=torch.float32)
        return x, y

    print("  Decoding frames to tensors...")
    x_train, y_train = to_tensors(train_examples)
    x_val,   y_val   = to_tensors(val_examples) if val_examples else (None, None)

    # Naive "always predict the train-set mean position, ignore the image
    # entirely" baseline — computed once, up front, and compared against the
    # trained model's val error below. A regressor that collapses to
    # near-constant output (a real risk under plain MSE, which treats
    # predicting the mean as a strong local minimum) would show val error
    # barely beating this number, which pure aggregate metrics alone
    # wouldn't reveal — same "don't trust a single number, ground it"
    # discipline as validate_pose_predictor.py's per-stage breakdown.
    train_mean_pos = y_train.mean(dim=0)
    if y_val is not None:
        baseline_error_cm = float((y_val - train_mean_pos).norm(dim=1).mean().item() * 100.0)
        print(f"  Baseline (predict train-mean position {train_mean_pos.tolist()}, "
              f"ignore image): val_mean_euclidean_error_cm={baseline_error_cm:.2f}")
    else:
        baseline_error_cm = float("nan")

    # ------------------------------------------------------------------
    # 2. Model + optimizer
    # ------------------------------------------------------------------
    model     = BlockPosePredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    def compute_val_metrics() -> tuple:
        if x_val is None:
            return float("nan"), float("nan")
        model.eval()
        with torch.no_grad():
            pred = model(x_val)
            loss = loss_fn(pred, y_val).item()
            # More interpretable against the reward thresholds this needs to
            # support (e.g. close_gripper_dxy_limit=0.03m) than raw MSE.
            mean_euclidean_error_cm = float((pred - y_val).norm(dim=1).mean().item() * 100.0)
        model.train()
        return loss, mean_euclidean_error_cm

    # ------------------------------------------------------------------
    # 3. Training loop
    # ------------------------------------------------------------------
    print(f"\n[2/3] Training: {n_epochs} epochs (upper bound)  lr={lr}  batch_size={batch_size}...")

    n_train = x_train.shape[0]
    loss_history, val_loss_history, val_err_cm_history = [], [], []
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    stopped_early      = False
    ckpt_path = os.path.join(MODEL_CACHE_DIR, "checkpoints", "block_pose_predictor.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for epoch in range(n_epochs):
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x_train[idx], y_train[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        val_loss, val_err_cm = compute_val_metrics()
        loss_history.append(round(avg_train_loss, 6))
        val_loss_history.append(round(val_loss, 6))
        val_err_cm_history.append(round(val_err_cm, 3))

        print(f"  epoch {epoch+1}/{n_epochs}  train_loss={avg_train_loss:.6f}  "
              f"val_loss={val_loss:.6f}  val_mean_euclidean_error_cm={val_err_cm:.2f}")

        val_signal = val_loss if val_examples else avg_train_loss
        if val_signal < best_val_loss - min_delta:
            best_val_loss     = val_signal
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            model_volume.commit()
            print(f"    [best] val_loss improved -> checkpoint saved")
        else:
            epochs_no_improve += 1

        if val_examples and epochs_no_improve >= patience:
            print(f"  [early stop] no val_loss improvement for {patience} epochs.")
            stopped_early = True
            break

    print(f"\n[3/3] Done. Train loss trend: {loss_history}")
    print(f"  Val loss trend               : {val_loss_history}")
    print(f"  Val mean euclidean error (cm): {val_err_cm_history}")
    print(f"  Baseline (predict mean, cm)  : {baseline_error_cm:.2f}")
    best_err_cm = min(val_err_cm_history) if val_err_cm_history else float("nan")
    beats_baseline = best_err_cm < baseline_error_cm - 0.5   # 0.5cm margin, not noise
    print(f"  Best model beats baseline by : {baseline_error_cm - best_err_cm:.2f}cm "
          f"({'YES — learning real signal from pixels' if beats_baseline else 'NO — model is not meaningfully better than ignoring the image'})")
    print(f"  Checkpoint -> {ckpt_path}")
    print("=" * 60)

    return {
        "status"          : "PASS",
        "stopped_early"   : stopped_early,
        "best_val_loss"   : round(best_val_loss, 6),
        "loss_trend"      : loss_history,
        "val_loss_trend"  : val_loss_history,
        "val_err_cm_trend": val_err_cm_history,
        "baseline_error_cm": round(baseline_error_cm, 3),
        "beats_baseline"   : beats_baseline,
        "ckpt_path"       : ckpt_path,
    }


@app.local_entrypoint()
def main(n_epochs: int = 20, lr: float = 1e-3, batch_size: int = 32):
    print(f"\nDispatching pose predictor training to Modal (CPU)...")
    print(f"  epochs<={n_epochs}  lr={lr}  batch_size={batch_size}\n")
    result = train_pose_estimator.remote(n_epochs=n_epochs, lr=lr, batch_size=batch_size)
    print(f"\nDone. best_val_loss={result['best_val_loss']}  stopped_early={result['stopped_early']}")
    if result["val_err_cm_trend"]:
        best_idx = result["val_loss_trend"].index(result["best_val_loss"])
        print(f"At best checkpoint: val_mean_euclidean_error_cm={result['val_err_cm_trend'][best_idx]}")
    print(f"Baseline (predict mean, ignore image): {result['baseline_error_cm']}cm  "
          f"beats_baseline={result['beats_baseline']}")
    print(f"Checkpoint: {result['ckpt_path']}")
