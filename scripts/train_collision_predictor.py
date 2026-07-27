"""
train_collision_predictor.py

Supervised training of perception/collision_predictor.py on the dataset
collected by collect_collision_data.py: (frame, table_collision) pairs,
BCE loss. Small model + small (64x64) images, so this runs fine on CPU —
no A10G needed, unlike the VLM stages.

Trains on the "table_collision" field, NOT "contact" — found empirically
(2026-07-12) that raw "any MuJoCo contact" is ~100% at every step, dominated
by structural contacts (robot base resting against floor/table, block
resting on table) that are always true regardless of policy. table_collision
(env.setup.is_meaningful_table_collision) excludes those, and is the actual
"did the arm hit the table" signal.

Reads : /model-cache/collision_data.jsonl
Saves : /model-cache/checkpoints/collision_predictor.pt (best val loss only,
        same "track best, not latest" convention as sft_train.py's
        sft_warmstart fix)

Run with:
    modal run scripts/train_collision_predictor.py
    modal run scripts/train_collision_predictor.py --n-epochs 30 --lr 5e-4
"""

import modal
from think_then_act.modal_app import app, rl_image, model_volume, MODEL_CACHE_DIR


@app.function(
    image=rl_image,
    gpu=None,
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=1800,
)
def train_collision_predictor(
    n_epochs     : int   = 20,
    lr           : float = 1e-3,
    batch_size   : int   = 32,
    val_frac     : float = 0.1,   # fraction of EPISODES held out, same leakage
                                  # rationale as sft_train.py's episode split
    patience     : int   = 3,
    min_delta    : float = 0.001,
    max_examples : int   = 0,     # 0 = use all
) -> dict:
    import os, json, random, base64, io
    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image

    from think_then_act.perception.collision_predictor import CollisionPredictor

    print("\n" + "=" * 60)
    print("  COLLISION PREDICTOR TRAINING")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load + split data (by episode, not example — same leakage rationale
    #    as sft_train.py: adjacent steps in one episode are correlated).
    # ------------------------------------------------------------------
    data_path = os.path.join(MODEL_CACHE_DIR, "collision_data.jsonl")
    print(f"\n[1/3] Loading collision data from {data_path}...")

    examples = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line))
    if max_examples > 0:
        examples = examples[:max_examples]

    n_pos = sum(1 for ex in examples if ex["table_collision"])
    print(f"  {len(examples)} examples  table_collision_rate={n_pos / max(len(examples), 1):.1%}")
    if n_pos == 0:
        raise ValueError(
            "0 positive table_collision examples in this dataset — the model "
            "would just learn to always predict 'no collision'. Re-run "
            "collect_collision_data.py (check its printed table_collision_rate "
            "before training on the result)."
        )

    episode_ids = sorted({ex["episode"] for ex in examples})
    rng = random.Random(0)
    rng.shuffle(episode_ids)
    n_val_eps = max(1, int(len(episode_ids) * val_frac))
    val_ep_ids = set(episode_ids[:n_val_eps])

    train_examples = [ex for ex in examples if ex["episode"] not in val_ep_ids]
    val_examples   = [ex for ex in examples if ex["episode"] in val_ep_ids]
    print(f"  train={len(train_examples)} examples ({len(episode_ids) - n_val_eps} episodes)"
          f"  val={len(val_examples)} examples ({n_val_eps} episodes)")

    # pos_weight for BCEWithLogitsLoss, computed from TRAIN split only (not
    # val/full dataset, to avoid any leakage into the loss the model is
    # optimized against). table_collision is rare (~11% here) — unweighted
    # BCE lets the model minimize loss by collapsing to "always predict no
    # collision", which still scores a deceptively high accuracy (~89%) at
    # the base rate while never learning to discriminate. Found empirically
    # 2026-07-13: val_acc pinned identical across every epoch regardless of
    # loss movement was the tell.
    n_pos_train = sum(1 for ex in train_examples if ex["table_collision"])
    n_neg_train = len(train_examples) - n_pos_train
    pos_weight  = n_neg_train / max(n_pos_train, 1)
    print(f"  pos_weight={pos_weight:.2f} (train: {n_pos_train} pos / {n_neg_train} neg)")

    def to_tensors(exs: list) -> tuple:
        frames = []
        for ex in exs:
            frame = np.array(Image.open(io.BytesIO(base64.b64decode(ex["frame_b64"]))))
            frames.append(CollisionPredictor.preprocess(frame))
        x = torch.stack(frames)
        y = torch.tensor([float(ex["table_collision"]) for ex in exs], dtype=torch.float32)
        return x, y

    print("  Decoding frames to tensors...")
    x_train, y_train = to_tensors(train_examples)
    x_val,   y_val   = to_tensors(val_examples) if val_examples else (None, None)

    # ------------------------------------------------------------------
    # 2. Model + optimizer
    # ------------------------------------------------------------------
    model     = CollisionPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    def compute_val_metrics() -> tuple:
        if x_val is None:
            return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
        model.eval()
        with torch.no_grad():
            logits = model(x_val)
            loss   = loss_fn(logits, y_val).item()
            preds  = (torch.sigmoid(logits) > 0.5).float()
            acc    = (preds == y_val).float().mean().item()
            # accuracy alone is misleading under class imbalance (a model that
            # always predicts "no collision" still scores ~89% here) — track
            # precision/recall/F1 on the positive (collision) class so a
            # collapse to the majority class is visible even if accuracy looks fine.
            tp = ((preds == 1) & (y_val == 1)).sum().item()
            fp = ((preds == 1) & (y_val == 0)).sum().item()
            fn = ((preds == 0) & (y_val == 1)).sum().item()
            precision = tp / max(tp + fp, 1)
            recall    = tp / max(tp + fn, 1)
            f1        = 2 * precision * recall / max(precision + recall, 1e-8)
        model.train()
        return loss, acc, precision, recall, f1

    # ------------------------------------------------------------------
    # 3. Training loop
    # ------------------------------------------------------------------
    print(f"\n[2/3] Training: {n_epochs} epochs (upper bound)  lr={lr}  batch_size={batch_size}...")

    n_train = x_train.shape[0]
    loss_history, val_loss_history, val_acc_history = [], [], []
    val_precision_history, val_recall_history, val_f1_history = [], [], []
    best_val_loss      = float("inf")
    epochs_no_improve  = 0
    stopped_early      = False
    ckpt_path = os.path.join(MODEL_CACHE_DIR, "checkpoints", "collision_predictor.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for epoch in range(n_epochs):
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x_train[idx], y_train[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss   = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        val_loss, val_acc, val_precision, val_recall, val_f1 = compute_val_metrics()
        loss_history.append(round(avg_train_loss, 5))
        val_loss_history.append(round(val_loss, 5))
        val_acc_history.append(round(val_acc, 4))
        val_precision_history.append(round(val_precision, 4))
        val_recall_history.append(round(val_recall, 4))
        val_f1_history.append(round(val_f1, 4))

        print(f"  epoch {epoch+1}/{n_epochs}  train_loss={avg_train_loss:.5f}  "
              f"val_loss={val_loss:.5f}  val_acc={val_acc:.1%}  "
              f"val_precision={val_precision:.1%}  val_recall={val_recall:.1%}  val_f1={val_f1:.3f}")

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
    print(f"  Val loss trend     : {val_loss_history}")
    print(f"  Val acc trend      : {val_acc_history}")
    print(f"  Val precision trend: {val_precision_history}")
    print(f"  Val recall trend   : {val_recall_history}")
    print(f"  Val f1 trend       : {val_f1_history}")
    print(f"  Checkpoint -> {ckpt_path}")
    print("=" * 60)

    return {
        "status"              : "PASS",
        "stopped_early"        : stopped_early,
        "best_val_loss"        : round(best_val_loss, 5),
        "loss_trend"           : loss_history,
        "val_loss_trend"       : val_loss_history,
        "val_acc_trend"        : val_acc_history,
        "val_precision_trend"  : val_precision_history,
        "val_recall_trend"     : val_recall_history,
        "val_f1_trend"         : val_f1_history,
        "ckpt_path"            : ckpt_path,
    }


@app.local_entrypoint()
def main(n_epochs: int = 20, lr: float = 1e-3, batch_size: int = 32):
    print(f"\nDispatching collision predictor training to Modal (CPU)...")
    print(f"  epochs<={n_epochs}  lr={lr}  batch_size={batch_size}\n")
    result = train_collision_predictor.remote(n_epochs=n_epochs, lr=lr, batch_size=batch_size)
    print(f"\nDone. best_val_loss={result['best_val_loss']}  stopped_early={result['stopped_early']}")
    if result["val_f1_trend"]:
        best_idx = result["val_loss_trend"].index(result["best_val_loss"])
        print(f"At best checkpoint: val_acc={result['val_acc_trend'][best_idx]:.1%}  "
              f"val_precision={result['val_precision_trend'][best_idx]:.1%}  "
              f"val_recall={result['val_recall_trend'][best_idx]:.1%}  "
              f"val_f1={result['val_f1_trend'][best_idx]:.3f}")
    print(f"Checkpoint: {result['ckpt_path']}")
