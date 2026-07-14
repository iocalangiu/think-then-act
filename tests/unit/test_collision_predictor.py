"""
Unit tests for perception.collision_predictor — pure PyTorch, no mujoco needed.
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from think_then_act.perception.collision_predictor import CollisionPredictor, FRAME_SIZE


def _random_frame(h=480, w=480):
    return (np.random.rand(h, w, 3) * 255).astype(np.uint8)


def test_forward_shape_and_finite():
    model = CollisionPredictor()
    x = torch.rand(4, 3, FRAME_SIZE, FRAME_SIZE)
    logits = model(x)
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()


def test_preprocess_shape_and_range():
    x = CollisionPredictor.preprocess(_random_frame())
    assert x.shape == (3, FRAME_SIZE, FRAME_SIZE)
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_preprocess_handles_non_square_frames():
    x = CollisionPredictor.preprocess(_random_frame(h=360, w=640))
    assert x.shape == (3, FRAME_SIZE, FRAME_SIZE)


def test_predict_proba_is_a_probability():
    model = CollisionPredictor()
    p = model.predict_proba(_random_frame())
    assert 0.0 <= p <= 1.0


def test_predict_proba_restores_training_mode():
    model = CollisionPredictor()
    model.train()
    model.predict_proba(_random_frame())
    assert model.training is True

    model.eval()
    model.predict_proba(_random_frame())
    assert model.training is False


def test_trainable_with_bce_loss():
    """Sanity check: a gradient step on a toy batch actually reduces loss."""
    torch.manual_seed(0)
    model = CollisionPredictor()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.rand(8, 3, FRAME_SIZE, FRAME_SIZE)
    y = torch.randint(0, 2, (8,)).float()
    loss_fn = torch.nn.BCEWithLogitsLoss()

    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
