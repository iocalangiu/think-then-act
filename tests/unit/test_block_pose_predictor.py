"""
Unit tests for perception.block_pose_predictor — pure PyTorch, no mujoco needed.
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from think_then_act.perception.block_pose_predictor import BlockPosePredictor, FRAME_SIZE


def _random_frame(h=480, w=480):
    return (np.random.rand(h, w, 3) * 255).astype(np.uint8)


def test_forward_shape_and_finite():
    model = BlockPosePredictor()
    x = torch.rand(4, 3, FRAME_SIZE, FRAME_SIZE)
    out = model(x)
    assert out.shape == (4, 3)
    assert torch.isfinite(out).all()


def test_preprocess_shape_and_range():
    x = BlockPosePredictor.preprocess(_random_frame())
    assert x.shape == (3, FRAME_SIZE, FRAME_SIZE)
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_preprocess_handles_non_square_frames():
    x = BlockPosePredictor.preprocess(_random_frame(h=360, w=640))
    assert x.shape == (3, FRAME_SIZE, FRAME_SIZE)


def test_predict_position_shape():
    model = BlockPosePredictor()
    pos = model.predict_position(_random_frame())
    assert pos.shape == (3,)
    assert np.isfinite(pos).all()


def test_predict_position_restores_training_mode():
    model = BlockPosePredictor()
    model.train()
    model.predict_position(_random_frame())
    assert model.training is True

    model.eval()
    model.predict_position(_random_frame())
    assert model.training is False


def test_trainable_with_mse_loss():
    """Sanity check: a gradient step on a toy batch actually reduces loss."""
    torch.manual_seed(0)
    model = BlockPosePredictor()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.rand(8, 3, FRAME_SIZE, FRAME_SIZE)
    y = torch.rand(8, 3) + torch.tensor([1.3, 0.75, 0.45])  # roughly table-scale targets
    loss_fn = torch.nn.MSELoss()

    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
