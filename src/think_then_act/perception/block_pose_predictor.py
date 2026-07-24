"""
think_then_act.perception.block_pose_predictor

Small self-supervised block-pose regressor: takes a single rendered frame
and outputs an estimate of the block's XYZ position in world coordinates.

Trained on (frame, achieved_goal) pairs collected during sim rollouts,
where achieved_goal is MuJoCo's own free block-position state — used only
as a training LABEL, never as an input. The model itself only ever sees
pixels, same split as perception.collision_predictor.CollisionPredictor
(see that module's docstring) — this is the piece that replaces the
privileged achieved_goal fed into training/subgoal_features.py's
build_subgoal_observation, so a checkpoint trained this way is the piece
that could plausibly run on a real robot later, where there's no privileged
sim state, only a camera.

Reward/done computation (reward/subgoal_reward.py) deliberately keeps using
the privileged ground-truth achieved_goal, not this model's output — only
the *observation* fed to the policy switches. Success criteria elsewhere in
this project (see memory: hierarchical_architecture.md, close_gripper) have
already been gamed by proxy signals multiple times; keeping done/reward on
ground truth keeps completion_rate trustworthy while this model's own
accuracy stays independently measurable (see scripts/validate_pose_predictor.py).

Deliberately does NOT end its conv backbone with a global average pool the
way collision_predictor.py does, despite the otherwise-identical backbone
shape. Global average pooling is the right inductive bias for THAT model's
task (translation-invariant "is any contact happening anywhere" — a global
yes/no), but it is the wrong one here: it destroys the very spatial
location information a position estimate depends on. Confirmed empirically
(2026-07-23), not assumed: a first version of this model using
AdaptiveAvgPool2d(1) never beat a "predict the training-set mean position,
ignore the image entirely" baseline (val error 14.87cm vs. baseline
14.99cm) after training loss plateaued immediately post-epoch-1 — the
classic signature of MSE collapsing to a near-constant output. Root-caused
via per-pixel Pearson correlation between raw pixel intensity and true
block x/y across the collected dataset: real signal exists (max |r|~0.6,
survives the 64x64 downsample) but is concentrated in a small, spatially
localized patch (~55 of 4096 pixels at 64x64) — global-average-pooling the
8x8 post-conv feature map into one value per channel dilutes that small
patch in with the other ~60 mostly-irrelevant cells (dominated by the
randomized arm pose, not the block) before the regression head ever sees
it. Flattening the full spatial feature map instead preserves that
localized signal for the head to actually use.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

FRAME_SIZE = 64  # same cost-budget rationale as collision_predictor.py —
                  # this also runs every low-level step
_FEATURE_MAP_SIZE = FRAME_SIZE // 8   # 3 stride-2 convs: 64 -> 32 -> 16 -> 8


class BlockPosePredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            # NO AdaptiveAvgPool2d(1) here — see module docstring for why
            # that's wrong for position regression specifically.
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * _FEATURE_MAP_SIZE * _FEATURE_MAP_SIZE, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, H, W) float tensor in [0, 1]. Returns (N, 3) raw (x, y, z) metres."""
        return self.head(self.features(x))

    @staticmethod
    def preprocess(frame: np.ndarray) -> torch.Tensor:
        """
        frame: (H, W, 3) uint8 RGB, as produced by ObservationHarness.last_frame().
        Returns a (3, FRAME_SIZE, FRAME_SIZE) float tensor in [0, 1].

        Same resize/normalize logic as CollisionPredictor.preprocess, kept
        as its own copy rather than a shared import — the two modules are
        otherwise fully independent and this one-line preprocessing isn't
        worth coupling them over.
        """
        from PIL import Image
        img = Image.fromarray(frame).resize((FRAME_SIZE, FRAME_SIZE), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0   # (H, W, 3)
        return torch.from_numpy(arr).permute(2, 0, 1)     # (3, H, W)

    def predict_position(self, frame: np.ndarray) -> np.ndarray:
        """Single-frame convenience wrapper: frame -> (3,) estimated block XYZ, metres."""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                x = self.preprocess(frame).unsqueeze(0)
                return self.forward(x).squeeze(0).numpy()
        finally:
            if was_training:
                self.train()
