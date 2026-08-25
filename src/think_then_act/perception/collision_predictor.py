"""
think_then_act.perception.collision_predictor

Small self-supervised collision-probability predictor: takes a single
rendered frame and outputs P(the arm is currently in unwanted contact).

Trained on (frame, contact_ground_truth) pairs collected during sim rollouts,
where the ground truth comes from MuJoCo's free contact state
(env.setup.has_contact) — used only as a training LABEL, never as an input.
The model itself only ever sees pixels, so a checkpoint trained this way is
the piece that could plausibly run on a real robot later, where there's no
privileged sim state, only a camera — unlike the ground-truth signal it's
trained against, which only exists in sim.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

FRAME_SIZE = 64  # deliberately small/cheap — this runs every low-level step,
                 # unlike the VLM's 224x224 which only runs every K steps


class CollisionPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, H, W) float tensor in [0, 1]. Returns (N,) raw logits (pre-sigmoid)."""
        return self.head(self.features(x)).squeeze(-1)

    @staticmethod
    def preprocess(frame: np.ndarray) -> torch.Tensor:
        """
        frame: (H, W, 3) uint8 RGB, as produced by ObservationHarness.last_frame().
        Returns a (3, FRAME_SIZE, FRAME_SIZE) float tensor in [0, 1].
        """
        from PIL import Image
        img = Image.fromarray(frame).resize((FRAME_SIZE, FRAME_SIZE), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0   # (H, W, 3)
        return torch.from_numpy(arr).permute(2, 0, 1)     # (3, H, W)

    def predict_proba(self, frame: np.ndarray) -> float:
        """Single-frame convenience wrapper: frame -> P(contact) in [0, 1]."""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                x = self.preprocess(frame).unsqueeze(0)
                logit = self.forward(x)
                return torch.sigmoid(logit).item()
        finally:
            if was_training:
                self.train()
