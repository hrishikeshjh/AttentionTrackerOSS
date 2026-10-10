"""
model.py — 1D-CNN temporal behavior classifier.

Architecture (ARCHITECTURE.md §4 — Temporal Classifier):
    Input  : (batch, FEATURE_DIM, WINDOW_SIZE)   i.e. (B, 15, 15)
             FEATURE_DIM = 15 = features/feature_vector.py FEATURE_VECTOR_VERSION 2
             (13 v1.0-compatible channels + posture_upright + posture_lean_norm).
             Older 13-channel checkpoints still load (load() reads the
             channel count from the checkpoint).
    Conv1D : F→32, kernel 3, pad 1  + BN + ReLU
    Conv1D : 32→64, kernel 3, pad 1  + BN + ReLU + Dropout(0.2)
    Conv1D : 64→64, kernel 3, pad 1  + BN + ReLU
    GAP    : AdaptiveAvgPool1d(1)    → (B, 64)
    FC     : 64 → 6
    Output : logits (6 classes)

Why 1D-CNN over GRU/LSTM:
  - Captures local temporal patterns via receptive field
  - Faster training and inference than recurrent models
  - 3 conv layers cover a receptive field of 7 frames = 1.4 seconds
  - ~20.7k parameters at 15 input channels (20,550 at 13) — ~90 KB ONNX
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .datasets.base_dataset import BEHAVIOR_CLASSES, FEATURE_DIM, WINDOW_SIZE

NUM_CLASSES = len(BEHAVIOR_CLASSES)


class BehaviorClassifierCNN(nn.Module):
    """
    Lightweight 1D-CNN for 15-frame behavioral window classification.

    Input tensor shape: (batch, feature_dim, WINDOW_SIZE) = (B, 15, 15) by default
    """

    def __init__(
        self,
        feature_dim: int  = FEATURE_DIM,
        window_size: int  = WINDOW_SIZE,
        num_classes: int  = NUM_CLASSES,
        dropout: float    = 0.2,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.num_classes = num_classes

        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv1d(feature_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Block 3
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)  # (B, 64, 1)
        self.classifier = nn.Linear(64, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, FEATURE_DIM, WINDOW_SIZE)

        Returns:
            logits: (batch, NUM_CLASSES)
        """
        x = self.conv_layers(x)       # (B, 64, W)
        x = self.pool(x).squeeze(-1)  # (B, 64)
        return self.classifier(x)     # (B, 6)

    # ── convenience ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls, ckpt_path: str, device: str = "cpu") -> "BehaviorClassifierCNN":
        """Load a saved checkpoint."""
        # weights_only=True: never unpickle arbitrary objects from a checkpoint
        # file (code execution); our checkpoints hold only tensors + plain
        # Python values (training/train_classifier.py), which this allows.
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        # Channel count from the weights, so 13-channel (pre-v3) checkpoints load.
        feature_dim = int(state["conv_layers.0.weight"].shape[1])
        model = cls(feature_dim=feature_dim)
        model.load_state_dict(state)
        model.eval()
        return model

    def predict_proba(self, x: Tensor) -> Tensor:
        """Return softmax probabilities. Input: (B, F, W) or (F, W)."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=-1)

    def predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Returns (class_indices, confidences).
        Input: (B, F, W) or (F, W) numpy/tensor.
        """
        if not isinstance(x, Tensor):
            x = torch.from_numpy(x)
        probs   = self.predict_proba(x)
        confs, indices = probs.max(dim=-1)
        return indices, confs
