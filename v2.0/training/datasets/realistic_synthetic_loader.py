"""
realistic_synthetic_loader.py — Class-conditional synthetic dataset generator.

Replaces pure-random noise with statistically grounded feature windows that
encode the biometric distributions identified in experiment_report.md §3.1.

Each window is a 15-frame AR(1) temporal sequence, so the model can learn
true temporal patterns (sustained vs momentary behavior).

Feature vector (13 dims, index order fixed):
  [0]  gaze_x          [-1,  1]
  [1]  gaze_y          [-1,  1]
  [2]  pitch_norm      [-1,  1]   pitch/90
  [3]  yaw_norm        [-1,  1]   yaw/90
  [4]  EAR             [ 0,  1]
  [5]  PERCLOS         [ 0,  1]   /100
  [6]  head_stability  [ 0,  1]   /100
  [7]  has_book        {0, 1}
  [8]  has_phone       {0, 1}
  [9]  has_laptop      {0, 1}
  [10] has_none        {0, 1}
  [11] fidget          [ 0,  1]   /100
  [12] proximity_code  {0.00, 0.25, 0.50, 0.75, 1.00}
"""

from __future__ import annotations
import numpy as np
from typing import Generator

from training.datasets.base_dataset import (
    BaseBehaviorDataset,
    BEHAVIOR_CLASSES,
    CLASS_TO_IDX,
    FEATURE_DIM,
    WINDOW_SIZE,
)

# ── Per-class feature distributions ───────────────────────────────────────────
# Each class has (mean, std) for continuous features and rules for binary features.
# Distributions derived from:
#   - experiment_report.md §3.1 overlap table
#   - Known biometric ranges (EAR thresholds, PERCLOS norms)
#   - Classroom observation literature

# Feature indices
_GX   = 0   # gaze_x
_GY   = 1   # gaze_y
_PNM  = 2   # pitch_norm (pitch/90)
_YNM  = 3   # yaw_norm (yaw/90)
_EAR  = 4   # eye aspect ratio
_PERC = 5   # perclos /100
_STAB = 6   # head_stability /100
_BOOK = 7   # has_book
_PHN  = 8   # has_phone
_LAP  = 9   # has_laptop
_NONE = 10  # has_none
_FID  = 11  # fidget /100
_PROX = 12  # proximity_code

# Temporal autocorrelation per feature (higher = smoother transitions)
_RHO: np.ndarray = np.array([
    0.70,  # gaze_x
    0.70,  # gaze_y
    0.85,  # pitch_norm  — head pose changes slowly
    0.80,  # yaw_norm
    0.60,  # EAR         — blinks add rapid variation
    0.90,  # PERCLOS     — computed over rolling window → very smooth
    0.88,  # stability
    1.00,  # has_book    — binary, held constant per window
    1.00,  # has_phone   — binary, held constant per window
    1.00,  # has_laptop  — binary, held constant per window
    1.00,  # has_none    — binary, held constant per window
    0.65,  # fidget      — movement varies quickly
    1.00,  # proximity   — set at window start, held
], dtype=np.float32)


class _ClassSpec:
    """
    Parameterises the generative distribution for one behavior class.
    """
    __slots__ = (
        "gaze_x_mu", "gaze_x_sd",
        "gaze_y_mu", "gaze_y_sd",
        "pitch_mu",  "pitch_sd",
        "yaw_mu",    "yaw_sd",
        "ear_mu",    "ear_sd",
        "perc_mu",   "perc_sd",
        "stab_mu",   "stab_sd",
        "fidget_mu", "fidget_sd",
        "p_book", "p_phone", "p_laptop",
        "prox_when_book", "prox_when_phone", "prox_baseline",
        "noise_scale",
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# Class distributions — carefully calibrated to match biometric observations
_SPECS: dict[str, _ClassSpec] = {
    "attentive_screen": _ClassSpec(
        gaze_x_mu=0.00, gaze_x_sd=0.12,
        gaze_y_mu=0.00, gaze_y_sd=0.10,
        pitch_mu=-0.03, pitch_sd=0.05,   # ≈ −3°
        yaw_mu=0.00,    yaw_sd=0.08,
        ear_mu=0.32,    ear_sd=0.03,
        perc_mu=0.03,   perc_sd=0.015,
        stab_mu=0.87,   stab_sd=0.07,
        fidget_mu=0.10, fidget_sd=0.06,
        p_book=0.00,  p_phone=0.00,  p_laptop=0.50,
        prox_when_book=0.75,  prox_when_phone=1.00,  prox_baseline=0.25,
        noise_scale=0.03,
    ),
    "attentive_reading": _ClassSpec(
        gaze_x_mu=0.00, gaze_x_sd=0.10,
        gaze_y_mu=0.65, gaze_y_sd=0.08,   # looking down (normalised)
        pitch_mu=-0.37, pitch_sd=0.05,    # ≈ −33° sustained downward
        yaw_mu=0.00,    yaw_sd=0.06,
        ear_mu=0.30,    ear_sd=0.02,
        perc_mu=0.04,   perc_sd=0.015,
        stab_mu=0.82,   stab_sd=0.08,
        fidget_mu=0.08, fidget_sd=0.05,
        p_book=0.82,  p_phone=0.00,  p_laptop=0.10,
        prox_when_book=0.75,  prox_when_phone=1.00,  prox_baseline=0.25,
        noise_scale=0.025,
    ),
    "using_phone": _ClassSpec(
        gaze_x_mu=0.05, gaze_x_sd=0.10,
        gaze_y_mu=0.68, gaze_y_sd=0.09,   # looking down at phone
        pitch_mu=-0.42, pitch_sd=0.05,    # ≈ −38° (slightly more than reading)
        yaw_mu=0.00,    yaw_sd=0.08,
        ear_mu=0.30,    ear_sd=0.025,
        perc_mu=0.04,   perc_sd=0.020,
        stab_mu=0.76,   stab_sd=0.10,
        fidget_mu=0.28, fidget_sd=0.10,   # higher: thumb scrolling
        p_book=0.00,  p_phone=0.87,  p_laptop=0.00,
        prox_when_book=0.75,  prox_when_phone=1.00,  prox_baseline=0.25,
        noise_scale=0.03,
    ),
    "distracted": _ClassSpec(
        gaze_x_mu=0.00, gaze_x_sd=0.32,   # wide gaze deviation
        gaze_y_mu=0.00, gaze_y_sd=0.22,
        pitch_mu=0.05,  pitch_sd=0.15,    # variable, can look sideways
        yaw_mu=0.00,    yaw_sd=0.20,      # turning head side-to-side
        ear_mu=0.31,    ear_sd=0.03,
        perc_mu=0.05,   perc_sd=0.02,
        stab_mu=0.48,   stab_sd=0.15,     # low stability
        fidget_mu=0.38, fidget_sd=0.12,   # high fidget
        p_book=0.00,  p_phone=0.00,  p_laptop=0.20,
        prox_when_book=0.75,  prox_when_phone=1.00,  prox_baseline=0.00,
        noise_scale=0.05,
    ),
    "drowsy": _ClassSpec(
        gaze_x_mu=0.00, gaze_x_sd=0.07,
        gaze_y_mu=0.02, gaze_y_sd=0.07,
        pitch_mu=-0.05, pitch_sd=0.08,
        yaw_mu=0.00,    yaw_sd=0.05,
        ear_mu=0.18,    ear_sd=0.04,      # EAR < 0.22 → drowsy threshold
        perc_mu=0.32,   perc_sd=0.08,     # PERCLOS > 20% → drowsy threshold
        stab_mu=0.58,   stab_sd=0.13,     # head nodding
        fidget_mu=0.04, fidget_sd=0.03,
        p_book=0.00,  p_phone=0.00,  p_laptop=0.05,
        prox_when_book=0.75,  prox_when_phone=1.00,  prox_baseline=0.00,
        noise_scale=0.02,
    ),
    "absent": _ClassSpec(
        gaze_x_mu=0.00, gaze_x_sd=0.02,   # no face → near-zero
        gaze_y_mu=0.00, gaze_y_sd=0.02,
        pitch_mu=0.00,  pitch_sd=0.01,
        yaw_mu=0.00,    yaw_sd=0.01,
        ear_mu=0.30,    ear_sd=0.01,       # default EAR when no face
        perc_mu=0.00,   perc_sd=0.005,
        stab_mu=0.00,   stab_sd=0.01,
        fidget_mu=0.00, fidget_sd=0.01,
        p_book=0.00,  p_phone=0.00,  p_laptop=0.10,
        prox_when_book=0.75,  prox_when_phone=1.00,  prox_baseline=0.00,
        noise_scale=0.01,
    ),
}


def _clamp(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(v, lo, hi)


def _sample_binary_context(spec: _ClassSpec, rng: np.random.Generator) -> np.ndarray:
    """Sample fixed binary context features for a window (held constant per window)."""
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)

    has_phone  = int(rng.random() < spec.p_phone)
    has_book   = int(rng.random() < spec.p_book)  if not has_phone else 0
    has_laptop = int(rng.random() < spec.p_laptop) if not (has_phone or has_book) else 0
    has_none   = int(not (has_phone or has_book or has_laptop))

    feat[_BOOK] = has_book
    feat[_PHN]  = has_phone
    feat[_LAP]  = has_laptop
    feat[_NONE] = has_none

    if has_phone:
        feat[_PROX] = spec.prox_when_phone
    elif has_book:
        feat[_PROX] = spec.prox_when_book
    else:
        # Randomly choose: hands_not_visible=0.0, hands_on_desk=0.25, holding_pen=0.50
        feat[_PROX] = float(rng.choice([0.0, 0.25, 0.50]))

    return feat


def _generate_window(
    cls_name: str,
    rng: np.random.Generator,
    window_size: int = WINDOW_SIZE,
) -> np.ndarray:
    """
    Generate a single (window_size, FEATURE_DIM) window via AR(1) process.

    Frame 0: sampled from class distribution.
    Frame t: rho * frame_{t-1} + (1-rho) * mean + noise
    Binary context features (book, phone, laptop, none, proximity) are held
    constant for the entire window (realistic — objects don't disappear mid-window).
    """
    spec = _SPECS[cls_name]

    # Stationary means for continuous features
    means = np.array([
        spec.gaze_x_mu, spec.gaze_y_mu,
        spec.pitch_mu,  spec.yaw_mu,
        spec.ear_mu,    spec.perc_mu,
        spec.stab_mu,
        0.0, 0.0, 0.0, 0.0,          # binary — handled separately
        spec.fidget_mu, 0.0,          # proximity handled separately
    ], dtype=np.float32)

    stds = np.array([
        spec.gaze_x_sd, spec.gaze_y_sd,
        spec.pitch_sd,  spec.yaw_sd,
        spec.ear_sd,    spec.perc_sd,
        spec.stab_sd,
        0.0, 0.0, 0.0, 0.0,
        spec.fidget_sd, 0.0,
    ], dtype=np.float32)

    # Sample fixed binary context once for entire window
    context = _sample_binary_context(spec, rng)

    # Initialise first frame from class distribution
    frames = np.zeros((window_size, FEATURE_DIM), dtype=np.float32)
    frame0 = rng.normal(means, stds).astype(np.float32)
    frame0[_BOOK] = context[_BOOK]
    frame0[_PHN]  = context[_PHN]
    frame0[_LAP]  = context[_LAP]
    frame0[_NONE] = context[_NONE]
    frame0[_PROX] = context[_PROX]
    frames[0] = frame0

    # AR(1) propagation for subsequent frames
    noise_std = stds * spec.noise_scale + 0.008  # floor noise
    for t in range(1, window_size):
        noise = rng.normal(0, noise_std).astype(np.float32)
        cont  = _RHO * frames[t - 1] + (1 - _RHO) * means + noise
        # Inject blink artefact (EAR dip for 1–2 frames, p=0.05 per frame)
        if rng.random() < 0.05:
            cont[_EAR] = float(rng.uniform(0.12, 0.20))
        # Re-stamp binary context (AR(1) would corrupt them otherwise)
        cont[_BOOK] = context[_BOOK]
        cont[_PHN]  = context[_PHN]
        cont[_LAP]  = context[_LAP]
        cont[_NONE] = context[_NONE]
        cont[_PROX] = context[_PROX]
        frames[t] = cont

    # Clamp continuous features to valid ranges
    frames[:, _GX]   = _clamp(frames[:, _GX],   -1.0,  1.0)
    frames[:, _GY]   = _clamp(frames[:, _GY],   -1.0,  1.0)
    frames[:, _PNM]  = _clamp(frames[:, _PNM],  -1.0,  1.0)
    frames[:, _YNM]  = _clamp(frames[:, _YNM],  -1.0,  1.0)
    frames[:, _EAR]  = _clamp(frames[:, _EAR],   0.0,  1.0)
    frames[:, _PERC] = _clamp(frames[:, _PERC],  0.0,  1.0)
    frames[:, _STAB] = _clamp(frames[:, _STAB],  0.0,  1.0)
    frames[:, _FID]  = _clamp(frames[:, _FID],   0.0,  1.0)

    return frames


# ── Temporal augmentation ─────────────────────────────────────────────────────

def _augment_time_warp(window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Time-warp: resample the temporal axis at ±20% speed.
    Simulates a student performing an action faster or slower than average.
    """
    w = window.shape[0]
    factor = rng.uniform(0.8, 1.2)
    src_len = max(2, int(w * factor))
    # Upsample/downsample source via linear interpolation, then crop/pad to w
    src_indices = np.linspace(0, w - 1, src_len)
    tgt_indices = np.linspace(0, src_len - 1, w)
    warped = np.zeros_like(window)
    for f in range(window.shape[1]):
        warped[:, f] = np.interp(tgt_indices, np.arange(src_len),
                                  np.interp(src_indices, np.arange(w), window[:, f]))
    return warped


def _augment_feature_dropout(window: np.ndarray, rng: np.random.Generator, p: float = 0.08) -> np.ndarray:
    """
    Randomly zero out continuous features for individual frames.
    Simulates sensor glitches or brief occlusion.
    Binary context features (7–12) are never dropped.
    """
    mask = rng.random((window.shape[0], 7)) > p
    result = window.copy()
    result[:, :7] *= mask
    return result


def _augment_gaze_jitter(window: np.ndarray, rng: np.random.Generator, sigma: float = 0.05) -> np.ndarray:
    """Add Gaussian noise to gaze estimates (simulates head-mounted vs remote gaze error)."""
    result = window.copy()
    result[:, _GX] = np.clip(result[:, _GX] + rng.normal(0, sigma, window.shape[0]), -1, 1)
    result[:, _GY] = np.clip(result[:, _GY] + rng.normal(0, sigma, window.shape[0]), -1, 1)
    return result


def _augment_object_context_injection(
    window: np.ndarray,
    cls_name: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    For phone/reading windows with no object detected, randomly inject
    the correct object context. This trains the model to rely on object
    context when it is available, and to be uncertain when it is not.
    """
    result = window.copy()
    if cls_name == "using_phone" and result[0, _PHN] == 0 and rng.random() < 0.35:
        result[:, _PHN]  = 1.0
        result[:, _NONE] = 0.0
        result[:, _PROX] = 1.00  # holding_phone
    elif cls_name == "attentive_reading" and result[0, _BOOK] == 0 and rng.random() < 0.30:
        result[:, _BOOK] = 1.0
        result[:, _NONE] = 0.0
        result[:, _PROX] = 0.75  # holding_book
    return result


def augment_window(
    window: np.ndarray,
    cls_name: str,
    rng: np.random.Generator,
    p_warp: float = 0.40,
    p_dropout: float = 0.50,
    p_jitter: float = 0.60,
    p_inject: float = 0.30,
) -> np.ndarray:
    """Apply a random subset of augmentations to a window."""
    w = window.copy()
    if rng.random() < p_warp:
        w = _augment_time_warp(w, rng)
    if rng.random() < p_dropout:
        w = _augment_feature_dropout(w, rng)
    if rng.random() < p_jitter:
        w = _augment_gaze_jitter(w, rng)
    if rng.random() < p_inject:
        w = _augment_object_context_injection(w, cls_name, rng)
    return w


# ── Dataset class ─────────────────────────────────────────────────────────────

class RealisticSyntheticDataset(BaseBehaviorDataset):
    """
    Class-conditional synthetic dataset with temporal autocorrelation.

    Generates `n_per_class` base windows per class, then creates
    `augment_factor` augmented copies of each. Total windows:
        n_per_class * 6 * (1 + augment_factor)

    Args:
        n_per_class:    Base windows per behavior class (default 2000).
        augment_factor: Additional augmented copies per base window (default 3).
        seed:           RNG seed for reproducibility.
        split:          "train" or "val" — split is done deterministically
                        by reserving the last `val_fraction` of each class.
        val_fraction:   Fraction held out for validation (default 0.15).
        window_size:    Temporal window length (default WINDOW_SIZE=15).
    """

    def __init__(
        self,
        root: str = "",            # unused (no files), kept for API compat
        split: str = "train",
        n_per_class: int = 2000,
        augment_factor: int = 3,
        seed: int = 42,
        val_fraction: float = 0.15,
        window_size: int = WINDOW_SIZE,
    ) -> None:
        super().__init__(root, split)
        self.n_per_class    = n_per_class
        self.augment_factor = augment_factor
        self.seed           = seed
        self.val_fraction   = val_fraction
        self.window_size    = window_size

        self._windows, self._labels = self._build_dataset()

    def _build_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        n   = self.n_per_class

        all_windows: list[np.ndarray] = []
        all_labels:  list[int]        = []

        for cls_name in BEHAVIOR_CLASSES:
            cls_idx = CLASS_TO_IDX[cls_name]
            n_val   = max(1, int(n * self.val_fraction))
            n_train = n - n_val

            # Generate base windows for this class
            base_wins = [_generate_window(cls_name, rng, self.window_size)
                         for _ in range(n)]

            if self.split == "val":
                selected = base_wins[-n_val:]   # last n_val are val
            else:
                selected = base_wins[:n_train]  # first n_train are train

            # Add base windows
            all_windows.extend(selected)
            all_labels.extend([cls_idx] * len(selected))

            # Add augmented copies (train only)
            if self.split == "train":
                for base_win in selected:
                    for _ in range(self.augment_factor):
                        aug = augment_window(base_win, cls_name, rng)
                        all_windows.append(aug)
                        all_labels.append(cls_idx)

        # Shuffle so classes interleave
        idx = rng.permutation(len(all_windows))
        windows = np.stack(all_windows)[idx].astype(np.float32)
        labels  = np.array(all_labels, dtype=np.int64)[idx]
        return windows, labels

    def __len__(self) -> int:
        return len(self._labels)

    def _iter_samples(self) -> Generator[tuple[np.ndarray, int, object], None, None]:
        # group_id=None: each synthetic window is independently generated,
        # no real subject/clip continuity to leak across a split.
        for window, label in zip(self._windows, self._labels):
            yield window, int(label), None

    def class_distribution(self) -> dict[str, int]:
        """Return count per class label."""
        counts = {}
        for i, cls in enumerate(BEHAVIOR_CLASSES):
            counts[cls] = int((self._labels == i).sum())
        return counts
