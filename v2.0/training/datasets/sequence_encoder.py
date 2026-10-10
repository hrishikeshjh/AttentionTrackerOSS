"""
sequence_encoder.py — turn a timestamped face-observation sequence into the
per-frame feature vectors the v2 classifier sees on the server.

Why: the loaders used to call ``extract_feature_vector_v2_from_landmarks`` on
each frame in isolation. That produced the LEGACY 68-point pitch with a fixed
neutral offset (the server uses the matrix pitch relative to a per-track
reference), the raw EAR (the server uses the gaze-corrected openness),
PERCLOS / head-pose stability / fidget frozen at the encoder defaults (5 % /
90 / 5; the server computes them over time), ``np.zeros(15)`` for a frame
without a face (the server encodes such a frame with the defaults: EAR 0.3,
has_none 1, posture unknown, ...), and it ran at the video's native fps
(15 frames = 0.5 s at 30 fps; the server window is 15 frames at ~5 Hz = 3 s).

Now every loader goes through ``TrackSequenceEncoder``:

  * one ``features.track_features.TrackMetricsState`` per clip / person
    (the same object RoomAnalyzer keeps per track), fed the frame
    timestamps, so PERCLOS, stability, fidget, the DACE pitch reference and
    the openness model evolve exactly as on the server;
  * ``features.feature_vector.encode_frame_v2`` (the encoder
    BehaviorClassifier._extract uses) for every frame, with or without a face;
  * a BehaviorClassifier stepped on the same frames: its label is the
    "attentive" hint that gates the DACE openness model (RoomAnalyzer uses the
    previous label of the deployed classifier). Training loaders use the
    rule-based classifier (no model -> no circularity, label-free);
    evaluate.py passes the DEPLOYED classifier (ONNX + context gate), so its
    outputs are the deployed predictions.

Cadence: ``subsample_by_timestamp`` keeps frames on a 200 ms grid (5 Hz =
server TARGET_FPS, 15 frames = 3 s) using frame timestamps, not frame counts.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

import numpy as np

from features.dace_measurement import FaceObservation
from features.face_metrics import FaceMetricsResult
from features.feature_vector import encode_frame_v2
from features.track_features import TrackMetricsState, observe_face
from inference.behavior_classifier import BehaviorClassifier, BehaviorResult

logger = logging.getLogger("cognen.datasets.sequence")

TARGET_PERIOD_MS = 200.0          # 5 Hz: server TARGET_FPS, WINDOW_SIZE 15 = 3 s
# Consecutive kept frames further apart than this start a new segment (a
# window must not span a hole in the recording / annotation).
MAX_GAP_FACTOR = 2.5
_TS_TOL_MS = 1.0                  # float timestamps (33.333.. ms steps) land on the grid


# ── Cadence ───────────────────────────────────────────────────────────────────

def subsample_by_timestamp(
    timestamps_ms: Sequence[float],
    period_ms: float = TARGET_PERIOD_MS,
) -> list[int]:
    """
    Indices of the frames kept at a fixed cadence: the first frame, then the
    first frame at or after each grid point t0 + k * period. Timestamps must
    be non-decreasing. A source slower than the cadence keeps every frame.
    """
    keep: list[int] = []
    next_t: float | None = None
    prev: float | None = None
    for i, t in enumerate(timestamps_ms):
        t = float(t)
        if prev is not None and t < prev:
            raise ValueError(f"timestamps must be non-decreasing (index {i}: {t} < {prev})")
        prev = t
        if next_t is None:
            keep.append(i)
            next_t = t + period_ms
            continue
        if t + _TS_TOL_MS >= next_t:
            keep.append(i)
            # advance the grid past t (skips grid points a slow source missed)
            steps = math.floor((t + _TS_TOL_MS - next_t) / period_ms) + 1
            next_t += steps * period_ms
    return keep


def split_on_gaps(timestamps_ms: Sequence[float], period_ms: float = TARGET_PERIOD_MS) -> list[slice]:
    """Contiguous segments of kept frames (a gap > MAX_GAP_FACTOR x period splits)."""
    if len(timestamps_ms) == 0:
        return []
    segs, start = [], 0
    for i in range(1, len(timestamps_ms)):
        if timestamps_ms[i] - timestamps_ms[i - 1] > MAX_GAP_FACTOR * period_ms:
            segs.append(slice(start, i))
            start = i
    segs.append(slice(start, len(timestamps_ms)))
    return segs


def iter_video_at_cadence(
    path: str | Path,
    period_ms: float = TARGET_PERIOD_MS,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """
    Decode *path* and yield (frame_index, timestamp_ms, BGR frame) for the
    frames on the *period_ms* grid. Timestamps are frame_index / fps (the
    container fps); without a usable fps, CAP_PROP_POS_MSEC is used.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    try:
        next_t: float | None = None
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fps > 0:
                t = idx * 1000.0 / fps
            else:
                t = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if next_t is None or t + _TS_TOL_MS >= next_t:
                if next_t is None:
                    next_t = t + period_ms
                else:
                    steps = math.floor((t + _TS_TOL_MS - next_t) / period_ms) + 1
                    next_t += steps * period_ms
                yield idx, t, frame
            idx += 1
    finally:
        cap.release()


# ── Encoder ───────────────────────────────────────────────────────────────────

def as_detected_objects(objs: Sequence[Any]) -> list[Any]:
    """Annotation object names (str) -> objects with ``.cls`` like ObjectDetection."""
    out = []
    for o in objs or ():
        out.append(SimpleNamespace(cls=o) if isinstance(o, str) else o)
    return out


@dataclass
class EncodedFrame:
    ts_ms: int
    vector: np.ndarray            # (FEATURE_DIM,) float32, encode_frame_v2
    no_face: bool                 # landmarker ran and found no face
    result: BehaviorResult        # label of the stepped classifier
    metrics: FaceMetricsResult


class TrackSequenceEncoder:
    """
    Per-clip / per-person encoder (one instance = one server track).

    classifier: BehaviorClassifier stepped on every frame. None -> a fresh
                rule-based BehaviorClassifier (no model).
    """

    def __init__(self, classifier: BehaviorClassifier | None = None) -> None:
        self.state = TrackMetricsState()
        self.classifier = classifier if classifier is not None else BehaviorClassifier(session=None)
        self._last_label: str | None = None

    def step(
        self,
        ts_ms: float,
        obs: FaceObservation | None,
        *,
        face_detector_available: bool = True,
        face_area: float | None = 1.0,
        detected_objects: Sequence[Any] = (),
        hand_proximity: str = "hands_not_visible",
        posture: Any = None,
    ) -> EncodedFrame:
        ts = int(round(ts_ms))
        metrics = self.state.measure(
            ts, obs,
            face_detector_available=face_detector_available,
            attentive=self._last_label in ("attentive_screen", "attentive_reading"),
            face_area=face_area,
        )
        objs = as_detected_objects(detected_objects)
        frame = {"metrics": metrics, "detected_objects": objs,
                 "hand_proximity": hand_proximity, "posture": posture}
        vec = encode_frame_v2(metrics, objs, hand_proximity, posture)
        res = self.classifier.update(frame)
        self._last_label = res.label
        return EncodedFrame(ts, vec, metrics.face_detected is False, res, metrics)


@dataclass
class LabelledSequence:
    """A contiguous, cadence-regular run of encoded frames of ONE person."""
    stream_id: str                       # unique per clip / person / segment
    group_id: str                        # leakage group (subject / participant)
    frames: list[EncodedFrame] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)

    @property
    def vectors(self) -> np.ndarray:
        return np.stack([f.vector for f in self.frames]).astype(np.float32)

    @property
    def no_face(self) -> np.ndarray:
        return np.array([f.no_face for f in self.frames], dtype=bool)

    @property
    def timestamps_ms(self) -> np.ndarray:
        return np.array([f.ts_ms for f in self.frames], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.frames)


def segment_sequence(seq: LabelledSequence, period_ms: float = TARGET_PERIOD_MS) -> list[LabelledSequence]:
    """Split *seq* where consecutive frames are further apart than the gap limit."""
    out = []
    for k, sl in enumerate(split_on_gaps(seq.timestamps_ms.tolist(), period_ms)):
        out.append(LabelledSequence(
            stream_id=f"{seq.stream_id}#{k}", group_id=seq.group_id,
            frames=seq.frames[sl], labels=seq.labels[sl],
        ))
    return out


def observe_whole_frame(detector, frame: np.ndarray) -> FaceObservation | None:
    """Face observation for a frame that IS the person (webcam datasets)."""
    h, w = frame.shape[:2]
    return observe_face(detector, frame, (0, 0, w, h))
