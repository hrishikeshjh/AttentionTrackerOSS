"""
daisee_loader.py — DAiSEE engagement video dataset loader.

DAiSEE (Dataset for Affective States in E-Environments)
  - 9,068 webcam clips of 112 participants, ~10 s each, recorded at 30 fps
  - 4 engagement levels: 0 = very low, 1 = low, 2 = high, 3 = very high
  - Also boredom, confusion, frustration labels (ignored here)

Download: https://iith.ac.in/~daisee-dataset/

Expected directory layout (as distributed):
    <root>/
        DataSet/
            Train/ Validation/ Test/
                <subject_id>/            6 digits, e.g. 110001
                    <clip_id>/           subject_id + 4 digits, e.g. 1100011002
                        <clip_id>.avi
        Labels/
            TrainLabels.csv  ValidationLabels.csv  TestLabels.csv
CSV columns: ClipID, Boredom, Engagement, Confusion, Frustration (0-3);
ClipID may carry the file extension ("1100011002.avi").

What DAiSEE can and cannot ground (GROUNDED_CLASSES)
----------------------------------------------------
Only an engagement level per clip, from a participant facing a laptop webcam
with no objects in view. It is used as a PROXY for three classes:
    3 / 2 (high, very high) -> attentive_screen
    1     (low)             -> distracted
    0     (very low)        -> drowsy   (the weakest proxy: "very low
                                         engagement" is not eye closure)
It never yields attentive_reading / using_phone (no book / phone context) or
absent (the participant is always recorded); windows whose face is missing
in >= ABSENT_MIN_NO_FACE_FRAMES of 15 frames are dropped
(BaseBehaviorDataset). Object channels are has_none = 1, hand proximity
"hands_not_visible", posture unknown -- what the server encodes when nothing
is detected.

Grouping / leakage: group_id = "daisee:<subject_id>". Every window of one
participant shares it, so no participant can sit on both sides of a split.
The subject comes from the directory layout, else from the first 6 digits of
the 10-digit ClipID.

Features: frames are decoded and subsampled to the server cadence (5 Hz by
timestamp, 15 frames = 3 s), each clip runs through one
TrackSequenceEncoder (the server's per-track measurement: DACE pitch
reference, gaze-corrected openness / PERCLOS, stability, fidget; see
training/datasets/sequence_encoder.py).
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Iterator

from .base_dataset import CLASS_TO_IDX, ENGAGEMENT_PROXY_CLASSES, WINDOW_SIZE, BaseBehaviorDataset
from .sequence_encoder import (
    TARGET_PERIOD_MS,
    LabelledSequence,
    TrackSequenceEncoder,
    iter_video_at_cadence,
    observe_whole_frame,
    segment_sequence,
)

logger = logging.getLogger("cognen.datasets.daisee")

# DAiSEE engagement → behavior class (proxy, see module docstring)
_ENGAGEMENT_TO_CLASS: dict[int, str] = {
    3: "attentive_screen",
    2: "attentive_screen",
    1: "distracted",
    0: "drowsy",
}

_SPLIT_DIR: dict[str, str] = {
    "train": "Train",
    "val":   "Validation",
    "test":  "Test",
}

_LABEL_FILE: dict[str, str] = {
    "train": "TrainLabels.csv",
    "val":   "ValidationLabels.csv",
    "test":  "TestLabels.csv",
}

_CLIP_ID_RE = re.compile(r"^(\d{6})\d{4}$")
_VIDEO_EXTS = (".avi", ".mp4", ".mkv", ".mov")


def clip_stem(clip_id: str) -> str:
    """'1100011002.avi' -> '1100011002'."""
    clip_id = clip_id.strip()
    for ext in _VIDEO_EXTS:
        if clip_id.lower().endswith(ext):
            return clip_id[: -len(ext)]
    return clip_id


def daisee_subject_id(clip_id: str, video_path: Path | None = None, split_dir: Path | None = None) -> str:
    """
    Participant id of a DAiSEE clip: the <subject_id> directory of the
    distributed layout (DataSet/<Split>/<subject>/<clip>/<clip>.avi), else
    the first 6 digits of the 10-digit ClipID. Raises ValueError when neither
    is available (grouping by clip would leak a participant across splits).
    """
    stem = clip_stem(clip_id)
    if video_path is not None and split_dir is not None:
        try:
            rel = video_path.relative_to(split_dir).parts
        except ValueError:
            rel = ()
        if len(rel) >= 3:              # <subject>/<clip>/<file>
            return rel[0]
    m = _CLIP_ID_RE.match(stem)
    if m:
        return m.group(1)
    raise ValueError(
        f"cannot derive a DAiSEE subject id from ClipID {clip_id!r} "
        f"(expected DataSet/<Split>/<subject>/<clip>/<file> or a 10-digit ClipID)"
    )


class DAiSEEDataset(BaseBehaviorDataset):
    """
    Args:
        root:          DAiSEE root.
        split:         "train" | "val" | "test" (the official split).
        window_size:   Frames per window (15 = 3 s at 5 Hz).
        stride:        Window stride in kept frames (5 = 1 s).
        max_clips:     Cap on clips (None = all).
        face_detector: Injected face landmarker (tests); default
                       perception.face_landmarks.FaceLandmarksDetector.
        period_ms:     Analysis cadence (200 ms = 5 Hz, the server's).
    """

    DATASET_NAME = "daisee"
    GROUNDED_CLASSES = ENGAGEMENT_PROXY_CLASSES

    def __init__(
        self,
        root: str,
        split: str = "train",
        window_size: int = WINDOW_SIZE,
        stride: int = 5,
        max_clips: int | None = None,
        face_detector=None,
        period_ms: float = TARGET_PERIOD_MS,
    ) -> None:
        super().__init__(root, split)
        self.window_size = window_size
        self.WINDOW_SIZE = window_size
        self.stride      = stride
        self.max_clips   = max_clips
        self.period_ms   = period_ms

        self._label_path = Path(root) / "Labels" / _LABEL_FILE[split]
        self._video_dir  = Path(root) / "DataSet" / _SPLIT_DIR[split]
        self._clip_labels = self._load_labels()
        self._face_det = face_detector

    # ── internal ─────────────────────────────────────────────────────────────

    def _load_labels(self) -> dict[str, int]:
        """Return {clip_id (no extension): engagement_label} from CSV."""
        labels: dict[str, int] = {}
        if not self._label_path.exists():
            logger.warning("DAiSEE label file not found: %s", self._label_path)
            return labels
        with open(self._label_path, newline="") as f:
            reader = csv.DictReader(f)
            # DAiSEE's CSV headers sometimes carry trailing spaces ("Frustration ")
            for row in reader:
                row = {(k or "").strip(): v for k, v in row.items()}
                labels[clip_stem(row["ClipID"])] = int(row["Engagement"])
        return labels

    def _get_face_detector(self):
        if self._face_det is None:
            from perception.face_landmarks import FaceLandmarksDetector
            self._face_det = FaceLandmarksDetector(max_faces=1)
        if not getattr(self._face_det, "available", False):
            raise RuntimeError(
                "face landmarker unavailable: DAiSEE features cannot be extracted "
                f"(status={getattr(self._face_det, 'status', None)})"
            )
        return self._face_det

    def _find_video(self, stem: str) -> Path | None:
        for ext in _VIDEO_EXTS:
            matches = sorted(self._video_dir.rglob(f"{stem}{ext}"))
            if matches:
                return matches[0]
        return None

    def iter_sequences(self, classifier_factory=None) -> Iterator[LabelledSequence]:
        """One LabelledSequence per contiguous run of a clip (labels constant per clip)."""
        clips_seen = 0
        for stem, engagement in self._clip_labels.items():
            if self.max_clips and clips_seen >= self.max_clips:
                break
            label_idx = CLASS_TO_IDX[_ENGAGEMENT_TO_CLASS[engagement]]
            video_path = self._find_video(stem)
            if video_path is None:
                continue
            subject = daisee_subject_id(stem, video_path, self._video_dir)
            det = self._get_face_detector()
            enc = TrackSequenceEncoder(classifier_factory() if classifier_factory else None)
            seq = LabelledSequence(stream_id=f"daisee:{stem}", group_id=f"daisee:{subject}")
            for _idx, ts, frame in iter_video_at_cadence(video_path, self.period_ms):
                obs = observe_whole_frame(det, frame)
                seq.frames.append(enc.step(ts, obs, face_detector_available=True, face_area=1.0))
                seq.labels.append(label_idx)
            clips_seen += 1
            yield from segment_sequence(seq, self.period_ms)
