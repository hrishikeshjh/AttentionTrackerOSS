"""
engagenet_loader.py — EngageNet-style frame-annotation loader.

The loader reads a frame-level annotation CSV (the format below; adapt an
EngageNet release to it once -- the public releases differ in layout) and
images extracted from the recordings:

    <root>/
        images/<relative image path>      e.g. images/S012/clip03/000123.jpg
        annotations.csv

CSV columns (header names; one row per frame):
    image | image_path          relative path under images/       (required)
    score | engagement_score    continuous engagement in [0, 1]   (one of
    label                       Not-Engaged / Barely-Engaged /     these two)
                                Engaged / Highly-Engaged
    subject | participant | subject_id   participant id        (recommended)
    clip | video | sequence     recording id                     (optional)
    timestamp_ms | frame        frame time (ms) or frame number  (optional)
    split                       train | val | test               (optional)

What EngageNet can and cannot ground (GROUNDED_CLASSES)
-------------------------------------------------------
Engagement only, webcam view, no objects -> the same three PROXY classes as
DAiSEE:
    score >= 0.50  / Engaged, Highly-Engaged -> attentive_screen
    score >= 0.25  / Barely-Engaged          -> distracted
    score <  0.25  / Not-Engaged             -> drowsy (weak proxy)
Never attentive_reading / using_phone / absent (windows that would be
"absent" on the server are dropped, see BaseBehaviorDataset).

Grouping / leakage: group_id = "engagenet:<participant>". The participant
comes from the subject column, else from the first directory of the image
path, else from the file-name prefix before "_" / "-". If none exists the
loader raises: grouping by row offset (the previous ``seq_start``) collided
across splits and let one participant land in train and val.

Temporal structure: rows are grouped into sequences per (participant, clip)
-- never across participants or clips -- and ordered by timestamp_ms /
frame (else CSV order). Timestamps: timestamp_ms, else frame * 1000 /
source_fps, else row order at source_fps. Frames are subsampled to the
server cadence (5 Hz) by timestamp and split at gaps; each sequence runs
through one TrackSequenceEncoder (same per-track measurement as the server).
"""

from __future__ import annotations

import csv
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .base_dataset import CLASS_TO_IDX, ENGAGEMENT_PROXY_CLASSES, WINDOW_SIZE, BaseBehaviorDataset
from .sequence_encoder import (
    TARGET_PERIOD_MS,
    LabelledSequence,
    TrackSequenceEncoder,
    observe_whole_frame,
    segment_sequence,
    subsample_by_timestamp,
)

logger = logging.getLogger("cognen.datasets.engagenet")

_SCORE_TO_CLASS = [
    (0.50, "attentive_screen"),
    (0.25, "distracted"),
    (0.00, "drowsy"),
]

_LABEL_TO_CLASS: dict[str, str] = {
    "highly-engaged": "attentive_screen",
    "engaged":        "attentive_screen",
    "barely-engaged": "distracted",
    "not-engaged":    "drowsy",
}

_PREFIX_RE = re.compile(r"^([A-Za-z0-9]+)[_-]")


def _score_to_label(score: float) -> int:
    for threshold, cls in _SCORE_TO_CLASS:
        if score >= threshold:
            return CLASS_TO_IDX[cls]
    return CLASS_TO_IDX["drowsy"]


def _text_label_to_idx(label: str) -> int:
    key = label.strip().lower().replace(" ", "-").replace("_", "-")
    if key not in _LABEL_TO_CLASS:
        raise ValueError(f"unknown EngageNet label {label!r} (expected one of {sorted(_LABEL_TO_CLASS)})")
    return CLASS_TO_IDX[_LABEL_TO_CLASS[key]]


def engagenet_subject_id(image: str, subject_field: str | None = None) -> str:
    """Participant id (see module docstring); ValueError if none can be derived."""
    if subject_field and subject_field.strip():
        return subject_field.strip()
    parts = Path(image).parts
    if len(parts) >= 2:
        return parts[0]
    m = _PREFIX_RE.match(Path(image).name)
    if m:
        return m.group(1)
    raise ValueError(
        f"cannot derive a participant id for {image!r}: add a 'subject' column or store images "
        f"as images/<participant>/...; grouping by row offset would leak participants across splits"
    )


@dataclass
class _Row:
    image: Path
    label: int
    subject: str
    clip: str
    ts_ms: float | None
    order: int


class EngageNetDataset(BaseBehaviorDataset):
    """
    Args:
        root:          EngageNet root.
        split:         "train" | "val" | "test" (filtered by the split column if present).
        window_size:   Frames per window (15 = 3 s at 5 Hz).
        stride:        Window stride in kept frames.
        max_images:    Cap on CSV rows read (None = all).
        csv_filename:  Annotation CSV name.
        source_fps:    Frame rate of the rows when the CSV has no timestamp_ms
                       (frame numbers / row order are converted with it).
        face_detector: Injected face landmarker (tests).
    """

    DATASET_NAME = "engagenet"
    GROUNDED_CLASSES = ENGAGEMENT_PROXY_CLASSES

    def __init__(
        self,
        root: str,
        split: str = "train",
        window_size: int = WINDOW_SIZE,
        stride: int = 5,
        max_images: int | None = None,
        csv_filename: str = "annotations.csv",
        source_fps: float = 5.0,
        face_detector=None,
        period_ms: float = TARGET_PERIOD_MS,
    ) -> None:
        super().__init__(root, split)
        self.window_size  = window_size
        self.WINDOW_SIZE  = window_size
        self.stride       = stride
        self.max_images   = max_images
        self.source_fps   = source_fps
        self.period_ms    = period_ms
        self._csv_path    = Path(root) / csv_filename
        self._image_dir   = Path(root) / "images"
        self._face_det    = face_detector

    # ── internal ─────────────────────────────────────────────────────────────

    def _load_rows(self) -> list[_Row]:
        rows: list[_Row] = []
        if not self._csv_path.exists():
            logger.warning("EngageNet CSV not found: %s", self._csv_path)
            return rows

        with open(self._csv_path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            has_split = "split" in fieldnames
            for i, row in enumerate(reader):
                if self.max_images is not None and i >= self.max_images:
                    break
                if has_split and (row.get("split") or "").strip() != self.split:
                    continue
                img = (row.get("image") or row.get("image_path") or "").strip()
                if not img:
                    continue
                if (row.get("label") or "").strip():
                    label = _text_label_to_idx(row["label"])
                else:
                    raw = row.get("score") or row.get("engagement_score")
                    if raw is None or not raw.strip():
                        raise ValueError(f"row {i}: no score / engagement_score / label")
                    label = _score_to_label(float(raw))
                subject = engagenet_subject_id(
                    img, row.get("subject") or row.get("participant") or row.get("subject_id"))
                clip = (row.get("clip") or row.get("video") or row.get("sequence") or "").strip() \
                    or str(Path(img).parent)
                ts: float | None = None
                if (row.get("timestamp_ms") or "").strip():
                    ts = float(row["timestamp_ms"])
                elif (row.get("frame") or "").strip():
                    ts = float(row["frame"]) * 1000.0 / self.source_fps
                rows.append(_Row(self._image_dir / img, label, subject, clip, ts, i))
        return rows

    def _get_face_detector(self):
        if self._face_det is None:
            from perception.face_landmarks import FaceLandmarksDetector
            self._face_det = FaceLandmarksDetector(max_faces=1)
        if not getattr(self._face_det, "available", False):
            raise RuntimeError(
                "face landmarker unavailable: EngageNet features cannot be extracted "
                f"(status={getattr(self._face_det, 'status', None)})"
            )
        return self._face_det

    def iter_sequences(self, classifier_factory=None) -> Iterator[LabelledSequence]:
        import cv2

        groups: OrderedDict[tuple[str, str], list[_Row]] = OrderedDict()
        for r in self._load_rows():
            groups.setdefault((r.subject, r.clip), []).append(r)

        period_src = 1000.0 / self.source_fps
        for (subject, clip), rows in groups.items():
            if any(r.ts_ms is None for r in rows):
                rows = sorted(rows, key=lambda r: r.order)
                ts = [k * period_src for k in range(len(rows))]
            else:
                rows = sorted(rows, key=lambda r: (r.ts_ms, r.order))
                ts = [float(r.ts_ms) for r in rows if r.ts_ms is not None]   # all set (checked above)
            keep = subsample_by_timestamp(ts, self.period_ms)

            det = self._get_face_detector()
            enc = TrackSequenceEncoder(classifier_factory() if classifier_factory else None)
            seq = LabelledSequence(stream_id=f"engagenet:{subject}:{clip}", group_id=f"engagenet:{subject}")
            for k in keep:
                frame = cv2.imread(str(rows[k].image))
                if frame is None:
                    # unreadable image: treat as a hole in the recording
                    logger.debug("unreadable image %s", rows[k].image)
                    continue
                obs = observe_whole_frame(det, frame)
                seq.frames.append(enc.step(ts[k], obs, face_detector_available=True, face_area=1.0))
                seq.labels.append(rows[k].label)
            yield from segment_sequence(seq, self.period_ms)
