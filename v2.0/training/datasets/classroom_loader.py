"""
classroom_loader.py — Custom classroom CCTV dataset loader.

Expected dataset layout:
    <root>/
        videos/
            <clip_id>.mp4          (or .avi / .mkv)
        annotations/
            <clip_id>.json         per-clip annotation file
        metadata.csv               index of all clips

Annotation JSON schema per clip (training/datasets/annotation_tools/
cvat_to_classroom_json.py writes it):
    {
      "clip_id": "clip_0001",
      "fps": 5,
      "persons": [
        {
          "track_id": 1,
          "subject_id": "student_17",       optional, see grouping
          "seat_id": "seat_3",
          "frames": [
            {
              "frame_idx": 0,                index into the VIDEO's frames
              "bbox_xyxy": [x1, y1, x2, y2], video pixel coordinates
              "behavior": "attentive_screen",  null = unlabelled (tracking only)
              "detected_objects": ["book"],  optional
              "hand_proximity": "holding_book"  optional
            },
            ...
          ]
        }
      ],
      "n_frames": 150                      optional: frames 0..n-1 all annotated
    }

metadata.csv format: clip_id, split (train|val|test), duration_sec, num_persons

Pipeline (same per-person path as inference/room_analyzer.py): the frame is
resized to the server input size (INPUT_WIDTH x INPUT_HEIGHT) and the box
scaled with it, the face landmarker runs on face_roi (upper square of the
clipped person box), and one TrackSequenceEncoder per annotated person keeps
the per-track state (DACE pitch reference, gaze-corrected openness / PERCLOS,
stability, fidget). Annotated frames are taken on the 5 Hz grid by video
timestamp; sequences are split at annotation gaps.

Grounding: annotators label all six classes, so GROUNDED_CLASSES is the full
taxonomy (an unknown behavior string raises).

Grouping: group_id = "classroom:<subject_id>" when the annotation has a
subject / student id (the same student across clips), else
"classroom:<clip_id>" (every person of one clip in one group: two tracks of
one clip share scene, lighting and often the same student re-identified).
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Iterator

from .base_dataset import CLASS_TO_IDX, WINDOW_SIZE, BaseBehaviorDataset
from .sequence_encoder import (
    TARGET_PERIOD_MS,
    LabelledSequence,
    TrackSequenceEncoder,
    segment_sequence,
    subsample_by_timestamp,
)

logger = logging.getLogger("cognen.datasets.classroom")

_VIDEO_EXTS = (".mp4", ".avi", ".mkv")


def find_clip_video(root: Path, clip_id: str) -> Path | None:
    for ext in _VIDEO_EXTS:
        p = root / "videos" / f"{clip_id}{ext}"
        if p.exists():
            return p
    return None


def load_split_clip_ids(root: str | Path, split: str) -> list[str]:
    """clip ids of *split* from metadata.csv (the ONLY source of the split)."""
    meta = Path(root) / "metadata.csv"
    clip_ids: list[str] = []
    if not meta.exists():
        logger.warning("Classroom metadata.csv not found: %s", meta)
        return clip_ids
    with open(meta, newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("split") or "").strip() == split:
                clip_ids.append(row["clip_id"].strip())
    return clip_ids


def person_subject_id(clip_id: str, person: dict) -> str:
    for key in ("subject_id", "student_id", "person_id"):
        v = person.get(key)
        if v not in (None, ""):
            return f"classroom:{v}"
    return f"classroom:{clip_id}"


class ClassroomDataset(BaseBehaviorDataset):
    """
    Args:
        root:          Dataset root directory.
        split:         "train" | "val" | "test" (from metadata.csv).
        window_size:   Frames per window (15 = 3 s at 5 Hz).
        stride:        Window stride in kept frames.
        max_clips:     Cap on number of clips (None = all).
        face_detector: Injected face landmarker (tests).
    """

    DATASET_NAME = "classroom"

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
        self._face_det   = face_detector

    def _load_metadata(self) -> list[str]:
        return load_split_clip_ids(self.root, self.split)

    def _get_face_detector(self):
        if self._face_det is None:
            from perception.face_landmarks import FaceLandmarksDetector
            self._face_det = FaceLandmarksDetector(max_faces=1)
        return self._face_det

    def iter_sequences(self, classifier_factory=None) -> Iterator[LabelledSequence]:
        """
        One LabelledSequence per annotated person and contiguous segment.
        classifier_factory: builds the BehaviorClassifier stepped per person
        (evaluate.py passes the deployed one; None = rule-based hint).
        """
        import cv2

        from features.track_features import face_roi, observe_face
        from server.config import INPUT_HEIGHT, INPUT_WIDTH

        root = Path(self.root)
        clip_ids = self._load_metadata()
        if self.max_clips:
            clip_ids = clip_ids[: self.max_clips]
        det = self._get_face_detector()
        det_available = bool(getattr(det, "available", False))
        if not det_available:
            logger.error("face landmarker unavailable: face_detected is unknown for every frame")

        for clip_id in clip_ids:
            ann_path = root / "annotations" / f"{clip_id}.json"
            video_path = find_clip_video(root, clip_id)
            if not ann_path.exists() or video_path is None:
                logger.debug("Skipping clip %s — missing video or annotation", clip_id)
                continue
            with open(ann_path) as f:
                ann = json.load(f)

            cap = cv2.VideoCapture(str(video_path))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or float(ann.get("fps") or 0.0)
            if fps <= 0:
                cap.release()
                raise ValueError(f"clip {clip_id}: unknown video fps")

            # Per person: annotated frames (valid behavior) on the 5 Hz grid.
            wanted: dict[int, list[tuple[int, dict]]] = {}
            persons = ann.get("persons", [])
            for p_idx, person in enumerate(persons):
                # frames with behavior null are unlabelled (tracking-only boxes)
                fr = sorted((fd for fd in person.get("frames", [])
                             if "frame_idx" in fd and fd.get("behavior") is not None),
                            key=lambda fd: fd["frame_idx"])
                for fd in fr:
                    if fd["behavior"] not in CLASS_TO_IDX:
                        raise ValueError(f"clip {clip_id}: unknown behavior {fd['behavior']!r}")
                keep = subsample_by_timestamp([fd["frame_idx"] * 1000.0 / fps for fd in fr], self.period_ms)
                for k in keep:
                    wanted.setdefault(int(fr[k]["frame_idx"]), []).append((p_idx, fr[k]))

            encoders = {i: TrackSequenceEncoder(classifier_factory() if classifier_factory else None)
                        for i in range(len(persons))}
            seqs = {
                i: LabelledSequence(
                    stream_id=f"classroom:{clip_id}:{persons[i].get('track_id', i)}",
                    group_id=person_subject_id(clip_id, persons[i]),
                )
                for i in range(len(persons))
            }

            fi = 0
            last_needed = max(wanted) if wanted else -1
            while fi <= last_needed:
                ok, frame = cap.read()
                if not ok:
                    break
                if fi in wanted:
                    vh, vw = frame.shape[:2]
                    sx, sy = INPUT_WIDTH / vw, INPUT_HEIGHT / vh
                    resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
                    for p_idx, fd in wanted[fi]:
                        x1, y1, x2, y2 = (float(v) for v in fd.get("bbox_xyxy", [0, 0, 0, 0]))
                        box = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
                        roi = face_roi(box, INPUT_WIDTH, INPUT_HEIGHT)
                        obs = observe_face(det, resized, roi)
                        cw = max(0.0, min(box[2], INPUT_WIDTH) - max(box[0], 0.0))
                        ch = max(0.0, min(box[3], INPUT_HEIGHT) - max(box[1], 0.0))
                        area = (cw * ch) / float(INPUT_WIDTH * INPUT_HEIGHT)
                        seqs[p_idx].frames.append(encoders[p_idx].step(
                            fi * 1000.0 / fps, obs,
                            face_detector_available=det_available,
                            face_area=area,
                            detected_objects=fd.get("detected_objects", []) or [],
                            hand_proximity=fd.get("hand_proximity", "hands_not_visible"),
                        ))
                        seqs[p_idx].labels.append(CLASS_TO_IDX[fd["behavior"]])
                fi += 1
            cap.release()

            for i in range(len(persons)):
                yield from segment_sequence(seqs[i], self.period_ms)
