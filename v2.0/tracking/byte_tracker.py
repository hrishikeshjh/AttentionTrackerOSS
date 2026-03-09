"""
ByteTrackWrapper — multi-person tracker (ARCHITECTURE.md §3 Tracking Layer)

Wraps boxmot's ByteTrack implementation.
Input:  person bounding boxes + frame
Output: tracks with persistent track_id + bbox
~1 ms/frame on CPU (ARCHITECTURE.md §4).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class Track:
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]   # pixel coords
    confidence: float


class ByteTrackWrapper:
    """
    Thin wrapper around boxmot.ByteTrack.
    Falls back to a stub if boxmot is not installed.
    """

    def __init__(self, frame_rate: int = 15):
        try:
            from boxmot import ByteTrack
            self._tracker = ByteTrack(
                track_thresh=0.25,
                track_buffer=30,
                match_thresh=0.8,
                frame_rate=frame_rate,
            )
            self._available = True
        except ImportError:
            self._tracker = None
            self._available = False
            import warnings
            warnings.warn(
                "[ByteTrackWrapper] boxmot not installed. "
                "Install with: pip install boxmot. Using stub tracker."
            )

    def update(
        self,
        detections: list,   # list of PersonDetection
        frame: np.ndarray,
    ) -> list[Track]:
        if not self._available:
            # Stub: assign sequential IDs with no tracking continuity
            return [
                Track(
                    track_id=i,
                    bbox_xyxy=d.bbox_xyxy,
                    confidence=d.confidence,
                )
                for i, d in enumerate(detections)
            ]

        if not detections:
            return []

        # boxmot expects [x1, y1, x2, y2, conf, cls_id] as float32
        dets = np.array(
            [[*d.bbox_xyxy, d.confidence, 0] for d in detections],
            dtype=np.float32,
        )
        tracked = self._tracker.update(dets, frame)

        tracks: list[Track] = []
        for row in tracked:
            x1, y1, x2, y2, tid = (
                float(row[0]), float(row[1]),
                float(row[2]), float(row[3]),
                int(row[4]),
            )
            tracks.append(Track(track_id=tid, bbox_xyxy=(x1, y1, x2, y2), confidence=1.0))
        return tracks
