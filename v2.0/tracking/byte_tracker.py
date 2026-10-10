"""
ByteTrackWrapper — multi-person tracker (ARCHITECTURE.md §3 Tracking Layer)

Wraps boxmot's ByteTrack implementation.
Input:  person bounding boxes + frame
Output: tracks with persistent track_id + bbox
~1 ms/frame on CPU (ARCHITECTURE.md §4).

boxmot API notes (pinned version: see server/requirements.txt):
  * boxmot 10.x exported ``BYTETracker`` only, so ``from boxmot import ByteTrack``
    raised ImportError and the stub below was silently used.
  * boxmot 16.x and 25.x export ``ByteTrack`` with the constructor
    ``ByteTrack(min_conf, track_thresh, match_thresh, track_buffer, frame_rate)``
    and ``update(dets[N,6] = x1,y1,x2,y2,conf,cls, img) -> [M,8]`` rows of
    ``x1,y1,x2,y2,track_id,conf,cls,det_ind``.
  * boxmot 17.x-18.x do not export ``ByteTrack`` at the package root; with
    those versions this wrapper falls back to the stub.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class Track:
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]   # pixel coords
    confidence: float


class ByteTrackWrapper:
    """
    Thin wrapper around boxmot.ByteTrack.
    Falls back to a stub if boxmot is not installed (or its API is incompatible).
    """

    def __init__(self, frame_rate: int = 15):
        self._tracker = None
        self._available = False
        try:
            from boxmot import ByteTrack
        except ImportError as exc:
            self._warn_stub(f"boxmot ByteTrack unavailable ({exc})")
            return
        try:
            self._tracker = ByteTrack(
                min_conf=0.1,
                track_thresh=0.25,
                match_thresh=0.8,
                track_buffer=30,
                frame_rate=frame_rate,
            )
            self._available = True
        except TypeError as exc:   # constructor signature changed in an unsupported version
            self._warn_stub(f"boxmot ByteTrack signature not supported ({exc})")

    @staticmethod
    def _warn_stub(reason: str) -> None:
        warnings.warn(
            f"[ByteTrackWrapper] {reason}. "
            "Install the pinned version from server/requirements.txt. Using stub tracker."
        )

    def update(
        self,
        detections: list,   # list of PersonDetection
        frame: np.ndarray,
    ) -> list[Track]:
        if not self._available or self._tracker is None:
            # Stub: assign sequential IDs with no tracking continuity
            return [
                Track(
                    track_id=i,
                    bbox_xyxy=d.bbox_xyxy,
                    confidence=d.confidence,
                )
                for i, d in enumerate(detections)
            ]

        # boxmot expects [x1, y1, x2, y2, conf, cls_id] as float32. An empty
        # (0, 6) array is still passed through so lost tracks keep ageing.
        if detections:
            dets = np.array(
                [[*d.bbox_xyxy, d.confidence, 0] for d in detections],
                dtype=np.float32,
            )
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        try:
            tracked = self._tracker.update(dets, frame)
        except Exception:
            if not detections:
                return []   # tolerate versions that reject empty input
            raise

        out = np.asarray(tracked)
        if out.ndim != 2 or out.shape[0] == 0 or out.shape[1] < 5:
            return []   # boxmot returns a 1-D empty array when nothing is tracked

        tracks: list[Track] = []
        for row in out:
            x1, y1, x2, y2, tid = (
                float(row[0]), float(row[1]),
                float(row[2]), float(row[3]),
                int(row[4]),
            )
            conf = float(row[5]) if out.shape[1] > 5 else 1.0
            tracks.append(Track(track_id=tid, bbox_xyxy=(x1, y1, x2, y2), confidence=conf))
        return tracks
