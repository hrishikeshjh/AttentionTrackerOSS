"""
track_features.py — per-track face metrics, shared by inference and training.

ONE implementation of "face observation sequence -> FaceMetricsResult" for
one tracked person, used by

  * inference/room_analyzer.py (``RoomAnalyzer._analyze_person``), and
  * the dataset loaders (training/datasets/sequence_encoder.py), so that the
    PERCLOS / head-pose stability / fidget / gaze-corrected openness channels
    seen in training are computed exactly as on the server, from the frame
    sequence, instead of being constants.

The stateful pieces are the ones RoomAnalyzer keeps per track: the DACE
measurer (pitch reference, openness model, corrected PERCLOS), the blink
detector (raw EAR, blink rate), the head-pose tracker (stability, fidget) and
the fatigue calculator. ``TrackMetricsState.measure`` is the body that used to
live inline in ``RoomAnalyzer._analyze_person``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from features.dace_measurement import DaceTrackMeasurer, FaceObservation
from features.face_metrics import (
    BlinkDetector,
    FaceMetricsResult,
    FatigueCalculator,
    HeadPoseTracker,
    calculate_mar,
)


@dataclass
class TrackMetricsState:
    """Per-track measurement state (one instance per tracked person / clip)."""
    blink_detector: BlinkDetector     = field(default_factory=BlinkDetector)
    pose_tracker:   HeadPoseTracker   = field(default_factory=HeadPoseTracker)
    fatigue_calc:   FatigueCalculator = field(default_factory=FatigueCalculator)
    measurer:       DaceTrackMeasurer = field(default_factory=DaceTrackMeasurer)

    def measure(
        self,
        ts_ms: int,
        obs: FaceObservation | None,
        *,
        face_detector_available: bool = True,
        attentive: bool = False,
        face_area: float | None = None,
    ) -> FaceMetricsResult:
        """
        One frame of measurement.

        ts_ms:                   frame timestamp (ms); drives every time window
                                 (PERCLOS 60 s, pose 30 s, reference 5 s).
        obs:                     the face observation, or None (no face found).
        face_detector_available: False when the landmarker is not running
                                 (face_detected is then None = unknown, never
                                 counted as "no face").
        attentive:               the previous behavior label was attentive
                                 (gates openness-model samples; DACE).
        face_area:               presence / size proxy carried through to the
                                 classifier's single-frame "absent" rule.
        """
        face_detected: bool | None = (obs is not None) if face_detector_available else None

        # DACE measurement (features/dace_measurement.py): matrix pitch relative
        # to this track's screen-facing reference, blendshape gaze with iris
        # cross-check, gaze-corrected openness / PERCLOS. Values handed on are
        # in the feature convention (down = negative).
        meas = self.measurer.update(ts_ms, obs, attentive=attentive)
        if obs is not None:
            mar = calculate_mar(obs.landmarks68)
            pitch, yaw, roll = meas.pitch_feature, meas.yaw_deg, meas.roll_deg
            gaze_x, gaze_y = meas.gaze_x_feature, meas.gaze_y_feature
            ear, ear_raw = meas.ear_feature, meas.ear_raw
        else:
            ear = ear_raw = mar = yaw = pitch = roll = gaze_x = gaze_y = None

        # Blinks / blink rate on the raw EAR (fast closures); PERCLOS is the
        # gaze-corrected one, so looking down is not counted as eye closure.
        blink_det, blink_rate, _legacy_perclos = self.blink_detector.update(ear_raw, ts_ms)
        perclos = meas.perclos * 100.0
        self.pose_tracker.add(yaw, pitch, roll, ts_ms)
        ear_mean, ear_var = self.blink_detector.ear_stats()
        fatigue = self.fatigue_calc.update(perclos, blink_rate, ear_mean, ts_ms)

        return FaceMetricsResult(
            ear=ear, mar=mar,
            pitch=pitch, yaw=yaw, roll=roll,
            gaze_x=gaze_x, gaze_y=gaze_y,
            blink_detected=blink_det,
            blink_rate=blink_rate, perclos=perclos,
            fatigue_index=fatigue,
            fidget_score=self.pose_tracker.fidget(),
            head_pose_stability=self.pose_tracker.stability(),
            face_distance=None,
            face_area=face_area,
            mouth_openness=None,
            ear_variance=ear_var,
            pose_variance=self.pose_tracker.pose_variance(),
            face_detected=face_detected,
            ear_raw=ear_raw,
            looking_down=meas.looking_down if obs is not None else None,
            pitch_source=meas.pitch_source,
            gaze_source=meas.gaze_source,
        )


# ── Person-box geometry (shared by RoomAnalyzer and the classroom loader) ────

def clip_bbox(
    bbox_xyxy, frame_w: int, frame_h: int,
) -> tuple[int, int, int, int] | None:
    """Integer bbox clipped to the frame, or None if nothing is left.
    Negative tracker coordinates used to wrap around in numpy slicing
    (frame[-5:100] is empty or the wrong region)."""
    x1, y1, x2, y2 = bbox_xyxy
    x1 = int(max(0, min(frame_w, np.floor(x1))))
    y1 = int(max(0, min(frame_h, np.floor(y1))))
    x2 = int(max(0, min(frame_w, np.ceil(x2))))
    y2 = int(max(0, min(frame_h, np.ceil(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def face_roi(
    bbox_xyxy, frame_w: int, frame_h: int,
) -> tuple[int, int, int, int] | None:
    """
    Region of the person box handed to FaceLandmarker: the clipped box,
    truncated to its upper square (height <= width) so the head is a larger
    fraction of the crop. MediaPipe's face detector misses faces below
    roughly 10-12% of the crop's longer side (measured on a real image with
    mediapipe 0.10.31), which a full standing/seated person box often is.
    """
    clipped = clip_bbox(bbox_xyxy, frame_w, frame_h)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    return x1, y1, x2, min(y2, y1 + (x2 - x1))


def observe_face(detector, frame: np.ndarray, roi: tuple[int, int, int, int] | None) -> FaceObservation | None:
    """
    Run the face landmarker inside *roi* (clipped, integer xyxy) of *frame*.
    Returns a FaceObservation (68 points, mesh, transformation matrix,
    blendshapes; FRAME pixel coordinates) or None. Detectors without
    ``detect_full`` (68 points only) give an observation with no
    matrix/blendshapes, i.e. the labelled legacy fallback. Same call the
    server makes (RoomAnalyzer._get_face) and the loaders make.
    """
    if not getattr(detector, "available", False) or roi is None:
        return None
    x1, y1, x2, y2 = roi
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    detect_full = getattr(detector, "detect_full", None)
    if callable(detect_full):
        obs = detect_full(crop, offset=(x1, y1))
        return obs[0] if obs else None
    results = detector.detect(crop, offset=(x1, y1))
    return FaceObservation(landmarks68=results[0]) if results else None
