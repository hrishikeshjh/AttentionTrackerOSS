"""
RoomAnalyzer — main orchestrator for v2.0 multi-person analysis.
(ARCHITECTURE.md §5.2 + §9 v2.0 Pipeline)

Instantiates all perception/tracking/feature/inference modules and
drives the per-frame loop described in ARCHITECTURE.md §5.2.

Frame sources
-------------
Use ``run_source(source, process_every_n)`` for full control over the
input type (file / RTSP / camera).  ``run_stream(url)`` is kept for
backwards compatibility and wraps an RTSPSource internally.

Threading
---------
``source.open()``, ``source.read()`` and the whole ``_process_frame`` (YOLO x3,
tracking, FaceLandmarker, classifier) run in a worker thread via
``asyncio.to_thread`` so one room's inference does not stall the event loop
(other rooms, REST, WebSocket pushes). Frames of one room are still
processed strictly one after another.

Heavy dependencies (cv2, ultralytics, shapely) are imported lazily, and every
component can be injected through the constructor, so the per-frame logic is
unit-testable without them (tests/test_room_analyzer.py).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from features.dace_measurement import FaceObservation
from features.hand_object import proximity_from_hands
from features.hand_proximity import estimate_hand_proximity
from features.track_features import (  # noqa: F401 (clip_bbox / face_roi re-exported)
    TrackMetricsState,
    clip_bbox,
    face_roi,
    observe_face,
)
from inference.behavior_classifier import BehaviorClassifier, BehaviorResult, load_onnx_session
from inference.contradiction_detector import ContradictionDetector
from input.frame_source import FrameSource, RTSPSource
from server.config import (
    BEHAVIOR_CLASSIFIER_MODEL,
    HAND_PROXIMITY_NORM,
    HAND_PROXIMITY_SOURCE,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    PROCESS_EVERY_N_FRAMES,
    TARGET_FPS,
)
from tracking.byte_tracker import ByteTrackWrapper, Track

if TYPE_CHECKING:
    from perception.pose_estimator import PoseDetection

logger = logging.getLogger("cognen.analyzer")

# Per-person state is dropped once its track has not been seen for this long
# (ByteTrack itself forgets a lost track after track_buffer=30 frames, ~2 s
# at 15 FPS, and never reuses the ID).
PERSON_STATE_TTL_MS = 30_000


# ── Per-person state ──────────────────────────────────────────────────────────

@dataclass
class PersonState:
    behavior_clf:   BehaviorClassifier
    # DACE measurer + blink / head-pose / fatigue trackers. Shared with the
    # training loaders (features/track_features.py), so training windows see
    # the same PERCLOS / stability / fidget / openness computation.
    track_metrics:  TrackMetricsState   = field(default_factory=TrackMetricsState)
    contradiction_detector: ContradictionDetector = field(default_factory=ContradictionDetector)
    last_result: BehaviorResult | None  = None
    last_seen_ms: int = 0


# ── RoomAnalyzer ─────────────────────────────────────────────────────────────

class RoomAnalyzer:
    """
    Processes a CCTV stream and produces per-seat behavior analytics.
    One instance per room.  Call run_stream(url) as an asyncio task.
    """

    def __init__(
        self,
        room_id: str,
        seat_map: dict,
        *,
        person_detector: Any = None,
        object_detector: Any = None,
        pose_estimator: Any = None,
        face_landmarks: Any = None,
        tracker: Any = None,
        seat_mapper: Any = None,
        behavior_session: Any = None,
        analyze_posture: Callable | None = None,
        hand_landmarks: Any = None,
        hand_proximity_source: str | None = None,
    ):
        self.room_id   = room_id
        self._running   = False
        self._snapshot: dict = {}
        self._last_frame: np.ndarray | None = None
        self._source: FrameSource | None = None

        # Perception (real modules unless injected, e.g. by tests)
        if person_detector is None:
            from perception.person_detector import PersonDetector
            person_detector = PersonDetector()
        if object_detector is None:
            from perception.object_detector import ObjectDetector
            object_detector = ObjectDetector(INPUT_WIDTH, INPUT_HEIGHT)
        if pose_estimator is None or analyze_posture is None:
            from perception.pose_estimator import PoseEstimator
            if pose_estimator is None:
                pose_estimator = PoseEstimator()
            if analyze_posture is None:
                analyze_posture = PoseEstimator.analyze_posture
        if face_landmarks is None:
            from perception.face_landmarks import FaceLandmarksDetector
            face_landmarks = FaceLandmarksDetector(max_faces=1)
        self._person_det   = person_detector
        self._object_det   = object_detector
        self._pose_est     = pose_estimator
        self._face_lms_det = face_landmarks
        self._analyze_posture = analyze_posture

        # Hand/object proximity source (server/config.py COGNEN_HAND_PROXIMITY):
        # "pose" (default) = YOLOv8-pose wrists; "hands" = MediaPipe
        # HandLandmarker + IoU holding / near (features/hand_object.py),
        # falling back to wrists when no hand is found; "off" = opt-out.
        self._hand_source = hand_proximity_source or HAND_PROXIMITY_SOURCE
        if self._hand_source == "hands" and hand_landmarks is None:
            from perception.hand_landmarks import HandLandmarksDetector
            hand_landmarks = HandLandmarksDetector(max_hands=2)
        self._hand_lms_det = hand_landmarks

        self.face_landmarks_status: str = getattr(face_landmarks, "status", "unknown")
        if not getattr(face_landmarks, "available", False):
            logger.error(
                "[%s] Face landmarks unavailable (status=%s, reason=%s): EAR/MAR/"
                "head pose/gaze are None for every person and 'absent' cannot be "
                "inferred from a missing face.",
                room_id, self.face_landmarks_status,
                getattr(face_landmarks, "init_error", None),
            )

        # Tracking
        if tracker is None:
            tracker = ByteTrackWrapper(frame_rate=TARGET_FPS)
        if seat_mapper is None:
            from tracking.seat_mapper import SeatMapper
            seat_mapper = SeatMapper(seat_map)
        self._tracker     = tracker
        self._seat_mapper = seat_mapper

        # One ONNX session shared by every tracked person (was one per track).
        self._behavior_session = (
            behavior_session if behavior_session is not None
            else load_onnx_session(BEHAVIOR_CLASSIFIER_MODEL)
        )

        # Per-person state registry
        self._person_states: dict[int, PersonState] = {}

    def _new_person_state(self) -> PersonState:
        return PersonState(behavior_clf=BehaviorClassifier(session=self._behavior_session))

    # ── Public ───────────────────────────────────────────────────────────────

    async def run_source(
        self,
        source: FrameSource,
        process_every_n: int = PROCESS_EVERY_N_FRAMES,
    ) -> None:
        """
        Main processing loop — reads frames from any FrameSource and runs the
        full detection → tracking → behavior pipeline on every Nth frame.

        Live sources (RTSP / camera) are drained at their native rate;
        frames between analysis ticks are discarded to avoid buffer build-up.
        Video files are paced to their encoded FPS.

        Args:
            source:          Open (or not yet opened) FrameSource.
            process_every_n: Analyse 1 out of every N frames.  Use N > 1 to
                             reduce CPU/GPU load on high-FPS streams without
                             affecting buffer freshness.
        """
        import cv2

        self._source = source
        # open() can block for the whole connection timeout (RTSP)
        if not await asyncio.to_thread(source.open):
            logger.error("[%s] Failed to open source: %s", self.room_id, source.name)
            return

        self._running   = True
        frame_count     = 0
        # How often to actually call _process_frame
        process_interval = process_every_n / max(source.fps, TARGET_FPS)

        logger.info(
            "[%s] Starting on %s  process_every_n=%d  ~%.1f analysis FPS",
            self.room_id, source.name, process_every_n,
            max(source.fps, TARGET_FPS) / process_every_n,
        )

        try:
            while self._running:
                t0 = time.perf_counter()

                # Blocking read — run in thread to avoid blocking event loop
                ok, frame = await asyncio.to_thread(source.read)

                if not ok:
                    # VideoFileSource returns False on EOF; live sources
                    # return False only after exhausting all reconnect attempts.
                    logger.info("[%s] Source ended: %s", self.room_id, source.name)
                    break

                frame_count += 1

                # Skip frames — still drain buffer, just don't process
                if frame_count % process_every_n != 0:
                    await asyncio.sleep(0)   # yield to event loop
                    continue

                resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
                self._last_frame = resized
                # Inference in a worker thread: keeps the event loop free
                self._snapshot = await asyncio.to_thread(self._process_frame, resized)

                elapsed = time.perf_counter() - t0
                await asyncio.sleep(max(0.0, process_interval - elapsed))

        finally:
            source.release()
            self._running = False
            logger.info("[%s] Source released: %s", self.room_id, source.name)

    async def run_stream(self, stream_url: str) -> None:
        """
        Backwards-compatible wrapper — opens ``stream_url`` as an RTSPSource.
        For new code prefer ``run_source()`` with an explicit FrameSource.
        """
        src = RTSPSource(stream_url)
        await self.run_source(src)

    def get_snapshot(self) -> dict:
        return self._snapshot.copy()

    def get_frame(self) -> np.ndarray | None:
        """Return the last processed frame, or None if no frame yet."""
        f = self._last_frame
        return f.copy() if f is not None else None

    def shutdown(self) -> None:
        self._running = False
        stop = getattr(self._source, "request_stop", None)
        if callable(stop):
            stop()   # abort an RTSP reconnect back-off blocking the reader thread

    # ── Core pipeline (ARCHITECTURE.md §5.2) ────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> dict:
        ts_ms = int(time.time() * 1000)

        # 1. PERCEPTION
        person_dets = self._person_det.detect(frame)
        object_dets = self._object_det.detect(frame)
        pose_dets   = self._pose_est.detect(frame)

        # 2. TRACKING
        tracks = self._tracker.update(person_dets, frame)

        # 3. PER-PERSON ANALYSIS
        results = []
        for track in tracks:
            state = self._person_states.get(track.track_id)
            if state is None:
                state = self._person_states[track.track_id] = self._new_person_state()
            state.last_seen_ms = ts_ms
            result = self._analyze_person(frame, track, object_dets, pose_dets, state, ts_ms)
            results.append(result)
        self._prune_person_states(ts_ms)

        # 4. SEAT OCCUPANCY
        seat_statuses = self._seat_mapper.assign(tracks)
        occupied = [s.seat_id for s in seat_statuses if s.occupied]
        absent   = [s.seat_id for s in seat_statuses if not s.occupied]

        # Attach seat_id to each person result
        # `is not None`: track_id 0 is a valid ID (the stub tracker starts at 0)
        track_seat = {s.track_id: s.seat_id for s in seat_statuses if s.track_id is not None}
        for r in results:
            r["seat_id"] = track_seat.get(r["track_id"])

        room_attention = (
            sum(r["attention_score"] for r in results) / len(results)
            if results else 0
        )

        return {
            "ts": ts_ms / 1000,
            "persons": results,
            "occupancy": {
                "occupied": occupied,
                "absent":   absent,
                "total":    len(seat_statuses),
            },
            "room_attention": round(room_attention, 1),
        }

    def _analyze_person(
        self,
        frame: np.ndarray,
        track: Track,
        object_dets: list,
        pose_dets: list,
        state: PersonState,
        ts_ms: int,
    ) -> dict:
        fh_total, fw_total = frame.shape[:2]
        clipped = clip_bbox(track.bbox_xyxy, fw_total, fh_total)
        if clipped is not None:
            cx1, cy1, cx2, cy2 = clipped
            bbox_face_area = ((cx2 - cx1) * (cy2 - cy1)) / max(fw_total * fh_total, 1)
        else:
            bbox_face_area = 0.0

        # Face landmarks: MediaPipe FaceLandmarker (IMAGE mode) on the upper
        # square of the clipped person box, mapped to 68 points in FRAME
        # coordinates (perception/face_landmarks.py _MP_TO_68).
        roi = face_roi(track.bbox_xyxy, fw_total, fh_total)
        obs = self._get_face(frame, roi)

        # Per-track measurement (features/track_features.py, shared with the
        # training loaders): DACE pitch / gaze / gaze-corrected openness and
        # PERCLOS, blink rate, head-pose stability, fidget, fatigue.
        last = state.last_result.label if state.last_result else None
        metrics = state.track_metrics.measure(
            ts_ms, obs,
            face_detector_available=bool(self._face_lms_det.available),
            attentive=last in ("attentive_screen", "attentive_reading"),
            face_area=bbox_face_area,
        )

        # Associate objects with this person's bbox
        person_objs = [
            o for o in object_dets
            if self._bbox_overlap(track.bbox_xyxy, o.bbox_xyxy)
        ]

        # Associate nearest pose
        person_pose = self._nearest_pose(track.bbox_xyxy, pose_dets)
        posture = self._analyze_posture(person_pose) if person_pose else None

        hand_proximity = self._hand_proximity(frame, track, person_objs, person_pose)
        behavior = state.behavior_clf.update({
            "metrics": metrics,
            "detected_objects": person_objs,
            "hand_proximity": hand_proximity,
            "posture": posture,
        })
        state.last_result = behavior

        # Cross-Modal Contradiction Vector -- computed alongside the
        # behavior classifier but never folded into attention_score (see
        # inference/contradiction_detector.py). The behavior label is passed so
        # an "attentive_reading" window counts as engagement with a visible
        # book (no "prop_without_engagement"). Categories are passed through
        # as-is: coherent / fatigue_masking / prop_without_engagement /
        # context_conflict / unclassified.
        contradiction = state.contradiction_detector.compute(
            metrics, person_objs, hand_proximity, behavior.label,
        )

        return {
            "track_id":      track.track_id,
            "seat_id":       None,  # filled in by caller
            "behavior":      behavior.label,
            "attention_score": behavior.attention_score,
            "confidence":    round(behavior.confidence, 2),
            "contradiction_index":    round(contradiction.contradiction_index, 3),
            "contradiction_category": contradiction.category,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_face(self, frame: np.ndarray, roi: tuple[int, int, int, int] | None) -> FaceObservation | None:
        """Face landmarker inside *roi* (features/track_features.observe_face,
        the same call the training loaders make)."""
        return observe_face(self._face_lms_det, frame, roi)

    def _get_landmarks(self, frame: np.ndarray, roi: tuple[int, int, int, int] | None):
        """68 facial landmarks (FRAME pixels) inside *roi*, or None."""
        obs = self._get_face(frame, roi)
        return obs.landmarks68 if obs is not None else None

    def _hand_proximity(self, frame: np.ndarray, track: Track, person_objs: list, person_pose) -> str:
        """HandObjectProximity label for one person (see __init__ for the sources)."""
        if self._hand_source == "off":
            return "hands_not_visible"
        if self._hand_source == "hands" and self._hand_lms_det is not None \
                and getattr(self._hand_lms_det, "available", False):
            fh, fw = frame.shape[:2]
            box = clip_bbox(track.bbox_xyxy, fw, fh)
            if box is not None:
                x1, y1, x2, y2 = box
                hands = self._hand_lms_det.detect(frame[y1:y2, x1:x2], offset=(x1, y1))
                label = proximity_from_hands([h.landmarks for h in hands], person_objs)
                if label is not None:
                    return label
        if person_pose is not None:
            return estimate_hand_proximity(
                person_pose.keypoints, track.bbox_xyxy, person_objs, max_dist=HAND_PROXIMITY_NORM,
            )
        return "hands_not_visible"

    def _prune_person_states(self, now_ms: int) -> None:
        stale = [
            tid for tid, st in self._person_states.items()
            if now_ms - st.last_seen_ms > PERSON_STATE_TTL_MS
        ]
        for tid in stale:
            del self._person_states[tid]

    def _bbox_overlap(self, person_xyxy, obj_xyxy) -> bool:
        """True if object bbox centre falls inside person bbox."""
        px1, py1, px2, py2 = person_xyxy
        ox1, oy1, ox2, oy2 = obj_xyxy
        cx, cy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
        return px1 <= cx <= px2 and py1 <= cy <= py2

    def _nearest_pose(self, bbox_xyxy, poses: list[PoseDetection]) -> PoseDetection | None:
        if not poses:
            return None
        px1, py1, px2, py2 = bbox_xyxy
        best, best_iou = None, 0.0
        for p in poses:
            bx1, by1, bx2, by2 = p.bbox_xyxy
            ix = max(0, min(px2, bx2) - max(px1, bx1))
            iy = max(0, min(py2, by2) - max(py1, by1))
            inter = ix * iy
            union = (px2-px1)*(py2-py1) + (bx2-bx1)*(by2-by1) - inter
            iou = inter / union if union else 0.0
            if iou > best_iou:
                best_iou, best = iou, p
        return best
