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
"""

from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from perception.person_detector   import PersonDetector
from perception.object_detector   import ObjectDetector
from perception.pose_estimator    import PoseEstimator, PoseDetection
from perception.face_landmarks    import FaceLandmarksDetector
from tracking.byte_tracker       import ByteTrackWrapper, Track
from tracking.seat_mapper        import SeatMapper
from features.face_metrics       import (
    BlinkDetector, HeadPoseTracker, FatigueCalculator,
    calculate_ear, calculate_mar, calculate_head_pose,
    FaceMetricsResult,
)
from inference.behavior_classifier import BehaviorClassifier, BehaviorResult
from input.frame_source import FrameSource, RTSPSource
from server.config import (
    INPUT_WIDTH, INPUT_HEIGHT, TARGET_FPS,
    BEHAVIOR_CLASSIFIER_MODEL,
    PROCESS_EVERY_N_FRAMES,
)

logger = logging.getLogger("cognen.analyzer")


# ── Per-person state ──────────────────────────────────────────────────────────

@dataclass
class PersonState:
    blink_detector: BlinkDetector       = field(default_factory=BlinkDetector)
    pose_tracker:   HeadPoseTracker     = field(default_factory=HeadPoseTracker)
    fatigue_calc:   FatigueCalculator   = field(default_factory=FatigueCalculator)
    behavior_clf:   BehaviorClassifier  = field(default_factory=lambda:
                        BehaviorClassifier(BEHAVIOR_CLASSIFIER_MODEL))
    last_result: BehaviorResult | None  = None


# ── RoomAnalyzer ─────────────────────────────────────────────────────────────

class RoomAnalyzer:
    """
    Processes a CCTV stream and produces per-seat behavior analytics.
    One instance per room.  Call run_stream(url) as an asyncio task.
    """

    def __init__(self, room_id: str, seat_map: dict):
        self.room_id   = room_id
        self._running  = False
        self._snapshot: dict = {}

        # Perception
        self._person_det   = PersonDetector()
        self._object_det   = ObjectDetector(INPUT_WIDTH, INPUT_HEIGHT)
        self._pose_est     = PoseEstimator()
        self._face_lms_det = FaceLandmarksDetector(max_faces=1)

        # Tracking
        self._tracker    = ByteTrackWrapper(frame_rate=TARGET_FPS)
        self._seat_mapper = SeatMapper(seat_map)

        # Per-person state registry
        self._person_states: dict[int, PersonState] = {}

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
        if not source.open():
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
                self._snapshot = self._process_frame(resized)

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

    def shutdown(self) -> None:
        self._running = False

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
            state = self._person_states.setdefault(track.track_id, PersonState())
            result = self._analyze_person(frame, track, object_dets, pose_dets, state, ts_ms)
            results.append(result)

        # 4. SEAT OCCUPANCY
        seat_statuses = self._seat_mapper.assign(tracks)
        occupied = [s.seat_id for s in seat_statuses if s.occupied]
        absent   = [s.seat_id for s in seat_statuses if not s.occupied]

        # Attach seat_id to each person result
        track_seat = {s.track_id: s.seat_id for s in seat_statuses if s.track_id}
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
        x1, y1, x2, y2 = (int(v) for v in track.bbox_xyxy)
        face_crop = frame[y1:y2, x1:x2]
        fh, fw = face_crop.shape[:2]

        # Face landmarks (placeholder — integrate face_alignment / dlib here)
        landmarks = self._get_landmarks(face_crop)

        if landmarks:
            ear   = calculate_ear(landmarks)
            mar   = calculate_mar(landmarks)
            yaw, pitch, roll = calculate_head_pose(landmarks)
        else:
            ear = mar = yaw = pitch = roll = None

        blink_det, blink_rate, perclos = state.blink_detector.update(ear, ts_ms)
        state.pose_tracker.add(yaw, pitch, roll, ts_ms)
        ear_mean, ear_var = state.blink_detector.ear_stats()
        fatigue = state.fatigue_calc.update(perclos, blink_rate, ear_mean, ts_ms)

        metrics = FaceMetricsResult(
            ear=ear, mar=mar,
            pitch=pitch, yaw=yaw, roll=roll,
            gaze_x=None, gaze_y=None,
            blink_detected=blink_det,
            blink_rate=blink_rate, perclos=perclos,
            fatigue_index=fatigue,
            fidget_score=state.pose_tracker.fidget(),
            head_pose_stability=state.pose_tracker.stability(),
            face_distance=None,
            face_area=None,
            mouth_openness=None,
            ear_variance=ear_var,
            pose_variance=state.pose_tracker.pose_variance(),
        )

        # Associate objects with this person's bbox
        person_objs = [
            o for o in object_dets
            if self._bbox_overlap(track.bbox_xyxy, o.bbox_xyxy)
        ]

        # Associate nearest pose
        person_pose = self._nearest_pose(track.bbox_xyxy, pose_dets)
        posture = PoseEstimator.analyze_posture(person_pose) if person_pose else None

        behavior = state.behavior_clf.update({
            "metrics": metrics,
            "detected_objects": person_objs,
            "hand_proximity": "hands_not_visible",  # extend with HandTracker if needed
            "posture": posture,
        })
        state.last_result = behavior

        return {
            "track_id":      track.track_id,
            "seat_id":       None,  # filled in by caller
            "behavior":      behavior.label,
            "attention_score": behavior.attention_score,
            "confidence":    round(behavior.confidence, 2),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_landmarks(self, face_crop: np.ndarray):
        """
        Detect 68 facial landmarks from a face crop using MediaPipe FaceMesh.
        Returns a list of Point2D objects (ibug 68-point convention) or None.
        """
        if not self._face_lms_det.available or face_crop.size == 0:
            return None
        results = self._face_lms_det.detect(face_crop)
        return results[0] if results else None

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
