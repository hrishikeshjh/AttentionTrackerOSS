"""
FaceMetrics — v1.0 face geometry logic ported to Python.
(ARCHITECTURE.md §3 Feature Extraction + §3 Summary: "Port to Python")

Computes EAR, MAR, head pose, gaze, blink rate, PERCLOS, and fatigue
from 68-point face-api.js-compatible landmarks extracted via a cropped
face region passed through dlib or MediaPipe FaceMesh.
"""

from __future__ import annotations
import math
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Point2D:
    x: float
    y: float


@dataclass
class FaceMetricsResult:
    ear: float | None
    mar: float | None
    pitch: float | None
    yaw: float | None
    roll: float | None
    gaze_x: float | None
    gaze_y: float | None
    blink_detected: bool
    blink_rate: float       # blinks per minute
    perclos: float          # 0–100
    fatigue_index: float    # 0–100
    fidget_score: float     # 0–100
    head_pose_stability: float  # 0–100
    face_distance: float | None
    face_area: float | None
    mouth_openness: float | None
    ear_variance: float
    pose_variance: float


# ── Constants (mirrored from v1.0 face-geometry.ts) ──────────────────────────

EAR_BLINK_THRESHOLD = 0.21
PERCLOS_WINDOW_MS   = 60_000
NORMAL_BLINK_RATE   = (15, 20)
FATIGUE_EAR_THRESH  = 0.25

LEFT_EYE  = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]
MOUTH_KPS = [48, 54, 51, 57]    # left, right, top, bottom


def _dist(a: Point2D, b: Point2D) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _variance(vals: list[float]) -> float:
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    return sum((v - mean) ** 2 for v in vals) / len(vals)


def calculate_ear(lm: list[Point2D]) -> float | None:
    def single(indices):
        pts = [lm[i] for i in indices]
        v = _dist(pts[1], pts[5]) + _dist(pts[2], pts[4])
        h = _dist(pts[0], pts[3])
        return v / (2 * h) if h else None

    left  = single(LEFT_EYE)
    right = single(RIGHT_EYE)
    if left is None or right is None:
        return None
    return (left + right) / 2


def calculate_mar(lm: list[Point2D]) -> float | None:
    pts = [lm[i] for i in MOUTH_KPS]
    v = _dist(pts[2], pts[3])
    h = _dist(pts[0], pts[1])
    return v / h if h else None


def calculate_head_pose(lm: list[Point2D]) -> tuple[float | None, float | None, float | None]:
    """Returns (yaw, pitch, roll) in approximate degrees."""
    nose, l_eye, r_eye, chin = lm[30], lm[36], lm[45], lm[8]
    eye_mid = Point2D((l_eye.x + r_eye.x) / 2, (l_eye.y + r_eye.y) / 2)
    eye_dist = _dist(l_eye, r_eye) or 1
    nc_dist  = _dist(nose, chin) or 1

    yaw   = (nose.x - eye_mid.x) / eye_dist * 50
    pitch = (nose.y - eye_mid.y) / nc_dist * 50
    roll  = math.degrees(math.atan2(r_eye.y - l_eye.y, r_eye.x - l_eye.x))
    return yaw, pitch, roll


# ── Stateful detectors (one instance per tracked person) ─────────────────────

class BlinkDetector:
    def __init__(self):
        self._ear_history: deque[tuple[int, float]] = deque(maxlen=10_000)
        self._blink_events: deque[int] = deque(maxlen=1_000)
        self._closed = False
        self._close_start: int | None = None

    def update(self, ear: float | None, ts_ms: int) -> tuple[bool, float, float]:
        if ear is None:
            return False, self._blink_rate(ts_ms), self._perclos(ts_ms)

        self._ear_history.append((ts_ms, ear))
        cutoff = ts_ms - 120_000
        while self._ear_history and self._ear_history[0][0] < cutoff:
            self._ear_history.popleft()
        while self._blink_events and self._blink_events[0] < cutoff:
            self._blink_events.popleft()

        is_closed = ear < EAR_BLINK_THRESHOLD
        detected = False

        if is_closed and not self._closed:
            self._close_start = ts_ms
        if self._closed and not is_closed and self._close_start:
            dur = ts_ms - self._close_start
            if 50 <= dur <= 500:
                detected = True
                self._blink_events.append(ts_ms)
            self._close_start = None

        self._closed = is_closed
        return detected, self._blink_rate(ts_ms), self._perclos(ts_ms)

    def _blink_rate(self, now: int) -> float:
        if len(self._blink_events) < 2 or len(self._ear_history) < 50:
            return 0.0
        span = min(60_000, now - self._ear_history[0][0])
        if span < 5_000:
            return 0.0
        return len(self._blink_events) / span * 60_000

    def _perclos(self, now: int) -> float:
        window = [e for t, e in self._ear_history if t > now - PERCLOS_WINDOW_MS]
        if len(window) < 10:
            return 0.0
        return sum(1 for e in window if e < EAR_BLINK_THRESHOLD) / len(window) * 100

    def ear_stats(self) -> tuple[float, float]:
        vals = [e for _, e in self._ear_history]
        if not vals:
            return 0.0, 0.0
        mean = sum(vals) / len(vals)
        return mean, _variance(vals)


class HeadPoseTracker:
    def __init__(self):
        self._history: deque[tuple[int, float, float, float]] = deque(maxlen=5_000)

    def add(self, yaw, pitch, roll, ts_ms: int):
        if None not in (yaw, pitch, roll):
            self._history.append((ts_ms, yaw, pitch, roll))
            cutoff = ts_ms - 30_000
            while self._history and self._history[0][0] < cutoff:
                self._history.popleft()

    def stability(self) -> float:
        if len(self._history) < 5:
            return 100.0
        yaws   = [h[1] for h in self._history]
        pitches = [h[2] for h in self._history]
        rolls  = [h[3] for h in self._history]
        combined = (_variance(yaws) + _variance(pitches) + _variance(rolls) * 0.5) / 2.5
        return max(0.0, min(100.0, 100 - combined / 10))

    def fidget(self) -> float:
        if len(self._history) < 10:
            return 0.0
        diffs = [
            abs(self._history[i][1] - self._history[i-1][1]) +
            abs(self._history[i][2] - self._history[i-1][2]) +
            abs(self._history[i][3] - self._history[i-1][3])
            for i in range(1, len(self._history))
        ]
        return min(100.0, sum(diffs) / len(diffs) * 5)

    def pose_variance(self) -> float:
        yaws   = [h[1] for h in self._history]
        pitches = [h[2] for h in self._history]
        rolls  = [h[3] for h in self._history]
        return (_variance(yaws) + _variance(pitches) + _variance(rolls)) / 3


class FatigueCalculator:
    def __init__(self):
        self._perclos_h: deque[tuple[int, float]] = deque(maxlen=5_000)
        self._blink_h:   deque[tuple[int, float]] = deque(maxlen=5_000)
        self._ear_h:     deque[tuple[int, float]] = deque(maxlen=5_000)

    def update(self, perclos: float, blink_rate: float, ear_mean: float, ts_ms: int) -> float:
        for buf, val in [
            (self._perclos_h, perclos),
            (self._blink_h,   blink_rate),
            (self._ear_h,     ear_mean),
        ]:
            buf.append((ts_ms, val))
            cutoff = ts_ms - 300_000
            while buf and buf[0][0] < cutoff:
                buf.popleft()
        return self._index()

    def _index(self) -> float:
        if len(self._perclos_h) < 5:
            return 0.0
        avg_p = sum(v for _, v in self._perclos_h) / len(self._perclos_h)
        avg_b = sum(v for _, v in self._blink_h) / len(self._blink_h) if self._blink_h else 0
        avg_e = sum(v for _, v in self._ear_h) / len(self._ear_h) if self._ear_h else 0

        p_score = min(40.0, avg_p * 2)
        b_score = 0.0
        if 0 < avg_b < NORMAL_BLINK_RATE[0]:
            b_score = min(30.0, (NORMAL_BLINK_RATE[0] - avg_b) * 2)
        elif avg_b > NORMAL_BLINK_RATE[1]:
            b_score = min(30.0, avg_b - NORMAL_BLINK_RATE[1])
        e_score = max(0.0, min(30.0, (FATIGUE_EAR_THRESH - avg_e) * 100)) if avg_e < FATIGUE_EAR_THRESH else 0

        return min(100.0, p_score + b_score + e_score)
