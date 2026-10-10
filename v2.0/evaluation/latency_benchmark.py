"""
latency_benchmark.py — Per-frame latency benchmarking for v1.0 and v2.0.

ARCHITECTURE.md §7 targets:
    v1.0 latency : < 200 ms/frame
    v2.0 latency : < 100 ms/frame (GPU)

Measures wall-clock time per pipeline call. For v2.0,
``benchmark_pipeline_stages`` times the FULL per-frame pipeline
(RoomAnalyzer._process_frame) and each stage inside it -- person detection,
object detection, pose, tracking, face landmarks, hand landmarks, behavior
classification (summed over the persons of the frame) -- with p50 / p95 per
stage. The previous runner timed only the 15x15 CNN (~0.02 ms) and reported
that as "v2.0 latency".

Nothing is invented: ``missing_pipeline_models`` lists the model files the
real pipeline needs; when any is missing, or no real video frames are given,
the result is {"status": "SKIPPED", "reason": ...}. A benchmark in which no
person was ever detected is "INCOMPLETE" (face / classifier stages never ran)
and its target check is None.

Usage:
    from evaluation.latency_benchmark import LatencyBenchmark, benchmark_v2_pipeline

    # Context-manager style:
    bench = LatencyBenchmark("v2.0 perception")
    with bench.measure():
        output = pipeline.process_frame(frame)
    print(bench.summary())

    # Or use the convenience function:
    benchmark_v2_pipeline(room_analyzer, test_frames)
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any, Callable, Generator, Sequence

import numpy as np

TARGETS_MS = {
    "v1.0": 200.0,
    "v2.0": 100.0,
}


class LatencyBenchmark:
    """
    Lightweight wall-clock timer for measuring per-call latency.
    """

    def __init__(self, name: str = "pipeline") -> None:
        self.name      = name
        self._times_ms: list[float] = []

    # ── context manager ───────────────────────────────────────────────────────

    @contextlib.contextmanager
    def measure(self) -> Generator[None, None, None]:
        """Use as `with bench.measure(): ...` to time one call."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._times_ms.append(elapsed_ms)

    # ── decorator ─────────────────────────────────────────────────────────────

    def wrap(self, fn: Callable) -> Callable:
        """Wrap a callable to automatically record its latency."""
        def _inner(*args, **kwargs):
            with self.measure():
                return fn(*args, **kwargs)
        return _inner

    # ── results ───────────────────────────────────────────────────────────────

    def record(self, elapsed_ms: float) -> None:
        self._times_ms.append(elapsed_ms)

    def reset(self) -> None:
        self._times_ms.clear()

    def compute(self) -> dict:
        if not self._times_ms:
            return {}
        t = np.array(self._times_ms)
        return {
            "name":        self.name,
            "n_frames":    len(t),
            "mean_ms":     round(float(t.mean()),              2),
            "median_ms":   round(float(np.median(t)),         2),
            "p50_ms":      round(float(np.percentile(t, 50)), 2),
            "p95_ms":      round(float(np.percentile(t, 95)), 2),
            "p99_ms":      round(float(np.percentile(t, 99)), 2),
            "max_ms":      round(float(t.max()),               2),
            "min_ms":      round(float(t.min()),               2),
        }

    def summary(self, target_version: str | None = None) -> str:
        m = self.compute()
        if not m:
            return f"[{self.name}] No measurements."
        lines = [
            f"\n{'=' * 55}",
            f"  Latency Benchmark: {m['name']}",
            f"{'=' * 55}",
            f"  Frames measured : {m['n_frames']}",
            f"  Mean            : {m['mean_ms']:.1f} ms",
            f"  Median          : {m['median_ms']:.1f} ms",
            f"  P95             : {m['p95_ms']:.1f} ms",
            f"  P99             : {m['p99_ms']:.1f} ms",
            f"  Max             : {m['max_ms']:.1f} ms",
        ]
        if target_version and target_version in TARGETS_MS:
            target  = TARGETS_MS[target_version]
            p95_ok  = m["p95_ms"] <= target
            symbol  = "✓" if p95_ok else "✗"
            lines.append(f"  {symbol} P95 target ({target_version}): < {target:.0f} ms")
        lines.append("=" * 55 + "\n")
        return "\n".join(lines)


# ── v2.0 full-pipeline stage benchmark ────────────────────────────────────────

STAGES: tuple[str, ...] = (
    "person_detect", "object_detect", "pose_detect", "track",
    "face_landmarks", "hand_landmarks", "classify",
)


def missing_pipeline_models() -> list[str]:
    """Model files the real v2.0 pipeline loads that are not on disk."""
    from server.config import (
        BEHAVIOR_CLASSIFIER_MODEL,
        OBJECT_DETECTOR_MODEL,
        PERSON_DETECTOR_MODEL,
        POSE_ESTIMATOR_MODEL,
    )
    models_dir = Path(PERSON_DETECTOR_MODEL).parent
    needed = [PERSON_DETECTOR_MODEL, OBJECT_DETECTOR_MODEL, POSE_ESTIMATOR_MODEL,
              BEHAVIOR_CLASSIFIER_MODEL, models_dir / "face_landmarker.task"]
    return [str(p) for p in needed if not Path(p).exists()]


class _StageClock:
    """Per-frame accumulators for the wrapped stage callables."""

    def __init__(self) -> None:
        self.frame: dict[str, float] = {}
        self.calls: dict[str, int] = {}

    def wrap(self, stage: str, fn: Callable) -> Callable:
        def _timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.frame[stage] = self.frame.get(stage, 0.0) + (time.perf_counter() - t0) * 1000.0
                self.calls[stage] = self.calls.get(stage, 0) + 1
        return _timed


def instrument_room_analyzer(analyzer: Any) -> _StageClock:
    """
    Wrap the stage methods of a RoomAnalyzer INSTANCE (its components) with
    timers. Per-person stages (face landmarks, hand landmarks, classify) are
    summed per frame. Returns the clock; the analyzer keeps working normally.
    """
    clock = _StageClock()

    def _patch(obj: Any, attr: str, stage: str) -> None:
        if obj is not None and callable(getattr(obj, attr, None)):
            setattr(obj, attr, clock.wrap(stage, getattr(obj, attr)))

    _patch(analyzer._person_det, "detect", "person_detect")
    _patch(analyzer._object_det, "detect", "object_detect")
    _patch(analyzer._pose_est, "detect", "pose_detect")
    _patch(analyzer._tracker, "update", "track")
    # FaceLandmarksDetector.detect calls detect_full: patch only one of them,
    # or a frame's landmark time would be counted twice.
    if callable(getattr(analyzer._face_lms_det, "detect_full", None)):
        _patch(analyzer._face_lms_det, "detect_full", "face_landmarks")
    else:
        _patch(analyzer._face_lms_det, "detect", "face_landmarks")
    _patch(getattr(analyzer, "_hand_lms_det", None), "detect", "hand_landmarks")

    new_state = analyzer._new_person_state

    def _new_person_state_timed():
        st = new_state()
        st.behavior_clf.update = clock.wrap("classify", st.behavior_clf.update)
        return st

    analyzer._new_person_state = _new_person_state_timed
    return clock


def _pct(values: Sequence[float]) -> dict:
    t = np.asarray(values, dtype=np.float64)
    return {
        "p50_ms": round(float(np.percentile(t, 50)), 3),
        "p95_ms": round(float(np.percentile(t, 95)), 3),
        "mean_ms": round(float(t.mean()), 3),
        "max_ms": round(float(t.max()), 3),
    }


def benchmark_pipeline_stages(
    room_analyzer: Any,
    frames: Sequence[np.ndarray],
    n_warmup: int = 5,
    target_version: str = "v2.0",
) -> dict:
    """
    Time RoomAnalyzer._process_frame on real *frames* (already resized to the
    server input size), total and per stage, after *n_warmup* frames.
    """
    if len(frames) <= n_warmup:
        return {"status": "SKIPPED",
                "reason": f"need more than {n_warmup} frames (got {len(frames)})"}
    clock = instrument_room_analyzer(room_analyzer)
    totals: list[float] = []
    per_stage: dict[str, list[float]] = {s: [] for s in STAGES}
    persons: list[int] = []
    for i, frame in enumerate(frames):
        clock.frame = {}
        t0 = time.perf_counter()
        snap = room_analyzer._process_frame(frame)
        dt = (time.perf_counter() - t0) * 1000.0
        if i < n_warmup:
            continue
        totals.append(dt)
        persons.append(len(snap.get("persons", [])))
        for st in STAGES:
            per_stage[st].append(clock.frame.get(st, 0.0))

    stages = {st: _pct(v) for st, v in per_stage.items() if any(x > 0 for x in v)}
    frames_with_persons = sum(1 for n in persons if n > 0)
    total = _pct(totals)
    status = "ok" if frames_with_persons > 0 else "INCOMPLETE"
    result: dict[str, Any] = {
        "status": status,
        "name": "v2.0 RoomAnalyzer._process_frame (full pipeline)",
        "n_frames": len(totals),
        "frames_with_persons": frames_with_persons,
        "mean_persons_per_frame": round(float(np.mean(persons)), 2),
        "total": total,
        "stages": stages,
        "stages_not_run": [st for st in STAGES if st not in stages],
    }
    if status != "ok":
        result["reason"] = ("no person was detected in the benchmark frames: face landmarks and the "
                            "behavior classifier never ran, so the total is not a per-person pipeline time")
    target = TARGETS_MS.get(target_version)
    result["target_ms"] = target
    result["target_met"] = (total["p95_ms"] <= target) if (status == "ok" and target) else None
    return result


def benchmark_v2_pipeline(
    room_analyzer,
    test_frames: list,
    n_warmup: int = 5,
) -> dict:
    """Full-pipeline stage benchmark (see benchmark_pipeline_stages)."""
    result = benchmark_pipeline_stages(room_analyzer, test_frames, n_warmup=n_warmup)
    print(format_stage_report(result))
    return result


def format_stage_report(result: dict) -> str:
    if result.get("status") == "SKIPPED":
        return f"[latency] SKIPPED: {result.get('reason')}"
    lines = [f"\n{'=' * 60}", f"  Latency: {result['name']}  [{result['status']}]", f"{'=' * 60}",
             f"  Frames: {result['n_frames']}  (with persons: {result['frames_with_persons']})",
             f"  {'stage':<16} {'p50 ms':>9} {'p95 ms':>9}",
             f"  {'TOTAL':<16} {result['total']['p50_ms']:>9.2f} {result['total']['p95_ms']:>9.2f}"]
    for st, m in result["stages"].items():
        lines.append(f"  {st:<16} {m['p50_ms']:>9.2f} {m['p95_ms']:>9.2f}")
    if result.get("stages_not_run"):
        lines.append(f"  not run: {', '.join(result['stages_not_run'])}")
    lines.append("=" * 60 + "\n")
    return "\n".join(lines)


def benchmark_v1_inference(
    ml_engine,
    sample_metrics: list,
    n_warmup: int = 10,
) -> dict:
    """
    Benchmark MLEngine.predict() over a list of ExtendedFaceMetrics samples.

    Args:
        ml_engine:       MLEngine instance.
        sample_metrics:  List of ExtendedFaceMetrics objects.
        n_warmup:        Warmup calls to exclude.

    Returns:
        Latency metrics dict.
    """
    bench = LatencyBenchmark("v1.0 MLEngine.predict")
    for i, metrics in enumerate(sample_metrics):
        with bench.measure():
            ml_engine.predict(metrics)
        if i == n_warmup - 1:
            bench.reset()

    result = bench.compute()
    print(bench.summary(target_version="v1.0"))
    return result
