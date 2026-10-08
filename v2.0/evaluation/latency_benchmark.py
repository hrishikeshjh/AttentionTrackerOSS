"""
latency_benchmark.py — Per-frame latency benchmarking for v1.0 and v2.0.

ARCHITECTURE.md §7 targets:
    v1.0 latency : < 200 ms/frame
    v2.0 latency : < 100 ms/frame (GPU)

Measures end-to-end wall-clock time per pipeline call.
Produces mean, median, P95, P99, and max latency statistics.

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
from typing import Callable, Generator

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


# ── convenience functions ─────────────────────────────────────────────────────

def benchmark_v2_pipeline(
    room_analyzer,
    test_frames: list,
    n_warmup: int = 5,
) -> dict:
    """
    Benchmark RoomAnalyzer._process_frame() over a list of test frames.

    Args:
        room_analyzer: RoomAnalyzer instance (v2.0).
        test_frames:   List of numpy BGR frames to feed.
        n_warmup:      Number of warmup frames (excluded from stats).

    Returns:
        Latency metrics dict.
    """
    bench = LatencyBenchmark("v2.0 RoomAnalyzer._process_frame")
    for i, frame in enumerate(test_frames):
        with bench.measure():
            room_analyzer._process_frame(frame)
        if i == n_warmup - 1:
            bench.reset()  # discard warmup measurements

    result = bench.compute()
    print(bench.summary(target_version="v2.0"))
    return result


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
