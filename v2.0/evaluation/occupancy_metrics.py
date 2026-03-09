"""
occupancy_metrics.py — Seat occupancy evaluation metrics.

Implements occupancy metrics from ARCHITECTURE.md §7:
    • Occupancy accuracy           (target ≥ 95%)
    • Absence detection latency    (target < 30 seconds)

Usage:
    from evaluation.occupancy_metrics import OccupancyEvaluator
    ev = OccupancyEvaluator(seat_ids=["seat_1", ..., "seat_6"], fps=5.0)
    ev.update(pred_occupied={"seat_1", "seat_3"}, gt_occupied={"seat_1", "seat_2", "seat_3"})
    report = ev.compute()
"""

from __future__ import annotations
from typing import Set

import numpy as np


TARGETS = {
    "occupancy_accuracy":         0.95,
    "absence_detection_latency_s": 30.0,
}


class OccupancyEvaluator:
    """
    Evaluates per-seat occupancy predictions against ground truth.

    Args:
        seat_ids: List of all seat identifiers in the room.
        fps:      Frame rate (for latency computation).
    """

    def __init__(self, seat_ids: list[str], fps: float = 5.0) -> None:
        self.seat_ids = seat_ids
        self.fps      = fps
        self._n_seats = len(seat_ids)

        self._correct: int = 0
        self._total:   int = 0

        # Absence latency: track when each seat first became empty in GT
        # and when the predictor finally called it empty.
        self._gt_empty_since:   dict[str, int | None] = {s: None for s in seat_ids}
        self._pred_empty_since: dict[str, int | None] = {s: None for s in seat_ids}
        self._latencies_frames: list[int] = []
        self._frame_idx: int = 0

    # ── accumulation ─────────────────────────────────────────────────────────

    def update(
        self,
        pred_occupied: Set[str],
        gt_occupied:   Set[str],
    ) -> None:
        """
        Record one frame's seat occupancy prediction.

        Args:
            pred_occupied: Set of seat IDs predicted as occupied.
            gt_occupied:   Set of seat IDs that are actually occupied.
        """
        fi = self._frame_idx
        self._frame_idx += 1

        for seat in self.seat_ids:
            pred_occ = seat in pred_occupied
            gt_occ   = seat in gt_occupied

            self._total   += 1
            self._correct += int(pred_occ == gt_occ)

            # Absence latency tracking
            if not gt_occ and self._gt_empty_since[seat] is None:
                self._gt_empty_since[seat] = fi   # seat just became empty in GT

            if not pred_occ and self._pred_empty_since[seat] is None:
                self._pred_empty_since[seat] = fi

            # Once predictor detects empty AND GT is empty, record latency
            if (self._gt_empty_since[seat] is not None
                    and self._pred_empty_since[seat] is not None):
                latency = self._pred_empty_since[seat] - self._gt_empty_since[seat]
                if latency >= 0:
                    self._latencies_frames.append(latency)
                # Reset to avoid double-counting
                self._gt_empty_since[seat]   = None
                self._pred_empty_since[seat] = None

            # Reset when seat becomes occupied again
            if gt_occ:
                self._gt_empty_since[seat]   = None
                self._pred_empty_since[seat] = None

    def reset(self) -> None:
        self._correct = self._total = self._frame_idx = 0
        self._latencies_frames.clear()
        for s in self.seat_ids:
            self._gt_empty_since[s] = self._pred_empty_since[s] = None

    # ── computation ──────────────────────────────────────────────────────────

    def compute(self) -> dict:
        accuracy = self._correct / max(self._total, 1)

        latencies_s = [lat / self.fps for lat in self._latencies_frames]
        mean_latency = float(np.mean(latencies_s)) if latencies_s else 0.0
        p95_latency  = float(np.percentile(latencies_s, 95)) if latencies_s else 0.0

        result = {
            "occupancy_accuracy":      round(accuracy,     4),
            "mean_absence_latency_s":  round(mean_latency, 2),
            "p95_absence_latency_s":   round(p95_latency,  2),
            "n_seat_frames":           self._total,
            "n_latency_events":        len(latencies_s),
        }
        result["targets_met"] = {
            "occupancy_accuracy":          accuracy     >= TARGETS["occupancy_accuracy"],
            "absence_detection_latency_s": p95_latency  <= TARGETS["absence_detection_latency_s"],
        }
        return result

    def print_report(self) -> None:
        m = self.compute()
        print("\n" + "=" * 50)
        print("  CognEn Seat Occupancy Metrics")
        print("=" * 50)
        print(f"  Occupancy accuracy   : {m['occupancy_accuracy']:.4f}"
              f"  (target ≥ {TARGETS['occupancy_accuracy']})")
        print(f"  Mean absence latency : {m['mean_absence_latency_s']:.1f} s")
        print(f"  P95  absence latency : {m['p95_absence_latency_s']:.1f} s"
              f"  (target < {TARGETS['absence_detection_latency_s']} s)")
        print(f"  Seat×frame samples   : {m['n_seat_frames']}")
        print()
        print("  Targets met:")
        for k, v in m["targets_met"].items():
            print(f"    {'✓' if v else '✗'} {k}")
        print("=" * 50 + "\n")
