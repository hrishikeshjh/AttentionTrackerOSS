"""
occupancy_metrics.py — Seat occupancy evaluation metrics.

Implements the occupancy metrics of ARCHITECTURE.md §7:
    • Occupancy accuracy           (target ≥ 0.95)  per seat × frame
    • Absence detection latency    (target < 30 s)  per absence EVENT

An absence event starts at the frame where a seat goes from occupied to
empty in the ground truth (an onset must be observed: a seat that is empty
from the first frame has no onset). Its latency is the time from the onset to
the first frame, at or after it, in which the prediction also says empty.
Each event is recorded once. If the seat is re-occupied (GT) or the stream
ends before the prediction says empty, the event is counted as MISSED and
contributes no latency.

The previous implementation re-recorded a 0-frame "latency" on every frame
in which GT and prediction were both empty (it reset its state after each
recording), so an empty seat flooded the statistic with zeros.

Not evaluable -> None: accuracy without frames, latency without any detected
event (status explains why).
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
    Args:
        seat_ids: All seat identifiers in the room.
        fps:      Rate of the evaluated frames (latency frames -> seconds).
    """

    def __init__(self, seat_ids: list[str], fps: float = 5.0) -> None:
        self.seat_ids = list(seat_ids)
        self.fps      = fps
        self.reset()

    # ── accumulation ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._correct = 0
        self._total = 0
        self._frame_idx = 0
        self._latencies_frames: list[int] = []
        self._n_events = 0
        self._n_missed = 0
        self._prev_gt: dict[str, bool | None] = {s: None for s in self.seat_ids}
        self._open_event: dict[str, int | None] = {s: None for s in self.seat_ids}

    def start_sequence(self) -> None:
        """New clip: close open events as missed, forget the previous GT state."""
        for s in self.seat_ids:
            if self._open_event[s] is not None:
                self._n_missed += 1
            self._open_event[s] = None
            self._prev_gt[s] = None

    def update(self, pred_occupied: Set[str], gt_occupied: Set[str]) -> None:
        """Record one frame's predicted / ground-truth occupied seat sets."""
        fi = self._frame_idx
        self._frame_idx += 1

        for seat in self.seat_ids:
            pred_occ = seat in pred_occupied
            gt_occ   = seat in gt_occupied
            self._total   += 1
            self._correct += int(pred_occ == gt_occ)

            prev = self._prev_gt[seat]
            if gt_occ:
                if self._open_event[seat] is not None:      # re-occupied before detection
                    self._n_missed += 1
                    self._open_event[seat] = None
            elif prev is True:                               # occupied -> empty: onset
                self._open_event[seat] = fi
                self._n_events += 1

            onset = self._open_event[seat]
            if onset is not None and not pred_occ:
                self._latencies_frames.append(fi - onset)
                self._open_event[seat] = None
            self._prev_gt[seat] = gt_occ

    # ── computation ──────────────────────────────────────────────────────────

    def compute(self) -> dict:
        open_events = sum(1 for s in self.seat_ids if self._open_event[s] is not None)
        missed = self._n_missed + open_events
        latencies_s = [lat / self.fps for lat in self._latencies_frames]

        accuracy = self._correct / self._total if self._total else None
        mean_latency = float(np.mean(latencies_s)) if latencies_s else None
        p95_latency  = float(np.percentile(latencies_s, 95)) if latencies_s else None

        if not latencies_s:
            lat_status = ("insufficient_data: no occupied->empty transition in the ground truth"
                          if self._n_events == 0 else
                          "no absence event was detected by the prediction (all missed)")
        else:
            lat_status = "ok"

        result: dict[str, object] = {
            "occupancy_accuracy":      round(accuracy, 4) if accuracy is not None else None,
            "mean_absence_latency_s":  round(mean_latency, 2) if mean_latency is not None else None,
            "p95_absence_latency_s":   round(p95_latency, 2) if p95_latency is not None else None,
            "absence_latency_status":  lat_status,
            "n_seat_frames":           self._total,
            "n_absence_events":        self._n_events,
            "n_latency_events":        len(latencies_s),
            "n_missed_events":         missed,
        }
        lat_ok: bool | None
        if p95_latency is not None and missed == 0:
            lat_ok = p95_latency <= TARGETS["absence_detection_latency_s"]
        elif missed > 0 and self._n_events > 0:
            lat_ok = False          # an undetected absence fails the latency target
        else:
            lat_ok = None
        result["targets_met"] = {
            "occupancy_accuracy": None if accuracy is None else accuracy >= TARGETS["occupancy_accuracy"],
            "absence_detection_latency_s": lat_ok,
        }
        return result

    def print_report(self) -> None:
        self.print_metrics(self.compute())

    @staticmethod
    def print_metrics(m: dict) -> None:
        """Print a compute() result."""
        def _f(v, fmt):
            return format(v, fmt) if v is not None else "n/a"

        print("\n" + "=" * 50)
        print("  CognEn Seat Occupancy Metrics")
        print("=" * 50)
        print(f"  Occupancy accuracy   : {_f(m['occupancy_accuracy'], '.4f')}"
              f"  (target ≥ {TARGETS['occupancy_accuracy']})")
        print(f"  Mean absence latency : {_f(m['mean_absence_latency_s'], '.1f')} s")
        print(f"  P95  absence latency : {_f(m['p95_absence_latency_s'], '.1f')} s"
              f"  (target < {TARGETS['absence_detection_latency_s']} s)  [{m['absence_latency_status']}]")
        print(f"  Absence events       : {m['n_absence_events']} "
              f"(detected {m['n_latency_events']}, missed {m['n_missed_events']})")
        print(f"  Seat×frame samples   : {m['n_seat_frames']}")
        print()
        print("  Targets met:")
        for k, v in m["targets_met"].items():
            print(f"    {'✓' if v is True else ('✗' if v is False else '–  (not evaluable)')} {k}")
        print("=" * 50 + "\n")
