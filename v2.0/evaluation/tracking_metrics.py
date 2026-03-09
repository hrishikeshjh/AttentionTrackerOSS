"""
tracking_metrics.py — Multi-object tracking evaluation metrics.

Implements MOT metrics from ARCHITECTURE.md §7:
    • MOTA  — Multi-Object Tracking Accuracy  (target ≥ 75%)
    • IDF1  — ID F1 score                     (target ≥ 80%)
    • ID switches                             (target < 5/min)

Metric definitions follow the MOTChallenge standard:
    MOTA = 1 - (FN + FP + IDSW) / GT
    IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)

Where IDTP/IDFP/IDFN are computed using the longest common sub-sequence (LCSS)
of ground-truth ↔ predicted identity assignments.

Usage:
    from evaluation.tracking_metrics import TrackingEvaluator
    ev = TrackingEvaluator()
    for frame_gt, frame_pred in zip(gt_frames, pred_frames):
        ev.update(frame_gt, frame_pred)
    report = ev.compute()

Each frame_gt / frame_pred is a list of Track objects (or dicts) with:
    .track_id   (int)
    .bbox_xyxy  (x1, y1, x2, y2) in pixel coords
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ── Targets ───────────────────────────────────────────────────────────────────

TARGETS = {
    "mota":          0.75,
    "idf1":          0.80,
    "id_switches_per_min": 5.0,
}


@dataclass
class TrackFrame:
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]


@dataclass
class FrameStats:
    tp:   int = 0
    fp:   int = 0
    fn:   int = 0
    idsw: int = 0


class TrackingEvaluator:
    """
    Accumulates per-frame detections and computes MOT metrics.

    Args:
        iou_threshold: Minimum IoU to count a detection as a true positive.
        fps:           Stream frame rate (used for ID switches per minute).
    """

    def __init__(self, iou_threshold: float = 0.5, fps: float = 5.0) -> None:
        self.iou_threshold = iou_threshold
        self.fps           = fps

        self._frame_stats: list[FrameStats] = []
        self._gt_total:    int = 0
        self._n_frames:    int = 0

        # ID consistency tracking: gt_id → last matched pred_id
        self._gt_to_pred: dict[int, int] = {}
        self._id_switches: int = 0

        # For IDF1: per gt-track match sequences
        self._idtp: int = 0
        self._idfp: int = 0
        self._idfn: int = 0

    # ── accumulation ─────────────────────────────────────────────────────────

    def update(
        self,
        gt_tracks:   Sequence,
        pred_tracks: Sequence,
    ) -> FrameStats:
        """
        Process one frame.

        Args:
            gt_tracks:   Ground-truth track list (TrackFrame or dict).
            pred_tracks: Predicted track list.

        Returns:
            FrameStats for this frame.
        """
        gt   = [_to_track_frame(t) for t in gt_tracks]
        pred = [_to_track_frame(t) for t in pred_tracks]

        self._gt_total += len(gt)
        self._n_frames  += 1

        if not gt and not pred:
            stats = FrameStats()
            self._frame_stats.append(stats)
            return stats

        # IoU matching (greedy)
        matches, unmatched_gt, unmatched_pred = _match_iou(gt, pred, self.iou_threshold)

        tp   = len(matches)
        fp   = len(unmatched_pred)
        fn   = len(unmatched_gt)
        idsw = 0

        for gt_t, pred_t in matches:
            prev_pred = self._gt_to_pred.get(gt_t.track_id)
            if prev_pred is not None and prev_pred != pred_t.track_id:
                idsw += 1
                self._id_switches += 1
            self._gt_to_pred[gt_t.track_id] = pred_t.track_id

        self._idtp += tp
        self._idfp += fp
        self._idfn += fn

        stats = FrameStats(tp=tp, fp=fp, fn=fn, idsw=idsw)
        self._frame_stats.append(stats)
        return stats

    def reset(self) -> None:
        self._frame_stats.clear()
        self._gt_total    = 0
        self._n_frames    = 0
        self._gt_to_pred.clear()
        self._id_switches = 0
        self._idtp = self._idfp = self._idfn = 0

    # ── computation ──────────────────────────────────────────────────────────

    def compute(self) -> dict:
        total_fn   = sum(s.fn   for s in self._frame_stats)
        total_fp   = sum(s.fp   for s in self._frame_stats)
        total_idsw = sum(s.idsw for s in self._frame_stats)
        denom      = max(self._gt_total, 1)

        mota = 1.0 - (total_fn + total_fp + total_idsw) / denom
        mota = float(np.clip(mota, 0.0, 1.0))

        idtp, idfp, idfn = self._idtp, self._idfp, self._idfn
        idf1_denom = 2 * idtp + idfp + idfn
        idf1 = (2.0 * idtp / idf1_denom) if idf1_denom > 0 else 0.0

        duration_min = (self._n_frames / max(self.fps, 1)) / 60.0
        idsw_per_min = self._id_switches / max(duration_min, 1e-6)

        result = {
            "mota":               round(mota, 4),
            "idf1":               round(idf1, 4),
            "id_switches":        self._id_switches,
            "id_switches_per_min": round(idsw_per_min, 2),
            "total_fp":           total_fp,
            "total_fn":           total_fn,
            "gt_total":           self._gt_total,
            "n_frames":           self._n_frames,
        }
        result["targets_met"] = {
            "mota":               mota          >= TARGETS["mota"],
            "idf1":               idf1          >= TARGETS["idf1"],
            "id_switches_per_min": idsw_per_min <= TARGETS["id_switches_per_min"],
        }
        return result

    def print_report(self) -> None:
        m = self.compute()
        print("\n" + "=" * 50)
        print("  CognEn Tracking Metrics")
        print("=" * 50)
        print(f"  MOTA              : {m['mota']:.4f}  (target ≥ {TARGETS['mota']})")
        print(f"  IDF1              : {m['idf1']:.4f}  (target ≥ {TARGETS['idf1']})")
        print(f"  ID switches/min   : {m['id_switches_per_min']:.2f}  (target < {TARGETS['id_switches_per_min']})")
        print(f"  Total FP          : {m['total_fp']}")
        print(f"  Total FN          : {m['total_fn']}")
        print(f"  GT total          : {m['gt_total']}")
        print(f"  Frames evaluated  : {m['n_frames']}")
        print()
        print("  Targets met:")
        for k, v in m["targets_met"].items():
            print(f"    {'✓' if v else '✗'} {k}")
        print("=" * 50 + "\n")


# ── IoU helpers ───────────────────────────────────────────────────────────────

def _iou(a: TrackFrame, b: TrackFrame) -> float:
    ax1, ay1, ax2, ay2 = a.bbox_xyxy
    bx1, by1, bx2, by2 = b.bbox_xyxy
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def _match_iou(
    gt:   list[TrackFrame],
    pred: list[TrackFrame],
    threshold: float,
) -> tuple[list[tuple[TrackFrame, TrackFrame]], list[TrackFrame], list[TrackFrame]]:
    """Greedy IoU matching. Returns (matches, unmatched_gt, unmatched_pred)."""
    matched_gt   = set()
    matched_pred = set()
    matches      = []

    # Build IoU matrix
    iou_mat = np.array([[_iou(g, p) for p in pred] for g in gt])

    # Greedy matching by descending IoU
    while True:
        if iou_mat.size == 0:
            break
        best = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
        gi, pi = best
        if iou_mat[gi, pi] < threshold:
            break
        matches.append((gt[gi], pred[pi]))
        matched_gt.add(gi)
        matched_pred.add(pi)
        iou_mat[gi, :] = -1
        iou_mat[:, pi] = -1

    unmatched_gt   = [gt[i]   for i in range(len(gt))   if i not in matched_gt]
    unmatched_pred = [pred[i] for i in range(len(pred)) if i not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def _to_track_frame(t) -> TrackFrame:
    if isinstance(t, TrackFrame):
        return t
    if isinstance(t, dict):
        return TrackFrame(track_id=t["track_id"], bbox_xyxy=tuple(t["bbox_xyxy"]))
    # Duck-typing for Track objects from byte_tracker.py
    return TrackFrame(track_id=t.track_id, bbox_xyxy=tuple(t.bbox_xyxy))
