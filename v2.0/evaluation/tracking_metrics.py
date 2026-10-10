"""
tracking_metrics.py — Multi-object tracking evaluation metrics.

Implements the MOT metrics of ARCHITECTURE.md §7:
    • MOTA  — Multi-Object Tracking Accuracy  (target ≥ 0.75)
    • IDF1  — identity F1                     (target ≥ 0.80)
    • ID switches per minute                  (target ≤ 5)

Definitions (MOTChallenge / CLEAR MOT and Ristani et al. 2016):

    MOTA = 1 - (FN + FP + IDSW) / GT              NOT clipped: it is negative
                                                  when the errors exceed GT.
    Per frame, GT and predicted boxes are matched one-to-one: GT objects keep
    last frame's partner while IoU >= threshold, the rest are matched by the
    Hungarian algorithm on IoU (pairs below the threshold are not matches).
    IDSW = a GT object matched to a different predicted id than at its
    previous match.

    IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN)
    computed from a GLOBAL one-to-one assignment between GT identities and
    predicted identities over the whole sequence: m[i, j] = number of frames
    where GT i and prediction j co-occur with IoU >= threshold; the Hungarian
    algorithm maximises sum m over the assignment (= minimises IDFP + IDFN),
    IDTP = sum of assigned m, IDFN = |GT boxes| - IDTP, IDFP = |pred boxes| -
    IDTP. (The previous "IDF1" was per-frame detection F1, which ignores
    identity entirely.)

Sequences: call ``start_sequence(name)`` before each clip; ids are scoped to
their sequence (a track id 1 in clip A is not track id 1 in clip B) and the
IDSW memory is reset. Metrics are pooled over all sequences.

When nothing can be measured (no ground-truth boxes) every metric is None
with status "insufficient_data" and targets_met values are None.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np

# ── Targets ───────────────────────────────────────────────────────────────────

TARGETS = {
    "mota":          0.75,
    "idf1":          0.80,
    "id_switches_per_min": 5.0,
}


@dataclass
class TrackFrame:
    track_id: Hashable
    bbox_xyxy: tuple[float, float, float, float]


@dataclass
class FrameStats:
    tp:   int = 0
    fp:   int = 0
    fn:   int = 0
    idsw: int = 0


class TrackingEvaluator:
    """
    Accumulates per-frame GT / predicted tracks and computes MOT metrics.

    Args:
        iou_threshold: Minimum IoU for a GT / prediction match.
        fps:           Frame rate of the EVALUATED frames (ID switches / minute).
    """

    def __init__(self, iou_threshold: float = 0.5, fps: float = 5.0) -> None:
        self.iou_threshold = iou_threshold
        self.fps           = fps
        self.reset()

    # ── accumulation ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._frame_stats: list[FrameStats] = []
        self._gt_total = 0
        self._pred_total = 0
        self._n_frames = 0
        self._seq = "seq0"
        self._n_sequences = 0
        self._gt_to_pred: dict[tuple, tuple] = {}       # last matched partner (scoped ids)
        self._pair_counts: dict[tuple, dict[tuple, int]] = defaultdict(lambda: defaultdict(int))
        self._gt_ids: set[tuple] = set()
        self._pred_ids: set[tuple] = set()

    def start_sequence(self, name: str) -> None:
        """Begin a new clip: ids are scoped to *name*, IDSW memory cleared."""
        self._seq = str(name)
        self._gt_to_pred = {}
        self._n_sequences += 1

    def update(self, gt_tracks: Sequence, pred_tracks: Sequence) -> FrameStats:
        """Process one evaluated frame (GT may be empty: predictions are then FPs)."""
        gt   = [_to_track_frame(t) for t in gt_tracks]
        pred = [_to_track_frame(t) for t in pred_tracks]
        self._gt_total += len(gt)
        self._pred_total += len(pred)
        self._n_frames += 1
        seq = self._seq

        gkeys = [(seq, g.track_id) for g in gt]
        pkeys = [(seq, p.track_id) for p in pred]
        self._gt_ids.update(gkeys)
        self._pred_ids.update(pkeys)

        iou = np.array([[_iou(g, p) for p in pred] for g in gt], dtype=np.float64).reshape(len(gt), len(pred))

        # identity co-occurrence counts for IDF1 (no per-frame one-to-one constraint)
        for gi in range(len(gt)):
            for pi in range(len(pred)):
                if iou[gi, pi] >= self.iou_threshold:
                    self._pair_counts[gkeys[gi]][pkeys[pi]] += 1

        matches = _match_frame(iou, gkeys, pkeys, self._gt_to_pred, self.iou_threshold)
        idsw = 0
        for gi, pi in matches:
            prev = self._gt_to_pred.get(gkeys[gi])
            if prev is not None and prev != pkeys[pi]:
                idsw += 1
            self._gt_to_pred[gkeys[gi]] = pkeys[pi]

        stats = FrameStats(tp=len(matches), fp=len(pred) - len(matches),
                           fn=len(gt) - len(matches), idsw=idsw)
        self._frame_stats.append(stats)
        return stats

    # ── computation ──────────────────────────────────────────────────────────

    def identity_counts(self) -> tuple[int, int, int]:
        """(IDTP, IDFP, IDFN) from the global Hungarian identity assignment."""
        gt_ids = sorted(self._gt_ids, key=repr)
        pred_ids = sorted(self._pred_ids, key=repr)
        idtp = 0
        if gt_ids and pred_ids:
            m = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
            pidx = {p: j for j, p in enumerate(pred_ids)}
            for i, g in enumerate(gt_ids):
                for p, c in self._pair_counts.get(g, {}).items():
                    m[i, pidx[p]] = c
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(-m)
            idtp = int(m[rows, cols].sum())
        return idtp, self._pred_total - idtp, self._gt_total - idtp

    def compute(self) -> dict:
        total_fn   = sum(s.fn   for s in self._frame_stats)
        total_fp   = sum(s.fp   for s in self._frame_stats)
        total_tp   = sum(s.tp   for s in self._frame_stats)
        total_idsw = sum(s.idsw for s in self._frame_stats)

        duration_min = (self._n_frames / self.fps) / 60.0 if self.fps > 0 else 0.0
        base: dict[str, object] = {
            "total_tp":            total_tp,
            "total_fp":            total_fp,
            "total_fn":            total_fn,
            "id_switches":         total_idsw,
            "gt_total":            self._gt_total,
            "pred_total":          self._pred_total,
            "n_frames":            self._n_frames,
            "n_sequences":         self._n_sequences,
            "n_gt_identities":     len(self._gt_ids),
            "n_pred_identities":   len(self._pred_ids),
            "iou_threshold":       self.iou_threshold,
            "fps":                 self.fps,
        }
        if self._gt_total == 0:
            base.update({
                "status": "insufficient_data: no ground-truth boxes",
                "mota": None, "idf1": None, "idtp": None, "idfp": None, "idfn": None,
                "id_switches_per_min": None,
                "targets_met": {k: None for k in TARGETS},
            })
            return base

        mota = 1.0 - (total_fn + total_fp + total_idsw) / self._gt_total
        idtp, idfp, idfn = self.identity_counts()
        idf1_denom = 2 * idtp + idfp + idfn
        idf1 = (2.0 * idtp / idf1_denom) if idf1_denom > 0 else 0.0
        idsw_per_min = total_idsw / duration_min if duration_min > 0 else None

        base.update({
            "status": "ok",
            "mota":   round(float(mota), 4),
            "idf1":   round(float(idf1), 4),
            "idtp": idtp, "idfp": idfp, "idfn": idfn,
            "id_switches_per_min": round(idsw_per_min, 2) if idsw_per_min is not None else None,
        })
        base["targets_met"] = {
            "mota":               bool(mota >= TARGETS["mota"]),
            "idf1":               bool(idf1 >= TARGETS["idf1"]),
            "id_switches_per_min": None if idsw_per_min is None
            else bool(idsw_per_min <= TARGETS["id_switches_per_min"]),
        }
        return base

    def print_report(self) -> None:
        self.print_metrics(self.compute())

    @staticmethod
    def print_metrics(m: dict) -> None:
        """Print a compute() result."""
        def _f(v, fmt=".4f"):
            return format(v, fmt) if v is not None else "n/a"

        print("\n" + "=" * 50)
        print("  CognEn Tracking Metrics")
        print("=" * 50)
        print(f"  Status            : {m['status']}")
        print(f"  MOTA              : {_f(m['mota'])}  (target ≥ {TARGETS['mota']})")
        print(f"  IDF1              : {_f(m['idf1'])}  (target ≥ {TARGETS['idf1']})")
        print(f"  ID switches/min   : {_f(m['id_switches_per_min'], '.2f')}  "
              f"(target ≤ {TARGETS['id_switches_per_min']})")
        print(f"  TP / FP / FN      : {m['total_tp']} / {m['total_fp']} / {m['total_fn']}")
        print(f"  GT total          : {m['gt_total']}")
        print(f"  Frames evaluated  : {m['n_frames']}  ({m['n_sequences']} sequences)")
        print()
        print("  Targets met:")
        for k, v in m["targets_met"].items():
            print(f"    {'✓' if v is True else ('✗' if v is False else '–  (not evaluable)')} {k}")
        print("=" * 50 + "\n")


# ── IoU / matching helpers ────────────────────────────────────────────────────

def _iou(a: TrackFrame, b: TrackFrame) -> float:
    ax1, ay1, ax2, ay2 = a.bbox_xyxy
    bx1, by1, bx2, by2 = b.bbox_xyxy
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def _match_frame(
    iou: np.ndarray,
    gkeys: list,
    pkeys: list,
    prev_partner: dict,
    threshold: float,
) -> list[tuple[int, int]]:
    """CLEAR-MOT frame matching: keep valid previous partners, Hungarian on the rest."""
    matches: list[tuple[int, int]] = []
    used_g: set[int] = set()
    used_p: set[int] = set()
    pidx = {k: j for j, k in enumerate(pkeys)}
    for gi, gk in enumerate(gkeys):
        pk = prev_partner.get(gk)
        pi = pidx.get(pk) if pk is not None else None
        if pi is not None and pi not in used_p and iou[gi, pi] >= threshold:
            matches.append((gi, pi))
            used_g.add(gi)
            used_p.add(pi)
    rest_g = [i for i in range(len(gkeys)) if i not in used_g]
    rest_p = [j for j in range(len(pkeys)) if j not in used_p]
    if rest_g and rest_p:
        from scipy.optimize import linear_sum_assignment
        sub = iou[np.ix_(rest_g, rest_p)]
        cost = np.where(sub >= threshold, 1.0 - sub, 1e6)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if sub[r, c] >= threshold:
                matches.append((rest_g[r], rest_p[c]))
    return matches


def _match_iou(
    gt:   list[TrackFrame],
    pred: list[TrackFrame],
    threshold: float,
) -> tuple[list[tuple[TrackFrame, TrackFrame]], list[TrackFrame], list[TrackFrame]]:
    """One-to-one IoU matching (Hungarian). Returns (matches, unmatched_gt, unmatched_pred)."""
    iou = np.array([[_iou(g, p) for p in pred] for g in gt], dtype=np.float64).reshape(len(gt), len(pred))
    pairs = _match_frame(iou, list(range(len(gt))), [("p", j) for j in range(len(pred))], {}, threshold)
    mg = {g for g, _ in pairs}
    mp = {p for _, p in pairs}
    return ([(gt[g], pred[p]) for g, p in pairs],
            [gt[i] for i in range(len(gt)) if i not in mg],
            [pred[j] for j in range(len(pred)) if j not in mp])


def _to_track_frame(t) -> TrackFrame:
    if isinstance(t, TrackFrame):
        return t
    if isinstance(t, dict):
        return TrackFrame(track_id=t["track_id"], bbox_xyxy=tuple(t["bbox_xyxy"]))
    # Duck-typing for Track objects from byte_tracker.py
    return TrackFrame(track_id=t.track_id, bbox_xyxy=tuple(t.bbox_xyxy))
