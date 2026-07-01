"""
evaluate.py — Main evaluation runner for CognEn v2.0.

Orchestrates all evaluators against a held-out test set and prints a
consolidated report with pass/fail status for every ARCHITECTURE target.

Usage:
    python3 v2.0/evaluation/evaluate.py \\
        --model      v2.0/models/behavior_classifier.onnx \\
        --classroom  /data/classroom \\
        --seat_map   v2.0/data/seat_maps/room_default.json \\
        --fps        5.0 \\
        [--skip_latency]

Output: JSON report written to v2.0/evaluation/reports/<timestamp>.json
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cognen.evaluate")

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.behavior_metrics  import BehaviorEvaluator
from evaluation.tracking_metrics  import TrackingEvaluator, TrackFrame
# OccupancyEvaluator import removed -- unused until a real per-seat
# ground-truth annotation format exists (see _eval_occupancy docstring).
from evaluation.latency_benchmark import LatencyBenchmark, TARGETS_MS
from training.datasets.base_dataset import WINDOW_SIZE, FEATURE_DIM


# ── Behavior evaluation ───────────────────────────────────────────────────────

def _skip(reason: str) -> dict:
    """Standard shape for a section that produced no real evaluation.

    IMPORTANT: never fill this in with fabricated numbers. This runner
    previously synthesized fake predictions here (`pred = label` when no
    real model was available, or `pred = label if rng.random() > 0.2 else
    random_class` when no real dataset was available) which made it
    indistinguishable from a real evaluation result in the output JSON —
    those numbers were quoted in README-overall.md as if they were
    measured. If you are tempted to add a synthetic fallback here, tag its
    output with "data_source": "synthetic_smoke_test" so it can never again
    be mistaken for a real result, and do not report it as "targets_met".
    """
    logger.warning("  SKIPPED: %s", reason)
    return {"status": "SKIPPED", "reason": reason, "targets_met": {}}


def _eval_behavior(args: argparse.Namespace) -> dict:
    """
    Run the real behavior classifier ONNX model on the classroom test split
    and compute F1 metrics. Returns a SKIPPED result (no numeric score) if
    either a real dataset or a real model is unavailable -- this function
    must never emit a score that isn't derived from an actual model
    prediction against real held-out labels.
    """
    logger.info("Evaluating behavior classifier …")

    if not args.classroom:
        return _skip("no --classroom dataset path provided")

    try:
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess = ort.InferenceSession(args.model, providers=providers)
        logger.info("  Using ORT session: %s", args.model)
    except Exception as exc:
        return _skip(f"no usable ONNX model at --model ({exc})")

    ev = BehaviorEvaluator()
    from training.datasets.classroom_loader import ClassroomDataset
    ds = ClassroomDataset(args.classroom, split="test")
    for window, label in ds:
        # window: (W, F) → (1, F, W)
        x = window.T[np.newaxis, :, :]  # (1, 13, 15)
        logits = sess.run(["logits"], {"input": x})[0]
        pred = int(np.argmax(logits))
        ev.update(pred, label)

    ev.print_report()
    result = ev.compute()
    result["data_source"] = "real"
    return result


# ── Tracking evaluation ───────────────────────────────────────────────────────

def _eval_tracking(args: argparse.Namespace) -> dict:
    """
    Evaluate the REAL ByteTrack tracker against ground-truth tracks in
    classroom test clips. Returns a SKIPPED result if no classroom dataset
    is available -- this must never run the tracker's ground truth through
    a fixed-offset noise injector and report the result as a tracking score
    (that measures the noise injector, not the tracker).
    """
    logger.info("Evaluating tracking …")

    if not args.classroom:
        return _skip("no --classroom dataset path provided")

    import json as _json
    import cv2
    from tracking.byte_tracker import ByteTrackWrapper
    from perception.person_detector import PersonDetector

    classroom_root = Path(args.classroom)
    ann_dir = classroom_root / "annotations"
    vid_dir = classroom_root / "videos"

    ev = TrackingEvaluator(iou_threshold=0.5, fps=args.fps)
    detector = PersonDetector()
    n_clips = 0

    for ann_path in sorted(ann_dir.glob("*.json")):
        clip_id = ann_path.stem
        video_path = vid_dir / f"{clip_id}.mp4"
        if not video_path.exists():
            logger.warning("  Skipping %s: no matching video file", clip_id)
            continue

        with open(ann_path) as f:
            ann = _json.load(f)

        frame_gt: dict[int, list[TrackFrame]] = {}
        for person in ann.get("persons", []):
            tid = person["track_id"]
            for fdata in person.get("frames", []):
                fi = fdata["frame_idx"]
                box = fdata.get("bbox_xyxy", [0, 0, 0, 0])
                frame_gt.setdefault(fi, []).append(TrackFrame(tid, tuple(box)))

        if not frame_gt:
            continue

        cap = cv2.VideoCapture(str(video_path))
        tracker = ByteTrackWrapper(frame_rate=int(args.fps))  # fresh tracker state per clip
        n_clips += 1
        fi = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            if fi in frame_gt:
                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame)
                pred_tracks = [TrackFrame(t.track_id, tuple(t.bbox_xyxy)) for t in tracks]
                ev.update(frame_gt[fi], pred_tracks)
            fi += 1
        cap.release()

    if n_clips == 0:
        return _skip("--classroom dataset has no annotation/video pairs to evaluate")

    ev.print_report()
    result = ev.compute()
    result["data_source"] = "real"
    return result


# ── Occupancy evaluation ──────────────────────────────────────────────────────

def _eval_occupancy(args: argparse.Namespace) -> dict:
    """
    No real-data occupancy evaluation path exists yet -- there is currently
    no annotation format or loader wiring per-seat ground-truth occupancy
    (unlike behavior/tracking, which have ClassroomDataset + annotation
    JSONs). This previously ran a synthetic ~95%-accuracy generator
    unconditionally and reported it as a real score; that has been removed.
    Building the real path requires a per-seat occupancy ground-truth
    annotation format, which is out of scope for this fix -- tracked
    separately.
    """
    logger.info("Evaluating seat occupancy …")
    return _skip(
        "no real-data occupancy evaluation path is implemented yet "
        "(needs a per-seat ground-truth annotation format)"
    )


# ── Latency benchmark ─────────────────────────────────────────────────────────

def _eval_latency(args: argparse.Namespace) -> dict:
    if args.skip_latency:
        logger.info("Latency benchmark skipped (--skip_latency).")
        return {}

    logger.info("Running latency benchmark …")
    bench = LatencyBenchmark("v2.0 inference (ONNX)")

    if args.model and Path(args.model).exists():
        try:
            import onnxruntime as ort
            sess  = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
            dummy = np.zeros((1, FEATURE_DIM, WINDOW_SIZE), dtype=np.float32)
            for i in range(120):
                with bench.measure():
                    sess.run(["logits"], {"input": dummy})
                if i == 9:
                    bench.reset()  # discard warmup
        except Exception as exc:
            logger.warning("  ORT benchmark failed: %s", exc)
    else:
        logger.warning("  Model not found; skipping latency.")

    result = bench.compute()
    print(bench.summary(target_version="v2.0"))
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "behavior":  _eval_behavior(args),
        "tracking":  _eval_tracking(args),
        "occupancy": _eval_occupancy(args),
        "latency":   _eval_latency(args),
    }

    # Overall pass/fail
    all_targets = {}
    for section in ("behavior", "tracking", "occupancy"):
        all_targets.update(report[section].get("targets_met", {}))

    latency = report["latency"]
    if latency and "p95_ms" in latency:
        all_targets["v2_latency"] = latency["p95_ms"] <= TARGETS_MS["v2.0"]

    n_pass = sum(all_targets.values())
    n_total = len(all_targets)
    report["summary"] = {"targets_met": all_targets, "score": f"{n_pass}/{n_total}"}

    print("\n" + "=" * 60)
    print(f"  OVERALL: {n_pass}/{n_total} ARCHITECTURE targets met")
    print("=" * 60)
    for k, v in all_targets.items():
        print(f"  {'✓' if v else '✗'} {k}")
    print("=" * 60 + "\n")

    # Write JSON report
    report_dir = Path(__file__).parent / "reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    class _Encoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (bool, int, float)):
                return super().default(o)
            try:
                import numpy as _np
                if isinstance(o, _np.bool_):
                    return bool(o)
                if isinstance(o, (_np.integer,)):
                    return int(o)
                if isinstance(o, (_np.floating,)):
                    return float(o)
            except ImportError:
                pass
            return super().default(o)

    report_path.write_text(json.dumps(report, indent=2, cls=_Encoder))
    logger.info("Report written → %s", report_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CognEn v2.0 evaluation runner")
    p.add_argument("--model",     type=str, default=None,
                   help="Path to behavior_classifier.onnx")
    p.add_argument("--classroom", type=str, default=None,
                   help="Classroom dataset root (for behavior + tracking)")
    p.add_argument("--seat_map",  type=str,
                   default=str(Path(__file__).parent.parent / "data" / "seat_maps" / "room_default.json"))
    p.add_argument("--fps",          type=float, default=5.0)
    p.add_argument("--skip_latency", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
