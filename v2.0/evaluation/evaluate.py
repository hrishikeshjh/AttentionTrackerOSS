"""
evaluate.py — Main evaluation runner for CognEn v2.0.

Runs every evaluator on the TEST split of a real classroom dataset
(training/datasets/classroom_loader.py layout; split from metadata.csv) and
writes a JSON report.

Usage:
    python3 v2.0/evaluation/evaluate.py \\
        --model      v2.0/models/behavior_classifier_v3.onnx \\
        --classroom  /data/classroom \\
        --seat_map   v2.0/data/seat_maps/room_default.json \\
        --fps        5.0 \\
        [--latency_video clip.mp4] [--skip_latency] [--out report.json]

Every report carries
    data_source:     "real" (some section measured real held-out data),
                     "none" (nothing measured);
                     synthetic numbers come only from
                     evaluation/synthetic_holdout_eval.py ("synthetic")
    claims_allowed:  True only for data_source "real" with at least one
                     evaluated target. Sections that could not run are
                     {"status": "SKIPPED", "reason": ...}; targets that cannot
                     be evaluated are None, never a placeholder pass.

Sections
--------
behavior   Ground-truth person boxes of the test clips -> the same per-person
           measurement as the server (TrackSequenceEncoder) -> BOTH
             deployed: BehaviorClassifier(session) = rules < 15 frames,
                       absent override, ONNX + context gate (what
                       RoomAnalyzer reports), and
             raw_onnx: the ONNX model alone on the same 15-frame windows.
           Both are scored on the same frames (full windows only) with the
           real stream id as person_id, so temporal consistency is measured
           (it was a vacuous 1.0 before: no person id was ever passed).
tracking   PersonDetector + ByteTrack on the test clips at the analysis
           cadence (--fps), frames resized to the server input size; MOTA
           (unclipped), real IDF1 (global identity assignment), ID
           switches / min.
occupancy  SeatMapper on the tracks vs the annotated seat_id of every GT
           person (skipped when the annotation has no seat ids).
latency    Full RoomAnalyzer._process_frame on real frames, per stage
           (evaluation/latency_benchmark.py).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

logger = logging.getLogger("cognen.evaluate")

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.behavior_metrics import BehaviorEvaluator
from evaluation.latency_benchmark import (
    benchmark_pipeline_stages,
    format_stage_report,
    missing_pipeline_models,
)
from evaluation.occupancy_metrics import OccupancyEvaluator
from evaluation.tracking_metrics import TrackFrame, TrackingEvaluator
from training.datasets.base_dataset import CLASS_TO_IDX, WINDOW_SIZE

DATA_SOURCES = ("real", "synthetic", "none")


def _skip(reason: str) -> dict:
    """A section that produced no evaluation. Never fill this with numbers."""
    logger.warning("  SKIPPED: %s", reason)
    return {"status": "SKIPPED", "reason": reason, "data_source": "none", "targets_met": {}}


# ── Behavior ──────────────────────────────────────────────────────────────────

def _onnx_window_predictions(session, vectors: np.ndarray, ends: list[int]) -> list[int]:
    """Raw ONNX argmax for the 15-frame windows ending at *ends*."""
    if not ends:
        return []
    dim = int(session.get_inputs()[0].shape[1]) if hasattr(session, "get_inputs") else vectors.shape[1]
    x = np.stack([vectors[e - WINDOW_SIZE + 1: e + 1, :dim].T for e in ends]).astype(np.float32)
    logits = session.run(None, {"input": np.ascontiguousarray(x)})[0]
    return [int(i) for i in np.argmax(logits, axis=1)]


def evaluate_behavior(sequences: Iterable, session) -> dict:
    """
    Score deployed and raw-ONNX predictions on labelled sequences produced
    with the DEPLOYED classifier as the encoder's classifier
    (LabelledSequence frames carry its results).
    """
    ev_dep, ev_raw = BehaviorEvaluator(), BehaviorEvaluator()
    n_seq = 0
    for seq in sequences:
        if len(seq) < WINDOW_SIZE:
            continue
        n_seq += 1
        ends = list(range(WINDOW_SIZE - 1, len(seq)))
        raw = _onnx_window_predictions(session, seq.vectors, ends)
        for e, rp in zip(ends, raw):
            y = int(seq.labels[e])
            ev_dep.update(CLASS_TO_IDX[seq.frames[e].result.label], y, person_id=seq.stream_id)
            ev_raw.update(rp, y, person_id=seq.stream_id)
    if n_seq == 0:
        return _skip("no test sequence with at least 15 frames at the analysis cadence")
    dep_metrics, raw_metrics = ev_dep.compute(), ev_raw.compute()
    return {
        "status": "ok",
        "data_source": "real",
        "primary": "deployed",
        "deployed": dep_metrics,
        "raw_onnx": raw_metrics,
        "n_sequences": n_seq,
        "note": "person boxes are the ground-truth boxes (isolates the classifier from detection/tracking)",
        "targets_met": dep_metrics["targets_met"],
    }


def _eval_behavior(args: argparse.Namespace) -> dict:
    logger.info("Evaluating behavior classifier …")
    if not args.classroom:
        return _skip("no --classroom dataset path provided")
    from inference.behavior_classifier import BehaviorClassifier, load_onnx_session
    session = load_onnx_session(args.model) if args.model else None
    if session is None:
        return _skip(f"no usable ONNX model at --model ({args.model})")
    from training.datasets.classroom_loader import ClassroomDataset
    ds = ClassroomDataset(args.classroom, split="test")
    result = evaluate_behavior(
        ds.iter_sequences(classifier_factory=lambda: BehaviorClassifier(session=session)), session)
    if result.get("status") == "ok":
        BehaviorEvaluator.print_metrics(result["deployed"], "deployed")
        BehaviorEvaluator.print_metrics(result["raw_onnx"], "raw ONNX")
    return result


# ── Tracking + occupancy (one pass over the test clips) ──────────────────────

def _clip_ground_truth(ann: dict, sx: float, sy: float) -> tuple[dict[int, list], dict[int, set], bool]:
    """frame_idx -> GT TrackFrames (scaled to the server input size), seat sets, has_seats."""
    gt: dict[int, list[TrackFrame]] = {}
    seats: dict[int, set[str]] = {}
    has_seats = False
    for person in ann.get("persons", []):
        tid = person.get("track_id")
        seat = person.get("seat_id")
        has_seats = has_seats or seat is not None
        for fd in person.get("frames", []):
            fi = int(fd["frame_idx"])
            x1, y1, x2, y2 = (float(v) for v in fd.get("bbox_xyxy", [0, 0, 0, 0]))
            gt.setdefault(fi, []).append(TrackFrame(tid, (x1 * sx, y1 * sy, x2 * sx, y2 * sy)))
            if seat is not None:
                seats.setdefault(fi, set()).add(seat)
    return gt, seats, has_seats


def evaluate_tracking_occupancy(
    root: str | Path,
    seat_map: dict | None,
    fps: float,
    detector,
    tracker_factory: Callable[[], Any],
    seat_mapper_factory: Callable[[dict], Any] | None = None,
    split: str = "test",
) -> tuple[dict, dict]:
    """Run detector + tracker (+ seat mapper) on the *split* clips; return (tracking, occupancy)."""
    import cv2

    from server.config import INPUT_HEIGHT, INPUT_WIDTH
    from training.datasets.classroom_loader import find_clip_video, load_split_clip_ids
    from training.datasets.sequence_encoder import subsample_by_timestamp

    root = Path(root)
    clip_ids = load_split_clip_ids(root, split)
    if not clip_ids:
        r = _skip(f"no '{split}' clips in {root / 'metadata.csv'}")
        return r, r

    trk = TrackingEvaluator(iou_threshold=0.5, fps=fps)
    occ = OccupancyEvaluator(sorted(seat_map), fps=fps) if seat_map else None
    any_seats, n_clips = False, 0

    for clip_id in clip_ids:
        ann_path = root / "annotations" / f"{clip_id}.json"
        video = find_clip_video(root, clip_id)
        if not ann_path.exists() or video is None:
            logger.warning("  Skipping %s: missing annotation or video", clip_id)
            continue
        ann = json.loads(ann_path.read_text())
        cap = cv2.VideoCapture(str(video))
        vfps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or float(ann.get("fps") or 0.0)
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if vfps <= 0 or vw <= 0 or vh <= 0:
            cap.release()
            logger.warning("  Skipping %s: unreadable video geometry / fps", clip_id)
            continue
        gt, seats, has_seats = _clip_ground_truth(ann, INPUT_WIDTH / vw, INPUT_HEIGHT / vh)
        any_seats = any_seats or has_seats
        if "n_frames" in ann:
            evaluated = list(range(int(ann["n_frames"])))      # empty frames count (FPs)
        else:
            evaluated = sorted(gt)
        keep = {evaluated[k] for k in subsample_by_timestamp([f * 1000.0 / vfps for f in evaluated],
                                                              1000.0 / fps)}
        if not keep:
            cap.release()
            continue
        n_clips += 1
        trk.start_sequence(clip_id)
        if occ is not None:
            occ.start_sequence()
        tracker = tracker_factory()
        mapper = seat_mapper_factory(seat_map) if (seat_mapper_factory and seat_map) else None
        fi, last = 0, max(keep)
        while fi <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if fi in keep:
                resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
                tracks = tracker.update(detector.detect(resized), resized)
                trk.update(gt.get(fi, []), [TrackFrame(t.track_id, tuple(t.bbox_xyxy)) for t in tracks])
                if occ is not None and mapper is not None and has_seats:
                    pred = {s.seat_id for s in mapper.assign(tracks) if s.occupied}
                    occ.update(pred, seats.get(fi, set()))
            fi += 1
        cap.release()

    if n_clips == 0:
        r = _skip(f"the '{split}' split has no annotation/video pairs to evaluate")
        return r, r
    t_res = trk.compute()
    t_res.update({"data_source": "real", "split": split})
    if t_res["status"] != "ok":
        t_res["data_source"] = "none"
    if occ is None or not any_seats:
        o_res = _skip("no seat map, or no seat_id in the test annotations")
    else:
        o_res = occ.compute()
        o_res.update({"status": "ok", "data_source": "real", "split": split})
    return t_res, o_res


def _eval_tracking_occupancy(args: argparse.Namespace) -> tuple[dict, dict]:
    logger.info("Evaluating tracking + occupancy (test split) …")
    if not args.classroom:
        r = _skip("no --classroom dataset path provided")
        return r, r
    missing = [m for m in missing_pipeline_models() if "yolov8m.pt" in m]
    if missing:
        r = _skip(f"person detector model missing: {missing}")
        return r, r
    from perception.person_detector import PersonDetector
    from tracking.byte_tracker import ByteTrackWrapper
    from tracking.seat_mapper import SeatMapper
    seat_map = json.loads(Path(args.seat_map).read_text()) if args.seat_map and Path(args.seat_map).exists() \
        else None
    t_res, o_res = evaluate_tracking_occupancy(
        args.classroom, seat_map, args.fps, PersonDetector(),
        tracker_factory=lambda: ByteTrackWrapper(frame_rate=int(round(args.fps))),
        seat_mapper_factory=SeatMapper,
    )
    if t_res.get("status") == "ok":
        TrackingEvaluator.print_metrics(t_res)
    if o_res.get("status") == "ok":
        OccupancyEvaluator.print_metrics(o_res)
    return t_res, o_res


# ── Latency ───────────────────────────────────────────────────────────────────

def _latency_frames(args: argparse.Namespace, n: int = 60) -> list[np.ndarray]:
    """Real frames for the latency benchmark: --latency_video, else the first test clip."""
    import cv2

    from server.config import INPUT_HEIGHT, INPUT_WIDTH
    from training.datasets.sequence_encoder import iter_video_at_cadence

    video = Path(args.latency_video) if args.latency_video else None
    if video is None and args.classroom:
        from training.datasets.classroom_loader import find_clip_video, load_split_clip_ids
        for cid in load_split_clip_ids(args.classroom, "test"):
            video = find_clip_video(Path(args.classroom), cid)
            if video is not None:
                break
    if video is None or not video.exists():
        return []
    frames = []
    for _i, _t, f in iter_video_at_cadence(video, 1000.0 / args.fps):
        frames.append(cv2.resize(f, (INPUT_WIDTH, INPUT_HEIGHT)))
        if len(frames) >= n:
            break
    return frames


def _eval_latency(args: argparse.Namespace) -> dict:
    if args.skip_latency:
        return _skip("--skip_latency")
    logger.info("Running full-pipeline latency benchmark …")
    missing = missing_pipeline_models()
    if missing:
        return _skip(f"pipeline models missing: {missing}")
    frames = _latency_frames(args)
    if not frames:
        return _skip("no real video frames (pass --latency_video or a --classroom dataset with test clips)")
    from inference.room_analyzer import RoomAnalyzer
    seat_map = json.loads(Path(args.seat_map).read_text()) if args.seat_map else {}
    analyzer = RoomAnalyzer("latency_benchmark", seat_map)
    result = benchmark_pipeline_stages(analyzer, frames)
    print(format_stage_report(result))
    result["data_source"] = "real" if result.get("status") == "ok" else "none"
    result["targets_met"] = {"v2_latency_p95": result.get("target_met")}
    return result


# ── Report ────────────────────────────────────────────────────────────────────

def build_report(sections: dict[str, dict]) -> dict:
    """Assemble the report: data_source / claims_allowed / target summary."""
    targets: dict[str, bool | None] = {}
    for name, sec in sections.items():
        for k, v in (sec.get("targets_met") or {}).items():
            targets[f"{name}.{k}"] = v
    evaluated = {k: v for k, v in targets.items() if v is not None}
    n_pass = sum(1 for v in evaluated.values() if v)
    sources = {sec.get("data_source", "none") for sec in sections.values()}
    data_source = "real" if "real" in sources else ("synthetic" if "synthetic" in sources else "none")
    claims_allowed = data_source == "real" and len(evaluated) > 0
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_source": data_source,
        "claims_allowed": claims_allowed,
        "claims_note": (
            "measured on real held-out (test split) data" if claims_allowed else
            "NO CLAIMS: nothing was measured on real held-out data; do not quote any number from this report"
        ),
        **sections,
        "summary": {
            "targets_met": targets,
            "n_evaluated": len(evaluated),
            "n_not_evaluable": len(targets) - len(evaluated),
            "score": f"{n_pass}/{len(evaluated)} evaluated targets met",
        },
    }
    assert report["data_source"] in DATA_SOURCES
    return report


def _json_default(o):
    """numpy scalars -> Python. NaN / inf floats are rejected by
    write_report (allow_nan=False): a metric that was not measured is None."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_report(report: dict, out: str | Path | None = None) -> Path:
    if out is None:
        report_dir = Path(__file__).parent / "reports"
        report_dir.mkdir(exist_ok=True)
        out = report_dir / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out = Path(out)
    out.write_text(json.dumps(report, indent=2, default=_json_default, allow_nan=False))
    return out


def main(args: argparse.Namespace) -> dict:
    tracking, occupancy = _eval_tracking_occupancy(args)
    report = build_report({
        "behavior":  _eval_behavior(args),
        "tracking":  tracking,
        "occupancy": occupancy,
        "latency":   _eval_latency(args),
    })
    print("\n" + "=" * 60)
    print(f"  data_source={report['data_source']}  claims_allowed={report['claims_allowed']}")
    print(f"  {report['summary']['score']}  ({report['summary']['n_not_evaluable']} not evaluable)")
    print("=" * 60)
    for k, v in report["summary"]["targets_met"].items():
        print(f"  {'✓' if v is True else ('✗' if v is False else '–')} {k}")
    print("=" * 60 + "\n")
    path = write_report(report, args.out)
    logger.info("Report written → %s", path)
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CognEn v2.0 evaluation runner")
    p.add_argument("--model",     type=str, default=None,
                   help="Behavior classifier ONNX (e.g. models/behavior_classifier_v3.onnx)")
    p.add_argument("--classroom", type=str, default=None,
                   help="Classroom dataset root (test split from metadata.csv)")
    p.add_argument("--seat_map",  type=str,
                   default=str(Path(__file__).parent.parent / "data" / "seat_maps" / "room_default.json"))
    p.add_argument("--fps",          type=float, default=5.0, help="analysis cadence (server TARGET_FPS)")
    p.add_argument("--latency_video", type=str, default=None)
    p.add_argument("--skip_latency", action="store_true")
    p.add_argument("--out", type=str, default=None, help="report path (default reports/eval_<ts>.json)")
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(_parse_args())

