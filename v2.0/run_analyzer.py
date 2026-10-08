"""
run_analyzer.py — CLI entry point for CognEn v2.0 room analyzer.

Runs the full detection → tracking → behavior pipeline on any video source
without requiring the FastAPI server.  Useful for local testing, debugging,
and offline video processing.

Usage
-----
# RTSP CCTV stream
python3 v2.0/run_analyzer.py \\
    --source rtsp \\
    --url "rtsp://192.168.1.100:554/channel1" \\
    --room room_default \\
    --process-every 3

# Local video file
python3 v2.0/run_analyzer.py \\
    --source file \\
    --url /data/classroom_clip.mp4 \\
    --room room_default \\
    --loop

# Webcam (device 0)
python3 v2.0/run_analyzer.py \\
    --source camera \\
    --device 0 \\
    --room room_default

# Show annotated preview window (adds ~2 ms overhead)
python3 v2.0/run_analyzer.py --source camera --show

Press Q (or Ctrl+C) to stop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure v2.0/ is on the import path
sys.path.insert(0, str(Path(__file__).parent))

from input.frame_source import make_source
from server.config import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    PROCESS_EVERY_N_FRAMES,
    RTSP_TRANSPORT,
    load_seat_map,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
logger = logging.getLogger("cognen.run")


# ── Annotation helpers ────────────────────────────────────────────────────────

_BEHAVIOR_COLORS: dict[str, tuple[int, int, int]] = {
    "attentive_screen":  (0, 200,  80),   # green
    "attentive_reading": (0, 160, 220),   # sky blue
    "using_phone":       (0,   0, 220),   # red
    "distracted":        (0, 140, 255),   # orange
    "drowsy":            (128, 0, 128),   # purple
    "absent":            (160, 160, 160), # grey
}
_DEFAULT_COLOR = (200, 200, 200)


def _draw_snapshot(frame: np.ndarray, snapshot: dict) -> np.ndarray:
    """Overlay bounding boxes and behavior labels on a frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    for person in snapshot.get("persons", []):
        tid   = person.get("track_id", "?")
        label = person.get("behavior", "unknown")
        score = person.get("attention_score", 0)
        conf  = person.get("confidence", 0.0)
        seat  = person.get("seat_id") or "—"
        color = _BEHAVIOR_COLORS.get(label, _DEFAULT_COLOR)

        # Person bboxes are not stored in snapshot (tracked internally);
        # draw a status block per person in the top-left corner instead.
        text = f"T{tid}  {label}  score={score:.0f}  conf={conf:.2f}  seat={seat}"
        y_off = 30 + list(snapshot["persons"]).index(person) * 22
        cv2.putText(out, text, (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # Room-level summary at bottom
    occ = snapshot.get("occupancy", {})
    summary = (
        f"Room attention: {snapshot.get('room_attention', 0):.1f}  |  "
        f"Present: {len(occ.get('occupied', []))}  "
        f"Absent: {len(occ.get('absent', []))}  "
        f"Total seats: {occ.get('total', 0)}"
    )
    cv2.putText(out, summary, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
    return out


# ── Main loop ─────────────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> None:
    from inference.room_analyzer import RoomAnalyzer

    seat_map = load_seat_map(args.room)
    if not seat_map:
        logger.warning(
            "No seat map found for room '%s'. "
            "Occupancy tracking disabled. "
            "Add v2.0/data/seat_maps/%s.json to enable it.",
            args.room, args.room,
        )

    analyzer = RoomAnalyzer(room_id=args.room, seat_map=seat_map)

    source = make_source(
        source_type    = args.source,
        url            = args.url,
        device         = args.device,
        loop_file      = args.loop,
        rtsp_transport = args.rtsp_transport,
        rtsp_max_reconnect = args.rtsp_max_reconnect,
        camera_width   = args.width,
        camera_height  = args.height,
        camera_fps     = args.camera_fps,
    )

    show = args.show
    if show:
        cv2.namedWindow("CognEn v2.0", cv2.WINDOW_NORMAL)

    # Run the analyzer pipeline in a background task
    pipeline_task = asyncio.create_task(
        analyzer.run_source(source, process_every_n=args.process_every)
    )

    logger.info("Running. Press Q (preview window) or Ctrl+C to stop.")

    try:
        while not pipeline_task.done():
            snapshot = analyzer.get_snapshot()

            # Print JSON snapshot to stdout if requested
            if args.print_json and snapshot:
                print(json.dumps(snapshot, indent=2))

            # Annotated preview (requires display)
            if show and snapshot:
                frame = analyzer.get_frame()
                canvas = frame if frame is not None else np.zeros(
                    (INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8
                )
                canvas = _draw_snapshot(canvas, snapshot)
                cv2.imshow("CognEn v2.0", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Q pressed — stopping.")
                    break

            await asyncio.sleep(0.2)  # 5 Hz UI refresh

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Interrupted.")
    finally:
        analyzer.shutdown()
        await asyncio.sleep(0.1)  # let pipeline_task notice the shutdown flag
        pipeline_task.cancel()
        if show:
            cv2.destroyAllWindows()
        logger.info("Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CognEn v2.0 room analyzer — run on file / RTSP / camera",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Source
    src = p.add_argument_group("source")
    src.add_argument(
        "--source", choices=["rtsp", "file", "camera"], default="camera",
        help="Input type: rtsp | file | camera  (default: camera)",
    )
    src.add_argument("--url",    default=None, help="RTSP URL or video file path")
    src.add_argument("--device", type=int, default=0, help="Camera device index (default: 0)")
    src.add_argument("--loop",   action="store_true", help="Loop video files at EOF")

    # RTSP options
    rtsp = p.add_argument_group("rtsp options")
    rtsp.add_argument(
        "--rtsp-transport", dest="rtsp_transport",
        choices=["tcp", "udp"], default=RTSP_TRANSPORT,
        help=f"RTSP transport protocol (default: {RTSP_TRANSPORT})",
    )
    rtsp.add_argument(
        "--rtsp-max-reconnect", dest="rtsp_max_reconnect",
        type=int, default=0,
        help="Max RTSP reconnect attempts; 0 = infinite (default: 0)",
    )

    # Camera options
    cam = p.add_argument_group("camera options")
    cam.add_argument("--width",      type=int,   default=None, help="Camera capture width")
    cam.add_argument("--height",     type=int,   default=None, help="Camera capture height")
    cam.add_argument("--camera-fps", type=float, dest="camera_fps",
                     default=None, help="Camera target FPS")

    # Processing
    proc = p.add_argument_group("processing")
    proc.add_argument(
        "--process-every", dest="process_every",
        type=int, default=PROCESS_EVERY_N_FRAMES,
        help=(
            f"Analyse every Nth frame (default: {PROCESS_EVERY_N_FRAMES}). "
            "Use 3 on a 15 FPS source for ~5 analysis FPS."
        ),
    )
    proc.add_argument(
        "--room", default="room_default",
        help="Room ID — used to load the matching seat map JSON (default: room_default)",
    )

    # Output
    out = p.add_argument_group("output")
    out.add_argument(
        "--show", action="store_true",
        help="Display annotated preview window (requires display)",
    )
    out.add_argument(
        "--print-json", dest="print_json", action="store_true",
        help="Print each snapshot as JSON to stdout",
    )

    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(_run(_parse_args()))
