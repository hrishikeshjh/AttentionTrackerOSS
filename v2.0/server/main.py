"""
CognEn v2.0 — FastAPI Entry Point
Streams CCTV frames through the multi-person analysis pipeline and
broadcasts per-room analytics to connected WebSocket clients.

Starting a room
---------------
POST /rooms/{room_id}/start
Body (JSON):
  {
    "source_type":    "rtsp" | "file" | "camera",
    "url":            "rtsp://192.168.1.100:554/ch1",   # rtsp / file
    "device":         0,                                 # camera only
    "process_every_n": 3,                                # skip N-1 frames
    "rtsp_transport": "tcp",                             # "tcp" | "udp"
    "loop_file":      false                              # loop video files
  }
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from inference.room_analyzer import RoomAnalyzer
from input.frame_source import make_source
from server.config import (
    API_HOST,
    API_PORT,
    MAX_CONCURRENT_ROOMS,
    PROCESS_EVERY_N_FRAMES,
    ROOM_ID_PATTERN,
    RTSP_TRANSPORT,
    WS_PATH,
    load_seat_map,
)

logger = logging.getLogger("cognen.v2")


def _validate_room_id(room_id: str) -> None:
    """room_id flows into filesystem paths (load_seat_map) -- reject
    anything outside a safe charset before it's used anywhere, closing the
    path-traversal hole (e.g. room_id="../../etc/passwd")."""
    if not ROOM_ID_PATTERN.match(room_id):
        raise HTTPException(status_code=422, detail=f"invalid room_id: {room_id!r}")


# ── Source configuration model ────────────────────────────────────────────────

class SourceConfig(BaseModel):
    """
    Describes the video input for a room.

    Fields
    ------
    source_type:     ``"rtsp"``, ``"file"``, or ``"camera"``
    url:             RTSP / HTTP URL or file path (required for rtsp/file)
    device:          Camera device index (camera only; default 0)
    process_every_n: Analyse 1 out of every N frames.  N=3 on a 15 FPS
                     stream yields ~5 analysis FPS with lower CPU load.
    rtsp_transport:  ``"tcp"`` (default, reliable) or ``"udp"``
    loop_file:       Loop video files when EOF is reached
    """
    source_type:     Literal["rtsp", "file", "camera"] = "rtsp"
    url:             Optional[str]  = Field(None, description="RTSP URL or file path")
    device:          int            = Field(0,    description="Camera device index")
    process_every_n: int            = Field(PROCESS_EVERY_N_FRAMES, ge=1)
    rtsp_transport:  str            = Field(RTSP_TRANSPORT)
    loop_file:       bool           = False


# ── Room analyzer registry (one per active room) ─────────────────────────────

_analyzers: dict[str, RoomAnalyzer] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("CognEn v2.0 starting…")
    yield
    for analyzer in _analyzers.values():
        analyzer.shutdown()
    logger.info("CognEn v2.0 stopped.")


app = FastAPI(title="CognEn v2.0", lifespan=lifespan)
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.parent / "dashboard/static"),
    name="static",
)


# ── REST: room management ────────────────────────────────────────────────────

@app.post("/rooms/{room_id}/start")
async def start_room(room_id: str, config: SourceConfig):
    """
    Register a video source for a room and start the analysis pipeline.

    Example — RTSP CCTV camera:
        curl -X POST "http://localhost:8000/rooms/room_a/start" \\
             -H "Content-Type: application/json" \\
             -d '{"source_type":"rtsp","url":"rtsp://192.168.1.100:554/ch1","process_every_n":3}'

    Example — local video file:
        curl -X POST "http://localhost:8000/rooms/lab/start" \\
             -H "Content-Type: application/json" \\
             -d '{"source_type":"file","url":"/data/class_session.mp4"}'

    Example — webcam:
        curl -X POST "http://localhost:8000/rooms/demo/start" \\
             -H "Content-Type: application/json" \\
             -d '{"source_type":"camera","device":0}'
    """
    _validate_room_id(room_id)

    if room_id in _analyzers and _analyzers[room_id]._running:
        return {"status": "already_running", "room_id": room_id}

    n_running = sum(1 for a in _analyzers.values() if a._running)
    if n_running >= MAX_CONCURRENT_ROOMS:
        raise HTTPException(
            status_code=429,
            detail=f"max concurrent rooms ({MAX_CONCURRENT_ROOMS}) reached; stop an existing room first",
        )

    try:
        seat_map = load_seat_map(room_id)
        analyzer = RoomAnalyzer(room_id=room_id, seat_map=seat_map)
    except FileNotFoundError as exc:
        # Model weights missing -- clear 500 with actionable message instead
        # of an opaque stack trace from deep inside ultralytics/onnxruntime.
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    _analyzers[room_id] = analyzer

    try:
        source = make_source(
            source_type=config.source_type,
            url=config.url,
            device=config.device,
            loop_file=config.loop_file,
            rtsp_transport=config.rtsp_transport,
        )
    except ValueError as exc:
        del _analyzers[room_id]
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    asyncio.create_task(
        analyzer.run_source(source, process_every_n=config.process_every_n)
    )
    logger.info(
        "Started room=%s  source=%s  process_every_n=%d",
        room_id, config.source_type, config.process_every_n,
    )
    return {
        "status":          "started",
        "room_id":         room_id,
        "source_type":     config.source_type,
        "process_every_n": config.process_every_n,
    }


@app.post("/rooms/{room_id}/stop")
async def stop_room(room_id: str):
    """Stop the analysis pipeline for a room."""
    _validate_room_id(room_id)
    analyzer = _analyzers.get(room_id)
    if analyzer is None:
        raise HTTPException(status_code=404, detail="Room not found")
    analyzer.shutdown()
    return {"status": "stopped", "room_id": room_id}


@app.get("/rooms/{room_id}/snapshot")
async def snapshot(room_id: str):
    """Return the latest analytics snapshot for a room."""
    _validate_room_id(room_id)
    if room_id not in _analyzers:
        raise HTTPException(status_code=404, detail="Room not found")
    return _analyzers[room_id].get_snapshot()


@app.get("/rooms")
async def list_rooms():
    """List all registered rooms and their running status."""
    return {
        rid: {"running": a._running, "snapshot_ts": a.get_snapshot().get("ts")}
        for rid, a in _analyzers.items()
    }


# ── WebSocket: real-time dashboard ───────────────────────────────────────────

@app.websocket(WS_PATH)
async def room_websocket(websocket: WebSocket, room_id: str):
    """
    Push room analytics at ~5 Hz to connected dashboard clients.
    Message format:
      { "ts": float, "persons": [...], "occupancy": {...}, "room_attention": float }
    """
    if not ROOM_ID_PATTERN.match(room_id):
        # Reject before accept -- same path-traversal-adjacent validation
        # as the REST endpoints (room_id isn't used in a path here, but
        # rejecting a malformed ID early is consistent and cheap).
        await websocket.close(code=1008)  # policy violation
        return

    await websocket.accept()

    if room_id not in _analyzers:
        await websocket.send_json({"error": "Room not found"})
        await websocket.close(code=1008)
        return

    analyzer = _analyzers[room_id]
    try:
        while True:
            try:
                payload = analyzer.get_snapshot()
                await websocket.send_text(json.dumps(payload))
            except (TypeError, ValueError) as exc:
                # Snapshot serialization failure -- log and skip this tick
                # rather than crashing the whole streaming task silently.
                logger.error("Snapshot serialization failed: room=%s err=%s", room_id, exc)
            await asyncio.sleep(0.2)   # 5 Hz push
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: room=%s", room_id)
    except Exception:
        logger.exception("Unexpected error in room_websocket: room=%s", room_id)
        await websocket.close(code=1011)  # internal error


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("server.main:app", host=API_HOST, port=API_PORT, reload=False)
