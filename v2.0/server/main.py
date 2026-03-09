"""
CognEn v2.0 — FastAPI Entry Point
Streams CCTV frames through the multi-person analysis pipeline and
broadcasts per-room analytics to connected WebSocket clients.
"""

from __future__ import annotations
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from server.config import API_HOST, API_PORT, WS_PATH, load_seat_map
from inference.room_analyzer import RoomAnalyzer

logger = logging.getLogger("cognen.v2")

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
app.mount("/static", StaticFiles(directory=Path(__file__).parent.parent / "dashboard/static"), name="static")


# ── REST: room management ────────────────────────────────────────────────────

@app.post("/rooms/{room_id}/start")
async def start_room(room_id: str, stream_url: str):
    """Register a CCTV stream URL for a room and start analysis."""
    if room_id not in _analyzers:
        seat_map = load_seat_map(room_id)
        _analyzers[room_id] = RoomAnalyzer(room_id=room_id, seat_map=seat_map)

    analyzer = _analyzers[room_id]
    asyncio.create_task(analyzer.run_stream(stream_url))
    return {"status": "started", "room_id": room_id}


@app.get("/rooms/{room_id}/snapshot")
async def snapshot(room_id: str):
    """Return the latest analytics snapshot for a room."""
    if room_id not in _analyzers:
        return {"error": "Room not found"}
    return _analyzers[room_id].get_snapshot()


# ── WebSocket: real-time dashboard ───────────────────────────────────────────

@app.websocket(WS_PATH)
async def room_websocket(websocket: WebSocket, room_id: str):
    """
    Push room analytics at ~5 Hz to connected dashboard clients.
    Message format:
      { "ts": float, "persons": [...], "occupancy": {...}, "room_attention": float }
    """
    await websocket.accept()

    if room_id not in _analyzers:
        await websocket.send_json({"error": "Room not found"})
        await websocket.close()
        return

    analyzer = _analyzers[room_id]
    try:
        while True:
            payload = analyzer.get_snapshot()
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.2)   # 5 Hz push
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: room=%s", room_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("server.main:app", host=API_HOST, port=API_PORT, reload=False)
