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

Security (server/security.py; env vars in server/config.py + README §24):
every route except /healthz and /static requires an API key (X-API-Key
header; WebSocket also ?api_key=). No COGNEN_API_KEY => 503 on protected
routes and non-loopback binds refused, unless COGNEN_ALLOW_INSECURE=1.
/start is rate limited per key and per IP; bodies are size-capped; CORS is an
explicit allowlist; error bodies never carry tracebacks or paths.

API source allowlist (input/frame_source.py::validate_api_source): file
paths must resolve (symlinks followed) inside COGNEN_VIDEO_DIRS; stream URLs
are rtsp/rtsps by default and pass an SSRF policy (no loopback / link-local /
metadata / private targets unless COGNEN_STREAM_HOSTS or
COGNEN_ALLOW_PRIVATE_STREAMS allow them); camera needs
COGNEN_ALLOW_CAMERA_SOURCE=1.

Each room's analysis runs as an asyncio Task kept in ``_tasks``; a crash is
logged (done-callback) instead of vanishing with an unobserved exception, and
shutdown waits for the tasks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import anyio
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from inference.room_analyzer import RoomAnalyzer
from input.frame_source import MAX_URL_LENGTH, make_source, validate_api_source
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
from server.security import (
    AuthError,
    BodySizeLimitMiddleware,
    DynamicCORSMiddleware,
    Principal,
    SecurityHeadersMiddleware,
    SecuritySettings,
    WSRegistry,
    authenticate,
    check_bind_allowed,
    check_start_rate,
    get_settings,
    install_log_redaction,
    origin_allowed,
)

logger = logging.getLogger("cognen.v2")


def _validate_room_id(room_id: str) -> None:
    """room_id flows into filesystem paths (load_seat_map) -- reject
    anything outside a safe charset before it's used anywhere, closing the
    path-traversal hole (e.g. room_id="../../etc/passwd"). The detail does
    not echo the input."""
    if not ROOM_ID_PATTERN.match(room_id):
        raise HTTPException(status_code=422, detail="invalid room_id (allowed: [A-Za-z0-9_-]{1,64})")


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

    Unknown fields are rejected (extra="forbid").
    """
    model_config = ConfigDict(extra="forbid")

    source_type:     Literal["rtsp", "file", "camera"] = "rtsp"
    url:             Optional[str]  = Field(None, max_length=MAX_URL_LENGTH,
                                            description="RTSP URL or file path")
    device:          int            = Field(0, ge=0, le=63, description="Camera device index")
    process_every_n: int            = Field(PROCESS_EVERY_N_FRAMES, ge=1, le=1000)
    # Validated again in RTSPSource (an invalid COGNEN_RTSP_TRANSPORT default -> 422)
    rtsp_transport:  Literal["tcp", "udp"] = Field(RTSP_TRANSPORT)
    loop_file:       bool           = False


# ── Room analyzer registry (one per active room) ─────────────────────────────

_analyzers: dict[str, RoomAnalyzer] = {}
_tasks: dict[str, asyncio.Task] = {}
_ws_registry = WSRegistry()

SHUTDOWN_TIMEOUT_S = 10.0


def _on_task_done(room_id: str, task: asyncio.Task) -> None:
    """Observe the task's outcome so a pipeline crash is logged, not lost."""
    if task.cancelled():
        logger.info("Room task cancelled: room=%s", room_id)
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Room pipeline crashed: room=%s", room_id, exc_info=exc)
    else:
        logger.info("Room pipeline finished: room=%s", room_id)


def _room_active(room_id: str) -> bool:
    task = _tasks.get(room_id)
    return task is not None and not task.done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    install_log_redaction()
    get_settings()          # parse + validate security env now (logs misconfig / insecure mode)
    logger.info("CognEn v2.0 starting…")
    yield
    for analyzer in _analyzers.values():
        analyzer.shutdown()
    pending = [t for t in _tasks.values() if not t.done()]
    if pending:
        _done, still = await asyncio.wait(pending, timeout=SHUTDOWN_TIMEOUT_S)
        for t in still:
            t.cancel()
    logger.info("CognEn v2.0 stopped.")


_docs = get_settings().enable_docs   # also fails fast on invalid security env
app = FastAPI(
    title="CognEn v2.0",
    lifespan=lifespan,
    docs_url="/docs" if _docs else None,
    redoc_url="/redoc" if _docs else None,
    openapi_url="/openapi.json" if _docs else None,
)
# Middleware order: the last added is outermost. Headers (and the sanitised
# 500) wrap everything; then the body cap; CORS innermost.
app.add_middleware(DynamicCORSMiddleware)
app.add_middleware(BodySizeLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
# Static dashboard assets are public (no analytics data; browsers can't attach
# the API-key header to <script>/<link> loads). Everything else is protected.
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.parent / "dashboard/static"),
    name="static",
)


@app.exception_handler(RequestValidationError)
async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 without echoing the submitted values (a url may carry credentials)."""
    errors = [
        {"loc": list(e.get("loc", ())), "msg": str(e.get("msg", "")), "type": str(e.get("type", ""))}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


# ── Auth dependency ──────────────────────────────────────────────────────────

def require_principal(request: Request) -> Principal:
    try:
        return authenticate(request)
    except AuthError as exc:
        headers = {"WWW-Authenticate": 'ApiKey header="X-API-Key"'} if exc.status == 401 else None
        if exc.status == 429:
            headers = {"Retry-After": "60"}
        raise HTTPException(status_code=exc.status, detail=exc.detail, headers=headers) from None


router = APIRouter(dependencies=[Depends(require_principal)])


@app.get("/healthz", include_in_schema=False)
async def healthz():
    """Unauthenticated liveness probe; reveals nothing about rooms or config."""
    return {"status": "ok"}


# ── REST: room management ────────────────────────────────────────────────────

@router.post("/rooms/{room_id}/start")
async def start_room(room_id: str, config: SourceConfig, principal: Principal = Depends(require_principal)):
    """
    Register a video source for a room and start the analysis pipeline.

    Example — RTSP CCTV camera (private IP: needs COGNEN_STREAM_HOSTS=192.168.1.0/24
    or COGNEN_ALLOW_PRIVATE_STREAMS=1):
        curl -X POST "http://localhost:8000/rooms/room_a/start" \\
             -H "X-API-Key: $COGNEN_API_KEY" -H "Content-Type: application/json" \\
             -d '{"source_type":"rtsp","url":"rtsp://192.168.1.100:554/ch1","process_every_n":3}'

    Example — video file inside COGNEN_VIDEO_DIRS:
        curl -X POST "http://localhost:8000/rooms/lab/start" \\
             -H "X-API-Key: $COGNEN_API_KEY" -H "Content-Type: application/json" \\
             -d '{"source_type":"file","url":"data/videos/class_session.mp4"}'

    Example — webcam (needs COGNEN_ALLOW_CAMERA_SOURCE=1):
        curl -X POST "http://localhost:8000/rooms/demo/start" \\
             -H "X-API-Key: $COGNEN_API_KEY" -H "Content-Type: application/json" \\
             -d '{"source_type":"camera","device":0}'
    """
    retry_after = check_start_rate(principal)
    if retry_after is not None:
        raise HTTPException(status_code=429, detail="rate limit exceeded for /start",
                            headers={"Retry-After": str(int(retry_after + 0.999))})

    _validate_room_id(room_id)

    if _room_active(room_id):
        if room_id in _analyzers and _analyzers[room_id]._running:
            return {"status": "already_running", "room_id": room_id}
        raise HTTPException(status_code=409, detail="room is starting or stopping; retry shortly")

    # Reject disallowed sources before loading any model. Validation resolves
    # DNS (SSRF policy), so it runs in a worker thread.
    try:
        guard = await asyncio.to_thread(validate_api_source, config.source_type, config.url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    n_running = sum(1 for rid in _tasks if _room_active(rid))
    if n_running >= MAX_CONCURRENT_ROOMS:
        raise HTTPException(
            status_code=429,
            detail=f"max concurrent rooms ({MAX_CONCURRENT_ROOMS}) reached; stop an existing room first",
        )

    try:
        seat_map = load_seat_map(room_id)
        analyzer = RoomAnalyzer(room_id=room_id, seat_map=seat_map)
    except FileNotFoundError:
        # Missing model weights: the message names a filesystem path, so it
        # goes to the server log only.
        logger.exception("Model weights missing while starting room=%s", room_id)
        raise HTTPException(status_code=500,
                            detail="analysis models are not installed; see server logs") from None
    except ValueError:
        logger.exception("Invalid room configuration: room=%s", room_id)
        raise HTTPException(status_code=422, detail="invalid room configuration (seat map)") from None

    _analyzers[room_id] = analyzer

    try:
        source = make_source(
            source_type=config.source_type,
            url=config.url,
            device=config.device,
            loop_file=config.loop_file,
            rtsp_transport=config.rtsp_transport,
            guard=guard,
        )
    except ValueError as exc:
        del _analyzers[room_id]
        raise HTTPException(status_code=422, detail=str(exc)) from None

    task = asyncio.create_task(
        analyzer.run_source(source, process_every_n=config.process_every_n),
        name=f"room:{room_id}",
    )
    task.add_done_callback(lambda t, rid=room_id: _on_task_done(rid, t))
    _tasks[room_id] = task
    logger.info(
        "Started room=%s  source=%s (%s)  process_every_n=%d  key=%s",
        room_id, config.source_type, getattr(guard, "redacted", "?"),
        config.process_every_n, principal.key_id,
    )
    return {
        "status":          "started",
        "room_id":         room_id,
        "source_type":     config.source_type,
        "process_every_n": config.process_every_n,
    }


@router.post("/rooms/{room_id}/stop")
async def stop_room(room_id: str):
    """Stop the analysis pipeline for a room."""
    _validate_room_id(room_id)
    analyzer = _analyzers.get(room_id)
    if analyzer is None:
        raise HTTPException(status_code=404, detail="Room not found")
    analyzer.shutdown()
    return {"status": "stopped", "room_id": room_id}


@router.get("/rooms/{room_id}/snapshot")
async def snapshot(room_id: str):
    """Return the latest analytics snapshot for a room."""
    _validate_room_id(room_id)
    if room_id not in _analyzers:
        raise HTTPException(status_code=404, detail="Room not found")
    return _analyzers[room_id].get_snapshot()


@router.get("/rooms")
async def list_rooms():
    """List all registered rooms and their running status."""
    return {
        rid: {
            "running": a._running,
            "snapshot_ts": a.get_snapshot().get("ts"),
            "face_landmarks": getattr(a, "face_landmarks_status", "unknown"),
        }
        for rid, a in _analyzers.items()
    }


app.include_router(router)


# ── WebSocket: real-time dashboard ───────────────────────────────────────────

class SlowClientError(Exception):
    """A send did not complete within COGNEN_WS_SEND_TIMEOUT_S."""


async def _safe_close(websocket: WebSocket, code: int, reason: str = "") -> None:
    try:
        await asyncio.wait_for(websocket.close(code=code, reason=reason), timeout=1.0)
    except Exception:   # already closed / peer gone / transport stuck
        pass


async def _send_bounded(websocket: WebSocket, text: str, lock: asyncio.Lock, timeout: float) -> None:
    try:
        async with lock:
            await asyncio.wait_for(websocket.send_text(text), timeout=timeout)
    except asyncio.TimeoutError:
        raise SlowClientError from None


async def _ws_pusher(websocket: WebSocket, room_id: str, s: SecuritySettings, lock: asyncio.Lock) -> None:
    """Push the *latest* snapshot every push interval (no queue => no
    unbounded buffering). Identical payloads are not resent except as a
    heartbeat. A send that exceeds the timeout drops the client."""
    loop = asyncio.get_running_loop()
    last_text: Optional[str] = None
    last_sent = 0.0
    while True:
        analyzer = _analyzers.get(room_id)
        if analyzer is None:
            await _safe_close(websocket, 1001, "room removed")
            return
        text: Optional[str] = None
        try:
            text = json.dumps(analyzer.get_snapshot())
        except (TypeError, ValueError) as exc:
            # Snapshot serialization failure -- log and skip this tick.
            logger.error("Snapshot serialization failed: room=%s err=%s", room_id, type(exc).__name__)
        now = loop.time()
        if text is not None and (text != last_text or now - last_sent >= s.ws_heartbeat_s):
            await _send_bounded(websocket, text, lock, s.ws_send_timeout_s)
            last_text, last_sent = text, now
        await asyncio.sleep(s.ws_push_interval_s)


async def _ws_receiver(websocket: WebSocket, s: SecuritySettings, lock: asyncio.Lock) -> None:
    """Drain client messages: enforce size and rate limits, answer "ping"."""
    loop = asyncio.get_running_loop()
    window, count = loop.time(), 0
    while True:
        msg = await websocket.receive()
        if msg["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(msg.get("code", 1000))
        data = msg.get("text")
        size = len(data.encode("utf-8")) if data is not None else len(msg.get("bytes") or b"")
        if size > s.ws_max_message_bytes:
            await _safe_close(websocket, 1009, "message too big")
            return
        now = loop.time()
        if now - window >= 1.0:
            window, count = now, 0
        count += 1
        if count > s.ws_max_client_msgs_per_s:
            await _safe_close(websocket, 1008, "too many messages")
            return
        if data is not None and data.strip().lower() == "ping":
            await _send_bounded(websocket, '{"type":"pong"}', lock, s.ws_send_timeout_s)


@app.websocket(WS_PATH)
async def room_websocket(websocket: WebSocket, room_id: str):
    """
    Push room analytics (latest snapshot, up to ~5 Hz) to dashboard clients.
    Message format:
      { "ts": float, "persons": [...], "occupancy": {...}, "room_attention": float }

    Handshake checks (all before accept; a pre-accept close becomes HTTP 403):
    room_id charset, Origin in COGNEN_CORS_ORIGINS (when an Origin is sent),
    API key (X-API-Key header or ?api_key=), connection caps per key / room /
    total. Clients may send the text "ping" (reply {"type":"pong"}); other
    messages are ignored, subject to size/rate limits.
    """
    s = get_settings()
    if not ROOM_ID_PATTERN.match(room_id):
        await websocket.close(code=1008)  # policy violation
        return
    if not origin_allowed(websocket.headers.get("origin"), s):
        logger.warning("WebSocket origin rejected: room=%s", room_id)
        await websocket.close(code=1008)
        return
    try:
        principal = authenticate(websocket, allow_query=True)
    except AuthError as exc:
        await websocket.close(code=1008 if exc.status in (401, 403) else 1013)
        return
    refusal = _ws_registry.try_acquire(principal.key_id, room_id, s)
    if refusal is not None:
        logger.warning("WebSocket refused: room=%s reason=%s", room_id, refusal)
        await websocket.close(code=1013)   # try again later
        return

    try:
        await websocket.accept()
        if room_id not in _analyzers:
            await websocket.send_json({"error": "Room not found"})
            await websocket.close(code=1008)
            return

        lock = asyncio.Lock()
        outcome: list[Optional[BaseException]] = []

        # anyio task group (structured concurrency, same as Starlette): when
        # either side ends, the other is cancelled; a cancellation of this
        # handler (server shutdown / client gone) cancels both cleanly.
        async with anyio.create_task_group() as tg:
            async def _run(coro) -> None:
                try:
                    await coro
                    outcome.append(None)
                except Exception as exc:   # noqa: BLE001 -- classified below
                    outcome.append(exc)
                finally:
                    tg.cancel_scope.cancel()

            tg.start_soon(_run, _ws_pusher(websocket, room_id, s, lock))
            tg.start_soon(_run, _ws_receiver(websocket, s, lock))

        exc = outcome[0] if outcome else None
        if exc is None or isinstance(exc, WebSocketDisconnect):
            logger.info("WebSocket closed: room=%s", room_id)
        elif isinstance(exc, SlowClientError):
            logger.warning("Dropping slow WebSocket client: room=%s key=%s", room_id, principal.key_id)
            await _safe_close(websocket, 1008, "client too slow")
        else:
            logger.error("Unexpected error in room_websocket: room=%s", room_id, exc_info=exc)
            await _safe_close(websocket, 1011)  # internal error
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: room=%s", room_id)
    finally:
        _ws_registry.release(principal.key_id, room_id)


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    install_log_redaction()
    try:
        settings = get_settings()
        check_bind_allowed(API_HOST, settings)
    except RuntimeError as exc:
        logger.critical("%s", exc)
        sys.exit(2)
    uvicorn.run(
        "server.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        server_header=False,
        ws_max_size=settings.ws_max_message_bytes,
        ws_ping_interval=20.0,
        ws_ping_timeout=20.0,
    )


if __name__ == "__main__":
    main()
