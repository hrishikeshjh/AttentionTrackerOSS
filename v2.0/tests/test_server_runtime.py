"""
Server-side runtime fixes: API source allowlist, RTSP transport validation,
FFmpeg env-var race, cancellable reconnect, one-to-one seat assignment,
room task bookkeeping in server/main.py, torch.load(weights_only=True).

Light tier: frame_source tests run with a fake cv2. Seat mapper tests need
shapely, server tests need fastapi + httpx, the checkpoint test needs torch
(each skipped via importorskip when absent).

Run: pytest v2.0/tests/test_server_runtime.py
"""

import asyncio
import logging
import os
import sys
import threading
import types
from pathlib import Path

import pytest

V2_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(V2_ROOT))

from input import frame_source as fs  # noqa: E402

# ── validate_api_source ──────────────────────────────────────────────────────


def test_file_source_must_be_inside_allowed_dirs(monkeypatch, tmp_path):
    allowed = tmp_path / "videos"
    allowed.mkdir()
    monkeypatch.setenv("COGNEN_VIDEO_DIRS", str(allowed))
    fs.validate_api_source("file", str(allowed / "clip.mp4"))
    for bad in ("/etc/passwd", str(allowed / ".." / "secret.mp4"), "file:///etc/passwd",
                "concat:/a|/b", "http://x/y.mp4"):
        with pytest.raises(ValueError):
            fs.validate_api_source("file", bad)


def test_default_video_dir_when_env_unset(monkeypatch):
    monkeypatch.delenv("COGNEN_VIDEO_DIRS", raising=False)
    assert fs.allowed_video_dirs() == [fs.DEFAULT_VIDEO_DIR]
    fs.validate_api_source("file", str(fs.DEFAULT_VIDEO_DIR / "a.mp4"))
    with pytest.raises(ValueError):
        fs.validate_api_source("file", "/data/class_session.mp4")


def test_stream_url_scheme_and_host_allowlist(monkeypatch):
    # Private targets and http(s) are opt-in now (SSRF defaults; see
    # test_server_security.py for the full policy matrix).
    monkeypatch.delenv("COGNEN_STREAM_HOSTS", raising=False)
    monkeypatch.setenv("COGNEN_ALLOW_PRIVATE_STREAMS", "1")
    monkeypatch.setenv("COGNEN_STREAM_SCHEMES", "rtsp,rtsps,http,https")
    monkeypatch.setattr(fs.socket, "getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))])
    fs.validate_api_source("rtsp", "rtsp://192.168.1.100:554/ch1")
    fs.validate_api_source("rtsp", "https://cam.example/stream")
    for bad in ("file:///etc/passwd", "/etc/passwd", "concat:a|b", "rtsp:///nohost",
                "pipe:0", "rtsp://cam/a\nb", None, ""):
        with pytest.raises(ValueError):
            fs.validate_api_source("rtsp", bad)
    monkeypatch.setenv("COGNEN_STREAM_HOSTS", "cam1.local, 10.0.0.5")
    fs.validate_api_source("rtsp", "rtsp://10.0.0.5/ch1")
    with pytest.raises(ValueError):
        fs.validate_api_source("rtsp", "rtsp://169.254.169.254/latest")


def test_camera_and_unknown_type(monkeypatch):
    monkeypatch.delenv("COGNEN_ALLOW_CAMERA_SOURCE", raising=False)
    with pytest.raises(ValueError):
        fs.validate_api_source("camera", None)
    monkeypatch.setenv("COGNEN_ALLOW_CAMERA_SOURCE", "1")
    fs.validate_api_source("camera", None)
    with pytest.raises(ValueError):
        fs.validate_api_source("ftp", "ftp://x")


# ── RTSPSource ────────────────────────────────────────────────────────────────

def test_rtsp_transport_validated():
    assert fs.RTSPSource("rtsp://h/x", transport="UDP")._transport == "udp"
    with pytest.raises(ValueError):
        fs.RTSPSource("rtsp://h/x", transport="tcp|evil;1")
    with pytest.raises(ValueError):
        fs.make_source("rtsp", url="rtsp://h/x", rtsp_transport="http")


class _FakeCap:
    opened = True

    def __init__(self, url, backend=None, params=None):
        self.url, self.params = url, params
        self.env_at_open = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")

    def set(self, *_):
        return True

    def get(self, *_):
        return 25.0

    def isOpened(self):
        return self.opened

    def read(self):
        return False, None

    def release(self):
        pass


def _cap_of(src: fs.FrameSource) -> _FakeCap | None:
    """The (fake) capture a source opened; sources keep it in the private ``_cap``."""
    cap = vars(src).get("_cap")
    assert cap is None or isinstance(cap, _FakeCap)
    return cap


def _fake_cv2(cap_cls=_FakeCap):
    return types.SimpleNamespace(
        VideoCapture=cap_cls, CAP_FFMPEG=1900, CAP_PROP_BUFFERSIZE=38,
        CAP_PROP_FPS=5, CAP_PROP_OPEN_TIMEOUT_MSEC=53, CAP_PROP_READ_TIMEOUT_MSEC=54,
    )


def test_ffmpeg_env_set_only_during_open_and_restored(monkeypatch):
    monkeypatch.setattr(fs, "cv2", _fake_cv2())
    monkeypatch.setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "user;value")
    src = fs.RTSPSource("rtsp://h/x", transport="udp", connection_timeout_s=2)
    assert src.open()
    cap = _cap_of(src)
    assert cap is not None
    assert cap.env_at_open == "rtsp_transport;udp"
    assert cap.params == [53, 2000, 54, 2000]
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "user;value"
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS")
    fs.RTSPSource("rtsp://h/y", transport="tcp").open()
    assert "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ


def test_concurrent_sources_see_their_own_transport(monkeypatch):
    seen = []

    class _SlowCap(_FakeCap):
        def __init__(self, url, backend=None, params=None):
            super().__init__(url, backend, params)
            threading.Event().wait(0.01)   # widen the race window
            seen.append((url, os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")))

    monkeypatch.setattr(fs, "cv2", _fake_cv2(_SlowCap))
    srcs = [fs.RTSPSource(f"rtsp://h/{i}", transport="tcp" if i % 2 else "udp") for i in range(8)]
    threads = [threading.Thread(target=s.open) for s in srcs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for url, env in seen:
        i = int(url.rsplit("/", 1)[1])
        assert env == f"rtsp_transport;{'tcp' if i % 2 else 'udp'}"


def test_request_stop_aborts_infinite_reconnect(monkeypatch):
    class _DeadCap(_FakeCap):
        opened = False

    monkeypatch.setattr(fs, "cv2", _fake_cv2(_DeadCap))
    src = fs.RTSPSource("rtsp://h/x", max_reconnect_attempts=0, max_reconnect_delay_s=30)
    monkeypatch.setattr(src, "_cap", _DeadCap("rtsp://h/x"))
    result: dict[str, object] = {}
    t = threading.Thread(target=lambda: result.setdefault("r", src.read()))
    t.start()
    threading.Event().wait(0.05)
    src.request_stop()
    t.join(timeout=5)
    assert not t.is_alive()
    assert result["r"] == (False, None)


# ── SeatMapper one-to-one ────────────────────────────────────────────────────

def _mapper(seat_map):
    pytest.importorskip("shapely")
    from tracking.seat_mapper import SeatMapper
    return SeatMapper(seat_map)


SEATS = {
    "seat_1": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
    "seat_2": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
}


def test_one_track_occupies_at_most_one_seat():
    from tracking.byte_tracker import Track
    m = _mapper(SEATS)
    # one wide box straddling both seats (IoU ~0.5 with each)
    wide = Track(5, (0, 0, 1280, 720), 0.9)
    st = m.assign([wide])
    assert sum(s.occupied for s in st) == 1


def test_second_person_not_starved_by_wide_box():
    from tracking.byte_tracker import Track
    m = _mapper(SEATS)
    wide = Track(5, (100, 0, 900, 720), 0.9)        # centre in seat_1, overlaps seat_2
    right = Track(6, (700, 0, 1280, 720), 0.9)      # centre in seat_2
    by_seat = {s.seat_id: s.track_id for s in m.assign([wide, right])}
    assert by_seat == {"seat_1": 5, "seat_2": 6}


def test_track_id_zero_seat_status():
    from tracking.byte_tracker import Track
    m = _mapper(SEATS)
    st = {s.seat_id: s for s in m.assign([Track(0, (50, 100, 500, 600), 0.9)])}
    assert st["seat_1"].occupied and st["seat_1"].track_id == 0


# ── server/main.py ───────────────────────────────────────────────────────────

class _FakeAnalyzer:
    behaviour = "ok"

    def __init__(self, room_id, seat_map):
        self.room_id = room_id
        self._running = False
        self.face_landmarks_status = "ready"
        self.stopped = False

    async def run_source(self, source, process_every_n=1):
        self._running = True
        if self.behaviour == "crash":
            raise RuntimeError("pipeline exploded")
        while not self.stopped:
            await asyncio.sleep(0.01)
        self._running = False

    def get_snapshot(self):
        return {}

    def shutdown(self):
        self.stopped = True


@pytest.fixture
def server(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import server.main as main
    from server import security
    monkeypatch.setattr(main, "RoomAnalyzer", _FakeAnalyzer)
    monkeypatch.setattr(main, "make_source", lambda **kw: object())
    monkeypatch.setattr(main, "_analyzers", {})
    monkeypatch.setattr(main, "_tasks", {})
    monkeypatch.setenv("COGNEN_VIDEO_DIRS", str(tmp_path))
    monkeypatch.setenv("COGNEN_API_KEY", "runtime-test-key-0123456789")
    monkeypatch.setenv("COGNEN_ALLOW_CAMERA_SOURCE", "1")
    security.reset_settings()
    _FakeAnalyzer.behaviour = "ok"
    with TestClient(main.app, headers={"X-API-Key": "runtime-test-key-0123456789"}) as client:
        yield main, client, tmp_path
    security.reset_settings()


def test_api_rejects_disallowed_file_and_bad_transport(server):
    main, client, _tmp = server
    r = client.post("/rooms/r1/start", json={"source_type": "file", "url": "/etc/passwd"})
    assert r.status_code == 422 and "r1" not in main._analyzers
    r = client.post("/rooms/r1/start", json={"source_type": "rtsp", "url": "rtsp://h/x", "rtsp_transport": "bogus"})
    assert r.status_code == 422
    r = client.post("/rooms/r1/start", json={"source_type": "rtsp", "url": "file:///etc/passwd"})
    assert r.status_code == 422


def test_task_is_stored_and_room_listed(server):
    main, client, tmp = server
    r = client.post("/rooms/r1/start", json={"source_type": "file", "url": str(tmp / "a.mp4")})
    assert r.status_code == 200, r.text
    assert "r1" in main._tasks
    assert client.get("/rooms").json()["r1"]["face_landmarks"] == "ready"
    assert client.post("/rooms/r1/stop").status_code == 200


def test_crash_is_logged_not_lost(server, caplog):
    main, client, tmp = server
    _FakeAnalyzer.behaviour = "crash"
    with caplog.at_level(logging.ERROR, logger="cognen.v2"):
        r = client.post("/rooms/r2/start", json={"source_type": "camera"})
        assert r.status_code == 200
        task = main._tasks["r2"]
        for _ in range(200):
            if task.done():
                break
            threading.Event().wait(0.01)
        client.get("/rooms")   # let the loop run the done-callback
    assert task.done()
    assert any("pipeline exploded" in (r.exc_text or "") or r.exc_info for r in caplog.records
               if "crashed" in r.getMessage())


# ── torch.load(weights_only=True) ────────────────────────────────────────────

class _Evil:
    def __reduce__(self):
        return (os.getcwd, ())


def test_checkpoint_load_refuses_pickled_objects(tmp_path):
    torch = pytest.importorskip("torch")
    from training.model import BehaviorClassifierCNN
    good = tmp_path / "good.pt"
    model = BehaviorClassifierCNN()
    torch.save({"epoch": 1, "model_state_dict": model.state_dict(), "val_acc": 0.5,
                "classes": ["a"], "feature_dim": 13, "window_size": 15}, good)
    BehaviorClassifierCNN.load(str(good))
    bad = tmp_path / "bad.pt"
    torch.save({"model_state_dict": model.state_dict(), "payload": _Evil()}, bad)
    with pytest.raises(Exception, match="(?i)weights_only|unpickl"):
        BehaviorClassifierCNN.load(str(bad))
