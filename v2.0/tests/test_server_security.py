"""
API security: auth (incl. rotation, WebSocket), insecure-mode opt-in, rate
limiting, body cap, CORS, security headers, sanitised errors, WebSocket
caps/origin/backpressure, SSRF policy for stream URLs, file-source escape.

The SSRF / file tests (input/frame_source.py) run in the light tier; the
server tests need fastapi + httpx and are skipped when absent.

Run: pytest v2.0/tests/test_server_security.py
"""

import asyncio
import logging
import os
import socket
import sys
from pathlib import Path

import pytest

V2_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(V2_ROOT))

from input import frame_source as fs  # noqa: E402

KEY_A = "key-alpha-0123456789abcdef"
KEY_B = "key-bravo-0123456789abcdef"


# ── SSRF policy (light tier) ─────────────────────────────────────────────────

_STREAM_ENV = ("COGNEN_STREAM_HOSTS", "COGNEN_ALLOW_PRIVATE_STREAMS", "COGNEN_STREAM_PORTS",
               "COGNEN_STREAM_SCHEMES", "COGNEN_ALLOW_URL_CREDENTIALS", "COGNEN_ALLOW_CAMERA_SOURCE")


@pytest.fixture
def clean_env(monkeypatch):
    for k in _STREAM_ENV:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _dns(monkeypatch, mapping):
    """Monkeypatch getaddrinfo: hostname -> list of IP strings."""
    calls = []

    def fake(host, port, *a, **k):
        calls.append(host)
        if host not in mapping:
            raise socket.gaierror("unknown host")
        out = []
        for ip in mapping[host]:
            fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
            sa = (ip, port, 0, 0) if fam == socket.AF_INET6 else (ip, port)
            out.append((fam, socket.SOCK_STREAM, 6, "", sa))
        return out

    monkeypatch.setattr(fs.socket, "getaddrinfo", fake)
    return calls


@pytest.mark.parametrize("url,category", [
    ("rtsp://127.0.0.1/x", "loopback"),
    ("rtsp://127.1/x", "loopback"),                 # short form, via resolver
    ("rtsp://0.0.0.0/x", "unspecified"),
    ("rtsp://[::1]/x", "loopback"),
    ("rtsp://[::]/x", "unspecified"),
    ("rtsp://10.1.2.3/x", "private"),
    ("rtsp://172.16.0.9/x", "private"),
    ("rtsp://192.168.1.100:554/ch1", "private"),
    ("rtsp://100.64.0.1/x", "private"),             # CGNAT
    ("rtsp://169.254.169.254/latest", "link-local"),  # cloud metadata
    ("rtsp://[fe80::1]/x", "link-local"),
    ("rtsp://[fd00::1]/x", "private"),              # IPv6 ULA
    ("rtsp://224.0.0.1/x", "multicast"),
    ("rtsp://[ff02::1]/x", "multicast"),
    ("rtsp://240.0.0.1/x", "reserved"),
    ("rtsp://[::ffff:127.0.0.1]/x", "loopback"),    # IPv4-mapped IPv6
    ("rtsp://[::ffff:169.254.169.254]/x", "link-local"),
    ("rtsp://[::ffff:10.0.0.1]/x", "private"),
    ("rtsp://[64:ff9b::a9fe:a9fe]/x", "link-local"),  # NAT64 of 169.254.169.254
    ("rtsp://[2002:7f00:1::]/x", "loopback"),       # 6to4 of 127.0.0.1
])
def test_ssrf_default_denies_non_public(clean_env, url, category):
    _dns(clean_env, {"127.1": ["127.0.0.1"]})
    with pytest.raises(ValueError, match=category):
        fs.validate_api_source("rtsp", url)


def test_public_literal_allowed_and_not_pinned(clean_env):
    v = fs.validate_api_source("rtsp", "rtsp://93.184.216.34:8554/live")
    assert v.connect_target == "rtsp://93.184.216.34:8554/live" and not v.pinned


def test_dns_name_resolving_to_private_is_denied(clean_env):
    _dns(clean_env, {"cam.evil.example": ["93.184.216.34", "10.0.0.7"]})   # any bad record fails
    with pytest.raises(ValueError, match="private"):
        fs.validate_api_source("rtsp", "rtsp://cam.evil.example/x")
    _dns(clean_env, {"meta.evil.example": ["169.254.169.254"]})
    with pytest.raises(ValueError, match="link-local"):
        fs.validate_api_source("rtsp", "rtsp://meta.evil.example/x")
    _dns(clean_env, {"v6.evil.example": ["::ffff:127.0.0.1"]})
    with pytest.raises(ValueError, match="loopback"):
        fs.validate_api_source("rtsp", "rtsp://v6.evil.example/x")


def test_unresolvable_host_rejected(clean_env):
    _dns(clean_env, {})
    with pytest.raises(ValueError, match="resolved"):
        fs.validate_api_source("rtsp", "rtsp://nope.invalid/x")


def test_public_name_is_pinned_to_validated_ip(clean_env):
    _dns(clean_env, {"cam.example.com": ["93.184.216.34"]})
    v = fs.validate_api_source("rtsp", "rtsp://cam.example.com/ch1?a=b")
    assert v.pinned and v.connect_target == "rtsp://93.184.216.34:554/ch1?a=b"
    _dns(clean_env, {"cam6.example.com": ["2606:2800:220:1:248:1893:25c8:1946"]})
    v6 = fs.validate_api_source("rtsp", "rtsp://cam6.example.com/ch1")
    assert v6.connect_target == "rtsp://[2606:2800:220:1:248:1893:25c8:1946]:554/ch1"


def test_allowlisted_cidr_admits_private_camera(clean_env):
    clean_env.setenv("COGNEN_STREAM_HOSTS", "192.168.1.0/24, cam.lan")
    fs.validate_api_source("rtsp", "rtsp://192.168.1.100:554/ch1")
    with pytest.raises(ValueError, match="COGNEN_STREAM_HOSTS"):
        fs.validate_api_source("rtsp", "rtsp://192.168.2.1/ch1")
    with pytest.raises(ValueError, match="COGNEN_STREAM_HOSTS"):
        fs.validate_api_source("rtsp", "rtsp://93.184.216.34/ch1")   # public but not listed
    # allowlisted hostname may resolve privately ...
    _dns(clean_env, {"cam.lan": ["10.9.9.9"]})
    v = fs.validate_api_source("rtsp", "rtsp://cam.lan/ch1")
    assert v.connect_target == "rtsp://10.9.9.9:554/ch1"
    # ... but never to metadata / loopback (DNS mistake or rebinding)
    _dns(clean_env, {"cam.lan": ["169.254.169.254"]})
    with pytest.raises(ValueError, match="link-local"):
        fs.validate_api_source("rtsp", "rtsp://cam.lan/ch1")


def test_explicit_cidr_can_admit_loopback(clean_env):
    clean_env.setenv("COGNEN_STREAM_HOSTS", "127.0.0.0/8")
    fs.validate_api_source("rtsp", "rtsp://127.0.0.1:8554/test")
    fs.validate_api_source("rtsp", "rtsp://[::ffff:127.0.0.1]:8554/test")   # mapped form


def test_allow_private_flag(clean_env):
    clean_env.setenv("COGNEN_ALLOW_PRIVATE_STREAMS", "1")
    fs.validate_api_source("rtsp", "rtsp://10.0.0.5/ch1")
    fs.validate_api_source("rtsp", "rtsp://[fd12::5]/ch1")
    for bad in ("rtsp://127.0.0.1/x", "rtsp://169.254.169.254/x", "rtsp://[::ffff:169.254.169.254]/x"):
        with pytest.raises(ValueError):
            fs.validate_api_source("rtsp", bad)


def test_credentials_schemes_ports(clean_env):
    with pytest.raises(ValueError, match="credentials"):
        fs.validate_api_source("rtsp", "rtsp://admin:pw@93.184.216.34/ch1")
    clean_env.setenv("COGNEN_ALLOW_URL_CREDENTIALS", "1")
    fs.validate_api_source("rtsp", "rtsp://admin:pw@93.184.216.34/ch1")
    for bad in ("http://93.184.216.34/x", "https://93.184.216.34/x", "ftp://93.184.216.34/x",
                "gopher://93.184.216.34/", "file:///etc/passwd", "concat:a|b", "rtsp://93.184.216.34:99999/x"):
        with pytest.raises(ValueError):
            fs.validate_api_source("rtsp", bad)
    clean_env.setenv("COGNEN_STREAM_SCHEMES", "rtsp,https")
    fs.validate_api_source("rtsp", "https://93.184.216.34/x")
    clean_env.setenv("COGNEN_STREAM_PORTS", "554,8554")
    fs.validate_api_source("rtsp", "rtsp://93.184.216.34/x")           # default 554
    with pytest.raises(ValueError, match="COGNEN_STREAM_PORTS"):
        fs.validate_api_source("rtsp", "rtsp://93.184.216.34:8080/x")


def test_redact_url():
    assert fs.redact_url("rtsp://admin:pw@10.0.0.1:554/ch1?token=abc") == "rtsp://***@10.0.0.1:554/ch1"
    assert "pw" not in fs.RTSPSource("rtsp://admin:pw@10.0.0.1/x").name


class _FakeCap:
    def __init__(self, url, backend=None, params=None):
        self.url = url
        self.env = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")

    def set(self, *_):
        return True

    def get(self, *_):
        return 25.0

    def isOpened(self):
        return True

    def read(self):
        return False, None

    def release(self):
        pass


def _fake_cv2():
    import types
    return types.SimpleNamespace(
        VideoCapture=_FakeCap, CAP_FFMPEG=1900, CAP_PROP_BUFFERSIZE=38, CAP_PROP_FPS=5,
        CAP_PROP_FRAME_COUNT=7, CAP_PROP_POS_FRAMES=1,
        CAP_PROP_OPEN_TIMEOUT_MSEC=53, CAP_PROP_READ_TIMEOUT_MSEC=54,
    )


def test_guarded_rtsp_revalidates_on_every_connect(clean_env):
    """DNS rebinding: a name that turns private after validation is refused at (re)connect."""
    clean_env.setattr(fs, "cv2", _fake_cv2())
    answers = {"cam.example.com": ["93.184.216.34"]}
    _dns(clean_env, answers)
    url = "rtsp://cam.example.com/ch1"
    guard = fs.validate_api_source("rtsp", url)
    src = fs.make_source("rtsp", url=url, guard=guard)
    assert src.open()
    assert src._cap.url == "rtsp://93.184.216.34:554/ch1"        # pinned IP literal
    assert "protocol_whitelist;rtsp" in src._cap.env and "format_whitelist" in src._cap.env
    answers["cam.example.com"] = ["127.0.0.1"]                   # rebinding
    assert src.open() is False
    assert src._cap is None


def test_guarded_http_source_disables_redirects(clean_env):
    clean_env.setattr(fs, "cv2", _fake_cv2())
    clean_env.setenv("COGNEN_STREAM_SCHEMES", "https")
    url = "https://93.184.216.34/live.ts"
    src = fs.make_source("rtsp", url=url, guard=fs.validate_api_source("rtsp", url))
    assert src.open()
    assert "max_redirects;0" in src._cap.env and "hls" not in src._cap.env


# ── File sources (light tier) ────────────────────────────────────────────────

def test_symlink_escape_rejected(clean_env, tmp_path):
    allowed = tmp_path / "videos"
    allowed.mkdir()
    secret = tmp_path / "secret.mp4"
    secret.write_bytes(b"x")
    (allowed / "link.mp4").symlink_to(secret)
    (allowed / "dirlink").symlink_to(tmp_path)
    clean_env.setenv("COGNEN_VIDEO_DIRS", str(allowed))
    for bad in (str(allowed / "link.mp4"), str(allowed / "dirlink" / "secret.mp4"),
                str(allowed / ".." / "secret.mp4")):
        with pytest.raises(ValueError, match="outside"):
            fs.validate_api_source("file", bad)


def test_file_extension_and_type(clean_env, tmp_path):
    clean_env.setenv("COGNEN_VIDEO_DIRS", str(tmp_path))
    (tmp_path / "list.m3u8").write_text("#EXTM3U\nhttp://169.254.169.254/\n")
    with pytest.raises(ValueError, match="extension"):
        fs.validate_api_source("file", str(tmp_path / "list.m3u8"))
    (tmp_path / "dir.mp4").mkdir()
    with pytest.raises(ValueError, match="regular file"):
        fs.validate_api_source("file", str(tmp_path / "dir.mp4"))


def test_guarded_file_rechecks_at_open(clean_env, tmp_path):
    """Symlink swapped after validation (TOCTOU) is refused at open."""
    clean_env.setattr(fs, "cv2", _fake_cv2())
    allowed = tmp_path / "videos"
    allowed.mkdir()
    clip = allowed / "clip.mp4"
    clip.write_bytes(b"x")
    clean_env.setenv("COGNEN_VIDEO_DIRS", str(allowed))
    guard = fs.validate_api_source("file", str(clip))
    src = fs.make_source("file", url=str(clip), guard=guard)
    assert src.open()
    assert src._cap.env.startswith("protocol_whitelist;file")
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"x")
    clip.unlink()
    clip.symlink_to(outside)
    assert fs.make_source("file", url=str(clip), guard=guard).open() is False


def test_camera_requires_opt_in(clean_env):
    with pytest.raises(ValueError, match="COGNEN_ALLOW_CAMERA_SOURCE"):
        fs.validate_api_source("camera", None)


# ── Server (fastapi + httpx) ─────────────────────────────────────────────────

class _FakeAnalyzer:
    snapshot = {"ts": 1.0, "persons": []}

    def __init__(self, room_id, seat_map):
        self.room_id = room_id
        self._running = False
        self.stopped = False
        self.face_landmarks_status = "ready"

    async def run_source(self, source, process_every_n=1):
        self._running = True
        while not self.stopped:
            await asyncio.sleep(0.01)
        self._running = False

    def get_snapshot(self):
        return dict(self.snapshot)

    def shutdown(self):
        self.stopped = True


def _server(monkeypatch, tmp_path, **env):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import server.main as main
    from server import security
    for k in list(os.environ):
        if k.startswith("COGNEN_"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("COGNEN_VIDEO_DIRS", str(tmp_path))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(main, "RoomAnalyzer", _FakeAnalyzer)
    monkeypatch.setattr(main, "make_source", lambda **kw: object())
    monkeypatch.setattr(main, "_analyzers", {})
    monkeypatch.setattr(main, "_tasks", {})
    monkeypatch.setattr(main, "_ws_registry", security.WSRegistry())
    security.reset_settings()
    return main, TestClient(main.app)


@pytest.fixture
def api(monkeypatch, tmp_path):
    main, client = _server(monkeypatch, tmp_path, COGNEN_API_KEY=f"{KEY_A},{KEY_B}")
    with client:
        yield main, client, tmp_path
    from server import security
    security.reset_settings()


def _start(client, tmp, key=KEY_A, room="r1"):
    return client.post(f"/rooms/{room}/start", json={"source_type": "file", "url": str(tmp / "a.mp4")},
                       headers={"X-API-Key": key} if key else {})


PROTECTED = [("get", "/rooms"), ("post", "/rooms/r1/start"), ("post", "/rooms/r1/stop"),
             ("get", "/rooms/r1/snapshot")]


@pytest.mark.parametrize("method,path", PROTECTED)
def test_every_route_requires_key(api, method, path):
    _main, client, _ = api
    r = getattr(client, method)(path)
    assert r.status_code == 401 and "WWW-Authenticate" in r.headers
    r = getattr(client, method)(path, headers={"X-API-Key": "wrong-key-wrong-key-wrong"})
    assert r.status_code == 403
    # query-string keys are only accepted on the WebSocket
    r = getattr(client, method)(path + f"?api_key={KEY_A}")
    assert r.status_code == 401


def test_key_rotation_both_keys_work(api):
    _main, client, tmp = api
    for key in (KEY_A, KEY_B):
        assert client.get("/rooms", headers={"X-API-Key": key}).status_code == 200
    assert _start(client, tmp, KEY_B).status_code == 200


def test_healthz_is_open_and_minimal(api):
    _main, client, _ = api
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    assert client.get("/openapi.json").status_code == 404      # docs off by default


def test_no_key_configured_returns_503(monkeypatch, tmp_path):
    main, client = _server(monkeypatch, tmp_path)
    with client:
        assert client.get("/rooms").status_code == 503
        assert client.get("/rooms", headers={"X-API-Key": KEY_A}).status_code == 503
        assert client.get("/healthz").status_code == 200


def test_insecure_opt_in_disables_auth_with_warning(monkeypatch, tmp_path, caplog):
    main, client = _server(monkeypatch, tmp_path, COGNEN_ALLOW_INSECURE="1")
    with caplog.at_level(logging.WARNING, logger="cognen.security"), client:
        assert client.get("/rooms").status_code == 200
        client.get("/rooms")
    assert sum("AUTHENTICATION IS DISABLED" in r.getMessage() for r in caplog.records) == 1


def test_bind_check_and_short_keys(monkeypatch):
    pytest.importorskip("starlette")
    from server import security
    s = security.SecuritySettings()
    security.check_bind_allowed("127.0.0.1", s)
    security.check_bind_allowed("::1", s)
    with pytest.raises(RuntimeError, match="non-loopback"):
        security.check_bind_allowed("0.0.0.0", s)
    security.check_bind_allowed("0.0.0.0", security.SecuritySettings(allow_insecure=True))
    monkeypatch.setenv("COGNEN_API_KEY", "short")
    with pytest.raises(RuntimeError, match="shorter"):
        security.SecuritySettings.from_env()
    monkeypatch.setenv("COGNEN_API_KEY", KEY_A)
    monkeypatch.setenv("COGNEN_CORS_ORIGINS", "*")
    with pytest.raises(RuntimeError, match="explicit"):
        security.SecuritySettings.from_env()


def test_start_rate_limited(api):
    _main, client, tmp = api
    codes = [_start(client, tmp, room=f"rl{i}").status_code for i in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3] == 429
    r = _start(client, tmp, room="rlx")
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


def test_auth_failures_throttled_per_ip(api):
    _main, client, _ = api
    for _ in range(10):
        assert client.get("/rooms", headers={"X-API-Key": "x" * 20}).status_code == 403
    assert client.get("/rooms", headers={"X-API-Key": "x" * 20}).status_code == 429
    # lockout applies to the IP even with a valid key until tokens refill
    assert client.get("/rooms", headers={"X-API-Key": KEY_A}).status_code == 429


def test_body_size_limit(api):
    _main, client, _ = api
    big = {"source_type": "file", "url": "a" * 20000}
    r = client.post("/rooms/r1/start", json=big, headers={"X-API-Key": KEY_A})
    assert r.status_code == 413


def test_cors_allowlist(monkeypatch, tmp_path):
    main, client = _server(monkeypatch, tmp_path, COGNEN_API_KEY=KEY_A,
                           COGNEN_CORS_ORIGINS="https://dash.example.edu")
    pre = {"Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "x-api-key"}
    with client:
        ok = client.options("/rooms", headers={"Origin": "https://dash.example.edu", **pre})
        assert ok.status_code == 200
        assert ok.headers["access-control-allow-origin"] == "https://dash.example.edu"
        assert "access-control-allow-credentials" not in ok.headers
        bad = client.options("/rooms", headers={"Origin": "https://evil.example", **pre})
        assert bad.status_code == 400 and "access-control-allow-origin" not in bad.headers
        r = client.get("/rooms", headers={"Origin": "https://evil.example", "X-API-Key": KEY_A})
        assert "access-control-allow-origin" not in r.headers


def test_security_headers(api):
    _main, client, _ = api
    r = client.get("/rooms", headers={"X-API-Key": KEY_A})
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert r.headers["cache-control"] == "no-store"
    assert client.get("/rooms").headers["x-frame-options"] == "DENY"   # also on 401


def test_errors_are_sanitised(api, monkeypatch):
    main, client, tmp = api
    h = {"X-API-Key": KEY_A}

    def boom(**_kw):
        raise RuntimeError("/secret/path/model.onnx exploded")
    monkeypatch.setattr(main, "load_seat_map", lambda rid: (_ for _ in ()).throw(
        FileNotFoundError("Model file not found: /srv/models/yolov8m.pt")))
    r = _start(client, tmp)
    assert r.status_code == 500 and "/srv" not in r.text and "yolov8m" not in r.text

    monkeypatch.setattr(main, "load_seat_map", lambda rid: {})
    monkeypatch.setattr(main, "make_source", boom)
    from fastapi.testclient import TestClient
    with TestClient(main.app, raise_server_exceptions=False) as c2:
        r = _start(c2, tmp, room="r9")
    assert r.status_code == 500
    assert "secret" not in r.text and "Traceback" not in r.text and "request_id" in r.json()

    # validation errors don't echo the submitted url (may contain credentials)
    r = client.post("/rooms/r1/start", headers=h,
                    json={"source_type": "rtsp", "url": "rtsp://u:hunter2@h/x", "bogus": 1})
    assert r.status_code == 422 and "hunter2" not in r.text
    r = client.post("/rooms/r1/start", headers=h,
                    json={"source_type": "file", "url": "/etc/shadow"})
    assert r.status_code == 422 and "/etc/shadow" not in r.text
    r = client.post("/rooms/..%2Fetc/start", headers=h, json={"source_type": "camera"})
    assert r.status_code in (404, 422) and "etc" not in r.json().get("detail", "")


def test_ssrf_via_api(api):
    _main, client, _ = api
    for url in ("rtsp://169.254.169.254/latest/meta-data", "rtsp://127.0.0.1/x", "rtsp://[::ffff:10.0.0.1]/x"):
        r = client.post("/rooms/r1/start", headers={"X-API-Key": KEY_A},
                        json={"source_type": "rtsp", "url": url})
        assert r.status_code == 422, url


def test_logs_never_contain_keys_or_url_credentials(caplog):
    pytest.importorskip("starlette")
    from server import security
    security.install_log_redaction()
    lg = logging.getLogger("uvicorn.error")
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        lg.info('%s - "WebSocket %s" [accepted]', "1.2.3.4", f"/ws/room/r1?api_key={KEY_A}")
        lg.info("opening %s", "rtsp://admin:hunter2@10.0.0.1/ch1")
    text = caplog.text
    assert KEY_A not in text and "hunter2" not in text and "[REDACTED]" in text


# ── WebSocket ────────────────────────────────────────────────────────────────

def _ws_disconnect_code(client, url, **kw):
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect) as ei:
        with client.websocket_connect(url, **kw) as ws:
            ws.receive_text()
    return ei.value.code


def test_ws_requires_auth_before_accept(api):
    main, client, tmp = api
    assert _start(client, tmp).status_code == 200
    assert _ws_disconnect_code(client, "/ws/room/r1") == 1008
    assert _ws_disconnect_code(client, "/ws/room/r1?api_key=nope-nope-nope-nope") == 1008
    with client.websocket_connect(f"/ws/room/r1?api_key={KEY_B}") as ws:
        assert ws.receive_json()["ts"] == 1.0
    with client.websocket_connect("/ws/room/r1", headers={"X-API-Key": KEY_A}) as ws:
        assert ws.receive_json()["ts"] == 1.0
        ws.send_text("ping")
        # unchanged snapshot is not resent, so the next frame is the pong
        assert ws.receive_json() == {"type": "pong"}


def test_ws_origin_check(monkeypatch, tmp_path):
    main, client = _server(monkeypatch, tmp_path, COGNEN_API_KEY=KEY_A,
                           COGNEN_CORS_ORIGINS="https://dash.example.edu")
    with client:
        assert _start(client, tmp_path).status_code == 200
        url = f"/ws/room/r1?api_key={KEY_A}"
        assert _ws_disconnect_code(client, url, headers={"Origin": "https://evil.example"}) == 1008
        with client.websocket_connect(url, headers={"Origin": "https://dash.example.edu"}) as ws:
            assert "ts" in ws.receive_json()


def test_ws_connection_cap_per_key(monkeypatch, tmp_path):
    main, client = _server(monkeypatch, tmp_path, COGNEN_API_KEY=KEY_A, COGNEN_WS_MAX_PER_KEY="1")
    with client:
        assert _start(client, tmp_path).status_code == 200
        url = f"/ws/room/r1?api_key={KEY_A}"
        with client.websocket_connect(url) as ws:
            ws.receive_json()
            assert _ws_disconnect_code(client, url) == 1013
        with client.websocket_connect(url) as ws:     # slot released on close
            ws.receive_json()


def test_ws_oversized_client_message_closes(api):
    main, client, tmp = api
    assert _start(client, tmp).status_code == 200
    from starlette.websockets import WebSocketDisconnect
    with client.websocket_connect(f"/ws/room/r1?api_key={KEY_A}") as ws:
        ws.receive_json()
        ws.send_text("x" * 10_000)
        with pytest.raises(WebSocketDisconnect) as ei:
            for _ in range(100):
                ws.receive_text()
        assert ei.value.code == 1009


def test_ws_backpressure_drops_slow_client(monkeypatch):
    pytest.importorskip("fastapi")
    import server.main as main
    from server import security

    class _StuckWS:
        async def send_text(self, _t):
            await asyncio.sleep(3600)          # client never drains

        async def close(self, code=1000, reason=""):
            pass

    monkeypatch.setattr(main, "_analyzers", {"r1": _FakeAnalyzer("r1", {})})
    s = security.SecuritySettings(ws_send_timeout_s=0.05, ws_push_interval_s=0.01)

    async def run():
        await asyncio.wait_for(main._ws_pusher(_StuckWS(), "r1", s, asyncio.Lock()), timeout=2)

    with pytest.raises(main.SlowClientError):
        asyncio.run(run())


def test_ws_sends_latest_snapshot_only(monkeypatch):
    """Unchanged snapshots aren't resent (no queue of stale frames)."""
    pytest.importorskip("fastapi")
    import server.main as main
    from server import security

    sent = []

    class _WS:
        async def send_text(self, t):
            sent.append(t)

        async def close(self, code=1000, reason=""):
            pass

    an = _FakeAnalyzer("r1", {})
    monkeypatch.setattr(main, "_analyzers", {"r1": an})
    s = security.SecuritySettings(ws_push_interval_s=0.01, ws_heartbeat_s=60)

    async def run():
        t = asyncio.create_task(main._ws_pusher(_WS(), "r1", s, asyncio.Lock()))
        await asyncio.sleep(0.1)
        an.snapshot = {"ts": 2.0}
        await asyncio.sleep(0.1)
        main._analyzers.pop("r1")
        await asyncio.wait_for(t, timeout=1)

    asyncio.run(run())
    assert len(sent) == 2 and '"ts": 2.0' in sent[1]
