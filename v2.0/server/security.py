"""
CognEn v2.0 — API security layer (server/main.py wires it in).

Everything here is configured from environment variables (documented in
server/config.py and README §24 "Security"); ``get_settings()`` parses them
once and caches the result (``reset_settings()`` re-reads, used by tests).

Pieces
------
* API-key auth   ``X-API-Key`` header (HTTP + WebSocket) or ``?api_key=``
                 (WebSocket only -- browsers cannot set WS headers). Keys come
                 from COGNEN_API_KEY (comma-separated => rotation: old and new
                 key both valid until the old one is removed). Every key is
                 compared with ``secrets.compare_digest`` (all keys, no early
                 exit). Keys shorter than 16 chars are refused at startup.
* Secure default no key configured => protected routes return 503 and
                 ``check_bind_allowed`` refuses non-loopback binds, unless
                 COGNEN_ALLOW_INSECURE=1 (auth disabled, loud warning once).
* Rate limiting  in-memory token buckets: /start per key and per client IP;
                 failed authentication per client IP (brute-force damper).
* Body size      ``BodySizeLimitMiddleware`` (COGNEN_MAX_BODY_BYTES, 16 KiB).
* CORS           explicit allowlist COGNEN_CORS_ORIGINS (default: none);
                 "*" is rejected; credentials are never allowed (auth is a
                 header, not a cookie).
* Headers        nosniff, X-Frame-Options DENY, Referrer-Policy no-referrer,
                 CSP default-src 'none' + Cache-Control no-store on API paths.
* Errors         unhandled exceptions => ``{"detail": "internal server
                 error", "request_id": ...}`` (traceback only in the server log).
* Logging        ``RedactingFilter`` scrubs api_key=..., X-API-Key values and
                 URL userinfo from log records (incl. uvicorn's access/WS lines).

Client IPs are taken from the ASGI ``client`` (the TCP peer). Behind a reverse
proxy run uvicorn with ``--proxy-headers --forwarded-allow-ips=<proxy>`` so
the peer is the real client; X-Forwarded-For is never trusted directly.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import os
import re
import secrets
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

from starlette.middleware.cors import CORSMiddleware
from starlette.requests import HTTPConnection
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("cognen.security")

API_KEY_HEADER = "x-api-key"
MIN_KEY_LENGTH = 16


def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int, lo: int = 0) -> int:
    try:
        return max(lo, int(os.getenv(name, str(default))))
    except ValueError:
        logger.warning("Invalid integer for %s; using default %d", name, default)
        return default


def _float(name: str, default: float, lo: float = 0.0) -> float:
    try:
        return max(lo, float(os.getenv(name, str(default))))
    except ValueError:
        logger.warning("Invalid number for %s; using default %s", name, default)
        return default


def key_id(key: str) -> str:
    """Non-reversible short identifier for a key (safe to log / use as a bucket key)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class SecuritySettings:
    api_keys: tuple[bytes, ...] = ()
    allow_insecure: bool = False
    cors_origins: tuple[str, ...] = ()
    max_body_bytes: int = 16 * 1024
    start_rate_per_min: float = 6.0
    start_burst: int = 3
    auth_fail_per_min: float = 10.0
    auth_fail_burst: int = 10
    ws_max_per_key: int = 8
    ws_max_per_room: int = 32
    ws_max_total: int = 256
    ws_send_timeout_s: float = 2.0
    ws_push_interval_s: float = 0.2
    ws_heartbeat_s: float = 15.0
    ws_max_message_bytes: int = 4096
    ws_max_client_msgs_per_s: float = 5.0
    enable_docs: bool = False
    key_ids: tuple[str, ...] = field(default=())

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)

    @property
    def misconfigured(self) -> bool:
        """No key and no explicit insecure opt-in => refuse protected routes."""
        return not self.api_keys and not self.allow_insecure

    @classmethod
    def from_env(cls) -> "SecuritySettings":
        raw_keys = [k.strip() for k in os.getenv("COGNEN_API_KEY", "").split(",") if k.strip()]
        short = [k for k in raw_keys if len(k) < MIN_KEY_LENGTH]
        if short:
            raise RuntimeError(
                f"COGNEN_API_KEY contains {len(short)} key(s) shorter than {MIN_KEY_LENGTH} "
                "characters; generate one with `python -c \"import secrets; print(secrets.token_urlsafe(32))\"`"
            )
        origins = tuple(o.strip().rstrip("/") for o in os.getenv("COGNEN_CORS_ORIGINS", "").split(",") if o.strip())
        if any(o == "*" or "*" in o for o in origins):
            raise RuntimeError("COGNEN_CORS_ORIGINS must list explicit origins; '*' is not allowed")
        for o in origins:
            if not re.match(r"^https?://[^/\s]+$", o):
                raise RuntimeError("COGNEN_CORS_ORIGINS entries must look like https://host[:port]")
        return cls(
            api_keys=tuple(k.encode("utf-8") for k in raw_keys),
            key_ids=tuple(key_id(k) for k in raw_keys),
            allow_insecure=_flag("COGNEN_ALLOW_INSECURE"),
            cors_origins=origins,
            max_body_bytes=_int("COGNEN_MAX_BODY_BYTES", 16 * 1024, lo=256),
            start_rate_per_min=_float("COGNEN_START_RATE_PER_MIN", 6.0, lo=0.01),
            start_burst=_int("COGNEN_START_BURST", 3, lo=1),
            auth_fail_per_min=_float("COGNEN_AUTH_FAIL_RATE_PER_MIN", 10.0, lo=0.01),
            auth_fail_burst=_int("COGNEN_AUTH_FAIL_BURST", 10, lo=1),
            ws_max_per_key=_int("COGNEN_WS_MAX_PER_KEY", 8, lo=1),
            ws_max_per_room=_int("COGNEN_WS_MAX_PER_ROOM", 32, lo=1),
            ws_max_total=_int("COGNEN_WS_MAX_TOTAL", 256, lo=1),
            ws_send_timeout_s=_float("COGNEN_WS_SEND_TIMEOUT_S", 2.0, lo=0.05),
            ws_push_interval_s=_float("COGNEN_WS_PUSH_INTERVAL_S", 0.2, lo=0.01),
            ws_heartbeat_s=_float("COGNEN_WS_HEARTBEAT_S", 15.0, lo=0.1),
            ws_max_message_bytes=_int("COGNEN_WS_MAX_MESSAGE_BYTES", 4096, lo=64),
            ws_max_client_msgs_per_s=_float("COGNEN_WS_MAX_CLIENT_MSGS_PER_S", 5.0, lo=0.1),
            enable_docs=_flag("COGNEN_ENABLE_DOCS"),
        )


_settings: Optional[SecuritySettings] = None
_settings_lock = threading.Lock()
_insecure_warned = False


def get_settings() -> SecuritySettings:
    global _settings, _insecure_warned
    if _settings is None:
        with _settings_lock:
            if _settings is None:
                _settings = SecuritySettings.from_env()
                if _settings.allow_insecure and not _settings.api_keys and not _insecure_warned:
                    _insecure_warned = True
                    logger.warning(
                        "!!! COGNEN_ALLOW_INSECURE=1 and no COGNEN_API_KEY: API AUTHENTICATION IS "
                        "DISABLED. Anyone who can reach this server can start/stop rooms and read "
                        "analytics. Use only for local development. !!!"
                    )
                elif _settings.misconfigured:
                    logger.error(
                        "COGNEN_API_KEY is not set: protected endpoints will return 503. "
                        "Set COGNEN_API_KEY (or COGNEN_ALLOW_INSECURE=1 for local dev only)."
                    )
    return _settings


def reset_settings() -> None:
    """Re-read the environment on next access (tests / config reload)."""
    global _settings, _insecure_warned
    with _settings_lock:
        _settings = None
        _insecure_warned = False
    reset_rate_limiters()


def is_loopback_host(host: str) -> bool:
    h = (host or "").strip().strip("[]").lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def check_bind_allowed(host: str, settings: Optional[SecuritySettings] = None) -> None:
    """Refuse a non-loopback bind when no API key is configured (secure default)."""
    s = settings or get_settings()
    if s.api_keys or is_loopback_host(host):
        return
    if s.allow_insecure:
        logger.warning("Binding %s WITHOUT authentication (COGNEN_ALLOW_INSECURE=1)", host)
        return
    raise RuntimeError(
        f"Refusing to bind non-loopback host {host!r} without COGNEN_API_KEY. "
        "Set COGNEN_API_KEY, bind 127.0.0.1, or set COGNEN_ALLOW_INSECURE=1 (local dev only)."
    )


# ── Authentication ───────────────────────────────────────────────────────────

class AuthError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class Principal:
    key_id: str          # "anonymous" when auth is disabled (insecure mode)
    client_ip: str


def match_key(candidate: Optional[str], settings: SecuritySettings) -> Optional[str]:
    """Constant-time check of ``candidate`` against every configured key.
    Returns the matching key's id or None. No early exit across keys."""
    if not candidate:
        return None
    cand = candidate.encode("utf-8", "surrogateescape")
    found: Optional[str] = None
    for k, kid in zip(settings.api_keys, settings.key_ids):
        if secrets.compare_digest(cand, k) and found is None:
            found = kid
    return found


def client_ip(conn: HTTPConnection) -> str:
    return conn.client.host if conn.client else "unknown"


def authenticate(conn: HTTPConnection, allow_query: bool = False) -> Principal:
    """Validate the API key on an HTTP request or WebSocket handshake.

    Raises AuthError(401 missing / 403 wrong / 429 too many failures /
    503 no key configured).
    """
    s = get_settings()
    ip = client_ip(conn)
    if s.misconfigured:
        raise AuthError(503, "server authentication is not configured")
    if not s.auth_enabled:          # explicit COGNEN_ALLOW_INSECURE=1
        return Principal("anonymous", ip)

    fail_bucket = auth_fail_limiter()
    if fail_bucket.peek_empty(ip):
        raise AuthError(429, "too many failed authentication attempts")

    candidate = conn.headers.get(API_KEY_HEADER)
    if not candidate and allow_query:
        candidate = conn.query_params.get("api_key")
    if not candidate:
        fail_bucket.consume(ip)
        raise AuthError(401, "missing API key")
    kid = match_key(candidate, s)
    if kid is None:
        fail_bucket.consume(ip)
        logger.warning("Rejected invalid API key from %s", ip)
        raise AuthError(403, "invalid API key")
    return Principal(kid, ip)


# ── Rate limiting ────────────────────────────────────────────────────────────

class TokenBucketLimiter:
    """In-memory token buckets keyed by string; LRU-bounded so an attacker
    cycling IPs cannot grow memory without bound. Thread-safe."""

    def __init__(self, rate_per_min: float, burst: int, max_keys: int = 10_000) -> None:
        self.rate = rate_per_min / 60.0
        self.burst = float(burst)
        self.max_keys = max_keys
        self._b: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()

    def _refill(self, key: str, now: float) -> float:
        tokens, last = self._b.get(key, (self.burst, now))
        tokens = min(self.burst, tokens + (now - last) * self.rate)
        self._b[key] = (tokens, now)
        self._b.move_to_end(key)
        while len(self._b) > self.max_keys:
            self._b.popitem(last=False)
        return tokens

    def consume(self, key: str) -> tuple[bool, float]:
        """Take one token. Returns (allowed, retry_after_seconds)."""
        now = time.monotonic()
        with self._lock:
            tokens = self._refill(key, now)
            if tokens >= 1.0:
                self._b[key] = (tokens - 1.0, now)
                return True, 0.0
            return False, (1.0 - tokens) / self.rate if self.rate > 0 else 60.0

    def peek_empty(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            if key not in self._b:
                return False
            return self._refill(key, now) < 1.0


_limiters: dict[str, TokenBucketLimiter] = {}
_limiters_lock = threading.Lock()


def _limiter(name: str, rate: float, burst: int) -> TokenBucketLimiter:
    with _limiters_lock:
        lim = _limiters.get(name)
        if lim is None:
            lim = _limiters[name] = TokenBucketLimiter(rate, burst)
        return lim


def start_limiter() -> TokenBucketLimiter:
    s = get_settings()
    return _limiter("start", s.start_rate_per_min, s.start_burst)


def auth_fail_limiter() -> TokenBucketLimiter:
    s = get_settings()
    return _limiter("auth_fail", s.auth_fail_per_min, s.auth_fail_burst)


def reset_rate_limiters() -> None:
    with _limiters_lock:
        _limiters.clear()


def check_start_rate(principal: Principal) -> Optional[float]:
    """Consume a /start token for both the key and the client IP.
    Returns None when allowed, else the Retry-After seconds."""
    lim = start_limiter()
    ok_k, wait_k = lim.consume(f"key:{principal.key_id}")
    ok_ip, wait_ip = lim.consume(f"ip:{principal.client_ip}")
    if ok_k and ok_ip:
        return None
    return max(wait_k, wait_ip, 1.0)


# ── WebSocket connection accounting ──────────────────────────────────────────

class WSRegistry:
    """Counts open WebSockets per key / per room / total (single event loop)."""

    def __init__(self) -> None:
        self.per_key: dict[str, int] = {}
        self.per_room: dict[str, int] = {}
        self.total = 0

    def try_acquire(self, kid: str, room: str, s: SecuritySettings) -> Optional[str]:
        if self.total >= s.ws_max_total:
            return "server websocket limit reached"
        if self.per_key.get(kid, 0) >= s.ws_max_per_key:
            return "too many websocket connections for this key"
        if self.per_room.get(room, 0) >= s.ws_max_per_room:
            return "too many websocket connections for this room"
        self.per_key[kid] = self.per_key.get(kid, 0) + 1
        self.per_room[room] = self.per_room.get(room, 0) + 1
        self.total += 1
        return None

    def release(self, kid: str, room: str) -> None:
        for d, k in ((self.per_key, kid), (self.per_room, room)):
            n = d.get(k, 0) - 1
            if n > 0:
                d[k] = n
            else:
                d.pop(k, None)
        self.total = max(0, self.total - 1)


def origin_allowed(origin: Optional[str], settings: SecuritySettings) -> bool:
    """WebSocket Origin check. Browsers always send Origin; a missing Origin
    means a non-browser client (still subject to API-key auth)."""
    if origin is None:
        return True
    return origin.rstrip("/") in settings.cors_origins


# ── ASGI middleware ──────────────────────────────────────────────────────────

_SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"cross-origin-resource-policy", b"same-origin"),
]
_API_ONLY_HEADERS = [
    (b"cache-control", b"no-store"),
    (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'"),
]


async def _send_json_error(send: Send, status: int, body: bytes, extra: Optional[list] = None) -> None:
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    await send({"type": "http.response.start", "status": status, "headers": headers + (extra or [])})
    await send({"type": "http.response.body", "body": body})


class SecurityHeadersMiddleware:
    """Adds security headers and turns any unhandled exception into a
    sanitised JSON 500 (no traceback / paths in the body)."""

    def __init__(self, app: ASGIApp, static_prefix: str = "/static") -> None:
        self.app = app
        self.static_prefix = static_prefix

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        is_static = scope.get("path", "").startswith(self.static_prefix)
        started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                headers = list(message.get("headers", []))
                present = {k.lower() for k, _ in headers}
                add = _SECURITY_HEADERS + ([] if is_static else _API_ONLY_HEADERS)
                headers += [(k, v) for k, v in add if k not in present]
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            rid = uuid.uuid4().hex[:12]
            logger.exception("Unhandled error request_id=%s path=%s", rid, scope.get("path"))
            if not started:
                await _send_json_error(
                    send_wrapper, 500,
                    b'{"detail":"internal server error","request_id":"' + rid.encode() + b'"}',
                )


class BodySizeLimitMiddleware:
    """Rejects request bodies larger than COGNEN_MAX_BODY_BYTES with 413,
    both by Content-Length and by counting streamed (chunked) bodies."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = get_settings().max_body_bytes
        too_big = b'{"detail":"request body too large"}'
        for k, v in scope.get("headers", []):
            if k == b"content-length":
                try:
                    if int(v) > limit:
                        await _send_json_error(send, 413, too_big)
                        return
                except ValueError:
                    await _send_json_error(send, 400, b'{"detail":"invalid content-length"}')
                    return
        received = 0

        class _TooLarge(Exception):
            pass

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _TooLarge
            return message

        started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, send_wrapper)
        except _TooLarge:
            if not started:
                await _send_json_error(send, 413, too_big)


class DynamicCORSMiddleware:
    """Starlette's CORSMiddleware built from the current settings (rebuilt
    when settings are reset). Allowlist only; never allow_credentials."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._built_for: Optional[SecuritySettings] = None
        self._cors: Optional[ASGIApp] = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        s = get_settings()
        if self._cors is None or self._built_for is not s:
            self._cors = CORSMiddleware(
                self.app,
                allow_origins=list(s.cors_origins),
                allow_methods=["GET", "POST"],
                allow_headers=["X-API-Key", "Content-Type"],
                allow_credentials=False,
                max_age=600,
            )
            self._built_for = s
        await self._cors(scope, receive, send)


# ── Log redaction ────────────────────────────────────────────────────────────

_REDACTIONS = [
    (re.compile(r"(?i)(api_key=)[^&\s\"']+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(x-api-key[\"']?\s*[:=,]\s*[\"']?)[^\s\"',)]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/@\s:]+(?::[^/@\s]*)?@"), r"\1***@"),
]


def redact_text(text: str) -> str:
    for pat, repl in _REDACTIONS:
        text = pat.sub(repl, text)
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        red = redact_text(msg)
        if red != msg:
            record.msg, record.args = red, None
        return True


_REDACTED_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access", "cognen", "cognen.v2",
                     "cognen.input", "cognen.security", "fastapi")


def install_log_redaction() -> None:
    """Attach RedactingFilter to the uvicorn/cognen loggers and their handlers
    (filters on a logger don't apply to records propagated from children, so
    handlers are covered too). Idempotent."""
    targets: list = [logging.getLogger(n) for n in _REDACTED_LOGGERS]
    for lg in list(targets):
        targets.extend(lg.handlers)
    targets.extend(logging.getLogger().handlers)
    for t in targets:
        if not any(isinstance(f, RedactingFilter) for f in t.filters):
            t.addFilter(RedactingFilter())
