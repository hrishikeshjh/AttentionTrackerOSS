"""
frame_source.py — Unified video input abstraction for CognEn v2.0.

Three concrete implementations share a common FrameSource interface so the
RoomAnalyzer pipeline is fully decoupled from the capture mechanism.

  VideoFileSource  — local video file (MP4, AVI, MKV, …)
  RTSPSource       — RTSP / HTTP network streams with auto-reconnect
  CameraSource     — local capture device (USB webcam, built-in camera)

Usage
-----
# File
with VideoFileSource("/data/lecture_clip.mp4") as src:
    ok, frame = src.read()

# RTSP
src = RTSPSource("rtsp://192.168.1.100:554/ch1", transport="tcp")
src.open()

# Camera
src = CameraSource(device_index=0, width=1280, height=720)
src.open()

API sources
-----------
``validate_api_source()`` is the gate for sources requested over the HTTP
API (server/main.py); the CLI (run_analyzer.py) is trusted and skips it.
It returns a ``ValidatedSource`` that ``make_source(..., guard=...)`` turns
into a *guarded* source (re-validated on every (re)connect / open).

  file    realpath (symlinks resolved) must lie inside one of
          COGNEN_VIDEO_DIRS (os.pathsep-separated; default v2.0/data/videos),
          extension in COGNEN_VIDEO_EXTENSIONS, a regular file if it exists.
          The guarded open re-checks containment, opens the realpath with the
          FFmpeg backend and protocol_whitelist=file + a demuxer whitelist
          (no hls/concat/dash playlists that could fetch other URLs).
  rtsp    see ``validate_stream_url``: scheme rtsp/rtsps (http/https only via
          COGNEN_STREAM_SCHEMES), no userinfo credentials unless
          COGNEN_ALLOW_URL_CREDENTIALS=1, port in COGNEN_STREAM_PORTS if set,
          every A/AAAA record of the host checked against an SSRF policy:
          loopback / link-local (incl. 169.254.169.254 metadata) / unspecified /
          multicast / reserved are always denied unless a COGNEN_STREAM_HOSTS
          CIDR explicitly covers them; private ranges (RFC1918, CGNAT
          100.64/10, IPv6 ULA) need COGNEN_ALLOW_PRIVATE_STREAMS=1 or a
          COGNEN_STREAM_HOSTS entry (hostname or CIDR). IPv4-mapped / NAT64 /
          6to4 / Teredo IPv6 addresses are judged by the embedded IPv4.
  camera  denied unless COGNEN_ALLOW_CAMERA_SOURCE=1.

DNS rebinding: plain rtsp:// URLs are *pinned* -- the connection uses the
validated IP literal, so FFmpeg never re-resolves. rtsps/http/https cannot be
pinned without breaking TLS/Host, so they are re-resolved and re-validated on
every (re)connect; a residual window remains between that check and FFmpeg's
own lookup. FFmpeg's HTTP redirects are disabled (max_redirects=0); RTSP 3xx
redirects cannot be disabled in FFmpeg (residual risk, bounded by the
rtsp/rtp protocol whitelist).
"""

from __future__ import annotations

import abc
import ipaddress
import logging
import os
import socket
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional
from urllib.parse import urlparse, urlunparse

import numpy as np

if TYPE_CHECKING:
    import cv2
else:
    try:  # cv2 is only needed to actually open sources; validation works without it
        import cv2
    except ImportError:  # pragma: no cover - exercised in the light CI tier
        cv2 = None

logger = logging.getLogger("cognen.input")

RTSP_TRANSPORTS = ("tcp", "udp")
# Every scheme the stream path can ever accept; the *enabled* subset comes from
# COGNEN_STREAM_SCHEMES (default: rtsp,rtsps only -- CCTV is RTSP, and FFmpeg's
# HTTP demuxers (HLS/DASH playlists) can be steered into fetching other URLs).
STREAM_SCHEMES = ("rtsp", "rtsps", "http", "https")
DEFAULT_STREAM_SCHEMES = ("rtsp", "rtsps")
DEFAULT_PORTS = {"rtsp": 554, "rtsps": 322, "http": 80, "https": 443}
DEFAULT_VIDEO_DIR = Path(__file__).resolve().parent.parent / "data" / "videos"
DEFAULT_VIDEO_EXTENSIONS = (".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".ts", ".mpg", ".mpeg", ".flv")
MAX_URL_LENGTH = 2048

# FFmpeg demuxer / protocol whitelists for guarded (API-requested) sources.
# They stop a crafted file or stream from making FFmpeg open other inputs
# (HLS/DASH/concat playlists, file: from a stream, nested protocols).
_FILE_FFMPEG_OPTS = (
    "protocol_whitelist;file"
    "|format_whitelist;mov,mp4,matroska,webm,avi,mpegts,mpeg,flv,h264,hevc,mjpeg"
)
_RTSP_FFMPEG_OPTS = "protocol_whitelist;rtsp,rtsps,rtp,srtp,udp,tcp,tls|format_whitelist;rtsp,sdp"
_HTTP_FFMPEG_OPTS = (
    "protocol_whitelist;http,https,tcp,tls"
    "|format_whitelist;mpjpeg,mjpeg,mpegts,flv,matroska,mp4,h264,hevc"
    "|max_redirects;0"
)

_CGNAT = ipaddress.ip_network("100.64.0.0/10")
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_PRIVATE_NETS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    _CGNAT,
    ipaddress.ip_network("fc00::/7"),     # IPv6 unique-local
    ipaddress.ip_network("fec0::/10"),    # deprecated IPv6 site-local
)

# OPENCV_FFMPEG_CAPTURE_OPTIONS is process-global and read by
# cv2.VideoCapture() while it opens; serialise set -> open -> restore so two
# sources with different options cannot pick up each other's options.
_FFMPEG_ENV_LOCK = threading.Lock()


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_list(name: str, sep: str = ",") -> list[str]:
    return [x.strip() for x in os.getenv(name, "").split(sep) if x.strip()]


def redact_url(url: Optional[str]) -> str:
    """Strip userinfo, query and fragment from a URL for logging."""
    if not url:
        return ""
    try:
        p = urlparse(url)
    except ValueError:
        return "<unparseable-url>"
    if not p.scheme or not p.netloc:
        return url if "://" not in url else "<redacted-url>"
    netloc = p.hostname or ""
    if ":" in netloc:
        netloc = f"[{netloc}]"
    try:
        if p.port is not None:
            netloc += f":{p.port}"
    except ValueError:
        pass
    if p.username is not None or p.password is not None:
        netloc = "***@" + netloc
    return urlunparse((p.scheme, netloc, p.path, "", "", ""))


def validate_rtsp_transport(transport: str) -> str:
    t = (transport or "").lower().strip()
    if t not in RTSP_TRANSPORTS:
        raise ValueError(f"rtsp_transport must be one of {RTSP_TRANSPORTS}")
    return t


def allowed_video_dirs() -> list[Path]:
    raw = os.getenv("COGNEN_VIDEO_DIRS", "")
    dirs = [Path(os.path.realpath(os.path.expanduser(d))) for d in raw.split(os.pathsep) if d.strip()]
    return dirs or [DEFAULT_VIDEO_DIR]


def allowed_video_extensions() -> tuple[str, ...]:
    exts = [e.lower() if e.startswith(".") else "." + e.lower() for e in _env_list("COGNEN_VIDEO_EXTENSIONS")]
    return tuple(exts) or DEFAULT_VIDEO_EXTENSIONS


def enabled_stream_schemes() -> tuple[str, ...]:
    raw = [s.lower() for s in _env_list("COGNEN_STREAM_SCHEMES")]
    return tuple(s for s in raw if s in STREAM_SCHEMES) or DEFAULT_STREAM_SCHEMES


# ── IP classification ────────────────────────────────────────────────────────

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


def _embedded_ipv4(ip: IPAddress) -> Optional[ipaddress.IPv4Address]:
    """IPv4 carried inside an IPv6 address (mapped, NAT64, 6to4, Teredo)."""
    if not isinstance(ip, ipaddress.IPv6Address):
        return None
    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if ip in _NAT64:
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    if ip.sixtofour is not None:
        return ip.sixtofour
    if ip.teredo is not None:
        return ip.teredo[1]
    # IPv4-compatible (deprecated ::a.b.c.d)
    if int(ip) >> 32 == 0 and int(ip) > 1:
        return ipaddress.IPv4Address(int(ip))
    return None


def classify_ip(ip: IPAddress) -> str:
    """
    Return one of: public, private, loopback, link_local, unspecified,
    multicast, reserved. Embedded IPv4 (mapped/NAT64/6to4/Teredo) wins.
    """
    inner = _embedded_ipv4(ip)
    if inner is not None:
        return classify_ip(inner)
    if ip.is_unspecified:
        return "unspecified"
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link_local"
    if ip.is_multicast:
        return "multicast"
    if any(ip in n for n in _PRIVATE_NETS if n.version == ip.version):
        return "private"
    if ip.is_global:
        return "public"
    return "reserved"


@dataclass(frozen=True)
class StreamPolicy:
    allow_private: bool = False
    allow_credentials: bool = False
    schemes: tuple[str, ...] = DEFAULT_STREAM_SCHEMES
    ports: frozenset[int] = frozenset()
    host_names: frozenset[str] = frozenset()
    host_nets: tuple = ()

    @classmethod
    def from_env(cls) -> "StreamPolicy":
        names, nets = set(), []
        for entry in _env_list("COGNEN_STREAM_HOSTS"):
            e = entry.lower().strip("[]")
            try:
                nets.append(ipaddress.ip_network(e, strict=False))
            except ValueError:
                names.add(e.rstrip("."))
        ports = set()
        for p in _env_list("COGNEN_STREAM_PORTS"):
            try:
                ports.add(int(p))
            except ValueError:
                logger.warning("Ignoring non-integer COGNEN_STREAM_PORTS entry")
        return cls(
            allow_private=_env_flag("COGNEN_ALLOW_PRIVATE_STREAMS"),
            allow_credentials=_env_flag("COGNEN_ALLOW_URL_CREDENTIALS"),
            schemes=enabled_stream_schemes(),
            ports=frozenset(ports),
            host_names=frozenset(names),
            host_nets=tuple(nets),
        )

    @property
    def has_allowlist(self) -> bool:
        return bool(self.host_names or self.host_nets)

    def ip_allowed(self, ip: IPAddress, host_listed: bool) -> tuple[bool, str]:
        cat = classify_ip(ip)
        inner = _embedded_ipv4(ip)
        candidates = [ip] + ([inner] if inner is not None else [])
        in_cidr = any(c in n for n in self.host_nets for c in candidates if c.version == n.version)
        if in_cidr:
            return True, cat                      # explicit operator CIDR
        if cat in ("loopback", "link_local", "unspecified", "multicast", "reserved"):
            return False, cat                     # never via name / blanket flag
        if self.has_allowlist and not host_listed:
            return False, "not_allowlisted"
        if cat == "private":
            return (self.allow_private or host_listed), cat
        return True, cat                          # public


@dataclass(frozen=True)
class ValidatedSource:
    """Result of validate_api_source(). ``connect_target`` is what to open."""
    source_type: str
    original: Optional[str] = None
    connect_target: Optional[str] = None      # pinned URL / realpath
    ips: tuple[str, ...] = field(default=())
    pinned: bool = False

    @property
    def redacted(self) -> str:
        return redact_url(self.original) if self.source_type == "rtsp" else (
            Path(self.original).name if self.original else self.source_type)


def _resolve(host: str, port: int) -> list[IPAddress]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError, OSError) as exc:
        raise ValueError("stream host could not be resolved") from exc
    out: list[IPAddress] = []
    for info in infos:
        addr = str(info[4][0])   # sockaddr[0] is the host string for AF_INET/AF_INET6
        try:
            ip = ipaddress.ip_address(addr.split("%", 1)[0])
        except ValueError:
            continue
        if ip not in out:
            out.append(ip)
    if not out:
        raise ValueError("stream host could not be resolved")
    return out


def validate_stream_url(url: str, policy: Optional[StreamPolicy] = None) -> ValidatedSource:
    """SSRF policy for API-requested stream URLs (see module docstring)."""
    policy = policy or StreamPolicy.from_env()
    if len(url) > MAX_URL_LENGTH:
        raise ValueError("url is too long")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F or ch in " \\" for ch in url):
        raise ValueError("url contains control characters or spaces")
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("stream url is malformed") from exc
    scheme = parsed.scheme.lower()
    if scheme not in policy.schemes or not parsed.hostname:
        raise ValueError(f"stream url must be {'/'.join(policy.schemes)}://host/... "
                         "(more schemes via COGNEN_STREAM_SCHEMES)")
    if (parsed.username is not None or parsed.password is not None or "@" in parsed.netloc) \
            and not policy.allow_credentials:
        raise ValueError("credentials in the stream url are not allowed "
                         "(set COGNEN_ALLOW_URL_CREDENTIALS=1)")
    port = port if port is not None else DEFAULT_PORTS[scheme]
    if not 0 < port < 65536:
        raise ValueError("stream port is out of range")
    if policy.ports and port not in policy.ports:
        raise ValueError("stream port is not in COGNEN_STREAM_PORTS")

    host = parsed.hostname.lower().rstrip(".")
    try:
        literal: Optional[IPAddress] = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        literal = None
    host_listed = literal is None and host in policy.host_names
    if literal is not None and str(literal) in policy.host_names:
        host_listed = True

    ips = [literal] if literal is not None else _resolve(host, port)
    for ip in ips:
        ok, why = policy.ip_allowed(ip, host_listed)
        if not ok:
            if why == "not_allowlisted":
                raise ValueError("stream host is not in COGNEN_STREAM_HOSTS")
            hint = (" (allow with COGNEN_ALLOW_PRIVATE_STREAMS=1 or COGNEN_STREAM_HOSTS)"
                    if why == "private" else " (only an explicit COGNEN_STREAM_HOSTS CIDR can allow it)")
            raise ValueError(f"stream host resolves to a disallowed {why.replace('_', '-')} address{hint}")

    connect = url
    pinned = False
    if scheme == "rtsp" and literal is None:
        # Pin plain RTSP to the validated address: FFmpeg won't re-resolve, so
        # a DNS-rebinding answer after validation cannot redirect the connect.
        ip0 = ips[0]
        ip_txt = f"[{ip0}]" if ip0.version == 6 else str(ip0)
        userinfo = parsed.netloc.rsplit("@", 1)[0] + "@" if "@" in parsed.netloc else ""
        netloc = f"{userinfo}{ip_txt}:{port}"
        connect = urlunparse(parsed._replace(netloc=netloc))
        pinned = True
    return ValidatedSource("rtsp", url, connect, tuple(str(i) for i in ips), pinned)


def validate_video_path(path_str: str) -> ValidatedSource:
    """realpath containment + extension + regular-file check for file sources."""
    if "://" in path_str or path_str.startswith(("file:", "concat:", "subfile:", "pipe:", "data:")):
        raise ValueError("source_type='file' takes a filesystem path, not a URL")
    if len(path_str) > 4096:
        raise ValueError("file path is too long")
    real = Path(os.path.realpath(os.path.expanduser(path_str)))   # resolves every symlink
    if not any(real == root or root in real.parents for root in allowed_video_dirs()):
        raise ValueError("file path is outside the allowed video directories "
                         "(set COGNEN_VIDEO_DIRS to allow more)")
    if real.suffix.lower() not in allowed_video_extensions():
        raise ValueError("file extension is not an allowed video type (COGNEN_VIDEO_EXTENSIONS)")
    if real.exists() and not real.is_file():
        raise ValueError("file path is not a regular file")
    return ValidatedSource("file", path_str, str(real))


def validate_api_source(source_type: str, url: Optional[str]) -> ValidatedSource:
    """
    Reject API-requested sources outside the allowlist (see module doc).
    Raises ValueError with a client-safe message (never echoes the url/path).
    May block on DNS -- call it from a worker thread in async code.
    """
    st = (source_type or "").lower().strip()
    if st == "camera":
        if not _env_flag("COGNEN_ALLOW_CAMERA_SOURCE"):
            raise ValueError("camera sources are disabled for the API "
                             "(set COGNEN_ALLOW_CAMERA_SOURCE=1)")
        return ValidatedSource("camera")
    if not url:
        raise ValueError(f"url is required for source_type={st!r}")
    if any(ch in url for ch in ("\x00", "\n", "\r")):
        raise ValueError("url contains control characters")

    if st == "file":
        return validate_video_path(url)
    if st in ("rtsp", "http", "stream"):
        return validate_stream_url(url)
    raise ValueError("unknown source_type")


def _open_capture(target: str, ffmpeg_options: Optional[str], *args):
    """cv2.VideoCapture(target, *args) with OPENCV_FFMPEG_CAPTURE_OPTIONS set
    to ``ffmpeg_options`` only for the duration of the open (under a lock)."""
    if ffmpeg_options is None:
        return cv2.VideoCapture(target, *args)
    key = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
    with _FFMPEG_ENV_LOCK:
        previous = os.environ.get(key)
        os.environ[key] = ffmpeg_options
        try:
            return cv2.VideoCapture(target, *args)
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


# ── Abstract base ─────────────────────────────────────────────────────────────

class FrameSource(abc.ABC):
    """
    Protocol for all video input sources.

    Lifecycle:
        source.open()          → bool   (True = ready to read)
        ok, frame = source.read()       (frame is BGR np.ndarray or None)
        source.release()
    """

    @abc.abstractmethod
    def open(self) -> bool:
        """Initialise the capture device / file / stream. Return True on success."""

    @abc.abstractmethod
    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Fetch the next frame.

        Returns:
            (True, frame_bgr)  — success
            (False, None)      — end-of-file, unrecoverable error, or timed-out
                                 after all reconnect attempts
        """

    @abc.abstractmethod
    def release(self) -> None:
        """Free the underlying capture resource."""

    @property
    @abc.abstractmethod
    def fps(self) -> float:
        """Native frames-per-second of the source (may be 0 if unknown)."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable identifier for logging."""

    # Context-manager support
    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.release()


# ── VideoFileSource ───────────────────────────────────────────────────────────

class VideoFileSource(FrameSource):
    """
    Reads frames sequentially from a local video file.

    Args:
        path:    Path to the video file.
        loop:    If True, seek back to frame 0 on EOF and continue reading.
        guarded: API-requested source -- re-validate the realpath against
                 COGNEN_VIDEO_DIRS at open time (closes a symlink-swap TOCTOU),
                 open with the FFmpeg backend and protocol/demuxer whitelists.
    """

    def __init__(self, path: str | Path, loop: bool = False, guarded: bool = False) -> None:
        self._path   = Path(path)
        self._loop   = loop
        self._guarded = guarded
        self._cap:   Optional[cv2.VideoCapture] = None
        self._fps:   float = 0.0

    # ── FrameSource ──────────────────────────────────────────────────────────

    def open(self) -> bool:
        target = self._path
        if self._guarded:
            try:
                target = Path(validate_video_path(str(self._path)).connect_target or "")
            except ValueError as exc:
                logger.error("[VideoFile] Refusing to open %s: %s", self._path.name, exc)
                return False
            if not target.is_file():
                logger.error("[VideoFile] Not a regular file: %s", self._path.name)
                return False
        if not target.exists():
            logger.error("[VideoFile] File not found: %s", self._path)
            return False

        if self._guarded:
            self._cap = _open_capture(str(target), _FILE_FFMPEG_OPTS, cv2.CAP_FFMPEG)
        else:
            self._cap = cv2.VideoCapture(str(target))
        if not self._cap.isOpened():
            logger.error("[VideoFile] Failed to open: %s", self._path)
            return False

        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames  = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("[VideoFile] Opened %s  fps=%.1f  frames=%d", self._path.name, self._fps, n_frames)
        return True

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()
        if not ret:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    return False, None
            else:
                logger.info("[VideoFile] EOF: %s", self._path.name)
                return False, None

        return True, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def name(self) -> str:
        return f"file:{self._path.name}"


# ── RTSPSource ────────────────────────────────────────────────────────────────

class RTSPSource(FrameSource):
    """
    Reads frames from an RTSP or HTTP stream with automatic reconnection.

    Reconnect strategy: exponential backoff capped at ``max_reconnect_delay_s``.
    Set ``max_reconnect_attempts=0`` for unlimited retries (typical for 24 h CCTV).

    Args:
        url:                    RTSP or HTTP stream URL.
        transport:              ``"tcp"`` (reliable) or ``"udp"`` (lower latency).
        connection_timeout_s:   Seconds to wait for initial connection.
        max_reconnect_attempts: Hard cap on retry count (0 = infinite).
        max_reconnect_delay_s:  Upper bound on back-off interval.
        buffer_size:            cv2 frame buffer size (1 = minimal latency).
        revalidate:             API-requested sources: called before every
                                (re)connect; returns the URL to open (pinned
                                IP for rtsp://) or raises ValueError, which
                                aborts that attempt (DNS-rebinding mitigation).
        ffmpeg_options:         Extra OPENCV_FFMPEG_CAPTURE_OPTIONS entries
                                (protocol/demuxer whitelists for guarded sources).

    URLs are logged via ``redact_url`` (no credentials / query strings).

    ``request_stop()`` (thread-safe) aborts a pending reconnect back-off, so
    an infinite-retry source can be shut down from another thread.
    """

    def __init__(
        self,
        url: str,
        transport: str = "tcp",
        connection_timeout_s: float = 10.0,
        max_reconnect_attempts: int = 0,
        max_reconnect_delay_s:  float = 30.0,
        buffer_size: int = 1,
        revalidate: Optional[Callable[[], str]] = None,
        ffmpeg_options: Optional[str] = None,
    ) -> None:
        self._url               = url
        self._safe_url          = redact_url(url)
        self._revalidate        = revalidate
        self._extra_opts        = ffmpeg_options
        self._transport         = validate_rtsp_transport(transport)
        self._stop              = threading.Event()
        self._conn_timeout_ms   = int(connection_timeout_s * 1000)
        self._max_attempts      = max_reconnect_attempts
        self._max_delay         = max_reconnect_delay_s
        self._buffer_size       = buffer_size
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 0.0
        self._reconnect_count   = 0

    # ── FrameSource ──────────────────────────────────────────────────────────

    def open(self) -> bool:
        return self._connect()

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Read one frame.  On failure, attempt reconnect with back-off.
        Returns (False, None) only when max_reconnect_attempts is exceeded.
        """
        if self._cap is None:
            return False, None

        ret, frame = self._cap.read()
        if ret:
            self._reconnect_count = 0  # reset on success
            return True, frame

        # Read failed — attempt reconnect
        while not self._stop.is_set():
            self._reconnect_count += 1
            if self._max_attempts and self._reconnect_count > self._max_attempts:
                logger.error(
                    "[RTSP] Max reconnect attempts (%d) exceeded: %s",
                    self._max_attempts, self._safe_url,
                )
                return False, None

            delay = min(2 ** (self._reconnect_count - 1), self._max_delay)
            logger.warning(
                "[RTSP] Stream interrupted (attempt %d). Reconnecting in %.1f s…",
                self._reconnect_count, delay,
            )
            if self._stop.wait(delay):
                break

            if self._connect():
                ret, frame = self._cap.read()
                if ret:
                    return True, frame
        logger.info("[RTSP] Stop requested; giving up reconnect: %s", self._safe_url)
        return False, None

    def request_stop(self) -> None:
        """Abort reconnect back-off (safe to call from any thread)."""
        self._stop.set()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def name(self) -> str:
        return f"rtsp:{self._safe_url}"

    # ── Internal ─────────────────────────────────────────────────────────────

    def _connect(self) -> bool:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        target = self._url
        if self._revalidate is not None:
            try:
                target = self._revalidate()
            except ValueError as exc:
                logger.warning("[RTSP] Re-validation failed, not connecting: %s (%s)", self._safe_url, exc)
                return False

        # RTSP transport can only be passed through the process-wide FFmpeg
        # options env var; hold a lock while it is set and restore it after
        # the open. Timeouts use the per-capture OpenCV params instead of
        # FFmpeg's `stimeout` (removed in FFmpeg 5+; the bundled FFmpeg
        # ignores it).
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._conn_timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._conn_timeout_ms,
        ]
        opts = f"rtsp_transport;{self._transport}"
        if self._extra_opts:
            opts += "|" + self._extra_opts
        self._cap = _open_capture(target, opts, cv2.CAP_FFMPEG, params)

        # Reduce internal buffer to minimise latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)

        if not self._cap.isOpened():
            logger.warning("[RTSP] Could not connect: %s", self._safe_url)
            return False

        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        logger.info("[RTSP] Connected: %s  fps=%.1f", self._safe_url, self._fps)
        return True


# ── CameraSource ──────────────────────────────────────────────────────────────

class CameraSource(FrameSource):
    """
    Reads frames from a local camera device via OpenCV VideoCapture.

    Sets the internal capture buffer to ``buffer_size`` (default 1) so
    ``read()`` always returns the most-recent frame rather than a queued one.

    Args:
        device_index:  OpenCV device index (0 = default system camera).
        width:         Requested capture width  (None = driver default).
        height:        Requested capture height (None = driver default).
        target_fps:    Requested capture FPS    (None = driver default).
        backend:       cv2.VideoCaptureAPIs backend constant
                       (None = auto-select by OS).
        buffer_size:   Internal frame buffer (1 = lowest latency).
    """

    def __init__(
        self,
        device_index: int = 0,
        width:  Optional[int] = None,
        height: Optional[int] = None,
        target_fps: Optional[float] = None,
        backend: Optional[int] = None,
        buffer_size: int = 1,
    ) -> None:
        self._device    = device_index
        self._width     = width
        self._height    = height
        self._target_fps = target_fps
        self._backend   = backend
        self._buf_size  = buffer_size
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 0.0

    # ── FrameSource ──────────────────────────────────────────────────────────

    def open(self) -> bool:
        args = [self._device]
        if self._backend is not None:
            args.append(self._backend)
        self._cap = cv2.VideoCapture(*args)

        if not self._cap.isOpened():
            logger.error("[Camera] Failed to open device %d", self._device)
            return False

        # Minimise latency — only buffer the most recent frame
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buf_size)

        if self._width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  float(self._width))
        if self._height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))
        if self._target_fps is not None:
            self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)

        # Read back actual values after driver negotiation
        actual_w   = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps  = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

        logger.info(
            "[Camera] Opened device %d  %dx%d @ %.1f fps",
            self._device, actual_w, actual_h, self._fps,
        )
        return True

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._cap is None or not self._cap.isOpened():
            return False, None
        return self._cap.read()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def name(self) -> str:
        return f"camera:{self._device}"


# ── Factory ───────────────────────────────────────────────────────────────────

def make_source(
    source_type: str,
    url: Optional[str] = None,
    device: int = 0,
    loop_file: bool = False,
    rtsp_transport: str = "tcp",
    rtsp_max_reconnect: int = 0,
    camera_width:  Optional[int] = None,
    camera_height: Optional[int] = None,
    camera_fps:    Optional[float] = None,
    guard: Optional[ValidatedSource] = None,
) -> FrameSource:
    """
    Convenience factory — create a FrameSource from a type string.

    Args:
        source_type: ``"file"``, ``"rtsp"``, or ``"camera"``
        url:         File path or RTSP/HTTP URL (required for file/rtsp)
        device:      Camera device index (for camera only)
        loop_file:   Loop video files (file only)
        rtsp_transport:     ``"tcp"`` or ``"udp"`` (anything else: ValueError)
        rtsp_max_reconnect: Reconnect attempts (0 = infinite)
        camera_width/height/fps: Optional resolution hints (camera only)
        guard:       ``validate_api_source()`` result for an API-requested
                     source. Builds a guarded source: file opens the
                     validated realpath (re-checked at open); rtsp opens the
                     pinned URL and re-validates on every reconnect; FFmpeg
                     protocol/demuxer whitelists and no HTTP redirects.

    Raises:
        ValueError: on unknown source_type or missing url for file/rtsp.
    """
    st = source_type.lower().strip()

    if st == "file":
        if not url:
            raise ValueError("url is required for source_type='file'")
        if guard is not None:
            return VideoFileSource(guard.connect_target or url, loop=loop_file, guarded=True)
        return VideoFileSource(url, loop=loop_file)

    if st in ("rtsp", "http", "stream"):
        if not url:
            raise ValueError("url is required for source_type='rtsp'")
        if guard is not None:
            scheme = urlparse(url).scheme.lower()
            return RTSPSource(
                url,
                transport=rtsp_transport,
                max_reconnect_attempts=rtsp_max_reconnect,
                revalidate=lambda: validate_stream_url(url).connect_target or url,
                ffmpeg_options=_RTSP_FFMPEG_OPTS if scheme.startswith("rtsp") else _HTTP_FFMPEG_OPTS,
            )
        return RTSPSource(
            url,
            transport=rtsp_transport,
            max_reconnect_attempts=rtsp_max_reconnect,
        )

    if st == "camera":
        return CameraSource(
            device_index=device,
            width=camera_width,
            height=camera_height,
            target_fps=camera_fps,
        )

    raise ValueError(
        f"Unknown source_type '{source_type}'. "
        "Expected 'file', 'rtsp', or 'camera'."
    )
