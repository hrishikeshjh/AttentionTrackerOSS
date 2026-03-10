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
"""

from __future__ import annotations

import abc
import logging
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("cognen.input")


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
    """

    def __init__(self, path: str | Path, loop: bool = False) -> None:
        self._path   = Path(path)
        self._loop   = loop
        self._cap:   Optional[cv2.VideoCapture] = None
        self._fps:   float = 0.0

    # ── FrameSource ──────────────────────────────────────────────────────────

    def open(self) -> bool:
        if not self._path.exists():
            logger.error("[VideoFile] File not found: %s", self._path)
            return False

        self._cap = cv2.VideoCapture(str(self._path))
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
    """

    def __init__(
        self,
        url: str,
        transport: str = "tcp",
        connection_timeout_s: float = 10.0,
        max_reconnect_attempts: int = 0,
        max_reconnect_delay_s:  float = 30.0,
        buffer_size: int = 1,
    ) -> None:
        self._url               = url
        self._transport         = transport
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
        while True:
            self._reconnect_count += 1
            if self._max_attempts and self._reconnect_count > self._max_attempts:
                logger.error(
                    "[RTSP] Max reconnect attempts (%d) exceeded: %s",
                    self._max_attempts, self._url,
                )
                return False, None

            delay = min(2 ** (self._reconnect_count - 1), self._max_delay)
            logger.warning(
                "[RTSP] Stream interrupted (attempt %d). Reconnecting in %.1f s…",
                self._reconnect_count, delay,
            )
            time.sleep(delay)

            if self._connect():
                ret, frame = self._cap.read()
                if ret:
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
        return f"rtsp:{self._url}"

    # ── Internal ─────────────────────────────────────────────────────────────

    def _connect(self) -> bool:
        if self._cap is not None:
            self._cap.release()

        # Set RTSP transport via FFmpeg options (must be done before open)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{self._transport}|"
            f"stimeout;{self._conn_timeout_ms * 1000}"  # stimeout is in µs
        )

        self._cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)

        # Reduce internal buffer to minimise latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)

        if not self._cap.isOpened():
            logger.warning("[RTSP] Could not connect: %s", self._url)
            return False

        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        logger.info("[RTSP] Connected: %s  fps=%.1f", self._url, self._fps)
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
) -> FrameSource:
    """
    Convenience factory — create a FrameSource from a type string.

    Args:
        source_type: ``"file"``, ``"rtsp"``, or ``"camera"``
        url:         File path or RTSP/HTTP URL (required for file/rtsp)
        device:      Camera device index (for camera only)
        loop_file:   Loop video files (file only)
        rtsp_transport:     ``"tcp"`` or ``"udp"``
        rtsp_max_reconnect: Reconnect attempts (0 = infinite)
        camera_width/height/fps: Optional resolution hints (camera only)

    Raises:
        ValueError: on unknown source_type or missing url for file/rtsp.
    """
    st = source_type.lower().strip()

    if st == "file":
        if not url:
            raise ValueError("url is required for source_type='file'")
        return VideoFileSource(url, loop=loop_file)

    if st in ("rtsp", "http", "stream"):
        if not url:
            raise ValueError("url is required for source_type='rtsp'")
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
