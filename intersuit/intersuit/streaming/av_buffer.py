"""Timestamped audio/video buffering utilities for AS-M4.

The buffer keeps scene-audio windows and video frames on a shared time axis.
It is intentionally CPU-friendly and model-agnostic so the streaming policy can
be smoke-tested before wiring it into M4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BufferedAudioWindow:
    """A scene-audio item stored in the streaming buffer."""

    samples: Any
    start_sec: float
    end_sec: float
    index: int | None = None


@dataclass(frozen=True)
class BufferedVideoFrame:
    """A video frame or precomputed frame feature with an explicit timestamp."""

    payload: Any
    timestamp_sec: float
    frame_id: int | None = None


@dataclass(frozen=True)
class AVWindow:
    """A time-bounded snapshot returned from ``StreamingAVBuffer``."""

    audio_windows: list[BufferedAudioWindow]
    video_frames: list[BufferedVideoFrame]
    start_sec: float
    end_sec: float
    current_time: float
    lookahead_sec: float


class StreamingAVBuffer:
    """Bounded audio/video history for causal streaming alignment.

    ``lookahead_sec=0`` is strict real-time mode and never returns items whose
    timestamps start after ``current_time``. Positive lookahead allows a bounded
    future buffer, which must be accounted for as latency by callers.
    """

    def __init__(self, history_sec: float = 4.0) -> None:
        if history_sec <= 0:
            raise ValueError("history_sec must be positive")
        self.history_sec = float(history_sec)
        self.audio_windows: list[BufferedAudioWindow] = []
        self.video_frames: list[BufferedVideoFrame] = []

    def push_audio(
        self,
        samples: Any,
        start_sec: float,
        end_sec: float,
        index: int | None = None,
    ) -> None:
        """Insert a scene-audio window."""

        if end_sec < start_sec:
            raise ValueError("audio end_sec must be greater than or equal to start_sec")
        self.audio_windows.append(
            BufferedAudioWindow(
                samples=samples,
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                index=index,
            )
        )
        self.audio_windows.sort(key=lambda item: (item.start_sec, item.end_sec))

    def push_audio_window(self, window: Any) -> None:
        """Insert an object exposing samples/start_sec/end_sec/index attributes."""

        self.push_audio(
            samples=getattr(window, "samples"),
            start_sec=float(getattr(window, "start_sec")),
            end_sec=float(getattr(window, "end_sec")),
            index=getattr(window, "index", None),
        )

    def push_frame(
        self,
        payload: Any,
        timestamp_sec: float,
        frame_id: int | None = None,
    ) -> None:
        """Insert a video frame or frame feature."""

        self.video_frames.append(
            BufferedVideoFrame(
                payload=payload,
                timestamp_sec=float(timestamp_sec),
                frame_id=frame_id,
            )
        )
        self.video_frames.sort(key=lambda item: item.timestamp_sec)

    add_audio_window = push_audio
    add_video_frame = push_frame

    def get_window(
        self,
        current_time: float,
        window_sec: float,
        lookahead_sec: float = 0.0,
    ) -> AVWindow:
        """Return buffered items in ``[current_time-window_sec, current_time+lookahead]``."""

        if window_sec < 0:
            raise ValueError("window_sec must be non-negative")
        if lookahead_sec < 0:
            raise ValueError("lookahead_sec must be non-negative")

        current = float(current_time)
        start = current - float(window_sec)
        end = current + float(lookahead_sec)

        audio = [
            item
            for item in self.audio_windows
            if item.end_sec > start and item.start_sec <= end
        ]
        frames = [
            item
            for item in self.video_frames
            if start <= item.timestamp_sec <= end
        ]
        return AVWindow(
            audio_windows=audio,
            video_frames=frames,
            start_sec=start,
            end_sec=end,
            current_time=current,
            lookahead_sec=float(lookahead_sec),
        )

    def prune(self, current_time: float) -> None:
        """Drop items older than the bounded history window."""

        cutoff = float(current_time) - self.history_sec
        self.audio_windows = [
            item for item in self.audio_windows if item.end_sec >= cutoff
        ]
        self.video_frames = [
            item for item in self.video_frames if item.timestamp_sec >= cutoff
        ]

    def remove_expired(self, current_time: float) -> None:
        """Compatibility alias used by the AS-M4 implementation plan."""

        self.prune(current_time)

    def __len__(self) -> int:
        return len(self.audio_windows) + len(self.video_frames)

