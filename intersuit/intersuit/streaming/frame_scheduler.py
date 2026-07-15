"""Event-aware frame scheduling for AS-M4 streaming input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class FrameCandidate:
    """A frame or frame feature available for scheduling."""

    timestamp_sec: float
    payload: Any = None
    frame_id: int | None = None


@dataclass(frozen=True)
class ScheduledFrame:
    """A selected frame plus the priority assigned by the scheduler."""

    candidate: FrameCandidate
    priority: float


class EventAwareFrameScheduler:
    """Select frames under a fixed budget, favoring audio-event neighborhoods."""

    def __init__(
        self,
        base_priority: float = 1.0,
        event_priority_scale: float = 5.0,
        pre_event_sec: float = 1.0,
        post_event_sec: float = 1.0,
    ) -> None:
        if pre_event_sec < 0 or post_event_sec < 0:
            raise ValueError("pre_event_sec and post_event_sec must be non-negative")
        self.base_priority = float(base_priority)
        self.event_priority_scale = float(event_priority_scale)
        self.pre_event_sec = float(pre_event_sec)
        self.post_event_sec = float(post_event_sec)

    def select(
        self,
        candidates: Iterable[FrameCandidate | float],
        event_timestamps: Iterable[float],
        eventness: Iterable[float],
        frame_budget: int,
    ) -> list[ScheduledFrame]:
        """Select at most ``frame_budget`` frames.

        ``candidates`` may be ``FrameCandidate`` objects or raw timestamps.
        """

        if frame_budget < 0:
            raise ValueError("frame_budget must be non-negative")
        if frame_budget == 0:
            return []

        normalized = [_normalize_candidate(item) for item in candidates]
        if not normalized:
            return []

        events = [(float(ts), float(score)) for ts, score in zip(event_timestamps, eventness)]
        scored = [
            ScheduledFrame(candidate=item, priority=self._priority(item.timestamp_sec, events))
            for item in normalized
        ]
        top = sorted(
            scored,
            key=lambda item: (-item.priority, item.candidate.timestamp_sec),
        )[:frame_budget]
        return sorted(top, key=lambda item: item.candidate.timestamp_sec)

    def select_timestamps(
        self,
        candidate_timestamps: Iterable[float],
        event_timestamps: Iterable[float],
        eventness: Iterable[float],
        frame_budget: int,
    ) -> list[float]:
        return [
            item.candidate.timestamp_sec
            for item in self.select(candidate_timestamps, event_timestamps, eventness, frame_budget)
        ]

    def _priority(self, frame_time: float, events: list[tuple[float, float]]) -> float:
        priority = self.base_priority
        for event_time, score in events:
            if score <= 0:
                continue
            start = event_time - self.pre_event_sec
            end = event_time + self.post_event_sec
            if frame_time < start or frame_time > end:
                continue
            radius = self.pre_event_sec if frame_time <= event_time else self.post_event_sec
            radius = max(radius, 1e-6)
            distance = abs(frame_time - event_time)
            decay = max(0.0, 1.0 - distance / radius)
            priority += self.event_priority_scale * score * decay
        return priority


def fixed_stride_select(candidate_timestamps: Iterable[float], frame_budget: int) -> list[float]:
    """Simple baseline selector spread over the full candidate list."""

    timestamps = [float(item) for item in candidate_timestamps]
    if frame_budget <= 0 or not timestamps:
        return []
    if frame_budget >= len(timestamps):
        return timestamps
    if frame_budget == 1:
        return [timestamps[0]]

    step = (len(timestamps) - 1) / (frame_budget - 1)
    indices = [round(i * step) for i in range(frame_budget)]
    deduped: list[int] = []
    for idx in indices:
        idx = max(0, min(len(timestamps) - 1, idx))
        if idx not in deduped:
            deduped.append(idx)
    cursor = 0
    while len(deduped) < frame_budget and cursor < len(timestamps):
        if cursor not in deduped:
            deduped.append(cursor)
        cursor += 1
    return [timestamps[idx] for idx in sorted(deduped[:frame_budget])]


def _normalize_candidate(item: FrameCandidate | float) -> FrameCandidate:
    if isinstance(item, FrameCandidate):
        return item
    return FrameCandidate(timestamp_sec=float(item))

