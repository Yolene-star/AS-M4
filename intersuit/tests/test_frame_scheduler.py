"""CPU harness for AS-M4 event-aware frame scheduling."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "intersuit" / "streaming" / "frame_scheduler.py"
SPEC = importlib.util.spec_from_file_location("as_m4_frame_scheduler", MODULE_PATH)
frame_scheduler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = frame_scheduler
SPEC.loader.exec_module(frame_scheduler)

EventAwareFrameScheduler = frame_scheduler.EventAwareFrameScheduler
fixed_stride_select = frame_scheduler.fixed_stride_select


def _event_recall(selected, event_time, radius):
    return any(abs(ts - event_time) <= radius for ts in selected)


def test_event_scheduler_improves_event_frame_recall_under_same_budget():
    candidates = [float(i) for i in range(10)]
    event_time = 5.0
    budget = 3
    scheduler = EventAwareFrameScheduler(pre_event_sec=1.0, post_event_sec=1.0)

    fixed = fixed_stride_select(candidates, budget)
    event_aware = scheduler.select_timestamps(candidates, [event_time], [1.0], budget)

    assert len(fixed) == budget
    assert len(event_aware) == budget
    assert not _event_recall(fixed, event_time, radius=0.1)
    assert _event_recall(event_aware, event_time, radius=0.1)


def test_scheduler_does_not_exceed_frame_budget():
    candidates = [float(i) * 0.5 for i in range(20)]
    scheduler = EventAwareFrameScheduler()

    selected = scheduler.select_timestamps(candidates, [2.0, 5.0], [1.0, 0.8], frame_budget=5)

    assert len(selected) <= 5
    assert selected == sorted(selected)


def test_silent_events_fall_back_to_base_priority_order():
    candidates = [0.0, 1.0, 2.0, 3.0]
    scheduler = EventAwareFrameScheduler()

    selected = scheduler.select_timestamps(candidates, [2.0], [0.0], frame_budget=2)

    assert selected == [0.0, 1.0]


def test_zero_budget_returns_empty_selection():
    scheduler = EventAwareFrameScheduler()

    assert scheduler.select_timestamps([0.0, 1.0], [0.5], [1.0], frame_budget=0) == []


if __name__ == "__main__":
    test_event_scheduler_improves_event_frame_recall_under_same_budget()
    test_scheduler_does_not_exceed_frame_budget()
    test_silent_events_fall_back_to_base_priority_order()
    test_zero_budget_returns_empty_selection()
    print("frame_scheduler harness passed")
