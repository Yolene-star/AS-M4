"""CPU harness for the AS-M4 streaming AV buffer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "intersuit" / "streaming" / "av_buffer.py"
SPEC = importlib.util.spec_from_file_location("as_m4_av_buffer", MODULE_PATH)
av_buffer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = av_buffer
SPEC.loader.exec_module(av_buffer)

StreamingAVBuffer = av_buffer.StreamingAVBuffer


def test_strict_realtime_does_not_return_future_items():
    buffer = StreamingAVBuffer(history_sec=4.0)
    buffer.push_audio("past-audio", 9.0, 10.0, index=0)
    buffer.push_audio("future-audio", 10.25, 11.25, index=1)
    buffer.push_frame("past-frame", 9.5, frame_id=0)
    buffer.push_frame("future-frame", 10.25, frame_id=1)

    window = buffer.get_window(current_time=10.0, window_sec=2.0, lookahead_sec=0.0)

    assert [item.samples for item in window.audio_windows] == ["past-audio"]
    assert [item.payload for item in window.video_frames] == ["past-frame"]
    assert window.end_sec == 10.0


def test_positive_lookahead_is_bounded():
    buffer = StreamingAVBuffer(history_sec=4.0)
    buffer.push_audio("in-lookahead", 10.25, 10.75, index=0)
    buffer.push_audio("outside-lookahead", 10.75, 11.25, index=1)
    buffer.push_frame("in-lookahead-frame", 10.5, frame_id=0)
    buffer.push_frame("outside-lookahead-frame", 10.75, frame_id=1)

    window = buffer.get_window(current_time=10.0, window_sec=1.0, lookahead_sec=0.5)

    assert [item.samples for item in window.audio_windows] == ["in-lookahead"]
    assert [item.payload for item in window.video_frames] == ["in-lookahead-frame"]
    assert window.end_sec == 10.5


def test_prune_bounds_history_and_keeps_recent_items():
    buffer = StreamingAVBuffer(history_sec=2.0)
    buffer.push_audio("old-audio", 0.0, 1.0, index=0)
    buffer.push_audio("recent-audio", 3.0, 4.0, index=1)
    buffer.push_frame("old-frame", 0.5, frame_id=0)
    buffer.push_frame("recent-frame", 3.5, frame_id=1)

    buffer.prune(current_time=5.0)

    assert [item.samples for item in buffer.audio_windows] == ["recent-audio"]
    assert [item.payload for item in buffer.video_frames] == ["recent-frame"]
    assert len(buffer) == 2


def test_unordered_inputs_are_returned_monotonically():
    buffer = StreamingAVBuffer(history_sec=4.0)
    buffer.push_audio("a2", 2.0, 3.0, index=2)
    buffer.push_audio("a1", 1.0, 2.0, index=1)
    buffer.push_frame("f2", 2.0, frame_id=2)
    buffer.push_frame("f1", 1.0, frame_id=1)

    window = buffer.get_window(current_time=3.0, window_sec=3.0)

    assert [item.samples for item in window.audio_windows] == ["a1", "a2"]
    assert [item.payload for item in window.video_frames] == ["f1", "f2"]


def test_missing_modalities_return_empty_lists():
    buffer = StreamingAVBuffer(history_sec=4.0)
    window = buffer.get_window(current_time=2.0, window_sec=1.0)

    assert window.audio_windows == []
    assert window.video_frames == []


if __name__ == "__main__":
    test_strict_realtime_does_not_return_future_items()
    test_positive_lookahead_is_bounded()
    test_prune_bounds_history_and_keeps_recent_items()
    test_unordered_inputs_are_returned_monotonically()
    test_missing_modalities_return_empty_lists()
    print("av_buffer harness passed")

