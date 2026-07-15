from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "intersuit" / "streaming" / "audio_stream.py"
SPEC = importlib.util.spec_from_file_location("audio_stream", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audio_stream = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audio_stream
SPEC.loader.exec_module(audio_stream)

split_audio_windows = audio_stream.split_audio_windows
stack_audio_windows = audio_stream.stack_audio_windows


def test_split_three_seconds_into_overlapping_windows():
    sample_rate = 16000
    waveform = torch.arange(3 * sample_rate, dtype=torch.float32)

    windows = split_audio_windows(waveform, sample_rate, window_sec=1.0, hop_sec=0.5)
    samples, timestamps = stack_audio_windows(windows)

    assert len(windows) == 5
    assert samples.shape == (5, sample_rate)
    assert torch.allclose(
        timestamps,
        torch.tensor(
            [
                [0.0, 1.0],
                [0.5, 1.5],
                [1.0, 2.0],
                [1.5, 2.5],
                [2.0, 3.0],
            ]
        ),
    )
    assert all(windows[idx].start_sec < windows[idx + 1].start_sec for idx in range(len(windows) - 1))


def test_short_audio_is_padded_to_one_window():
    sample_rate = 16000
    waveform = torch.ones(sample_rate // 4)

    windows = split_audio_windows(waveform, sample_rate, window_sec=1.0, hop_sec=0.5)

    assert len(windows) == 1
    assert windows[0].samples.shape == (sample_rate,)
    assert torch.allclose(windows[0].samples[: sample_rate // 4], torch.ones(sample_rate // 4))
    assert torch.count_nonzero(windows[0].samples[sample_rate // 4 :]) == 0


def test_empty_audio_returns_no_windows():
    windows = split_audio_windows(torch.empty(0), sample_rate=16000)
    samples, timestamps = stack_audio_windows(windows)

    assert windows == []
    assert samples.shape == (0, 0)
    assert timestamps.shape == (0, 2)


def test_stereo_audio_is_mixed_to_mono_before_windowing():
    sample_rate = 10
    left = torch.ones(20)
    right = torch.zeros(20)
    waveform = torch.stack([left, right], dim=0)

    windows = split_audio_windows(waveform, sample_rate, window_sec=1.0, hop_sec=1.0)

    assert len(windows) == 2
    assert torch.allclose(windows[0].samples, torch.full((10,), 0.5))


if __name__ == "__main__":
    test_split_three_seconds_into_overlapping_windows()
    test_short_audio_is_padded_to_one_window()
    test_empty_audio_returns_no_windows()
    test_stereo_audio_is_mixed_to_mono_before_windowing()
    print("audio_stream harness passed")
