"""Scene audio loading and windowing utilities for AS-M4.

This module handles video scene audio only. It must not be used for the
existing M4 ``speech`` / query-speech path, which represents user questions.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio


@dataclass(frozen=True)
class AudioWindow:
    """A scene-audio window with an explicit timestamp."""

    samples: torch.Tensor
    start_sec: float
    end_sec: float
    index: int


def _as_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform with 1 or 2 dims, got shape {tuple(waveform.shape)}")
    return waveform.mean(dim=0)


def load_scene_audio(
    media_path: str | Path,
    sample_rate: int = 16000,
    mono: bool = True,
) -> tuple[torch.Tensor, int]:
    """Load scene audio from an audio/video file.

    Audio files are loaded directly through torchaudio. Other media files are
    decoded by ffmpeg into a temporary wav first. Returned tensors are shaped
    ``[num_samples]`` when ``mono=True`` and ``[channels, num_samples]`` when
    ``mono=False``.
    """

    path = Path(media_path)
    if not path.exists():
        raise FileNotFoundError(f"Scene audio source does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}:
        try:
            waveform, source_rate = torchaudio.load(str(path))
        except (RuntimeError, OSError):
            # Some CPU environments ship torchaudio without an IO backend.
            # Keep scene-audio loading functional through the same ffmpeg path
            # used for video containers.
            channels = 1 if mono else 2
            command = [
                "ffmpeg", "-v", "error", "-i", str(path), "-vn",
                "-ac", str(channels), "-ar", str(sample_rate),
                "-acodec", "pcm_f32le", "-f", "f32le", "pipe:1",
            ]
            decoded = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if decoded.returncode != 0 or not decoded.stdout:
                raise RuntimeError(
                    f"ffmpeg scene-audio decode failed for {path}: "
                    f"{decoded.stderr.decode(errors='replace').strip()}"
                )
            waveform = torch.frombuffer(bytearray(decoded.stdout), dtype=torch.float32).clone()
            if channels > 1:
                waveform = waveform.reshape(-1, channels).transpose(0, 1).contiguous()
            source_rate = sample_rate
    else:
        channels = 1 if mono else 2
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-acodec",
            "pcm_f32le",
            "-f",
            "f32le",
            "pipe:1",
        ]
        decoded = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if decoded.returncode != 0:
            raise RuntimeError(f"ffmpeg scene-audio decode failed for {path}: {decoded.stderr.decode(errors='replace').strip()}")
        if not decoded.stdout:
            raise ValueError(f"ffmpeg decoded empty scene audio from {path}")
        waveform = torch.frombuffer(bytearray(decoded.stdout), dtype=torch.float32).clone()
        if channels > 1:
            if waveform.numel() % channels:
                raise ValueError(f"Decoded PCM sample count is not divisible by channels for {path}")
            waveform = waveform.reshape(-1, channels).transpose(0, 1).contiguous()
        source_rate = sample_rate

    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)

    if mono:
        waveform = _as_mono(waveform)

    return waveform.contiguous(), sample_rate


def split_audio_windows(
    waveform: torch.Tensor,
    sample_rate: int,
    window_sec: float = 1.0,
    hop_sec: float = 0.5,
    pad_short: bool = True,
) -> list[AudioWindow]:
    """Split a waveform into timestamped scene-audio windows.

    Empty inputs return an empty list. Short non-empty inputs are padded to one
    full window by default so downstream modules can smoke-test safely.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if window_sec <= 0 or hop_sec <= 0:
        raise ValueError("window_sec and hop_sec must be positive")

    mono = _as_mono(waveform).to(dtype=torch.float32)
    num_samples = int(mono.numel())
    if num_samples == 0:
        return []

    window_samples = max(1, int(round(window_sec * sample_rate)))
    hop_samples = max(1, int(round(hop_sec * sample_rate)))

    if num_samples < window_samples:
        if not pad_short:
            return []
        mono = F.pad(mono, (0, window_samples - num_samples))
        num_samples = window_samples

    num_windows = 1 + math.ceil((num_samples - window_samples) / hop_samples)
    total_needed = (num_windows - 1) * hop_samples + window_samples
    if total_needed > num_samples:
        mono = F.pad(mono, (0, total_needed - num_samples))

    windows: list[AudioWindow] = []
    for idx in range(num_windows):
        start_sample = idx * hop_samples
        end_sample = start_sample + window_samples
        windows.append(
            AudioWindow(
                samples=mono[start_sample:end_sample].contiguous(),
                start_sec=start_sample / sample_rate,
                end_sec=end_sample / sample_rate,
                index=idx,
            )
        )
    return windows


def stack_audio_windows(windows: list[AudioWindow]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack windows into tensors for tests and simple harnesses."""

    if not windows:
        return torch.empty(0, 0), torch.empty(0, 2)
    samples = torch.stack([window.samples for window in windows], dim=0)
    timestamps = torch.tensor(
        [[window.start_sec, window.end_sec] for window in windows],
        dtype=torch.float32,
    )
    return samples, timestamps
