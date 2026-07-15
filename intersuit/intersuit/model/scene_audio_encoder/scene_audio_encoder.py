"""Lightweight scene-audio encoder backends for AS-M4.

These classes define the stable interface used by downstream AS-M4 modules.
Heavy pretrained encoders such as BEATs, PANNs, or CLAP can later implement the
same contract without changing event detection, alignment, or fusion code.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F


class SceneAudioEncoderOutput(NamedTuple):
    """Scene-audio features plus masks and optional timestamps."""

    features: torch.Tensor
    mask: torch.Tensor
    timestamps: torch.Tensor | None = None


class DummySceneAudioEncoder(nn.Module):
    """Deterministic waveform-statistics encoder used for smoke tests.

    Input shape is ``[B, T, S]`` where ``T`` is the number of audio windows and
    ``S`` is samples per window. The encoder has no trainable parameters.
    """

    def __init__(self, hidden_size: int = 768) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = int(hidden_size)

    def forward(
        self,
        audio_windows: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
        timestamps: torch.Tensor | None = None,
    ) -> SceneAudioEncoderOutput:
        windows = _ensure_batched_windows(audio_windows)
        batch, steps, _ = windows.shape

        if sample_mask is None:
            mask = torch.ones(batch, steps, dtype=torch.bool, device=windows.device)
        else:
            mask = sample_mask.to(device=windows.device, dtype=torch.bool)
            if mask.shape != (batch, steps):
                raise ValueError(f"sample_mask shape {tuple(mask.shape)} does not match {(batch, steps)}")

        stats = _window_statistics(windows)
        features = _expand_to_hidden(stats, self.hidden_size)
        features = features.masked_fill(~mask.unsqueeze(-1), 0.0)
        return SceneAudioEncoderOutput(features=features, mask=mask, timestamps=timestamps)


class PrecomputedSceneAudioEncoder(nn.Module):
    """Adapter for precomputed scene-audio features.

    If ``input_dim`` differs from ``hidden_size``, a frozen deterministic linear
    projection is applied. This keeps first-pass harnesses dependency-free while
    exposing the same shape as future pretrained encoders.
    """

    def __init__(self, hidden_size: int = 768, input_dim: int | None = None) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = int(hidden_size)
        self.input_dim = int(input_dim) if input_dim is not None else None
        self.proj: nn.Linear | None = None
        if self.input_dim is not None and self.input_dim != self.hidden_size:
            self.proj = nn.Linear(self.input_dim, self.hidden_size, bias=False)
            nn.init.zeros_(self.proj.weight)
            diag = min(self.input_dim, self.hidden_size)
            with torch.no_grad():
                self.proj.weight[:diag, :diag] = torch.eye(diag)
            for param in self.proj.parameters():
                param.requires_grad = False

    def forward(
        self,
        features: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
        timestamps: torch.Tensor | None = None,
    ) -> SceneAudioEncoderOutput:
        encoded = _ensure_batched_features(features).to(dtype=torch.float32)
        batch, steps, dim = encoded.shape

        if self.proj is not None:
            if dim != self.input_dim:
                raise ValueError(f"Expected precomputed dim {self.input_dim}, got {dim}")
            encoded = self.proj(encoded)
        elif dim != self.hidden_size:
            encoded = _expand_to_hidden(encoded, self.hidden_size)

        if sample_mask is None:
            mask = torch.ones(batch, steps, dtype=torch.bool, device=encoded.device)
        else:
            mask = sample_mask.to(device=encoded.device, dtype=torch.bool)
            if mask.shape != (batch, steps):
                raise ValueError(f"sample_mask shape {tuple(mask.shape)} does not match {(batch, steps)}")

        encoded = encoded.masked_fill(~mask.unsqueeze(-1), 0.0)
        return SceneAudioEncoderOutput(features=encoded, mask=mask, timestamps=timestamps)


def _ensure_batched_windows(audio_windows: torch.Tensor) -> torch.Tensor:
    if audio_windows.ndim == 2:
        audio_windows = audio_windows.unsqueeze(0)
    if audio_windows.ndim != 3:
        raise ValueError(f"Expected audio windows shaped [B,T,S] or [T,S], got {tuple(audio_windows.shape)}")
    return audio_windows.to(dtype=torch.float32)


def _ensure_batched_features(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 2:
        features = features.unsqueeze(0)
    if features.ndim != 3:
        raise ValueError(f"Expected features shaped [B,T,D] or [T,D], got {tuple(features.shape)}")
    return features


def _window_statistics(windows: torch.Tensor) -> torch.Tensor:
    mean = windows.mean(dim=-1)
    std = windows.std(dim=-1, unbiased=False)
    rms = torch.sqrt(torch.clamp((windows * windows).mean(dim=-1), min=0.0))
    max_abs = windows.abs().amax(dim=-1)
    min_value = windows.amin(dim=-1)
    max_value = windows.amax(dim=-1)
    zero_cross = ((windows[..., 1:] * windows[..., :-1]) < 0).float().mean(dim=-1)
    return torch.stack([mean, std, rms, max_abs, min_value, max_value, zero_cross], dim=-1)


def _expand_to_hidden(features: torch.Tensor, hidden_size: int) -> torch.Tensor:
    if features.shape[-1] == hidden_size:
        return features
    if features.shape[-1] > hidden_size:
        return features[..., :hidden_size]
    repeat = (hidden_size + features.shape[-1] - 1) // features.shape[-1]
    expanded = features.repeat(1, 1, repeat)
    return expanded[..., :hidden_size]
