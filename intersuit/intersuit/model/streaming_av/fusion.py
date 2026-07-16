"""Gated residual audio-video fusion for AS-M4."""

from __future__ import annotations

import math

import torch
from torch import nn


class GatedAVFusion(nn.Module):
    """Fuse aligned scene-audio features into video tokens without adding tokens."""

    def __init__(self, hidden_size: int, fusion_init: str = "zero") -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if fusion_init not in {"zero", "identity"}:
            raise ValueError(f"fusion_init must be 'zero' or 'identity', got {fusion_init!r}")
        self.hidden_size = int(hidden_size)
        self.fusion_init = fusion_init
        self.audio_projector = nn.Linear(hidden_size, hidden_size, bias=False)
        if fusion_init == "zero":
            nn.init.zeros_(self.audio_projector.weight)
        else:
            nn.init.eye_(self.audio_projector.weight)

    def forward(
        self,
        video_tokens: torch.Tensor,
        aligned_audio: torch.Tensor,
        gate: torch.Tensor,
        residual_scale: float = 1.0,
    ) -> torch.Tensor:
        """Return video-shaped fused tokens.

        Args:
            video_tokens: ``[B,T,N,H]`` visual tokens.
            aligned_audio: ``[B,T,H]`` scene-audio features aligned to frames.
            gate: ``[B,T]`` or broadcast-compatible audio weight in ``[0,1]``.
        """

        if video_tokens.ndim != 4:
            raise ValueError(f"Expected video_tokens shaped [B,T,N,H], got {tuple(video_tokens.shape)}")
        if aligned_audio.ndim != 3:
            raise ValueError(f"Expected aligned_audio shaped [B,T,H], got {tuple(aligned_audio.shape)}")
        if video_tokens.shape[:2] != aligned_audio.shape[:2] or video_tokens.shape[-1] != self.hidden_size:
            raise ValueError("video/audio batch, time, or hidden dimensions do not match")
        if aligned_audio.shape[-1] != self.hidden_size:
            raise ValueError("aligned_audio hidden dimension does not match fusion hidden_size")

        residual_scale = float(residual_scale)
        if not math.isfinite(residual_scale) or residual_scale < 0:
            raise ValueError("residual_scale must be a finite non-negative number")
        aligned_audio = torch.nan_to_num(aligned_audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio_delta = self.audio_delta(aligned_audio)
        audio_delta = torch.nan_to_num(audio_delta, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(2)
        gate_view = _broadcast_gate(gate, video_tokens)
        gate_view = torch.nan_to_num(gate_view, nan=0.0, posinf=1.0, neginf=0.0)
        return video_tokens + residual_scale * gate_view * audio_delta

    def audio_delta(self, aligned_audio: torch.Tensor) -> torch.Tensor:
        aligned_audio = torch.nan_to_num(aligned_audio, nan=0.0, posinf=0.0, neginf=0.0)
        return self.audio_projector(aligned_audio)


def _broadcast_gate(gate: torch.Tensor, video_tokens: torch.Tensor) -> torch.Tensor:
    gate = gate.to(device=video_tokens.device, dtype=video_tokens.dtype)
    batch, steps, num_tokens, _ = video_tokens.shape
    if gate.ndim == 2:
        gate = gate.view(batch, steps, 1, 1)
    elif gate.ndim == 3:
        if gate.shape == (batch, steps, 1):
            gate = gate.view(batch, steps, 1, 1)
        elif gate.shape == (batch, steps, num_tokens):
            gate = gate.unsqueeze(-1)
        else:
            raise ValueError(f"Cannot broadcast gate shape {tuple(gate.shape)} to video tokens")
    elif gate.ndim == 4:
        if gate.shape not in {(batch, steps, 1, 1), (batch, steps, num_tokens, 1)}:
            raise ValueError(f"Cannot broadcast gate shape {tuple(gate.shape)} to video tokens")
    else:
        raise ValueError(f"Cannot broadcast gate shape {tuple(gate.shape)} to video tokens")
    return gate
