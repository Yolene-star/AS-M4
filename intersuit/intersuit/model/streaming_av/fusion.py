"""Gated residual audio-video fusion for AS-M4."""

from __future__ import annotations

import torch
from torch import nn


class GatedAVFusion(nn.Module):
    """Fuse aligned scene-audio features into video tokens without adding tokens."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = int(hidden_size)
        self.audio_projector = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.audio_projector.weight)

    def forward(
        self,
        video_tokens: torch.Tensor,
        aligned_audio: torch.Tensor,
        gate: torch.Tensor,
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

        audio_delta = self.audio_projector(aligned_audio).unsqueeze(2)
        gate_view = _broadcast_gate(gate, video_tokens)
        return video_tokens + gate_view * audio_delta


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

