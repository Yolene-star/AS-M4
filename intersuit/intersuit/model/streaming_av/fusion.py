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
        delta_ratio_cap: float = 0.0,
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
        raw_delta = residual_scale * gate_view * audio_delta
        capped_delta, _ = apply_audio_delta_ratio_cap(video_tokens, raw_delta, delta_ratio_cap)
        return video_tokens + capped_delta

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


def apply_audio_delta_ratio_cap(
    video_tokens: torch.Tensor,
    raw_delta: torch.Tensor,
    ratio_cap: float = 0.0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Optionally cap each sample's audio residual norm relative to video norm.

    ``ratio_cap <= 0`` is an exact no-op: the original ``raw_delta`` tensor is
    returned without a cast or multiplication. Norms are accumulated in
    float32, while the scale applied to the residual uses its original dtype.
    """

    ratio_cap = float(ratio_cap)
    if not math.isfinite(ratio_cap) or ratio_cap < 0:
        raise ValueError("ratio_cap must be a finite non-negative number")
    if video_tokens.ndim != 4 or raw_delta.ndim != 4:
        raise ValueError("video_tokens and raw_delta must be shaped [B,T,N,H]")
    if video_tokens.shape[0] != raw_delta.shape[0]:
        raise ValueError("video_tokens and raw_delta batch dimensions do not match")

    reduce_dims = tuple(range(1, video_tokens.ndim))
    video_norm = torch.linalg.vector_norm(video_tokens.detach().float(), dim=reduce_dims)
    raw_delta_norm = torch.linalg.vector_norm(raw_delta.detach().float(), dim=reduce_dims)
    raw_ratio = raw_delta_norm / video_norm.clamp_min(float(eps))

    if ratio_cap == 0.0:
        applied_scale = torch.ones_like(raw_ratio)
        capped_delta = raw_delta
    else:
        cap = torch.full_like(raw_ratio, ratio_cap)
        needs_cap = raw_ratio > cap
        applied_scale = torch.where(
            needs_cap,
            cap / raw_ratio.clamp_min(float(eps)),
            torch.ones_like(raw_ratio),
        )
        mask_view = needs_cap.view(raw_delta.shape[0], *([1] * (raw_delta.ndim - 1)))
        # Low-precision conversion can round a mathematically exact cap a few
        # ulps above its limit. Re-measure and refine the float32 scale before
        # keeping the final tensor in the original dtype.
        for _ in range(3):
            scale_view = applied_scale.to(device=raw_delta.device).view(
                raw_delta.shape[0], *([1] * (raw_delta.ndim - 1))
            )
            candidate = (raw_delta.float() * scale_view).to(dtype=raw_delta.dtype)
            capped_delta = torch.where(mask_view, candidate, raw_delta)
            measured_norm = torch.linalg.vector_norm(capped_delta.detach().float(), dim=reduce_dims)
            measured_ratio = measured_norm / video_norm.clamp_min(float(eps))
            correction = torch.where(
                needs_cap & (measured_ratio > cap),
                cap / measured_ratio.clamp_min(float(eps)),
                torch.ones_like(measured_ratio),
            )
            applied_scale = applied_scale * correction
        scale_view = applied_scale.to(device=raw_delta.device).view(
            raw_delta.shape[0], *([1] * (raw_delta.ndim - 1))
        )
        candidate = (raw_delta.float() * scale_view).to(dtype=raw_delta.dtype)
        capped_delta = torch.where(mask_view, candidate, raw_delta)

    capped_delta_norm = torch.linalg.vector_norm(capped_delta.detach().float(), dim=reduce_dims)
    capped_ratio = capped_delta_norm / video_norm.clamp_min(float(eps))
    return capped_delta, {
        "video_norm": video_norm,
        "raw_delta_norm": raw_delta_norm,
        "raw_delta_to_video_ratio": raw_ratio,
        "audio_delta_cap": ratio_cap,
        "audio_delta_applied_scale": applied_scale,
        "capped_delta_norm": capped_delta_norm,
        "capped_delta_to_video_ratio": capped_ratio,
    }
