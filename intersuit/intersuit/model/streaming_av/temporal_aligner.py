"""Causal temporal aligner for AS-M4 streaming audio/video features."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F


class TemporalAlignerOutput(NamedTuple):
    """Soft alignment results for scene audio and video windows."""

    alignment_weights: torch.Tensor
    aligned_video_features: torch.Tensor
    offset_sec: torch.Tensor
    offset_confidence: torch.Tensor
    valid_mask: torch.Tensor


class CausalTemporalAligner(nn.Module):
    """Local-window audio/video temporal aligner.

    Offset convention: ``offset = audio_time - video_time``. A positive offset
    means the audio arrives after the matching video frame.
    """

    def __init__(
        self,
        hidden_size: int,
        align_dim: int | None = None,
        max_offset_sec: float = 1.5,
        temperature: float = 1.0,
        similarity_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if max_offset_sec < 0:
            raise ValueError("max_offset_sec must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.hidden_size = int(hidden_size)
        self.align_dim = int(align_dim or hidden_size)
        self.max_offset_sec = float(max_offset_sec)
        self.temperature = float(temperature)
        self.similarity_chunk_size = similarity_chunk_size

        if self.align_dim == self.hidden_size:
            self.audio_proj = nn.Identity()
            self.video_proj = nn.Identity()
        else:
            self.audio_proj = nn.Linear(self.hidden_size, self.align_dim, bias=False)
            self.video_proj = nn.Linear(self.hidden_size, self.align_dim, bias=False)

    def forward(
        self,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        audio_timestamps: torch.Tensor,
        video_timestamps: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
        video_mask: torch.Tensor | None = None,
        lookahead_sec: float = 0.0,
    ) -> TemporalAlignerOutput:
        if lookahead_sec < 0:
            raise ValueError("lookahead_sec must be non-negative")
        audio = _ensure_batched_features(audio_features)
        video = _summarize_video_features(video_features)
        if audio.shape[0] != video.shape[0] or audio.shape[-1] != self.hidden_size or video.shape[-1] != self.hidden_size:
            raise ValueError("audio/video batch or hidden dimensions do not match aligner configuration")

        batch, audio_steps, _ = audio.shape
        video_steps = video.shape[1]
        audio_times = _ensure_batched_times(audio_timestamps, batch, audio_steps, audio.device)
        video_times = _ensure_batched_times(video_timestamps, batch, video_steps, video.device)

        if audio_mask is None:
            a_mask = torch.ones(batch, audio_steps, dtype=torch.bool, device=audio.device)
        else:
            a_mask = audio_mask.to(device=audio.device, dtype=torch.bool)
        if video_mask is None:
            v_mask = torch.ones(batch, video_steps, dtype=torch.bool, device=video.device)
        else:
            v_mask = video_mask.to(device=video.device, dtype=torch.bool)

        audio_key = F.normalize(self.audio_proj(audio), dim=-1)
        video_key = F.normalize(self.video_proj(video), dim=-1)
        similarity = _chunked_similarity(audio_key, video_key, self.similarity_chunk_size)
        similarity = similarity / self.temperature

        offset_matrix = audio_times.unsqueeze(-1) - video_times.unsqueeze(1)
        causal = video_times.unsqueeze(1) <= audio_times.unsqueeze(-1) + float(lookahead_sec)
        local = offset_matrix.abs() <= self.max_offset_sec
        valid_pair = causal & local & a_mask.unsqueeze(-1) & v_mask.unsqueeze(1)
        has_valid = valid_pair.any(dim=-1)

        masked_similarity = similarity.masked_fill(~valid_pair, -1e9)
        weights = torch.softmax(masked_similarity, dim=-1)
        weights = weights.masked_fill(~valid_pair, 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = weights.masked_fill(~has_valid.unsqueeze(-1), 0.0)

        aligned_video = torch.matmul(weights, video)
        expected_offset = (weights * offset_matrix).sum(dim=-1)
        best_score = masked_similarity.max(dim=-1).values
        confidence = torch.sigmoid(best_score).masked_fill(~has_valid, 0.0)
        expected_offset = expected_offset.masked_fill(~has_valid, 0.0)

        return TemporalAlignerOutput(
            alignment_weights=weights,
            aligned_video_features=aligned_video,
            offset_sec=expected_offset,
            offset_confidence=confidence,
            valid_mask=has_valid,
        )


def _ensure_batched_features(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 2:
        features = features.unsqueeze(0)
    if features.ndim != 3:
        raise ValueError(f"Expected features shaped [B,T,D] or [T,D], got {tuple(features.shape)}")
    return features.to(dtype=torch.float32)


def _summarize_video_features(video_features: torch.Tensor) -> torch.Tensor:
    if video_features.ndim == 4:
        video_features = video_features.mean(dim=2)
    return _ensure_batched_features(video_features)


def _ensure_batched_times(
    timestamps: torch.Tensor,
    batch: int,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    times = timestamps.to(device=device, dtype=torch.float32)
    if times.ndim == 1:
        times = times.unsqueeze(0).expand(batch, -1)
    if times.shape != (batch, steps):
        raise ValueError(f"timestamps shape {tuple(times.shape)} does not match {(batch, steps)}")
    return times


def _chunked_similarity(
    audio_key: torch.Tensor,
    video_key: torch.Tensor,
    chunk_size: int | None,
) -> torch.Tensor:
    if chunk_size is None or chunk_size <= 0 or chunk_size >= audio_key.shape[1]:
        return torch.matmul(audio_key, video_key.transpose(-1, -2))

    chunks = []
    for start in range(0, audio_key.shape[1], chunk_size):
        end = min(start + chunk_size, audio_key.shape[1])
        chunks.append(torch.matmul(audio_key[:, start:end], video_key.transpose(-1, -2)))
    return torch.cat(chunks, dim=1)
