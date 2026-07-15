"""Audio confidence gate for AS-M4 scene-audio fusion."""

from __future__ import annotations

from contextlib import nullcontext
from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F


class AudioConfidenceGateOutput(NamedTuple):
    """Quality, relevance, and final audio gate."""

    quality: torch.Tensor
    relevance: torch.Tensor
    gate: torch.Tensor
    quality_logits: torch.Tensor
    relevance_logits: torch.Tensor


class AudioConfidenceGate(nn.Module):
    """Estimate whether scene audio should affect video features."""

    def __init__(
        self,
        hidden_size: int,
        quality_dim: int = 0,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if quality_dim < 0:
            raise ValueError("quality_dim must be non-negative")
        hidden = int(hidden_dim or max(32, min(512, hidden_size)))
        self.hidden_size = int(hidden_size)
        self.quality_dim = int(quality_dim)

        quality_input_dim = hidden_size + self.quality_dim
        relevance_input_dim = hidden_size * 3 + 2
        self.quality_mlp = nn.Sequential(
            _DTypeSafeLayerNorm(quality_input_dim),
            nn.Linear(quality_input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.relevance_mlp = nn.Sequential(
            _DTypeSafeLayerNorm(relevance_input_dim),
            nn.Linear(relevance_input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self._zero_last_layers()

    def forward(
        self,
        audio_feature: torch.Tensor,
        video_feature: torch.Tensor,
        question_feature: torch.Tensor | None = None,
        quality_features: torch.Tensor | None = None,
        alignment_confidence: torch.Tensor | None = None,
    ) -> AudioConfidenceGateOutput:
        audio = _ensure_batched(audio_feature)
        video = _ensure_batched(video_feature)
        if audio.shape != video.shape or audio.shape[-1] != self.hidden_size:
            raise ValueError("audio/video features must share [B,T,H] with configured hidden_size")
        batch, steps, _ = audio.shape

        if question_feature is None:
            question = torch.zeros_like(audio)
        else:
            question = _broadcast_question(question_feature, batch, steps, self.hidden_size, audio.device)

        if quality_features is None:
            quality = torch.empty(batch, steps, 0, device=audio.device, dtype=audio.dtype)
        else:
            quality = quality_features.to(device=audio.device, dtype=audio.dtype)
            if quality.ndim == 2:
                quality = quality.unsqueeze(-1)
            if quality.shape[:2] != (batch, steps) or quality.shape[-1] != self.quality_dim:
                raise ValueError(
                    f"quality_features shape {tuple(quality.shape)} does not match {(batch, steps, self.quality_dim)}"
                )

        if alignment_confidence is None:
            align_conf = torch.ones(batch, steps, device=audio.device, dtype=audio.dtype)
        else:
            align_conf = alignment_confidence.to(device=audio.device, dtype=audio.dtype)
            if align_conf.shape != (batch, steps):
                raise ValueError(f"alignment_confidence shape {tuple(align_conf.shape)} does not match {(batch, steps)}")

        autocast_ctx = (
            torch.amp.autocast(device_type=audio.device.type, enabled=False)
            if hasattr(torch, "amp") and audio.device.type in {"cuda", "cpu"}
            else nullcontext()
        )
        with autocast_ctx:
            audio = audio.float()
            video = video.float()
            question = question.float()
            quality = quality.float()
            align_conf = align_conf.float()

            audio_energy = audio.abs().mean(dim=-1)
            quality_hint = torch.logit(audio_energy.clamp(1e-4, 1.0 - 1e-4))
            if self.quality_dim > 0:
                quality_hint = quality_hint + quality[..., 0]
            quality_input = torch.cat([audio, quality], dim=-1)
            quality_logits = self.quality_mlp(quality_input).squeeze(-1) + quality_hint

            av_cos = F.cosine_similarity(audio, video, dim=-1)
            aq_cos = F.cosine_similarity(audio, question, dim=-1)
            relevance_hint = 4.0 * av_cos + 1.0 * aq_cos + torch.logit(align_conf.clamp(1e-4, 1.0 - 1e-4))
            relevance_input = torch.cat(
                [audio, video, question, av_cos.unsqueeze(-1), align_conf.unsqueeze(-1)],
                dim=-1,
            )
            relevance_logits = self.relevance_mlp(relevance_input).squeeze(-1) + relevance_hint

            q = torch.sigmoid(quality_logits)
            r = torch.sigmoid(relevance_logits)
            gate = q * r
        return AudioConfidenceGateOutput(
            quality=q,
            relevance=r,
            gate=gate,
            quality_logits=quality_logits,
            relevance_logits=relevance_logits,
        )

    def _zero_last_layers(self) -> None:
        for module in (self.quality_mlp[-1], self.relevance_mlp[-1]):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)


class _DTypeSafeLayerNorm(nn.LayerNorm):
    """LayerNorm that honors its parameter dtype under ZeRO/bf16 autocast."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        target_dtype = self.weight.dtype if self.weight is not None else torch.float32
        autocast_ctx = (
            torch.cuda.amp.autocast(enabled=False)
            if input.device.type == "cuda"
            else nullcontext()
        )
        with autocast_ctx:
            return F.layer_norm(
                input.to(dtype=target_dtype),
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            )


def _ensure_batched(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 2:
        features = features.unsqueeze(0)
    if features.ndim != 3:
        raise ValueError(f"Expected features shaped [B,T,H] or [T,H], got {tuple(features.shape)}")
    return features.to(dtype=torch.float32)


def _broadcast_question(
    question_feature: torch.Tensor,
    batch: int,
    steps: int,
    hidden_size: int,
    device: torch.device,
) -> torch.Tensor:
    question = question_feature.to(device=device, dtype=torch.float32)
    if question.ndim == 1:
        question = question.view(1, 1, hidden_size).expand(batch, steps, -1)
    elif question.ndim == 2:
        if question.shape == (batch, hidden_size):
            question = question.unsqueeze(1).expand(-1, steps, -1)
        elif question.shape == (steps, hidden_size):
            question = question.unsqueeze(0).expand(batch, -1, -1)
        else:
            raise ValueError(f"Cannot broadcast question_feature shape {tuple(question.shape)}")
    elif question.ndim == 3:
        if question.shape != (batch, steps, hidden_size):
            raise ValueError(f"question_feature shape {tuple(question.shape)} does not match {(batch, steps, hidden_size)}")
    else:
        raise ValueError(f"Unexpected question_feature shape {tuple(question.shape)}")
    return question
