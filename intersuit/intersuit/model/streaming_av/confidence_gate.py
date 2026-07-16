"""Audio confidence gate for AS-M4 scene-audio fusion."""

from __future__ import annotations

from contextlib import nullcontext
import math
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


class AudioSignalFeatures(NamedTuple):
    """Per-window waveform statistics consumed by Gate v1."""

    rms: torch.Tensor
    loudness_dbfs: torch.Tensor
    silence_ratio: torch.Tensor
    norm: torch.Tensor


class AudioConfidenceGate(nn.Module):
    """Estimate whether scene audio should affect video features."""

    def __init__(
        self,
        hidden_size: int,
        quality_dim: int = 0,
        hidden_dim: int | None = None,
        gate_logit_bias: float = -5.0,
        enable_v1: bool = False,
        silence_threshold: float = 1e-4,
        rms_reference: float = 0.05,
        max_offset_sec: float = 1.5,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if quality_dim < 0:
            raise ValueError("quality_dim must be non-negative")
        hidden = int(hidden_dim or max(32, min(512, hidden_size)))
        self.hidden_size = int(hidden_size)
        self.quality_dim = int(quality_dim)
        self.gate_logit_bias = float(gate_logit_bias)
        self.enable_v1 = bool(enable_v1)
        self.silence_threshold = float(silence_threshold)
        self.rms_reference = float(rms_reference)
        self.max_offset_sec = float(max_offset_sec)
        self._last_v1_diagnostics: dict[str, torch.Tensor] = {}
        if self.silence_threshold < 0:
            raise ValueError("silence_threshold must be non-negative")
        if self.rms_reference <= 0:
            raise ValueError("rms_reference must be positive")

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
        signal_features: AudioSignalFeatures | None = None,
        offset_sec: torch.Tensor | None = None,
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

            q = torch.sigmoid(quality_logits + self.gate_logit_bias)
            r = torch.sigmoid(relevance_logits + self.gate_logit_bias)
            gate = q * r

            if self.enable_v1:
                diagnostic_zeros = torch.zeros_like(gate)
                diagnostic_ones = torch.ones_like(gate)
                question_similarity = diagnostic_zeros
                offset_score = diagnostic_ones
                signal = signal_features or compute_audio_signal_features(
                    audio,
                    silence_threshold=self.silence_threshold,
                )
                audio_rms = _validate_scalar_feature("rms", signal.rms, batch, steps, audio.device)
                audio_loudness_dbfs = _validate_scalar_feature(
                    "loudness_dbfs", signal.loudness_dbfs, batch, steps, audio.device
                )
                silence_ratio = _validate_scalar_feature(
                    "silence_ratio", signal.silence_ratio, batch, steps, audio.device
                ).clamp(0.0, 1.0)
                audio_norm = _validate_scalar_feature("norm", signal.norm, batch, steps, audio.device)

                rms_score = (audio_rms / self.rms_reference).clamp(0.0, 1.0)
                normalized_norm = audio_norm / math.sqrt(max(1, audio.shape[-1]))
                norm_score = (normalized_norm / self.rms_reference).clamp(0.0, 1.0)
                non_silent_score = (1.0 - silence_ratio).clamp(0.0, 1.0)
                v1_quality_factor = rms_score * norm_score * non_silent_score

                if question_feature is not None:
                    question_similarity = F.cosine_similarity(audio, question, dim=-1)
                    question_score = ((question_similarity + 1.0) * 0.5).clamp(0.0, 1.0)
                else:
                    question_score = diagnostic_ones

                if offset_sec is not None:
                    offset = _validate_scalar_feature("offset_sec", offset_sec, batch, steps, audio.device)
                    offset_scale = max(1e-6, self.max_offset_sec)
                    offset_score = torch.exp(-offset.abs() / offset_scale)
                offset_score = offset_score * align_conf.clamp(0.0, 1.0)
                v1_relevance_factor = question_score * offset_score
                gate = gate * v1_quality_factor * v1_relevance_factor
                gate = torch.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                self._last_v1_diagnostics = {
                    "audio_rms": audio_rms,
                    "audio_loudness_dbfs": audio_loudness_dbfs,
                    "silence_ratio": silence_ratio,
                    "audio_norm": audio_norm,
                    "question_similarity": question_similarity,
                    "offset_score": offset_score,
                    "v1_quality_factor": v1_quality_factor,
                    "v1_relevance_factor": v1_relevance_factor,
                }
        return AudioConfidenceGateOutput(
            quality=q,
            relevance=r,
            gate=gate,
            quality_logits=quality_logits,
            relevance_logits=relevance_logits,
        )

    @property
    def last_v1_diagnostics(self) -> dict[str, torch.Tensor]:
        return self._last_v1_diagnostics

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


def compute_audio_signal_features(
    audio_windows: torch.Tensor,
    sample_mask: torch.Tensor | None = None,
    silence_threshold: float = 1e-4,
) -> AudioSignalFeatures:
    """Compute dependency-free waveform statistics for Gate v1."""

    windows = _ensure_batched(audio_windows).float()
    batch, steps, samples = windows.shape
    if samples == 0:
        raise ValueError("audio_windows must contain at least one sample per window")
    if silence_threshold < 0:
        raise ValueError("silence_threshold must be non-negative")
    if sample_mask is None:
        mask = torch.ones(batch, steps, dtype=torch.bool, device=windows.device)
    else:
        mask = sample_mask.to(device=windows.device, dtype=torch.bool)
        if mask.shape != (batch, steps):
            raise ValueError(f"sample_mask shape {tuple(mask.shape)} does not match {(batch, steps)}")

    finite = torch.nan_to_num(windows, nan=0.0, posinf=0.0, neginf=0.0)
    rms = finite.square().mean(dim=-1).sqrt()
    loudness_dbfs = 20.0 * torch.log10(rms.clamp_min(1e-12))
    silence_ratio = finite.abs().le(float(silence_threshold)).float().mean(dim=-1)
    norm = finite.norm(dim=-1)
    zeros = torch.zeros_like(rms)
    return AudioSignalFeatures(
        rms=torch.where(mask, rms, zeros),
        loudness_dbfs=torch.where(mask, loudness_dbfs, zeros),
        silence_ratio=torch.where(mask, silence_ratio, torch.ones_like(silence_ratio)),
        norm=torch.where(mask, norm, zeros),
    )


def _validate_scalar_feature(
    name: str,
    value: torch.Tensor,
    batch: int,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    feature = value.to(device=device, dtype=torch.float32)
    if feature.shape != (batch, steps):
        raise ValueError(f"{name} shape {tuple(feature.shape)} does not match {(batch, steps)}")
    return torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)


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
