"""Diagnostic-only local audio event alignment for AS-M4."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .offset_stabilizer import stabilize_offset_scores


class AudioEventFeatures(NamedTuple):
    event_strength: torch.Tensor
    is_silent_window: torch.Tensor
    audio_rms: torch.Tensor
    audio_peak: torch.Tensor
    non_silent_ratio: torch.Tensor
    feature_change: torch.Tensor
    mask: torch.Tensor


class LocalAudioAlignmentOutput(NamedTuple):
    candidate_offsets: torch.Tensor
    candidate_indices: torch.Tensor
    candidate_valid: torch.Tensor
    semantic_similarity: torch.Tensor
    video_event_strength: torch.Tensor
    candidate_scores: torch.Tensor
    best_offset: torch.Tensor
    best_alignment_score: torch.Tensor
    second_best_alignment_score: torch.Tensor
    alignment_margin: torch.Tensor
    alignment_confidence: torch.Tensor
    offset_scorer_candidate_scores: torch.Tensor
    offset_scorer_best_offset: torch.Tensor
    offset_scorer_margin: torch.Tensor
    offset_scorer_accepted: torch.Tensor
    offset_scorer_suggested_offset: torch.Tensor
    offset_scorer_available: torch.Tensor
    offset_scorer_stable_candidate_scores: torch.Tensor
    offset_scorer_stable_best_offset: torch.Tensor
    offset_scorer_stable_margin: torch.Tensor
    offset_scorer_stable_accepted: torch.Tensor
    offset_scorer_stable_suggested_offset: torch.Tensor
    offset_scorer_stable_delay_windows: torch.Tensor


class FrozenOffsetScorerInputs(NamedTuple):
    """Exact frozen feature streams required by the accepted AVE scorer."""

    audio_features: torch.Tensor
    clip_features: torch.Tensor
    rgb_features: torch.Tensor
    audio_rms: torch.Tensor
    non_silent_ratio: torch.Tensor


class FrozenOffsetScorerOutput(NamedTuple):
    candidate_scores: torch.Tensor
    best_offset: torch.Tensor
    margin: torch.Tensor
    accepted: torch.Tensor
    suggested_offset: torch.Tensor
    available: torch.Tensor


class FrozenTemporalOffsetScorer(nn.Module):
    """Frozen diagnostic scorer matching the accepted AVE feature schema."""

    FORMAT_VERSION = 1

    def __init__(self, bundle_path: str | Path, margin_threshold: float = 0.15) -> None:
        super().__init__()
        if margin_threshold < 0:
            raise ValueError("margin_threshold must be non-negative")
        path = Path(bundle_path)
        if not path.is_file():
            raise FileNotFoundError(f"Frozen offset scorer bundle not found: {path}")
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        if int(bundle.get("format_version", -1)) != self.FORMAT_VERSION:
            raise ValueError("Unsupported frozen offset scorer bundle format")
        metadata = bundle.get("metadata")
        state = bundle.get("state_dict")
        scalar_mean = bundle.get("scalar_mean")
        scalar_std = bundle.get("scalar_std")
        if not isinstance(metadata, dict) or not isinstance(state, dict):
            raise ValueError("Frozen offset scorer bundle is missing metadata/state_dict")
        if not torch.is_tensor(scalar_mean) or not torch.is_tensor(scalar_std):
            raise ValueError("Frozen offset scorer bundle is missing scalar normalization statistics")

        self.audio_dim = int(metadata["audio_dim"])
        self.clip_dim = int(metadata["clip_dim"])
        self.rgb_dim = int(metadata["rgb_dim"])
        self.context_radius = int(metadata["context_radius"])
        self.scalar_dim = int(metadata["scalar_dim"])
        self.hidden_dim = int(metadata["hidden_dim"])
        if "seed" in metadata and int(metadata["seed"]) != 20260719:
            raise ValueError("Frozen offset scorer bundle must use accepted seed 20260719")
        if "margin_threshold" in metadata and abs(
            float(metadata["margin_threshold"]) - float(margin_threshold)
        ) > 1e-12:
            raise ValueError(
                "Frozen offset scorer margin threshold is locked by the runtime bundle: "
                f"{metadata['margin_threshold']}"
            )
        if "zero_class_weight" in metadata and float(metadata["zero_class_weight"]) != 1.25:
            raise ValueError("Frozen offset scorer bundle must keep zero_class_weight=1.25")
        if "require_center_peak" in metadata and not bool(metadata["require_center_peak"]):
            raise ValueError("Frozen offset scorer bundle must keep center-peak filtering")
        offsets = tuple(float(value) for value in metadata["candidate_offsets"])
        if offsets != (-0.5, 0.0, 0.5):
            raise ValueError(f"Frozen scorer candidate offsets are not accepted: {offsets}")
        context_count = self.context_radius * 2 + 1
        pair_dim = (
            context_count * self.audio_dim * 2
            + context_count * (self.clip_dim + self.rgb_dim) * 2
            + self.scalar_dim
        )
        first_weight = state.get("net.0.weight")
        if not torch.is_tensor(first_weight) or tuple(first_weight.shape) != (self.hidden_dim, pair_dim):
            raise ValueError(
                f"Frozen scorer input shape mismatch: expected {(self.hidden_dim, pair_dim)}, "
                f"got {None if not torch.is_tensor(first_weight) else tuple(first_weight.shape)}"
            )
        if scalar_mean.numel() != self.scalar_dim or scalar_std.numel() != self.scalar_dim:
            raise ValueError("Frozen scorer scalar statistics do not match scalar_dim")

        self.net = nn.Sequential(
            nn.Linear(pair_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.net.load_state_dict(
            {
                (key[len("net.") :] if key.startswith("net.") else key): value
                for key, value in state.items()
            }
        )
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.register_buffer("scalar_mean", scalar_mean.float().reshape(self.scalar_dim), persistent=True)
        self.register_buffer("scalar_std", scalar_std.float().reshape(self.scalar_dim).clamp_min(1e-6), persistent=True)
        self.register_buffer("candidate_offsets", torch.tensor(offsets, dtype=torch.float32), persistent=True)
        self.margin_threshold = float(margin_threshold)
        self.bundle_path = str(path.resolve())
        self.bundle_metadata = metadata

    @torch.no_grad()
    def forward(self, inputs: FrozenOffsetScorerInputs) -> FrozenOffsetScorerOutput:
        audio = _ensure_frozen_stream(inputs.audio_features, self.audio_dim, "audio")
        clip = _ensure_frozen_stream(inputs.clip_features, self.clip_dim, "clip")
        rgb = _ensure_frozen_stream(inputs.rgb_features, self.rgb_dim, "rgb")
        if audio.shape[:2] != clip.shape[:2] or audio.shape[:2] != rgb.shape[:2]:
            raise ValueError("Frozen offset scorer audio/CLIP/RGB windows must be aligned")
        batch, steps, _ = audio.shape
        rms = _ensure_frozen_scalar(inputs.audio_rms, batch, steps, audio.device, "audio_rms")
        nonsilent = _ensure_frozen_scalar(
            inputs.non_silent_ratio,
            batch,
            steps,
            audio.device,
            "non_silent_ratio",
        )

        audio_change = _frozen_diff_norm(audio)
        clip_change = _frozen_diff_norm(clip)
        rgb_change = _frozen_diff_norm(rgb)
        energy_change = _frozen_diff_norm(rms.unsqueeze(-1))
        audio = F.normalize(audio, dim=-1, eps=1e-6)
        clip = F.normalize(clip, dim=-1, eps=1e-6)
        rgb = F.normalize(rgb, dim=-1, eps=1e-6)
        scores = torch.zeros(batch, steps, 3, device=audio.device, dtype=torch.float32)
        valid = torch.zeros(batch, steps, 3, device=audio.device, dtype=torch.bool)

        for video_index in range(steps):
            video_context = torch.cat(
                [
                    _frozen_context_with_deltas(clip, video_index, self.context_radius),
                    _frozen_context_with_deltas(rgb, video_index, self.context_radius),
                ],
                dim=-1,
            )
            for candidate_slot, audio_index in enumerate(
                (video_index - 1, video_index, video_index + 1)
            ):
                if audio_index < 0 or audio_index >= steps:
                    continue
                scalar_context = _frozen_scalar_context(
                    rms,
                    nonsilent,
                    audio_change,
                    energy_change,
                    clip_change,
                    rgb_change,
                    audio_index,
                    video_index,
                    self.context_radius,
                )
                evidence = _frozen_sync_evidence(
                    audio_change + energy_change,
                    clip_change + rgb_change,
                    audio_index,
                    video_index,
                    self.context_radius,
                )
                offset = self.candidate_offsets[candidate_slot].to(device=audio.device)
                scalar = torch.cat(
                    [
                        scalar_context,
                        torch.stack(
                            [
                                offset.expand(batch),
                                offset.abs().expand(batch),
                                evidence["sync_peak_delta"],
                                evidence["sync_peak_delta"].abs(),
                                evidence["center_audio_dominance"],
                                evidence["center_video_dominance"],
                                evidence["center_audio_dominance"]
                                * evidence["center_video_dominance"],
                            ],
                            dim=-1,
                        ),
                    ],
                    dim=-1,
                )
                scalar = (scalar - self.scalar_mean.to(audio.device)) / self.scalar_std.to(audio.device)
                values = torch.cat(
                    [
                        _frozen_context_with_deltas(audio, audio_index, self.context_radius),
                        video_context,
                        scalar,
                    ],
                    dim=-1,
                )
                scores[:, video_index, candidate_slot] = self.net(values).squeeze(-1)
                valid[:, video_index, candidate_slot] = True

        selection = scores.masked_fill(~valid, -1e9)
        tie_break = self.candidate_offsets.abs().to(audio.device).view(1, 1, 3) * 1e-6
        best_index = (selection - tie_break).argmax(dim=-1)
        best_score = selection.gather(2, best_index.unsqueeze(-1)).squeeze(-1)
        offsets = self.candidate_offsets.to(audio.device)
        best_offset = offsets[best_index]
        second_score = selection.topk(k=2, dim=-1).values[..., 1]
        enough_candidates = valid.sum(dim=-1) >= 2
        margin = torch.where(enough_candidates, best_score - second_score, torch.zeros_like(best_score))
        accepted = enough_candidates & margin.ge(self.margin_threshold)
        suggested = torch.where(accepted, best_offset, torch.zeros_like(best_offset))
        available = torch.ones_like(accepted)
        return FrozenOffsetScorerOutput(
            candidate_scores=scores.masked_fill(~valid, 0.0),
            best_offset=best_offset,
            margin=margin.clamp_min(0.0),
            accepted=accepted,
            suggested_offset=suggested,
            available=available,
        )


def compute_audio_event_features(
    audio_windows: torch.Tensor,
    audio_features: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
    silence_threshold: float = 1e-4,
    rms_reference: float = 0.05,
) -> AudioEventFeatures:
    """Compute bounded, interpretable event statistics for audio windows."""

    windows = _ensure_audio_windows(audio_windows)
    if silence_threshold < 0:
        raise ValueError("silence_threshold must be non-negative")
    if rms_reference <= 0:
        raise ValueError("rms_reference must be positive")

    batch, steps, samples = windows.shape
    if samples == 0:
        raise ValueError("audio_windows must contain at least one sample per window")
    mask = _ensure_mask(sample_mask, batch, steps, windows.device)
    finite = torch.nan_to_num(windows.float(), nan=0.0, posinf=0.0, neginf=0.0)
    audio_rms = finite.square().mean(dim=-1).sqrt()
    audio_peak = finite.abs().amax(dim=-1)
    non_silent_ratio = finite.abs().gt(float(silence_threshold)).float().mean(dim=-1)
    feature_change = _adjacent_feature_change(audio_features, audio_rms, mask)

    rms_level = (audio_rms / float(rms_reference)).clamp(0.0, 1.0)
    peak_level = (audio_peak / float(2.0 * rms_reference)).clamp(0.0, 1.0)
    event_strength = (
        0.40 * rms_level
        + 0.20 * peak_level
        + 0.25 * non_silent_ratio
        + 0.15 * feature_change
    ).clamp(0.0, 1.0)
    is_silent = (audio_rms <= float(silence_threshold)) & (audio_peak <= float(silence_threshold))
    event_strength = torch.where(is_silent, torch.zeros_like(event_strength), event_strength)

    zeros = torch.zeros_like(audio_rms)
    return AudioEventFeatures(
        event_strength=torch.where(mask, event_strength, zeros),
        is_silent_window=torch.where(mask, is_silent, torch.ones_like(is_silent)),
        audio_rms=torch.where(mask, audio_rms, zeros),
        audio_peak=torch.where(mask, audio_peak, zeros),
        non_silent_ratio=torch.where(mask, non_silent_ratio, zeros),
        feature_change=torch.where(mask, feature_change, zeros),
        mask=mask,
    )


class LocalAudioEventAligner(nn.Module):
    """Score nearby audio windows without changing the fused audio window."""

    SEMANTIC_DISABLED = "disabled"
    SEMANTIC_SHARED_PRECOMPUTED = "shared_precomputed"
    SEMANTIC_CHECKPOINT_LOADED = "checkpoint_loaded"

    def __init__(
        self,
        hidden_size: int,
        align_dim: int | None = None,
        candidate_offsets: Sequence[float] = (-0.5, 0.0, 0.5),
        event_strength_weight: float = 0.05,
        semantic_feature_mode: str = SEMANTIC_DISABLED,
        projector_checkpoint_path: str | None = None,
        offset_scorer_bundle_path: str | None = None,
        offset_scorer_margin_threshold: float = 0.15,
        offset_scorer_stabilization_strategy: str = "none",
        offset_scorer_consecutive_windows: int = 2,
        offset_scorer_hold_margin: float = 0.10,
        offset_scorer_switch_margin: float = 0.30,
        offset_scorer_moving_average_windows: int = 3,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        offsets = tuple(float(value) for value in candidate_offsets)
        if len(offsets) < 2:
            raise ValueError("candidate_offsets must contain at least two values")
        if 0.0 not in offsets:
            raise ValueError("candidate_offsets must include 0.0")
        if event_strength_weight < 0:
            raise ValueError("event_strength_weight must be non-negative")
        self.hidden_size = int(hidden_size)
        self.align_dim = int(align_dim or hidden_size)
        self.candidate_offsets = offsets
        self.event_strength_weight = float(event_strength_weight)
        self.audio_proj: nn.Linear | None = None
        self.video_proj: nn.Linear | None = None
        self.semantic_feature_mode = self._normalize_semantic_feature_mode(semantic_feature_mode)
        if self.semantic_feature_mode == self.SEMANTIC_CHECKPOINT_LOADED:
            if projector_checkpoint_path is None:
                raise ValueError("projector_checkpoint_path is required for checkpoint_loaded semantic mode")
            self.load_projector_checkpoint(projector_checkpoint_path)
        elif projector_checkpoint_path is not None:
            self.load_projector_checkpoint(projector_checkpoint_path)
        self.frozen_offset_scorer = (
            FrozenTemporalOffsetScorer(
                offset_scorer_bundle_path,
                margin_threshold=offset_scorer_margin_threshold,
            )
            if offset_scorer_bundle_path
            else None
        )
        self.offset_scorer_stabilization_strategy = str(
            offset_scorer_stabilization_strategy
        ).lower()
        if self.offset_scorer_stabilization_strategy not in {
            "none",
            "consecutive",
            "hysteresis",
            "moving_average",
        }:
            raise ValueError("Unsupported frozen offset scorer stabilization strategy")
        self.offset_scorer_stabilization_kwargs = {
            "margin_threshold": float(offset_scorer_margin_threshold),
            "consecutive_windows": int(offset_scorer_consecutive_windows),
            "hold_margin": float(offset_scorer_hold_margin),
            "switch_margin": float(offset_scorer_switch_margin),
            "moving_average_windows": int(offset_scorer_moving_average_windows),
        }

    def forward(
        self,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        audio_timestamps: torch.Tensor,
        video_timestamps: torch.Tensor,
        event_features: AudioEventFeatures,
        audio_mask: torch.Tensor | None = None,
        video_mask: torch.Tensor | None = None,
        frozen_offset_inputs: FrozenOffsetScorerInputs | None = None,
    ) -> LocalAudioAlignmentOutput:
        audio = _ensure_features(audio_features)
        video = _summarize_video(video_features)
        if audio.shape[0] != video.shape[0] or audio.shape[-1] != video.shape[-1]:
            raise ValueError("audio/video batch or hidden dimensions do not match")
        if audio.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected hidden_size {self.hidden_size}, got {audio.shape[-1]}")

        batch, audio_steps, hidden = audio.shape
        video_steps = video.shape[1]
        audio_times = _ensure_times(audio_timestamps, batch, audio_steps, audio.device)
        video_times = _ensure_times(video_timestamps, batch, video_steps, video.device)
        a_mask = _ensure_mask(audio_mask, batch, audio_steps, audio.device) & event_features.mask.to(audio.device)
        v_mask = _ensure_mask(video_mask, batch, video_steps, video.device)
        offsets = torch.tensor(self.candidate_offsets, device=audio.device, dtype=torch.float32)

        target_audio_times = video_times.unsqueeze(-1) - offsets.view(1, 1, -1)
        distances = (target_audio_times.unsqueeze(-1) - audio_times.unsqueeze(1).unsqueeze(1)).abs()
        distances = distances.masked_fill(~a_mask.unsqueeze(1).unsqueeze(1), float("inf"))
        candidate_indices = distances.argmin(dim=-1)

        min_time = audio_times.masked_fill(~a_mask, float("inf")).amin(dim=-1, keepdim=True)
        max_time = audio_times.masked_fill(~a_mask, float("-inf")).amax(dim=-1, keepdim=True)
        has_audio = a_mask.any(dim=-1, keepdim=True)
        candidate_valid = (
            has_audio.unsqueeze(1)
            & v_mask.unsqueeze(-1)
            & (target_audio_times >= min_time.unsqueeze(1))
            & (target_audio_times <= max_time.unsqueeze(1))
        )

        gather_index = candidate_indices.unsqueeze(-1).expand(-1, -1, -1, hidden)
        expanded_audio = audio.unsqueeze(1).expand(-1, video_steps, -1, -1)
        candidate_audio = torch.gather(expanded_audio, 2, gather_index)
        candidate_strength = torch.gather(
            event_features.event_strength.to(audio.device).unsqueeze(1).expand(-1, video_steps, -1),
            2,
            candidate_indices,
        )
        video_event_strength = _temporal_feature_change(video, v_mask)
        semantic_similarity = self._semantic_similarity(candidate_audio, video, candidate_valid)
        semantic_score = ((semantic_similarity + 1.0) * 0.5).clamp(0.0, 1.0)
        if self.semantic_feature_mode == self.SEMANTIC_DISABLED:
            semantic_score = torch.zeros_like(semantic_score)
        event_match = 1.0 - (candidate_strength - video_event_strength.unsqueeze(-1)).abs().clamp(0.0, 1.0)
        candidate_scores = torch.nan_to_num(
            semantic_score
            + self.event_strength_weight * (0.5 * event_match + 0.5 * candidate_strength),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).masked_fill(~candidate_valid, 0.0)
        candidate_scores = candidate_scores.masked_fill(candidate_strength <= 0.0, 0.0)

        tie_break = offsets.abs().view(1, 1, -1) * 1e-6
        selection_scores = (candidate_scores - tie_break).masked_fill(~candidate_valid, -1e9)
        best_index = selection_scores.argmax(dim=-1)
        best_score = torch.gather(candidate_scores, 2, best_index.unsqueeze(-1)).squeeze(-1)
        best_offset = offsets[best_index]

        sorted_scores = selection_scores.topk(k=2, dim=-1).indices
        second_index = sorted_scores[..., 1]
        second_score = torch.gather(candidate_scores, 2, second_index.unsqueeze(-1)).squeeze(-1)
        second_score = torch.where(candidate_valid.sum(dim=-1) >= 2, second_score, best_score)
        has_valid_candidate = candidate_valid.any(dim=-1)
        best_score = torch.where(has_valid_candidate, best_score, torch.zeros_like(best_score))
        best_offset = torch.where(has_valid_candidate, best_offset, torch.zeros_like(best_offset))
        margin = (best_score - second_score).clamp_min(0.0)
        confidence = (best_score * (0.5 + 0.5 * margin)).clamp(0.0, 1.0)
        frozen = self._run_frozen_offset_scorer(
            frozen_offset_inputs,
            batch=batch,
            fallback_steps=video_steps,
            device=audio.device,
        )
        stable = stabilize_offset_scores(
            frozen.candidate_scores,
            self.offset_scorer_stabilization_strategy,
            **self.offset_scorer_stabilization_kwargs,
        )
        stable = stable._replace(
            accepted=stable.accepted & frozen.available,
            suggested_offset=torch.where(
                frozen.available,
                stable.suggested_offset,
                torch.zeros_like(stable.suggested_offset),
            ),
        )

        return LocalAudioAlignmentOutput(
            candidate_offsets=offsets.view(1, 1, -1).expand(batch, video_steps, -1),
            candidate_indices=candidate_indices,
            candidate_valid=candidate_valid,
            semantic_similarity=semantic_similarity.masked_fill(~candidate_valid, 0.0),
            video_event_strength=video_event_strength,
            candidate_scores=candidate_scores,
            best_offset=best_offset,
            best_alignment_score=best_score,
            second_best_alignment_score=second_score,
            alignment_margin=margin,
            alignment_confidence=confidence,
            offset_scorer_candidate_scores=frozen.candidate_scores,
            offset_scorer_best_offset=frozen.best_offset,
            offset_scorer_margin=frozen.margin,
            offset_scorer_accepted=frozen.accepted,
            offset_scorer_suggested_offset=frozen.suggested_offset,
            offset_scorer_available=frozen.available,
            offset_scorer_stable_candidate_scores=stable.candidate_scores,
            offset_scorer_stable_best_offset=stable.best_offset,
            offset_scorer_stable_margin=stable.margin,
            offset_scorer_stable_accepted=stable.accepted,
            offset_scorer_stable_suggested_offset=stable.suggested_offset,
            offset_scorer_stable_delay_windows=stable.decision_delay_windows,
        )

    def _run_frozen_offset_scorer(
        self,
        inputs: FrozenOffsetScorerInputs | None,
        batch: int,
        fallback_steps: int,
        device: torch.device,
    ) -> FrozenOffsetScorerOutput:
        if self.frozen_offset_scorer is not None and inputs is not None:
            scorer = self.frozen_offset_scorer.to(device=device)
            return scorer(inputs)
        scores = torch.zeros(batch, fallback_steps, 3, device=device, dtype=torch.float32)
        offsets = torch.zeros(batch, fallback_steps, device=device, dtype=torch.float32)
        flags = torch.zeros(batch, fallback_steps, device=device, dtype=torch.bool)
        return FrozenOffsetScorerOutput(
            candidate_scores=scores,
            best_offset=offsets,
            margin=offsets,
            accepted=flags,
            suggested_offset=offsets,
            available=flags,
        )

    def load_projector_checkpoint(self, checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> None:
        """Load trained audio/video projectors before semantic scoring is enabled."""

        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Audio event aligner projector checkpoint not found: {path}. "
                "Semantic alignment with trainable projections requires a trained checkpoint."
            )
        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location=map_location)
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state, dict):
            raise ValueError("Projector checkpoint must contain a state_dict-like mapping")

        audio_weight = _find_weight(state, ("audio_proj.weight", "audio_projector.weight"))
        video_weight = _find_weight(state, ("video_proj.weight", "video_projector.weight"))
        if audio_weight is None or video_weight is None:
            raise ValueError("Projector checkpoint must contain audio_proj.weight and video_proj.weight")
        if audio_weight.ndim != 2 or video_weight.ndim != 2:
            raise ValueError("Projector weights must be rank-2 tensors")
        if audio_weight.shape[1] != self.hidden_size or video_weight.shape[1] != self.hidden_size:
            raise ValueError(
                "Projector input dimensions do not match aligner hidden_size: "
                f"audio={tuple(audio_weight.shape)}, video={tuple(video_weight.shape)}, hidden_size={self.hidden_size}"
            )
        if audio_weight.shape[0] != video_weight.shape[0]:
            raise ValueError("Audio and video projector output dimensions must match")

        self.align_dim = int(audio_weight.shape[0])
        self.audio_proj = nn.Linear(self.hidden_size, self.align_dim, bias=False)
        self.video_proj = nn.Linear(self.hidden_size, self.align_dim, bias=False)
        with torch.no_grad():
            self.audio_proj.weight.copy_(audio_weight.to(dtype=self.audio_proj.weight.dtype))
            self.video_proj.weight.copy_(video_weight.to(dtype=self.video_proj.weight.dtype))
        for module in (self.audio_proj, self.video_proj):
            for param in module.parameters():
                param.requires_grad = False
        self.semantic_feature_mode = self.SEMANTIC_CHECKPOINT_LOADED

    def enable_shared_precomputed_semantic_features(self) -> None:
        """Allow cosine scoring when audio/video features are already in one semantic space."""

        self.semantic_feature_mode = self.SEMANTIC_SHARED_PRECOMPUTED

    def _semantic_similarity(
        self,
        candidate_audio: torch.Tensor,
        video: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> torch.Tensor:
        if self.semantic_feature_mode == self.SEMANTIC_DISABLED:
            return torch.zeros(candidate_audio.shape[:-1], device=candidate_audio.device, dtype=torch.float32)
        if self.semantic_feature_mode == self.SEMANTIC_SHARED_PRECOMPUTED:
            audio_key = _normalize_projected_features(candidate_audio)
            video_key = _normalize_projected_features(video).unsqueeze(2)
        elif self.semantic_feature_mode == self.SEMANTIC_CHECKPOINT_LOADED:
            if self.audio_proj is None or self.video_proj is None:
                raise ValueError("Semantic projector checkpoint was not loaded")
            audio_key = _normalize_projected_features(self.audio_proj(candidate_audio))
            video_key = _normalize_projected_features(self.video_proj(video)).unsqueeze(2)
        else:
            raise ValueError(f"Unknown semantic_feature_mode: {self.semantic_feature_mode}")
        similarity = F.cosine_similarity(audio_key, video_key, dim=-1, eps=1e-6)
        return similarity.masked_fill(~candidate_valid, 0.0)

    def _normalize_semantic_feature_mode(self, mode: str) -> str:
        normalized = str(mode or self.SEMANTIC_DISABLED).lower()
        allowed = {
            self.SEMANTIC_DISABLED,
            self.SEMANTIC_SHARED_PRECOMPUTED,
            self.SEMANTIC_CHECKPOINT_LOADED,
        }
        if normalized not in allowed:
            raise ValueError(
                "semantic_feature_mode must be one of "
                f"{sorted(allowed)}; random or untrained projectors are not allowed"
            )
        return normalized


def _ensure_audio_windows(audio_windows: torch.Tensor) -> torch.Tensor:
    windows = audio_windows
    if windows.ndim == 2:
        windows = windows.unsqueeze(0)
    if windows.ndim != 3:
        raise ValueError(f"Expected audio windows shaped [B,T,S] or [T,S], got {tuple(windows.shape)}")
    return windows


def _ensure_features(features: torch.Tensor) -> torch.Tensor:
    values = features
    if values.ndim == 2:
        values = values.unsqueeze(0)
    if values.ndim != 3:
        raise ValueError(f"Expected features shaped [B,T,H] or [T,H], got {tuple(values.shape)}")
    return torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _summarize_video(video_features: torch.Tensor) -> torch.Tensor:
    values = video_features.mean(dim=2) if video_features.ndim == 4 else video_features
    return _ensure_features(values)


def _normalize_projected_features(features: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)
    values = values - values.mean(dim=-1, keepdim=True)
    scale = values.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    return F.normalize(values / scale, dim=-1, eps=1e-6)


def _find_weight(state: dict, keys: Sequence[str]) -> torch.Tensor | None:
    for key in keys:
        value = state.get(key)
        if torch.is_tensor(value):
            return value.detach().to(dtype=torch.float32)
    for prefix in ("audio_event_aligner.", "streaming_av_module.audio_event_aligner."):
        for key in keys:
            value = state.get(f"{prefix}{key}")
            if torch.is_tensor(value):
                return value.detach().to(dtype=torch.float32)
    return None


def _temporal_feature_change(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    change = torch.zeros(features.shape[:2], device=features.device, dtype=torch.float32)
    if features.shape[1] <= 1:
        return change
    pair_valid = mask[:, 1:] & mask[:, :-1]
    numerator = (features[:, 1:] - features[:, :-1]).norm(dim=-1)
    denominator = features[:, 1:].norm(dim=-1) + features[:, :-1].norm(dim=-1) + 1e-6
    pair_change = (numerator / denominator).clamp(0.0, 1.0)
    pair_change = torch.where(pair_valid, pair_change, torch.zeros_like(pair_change))
    change[:, 1:] = torch.maximum(change[:, 1:], pair_change)
    change[:, :-1] = torch.maximum(change[:, :-1], pair_change)
    return change


def _ensure_times(
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
    return torch.nan_to_num(times, nan=0.0, posinf=0.0, neginf=0.0)


def _ensure_mask(
    mask: torch.Tensor | None,
    batch: int,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    if mask is None:
        return torch.ones(batch, steps, dtype=torch.bool, device=device)
    values = mask.to(device=device, dtype=torch.bool)
    if values.shape != (batch, steps):
        raise ValueError(f"mask shape {tuple(values.shape)} does not match {(batch, steps)}")
    return values


def _adjacent_feature_change(
    audio_features: torch.Tensor | None,
    audio_rms: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if audio_features is None:
        values = audio_rms.unsqueeze(-1)
    else:
        values = _ensure_features(audio_features).to(device=audio_rms.device)
        if values.shape[:2] != audio_rms.shape:
            raise ValueError("audio_features windows do not match audio_windows")

    pair_valid = mask[:, 1:] & mask[:, :-1]
    numerator = (values[:, 1:] - values[:, :-1]).norm(dim=-1)
    denominator = values[:, 1:].norm(dim=-1) + values[:, :-1].norm(dim=-1) + 1e-6
    pair_change = (numerator / denominator).clamp(0.0, 1.0)
    pair_change = torch.where(pair_valid, pair_change, torch.zeros_like(pair_change))
    change = torch.zeros_like(audio_rms)
    if audio_rms.shape[1] > 1:
        change[:, 1:] = torch.maximum(change[:, 1:], pair_change)
        change[:, :-1] = torch.maximum(change[:, :-1], pair_change)
    return change


def _ensure_frozen_stream(values: torch.Tensor, expected_dim: int, name: str) -> torch.Tensor:
    stream = _ensure_features(values)
    if stream.shape[-1] != expected_dim:
        raise ValueError(f"Frozen offset scorer {name} dim must be {expected_dim}, got {stream.shape[-1]}")
    if not torch.isfinite(stream).all():
        raise ValueError(f"Frozen offset scorer {name} features must be finite")
    return stream


def _ensure_frozen_scalar(
    values: torch.Tensor,
    batch: int,
    steps: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    scalar = values.to(device=device, dtype=torch.float32)
    if scalar.ndim == 1:
        scalar = scalar.unsqueeze(0)
    if scalar.shape != (batch, steps):
        raise ValueError(f"Frozen offset scorer {name} shape must be {(batch, steps)}, got {tuple(scalar.shape)}")
    if not torch.isfinite(scalar).all():
        raise ValueError(f"Frozen offset scorer {name} must be finite")
    return scalar


def _frozen_diff_norm(values: torch.Tensor) -> torch.Tensor:
    change = torch.zeros(values.shape[:2], device=values.device, dtype=torch.float32)
    if values.shape[1] <= 1:
        return change
    step = (values[:, 1:] - values[:, :-1]).norm(dim=-1)
    change[:, 1:] = torch.maximum(change[:, 1:], step)
    change[:, :-1] = torch.maximum(change[:, :-1], step)
    return change


def _frozen_context_indices(index: int, length: int, radius: int) -> list[int]:
    return [max(0, min(length - 1, index + delta)) for delta in range(-radius, radius + 1)]


def _frozen_context_with_deltas(values: torch.Tensor, index: int, radius: int) -> torch.Tensor:
    indices = _frozen_context_indices(index, values.shape[1], radius)
    context = values[:, indices]
    deltas = torch.zeros_like(context)
    if context.shape[1] > 1:
        step = context[:, 1:] - context[:, :-1]
        deltas[:, 1:] = step
        deltas[:, :-1] = deltas[:, :-1] + step
    return torch.cat([context.flatten(1), deltas.flatten(1)], dim=-1)


def _frozen_scalar_context(
    rms: torch.Tensor,
    nonsilent: torch.Tensor,
    audio_change: torch.Tensor,
    energy_change: torch.Tensor,
    clip_change: torch.Tensor,
    rgb_change: torch.Tensor,
    audio_index: int,
    video_index: int,
    radius: int,
) -> torch.Tensor:
    audio_indices = _frozen_context_indices(audio_index, rms.shape[1], radius)
    video_indices = _frozen_context_indices(video_index, rms.shape[1], radius)
    return torch.stack(
        [
            rms[:, audio_indices],
            nonsilent[:, audio_indices],
            audio_change[:, audio_indices],
            energy_change[:, audio_indices],
            clip_change[:, video_indices],
            rgb_change[:, video_indices],
        ],
        dim=-1,
    ).flatten(1)


def _frozen_local_peak_lag(values: torch.Tensor, index: int, radius: int) -> torch.Tensor:
    indices = _frozen_context_indices(index, values.shape[1], radius)
    peak = values[:, indices].argmax(dim=-1).float()
    return (peak - float(radius)) * 0.5


def _frozen_center_dominance(values: torch.Tensor, index: int) -> torch.Tensor:
    neighbors = []
    if index > 0:
        neighbors.append(values[:, index - 1])
    if index + 1 < values.shape[1]:
        neighbors.append(values[:, index + 1])
    if not neighbors:
        return torch.zeros(values.shape[0], device=values.device, dtype=torch.float32)
    return values[:, index] - torch.stack(neighbors, dim=0).mean(dim=0)


def _frozen_sync_evidence(
    audio_signal: torch.Tensor,
    video_signal: torch.Tensor,
    audio_index: int,
    video_index: int,
    radius: int,
) -> dict[str, torch.Tensor]:
    return {
        "sync_peak_delta": (
            _frozen_local_peak_lag(audio_signal, audio_index, radius)
            - _frozen_local_peak_lag(video_signal, video_index, radius)
        ),
        "center_audio_dominance": _frozen_center_dominance(audio_signal, audio_index),
        "center_video_dominance": _frozen_center_dominance(video_signal, video_index),
    }
