"""可学习的因果 offset 时序诊断模块。

该模块只生成诊断建议，不移动音频窗口、不参与 Gate，也不修改融合输出。
调用方显式持有并重置 GRU 状态，避免不同视频之间发生状态泄漏。
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import torch
from torch import nn


class TemporalOffsetGRUOutput(NamedTuple):
    offset_logits: torch.Tensor
    synchronizability_logits: torch.Tensor
    synchronizability_prob: torch.Tensor
    predicted_offset: torch.Tensor
    accepted: torch.Tensor
    suggested_offset: torch.Tensor
    state: torch.Tensor


class TemporalOffsetGRUDiagnostic(nn.Module):
    """从冻结 scorer 证据和历史状态预测稳定的三分类 offset。"""

    def __init__(
        self,
        candidate_feature_dim: int = 128,
        evidence_dim: int = 8,
        hidden_size: int = 128,
        candidate_projection_dim: int = 32,
        synchronizability_threshold: float = 0.5,
        frozen_margin_threshold: float = 0.15,
        evidence_mean: torch.Tensor | None = None,
        evidence_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if candidate_feature_dim <= 0 or evidence_dim < 0 or hidden_size <= 0:
            raise ValueError("feature dimensions and hidden_size must be valid")
        if candidate_projection_dim <= 0:
            raise ValueError("candidate_projection_dim must be positive")
        if not 0.0 <= synchronizability_threshold <= 1.0:
            raise ValueError("synchronizability_threshold must be in [0, 1]")
        if frozen_margin_threshold < 0:
            raise ValueError("frozen_margin_threshold must be non-negative")

        self.candidate_feature_dim = int(candidate_feature_dim)
        self.evidence_dim = int(evidence_dim)
        self.hidden_size = int(hidden_size)
        self.candidate_projection_dim = int(candidate_projection_dim)
        self.synchronizability_threshold = float(synchronizability_threshold)
        self.frozen_margin_threshold = float(frozen_margin_threshold)

        self.candidate_projection = nn.Sequential(
            nn.Linear(self.candidate_feature_dim, self.candidate_projection_dim),
            nn.GELU(),
        )
        token_dim = 3 + 3 + 2 + 3 * self.candidate_projection_dim + self.evidence_dim
        self.input_projection = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, self.hidden_size),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.offset_head = nn.Linear(self.hidden_size, 3)
        self.synchronizability_head = nn.Linear(self.hidden_size, 1)
        self.register_buffer(
            "candidate_offsets",
            torch.tensor((-0.5, 0.0, 0.5), dtype=torch.float32),
            persistent=True,
        )
        mean = (
            torch.zeros(self.evidence_dim)
            if evidence_mean is None
            else torch.as_tensor(evidence_mean).float().reshape(-1)
        )
        std = (
            torch.ones(self.evidence_dim)
            if evidence_std is None
            else torch.as_tensor(evidence_std).float().reshape(-1)
        )
        if mean.numel() != self.evidence_dim or std.numel() != self.evidence_dim:
            raise ValueError("evidence normalization does not match evidence_dim")
        self.register_buffer("evidence_mean", mean, persistent=True)
        self.register_buffer("evidence_std", std.clamp_min(1e-6), persistent=True)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        reference = next(self.parameters())
        return torch.zeros(
            1,
            batch_size,
            self.hidden_size,
            device=device if device is not None else reference.device,
            dtype=dtype if dtype is not None else reference.dtype,
        )

    def reset_state(
        self,
        state: torch.Tensor,
        reset_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state = self._validate_state(state)
        if reset_mask is None:
            return torch.zeros_like(state)
        mask = torch.as_tensor(reset_mask, device=state.device, dtype=torch.bool)
        if mask.ndim != 1 or mask.numel() != state.shape[1]:
            raise ValueError("reset_mask must have shape [batch]")
        return state.masked_fill(mask.view(1, -1, 1), 0.0)

    def forward_step(
        self,
        candidate_logits: torch.Tensor,
        candidate_features: torch.Tensor,
        evidence_features: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> TemporalOffsetGRUOutput:
        if candidate_logits.ndim != 2:
            raise ValueError("forward_step candidate_logits must have shape [B,3]")
        batch = candidate_logits.shape[0]
        if state is None:
            state = self.initial_state(
                batch,
                device=candidate_logits.device,
                dtype=candidate_logits.dtype,
            )
        output = self.forward(
            candidate_logits.unsqueeze(1),
            candidate_features.unsqueeze(1),
            evidence_features.unsqueeze(1),
            state=state,
        )
        return TemporalOffsetGRUOutput(
            offset_logits=output.offset_logits[:, 0],
            synchronizability_logits=output.synchronizability_logits[:, 0],
            synchronizability_prob=output.synchronizability_prob[:, 0],
            predicted_offset=output.predicted_offset[:, 0],
            accepted=output.accepted[:, 0],
            suggested_offset=output.suggested_offset[:, 0],
            state=output.state,
        )

    def forward(
        self,
        candidate_logits: torch.Tensor,
        candidate_features: torch.Tensor,
        evidence_features: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> TemporalOffsetGRUOutput:
        self._validate_inputs(candidate_logits, candidate_features, evidence_features)
        batch = candidate_logits.shape[0]
        if state is None:
            state = self.initial_state(
                batch,
                device=candidate_logits.device,
                dtype=candidate_logits.dtype,
            )
        else:
            state = self._validate_state(state)
            if state.shape[1] != batch:
                raise ValueError("state batch dimension does not match inputs")

        token = self._build_token(
            candidate_logits.float(),
            candidate_features.float(),
            evidence_features.float(),
        )
        recurrent, next_state = self.gru(self.input_projection(token), state.float())
        offset_logits = self.offset_head(recurrent)
        sync_logits = self.synchronizability_head(recurrent).squeeze(-1)
        sync_prob = torch.sigmoid(sync_logits)
        predicted_index = offset_logits.argmax(dim=-1)
        offsets = self.candidate_offsets.to(offset_logits.device)
        predicted_offset = offsets[predicted_index]

        sorted_frozen = candidate_logits.float().topk(k=2, dim=-1).values
        frozen_margin = (sorted_frozen[..., 0] - sorted_frozen[..., 1]).clamp_min(0.0)
        syncable = sync_prob.ge(self.synchronizability_threshold)
        nonzero = predicted_index.ne(1)
        accepted = syncable & nonzero & frozen_margin.ge(self.frozen_margin_threshold)
        suggested = torch.where(accepted, predicted_offset, torch.zeros_like(predicted_offset))
        return TemporalOffsetGRUOutput(
            offset_logits=offset_logits,
            synchronizability_logits=sync_logits,
            synchronizability_prob=sync_prob,
            predicted_offset=predicted_offset,
            accepted=accepted,
            suggested_offset=suggested,
            state=next_state,
        )

    def _build_token(
        self,
        candidate_logits: torch.Tensor,
        candidate_features: torch.Tensor,
        evidence_features: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.softmax(candidate_logits, dim=-1)
        top = candidate_logits.topk(k=2, dim=-1).values
        margin = (top[..., 0] - top[..., 1]).unsqueeze(-1)
        entropy = -(
            probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1, keepdim=True)
        projected = self.candidate_projection(candidate_features).flatten(start_dim=-2)
        evidence_features = (
            evidence_features - self.evidence_mean.to(evidence_features.device)
        ) / self.evidence_std.to(evidence_features.device)
        return torch.cat(
            [
                candidate_logits,
                probabilities,
                margin,
                entropy,
                projected,
                evidence_features,
            ],
            dim=-1,
        )

    def _validate_inputs(
        self,
        candidate_logits: torch.Tensor,
        candidate_features: torch.Tensor,
        evidence_features: torch.Tensor,
    ) -> None:
        if candidate_logits.ndim != 3 or candidate_logits.shape[-1] != 3:
            raise ValueError("candidate_logits must have shape [B,T,3]")
        expected = (*candidate_logits.shape, self.candidate_feature_dim)
        if tuple(candidate_features.shape) != expected:
            raise ValueError(
                "candidate_features must have shape "
                f"[B,T,3,{self.candidate_feature_dim}]"
            )
        if tuple(evidence_features.shape) != (
            candidate_logits.shape[0],
            candidate_logits.shape[1],
            self.evidence_dim,
        ):
            raise ValueError(
                f"evidence_features must have shape [B,T,{self.evidence_dim}]"
            )
        if not (
            torch.isfinite(candidate_logits).all()
            and torch.isfinite(candidate_features).all()
            and torch.isfinite(evidence_features).all()
        ):
            raise ValueError("temporal diagnostic inputs must be finite")

    def _validate_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3 or state.shape[0] != 1 or state.shape[2] != self.hidden_size:
            raise ValueError(f"state must have shape [1,B,{self.hidden_size}]")
        if not torch.isfinite(state).all():
            raise ValueError("state must be finite")
        return state


def build_temporal_offset_evidence(
    audio_features: torch.Tensor,
    clip_features: torch.Tensor,
    rgb_features: torch.Tensor,
    audio_rms: torch.Tensor,
    non_silent_ratio: torch.Tensor,
) -> torch.Tensor:
    """构造与 GRU 训练一致的八维逐窗口因果诊断证据。"""

    if not (
        audio_features.shape[:-1] == clip_features.shape[:-1] == rgb_features.shape[:-1]
    ):
        raise ValueError("audio/CLIP/RGB evidence streams must be aligned")
    if tuple(audio_rms.shape) != tuple(audio_features.shape[:-1]):
        raise ValueError("audio_rms shape does not match feature streams")
    if tuple(non_silent_ratio.shape) != tuple(audio_features.shape[:-1]):
        raise ValueError("non_silent_ratio shape does not match feature streams")

    audio_change = _sequence_diff_norm(audio_features)
    clip_change = _sequence_diff_norm(clip_features)
    rgb_change = _sequence_diff_norm(rgb_features)
    energy_change = _sequence_diff_norm(audio_rms.unsqueeze(-1))
    visual_change = clip_change + rgb_change
    av_difference = (audio_change + energy_change - visual_change).abs()
    event_strength = (
        0.5 * (audio_rms.float() / 0.05).clamp(0.0, 1.0)
        + 0.3 * non_silent_ratio.float().clamp(0.0, 1.0)
        + 0.2 * torch.log1p(audio_change)
    )
    boundary_change = torch.zeros_like(event_strength)
    if event_strength.shape[-1] > 1:
        boundary_change[..., 1:] = (
            event_strength[..., 1:] - event_strength[..., :-1]
        ).abs()
    evidence = torch.stack(
        [
            audio_rms.float(),
            non_silent_ratio.float(),
            torch.log1p(audio_change),
            torch.log1p(clip_change),
            torch.log1p(rgb_change),
            torch.log1p(energy_change),
            torch.log1p(av_difference),
            boundary_change,
        ],
        dim=-1,
    )
    return torch.nan_to_num(evidence, nan=0.0, posinf=0.0, neginf=0.0)


def _sequence_diff_norm(values: torch.Tensor) -> torch.Tensor:
    output = torch.zeros(values.shape[:-1], device=values.device, dtype=torch.float32)
    if values.shape[-2] > 1:
        step = (values[..., 1:, :].float() - values[..., :-1, :].float()).norm(dim=-1)
        output[..., 1:] = torch.maximum(output[..., 1:], step)
        output[..., :-1] = torch.maximum(output[..., :-1], step)
    return output


def ordered_offset_emd_loss(
    offset_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """三类有序 offset 的一阶 Wasserstein/EMD 损失。"""

    if offset_logits.shape[-1] != 3:
        raise ValueError("offset_logits must end in three ordered classes")
    target = torch.nn.functional.one_hot(targets.long(), num_classes=3).to(
        dtype=offset_logits.dtype
    )
    probabilities = torch.softmax(offset_logits, dim=-1)
    distance = (
        probabilities.cumsum(dim=-1)[..., :-1]
        - target.cumsum(dim=-1)[..., :-1]
    ).abs().mean(dim=-1)
    if reduction == "none":
        return distance
    if reduction == "sum":
        return distance.sum()
    if reduction != "mean":
        raise ValueError(f"unsupported reduction: {reduction}")
    return distance.mean()


def load_temporal_offset_gru_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    freeze: bool = True,
) -> TemporalOffsetGRUDiagnostic:
    """从独立实验 checkpoint 恢复诊断模块。"""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Temporal offset GRU checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if int(payload.get("format_version", -1)) != 1:
        raise ValueError("Unsupported temporal offset GRU checkpoint format")
    metadata = payload.get("metadata")
    state = payload.get("state_dict")
    if not isinstance(metadata, dict) or not isinstance(state, dict):
        raise ValueError("Temporal offset GRU checkpoint is missing metadata/state_dict")
    model = TemporalOffsetGRUDiagnostic(
        candidate_feature_dim=int(metadata.get("candidate_feature_dim", 128)),
        evidence_dim=int(metadata.get("evidence_dim", 8)),
        hidden_size=int(metadata.get("hidden_size", 128)),
        candidate_projection_dim=int(metadata.get("candidate_projection_dim", 32)),
        synchronizability_threshold=float(
            metadata.get("synchronizability_threshold", 0.5)
        ),
        frozen_margin_threshold=float(metadata.get("frozen_margin_threshold", 0.15)),
    )
    model.load_state_dict(state, strict=True)
    if freeze:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
    return model
