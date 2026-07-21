"""离线视频时间窗口加权。

本模块只对已经提取的视频特征做凸组合，不接入正式流式推理、Gate 或融合。
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class VideoWindowWeightingOutput(NamedTuple):
    features: torch.Tensor
    weights: torch.Tensor
    adjustment: torch.Tensor
    changed: torch.Tensor


def resample_window_signal(values: torch.Tensor, target_steps: int) -> torch.Tensor:
    """按归一化时间轴将 ``[..., T]`` 最近邻映射为 ``[..., target_steps]``。"""

    if values.ndim < 1:
        raise ValueError("values must have at least one dimension")
    if values.shape[-1] < 1 or target_steps < 1:
        raise ValueError("source and target steps must be positive")
    if values.shape[-1] == target_steps:
        return values.clone()
    positions = torch.linspace(
        0,
        values.shape[-1] - 1,
        target_steps,
        device=values.device,
        dtype=torch.float32,
    )
    indices = positions.round().long()
    return values.index_select(-1, indices)


def apply_video_window_weighting(
    features: torch.Tensor,
    best_offset: torch.Tensor,
    margin: torch.Tensor,
    *,
    event_strength: torch.Tensor | None = None,
    mode: str = "offset_event_soft",
    margin_threshold: float = 0.15,
    max_neighbor_weight: float = 0.35,
    enabled: bool = True,
) -> VideoWindowWeightingOutput:
    """对 ``[B,T,...]`` 或 ``[T,...]`` 视频特征应用离线时间邻窗加权。

    ``weights[..., 0:3]`` 分别表示前一、当前和后一窗口。软加权幅度为
    ``max_neighbor_weight * clamp(margin, 0, 1)``；联合模式再乘以有界的
    ``event_strength``。因此中心窗口始终至少保留 ``1-max_neighbor_weight``。
    """

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"baseline", "hard_move", "offset_soft", "offset_event_soft"}:
        raise ValueError(f"unsupported video window weighting mode: {mode}")
    if margin_threshold < 0:
        raise ValueError("margin_threshold must be non-negative")
    if not 0.0 <= max_neighbor_weight < 1.0:
        raise ValueError("max_neighbor_weight must be in [0,1)")
    if features.ndim < 2:
        raise ValueError("features must have shape [T,...] or [B,T,...]")
    if not torch.isfinite(features).all():
        raise ValueError("features must be finite")

    unbatched = best_offset.ndim == 1
    if unbatched:
        if features.shape[0] != best_offset.shape[0]:
            raise ValueError("unbatched features and decisions must share T")
        feature_batch = features.unsqueeze(0)
        offset_batch = best_offset.unsqueeze(0)
        margin_batch = margin.unsqueeze(0)
        event_batch = None if event_strength is None else event_strength.unsqueeze(0)
    else:
        feature_batch = features
        offset_batch = best_offset
        margin_batch = margin
        event_batch = event_strength

    batch, steps = feature_batch.shape[:2]
    if offset_batch.shape != (batch, steps) or margin_batch.shape != (batch, steps):
        raise ValueError("best_offset and margin must have shape [B,T]")
    if not torch.isfinite(offset_batch).all() or not torch.isfinite(margin_batch).all():
        raise ValueError("offset and margin must be finite")
    if event_batch is not None:
        if event_batch.shape != (batch, steps):
            raise ValueError("event_strength must have shape [B,T]")
        if not torch.isfinite(event_batch).all():
            raise ValueError("event_strength must be finite")

    weights = torch.zeros(
        batch,
        steps,
        3,
        device=feature_batch.device,
        dtype=feature_batch.dtype,
    )
    weights[..., 1] = 1.0
    adjustment = torch.zeros(
        batch,
        steps,
        device=feature_batch.device,
        dtype=feature_batch.dtype,
    )
    if not enabled or normalized_mode == "baseline":
        result = feature_batch.clone()
        changed = torch.zeros_like(adjustment, dtype=torch.bool)
        return _restore_batch(result, weights, adjustment, changed, unbatched)

    confident = margin_batch.ge(float(margin_threshold))
    previous = confident & offset_batch.eq(-0.5)
    following = confident & offset_batch.eq(0.5)
    if steps:
        previous[:, 0] = False
        following[:, -1] = False

    if normalized_mode == "hard_move":
        adjustment = (previous | following).to(feature_batch.dtype)
    else:
        confidence = margin_batch.to(feature_batch.dtype).clamp(0.0, 1.0)
        if normalized_mode == "offset_event_soft":
            if event_batch is None:
                raise ValueError("offset_event_soft requires event_strength")
            confidence = confidence * event_batch.to(feature_batch.dtype).clamp(0.0, 1.0)
        adjustment = float(max_neighbor_weight) * confidence
        adjustment = adjustment * (previous | following).to(feature_batch.dtype)

    weights[..., 1] = 1.0 - adjustment
    weights[..., 0] = torch.where(previous, adjustment, torch.zeros_like(adjustment))
    weights[..., 2] = torch.where(following, adjustment, torch.zeros_like(adjustment))

    before = torch.cat((feature_batch[:, :1], feature_batch[:, :-1]), dim=1)
    after = torch.cat((feature_batch[:, 1:], feature_batch[:, -1:]), dim=1)
    expand = (1,) * (feature_batch.ndim - 2)
    result = (
        before * weights[..., 0].reshape(batch, steps, *expand)
        + feature_batch * weights[..., 1].reshape(batch, steps, *expand)
        + after * weights[..., 2].reshape(batch, steps, *expand)
    )
    if not torch.isfinite(result).all():
        raise ValueError("weighted features contain NaN/Inf")
    changed = adjustment.gt(0)
    return _restore_batch(result, weights, adjustment, changed, unbatched)


def _restore_batch(
    features: torch.Tensor,
    weights: torch.Tensor,
    adjustment: torch.Tensor,
    changed: torch.Tensor,
    unbatched: bool,
) -> VideoWindowWeightingOutput:
    if unbatched:
        return VideoWindowWeightingOutput(
            features[0],
            weights[0],
            adjustment[0],
            changed[0],
        )
    return VideoWindowWeightingOutput(features, weights, adjustment, changed)
