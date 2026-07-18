"""冻结 offset scorer 的因果诊断稳定策略。

本模块只处理 scorer 已经输出的三候选分数，不训练参数、不移动音频窗口，
也不参与 Gate 或融合。所有策略只使用当前及历史窗口，避免离线前视泄漏。
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class OffsetStabilizerOutput(NamedTuple):
    candidate_scores: torch.Tensor
    best_offset: torch.Tensor
    margin: torch.Tensor
    accepted: torch.Tensor
    suggested_offset: torch.Tensor
    decision_delay_windows: torch.Tensor


def stabilize_offset_scores(
    candidate_scores: torch.Tensor,
    strategy: str,
    *,
    margin_threshold: float = 0.15,
    consecutive_windows: int = 2,
    hold_margin: float = 0.10,
    switch_margin: float = 0.30,
    moving_average_windows: int = 3,
) -> OffsetStabilizerOutput:
    """对 `[B,T,3]` scorer 分数应用一种因果诊断稳定策略。"""

    scores = candidate_scores.float()
    if scores.ndim != 3 or scores.shape[-1] != 3:
        raise ValueError("candidate_scores must have shape [B,T,3]")
    if not torch.isfinite(scores).all():
        raise ValueError("candidate_scores must be finite")
    if margin_threshold < 0 or hold_margin < 0 or switch_margin < 0:
        raise ValueError("margin thresholds must be non-negative")
    if consecutive_windows < 1 or moving_average_windows < 1:
        raise ValueError("window counts must be positive")
    normalized = str(strategy).strip().lower()
    if normalized not in {"none", "consecutive", "hysteresis", "moving_average"}:
        raise ValueError(f"unsupported offset stabilization strategy: {strategy}")

    if normalized == "moving_average":
        decision_scores = _causal_moving_average(scores, moving_average_windows)
    else:
        decision_scores = scores.clone()
    selection_scores = decision_scores.clone()
    if scores.shape[1]:
        selection_scores[:, 0, 0] = -1e9
        selection_scores[:, -1, 2] = -1e9
    raw_index, raw_margin = _best_and_margin(selection_scores)
    enough_candidates = torch.ones_like(raw_index, dtype=torch.bool)
    if scores.shape[1] == 1:
        enough_candidates.zero_()

    if normalized in {"none", "moving_average"}:
        accepted = raw_margin.ge(margin_threshold) & enough_candidates
        stable_index = torch.where(accepted, raw_index, torch.ones_like(raw_index))
        delays = torch.zeros_like(raw_index)
    elif normalized == "consecutive":
        stable_index, accepted, delays = _consecutive(
            raw_index,
            raw_margin,
            threshold=margin_threshold,
            windows=consecutive_windows,
        )
    else:
        stable_index, accepted, delays = _hysteresis(
            raw_index,
            raw_margin,
            entry_margin=margin_threshold,
            hold_margin=hold_margin,
            switch_margin=switch_margin,
        )
        accepted &= enough_candidates

    offsets = torch.tensor((-0.5, 0.0, 0.5), device=scores.device, dtype=torch.float32)
    best_offset = offsets[raw_index]
    suggested = offsets[stable_index]
    suggested = torch.where(accepted, suggested, torch.zeros_like(suggested))
    return OffsetStabilizerOutput(
        candidate_scores=decision_scores,
        best_offset=best_offset,
        margin=raw_margin,
        accepted=accepted,
        suggested_offset=suggested,
        decision_delay_windows=delays,
    )


def _best_and_margin(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tie_break = torch.tensor((0.5, 0.0, 0.5), device=scores.device).view(1, 1, 3) * 1e-6
    selection = scores - tie_break
    best = selection.argmax(dim=-1)
    top = selection.topk(k=2, dim=-1).values
    return best, (top[..., 0] - top[..., 1]).clamp_min(0.0)


def _causal_moving_average(scores: torch.Tensor, windows: int) -> torch.Tensor:
    output = torch.empty_like(scores)
    cumulative = scores.cumsum(dim=1)
    for index in range(scores.shape[1]):
        left = max(0, index - windows + 1)
        total = cumulative[:, index]
        if left:
            total = total - cumulative[:, left - 1]
        output[:, index] = total / float(index - left + 1)
    return output


def _consecutive(
    best: torch.Tensor,
    margin: torch.Tensor,
    *,
    threshold: float,
    windows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stable = torch.ones_like(best)
    accepted = torch.zeros_like(best, dtype=torch.bool)
    delays = torch.zeros_like(best)
    confident = margin.ge(threshold)
    for batch in range(best.shape[0]):
        current = 1
        pending = 1
        pending_length = 0
        for step in range(best.shape[1]):
            candidate = int(best[batch, step].item())
            proposed = candidate if confident[batch, step] else 1
            if proposed == current:
                pending, pending_length = current, 0
            else:
                if proposed == pending:
                    pending_length += 1
                else:
                    pending, pending_length = proposed, 1
            if pending_length >= windows:
                current = pending
                pending_length = 0
                delays[batch, step] = windows - 1
            stable[batch, step] = current
            accepted[batch, step] = current != 1 or (
                proposed == 1 and bool(confident[batch, step])
            )
    return stable, accepted, delays


def _hysteresis(
    best: torch.Tensor,
    margin: torch.Tensor,
    *,
    entry_margin: float,
    hold_margin: float,
    switch_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stable = torch.ones_like(best)
    accepted = torch.zeros_like(best, dtype=torch.bool)
    delays = torch.zeros_like(best)
    for batch in range(best.shape[0]):
        current = 1
        pending_since = 0
        for step in range(best.shape[1]):
            candidate = int(best[batch, step].item())
            confidence = float(margin[batch, step].item())
            if current == 1:
                if candidate == 1:
                    accepted[batch, step] = confidence >= entry_margin
                    pending_since = step
                elif confidence >= entry_margin:
                    current = candidate
                    accepted[batch, step] = True
                    pending_since = step
            elif candidate == current and confidence >= hold_margin:
                accepted[batch, step] = True
            elif candidate != current and confidence >= switch_margin:
                current = candidate
                accepted[batch, step] = True
                pending_since = step
            else:
                # 低置信冲突时维持上一方向，仅作为稳定诊断建议。
                accepted[batch, step] = True
            stable[batch, step] = current
            if accepted[batch, step] and current != 1:
                delays[batch, step] = max(0, step - pending_since)
    return stable, accepted, delays
