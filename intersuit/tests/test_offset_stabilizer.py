"""冻结 offset scorer 因果稳定策略测试。"""

from __future__ import annotations

import pytest
import torch

from intersuit.model.streaming_av.offset_stabilizer import stabilize_offset_scores


def _scores(best_slots: list[int], margins: list[float]) -> torch.Tensor:
    values = torch.zeros(1, len(best_slots), 3)
    for step, (slot, margin) in enumerate(zip(best_slots, margins)):
        values[0, step, slot] = margin
    return values


def test_consecutive_requires_two_matching_nonzero_windows():
    scores = _scores([1, 2, 1, 0, 0, 2], [0.3] * 6)

    output = stabilize_offset_scores(scores, "consecutive", consecutive_windows=2)

    assert output.suggested_offset.tolist() == [[0.0, 0.0, 0.0, 0.0, -0.5, -0.5]]
    assert output.accepted.tolist() == [[True, False, True, False, True, True]]
    assert output.decision_delay_windows[0, 4].item() == 1


def test_hysteresis_uses_lower_hold_and_higher_switch_margin():
    scores = _scores([2, 2, 0, 0, 0], [0.2, 0.11, 0.2, 0.29, 0.31])

    output = stabilize_offset_scores(
        scores,
        "hysteresis",
        hold_margin=0.1,
        switch_margin=0.3,
    )

    assert output.suggested_offset.tolist() == [[0.5, 0.5, 0.5, 0.5, -0.5]]


def test_moving_average_is_causal_and_removes_single_window_spike():
    scores = torch.tensor(
        [[[0.0, 0.4, 0.0], [0.0, 0.0, 0.8], [0.0, 0.4, 0.0]]],
        dtype=torch.float32,
    )

    output = stabilize_offset_scores(
        scores,
        "moving_average",
        moving_average_windows=3,
        margin_threshold=0.15,
    )

    assert torch.allclose(output.candidate_scores[0, 1], torch.tensor([0.0, 0.2, 0.4]))
    assert output.suggested_offset[0, 1].item() == pytest.approx(0.5)
    assert output.suggested_offset[0, 2].item() == pytest.approx(0.0)


def test_none_matches_frozen_margin_policy():
    scores = _scores([1, 0, 1, 2, 1], [0.2, 0.2, 0.1, 0.3, 0.2])

    output = stabilize_offset_scores(scores, "none", margin_threshold=0.15)

    assert output.accepted.tolist() == [[True, True, False, True, True]]
    assert output.suggested_offset.tolist() == [[0.0, -0.5, 0.0, 0.5, 0.0]]


def test_invalid_scores_and_strategy_are_rejected():
    with pytest.raises(ValueError, match="shape"):
        stabilize_offset_scores(torch.zeros(2, 3), "none")
    with pytest.raises(ValueError, match="unsupported"):
        stabilize_offset_scores(torch.zeros(1, 2, 3), "future_vote")
