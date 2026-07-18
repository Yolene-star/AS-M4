"""离线视频时间窗口软加权测试。"""

from __future__ import annotations

import pytest
import torch

from intersuit.model.streaming_av.video_window_weighting import (
    apply_video_window_weighting,
    resample_window_signal,
)


def _features() -> torch.Tensor:
    return torch.arange(5, dtype=torch.float32).reshape(5, 1, 1)


def test_disabled_and_low_confidence_are_elementwise_identical():
    features = _features()
    offsets = torch.tensor([0.0, -0.5, 0.5, -0.5, 0.5])
    low_margin = torch.full((5,), 0.149)
    event = torch.ones(5)

    disabled = apply_video_window_weighting(
        features,
        offsets,
        torch.ones(5),
        event_strength=event,
        enabled=False,
    )
    low_confidence = apply_video_window_weighting(
        features,
        offsets,
        low_margin,
        event_strength=event,
    )

    assert torch.equal(disabled.features, features)
    assert torch.equal(low_confidence.features, features)
    assert torch.equal(disabled.weights[:, 1], torch.ones(5))
    assert not low_confidence.changed.any()


def test_offset_soft_uses_requested_neighbor_and_preserves_center():
    features = _features()
    offsets = torch.tensor([0.0, -0.5, 0.0, 0.5, 0.0])
    margins = torch.tensor([1.0, 0.5, 1.0, 1.0, 1.0])

    output = apply_video_window_weighting(
        features,
        offsets,
        margins,
        mode="offset_soft",
        max_neighbor_weight=0.4,
    )

    assert output.weights[1].tolist() == pytest.approx([0.2, 0.8, 0.0])
    assert output.features[1].item() == pytest.approx(0.8)
    assert output.weights[3].tolist() == pytest.approx([0.0, 0.6, 0.4])
    assert output.features[3].item() == pytest.approx(3.4)
    assert torch.all(output.weights[:, 1] >= 0.6)
    assert torch.allclose(output.weights.sum(dim=-1), torch.ones(5))


def test_event_strength_scales_adjustment_and_zero_strength_is_identity():
    features = _features()
    offsets = torch.tensor([0.0, -0.5, 0.5, 0.5, 0.0])
    margins = torch.ones(5)
    event = torch.tensor([1.0, 0.0, 0.5, 1.0, 1.0])

    output = apply_video_window_weighting(
        features,
        offsets,
        margins,
        event_strength=event,
        max_neighbor_weight=0.4,
    )

    assert torch.equal(output.features[1], features[1])
    assert output.adjustment[2].item() == pytest.approx(0.2)
    assert output.features[2].item() == pytest.approx(2.2)
    assert not output.changed[1]


def test_hard_move_and_boundaries():
    features = _features()
    offsets = torch.tensor([-0.5, -0.5, 0.0, 0.5, 0.5])
    margins = torch.ones(5)

    output = apply_video_window_weighting(
        features,
        offsets,
        margins,
        mode="hard_move",
    )

    assert torch.equal(output.features[0], features[0])
    assert output.features[1].item() == 0.0
    assert output.features[3].item() == 4.0
    assert torch.equal(output.features[4], features[4])


def test_batched_features_and_resampling():
    features = _features().unsqueeze(0).repeat(2, 1, 1, 1)
    offsets = torch.zeros(2, 5)
    margins = torch.ones(2, 5)
    output = apply_video_window_weighting(
        features,
        offsets,
        margins,
        mode="offset_soft",
    )

    assert output.features.shape == features.shape
    assert torch.equal(output.features, features)
    assert resample_window_signal(torch.tensor([0.0, 1.0, 2.0]), 5).tolist() == [
        0.0,
        0.0,
        1.0,
        2.0,
        2.0,
    ]


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="event_strength"):
        apply_video_window_weighting(
            _features(),
            torch.zeros(5),
            torch.ones(5),
            mode="offset_event_soft",
        )
    with pytest.raises(ValueError, match="finite"):
        apply_video_window_weighting(
            torch.tensor([[float("nan")]]),
            torch.zeros(1),
            torch.ones(1),
            mode="offset_soft",
        )
