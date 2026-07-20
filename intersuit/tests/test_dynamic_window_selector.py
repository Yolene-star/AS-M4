"""CPU tests for AS-M4 Dynamic Window Selector V1."""

from __future__ import annotations

import torch

from intersuit.model.streaming_av.dynamic_window_selector import DynamicWindowSelector


def _inputs(steps: int = 8, dim: int = 4):
    features = torch.zeros(1, steps, dim)
    timestamps = torch.tensor(
        [[index * 0.5, index * 0.5 + 1.0] for index in range(steps)],
        dtype=torch.float32,
    )
    windows = torch.zeros(1, steps, 160)
    mask = torch.ones(1, steps, dtype=torch.bool)
    return features, timestamps, windows, mask


def test_silence_produces_no_selected_windows():
    features, timestamps, windows, mask = _inputs()
    selector = DynamicWindowSelector(input_dim=4, top_k=4)

    output = selector(features, timestamps, mask, windows)

    assert not output.selection_mask.any()
    assert torch.all(output.selected_features == 0)
    assert torch.isfinite(output.dynamic_thresholds).all()


def test_burst_event_is_selected_and_source_weights_are_normalized():
    features, timestamps, windows, mask = _inputs()
    features[0, 3] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    windows[0, 3] = 0.2
    selector = DynamicWindowSelector(input_dim=4, top_k=4, nms_iou=1.0)

    output = selector(features, timestamps, mask, windows)

    assert output.selection_mask.any()
    valid_weights = output.source_weights[output.selection_mask]
    assert torch.allclose(valid_weights.sum(dim=-1), torch.ones(valid_weights.shape[0]), atol=1e-5)
    selected_starts = output.selected_timestamps[0, output.selection_mask[0], 0]
    assert torch.all(selected_starts[1:] >= selected_starts[:-1])


def test_candidates_are_causal_and_limited_to_top_k():
    features, timestamps, windows, mask = _inputs(steps=12)
    features[:, 2:8] = 0.5
    windows[:, 2:8] = 0.1
    selector = DynamicWindowSelector(input_dim=4, top_k=3, nms_iou=0.0)

    output = selector(features, timestamps, mask, windows)

    assert output.selection_mask.shape == (1, 3)
    assert int(output.selection_mask.sum()) <= 3
    valid_times = output.selected_timestamps[output.selection_mask]
    assert torch.all(valid_times[:, 1] >= valid_times[:, 0])
    assert torch.all(valid_times[:, 1] <= timestamps[..., 1].max() + 1e-6)


def test_padding_mask_is_never_selected():
    features, timestamps, windows, mask = _inputs()
    mask[:, 5:] = False
    features[0, 5:] = 10.0
    windows[0, 5:] = 1.0
    selector = DynamicWindowSelector(input_dim=4, top_k=8, nms_iou=1.0)

    output = selector(features, timestamps, mask, windows)

    assert torch.all(output.source_weights[:, :, 5:] == 0)
    if output.selection_mask.any():
        valid_times = output.selected_timestamps[output.selection_mask]
        assert torch.all(valid_times[:, 1] <= timestamps[0, 4, 1] + 1e-6)


def test_causal_state_carries_boundary_feature_and_rms():
    features, timestamps, windows, mask = _inputs(steps=4)
    features[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    windows[0, 0] = 0.2
    selector = DynamicWindowSelector(input_dim=4, top_k=4, nms_iou=1.0)

    first = selector(features[:, :2], timestamps[:2], mask[:, :2], windows[:, :2])
    second_features = features[:, 2:]
    second_windows = windows[:, 2:]
    second = selector(
        second_features,
        timestamps[2:],
        mask[:, 2:],
        second_windows,
        state=first.state,
    )

    assert second.state.last_feature.shape == (1, 4)
    assert second.state.last_rms.shape == (1,)
    assert torch.isfinite(second.micro_scores).all()
