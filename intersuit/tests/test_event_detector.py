"""CPU harness for the AS-M4 audio event detector."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "intersuit" / "model" / "streaming_av" / "event_detector.py"
SPEC = importlib.util.spec_from_file_location("as_m4_event_detector", MODULE_PATH)
event_detector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = event_detector
SPEC.loader.exec_module(event_detector)

AudioEventDetector = event_detector.AudioEventDetector
compute_event_loss = event_detector.compute_event_loss


def test_event_detector_output_shapes():
    detector = AudioEventDetector(input_dim=8, num_events=5, hidden_dim=16)
    features = torch.randn(2, 4, 8)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)

    output = detector(features, mask=mask)

    assert output.event_logits.shape == (2, 4, 5)
    assert output.eventness.shape == (2, 4)
    assert output.boundary_logits.shape == (2, 4, 2)
    assert output.mask.equal(mask)
    assert torch.all(output.eventness[~mask] == 0)


def test_masked_loss_ignores_invalid_windows():
    detector = AudioEventDetector(input_dim=4, num_events=3, hidden_dim=8)
    features = torch.randn(1, 3, 4)
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    output = detector(features, mask=mask)

    event_labels = torch.tensor([[0, 1, 2]])
    eventness_labels = torch.tensor([[1.0, 0.0, 1.0]])
    boundary_labels = torch.tensor([[0, 1, 1]])
    loss = compute_event_loss(output, event_labels, eventness_labels, boundary_labels)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_eventness_sorts_high_energy_window_above_silence():
    detector = AudioEventDetector(
        input_dim=4,
        num_events=2,
        hidden_dim=8,
        eventness_prior_scale=5.0,
    )
    features = torch.zeros(1, 3, 4)
    features[0, 1] = 10.0

    output = detector(features)

    assert output.eventness[0, 1] > output.eventness[0, 0]
    assert output.eventness[0, 1] > output.eventness[0, 2]


def test_event_detector_rejects_wrong_shape():
    detector = AudioEventDetector(input_dim=4, num_events=2)
    try:
        detector(torch.randn(3, 4))
    except ValueError:
        return
    raise AssertionError("Expected ValueError for missing batch dimension")


if __name__ == "__main__":
    test_event_detector_output_shapes()
    test_masked_loss_ignores_invalid_windows()
    test_eventness_sorts_high_energy_window_above_silence()
    test_event_detector_rejects_wrong_shape()
    print("event_detector harness passed")

