"""CPU harness for dependency-free AS-M4 scene-audio encoders."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "intersuit" / "model" / "scene_audio_encoder" / "scene_audio_encoder.py"
SPEC = importlib.util.spec_from_file_location("as_m4_scene_audio_encoder", MODULE_PATH)
scene_audio_encoder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scene_audio_encoder
SPEC.loader.exec_module(scene_audio_encoder)

DummySceneAudioEncoder = scene_audio_encoder.DummySceneAudioEncoder
PrecomputedSceneAudioEncoder = scene_audio_encoder.PrecomputedSceneAudioEncoder


def test_dummy_encoder_outputs_shape_and_mask():
    encoder = DummySceneAudioEncoder(hidden_size=16)
    audio = torch.randn(2, 3, 8)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)

    output = encoder(audio, sample_mask=mask)

    assert output.features.shape == (2, 3, 16)
    assert output.mask.equal(mask)
    assert torch.all(output.features[0, 2] == 0)
    assert torch.all(output.features[1, 1:] == 0)


def test_dummy_encoder_handles_silence_without_nan():
    encoder = DummySceneAudioEncoder(hidden_size=8)
    output = encoder(torch.zeros(1, 2, 16))

    assert output.features.shape == (1, 2, 8)
    assert not torch.isnan(output.features).any()


def test_dummy_encoder_has_no_trainable_parameters():
    encoder = DummySceneAudioEncoder(hidden_size=8)

    assert list(encoder.parameters()) == []


def test_precomputed_encoder_preserves_or_expands_features():
    encoder = PrecomputedSceneAudioEncoder(hidden_size=6)
    features = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)

    output = encoder(features)

    assert output.features.shape == (2, 3, 6)
    assert torch.allclose(output.features[..., :4], features)
    assert output.mask.shape == (2, 3)


def test_precomputed_projection_is_frozen():
    encoder = PrecomputedSceneAudioEncoder(hidden_size=6, input_dim=4)
    features = torch.randn(2, 3, 4)

    output = encoder(features)

    assert output.features.shape == (2, 3, 6)
    assert all(not param.requires_grad for param in encoder.parameters())


if __name__ == "__main__":
    test_dummy_encoder_outputs_shape_and_mask()
    test_dummy_encoder_handles_silence_without_nan()
    test_dummy_encoder_has_no_trainable_parameters()
    test_precomputed_encoder_preserves_or_expands_features()
    test_precomputed_projection_is_frozen()
    print("scene_audio_encoder harness passed")

