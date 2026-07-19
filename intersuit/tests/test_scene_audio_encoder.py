"""CPU harness for dependency-free AS-M4 scene-audio encoders."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "intersuit" / "model" / "scene_audio_encoder" / "scene_audio_encoder.py"
SPEC = importlib.util.spec_from_file_location("as_m4_scene_audio_encoder", MODULE_PATH)
scene_audio_encoder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scene_audio_encoder
SPEC.loader.exec_module(scene_audio_encoder)

DummySceneAudioEncoder = scene_audio_encoder.DummySceneAudioEncoder
FrozenTorchaudioSceneAudioEncoder = scene_audio_encoder.FrozenTorchaudioSceneAudioEncoder
FrozenBEATsSceneAudioEncoder = scene_audio_encoder.FrozenBEATsSceneAudioEncoder
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
    assert output.feature_kind == "dummy_waveform_statistics"


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
    timestamps = torch.tensor([[0.0, 0.5], [0.5, 1.0], [1.0, 1.5]])

    output = encoder(features, timestamps=timestamps)

    assert output.features.shape == (2, 3, 6)
    assert torch.allclose(output.features[..., :4], features)
    assert output.mask.shape == (2, 3)
    assert output.timestamps.shape == (2, 3, 2)
    assert output.feature_kind == "precomputed_audio_features"


def test_precomputed_encoder_marks_shared_semantic_space():
    encoder = PrecomputedSceneAudioEncoder(hidden_size=4, shared_semantic_space=True)
    output = encoder(torch.randn(1, 2, 4))

    assert output.feature_kind == "shared_precomputed_semantic"


def test_precomputed_projection_is_frozen():
    encoder = PrecomputedSceneAudioEncoder(hidden_size=6, input_dim=4)
    features = torch.randn(2, 3, 4)

    output = encoder(features)

    assert output.features.shape == (2, 3, 6)
    assert all(not param.requires_grad for param in encoder.parameters())


def test_precomputed_encoder_rejects_bad_timestamps_and_nonfinite_features():
    encoder = PrecomputedSceneAudioEncoder(hidden_size=4)
    features = torch.randn(1, 2, 4)

    with pytest.raises(ValueError, match="timestamps shape"):
        encoder(features, timestamps=torch.zeros(1, 3, 2))
    with pytest.raises(ValueError, match="start <= end"):
        encoder(features, timestamps=torch.tensor([[[0.5, 0.0], [0.5, 1.0]]]))
    with pytest.raises(ValueError, match="must be finite"):
        encoder(features, timestamps=torch.tensor([[[0.0, float("nan")], [0.5, 1.0]]]))
    bad_features = features.clone()
    bad_features[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="features must be finite"):
        encoder(bad_features)


def test_frozen_torchaudio_encoder_rejects_unknown_bundle():
    try:
        FrozenTorchaudioSceneAudioEncoder(hidden_size=4, bundle_name="DOES_NOT_EXIST")
    except ValueError as exc:
        assert "Unknown torchaudio pipeline bundle" in str(exc)
    else:
        raise AssertionError("unknown torchaudio bundle should fail")


def test_frozen_torchaudio_encoder_requires_cached_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))

    with pytest.raises(FileNotFoundError, match="Automatic downloads are disabled"):
        FrozenTorchaudioSceneAudioEncoder(hidden_size=4, bundle_name="WAV2VEC2_BASE")


def test_frozen_torchaudio_encoder_exposes_speech_acoustic_baseline_label():
    assert "speech acoustic baseline" in (FrozenTorchaudioSceneAudioEncoder.__doc__ or "")


def test_frozen_beats_encoder_requires_local_checkpoint_and_source(tmp_path):
    with pytest.raises(FileNotFoundError, match="Automatic downloads are disabled"):
        FrozenBEATsSceneAudioEncoder(
            hidden_size=4,
            checkpoint_path=str(tmp_path / "missing.pt"),
            code_root=str(tmp_path / "missing_source"),
        )


if __name__ == "__main__":
    test_dummy_encoder_outputs_shape_and_mask()
    test_dummy_encoder_handles_silence_without_nan()
    test_dummy_encoder_has_no_trainable_parameters()
    test_precomputed_encoder_preserves_or_expands_features()
    test_precomputed_encoder_marks_shared_semantic_space()
    test_precomputed_projection_is_frozen()
    test_precomputed_encoder_rejects_bad_timestamps_and_nonfinite_features()
    test_frozen_torchaudio_encoder_rejects_unknown_bundle()
    test_frozen_beats_encoder_requires_local_checkpoint_and_source(Path("/tmp"))
    print("scene_audio_encoder harness passed")
