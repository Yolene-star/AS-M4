"""Factory for AS-M4 scene-audio encoders."""

from __future__ import annotations

from .scene_audio_encoder import DummySceneAudioEncoder, PrecomputedSceneAudioEncoder


def build_scene_audio_encoder(config):
    """Build a scene-audio encoder from config.

    Supported first-pass backends intentionally avoid external model downloads:

    - ``dummy``: deterministic waveform statistics, useful for harnesses.
    - ``precomputed``: accepts already extracted features.
    """

    encoder_type = getattr(config, "scene_audio_encoder_type", "dummy")
    hidden_size = int(getattr(config, "scene_audio_hidden_size", 768))
    if encoder_type is None:
        encoder_type = "dummy"
    encoder_type = encoder_type.lower()

    if encoder_type == "dummy":
        return DummySceneAudioEncoder(hidden_size=hidden_size)
    if encoder_type == "precomputed":
        input_dim = getattr(config, "scene_audio_precomputed_dim", None)
        return PrecomputedSceneAudioEncoder(hidden_size=hidden_size, input_dim=input_dim)

    raise ValueError(f"Unknown scene audio encoder: {encoder_type}")

