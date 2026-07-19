"""Scene-audio encoder interfaces for AS-M4."""

from .builder import build_scene_audio_encoder
from .scene_audio_encoder import (
    DummySceneAudioEncoder,
    FrozenBEATsSceneAudioEncoder,
    PrecomputedSceneAudioEncoder,
    SceneAudioEncoderOutput,
)

__all__ = [
    "build_scene_audio_encoder",
    "DummySceneAudioEncoder",
    "FrozenBEATsSceneAudioEncoder",
    "PrecomputedSceneAudioEncoder",
    "SceneAudioEncoderOutput",
]
