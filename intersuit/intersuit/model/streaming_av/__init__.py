"""Streaming audio-video modules for AS-M4."""

from .builder import StreamingAVModule, build_streaming_av_module
from .confidence_gate import (
    AudioConfidenceGate,
    AudioConfidenceGateOutput,
    AudioSignalFeatures,
    compute_audio_signal_features,
)
from .event_detector import AudioEventDetector, AudioEventDetectorOutput
from .fusion import GatedAVFusion
from .temporal_aligner import CausalTemporalAligner, TemporalAlignerOutput

__all__ = [
    "StreamingAVModule",
    "build_streaming_av_module",
    "AudioConfidenceGate",
    "AudioConfidenceGateOutput",
    "AudioEventDetector",
    "AudioEventDetectorOutput",
    "GatedAVFusion",
    "CausalTemporalAligner",
    "TemporalAlignerOutput",
]
