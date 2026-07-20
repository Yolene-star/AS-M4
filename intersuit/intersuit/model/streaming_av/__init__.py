"""Streaming audio-video modules for AS-M4."""

from .builder import StreamingAVModule, build_streaming_av_module
from .audio_event_aligner import (
    AudioEventFeatures,
    FrozenOffsetScorerInputs,
    FrozenOffsetScorerOutput,
    FrozenTemporalOffsetScorer,
    LocalAudioAlignmentOutput,
    LocalAudioEventAligner,
    compute_audio_event_features,
)
from .offset_stabilizer import OffsetStabilizerOutput, stabilize_offset_scores
from .temporal_offset_gru import (
    TemporalOffsetGRUDiagnostic,
    TemporalOffsetGRUOutput,
    build_temporal_offset_evidence,
    load_temporal_offset_gru_checkpoint,
    ordered_offset_emd_loss,
)
from .confidence_gate import (
    AudioConfidenceGate,
    AudioConfidenceGateOutput,
    AudioSignalFeatures,
    compute_audio_signal_features,
)
from .event_detector import AudioEventDetector, AudioEventDetectorOutput
from .dynamic_window_selector import (
    DynamicWindowSelector,
    DynamicWindowSelectorOutput,
    SelectorState,
)
from .fusion import GatedAVFusion
from .temporal_aligner import CausalTemporalAligner, TemporalAlignerOutput

__all__ = [
    "StreamingAVModule",
    "build_streaming_av_module",
    "AudioConfidenceGate",
    "AudioConfidenceGateOutput",
    "AudioEventDetector",
    "AudioEventDetectorOutput",
    "DynamicWindowSelector",
    "DynamicWindowSelectorOutput",
    "SelectorState",
    "AudioEventFeatures",
    "FrozenOffsetScorerInputs",
    "FrozenOffsetScorerOutput",
    "FrozenTemporalOffsetScorer",
    "OffsetStabilizerOutput",
    "stabilize_offset_scores",
    "TemporalOffsetGRUDiagnostic",
    "TemporalOffsetGRUOutput",
    "build_temporal_offset_evidence",
    "load_temporal_offset_gru_checkpoint",
    "ordered_offset_emd_loss",
    "LocalAudioAlignmentOutput",
    "LocalAudioEventAligner",
    "compute_audio_event_features",
    "GatedAVFusion",
    "CausalTemporalAligner",
    "TemporalAlignerOutput",
]
