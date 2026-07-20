"""Builder for AS-M4 streaming audio-video modules."""

from __future__ import annotations

from torch import nn

from .audio_event_aligner import LocalAudioEventAligner
from .confidence_gate import AudioConfidenceGate
from .event_detector import AudioEventDetector
from .fusion import GatedAVFusion
from .temporal_aligner import CausalTemporalAligner


class StreamingAVModule(nn.Module):
    """Container for AS-M4 lightweight streaming AV modules."""

    def __init__(self, config) -> None:
        super().__init__()
        hidden_size = int(getattr(config, "hidden_size"))
        num_events = int(getattr(config, "num_audio_events", 25))
        quality_dim = int(getattr(config, "audio_quality_dim", 1))
        align_dim = int(getattr(config, "streaming_av_align_dim", hidden_size))
        max_offset_sec = float(getattr(config, "max_av_offset_sec", 1.5))
        gate_logit_bias = float(getattr(config, "as_m4_gate_logit_bias", -5.0))
        enable_gate_v1 = bool(getattr(config, "enable_audio_confidence_gate_v1", False))
        silence_threshold = float(getattr(config, "audio_gate_silence_threshold", 1e-4))
        rms_reference = float(getattr(config, "audio_gate_rms_reference", 0.05))
        fusion_init = str(getattr(config, "as_m4_fusion_init", "zero"))

        self.event_detector = AudioEventDetector(hidden_size, num_events)
        self.temporal_aligner = CausalTemporalAligner(
            hidden_size=hidden_size,
            align_dim=align_dim,
            max_offset_sec=max_offset_sec,
            similarity_chunk_size=getattr(config, "av_similarity_chunk_size", None),
        )
        local_offset_sec = float(getattr(config, "audio_event_local_offset_sec", 0.5))
        enable_offset_scorer = bool(getattr(config, "enable_audio_event_offset_scorer", False))
        offset_scorer_bundle_path = getattr(config, "audio_event_offset_scorer_bundle_path", None) or None
        if enable_offset_scorer and offset_scorer_bundle_path is None:
            raise ValueError(
                "audio_event_offset_scorer_bundle_path is required when the frozen offset scorer is enabled"
            )
        self.audio_event_aligner = LocalAudioEventAligner(
            hidden_size=hidden_size,
            align_dim=int(getattr(config, "audio_event_align_dim", None) or align_dim),
            candidate_offsets=(-local_offset_sec, 0.0, local_offset_sec),
            event_strength_weight=float(getattr(config, "audio_event_strength_weight", 0.05)),
            semantic_feature_mode=str(getattr(config, "audio_event_semantic_feature_mode", "disabled")),
            projector_checkpoint_path=getattr(config, "audio_event_projector_checkpoint_path", None) or None,
            offset_scorer_bundle_path=offset_scorer_bundle_path if enable_offset_scorer else None,
            offset_scorer_margin_threshold=float(
                getattr(config, "audio_event_offset_scorer_margin_threshold", 0.15)
            ),
            offset_scorer_stabilization_strategy=str(
                getattr(config, "audio_event_offset_scorer_stabilization_strategy", "none")
            ),
            offset_scorer_consecutive_windows=int(
                getattr(config, "audio_event_offset_scorer_consecutive_windows", 2)
            ),
            offset_scorer_hold_margin=float(
                getattr(config, "audio_event_offset_scorer_hold_margin", 0.10)
            ),
            offset_scorer_switch_margin=float(
                getattr(config, "audio_event_offset_scorer_switch_margin", 0.30)
            ),
            offset_scorer_moving_average_windows=int(
                getattr(config, "audio_event_offset_scorer_moving_average_windows", 3)
            ),
            enable_temporal_offset_gru_diagnostic=bool(
                getattr(config, "enable_temporal_offset_gru_diagnostic", False)
            ),
            temporal_offset_gru_checkpoint_path=(
                getattr(config, "temporal_offset_gru_checkpoint_path", None) or None
            ),
        )
        self.confidence_gate = AudioConfidenceGate(
            hidden_size,
            quality_dim=quality_dim,
            gate_logit_bias=gate_logit_bias,
            enable_v1=enable_gate_v1,
            silence_threshold=silence_threshold,
            rms_reference=rms_reference,
            max_offset_sec=max_offset_sec,
        )
        self.fusion = GatedAVFusion(hidden_size, fusion_init=fusion_init)


def build_streaming_av_module(config) -> StreamingAVModule:
    return StreamingAVModule(config)
