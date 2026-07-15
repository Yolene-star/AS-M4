"""Builder for AS-M4 streaming audio-video modules."""

from __future__ import annotations

from torch import nn

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

        self.event_detector = AudioEventDetector(hidden_size, num_events)
        self.temporal_aligner = CausalTemporalAligner(
            hidden_size=hidden_size,
            align_dim=align_dim,
            max_offset_sec=max_offset_sec,
            similarity_chunk_size=getattr(config, "av_similarity_chunk_size", None),
        )
        self.confidence_gate = AudioConfidenceGate(hidden_size, quality_dim=quality_dim)
        self.fusion = GatedAVFusion(hidden_size)


def build_streaming_av_module(config) -> StreamingAVModule:
    return StreamingAVModule(config)

