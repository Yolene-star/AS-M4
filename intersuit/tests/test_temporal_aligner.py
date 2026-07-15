"""CPU harness for the AS-M4 causal temporal aligner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "intersuit" / "model" / "streaming_av" / "temporal_aligner.py"
SPEC = importlib.util.spec_from_file_location("as_m4_temporal_aligner", MODULE_PATH)
temporal_aligner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = temporal_aligner
SPEC.loader.exec_module(temporal_aligner)

CausalTemporalAligner = temporal_aligner.CausalTemporalAligner


def _features(num_steps, dim):
    return torch.eye(num_steps, dim).unsqueeze(0)


def _run_known_offset(offset, lookahead):
    dim = 8
    steps = 5
    video = _features(steps, dim)
    audio = video.clone()
    video_times = torch.arange(steps, dtype=torch.float32) * 0.5
    audio_times = video_times + offset
    aligner = CausalTemporalAligner(
        hidden_size=dim,
        align_dim=dim,
        max_offset_sec=1.25,
        temperature=0.05,
        similarity_chunk_size=2,
    )

    return aligner(audio, video, audio_times, video_times, lookahead_sec=lookahead)


def test_predicts_positive_offset_with_strict_causality():
    output = _run_known_offset(offset=0.5, lookahead=0.0)

    assert torch.allclose(output.offset_sec[output.valid_mask].mean(), torch.tensor(0.5), atol=0.26)
    assert output.offset_confidence[output.valid_mask].mean() > 0


def test_predicts_negative_offset_when_lookahead_allows_future_video():
    output = _run_known_offset(offset=-0.5, lookahead=0.5)

    assert torch.allclose(output.offset_sec[output.valid_mask].mean(), torch.tensor(-0.5), atol=0.26)


def test_strict_causality_blocks_future_video_match():
    dim = 4
    video = torch.eye(2, dim).unsqueeze(0)
    audio = video[:, 1:2].clone()
    audio_times = torch.tensor([0.0])
    video_times = torch.tensor([0.0, 0.5])
    aligner = CausalTemporalAligner(hidden_size=dim, align_dim=dim, max_offset_sec=1.0)

    output = aligner(audio, video, audio_times, video_times, lookahead_sec=0.0)
    best_index = output.alignment_weights[0, 0].argmax().item()

    assert best_index == 0
    assert output.alignment_weights[0, 0, 1] == 0


def test_accepts_patch_video_features_and_chunked_similarity():
    dim = 4
    audio = torch.randn(1, 3, dim)
    video = torch.randn(1, 3, 2, dim)
    times = torch.tensor([0.0, 0.5, 1.0])
    aligner = CausalTemporalAligner(hidden_size=dim, align_dim=dim, similarity_chunk_size=1)

    output = aligner(audio, video, times, times, lookahead_sec=0.0)

    assert output.alignment_weights.shape == (1, 3, 3)
    assert output.aligned_video_features.shape == (1, 3, dim)


if __name__ == "__main__":
    test_predicts_positive_offset_with_strict_causality()
    test_predicts_negative_offset_when_lookahead_allows_future_video()
    test_strict_causality_blocks_future_video_match()
    test_accepts_patch_video_features_and_chunked_similarity()
    print("temporal_aligner harness passed")
