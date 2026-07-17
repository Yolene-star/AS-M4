"""CPU tests for diagnostic-only local audio event alignment."""

from __future__ import annotations

import torch

from intersuit.model.streaming_av.audio_event_aligner import (
    LocalAudioEventAligner,
    compute_audio_event_features,
)


def _timestamps(steps: int) -> torch.Tensor:
    return torch.arange(steps, dtype=torch.float32) * 0.5


def test_silent_window_event_strength_is_zero():
    features = compute_audio_event_features(torch.zeros(1, 3, 160))

    assert torch.equal(features.event_strength, torch.zeros_like(features.event_strength))
    assert features.is_silent_window.all()
    assert torch.equal(features.audio_rms, torch.zeros_like(features.audio_rms))
    assert torch.equal(features.audio_peak, torch.zeros_like(features.audio_peak))


def test_sound_event_strength_exceeds_silence():
    windows = torch.zeros(1, 2, 160)
    windows[:, 1] = 0.1

    features = compute_audio_event_features(windows)

    assert features.event_strength[0, 1] > features.event_strength[0, 0]
    assert features.audio_rms[0, 1] > features.audio_rms[0, 0]
    assert features.audio_peak[0, 1] > features.audio_peak[0, 0]


def test_three_offsets_and_boundary_candidates_are_correct():
    aligner = LocalAudioEventAligner()
    audio = torch.eye(5).unsqueeze(0)
    video = audio.unsqueeze(2)
    times = _timestamps(5)
    events = compute_audio_event_features(torch.ones(1, 5, 80) * 0.1, audio_features=audio)

    output = aligner(audio, video, times, times, events)

    assert torch.equal(output.candidate_offsets[0, 2], torch.tensor([-0.5, 0.0, 0.5]))
    assert output.candidate_valid[0, 0].tolist() == [True, True, False]
    assert output.candidate_valid[0, -1].tolist() == [False, True, True]
    assert output.candidate_indices.min() >= 0
    assert output.candidate_indices.max() < audio.shape[1]


def test_single_valid_candidate_has_zero_margin():
    aligner = LocalAudioEventAligner()
    audio = torch.ones(1, 1, 4)
    video = audio.unsqueeze(2)
    times = torch.tensor([0.0])
    events = compute_audio_event_features(torch.ones(1, 1, 80) * 0.1, audio_features=audio)

    output = aligner(audio, video, times, times, events)

    assert output.candidate_valid[0, 0].tolist() == [False, True, False]
    assert output.alignment_margin[0, 0].item() == 0.0


def test_best_offset_and_alignment_margin_select_matching_window():
    aligner = LocalAudioEventAligner()
    audio = torch.eye(5).unsqueeze(0)
    video = audio[:, 3:4].unsqueeze(2)
    audio_times = _timestamps(5)
    video_times = torch.tensor([1.0])
    events = compute_audio_event_features(torch.ones(1, 5, 80) * 0.1, audio_features=audio)

    output = aligner(audio, video, audio_times, video_times, events)

    assert output.best_offset.item() == -0.5
    assert output.best_alignment_score.item() > output.second_best_alignment_score.item()
    assert torch.allclose(
        output.alignment_margin,
        output.best_alignment_score - output.second_best_alignment_score,
    )
    assert output.alignment_confidence.item() > 0.0


def test_semantic_similarity_dominates_event_strength():
    aligner = LocalAudioEventAligner(event_strength_weight=0.05)
    audio = torch.tensor([[[0.0, 1.0, -1.0, 0.0], [1.0, -1.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0]]])
    video = audio[:, 1:2].unsqueeze(2)
    audio_times = _timestamps(3)
    video_times = torch.tensor([0.5])
    events = compute_audio_event_features(torch.ones(1, 3, 80) * 0.1, audio_features=audio)
    events = events._replace(event_strength=torch.tensor([[1.0, 0.1, 1.0]]))

    output = aligner(audio, video, audio_times, video_times, events)

    assert output.best_offset.item() == 0.0
    assert output.semantic_similarity[0, 0, 1] > output.semantic_similarity[0, 0, 0]
    assert output.semantic_similarity[0, 0, 1] > output.semantic_similarity[0, 0, 2]


def test_silent_candidates_have_zero_score_margin_and_confidence():
    aligner = LocalAudioEventAligner()
    audio = torch.randn(1, 3, 4)
    video = torch.randn(1, 3, 1, 4)
    times = _timestamps(3)
    events = compute_audio_event_features(torch.zeros(1, 3, 80), audio_features=audio)

    output = aligner(audio, video, times, times, events)

    assert torch.equal(output.candidate_scores, torch.zeros_like(output.candidate_scores))
    assert torch.equal(output.alignment_margin, torch.zeros_like(output.alignment_margin))
    assert torch.equal(output.alignment_confidence, torch.zeros_like(output.alignment_confidence))


def test_bf16_inputs_are_finite():
    aligner = LocalAudioEventAligner()
    audio = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    video = torch.randn(1, 3, 2, 4, dtype=torch.bfloat16)
    windows = torch.randn(1, 3, 80, dtype=torch.bfloat16) * 0.05
    times = _timestamps(3)
    events = compute_audio_event_features(windows, audio_features=audio)

    output = aligner(audio, video, times, times, events)

    for value in (*events[:-1], *output):
        if torch.is_tensor(value) and value.is_floating_point():
            assert torch.isfinite(value).all()


def test_nan_and_inf_inputs_are_sanitized():
    aligner = LocalAudioEventAligner()
    audio = torch.tensor([[[float("nan"), 0.0], [float("inf"), 1.0], [1.0, 0.0]]])
    video = torch.ones(1, 3, 1, 2)
    windows = torch.tensor([[[float("nan"), 0.0], [float("inf"), 0.0], [0.1, -0.1]]])
    times = _timestamps(3)
    events = compute_audio_event_features(windows, audio_features=audio)

    output = aligner(audio, video, times, times, events)

    for value in (*events[:-1], *output):
        if torch.is_tensor(value) and value.is_floating_point():
            assert torch.isfinite(value).all()
