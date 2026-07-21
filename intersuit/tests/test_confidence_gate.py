"""CPU harness for AS-M4 audio confidence gate."""

from __future__ import annotations

import importlib.util
from functools import wraps
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "intersuit" / "model" / "streaming_av" / "confidence_gate.py"
SPEC = importlib.util.spec_from_file_location("as_m4_confidence_gate", MODULE_PATH)
confidence_gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = confidence_gate
SPEC.loader.exec_module(confidence_gate)

AudioConfidenceGate = confidence_gate.AudioConfidenceGate
compute_audio_signal_features = confidence_gate.compute_audio_signal_features


def preserve_torch_rng(test_fn):
    @wraps(test_fn)
    def wrapped():
        with torch.random.fork_rng(devices=[]):
            return test_fn()

    return wrapped


def test_gate_outputs_are_in_unit_interval():
    gate = AudioConfidenceGate(hidden_size=4, quality_dim=1)
    audio = torch.ones(1, 2, 4) * 0.5
    video = audio.clone()
    quality = torch.ones(1, 2, 1)

    output = gate(audio, video, quality_features=quality)

    assert torch.all((0 <= output.quality) & (output.quality <= 1))
    assert torch.all((0 <= output.relevance) & (output.relevance <= 1))
    assert torch.all((0 <= output.gate) & (output.gate <= 1))


def test_clean_audio_gate_is_higher_than_bad_audio():
    gate = AudioConfidenceGate(hidden_size=4, quality_dim=1)
    clean_audio = torch.ones(1, 1, 4) * 0.5
    clean_video = clean_audio.clone()
    clean_quality = torch.ones(1, 1, 1) * 3.0

    mute_audio = torch.zeros(1, 1, 4)
    noisy_quality = torch.ones(1, 1, 1) * -3.0
    mismatch_video = -clean_audio

    clean = gate(clean_audio, clean_video, quality_features=clean_quality, alignment_confidence=torch.ones(1, 1) * 0.95)
    mute = gate(mute_audio, clean_video, quality_features=clean_quality, alignment_confidence=torch.ones(1, 1) * 0.95)
    noisy = gate(clean_audio, clean_video, quality_features=noisy_quality, alignment_confidence=torch.ones(1, 1) * 0.95)
    mismatch = gate(clean_audio, mismatch_video, quality_features=clean_quality, alignment_confidence=torch.ones(1, 1) * 0.95)

    assert clean.gate.item() > mute.gate.item()
    assert clean.gate.item() > noisy.gate.item()
    assert clean.gate.item() > mismatch.gate.item()


def test_alignment_confidence_affects_relevance():
    gate = AudioConfidenceGate(hidden_size=4, quality_dim=1)
    audio = torch.ones(1, 1, 4) * 0.5
    video = audio.clone()
    quality = torch.ones(1, 1, 1) * 3.0

    high = gate(audio, video, quality_features=quality, alignment_confidence=torch.ones(1, 1) * 0.95)
    low = gate(audio, video, quality_features=quality, alignment_confidence=torch.ones(1, 1) * 0.05)

    assert high.relevance.item() > low.relevance.item()
    assert high.gate.item() > low.gate.item()


def test_gate_is_not_saturated_for_clean_audio():
    gate = AudioConfidenceGate(hidden_size=4, quality_dim=1)
    audio = torch.ones(1, 1, 4) * 0.25
    output = gate(audio, audio, quality_features=torch.ones(1, 1, 1))

    assert 0.0 < output.gate.item() < 1.0


def test_default_gate_starts_nearly_closed_for_clean_audio():
    gate = AudioConfidenceGate(hidden_size=4, quality_dim=1)
    audio = torch.ones(1, 1, 4) * 0.25
    output = gate(audio, audio, quality_features=torch.ones(1, 1, 1))

    assert 0.0 < output.gate.item() < 0.02


@preserve_torch_rng
def test_gate_v1_silent_audio_is_exactly_closed_and_finite():
    gate = AudioConfidenceGate(hidden_size=4, quality_dim=1, enable_v1=True)
    audio = torch.zeros(1, 1, 4)
    signal = compute_audio_signal_features(torch.zeros(1, 1, 160))

    output = gate(
        audio,
        torch.ones_like(audio),
        question_feature=torch.ones(1, 4),
        quality_features=torch.ones(1, 1, 1),
        alignment_confidence=torch.ones(1, 1) * 0.95,
        signal_features=signal,
        offset_sec=torch.zeros(1, 1),
    )

    assert output.gate.item() == 0.0
    assert gate.last_v1_diagnostics["audio_rms"].item() == 0.0
    assert gate.last_v1_diagnostics["silence_ratio"].item() == 1.0
    assert all(torch.isfinite(value).all() for value in output)
    assert all(torch.isfinite(value).all() for value in gate.last_v1_diagnostics.values())


@preserve_torch_rng
def test_gate_v1_ranks_correct_above_wrong_and_shifted_audio():
    gate = AudioConfidenceGate(hidden_size=4, quality_dim=1, enable_v1=True, max_offset_sec=1.0)
    correct_audio = torch.tensor([[[0.5, 0.5, 0.5, 0.5]]])
    wrong_audio = -correct_audio
    video = correct_audio.clone()
    question = correct_audio[:, 0]
    signal = compute_audio_signal_features(torch.ones(1, 1, 160) * 0.1)
    common = {
        "question_feature": question,
        "quality_features": torch.ones(1, 1, 1),
        "signal_features": signal,
    }

    correct = gate(
        correct_audio,
        video,
        alignment_confidence=torch.ones(1, 1) * 0.95,
        offset_sec=torch.zeros(1, 1),
        **common,
    )
    wrong = gate(
        wrong_audio,
        video,
        alignment_confidence=torch.ones(1, 1) * 0.95,
        offset_sec=torch.zeros(1, 1),
        **common,
    )
    shifted = gate(
        correct_audio,
        video,
        alignment_confidence=torch.ones(1, 1) * 0.2,
        offset_sec=torch.ones(1, 1),
        **common,
    )

    assert correct.gate.item() > wrong.gate.item()
    assert correct.gate.item() > shifted.gate.item()


@preserve_torch_rng
def test_gate_v1_disabled_preserves_legacy_outputs_bit_exactly():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        legacy = AudioConfidenceGate(hidden_size=4, quality_dim=1)
        explicit_off = AudioConfidenceGate(hidden_size=4, quality_dim=1, enable_v1=False)
        explicit_off.load_state_dict(legacy.state_dict())
        audio = torch.randn(1, 2, 4)
        video = torch.randn(1, 2, 4)
        question = torch.randn(1, 4)
        quality = torch.randn(1, 2, 1)
        confidence = torch.rand(1, 2)

        expected = legacy(audio, video, question, quality, confidence)
        actual = explicit_off(audio, video, question, quality, confidence)

    assert len(expected) == len(actual) == 5
    for expected_tensor, actual_tensor in zip(expected, actual):
        assert torch.equal(expected_tensor, actual_tensor)


if __name__ == "__main__":
    test_gate_outputs_are_in_unit_interval()
    test_clean_audio_gate_is_higher_than_bad_audio()
    test_alignment_confidence_affects_relevance()
    test_gate_is_not_saturated_for_clean_audio()
    test_default_gate_starts_nearly_closed_for_clean_audio()
    test_gate_v1_silent_audio_is_exactly_closed_and_finite()
    test_gate_v1_ranks_correct_above_wrong_and_shifted_audio()
    test_gate_v1_disabled_preserves_legacy_outputs_bit_exactly()
    print("confidence_gate harness passed")
