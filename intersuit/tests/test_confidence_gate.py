"""CPU harness for AS-M4 audio confidence gate."""

from __future__ import annotations

import importlib.util
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


if __name__ == "__main__":
    test_gate_outputs_are_in_unit_interval()
    test_clean_audio_gate_is_higher_than_bad_audio()
    test_alignment_confidence_affects_relevance()
    test_gate_is_not_saturated_for_clean_audio()
    print("confidence_gate harness passed")

