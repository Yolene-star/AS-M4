"""CPU tests for AVE_HF window feature extraction helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "extract_ave_hf_window_features.py"
SPEC = importlib.util.spec_from_file_location("extract_ave_hf_window_features", SCRIPT_PATH)
features = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = features
SPEC.loader.exec_module(features)


def test_load_wav_mono_reads_int16_stereo_and_resamples(tmp_path):
    path = tmp_path / "stereo.wav"
    rate = 8000
    left = torch.linspace(-0.5, 0.5, rate)
    right = -left
    stereo = torch.stack([left, right], dim=1)
    wavfile.write(path, rate, (stereo.numpy() * 32767).astype("int16"))

    waveform, sample_rate = features.load_wav_mono(path, target_sample_rate=16000)

    assert sample_rate == 16000
    assert waveform.ndim == 1
    assert waveform.numel() == 16000
    assert torch.isfinite(waveform).all()


def test_validate_payload_rejects_timestamp_mismatch(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"audio_embedding": torch.ones(2, 4), "timestamps": torch.ones(1, 2)}, path)

    try:
        features.validate_payload(path, "audio_embedding")
    except ValueError as exc:
        assert "时间戳" in str(exc)
    else:
        raise AssertionError("validate_payload should reject timestamp mismatch")


def test_load_existing_audio_feature_validates_sample_id(tmp_path):
    path = tmp_path / "precomputed_audio_features" / "abc" / "original.pt"
    path.parent.mkdir(parents=True)
    timestamps = torch.tensor([[0.0, 1.0], [0.5, 1.5]])
    torch.save(
        {
            "sample_id": "abc",
            "audio_embedding": torch.ones(2, 4),
            "timestamps": timestamps,
            "metadata": {"encoder_name": "BEATs"},
        },
        path,
    )

    result_path, result_timestamps, window_count = features.load_existing_audio_feature(
        {"youtube_id": "abc"}, tmp_path
    )

    assert result_path == path
    assert torch.equal(result_timestamps, timestamps)
    assert window_count == 2


def test_find_existing_audio_feature_returns_none_when_missing(tmp_path):
    result = features.find_existing_audio_feature({"youtube_id": "missing"}, tmp_path)

    assert result is None
