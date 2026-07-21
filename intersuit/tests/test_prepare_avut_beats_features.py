"""CPU tests for AVUT BEATs precomputed feature preparation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "prepare_avut_beats_features.py"
SPEC = importlib.util.spec_from_file_location("prepare_avut_beats_features", SCRIPT_PATH)
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_missing_beats_checkpoint_fails_without_download(tmp_path):
    missing = tmp_path / "BEATs_iter3_plus_AS2M.pt"

    with pytest.raises(FileNotFoundError, match="Automatic downloads are disabled"):
        prepare.BEATsWindowEncoder(checkpoint_path=missing, beats_code_root=ROOT / "missing_code")


def test_shift_waveform_preserves_length_and_zero_pads():
    waveform = torch.arange(5, dtype=torch.float32)

    delayed = prepare.shift_waveform(waveform, sample_rate=2, shift_seconds=1.0)
    advanced = prepare.shift_waveform(waveform, sample_rate=2, shift_seconds=-1.0)

    assert delayed.tolist() == [0.0, 0.0, 0.0, 1.0, 2.0]
    assert advanced.tolist() == [2.0, 3.0, 4.0, 0.0, 0.0]
    assert delayed.shape == waveform.shape
    assert advanced.shape == waveform.shape


def test_validate_feature_file_rejects_bad_payload(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"sample_id": "s0"}, path)

    with pytest.raises(ValueError, match="missing keys"):
        prepare.validate_feature_file(path)

    mismatch = tmp_path / "mismatch.pt"
    torch.save(
        {
            "sample_id": "s0",
            "condition": "original",
            "timestamps": torch.zeros(3, 2),
            "audio_embedding": torch.zeros(2, 4),
            "metadata": {},
        },
        mismatch,
    )
    with pytest.raises(ValueError, match="timestamp/window count mismatch"):
        prepare.validate_feature_file(mismatch)


def test_fake_encoder_pipeline_writes_all_conditions(monkeypatch, tmp_path):
    manifest = tmp_path / "avut_audio_smoke.json"
    media0 = tmp_path / "a.mp4"
    media1 = tmp_path / "b.mp4"
    media0.write_bytes(b"fake")
    media1.write_bytes(b"fake")
    _write_json(
        manifest,
        [
            {
                "id": "avut_0001",
                "scene_audio_path": str(media0),
                "scene_audio_sample_rate": 4,
                "scene_audio_window_sec": 1.0,
                "scene_audio_hop_sec": 0.5,
            },
            {
                "id": "avut_0002",
                "scene_audio_path": str(media1),
                "scene_audio_sample_rate": 4,
                "scene_audio_window_sec": 1.0,
                "scene_audio_hop_sec": 0.5,
            },
        ],
    )

    def fake_load_sample_waveforms(samples):
        return {
            "avut_0001": (torch.arange(8, dtype=torch.float32), 4, media0),
            "avut_0002": (torch.arange(8, dtype=torch.float32).flip(0), 4, media1),
        }

    monkeypatch.setattr(prepare, "load_sample_waveforms", fake_load_sample_waveforms)
    args = argparse.Namespace(
        manifest=str(manifest),
        output_root=str(tmp_path / "features"),
        beats_checkpoint=str(tmp_path / "unused.pt"),
        beats_code_root=str(tmp_path),
        device="cpu",
        limit=5,
        sample_rate=4,
        window_sec=1.0,
        hop_sec=0.5,
        git_commit="test",
    )

    result = prepare.run_feature_extraction(args, encoder=prepare.WaveformStatsWindowEncoder())

    output_root = tmp_path / "features"
    assert (output_root / "beats_feature_config.json").is_file()
    assert (output_root / "feature_validation_summary.json").is_file()
    assert (output_root / "feature_validation_report.md").is_file()
    assert (output_root / "run_metadata.json").is_file()
    assert result["summary"]["feature_file_count"] == 2 * len(prepare.CONDITIONS)
    assert result["summary"]["embedding_dims"] == [4]
    assert result["summary"]["meets_feature_preparation_criteria"] is True

    feature_path = output_root / "precomputed_audio_features" / "avut_0001" / "original.pt"
    validation = prepare.validate_feature_file(feature_path)
    payload = torch.load(feature_path, map_location="cpu", weights_only=True)
    assert validation["window_count"] == 3
    assert payload["metadata"]["encoder_name"] == "test_waveform_stats"
    assert payload["metadata"]["sample_rate"] == 4
    assert torch.isfinite(payload["audio_embedding"]).all()


def test_window_condition_rejects_nonfinite():
    with pytest.raises(ValueError, match="NaN or Inf"):
        prepare.validate_windows(torch.tensor([[float("inf")]]), torch.tensor([[0.0, 1.0]]))
