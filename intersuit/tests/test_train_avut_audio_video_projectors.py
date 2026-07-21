"""CPU tests for lightweight AVUT audio/video projector training."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "train_avut_audio_video_projectors.py"
SPEC = importlib.util.spec_from_file_location("train_avut_audio_video_projectors", SCRIPT_PATH)
projectors = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = projectors
SPEC.loader.exec_module(projectors)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _write_audio_feature(root: Path, sample_id: str, condition: str, values: torch.Tensor, timestamps: torch.Tensor, source: str | None = None) -> None:
    path = root / sample_id / f"{condition}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sample_id": sample_id,
            "condition": condition,
            "timestamps": timestamps,
            "audio_embedding": values,
            "metadata": {
                "source_sample_id": source or sample_id,
                "encoder_name": "BEATs",
                "embedding_dim": int(values.shape[-1]),
            },
        },
        path,
    )


def _make_inputs(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "avut.json"
    _write_json(
        manifest,
        [
            {"id": "avut_0001", "video_path": str(tmp_path / "a.mp4")},
            {"id": "avut_0002", "video_path": str(tmp_path / "b.mp4")},
        ],
    )
    audio_root = tmp_path / "audio"
    timestamps = torch.tensor([[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]], dtype=torch.float32)
    base = torch.arange(3 * 8, dtype=torch.float32).view(3, 8) / 10.0
    for sample_idx, sample_id in enumerate(("avut_0001", "avut_0002")):
        original = base + sample_idx
        _write_audio_feature(audio_root, sample_id, "original", original, timestamps)
        _write_audio_feature(audio_root, sample_id, "silence", torch.zeros_like(original), timestamps)
        _write_audio_feature(audio_root, sample_id, "wrong_audio", original.flip(0), timestamps, source="other")
        _write_audio_feature(audio_root, sample_id, "shift_plus_0_5", torch.roll(original, shifts=1, dims=0), timestamps)
        _write_audio_feature(audio_root, sample_id, "shift_minus_0_5", torch.roll(original, shifts=-1, dims=0), timestamps)
    return manifest, audio_root


def test_timestamp_match_rejects_mismatch():
    with pytest.raises(ValueError, match="not aligned"):
        projectors.validate_timestamp_match(
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([[0.0, 1.1]]),
        )


def test_prepare_and_train_projectors_with_frozen_report(monkeypatch, tmp_path):
    manifest, audio_root = _make_inputs(tmp_path)
    output_root = tmp_path / "out"

    def fake_save_video_feature(out_root, sample, timestamps, target_dim):
        sample_id = sample["id"]
        features = projectors.expand_to_dim(
            torch.stack(
                [
                    timestamps[:, 0],
                    timestamps[:, 1],
                    timestamps.mean(dim=-1),
                    torch.ones(timestamps.shape[0]),
                ],
                dim=-1,
            ),
            target_dim,
        )
        path = out_root / "video_window_features" / f"{sample_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sample_id": sample_id,
                "video_features": features,
                "timestamps": timestamps,
                "metadata": {"feature_kind": "test_video", "embedding_dim": target_dim},
            },
            path,
        )
        return path

    monkeypatch.setattr(projectors, "save_video_feature", fake_save_video_feature)
    args = argparse.Namespace(
        manifest=str(manifest),
        audio_feature_root=str(audio_root),
        output_root=str(output_root),
        limit=5,
        input_dim=8,
        project_dim=4,
        lr=1e-3,
        seed=123,
        steps=[2],
        prepare_only=False,
    )

    result = projectors.run(args)

    pair_manifest = output_root / "projector_pair_manifest.jsonl"
    assert pair_manifest.is_file()
    rows = projectors.read_pair_manifest(pair_manifest)
    assert len(rows) == 2 * len(projectors.CONDITIONS)
    assert sum(row["label"] == 1 for row in rows) == 2
    assert sum(row["label"] == 0 for row in rows) == 8
    report = result["training_reports"][0]
    assert report["checkpoint_reload_passed"] is True
    assert report["frozen_components"]["BEATs"] == "frozen_precomputed_audio_features_only"
    assert report["frozen_components"]["M4"] == "not_loaded_frozen_no_grad"
    assert report["frozen_components"]["Gate"] == "not_loaded_frozen_no_grad"
    assert report["frozen_components"]["trainable_parameter_names"] == ["audio_proj.weight", "video_proj.weight"]
    checkpoint = torch.load(report["checkpoint_path"], map_location="cpu", weights_only=True)
    assert checkpoint["audio_proj.weight"].shape == (4, 8)
    assert checkpoint["video_proj.weight"].shape == (4, 8)
    assert (output_root / "training_report_2step.md").is_file()


def test_load_pair_tensors_requires_matching_shapes(tmp_path):
    timestamps = torch.tensor([[0.0, 1.0]])
    audio_path = tmp_path / "audio.pt"
    video_path = tmp_path / "video.pt"
    torch.save(
        {
            "sample_id": "s0",
            "condition": "original",
            "timestamps": timestamps,
            "audio_embedding": torch.ones(1, 4),
            "metadata": {},
        },
        audio_path,
    )
    torch.save({"video_features": torch.ones(1, 5), "timestamps": timestamps}, video_path)

    with pytest.raises(ValueError, match="feature shape mismatch"):
        projectors.load_pair_tensors(
            {
                "sample_id": "s0",
                "audio_condition": "original",
                "audio_feature_path": str(audio_path),
                "video_feature_path": str(video_path),
                "label": 1,
            }
        )
