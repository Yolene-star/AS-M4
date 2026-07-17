"""CPU tests for AVE_HF projector baseline helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "train_ave_hf_projector_baseline.py"
SPEC = importlib.util.spec_from_file_location("train_ave_hf_projector_baseline", SCRIPT_PATH)
baseline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = baseline
SPEC.loader.exec_module(baseline)


def _write_feature(path: Path, key: str, values: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = torch.stack([torch.arange(values.shape[0]).float(), torch.arange(values.shape[0]).float() + 1.0], dim=1)
    torch.save({key: values.float(), "timestamps": timestamps}, path)


def _make_rows(tmp_path: Path) -> list[dict]:
    rows = []
    for index, label in enumerate(["dog", "dog", "car"]):
        youtube_id = f"v{index}"
        audio_path = tmp_path / f"{youtube_id}_audio.pt"
        video_path = tmp_path / f"{youtube_id}_video.pt"
        base = torch.eye(4)[:3] + index
        _write_feature(audio_path, "audio_embedding", base)
        _write_feature(video_path, "video_features", base + torch.linspace(0, 0.2, 3).unsqueeze(1))
        rows.append(
            {
                "youtube_id": youtube_id,
                "label": label,
                "split": "train",
                "audio_feature_path": str(audio_path),
                "video_feature_path": str(video_path),
            }
        )
    return rows


def test_diagnose_video_features_detects_non_collapsed_features(tmp_path):
    rows = _make_rows(tmp_path)

    diag = baseline.diagnose_video_features(rows, max_pairs=10, seed=1)

    assert diag["collapse_reasons"] == []
    assert diag["different_video_window_distance_mean"] > diag["adjacent_window_distance_mean"]
    assert diag["meets_minimum_training_gate"] is True


def test_build_pairs_and_split_keep_video_ids_disjoint(tmp_path):
    rows = _make_rows(tmp_path)

    pairs = baseline.build_pairs(rows, seed=1)
    split = baseline.split_videos(rows, train_ratio=0.67, seed=1)

    assert baseline.Counter(pair.pair_type for pair in pairs)["positive"] == 9
    assert baseline.Counter(pair.pair_type for pair in pairs)["silence_negative"] == 9
    assert not (set(split["train_ids"]) & set(split["val_ids"]))
    assert split["train_count"] + split["val_count"] == 3
