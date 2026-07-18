"""CPU tests for M4 CLIP window feature extraction helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "extract_ave_hf_clip_window_features.py"
SPEC = importlib.util.spec_from_file_location("extract_ave_hf_clip_window_features", SCRIPT_PATH)
clip_features = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = clip_features
SPEC.loader.exec_module(clip_features)


def test_select_rows_by_label_keeps_label_balanced_prefix():
    rows = [
        {"youtube_id": "a1", "label": "a"},
        {"youtube_id": "a2", "label": "a"},
        {"youtube_id": "a3", "label": "a"},
        {"youtube_id": "b1", "label": "b"},
        {"youtube_id": "b2", "label": "b"},
    ]

    selected = clip_features.select_rows_by_label(rows, samples_per_label=2)

    assert [row["youtube_id"] for row in selected] == ["a1", "a2", "b1", "b2"]


def test_slice_rows_uses_non_overlapping_half_open_range():
    rows = [{"youtube_id": str(index)} for index in range(6)]

    selected = clip_features.slice_rows(rows, start_index=2, end_index=5)

    assert [row["youtube_id"] for row in selected] == ["2", "3", "4"]


def test_frame_indices_for_windows_clamps_to_video_bounds():
    timestamps = torch.tensor([[0.0, 1.0], [9.5, 10.5]])

    indices = clip_features.frame_indices_for_windows(timestamps, fps=10.0, frame_count=100, frames_per_window=2)

    assert indices[0] == [2, 8]
    assert indices[1] == [98, 99]


def test_load_existing_video_feature_validates_and_reuses_output(tmp_path):
    timestamps = torch.tensor([[0.0, 1.0], [0.5, 1.5]])
    feature_dir = tmp_path / "video_window_features"
    feature_dir.mkdir()
    feature_path = feature_dir / "abc.pt"
    torch.save(
        {
            "video_features": torch.ones(2, 4),
            "timestamps": timestamps,
            "metadata": {"feature_kind": "test"},
        },
        feature_path,
    )

    result = clip_features.load_existing_video_feature({"youtube_id": "abc"}, timestamps, tmp_path)

    assert result is not None
    path, embeddings = result
    assert path == feature_path
    assert embeddings.shape == (2, 4)


def test_load_existing_video_feature_rejects_mismatched_timestamps(tmp_path):
    timestamps = torch.tensor([[0.0, 1.0]])
    feature_dir = tmp_path / "video_window_features"
    feature_dir.mkdir()
    torch.save(
        {
            "video_features": torch.ones(1, 4),
            "timestamps": torch.tensor([[1.0, 2.0]]),
        },
        feature_dir / "abc.pt",
    )

    try:
        clip_features.load_existing_video_feature({"youtube_id": "abc"}, timestamps, tmp_path)
    except ValueError as exc:
        assert "时间戳内容不一致" in str(exc)
    else:
        raise AssertionError("时间戳不一致时应拒绝复用已有特征")
