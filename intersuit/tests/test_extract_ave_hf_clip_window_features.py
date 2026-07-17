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


def test_frame_indices_for_windows_clamps_to_video_bounds():
    timestamps = torch.tensor([[0.0, 1.0], [9.5, 10.5]])

    indices = clip_features.frame_indices_for_windows(timestamps, fps=10.0, frame_count=100, frames_per_window=2)

    assert indices[0] == [2, 8]
    assert indices[1] == [98, 99]
