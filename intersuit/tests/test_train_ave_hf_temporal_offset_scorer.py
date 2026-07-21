"""CPU tests for AVE_HF temporal offset scorer helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "train_ave_hf_temporal_offset_scorer.py"
SPEC = importlib.util.spec_from_file_location("train_ave_hf_temporal_offset_scorer", SCRIPT_PATH)
offsets = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = offsets
SPEC.loader.exec_module(offsets)


def _cache() -> dict:
    values = torch.eye(6, 4)
    return {
        "youtube_id": "v1",
        "label": "10",
        "split": "train",
        "audio": values,
        "clip": values + 0.1,
        "rgb": values + 0.2,
        "timestamps": torch.stack([torch.arange(6).float(), torch.arange(6).float() + 1.0], dim=1),
        "rms": torch.linspace(0.1, 0.6, 6),
        "nonsilent": torch.ones(6),
        "audio_change": torch.tensor([0.0, 0.2, 2.0, 0.1, 0.3, 0.0]),
        "clip_change": torch.tensor([0.0, 0.1, 1.5, 0.2, 0.1, 0.0]),
        "rgb_change": torch.tensor([0.0, 0.1, 1.2, 0.2, 0.1, 0.0]),
        "energy_change": torch.tensor([0.0, 0.1, 1.0, 0.2, 0.1, 0.0]),
        "combined_change": torch.tensor([0.0, 0.5, 5.0, 0.4, 0.6, 0.0]),
    }


def test_select_change_windows_prefers_change_peak():
    selected = offsets.select_change_windows(_cache(), top_k=2, min_change_quantile=0.0)

    assert selected[0][0] == 2
    assert selected[0][1] == "audio_change"


def test_build_records_assigns_offsets_for_shift_conditions():
    records = offsets.build_records(
        [_cache()],
        {"v1"},
        top_k=1,
        min_change_quantile=0.0,
        min_candidate_change_margin=0.0,
        require_center_peak=False,
        min_center_audio_dominance=-1e9,
        min_center_video_dominance=-1e9,
        context_radius=2,
    )
    by_condition = {record.condition: record for record in records}

    assert by_condition["original"].correct_offset == 0.0
    assert by_condition["shift_plus_0.5"].correct_offset == -0.5
    assert by_condition["shift_minus_0.5"].correct_offset == 0.5
    assert by_condition["shift_plus_0.5"].audio_candidate_windows == [2, 3, 4]
    assert by_condition["original"].sync_peak_delta == 0.0


def test_offset_scorer_outputs_three_scores():
    model = offsets.OffsetScorer(audio_dim=4, video_dim=5, scalar_dim=6, hidden_dim=8)

    scores = model(torch.ones(2, 3, 4), torch.ones(2, 5), torch.ones(2, 3, 6))

    assert scores.shape == (2, 3)
    assert torch.isfinite(scores).all()


def test_evaluate_reports_accuracy_and_margin():
    class FixedModel(torch.nn.Module):
        def forward(self, audio_candidates, video_context, scalar_features):
            return torch.tensor([[0.0, 2.0, 1.0], [3.0, 1.0, 0.0]])

    records = [
        offsets.OffsetRecord("v1", "10", "val", "original", 2, [1, 2, 3], 0.0, 1, "audio_change", 1, 1, 1, 1, 1, 0, 1, 1),
        offsets.OffsetRecord("v2", "11", "val", "shift_plus_0.5", 2, [2, 3, 4], -0.5, 0, "clip_change", 1, 1, 1, 1, 1, 0, 1, 1),
    ]

    result = offsets.evaluate(FixedModel(), torch.ones(2, 3, 4), torch.ones(2, 5), torch.ones(2, 3, 6), torch.tensor([1, 0]), records)

    assert result["accuracy"] == 1.0
    assert result["margin_mean"] > 0
    assert result["condition_sample_count"] == {"original": 1, "shift_plus_0.5": 1}


def test_context_features_include_multiple_windows_and_deltas():
    values = torch.arange(20, dtype=torch.float32).reshape(5, 4)

    feature = offsets.context_with_deltas(values, index=2, radius=2)

    assert feature.shape == (5 * 4 * 2,)
    assert torch.isfinite(feature).all()


def test_keep_zero_margin_threshold_prefers_no_correction():
    class AmbiguousShiftModel(torch.nn.Module):
        def forward(self, audio_candidates, video_context, scalar_features):
            return torch.tensor([[1.05, 1.0, 0.2]])

    records = [
        offsets.OffsetRecord("v1", "10", "val", "original", 2, [1, 2, 3], 0.0, 1, "audio_change", 1, 1, 1, 1, 1, 0, 1, 1),
    ]

    result = offsets.evaluate(
        AmbiguousShiftModel(),
        torch.ones(1, 3, 4),
        torch.ones(1, 5),
        torch.ones(1, 3, 6),
        torch.tensor([1]),
        records,
        keep_zero_margin_threshold=0.1,
    )

    assert result["accuracy"] == 1.0
    assert result["prediction_distribution"] == {"0.0": 1}
