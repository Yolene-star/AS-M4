"""冻结 AVE 时间同步 scorer 离线分析 helper 测试。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "analyze_ave_hf_temporal_offset_frozen.py"
SPEC = importlib.util.spec_from_file_location("analyze_ave_hf_temporal_offset_frozen", SCRIPT_PATH)
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def _row(key: str, condition: str, target: int, prediction: int, margin: float) -> dict:
    return {
        "record_key": key,
        "condition": condition,
        "target_index": target,
        "prediction_index": prediction,
        "correct": target == prediction,
        "margin": margin,
    }


def test_canonical_record_set_hash_ignores_order():
    first = [
        {
            "youtube_id": "a",
            "condition": "original",
            "video_window": 2,
            "audio_candidate_windows": [1, 2, 3],
            "target_index": 1,
        },
        {
            "youtube_id": "b",
            "condition": "shift_plus_0.5",
            "video_window": 3,
            "audio_candidate_windows": [3, 4, 5],
            "target_index": 0,
        },
    ]

    assert analysis.canonical_record_set_sha256(first) == analysis.canonical_record_set_sha256(list(reversed(first)))


def test_rejection_metrics_keeps_low_margin_at_zero():
    rows = [
        _row("a", "original", 1, 0, 0.05),
        _row("b", "shift_plus_0.5", 0, 0, 0.3),
        _row("c", "shift_minus_0.5", 2, 2, 0.4),
    ]

    result = analysis.rejection_metrics(rows, threshold=0.2)

    assert result["accepted_count"] == 2
    assert result["accepted_accuracy"] == 1.0
    assert result["conservative_overall_accuracy"] == 1.0
    assert result["original_false_correction_rate"] == 0.0


def test_prediction_consistency_reports_unanimous_and_majority():
    seed_rows = {
        "1": [_row("a", "original", 1, 1, 0.2), _row("b", "original", 1, 0, 0.2)],
        "2": [_row("a", "original", 1, 1, 0.2), _row("b", "original", 1, 0, 0.2)],
        "3": [_row("a", "original", 1, 1, 0.2), _row("b", "original", 1, 1, 0.2)],
    }

    result = analysis.prediction_consistency(seed_rows)

    assert result["category_counts"] == {"unanimous": 1, "two_of_three": 1}
    assert result["category_accuracy"]["unanimous"] == 1.0
    assert result["category_accuracy"]["two_of_three"] == 0.0


def test_aggregate_grouped_accuracy_reports_seed_variation():
    summaries = {
        "1": {"event_label": {"Bell": {"sample_count": 3, "accuracy": 1.0}}},
        "2": {"event_label": {"Bell": {"sample_count": 3, "accuracy": 0.5}}},
        "3": {"event_label": {"Bell": {"sample_count": 3, "accuracy": 0.0}}},
    }

    result = analysis.aggregate_grouped_accuracy(summaries, "event_label")

    assert result["Bell"]["sample_count_values"] == [3, 3, 3]
    assert result["Bell"]["accuracy_mean"] == 0.5
