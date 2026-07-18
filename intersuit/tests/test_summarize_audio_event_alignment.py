"""冻结 offset scorer 诊断汇总测试。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/summarize_audio_event_alignment.py"
SPEC = importlib.util.spec_from_file_location("summarize_audio_event_alignment_test", SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary
SPEC.loader.exec_module(summary)


def test_offset_scorer_summary_records_acceptance_distribution_and_jumps():
    row = {
        "sample_id": "v1",
        "experiment_id": "F1_ORIGINAL",
        "as_m4_diagnostics": [
            {
                "audio_event_aligner_v1_enabled": True,
                "offset_scorer_available": [[True, True, True, True]],
                "offset_scorer_accepted": [[False, True, False, True]],
                "offset_scorer_best_offset": [[0.0, -0.5, 0.5, 0.5]],
                "offset_scorer_suggested_offset": [[0.0, -0.5, 0.0, 0.5]],
                "offset_scorer_margin": [[0.1, 0.3, 0.05, 0.4]],
                "offset_scorer_candidate_scores": [[[0.0, 1.0, 0.0]]],
                "offset_scorer_stable_accepted": [[False, False, False, True]],
                "offset_scorer_stable_suggested_offset": [[0.0, 0.0, 0.0, 0.5]],
            }
        ],
    }

    result = summary.summarize_row(row)

    assert result["offset_scorer_available_ratio"] == 1.0
    assert result["offset_scorer_accepted_ratio"] == 0.5
    assert result["suggested_offset_distribution"] == {"-0.5": 1, "0.0": 2, "0.5": 1}
    assert result["offset_scorer_jump_rate"] == 1.0
    assert result["offset_scorer_stable_accepted_ratio"] == 0.25
    assert result["offset_scorer_stable_jump_rate"] == pytest.approx(1 / 3)
