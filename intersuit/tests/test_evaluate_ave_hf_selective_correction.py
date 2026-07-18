"""高置信度选择性修正验收 helper 测试。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_ave_hf_selective_correction.py"
SPEC = importlib.util.spec_from_file_location("evaluate_ave_hf_selective_correction", SCRIPT)
evaluation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluation
SPEC.loader.exec_module(evaluation)


class Record:
    def __init__(self, condition: str):
        self.condition = condition


def test_strategy_metrics_keeps_rejected_samples_at_zero():
    records = [Record("original"), Record("shift_plus_0.5"), Record("shift_minus_0.5")]
    metrics = evaluation.strategy_metrics(
        records,
        torch.tensor([1, 0, 2]),
        torch.tensor([0, 0, 2]),
        torch.tensor([False, True, False]),
    )

    assert metrics["coverage"] == 1 / 3
    assert metrics["accepted_accuracy"] == 1.0
    assert metrics["conservative_overall_accuracy"] == pytest.approx(2 / 3)
    assert metrics["original_false_correction_rate"] == 0.0
    assert metrics["true_shift_recall"] == 0.5


def test_majority_prediction_uses_vote_and_probability_tiebreak():
    outputs = {
        "20260718": {
            "predictions": torch.tensor([0, 0]),
            "probabilities": torch.tensor([[0.8, 0.1, 0.1], [0.4, 0.3, 0.3]]),
            "scores": torch.log(torch.tensor([[0.8, 0.1, 0.1], [0.4, 0.3, 0.3]])),
        },
        "20260719": {
            "predictions": torch.tensor([0, 1]),
            "probabilities": torch.tensor([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]]),
            "scores": torch.log(torch.tensor([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]])),
        },
        "20260720": {
            "predictions": torch.tensor([2, 2]),
            "probabilities": torch.tensor([[0.1, 0.1, 0.8], [0.1, 0.2, 0.7]]),
            "scores": torch.log(torch.tensor([[0.1, 0.1, 0.8], [0.1, 0.2, 0.7]])),
        },
    }

    predictions, margins = evaluation.majority_prediction(outputs)

    assert predictions.tolist() == [0, 2]
    assert torch.all(margins >= 0)


def test_false_correction_reduction_has_predeclared_materiality():
    assert evaluation.false_correction_reduced(0.20, 0.25)
    assert evaluation.false_correction_reduced(0.14, 0.20)
    assert not evaluation.false_correction_reduced(0.23, 0.25)
