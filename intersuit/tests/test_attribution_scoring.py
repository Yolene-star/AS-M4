"""CPU harness tests for AS-M4 E0-E7 attribution scoring."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNNER_PATH = ROOT / "intersuit" / "harness" / "runners" / "score_ablation_matrix.py"
SPEC = importlib.util.spec_from_file_location("as_m4_score_ablation_matrix", RUNNER_PATH)
score_runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = score_runner
SPEC.loader.exec_module(score_runner)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rows(num_correct: int, total: int = 10, gate: float = 0.5, offset_error: float = 0.0) -> list[dict]:
    rows = []
    for idx in range(total):
        correct = idx < num_correct
        rows.append(
            {
                "id": f"q{idx}",
                "prediction": "yes" if correct else "no",
                "answer": "yes",
                "gate": gate,
                "quality_gate": gate,
                "relevance_gate": gate,
                "pred_offset_sec": offset_error,
                "target_offset_sec": 0.0,
            }
        )
    return rows


def _plan(tmp_path: Path) -> Path:
    plan = tmp_path / "matrix_plan.jsonl"
    pred_root = tmp_path / "predictions"
    with plan.open("w", encoding="utf-8") as f:
        for idx in range(8):
            exp_id = f"E{idx}"
            f.write(
                json.dumps(
                    {
                        "id": exp_id,
                        "output_jsonl": str(pred_root / f"{exp_id.lower()}_predictions.jsonl"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return plan


def test_score_passes_when_core_attribution_relations_hold(tmp_path):
    plan = _plan(tmp_path)
    pred_root = tmp_path / "predictions"
    correct_by_exp = {
        "E0": 5,
        "E1": 5,
        "E2": 8,
        "E3": 4,
        "E4": 6,
        "E5": 4,
        "E6": 7,
        "E7": 5,
    }
    for exp_id, num_correct in correct_by_exp.items():
        _write_jsonl(pred_root / f"{exp_id.lower()}_predictions.jsonl", _rows(num_correct))

    result = score_runner.score_plan(plan, tmp_path / "score")

    assert result["status"] == "pass"
    assert result["comparison"]["e2_gt_e1"]
    assert result["comparison"]["e6_gt_e5"]
    assert result["summaries"]["E2"]["accuracy"] == 0.8


def test_score_fails_when_prediction_file_missing(tmp_path):
    plan = _plan(tmp_path)
    result = score_runner.score_plan(plan, tmp_path / "score")

    assert result["status"] == "fail"
    assert result["missing_prediction_files"]


def test_score_fails_when_e2_does_not_beat_e1(tmp_path):
    plan = _plan(tmp_path)
    pred_root = tmp_path / "predictions"
    for idx in range(8):
        exp_id = f"E{idx}"
        _write_jsonl(pred_root / f"{exp_id.lower()}_predictions.jsonl", _rows(5))

    result = score_runner.score_plan(plan, tmp_path / "score")

    assert result["status"] == "fail"
    assert not result["comparison"]["e2_gt_e1"]
