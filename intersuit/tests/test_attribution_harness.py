"""CPU harness tests for AS-M4 E0-E7 attribution matrix planning."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "intersuit" / "harness" / "runners" / "run_ablation_matrix.py"
SPEC = importlib.util.spec_from_file_location("as_m4_run_ablation_matrix", MODULE_PATH)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _base_config(tmp_path: Path) -> dict:
    manifest = tmp_path / "manifest.json"
    baseline = tmp_path / "baseline"
    as_m4 = tmp_path / "as_m4"
    manifest.write_text("[]\n", encoding="utf-8")
    baseline.mkdir()
    as_m4.mkdir()
    return {
        "dataset": {
            "name": "unit",
            "split": "smoke",
            "manifest": str(manifest),
            "scorer": "exact_match",
        },
        "models": {
            "baseline_m4": str(baseline),
            "as_m4": str(as_m4),
        },
        "output_root": str(tmp_path / "predictions"),
    }


def test_default_matrix_contains_required_e0_to_e7(tmp_path):
    config = runner.parse_config(_base_config(tmp_path))
    errors = runner.validate_matrix(config, strict_paths=True)
    plan = runner.build_plan(config)

    assert errors == []
    assert [record["id"] for record in plan] == list(runner.REQUIRED_IDS)
    assert plan[0]["audio_condition"] == "none"
    assert plan[1]["env"]["AS_M4_ROLLBACK_MODE"] == "behavior"
    assert plan[6]["alignment"] == "on"
    assert plan[7]["env"]["AS_M4_FORCE_AUDIO_GATE"] == "0"


def test_invalid_matrix_order_is_rejected(tmp_path):
    data = _base_config(tmp_path)
    data["experiments"] = list(reversed(runner.DEFAULT_EXPERIMENTS))
    config = runner.parse_config(data)
    errors = runner.validate_matrix(config)

    assert errors
    assert "experiments must be exactly" in errors[0]


def test_write_outputs_creates_plan_and_summary(tmp_path):
    config = runner.parse_config(_base_config(tmp_path))
    plan = runner.build_plan(config)
    out = tmp_path / "artifacts"

    runner.write_outputs(plan, out, [])

    rows = [json.loads(line) for line in (out / "matrix_plan.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert len(rows) == 8
    assert summary["status"] == "pass"
    assert summary["experiment_ids"] == list(runner.REQUIRED_IDS)


def test_build_plan_preserves_additional_experiment_environment(tmp_path):
    data = _base_config(tmp_path)
    experiments = [dict(item) for item in runner.DEFAULT_EXPERIMENTS]
    experiments[2]["env"] = {
        "AS_M4_DEBUG_AUDIO_CONDITION": "silence",
        "AS_M4_ENABLE_SCENE_AUDIO": "0",
    }
    data["experiments"] = experiments

    plan = runner.build_plan(runner.parse_config(data))

    assert plan[2]["env"]["AS_M4_DEBUG_AUDIO_CONDITION"] == "silence"
    assert plan[2]["env"]["AS_M4_ENABLE_SCENE_AUDIO"] == "1"
