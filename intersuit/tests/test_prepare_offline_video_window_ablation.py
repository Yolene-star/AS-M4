"""离线视频窗口四组消融准备脚本的 CPU 测试。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "prepare_offline_video_window_ablation.py"
SPEC = importlib.util.spec_from_file_location(
    "prepare_offline_video_window_ablation",
    SCRIPT_PATH,
)
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


def test_prepare_four_conditions_and_exact_fallbacks(tmp_path):
    feature_root = tmp_path / "source"
    feature_root.mkdir()
    features = torch.arange(20, dtype=torch.bfloat16).reshape(5, 2, 2)
    torch.save({"features": features}, feature_root / "sample.pt")
    qa_manifest = tmp_path / "qa.json"
    qa_manifest.write_text(
        json.dumps(
            [
                {
                    "id": "sample",
                    "video_features": "sample.pt",
                    "evaluation_category": "audio_necessary",
                    "conversations": [
                        {"from": "human", "value": "<image>\nQuestion?"},
                        {"from": "gpt", "value": "Answer"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    diagnostics = tmp_path / "diagnostics.jsonl"
    diagnostics.write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "best_offset": [0.0, -0.5, 0.5, 0.5, 0.0],
                "margin": [0.9, 0.1, 0.5, 0.7, 0.8],
                "event_strength": [1.0, 1.0, 0.0, 0.5, 1.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "output"
    args = argparse.Namespace(
        qa_manifest=str(qa_manifest),
        feature_root=str(feature_root),
        diagnostics=str(diagnostics),
        output_root=str(output_root),
        model_path=str(tmp_path / "model"),
        scorer_seed=20260719,
        margin_threshold=0.15,
        max_neighbor_weight=0.35,
    )

    summary = prepare.run(args)

    assert summary["low_confidence_elementwise_identical"]
    assert summary["zero_offset_elementwise_identical"]
    assert summary["all_finite"]
    assert summary["conditions"]["baseline"]["changed_window_count"] == 0
    assert summary["conditions"]["hard_move"]["changed_window_count"] == 2
    assert summary["conditions"]["offset_event_soft"]["changed_window_count"] == 1
    plan = [
        json.loads(line)
        for line in (output_root / "ablation_plan.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [row["id"] for row in plan] == list(prepare.MODES)
    assert all(row["env"]["AS_M4_ENABLE_SCENE_AUDIO"] == "0" for row in plan)
    category_plan = [
        json.loads(line)
        for line in (output_root / "ablation_plan_by_category.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(category_plan) == len(prepare.MODES)
    assert all("audio_necessary" in row["id"] for row in category_plan)
    weighted = torch.load(
        output_root / "features/offset_soft/sample.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert weighted["features"].dtype == torch.bfloat16


def test_frozen_scorer_configuration_cannot_change(tmp_path):
    args = argparse.Namespace(
        qa_manifest=str(tmp_path / "missing.json"),
        feature_root=str(tmp_path),
        diagnostics=str(tmp_path / "missing.jsonl"),
        output_root=str(tmp_path / "output"),
        model_path=str(tmp_path / "model"),
        scorer_seed=123,
        margin_threshold=0.15,
        max_neighbor_weight=0.35,
    )

    try:
        prepare.run(args)
    except ValueError as exc:
        assert "冻结" in str(exc)
    else:
        raise AssertionError("修改冻结 scorer seed 时必须拒绝运行")
