"""最终任务 60 条混合评测清单的 CPU 测试。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/build_mixed_final_task_weighting_eval.py"
SPEC = importlib.util.spec_from_file_location(
    "build_mixed_final_task_weighting_eval",
    SCRIPT_PATH,
)
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def write_json(path: Path, value) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def write_jsonl(path: Path, values) -> Path:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )
    return path


def test_build_exact_three_category_mixture(tmp_path):
    audio = [
        {
            "id": f"a{index}",
            "answer": "yes",
            "evaluation_category": "audio_necessary",
            "scene_audio_path": f"/audio/a{index}.mp4",
            "video_path": f"/video/a{index}.mp4",
        }
        for index in range(20)
    ]
    visual = [
        {"id": f"v{index}", "answer": "yes", "evaluation_category": "pure_visual"}
        for index in range(20)
    ]
    interference = [
        {
            "id": f"i{index}",
            "answer": "yes",
            "evaluation_category": "audio_interference",
        }
        for index in range(20)
    ]
    old = visual + interference
    old_clip = [{"sample_id": row["id"]} for row in old]
    old_rgb = [{"sample_id": row["id"]} for row in old]
    avut_clip = [
        {
            "sample_id": row["id"],
            "audio_feature_path": f"/features/{row['id']}/original.pt",
            "video_feature_path": f"/clip/{row['id']}.pt",
        }
        for row in audio
    ]
    avut_pairs = [
        {
            "sample_id": row["id"],
            "audio_condition": "original",
            "audio_feature_path": f"/features/{row['id']}/original.pt",
            "video_feature_path": f"/rgb/{row['id']}.pt",
        }
        for row in audio
    ]
    args = argparse.Namespace(
        avut_qa_manifest=str(write_json(tmp_path / "avut.json", audio)),
        avut_clip_manifest=str(write_jsonl(tmp_path / "avut_clip.jsonl", avut_clip)),
        avut_pair_manifest=str(write_jsonl(tmp_path / "avut_pairs.jsonl", avut_pairs)),
        avut_m4_manifest=None,
        ave_m4_manifest=None,
        ave_qa_manifest=str(write_json(tmp_path / "ave.json", old)),
        ave_clip_manifest=str(write_jsonl(tmp_path / "ave_clip.jsonl", old_clip)),
        ave_rgb_manifest=str(write_jsonl(tmp_path / "ave_rgb.jsonl", old_rgb)),
        output_root=str(tmp_path / "output"),
        margin_threshold=0.15,
        max_neighbor_weight=0.35,
    )

    summary = builder.run(args)

    assert summary["sample_count"] == 60
    assert summary["unique_sample_count"] == 60
    assert summary["category_counts"] == {
        "audio_necessary": 20,
        "pure_visual": 20,
        "audio_interference": 20,
    }
    assert len(json.loads((tmp_path / "output/mixed_final_task_eval.json").read_text())) == 60
    assert len((tmp_path / "output/clip_manifest.jsonl").read_text().splitlines()) == 60


def test_frozen_weighting_parameters_are_rejected_before_file_reads(tmp_path):
    args = argparse.Namespace(
        margin_threshold=0.2,
        max_neighbor_weight=0.35,
    )
    with pytest.raises(ValueError, match="冻结"):
        builder.run(args)
