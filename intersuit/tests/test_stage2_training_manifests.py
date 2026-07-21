from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_stage2_training_manifests import (
    apportion,
    assign_groups,
    deduplicate_groups,
    normalize_task_type,
)
from finalize_stage2_gate_report import finalize
from new_dataset_common import sha256_file


def group(key: str, source: str, weight: int, task: str = "audio", origin: str = "template"):
    return {
        "group_key": key,
        "source_dataset": source,
        "youtube_id": key.casefold(),
        "physical_media_id": f"physical_{key}",
        "rows": [],
        "media_sha256": {f"{key}_{index}" for index in range(weight)},
        "video_paths": {f"/{key}_{index}.mp4" for index in range(weight)},
        "derived_ids": {f"{source}:{key}_{index}" for index in range(weight)},
        "video_ids": {f"{key}_{index}" for index in range(weight)},
        "task_types": {task},
        "qa_origins": {origin},
        "qa_count": weight,
        "weight": weight,
    }


def test_task_type_normalization():
    assert normalize_task_type("audio:counting") == "audio"
    assert normalize_task_type("visual") == "visual"
    assert normalize_task_type("audio_visual:temporal") == "audio_visual"


def test_apportion_matches_stage2_source_targets():
    counts = {"AVE_HF_EXPANDED": 852, "AVUT": 424, "MUSIC-AVQA-v2.0": 144}
    assert apportion(counts, 1000) == {
        "AVE_HF_EXPANDED": 600,
        "AVUT": 299,
        "MUSIC-AVQA-v2.0": 101,
    }
    assert apportion(counts, 300) == {
        "AVE_HF_EXPANDED": 180,
        "AVUT": 90,
        "MUSIC-AVQA-v2.0": 30,
    }


def test_weighted_groups_never_cross_splits():
    groups = [
        *[group(f"ave_{index}", "AVE_HF_EXPANDED", 1, "audio") for index in range(12)],
        *[group(f"avut_{index}", "AVUT", 1, "audio_visual", "human") for index in range(6)],
        group("music_flip_0", "MUSIC-AVQA-v2.0", 2, "visual"),
        group("music_0", "MUSIC-AVQA-v2.0", 1, "audio_visual"),
        group("music_1", "MUSIC-AVQA-v2.0", 1, "audio"),
    ]
    assigned = assign_groups(groups, train_target=14, dev_target=5)
    assert sum(item["weight"] for item in assigned["train"]) == 14
    assert sum(item["weight"] for item in assigned["dev"]) == 5
    all_keys = [item["group_key"] for values in assigned.values() for item in values]
    assert len(all_keys) == len(set(all_keys)) == len(groups)


def test_weighted_variants_are_not_all_consumed_by_train():
    groups = [
        *[group(f"music_{index}", "MUSIC-AVQA-v2.0", 1, "audio") for index in range(24)],
        *[group(f"music_flip_{index}", "MUSIC-AVQA-v2.0", 2, "audio") for index in range(6)],
    ]
    assigned = assign_groups(groups, train_target=24, dev_target=8)
    assert any(item["weight"] == 2 for item in assigned["dev"])
    assert any(item["weight"] == 2 for item in assigned["reserve"])


def test_content_dedup_drops_lower_priority_group():
    human = group("shared", "AVUT", 1, "audio_visual", "human")
    template = group("shared", "AVE_HF_EXPANDED", 1, "visual")
    template["youtube_id"] = human["youtube_id"]
    kept, dropped = deduplicate_groups([template, human])
    assert [item["source_dataset"] for item in kept] == ["AVUT"]
    assert dropped[0]["source_dataset"] == "AVE_HF_EXPANDED"
    assert dropped[0]["collisions"][0]["field"] == "youtube_id"


def test_stage2_validator_accepts_disjoint_tiny_splits(tmp_path):
    manifests = {}
    for split, frequency in (("train", 440), ("dev", 550), ("reserve", 660)):
        video = tmp_path / f"{split}.mp4"
        audio = tmp_path / f"{split}.wav"
        subprocess.run([
            "ffmpeg", "-nostdin", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=size=32x32:rate=5",
            "-f", "lavfi", "-i", f"sine=frequency={frequency}:duration=1",
            "-t", "1", "-c:v", "mpeg4", "-c:a", "aac", str(video),
        ], check=True)
        subprocess.run([
            "ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(video),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio),
        ], check=True)
        sample_id = f"{split}_sample"
        row = {
            "id": sample_id,
            "sample_id": sample_id,
            "source_dataset": "fixture",
            "source_revision": "fixture-v1",
            "video_id": f"{split}_video",
            "youtube_id": f"{split}_youtube",
            "physical_media_id": f"{split}_physical",
            "derived_media_id": f"{split}_derived",
            "video_path": str(video),
            "scene_audio_path": str(audio),
            "question": "What is shown?",
            "answer": "fixture",
            "task_type": "audio_visual",
            "media_sha256": sha256_file(video),
            "audio_sha256": sha256_file(audio),
            "qa_origin": "human",
            "conversations": [
                {"from": "human", "value": "<image>\nWhat is shown?"},
                {"from": "gpt", "value": "fixture"},
            ],
        }
        manifest_rows = []
        for task_type in ("audio", "visual", "audio_visual"):
            item = dict(row)
            item["id"] = f"{sample_id}_{task_type}"
            item["sample_id"] = f"{sample_id}_{task_type}"
            item["task_type"] = task_type
            manifest_rows.append(item)
        manifest = tmp_path / f"{split}_manifest.json"
        manifest.write_text(json.dumps(manifest_rows), encoding="utf-8")
        digest = sha256_file(manifest)
        manifest.with_suffix(".sha256").write_text(f"{digest}  {manifest.name}\n", encoding="ascii")
        manifests[split] = manifest
    exclusion = tmp_path / "exclusion.json"
    exclusion.write_text("[]\n", encoding="utf-8")
    output = tmp_path / "gate.json"
    audit = tmp_path / "audit.json"
    script = SCRIPTS / "validate_stage2_training_manifests.py"
    proc = subprocess.run([
        sys.executable, str(script),
        "--train", str(manifests["train"]),
        "--dev", str(manifests["dev"]),
        "--reserve", str(manifests["reserve"]),
        "--exclude-manifest", str(exclusion),
        "--output", str(output),
        "--train-audit-output", str(audit),
        "--require-train", "1",
        "--require-dev", "1",
        "--require-reserve", "1",
        "--min-dev-sources", "1",
        "--skip-runtime-smoke",
    ], env={**__import__("os").environ, "PYTHONPATH": f"{SCRIPTS}:{SCRIPTS.parent}"},
       text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(output.read_text())["status"] == "PASS"
    assert json.loads(audit.read_text())["status"] == "PASS"


def test_finalize_stage2_gate_requires_launcher_pass_markers(tmp_path):
    gate_path = tmp_path / "gate.json"
    audit_path = tmp_path / "audit.json"
    log_path = tmp_path / "launcher.log"
    report_path = tmp_path / "report.md"
    build_path = tmp_path / "build.json"
    dedup_path = tmp_path / "dedup.json"
    gate_path.write_text(json.dumps({
        "status": "PASS",
        "error_count": 0,
        "full_decode": True,
        "video_audio_decode_rate": 1.0,
        "scene_audio_path_valid_rate": 1.0,
        "split_physical_media_count": {"train": 1000, "dev": 300, "reserve": 120},
        "split_qa_count": {"train": 10, "dev": 3, "reserve": 2},
        "task_distribution": {
            split: {"audio": 1, "visual": 1, "audio_visual": 1}
            for split in ("train", "dev", "reserve")
        },
        "source_distribution": {
            split: {"AVE": 1, "AVUT": 1}
            for split in ("train", "dev", "reserve")
        },
    }), encoding="utf-8")
    audit_path.write_text(json.dumps({"status": "PASS", "error_count": 0}), encoding="utf-8")
    build_path.write_text(json.dumps({
        "status": "PASS",
        "manifest_sha256": {
            split: f"{split}_sha" for split in ("train", "dev", "reserve")
        },
        "input_sha256": {"source": "locked"},
    }), encoding="utf-8")
    dedup_path.write_text(json.dumps({
        "status": "PASS",
        "pre_dedup_physical_media_count": 1420,
        "post_dedup_physical_media_count": 1420,
        "dropped_group_count": 0,
    }), encoding="utf-8")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["manifest_sha256"] = {
        split: {"sha256": f"{split}_sha"} for split in ("train", "dev", "reserve")
    }
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    log_path.write_text(
        "训练 manifest 内容门禁：PASS\n"
        "AS-M4 BEATs 启动器内容门禁：PASS（仅检查，不启动训练）\n",
        encoding="utf-8",
    )
    result = finalize(
        gate_path, audit_path, log_path, report_path, build_path, dedup_path,
    )
    assert result["launcher_content_gate"]["status"] == "PASS"
    assert result["training_started"] is False
    assert "**PASS**" in report_path.read_text(encoding="utf-8")
