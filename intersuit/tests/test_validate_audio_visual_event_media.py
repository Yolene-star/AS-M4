"""CPU tests for local LLP/AVE media validation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "validate_audio_visual_event_media.py"
SPEC = importlib.util.spec_from_file_location("validate_audio_visual_event_media", SCRIPT_PATH)
media = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = media
SPEC.loader.exec_module(media)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _candidate(dataset: str, sample_id: str, start: float = 0.0, end: float = 1.0) -> dict:
    return {
        "dataset": dataset,
        "sample_id": sample_id,
        "video_path": None,
        "window_start": start,
        "window_end": end,
        "modality_role": "audio_visual",
        "event_label": "Dog",
        "audio_required": True,
        "visible_event_present": True,
        "split": "train",
        "source_annotation": "fixture",
    }


def test_infer_video_id_supports_llp_segment_names():
    assert media.infer_video_id("LLP", "BjCEufrlXm4_20_30") == "BjCEufrlXm4"
    assert media.infer_video_id("AVE", "RUhOCu3LNXM") == "RUhOCu3LNXM"


def test_find_media_path_uses_dataset_video_id_for_llp(tmp_path):
    path = tmp_path / "LLP" / "data" / "LLP_dataset" / "video" / "BjCEufrlXm4.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not real media")
    media_files = media.scan_media_files(tmp_path)
    index = media.build_media_index(media_files)

    found, rule = media.find_media_path(_candidate("LLP", "BjCEufrlXm4_20_30"), index)

    assert found == path.resolve()
    assert rule == "dataset_video_id"


def test_validate_one_reports_missing_file(tmp_path):
    index = media.build_media_index(media.scan_media_files(tmp_path))

    result = media.validate_one(_candidate("AVE", "missing"), index, decode=False)

    assert result.valid is False
    assert result.failure_reason == media.FAIL_MISSING
    assert result.media_path is None


def test_run_writes_valid_and_invalid_manifests(monkeypatch, tmp_path):
    llp_path = tmp_path / "datasets" / "LLP" / "data" / "LLP_dataset" / "video" / "BjCEufrlXm4.mp4"
    ave_path = tmp_path / "datasets" / "AVE" / "videos" / "RUhOCu3LNXM.mp4"
    llp_path.parent.mkdir(parents=True)
    ave_path.parent.mkdir(parents=True)
    llp_path.write_bytes(b"fake")
    ave_path.write_bytes(b"fake")
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(
        candidates,
        [
            _candidate("LLP", "BjCEufrlXm4_20_30", 0.0, 1.0),
            _candidate("AVE", "RUhOCu3LNXM", 0.0, 1.0),
            _candidate("AVE", "missing_video", 0.0, 1.0),
            _candidate("AVE", "RUhOCu3LNXM", 3.0, 4.0),
        ],
    )

    def fake_probe(path):
        return {"probe_ok": True, "duration_sec": 2.0, "has_video": True, "has_audio": True}

    monkeypatch.setattr(media, "ffprobe_media", fake_probe)
    args = argparse.Namespace(
        candidates=str(candidates),
        dataset_root=str(tmp_path / "datasets"),
        output_root=str(tmp_path / "out"),
        small_per_dataset=20,
        decode=False,
    )

    summary = media.run(args)

    assert summary["scan"]["local_media"]["LLP"]["video_count"] == 1
    assert summary["scan"]["local_media"]["AVE"]["video_count"] == 1
    assert summary["full_validation"]["valid_count"] == 2
    assert summary["full_validation"]["failure_counts"][media.FAIL_MISSING] == 1
    assert summary["full_validation"]["failure_counts"][media.FAIL_DURATION] == 1
    assert (tmp_path / "out" / "projector_media_valid_manifest.jsonl").is_file()
    assert (tmp_path / "out" / "media_validation_report.md").is_file()
