"""CPU tests for AVE Hugging Face pilot preparation helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "prepare_ave_hf_pilot.py"
SPEC = importlib.util.spec_from_file_location("prepare_ave_hf_pilot", SCRIPT_PATH)
pilot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pilot
SPEC.loader.exec_module(pilot)


def test_select_diverse_rows_round_robins_labels_and_deduplicates_youtube_ids():
    rows = [
        {"label": 0, "youtube_id": "a"},
        {"label": 0, "youtube_id": "a"},
        {"label": 0, "youtube_id": "b"},
        {"label": 1, "youtube_id": "c"},
        {"label": 1, "youtube_id": "d"},
        {"label": 2, "youtube_id": "e"},
    ]

    selected = pilot.select_diverse_rows(rows, limit=5)

    assert [row["youtube_id"] for row in selected] == ["a", "c", "e", "d", "b"]
    assert len({row["youtube_id"] for row in selected}) == len(selected)


def test_select_diverse_rows_excludes_existing_youtube_ids():
    rows = [
        {"label": 0, "youtube_id": "a"},
        {"label": 0, "youtube_id": "b"},
        {"label": 1, "youtube_id": "c"},
        {"label": 1, "youtube_id": "d"},
    ]

    selected = pilot.select_diverse_rows(rows, limit=3, exclude_ids={"a", "c"})

    assert [row["youtube_id"] for row in selected] == ["b", "d"]


def test_normalizers_accept_ave_hf_schema():
    row = {
        "youtube_id": "abc123",
        "start_seconds": 7,
        "label": 8,
        "video": {"path": "abc123.mp4", "bytes": b"video"},
        "audio": {"path": "abc123.wav", "bytes": b"audio"},
    }

    assert pilot.normalize_youtube_id(row, 0) == "abc123"
    assert pilot.normalize_start(row) == 7.0
    assert pilot.normalize_label(row) == "8"
    assert pilot.extract_video_source(row) == (b"video", "abc123.mp4")
    assert pilot.extract_audio_source(row) == (b"audio", "abc123.wav")


def test_load_exclude_ids_unions_multiple_manifests(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps({"youtube_id": "a"}) + "\n", encoding="utf-8")
    second.write_text(
        json.dumps({"sample_id": "b"}) + "\n" + json.dumps({"youtube_id": "a"}) + "\n",
        encoding="utf-8",
    )

    paths = pilot.resolve_manifest_paths(f"{first},{second}")
    ids = pilot.load_exclude_ids(paths)

    assert paths == [first.resolve(), second.resolve()]
    assert ids == {"a", "b"}
