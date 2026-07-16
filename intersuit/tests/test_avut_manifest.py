import csv
import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from avut_common import build_accept_contains  # noqa: E402
from prepare_avut_manifest import build_manifest  # noqa: E402


FIELDS = ["sample_id", "muted_answerable", "audio_answerable", "keep", "review_level", "manual_note"]


def candidates(path: Path) -> Path:
    payload = {"selected": [{"sample_id": "avut_0001", "video_path": "datasets/AVUT/smoke_videos/a.mp4", "question": "What is heard?", "answer": "dog barking", "choices": {"A": "dog barking"}, "source_video_id": 1, "source_qa_id": 2}]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def review(path: Path, **updates) -> Path:
    row = {"sample_id": "avut_0001", "muted_answerable": "no", "audio_answerable": "unknown", "keep": "yes", "review_level": "muted_only", "manual_note": "仅完成静音审核"}
    row.update(updates)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)
    return path


def test_review_csv_incomplete(tmp_path):
    video_root = tmp_path / "videos"
    video_root.mkdir()
    (video_root / "a.mp4").touch()
    with pytest.raises(ValueError, match="尚未填写完整"):
        build_manifest(candidates(tmp_path / "c.json"), review(tmp_path / "r.csv", keep=""), video_root)


def test_keep_no_is_excluded(tmp_path):
    video_root = tmp_path / "videos"
    video_root.mkdir()
    (video_root / "a.mp4").touch()
    with pytest.raises(ValueError, match="没有 keep=yes"):
        build_manifest(candidates(tmp_path / "c.json"), review(tmp_path / "r.csv", keep="no"), video_root)


def test_accept_contains_is_narrow():
    assert build_accept_contains("dog barking.") == ["dog barking"]


def test_manifest_video_missing(tmp_path):
    video_root = tmp_path / "videos"
    video_root.mkdir()
    with pytest.raises(FileNotFoundError, match="视频不存在"):
        build_manifest(candidates(tmp_path / "c.json"), review(tmp_path / "r.csv"), video_root)


def test_muted_only_manifest_flags_and_conversations(tmp_path):
    video_root = tmp_path / "videos"
    video_root.mkdir()
    (video_root / "a.mp4").touch()
    manifest = build_manifest(candidates(tmp_path / "c.json"), review(tmp_path / "r.csv"), video_root)
    item = manifest[0]
    assert item["muted_answerable"] is False
    assert item["audio_answerable"] is None
    assert item["review_level"] == "muted_only"
    assert item["audio_required_candidate"] is True
    assert item["manually_verified_audio_required"] is False
    assert item["scene_audio_path"] == str(video_root / "a.mp4")
    assert item["generation_mode"] == "generate"
    assert "context" not in item
    assert "new_query_pos" not in item
    assert item["conversations"][0]["from"] == "human"


def test_non_audio_question_is_rejected(tmp_path):
    video_root = tmp_path / "videos"
    video_root.mkdir()
    (video_root / "a.mp4").touch()
    path = candidates(tmp_path / "c.json")
    payload = json.loads(path.read_text())
    payload["selected"][0]["question"] = "What color is the shirt?"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="没有显式声音语义"):
        build_manifest(path, review(tmp_path / "r.csv"), video_root)
