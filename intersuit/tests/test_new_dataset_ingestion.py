from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
import sys
sys.path.insert(0, str(SCRIPTS))

from download_new_dataset_allowlist import _download
from new_dataset_common import replace_music_template, sha256_file, task_type_from_music
from validate_new_dataset_media import validate_entry
from validate_training_data_leakage import audit_manifest


def make_media(tmp_path: Path, *, audio: bool = True) -> Path:
    out = tmp_path / ("with_audio.mp4" if audio else "no_audio.mp4")
    command = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=32x32:rate=5",
               "-t", "1"]
    if audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-shortest"]
    command += ["-c:v", "mpeg4"]
    if audio:
        command += ["-c:a", "aac"]
    command += [str(out)]
    assert subprocess.run(command, check=False).returncode == 0
    return out


def test_corrupt_and_no_audio_media_rejected(tmp_path):
    corrupt = tmp_path / "bad.mp4"
    corrupt.write_bytes(b"not media")
    base = {"candidate_id": "x", "source_dataset": "AVUT", "video_id": "1", "youtube_id": "yt"}
    bad = validate_entry({**base, "path": str(corrupt)}, set(), set(), set(), 2)
    assert bad["validation_status"] == "rejected"
    no_audio = make_media(tmp_path, audio=False)
    result = validate_entry({**base, "path": str(no_audio)}, set(), set(), set(), 2)
    assert "missing audio stream" in " ".join(result["reasons"]) or result["validation_status"] == "rejected"


def test_download_is_idempotent_and_rejects_wrong_sha(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    source.write_bytes(b"hello")
    target = tmp_path / "out.bin"
    calls = []

    class Result:
        returncode = 0
        stderr = ""

    def fake_run(command, **kwargs):
        calls.append(command)
        target.with_suffix(".bin.part").write_bytes(source.read_bytes())
        return Result()

    monkeypatch.setattr("download_new_dataset_allowlist.subprocess.run", fake_run)
    assert _download("https://example.test/a", target, expected_bytes=5, timeout=1, retries=0)[0] == "downloaded"
    assert "-C" in calls[0] and "-" in calls[0]
    assert _download("https://example.test/a", target, expected_bytes=5, timeout=1, retries=0)[0] == "skipped"
    assert len(calls) == 1


def test_non_allowlist_artifact_rejected(tmp_path):
    source = tmp_path / "ordinary_manifest.json"
    source.write_text(json.dumps({"entries": []}), encoding="utf-8")
    script = SCRIPTS / "download_new_dataset_allowlist.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--allowlist", str(source), "--output-root", str(tmp_path / "q")],
        text=True, capture_output=True,
    )
    assert proc.returncode != 0
    assert "不是预检白名单" in proc.stderr


def test_leakage_gate_rejects_media_hash_collision_and_missing_fields(tmp_path):
    media = make_media(tmp_path)
    audio = tmp_path / "audio.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(media), "-vn", "-ar", "16000", "-ac", "1", str(audio)], check=True)
    digest = sha256_file(media)
    row = {
        "sample_id": "s", "source_dataset": "AVUT", "source_revision": "r", "video_id": "1",
        "youtube_id": "yt", "video_path": str(media), "scene_audio_path": str(audio),
        "question": "q", "answer": "a", "task_type": "audio",
        "media_sha256": digest, "audio_sha256": sha256_file(audio),
    }
    manifest = tmp_path / "train.json"
    manifest.write_text(json.dumps([row]), encoding="utf-8")
    report = audit_manifest(manifest, [], exclude_media_roots=[tmp_path])
    assert report["status"] == "FAIL"
    assert report["media_sha256_overlap_count"] >= 1
    incomplete = dict(row)
    incomplete.pop("audio_sha256")
    manifest.write_text(json.dumps([incomplete]), encoding="utf-8")
    assert audit_manifest(manifest, [])["status"] == "FAIL"


def test_train_dev_physical_media_leak_rejected(tmp_path):
    media = make_media(tmp_path)
    audio = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(media), "-vn", "-ar", "16000", "-ac", "1", str(audio)], check=True)
    row = {
        "sample_id": "s", "source_dataset": "AVUT", "source_revision": "r", "video_id": "1",
        "youtube_id": "yt", "video_path": str(media), "scene_audio_path": str(audio),
        "question": "q", "answer": "a", "task_type": "audio",
        "media_sha256": sha256_file(media), "audio_sha256": sha256_file(audio),
    }
    train = tmp_path / "train.json"
    dev = tmp_path / "dev.json"
    train.write_text(json.dumps([row]), encoding="utf-8")
    dev.write_text(json.dumps([{**row, "sample_id": "dev"}]), encoding="utf-8")
    report = audit_manifest(train, [], dev_manifest=dev)
    assert report["status"] == "FAIL"
    assert report["error_counts"]["train_dev_media_sha256_overlap"] == 1


def test_music_frozen_field_conversion_rules():
    assert replace_music_template("Is <Object> before <FL>?", ["violin", "flute"]) == "Is violin before flute?"
    assert task_type_from_music(["Audio-Visual", "Temporal"]) == "audio_visual"


def test_launcher_blocks_before_training_process(tmp_path):
    manifest = tmp_path / "train.json"
    manifest.write_text("[]", encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"status": "FAIL"}), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_as_m4_beats_stage.sh"
    proc = subprocess.run(
        ["bash", str(script), "12k-smoke"],
        cwd=script.parent.parent,
        env={**__import__("os").environ, "DATA_PATH": str(manifest), "TRAIN_MANIFEST_AUDIT": str(audit)},
        text=True, capture_output=True,
    )
    assert proc.returncode != 0
    assert "审计" in proc.stderr
