#!/usr/bin/env python
"""新增 AVUT/MUSIC 数据落地的共享校验与转换工具。"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_id(value: Any) -> str:
    return str(value).strip().casefold()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
        return records
    data = load_json(path)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("entries", "samples", "data", "train", "selected"):
            rows = data.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
        return [data]
    return []


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_probe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(path),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode:
        raise ValueError(f"ffprobe failed: {proc.stderr.strip()[:300]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("ffprobe output is not JSON") from exc


def media_metadata(path: Path) -> dict[str, Any]:
    probe = run_probe(path)
    streams = probe.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise ValueError("missing video stream")
    if audio is None:
        raise ValueError("missing audio stream")
    duration = float((probe.get("format") or {}).get("duration") or 0.0)
    if duration <= 0:
        duration = max(float(video.get("duration") or 0), float(audio.get("duration") or 0))
    if duration <= 0:
        raise ValueError("non-positive duration")
    return {
        "duration": duration,
        "video_codec": video.get("codec_name"),
        "audio_codec": audio.get("codec_name"),
        "audio_sample_rate": int(audio.get("sample_rate") or 0),
        "audio_channels": int(audio.get("channels") or 0),
    }


def audio_probe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=duration,sample_rate,channels",
        "-of", "json", str(path),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode:
        raise ValueError(f"audio ffprobe failed: {proc.stderr.strip()[:300]}")
    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise ValueError("missing decodable audio stream")
    return streams[0]


def extract_wav(media: Path, wav: Path) -> dict[str, Any]:
    wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(media),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode or not wav.is_file() or wav.stat().st_size == 0:
        raise ValueError(f"audio extraction failed: {proc.stderr.strip()[:300]}")
    meta = audio_probe(wav)
    if int(meta.get("sample_rate") or 0) != 16000 or int(meta.get("channels") or 0) != 1:
        raise ValueError("extracted audio is not 16 kHz mono")
    return {
        "audio_sha256": sha256_file(wav),
        "sample_rate": 16000,
        "channels": 1,
        "duration": float(meta.get("duration") or 0.0),
        "sample_count": int(round(float(meta.get("duration") or 0.0) * 16000)),
    }


def exclusion_sets(paths: Iterable[Path], media_roots: Iterable[Path] = ()) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    hashes: set[str] = set()
    for path in paths:
        if not path or not Path(path).is_file():
            continue
        rows = load_records(Path(path))
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in ("video_id", "youtube_id", "source_video_id", "id"):
                if row.get(key) not in (None, ""):
                    ids.add(canonical_id(row[key]))
            for key in ("media_sha256", "video_sha256", "sha256"):
                if row.get(key):
                    hashes.add(canonical_id(row[key]))
            if row.get("artifact_kind") == "new_dataset_exclusion_inventory":
                ids.update(canonical_id(value) for value in row.get("ids", []))
                hashes.update(canonical_id(value) for value in row.get("media_sha256", []))
    media_suffixes = {".mp4", ".webm", ".mkv", ".mov", ".avi"}
    for root in media_roots:
        root = Path(root)
        files = [root] if root.is_file() else root.rglob("*") if root.is_dir() else []
        for path in files:
            if path.is_file() and path.suffix.casefold() in media_suffixes:
                hashes.add(sha256_file(path))
                ids.add(canonical_id(path.stem.removesuffix("_flip")))
    return ids, hashes


def task_type_from_avut(value: Any) -> str:
    text = str(value or "").casefold()
    if any(token in text for token in ("character matching", "object matching", "ocr matching")):
        return "audio_visual"
    return "audio"


def task_type_from_music(value: Any) -> str:
    raw = value[0] if isinstance(value, list) and value else value
    text = str(raw or "").casefold()
    if "audio_visual" in text or "audio-visual" in text or text in {"av", "audio visual"}:
        return "audio_visual"
    if "audio" in text:
        return "audio"
    return "visual"


def replace_music_template(question: str, values: list[Any]) -> str:
    result = str(question)
    for value in values:
        replaced = False
        for token in ("<Object>", "<FL>"):
            if token in result:
                result = result.replace(token, str(value), 1)
                replaced = True
                break
        if not replaced:
            raise ValueError("templ_values count exceeds question placeholders")
    if "<Object>" in result or "<FL>" in result:
        raise ValueError("unresolved MUSIC question placeholder")
    return result
