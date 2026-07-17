#!/usr/bin/env python
"""验证 LLP/AVE projector 候选窗口的本地媒体可用性。

本脚本只扫描本地文件并调用 ffprobe/ffmpeg 做解码检查，不下载媒体、
不提取 BEATs/M4 特征、不训练 projector，也不修改 Gate 或动态窗口路径。
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"
DEFAULT_CANDIDATES = INTERSUIT_ROOT / "harness/artifacts/audio_visual_event_manifests/projector_positive_candidates.jsonl"
DEFAULT_DATASET_ROOT = INTERSUIT_ROOT / "datasets"
DEFAULT_OUTPUT_ROOT = INTERSUIT_ROOT / "harness/artifacts/audio_visual_event_media_validation"
VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".flv", ".m4v"}
FAIL_MISSING = "missing_file"
FAIL_NO_VIDEO = "no_video"
FAIL_NO_AUDIO = "no_audio"
FAIL_DECODE = "decode_failed"
FAIL_TIMESTAMP = "invalid_timestamp"
FAIL_DURATION = "duration_too_short"
FAIL_EMPTY_AUDIO = "empty_audio"


@dataclass(frozen=True)
class MediaValidationRecord:
    dataset: str
    sample_id: str
    video_id: str
    window_start: float
    window_end: float
    event_label: str | None
    split: str
    media_path: str | None
    valid: bool
    failure_reason: str | None
    duration_sec: float | None
    has_video: bool
    has_audio: bool
    audio_non_empty: bool | None
    decode_ok: bool
    naming_rule: str | None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"候选 manifest 不存在：{path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 第 {line_number} 行非法：{path}: {exc}") from exc
    if not rows:
        raise ValueError(f"候选 manifest 为空：{path}")
    return rows


def infer_video_id(dataset: str, sample_id: str) -> str:
    if dataset == "LLP":
        parts = sample_id.rsplit("_", 2)
        if len(parts) == 3 and parts[-1].replace(".", "", 1).isdigit() and parts[-2].replace(".", "", 1).isdigit():
            return parts[0]
        return sample_id[:11]
    return sample_id


def candidate_stems(dataset: str, sample_id: str) -> list[tuple[str, str]]:
    video_id = infer_video_id(dataset, sample_id)
    stems = [("exact_sample_id", sample_id)]
    if video_id != sample_id:
        stems.append(("dataset_video_id", video_id))
    if dataset == "LLP" and sample_id[:11] != video_id:
        stems.append(("llp_first_11_chars", sample_id[:11]))
    deduped: list[tuple[str, str]] = []
    seen = set()
    for rule, stem in stems:
        if stem not in seen:
            deduped.append((rule, stem))
            seen.add(stem)
    return deduped


def scan_media_files(dataset_root: Path) -> dict[str, list[Path]]:
    media: dict[str, list[Path]] = {"LLP": [], "AVE": []}
    for dataset in media:
        root = dataset_root / dataset
        if not root.exists():
            continue
        files = [path.resolve() for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES]
        media[dataset] = sorted(files)
    return media


def build_media_index(media_files: dict[str, list[Path]]) -> dict[str, dict[str, Path]]:
    index: dict[str, dict[str, Path]] = {}
    for dataset, files in media_files.items():
        dataset_index: dict[str, Path] = {}
        for path in files:
            dataset_index.setdefault(path.stem, path)
        index[dataset] = dataset_index
    return index


def find_media_path(record: dict[str, Any], media_index: dict[str, dict[str, Path]]) -> tuple[Path | None, str | None]:
    dataset = str(record["dataset"])
    sample_id = str(record["sample_id"])
    dataset_index = media_index.get(dataset, {})
    for rule, stem in candidate_stems(dataset, sample_id):
        if stem in dataset_index:
            return dataset_index[stem], rule
    return None, None


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def ffprobe_media(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        return {"probe_ok": False, "error": result.stderr.strip()[:500]}
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {"probe_ok": False, "error": f"ffprobe JSON 解析失败：{exc}"}
    streams = payload.get("streams", [])
    format_info = payload.get("format", {})
    duration = None
    if format_info.get("duration") is not None:
        duration = float(format_info["duration"])
    has_video = any(stream.get("codec_type") == "video" for stream in streams)
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    return {"probe_ok": True, "duration_sec": duration, "has_video": has_video, "has_audio": has_audio}


def ffmpeg_decode_ok(path: Path) -> tuple[bool, str]:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-f",
        "null",
        "-",
    ]
    result = run_command(cmd)
    return result.returncode == 0, result.stderr.strip()[:500]


def ffmpeg_audio_non_empty(path: Path) -> tuple[bool | None, str]:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        return None, result.stderr.strip()[:500]
    stderr = result.stderr
    if "max_volume: -inf" in stderr or "mean_volume: -inf" in stderr:
        return False, ""
    return True, ""


def validate_one(record: dict[str, Any], media_index: dict[str, dict[str, Path]], *, decode: bool) -> MediaValidationRecord:
    dataset = str(record["dataset"])
    sample_id = str(record["sample_id"])
    video_id = infer_video_id(dataset, sample_id)
    window_start = float(record["window_start"])
    window_end = float(record["window_end"])
    media_path, naming_rule = find_media_path(record, media_index)
    if media_path is None:
        return MediaValidationRecord(dataset, sample_id, video_id, window_start, window_end, record.get("event_label"), str(record.get("split", "")), None, False, FAIL_MISSING, None, False, False, None, False, None)

    probe = ffprobe_media(media_path)
    if not probe.get("probe_ok"):
        return MediaValidationRecord(dataset, sample_id, video_id, window_start, window_end, record.get("event_label"), str(record.get("split", "")), str(media_path), False, FAIL_DECODE, None, False, False, None, False, naming_rule)
    duration = probe.get("duration_sec")
    has_video = bool(probe.get("has_video"))
    has_audio = bool(probe.get("has_audio"))
    if not has_video:
        return MediaValidationRecord(dataset, sample_id, video_id, window_start, window_end, record.get("event_label"), str(record.get("split", "")), str(media_path), False, FAIL_NO_VIDEO, duration, False, has_audio, None, False, naming_rule)
    if not has_audio:
        return MediaValidationRecord(dataset, sample_id, video_id, window_start, window_end, record.get("event_label"), str(record.get("split", "")), str(media_path), False, FAIL_NO_AUDIO, duration, has_video, False, None, False, naming_rule)
    if window_end <= window_start or window_start < 0:
        return MediaValidationRecord(dataset, sample_id, video_id, window_start, window_end, record.get("event_label"), str(record.get("split", "")), str(media_path), False, FAIL_TIMESTAMP, duration, has_video, has_audio, None, False, naming_rule)
    if duration is None or duration + 1e-3 < window_end:
        return MediaValidationRecord(dataset, sample_id, video_id, window_start, window_end, record.get("event_label"), str(record.get("split", "")), str(media_path), False, FAIL_DURATION, duration, has_video, has_audio, None, False, naming_rule)
    if not decode:
        return MediaValidationRecord(dataset, sample_id, video_id, window_start, window_end, record.get("event_label"), str(record.get("split", "")), str(media_path), True, None, duration, has_video, has_audio, None, True, naming_rule)

    decode_ok, _ = ffmpeg_decode_ok(media_path)
    if not decode_ok:
        return MediaValidationRecord(dataset, sample_id, video_id, window_start, window_end, record.get("event_label"), str(record.get("split", "")), str(media_path), False, FAIL_DECODE, duration, has_video, has_audio, None, False, naming_rule)
    audio_non_empty, _ = ffmpeg_audio_non_empty(media_path)
    if audio_non_empty is False:
        return MediaValidationRecord(dataset, sample_id, video_id, window_start, window_end, record.get("event_label"), str(record.get("split", "")), str(media_path), False, FAIL_EMPTY_AUDIO, duration, has_video, has_audio, False, True, naming_rule)
    if audio_non_empty is None:
        return MediaValidationRecord(dataset, sample_id, video_id, window_start, window_end, record.get("event_label"), str(record.get("split", "")), str(media_path), False, FAIL_DECODE, duration, has_video, has_audio, None, True, naming_rule)
    return MediaValidationRecord(dataset, sample_id, video_id, window_start, window_end, record.get("event_label"), str(record.get("split", "")), str(media_path), True, None, duration, has_video, has_audio, True, True, naming_rule)


def select_small_batch(records: list[dict[str, Any]], per_dataset: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_video: set[tuple[str, str]] = set()
    for record in records:
        dataset = str(record["dataset"])
        if dataset not in {"LLP", "AVE"} or counts[dataset] >= per_dataset:
            continue
        key = (dataset, infer_video_id(dataset, str(record["sample_id"])))
        if key in seen_video:
            continue
        selected.append(record)
        seen_video.add(key)
        counts[dataset] += 1
    return selected


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[MediaValidationRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def summarize_media_dirs(media_files: dict[str, list[Path]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for dataset, files in media_files.items():
        dirs = Counter(str(path.parent) for path in files)
        summary[dataset] = {
            "video_count": len(files),
            "media_dirs": [{"path": path, "count": count} for path, count in dirs.most_common()],
            "example_files": [str(path) for path in files[:5]],
        }
    return summary


def summarize_records(records: list[MediaValidationRecord], media_files: dict[str, list[Path]]) -> dict[str, Any]:
    failure_counts = Counter(record.failure_reason for record in records if not record.valid)
    dataset_counts: dict[str, Any] = {}
    for dataset in ("LLP", "AVE"):
        subset = [record for record in records if record.dataset == dataset]
        dataset_counts[dataset] = {
            "checked": len(subset),
            "valid": sum(1 for record in subset if record.valid),
            "missing": sum(1 for record in subset if record.failure_reason == FAIL_MISSING),
            "invalid": sum(1 for record in subset if not record.valid),
        }
    matched_paths = [record.media_path for record in records if record.media_path]
    duplicate_videos = sum(1 for _, count in Counter(matched_paths).items() if count > 1)
    return {
        "local_media": summarize_media_dirs(media_files),
        "checked_count": len(records),
        "valid_count": sum(1 for record in records if record.valid),
        "invalid_count": sum(1 for record in records if not record.valid),
        "dataset_counts": dataset_counts,
        "failure_counts": dict(failure_counts),
        "duplicate_video_count": duplicate_videos,
        "unique_matched_video_count": len(set(matched_paths)),
        "naming_rule_counts": dict(Counter(record.naming_rule for record in records if record.naming_rule)),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    local = summary["scan"]["local_media"]
    full = summary["full_validation"]
    lines = [
        "# LLP/AVE 本地媒体验证报告",
        "",
        "本轮只扫描和解码本地媒体，不自动联网下载，不提取特征，不训练 projector，不修改 Gate 或动态窗口。",
        "",
        "## 本地视频扫描",
        f"- LLP 本地视频数量：{local['LLP']['video_count']}",
        f"- AVE 本地视频数量：{local['AVE']['video_count']}",
        f"- LLP 实际媒体目录：`{local['LLP']['media_dirs']}`",
        f"- AVE 实际媒体目录：`{local['AVE']['media_dirs']}`",
        "",
        "## 命名匹配规则",
        "- LLP：优先匹配完整 `sample_id`，其次匹配去掉 `_start_end` 后的视频 ID。",
        "- AVE：匹配 `Annotations.txt` 中的 `video_id`。",
        f"- 实际命中规则统计：`{full['naming_rule_counts']}`",
        "",
        "## 小批量 20+20 验证",
        f"- 检查数量：{summary['small_validation']['checked_count']}",
        f"- 可用数量：{summary['small_validation']['valid_count']}",
        f"- 失败统计：`{summary['small_validation']['failure_counts']}`",
        "",
        "## 400 条候选验证",
        f"- 匹配/可用数量：{full['valid_count']}",
        f"- 缺失/不可用数量：{full['invalid_count']}",
        f"- LLP 统计：`{full['dataset_counts']['LLP']}`",
        f"- AVE 统计：`{full['dataset_counts']['AVE']}`",
        f"- 失败原因：`{full['failure_counts']}`",
        f"- 重复视频数量：{full['duplicate_video_count']}",
        f"- 最终可用事件窗口数量：{full['valid_count']}",
        "",
        "## 阶段判断",
    ]
    if full["valid_count"] == 0:
        lines.append("- 停止：当前 LLP/AVE 本地媒体不足，不能进入 BEATs/M4 窗口特征提取。")
    else:
        lines.append("- 可以仅对 `projector_media_valid_manifest.jsonl` 中的本地可用媒体进入下一步特征提取。")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidates = load_jsonl(Path(args.candidates).resolve())
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    media_files = scan_media_files(dataset_root)
    media_index = build_media_index(media_files)

    small_records = select_small_batch(candidates, args.small_per_dataset)
    small_validation = [validate_one(record, media_index, decode=args.decode) for record in small_records]
    full_validation = [validate_one(record, media_index, decode=args.decode) for record in candidates]
    valid = [record for record in full_validation if record.valid]
    invalid = [record for record in full_validation if not record.valid]

    valid_path = output_root / "projector_media_valid_manifest.jsonl"
    invalid_path = output_root / "projector_media_invalid_manifest.jsonl"
    summary_path = output_root / "media_validation_summary.json"
    report_path = output_root / "media_validation_report.md"
    small_path = output_root / "small_batch_media_validation.jsonl"
    write_jsonl(valid_path, valid)
    write_jsonl(invalid_path, invalid)
    write_jsonl(small_path, small_validation)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidates_path": str(Path(args.candidates).resolve()),
        "dataset_root": str(dataset_root),
        "decode_enabled": bool(args.decode),
        "scan": summarize_records([], media_files),
        "small_validation": summarize_records(small_validation, media_files),
        "full_validation": summarize_records(full_validation, media_files),
        "paths": {
            "valid_manifest": str(valid_path),
            "invalid_manifest": str(invalid_path),
            "small_batch": str(small_path),
            "summary": str(summary_path),
            "report": str(report_path),
        },
    }
    write_json(summary_path, summary)
    write_report(report_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--small-per-dataset", type=int, default=20)
    parser.add_argument("--decode", action="store_true", help="对匹配成功的媒体执行 ffmpeg 完整解码和空音轨检查")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps({"ok": True, "summary": summary["paths"]["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
