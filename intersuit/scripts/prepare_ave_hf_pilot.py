#!/usr/bin/env python
"""从 Hugging Face `mteb/AVE-Dataset` 准备 20 条 AVE pilot 媒体并校验。

本脚本只处理小批量样本，不修改 Gate、动态窗口、视频关注权重或 M4 主体。
默认保存原始视频、抽取 16 kHz mono wav、记录样本元数据，并输出 valid/invalid
JSONL 与中文报告。若数据源字段结构变化，脚本会清晰报错而不是静默伪造媒体。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"
DEFAULT_OUTPUT_ROOT = INTERSUIT_ROOT / "datasets" / "AVE_HF"
DATASET_NAME = "mteb/AVE-Dataset"
DEFAULT_TRAIN_SHARD = "data/train-00000-of-00011.parquet"


@dataclass(frozen=True)
class PilotRecord:
    index: int
    youtube_id: str
    start_seconds: float | None
    label: str | None
    split: str
    video_path: str | None
    audio_path: str | None
    valid: bool
    failure_reason: str | None
    video_duration_sec: float | None
    audio_duration_sec: float | None
    has_video: bool
    has_audio: bool
    audio_non_empty: bool | None
    source_keys: list[str]


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def require_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise FileNotFoundError(f"缺少必需工具：{missing}")


def first_present(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def normalize_youtube_id(row: dict[str, Any], index: int) -> str:
    value = first_present(row, ("youtube_id", "video_id", "id", "YTID", "ytid"))
    if value is None:
        video_value = first_present(row, ("video", "video_path", "path", "file"))
        if isinstance(video_value, dict):
            video_value = video_value.get("path") or video_value.get("bytes")
        if isinstance(video_value, str) and video_value:
            value = Path(video_value).stem
    if value is None:
        value = f"ave_hf_{index:05d}"
    return str(value).replace("/", "_")


def normalize_start(row: dict[str, Any]) -> float | None:
    value = first_present(row, ("start_seconds", "start", "event_start", "onset"))
    if value is None:
        return None
    return float(value)


def normalize_label(row: dict[str, Any]) -> str | None:
    value = first_present(row, ("label", "event_label", "category", "class", "event"))
    return None if value is None else str(value)


def extract_video_source(row: dict[str, Any]) -> tuple[bytes | None, str | None]:
    for key in ("video", "video_file", "mp4", "file", "path", "video_path"):
        if key not in row or row[key] is None:
            continue
        value = row[key]
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                return value["bytes"], value.get("path")
            if value.get("path"):
                return None, str(value["path"])
        if isinstance(value, (bytes, bytearray)):
            return bytes(value), None
        if isinstance(value, str):
            return None, value
    return None, None


def extract_audio_source(row: dict[str, Any]) -> tuple[bytes | None, str | None]:
    for key in ("audio", "audio_file", "wav", "audio_path"):
        if key not in row or row[key] is None:
            continue
        value = row[key]
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                return value["bytes"], value.get("path")
            if value.get("path"):
                return None, str(value["path"])
        if isinstance(value, (bytes, bytearray)):
            return bytes(value), None
        if isinstance(value, str):
            return None, value
    return None, None


def copy_or_write_video(row: dict[str, Any], output_path: Path) -> None:
    video_bytes, source_path = extract_video_source(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if video_bytes is not None:
        output_path.write_bytes(video_bytes)
        return
    if source_path is None:
        raise ValueError(f"无法从样本中识别视频字段，字段为：{sorted(row)}")
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"视频源文件不存在：{source_path}")
    shutil.copyfile(source, output_path)


def copy_or_write_audio(row: dict[str, Any], output_path: Path) -> None:
    audio_bytes, source_path = extract_audio_source(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if audio_bytes is not None:
        output_path.write_bytes(audio_bytes)
        return
    if source_path is None:
        raise ValueError(f"无法从样本中识别音频字段，字段为：{sorted(row)}")
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"音频源文件不存在：{source_path}")
    shutil.copyfile(source, output_path)


def ffprobe_streams(path: Path) -> dict[str, Any]:
    result = run_command(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    if result.returncode != 0:
        return {"ok": False, "error": result.stderr.strip()[:500]}
    payload = json.loads(result.stdout)
    duration = payload.get("format", {}).get("duration")
    streams = payload.get("streams", [])
    return {
        "ok": True,
        "duration_sec": float(duration) if duration is not None else None,
        "has_video": any(stream.get("codec_type") == "video" for stream in streams),
        "has_audio": any(stream.get("codec_type") == "audio" for stream in streams),
    }


def extract_audio(video_path: Path, audio_path: Path) -> tuple[bool, str]:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    result = run_command(["ffmpeg", "-v", "error", "-y", "-i", str(video_path), "-ac", "1", "-ar", "16000", str(audio_path)])
    return result.returncode == 0, result.stderr.strip()[:500]


def decode_full(video_path: Path, audio_path: Path) -> tuple[bool, bool]:
    video_result = run_command(["ffmpeg", "-v", "error", "-i", str(video_path), "-map", "0:v:0", "-f", "null", "-"])
    audio_result = run_command(["ffmpeg", "-v", "error", "-i", str(audio_path), "-f", "null", "-"])
    return video_result.returncode == 0, audio_result.returncode == 0


def audio_non_empty(audio_path: Path) -> bool | None:
    result = run_command(["ffmpeg", "-v", "error", "-i", str(audio_path), "-af", "volumedetect", "-f", "null", "-"])
    if result.returncode != 0:
        return None
    text = result.stderr
    if "max_volume: -inf" in text or "mean_volume: -inf" in text:
        return False
    return True


def select_diverse_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_label: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(row.get("label"), []).append(row)
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    labels = sorted(by_label, key=lambda value: str(value))
    while len(selected) < limit:
        progressed = False
        for label in labels:
            bucket = by_label[label]
            if not bucket:
                continue
            row = bucket.pop(0)
            youtube_id = normalize_youtube_id(row, len(selected))
            if youtube_id in seen_ids:
                continue
            selected.append(row)
            seen_ids.add(youtube_id)
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def load_pilot_rows(split: str, limit: int, parquet_file: str, selection: str) -> list[dict[str, Any]]:
    if split != "train":
        raise ValueError("当前 pilot 默认只支持 train split 的首个 parquet 分片")
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET_NAME, parquet_file, repo_type="dataset")
    table = pq.read_table(path)
    rows = [dict(row) for row in table.to_pylist()]
    if selection == "first":
        return rows[:limit]
    if selection == "diverse_label":
        return select_diverse_rows(rows, limit)
    raise ValueError(f"未知 selection：{selection}")


def validate_row(row: dict[str, Any], index: int, split: str, output_root: Path) -> PilotRecord:
    youtube_id = normalize_youtube_id(row, index)
    label = normalize_label(row)
    start_seconds = normalize_start(row)
    video_path = output_root / "videos" / f"{youtube_id}.mp4"
    audio_path = output_root / "audio_16k_mono" / f"{youtube_id}.wav"
    source_keys = sorted(row)
    try:
        copy_or_write_video(row, video_path)
        copy_or_write_audio(row, audio_path)
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            raise ValueError("zero_byte_video")
        if not audio_path.is_file() or audio_path.stat().st_size <= 0:
            raise ValueError("zero_byte_audio")
        probe = ffprobe_streams(video_path)
        if not probe.get("ok"):
            raise ValueError("video_probe_failed")
        duration = probe.get("duration_sec")
        has_video = bool(probe.get("has_video"))
        has_audio = bool(probe.get("has_audio"))
        if not has_video:
            raise ValueError("no_video")
        if not has_audio:
            raise ValueError("no_audio")
        if duration is None or not math.isfinite(duration) or duration <= 0:
            raise ValueError("invalid_duration")
        if start_seconds is not None and (start_seconds < 0 or start_seconds > duration + 1e-3):
            raise ValueError("invalid_start_seconds")
        audio_probe = ffprobe_streams(audio_path)
        audio_duration = audio_probe.get("duration_sec") if audio_probe.get("ok") else None
        video_decode_ok, audio_decode_ok = decode_full(video_path, audio_path)
        if not video_decode_ok:
            raise ValueError("video_decode_failed")
        if not audio_decode_ok:
            raise ValueError("audio_decode_failed")
        non_empty = audio_non_empty(audio_path)
        if non_empty is False:
            raise ValueError("empty_audio")
        if non_empty is None:
            raise ValueError("audio_volume_check_failed")
        return PilotRecord(index, youtube_id, start_seconds, label, split, str(video_path), str(audio_path), True, None, duration, audio_duration, True, True, non_empty, source_keys)
    except Exception as exc:
        return PilotRecord(index, youtube_id, start_seconds, label, split, str(video_path) if video_path.exists() else None, str(audio_path) if audio_path.exists() else None, False, str(exc), None, None, False, False, None, source_keys)


def write_jsonl(path: Path, rows: list[PilotRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# AVE Hugging Face {summary['requested_count']} 条媒体验证报告",
        "",
        f"本轮只下载和验证 `mteb/AVE-Dataset` 的 {summary['requested_count']} 条训练样本，不修改 Gate、动态窗口、视频关注权重或 M4 主体。",
        "",
        f"- 数据源：`{DATASET_NAME}`",
        f"- split：`{summary['split']}`",
        f"- parquet 分片：`{summary['parquet_file']}`",
        f"- 抽样方式：`{summary['selection']}`",
        f"- 请求数量：{summary['requested_count']}",
        f"- 有效数量：{summary['valid_count']}",
        f"- 无效数量：{summary['invalid_count']}",
        f"- 失败原因：`{summary['failure_counts']}`",
        f"- label 分布：`{summary['label_counts']}`",
        f"- 继续门槛：至少 {summary['threshold']} 条可正常解码",
        f"- 是否通过：{summary['passed_threshold']}",
        "",
        "## 阶段判断",
    ]
    if summary["passed_threshold"]:
        lines.append("- 通过当前媒体门槛，可以下一步进入样本划分与窗口特征提取准备。")
    else:
        lines.append("- 未通过 pilot 门槛，本轮停止；不要提取 BEATs/M4 窗口特征，也不要训练 projector。")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    require_tools()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    rows = load_pilot_rows(args.split, args.limit, args.parquet_file, args.selection)
    records = [validate_row(row, index, args.split, output_root) for index, row in enumerate(rows)]
    valid = [record for record in records if record.valid]
    invalid = [record for record in records if not record.valid]
    valid_path = output_root / "ave_hf_pilot_valid.jsonl"
    invalid_path = output_root / "ave_hf_pilot_invalid.jsonl"
    summary_path = output_root / "ave_hf_pilot_summary.json"
    report_path = output_root / "ave_hf_pilot_report.md"
    metadata_path = output_root / "ave_hf_pilot_metadata.jsonl"
    write_jsonl(valid_path, valid)
    write_jsonl(invalid_path, invalid)
    write_jsonl(metadata_path, records)
    threshold = args.threshold
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_NAME,
        "split": args.split,
        "parquet_file": args.parquet_file,
        "selection": args.selection,
        "requested_count": args.limit,
        "loaded_count": len(rows),
        "valid_count": len(valid),
        "invalid_count": len(invalid),
        "failure_counts": dict(Counter(record.failure_reason for record in invalid)),
        "label_counts": dict(Counter(record.label for record in valid)),
        "threshold": threshold,
        "passed_threshold": len(valid) >= threshold,
        "output_root": str(output_root),
        "paths": {
            "valid": str(valid_path),
            "invalid": str(invalid_path),
            "metadata": str(metadata_path),
            "summary": str(summary_path),
            "report": str(report_path),
        },
    }
    write_json(summary_path, summary)
    write_report(report_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--threshold", type=int, default=18)
    parser.add_argument("--parquet-file", default=DEFAULT_TRAIN_SHARD)
    parser.add_argument("--selection", choices=("first", "diverse_label"), default="first")
    parser.add_argument("--hf-endpoint", default="https://huggingface.co")
    parser.add_argument("--disable-xet", action="store_true", default=True)
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps({"ok": True, "summary": summary["paths"]["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
