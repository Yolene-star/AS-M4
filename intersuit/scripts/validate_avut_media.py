#!/usr/bin/env python
"""使用 ffprobe/ffmpeg 严格校验 AVUT 候选视频。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def validate_video(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"video_path": str(path), "video_exists": path.is_file(), "file_size": path.stat().st_size if path.is_file() else 0}
    if not path.is_file() or result["file_size"] <= 0:
        result.update({"valid": False, "error": "视频不存在或大小为 0"})
        return result
    probe = run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    if probe.returncode != 0:
        result.update({"valid": False, "error": f"ffprobe 失败：{probe.stderr.strip()}"})
        return result
    try:
        metadata = json.loads(probe.stdout)
    except json.JSONDecodeError as exc:
        result.update({"valid": False, "error": f"ffprobe 返回非法 JSON：{exc}"})
        return result
    streams = metadata.get("streams", [])
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    duration = float(metadata.get("format", {}).get("duration") or 0)
    audio_duration = float(audio_streams[0].get("duration") or duration) if audio_streams else 0.0
    decode_video = run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"]) if video_streams else None
    decode_audio = run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "null", "-"]) if audio_streams else None
    audio = audio_streams[0] if audio_streams else {}
    result.update({
        "has_video_stream": bool(video_streams), "has_audio_stream": bool(audio_streams), "duration": duration,
        "audio_duration": audio_duration, "audio_codec": audio.get("codec_name"), "audio_sample_rate": int(audio.get("sample_rate") or 0),
        "audio_channels": int(audio.get("channels") or 0), "av_duration_difference": abs(duration - audio_duration),
        "video_decodable": bool(decode_video and decode_video.returncode == 0), "audio_decodable": bool(decode_audio and decode_audio.returncode == 0),
    })
    finite = all(math.isfinite(value) for value in (duration, audio_duration, result["av_duration_difference"]))
    result["valid"] = bool(result["video_exists"] and video_streams and audio_streams and duration > 0 and result["audio_decodable"] and result["video_decodable"] and finite)
    if not result["valid"]:
        errors = []
        if not video_streams: errors.append("没有视频流")
        if not audio_streams: errors.append("没有音频流")
        if duration <= 0: errors.append("时长无效")
        if decode_video and decode_video.returncode != 0: errors.append(f"视频解码失败：{decode_video.stderr.strip()}")
        if decode_audio and decode_audio.returncode != 0: errors.append(f"音频解码失败：{decode_audio.stderr.strip()}")
        result["error"] = "; ".join(errors)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="校验 AVUT 候选媒体。")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review-csv", type=Path, help="可选：仅回填 video_exists/has_audio，不触碰四个人工审核字段。")
    args = parser.parse_args()
    if not shutil.which("ffprobe") or not shutil.which("ffmpeg"):
        raise RuntimeError("ffprobe 或 ffmpeg 不可用")
    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    selected = payload.get("selected") if isinstance(payload, dict) else payload
    if not isinstance(selected, list) or not selected:
        raise ValueError("候选 JSON 中没有 selected 样本")
    results = []
    for item in selected:
        path = args.video_root / Path(item["video_path"]).name
        result = validate_video(path)
        result["sample_id"] = item["sample_id"]
        results.append(result)
        print(f"[{item['sample_id']}] {path}: {'PASS' if result['valid'] else 'FAIL'}")
    report = {"candidate_count": len(results), "valid_count": sum(bool(item["valid"]) for item in results), "all_valid": all(item["valid"] for item in results), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not report["all_valid"]:
        raise SystemExit("存在媒体校验失败样本；禁止静默跳过")
    if args.review_csv:
        if not args.review_csv.is_file():
            raise FileNotFoundError(f"审核 CSV 不存在：{args.review_csv}")
        with args.review_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            rows = list(reader)
        if not fieldnames or "video_exists" not in fieldnames or "has_audio" not in fieldnames:
            raise ValueError("审核 CSV 缺少 video_exists/has_audio 列")
        by_id = {item["sample_id"]: item for item in results}
        for row in rows:
            result = by_id.get(row.get("sample_id", ""))
            if result is None:
                raise ValueError(f"审核 CSV 出现未知 sample_id：{row.get('sample_id')}")
            row["video_exists"] = "yes" if result["video_exists"] else "no"
            row["has_audio"] = "yes" if result["has_audio_stream"] else "no"
        with args.review_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"已回填媒体机器检查列：{args.review_csv}")


if __name__ == "__main__":
    main()
