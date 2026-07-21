#!/usr/bin/env python
"""按冻结白名单下载新增媒体到 quarantine。"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse
from new_dataset_common import load_json, sha256_file, write_json


def _music_filename(entry: dict) -> str:
    video_id = str(entry["video_id"])
    suffix = str(entry.get("suffix") or "")
    if suffix and not video_id.endswith(suffix):
        video_id += suffix
    return f"{video_id}.mp4"


def _download(url: str, target: Path, *, expected_bytes: int | None, timeout: int, retries: int) -> tuple[str, int, str]:
    part = target.with_suffix(target.suffix + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size and (expected_bytes is None or target.stat().st_size == expected_bytes):
        return "skipped", 0, ""
    last = ""
    for attempt in range(retries + 1):
        command = ["curl", "-L", "--fail", "--retry", "0", "--connect-timeout", str(timeout),
                   "--max-time", str(timeout * 10), "-C", "-", "-o", str(part), url]
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and part.exists() and part.stat().st_size:
            os.replace(part, target)
            return "downloaded", 0, ""
        last = proc.stderr.strip()[-500:]
        if attempt < retries:
            time.sleep(1 + attempt)
    return "failed", 1, last


def _download_music(entry: dict, target: Path, *, timeout: int, retries: int) -> tuple[str, int, str]:
    if target.exists() and target.stat().st_size:
        return "skipped", 0, ""
    target.parent.mkdir(parents=True, exist_ok=True)
    start = float(entry.get("start_seconds") or 0)
    duration = float(entry.get("duration_seconds") or 0)
    if duration <= 0:
        return "failed", 1, "invalid MUSIC duration"
    command = [
        "yt-dlp", "--continue", "--retries", str(retries), "--socket-timeout", str(timeout),
        "--extractor-args", "youtube:player_client=android",
        "--download-sections", f"*{start}-{start + duration}",
        "--force-keyframes-at-cuts", "--merge-output-format", "mp4",
        "-f", "18/b", "-o", str(target), entry["source_url"],
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode or not target.exists() or target.stat().st_size == 0:
        return "failed", proc.returncode or 1, proc.stderr.strip()[-500:]
    if entry.get("has_flip"):
        flipped = target.with_name(target.stem + "_flip" + target.suffix)
        if not flipped.exists():
            flip = subprocess.run(
                ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(target),
                 "-vf", "hflip", "-c:v", "libx264", "-preset", "veryfast", "-c:a", "copy", str(flipped)],
                capture_output=True, text=True, check=False,
            )
            if flip.returncode or not flipped.exists():
                return "failed", flip.returncode or 1, f"flip failed: {flip.stderr.strip()[-400:]}"
    return "downloaded", 0, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--avut-limit", type=int)
    parser.add_argument("--music-limit", type=int)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--status-output", type=Path)
    parser.add_argument("--failures-output", type=Path)
    parser.add_argument("--reserve", type=Path, help="仅用于失败后的顺序补位")
    args = parser.parse_args()
    payload = load_json(args.allowlist)
    kind = payload.get("artifact_kind", "")
    if kind not in {"download_preflight_allowlist_not_training_manifest", "download_preflight_reserve_not_training_manifest"}:
        raise SystemExit("拒绝：输入不是预检白名单")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise SystemExit("白名单缺少 entries")
    if args.avut_limit is not None or args.music_limit is not None:
        avut = [e for e in entries if e.get("source_dataset") == "AVUT"][:args.avut_limit or 0]
        music = [e for e in entries if e.get("source_dataset") == "MUSIC-AVQA-v2.0"][:args.music_limit or 0]
        entries = avut + music
    elif args.limit is not None:
        entries = entries[:args.limit]
    results, failures = [], []
    def process(entry):
        source = entry.get("source_dataset")
        if source not in {"AVUT", "MUSIC-AVQA-v2.0"}:
            return None, {"candidate_id": entry.get("candidate_id"), "reason": "unsupported source"}
        if source == "AVUT":
            url = entry.get("download_url", "")
            expected = entry.get("expected_bytes")
            filename = Path(entry.get("remote_path") or f"{entry['youtube_id']}.mp4").name
        else:
            url = entry.get("source_url", "")
            expected = None
            filename = _music_filename(entry)
        if not url or urlparse(url).scheme not in {"http", "https"}:
            return None, {"candidate_id": entry.get("candidate_id"), "reason": "invalid URL"}
        target = args.output_root / source.replace("/", "_") / filename
        started = time.time()
        if source == "MUSIC-AVQA-v2.0":
            status, code, error = _download_music(entry, target, timeout=args.timeout, retries=args.retries)
        else:
            status, code, error = _download(url, target, expected_bytes=expected, timeout=args.timeout, retries=args.retries)
        record = dict(entry)
        record.update({"status": status, "path": str(target), "elapsed_sec": round(time.time() - started, 3)})
        if status == "failed":
            record["error"] = error
            return None, record
        elif entry.get("expected_sha256") and sha256_file(target) != entry["expected_sha256"]:
            record.update({"status": "failed", "error": "AVUT SHA256 mismatch"})
            return None, record
        else:
            return record, None
    for entry in entries:
        result, failure = process(entry)
        if result:
            results.append(result)
        if failure:
            failures.append(failure)
    unresolved = list(failures)
    failure_records = list(failures)
    if unresolved and args.reserve:
        reserve = load_json(args.reserve)
        if reserve.get("artifact_kind") != "download_preflight_reserve_not_training_manifest":
            raise SystemExit("拒绝：reserve 不是冻结替补清单")
        reserve_entries = reserve.get("entries", [])
        music_failures = [failure for failure in unresolved if failure.get("source_dataset") == "MUSIC-AVQA-v2.0"]
        unresolved = [failure for failure in unresolved if failure.get("source_dataset") != "MUSIC-AVQA-v2.0"]
        for entry in reserve_entries:
            if not music_failures:
                break
            if entry.get("source_dataset") != "MUSIC-AVQA-v2.0":
                continue
            result, failure = process(entry)
            if result:
                original = music_failures.pop(0)
                original["resolved_by"] = result.get("candidate_id")
                results.append(dict(result, replacement_for=original.get("candidate_id")))
            elif failure:
                failure_records.append(failure)
        unresolved.extend(music_failures)
    status_path = args.status_output or args.output_root / "download_status.json"
    failures_path = args.failures_output or args.output_root / "download_failures.json"
    write_json(status_path, {"status": "PASS" if not unresolved else "FAIL", "count": len(results), "entries": results})
    write_json(failures_path, {
        "status": "PASS" if not unresolved else "FAIL",
        "count": len(failure_records), "unresolved_count": len(unresolved), "entries": failure_records,
    })
    if unresolved:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
