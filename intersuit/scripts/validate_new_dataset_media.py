#!/usr/bin/env python
"""验证 quarantine 媒体，合格文件仍保持隔离并输出验收记录。"""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
from new_dataset_common import exclusion_sets, load_json, media_metadata, sha256_file, write_json


def validate_entry(entry: dict, excluded_ids: set[str], excluded_hashes: set[str], seen_hashes: set[str],
                   duration_tolerance: float) -> dict:
    record = dict(entry)
    reasons: list[str] = []
    path = Path(str(entry.get("path", "")))
    if not path.is_file() or path.stat().st_size == 0:
        reasons.append("missing_or_empty_media")
    meta = None
    if not reasons:
        try:
            meta = media_metadata(path)
            decode = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0", "-map", "0:a:0",
                 "-f", "null", "-"], capture_output=True, text=True, check=False,
            )
            if decode.returncode:
                reasons.append("media_decode_failed")
        except (ValueError, OSError) as exc:
            reasons.append(f"probe_failed:{exc}")
    media_hash = sha256_file(path) if path.is_file() and path.stat().st_size else None
    for key in ("video_id", "youtube_id"):
        if str(entry.get(key, "")).strip().casefold() in excluded_ids:
            reasons.append(f"excluded_{key}")
    if media_hash in excluded_hashes:
        reasons.append("excluded_media_sha256")
    if media_hash and media_hash in seen_hashes:
        reasons.append("duplicate_new_media_sha256")
    expected_hash = entry.get("expected_sha256")
    expected_bytes = entry.get("expected_bytes")
    if expected_hash and media_hash != expected_hash:
        reasons.append("source_sha256_mismatch")
    if expected_bytes is not None and path.is_file() and path.stat().st_size != int(expected_bytes):
        reasons.append("source_size_mismatch")
    if meta and entry.get("source_dataset") == "MUSIC-AVQA-v2.0":
        target_duration = float(entry.get("duration_seconds") or 0)
        if target_duration and abs(meta["duration"] - target_duration) > duration_tolerance:
            reasons.append("music_duration_mismatch")
        suffix = str(entry.get("suffix") or "")
        if suffix and not path.stem.endswith(suffix):
            reasons.append("music_suffix_mismatch")
        if entry.get("has_flip"):
            flipped = path.with_name(path.stem + "_flip" + path.suffix)
            if not flipped.is_file() or flipped.stat().st_size == 0:
                reasons.append("music_flip_missing")
            else:
                try:
                    media_metadata(flipped)
                    record["flip_media_sha256"] = sha256_file(flipped)
                    record["flip_path"] = str(flipped)
                except ValueError as exc:
                    reasons.append(f"music_flip_invalid:{exc}")
    if media_hash:
        seen_hashes.add(media_hash)
    record.update({
        "validation_status": "accepted" if not reasons else "rejected",
        "reasons": reasons,
        "media_sha256": media_hash,
        "original_media_sha256": media_hash,
        "metadata": meta,
    })
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-status", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--exclude-media-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-tolerance", type=float, default=2.0)
    args = parser.parse_args()
    payload = load_json(args.download_status)
    entries = payload.get("entries", [])
    excluded_ids, excluded_hashes = exclusion_sets(args.exclude_manifest, args.exclude_media_root)
    seen_hashes: set[str] = set()
    results = [validate_entry(e, excluded_ids, excluded_hashes, seen_hashes, args.duration_tolerance) for e in entries]
    rejected = [e for e in results if e["validation_status"] != "accepted"]
    write_json(args.output, {
        "status": "PASS" if not rejected else "FAIL",
        "candidate_count": len(results),
        "accepted_count": len(results) - len(rejected),
        "rejected_count": len(rejected),
        "entries": results,
    })
    if rejected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
