#!/usr/bin/env python
"""训练 manifest 内容门禁；不导入 torch/CUDA。"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from new_dataset_common import audio_probe, canonical_id, exclusion_sets, load_records, sha256_file, write_json

REQUIRED = {
    "sample_id", "source_dataset", "source_revision", "video_id", "youtube_id",
    "video_path", "scene_audio_path", "question", "answer", "task_type",
    "media_sha256", "audio_sha256",
}


def audit_manifest(manifest: Path, exclusions: list[Path], dev_manifest: Path | None = None,
                   frozen_paths: list[Path] | None = None, exclude_media_roots: list[Path] | None = None) -> dict:
    errors: list[dict] = []
    resolved = manifest.resolve()
    if any(resolved == p.resolve() for p in (frozen_paths or []) if p.exists()):
        errors.append({"reason": "manifest_is_frozen_evaluation"})
    rows = load_records(manifest)
    excluded_ids, excluded_hashes = exclusion_sets(exclusions, exclude_media_roots or [])
    exclusion_set_sha256 = hashlib.sha256(
        ("\n".join(sorted(excluded_ids)) + "\n" + "\n".join(sorted(excluded_hashes)) + "\n").encode()
    ).hexdigest()
    seen_physical: dict[tuple[str, str], str] = {}
    ids, youtube_ids, hashes = set(), set(), set()
    for index, row in enumerate(rows):
        missing = sorted(REQUIRED - set(row))
        if missing:
            errors.append({"index": index, "reason": "missing_fields", "fields": missing})
            continue
        vid, yid, media_hash = canonical_id(row["video_id"]), canonical_id(row["youtube_id"]), canonical_id(row["media_sha256"])
        media, audio = Path(row["video_path"]), Path(row["scene_audio_path"])
        if vid in excluded_ids:
            errors.append({"index": index, "reason": "video_id_overlap"})
        if yid in excluded_ids:
            errors.append({"index": index, "reason": "youtube_id_overlap"})
        if media_hash in excluded_hashes:
            errors.append({"index": index, "reason": "media_sha256_overlap"})
        if not media.is_file() or sha256_file(media) != media_hash:
            errors.append({"index": index, "reason": "media_sha256_mismatch_or_missing"})
        if not audio.is_file() or audio.stat().st_size == 0:
            errors.append({"index": index, "reason": "scene_audio_missing_or_empty"})
        else:
            try:
                audio_probe(audio)
                if sha256_file(audio) != canonical_id(row["audio_sha256"]):
                    errors.append({"index": index, "reason": "audio_sha256_mismatch"})
            except ValueError:
                errors.append({"index": index, "reason": "scene_audio_not_decodable"})
        physical = (row["source_dataset"], media_hash)
        split = str(row.get("split", "train"))
        prior = seen_physical.get(physical)
        if prior is not None and prior != split:
            errors.append({"index": index, "reason": "physical_media_in_multiple_splits"})
        seen_physical[physical] = split
        ids.add((row["source_dataset"], vid)); youtube_ids.add(yid); hashes.add(media_hash)
    if dev_manifest:
        dev_rows = load_records(dev_manifest)
        for index, row in enumerate(dev_rows):
            if (row.get("source_dataset"), canonical_id(row.get("video_id"))) in ids:
                errors.append({"index": index, "reason": "train_dev_video_id_overlap"})
            if canonical_id(row.get("youtube_id")) in youtube_ids:
                errors.append({"index": index, "reason": "train_dev_youtube_id_overlap"})
            if canonical_id(row.get("media_sha256")) in hashes:
                errors.append({"index": index, "reason": "train_dev_media_sha256_overlap"})
    counts = {reason: sum(e["reason"] == reason for e in errors) for reason in sorted({e["reason"] for e in errors})}
    return {
        "status": "PASS" if not errors else "FAIL",
        "manifest_path": str(resolved),
        "manifest_sha256": sha256_file(manifest),
        "candidate_set_sha256": hashlib.sha256(
            "\n".join(sorted(hashes)).encode()
        ).hexdigest(),
        "exclusion_set_sha256": exclusion_set_sha256,
        "candidate_count": len(rows),
        "video_id_overlap_count": counts.get("video_id_overlap", 0),
        "youtube_id_overlap_count": counts.get("youtube_id_overlap", 0),
        "media_sha256_overlap_count": counts.get("media_sha256_overlap", 0),
        "error_count": len(errors), "error_counts": counts, "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--exclude-media-root", type=Path, action="append", default=[])
    parser.add_argument("--frozen-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_manifest(
        args.manifest, args.exclude_manifest, args.dev_manifest,
        args.frozen_manifest, args.exclude_media_root,
    )
    write_json(args.output, report)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
