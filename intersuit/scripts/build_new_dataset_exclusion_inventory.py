#!/usr/bin/env python
"""从显式根目录重建训练数据泄漏排除清单。"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from new_dataset_common import canonical_id, load_records, sha256_file, write_json

MEDIA_SUFFIXES = {".mp4", ".webm", ".mkv", ".mov", ".avi"}
MANIFEST_MARKERS = ("frozen", "dev", "test", "eval", "historical")


def set_digest(values: set[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--exclude-path", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    excluded = [p.resolve() for p in args.exclude_path]

    def allowed(path: Path) -> bool:
        resolved = path.resolve()
        return not any(resolved == item or item in resolved.parents for item in excluded)

    ids: set[str] = set()
    hashes: set[str] = set()
    hashed_paths = []
    manifests = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink() or not allowed(path):
            continue
        suffix = path.suffix.casefold()
        if suffix in MEDIA_SUFFIXES:
            digest = sha256_file(path)
            hashes.add(digest)
            ids.add(canonical_id(path.stem.removesuffix("_flip")))
            hashed_paths.append({"path": str(path), "sha256": digest, "bytes": path.stat().st_size})
        elif suffix in {".json", ".jsonl"} and any(marker in str(path.relative_to(root)).casefold() for marker in MANIFEST_MARKERS):
            try:
                rows = load_records(path)
            except (OSError, ValueError):
                continue
            if not rows:
                continue
            manifests.append(str(path))
            for row in rows:
                for key in ("video_id", "youtube_id", "source_video_id", "id"):
                    if row.get(key) not in (None, ""):
                        ids.add(canonical_id(row[key]))
                for key in ("media_sha256", "video_sha256", "sha256"):
                    if row.get(key):
                        hashes.add(canonical_id(row[key]))
    write_json(args.output, {
        "status": "PASS",
        "artifact_kind": "new_dataset_exclusion_inventory",
        "id_count": len(ids), "id_set_sha256": set_digest(ids), "ids": sorted(ids),
        "media_hash_count": len(hashes), "media_hash_set_sha256": set_digest(hashes),
        "media_sha256": sorted(hashes), "hashed_path_count": len(hashed_paths),
        "hashed_bytes": sum(item["bytes"] for item in hashed_paths),
        "hashed_paths": hashed_paths, "manifest_count": len(manifests), "manifests": manifests,
    })


if __name__ == "__main__":
    main()
