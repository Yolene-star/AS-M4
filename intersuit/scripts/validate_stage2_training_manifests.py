#!/usr/bin/env python
"""验证阶段 2 正式 train/dev/reserve manifest 和运行时门禁。"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from new_dataset_common import audio_probe, load_records, sha256_file, write_json


REQUIRED_FIELDS = {
    "sample_id", "source_dataset", "source_revision", "video_id", "youtube_id",
    "physical_media_id", "derived_media_id", "video_path", "scene_audio_path",
    "question", "answer", "task_type", "media_sha256", "audio_sha256",
    "qa_origin", "conversations",
}
TASK_TYPES = {"audio", "visual", "audio_visual"}


def canonical(value: Any) -> str:
    return str(value).strip().casefold()


def external_sets(paths: list[Path]) -> tuple[set[str], set[str], str]:
    ids: set[str] = set()
    hashes: set[str] = set()
    inputs = []
    for path in paths:
        inputs.append(f"{path.resolve()}:{sha256_file(path)}")
        for row in load_records(path):
            for key in ("video_id", "youtube_id", "source_video_id", "id"):
                if row.get(key) not in (None, ""):
                    ids.add(canonical(row[key]))
            for key in ("media_sha256", "video_sha256", "sha256"):
                if row.get(key):
                    hashes.add(canonical(row[key]))
    digest = hashlib.sha256(
        ("\n".join(sorted(ids)) + "\n" + "\n".join(sorted(hashes)) + "\n"
         + "\n".join(sorted(inputs))).encode()
    ).hexdigest()
    return ids, hashes, digest


def verify_sidecar(path: Path) -> tuple[bool, str]:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        return False, "missing_sha256_sidecar"
    expected = sidecar.read_text(encoding="ascii").split()[0]
    actual = sha256_file(path)
    return expected == actual, actual


def decode_media(path: Path) -> str | None:
    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
            "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return None if proc.returncode == 0 else proc.stderr[-300:]


def cached_sha256(path: Path, cache: dict[Path, str]) -> str:
    if path not in cache:
        cache[path] = sha256_file(path)
    return cache[path]


def cached_audio_probe(path: Path, cache: dict[Path, dict[str, Any]]) -> dict[str, Any]:
    if path not in cache:
        cache[path] = audio_probe(path)
    return cache[path]


def runtime_smoke(train_manifest: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    import importlib.machinery
    import sys
    import types

    import torch

    deepspeed_stub = types.ModuleType("deepspeed")
    deepspeed_stub.__spec__ = importlib.machinery.ModuleSpec("deepspeed", loader=None)
    sys.modules.setdefault("deepspeed", deepspeed_stub)
    trainer_name = "intersuit.train.llava_trainer"
    if trainer_name not in sys.modules:
        trainer_stub = types.ModuleType(trainer_name)
        trainer_stub.__spec__ = importlib.machinery.ModuleSpec(trainer_name, loader=None)
        trainer_stub.LLaVATrainer = object
        sys.modules[trainer_name] = trainer_stub
    from intersuit.train.train import DataCollatorForSupervisedDataset, LazySupervisedDataset

    class DummyTokenizer:
        padding_side = "right"
        pad_token_id = 0
        model_max_length = 64

    data_args = types.SimpleNamespace(
        dataset_paths=[],
        scene_audio_folder=None,
        scene_audio_feature_folder=None,
        scene_audio_sample_rate=16000,
    )
    tokenizer = DummyTokenizer()
    dataset = LazySupervisedDataset(str(train_manifest), tokenizer, data_args)
    if len(dataset) != len(rows):
        raise ValueError(f"Dataset 长度不一致：{len(dataset)} != {len(rows)}")
    selected = []
    seen_audio = set()
    for row in rows:
        if row["scene_audio_path"] in seen_audio:
            continue
        seen_audio.add(row["scene_audio_path"])
        selected.append(row)
        if len(selected) == 2:
            break
    instances = []
    for index, row in enumerate(selected):
        fields = dataset._load_scene_audio_fields(row)
        token_ids = torch.tensor([1 + index, 2 + index], dtype=torch.long)
        instances.append({"input_ids": token_ids, "labels": token_ids.clone(), **fields})
    batch = DataCollatorForSupervisedDataset(tokenizer=tokenizer)(instances)
    if batch["scene_audios"].shape[0] != len(instances):
        raise ValueError("DataCollator scene audio batch 大小错误")
    if not bool(batch["scene_audio_mask"].any()):
        raise ValueError("DataCollator scene audio mask 全为空")
    return {
        "dataset_length": len(dataset),
        "collator_batch_size": len(instances),
        "scene_audio_shape": list(batch["scene_audios"].shape),
        "scene_audio_mask_nonempty": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--reserve", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-audit-output", type=Path, required=True)
    parser.add_argument("--require-train", type=int, default=1000)
    parser.add_argument("--require-dev", type=int, default=300)
    parser.add_argument("--require-reserve", type=int, default=50)
    parser.add_argument("--min-dev-sources", type=int, default=2)
    parser.add_argument("--full-decode", action="store_true")
    parser.add_argument("--skip-runtime-smoke", action="store_true")
    args = parser.parse_args()

    manifests = {"train": args.train.resolve(), "dev": args.dev.resolve(), "reserve": args.reserve.resolve()}
    rows_by_split = {split: load_records(path) for split, path in manifests.items()}
    errors: list[dict[str, Any]] = []
    external_ids, external_hashes, exclusion_sha256 = external_sets(args.exclude_manifest)
    path_hash_cache: dict[Path, str] = {}
    audio_meta_cache: dict[Path, dict[str, Any]] = {}
    decode_cache: dict[Path, str | None] = {}
    split_assets: dict[str, set[str]] = {}
    split_groups: dict[str, set[str]] = {}
    split_youtube: dict[str, set[str]] = {}
    split_derived: dict[str, set[str]] = {}
    split_video: dict[str, set[tuple[str, str]]] = {}
    global_sample_ids: set[str] = set()
    media_owner: dict[str, tuple[str, str]] = {}
    bad_audio_paths: set[Path] = set()
    bad_decode_paths: set[Path] = set()

    sidecars = {}
    for split, path in manifests.items():
        ok, digest = verify_sidecar(path)
        sidecars[split] = {"status": "PASS" if ok else "FAIL", "sha256": digest}
        if not ok:
            errors.append({"split": split, "reason": "manifest_sha256_sidecar_mismatch"})
        assets: set[str] = set()
        groups: set[str] = set()
        youtube: set[str] = set()
        derived: set[str] = set()
        video_ids: set[tuple[str, str]] = set()
        for index, row in enumerate(rows_by_split[split]):
            missing = sorted(REQUIRED_FIELDS - set(row))
            if missing:
                errors.append({"split": split, "index": index, "reason": "missing_fields", "fields": missing})
                continue
            sample_id = canonical(row["sample_id"])
            if sample_id in global_sample_ids:
                errors.append({"split": split, "index": index, "reason": "duplicate_sample_id"})
            global_sample_ids.add(sample_id)
            task_type = canonical(row["task_type"])
            if task_type not in TASK_TYPES:
                errors.append({"split": split, "index": index, "reason": "invalid_task_type"})
            media_hash = canonical(row["media_sha256"])
            audio_hash = canonical(row["audio_sha256"])
            video_path = Path(row["video_path"]).resolve()
            audio_path = Path(row["scene_audio_path"]).resolve()
            if not video_path.is_file() or not video_path.stat().st_size:
                errors.append({"split": split, "index": index, "reason": "video_missing_or_empty"})
            else:
                actual = cached_sha256(video_path, path_hash_cache)
                if actual != media_hash:
                    errors.append({"split": split, "index": index, "reason": "media_sha256_mismatch"})
            if not audio_path.is_file() or not audio_path.stat().st_size:
                errors.append({"split": split, "index": index, "reason": "scene_audio_missing_or_empty"})
                bad_audio_paths.add(audio_path)
            else:
                actual_audio = cached_sha256(audio_path, path_hash_cache)
                if actual_audio != audio_hash:
                    errors.append({"split": split, "index": index, "reason": "audio_sha256_mismatch"})
                    bad_audio_paths.add(audio_path)
                try:
                    meta = cached_audio_probe(audio_path, audio_meta_cache)
                    if int(meta.get("sample_rate") or 0) != 16000 or int(meta.get("channels") or 0) != 1:
                        errors.append({"split": split, "index": index, "reason": "scene_audio_not_16k_mono"})
                        bad_audio_paths.add(audio_path)
                except ValueError as exc:
                    errors.append({"split": split, "index": index, "reason": "scene_audio_not_decodable", "error": str(exc)})
                    bad_audio_paths.add(audio_path)
            if args.full_decode and video_path not in decode_cache:
                decode_cache[video_path] = decode_media(video_path)
                if len(decode_cache) % 100 == 0:
                    print(f"完整媒体解码进度：{len(decode_cache)}", flush=True)
            if args.full_decode and decode_cache.get(video_path):
                errors.append({"split": split, "index": index, "reason": "video_or_audio_decode_failed", "error": decode_cache[video_path]})
                bad_decode_paths.add(video_path)
            yid = canonical(row["youtube_id"])
            group_id = canonical(row["physical_media_id"])
            derived_id = canonical(row["derived_media_id"])
            video_id = (str(row["source_dataset"]), canonical(row["video_id"]))
            if yid in external_ids:
                errors.append({"split": split, "index": index, "reason": "youtube_id_overlap"})
            if canonical(row["video_id"]) in external_ids:
                errors.append({"split": split, "index": index, "reason": "video_id_overlap"})
            if media_hash in external_hashes:
                errors.append({"split": split, "index": index, "reason": "media_sha256_overlap"})
            owner = media_owner.get(media_hash)
            if owner is not None and owner != (split, derived_id):
                errors.append({"split": split, "index": index, "reason": "media_sha256_multiple_derived_ids"})
            media_owner[media_hash] = (split, derived_id)
            assets.add(media_hash)
            groups.add(group_id)
            youtube.add(yid)
            derived.add(derived_id)
            video_ids.add(video_id)
        split_assets[split] = assets
        split_groups[split] = groups
        split_youtube[split] = youtube
        split_derived[split] = derived
        split_video[split] = video_ids

    required_counts = {"train": args.require_train, "dev": args.require_dev}
    for split, required in required_counts.items():
        if len(split_assets[split]) != required:
            errors.append({"split": split, "reason": "physical_media_count_mismatch", "expected": required, "actual": len(split_assets[split])})
    if len(split_assets["reserve"]) < args.require_reserve:
        errors.append({"split": "reserve", "reason": "reserve_physical_media_below_minimum", "actual": len(split_assets["reserve"])})

    pairwise = {}
    for left, right in (("train", "dev"), ("train", "reserve"), ("dev", "reserve")):
        values = {
            "physical_media_id_overlap": len(split_groups[left] & split_groups[right]),
            "youtube_id_overlap": len(split_youtube[left] & split_youtube[right]),
            "derived_media_id_overlap": len(split_derived[left] & split_derived[right]),
            "video_id_overlap": len(split_video[left] & split_video[right]),
            "media_sha256_overlap": len(split_assets[left] & split_assets[right]),
        }
        pairwise[f"{left}_{right}"] = values
        for reason, count in values.items():
            if count:
                errors.append({"pair": f"{left}_{right}", "reason": reason, "count": count})

    task_distribution = {
        split: dict(sorted(Counter(canonical(row.get("task_type")) for row in rows).items()))
        for split, rows in rows_by_split.items()
    }
    source_distribution = {
        split: dict(sorted(Counter(str(row.get("source_dataset")) for row in rows).items()))
        for split, rows in rows_by_split.items()
    }
    for split in ("train", "dev"):
        for task in TASK_TYPES:
            if task_distribution[split].get(task, 0) == 0:
                errors.append({"split": split, "reason": "missing_task_type", "task_type": task})
    if len(source_distribution["dev"]) < args.min_dev_sources:
        errors.append({"split": "dev", "reason": "dev_single_source"})
    if not any(row.get("qa_origin") == "human" for row in rows_by_split["dev"]):
        errors.append({"split": "dev", "reason": "dev_missing_human_qa"})

    runtime = None
    if not args.skip_runtime_smoke:
        try:
            runtime = runtime_smoke(manifests["train"], rows_by_split["train"])
        except Exception as exc:
            errors.append({"reason": "dataset_or_collator_smoke_failed", "error": str(exc)})

    counts = Counter(error["reason"] for error in errors)
    report = {
        "status": "PASS" if not errors else "FAIL",
        "manifest_sha256": sidecars,
        "split_qa_count": {split: len(rows) for split, rows in rows_by_split.items()},
        "split_physical_media_count": {split: len(values) for split, values in split_assets.items()},
        "split_physical_group_count": {split: len(values) for split, values in split_groups.items()},
        "pairwise_overlap": pairwise,
        "task_distribution": task_distribution,
        "source_distribution": source_distribution,
        "external_exclusion_set_sha256": exclusion_sha256,
        "external_video_id_overlap_count": counts.get("video_id_overlap", 0),
        "external_youtube_id_overlap_count": counts.get("youtube_id_overlap", 0),
        "external_media_sha256_overlap_count": counts.get("media_sha256_overlap", 0),
        "scene_audio_path_valid_rate": (
            (len(audio_meta_cache) - len(bad_audio_paths)) / len(audio_meta_cache)
            if audio_meta_cache else 0.0
        ),
        "video_audio_decode_rate": (
            (len(decode_cache) - len(bad_decode_paths)) / len(decode_cache)
            if args.full_decode and decode_cache else None
        ),
        "sample_id_duplicate_count": counts.get("duplicate_sample_id", 0),
        "runtime_smoke": runtime,
        "full_decode": bool(args.full_decode),
        "error_count": len(errors),
        "error_counts": dict(sorted(counts.items())),
        "errors": errors,
        "training_started": False,
    }
    write_json(args.output, report)
    train_audit = {
        "status": report["status"],
        "manifest_path": str(manifests["train"]),
        "manifest_sha256": sidecars["train"]["sha256"],
        "candidate_set_sha256": hashlib.sha256(
            "\n".join(sorted(split_assets["train"])).encode()
        ).hexdigest(),
        "exclusion_set_sha256": exclusion_sha256,
        "candidate_count": len(rows_by_split["train"]),
        "video_id_overlap_count": counts.get("video_id_overlap", 0),
        "youtube_id_overlap_count": counts.get("youtube_id_overlap", 0),
        "media_sha256_overlap_count": counts.get("media_sha256_overlap", 0),
        "error_count": len(errors),
        "error_counts": dict(sorted(counts.items())),
        "errors": errors,
    }
    write_json(args.train_audit_output, train_audit)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
