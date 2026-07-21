#!/usr/bin/env python
"""把已验收的 AVE offset scorer 封装成只读诊断运行 bundle。

bundle 只补入训练集 scalar mean/std 和严格 schema；不训练、不调阈值，
也不修改 Gate、窗口或正式推理路径。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"
DEFAULT_FROZEN_ROOT = INTERSUIT_ROOT / "harness/artifacts/ave_hf_temporal_offset_zero125_centerpeak_expanded_frozen"
DEFAULT_OUTPUT = DEFAULT_FROZEN_ROOT / "seed_20260719/temporal_offset_scorer_runtime_bundle.pt"
FROZEN_SEED = "20260719"
FROZEN_MARGIN = 0.15


def import_temporal_module():
    path = INTERSUIT_ROOT / "scripts/train_ave_hf_temporal_offset_scorer.py"
    spec = importlib.util.spec_from_file_location("ave_hf_runtime_bundle_temporal", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


temporal = import_temporal_module()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_filtered_caches(
    clip_manifest: Path,
    rgb_manifest: Path,
    ids: set[str],
) -> dict[str, dict[str, Any]]:
    clip_rows = [row for row in temporal.load_jsonl(clip_manifest) if str(row["youtube_id"]) in ids]
    rgb_rows = {
        str(row["youtube_id"]): row
        for row in temporal.load_jsonl(rgb_manifest)
        if str(row["youtube_id"]) in ids
    }
    if {str(row["youtube_id"]) for row in clip_rows} != ids or set(rgb_rows) != ids:
        raise ValueError("冻结训练记录与原始特征清单不完整或不一致")
    caches = [temporal.build_row_cache(row, rgb_rows[str(row["youtube_id"])]) for row in clip_rows]
    return {cache["youtube_id"]: cache for cache in caches}


def build_bundle(frozen_root: Path, output_path: Path) -> dict[str, Any]:
    summary_path = frozen_root / "frozen_seed_summary.json"
    summary = load_json(summary_path)
    config = summary["config"]
    if [str(seed) for seed in summary["seeds"]] != ["20260718", "20260719", "20260720"]:
        raise ValueError("冻结 seed 集合发生变化")
    if float(config["zero_class_weight"]) != 1.25 or not bool(config["require_center_peak"]):
        raise ValueError("冻结配置不再是 zero_weight=1.25 + center peak")

    seed_root = frozen_root / f"seed_{FROZEN_SEED}"
    train_manifest = seed_root / "temporal_offset_train_manifest.jsonl"
    checkpoint_path = seed_root / "temporal_offset_scorer_one_epoch.pt"
    train_records = [temporal.OffsetRecord(**row) for row in load_jsonl(train_manifest)]
    train_ids = {record.youtube_id for record in train_records}
    cache_by_id = build_filtered_caches(
        (REPO_ROOT / config["clip_manifest"]).resolve(),
        (REPO_ROOT / config["rgb_manifest"]).resolve(),
        train_ids,
    )
    audio, video, scalars, _, _, scalar_stats = temporal.make_tensor_dataset(
        train_records,
        cache_by_id,
        context_radius=int(config["context_radius"]),
    )
    first_cache = next(iter(cache_by_id.values()))
    scalar_mean, scalar_std = scalar_stats
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    bundle = {
        "format_version": 1,
        "state_dict": state,
        "scalar_mean": scalar_mean.cpu(),
        "scalar_std": scalar_std.cpu(),
        "metadata": {
            "seed": int(FROZEN_SEED),
            "margin_threshold": FROZEN_MARGIN,
            "zero_class_weight": float(config["zero_class_weight"]),
            "require_center_peak": bool(config["require_center_peak"]),
            "context_radius": int(config["context_radius"]),
            "candidate_offsets": list(temporal.OFFSETS),
            "audio_dim": int(first_cache["audio"].shape[-1]),
            "clip_dim": int(first_cache["clip"].shape[-1]),
            "rgb_dim": int(first_cache["rgb"].shape[-1]),
            "scalar_dim": int(scalars.shape[-1]),
            "hidden_dim": int(config["hidden_dim"]),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "train_manifest_sha256": sha256_file(train_manifest),
            "frozen_summary_sha256": sha256_file(summary_path),
            "diagnostic_only": True,
            "moves_audio_window": False,
            "feeds_gate": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)
    reloaded = torch.load(output_path, map_location="cpu", weights_only=True)
    if reloaded["metadata"] != bundle["metadata"]:
        raise ValueError("runtime bundle 保存后元数据不一致")
    return {
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": bundle["metadata"]["checkpoint_sha256"],
        "train_record_count": len(train_records),
        "scalar_dim": int(scalars.shape[-1]),
        "pair_dim": int(state["net.0.weight"].shape[1]),
        "margin_threshold": FROZEN_MARGIN,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-root", default=str(DEFAULT_FROZEN_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_bundle(Path(args.frozen_root).resolve(), Path(args.output).resolve())
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
