#!/usr/bin/env python
"""用全新开发集选择 AVE 时间同步拒绝策略，并对独立测试集只评一次。

`select-dev` 只加载开发集特征并写出锁定策略；`evaluate-test` 必须读取该
锁定文件，且测试结果存在时拒绝重复执行。本脚本不训练模型、不接 Gate、
不启用动态窗口，也不修改正式推理路径。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"
DEFAULT_FROZEN_ROOT = INTERSUIT_ROOT / "harness/artifacts/ave_hf_temporal_offset_zero125_centerpeak_expanded_frozen"
DEFAULT_SPLIT_ROOT = INTERSUIT_ROOT / "harness/artifacts/ave_hf_selective_1200_split"
DEFAULT_CLIP_MANIFEST = INTERSUIT_ROOT / "harness/artifacts/ave_hf_selective_1200_clip_window_features/ave_hf_clip_window_feature_manifest.jsonl"
DEFAULT_RGB_MANIFEST = INTERSUIT_ROOT / "harness/artifacts/ave_hf_selective_1200_window_features/ave_hf_window_feature_manifest.jsonl"
DEFAULT_OUTPUT_ROOT = INTERSUIT_ROOT / "harness/artifacts/ave_hf_selective_correction_acceptance"
MARGIN_THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)
MIN_ACCEPTED_ACCURACY = 0.70
MIN_COVERAGE = 0.30
SEEDS = ("20260718", "20260719", "20260720")


def import_temporal_module():
    path = INTERSUIT_ROOT / "scripts/train_ave_hf_temporal_offset_scorer.py"
    spec = importlib.util.spec_from_file_location("ave_hf_selective_correction_temporal", path)
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_key(record: Any) -> str:
    return "|".join(
        (
            record.youtube_id,
            record.condition,
            str(record.video_window),
            ",".join(str(value) for value in record.audio_candidate_windows),
            str(record.target_index),
        )
    )


def build_filtered_caches(clip_manifest: Path, rgb_manifest: Path, ids: set[str]) -> dict[str, dict[str, Any]]:
    clip_rows = [row for row in temporal.load_jsonl(clip_manifest) if str(row["youtube_id"]) in ids]
    rgb_rows = {
        str(row["youtube_id"]): row
        for row in temporal.load_jsonl(rgb_manifest)
        if str(row["youtube_id"]) in ids
    }
    if {str(row["youtube_id"]) for row in clip_rows} != ids or set(rgb_rows) != ids:
        raise ValueError("请求的数据 ID 与 CLIP/RGB 特征清单不完整或不一致")
    caches = [temporal.build_row_cache(row, rgb_rows[str(row["youtube_id"])]) for row in clip_rows]
    return {cache["youtube_id"]: cache for cache in caches}


def load_frozen_train_records(frozen_root: Path, seed: str) -> list[Any]:
    path = frozen_root / f"seed_{seed}/temporal_offset_train_manifest.jsonl"
    return [temporal.OffsetRecord(**row) for row in load_jsonl(path)]


def build_eval_records(
    cache_by_id: dict[str, dict[str, Any]],
    ids: set[str],
    config: dict[str, Any],
) -> list[Any]:
    records = temporal.build_records(
        list(cache_by_id.values()),
        ids,
        top_k=int(config["top_k_windows"]),
        min_change_quantile=float(config["min_change_quantile"]),
        min_candidate_change_margin=float(config["min_candidate_change_margin"]),
        require_center_peak=bool(config["require_center_peak"]),
        min_center_audio_dominance=float(config["min_center_audio_dominance"]),
        min_center_video_dominance=float(config["min_center_video_dominance"]),
        context_radius=int(config["context_radius"]),
    )
    return temporal.balance_records_by_target(records, seed=20260718)


@torch.no_grad()
def predict_seeds(
    records: list[Any],
    eval_caches: dict[str, dict[str, Any]],
    frozen_root: Path,
    frozen_config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    train_records_by_seed = {
        seed: load_frozen_train_records(frozen_root, seed)
        for seed in SEEDS
    }
    train_ids = {
        record.youtube_id
        for records_for_seed in train_records_by_seed.values()
        for record in records_for_seed
    }
    old_clip = (REPO_ROOT / frozen_config["clip_manifest"]).resolve()
    old_rgb = (REPO_ROOT / frozen_config["rgb_manifest"]).resolve()
    train_caches = build_filtered_caches(old_clip, old_rgb, train_ids)
    outputs: dict[str, dict[str, Any]] = {}
    dataset_info: dict[str, Any] = {}
    for seed in SEEDS:
        train_records = train_records_by_seed[seed]
        _, _, _, _, _, scalar_stats = temporal.make_tensor_dataset(
            train_records,
            train_caches,
            context_radius=int(frozen_config["context_radius"]),
        )
        audio, video, scalars, targets, kept, _ = temporal.make_tensor_dataset(
            records,
            eval_caches,
            context_radius=int(frozen_config["context_radius"]),
            scalar_stats=scalar_stats,
        )
        checkpoint = torch.load(
            frozen_root / f"seed_{seed}/temporal_offset_scorer_one_epoch.pt",
            map_location="cpu",
            weights_only=True,
        )
        model = temporal.OffsetScorer(
            audio_dim=audio.shape[-1],
            video_dim=video.shape[-1],
            scalar_dim=scalars.shape[-1],
            hidden_dim=int(frozen_config["hidden_dim"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        scores = model(audio, video, scalars)
        probs = torch.softmax(scores, dim=-1)
        sorted_scores = scores.sort(dim=-1, descending=True).values
        outputs[seed] = {
            "scores": scores,
            "probabilities": probs,
            "predictions": scores.argmax(dim=-1),
            "margins": sorted_scores[:, 0] - sorted_scores[:, 1],
        }
        dataset_info = {
            "records": kept,
            "targets": targets,
            "record_keys": [record_key(record) for record in kept],
        }
    return outputs, dataset_info


def false_correction_reduced(rate: float | None, forced_rate: float | None) -> bool:
    if rate is None or forced_rate is None or forced_rate <= 0:
        return False
    return rate <= forced_rate * 0.8 or forced_rate - rate >= 0.05


def strategy_metrics(
    records: list[Any],
    targets: torch.Tensor,
    raw_predictions: torch.Tensor,
    accepted: torch.Tensor,
) -> dict[str, Any]:
    final_predictions = torch.where(accepted, raw_predictions, torch.ones_like(raw_predictions))
    accepted_count = int(accepted.sum().item())
    original = targets.eq(1)
    shifted = ~original
    original_false = original & final_predictions.ne(1)
    shifted_correct = shifted & final_predictions.eq(targets)
    condition: dict[str, list[float]] = defaultdict(list)
    for index, record in enumerate(records):
        condition[record.condition].append(float(final_predictions[index].item() == targets[index].item()))
    return {
        "sample_count": int(targets.numel()),
        "accepted_count": accepted_count,
        "coverage": accepted_count / int(targets.numel()),
        "accepted_accuracy": (
            float(raw_predictions[accepted].eq(targets[accepted]).float().mean().item())
            if accepted_count
            else None
        ),
        "conservative_overall_accuracy": float(final_predictions.eq(targets).float().mean().item()),
        "original_false_correction_count": int(original_false.sum().item()),
        "original_false_correction_rate": float(original_false.float().sum().item() / original.sum().item()),
        "true_shift_recall": float(shifted_correct.float().sum().item() / shifted.sum().item()),
        "condition_accuracy": {
            key: mean(values)
            for key, values in sorted(condition.items())
        },
        "prediction_distribution": dict(
            Counter(str(temporal.OFFSETS[int(value)]) for value in final_predictions.tolist())
        ),
    }


def majority_prediction(outputs: dict[str, dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    votes = torch.stack([outputs[seed]["predictions"] for seed in SEEDS], dim=0)
    mean_probs = torch.stack([outputs[seed]["probabilities"] for seed in SEEDS], dim=0).mean(dim=0)
    mean_scores = torch.stack([outputs[seed]["scores"] for seed in SEEDS], dim=0).mean(dim=0)
    counts = torch.stack([(votes == index).sum(dim=0) for index in range(3)], dim=1)
    prediction = counts.argmax(dim=1)
    ties = counts.max(dim=1).values.eq(1)
    prediction = torch.where(ties, mean_probs.argmax(dim=1), prediction)
    chosen_score = mean_scores.gather(1, prediction.unsqueeze(1)).squeeze(1)
    other_scores = mean_scores.masked_fill(
        torch.nn.functional.one_hot(prediction, num_classes=3).bool(),
        float("-inf"),
    )
    return prediction, chosen_score - other_scores.max(dim=1).values


def evaluate_candidates(
    outputs: dict[str, dict[str, Any]],
    info: dict[str, Any],
) -> list[dict[str, Any]]:
    records, targets = info["records"], info["targets"]
    baseline = float(targets.eq(1).float().mean().item())
    candidates: list[dict[str, Any]] = []
    forced_rates: dict[str, float] = {}
    for seed in SEEDS:
        raw = outputs[seed]["predictions"]
        forced = strategy_metrics(records, targets, raw, torch.ones_like(raw, dtype=torch.bool))
        forced_rates[f"single_{seed}"] = forced["original_false_correction_rate"]
        candidates.append({"name": f"single_{seed}_forced", "kind": "single", "seed": seed, "threshold": 0.0, **forced})
        for threshold in MARGIN_THRESHOLDS:
            metrics = strategy_metrics(records, targets, raw, outputs[seed]["margins"].ge(threshold))
            candidates.append(
                {"name": f"single_{seed}_margin_{threshold}", "kind": "single", "seed": seed, "threshold": threshold, **metrics}
            )
    majority, majority_margin = majority_prediction(outputs)
    forced = strategy_metrics(records, targets, majority, torch.ones_like(majority, dtype=torch.bool))
    forced_rates["majority"] = forced["original_false_correction_rate"]
    candidates.append({"name": "majority_forced", "kind": "majority", "seed": None, "threshold": 0.0, **forced})
    for threshold in MARGIN_THRESHOLDS:
        metrics = strategy_metrics(records, targets, majority, majority_margin.ge(threshold))
        candidates.append(
            {"name": f"majority_margin_{threshold}", "kind": "majority", "seed": None, "threshold": threshold, **metrics}
        )
    for row in candidates:
        source = f"single_{row['seed']}" if row["kind"] == "single" else "majority"
        row["always_zero_baseline_accuracy"] = baseline
        row["net_gain_over_always_zero"] = row["conservative_overall_accuracy"] - baseline
        row["false_correction_reduced"] = false_correction_reduced(
            row["original_false_correction_rate"],
            forced_rates[source],
        )
        row["meets_dev_acceptance"] = bool(
            row["threshold"] > 0
            and row["accepted_accuracy"] is not None
            and row["accepted_accuracy"] >= MIN_ACCEPTED_ACCURACY
            and row["coverage"] >= MIN_COVERAGE
            and row["net_gain_over_always_zero"] > 0
            and row["false_correction_reduced"]
        )
    return candidates


def strategy_predictions(
    lock: dict[str, Any],
    outputs: dict[str, dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = lock["selected_strategy"]
    threshold = float(selected["threshold"])
    if selected["kind"] == "single":
        seed = str(selected["seed"])
        return outputs[seed]["predictions"], outputs[seed]["margins"].ge(threshold)
    raw, margin = majority_prediction(outputs)
    return raw, margin.ge(threshold)


def seed_sensitivity(
    outputs: dict[str, dict[str, Any]],
    info: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    rows = {}
    for seed in SEEDS:
        rows[seed] = strategy_metrics(
            info["records"],
            info["targets"],
            outputs[seed]["predictions"],
            outputs[seed]["margins"].ge(threshold),
        )
    values = [row["conservative_overall_accuracy"] for row in rows.values()]
    baseline = float(info["targets"].eq(1).float().mean().item())
    return {
        "seeds": rows,
        "overall_accuracy_mean": mean(values),
        "overall_accuracy_pstdev": pstdev(values),
        "all_seeds_net_positive": all(
            row["conservative_overall_accuracy"] > baseline
            for row in rows.values()
        ),
        "all_seeds_high_confidence_acceptable": all(
            row["accepted_accuracy"] is not None
            and row["accepted_accuracy"] >= MIN_ACCEPTED_ACCURACY
            and row["coverage"] >= MIN_COVERAGE
            for row in rows.values()
        ),
    }


def load_split_ids(path: Path) -> set[str]:
    return {str(row["youtube_id"]) for row in load_jsonl(path)}


def prepare_split(args: argparse.Namespace, split: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    frozen_root = Path(args.frozen_root).resolve()
    frozen_summary = load_json(frozen_root / "frozen_seed_summary.json")
    split_manifest = Path(args.split_root).resolve() / f"{split}_manifest.jsonl"
    ids = load_split_ids(split_manifest)
    other_split = "test" if split == "dev" else "dev"
    other_ids = load_split_ids(Path(args.split_root).resolve() / f"{other_split}_manifest.jsonl")
    frozen_train_ids = {
        str(row["youtube_id"])
        for row in load_jsonl(frozen_root / f"seed_{SEEDS[0]}/temporal_offset_train_manifest.jsonl")
    }
    old_201_ids = {
        str(row["youtube_id"])
        for row in load_jsonl(frozen_root / f"seed_{SEEDS[0]}/temporal_offset_val_manifest.jsonl")
    }
    overlaps = {
        "other_new_split": len(ids & other_ids),
        "frozen_train": len(ids & frozen_train_ids),
        "old_201": len(ids & old_201_ids),
    }
    if any(overlaps.values()):
        raise ValueError(f"{split} 数据隔离失败：{overlaps}")
    caches = build_filtered_caches(Path(args.clip_manifest).resolve(), Path(args.rgb_manifest).resolve(), ids)
    records = build_eval_records(caches, ids, frozen_summary["config"])
    outputs, info = predict_seeds(records, caches, frozen_root, frozen_summary["config"])
    metadata = {
        "split": split,
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": sha256_file(split_manifest),
        "youtube_id_count": len(ids),
        "record_count": len(records),
        "target_counts": dict(Counter(str(record.target_index) for record in records)),
        "youtube_id_overlaps": overlaps,
    }
    return outputs, info, metadata


def select_dev(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    lock_path = output_root / "strategy_lock.json"
    if lock_path.exists():
        raise FileExistsError(f"策略已经锁定，拒绝覆盖：{lock_path}")
    outputs, info, metadata = prepare_split(args, "dev")
    candidates = evaluate_candidates(outputs, info)
    sensitivities = {
        str(threshold): seed_sensitivity(outputs, info, threshold)
        for threshold in MARGIN_THRESHOLDS
    }
    for row in candidates:
        if row["threshold"] > 0:
            sensitivity = sensitivities[str(row["threshold"])]
            row["seed_consistent"] = bool(
                sensitivity["all_seeds_net_positive"]
                and sensitivity["all_seeds_high_confidence_acceptable"]
            )
        else:
            row["seed_consistent"] = False
        row["formal_dev_eligible"] = bool(row["meets_dev_acceptance"] and row["seed_consistent"])
    eligible = [row for row in candidates if row["formal_dev_eligible"]]
    pool = eligible or [row for row in candidates if row["threshold"] > 0]
    selected = max(
        pool,
        key=lambda row: (
            row["conservative_overall_accuracy"],
            row["accepted_accuracy"] if row["accepted_accuracy"] is not None else -1.0,
            row["coverage"],
            -row["threshold"],
        ),
    )
    sensitivity = sensitivities[str(selected["threshold"])]
    dev_passed = bool(selected["formal_dev_eligible"])
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "development_strategy_locked",
        "dev_acceptance_passed": dev_passed,
        "selected_strategy": selected,
        "selection_rule": {
            "threshold_grid": list(MARGIN_THRESHOLDS),
            "minimum_accepted_accuracy": MIN_ACCEPTED_ACCURACY,
            "minimum_coverage": MIN_COVERAGE,
            "requires_positive_net_gain": True,
            "false_correction_reduction": "相对下降至少20%或绝对下降至少5个百分点",
            "ranking": "总体准确率、接受准确率、覆盖率、较低阈值依次优先",
        },
        "dev": metadata,
        "test_manifest_sha256_before_evaluation": sha256_file(Path(args.split_root).resolve() / "test_manifest.jsonl"),
        "seed_sensitivity": sensitivity,
        "all_candidates": candidates,
        "forbidden_paths_unchanged": {
            "gate": True,
            "dynamic_window": True,
            "video_attention_weight": True,
            "pseudo_label_training": True,
        },
    }
    write_json(lock_path, payload)
    write_jsonl(
        output_root / "dev_predictions.jsonl",
        [
            {
                **asdict(record),
                "record_key": info["record_keys"][index],
                "target_index": int(info["targets"][index].item()),
                "seed_predictions": {
                    seed: int(outputs[seed]["predictions"][index].item())
                    for seed in SEEDS
                },
            }
            for index, record in enumerate(info["records"])
        ],
    )
    return payload


def evaluate_test(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    lock_path = output_root / "strategy_lock.json"
    result_path = output_root / "test_result.json"
    started_path = output_root / "test_evaluation_started.json"
    if not lock_path.exists():
        raise FileNotFoundError("必须先运行 select-dev 并锁定策略")
    if result_path.exists():
        raise FileExistsError(f"测试集已经评过一次，拒绝重复执行：{result_path}")
    if started_path.exists():
        raise FileExistsError(f"测试集评估已经启动过，拒绝二次读取：{started_path}")
    lock = load_json(lock_path)
    test_manifest = Path(args.split_root).resolve() / "test_manifest.jsonl"
    if sha256_file(test_manifest) != lock["test_manifest_sha256_before_evaluation"]:
        raise ValueError("测试集指纹与锁定策略时不一致")
    write_json(
        started_path,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "strategy_lock_sha256": sha256_file(lock_path),
            "test_manifest_sha256": sha256_file(test_manifest),
            "policy": "测试特征读取前落盘；失败时也不得自动重跑。",
        },
    )
    outputs, info, metadata = prepare_split(args, "test")
    raw, accepted = strategy_predictions(lock, outputs)
    metrics = strategy_metrics(info["records"], info["targets"], raw, accepted)
    baseline = float(info["targets"].eq(1).float().mean().item())
    forced = strategy_metrics(info["records"], info["targets"], raw, torch.ones_like(raw, dtype=torch.bool))
    metrics["always_zero_baseline_accuracy"] = baseline
    metrics["net_gain_over_always_zero"] = metrics["conservative_overall_accuracy"] - baseline
    metrics["false_correction_reduced"] = false_correction_reduced(
        metrics["original_false_correction_rate"],
        forced["original_false_correction_rate"],
    )
    sensitivity = seed_sensitivity(outputs, info, float(lock["selected_strategy"]["threshold"]))
    formal_passed = bool(
        lock["dev_acceptance_passed"]
        and metrics["accepted_accuracy"] is not None
        and metrics["accepted_accuracy"] >= MIN_ACCEPTED_ACCURACY
        and metrics["coverage"] >= MIN_COVERAGE
        and metrics["net_gain_over_always_zero"] > 0
        and metrics["false_correction_reduced"]
        and sensitivity["all_seeds_net_positive"]
        and sensitivity["all_seeds_high_confidence_acceptable"]
    )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "independent_test_evaluated_once",
        "formal_acceptance_passed": formal_passed,
        "selected_strategy": lock["selected_strategy"],
        "test": metadata,
        "metrics": metrics,
        "forced_same_predictor": forced,
        "seed_sensitivity": sensitivity,
        "conclusion": (
            "选择性修正净收益为正，可进入诊断旁路准备；仍未授权接 Gate 或正式动态窗口。"
            if formal_passed
            else "选择性修正正式验收失败，继续禁止接 Gate、动态窗口和伪标签。"
        ),
    }
    write_json(result_path, payload)
    write_report(output_root / "selective_correction_acceptance_report.md", lock, payload)
    return payload


def write_report(path: Path, lock: dict[str, Any], result: dict[str, Any]) -> None:
    selected = lock["selected_strategy"]
    metrics = result["metrics"]
    lines = [
        "# AVE 时间同步高置信度选择性修正验收",
        "",
        "开发集只用于选择策略；测试集在策略锁定后只评一次。未训练、未接 Gate、未启用动态窗口。",
        "",
        "## 冻结与隔离",
        f"- 开发视频数：{lock['dev']['youtube_id_count']}，开发记录数：{lock['dev']['record_count']}",
        f"- 测试视频数：{result['test']['youtube_id_count']}，测试记录数：{result['test']['record_count']}",
        f"- 锁定策略：`{selected['name']}`",
        f"- 开发集验收：{lock['dev_acceptance_passed']}",
        "",
        "## 独立测试结果",
        f"- 覆盖率：{metrics['coverage']:.4f}（{metrics['accepted_count']}/{metrics['sample_count']}）",
        f"- 接受样本准确率：{metrics['accepted_accuracy']}",
        f"- 保守策略总体准确率：{metrics['conservative_overall_accuracy']:.4f}",
        f"- 不修正基线：{metrics['always_zero_baseline_accuracy']:.4f}",
        f"- 净收益：{metrics['net_gain_over_always_zero']:+.4f}",
        f"- 0s 误修正率：{metrics['original_false_correction_rate']:.4f}",
        f"- 真实偏移召回率：{metrics['true_shift_recall']:.4f}",
        f"- 三方向：`{metrics['condition_accuracy']}`",
        f"- 三 seed 总体均值/波动：{result['seed_sensitivity']['overall_accuracy_mean']:.4f} / "
        f"{result['seed_sensitivity']['overall_accuracy_pstdev']:.4f}",
        "",
        "## 结论",
        f"- 正式验收：{result['formal_acceptance_passed']}",
        f"- {result['conclusion']}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select-dev", "evaluate-test"))
    parser.add_argument("--frozen-root", default=str(DEFAULT_FROZEN_ROOT))
    parser.add_argument("--split-root", default=str(DEFAULT_SPLIT_ROOT))
    parser.add_argument("--clip-manifest", default=str(DEFAULT_CLIP_MANIFEST))
    parser.add_argument("--rgb-manifest", default=str(DEFAULT_RGB_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = select_dev(args) if args.command == "select-dev" else evaluate_test(args)
    print(json.dumps({"ok": True, "phase": result["phase"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
