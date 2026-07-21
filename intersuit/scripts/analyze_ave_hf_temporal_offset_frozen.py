#!/usr/bin/env python
"""只读分析 AVE_HF 冻结时间同步 scorer 的错误与拒绝修正曲线。

本脚本只重放已经保存的 one_epoch checkpoint，不训练、不修改冻结验证
manifest、不选择部署阈值，也不接 Gate 或动态窗口。
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
DEFAULT_INPUT_ROOT = INTERSUIT_ROOT / "harness/artifacts/ave_hf_temporal_offset_zero125_centerpeak_expanded_frozen"
DEFAULT_ANNOTATIONS = INTERSUIT_ROOT / "datasets/AVE/data/Annotations.txt"
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "offline_error_analysis"
MARGIN_THRESHOLDS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)


def import_temporal_module():
    path = INTERSUIT_ROOT / "scripts/train_ave_hf_temporal_offset_scorer.py"
    spec = importlib.util.spec_from_file_location("ave_hf_temporal_offset_offline_analysis", path)
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


def record_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row["youtube_id"]),
            str(row["condition"]),
            str(row["video_window"]),
            ",".join(str(value) for value in row["audio_candidate_windows"]),
            str(row["target_index"]),
        ]
    )


def canonical_record_set_sha256(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(sorted(record_key(row) for row in rows)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_annotations(path: Path) -> dict[str, dict[str, Any]]:
    annotations: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("&")
        if len(parts) < 5:
            continue
        try:
            start, end = int(parts[3]), int(parts[4])
        except ValueError:
            continue
        annotations[parts[1]] = {
            "event_label": parts[0],
            "quality": parts[2],
            "event_start": start,
            "event_end": end,
        }
    return annotations


def boundary_relation(annotation: dict[str, Any] | None, window_center: float) -> dict[str, Any]:
    if annotation is None:
        return {"boundary_relation": "annotation_missing"}
    start = float(annotation["event_start"])
    end = float(annotation["event_end"])
    internal: list[tuple[str, float]] = []
    if start > 0:
        internal.append(("event_start", abs(window_center - start)))
    if end < 10:
        internal.append(("event_end", abs(window_center - end)))
    if not internal:
        return {
            "boundary_relation": "full_clip_event",
            "distance_to_event_start": abs(window_center - start),
            "distance_to_event_end": abs(window_center - end),
        }
    relation, distance = min(internal, key=lambda item: item[1])
    return {
        "boundary_relation": relation,
        "nearest_internal_boundary_distance": distance,
        "near_internal_boundary_0.75s": distance <= 0.75,
        "distance_to_event_start": abs(window_center - start),
        "distance_to_event_end": abs(window_center - end),
    }


def accuracy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0, "accuracy": None}
    return {
        "sample_count": len(rows),
        "accuracy": mean(float(row["correct"]) for row in rows),
    }


def grouped_accuracy(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    return {name: accuracy_summary(items) for name, items in sorted(grouped.items())}


def rejection_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    accepted = [row for row in rows if float(row["margin"]) >= threshold]
    conservative_correct = []
    original_rows = [row for row in rows if row["condition"] == "original"]
    original_false_corrections = 0
    for row in rows:
        prediction = int(row["prediction_index"]) if float(row["margin"]) >= threshold else 1
        conservative_correct.append(prediction == int(row["target_index"]))
        if row["condition"] == "original" and prediction != 1:
            original_false_corrections += 1
    return {
        "threshold": threshold,
        "total_count": len(rows),
        "accepted_count": len(accepted),
        "coverage": len(accepted) / len(rows) if rows else 0.0,
        "accepted_accuracy": mean(float(row["correct"]) for row in accepted) if accepted else None,
        "conservative_overall_accuracy": mean(conservative_correct) if conservative_correct else None,
        "original_false_correction_count": original_false_corrections,
        "original_false_correction_rate": (
            original_false_corrections / len(original_rows) if original_rows else None
        ),
    }


def prediction_consistency(seed_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    by_seed = {
        seed: {row["record_key"]: row for row in rows}
        for seed, rows in seed_rows.items()
    }
    key_sets = [set(rows) for rows in by_seed.values()]
    if not key_sets or any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("三个 seed 的验证记录集合不一致")
    categories: Counter[str] = Counter()
    category_correct: dict[str, list[float]] = defaultdict(list)
    for key in sorted(key_sets[0]):
        rows = [by_seed[seed][key] for seed in sorted(by_seed)]
        predictions = [int(row["prediction_index"]) for row in rows]
        counts = Counter(predictions)
        if len(counts) == 1:
            category = "unanimous"
            consensus = predictions[0]
        elif max(counts.values()) == 2:
            category = "two_of_three"
            consensus = counts.most_common(1)[0][0]
        else:
            category = "all_different"
            consensus = None
        categories[category] += 1
        if consensus is not None:
            category_correct[category].append(float(consensus == int(rows[0]["target_index"])))
    return {
        "record_count": len(key_sets[0]),
        "category_counts": dict(categories),
        "category_accuracy": {
            key: mean(values) if values else None
            for key, values in sorted(category_correct.items())
        },
    }


def aggregate_seed_curves(curves: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result = []
    for index, threshold in enumerate(MARGIN_THRESHOLDS):
        rows = [curves[seed][index] for seed in sorted(curves)]
        item: dict[str, Any] = {"threshold": threshold}
        for metric in (
            "coverage",
            "accepted_accuracy",
            "conservative_overall_accuracy",
            "original_false_correction_rate",
        ):
            values = [float(row[metric]) for row in rows if row[metric] is not None]
            item[metric] = {
                "mean": mean(values) if values else None,
                "pstdev": pstdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
        item["accepted_count_values"] = [row["accepted_count"] for row in rows]
        result.append(item)
    return result


def aggregate_grouped_accuracy(
    seed_summaries: dict[str, dict[str, Any]],
    section: str,
) -> dict[str, dict[str, Any]]:
    names = sorted(
        {
            name
            for summary in seed_summaries.values()
            for name in summary[section]
        }
    )
    result = {}
    for name in names:
        rows = [
            seed_summaries[seed][section][name]
            for seed in sorted(seed_summaries)
            if name in seed_summaries[seed][section]
        ]
        values = [float(row["accuracy"]) for row in rows if row["accuracy"] is not None]
        result[name] = {
            "sample_count_values": [int(row["sample_count"]) for row in rows],
            "accuracy_mean": mean(values) if values else None,
            "accuracy_pstdev": pstdev(values) if len(values) > 1 else 0.0,
            "accuracy_values": values,
        }
    return result


def build_caches(clip_manifest: Path, rgb_manifest: Path) -> dict[str, dict[str, Any]]:
    clip_rows = temporal.load_jsonl(clip_manifest)
    rgb_rows = {str(row["youtube_id"]): row for row in temporal.load_jsonl(rgb_manifest)}
    caches = []
    for row in clip_rows:
        youtube_id = str(row["youtube_id"])
        if youtube_id not in rgb_rows:
            raise ValueError(f"缺少 RGB 特征行：{youtube_id}")
        caches.append(temporal.build_row_cache(row, rgb_rows[youtube_id]))
    return {cache["youtube_id"]: cache for cache in caches}


def load_records(path: Path) -> list[Any]:
    return [temporal.OffsetRecord(**row) for row in load_jsonl(path)]


@torch.no_grad()
def replay_seed(
    seed: str,
    seed_root: Path,
    cache_by_id: dict[str, dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    context_radius: int,
    hidden_dim: int,
) -> list[dict[str, Any]]:
    train_records = load_records(seed_root / "temporal_offset_train_manifest.jsonl")
    val_records = load_records(seed_root / "temporal_offset_val_manifest.jsonl")
    _, _, _, _, _, scalar_stats = temporal.make_tensor_dataset(
        train_records, cache_by_id, context_radius=context_radius
    )
    audio, video, scalars, targets, kept, _ = temporal.make_tensor_dataset(
        val_records,
        cache_by_id,
        context_radius=context_radius,
        scalar_stats=scalar_stats,
    )
    checkpoint_path = seed_root / "temporal_offset_scorer_one_epoch.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = temporal.OffsetScorer(
        audio_dim=audio.shape[-1],
        video_dim=video.shape[-1],
        scalar_dim=scalars.shape[-1],
        hidden_dim=hidden_dim,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    scores = model(audio, video, scalars)
    probabilities = torch.softmax(scores, dim=-1)
    predictions = scores.argmax(dim=-1)
    sorted_scores = scores.sort(dim=-1, descending=True).values
    margins = sorted_scores[:, 0] - sorted_scores[:, 1]
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(kept):
        cache = cache_by_id[record.youtube_id]
        timestamp = cache["timestamps"][record.video_window]
        window_center = float(timestamp.mean().item())
        base = asdict(record)
        base.update(
            {
                "seed": seed,
                "record_key": record_key(base),
                "target_offset": temporal.OFFSETS[record.target_index],
                "prediction_index": int(predictions[index].item()),
                "predicted_offset": temporal.OFFSETS[int(predictions[index].item())],
                "correct": bool(int(predictions[index].item()) == record.target_index),
                "scores": [float(value) for value in scores[index].tolist()],
                "probabilities": [float(value) for value in probabilities[index].tolist()],
                "confidence": float(probabilities[index].max().item()),
                "margin": float(margins[index].item()),
                "window_center_sec": window_center,
            }
        )
        annotation = annotations.get(record.youtube_id)
        if annotation:
            base.update(annotation)
        base.update(boundary_relation(annotation, window_center))
        rows.append(base)
    if len(rows) != int(targets.numel()):
        raise ValueError(f"{seed} 预测数和验证样本数不一致")
    return rows


def summarize_seed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wrong = [row for row in rows if not row["correct"]]
    adjacent_wrong = [
        row
        for row in wrong
        if abs(int(row["prediction_index"]) - int(row["target_index"])) == 1
    ]
    curves = [rejection_metrics(rows, threshold) for threshold in MARGIN_THRESHOLDS]
    return {
        "overall": accuracy_summary(rows),
        "condition": grouped_accuracy(rows, "condition"),
        "label": grouped_accuracy(rows, "label"),
        "event_label": grouped_accuracy(rows, "event_label"),
        "boundary_relation": grouped_accuracy(rows, "boundary_relation"),
        "boundary_type": grouped_accuracy(rows, "event_boundary_type"),
        "sync_peak_delta": grouped_accuracy(rows, "sync_peak_delta"),
        "error_count": len(wrong),
        "adjacent_error_count": len(adjacent_wrong),
        "adjacent_error_ratio": len(adjacent_wrong) / len(wrong) if wrong else None,
        "correct_abs_sync_peak_delta_mean": mean(
            abs(float(row["sync_peak_delta"])) for row in rows if row["correct"]
        ),
        "wrong_abs_sync_peak_delta_mean": mean(
            abs(float(row["sync_peak_delta"])) for row in wrong
        ) if wrong else None,
        "confidence_curve": curves,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    consistency = summary["prediction_consistency"]
    lines = [
        "# AVE 时间同步 scorer 冻结验证集离线错误分析",
        "",
        "本分析只重放已有 checkpoint；未训练、未改阈值、未筛样本、未接 Gate 或动态窗口。",
        "",
        "## 冻结确认",
        f"- 三个 seed 的规范化验证记录集合指纹一致：{summary['freeze']['canonical_record_sets_identical']}",
        f"- 每个 seed 验证记录数：{summary['freeze']['validation_record_counts']}",
        "",
        "## 跨 seed 一致性",
        f"- 记录数：{consistency['record_count']}",
        f"- 一致性分布：`{consistency['category_counts']}`",
        f"- 一致预测准确率：`{consistency['category_accuracy']}`",
        "",
        "## 跨 seed 事件类别与边界",
        "",
        "表现最差的事件类别（按三 seed 均值排序）：",
        "",
    ]
    worst_event_labels = sorted(
        summary["aggregate_event_label"].items(),
        key=lambda item: (
            item[1]["accuracy_mean"] if item[1]["accuracy_mean"] is not None else 9.0,
            item[0],
        ),
    )[:10]
    for name, item in worst_event_labels:
        lines.append(
            f"- {name}：mean={item['accuracy_mean']:.4f}，"
            f"std={item['accuracy_pstdev']:.4f}，n={item['sample_count_values']}"
        )
    lines.extend(
        [
            "",
            f"- 最近事件开始：`{summary['aggregate_boundary_relation'].get('event_start')}`",
            f"- 最近事件结束：`{summary['aggregate_boundary_relation'].get('event_end')}`",
            f"- 覆盖全片事件：`{summary['aggregate_boundary_relation'].get('full_clip_event')}`",
            "",
        "## 单 seed 错误概览",
        ]
    )
    for seed, item in summary["seeds"].items():
        lines.extend(
            [
                f"### {seed}",
                f"- 总体：`{item['overall']}`",
                f"- 三类：`{item['condition']}`",
                f"- 相邻候选错误：{item['adjacent_error_count']}/{item['error_count']}，比例={item['adjacent_error_ratio']:.4f}",
                f"- 正确/错误样本绝对峰值时差均值：{item['correct_abs_sync_peak_delta_mean']:.4f}/{item['wrong_abs_sync_peak_delta_mean']:.4f}",
                f"- 最近事件边界：`{item['boundary_relation']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 置信度—准确率—覆盖率",
            "",
            "下面阈值为预先固定的描述性网格，不代表已选择部署阈值。",
            "",
            "| margin 阈值 | 覆盖率均值 | 接受样本准确率均值 | 低置信保持 0s 后总体准确率 | 0s 误修正率 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["aggregate_confidence_curve"]:
        accepted = row["accepted_accuracy"]["mean"]
        lines.append(
            f"| {row['threshold']:.2f} | {row['coverage']['mean']:.4f} | "
            f"{accepted:.4f} | {row['conservative_overall_accuracy']['mean']:.4f} | "
            f"{row['original_false_correction_rate']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 本报告可以判断高 margin 子集是否更可靠，但不能用这 201 条记录选择新阈值。",
            "- 若后续决定采用拒绝修正策略，阈值必须在新的开发集上确定，并在新的独立测试集上一次性验收。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    annotations_path = Path(args.annotations).resolve()
    frozen_summary = load_json(input_root / "frozen_seed_summary.json")
    seeds = [str(seed) for seed in frozen_summary["seeds"]]
    config_path = Path(frozen_summary["config_path"])
    split_path = Path(frozen_summary["config"]["split_summary"])
    file_hashes = {
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "split": {"path": str(split_path), "sha256": sha256_file(split_path)},
    }
    validation_sets = {}
    checkpoint_hashes = {}
    validation_record_counts = {}
    for seed in seeds:
        seed_root = input_root / f"seed_{seed}"
        manifest = seed_root / "temporal_offset_val_manifest.jsonl"
        checkpoint = seed_root / "temporal_offset_scorer_one_epoch.pt"
        rows = load_jsonl(manifest)
        validation_sets[seed] = canonical_record_set_sha256(rows)
        validation_record_counts[seed] = len(rows)
        file_hashes[f"validation_manifest_{seed}"] = {
            "path": str(manifest),
            "sha256": sha256_file(manifest),
        }
        checkpoint_hashes[seed] = {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}
    if len(set(validation_sets.values())) != 1:
        raise ValueError("三个 seed 的规范化验证记录集合不一致")
    freeze = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "file_hashes": file_hashes,
        "checkpoint_hashes": checkpoint_hashes,
        "canonical_validation_record_set_sha256": validation_sets,
        "canonical_record_sets_identical": True,
        "validation_record_counts": validation_record_counts,
        "policy": "只读分析；禁止用本验证集调权重、阈值、模型或筛样本。",
    }
    write_json(output_root / "frozen_input_hashes.json", freeze)

    clip_manifest = Path(frozen_summary["config"]["clip_manifest"])
    rgb_manifest = Path(frozen_summary["config"]["rgb_manifest"])
    cache_by_id = build_caches(clip_manifest, rgb_manifest)
    annotations = load_annotations(annotations_path)
    seed_predictions: dict[str, list[dict[str, Any]]] = {}
    seed_summaries = {}
    seed_curves = {}
    for seed in seeds:
        rows = replay_seed(
            seed,
            input_root / f"seed_{seed}",
            cache_by_id,
            annotations,
            context_radius=int(frozen_summary["config"]["context_radius"]),
            hidden_dim=int(frozen_summary["config"]["hidden_dim"]),
        )
        seed_predictions[seed] = rows
        seed_summaries[seed] = summarize_seed(rows)
        seed_curves[seed] = seed_summaries[seed]["confidence_curve"]
        write_jsonl(output_root / f"predictions_seed_{seed}.jsonl", rows)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_only": True,
        "retrained": False,
        "gate_or_dynamic_window_modified": False,
        "freeze": freeze,
        "seeds": seed_summaries,
        "aggregate_event_label": aggregate_grouped_accuracy(seed_summaries, "event_label"),
        "aggregate_boundary_relation": aggregate_grouped_accuracy(seed_summaries, "boundary_relation"),
        "aggregate_boundary_type": aggregate_grouped_accuracy(seed_summaries, "boundary_type"),
        "prediction_consistency": prediction_consistency(seed_predictions),
        "aggregate_confidence_curve": aggregate_seed_curves(seed_curves),
        "paths": {
            "summary": str(output_root / "offline_error_analysis_summary.json"),
            "report": str(output_root / "offline_error_analysis_report.md"),
        },
    }
    write_json(output_root / "offline_error_analysis_summary.json", summary)
    write_report(output_root / "offline_error_analysis_report.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps({"ok": True, "paths": summary["paths"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
