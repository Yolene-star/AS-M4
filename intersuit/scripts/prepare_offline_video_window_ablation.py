#!/usr/bin/env python
"""准备四组离线视频时间窗口加权消融特征与评测清单。

输入视频特征必须是已经经过 M4 视觉塔与 projector 的 ``[T,...]`` 张量；
诊断 JSONL 必须来自冻结 seed=20260719、margin=0.15 的 offset scorer。
本脚本不加载或修改 M4，不接 Gate，也不修改正式推理路径。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from intersuit.model.streaming_av.video_window_weighting import (
    apply_video_window_weighting,
    resample_window_signal,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"
DEFAULT_MODEL = INTERSUIT_ROOT / "checkpoints/M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze"
FROZEN_SEED = 20260719
FROZEN_MARGIN = 0.15
MODES = ("baseline", "hard_move", "offset_soft", "offset_event_soft")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_feature(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        payload = payload.get("features", payload.get("video_features"))
    if not isinstance(payload, torch.Tensor) or payload.ndim < 2:
        raise ValueError(f"视频特征必须是 [T,...] tensor：{path}")
    values = payload.detach()
    if not values.is_floating_point():
        raise ValueError(f"视频特征必须是浮点 tensor：{path}")
    if not torch.isfinite(values).all():
        raise ValueError(f"视频特征包含 NaN/Inf：{path}")
    return values


def diagnostic_key(row: dict[str, Any]) -> str:
    value = row.get("sample_id", row.get("youtube_id", row.get("id")))
    if value is None:
        raise ValueError("诊断记录缺少 sample_id/youtube_id/id")
    return str(value)


def qa_key(row: dict[str, Any]) -> str:
    value = row.get("id", row.get("sample_id", row.get("youtube_id")))
    if value is None:
        raise ValueError("QA 记录缺少 id/sample_id/youtube_id")
    return str(value)


def signal(row: dict[str, Any], key: str, *, dtype: torch.dtype) -> torch.Tensor:
    value = row.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"诊断记录 {diagnostic_key(row)} 缺少非空 {key}")
    tensor = torch.tensor(value, dtype=dtype)
    if tensor.ndim != 1 or not torch.isfinite(tensor.float()).all():
        raise ValueError(f"诊断记录 {diagnostic_key(row)} 的 {key} 非法")
    return tensor


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.scorer_seed) != FROZEN_SEED:
        raise ValueError(f"scorer seed 已冻结为 {FROZEN_SEED}")
    if abs(float(args.margin_threshold) - FROZEN_MARGIN) > 1e-12:
        raise ValueError(f"margin 阈值已冻结为 {FROZEN_MARGIN}")

    qa_manifest = Path(args.qa_manifest).resolve()
    feature_root = Path(args.feature_root).resolve()
    diagnostics_path = Path(args.diagnostics).resolve()
    output_root = Path(args.output_root).resolve()
    model_path = Path(args.model_path).resolve()
    qa_rows = load_json(qa_manifest)
    if not isinstance(qa_rows, list) or not qa_rows:
        raise ValueError("QA manifest 必须是非空 JSON list")
    diagnostic_rows = {diagnostic_key(row): row for row in load_jsonl(diagnostics_path)}
    if len(diagnostic_rows) == 0:
        raise ValueError("诊断 JSONL 为空")

    manifests: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    stats: dict[str, dict[str, Any]] = {
        mode: {
            "sample_count": 0,
            "window_count": 0,
            "changed_window_count": 0,
            "max_adjustment": 0.0,
            "min_center_weight": 1.0,
            "finite": True,
        }
        for mode in MODES
    }
    exact_low_confidence = True
    exact_zero_offset = True
    offset_counts: Counter[str] = Counter()

    for qa in qa_rows:
        sample_id = qa_key(qa)
        if sample_id not in diagnostic_rows:
            raise ValueError(f"缺少 {sample_id} 的冻结 scorer 诊断")
        feature_value = qa.get("video_features")
        if not feature_value:
            raise ValueError(f"{sample_id} 未提供离线 video_features")
        feature_path = Path(str(feature_value))
        if not feature_path.is_absolute():
            feature_path = feature_root / feature_path
        features = load_feature(feature_path)
        steps = int(features.shape[0])
        diagnostic = diagnostic_rows[sample_id]
        offsets = resample_window_signal(
            signal(diagnostic, "best_offset", dtype=torch.float32),
            steps,
        )
        margins = resample_window_signal(
            signal(diagnostic, "margin", dtype=torch.float32),
            steps,
        )
        events = resample_window_signal(
            signal(diagnostic, "event_strength", dtype=torch.float32),
            steps,
        ).clamp(0.0, 1.0)
        for value in offsets.tolist():
            offset_counts[f"{value:.1f}"] += 1

        low_confidence = margins.lt(FROZEN_MARGIN)
        zero_offset = offsets.eq(0.0)
        for mode in MODES:
            output = apply_video_window_weighting(
                features,
                offsets,
                margins,
                event_strength=events,
                mode=mode,
                margin_threshold=FROZEN_MARGIN,
                max_neighbor_weight=float(args.max_neighbor_weight),
            )
            condition_dir = output_root / "features" / mode
            condition_dir.mkdir(parents=True, exist_ok=True)
            output_path = condition_dir / f"{sample_id}.pt"
            torch.save(
                {
                    "features": output.features,
                    "metadata": {
                        "sample_id": sample_id,
                        "mode": mode,
                        "diagnostic_only": True,
                        "scorer_seed": FROZEN_SEED,
                        "margin_threshold": FROZEN_MARGIN,
                        "max_neighbor_weight": float(args.max_neighbor_weight),
                    },
                },
                output_path,
            )
            manifest_row = dict(qa)
            manifest_row["video_features"] = str(output_path)
            manifest_row.pop("video_path", None)
            manifest_row["scene_audio"] = None
            manifest_row["scene_audio_path"] = None
            manifests[mode].append(manifest_row)

            mode_stats = stats[mode]
            mode_stats["sample_count"] += 1
            mode_stats["window_count"] += steps
            mode_stats["changed_window_count"] += int(output.changed.sum().item())
            mode_stats["max_adjustment"] = max(
                mode_stats["max_adjustment"],
                float(output.adjustment.max().item()),
            )
            mode_stats["min_center_weight"] = min(
                mode_stats["min_center_weight"],
                float(output.weights[:, 1].min().item()),
            )
            mode_stats["finite"] &= bool(torch.isfinite(output.features).all().item())
            exact_low_confidence &= torch.equal(
                output.features[low_confidence],
                features[low_confidence],
            )
            exact_zero_offset &= torch.equal(
                output.features[zero_offset],
                features[zero_offset],
            )

    plan_rows = []
    category_plan_rows = []
    categories = sorted(
        {
            str(row["evaluation_category"])
            for rows in manifests.values()
            for row in rows
            if row.get("evaluation_category")
        }
    )
    for mode in MODES:
        manifest_path = output_root / "manifests" / f"{mode}.json"
        write_json(manifest_path, manifests[mode])
        condition = stats[mode]
        condition["changed_rate"] = (
            condition["changed_window_count"] / condition["window_count"]
            if condition["window_count"]
            else 0.0
        )
        plan_rows.append(
            {
                "id": mode,
                "description": f"离线视频时间窗口消融：{mode}",
                "manifest": str(manifest_path),
                "model_key": "baseline_m4",
                "model_path": str(model_path),
                "audio_condition": "none",
                "alignment": "off",
                "gate_ablation": "none",
                "env": {
                    "AS_M4_ROLLBACK_MODE": "weights12k",
                    "AS_M4_ENABLE_SCENE_AUDIO": "0",
                    "AS_M4_FORCE_AUDIO_GATE": "0",
                    "AS_M4_TEMPORAL_OFFSET_GRU_ENABLED": "0",
                },
                "output_jsonl": str(output_root / "predictions" / f"{mode}.jsonl"),
            }
        )
        for category in categories:
            category_manifest = (
                output_root / "manifests" / "by_category" / f"{mode}__{category}.json"
            )
            write_json(
                category_manifest,
                [
                    row
                    for row in manifests[mode]
                    if str(row.get("evaluation_category")) == category
                ],
            )
            category_plan_rows.append(
                {
                    **plan_rows[-1],
                    "id": f"{mode}__{category}",
                    "description": f"离线视频时间窗口消融：{mode} / {category}",
                    "manifest": str(category_manifest),
                    "output_jsonl": str(
                        output_root
                        / "predictions_by_category"
                        / f"{mode}__{category}.jsonl"
                    ),
                }
            )
    plan_path = output_root / "ablation_plan.jsonl"
    write_jsonl(plan_path, plan_rows)
    category_plan_path = output_root / "ablation_plan_by_category.jsonl"
    if category_plan_rows:
        write_jsonl(category_plan_path, category_plan_rows)

    summary = {
        "diagnostic_only": True,
        "formal_streaming_path_modified": False,
        "moves_audio_window": False,
        "feeds_gate": False,
        "trains_m4": False,
        "scorer_seed": FROZEN_SEED,
        "margin_threshold": FROZEN_MARGIN,
        "max_neighbor_weight": float(args.max_neighbor_weight),
        "sample_count": len(qa_rows),
        "offset_distribution": dict(sorted(offset_counts.items())),
        "conditions": stats,
        "low_confidence_elementwise_identical": exact_low_confidence,
        "zero_offset_elementwise_identical": exact_zero_offset,
        "all_finite": all(item["finite"] for item in stats.values()),
        "paths": {
            "plan": str(plan_path),
            "category_plan": str(category_plan_path) if category_plan_rows else None,
            "summary": str(output_root / "preparation_summary.json"),
            "report": str(output_root / "preparation_report.md"),
        },
    }
    write_json(output_root / "preparation_summary.json", summary)
    write_report(output_root / "preparation_report.md", summary)
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# 离线视频时间窗口加权消融准备报告",
        "",
        "本实验只修改离线视频特征，不接入正式流式推理，不移动音频窗口，不接 Gate。",
        "",
        f"- 冻结 scorer seed：{summary['scorer_seed']}",
        f"- 冻结 margin：{summary['margin_threshold']}",
        f"- 邻窗最大软权重：{summary['max_neighbor_weight']}",
        f"- 样本数：{summary['sample_count']}",
        f"- offset 分布：`{summary['offset_distribution']}`",
        f"- 低置信逐元素一致：{summary['low_confidence_elementwise_identical']}",
        f"- `0s` 窗口逐元素一致：{summary['zero_offset_elementwise_identical']}",
        f"- 所有输出有限：{summary['all_finite']}",
        "",
        "| 条件 | 调整窗口数 | 调整率 | 最大邻窗权重 | 最小中心权重 |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        item = summary["conditions"][mode]
        lines.append(
            f"| {mode} | {item['changed_window_count']} | {item['changed_rate']:.4f} | "
            f"{item['max_adjustment']:.4f} | {item['min_center_weight']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-manifest", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--scorer-seed", type=int, default=FROZEN_SEED)
    parser.add_argument("--margin-threshold", type=float, default=FROZEN_MARGIN)
    parser.add_argument("--max-neighbor-weight", type=float, default=0.35)
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps({"ok": True, **summary["paths"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
