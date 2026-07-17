#!/usr/bin/env python
"""按冻结配置运行 AVE_HF 时间同步 offset scorer 多随机种子复验。

本脚本只复用 `train_ave_hf_temporal_offset_scorer.py` 的独立诊断训练入口，
不接 Gate、不启用动态窗口、不修改正式 M4 推理路径。默认配置固定为
中心峰过滤和 0s 类权重 1.25；扩展 AVE 数据后只应替换 manifest 路径。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"
DEFAULT_CONFIG = INTERSUIT_ROOT / "harness/configs/ave_hf_temporal_offset_zero125_centerpeak.json"


def import_temporal_module():
    path = INTERSUIT_ROOT / "scripts/train_ave_hf_temporal_offset_scorer.py"
    spec = importlib.util.spec_from_file_location("ave_hf_temporal_offset_scorer_frozen", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


temporal = import_temporal_module()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def namespace_from_config(config: dict[str, Any], seed: int, output_root: Path) -> argparse.Namespace:
    parser = temporal.build_parser()
    args = parser.parse_args([])
    mapping = {
        "clip_manifest": "clip_manifest",
        "rgb_manifest": "rgb_manifest",
        "split_summary": "split_summary",
        "top_k_windows": "top_k_windows",
        "min_change_quantile": "min_change_quantile",
        "min_candidate_change_margin": "min_candidate_change_margin",
        "require_center_peak": "require_center_peak",
        "min_center_audio_dominance": "min_center_audio_dominance",
        "min_center_video_dominance": "min_center_video_dominance",
        "context_radius": "context_radius",
        "hidden_dim": "hidden_dim",
        "lr": "lr",
        "zero_class_weight": "zero_class_weight",
        "loss_type": "loss_type",
        "focal_gamma": "focal_gamma",
        "keep_zero_margin_threshold": "keep_zero_margin_threshold",
        "overfit_steps": "overfit_steps",
        "overfit_samples": "overfit_samples",
        "batch_equivalent_size": "batch_equivalent_size",
        "pass_accuracy": "pass_accuracy",
        "min_val_samples": "min_val_samples",
    }
    for config_key, arg_key in mapping.items():
        if config_key in config:
            setattr(args, arg_key, config[config_key])
    args.clip_manifest = resolve_path(args.clip_manifest)
    args.rgb_manifest = resolve_path(args.rgb_manifest)
    args.split_summary = resolve_path(args.split_summary)
    args.seed = int(seed)
    args.output_root = str(output_root)
    return args


def summarize_seed(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    final = summary["runs"][-1]["validation"]
    return {
        "name": name,
        "decision": summary["decision"],
        "sample_count": final["sample_count"],
        "accuracy": final["accuracy"],
        "margin_mean": final["margin_mean"],
        "confidence_mean": final["confidence_mean"],
        "condition_accuracy": final["condition_accuracy"],
        "prediction_distribution": final["prediction_distribution"],
        "zero_false_shift_count": final.get("zero_false_shift_count"),
        "all_scores_finite": final["all_scores_finite"],
        "collapsed_scores": final["collapsed_scores"],
    }


def metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def condition_values(rows: list[dict[str, Any]], condition: str) -> list[float]:
    return [float(row["condition_accuracy"].get(condition, 0.0)) for row in rows]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "accuracy": metric_values(rows, "accuracy"),
        "margin_mean": metric_values(rows, "margin_mean"),
        "original_accuracy": condition_values(rows, "original"),
        "shift_plus_0.5_accuracy": condition_values(rows, "shift_plus_0.5"),
        "shift_minus_0.5_accuracy": condition_values(rows, "shift_minus_0.5"),
    }
    return {
        key: {
            "mean": mean(values),
            "pstdev": pstdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
        for key, values in metrics.items()
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    agg = payload["aggregate"]
    lines = [
        "# AVE_HF 时间同步冻结配置多种子复验",
        "",
        "本报告只复验独立 offset scorer；未接 Gate、未启用动态窗口、未修改正式 M4 推理路径。",
        "",
        f"- 配置文件：`{payload['config_path']}`",
        f"- seeds：`{payload['seeds']}`",
        f"- 样本数要求：验证集至少 {payload['config']['min_val_samples']} 条",
        f"- 冻结配置：中心峰过滤={payload['config']['require_center_peak']}，0s 权重={payload['config']['zero_class_weight']}",
        "",
        "## 汇总",
        f"- accuracy：mean={agg['accuracy']['mean']:.4f}, std={agg['accuracy']['pstdev']:.4f}, values={agg['accuracy']['values']}",
        f"- margin：mean={agg['margin_mean']['mean']:.4f}, std={agg['margin_mean']['pstdev']:.4f}, values={agg['margin_mean']['values']}",
        f"- original：mean={agg['original_accuracy']['mean']:.4f}, std={agg['original_accuracy']['pstdev']:.4f}, values={agg['original_accuracy']['values']}",
        f"- shift_plus_0.5：mean={agg['shift_plus_0.5_accuracy']['mean']:.4f}, std={agg['shift_plus_0.5_accuracy']['pstdev']:.4f}, values={agg['shift_plus_0.5_accuracy']['values']}",
        f"- shift_minus_0.5：mean={agg['shift_minus_0.5_accuracy']['mean']:.4f}, std={agg['shift_minus_0.5_accuracy']['pstdev']:.4f}, values={agg['shift_minus_0.5_accuracy']['values']}",
        "",
        "## 单种子结果",
    ]
    for row in payload["runs"]:
        lines.append(
            f"- {row['name']}：decision={row['decision']}，n={row['sample_count']}，"
            f"acc={row['accuracy']:.4f}，margin={row['margin_mean']:.4f}，"
            f"condition={row['condition_accuracy']}，pred={row['prediction_distribution']}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    if args.clip_manifest:
        config["clip_manifest"] = args.clip_manifest
    if args.rgb_manifest:
        config["rgb_manifest"] = args.rgb_manifest
    if args.split_summary:
        config["split_summary"] = args.split_summary
    if args.output_root:
        config["output_root"] = args.output_root
    seeds = [int(seed) for seed in (args.seeds.split(",") if args.seeds else config["seeds"])]
    output_root = Path(resolve_path(config["output_root"]))
    runs = []
    for seed in seeds:
        seed_root = output_root / f"seed_{seed}"
        run_config = deepcopy(config)
        summary = temporal.run(namespace_from_config(run_config, seed, seed_root))
        runs.append(summarize_seed(f"seed_{seed}", summary))
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config": config,
        "seeds": seeds,
        "runs": runs,
        "aggregate": aggregate(runs),
        "gate_or_dynamic_window_modified": False,
        "paths": {
            "summary": str(output_root / "frozen_seed_summary.json"),
            "report": str(output_root / "frozen_seed_report.md"),
        },
    }
    write_json(output_root / "frozen_seed_summary.json", payload)
    write_report(output_root / "frozen_seed_report.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--clip-manifest", default=None)
    parser.add_argument("--rgb-manifest", default=None)
    parser.add_argument("--split-summary", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--seeds", default=None, help="逗号分隔随机种子；默认读取配置文件")
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps({"ok": True, "summary": summary["paths"]["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
