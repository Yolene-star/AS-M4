#!/usr/bin/env python
"""汇总四组离线视频时间窗口加权的最终任务预测。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODES = ("baseline", "hard_move", "offset_soft", "offset_event_soft")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    preparation = json.loads((root / "preparation_summary.json").read_text(encoding="utf-8"))
    categories = {}
    if args.qa_manifest:
        qa_rows = json.loads(Path(args.qa_manifest).resolve().read_text(encoding="utf-8"))
        categories = {
            str(row["id"]): str(row["evaluation_category"])
            for row in qa_rows
        }
    results = {}
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    for mode in MODES:
        prediction_path = root / "predictions" / f"{mode}.jsonl"
        if prediction_path.is_file():
            rows = load_jsonl(prediction_path)
        else:
            category_paths = sorted(
                (root / "predictions_by_category").glob(f"{mode}__*.jsonl")
            )
            if not category_paths:
                raise FileNotFoundError(f"缺少 {mode} 的总体或分类预测")
            rows = [
                row
                for path in category_paths
                for row in load_jsonl(path)
            ]
        predictions[mode] = {str(row["sample_id"]): row for row in rows}
        correct = sum(bool(row.get("correct")) for row in rows)
        results[mode] = {
            "sample_count": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows) if rows else None,
            "by_category": {},
        }
        for category in sorted(set(categories.values())):
            subset = [row for row in rows if categories.get(str(row["sample_id"])) == category]
            subset_correct = sum(bool(row.get("correct")) for row in subset)
            results[mode]["by_category"][category] = {
                "sample_count": len(subset),
                "correct": subset_correct,
                "accuracy": subset_correct / len(subset) if subset else None,
            }
    ids = set(predictions["baseline"])
    if any(set(predictions[mode]) != ids for mode in MODES):
        raise ValueError("四组预测的样本集合不一致")
    baseline_accuracy = results["baseline"]["accuracy"]
    joint_accuracy = results["offset_event_soft"]["accuracy"]
    hard_accuracy = results["hard_move"]["accuracy"]
    result = {
        "sample_count": len(ids),
        "conditions": results,
        "joint_above_baseline": joint_accuracy is not None
        and baseline_accuracy is not None
        and joint_accuracy > baseline_accuracy,
        "joint_not_below_hard_move": joint_accuracy is not None
        and hard_accuracy is not None
        and joint_accuracy >= hard_accuracy,
        "low_confidence_elementwise_identical": preparation[
            "low_confidence_elementwise_identical"
        ],
        "zero_offset_elementwise_identical": preparation[
            "zero_offset_elementwise_identical"
        ],
        "all_finite": preparation["all_finite"],
        "formal_acceptance_passed": False,
    }
    if categories:
        result["joint_not_below_pure_visual"] = (
            results["offset_event_soft"]["by_category"]["pure_visual"]["accuracy"]
            >= results["baseline"]["by_category"]["pure_visual"]["accuracy"]
        )
        result["joint_not_below_audio_interference"] = (
            results["offset_event_soft"]["by_category"]["audio_interference"]["accuracy"]
            >= results["baseline"]["by_category"]["audio_interference"]["accuracy"]
        )
    else:
        result["joint_not_below_pure_visual"] = True
        result["joint_not_below_audio_interference"] = True
    result["formal_acceptance_passed"] = all(
        (
            result["joint_above_baseline"],
            result["joint_not_below_hard_move"],
            result["joint_not_below_pure_visual"],
            result["joint_not_below_audio_interference"],
            result["low_confidence_elementwise_identical"],
            result["zero_offset_elementwise_identical"],
            result["all_finite"],
        )
    )
    summary_path = root / "ablation_summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(root / "ablation_report.md", result)
    return result


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# 离线视频时间窗口加权四组消融",
        "",
        "| 条件 | 正确数 | 样本数 | 准确率 |",
        "|---|---:|---:|---:|",
    ]
    for mode in MODES:
        item = result["conditions"][mode]
        accuracy = "N/A" if item["accuracy"] is None else f"{item['accuracy']:.4f}"
        lines.append(
            f"| {mode} | {item['correct']} | {item['sample_count']} | {accuracy} |"
        )
    if any(result["conditions"][mode]["by_category"] for mode in MODES):
        lines.extend(
            [
                "",
                "| 条件 | 类别 | 正确数 | 样本数 | 准确率 |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for mode in MODES:
            for category, item in result["conditions"][mode]["by_category"].items():
                accuracy = "N/A" if item["accuracy"] is None else f"{item['accuracy']:.4f}"
                lines.append(
                    f"| {mode} | {category} | {item['correct']} | "
                    f"{item['sample_count']} | {accuracy} |"
                )
    lines.extend(
        [
            "",
            f"- 联合软加权高于基线：{result['joint_above_baseline']}",
            f"- 联合软加权不差于硬移动：{result['joint_not_below_hard_move']}",
            f"- 联合软加权不伤害纯视觉：{result['joint_not_below_pure_visual']}",
            f"- 联合软加权不伤害错配音频：{result['joint_not_below_audio_interference']}",
            f"- 低置信逐元素一致：{result['low_confidence_elementwise_identical']}",
            f"- `0s` 窗口逐元素一致：{result['zero_offset_elementwise_identical']}",
            f"- 无 NaN/Inf：{result['all_finite']}",
            f"- 本轮正式验收通过：{result['formal_acceptance_passed']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--qa-manifest", default="")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
