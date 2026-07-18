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
    results = {}
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    for mode in MODES:
        rows = load_jsonl(root / "predictions" / f"{mode}.jsonl")
        predictions[mode] = {str(row["sample_id"]): row for row in rows}
        correct = sum(bool(row.get("correct")) for row in rows)
        results[mode] = {
            "sample_count": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows) if rows else None,
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
    result["formal_acceptance_passed"] = all(
        (
            result["joint_above_baseline"],
            result["joint_not_below_hard_move"],
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
    lines.extend(
        [
            "",
            f"- 联合软加权高于基线：{result['joint_above_baseline']}",
            f"- 联合软加权不差于硬移动：{result['joint_not_below_hard_move']}",
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
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
