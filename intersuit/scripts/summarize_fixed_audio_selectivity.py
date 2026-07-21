#!/usr/bin/env python
"""汇总 Fixed BEATs 正确/错配/静音配对评测。"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


CONDITIONS = ("video_only", "correct", "mismatched", "silence", "gate0")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def index_rows(rows: list[dict[str, Any]], condition: str) -> dict[str, dict[str, Any]]:
    indexed = {str(row["id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"{condition} 存在重复预测 id")
    return indexed


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True


def task_type_map(manifest: Path) -> dict[str, str]:
    records = json.loads(manifest.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for record in records:
        conversations = record.get("conversations") or []
        for turn in range(len(conversations) // 2):
            result[f"{record['id']}_turn{turn}"] = str(record.get("task_type") or "unknown")
    return result


def row_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gates = [float(row["gate"]) for row in rows if row.get("gate") is not None]
    ratios = [float(row["delta_to_video_ratio"]) for row in rows if row.get("delta_to_video_ratio") is not None]
    return {
        "count": len(rows),
        "correct": sum(bool(row.get("correct")) for row in rows),
        "accuracy": sum(bool(row.get("correct")) for row in rows) / len(rows),
        "empty_output_count": sum(not str(row.get("prediction") or "").strip() for row in rows),
        "first_token_eos_count": sum(
            bool(((row.get("generation_debug") or {}).get("tokens") or {}).get("first_token_is_eos"))
            for row in rows
        ),
        "nonfinite_diagnostic_count": sum(not finite_tree(row.get("as_m4_diagnostics")) for row in rows),
        "mean_gate": mean(gates) if gates else None,
        "mean_delta_to_video_ratio": mean(ratios) if ratios else None,
    }


def paired_delta(candidate: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ids = sorted(baseline)
    gain = sum(bool(candidate[i]["correct"]) and not bool(baseline[i]["correct"]) for i in ids)
    harm = sum(bool(baseline[i]["correct"]) and not bool(candidate[i]["correct"]) for i in ids)
    same = sum(str(candidate[i].get("prediction") or "") == str(baseline[i].get("prediction") or "") for i in ids)
    return {"gain": gain, "harm": harm, "net_gain": gain - harm, "same_prediction_count": same, "same_prediction_rate": same / len(ids)}


def summarize(predictions_root: Path, manifest: Path) -> dict[str, Any]:
    rows = {
        condition: read_jsonl(predictions_root / f"dev760_{condition}_predictions.jsonl")
        for condition in CONDITIONS
    }
    indexed = {condition: index_rows(value, condition) for condition, value in rows.items()}
    expected_ids = set(indexed["video_only"])
    for condition in CONDITIONS[1:]:
        if set(indexed[condition]) != expected_ids:
            raise ValueError(f"{condition} 与 video_only 的预测 id 不一致")

    types = task_type_map(manifest)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for condition in CONDITIONS:
        for row in rows[condition]:
            grouped[types.get(str(row["id"]), "unknown")][condition].append(row)

    base = indexed["video_only"]
    correct = indexed["correct"]
    mismatch = indexed["mismatched"]
    ids = sorted(expected_ids)
    correct_unique = sum(bool(correct[i]["correct"]) and not bool(mismatch[i]["correct"]) for i in ids)
    mismatch_unique = sum(bool(mismatch[i]["correct"]) and not bool(correct[i]["correct"]) for i in ids)
    deltas = {condition: paired_delta(indexed[condition], base) for condition in CONDITIONS[1:]}
    summaries = {condition: row_summary(value) for condition, value in rows.items()}

    checks = {
        "correct_accuracy_ge_video_only": summaries["correct"]["accuracy"] >= summaries["video_only"]["accuracy"],
        "correct_accuracy_gt_mismatch": summaries["correct"]["accuracy"] > summaries["mismatched"]["accuracy"],
        "correct_unique_gain_positive": correct_unique > 0,
        "correct_net_gain_nonnegative": deltas["correct"]["net_gain"] >= 0,
        "correct_net_gain_gt_mismatch": deltas["correct"]["net_gain"] > deltas["mismatched"]["net_gain"],
        "silence_exact_fallback_ge_99pct": deltas["silence"]["same_prediction_rate"] >= 0.99,
        "gate0_exact_fallback": deltas["gate0"]["same_prediction_rate"] == 1.0,
        "outputs_and_diagnostics_valid": (
            summaries["correct"]["empty_output_count"] <= summaries["video_only"]["empty_output_count"]
            and summaries["mismatched"]["empty_output_count"] <= summaries["video_only"]["empty_output_count"]
            and summaries["correct"]["first_token_eos_count"] <= summaries["video_only"]["first_token_eos_count"]
            and summaries["mismatched"]["first_token_eos_count"] <= summaries["video_only"]["first_token_eos_count"]
            and all(summary["nonfinite_diagnostic_count"] == 0 for summary in summaries.values())
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "condition_summaries": summaries,
        "paired_vs_video_only": deltas,
        "correct_unique_vs_mismatch": correct_unique,
        "mismatch_unique_vs_correct": mismatch_unique,
        "task_type_summaries": {
            task_type: {condition: row_summary(task_rows) for condition, task_rows in conditions.items()}
            for task_type, conditions in sorted(grouped.items())
        },
    }


def write_report(result: dict[str, Any], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "selectivity_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = ["# Fixed BEATs dev760 语义选择性报告", "", f"状态：**{result['status']}**", "", "## 条件汇总", "", "| 条件 | 正确数 | 准确率 | 相对 video-only gain/harm/net | 相同预测率 |", "| --- | ---: | ---: | ---: | ---: |"]
    for condition in CONDITIONS:
        summary = result["condition_summaries"][condition]
        delta = result["paired_vs_video_only"].get(condition)
        delta_text = "-" if delta is None else f"{delta['gain']}/{delta['harm']}/{delta['net_gain']}"
        same_text = "-" if delta is None else f"{delta['same_prediction_rate']:.4f}"
        lines.append(f"| {condition} | {summary['correct']} | {summary['accuracy']:.4f} | {delta_text} | {same_text} |")
    lines.extend(["", "## 选择性", "", f"- 正确音频独有答对：{result['correct_unique_vs_mismatch']}", f"- 错配音频独有答对：{result['mismatch_unique_vs_correct']}", "", "## 门禁", ""])
    lines.extend(f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in result["checks"].items())
    (output_root / "selectivity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = summarize(Path(args.predictions_root), Path(args.manifest))
    write_report(result, Path(args.output_root))
    print(json.dumps({"status": result["status"], "checks": result["checks"]}, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
