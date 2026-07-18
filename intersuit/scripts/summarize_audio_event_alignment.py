#!/usr/bin/env python
"""汇总 AS-M4 音频事件局部对齐诊断结果。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


DIAGNOSTIC_KEYS = (
    "event_strength",
    "is_silent_window",
    "audio_rms",
    "audio_peak",
    "candidate_offsets",
    "candidate_valid",
    "semantic_similarity",
    "video_event_strength",
    "candidate_scores",
    "best_offset",
    "best_alignment_score",
    "second_best_alignment_score",
    "alignment_margin",
    "alignment_confidence",
    "offset_scorer_candidate_scores",
    "offset_scorer_best_offset",
    "offset_scorer_margin",
    "offset_scorer_accepted",
    "offset_scorer_suggested_offset",
    "offset_scorer_available",
)

CONDITION_LABELS = {
    "S0_DEFAULT_OFF": "default_off",
    "S1_ORIGINAL": "original",
    "S2_SILENCE": "silence",
    "S3_WRONG": "wrong",
    "S4_SHIFT_PLUS_05": "shift_plus_0.5",
    "S5_SHIFT_MINUS_05": "shift_minus_0.5",
    "S6_SHIFT_PLUS_1": "shift_plus_1.0",
    "S7_SHIFT_MINUS_1": "shift_minus_1.0",
    "S8_E7_GATE0": "e7_gate0",
    "F1_ORIGINAL": "original",
    "F2_SILENCE": "silence",
    "F3_WRONG": "wrong",
    "F4_SHIFT_PLUS_05": "shift_plus_0.5",
    "F5_SHIFT_MINUS_05": "shift_minus_0.5",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def collect_rows(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not directory.exists():
        return rows
    for path in sorted(directory.glob("*.jsonl")):
        for row in read_jsonl(path):
            row["source_jsonl"] = str(path)
            rows.append(row)
    return rows


def flatten_numbers(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        values: list[float] = []
        for item in value:
            values.extend(flatten_numbers(item))
        return values
    return []


def first_alignment_diagnostic(row: dict[str, Any]) -> dict[str, Any] | None:
    diagnostics = row.get("as_m4_diagnostics")
    if not isinstance(diagnostics, list):
        return None
    for item in diagnostics:
        if isinstance(item, dict) and item.get("audio_event_aligner_v1_enabled") is True:
            return item
    return None


def numeric_mean(value: Any) -> float | None:
    values = [number for number in flatten_numbers(value) if math.isfinite(number)]
    return mean(values) if values else None


def numeric_max(value: Any) -> float | None:
    values = [number for number in flatten_numbers(value) if math.isfinite(number)]
    return max(values) if values else None


def offset_distribution(value: Any) -> dict[str, int]:
    values = flatten_numbers(value)
    counts = Counter(f"{number:.1f}" for number in values if math.isfinite(number))
    return dict(sorted(counts.items()))


def true_ratio(value: Any) -> float | None:
    if not isinstance(value, list):
        return None
    flattened = []
    stack = list(value)
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, bool):
            flattened.append(item)
    return mean(flattened) if flattened else None


def offset_jump_rate(value: Any) -> float | None:
    values = [number for number in flatten_numbers(value) if math.isfinite(number)]
    if len(values) < 2:
        return 0.0 if values else None
    return sum(left != right for left, right in zip(values, values[1:])) / (len(values) - 1)


def count_non_finite(value: Any) -> tuple[int, int]:
    nan_count = 0
    inf_count = 0
    for number in flatten_numbers(value):
        if math.isnan(number):
            nan_count += 1
        elif math.isinf(number):
            inf_count += 1
    return nan_count, inf_count


def summarize_row(row: dict[str, Any]) -> dict[str, Any]:
    diagnostic = first_alignment_diagnostic(row)
    experiment_id = str(row.get("experiment_id") or "")
    summary = {
        "sample_id": row.get("sample_id"),
        "experiment_id": experiment_id,
        "condition": CONDITION_LABELS.get(experiment_id, experiment_id.lower()),
        "prediction": row.get("prediction"),
        "correct": row.get("correct"),
        "gate": row.get("gate"),
        "delta_to_video_ratio": row.get("delta_to_video_ratio"),
        "source_jsonl": row.get("source_jsonl"),
        "diagnostics_present": diagnostic is not None,
    }
    if diagnostic is None:
        summary.update(
            {
                "event_strength_mean": None,
                "event_strength_max": None,
                "best_offset_distribution": {},
                "suggested_offset_distribution": {},
                "offset_scorer_best_offset_distribution": {},
                "offset_scorer_available_ratio": None,
                "offset_scorer_accepted_ratio": None,
                "offset_scorer_margin_mean": None,
                "offset_scorer_jump_rate": None,
                "alignment_score_mean": None,
                "alignment_margin_mean": None,
                "alignment_confidence_mean": None,
                "nan_count": 0,
                "inf_count": 0,
            }
        )
        return summary

    selected = {key: diagnostic.get(key) for key in DIAGNOSTIC_KEYS}
    nan_count, inf_count = count_non_finite(selected)
    summary.update(
        {
            "event_strength_mean": numeric_mean(diagnostic.get("event_strength")),
            "event_strength_max": numeric_max(diagnostic.get("event_strength")),
            "best_offset_distribution": offset_distribution(diagnostic.get("best_offset")),
            "alignment_score_mean": numeric_mean(diagnostic.get("best_alignment_score")),
            "alignment_margin_mean": numeric_mean(diagnostic.get("alignment_margin")),
            "alignment_confidence_mean": numeric_mean(diagnostic.get("alignment_confidence")),
            "offset_scorer_available_ratio": true_ratio(diagnostic.get("offset_scorer_available")),
            "offset_scorer_accepted_ratio": true_ratio(diagnostic.get("offset_scorer_accepted")),
            "offset_scorer_best_offset_distribution": offset_distribution(
                diagnostic.get("offset_scorer_best_offset")
            ),
            "suggested_offset_distribution": offset_distribution(
                diagnostic.get("offset_scorer_suggested_offset")
            ),
            "offset_scorer_margin_mean": numeric_mean(diagnostic.get("offset_scorer_margin")),
            "offset_scorer_jump_rate": offset_jump_rate(
                diagnostic.get("offset_scorer_suggested_offset")
            ),
            "nan_count": nan_count,
            "inf_count": inf_count,
        }
    )
    return summary


def generated_signature(row: dict[str, Any]) -> Any:
    tokens = ((row.get("generation_debug") or {}).get("tokens") or {})
    return {
        "prediction": row.get("prediction"),
        "full_generated_token_count": tokens.get("full_generated_token_count"),
        "first_20_full_generated_ids": tokens.get("first_20_full_generated_ids"),
    }


def find_condition(summaries: list[dict[str, Any]], condition: str) -> dict[str, Any] | None:
    return next((item for item in summaries if item.get("condition") == condition), None)


def directional_fraction(summary: dict[str, Any] | None, sign: int) -> float:
    if not summary:
        return 0.0
    distribution = summary.get("best_offset_distribution") or {}
    total = sum(int(value) for value in distribution.values())
    if total == 0:
        return 0.0
    matched = sum(
        int(count)
        for offset, count in distribution.items()
        if (float(offset) < 0 if sign < 0 else float(offset) > 0)
    )
    return matched / total


def build_checks(
    single_rows: list[dict[str, Any]],
    single_summaries: list[dict[str, Any]],
    five_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    rows_by_condition = {
        CONDITION_LABELS.get(str(row.get("experiment_id") or ""), ""): row
        for row in single_rows
    }
    off = find_condition(single_summaries, "default_off")
    original = find_condition(single_summaries, "original")
    silence = find_condition(single_summaries, "silence")
    wrong = find_condition(single_summaries, "wrong")
    plus = find_condition(single_summaries, "shift_plus_0.5")
    minus = find_condition(single_summaries, "shift_minus_0.5")
    e7 = find_condition(single_summaries, "e7_gate0")

    off_row = rows_by_condition.get("default_off")
    original_row = rows_by_condition.get("original")
    default_noop = bool(off_row and original_row and generated_signature(off_row) == generated_signature(original_row))
    silence_low = bool(
        silence
        and (silence.get("event_strength_mean") or 0.0) <= 1e-8
        and (silence.get("alignment_confidence_mean") or 0.0) <= (original or {}).get("alignment_confidence_mean", 0.0)
    )
    single_correct_beats_wrong = bool(
        original
        and wrong
        and original.get("alignment_score_mean") is not None
        and wrong.get("alignment_score_mean") is not None
        and original["alignment_score_mean"] > wrong["alignment_score_mean"]
    )
    plus_corrects = directional_fraction(plus, -1) > directional_fraction(plus, 1)
    minus_corrects = directional_fraction(minus, 1) > directional_fraction(minus, -1)
    e7_exact = bool(e7 and float(e7.get("gate") or 0.0) == 0.0 and float(e7.get("delta_to_video_ratio") or 0.0) == 0.0)

    five_by_sample: dict[str, dict[str, dict[str, Any]]] = {}
    for item in five_summaries:
        five_by_sample.setdefault(str(item["sample_id"]), {})[str(item["condition"])] = item
    score_wins = 0
    comparable = 0
    for conditions in five_by_sample.values():
        correct = conditions.get("original")
        mismatch = conditions.get("wrong")
        if correct and mismatch and correct.get("alignment_score_mean") is not None and mismatch.get("alignment_score_mean") is not None:
            comparable += 1
            score_wins += int(correct["alignment_score_mean"] > mismatch["alignment_score_mean"])

    confidence_values = [
        float(item["alignment_confidence_mean"])
        for item in [*single_summaries, *five_summaries]
        if item.get("alignment_confidence_mean") is not None
    ]
    all_summaries = [*single_summaries, *five_summaries]
    no_numeric_anomalies = all(item.get("nan_count", 0) == 0 and item.get("inf_count", 0) == 0 for item in all_summaries)
    checks = {
        "default_disabled_exact_noop": default_noop,
        "silence_event_and_confidence_low": silence_low,
        "single_correct_score_above_wrong": single_correct_beats_wrong,
        "shift_plus_0.5_prefers_negative_correction": plus_corrects,
        "shift_minus_0.5_prefers_positive_correction": minus_corrects,
        "five_sample_correct_score_win_count": score_wins,
        "five_sample_comparable_count": comparable,
        "five_sample_correct_score_usually_above_wrong": comparable > 0 and score_wins > comparable / 2,
        "alignment_confidence_not_constant": len(confidence_values) > 1 and pstdev(confidence_values) > 1e-8,
        "no_nan_or_inf": no_numeric_anomalies,
        "e7_gate_and_residual_exact_zero": e7_exact,
        "five_sample_status": "completed" if five_summaries else "not_run_due_single_sample_pause",
    }
    required = (
        "default_disabled_exact_noop",
        "silence_event_and_confidence_low",
        "single_correct_score_above_wrong",
        "shift_plus_0.5_prefers_negative_correction",
        "shift_minus_0.5_prefers_positive_correction",
        "five_sample_correct_score_usually_above_wrong",
        "alignment_confidence_not_constant",
        "no_nan_or_inf",
        "e7_gate_and_residual_exact_zero",
    )
    checks["stage_decision"] = "可以进入动态修正窗口阶段" if all(checks[key] for key in required) else "暂停接入正式融合"
    return checks


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "sample_id",
        "experiment_id",
        "condition",
        "event_strength_mean",
        "event_strength_max",
        "best_offset_distribution",
        "alignment_score_mean",
        "alignment_margin_mean",
        "alignment_confidence_mean",
        "offset_scorer_available_ratio",
        "offset_scorer_accepted_ratio",
        "offset_scorer_margin_mean",
        "offset_scorer_best_offset_distribution",
        "suggested_offset_distribution",
        "offset_scorer_jump_rate",
        "nan_count",
        "inf_count",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {key: row.get(key) for key in fields}
            output["best_offset_distribution"] = json.dumps(output["best_offset_distribution"], ensure_ascii=False, sort_keys=True)
            output["offset_scorer_best_offset_distribution"] = json.dumps(
                output["offset_scorer_best_offset_distribution"],
                ensure_ascii=False,
                sort_keys=True,
            )
            output["suggested_offset_distribution"] = json.dumps(
                output["suggested_offset_distribution"],
                ensure_ascii=False,
                sort_keys=True,
            )
            writer.writerow(output)


def write_report(
    path: Path,
    single_summaries: list[dict[str, Any]],
    five_summaries: list[dict[str, Any]],
    checks: dict[str, Any],
) -> None:
    lines = [
        "# 音频事件感知的局部动态对齐 v1 报告",
        "",
        "## 实现说明",
        "",
        "事件强度由 RMS、峰值幅度、非静音比例和相邻编码特征变化加权组成，权重分别为 0.40、0.20、0.25、0.15。静音窗口最终强度硬归零，避免邻接事件变化抬高静音评分。",
        "",
        "候选偏移固定为 `[-0.5, 0, +0.5]` 秒。偏移表示对观测音频时间轴施加的修正量，候选音频中心按 `video_time - offset` 查找最近窗口；越过首尾边界的候选标记为无效，不循环音频。",
        "",
        "候选分数以跨模态语义相似度为主：先分别对音频候选特征和视频窗口特征做无参数标准化投影与 L2 归一化，再计算余弦相似度并映射到 `[0, 1]`。事件强度只以小权重 `0.05` 参与，包括候选事件强度和与视频时序变化强度的一致性；静音候选分数仍归零。置信度为 `best_score * (0.5 + 0.5 * margin)`。本阶段只写诊断，不替换正式音频窗口，不改变 Gate v1、残差或预测路径。",
        "",
        "## 单样本结果",
        "",
        "| 条件 | 事件强度均值 | 最佳分数均值 | margin 均值 | 置信度均值 | best_offset 分布 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in single_summaries:
        lines.append(
            "| {condition} | {event} | {score} | {margin} | {confidence} | `{offsets}` |".format(
                condition=item["condition"],
                event=_format_number(item.get("event_strength_mean")),
                score=_format_number(item.get("alignment_score_mean")),
                margin=_format_number(item.get("alignment_margin_mean")),
                confidence=_format_number(item.get("alignment_confidence_mean")),
                offsets=json.dumps(item.get("best_offset_distribution") or {}, ensure_ascii=False, sort_keys=True),
            )
        )
    scorer_rows = [
        item
        for item in [*single_summaries, *five_summaries]
        if (item.get("offset_scorer_available_ratio") or 0.0) > 0
    ]
    lines.extend(
        [
            "",
            "## 冻结 offset scorer 诊断旁路",
            "",
            "该表只记录建议，不移动音频窗口，也不把结果接入 Gate。",
            "",
            "| 样本/条件 | 可用率 | 接受率 | margin 均值 | raw offset 分布 | 建议 offset 分布 | 相邻建议跳变率 |",
            "|---|---:|---:|---:|---|---|---:|",
        ]
    )
    for item in scorer_rows:
        lines.append(
            "| {sample}/{condition} | {available} | {accepted} | {margin} | `{raw}` | `{suggested}` | {jump} |".format(
                sample=item.get("sample_id"),
                condition=item.get("condition"),
                available=_format_number(item.get("offset_scorer_available_ratio")),
                accepted=_format_number(item.get("offset_scorer_accepted_ratio")),
                margin=_format_number(item.get("offset_scorer_margin_mean")),
                raw=json.dumps(
                    item.get("offset_scorer_best_offset_distribution") or {},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                suggested=json.dumps(
                    item.get("suggested_offset_distribution") or {},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                jump=_format_number(item.get("offset_scorer_jump_rate")),
            )
        )

    lines.extend(
        [
            "",
            "## 五样本结果",
            "",
            (
                f"共汇总 `{len(five_summaries)}` 条样本-条件记录，逐条数据见 `five_sample_alignment_summary.csv`。"
                if five_summaries
                else "未运行固定 5 条推理。原因是单样本已触发暂停条件：正确/错误音频不可分，且 `-0.5s` 人工错位未产生合理正向修正。"
            ),
            "",
            "## 阶段判断",
            "",
        ]
    )
    for key, value in checks.items():
        if key == "stage_decision":
            continue
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            f"结论：**{checks['stage_decision']}**。",
            "",
            "若结论为暂停，应保持本阶段诊断旁路，不根据 `best_offset` 改写音频窗口，也不把 alignment confidence 接入 Gate。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_number(value: Any) -> str:
    return "-" if value is None else f"{float(value):.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总音频事件局部对齐诊断结果。")
    parser.add_argument(
        "--artifact_dir",
        default="intersuit/harness/artifacts/audio_event_aligner_v1",
    )
    args = parser.parse_args()
    artifact_dir = Path(args.artifact_dir)
    single_rows = collect_rows(artifact_dir / "single_predictions")
    five_rows = collect_rows(artifact_dir / "five_predictions")
    single_summaries = [summarize_row(row) for row in single_rows]
    five_summaries = [summarize_row(row) for row in five_rows]
    checks = build_checks(single_rows, single_summaries, five_summaries)

    compact_diagnostics = []
    for row in [*single_rows, *five_rows]:
        diagnostic = first_alignment_diagnostic(row)
        compact_diagnostics.append(
            {
                "sample_id": row.get("sample_id"),
                "experiment_id": row.get("experiment_id"),
                "condition": CONDITION_LABELS.get(str(row.get("experiment_id") or "")),
                "diagnostics": None if diagnostic is None else {key: diagnostic.get(key) for key in DIAGNOSTIC_KEYS},
            }
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "single_sample_alignment_summary.json").write_text(
        json.dumps({"records": single_summaries, "checks": checks}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(artifact_dir / "five_sample_alignment_summary.csv", five_summaries)
    (artifact_dir / "alignment_diagnostics.json").write_text(
        json.dumps({"records": compact_diagnostics}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(artifact_dir / "report.md", single_summaries, five_summaries, checks)
    print(json.dumps({"single_records": len(single_summaries), "five_records": len(five_summaries), **checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
