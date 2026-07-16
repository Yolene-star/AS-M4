"""Metrics for AS-M4 E0-E7 attribution experiments."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"JSONL rows must be objects: {path}")
                records.append(value)
    return records


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def record_correct(record: dict[str, Any]) -> bool:
    if "correct" in record:
        return bool(record["correct"])
    return normalize_text(record.get("prediction")) == normalize_text(record.get("answer"))


def accuracy(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return sum(1 for record in records if record_correct(record)) / len(records)


def mean_optional(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    if not values:
        return None
    return mean(values)


def offset_mae(records: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for record in records:
        pred = record.get("pred_offset_sec")
        target = record.get("target_offset_sec")
        if pred is not None and target is not None:
            values.append(abs(float(pred) - float(target)))
    if not values:
        return None
    return mean(values)


def summarize_predictions(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_samples": len(records),
        "accuracy": accuracy(records),
        "mean_gate": mean_optional(records, "gate"),
        "mean_quality": mean_optional(records, "quality_gate"),
        "mean_relevance": mean_optional(records, "relevance_gate"),
        "offset_mae": offset_mae(records),
    }


def compare(summary: dict[str, dict[str, Any]], tolerance: float = 0.0) -> dict[str, Any]:
    def acc(exp_id: str) -> float:
        return float(summary.get(exp_id, {}).get("accuracy", 0.0))

    checks = {
        "e2_gt_e1": acc("E2") > acc("E1") + tolerance,
        "e2_gt_e3": acc("E2") > acc("E3") + tolerance,
        "e6_gt_e5": acc("E6") > acc("E5") + tolerance,
        "e2_gt_e7": acc("E2") > acc("E7") + tolerance,
        "e1_close_e0": abs(acc("E1") - acc("E0")) <= max(tolerance, 0.05),
    }
    checks["all_core_passed"] = all(checks.values())
    return checks
