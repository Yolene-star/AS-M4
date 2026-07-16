#!/usr/bin/env python
"""Score AS-M4 E0-E7 attribution predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intersuit.harness.metrics.attribution_metrics import compare, read_jsonl, summarize_predictions


def load_plan(plan_path: Path) -> list[dict]:
    records = read_jsonl(plan_path)
    ids = [record.get("id") for record in records]
    expected = [f"E{i}" for i in range(8)]
    if ids != expected:
        raise ValueError(f"plan must contain E0-E7 in order, got {ids}")
    return records


def score_plan(plan_path: Path, output_dir: Path, tolerance: float = 0.0) -> dict:
    plan = load_plan(plan_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}
    missing: list[str] = []
    for item in plan:
        exp_id = str(item["id"])
        pred_path = Path(str(item["output_jsonl"]))
        if not pred_path.exists():
            missing.append(str(pred_path))
            continue
        summaries[exp_id] = summarize_predictions(read_jsonl(pred_path))

    comparison = compare(summaries, tolerance=tolerance) if not missing else {"all_core_passed": False}
    result = {
        "status": "pass" if not missing and comparison.get("all_core_passed") else "fail",
        "missing_prediction_files": missing,
        "summaries": summaries,
        "comparison": comparison,
        "tolerance": tolerance,
    }
    (output_dir / "score_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Score AS-M4 E0-E7 prediction JSONL files.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output_dir", default="intersuit/harness/artifacts/as_m4_e0_e7_score")
    parser.add_argument("--tolerance", type=float, default=0.0)
    args = parser.parse_args()

    result = score_plan(Path(args.plan), Path(args.output_dir), tolerance=args.tolerance)
    print(json.dumps({"status": result["status"], "comparison": result["comparison"], "missing": result["missing_prediction_files"]}, ensure_ascii=False))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
