#!/usr/bin/env python
"""合并 AVUT 音频任务与冻结的 AVE 视觉/干扰样本，构建 60 条评测集。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


INTERSUIT_ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = INTERSUIT_ROOT / "harness/artifacts/video_window_weighting_ave_final60/eval_definition"
DEFAULT_OUTPUT = INTERSUIT_ROOT / "harness/artifacts/video_window_weighting_mixed_final60/eval_definition"
FROZEN_MARGIN = 0.15
MAX_NEIGHBOR_WEIGHT = 0.35


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if abs(float(args.margin_threshold) - FROZEN_MARGIN) > 1e-12:
        raise ValueError(f"margin 已冻结为 {FROZEN_MARGIN}")
    if abs(float(args.max_neighbor_weight) - MAX_NEIGHBOR_WEIGHT) > 1e-12:
        raise ValueError(f"邻窗权重上限已冻结为 {MAX_NEIGHBOR_WEIGHT}")

    old_qa = load_json(Path(args.ave_qa_manifest).resolve())
    old_clip = load_jsonl(Path(args.ave_clip_manifest).resolve())
    old_rgb = load_jsonl(Path(args.ave_rgb_manifest).resolve())
    avut_qa = load_json(Path(args.avut_qa_manifest).resolve())
    avut_clip = {
        str(row["sample_id"]): row
        for row in load_jsonl(Path(args.avut_clip_manifest).resolve())
    }
    avut_pairs = {
        str(row["sample_id"]): row
        for row in load_jsonl(Path(args.avut_pair_manifest).resolve())
        if row["audio_condition"] == "original"
    }

    kept_qa = [row for row in old_qa if row["evaluation_category"] != "audio_necessary"]
    kept_ids = {str(row["id"]) for row in kept_qa}
    qa_rows = [*avut_qa, *kept_qa]
    clip_rows = [row for row in old_clip if str(row["sample_id"]) in kept_ids]
    rgb_rows = [row for row in old_rgb if str(row["sample_id"]) in kept_ids]
    dev_rows = [
        {
            "youtube_id": str(row["id"]),
            "sample_id": str(row["id"]),
            "label": row["answer"],
            "evaluation_category": row["evaluation_category"],
        }
        for row in qa_rows
    ]
    for row in avut_qa:
        sample_id = str(row["id"])
        if sample_id not in avut_clip or sample_id not in avut_pairs:
            raise ValueError(f"AVUT 特征清单缺少 {sample_id}")
        clip_rows.append(
            {
                **avut_clip[sample_id],
                "youtube_id": sample_id,
                "sample_id": sample_id,
                "label": row["answer"],
                "evaluation_category": "audio_necessary",
                "audio_condition": "original",
            }
        )
        pair = avut_pairs[sample_id]
        rgb_rows.append(
            {
                "youtube_id": sample_id,
                "sample_id": sample_id,
                "split": "final_eval",
                "label": row["answer"],
                "audio_feature_path": pair["audio_feature_path"],
                "video_feature_path": pair["video_feature_path"],
                "audio_path": row["scene_audio_path"],
                "video_path": row["video_path"],
                "evaluation_category": "audio_necessary",
                "audio_condition": "original",
            }
        )

    categories = Counter(row["evaluation_category"] for row in qa_rows)
    ids = [str(row["id"]) for row in qa_rows]
    if len(qa_rows) != 60 or categories != Counter(
        {"audio_necessary": 20, "pure_visual": 20, "audio_interference": 20}
    ):
        raise ValueError(f"混合集规模或类别不符合冻结要求：{dict(categories)}")
    if len(set(ids)) != len(ids):
        raise ValueError("混合集 sample id 不唯一")

    output = Path(args.output_root).resolve()
    paths = {
        "qa_manifest": output / "mixed_final_task_eval.json",
        "dev_manifest": output / "dev_manifest.jsonl",
        "clip_manifest": output / "clip_manifest.jsonl",
        "rgb_manifest": output / "rgb_manifest.jsonl",
        "summary": output / "selection_summary.json",
    }
    if bool(args.avut_m4_manifest) != bool(args.ave_m4_manifest):
        raise ValueError("AVUT 与 AVE 的 M4 特征清单必须同时提供")
    if args.avut_m4_manifest:
        old_m4 = load_json(Path(args.ave_m4_manifest).resolve())
        avut_m4 = load_json(Path(args.avut_m4_manifest).resolve())
        mixed_m4 = [
            *[row for row in avut_m4 if str(row["id"]) in set(ids)],
            *[row for row in old_m4 if str(row["id"]) in kept_ids],
        ]
        if len(mixed_m4) != len(qa_rows) or {str(row["id"]) for row in mixed_m4} != set(ids):
            raise ValueError("混合 M4 特征清单与 QA 清单不一致")
        paths["m4_feature_manifest"] = output / "m4_projected_feature_manifest.json"
        write_json(paths["m4_feature_manifest"], mixed_m4)
    write_json(paths["qa_manifest"], qa_rows)
    write_jsonl(paths["dev_manifest"], dev_rows)
    write_jsonl(paths["clip_manifest"], clip_rows)
    write_jsonl(paths["rgb_manifest"], rgb_rows)
    summary = {
        "sample_count": len(qa_rows),
        "category_counts": dict(categories),
        "unique_sample_count": len(set(ids)),
        "margin_threshold": FROZEN_MARGIN,
        "max_neighbor_weight": MAX_NEIGHBOR_WEIGHT,
        "audio_necessary_source": "AVUT_official_audio_tasks",
        "visual_and_interference_source": "frozen_AVE_selection",
        "paths": {key: str(value) for key, value in paths.items()},
    }
    write_json(paths["summary"], summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--avut-qa-manifest", required=True)
    parser.add_argument("--avut-clip-manifest", required=True)
    parser.add_argument("--avut-pair-manifest", required=True)
    parser.add_argument("--avut-m4-manifest")
    parser.add_argument("--ave-m4-manifest")
    parser.add_argument("--ave-qa-manifest", default=str(OLD_ROOT / "ave_final_task_eval.json"))
    parser.add_argument("--ave-clip-manifest", default=str(OLD_ROOT / "clip_manifest.jsonl"))
    parser.add_argument("--ave-rgb-manifest", default=str(OLD_ROOT / "rgb_manifest.jsonl"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--margin-threshold", type=float, default=FROZEN_MARGIN)
    parser.add_argument("--max-neighbor-weight", type=float, default=MAX_NEIGHBOR_WEIGHT)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False))
