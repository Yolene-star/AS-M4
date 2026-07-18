#!/usr/bin/env python
"""固定构建 20 条 AVUT 官方音频任务评测清单，不运行模型或调整参数。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from avut_common import find_human_annotation, inspect_schema
from select_avut_smoke_candidates import SEED, choose


INTERSUIT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = INTERSUIT_ROOT / "datasets/AVUT/raw/AV_Human_data.json"
DEFAULT_OUTPUT = INTERSUIT_ROOT / (
    "harness/artifacts/video_window_weighting_mixed_final60/eval_definition/"
    "avut_audio_necessary_eval.json"
)
VIDEO_ROOTS = (
    INTERSUIT_ROOT / "datasets/AVUT/smoke_videos",
    INTERSUIT_ROOT / "datasets/AVUT/final_eval_videos",
)


def youtube_id(item: dict[str, Any]) -> str:
    return Path(str(item["source_video_path"])).stem


def resolve_video(item: dict[str, Any]) -> Path | None:
    name = f"{youtube_id(item)}.mp4"
    return next((root / name for root in VIDEO_ROOTS if (root / name).is_file()), None)


def run(args: argparse.Namespace) -> dict[str, Any]:
    annotation = find_human_annotation(Path(args.input).resolve())
    records, schema = inspect_schema(annotation)
    ranked, _ = choose(records, schema, count=max(int(args.count) * 3, 60), seed=int(args.seed))
    selected: list[dict[str, Any]] = []
    unavailable: list[str] = []
    for item in ranked:
        video = resolve_video(item)
        if video is None:
            unavailable.append(youtube_id(item))
            continue
        row = {
            "id": item["sample_id"],
            "video_path": str(video.resolve()),
            "scene_audio_path": str(video.resolve()),
            "question": item["question"],
            "answer": item["answer"],
            "choices": item["choices"],
            "accept_contains": [item["answer"]],
            "generation_mode": "generate",
            "context": "Please watch the complete video and answer the multiple-choice question.",
            "new_query": item["question"],
            "new_query_pos": 20,
            "video_max_frames": 32,
            "source_dataset": "AVUT",
            "source_video_id": item["source_video_id"],
            "source_qa_id": item["source_qa_id"],
            "youtube_id": youtube_id(item),
            "task_type": item["task_type"],
            "selection_reason": item["selection_reason"],
            "evaluation_category": "audio_necessary",
            "diagnostic_audio_condition": "original",
            "scene_audio_sample_rate": 16000,
            "scene_audio_window_sec": 1.0,
            "scene_audio_hop_sec": 0.5,
            "audio_required_candidate": True,
            "conversations": [
                {"from": "human", "value": f"<image>\n{item['question']}"},
                {"from": "gpt", "value": item["answer"]},
            ],
        }
        selected.append(row)
        if len(selected) == int(args.count):
            break
    if len(selected) != int(args.count):
        raise ValueError(f"只有 {len(selected)} 条已下载且可用的 AVUT 候选")
    if len({row["youtube_id"] for row in selected}) != len(selected):
        raise ValueError("AVUT 评测清单未保持 youtube_id 唯一")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "sample_count": len(selected),
        "seed": int(args.seed),
        "youtube_ids": [row["youtube_id"] for row in selected],
        "sample_ids": [row["id"] for row in selected],
        "unavailable_ranked_candidates_skipped": unavailable,
        "official_audio_task_count": sum(
            row["task_type"] in {"Audio Content Counting", "Audio Event Location", "Audio Information Extraction"}
            for row in selected
        ),
        "output": str(output),
    }
    (output.parent / "avut_audio_necessary_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False))
