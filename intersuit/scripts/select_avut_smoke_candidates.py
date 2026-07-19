#!/usr/bin/env python
"""从 AVUT 人工标注中可复现地选择声音依赖候选，不运行模型。"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any

from avut_common import extract_choices, find_human_annotation, inspect_schema, resolve_answer


SEED = 20260716
SOUND_RE = re.compile(r"\b(sound|hear|heard|audio|singing|speaking|barking|noise|music|voice|instrument|alarm|explosion|clapping|crying|laughing|lyric|says?|said|word|ringtone|beep)\b", re.I)
VISUAL_RE = re.compile(r"\b(colou?r|how many (?:people|persons)|wearing|clothes|appearance|located|position|left side|right side)\b", re.I)
PREFERRED_TASKS = {"Audio Content Counting", "Audio Event Location", "Audio Information Extraction"}
CSV_FIELDS = ("sample_id", "source_video_id", "video_path", "question", "answer", "choices", "selection_reason", "video_exists", "has_audio", "muted_answerable", "audio_answerable", "keep", "review_level", "manual_note")


def choose(records: list[dict[str, Any]], schema: dict[str, Any], count: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    ranked: list[tuple[int, float, dict[str, Any], list[str]]] = []
    skipped: list[dict[str, Any]] = []
    qf, af = schema["question_field"], schema["answer_field"]
    vf, vif = schema["video_path_field"], schema["video_id_field"]
    cf, ofs = schema["choices_field"], schema["option_fields"]
    tf = schema.get("question_category_field")
    for record in records:
        question = str(record.get(qf, "")).strip()
        task = str(record.get(tf, "")) if tf else ""
        reasons: list[str] = []
        score = 0
        if task in PREFERRED_TASKS:
            score += 3
            reasons.append(f"官方任务类型={task}")
        if SOUND_RE.search(question):
            score += 3
            reasons.append("问题包含声音语义词")
        if VISUAL_RE.search(question):
            score -= 4
            reasons.append("包含明显视觉属性词")
        if score < 3:
            skipped.append({"sample_id": record.get("QA_id", record.get(vif)), "reason": "; ".join(reasons) or "未命中声音候选规则"})
            continue
        try:
            choices = extract_choices(record, cf, ofs)
            answer, method = resolve_answer(record.get(af), choices)
        except (TypeError, ValueError) as exc:
            skipped.append({"sample_id": record.get("QA_id", record.get(vif)), "reason": f"字段转换失败：{exc}"})
            continue
        item = {
            "sample_id": f"avut_{int(record.get('QA_id', len(ranked) + 1)):04d}",
            "source_qa_id": record.get("QA_id"),
            "source_video_id": record.get(vif),
            "source_video_path": str(record.get(vf)),
            "video_path": str(Path("datasets/AVUT/smoke_videos") / Path(str(record.get(vf))).name),
            "question": question,
            "answer_label": str(record.get(af)),
            "answer": answer,
            "answer_mapping_method": method,
            "choices": choices,
            "task_type": task,
            "selection_reason": "; ".join(reasons),
        }
        ranked.append((score, rng.random(), item, reasons))
    ranked.sort(key=lambda value: (-value[0], value[1], value[2]["sample_id"]))
    selected: list[dict[str, Any]] = []
    used_videos: set[str] = set()
    for _, _, item, _ in ranked:
        if item["source_video_path"] in used_videos:
            continue
        selected.append(item)
        used_videos.add(item["source_video_path"])
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(f"仅找到 {len(selected)} 条不同视频的候选，少于请求的 {count} 条")
    return selected, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="选择 3–5 条 AVUT smoke 候选。")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=Path("inputs/texts/avut/avut_smoke_candidates.json"))
    parser.add_argument("--review-csv", type=Path, default=Path("inputs/texts/avut/avut_smoke_review.csv"))
    args = parser.parse_args()
    if not 3 <= args.count <= 5:
        raise ValueError("第一轮候选数量必须在 3–5 条之间")
    annotation = find_human_annotation(args.input)
    records, schema = inspect_schema(annotation)
    selected, skipped = choose(records, schema, args.count, args.seed)
    payload = {"annotation_file": str(annotation), "seed": args.seed, "count": len(selected), "selected": selected, "skipped_count": len(skipped), "skipped": skipped}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.review_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.review_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for item in selected:
            writer.writerow({
                "sample_id": item["sample_id"], "source_video_id": item["source_video_id"], "video_path": item["video_path"],
                "question": item["question"], "answer": item["answer"], "choices": json.dumps(item["choices"], ensure_ascii=False),
                "selection_reason": item["selection_reason"], "video_exists": "", "has_audio": "",
                "muted_answerable": "", "audio_answerable": "", "keep": "", "review_level": "", "manual_note": "",
            })
    print(f"固定随机种子：{args.seed}")
    print(json.dumps(selected, ensure_ascii=False, indent=2))
    print(f"候选文件：{args.output}")
    print(f"人工审核模板：{args.review_csv}")


if __name__ == "__main__":
    main()
