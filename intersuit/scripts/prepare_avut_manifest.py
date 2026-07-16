#!/usr/bin/env python
"""在人工审核完成后生成 AS-M4 AVUT smoke manifest。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avut_common import build_accept_contains, extract_choices, inspect_schema, is_audio_related_question, load_review_csv, normalize_yes_no, resolve_answer


REVIEW_FIELDS = ("muted_answerable", "audio_answerable", "keep", "review_level", "manual_note")


def build_manifest(candidates_path: Path, review_path: Path, video_root: Path, annotations_path: Path | None = None) -> list[dict[str, object]]:
    payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    selected = payload.get("selected") if isinstance(payload, dict) else payload
    if not isinstance(selected, list):
        raise ValueError("候选 JSON 缺少 selected 列表")
    by_id = {str(item["sample_id"]): item for item in selected}
    source_records = None
    source_schema = None
    if annotations_path is not None:
        source_records, source_schema = inspect_schema(annotations_path)
        source_records = {str(item.get("QA_id")): item for item in source_records}
    rows = load_review_csv(review_path, REVIEW_FIELDS)
    output = []
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id not in by_id:
            raise ValueError(f"审核 CSV 中出现未知 sample_id：{sample_id}")
        if normalize_yes_no(row["keep"]) != "yes":
            continue
        if normalize_yes_no(row["muted_answerable"]) != "no":
            raise ValueError(f"{sample_id} 的 muted-only 保留条件要求 muted_answerable=no")
        if normalize_yes_no(row["audio_answerable"]) != "unknown":
            raise ValueError(f"{sample_id} 未做有声审核时 audio_answerable 必须为 unknown")
        if normalize_yes_no(row["review_level"]) != "muted_only":
            raise ValueError(f"{sample_id} 的 review_level 必须为 muted_only")
        item = by_id[sample_id]
        if source_records is not None and source_schema is not None:
            source = source_records.get(str(item.get("source_qa_id")))
            if source is None:
                raise ValueError(f"{sample_id} 在人工标注中找不到 source_qa_id={item.get('source_qa_id')}")
            source_choices = extract_choices(source, source_schema["choices_field"], source_schema["option_fields"])
            source_answer, _ = resolve_answer(source[source_schema["answer_field"]], source_choices)
            if str(source[source_schema["question_field"]]).strip() != str(item["question"]).strip() or source_answer != str(item["answer"]).strip() or source_choices != item["choices"]:
                raise ValueError(f"{sample_id} 与原始人工标注的问题、答案或 choices 不一致")
        video_path = video_root / Path(str(item["video_path"])).name
        if not video_path.is_file():
            raise FileNotFoundError(f"保留样本的视频不存在：{video_path}")
        question, answer = str(item["question"]).strip(), str(item["answer"]).strip()
        if not question or not answer:
            raise ValueError(f"{sample_id} 的问题或答案为空")
        if not is_audio_related_question(question):
            raise ValueError(f"{sample_id} 的问题没有显式声音语义，不能进入本轮候选 manifest")
        output.append({
            "id": sample_id, "video_path": str(video_path), "question": question, "answer": answer,
            "choices": item["choices"], "accept_contains": build_accept_contains(answer), "generation_mode": "generate",
            "video_max_frames": 32, "source_dataset": "AVUT", "source_video_id": item["source_video_id"],
            "source_qa_id": item.get("source_qa_id"), "scene_audio_path": str(video_path),
            "scene_audio_sample_rate": 16000, "scene_audio_window_sec": 1.0, "scene_audio_hop_sec": 0.5,
            "muted_answerable": False, "audio_answerable": None, "review_level": "muted_only",
            "audio_required_candidate": True, "manually_verified_audio_required": False, "manual_mute_checked": True,
            "conversations": [
                {"from": "human", "value": f"<image>\n{question}"},
                {"from": "gpt", "value": answer},
            ],
        })
    if not output:
        raise ValueError("人工审核后没有 keep=yes 的样本")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 AVUT AS-M4 smoke manifest。")
    parser.add_argument("--annotations", required=True, type=Path, help="保留接口一致性；字段映射已记录在 candidates 中。")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("inputs/texts/avut/avut_audio_smoke.json"))
    args = parser.parse_args()
    if not args.annotations.exists():
        raise FileNotFoundError(f"人工标注不存在：{args.annotations}")
    manifest = build_manifest(args.candidates, args.review, args.video_root, args.annotations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写入 {len(manifest)} 条样本：{args.output}")


if __name__ == "__main__":
    main()
