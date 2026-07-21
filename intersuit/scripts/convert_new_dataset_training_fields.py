#!/usr/bin/env python
"""把已验收 AVUT/MUSIC 媒体与冻结 QA 规则转换为训练器记录。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from avut_common import extract_choices, inspect_schema
from new_dataset_common import (
    extract_wav, load_json, replace_music_template, sha256_file,
    task_type_from_avut, task_type_from_music, write_json,
)


def _prompt(question: str, choices: dict[str, str] | None = None) -> str:
    if choices:
        options = "\n".join(f"{key}. {value}" for key, value in choices.items())
        question = f"{question}\n{options}"
    return f"<image>\n{question}"


def convert_avut(entry: dict, annotations: Path, wav_root: Path) -> list[dict]:
    records, schema = inspect_schema(annotations)
    result = []
    video_id = str(entry["video_id"])
    for row in records:
        if str(row.get(schema["video_id_field"])) != video_id:
            continue
        choices = extract_choices(row, schema["choices_field"], schema["option_fields"])
        answer = str(row[schema["answer_field"]]).strip().upper()
        if answer not in choices or not choices[answer].strip():
            continue
        qa_id = str(row.get("QA_id", len(result)))
        media = Path(entry["path"]).resolve()
        wav = (wav_root / "AVUT" / f"{entry['youtube_id']}.wav").resolve()
        audio = extract_wav(media, wav)
        question = str(row[schema["question_field"]]).strip()
        modality = task_type_from_avut(row.get(schema.get("question_category_field") or "task_type"))
        result.append({
            "id": f"avut_{qa_id}", "sample_id": f"avut_{qa_id}",
            "source_dataset": "AVUT", "source_revision": entry["source_revision"],
            "video_id": video_id, "youtube_id": entry["youtube_id"],
            "video_path": str(media), "scene_audio_path": str(wav),
            "question": question, "answer": answer,
            "task_type": f"{modality}:{row.get(schema.get('question_category_field') or 'task_type','')}",
            "media_sha256": entry["media_sha256"], "audio_sha256": audio["audio_sha256"],
            "scene_audio_sample_rate": 16000,
            "conversations": [{"from": "human", "value": _prompt(question, choices)}, {"from": "gpt", "value": answer}],
        })
    return result


def convert_music(entry: dict, annotations: Path, wav_root: Path) -> list[dict]:
    payload = load_json(annotations)
    rows = payload if isinstance(payload, list) else payload.get("data", payload.get("train", []))
    result = []
    for index, row in enumerate(rows):
        if str(row.get("video_id")) != str(entry["youtube_id"]) or int(row.get("question_deleted", 0) or 0) != 0:
            continue
        raw_values = row.get("templ_values") or []
        if isinstance(raw_values, str):
            raw_values = json.loads(raw_values)
        question = replace_music_template(row.get("question_content", ""), list(raw_values))
        answer = str(row.get("anser", "")).strip()
        if not question or not answer:
            raise ValueError("MUSIC question/answer is empty")
        media = Path(entry["path"]).resolve()
        derived = [("", media)]
        if entry.get("has_flip"):
            flipped = media.with_name(media.stem + "_flip" + media.suffix)
            if not flipped.is_file():
                raise ValueError(f"missing required flip media: {flipped}")
            derived.append(("_flip", flipped))
        for suffix, derived_media in derived:
            media_hash = sha256_file(derived_media)
            wav = (wav_root / "MUSIC_AVQA_V2" / f"{derived_media.stem}.wav").resolve()
            audio = extract_wav(derived_media, wav)
            raw_type = row.get("type")
            if isinstance(raw_type, str):
                raw_type = json.loads(raw_type)
            modality = task_type_from_music(raw_type)
            subtype = raw_type[1] if isinstance(raw_type, list) and len(raw_type) > 1 else ""
            sample_id = f"music_{entry['video_id']}{suffix}_{index:06d}"
            result.append({
                "id": sample_id, "sample_id": sample_id,
                "source_dataset": "MUSIC-AVQA-v2.0", "source_revision": entry["source_revision"],
                "video_id": f"{entry['video_id']}{suffix}", "youtube_id": entry["youtube_id"],
                "video_path": str(derived_media.resolve()), "scene_audio_path": str(wav),
                "question": question, "answer": answer, "task_type": f"{modality}:{subtype}",
                "media_sha256": media_hash, "audio_sha256": audio["audio_sha256"],
                "scene_audio_sample_rate": 16000,
                "conversations": [{"from": "human", "value": _prompt(question)}, {"from": "gpt", "value": answer}],
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--avut-annotations", type=Path, required=True)
    parser.add_argument("--music-annotations", type=Path)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validation = load_json(args.validation_report)
    entries = [e for e in validation.get("entries", []) if e.get("validation_status") == "accepted"]
    output = []
    for entry in entries:
        if entry["source_dataset"] == "AVUT":
            output.extend(convert_avut(entry, args.avut_annotations, args.audio_root))
        elif args.music_annotations:
            output.extend(convert_music(entry, args.music_annotations, args.audio_root))
    if not output:
        raise SystemExit("没有可输出的 QA 记录")
    write_json(args.output, output)
    digest = sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(f"{digest}  {args.output.name}\n", encoding="ascii")
    print(f"已写入 {len(output)} 条：{args.output}")


if __name__ == "__main__":
    main()
