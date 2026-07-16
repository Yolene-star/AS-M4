#!/usr/bin/env python
"""AVUT 人工标注解析的共享工具。"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


HUMAN_FILENAMES = ("AV_Human_data.json", "AV_Human_filtered_data.json")
QUESTION_FIELDS = ("question", "query", "prompt")
ANSWER_FIELDS = ("answer", "correct_answer", "label")
VIDEO_ID_FIELDS = ("video_id", "source_video_id", "vid")
VIDEO_PATH_FIELDS = ("video_path", "video", "path", "file")
SPLIT_FIELDS = ("split", "subset", "partition")
CATEGORY_FIELDS = ("task_type", "question_type", "category", "type")


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"非法 JSON：{path}: {exc}") from exc


def find_human_annotation(input_path: Path) -> Path:
    if input_path.is_file():
        if "Human" not in input_path.name or "Gemini" in input_path.name:
            raise ValueError(f"输入不是 AVUT 人工标注文件：{input_path}")
        return input_path.resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"输入路径不存在：{input_path}")
    matches = [p for p in input_path.rglob("*Human*.json") if "Gemini" not in p.name]
    if not matches:
        raise FileNotFoundError(f"未在 {input_path} 中找到 *Human*.json")
    ranked = sorted(matches, key=lambda p: (HUMAN_FILENAMES.index(p.name) if p.name in HUMAN_FILENAMES else 99, str(p)))
    return ranked[0].resolve()


def records_from_json(data: Any) -> tuple[list[dict[str, Any]], str]:
    if isinstance(data, list):
        records = data
        top_level = "list"
    elif isinstance(data, dict):
        list_items = [(key, value) for key, value in data.items() if isinstance(value, list)]
        if len(list_items) != 1:
            raise ValueError("JSON 顶层为 dict，但无法唯一识别其中的样本列表")
        records = list_items[0][1]
        top_level = "dict"
    else:
        raise TypeError(f"JSON 顶层必须为 list 或 dict，实际为 {type(data).__name__}")
    if not all(isinstance(item, dict) for item in records):
        raise TypeError("样本列表中存在非对象条目")
    return records, top_level


def detect_field(records: list[dict[str, Any]], candidates: Iterable[str], *, required: bool = False) -> str | None:
    counts = Counter(key for record in records for key in record)
    for key in candidates:
        if counts[key] == len(records):
            return key
    if required:
        raise ValueError(f"无法识别必需字段，候选字段为：{list(candidates)}")
    return None


def detect_choices(records: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
    for key in ("choices", "options", "candidates"):
        if records and all(key in record for record in records):
            return key, []
    option_fields = sorted(
        {key for record in records for key in record if re.fullmatch(r"(?:option|choice)_[A-Za-z0-9]+", key)},
        key=lambda key: key.rsplit("_", 1)[-1],
    )
    if option_fields and all(all(key in record for key in option_fields) for record in records):
        return None, option_fields
    raise ValueError("无法识别 choices：既没有 choices/options 字段，也没有完整 option_* 字段")


def extract_choices(record: dict[str, Any], choices_field: str | None, option_fields: list[str]) -> dict[str, str]:
    if choices_field:
        raw = record.get(choices_field)
        if isinstance(raw, list):
            return {chr(ord("A") + index): str(value) for index, value in enumerate(raw)}
        if isinstance(raw, dict):
            return {str(key): str(value) for key, value in raw.items()}
        raise ValueError(f"choices 字段必须为 list 或 dict，实际为 {type(raw).__name__}")
    choices: dict[str, str] = {}
    for field in option_fields:
        label = field.rsplit("_", 1)[-1].upper()
        choices[label] = str(record.get(field, ""))
    if not choices:
        raise ValueError("选项为空")
    return choices


def resolve_answer(raw_answer: Any, choices: dict[str, str]) -> tuple[str, str]:
    if raw_answer is None or str(raw_answer).strip() == "":
        raise ValueError("答案为空")
    answer = str(raw_answer).strip()
    upper = answer.upper()
    normalized = {str(key).upper(): value for key, value in choices.items()}
    if upper in normalized:
        resolved = normalized[upper].strip()
        if not resolved:
            raise ValueError(f"答案 {answer!r} 对应的选项文本为空")
        return resolved, "letter_to_option_text"
    if re.fullmatch(r"\d+", answer):
        index = int(answer)
        labels = list(choices)
        if 0 <= index < len(labels):
            return choices[labels[index]], "zero_based_index_to_option_text"
        if 1 <= index <= len(labels):
            return choices[labels[index - 1]], "one_based_index_to_option_text"
        raise ValueError(f"答案索引越界：{answer}")
    for value in choices.values():
        if answer.casefold() == value.strip().casefold():
            return value, "answer_is_option_text"
    raise ValueError(f"答案 {answer!r} 无法映射到 choices")


def inspect_schema(annotation_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = load_json(annotation_path)
    records, top_level = records_from_json(data)
    if not records:
        raise ValueError("人工标注没有样本")
    question_field = detect_field(records, QUESTION_FIELDS, required=True)
    answer_field = detect_field(records, ANSWER_FIELDS, required=True)
    video_id_field = detect_field(records, VIDEO_ID_FIELDS, required=True)
    video_path_field = detect_field(records, VIDEO_PATH_FIELDS, required=True)
    choices_field, option_fields = detect_choices(records)
    mapping_methods = Counter()
    conversion_errors = []
    for index, record in enumerate(records):
        try:
            choices = extract_choices(record, choices_field, option_fields)
            _, method = resolve_answer(record[answer_field], choices)
            mapping_methods[method] += 1
        except (TypeError, ValueError) as exc:
            conversion_errors.append({"index": index, "sample_id": record.get("QA_id", record.get(video_id_field)), "reason": str(exc)})
    all_fields = sorted({key for record in records for key in record})
    structure = {
        "annotation_file": str(annotation_path.resolve()),
        "sample_count": len(records),
        "top_level_type": top_level,
        "all_fields": all_fields,
        "field_counts": dict(Counter(key for record in records for key in record)),
        "first_three_samples": records[:3],
        "video_id_field": video_id_field,
        "video_path_field": video_path_field,
        "question_field": question_field,
        "answer_field": answer_field,
        "choices_field": choices_field,
        "option_fields": option_fields,
        "choices_representation": "multiple_fields" if option_fields else "list_or_dict_field",
        "answer_mapping_method": dict(mapping_methods),
        "conversion_errors": conversion_errors,
        "question_category_field": detect_field(records, CATEGORY_FIELDS),
        "split_field": detect_field(records, SPLIT_FIELDS),
    }
    return records, structure


def normalize_yes_no(value: str) -> str:
    return value.strip().casefold()


def load_review_csv(path: Path, required_fields: Iterable[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"人工审核 CSV 不存在：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("人工审核 CSV 没有数据行")
    missing_columns = [field for field in required_fields if field not in (rows[0].keys() if rows else [])]
    if missing_columns:
        raise ValueError(f"人工审核 CSV 缺少列：{missing_columns}")
    incomplete = [row.get("sample_id", "<unknown>") for row in rows if any(not row.get(field, "").strip() for field in required_fields)]
    if incomplete:
        raise ValueError(f"人工审核尚未填写完整：{incomplete}")
    return rows


def build_accept_contains(answer: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", answer).strip().strip(".?!")
    if not cleaned:
        raise ValueError("不能从空答案生成 accept_contains")
    return [cleaned]


def is_audio_related_question(question: str) -> bool:
    """保守判断问题是否显式涉及声音、言语或音频时间。"""

    pattern = re.compile(
        r"\b(audio|sound|hear|heard|say|says|said|speaks?|voice|music|drum|alarm|noise|word|lyric|ringtone|beep|explosion)\b",
        flags=re.IGNORECASE,
    )
    return bool(pattern.search(str(question)))
