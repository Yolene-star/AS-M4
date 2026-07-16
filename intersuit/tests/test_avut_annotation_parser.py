import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from avut_common import extract_choices, inspect_schema, resolve_answer  # noqa: E402


def record(**updates):
    value = {
        "video_id": 1,
        "video_path": "data/example.mp4",
        "question": "What sound is heard?",
        "option_A": "dog",
        "option_B": "cat",
        "option_C": "bell",
        "option_D": "music",
        "answer": "A",
        "task_type": "Audio Information Extraction",
        "QA_id": 1,
    }
    value.update(updates)
    return value


def write_json(path: Path, value) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_normal_annotation_and_letter_mapping(tmp_path):
    records, schema = inspect_schema(write_json(tmp_path / "AV_Human_data.json", [record()]))
    choices = extract_choices(records[0], schema["choices_field"], schema["option_fields"])
    assert resolve_answer(records[0]["answer"], choices) == ("dog", "letter_to_option_text")


def test_missing_question_field(tmp_path):
    item = record()
    item.pop("question")
    with pytest.raises(ValueError, match="必需字段"):
        inspect_schema(write_json(tmp_path / "AV_Human_data.json", [item]))


def test_missing_answer_field(tmp_path):
    item = record()
    item.pop("answer")
    with pytest.raises(ValueError, match="必需字段"):
        inspect_schema(write_json(tmp_path / "AV_Human_data.json", [item]))


def test_choices_zero_based_index_mapping():
    assert resolve_answer(0, {"A": "dog", "B": "cat"}) == ("dog", "zero_based_index_to_option_text")


def test_illegal_json(tmp_path):
    path = tmp_path / "AV_Human_data.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="非法 JSON"):
        inspect_schema(path)
