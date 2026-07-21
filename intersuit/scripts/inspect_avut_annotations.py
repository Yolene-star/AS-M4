#!/usr/bin/env python
"""检查 AVUT 人工标注结构，不修改源数据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avut_common import find_human_annotation, inspect_schema


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 AVUT 人工标注字段结构。")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("harness/artifacts/as_m4_avut_smoke/annotation_structure.json"))
    args = parser.parse_args()
    annotation = find_human_annotation(args.input)
    records, structure = inspect_schema(annotation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(structure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"人工标注文件：{annotation}")
    print(f"JSON 顶层类型：{structure['top_level_type']}")
    print(f"样本总数：{len(records)}")
    print("前 3 条完整样本：")
    print(json.dumps(records[:3], ensure_ascii=False, indent=2))
    print(f"全部字段：{', '.join(structure['all_fields'])}")
    print(f"字段识别结果：{json.dumps({k: structure[k] for k in ('video_id_field', 'video_path_field', 'question_field', 'answer_field', 'choices_field', 'option_fields', 'question_category_field', 'split_field')}, ensure_ascii=False)}")
    print(f"答案映射：{json.dumps(structure['answer_mapping_method'], ensure_ascii=False)}")
    print(f"结构报告已写入：{args.output}")


if __name__ == "__main__":
    main()
