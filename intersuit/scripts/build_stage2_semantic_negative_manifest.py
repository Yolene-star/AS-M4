#!/usr/bin/env python
"""为 Fixed BEATs 训练清单确定性配对语义错配音频。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def stable_index(sample_id: str, size: int) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % size


def build_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("scene_audio_path"):
            pools[(str(row.get("source_dataset")), str(row.get("task_type")))].append(row)

    output = []
    for row in rows:
        candidates = [
            candidate
            for candidate in pools[(str(row.get("source_dataset")), str(row.get("task_type")))]
            if candidate.get("physical_media_id") != row.get("physical_media_id")
            and candidate.get("scene_audio_path") != row.get("scene_audio_path")
            and (
                not row.get("event_label")
                or not candidate.get("event_label")
                or candidate.get("event_label") != row.get("event_label")
            )
        ]
        if not candidates:
            raise ValueError(f"样本 {row.get('id')} 没有满足约束的语义负音频")
        candidates.sort(key=lambda item: str(item.get("id")))
        negative = candidates[stable_index(str(row.get("id")), len(candidates))]
        paired = dict(row)
        paired.update(
            {
                "scene_audio_negative_path": negative["scene_audio_path"],
                "scene_audio_negative_sample_rate": negative.get("scene_audio_sample_rate", 16000),
                "scene_audio_negative_source_id": negative.get("id"),
                "scene_audio_negative_physical_media_id": negative.get("physical_media_id"),
                "scene_audio_negative_event_label": negative.get("event_label"),
            }
        )
        output.append(paired)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    paired = build_pairs(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(paired, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(json.dumps({"status": "PASS", "count": len(paired), "sha256": digest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
