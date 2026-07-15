#!/usr/bin/env python
"""从 M4-IT 文本数据和已生成 wav 文件构建音频训练 JSON。

规则：对每条样本的人类轮次按 0,1,2... 编号，查找
`{样本id}_{人类轮次}.wav`。存在 wav 时，把该轮用户文本替换成
`<speech>`；如果原文本含 `<image>`，保留 `<image>`，生成 `<image>\n<speech>\n`。
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} 不是 list 格式")
    return data


def speech_value(original: str) -> str:
    prefix = ""
    if "<image>" in original:
        prefix = "<image>\n"
    return prefix + "<speech>\n"


def build_audio_samples(samples: list[dict[str, Any]], speech_root: Path, require_all_human: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    total_human_turns = 0
    total_speech_files = 0
    samples_with_speech = 0
    samples_missing_some = 0
    missing_head: list[str] = []

    for item in samples:
        new_item = copy.deepcopy(item)
        item_id = str(new_item.get("id"))
        conversations = new_item.get("conversations", [])
        speech_files: list[str] = []
        missing_for_item = False
        human_idx = 0

        for turn in conversations:
            if not isinstance(turn, dict) or turn.get("from") != "human":
                continue
            total_human_turns += 1
            rel = f"{item_id}_{human_idx}.wav"
            if (speech_root / rel).exists():
                turn["value"] = speech_value(str(turn.get("value", "")))
                speech_files.append(rel)
                total_speech_files += 1
            else:
                missing_for_item = True
                if len(missing_head) < 50:
                    missing_head.append(rel)
                if require_all_human:
                    turn["value"] = speech_value(str(turn.get("value", "")))
            human_idx += 1

        if speech_files:
            new_item["speech"] = speech_files
            samples_with_speech += 1
        if missing_for_item:
            samples_missing_some += 1
        output.append(new_item)

    stats = {
        "sample_count": len(samples),
        "samples_with_speech": samples_with_speech,
        "samples_missing_some_speech": samples_missing_some,
        "total_human_turns": total_human_turns,
        "total_speech_files_attached": total_speech_files,
        "missing_speech_head": missing_head,
        "status": "pass" if samples_with_speech == len(samples) and not missing_head else "warn",
    }
    return output, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 M4-IT 音频训练 JSON。")
    parser.add_argument("--input_json", default="intersuit/inputs/texts/m4-it-qwen.json")
    parser.add_argument("--speech_root", default="intersuit/inputs/images/llava-next")
    parser.add_argument("--output_json", default="intersuit/inputs/texts/m4-it-qwen-audio.generated.json")
    parser.add_argument("--report_json", default="intersuit/train_logs/m4_audio_json_build_report.json")
    parser.add_argument("--require_all_human", action="store_true", help="即使缺少 wav，也把人类轮次替换成 <speech>；通常不建议。")
    args = parser.parse_args()

    samples = load_json(Path(args.input_json))
    output, stats = build_audio_samples(samples, Path(args.speech_root), args.require_all_human)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False) + "\n", encoding="utf-8")

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
