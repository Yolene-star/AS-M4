#!/usr/bin/env python
"""构建固定的 AVE 三类候选评测集及 offset 诊断输入清单。

评测集只从未进入冻结 offset scorer 训练/旧验证的 AVE_HF_EXPANDED 视频中
抽取。三类分别使用原音频、静音诊断和错配音频；最终答案来自 AVE 官方
事件标签。这里的 ``audio_necessary`` 只是声音类别代理，不能作为正式的
音频必要性验收集；正式混合集只复用本脚本固定的纯视觉与干扰音频子集。
脚本不提取特征、不运行模型，也不调整冻结阈值或软权重上限。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"
DEFAULT_VALID = INTERSUIT_ROOT / "datasets/AVE_HF_EXPANDED/ave_hf_pilot_valid.jsonl"
DEFAULT_ANNOTATIONS = INTERSUIT_ROOT / "datasets/AVE/data/Annotations.txt"
DEFAULT_CLIP = INTERSUIT_ROOT / (
    "harness/artifacts/ave_hf_expanded_clip_window_features/"
    "ave_hf_clip_window_feature_manifest.jsonl"
)
DEFAULT_RGB = INTERSUIT_ROOT / (
    "harness/artifacts/ave_hf_expanded_window_features/"
    "ave_hf_window_feature_manifest.jsonl"
)
DEFAULT_SCORER_ROOT = INTERSUIT_ROOT / (
    "harness/artifacts/ave_hf_temporal_offset_zero125_centerpeak_expanded_frozen/"
    "seed_20260719"
)
DEFAULT_OUTPUT = INTERSUIT_ROOT / (
    "harness/artifacts/video_window_weighting_ave_final60/eval_definition"
)
FROZEN_SEED = 20260719
FROZEN_MARGIN = 0.15
MAX_NEIGHBOR_WEIGHT = 0.35

AUDIO_NECESSARY_LABELS = (
    "Baby cry, infant cry",
    "Bark",
    "Church bell",
    "Female speech, woman speaking",
    "Male speech, man speaking",
    "Shofar",
    "Toilet flush",
    "Train horn",
)
VISUAL_LABELS = (
    "Bus",
    "Fixed-wing aircraft, airplane",
    "Helicopter",
    "Horse",
    "Motorcycle",
    "Race car, auto racing",
    "Rodents, rats, mice",
    "Truck",
)
CATEGORIES = ("audio_necessary", "pure_visual", "audio_interference")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def annotation_map(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        parts = line.split("&")
        if len(parts) != 5:
            raise ValueError(f"AVE annotation 非法：{line}")
        result[parts[1]] = parts[0]
    return result


def stable_key(value: str) -> str:
    return hashlib.sha256(f"{FROZEN_SEED}:{value}".encode()).hexdigest()


def used_scorer_ids(root: Path) -> set[str]:
    result = set()
    for name in ("temporal_offset_train_manifest.jsonl", "temporal_offset_val_manifest.jsonl"):
        for row in load_jsonl(root / name):
            result.add(str(row["youtube_id"]))
    return result


def balanced_select(
    rows: list[dict[str, Any]],
    allowed_labels: tuple[str, ...],
    count: int,
    excluded: set[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["event_label"] in allowed_labels and row["youtube_id"] not in excluded:
            grouped[row["event_label"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: stable_key(str(row["youtube_id"])))
    selected = []
    cursor = Counter()
    labels = [label for label in allowed_labels if grouped[label]]
    while len(selected) < count:
        progressed = False
        for label in labels:
            index = cursor[label]
            if index < len(grouped[label]):
                selected.append(grouped[label][index])
                cursor[label] += 1
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise ValueError(f"标签池不足，无法选择 {count} 条：{allowed_labels}")
    return selected


def choices_for(answer: str, category: str) -> dict[str, str]:
    label_pool = (
        list(AUDIO_NECESSARY_LABELS)
        if category == "audio_necessary"
        else list(VISUAL_LABELS)
    )
    distractors = sorted(
        (label for label in label_pool if label != answer),
        key=lambda label: stable_key(f"{answer}:{label}"),
    )[:3]
    values = [answer, *distractors]
    values.sort(key=lambda label: stable_key(f"choice:{answer}:{label}"))
    return dict(zip(("A", "B", "C", "D"), values))


def qa_row(row: dict[str, Any], category: str, donor_id: str | None) -> dict[str, Any]:
    sample_id = str(row["youtube_id"])
    answer = str(row["event_label"])
    question = (
        "Which sound event occurs in this video?"
        if category == "audio_necessary"
        else "Which visible event or object is primarily shown in this video?"
    )
    return {
        "id": sample_id,
        "video_path": row["video_path"],
        "question": question,
        "answer": answer,
        "choices": choices_for(answer, category),
        "generation_mode": "generate",
        "context": "Watch the complete video and answer the multiple-choice question.",
        "video_max_frames": 32,
        "source_dataset": "AVE",
        "youtube_id": sample_id,
        "event_label": answer,
        "evaluation_category": category,
        "diagnostic_audio_condition": {
            "audio_necessary": "original",
            "pure_visual": "silence",
            "audio_interference": "mismatched",
        }[category],
        "audio_donor_youtube_id": donor_id,
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": answer},
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.scorer_seed) != FROZEN_SEED:
        raise ValueError(f"scorer seed 已冻结为 {FROZEN_SEED}")
    if abs(float(args.margin_threshold) - FROZEN_MARGIN) > 1e-12:
        raise ValueError(f"margin 已冻结为 {FROZEN_MARGIN}")
    if abs(float(args.max_neighbor_weight) - MAX_NEIGHBOR_WEIGHT) > 1e-12:
        raise ValueError(f"邻窗权重上限已冻结为 {MAX_NEIGHBOR_WEIGHT}")
    per_category = int(args.per_category)
    if per_category < 1:
        raise ValueError("per_category 必须为正数")

    valid = load_jsonl(Path(args.valid_manifest).resolve())
    labels = annotation_map(Path(args.annotations).resolve())
    clip_rows = {
        str(row["youtube_id"]): row
        for row in load_jsonl(Path(args.clip_manifest).resolve())
    }
    rgb_rows = {
        str(row["youtube_id"]): row
        for row in load_jsonl(Path(args.rgb_manifest).resolve())
    }
    scorer_used = used_scorer_ids(Path(args.scorer_manifest_root).resolve())
    candidates = []
    for row in valid:
        youtube_id = str(row["youtube_id"])
        if youtube_id in scorer_used:
            continue
        if youtube_id not in labels or youtube_id not in clip_rows or youtube_id not in rgb_rows:
            continue
        candidates.append(
            {
                **row,
                "youtube_id": youtube_id,
                "event_label": labels[youtube_id],
            }
        )

    selected_audio = balanced_select(
        candidates,
        AUDIO_NECESSARY_LABELS,
        per_category,
        scorer_used,
    )
    selected_ids = {row["youtube_id"] for row in selected_audio}
    selected_visual = balanced_select(
        candidates,
        VISUAL_LABELS,
        per_category,
        scorer_used | selected_ids,
    )
    selected_ids.update(row["youtube_id"] for row in selected_visual)
    selected_interference = balanced_select(
        candidates,
        VISUAL_LABELS,
        per_category,
        scorer_used | selected_ids,
    )
    selected_ids.update(row["youtube_id"] for row in selected_interference)
    if len(selected_ids) != per_category * 3:
        raise ValueError("三类评测视频没有保持 youtube_id 唯一")

    donor_pool = sorted(selected_audio, key=lambda row: stable_key(f"donor:{row['youtube_id']}"))
    category_rows = {
        "audio_necessary": selected_audio,
        "pure_visual": selected_visual,
        "audio_interference": selected_interference,
    }
    qa_rows, dev_rows, diagnostic_clip, diagnostic_rgb = [], [], [], []
    for category in CATEGORIES:
        for index, target in enumerate(category_rows[category]):
            target_id = str(target["youtube_id"])
            donor = donor_pool[index % len(donor_pool)] if category == "audio_interference" else target
            donor_id = str(donor["youtube_id"]) if category == "audio_interference" else None
            condition = {
                "audio_necessary": "original",
                "pure_visual": "silence",
                "audio_interference": "mismatched",
            }[category]
            qa_rows.append(qa_row(target, category, donor_id))
            dev_rows.append(
                {
                    "youtube_id": target_id,
                    "sample_id": target_id,
                    "label": target["event_label"],
                    "evaluation_category": category,
                }
            )
            target_clip = clip_rows[target_id]
            target_rgb = rgb_rows[target_id]
            donor_clip = clip_rows[str(donor["youtube_id"])]
            donor_rgb = rgb_rows[str(donor["youtube_id"])]
            diagnostic_clip.append(
                {
                    **target_clip,
                    "youtube_id": target_id,
                    "sample_id": target_id,
                    "audio_feature_path": donor_clip["audio_feature_path"],
                    "evaluation_category": category,
                    "audio_condition": condition,
                    "audio_donor_youtube_id": donor_id,
                }
            )
            diagnostic_rgb.append(
                {
                    **target_rgb,
                    "youtube_id": target_id,
                    "sample_id": target_id,
                    "audio_path": donor_rgb["audio_path"],
                    "evaluation_category": category,
                    "audio_condition": condition,
                    "audio_donor_youtube_id": donor_id,
                }
            )

    output_root = Path(args.output_root).resolve()
    paths = {
        "qa_manifest": str(output_root / "ave_final_task_eval.json"),
        "dev_manifest": str(output_root / "dev_manifest.jsonl"),
        "clip_manifest": str(output_root / "clip_manifest.jsonl"),
        "rgb_manifest": str(output_root / "rgb_manifest.jsonl"),
        "summary": str(output_root / "selection_summary.json"),
    }
    write_json(Path(paths["qa_manifest"]), qa_rows)
    write_jsonl(Path(paths["dev_manifest"]), dev_rows)
    write_jsonl(Path(paths["clip_manifest"]), diagnostic_clip)
    write_jsonl(Path(paths["rgb_manifest"]), diagnostic_rgb)
    summary = {
        "sample_count": len(qa_rows),
        "per_category": per_category,
        "category_counts": dict(Counter(row["evaluation_category"] for row in qa_rows)),
        "category_label_counts": {
            category: dict(
                Counter(row["event_label"] for row in qa_rows if row["evaluation_category"] == category)
            )
            for category in CATEGORIES
        },
        "unique_youtube_id_count": len(selected_ids),
        "expanded_candidate_count_after_scorer_exclusion": len(candidates),
        "excluded_frozen_scorer_youtube_id_count": len(scorer_used),
        "overlap_with_frozen_scorer_inputs": len(selected_ids & scorer_used),
        "scorer_seed": FROZEN_SEED,
        "margin_threshold": FROZEN_MARGIN,
        "max_neighbor_weight": MAX_NEIGHBOR_WEIGHT,
        "selection_is_deterministic": True,
        "audio_necessary_is_proxy_only": True,
        "formal_mixed_eval_reuses_categories": [
            "pure_visual",
            "audio_interference",
        ],
        "paths": paths,
    }
    write_json(Path(paths["summary"]), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-manifest", default=str(DEFAULT_VALID))
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--clip-manifest", default=str(DEFAULT_CLIP))
    parser.add_argument("--rgb-manifest", default=str(DEFAULT_RGB))
    parser.add_argument("--scorer-manifest-root", default=str(DEFAULT_SCORER_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--per-category", type=int, default=20)
    parser.add_argument("--scorer-seed", type=int, default=FROZEN_SEED)
    parser.add_argument("--margin-threshold", type=float, default=FROZEN_MARGIN)
    parser.add_argument("--max-neighbor-weight", type=float, default=MAX_NEIGHBOR_WEIGHT)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
