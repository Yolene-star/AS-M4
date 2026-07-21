#!/usr/bin/env python
"""构建阶段 2 的正式 train/dev/reserve manifest。"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from new_dataset_common import load_records, sha256_file, write_json


SEED = 20260720
TASK_TYPES = {"audio", "visual", "audio_visual"}
SOURCE_PRIORITY = {
    "AVUT": 0,
    "MUSIC-AVQA-v2.0": 1,
    "AVE_HF_EXPANDED": 2,
}
HISTORICAL_AVUT_IDS = {
    "jkgJiV-l0k0", "4obNozIkugU", "x_x1lv_J3Ko", "JrWfGxd0J_4",
    "48AB70FSz7g", "g-jRGhwJ7BM", "6lbk8ihhYw0", "hVxbhnC302w",
    "ocMYLIDBs3U", "1rPehAUrYsQ", "uNkn_x2wFsA", "yMR2YHOdPno",
    "D4oMZ006Gos", "YvM7H2D2Zcg", "aZTS8zNHKq4", "Dkk6TQ5e0sc",
    "7kFjSrlWoZE", "w0WD7UUWKnQ", "KZJqJkbMCzc", "fRTTWZshfws",
    "rKj1CN9Ttok",
}


def canonical(value: Any) -> str:
    return str(value).strip().casefold()


def stable_key(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def annotation_map(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("&")
        if len(parts) != 5:
            raise ValueError(f"AVE 标注第 {line_number} 行不是 5 列")
        result[parts[1].strip()] = parts[0].strip()
    return result


def collect_excluded_ids(paths: Iterable[Path]) -> set[str]:
    result = {canonical(value) for value in HISTORICAL_AVUT_IDS}
    for path in paths:
        for row in load_records(path):
            for key in ("youtube_id", "video_id", "source_video_id", "id"):
                if row.get(key) not in (None, ""):
                    result.add(canonical(row[key]))
    return result


def normalize_task_type(value: Any) -> str:
    task = canonical(value).split(":", 1)[0]
    if task not in TASK_TYPES:
        raise ValueError(f"不支持的 task_type：{value}")
    return task


def physical_media_id(youtube_id: str) -> str:
    digest = hashlib.sha256(canonical(youtube_id).encode()).hexdigest()[:20]
    return f"physical_{digest}"


def _prompt(question: str) -> list[dict[str, str]]:
    return [
        {"from": "human", "value": f"<image>\n{question}"},
        {"from": "gpt", "value": ""},
    ]


def build_ave_rows(
    valid_manifest: Path,
    annotations: Path,
    excluded_ids: set[str],
    expected_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = annotation_map(annotations)
    source_revision = f"sha256:{sha256_file(valid_manifest)}"
    rows = load_jsonl(valid_manifest)
    excluded = []
    duration_rejected = []
    strict = []
    for row in rows:
        youtube_id = str(row["youtube_id"])
        if canonical(youtube_id) in excluded_ids:
            excluded.append(youtube_id)
        elif abs(float(row["video_duration_sec"]) - float(row["audio_duration_sec"])) > 0.1:
            duration_rejected.append(youtube_id)
        else:
            strict.append(row)
    if len(strict) != expected_count:
        raise ValueError(
            f"AVE 严格候选数量应为 {expected_count}，实际为 {len(strict)}；"
            f"ID 排除 {len(excluded)}，时长排除 {len(duration_rejected)}"
        )

    output = []
    for index, row in enumerate(strict, 1):
        youtube_id = str(row["youtube_id"])
        if youtube_id not in labels:
            raise ValueError(f"AVE 缺少事件标签：{youtube_id}")
        video = Path(row["video_path"]).resolve()
        audio = Path(row["audio_path"]).resolve()
        if not video.is_file() or not audio.is_file():
            raise FileNotFoundError(f"AVE 媒体缺失：{youtube_id}")
        media_sha256 = sha256_file(video)
        audio_sha256 = sha256_file(audio)
        common = {
            "source_dataset": "AVE_HF_EXPANDED",
            "source_revision": source_revision,
            "source_pool": "existing",
            "video_id": youtube_id,
            "youtube_id": youtube_id,
            "physical_media_id": physical_media_id(youtube_id),
            "derived_media_id": f"AVE_HF_EXPANDED:{youtube_id}",
            "video_path": str(video),
            "scene_audio_path": str(audio),
            "scene_audio_sample_rate": 16000,
            "media_sha256": media_sha256,
            "audio_sha256": audio_sha256,
            "qa_origin": "template",
            "event_label": labels[youtube_id],
        }
        templates = (
            ("audio", "Which sound event is audible in this video?"),
            ("visual", "Which visible event or object is primarily shown in this video?"),
        )
        for task_type, question in templates:
            sample_id = f"ave_stage2_{youtube_id}_{task_type}"
            conversations = _prompt(question)
            conversations[1]["value"] = labels[youtube_id]
            output.append({
                **common,
                "id": sample_id,
                "sample_id": sample_id,
                "question": question,
                "answer": labels[youtube_id],
                "task_type": task_type,
                "conversations": conversations,
            })
        if index % 100 == 0:
            print(f"AVE 哈希进度：{index}/{len(strict)}", flush=True)
    return output, {
        "input_count": len(rows),
        "excluded_id_count": len(excluded),
        "duration_rejected_count": len(duration_rejected),
        "strict_count": len(strict),
        "excluded_ids": sorted(excluded),
        "duration_rejected_ids": sorted(duration_rejected),
    }


def normalize_stage1_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    output = []
    for row in rows:
        source = str(row["source_dataset"])
        task_type = normalize_task_type(row["task_type"])
        video_id = str(row["video_id"])
        youtube_id = str(row["youtube_id"])
        subtype = str(row["task_type"]).split(":", 1)[1] if ":" in str(row["task_type"]) else ""
        normalized = dict(row)
        normalized.update({
            "source_pool": "stage1_new",
            "physical_media_id": physical_media_id(youtube_id),
            "derived_media_id": f"{source}:{video_id}",
            "task_type": task_type,
            "task_subtype": subtype,
            "qa_origin": "human" if source == "AVUT" else "template",
        })
        output.append(normalized)
    return output


def build_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source_dataset"]), canonical(row["youtube_id"]))].append(row)
    groups = []
    for (source, youtube_id), values in grouped.items():
        hashes = {canonical(row["media_sha256"]) for row in values}
        paths = {str(Path(row["video_path"]).resolve()) for row in values}
        derived_ids = {canonical(row["derived_media_id"]) for row in values}
        video_ids = {canonical(row["video_id"]) for row in values}
        tasks = {normalize_task_type(row["task_type"]) for row in values}
        origins = {str(row["qa_origin"]) for row in values}
        groups.append({
            "group_key": f"{source}:{youtube_id}",
            "source_dataset": source,
            "youtube_id": youtube_id,
            "physical_media_id": physical_media_id(youtube_id),
            "rows": values,
            "media_sha256": hashes,
            "video_paths": paths,
            "derived_ids": derived_ids,
            "video_ids": video_ids,
            "task_types": tasks,
            "qa_origins": origins,
            "qa_count": len(values),
            "weight": len(hashes),
        })
    return groups


def deduplicate_groups(groups: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(
        groups,
        key=lambda group: (
            SOURCE_PRIORITY.get(group["source_dataset"], 99),
            stable_key(group["group_key"]),
        ),
    )
    kept = []
    dropped = []
    seen_youtube: dict[str, str] = {}
    seen_hash: dict[str, str] = {}
    seen_derived: dict[str, str] = {}
    seen_video: dict[tuple[str, str], str] = {}
    for group in ordered:
        collisions = []
        youtube_id = group["youtube_id"]
        if youtube_id in seen_youtube:
            collisions.append({"field": "youtube_id", "value": youtube_id, "kept_group": seen_youtube[youtube_id]})
        for value in group["media_sha256"]:
            if value in seen_hash:
                collisions.append({"field": "media_sha256", "value": value, "kept_group": seen_hash[value]})
        for value in group["derived_ids"]:
            if value in seen_derived:
                collisions.append({"field": "derived_media_id", "value": value, "kept_group": seen_derived[value]})
        for value in group["video_ids"]:
            key = (group["source_dataset"], value)
            if key in seen_video:
                collisions.append({"field": "video_id", "value": value, "kept_group": seen_video[key]})
        if collisions:
            dropped.append({
                "group_key": group["group_key"],
                "source_dataset": group["source_dataset"],
                "physical_media_count": group["weight"],
                "collisions": collisions,
            })
            continue
        kept.append(group)
        seen_youtube[youtube_id] = group["group_key"]
        for value in group["media_sha256"]:
            seen_hash[value] = group["group_key"]
        for value in group["derived_ids"]:
            seen_derived[value] = group["group_key"]
        for value in group["video_ids"]:
            seen_video[(group["source_dataset"], value)] = group["group_key"]
    return kept, dropped


def apportion(counts: dict[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values())
    if target > total:
        raise ValueError(f"目标 {target} 超过可用物理媒体 {total}")
    raw = {key: counts[key] * target / total for key in counts}
    result = {key: int(raw[key]) for key in counts}
    remaining = target - sum(result.values())
    order = sorted(counts, key=lambda key: (-(raw[key] - result[key]), stable_key(key)))
    for key in order[:remaining]:
        result[key] += 1
    return result


def stratified_order(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        qa_bin = min(group["qa_count"] // 3, 4)
        signature = (
            ",".join(sorted(group["task_types"])),
            ",".join(sorted(group["qa_origins"])),
            qa_bin,
        )
        buckets[signature].append(group)
    for values in buckets.values():
        values.sort(key=lambda group: stable_key(group["group_key"]))
    keys = sorted(buckets, key=lambda value: stable_key(repr(value)))
    result = []
    while True:
        progressed = False
        for key in keys:
            if buckets[key]:
                result.append(buckets[key].pop(0))
                progressed = True
        if not progressed:
            return result


def exact_weight_subset(groups: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    ordered = stratified_order(groups)
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for index, group in enumerate(ordered):
        weight = int(group["weight"])
        additions = {}
        for current, selected in list(reachable.items()):
            candidate = current + weight
            if candidate <= target and candidate not in reachable and candidate not in additions:
                additions[candidate] = selected + (index,)
        reachable.update(additions)
        if target in reachable:
            return [ordered[index] for index in reachable[target]]
    raise ValueError(f"无法按物理媒体权重精确选择 {target}，可达最大值 {max(reachable)}")


def assign_groups(
    groups: list[dict[str, Any]],
    train_target: int,
    dev_target: int,
) -> dict[str, list[dict[str, Any]]]:
    source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        source_groups[group["source_dataset"]].append(group)
    counts = {
        source: sum(int(group["weight"]) for group in values)
        for source, values in source_groups.items()
    }
    train_quota = apportion(counts, train_target)
    dev_quota = apportion(counts, dev_target)
    result = {"train": [], "dev": [], "reserve": []}
    for source, values in sorted(source_groups.items()):
        if train_quota[source] + dev_quota[source] > counts[source]:
            raise ValueError(f"{source} 的 train/dev 配额超过可用媒体")
        train = exact_weight_subset(values, train_quota[source])
        train_keys = {group["group_key"] for group in train}
        remaining = [group for group in values if group["group_key"] not in train_keys]
        dev = exact_weight_subset(remaining, dev_quota[source])
        dev_keys = {group["group_key"] for group in dev}
        reserve = [group for group in remaining if group["group_key"] not in dev_keys]
        result["train"].extend(train)
        result["dev"].extend(dev)
        result["reserve"].extend(reserve)
    return result


def flatten_split(groups: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    rows = []
    for group in groups:
        for row in group["rows"]:
            item = dict(row)
            item["physical_media_id"] = group["physical_media_id"]
            item["split"] = split
            rows.append(item)
    rows.sort(key=lambda row: stable_key(str(row["sample_id"])))
    return rows


def distribution(rows_by_split: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, Any]]:
    task_report = {}
    source_report = {}
    for split, rows in rows_by_split.items():
        task_report[split] = {
            "qa_count": dict(sorted(Counter(row["task_type"] for row in rows).items())),
            "physical_media_count": {
                task: len({
                    row["media_sha256"] for row in rows if row["task_type"] == task
                })
                for task in sorted(TASK_TYPES)
            },
        }
        source_report[split] = {}
        for source in sorted({row["source_dataset"] for row in rows}):
            selected = [row for row in rows if row["source_dataset"] == source]
            source_report[split][source] = {
                "qa_count": len(selected),
                "physical_group_count": len({row["physical_media_id"] for row in selected}),
                "physical_media_count": len({row["media_sha256"] for row in selected}),
                "qa_origin_count": dict(sorted(Counter(row["qa_origin"] for row in selected).items())),
            }
    return task_report, source_report


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> str:
    write_json(path, rows)
    digest = sha256_file(path)
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def write_split_report(path: Path, summary: dict[str, Any]) -> None:
    split = summary["split_physical_media_count"]
    lines = [
        "# 阶段 2 物理媒体划分报告",
        "",
        f"- 固定种子：`{SEED}`",
        f"- 内容级去重前物理媒体：{summary['pre_dedup_physical_media_count']}",
        f"- 内容级去重后物理媒体：{summary['post_dedup_physical_media_count']}",
        f"- 去重丢弃物理组：{summary['deduplicated_group_count']}",
        f"- train/dev/reserve：{split['train']}/{split['dev']}/{split['reserve']}",
        "",
        "划分在 `physical_media_id` 组级完成；同一 YouTube 来源的 flip、裁剪和 suffix "
        "派生媒体不会跨集合。AVE 使用 audio/visual 两种冻结模板，AVUT 保留人工 QA，"
        "MUSIC 保留原模板 QA。",
        "",
        "正式 manifest 已生成，但本脚本不启动训练。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ave-valid", type=Path, required=True)
    parser.add_argument("--ave-annotations", type=Path, required=True)
    parser.add_argument("--stage1-fields", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-ave-count", type=int, default=852)
    parser.add_argument("--train-media", type=int, default=1000)
    parser.add_argument("--dev-media", type=int, default=300)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    excluded_ids = collect_excluded_ids(args.exclude_manifest)
    ave_rows, ave_summary = build_ave_rows(
        args.ave_valid.resolve(),
        args.ave_annotations.resolve(),
        excluded_ids,
        args.expected_ave_count,
    )
    stage1_rows = normalize_stage1_rows(args.stage1_fields.resolve())
    groups = build_groups([*ave_rows, *stage1_rows])
    pre_dedup_count = sum(group["weight"] for group in groups)
    groups, dropped = deduplicate_groups(groups)
    post_dedup_count = sum(group["weight"] for group in groups)
    if post_dedup_count < args.train_media + args.dev_media + 50:
        raise ValueError(
            f"内容去重后只有 {post_dedup_count} 个物理媒体，"
            f"不足 train={args.train_media}、dev={args.dev_media}、reserve>=50"
        )
    assigned = assign_groups(groups, args.train_media, args.dev_media)
    rows_by_split = {
        split: flatten_split(values, split)
        for split, values in assigned.items()
    }
    manifest_sha256 = {}
    split_counts = {}
    for split, rows in rows_by_split.items():
        path = output_root / f"{split}_manifest.json"
        manifest_sha256[split] = write_manifest(path, rows)
        split_counts[split] = len({row["media_sha256"] for row in rows})
    task_report, source_report = distribution(rows_by_split)
    write_json(output_root / "task_distribution.json", task_report)
    write_json(output_root / "source_distribution.json", source_report)
    write_json(output_root / "deduplication_report.json", {
        "status": "PASS",
        "pre_dedup_group_count": len(groups) + len(dropped),
        "post_dedup_group_count": len(groups),
        "pre_dedup_physical_media_count": pre_dedup_count,
        "post_dedup_physical_media_count": post_dedup_count,
        "dropped_group_count": len(dropped),
        "dropped_groups": dropped,
    })
    exclusions = {
        "artifact_kind": "stage2_external_exclusions",
        "historical_avut_ids": sorted(HISTORICAL_AVUT_IDS),
        "entries": [{"youtube_id": value} for value in sorted(HISTORICAL_AVUT_IDS)],
        "manifests": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.exclude_manifest
        ],
    }
    write_json(output_root / "stage2_exclusions.json", exclusions)
    summary = {
        "status": "PASS",
        "seed": SEED,
        "input_sha256": {
            "ave_valid": sha256_file(args.ave_valid),
            "ave_annotations": sha256_file(args.ave_annotations),
            "stage1_fields": sha256_file(args.stage1_fields),
        },
        "ave_strict_rebuild": ave_summary,
        "pre_dedup_physical_media_count": pre_dedup_count,
        "post_dedup_physical_media_count": post_dedup_count,
        "deduplicated_group_count": len(dropped),
        "split_physical_media_count": split_counts,
        "split_qa_count": {split: len(rows) for split, rows in rows_by_split.items()},
        "manifest_sha256": manifest_sha256,
        "training_started": False,
    }
    write_json(output_root / "stage2_build_summary.json", summary)
    write_split_report(output_root / "stage2_split_report.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
