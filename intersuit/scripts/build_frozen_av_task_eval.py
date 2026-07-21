#!/usr/bin/env python
"""构建三类音视频最终任务评测候选；媒体齐全后才允许标记为冻结。"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from avut_common import extract_choices, inspect_schema, resolve_answer


INTERSUIT_ROOT = Path(__file__).resolve().parents[1]
SEED = 20260719
AUDIO_TASKS = {
    "Audio Information Extraction",
    "Audio Event Location",
    "Audio Content Counting",
}
JOINT_TASKS = {
    "Audio Character Matching",
    "Audio Object Matching",
    "Audio OCR Matching",
}
SOUND_RE = re.compile(
    r"\b(sound|hear|heard|audio|singing|speaking|barking|noise|music|voice|"
    r"instrument|alarm|explosion|clapping|crying|laughing|giggling|lyric|"
    r"says?|said|word|ringtone|beep)\b",
    re.I,
)
VISUAL_ONLY_RE = re.compile(
    r"\b(colou?r|wearing|clothes|appearance|left side|right side|look like|"
    r"doing when|text on)\b",
    re.I,
)
TIME_RE = re.compile(
    r"\b(when|while|during|before|after|first|last|time|period|throughout|"
    r"how many times|as the audio|audio says|audio mentions)\b",
    re.I,
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
HISTORICAL_AVUT_IDS = {
    "jkgJiV-l0k0",
    "4obNozIkugU",
    "x_x1lv_J3Ko",
    "JrWfGxd0J_4",
    "48AB70FSz7g",
    "g-jRGhwJ7BM",
    "6lbk8ihhYw0",
    "hVxbhnC302w",
    "ocMYLIDBs3U",
    "1rPehAUrYsQ",
    "uNkn_x2wFsA",
    "yMR2YHOdPno",
    "D4oMZ006Gos",
    "YvM7H2D2Zcg",
    "aZTS8zNHKq4",
    "Dkk6TQ5e0sc",
    "7kFjSrlWoZE",
    "w0WD7UUWKnQ",
    "KZJqJkbMCzc",
    "fRTTWZshfws",
    "rKj1CN9Ttok",
}


def stable_key(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def validate_media(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(row["video_path"])
    require_audio = row["evaluation_category"] != "video_necessary"
    result: dict[str, Any] = {
        "id": row["id"],
        "youtube_id": row["youtube_id"],
        "evaluation_category": row["evaluation_category"],
        "video_path": str(path),
        "require_audio": require_audio,
        "video_exists": path.is_file(),
        "file_size": path.stat().st_size if path.is_file() else 0,
    }
    if not path.is_file() or result["file_size"] <= 0:
        return {**result, "valid": False, "error": "媒体不存在或大小为 0"}
    probe = run_command(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    )
    if probe.returncode != 0:
        return {**result, "valid": False, "error": f"ffprobe 失败：{probe.stderr.strip()}"}
    try:
        metadata = json.loads(probe.stdout)
    except json.JSONDecodeError as exc:
        return {**result, "valid": False, "error": f"ffprobe 返回非法 JSON：{exc}"}
    streams = metadata.get("streams", [])
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    duration = float(metadata.get("format", {}).get("duration") or 0)
    decode_video = (
        run_command(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"])
        if video_streams
        else None
    )
    decode_audio = (
        run_command(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "null", "-"])
        if require_audio and audio_streams
        else None
    )
    video_decodable = bool(decode_video and decode_video.returncode == 0)
    audio_decodable = bool(not require_audio or (decode_audio and decode_audio.returncode == 0))
    valid = bool(
        video_streams
        and video_decodable
        and duration > 0
        and math.isfinite(duration)
        and (not require_audio or (audio_streams and audio_decodable))
    )
    errors = []
    if not video_streams:
        errors.append("没有视频流")
    if not video_decodable:
        errors.append(
            "视频解码失败"
            + (f"：{decode_video.stderr.strip()}" if decode_video and decode_video.stderr.strip() else "")
        )
    if duration <= 0 or not math.isfinite(duration):
        errors.append("时长无效")
    if require_audio and not audio_streams:
        errors.append("没有音频流")
    if require_audio and not audio_decodable:
        errors.append(
            "音频解码失败"
            + (f"：{decode_audio.stderr.strip()}" if decode_audio and decode_audio.stderr.strip() else "")
        )
    return {
        **result,
        "duration": duration,
        "has_video_stream": bool(video_streams),
        "has_audio_stream": bool(audio_streams),
        "video_decodable": video_decodable,
        "audio_decodable": audio_decodable,
        "valid": valid,
        "error": "; ".join(errors),
    }


def validate_rows(rows: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(validate_media, rows))


def youtube_id(record: dict[str, Any], video_path_field: str) -> str:
    return Path(str(record[video_path_field])).stem


def balanced_unique(
    records: list[dict[str, Any]],
    count: int,
    group_key: str,
    excluded_ids: set[str],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["youtube_id"] not in excluded_ids:
            groups[str(row[group_key])].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: stable_key(f"{row[group_key]}:{row['youtube_id']}:{row['id']}"))
    selected: list[dict[str, Any]] = []
    used = set(excluded_ids)
    indexes = Counter()
    labels = sorted(groups, key=stable_key)
    while len(selected) < count:
        progressed = False
        for label in labels:
            rows = groups[label]
            while indexes[label] < len(rows) and rows[indexes[label]]["youtube_id"] in used:
                indexes[label] += 1
            if indexes[label] >= len(rows):
                continue
            row = rows[indexes[label]]
            indexes[label] += 1
            selected.append(row)
            used.add(row["youtube_id"])
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError(f"只有 {len(selected)} 条不同视频候选，少于请求的 {count}")
    return selected


def avut_candidates(path: Path, media_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records, schema = inspect_schema(path)
    qf, af = schema["question_field"], schema["answer_field"]
    vf, vif = schema["video_path_field"], schema["video_id_field"]
    cf, ofs = schema["choices_field"], schema["option_fields"]
    tf = schema["question_category_field"]
    audio, joint = [], []
    for record in records:
        question = str(record[qf]).strip()
        task = str(record[tf])
        if not SOUND_RE.search(question):
            continue
        if task in AUDIO_TASKS and VISUAL_ONLY_RE.search(question):
            continue
        if task not in AUDIO_TASKS | JOINT_TASKS:
            continue
        choices = extract_choices(record, cf, ofs)
        answer, mapping = resolve_answer(record[af], choices)
        yt_id = youtube_id(record, vf)
        row = {
            "id": f"avut_{int(record['QA_id']):04d}",
            "youtube_id": yt_id,
            "source_video_id": record[vif],
            "source_qa_id": record["QA_id"],
            "source_url": record.get("url"),
            "video_path": str((media_root / f"{yt_id}.mp4").resolve()),
            "scene_audio_path": str((media_root / f"{yt_id}.mp4").resolve()),
            "question": question,
            "answer": answer,
            "choices": choices,
            "answer_mapping_method": mapping,
            "task_type": task,
            "time_alignment_sensitive": bool(TIME_RE.search(question)),
            "source_dataset": "AVUT",
            "generation_mode": "generate",
            "video_max_frames": 32,
            "scene_audio_sample_rate": 16000,
            "scene_audio_window_sec": 1.0,
            "scene_audio_hop_sec": 0.5,
            "conversations": [
                {"from": "human", "value": f"<image>\n{question}"},
                {"from": "gpt", "value": answer},
            ],
        }
        (audio if task in AUDIO_TASKS else joint).append(row)
    return audio, joint


def ave_annotation_map(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split("&")
        if len(parts) == 5:
            result[parts[1]] = parts[0]
    return result


def ave_choices(answer: str) -> dict[str, str]:
    distractors = sorted(
        (label for label in VISUAL_LABELS if label != answer),
        key=lambda label: stable_key(f"{answer}:{label}"),
    )[:3]
    values = sorted([answer, *distractors], key=lambda label: stable_key(f"choice:{answer}:{label}"))
    return dict(zip(("A", "B", "C", "D"), values))


def ave_candidates(valid_path: Path, annotation_path: Path) -> list[dict[str, Any]]:
    labels = ave_annotation_map(annotation_path)
    rows = []
    for record in load_jsonl(valid_path):
        yt_id = str(record["youtube_id"])
        label = labels.get(yt_id)
        if label not in VISUAL_LABELS:
            continue
        question = "Which visible event or object is primarily shown in this video?"
        rows.append(
            {
                "id": f"ave_{yt_id}",
                "youtube_id": yt_id,
                "video_path": str(Path(record["video_path"]).resolve()),
                "scene_audio_path": None,
                "question": question,
                "answer": label,
                "choices": ave_choices(label),
                "task_type": "Visual Event Classification",
                "time_alignment_sensitive": False,
                "source_dataset": "AVE",
                "generation_mode": "generate",
                "video_max_frames": 32,
                "conversations": [
                    {"from": "human", "value": f"<image>\n{question}"},
                    {"from": "gpt", "value": label},
                ],
            }
        )
    return rows


def replacement_group(row: dict[str, Any]) -> tuple[str, str]:
    if row["evaluation_category"] == "video_necessary":
        return row["evaluation_category"], row["answer"]
    return row["evaluation_category"], row["task_type"]


def replace_invalid_rows(
    rows: list[dict[str, Any]],
    invalid_ids: set[str],
    pools: dict[str, list[dict[str, Any]]],
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    used_youtube_ids = {row["youtube_id"] for row in rows}
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for category, pool in pools.items():
        for original in pool:
            row = {**original, "evaluation_category": category}
            if row["youtube_id"] in used_youtube_ids or row["youtube_id"] in HISTORICAL_AVUT_IDS:
                continue
            candidates[replacement_group(row)].append(row)
    for values in candidates.values():
        values.sort(key=lambda row: stable_key(f"replacement:{row['youtube_id']}:{row['id']}"))

    replacements = []
    replacement_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["id"] not in invalid_ids:
            continue
        group = replacement_group(row)
        while candidates[group]:
            candidate = candidates[group].pop(0)
            if not Path(candidate["video_path"]).is_file():
                continue
            validation = validate_rows([candidate], workers)[0]
            if not validation["valid"]:
                continue
            replacement_by_id[row["id"]] = candidate
            used_youtube_ids.add(candidate["youtube_id"])
            replacements.append(
                {
                    "replaced_id": row["id"],
                    "replaced_youtube_id": row["youtube_id"],
                    "replacement_id": candidate["id"],
                    "replacement_youtube_id": candidate["youtube_id"],
                    "evaluation_category": row["evaluation_category"],
                    "task_type": row["task_type"],
                }
            )
            break
        if row["id"] not in replacement_by_id:
            raise FileNotFoundError(
                f"{row['id']} 没有通过媒体校验的同组备用样本，禁止冻结"
            )
    return [replacement_by_id.get(row["id"], row) for row in rows], replacements


def run(args: argparse.Namespace) -> dict[str, Any]:
    per_category = int(args.per_category)
    if not 100 <= per_category <= 200:
        raise ValueError("per_category 必须在 100～200 之间")
    output_root = Path(args.output_root).resolve()
    media_root = Path(args.avut_media_root).resolve()
    audio_pool, joint_pool = avut_candidates(Path(args.avut_annotations).resolve(), media_root)
    selection_audio_pool = audio_pool
    audio = balanced_unique(
        selection_audio_pool,
        per_category,
        "task_type",
        HISTORICAL_AVUT_IDS,
    )
    audio_ids = {row["youtube_id"] for row in audio}
    selection_joint_pool = joint_pool
    joint = balanced_unique(
        selection_joint_pool,
        per_category,
        "task_type",
        HISTORICAL_AVUT_IDS | audio_ids,
    )
    ave = balanced_unique(
        ave_candidates(
            Path(args.ave_valid_manifest).resolve(),
            Path(args.ave_annotations).resolve(),
        ),
        per_category,
        "answer",
        set(),
    )
    categories = {
        "audio_necessary": audio,
        "video_necessary": ave,
        "audio_visual_joint": joint,
    }
    rows = []
    for category, values in categories.items():
        for row in values:
            rows.append({**row, "evaluation_category": category})
    missing = [
        row
        for row in rows
        if not Path(row["video_path"]).is_file()
        or (
            row["evaluation_category"] != "video_necessary"
            and not Path(row["scene_audio_path"]).is_file()
        )
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    frozen = False
    manifest_name = "candidate_eval.json"
    replacements: list[dict[str, Any]] = []
    media_validation: dict[str, Any] | None = None
    if args.require_media:
        if not shutil.which("ffprobe") or not shutil.which("ffmpeg"):
            raise RuntimeError("ffprobe 或 ffmpeg 不可用，禁止冻结")
        initial_validation = validate_rows(rows, args.media_workers)
        invalid_ids = {item["id"] for item in initial_validation if not item["valid"]}
        if invalid_ids:
            rows, replacements = replace_invalid_rows(
                rows,
                invalid_ids,
                {
                    "audio_necessary": audio_pool,
                    "video_necessary": ave_candidates(
                        Path(args.ave_valid_manifest).resolve(),
                        Path(args.ave_annotations).resolve(),
                    ),
                    "audio_visual_joint": joint_pool,
                },
                args.media_workers,
            )
        final_validation = validate_rows(rows, args.media_workers)
        unique_ids = len({row["id"] for row in rows})
        unique_youtube_ids = len({row["youtube_id"] for row in rows})
        audio_rows = [
            row for row in rows if row["evaluation_category"] != "video_necessary"
        ]
        media_validation = {
            "candidate_count": len(rows),
            "video_decodable_count": sum(bool(item["video_decodable"]) for item in final_validation),
            "audio_required_count": len(audio_rows),
            "audio_stream_count": sum(
                bool(item["has_audio_stream"]) for item in final_validation if item["require_audio"]
            ),
            "audio_decodable_count": sum(
                bool(item["audio_decodable"]) for item in final_validation if item["require_audio"]
            ),
            "unique_id_count": unique_ids,
            "unique_youtube_id_count": unique_youtube_ids,
            "replacement_count": len(replacements),
            "replacements": replacements,
            "all_valid": all(item["valid"] for item in final_validation),
            "results": final_validation,
        }
        write_json(output_root / "media_validation.json", media_validation)
        hard_checks_passed = bool(
            len(rows) == per_category * 3
            and media_validation["video_decodable_count"] == len(rows)
            and len(audio_rows) == per_category * 2
            and media_validation["audio_stream_count"] == len(audio_rows)
            and media_validation["audio_decodable_count"] == len(audio_rows)
            and unique_ids == len(rows)
            and unique_youtube_ids == len(rows)
            and media_validation["all_valid"]
        )
        if not hard_checks_passed:
            raise RuntimeError("媒体或 ID 硬校验未全部通过，禁止生成 frozen_eval.json")
        frozen = True
        manifest_name = "frozen_eval.json"
        missing = [
            row
            for row in rows
            if not Path(row["video_path"]).is_file()
            or (
                row["evaluation_category"] != "video_necessary"
                and not Path(row["scene_audio_path"]).is_file()
            )
        ]
    write_json(output_root / manifest_name, rows)
    download_rows = []
    if not frozen:
        for category, values in (
            ("audio_necessary", audio),
            ("audio_visual_joint", joint),
        ):
            for row in values:
                if not Path(row["video_path"]).is_file():
                    download_rows.append(
                        {
                            "id": row["id"],
                            "youtube_id": row["youtube_id"],
                            "url": row["source_url"],
                            "output_path": row["video_path"],
                            "evaluation_category": category,
                        }
                    )
    write_json(output_root / "download_jobs.json", download_rows)
    (output_root / "download_urls.txt").write_text(
        "".join(f"{row['url']}\n" for row in download_rows),
        encoding="utf-8",
    )
    reserve_download_rows = []
    if not args.require_media and missing:
        selected_avut_ids = {
            row["youtube_id"]
            for row in [*audio, *joint]
        }
        missing_by_group = Counter(
            (row["evaluation_category"], row["task_type"])
            for row in missing
            if row["evaluation_category"] != "video_necessary"
        )
        reserve_used = set(HISTORICAL_AVUT_IDS) | selected_avut_ids
        for category, pool in (
            ("audio_necessary", audio_pool),
            ("audio_visual_joint", joint_pool),
        ):
            for task_type in sorted(
                {task for (name, task), count in missing_by_group.items() if name == category and count},
                key=stable_key,
            ):
                needed = missing_by_group[(category, task_type)] * 3
                candidates = sorted(
                    (
                        row
                        for row in pool
                        if row["task_type"] == task_type
                        and row["youtube_id"] not in reserve_used
                    ),
                    key=lambda row: stable_key(f"reserve:{row['youtube_id']}:{row['id']}"),
                )
                for row in candidates[:needed]:
                    reserve_used.add(row["youtube_id"])
                    reserve_download_rows.append(
                        {
                            "id": row["id"],
                            "youtube_id": row["youtube_id"],
                            "url": row["source_url"],
                            "output_path": row["video_path"],
                            "evaluation_category": category,
                            "task_type": task_type,
                        }
                    )
        write_json(output_root / "reserve_download_jobs.json", reserve_download_rows)
        (output_root / "reserve_download_urls.txt").write_text(
            "".join(f"{row['url']}\n" for row in reserve_download_rows),
            encoding="utf-8",
        )
    manifest_sha256 = None
    sample_ids_sha256 = None
    if frozen:
        manifest_path = output_root / manifest_name
        manifest_sha256 = sha256_file(manifest_path)
        ordered_ids = "".join(f"{row['id']}\n" for row in rows).encode("utf-8")
        sample_ids_sha256 = sha256_bytes(ordered_ids)
        (output_root / "frozen_eval.sha256").write_text(
            f"{manifest_sha256}  frozen_eval.json\n",
            encoding="ascii",
        )
        (output_root / "frozen_eval_ids.sha256").write_text(
            f"{sample_ids_sha256}  frozen_eval.ids\n",
            encoding="ascii",
        )
        write_json(
            output_root / "frozen_eval_lock.json",
            {
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha256,
                "ordered_sample_ids_sha256": sample_ids_sha256,
                "sample_count": len(rows),
                "ordered_sample_ids": [row["id"] for row in rows],
            },
        )
    summary = {
        "status": "frozen" if frozen else "candidate_not_frozen",
        "seed": SEED,
        "sample_count": len(rows),
        "per_category": per_category,
        "category_counts": {key: len(value) for key, value in categories.items()},
        "unique_youtube_id_count": len({row["youtube_id"] for row in rows}),
        "avut_cross_category_overlap": len(audio_ids & {row["youtube_id"] for row in joint}),
        "historical_avut_overlap": len(
            HISTORICAL_AVUT_IDS & {row["youtube_id"] for row in [*audio, *joint]}
        ),
        "time_alignment_sensitive_counts": {
            key: sum(bool(row["time_alignment_sensitive"]) for row in value)
            for key, value in categories.items()
        },
        "missing_media_count": len(missing),
        "download_job_count": len(download_rows),
        "reserve_download_job_count": len(reserve_download_rows),
        "manifest": str(output_root / manifest_name),
        "manifest_sha256": manifest_sha256,
        "ordered_sample_ids_sha256": sample_ids_sha256,
        "media_validation": (
            {
                key: value
                for key, value in media_validation.items()
                if key not in {"results", "replacements"}
            }
            if media_validation
            else None
        ),
    }
    write_json(output_root / "selection_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--avut-annotations", default="datasets/AVUT/raw/AV_Human_data.json")
    parser.add_argument("--avut-media-root", default="datasets/AVUT/frozen_dev_videos")
    parser.add_argument("--ave-valid-manifest", default="datasets/AVE_HF_EXPANDED/ave_hf_pilot_valid.jsonl")
    parser.add_argument("--ave-annotations", default="datasets/AVE/data/Annotations.txt")
    parser.add_argument("--output-root", default="harness/artifacts/frozen_av_task_eval_dev300")
    parser.add_argument("--per-category", type=int, default=100)
    parser.add_argument("--require-media", action="store_true")
    parser.add_argument("--media-workers", type=int, default=8)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False))
