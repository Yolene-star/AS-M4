#!/usr/bin/env python
"""准备 LLP、AVE、AVQA 的统一音画事件标注 manifest。

本脚本只读取标注并生成小规模候选清单，不下载媒体、不训练 projector、
不修改 Gate 或动态窗口路径。LLP/AVE 的音画同步窗口可作为后续重新提取
BEATs 与 M4 视频窗口特征的输入；AVQA 目前只做 Sound/View/Both 统计。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import h5py
except ImportError:  # pragma: no cover - exercised only in minimal envs
    h5py = None


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"
DEFAULT_DATASET_ROOT = INTERSUIT_ROOT / "datasets"
DEFAULT_OUTPUT_ROOT = INTERSUIT_ROOT / "harness/artifacts/audio_visual_event_manifests"
WINDOW_SEC = 1.0
HOP_SEC = 0.5
CLIP_END_SEC = 10.0


@dataclass(frozen=True)
class UnifiedEventRecord:
    dataset: str
    sample_id: str
    video_path: str | None
    window_start: float
    window_end: float
    modality_role: str
    event_label: str | None
    audio_required: bool
    visible_event_present: bool
    split: str
    source_annotation: str


def round_time(value: float) -> float:
    return round(float(value), 6)


def iter_windows(start: float = 0.0, end: float = CLIP_END_SEC, window: float = WINDOW_SEC, hop: float = HOP_SEC) -> Iterable[tuple[float, float]]:
    current = float(start)
    last_start = float(end) - float(window)
    while current <= last_start + 1e-9:
        yield round_time(current), round_time(current + window)
        current += hop


def interval_overlaps(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    return min(end_a, end_b) - max(start_a, start_b) > 1e-9


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"标注文件为空：{path}")
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    sample = path.read_text(encoding="utf-8-sig")[:2048]
    delimiter = "\t" if sample.splitlines()[0].count("\t") > sample.splitlines()[0].count(",") else ","
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    if not rows:
        raise ValueError(f"标注文件为空：{path}")
    return rows


def load_llp_split_map(llp_root: Path) -> tuple[dict[str, str], dict[str, int]]:
    files = {
        "train": llp_root / "data" / "AVVP_train.csv",
        "val": llp_root / "data" / "AVVP_val_pd.csv",
        "test": llp_root / "data" / "AVVP_test_pd.csv",
    }
    split_map: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split, path in files.items():
        rows = read_tsv(path)
        counts[split] = len(rows)
        for row in rows:
            filename = row.get("filename", "").strip()
            if filename:
                split_map[filename] = split
    return split_map, counts


def load_llp_dense(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    rows = read_csv(path)
    grouped: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for row in rows:
        filename = row.get("filename", "").strip()
        label = row.get("event_labels", "").strip()
        if not filename or not label:
            continue
        onset = float(row["onset"])
        offset = float(row["offset"])
        if offset <= onset:
            continue
        grouped[filename].append((onset, offset, label))
    return dict(grouped)


def active_labels(intervals: list[tuple[float, float, str]], start: float, end: float) -> set[str]:
    return {label for onset, offset, label in intervals if interval_overlaps(onset, offset, start, end)}


def build_llp_records(llp_root: Path) -> tuple[list[UnifiedEventRecord], dict[str, Any]]:
    split_map, weak_counts = load_llp_split_map(llp_root)
    audio = load_llp_dense(llp_root / "data" / "AVVP_eval_audio.csv")
    visual = load_llp_dense(llp_root / "data" / "AVVP_eval_visual.csv")
    filenames = sorted(set(audio) | set(visual))
    records: list[UnifiedEventRecord] = []

    for filename in filenames:
        split = split_map.get(filename, "unknown")
        for start, end in iter_windows():
            audio_labels = active_labels(audio.get(filename, []), start, end)
            visual_labels = active_labels(visual.get(filename, []), start, end)
            shared = sorted(audio_labels & visual_labels)
            audio_only = sorted(audio_labels - visual_labels)
            visual_only = sorted(visual_labels - audio_labels)
            if shared:
                for label in shared:
                    records.append(
                        UnifiedEventRecord(
                            dataset="LLP",
                            sample_id=filename,
                            video_path=None,
                            window_start=start,
                            window_end=end,
                            modality_role="audio_visual",
                            event_label=label,
                            audio_required=True,
                            visible_event_present=True,
                            split=split,
                            source_annotation="AVVP_eval_audio.csv+AVVP_eval_visual.csv",
                        )
                    )
            for label in audio_only:
                records.append(
                    UnifiedEventRecord(
                        dataset="LLP",
                        sample_id=filename,
                        video_path=None,
                        window_start=start,
                        window_end=end,
                        modality_role="audio_only",
                        event_label=label,
                        audio_required=True,
                        visible_event_present=False,
                        split=split,
                        source_annotation="AVVP_eval_audio.csv",
                    )
                )
            for label in visual_only:
                records.append(
                    UnifiedEventRecord(
                        dataset="LLP",
                        sample_id=filename,
                        video_path=None,
                        window_start=start,
                        window_end=end,
                        modality_role="visual_only",
                        event_label=label,
                        audio_required=False,
                        visible_event_present=True,
                        split=split,
                        source_annotation="AVVP_eval_visual.csv",
                    )
                )
            if not shared and not audio_only and not visual_only:
                records.append(
                    UnifiedEventRecord(
                        dataset="LLP",
                        sample_id=filename,
                        video_path=None,
                        window_start=start,
                        window_end=end,
                        modality_role="background",
                        event_label=None,
                        audio_required=False,
                        visible_event_present=False,
                        split=split,
                        source_annotation="AVVP_eval_audio.csv+AVVP_eval_visual.csv",
                    )
                )

    summary = {
        "weak_split_counts": weak_counts,
        "dense_audio_video_count": len(audio),
        "dense_visual_video_count": len(visual),
        "dense_union_video_count": len(filenames),
        "record_count": len(records),
        "role_counts": dict(Counter(record.modality_role for record in records)),
    }
    return records, summary


def read_h5_order(path: Path) -> list[int]:
    if h5py is None:
        raise ImportError("读取 AVE split 需要 h5py，但当前环境未安装")
    with h5py.File(path, "r") as handle:
        if "order" not in handle:
            raise ValueError(f"AVE split 文件缺少 order key：{path}")
        return [int(value) for value in handle["order"][:].reshape(-1)]


def load_ave_splits(ave_root: Path, annotation_count: int) -> tuple[dict[int, str], dict[str, int]]:
    split_map: dict[int, str] = {}
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        order_path = ave_root / "data" / f"{split}_order.h5"
        order = read_h5_order(order_path)
        counts[split] = len(order)
        for index in order:
            if index < 0 or index >= annotation_count:
                raise ValueError(f"{order_path} 中存在越界索引 {index}，标注总数 {annotation_count}")
            if index in split_map:
                raise ValueError(f"AVE split 重复包含行号 {index}")
            split_map[index] = split
    missing = sorted(set(range(annotation_count)) - set(split_map))
    if missing:
        raise ValueError(f"AVE split 未覆盖全部标注行，缺失前 5 个索引：{missing[:5]}")
    return split_map, counts


def parse_ave_annotations(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle):
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split("&")
            if len(parts) != 5:
                raise ValueError(f"AVE 标注第 {line_number + 1} 行不是 5 列：{raw}")
            label, video_id, quality, start, end = parts
            rows.append(
                {
                    "line_index": line_number,
                    "event_label": label.strip(),
                    "video_id": video_id.strip(),
                    "quality": quality.strip(),
                    "event_start": float(start),
                    "event_end": float(end),
                }
            )
    if not rows:
        raise ValueError(f"AVE 标注文件为空：{path}")
    return rows


def build_ave_records(ave_root: Path) -> tuple[list[UnifiedEventRecord], dict[str, Any]]:
    annotations = parse_ave_annotations(ave_root / "data" / "Annotations.txt")
    split_map, split_counts = load_ave_splits(ave_root, len(annotations))
    records: list[UnifiedEventRecord] = []
    skipped = Counter()

    for row in annotations:
        start = float(row["event_start"])
        end = float(row["event_end"])
        if row["quality"] != "good":
            skipped["non_good_quality"] += 1
            continue
        if end <= start:
            skipped["empty_or_invalid_interval"] += 1
            continue
        for window_start, window_end in iter_windows():
            center = (window_start + window_end) / 2.0
            if center < start or center >= end:
                continue
            records.append(
                UnifiedEventRecord(
                    dataset="AVE",
                    sample_id=row["video_id"],
                    video_path=None,
                    window_start=window_start,
                    window_end=window_end,
                    modality_role="audio_visual",
                    event_label=row["event_label"],
                    audio_required=True,
                    visible_event_present=True,
                    split=split_map[int(row["line_index"])],
                    source_annotation="Annotations.txt",
                )
            )

    summary = {
        "annotation_count": len(annotations),
        "split_counts": split_counts,
        "record_count": len(records),
        "role_counts": dict(Counter(record.modality_role for record in records)),
        "skipped": dict(skipped),
    }
    return records, summary


def normalize_avqa_relation(value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    if text in {"sound", "audio"}:
        return "audio_only"
    if text in {"view", "visual"}:
        return "visual_only"
    if text in {"both", "audio-visual", "audio_visual", "av"}:
        return "both_required"
    return None


def summarize_avqa(avqa_root: Path, per_role_limit: int) -> dict[str, Any]:
    files = [avqa_root / "train_qa.json", avqa_root / "val_qa.json"]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        return {"available": False, "missing_files": missing}

    counts = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in files:
        split = path.stem.replace("_qa", "")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        if not isinstance(rows, list):
            raise ValueError(f"AVQA 标注顶层无法解析为列表：{path}")
        for row in rows:
            relation = normalize_avqa_relation(row.get("question_relation"))
            if relation is None:
                counts["unknown"] += 1
                continue
            counts[relation] += 1
            if len(examples[relation]) < per_role_limit:
                examples[relation].append({"split": split, **row})
    return {"available": True, "relation_counts": dict(counts), "examples": dict(examples)}


def select_records(records: list[UnifiedEventRecord], dataset: str, role: str, limit: int) -> list[UnifiedEventRecord]:
    selected = [record for record in records if record.dataset == dataset and record.modality_role == role]
    return selected[:limit]


def resolve_media_path(record: UnifiedEventRecord, media_roots: list[Path]) -> Path | None:
    candidates: list[Path] = []
    if record.video_path:
        candidates.append(Path(record.video_path))
    suffixes = (".mp4", ".webm", ".mkv", ".avi", ".mov")
    for root in media_roots:
        for suffix in suffixes:
            candidates.append(root / record.dataset / f"{record.sample_id}{suffix}")
            candidates.append(root / f"{record.sample_id}{suffix}")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def ffmpeg_probe(path: Path) -> dict[str, Any]:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {"ok": result.returncode == 0, "stderr": result.stderr.strip()[:500]}


def validate_media(records: list[UnifiedEventRecord], media_roots: list[Path], limit: int) -> dict[str, Any]:
    if not media_roots:
        return {"enabled": False, "reason": "未提供 --media-root，本轮只验证标注"}
    checked = []
    for record in records[:limit]:
        path = resolve_media_path(record, media_roots)
        if path is None:
            checked.append({"dataset": record.dataset, "sample_id": record.sample_id, "exists": False, "audio_decode_ok": False})
            continue
        probe = ffmpeg_probe(path)
        checked.append(
            {
                "dataset": record.dataset,
                "sample_id": record.sample_id,
                "path": str(path),
                "exists": True,
                "audio_decode_ok": probe["ok"],
                "ffmpeg_error": probe["stderr"],
            }
        )
    return {
        "enabled": True,
        "checked_count": len(checked),
        "available_count": sum(1 for item in checked if item["exists"]),
        "audio_decode_ok_count": sum(1 for item in checked if item["audio_decode_ok"]),
        "items": checked,
    }


def write_jsonl(path: Path, records: Iterable[UnifiedEventRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# LLP + AVE 音画事件标注准备报告",
        "",
        "本轮只读取和统一标注，不修改 Gate、不接入动态窗口、不训练 M4 或 projector。",
        "",
        "## 输出",
        f"- 统一 manifest：`{summary['paths']['unified_manifest']}`",
        f"- projector 正样本候选：`{summary['paths']['projector_positive_candidates']}`",
        f"- AVQA 统计：`{summary['paths']['avqa_summary']}`",
        "",
        "## LLP",
        f"- dense 视频数：{summary['LLP']['dense_union_video_count']}",
        f"- 记录数：{summary['LLP']['record_count']}",
        f"- role 分布：`{summary['LLP']['role_counts']}`",
        "",
        "## AVE",
        f"- 标注条数：{summary['AVE']['annotation_count']}",
        f"- 展开窗口数：{summary['AVE']['record_count']}",
        f"- split 分布：`{summary['AVE']['split_counts']}`",
        f"- 跳过项：`{summary['AVE']['skipped']}`",
        "",
        "## AVQA",
    ]
    if summary["AVQA"].get("available"):
        lines.append(f"- Sound/View/Both 统计：`{summary['AVQA']['relation_counts']}`")
    else:
        lines.append(f"- 暂未读取：`{summary['AVQA'].get('missing_files')}`")
    lines.extend(
        [
            "",
            "## 媒体验证",
            f"- 状态：`{summary['media_validation']}`",
            "",
            "## 结论",
            "- 已准备 LLP audio_visual 窗口和 AVE 同步事件窗口，适合进入小规模媒体可用性验证。",
            "- LLP audio_only 与 AVQA Sound 只保留为后续音频独立理解/问答评测资源，不作为音画 projector 正样本。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    llp_root = Path(args.llp_root).resolve() if args.llp_root else dataset_root / "LLP"
    ave_root = Path(args.ave_root).resolve() if args.ave_root else dataset_root / "AVE"
    avqa_root = Path(args.avqa_root).resolve() if args.avqa_root else dataset_root / "AVQA"

    llp_records, llp_summary = build_llp_records(llp_root)
    ave_records, ave_summary = build_ave_records(ave_root)
    all_records = llp_records + ave_records

    llp_positive = select_records(all_records, "LLP", "audio_visual", args.llp_audio_visual_limit)
    ave_positive = select_records(all_records, "AVE", "audio_visual", args.ave_video_limit)
    projector_candidates = llp_positive + ave_positive

    paths = {
        "unified_manifest": str(output_root / "unified_audio_visual_manifest.jsonl"),
        "projector_positive_candidates": str(output_root / "projector_positive_candidates.jsonl"),
        "llp_audio_visual_candidates": str(output_root / "llp_audio_visual_candidates.jsonl"),
        "ave_audio_visual_candidates": str(output_root / "ave_audio_visual_candidates.jsonl"),
        "avqa_summary": str(output_root / "avqa_annotation_summary.json"),
        "summary": str(output_root / "annotation_summary.json"),
        "report": str(output_root / "annotation_report.md"),
    }

    write_jsonl(Path(paths["unified_manifest"]), all_records)
    write_jsonl(Path(paths["projector_positive_candidates"]), projector_candidates)
    write_jsonl(Path(paths["llp_audio_visual_candidates"]), llp_positive)
    write_jsonl(Path(paths["ave_audio_visual_candidates"]), ave_positive)
    avqa_summary = summarize_avqa(avqa_root, args.avqa_per_role_limit)
    write_json(Path(paths["avqa_summary"]), avqa_summary)

    media_roots = [Path(value).resolve() for value in args.media_root]
    media_summary = validate_media(projector_candidates, media_roots, args.media_check_limit)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "window_sec": args.window_sec,
        "hop_sec": args.hop_sec,
        "LLP": llp_summary,
        "AVE": ave_summary,
        "AVQA": avqa_summary,
        "projector_candidate_counts": {
            "LLP_audio_visual": len(llp_positive),
            "AVE_audio_visual": len(ave_positive),
            "total": len(projector_candidates),
        },
        "media_validation": media_summary,
        "paths": paths,
    }
    write_json(Path(paths["summary"]), summary)
    write_report(Path(paths["report"]), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--llp-root", default=None)
    parser.add_argument("--ave-root", default=None)
    parser.add_argument("--avqa-root", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--llp-audio-visual-limit", type=int, default=300)
    parser.add_argument("--ave-video-limit", type=int, default=100)
    parser.add_argument("--avqa-per-role-limit", type=int, default=50)
    parser.add_argument("--media-root", action="append", default=[])
    parser.add_argument("--media-check-limit", type=int, default=50)
    parser.add_argument("--window-sec", type=float, default=WINDOW_SEC)
    parser.add_argument("--hop-sec", type=float, default=HOP_SEC)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not math.isclose(args.window_sec, WINDOW_SEC) or not math.isclose(args.hop_sec, HOP_SEC):
        raise ValueError("本阶段固定使用 window=1.0s、hop=0.5s；如需变更请先同步特征提取路径")
    summary = run(args)
    print(json.dumps({"ok": True, "summary": summary["paths"]["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
