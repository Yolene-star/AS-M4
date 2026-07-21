#!/usr/bin/env python
"""使用 M4 本地 CLIP 视觉塔提取 AVE_HF 窗口级语义视频特征。

本脚本只加载本地 `clip-vit-large-patch14-336` 视觉塔，不加载语言模型，
不修改正式 M4 推理路径，不修改 Gate 或动态窗口。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"
DEFAULT_FEATURE_MANIFEST = INTERSUIT_ROOT / "harness/artifacts/ave_hf_300_window_features/ave_hf_window_feature_manifest.jsonl"
DEFAULT_OUTPUT_ROOT = INTERSUIT_ROOT / "harness/artifacts/ave_hf_300_clip_window_features"
DEFAULT_VISION_TOWER = INTERSUIT_ROOT / "checkpoints/clip-vit-large-patch14-336"


@dataclass(frozen=True)
class ClipFeatureRecord:
    dataset: str
    sample_id: str
    youtube_id: str
    split: str
    label: str | None
    start_seconds: float | None
    audio_feature_path: str
    video_feature_path: str
    source_video_path: str
    window_count: int
    window_start: float
    window_end: float
    audio_embedding_dim: int
    video_embedding_dim: int


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"manifest 为空：{path}")
    return rows


def select_rows_by_label(rows: list[dict[str, Any]], samples_per_label: int | None) -> list[dict[str, Any]]:
    if samples_per_label is None:
        return rows
    if samples_per_label <= 0:
        raise ValueError("--samples-per-label 必须为正数")
    counts: Counter[str] = Counter()
    selected = []
    for row in rows:
        label = str(row.get("label"))
        if counts[label] < samples_per_label:
            selected.append(row)
            counts[label] += 1
    if not selected:
        raise ValueError("按 label 抽样后没有可用样本")
    return selected


def slice_rows(
    rows: list[dict[str, Any]],
    start_index: int | None,
    end_index: int | None,
) -> list[dict[str, Any]]:
    start = 0 if start_index is None else start_index
    end = len(rows) if end_index is None else end_index
    if start < 0 or end < start or end > len(rows):
        raise ValueError(f"分片范围非法：start={start}, end={end}, total={len(rows)}")
    selected = rows[start:end]
    if not selected:
        raise ValueError(f"分片范围为空：start={start}, end={end}, total={len(rows)}")
    return selected


def load_audio_timestamps(path: Path) -> tuple[torch.Tensor, int]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    timestamps = payload["timestamps"].float()
    audio = payload["audio_embedding"].float()
    if timestamps.shape != (audio.shape[0], 2):
        raise ValueError(f"音频窗口和时间戳不一致：{path}")
    if not torch.isfinite(timestamps).all() or not torch.isfinite(audio).all():
        raise ValueError(f"音频特征包含 NaN/Inf：{path}")
    return timestamps, int(audio.shape[-1])


def load_local_clip_vision_model(vision_tower: Path) -> CLIPVisionModel:
    try:
        return CLIPVisionModel.from_pretrained(vision_tower, local_files_only=True)
    except ValueError as exc:
        if "torch.load" not in str(exc) or "safetensors" not in str(exc):
            raise
    config = CLIPVisionConfig.from_pretrained(vision_tower, local_files_only=True)
    model = CLIPVisionModel(config)
    checkpoint_path = vision_tower / "pytorch_model.bin"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"本地 CLIP 权重缺失：{checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    vision_state = {key: value for key, value in state_dict.items() if key.startswith("vision_model.")}
    missing, unexpected = model.load_state_dict(vision_state, strict=False)
    unexpected = [key for key in unexpected if not key.endswith("position_ids")]
    if unexpected:
        raise ValueError(f"CLIP 视觉塔权重存在未预期字段：{unexpected[:5]}")
    critical_missing = [key for key in missing if not key.endswith("position_ids")]
    if critical_missing:
        raise ValueError(f"CLIP 视觉塔权重缺失字段：{critical_missing[:5]}")
    return model


def frame_indices_for_windows(timestamps: torch.Tensor, fps: float, frame_count: int, frames_per_window: int) -> list[list[int]]:
    all_indices = []
    for start, end in timestamps.tolist():
        if frames_per_window <= 1:
            centers = [(start + end) / 2.0]
        else:
            step = (end - start) / frames_per_window
            centers = [start + step * (idx + 0.5) for idx in range(frames_per_window)]
        indices = [max(0, min(frame_count - 1, int(round(center * fps)))) for center in centers]
        all_indices.append(indices)
    return all_indices


def load_video_frames_with_decord(video_path: Path, timestamps: torch.Tensor, frames_per_window: int) -> list[list[Image.Image]]:
    import decord

    reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
    frame_count = len(reader)
    fps = float(reader.get_avg_fps() or 0.0)
    if frame_count <= 0 or fps <= 0:
        raise ValueError(f"视频无法读取帧或 fps 非法：{video_path}")
    grouped_indices = frame_indices_for_windows(timestamps, fps, frame_count, frames_per_window)
    flat_indices = [idx for group in grouped_indices for idx in group]
    # Keep video decoding independent from CUDA. The decord torch bridge can
    # segfault in this environment when CUDA is visible.
    frames = reader.get_batch(flat_indices).asnumpy()
    pil_frames = [Image.fromarray(frame) for frame in frames]
    grouped = []
    cursor = 0
    for group in grouped_indices:
        grouped.append(pil_frames[cursor : cursor + len(group)])
        cursor += len(group)
    return grouped


def load_video_frames_with_opencv(video_path: Path, timestamps: torch.Tensor, frames_per_window: int) -> list[list[Image.Image]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV 无法打开视频：{video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0 or fps <= 0:
        capture.release()
        raise ValueError(f"视频无法读取帧或 fps 非法：{video_path}")
    grouped_indices = frame_indices_for_windows(timestamps, fps, frame_count, frames_per_window)
    frame_cache: dict[int, Image.Image] = {}
    for index in sorted({idx for group in grouped_indices for idx in group}):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise ValueError(f"OpenCV 读取视频帧失败：{video_path}#{index}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_cache[index] = Image.fromarray(frame_rgb)
    capture.release()
    return [[frame_cache[idx] for idx in group] for group in grouped_indices]


def load_video_frames(video_path: Path, timestamps: torch.Tensor, frames_per_window: int, decoder: str) -> list[list[Image.Image]]:
    if decoder == "opencv":
        return load_video_frames_with_opencv(video_path, timestamps, frames_per_window)
    if decoder == "decord":
        return load_video_frames_with_decord(video_path, timestamps, frames_per_window)
    raise ValueError(f"未知视频解码器：{decoder}")


@torch.no_grad()
def encode_windows(
    grouped_frames: list[list[Image.Image]],
    processor: CLIPImageProcessor,
    model: CLIPVisionModel,
    device: torch.device,
    select_layer: int,
    batch_size: int,
) -> torch.Tensor:
    flat_frames = [frame for group in grouped_frames for frame in group]
    window_sizes = [len(group) for group in grouped_frames]
    frame_embeddings = []
    for start in range(0, len(flat_frames), batch_size):
        batch = flat_frames[start : start + batch_size]
        pixel_values = processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
        outputs = model(pixel_values, output_hidden_states=True)
        patch_tokens = outputs.hidden_states[select_layer][:, 1:, :].float()
        frame_embeddings.append(patch_tokens.mean(dim=1).cpu())
    values = torch.cat(frame_embeddings, dim=0)
    windows = []
    cursor = 0
    for size in window_sizes:
        windows.append(values[cursor : cursor + size].mean(dim=0))
        cursor += size
    embeddings = torch.stack(windows, dim=0)
    if not torch.isfinite(embeddings).all():
        raise ValueError("CLIP 视觉特征包含 NaN/Inf")
    return embeddings


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[ClipFeatureRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def save_video_feature(
    row: dict[str, Any],
    timestamps: torch.Tensor,
    embeddings: torch.Tensor,
    output_root: Path,
    args: argparse.Namespace,
) -> Path:
    youtube_id = str(row["youtube_id"])
    out_dir = output_root / "video_window_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{youtube_id}.pt"
    metadata = {
        "dataset": "AVE_HF",
        "feature_kind": "m4_clip_vision_tower_patch_mean_frame_mean",
        "vision_tower": str(Path(args.vision_tower).resolve()),
        "select_layer": int(args.select_layer),
        "frames_per_window": int(args.frames_per_window),
        "decoder": str(args.decoder),
        "source_video_path": row["video_path"],
        "embedding_dim": int(embeddings.shape[-1]),
        "window_count": int(embeddings.shape[0]),
        "label": row.get("label"),
        "start_seconds": row.get("start_seconds"),
    }
    torch.save({"sample_id": youtube_id, "video_features": embeddings.cpu(), "timestamps": timestamps.cpu(), "metadata": metadata}, out_path)
    payload = torch.load(out_path, map_location="cpu", weights_only=True)
    if payload["video_features"].shape != embeddings.shape or payload["timestamps"].shape != timestamps.shape:
        raise ValueError(f"保存后重载形状不一致：{out_path}")
    return out_path


def load_existing_video_feature(
    row: dict[str, Any],
    timestamps: torch.Tensor,
    output_root: Path,
) -> tuple[Path, torch.Tensor] | None:
    out_path = output_root / "video_window_features" / f"{row['youtube_id']}.pt"
    if not out_path.is_file():
        return None
    payload = torch.load(out_path, map_location="cpu", weights_only=True)
    embeddings = payload.get("video_features")
    saved_timestamps = payload.get("timestamps")
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError(f"已有 CLIP 特征形状非法：{out_path}")
    if not isinstance(saved_timestamps, torch.Tensor) or saved_timestamps.shape != timestamps.shape:
        raise ValueError(f"已有 CLIP 特征时间戳形状不一致：{out_path}")
    if not torch.allclose(saved_timestamps.float(), timestamps.float(), atol=1e-5, rtol=0.0):
        raise ValueError(f"已有 CLIP 特征时间戳内容不一致：{out_path}")
    if embeddings.shape[0] != timestamps.shape[0]:
        raise ValueError(f"已有 CLIP 特征窗口数不一致：{out_path}")
    if not torch.isfinite(embeddings).all():
        raise ValueError(f"已有 CLIP 特征包含 NaN/Inf：{out_path}")
    return out_path, embeddings.float()


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# AVE_HF M4 CLIP 视觉塔窗口特征报告",
        "",
        "本轮只使用本地 M4 CLIP 视觉塔离线提取窗口级视频语义特征，不加载语言模型，不修改 Gate、动态窗口或正式推理路径。",
        "",
        f"- 样本数：{summary['sample_count']}",
        f"- 视觉塔：`{summary['vision_tower']}`",
        f"- select layer：{summary['select_layer']}",
        f"- 每窗口帧数：{summary['frames_per_window']}",
        f"- 视频解码器：`{summary['decoder']}`",
        f"- 特征维度：{summary['video_embedding_dims']}",
        f"- 窗口数量分布：`{summary['window_count_counts']}`",
        f"- label 分布：`{summary['label_counts']}`",
        f"- 无 NaN/Inf：{summary['finite_check_passed']}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    all_rows = select_rows_by_label(
        load_jsonl(Path(args.feature_manifest).resolve(), limit=args.limit),
        args.samples_per_label,
    )
    rows = (
        all_rows
        if args.finalize_existing
        else slice_rows(all_rows, args.start_index, args.end_index)
    )
    output_root = Path(args.output_root).resolve()
    device = torch.device(args.device)
    vision_tower = Path(args.vision_tower).resolve()
    processor = None
    model = None
    if not args.finalize_existing:
        processor = CLIPImageProcessor.from_pretrained(vision_tower, local_files_only=True)
        model = load_local_clip_vision_model(vision_tower)
        model = model.cuda() if device.type == "cuda" else model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    records: list[ClipFeatureRecord] = []
    resumed_count = 0
    for row in rows:
        timestamps, audio_dim = load_audio_timestamps(Path(row["audio_feature_path"]))
        existing = (
            load_existing_video_feature(row, timestamps, output_root)
            if args.resume or args.finalize_existing
            else None
        )
        if existing is None:
            if args.finalize_existing:
                raise FileNotFoundError(f"finalize 时缺少 CLIP 特征：{row['youtube_id']}")
            grouped_frames = load_video_frames(Path(row["video_path"]), timestamps, args.frames_per_window, args.decoder)
            embeddings = encode_windows(grouped_frames, processor, model, device, args.select_layer, args.batch_size)
            video_feature_path = save_video_feature(row, timestamps, embeddings, output_root, args)
        else:
            video_feature_path, embeddings = existing
            resumed_count += 1
        records.append(
            ClipFeatureRecord(
                dataset="AVE_HF",
                sample_id=str(row["youtube_id"]),
                youtube_id=str(row["youtube_id"]),
                split=str(row.get("split", "train")),
                label=row.get("label"),
                start_seconds=row.get("start_seconds"),
                audio_feature_path=row["audio_feature_path"],
                video_feature_path=str(video_feature_path),
                source_video_path=row["video_path"],
                window_count=int(embeddings.shape[0]),
                window_start=float(timestamps[0, 0].item()),
                window_end=float(timestamps[-1, 1].item()),
                audio_embedding_dim=audio_dim,
                video_embedding_dim=int(embeddings.shape[-1]),
            )
        )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(records),
        "source_sample_count": len(all_rows),
        "start_index": None if args.finalize_existing else args.start_index,
        "end_index": None if args.finalize_existing else args.end_index,
        "finalize_existing": bool(args.finalize_existing),
        "resumed_count": resumed_count,
        "newly_extracted_count": len(records) - resumed_count,
        "vision_tower": str(vision_tower),
        "select_layer": int(args.select_layer),
        "frames_per_window": int(args.frames_per_window),
        "decoder": str(args.decoder),
        "video_embedding_dims": sorted({record.video_embedding_dim for record in records}),
        "audio_embedding_dims": sorted({record.audio_embedding_dim for record in records}),
        "window_count_counts": dict(Counter(str(record.window_count) for record in records)),
        "label_counts": dict(Counter(str(record.label) for record in records)),
        "finite_check_passed": True,
        "gate_or_dynamic_window_modified": False,
        "paths": {
            "feature_manifest": str(output_root / "ave_hf_clip_window_feature_manifest.jsonl"),
            "summary": str(output_root / "ave_hf_clip_window_feature_summary.json"),
            "report": str(output_root / "ave_hf_clip_window_feature_report.md"),
        },
    }
    write_jsonl(output_root / "ave_hf_clip_window_feature_manifest.jsonl", records)
    write_json(output_root / "ave_hf_clip_window_feature_summary.json", summary)
    write_report(output_root / "ave_hf_clip_window_feature_report.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", default=str(DEFAULT_FEATURE_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--vision-tower", default=str(DEFAULT_VISION_TOWER))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples-per-label", type=int, default=None)
    parser.add_argument("--select-layer", type=int, default=-2)
    parser.add_argument("--frames-per-window", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--decoder", choices=["opencv", "decord"], default="opencv")
    parser.add_argument("--resume", action="store_true", help="复用输出目录中已存在且通过校验的 CLIP 特征")
    parser.add_argument("--start-index", type=int, default=None, help="可选分片起始索引（含）")
    parser.add_argument("--end-index", type=int, default=None, help="可选分片结束索引（不含）")
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="不加载 CLIP 模型，只校验已有特征并生成完整 manifest/summary",
    )
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps({"ok": True, "summary": summary["paths"]["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
