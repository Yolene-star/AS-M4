#!/usr/bin/env python
"""为 AVE_HF 可用样本提取窗口级音频/视频特征。

音频使用本地冻结 BEATs checkpoint，视频使用当前 projector smoke 中已有的
冻结离线视频窗口特征路径。脚本只生成 precomputed 特征和校验报告，不修改
Gate、动态窗口、视频关注权重或 M4 主体，也不训练 projector。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"
DEFAULT_INPUT = INTERSUIT_ROOT / "datasets/AVE_HF_300/ave_hf_pilot_valid.jsonl"
DEFAULT_OUTPUT_ROOT = INTERSUIT_ROOT / "harness/artifacts/ave_hf_300_window_features"
DEFAULT_BEATS_CHECKPOINT = INTERSUIT_ROOT / "checkpoints/BEATs_iter3_plus_AS2M.pt"
DEFAULT_BEATS_CODE_ROOT = REPO_ROOT / "third_party/OmniMMI/baselines/videollama2/model"


def import_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


avut_beats = import_script_module("prepare_avut_beats_features_for_ave_hf", INTERSUIT_ROOT / "scripts/prepare_avut_beats_features.py")
avut_projector = None


def get_avut_projector_module():
    global avut_projector
    if avut_projector is None:
        avut_projector = import_script_module(
            "train_avut_audio_video_projectors_for_ave_hf",
            INTERSUIT_ROOT / "scripts/train_avut_audio_video_projectors.py",
        )
    return avut_projector


@dataclass(frozen=True)
class FeatureRecord:
    dataset: str
    sample_id: str
    youtube_id: str
    split: str
    label: str | None
    start_seconds: float | None
    audio_feature_path: str
    video_feature_path: str
    audio_path: str
    video_path: str
    window_count: int
    window_start: float
    window_end: float
    audio_embedding_dim: int
    video_embedding_dim: int


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"输入 manifest 为空：{path}")
    return rows


def validate_payload(path: Path, key: str) -> tuple[int, int, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    values = payload[key].float()
    timestamps = payload["timestamps"].float()
    if values.ndim != 2:
        raise ValueError(f"{path} {key} 必须是 [T,D]，实际 {tuple(values.shape)}")
    if timestamps.shape != (values.shape[0], 2):
        raise ValueError(f"{path} 时间戳与窗口数不一致")
    if not torch.isfinite(values).all() or not torch.isfinite(timestamps).all():
        raise ValueError(f"{path} 包含 NaN/Inf")
    return int(values.shape[0]), int(values.shape[1]), timestamps


def load_wav_mono(path: Path, target_sample_rate: int = 16000) -> tuple[torch.Tensor, int]:
    sample_rate, data = wavfile.read(path)
    values = torch.as_tensor(data)
    if values.ndim == 2:
        values = values.float().mean(dim=1)
    else:
        values = values.float()
    if values.numel() == 0:
        raise ValueError(f"音频为空：{path}")
    if data.dtype.kind in {"i", "u"}:
        info = np.iinfo(data.dtype)
        scale = float(max(abs(info.min), abs(info.max)))
        values = values / scale
    elif values.abs().max().item() > 1.0:
        values = values / values.abs().max().clamp_min(1.0)
    if sample_rate != target_sample_rate:
        gcd = math.gcd(int(sample_rate), int(target_sample_rate))
        resampled = resample_poly(values.numpy(), target_sample_rate // gcd, sample_rate // gcd)
        values = torch.from_numpy(resampled).float()
        sample_rate = target_sample_rate
    if not torch.isfinite(values).all():
        raise ValueError(f"音频包含 NaN/Inf：{path}")
    return values.contiguous(), int(sample_rate)


def save_audio_feature(
    row: dict[str, Any],
    encoder,
    output_root: Path,
    window_sec: float,
    hop_sec: float,
) -> tuple[Path, torch.Tensor, int]:
    youtube_id = str(row["youtube_id"])
    audio_path = Path(row["audio_path"])
    waveform, sample_rate = load_wav_mono(audio_path, target_sample_rate=16000)
    windows, timestamps = avut_beats.window_condition(waveform, sample_rate, window_sec, hop_sec)
    embeddings = encoder.encode_windows(windows)
    if embeddings.shape[0] != timestamps.shape[0]:
        raise ValueError(f"{youtube_id} BEATs 输出窗口数不一致")
    out_dir = output_root / "precomputed_audio_features" / youtube_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "original.pt"
    metadata = {
        "dataset": "AVE_HF",
        "sample_id": youtube_id,
        "condition": "original",
        "source_sample_id": youtube_id,
        "source_audio_path": str(audio_path),
        "encoder_name": encoder.encoder_name,
        "checkpoint_name": encoder.checkpoint_name,
        "checkpoint_sha256": encoder.checkpoint_sha256,
        "sample_rate": int(sample_rate),
        "window_sec": float(window_sec),
        "hop_sec": float(hop_sec),
        "window_count": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "label": row.get("label"),
        "start_seconds": row.get("start_seconds"),
    }
    torch.save(
        {
            "sample_id": youtube_id,
            "condition": "original",
            "timestamps": timestamps.cpu(),
            "audio_embedding": embeddings.cpu(),
            "metadata": metadata,
        },
        out_path,
    )
    window_count, _, _ = validate_payload(out_path, "audio_embedding")
    return out_path, timestamps, window_count


def find_existing_audio_feature(
    row: dict[str, Any],
    output_root: Path,
) -> tuple[Path, torch.Tensor, int] | None:
    youtube_id = str(row["youtube_id"])
    path = output_root / "precomputed_audio_features" / youtube_id / "original.pt"
    if not path.is_file():
        return None
    window_count, _, timestamps = validate_payload(path, "audio_embedding")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if str(payload.get("sample_id")) != youtube_id:
        raise ValueError(f"已有音频特征 sample_id 不匹配：{path}")
    return path, timestamps, window_count


def load_existing_audio_feature(
    row: dict[str, Any],
    output_root: Path,
) -> tuple[Path, torch.Tensor, int]:
    existing = find_existing_audio_feature(row, output_root)
    if existing is None:
        youtube_id = str(row["youtube_id"])
        path = output_root / "precomputed_audio_features" / youtube_id / "original.pt"
        raise FileNotFoundError(f"要求复用音频特征，但文件不存在：{path}")
    return existing


def save_video_feature(
    row: dict[str, Any],
    timestamps: torch.Tensor,
    output_root: Path,
    target_dim: int,
) -> Path:
    projector = get_avut_projector_module()
    youtube_id = str(row["youtube_id"])
    video_path = Path(row["video_path"])
    features = projector.extract_video_window_features(video_path, timestamps, target_dim=target_dim)
    if features.shape[0] != timestamps.shape[0]:
        raise ValueError(f"{youtube_id} 视频窗口数不一致")
    out_dir = output_root / "video_window_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{youtube_id}.pt"
    torch.save(
        {
            "sample_id": youtube_id,
            "video_features": features.cpu(),
            "timestamps": timestamps.cpu(),
            "metadata": {
                "dataset": "AVE_HF",
                "feature_kind": "frozen_rgb_frame_statistics_expanded",
                "source_video_path": str(video_path),
                "embedding_dim": int(features.shape[-1]),
                "window_count": int(features.shape[0]),
                "label": row.get("label"),
                "start_seconds": row.get("start_seconds"),
            },
        },
        out_path,
    )
    validate_payload(out_path, "video_features")
    return out_path


def load_existing_video_feature(
    row: dict[str, Any],
    timestamps: torch.Tensor,
    output_root: Path,
) -> tuple[Path, int] | None:
    youtube_id = str(row["youtube_id"])
    path = output_root / "video_window_features" / f"{youtube_id}.pt"
    if not path.is_file():
        return None
    window_count, video_dim, video_ts = validate_payload(path, "video_features")
    if window_count != timestamps.shape[0]:
        raise ValueError(f"已有 RGB 特征窗口数不一致：{path}")
    get_avut_projector_module().validate_timestamp_match(timestamps, video_ts)
    return path, video_dim


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[FeatureRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# AVE_HF 300 条窗口特征提取报告",
        "",
        "本轮只生成离线窗口特征，不修改 Gate、动态窗口、视频关注权重或 M4 主体。",
        "",
        f"- 输入 manifest：`{summary['input_manifest']}`",
        f"- 样本数：{summary['sample_count']}",
        f"- 音频编码器：{summary['audio_encoder']}",
        f"- 音频特征维度：{summary['audio_embedding_dims']}",
        f"- 视频特征类型：`{summary['video_feature_kind']}`",
        f"- 视频特征维度：{summary['video_embedding_dims']}",
        f"- 窗口：{summary['window_sec']} 秒，hop={summary['hop_sec']} 秒",
        f"- label 分布：`{summary['label_counts']}`",
        f"- 窗口数量分布：`{summary['window_count_counts']}`",
        f"- 无 NaN/Inf：{summary['finite_check_passed']}",
        f"- 时间戳一致：{summary['timestamp_check_passed']}",
        "",
        "## 说明",
        "- 音频特征来自冻结 BEATs，本轮未训练编码器。",
        "- 视频特征使用现有 projector smoke 的冻结 RGB 帧统计 expanded 路径，不冒充完整 M4 主干语义特征。",
        "- 若后续需要正式 M4 视频特征，应在同一 manifest 上替换 video feature backend 后重新提取。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_manifest = Path(args.input_manifest).resolve()
    output_root = Path(args.output_root).resolve()
    rows = load_jsonl(input_manifest, limit=args.limit)
    encoder = None
    if not args.reuse_existing_audio_features:
        encoder = avut_beats.BEATsWindowEncoder(
            checkpoint_path=Path(args.beats_checkpoint).resolve(),
            beats_code_root=Path(args.beats_code_root).resolve(),
            device=args.device,
        )

    records: list[FeatureRecord] = []
    resumed_audio_count = 0
    resumed_rgb_count = 0
    video_feature_kind = "skipped_rgb_frame_statistics_audio_manifest_only" if args.skip_rgb_video_features else "frozen_rgb_frame_statistics_expanded"
    video_dims_for_summary = [0] if args.skip_rgb_video_features else None
    timestamp_check_passed = True
    for row in rows:
        if args.reuse_existing_audio_features:
            audio_feature_path, timestamps, window_count = load_existing_audio_feature(row, output_root)
            resumed_audio_count += 1
        elif args.resume_audio_features:
            existing_audio = find_existing_audio_feature(row, output_root)
            if existing_audio is None:
                audio_feature_path, timestamps, window_count = save_audio_feature(
                    row, encoder, output_root, args.window_sec, args.hop_sec
                )
            else:
                audio_feature_path, timestamps, window_count = existing_audio
                resumed_audio_count += 1
        else:
            audio_feature_path, timestamps, window_count = save_audio_feature(
                row, encoder, output_root, args.window_sec, args.hop_sec
            )
        audio_windows, audio_dim, audio_ts = validate_payload(audio_feature_path, "audio_embedding")
        if args.skip_rgb_video_features:
            video_feature_path_text = ""
            video_dim_for_record = 0
        else:
            existing_video = (
                load_existing_video_feature(row, timestamps, output_root)
                if args.resume_rgb_video_features
                else None
            )
            if existing_video is None:
                video_feature_path = save_video_feature(row, timestamps, output_root, args.video_dim)
                video_windows, video_dim, video_ts = validate_payload(video_feature_path, "video_features")
            else:
                video_feature_path, video_dim = existing_video
                video_windows, _, video_ts = validate_payload(video_feature_path, "video_features")
                resumed_rgb_count += 1
            if audio_windows != video_windows:
                raise ValueError(f"{row['youtube_id']} 音视频窗口数不一致")
            get_avut_projector_module().validate_timestamp_match(audio_ts, video_ts)
            video_feature_path_text = str(video_feature_path)
            video_dim_for_record = video_dim
        records.append(
            FeatureRecord(
                dataset="AVE_HF",
                sample_id=str(row["youtube_id"]),
                youtube_id=str(row["youtube_id"]),
                split=str(row.get("split", "train")),
                label=row.get("label"),
                start_seconds=row.get("start_seconds"),
                audio_feature_path=str(audio_feature_path),
                video_feature_path=video_feature_path_text,
                audio_path=str(row["audio_path"]),
                video_path=str(row["video_path"]),
                window_count=window_count,
                window_start=float(audio_ts[0, 0].item()),
                window_end=float(audio_ts[-1, 1].item()),
                audio_embedding_dim=audio_dim,
                video_embedding_dim=video_dim_for_record,
            )
        )

    audio_dims = sorted({record.audio_embedding_dim for record in records})
    video_dims = video_dims_for_summary or sorted({record.video_embedding_dim for record in records})
    window_counts = Counter(record.window_count for record in records)
    label_counts = Counter(record.label for record in records)
    first_audio = torch.load(records[0].audio_feature_path, map_location="cpu", weights_only=True)
    audio_metadata = first_audio.get("metadata", {})
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(input_manifest),
        "output_root": str(output_root),
        "sample_count": len(records),
        "audio_encoder": encoder.encoder_name if encoder is not None else audio_metadata.get("encoder_name"),
        "checkpoint_name": encoder.checkpoint_name if encoder is not None else audio_metadata.get("checkpoint_name"),
        "checkpoint_sha256": encoder.checkpoint_sha256 if encoder is not None else audio_metadata.get("checkpoint_sha256"),
        "reused_audio_feature_count": resumed_audio_count,
        "new_audio_feature_count": len(records) - resumed_audio_count,
        "resumed_rgb_feature_count": resumed_rgb_count,
        "new_rgb_feature_count": 0 if args.skip_rgb_video_features else len(records) - resumed_rgb_count,
        "audio_embedding_dims": audio_dims,
        "video_feature_kind": video_feature_kind,
        "video_embedding_dims": video_dims,
        "window_sec": float(args.window_sec),
        "hop_sec": float(args.hop_sec),
        "label_counts": dict(label_counts),
        "window_count_counts": {str(key): value for key, value in sorted(window_counts.items())},
        "finite_check_passed": True,
        "timestamp_check_passed": timestamp_check_passed,
        "gate_or_dynamic_window_modified": False,
        "paths": {
            "feature_manifest": str(output_root / "ave_hf_window_feature_manifest.jsonl"),
            "summary": str(output_root / "ave_hf_window_feature_summary.json"),
            "report": str(output_root / "ave_hf_window_feature_report.md"),
        },
    }
    write_jsonl(output_root / "ave_hf_window_feature_manifest.jsonl", records)
    write_json(output_root / "ave_hf_window_feature_summary.json", summary)
    write_report(output_root / "ave_hf_window_feature_report.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--beats-checkpoint", default=str(DEFAULT_BEATS_CHECKPOINT))
    parser.add_argument("--beats-code-root", default=str(DEFAULT_BEATS_CODE_ROOT))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--hop-sec", type=float, default=0.5)
    parser.add_argument("--video-dim", type=int, default=768)
    parser.add_argument(
        "--reuse-existing-audio-features",
        action="store_true",
        help="复用并校验输出目录中已有的 BEATs 音频特征，不重新加载或运行 BEATs。",
    )
    parser.add_argument(
        "--resume-audio-features",
        action="store_true",
        help="复用已存在且通过校验的 BEATs 特征，仅计算缺失样本。",
    )
    parser.add_argument(
        "--resume-rgb-video-features",
        action="store_true",
        help="复用输出目录中已存在且通过时间戳校验的 RGB 窗口特征。",
    )
    parser.add_argument(
        "--skip-rgb-video-features",
        action="store_true",
        help="只提取 BEATs 音频窗口特征并保留 manifest，视频语义特征由独立 CLIP 脚本生成。",
    )
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps({"ok": True, "summary": summary["paths"]["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
