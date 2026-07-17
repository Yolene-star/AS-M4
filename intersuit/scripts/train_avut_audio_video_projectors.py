#!/usr/bin/env python
"""训练 AVUT 5 条样本的轻量 audio/video projector 烟测入口。

本脚本只消费离线 BEATs 音频窗口特征，并从视频中提取冻结的轻量帧统计
窗口特征，用于验证 projector 训练、checkpoint 保存/重载和冻结报告。
它不会加载或修改 M4、Gate、BEATs 权重，也不会接入正式融合路径。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import decord
import torch
from torch import nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"
DEFAULT_MANIFEST = INTERSUIT_ROOT / "inputs/texts/avut/avut_audio_smoke.json"
DEFAULT_AUDIO_FEATURE_ROOT = INTERSUIT_ROOT / "harness/artifacts/avut_beats_features_real/precomputed_audio_features"
DEFAULT_OUTPUT_ROOT = INTERSUIT_ROOT / "harness/artifacts/avut_projector_training"
CONDITIONS = ("original", "silence", "wrong_audio", "shift_plus_0_5", "shift_minus_0_5")
NEGATIVE_CONDITIONS = ("silence", "wrong_audio", "shift_plus_0_5", "shift_minus_0_5")


@dataclass(frozen=True)
class PairRecord:
    sample_id: str
    pair_type: str
    audio_condition: str
    label: int
    audio_feature_path: str
    video_feature_path: str
    window_count: int
    embedding_dim: int
    timestamp_start: float
    timestamp_end: float
    wrong_source_sample_id: str | None = None


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidates = (
        (Path.cwd() / path).resolve(),
        (REPO_ROOT / path).resolve(),
        (INTERSUIT_ROOT / path).resolve(),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise TypeError("AVUT manifest must be a JSON list")
    rows = data[:limit] if limit is not None else data
    if not rows:
        raise ValueError("AVUT manifest is empty")
    return rows


def load_feature_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"sample_id", "condition", "timestamps", "audio_embedding", "metadata"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"feature file {path} is missing keys: {sorted(missing)}")
    embeddings = payload["audio_embedding"].float()
    timestamps = payload["timestamps"].float()
    if embeddings.ndim != 2:
        raise ValueError(f"audio_embedding must be [T,D], got {tuple(embeddings.shape)}")
    if timestamps.shape != (embeddings.shape[0], 2):
        raise ValueError(f"timestamp/window mismatch in {path}")
    if not torch.isfinite(embeddings).all() or not torch.isfinite(timestamps).all():
        raise ValueError(f"NaN/Inf in feature file {path}")
    return payload


def expand_to_dim(features: torch.Tensor, target_dim: int) -> torch.Tensor:
    if features.shape[-1] == target_dim:
        return features
    if features.shape[-1] > target_dim:
        return features[..., :target_dim]
    repeat = math.ceil(target_dim / features.shape[-1])
    return features.repeat(1, repeat)[..., :target_dim]


def extract_video_window_features(
    video_path: Path,
    timestamps: torch.Tensor,
    target_dim: int = 768,
) -> torch.Tensor:
    """Extract frozen per-window RGB statistics from nearest video frames."""

    if not video_path.is_file():
        raise FileNotFoundError(f"video source not found: {video_path}")
    decord.bridge.set_bridge("torch")
    reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
    frame_count = len(reader)
    if frame_count <= 0:
        raise ValueError(f"video has no frames: {video_path}")
    fps = float(reader.get_avg_fps() or 0.0)
    if fps <= 0:
        raise ValueError(f"video fps is invalid: {video_path}")

    centers = timestamps.float().mean(dim=-1)
    indices = torch.clamp((centers * fps).round().long(), 0, frame_count - 1)
    frames = reader.get_batch(indices.tolist()).float() / 255.0
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"decoded frames must be [T,H,W,3], got {tuple(frames.shape)}")
    flat = frames.reshape(frames.shape[0], -1, 3)
    mean = flat.mean(dim=1)
    std = flat.std(dim=1, unbiased=False)
    min_value = flat.amin(dim=1)
    max_value = flat.amax(dim=1)
    center_pixel = frames[:, frames.shape[1] // 2, frames.shape[2] // 2, :]
    base = torch.cat([mean, std, min_value, max_value, center_pixel], dim=-1)
    if not torch.isfinite(base).all():
        raise ValueError(f"video feature contains NaN/Inf: {video_path}")
    return expand_to_dim(base, target_dim).contiguous()


def save_video_feature(
    output_root: Path,
    sample: dict[str, Any],
    timestamps: torch.Tensor,
    target_dim: int,
) -> Path:
    sample_id = str(sample.get("id"))
    video_path = resolve_repo_path(str(sample.get("video_path")))
    features = extract_video_window_features(video_path, timestamps, target_dim=target_dim)
    if features.shape[0] != timestamps.shape[0]:
        raise ValueError("video feature/window count mismatch")
    out_dir = output_root / "video_window_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{sample_id}.pt"
    torch.save(
        {
            "sample_id": sample_id,
            "video_features": features,
            "timestamps": timestamps.float(),
            "metadata": {
                "feature_kind": "frozen_rgb_frame_statistics_expanded",
                "source_video_path": str(video_path),
                "embedding_dim": int(features.shape[-1]),
                "window_count": int(features.shape[0]),
            },
        },
        path,
    )
    return path


def validate_timestamp_match(audio_ts: torch.Tensor, video_ts: torch.Tensor, atol: float = 1e-5) -> None:
    if audio_ts.shape != video_ts.shape:
        raise ValueError(f"audio/video timestamps shape mismatch: {tuple(audio_ts.shape)} vs {tuple(video_ts.shape)}")
    if not torch.allclose(audio_ts.float(), video_ts.float(), atol=atol, rtol=0.0):
        diff = (audio_ts.float() - video_ts.float()).abs().max().item()
        raise ValueError(f"audio/video timestamps are not aligned; max diff={diff}")


def build_pair_manifest(
    samples: list[dict[str, Any]],
    audio_feature_root: Path,
    output_root: Path,
    target_dim: int,
) -> tuple[list[PairRecord], dict[str, Path]]:
    records: list[PairRecord] = []
    video_paths: dict[str, Path] = {}
    for sample in samples:
        sample_id = str(sample.get("id"))
        original_audio_path = audio_feature_root / sample_id / "original.pt"
        original_payload = load_feature_payload(original_audio_path)
        timestamps = original_payload["timestamps"].float()
        video_feature_path = save_video_feature(output_root, sample, timestamps, target_dim)
        video_payload = torch.load(video_feature_path, map_location="cpu", weights_only=True)
        validate_timestamp_match(timestamps, video_payload["timestamps"].float())
        video_paths[sample_id] = video_feature_path
        for condition in CONDITIONS:
            audio_path = audio_feature_root / sample_id / f"{condition}.pt"
            audio_payload = load_feature_payload(audio_path)
            validate_timestamp_match(audio_payload["timestamps"].float(), video_payload["timestamps"].float())
            label = 1 if condition == "original" else 0
            pair_type = "positive" if label else condition
            metadata = audio_payload["metadata"]
            records.append(
                PairRecord(
                    sample_id=sample_id,
                    pair_type=pair_type,
                    audio_condition=condition,
                    label=label,
                    audio_feature_path=str(audio_path),
                    video_feature_path=str(video_feature_path),
                    window_count=int(audio_payload["audio_embedding"].shape[0]),
                    embedding_dim=int(audio_payload["audio_embedding"].shape[1]),
                    timestamp_start=float(audio_payload["timestamps"][0, 0].item()),
                    timestamp_end=float(audio_payload["timestamps"][-1, 1].item()),
                    wrong_source_sample_id=metadata.get("source_sample_id") if condition == "wrong_audio" else None,
                )
            )
    return records, video_paths


def write_pair_manifest(records: list[PairRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")


def read_pair_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"pair manifest is empty: {path}")
    return rows


class AVProjector(nn.Module):
    def __init__(self, input_dim: int = 768, project_dim: int = 128) -> None:
        super().__init__()
        self.audio_proj = nn.Linear(input_dim, project_dim, bias=False)
        self.video_proj = nn.Linear(input_dim, project_dim, bias=False)

    def forward(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        audio_z = F.normalize(self.audio_proj(audio.float()), dim=-1, eps=1e-6)
        video_z = F.normalize(self.video_proj(video.float()), dim=-1, eps=1e-6)
        return (audio_z * video_z).sum(dim=-1)


def load_pair_tensors(record: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    audio_payload = load_feature_payload(Path(record["audio_feature_path"]))
    video_payload = torch.load(record["video_feature_path"], map_location="cpu", weights_only=True)
    audio = audio_payload["audio_embedding"].float()
    video = video_payload["video_features"].float()
    validate_timestamp_match(audio_payload["timestamps"].float(), video_payload["timestamps"].float())
    if audio.shape != video.shape:
        raise ValueError(f"audio/video feature shape mismatch for {record['sample_id']}/{record['audio_condition']}")
    labels = torch.full((audio.shape[0],), float(record["label"]), dtype=torch.float32)
    return audio, video, labels


def collect_training_tensors(records: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    audio_rows, video_rows, label_rows = [], [], []
    for record in records:
        audio, video, labels = load_pair_tensors(record)
        audio_rows.append(audio)
        video_rows.append(video)
        label_rows.append(labels)
    return torch.cat(audio_rows), torch.cat(video_rows), torch.cat(label_rows)


def train_projectors(
    pair_manifest: Path,
    output_root: Path,
    steps: int,
    project_dim: int = 128,
    lr: float = 1e-3,
    seed: int = 20260717,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    rows = read_pair_manifest(pair_manifest)
    audio, video, labels = collect_training_tensors(rows)
    model = AVProjector(input_dim=audio.shape[-1], project_dim=project_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    trainable = {name: int(param.numel()) for name, param in model.named_parameters() if param.requires_grad}
    frozen_report = {
        "BEATs": "frozen_precomputed_audio_features_only",
        "M4": "not_loaded_frozen_no_grad",
        "Gate": "not_loaded_frozen_no_grad",
        "trainable_parameter_names": sorted(trainable),
        "trainable_parameter_count": sum(trainable.values()),
    }
    history = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(audio, video) / 0.07
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pos = logits[labels > 0.5]
            neg = logits[labels < 0.5]
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "positive_logit_mean": float(pos.mean().item()) if pos.numel() else None,
                    "negative_logit_mean": float(neg.mean().item()) if neg.numel() else None,
                }
            )

    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / f"projector_checkpoint_{steps}step.pt"
    torch.save(
        {
            "audio_proj.weight": model.audio_proj.weight.detach().cpu(),
            "video_proj.weight": model.video_proj.weight.detach().cpu(),
            "metadata": {
                "input_dim": int(audio.shape[-1]),
                "project_dim": int(project_dim),
                "steps": int(steps),
                "lr": float(lr),
                "seed": int(seed),
            },
        },
        checkpoint_path,
    )
    reloaded = AVProjector(input_dim=audio.shape[-1], project_dim=project_dim)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reloaded.audio_proj.weight.data.copy_(state["audio_proj.weight"])
    reloaded.video_proj.weight.data.copy_(state["video_proj.weight"])
    with torch.no_grad():
        reload_max_diff = float((model(audio, video) - reloaded(audio, video)).abs().max().item())
    report = {
        "status": "complete",
        "steps": int(steps),
        "pair_manifest": str(pair_manifest),
        "checkpoint_path": str(checkpoint_path),
        "history": history,
        "reload_max_abs_logit_diff": reload_max_diff,
        "checkpoint_reload_passed": reload_max_diff <= 1e-6,
        "frozen_components": frozen_report,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_root / f"training_report_{steps}step.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_training_markdown(output_root / f"training_report_{steps}step.md", report)
    return report


def write_training_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# AVUT audio/video projector 训练与冻结报告",
        "",
        f"- 状态：{report['status']}",
        f"- 训练步数：{report['steps']}",
        f"- checkpoint：{report['checkpoint_path']}",
        f"- reload 通过：{report['checkpoint_reload_passed']}",
        f"- reload 最大差异：{report['reload_max_abs_logit_diff']}",
        "",
        "## 冻结范围",
        "",
    ]
    frozen = report["frozen_components"]
    for name in ("BEATs", "M4", "Gate"):
        lines.append(f"- {name}: {frozen[name]}")
    lines.extend(
        [
            f"- 可训练参数：{', '.join(frozen['trainable_parameter_names'])}",
            f"- 可训练参数量：{frozen['trainable_parameter_count']}",
            "",
            "## Loss 历史",
            "",
        ]
    )
    for item in report["history"]:
        lines.append(
            f"- step {item['step']}: loss={item['loss']:.6f}, "
            f"pos={item['positive_logit_mean']:.6f}, neg={item['negative_logit_mean']:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest_summary(records: list[PairRecord], output_root: Path) -> dict[str, Any]:
    summary = {
        "sample_count": len({record.sample_id for record in records}),
        "record_count": len(records),
        "positive_count": sum(record.label == 1 for record in records),
        "negative_count": sum(record.label == 0 for record in records),
        "conditions": sorted({record.audio_condition for record in records}),
        "embedding_dims": sorted({record.embedding_dim for record in records}),
        "window_counts": {record.sample_id: record.window_count for record in records if record.audio_condition == "original"},
        "timestamp_alignment": "passed",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "pair_manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_root / "pair_manifest_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(PairRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)
    return summary


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    samples = load_manifest(resolve_repo_path(args.manifest), limit=args.limit)
    output_root = resolve_repo_path(args.output_root)
    records, video_paths = build_pair_manifest(
        samples,
        resolve_repo_path(args.audio_feature_root),
        output_root,
        target_dim=args.input_dim,
    )
    manifest_path = output_root / "projector_pair_manifest.jsonl"
    write_pair_manifest(records, manifest_path)
    summary = write_manifest_summary(records, output_root)
    metadata = {
        "status": "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "video_feature_files": {key: str(value) for key, value in video_paths.items()},
        "audio_feature_root": str(resolve_repo_path(args.audio_feature_root)),
        "freeze_policy": {
            "BEATs": "precomputed_only",
            "M4": "not_loaded",
            "Gate": "not_loaded",
        },
    }
    (output_root / "projector_prepare_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest": str(manifest_path), "summary": summary, "metadata": metadata}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = resolve_repo_path(args.output_root)
    prepared = prepare(args)
    reports = []
    for steps in args.steps:
        reports.append(
            train_projectors(
                Path(prepared["manifest"]),
                output_root,
                steps=int(steps),
                project_dim=args.project_dim,
                lr=args.lr,
                seed=args.seed,
            )
        )
    result = {"prepared": prepared, "training_reports": reports}
    (output_root / "projector_training_run_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--audio-feature-root", default=str(DEFAULT_AUDIO_FEATURE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--input-dim", type=int, default=768)
    parser.add_argument("--project-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--steps", type=int, nargs="+", default=[2, 20])
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        result = prepare(args) if args.prepare_only else run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
