#!/usr/bin/env python
"""训练 AVE 连续序列上的轻量因果 offset GRU 诊断模块。

本脚本只读取既有 development manifest，并在其 youtube_id 内部重新划分
train/dev。它不会读取冻结 test manifest，不移动音频窗口、不接 Gate，
也不训练冻结 offset scorer、BEATs、CLIP、RGB 或 M4 主体。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile

from intersuit.model.streaming_av.audio_event_aligner import (
    FrozenOffsetScorerInputs,
    FrozenTemporalOffsetScorer,
)
from intersuit.model.streaming_av.temporal_offset_gru import (
    TemporalOffsetGRUDiagnostic,
    build_temporal_offset_evidence,
    ordered_offset_emd_loss,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"
DEFAULT_BUNDLE = INTERSUIT_ROOT / (
    "harness/artifacts/ave_hf_temporal_offset_zero125_centerpeak_expanded_frozen/"
    "seed_20260719/temporal_offset_scorer_runtime_bundle.pt"
)
DEFAULT_SOURCE = INTERSUIT_ROOT / "harness/artifacts/ave_hf_selective_1200_split/dev_manifest.jsonl"
FROZEN_TEST = INTERSUIT_ROOT / "harness/artifacts/ave_hf_selective_1200_split/test_manifest.jsonl"
DEFAULT_CLIP = INTERSUIT_ROOT / (
    "harness/artifacts/ave_hf_selective_1200_clip_window_features/"
    "ave_hf_clip_window_feature_manifest.jsonl"
)
DEFAULT_RGB = INTERSUIT_ROOT / (
    "harness/artifacts/ave_hf_selective_1200_window_features/"
    "ave_hf_window_feature_manifest.jsonl"
)
DEFAULT_ANNOTATIONS = INTERSUIT_ROOT / "datasets/AVE/data/Annotations.txt"
DEFAULT_OUTPUT = INTERSUIT_ROOT / "harness/artifacts/ave_hf_temporal_offset_gru_dev_v1"
OFFSETS = (-0.5, 0.0, 0.5)
CONDITIONS = (
    ("original", 0, 1, 1.0),
    ("shift_plus_0.5", 1, 0, 1.0),
    ("shift_minus_0.5", -1, 2, 1.0),
    ("cross_video_mismatch", 0, 1, 0.0),
)


@dataclass(frozen=True)
class SequenceMetadata:
    youtube_id: str
    partner_youtube_id: str | None
    partition: str
    condition: str
    start_window: int
    length: int
    target_index: int
    synchronizable: float


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_annotations(path: Path) -> dict[str, tuple[float, float]]:
    annotations: dict[str, tuple[float, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("&")
        if len(parts) != 5:
            continue
        try:
            start, end = float(parts[3]), float(parts[4])
        except ValueError:
            continue
        annotations[parts[1]] = (start, end)
    if not annotations:
        raise ValueError(f"AVE 标注为空：{path}")
    return annotations


def assert_development_source(path: Path) -> None:
    resolved = path.resolve()
    if resolved == FROZEN_TEST.resolve() or resolved.name == "test_manifest.jsonl":
        raise ValueError("冻结 test manifest 禁止用于 GRU 训练或开发评估")
    if resolved != DEFAULT_SOURCE.resolve():
        raise ValueError(
            "第一版只允许读取冻结测试集之外的既有 development manifest："
            f"{DEFAULT_SOURCE.resolve()}"
        )


def split_youtube_ids(
    youtube_ids: list[str],
    *,
    dev_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    if not 0.0 < dev_ratio < 1.0:
        raise ValueError("dev_ratio must be in (0,1)")
    unique = sorted(set(youtube_ids))
    random.Random(seed).shuffle(unique)
    dev_count = max(1, int(round(len(unique) * dev_ratio)))
    train_ids = sorted(unique[dev_count:])
    dev_ids = sorted(unique[:dev_count])
    if not train_ids or set(train_ids) & set(dev_ids):
        raise ValueError("youtube_id train/dev 划分无效")
    return train_ids, dev_ids


def load_feature(path: str | Path, key: str) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    values = payload[key].float()
    timestamps = payload["timestamps"].float()
    if values.ndim != 2 or timestamps.shape != (values.shape[0], 2):
        raise ValueError(f"特征形状非法：{path}")
    if not torch.isfinite(values).all() or not torch.isfinite(timestamps).all():
        raise ValueError(f"特征包含 NaN/Inf：{path}")
    return values, timestamps


def window_audio_stats(path: str | Path, timestamps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rate, data = wavfile.read(Path(path))
    if rate != 16000:
        raise ValueError(f"音频采样率必须为 16000：{path}")
    values = torch.as_tensor(data).float()
    if values.ndim == 2:
        values = values.mean(dim=1)
    if data.dtype.kind in {"i", "u"}:
        info = np.iinfo(data.dtype)
        values = values / float(max(abs(info.min), abs(info.max)))
    rms = []
    nonsilent = []
    for start, end in timestamps.tolist():
        left = max(0, int(round(start * rate)))
        right = min(values.numel(), int(round(end * rate)))
        window = values[left:right]
        rms.append(float(window.square().mean().sqrt().item()) if window.numel() else 0.0)
        nonsilent.append(float(window.abs().gt(1e-4).float().mean().item()) if window.numel() else 0.0)
    return torch.tensor(rms), torch.tensor(nonsilent)


def diff_norm(values: torch.Tensor) -> torch.Tensor:
    output = torch.zeros(values.shape[0], dtype=torch.float32)
    if values.shape[0] > 1:
        step = (values[1:].float() - values[:-1].float()).norm(dim=-1)
        output[1:] = torch.maximum(output[1:], step)
        output[:-1] = torch.maximum(output[:-1], step)
    return output


def shift_stream(values: torch.Tensor, shift: int) -> torch.Tensor:
    indices = torch.arange(values.shape[0]) + int(shift)
    return values[indices.clamp(0, values.shape[0] - 1)]


def build_evidence(
    audio: torch.Tensor,
    clip: torch.Tensor,
    rgb: torch.Tensor,
    rms: torch.Tensor,
    nonsilent: torch.Tensor,
) -> torch.Tensor:
    return build_temporal_offset_evidence(
        audio,
        clip,
        rgb,
        rms,
        nonsilent,
    )


def load_streams(
    youtube_id: str,
    clip_rows: dict[str, dict[str, Any]],
    rgb_rows: dict[str, dict[str, Any]],
) -> dict[str, torch.Tensor]:
    clip_row = clip_rows[youtube_id]
    rgb_row = rgb_rows[youtube_id]
    audio, audio_ts = load_feature(clip_row["audio_feature_path"], "audio_embedding")
    clip, clip_ts = load_feature(clip_row["video_feature_path"], "video_features")
    rgb, rgb_ts = load_feature(rgb_row["video_feature_path"], "video_features")
    if not (
        torch.allclose(audio_ts, clip_ts, atol=1e-5, rtol=0.0)
        and torch.allclose(audio_ts, rgb_ts, atol=1e-5, rtol=0.0)
    ):
        raise ValueError(f"{youtube_id} 三路时间戳不一致")
    rms, nonsilent = window_audio_stats(rgb_row["audio_path"], audio_ts)
    return {
        "audio": audio,
        "clip": clip,
        "rgb": rgb,
        "rms": rms,
        "nonsilent": nonsilent,
        "timestamps": audio_ts,
    }


@torch.no_grad()
def build_partition_sequences(
    youtube_ids: list[str],
    partition: str,
    clip_rows: dict[str, dict[str, Any]],
    rgb_rows: dict[str, dict[str, Any]],
    scorer: FrozenTemporalOffsetScorer,
    annotations: dict[str, tuple[float, float]],
    *,
    min_length: int,
    max_length: int,
    seed: int,
    scorer_batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rng = random.Random(seed + (0 if partition == "train" else 1000))
    streams = {
        youtube_id: load_streams(youtube_id, clip_rows, rgb_rows)
        for youtube_id in youtube_ids
    }
    partners = list(youtube_ids[1:]) + list(youtube_ids[:1])
    pending: list[dict[str, Any]] = []
    for youtube_id, partner_id in zip(youtube_ids, partners):
        source = streams[youtube_id]
        if youtube_id not in annotations:
            raise ValueError(f"AVE 标注缺少 youtube_id：{youtube_id}")
        event_start, event_end = annotations[youtube_id]
        for condition, shift, target, syncable in CONDITIONS:
            if condition == "cross_video_mismatch":
                partner = streams[partner_id]
                audio = partner["audio"]
                rms = partner["rms"]
                nonsilent = partner["nonsilent"]
                partner_value: str | None = partner_id
            else:
                audio = shift_stream(source["audio"], shift)
                rms = shift_stream(source["rms"], shift)
                nonsilent = shift_stream(source["nonsilent"], shift)
                partner_value = None
            steps = min(
                audio.shape[0],
                source["clip"].shape[0],
                source["rgb"].shape[0],
                rms.shape[0],
            )
            if steps < min_length:
                continue
            audio = audio[:steps]
            clip = source["clip"][:steps]
            rgb = source["rgb"][:steps]
            rms = rms[:steps]
            nonsilent = nonsilent[:steps]
            centers = source["timestamps"][:steps].mean(dim=-1)
            if syncable:
                sync_targets = centers.ge(event_start) & centers.lt(event_end)
            else:
                sync_targets = torch.zeros(steps, dtype=torch.bool)
            length = rng.randint(min_length, min(max_length, steps))
            active = sync_targets.nonzero(as_tuple=False).flatten().tolist()
            if active:
                anchor = rng.choice(active)
                start = max(0, min(steps - length, anchor - rng.randrange(length)))
            else:
                start = rng.randint(0, steps - length)
            pending.append(
                {
                    "audio": audio,
                    "clip": clip,
                    "rgb": rgb,
                    "rms": rms,
                    "nonsilent": nonsilent,
                    "evidence": build_evidence(audio, clip, rgb, rms, nonsilent),
                    "start": start,
                    "length": length,
                    "target": target,
                    "sync_targets": sync_targets.float(),
                    "metadata": SequenceMetadata(
                        youtube_id=youtube_id,
                        partner_youtube_id=partner_value,
                        partition=partition,
                        condition=condition,
                        start_window=start,
                        length=length,
                        target_index=target,
                        synchronizable=syncable,
                    ),
                }
            )
    records: list[dict[str, Any]] = []
    by_steps: dict[int, list[dict[str, Any]]] = {}
    for item in pending:
        by_steps.setdefault(int(item["audio"].shape[0]), []).append(item)
    scorer = scorer.to(device)
    for steps, items in sorted(by_steps.items()):
        for batch_start in range(0, len(items), scorer_batch_size):
            selected = items[batch_start : batch_start + scorer_batch_size]
            frozen = scorer(
                FrozenOffsetScorerInputs(
                    torch.stack([item["audio"] for item in selected]).to(device),
                    torch.stack([item["clip"] for item in selected]).to(device),
                    torch.stack([item["rgb"] for item in selected]).to(device),
                    torch.stack([item["rms"] for item in selected]).to(device),
                    torch.stack([item["nonsilent"] for item in selected]).to(device),
                )
            )
            for row, item in enumerate(selected):
                start = int(item["start"])
                end = start + int(item["length"])
                records.append(
                    {
                        "candidate_logits": frozen.candidate_scores[row, start:end].cpu(),
                        "candidate_features": frozen.candidate_features[row, start:end].cpu(),
                        "evidence": item["evidence"][start:end],
                        "offset_targets": torch.full(
                            (item["length"],),
                            item["target"],
                            dtype=torch.long,
                        ),
                        "sync_targets": torch.full(
                            (item["length"],), 0.0, dtype=torch.float32
                        )
                        + item["sync_targets"][start:end],
                        "metadata": item["metadata"],
                    }
                )
            print(
                f"[{partition}] scorer {min(batch_start + len(selected), len(items))}/"
                f"{len(items)}（窗口数 {steps}）",
                flush=True,
            )
    scorer.to("cpu")
    return records


def normalize_evidence(
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.cat([record["evidence"] for record in train_records])
    mean_value = values.mean(dim=0)
    std_value = values.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean_value, std_value


def collate(records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    batch = len(records)
    steps = max(record["candidate_logits"].shape[0] for record in records)
    feature_dim = records[0]["candidate_features"].shape[-1]
    evidence_dim = records[0]["evidence"].shape[-1]
    output = {
        "candidate_logits": torch.zeros(batch, steps, 3),
        "candidate_features": torch.zeros(batch, steps, 3, feature_dim),
        "evidence": torch.zeros(batch, steps, evidence_dim),
        "offset_targets": torch.ones(batch, steps, dtype=torch.long),
        "sync_targets": torch.zeros(batch, steps),
        "mask": torch.zeros(batch, steps, dtype=torch.bool),
    }
    for index, record in enumerate(records):
        length = record["candidate_logits"].shape[0]
        for key in (
            "candidate_logits",
            "candidate_features",
            "evidence",
            "offset_targets",
            "sync_targets",
        ):
            output[key][index, :length] = record[key]
        output["mask"][index, :length] = True
    return output


def compute_loss(
    model: TemporalOffsetGRUDiagnostic,
    batch: dict[str, torch.Tensor],
    *,
    emd_weight: float,
    sync_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(
        batch["candidate_logits"],
        batch["candidate_features"],
        batch["evidence"],
    )
    mask = batch["mask"]
    offset_mask = mask & batch["sync_targets"].bool()
    ce = F.cross_entropy(
        output.offset_logits[offset_mask],
        batch["offset_targets"][offset_mask],
    )
    emd = ordered_offset_emd_loss(
        output.offset_logits[offset_mask],
        batch["offset_targets"][offset_mask],
    )
    sync_targets = batch["sync_targets"][mask]
    positive = sync_targets.sum()
    negative = sync_targets.numel() - positive
    positive_weight = (negative / positive.clamp_min(1.0)).clamp(0.25, 4.0)
    sync = F.binary_cross_entropy_with_logits(
        output.synchronizability_logits[mask],
        sync_targets,
        pos_weight=positive_weight,
    )
    total = ce + float(emd_weight) * emd + float(sync_weight) * sync
    return total, {
        "loss": float(total.detach().item()),
        "offset_ce": float(ce.detach().item()),
        "offset_emd": float(emd.detach().item()),
        "sync_bce": float(sync.detach().item()),
    }


def binary_auc(targets: list[int], scores: list[float]) -> float | None:
    positives = [score for target, score in zip(targets, scores) if target == 1]
    negatives = [score for target, score in zip(targets, scores) if target == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


def jump_rate(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return sum(left != right for left, right in zip(values, values[1:])) / (len(values) - 1)


def nonzero_runs(values: list[float]) -> list[int]:
    runs = []
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[end] == values[index]:
            end += 1
        if values[index] != 0.0:
            runs.append(end - index)
        index = end
    return runs


@torch.no_grad()
def evaluate(
    model: TemporalOffsetGRUDiagnostic,
    records: list[dict[str, Any]],
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    offset_correct: list[bool] = []
    sync_targets: list[int] = []
    sync_scores: list[float] = []
    accepted_correct: list[bool] = []
    accepted_values: list[bool] = []
    jumps = []
    runs: list[int] = []
    isolated = 0
    nonzero_windows = 0
    direction_correct: dict[int, list[bool]] = {0: [], 1: [], 2: []}
    zero_false_shift = 0
    true_shift_total = 0
    true_shift_recalled = 0
    long_wrong_locks = 0
    score_values = []
    for start in range(0, len(records), batch_size):
        selected = records[start : start + batch_size]
        batch = collate(selected)
        output = model(
            batch["candidate_logits"],
            batch["candidate_features"],
            batch["evidence"],
        )
        prediction = output.offset_logits.argmax(dim=-1)
        for row_index, record in enumerate(selected):
            length = record["candidate_logits"].shape[0]
            target = batch["offset_targets"][row_index, :length]
            sync_target = batch["sync_targets"][row_index, :length].bool()
            pred = prediction[row_index, :length]
            accepted = output.accepted[row_index, :length]
            suggested = output.suggested_offset[row_index, :length]
            score_values.extend(output.offset_logits[row_index, :length].reshape(-1).tolist())
            sync_targets.extend(sync_target.long().tolist())
            sync_scores.extend(output.synchronizability_prob[row_index, :length].tolist())
            accepted_values.extend(accepted.tolist())
            if sync_target.any():
                correct = pred[sync_target].eq(target[sync_target])
                offset_correct.extend(correct.tolist())
                for direction in (0, 1, 2):
                    direction_mask = sync_target & target.eq(direction)
                    direction_correct[direction].extend(pred[direction_mask].eq(direction).tolist())
                accepted_sync = accepted & sync_target
                accepted_correct.extend(pred[accepted_sync].eq(target[accepted_sync]).tolist())
                zero_mask = sync_target & target.eq(1)
                zero_false_shift += int((accepted & zero_mask).sum().item())
                shift_mask = sync_target & target.ne(1)
                true_shift_total += int(shift_mask.sum().item())
                true_shift_recalled += int((accepted & shift_mask & pred.eq(target)).sum().item())
                wrong_nonzero = accepted & shift_mask & pred.ne(target)
                if int(wrong_nonzero.sum().item()) >= 4:
                    long_wrong_locks += 1
            values = suggested.tolist()
            jumps.append(jump_rate(values))
            row_runs = nonzero_runs(values)
            runs.extend(row_runs)
            isolated += sum(value == 1 for value in row_runs)
            nonzero_windows += sum(value != 0.0 for value in values)
    sync_pred = [score >= model.synchronizability_threshold for score in sync_scores]
    tp = sum(pred and target for pred, target in zip(sync_pred, sync_targets))
    fp = sum(pred and not target for pred, target in zip(sync_pred, sync_targets))
    fn = sum(not pred and target for pred, target in zip(sync_pred, sync_targets))
    return {
        "sequence_count": len(records),
        "window_count": len(sync_targets),
        "offset_accuracy": mean(offset_correct),
        "high_confidence_accuracy": mean(accepted_correct) if accepted_correct else None,
        "synchronizability_auc": binary_auc(sync_targets, sync_scores),
        "synchronizability_f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "mean_sequence_jump_rate": mean(jumps),
        "isolated_nonzero_window_ratio": isolated / max(1, nonzero_windows),
        "nonzero_run_length_mean": mean(runs) if runs else 0.0,
        "zero_false_correction_rate": zero_false_shift / max(1, len(direction_correct[1])),
        "true_shift_recall": true_shift_recalled / max(1, true_shift_total),
        "direction_accuracy": {
            str(OFFSETS[index]): mean(values) if values else None
            for index, values in direction_correct.items()
        },
        "acceptance_rate": mean(accepted_values),
        "maximum_extra_delay_seconds": 0.0,
        "long_wrong_direction_lock_count": long_wrong_locks,
        "all_outputs_finite": bool(torch.isfinite(torch.tensor(score_values)).all().item()),
        "collapsed_offset_logits": float(torch.tensor(score_values).std(unbiased=False).item()) <= 1e-6,
    }


def train_steps(
    model: TemporalOffsetGRUDiagnostic,
    records: list[dict[str, Any]],
    *,
    steps: int,
    batch_size: int,
    lr: float,
    emd_weight: float,
    sync_weight: float,
    seed: int,
) -> list[dict[str, float]]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(seed)
    history = []
    for step in range(1, steps + 1):
        selected = (
            records
            if len(records) <= batch_size
            else rng.sample(records, batch_size)
        )
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = compute_loss(
            model,
            collate(selected),
            emd_weight=emd_weight,
            sync_weight=sync_weight,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        metrics.update(
            {
                "step": step,
                "grad_norm": float(grad_norm),
                "finite": bool(torch.isfinite(loss).item()),
            }
        )
        history.append(metrics)
    return history


def new_model(
    seed: int,
    sync_threshold: float,
    evidence_mean: torch.Tensor,
    evidence_std: torch.Tensor,
) -> TemporalOffsetGRUDiagnostic:
    torch.manual_seed(seed)
    return TemporalOffsetGRUDiagnostic(
        hidden_size=128,
        synchronizability_threshold=sync_threshold,
        frozen_margin_threshold=0.15,
        evidence_mean=evidence_mean,
        evidence_std=evidence_std,
    )


def save_stage(
    path: Path,
    model: TemporalOffsetGRUDiagnostic,
    stage: str,
    history: list[dict[str, float]],
    validation: dict[str, Any],
) -> None:
    torch.save(
        {
            "format_version": 1,
            "state_dict": model.state_dict(),
            "metadata": {
                "stage": stage,
                "candidate_feature_dim": model.candidate_feature_dim,
                "evidence_dim": model.evidence_dim,
                "hidden_size": model.hidden_size,
                "candidate_projection_dim": model.candidate_projection_dim,
                "synchronizability_threshold": model.synchronizability_threshold,
                "frozen_margin_threshold": model.frozen_margin_threshold,
            },
            "history": history,
            "validation": validation,
        },
        path,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.source_manifest).resolve()
    assert_development_source(source_path)
    output_root = Path(args.output_root).resolve()
    if args.scorer_batch_size <= 0:
        raise ValueError("scorer_batch_size must be positive")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"请求的设备不可用：{args.device}")
    if output_root.exists() and any(output_root.iterdir()) and not args.reuse_cache:
        raise FileExistsError(f"独立实验目录已存在且非空：{output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    source_rows = load_jsonl(source_path)
    annotations = load_annotations(Path(args.annotations).resolve())
    source_ids = [str(row["youtube_id"]) for row in source_rows]
    if args.max_videos > 0:
        source_ids = sorted(source_ids)[: args.max_videos]
    train_ids, dev_ids = split_youtube_ids(
        source_ids,
        dev_ratio=args.dev_ratio,
        seed=args.split_seed,
    )
    clip_all = {str(row["youtube_id"]): row for row in load_jsonl(Path(args.clip_manifest).resolve())}
    rgb_all = {str(row["youtube_id"]): row for row in load_jsonl(Path(args.rgb_manifest).resolve())}
    selected_ids = set(train_ids) | set(dev_ids)
    if not selected_ids <= set(clip_all) or not selected_ids <= set(rgb_all):
        raise ValueError("development youtube_id 与三路特征 manifest 不一致")
    clip_rows = {key: clip_all[key] for key in selected_ids}
    rgb_rows = {key: rgb_all[key] for key in selected_ids}

    cache_path = output_root / "sequence_cache.pt"
    if args.reuse_cache and cache_path.is_file():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        train_records = cache["train_records"]
        dev_records = cache["dev_records"]
        evidence_mean = cache["evidence_mean"]
        evidence_std = cache["evidence_std"]
    else:
        scorer = FrozenTemporalOffsetScorer(
            Path(args.bundle).resolve(),
            margin_threshold=0.15,
        ).eval()
        train_records = build_partition_sequences(
            train_ids,
            "train",
            clip_rows,
            rgb_rows,
            scorer,
            annotations,
            min_length=args.min_sequence_length,
            max_length=args.max_sequence_length,
            seed=args.seed,
            scorer_batch_size=args.scorer_batch_size,
            device=torch.device(args.device),
        )
        dev_records = build_partition_sequences(
            dev_ids,
            "dev",
            clip_rows,
            rgb_rows,
            scorer,
            annotations,
            min_length=args.min_sequence_length,
            max_length=args.max_sequence_length,
            seed=args.seed,
            scorer_batch_size=args.scorer_batch_size,
            device=torch.device(args.device),
        )
        evidence_mean, evidence_std = normalize_evidence(train_records, dev_records)
        torch.save(
            {
                "train_records": train_records,
                "dev_records": dev_records,
                "evidence_mean": evidence_mean,
                "evidence_std": evidence_std,
            },
            cache_path,
        )

    stages = []
    definitions = [
        ("2step", train_records, 2),
        (
            "overfit_small",
            train_records[: min(args.overfit_sequences, len(train_records))],
            args.overfit_steps,
        ),
        ("20step", train_records, 20),
        (
            "one_epoch",
            train_records,
            max(1, math.ceil(len(train_records) / args.batch_size)),
        ),
    ]
    for index, (name, records, steps) in enumerate(definitions):
        model = new_model(
            args.seed + index,
            args.sync_threshold,
            evidence_mean,
            evidence_std,
        )
        history = train_steps(
            model,
            records,
            steps=steps,
            batch_size=args.batch_size,
            lr=args.lr,
            emd_weight=args.emd_weight,
            sync_weight=args.sync_weight,
            seed=args.seed + index,
        )
        validation = evaluate(model, dev_records, args.batch_size)
        training = evaluate(model, records, args.batch_size)
        save_stage(
            output_root / f"temporal_offset_gru_{name}.pt",
            model,
            name,
            history,
            validation,
        )
        stages.append(
            {
                "name": name,
                "steps": steps,
                "final_training": history[-1],
                "training_evaluation": training,
                "validation": validation,
            }
        )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "moves_audio_window": False,
        "feeds_gate": False,
        "modifies_fusion": False,
        "trains_m4": False,
        "source_manifest": str(source_path),
        "output_root": str(output_root),
        "frozen_test_manifest_read": False,
        "train_youtube_count": len(train_ids),
        "dev_youtube_count": len(dev_ids),
        "youtube_overlap": sorted(set(train_ids) & set(dev_ids)),
        "train_sequence_count": len(train_records),
        "dev_sequence_count": len(dev_records),
        "sequence_length_range": [
            min(record["candidate_logits"].shape[0] for record in train_records + dev_records),
            max(record["candidate_logits"].shape[0] for record in train_records + dev_records),
        ],
        "condition_counts": dict(
            Counter(asdict(record["metadata"])["condition"] for record in train_records + dev_records)
        ),
        "zero_syncable_positive_count": sum(
            int(record["sync_targets"].sum().item())
            for record in train_records + dev_records
            if record["metadata"].condition == "original"
        ),
        "config": {
            "seed": args.seed,
            "split_seed": args.split_seed,
            "hidden_size": 128,
            "frozen_margin_threshold": 0.15,
            "synchronizability_threshold": args.sync_threshold,
            "losses": ["offset_cross_entropy", "ordered_emd", "synchronizability_bce"],
            "temporal_consistency_loss": False,
            "evidence_mean": evidence_mean.tolist(),
            "evidence_std": evidence_std.tolist(),
        },
        "baseline": {
            "jump_rate": 0.3505,
            "high_confidence_accuracy": 0.8412,
            "additional_delay_seconds": 0.0,
        },
        "stages": stages,
    }
    (output_root / "temporal_offset_gru_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE))
    parser.add_argument("--clip-manifest", default=str(DEFAULT_CLIP))
    parser.add_argument("--rgb-manifest", default=str(DEFAULT_RGB))
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--split-seed", type=int, default=20260719)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--min-sequence-length", type=int, default=5)
    parser.add_argument("--max-sequence-length", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emd-weight", type=float, default=0.25)
    parser.add_argument("--sync-weight", type=float, default=1.0)
    parser.add_argument("--sync-threshold", type=float, default=0.5)
    parser.add_argument("--overfit-sequences", type=int, default=16)
    parser.add_argument("--overfit-steps", type=int, default=200)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--scorer-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-cache", action="store_true")
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    final = summary["stages"][-1]["validation"]
    print(
        json.dumps(
            {
                "ok": True,
                "offset_accuracy": final["offset_accuracy"],
                "high_confidence_accuracy": final["high_confidence_accuracy"],
                "jump_rate": final["mean_sequence_jump_rate"],
                "report": str(
                    Path(summary["output_root"]) / "temporal_offset_gru_summary.json"
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
