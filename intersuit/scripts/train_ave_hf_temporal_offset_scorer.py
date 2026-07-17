#!/usr/bin/env python
"""训练 AVE_HF 独立时间同步 offset scorer。

本脚本只训练轻量三候选 scorer，用于判断音频相对视频应选择 -0.5s/0/+0.5s。
它固定现有 AVE_HF 300 媒体、BEATs 音频特征、M4 CLIP 视频特征和已有
train/val 划分；不接 Gate，不修改动态窗口，不修改正式 M4 推理路径。

注意：当前本地 AVE_HF manifest 只有 YouTube segment 的 start_seconds，
没有可靠的本地事件起止区间。因此本轮候选窗口按 BEATs/CLIP/RGB/能量变化
峰值筛选，报告中不冒充官方事件边界监督。v2 默认使用 5-window 上下文，
并过滤候选间变化强度过于接近的稳定窗口。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import wavfile
from torch import nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"
DEFAULT_CLIP_MANIFEST = INTERSUIT_ROOT / "harness/artifacts/ave_hf_300_clip_window_features/ave_hf_clip_window_feature_manifest.jsonl"
DEFAULT_RGB_MANIFEST = INTERSUIT_ROOT / "harness/artifacts/ave_hf_300_window_features/ave_hf_window_feature_manifest.jsonl"
DEFAULT_SPLIT_SUMMARY = INTERSUIT_ROOT / "harness/artifacts/ave_hf_semantic_projector_clip300/semantic_projector_run_summary.json"
DEFAULT_OUTPUT_ROOT = INTERSUIT_ROOT / "harness/artifacts/ave_hf_temporal_offset_context_scorer_clip300"
OFFSETS = (-0.5, 0.0, 0.5)
CONDITION_TO_BASE_SHIFT = {"original": 0, "shift_plus_0.5": 1, "shift_minus_0.5": -1}
CONDITION_TO_TARGET = {"original": 1, "shift_plus_0.5": 0, "shift_minus_0.5": 2}
CONDITIONS = ("original", "shift_plus_0.5", "shift_minus_0.5")


def import_semantic_module():
    path = INTERSUIT_ROOT / "scripts/train_ave_hf_semantic_projector.py"
    spec = importlib.util.spec_from_file_location("ave_hf_semantic_projector_for_temporal_offset", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


semantic = import_semantic_module()


@dataclass(frozen=True)
class OffsetRecord:
    youtube_id: str
    label: str
    split: str
    condition: str
    video_window: int
    audio_candidate_windows: list[int]
    correct_offset: float
    target_index: int
    event_boundary_type: str
    audio_change_strength: float
    clip_change_strength: float
    rgb_change_strength: float
    energy_change_strength: float
    candidate_change_margin: float


class OffsetScorer(nn.Module):
    def __init__(self, audio_dim: int, video_dim: int, scalar_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        pair_dim = audio_dim + video_dim + scalar_dim
        self.net = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, audio_candidates: torch.Tensor, video_context: torch.Tensor, scalar_features: torch.Tensor) -> torch.Tensor:
        video = video_context.unsqueeze(1).expand(-1, audio_candidates.shape[1], -1)
        values = torch.cat([audio_candidates.float(), video.float(), scalar_features.float()], dim=-1)
        return self.net(values).squeeze(-1)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"manifest 为空：{path}")
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[OffsetRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def load_feature(path: Path, key: str) -> tuple[torch.Tensor, torch.Tensor]:
    return semantic.load_feature(path, key)


def l2_normalize(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(values.float(), dim=-1, eps=1e-6)


def diff_norm(values: torch.Tensor) -> torch.Tensor:
    if values.shape[0] < 2:
        return torch.zeros(values.shape[0])
    diffs = torch.zeros(values.shape[0])
    step = (values[1:] - values[:-1]).norm(dim=-1)
    diffs[1:] = torch.maximum(diffs[1:], step)
    diffs[:-1] = torch.maximum(diffs[:-1], step)
    return diffs


def window_audio_stats(audio_path: Path, timestamps: torch.Tensor, sample_rate: int = 16000) -> tuple[torch.Tensor, torch.Tensor]:
    rate, data = wavfile.read(audio_path)
    if rate != sample_rate:
        raise ValueError(f"音频采样率不是 {sample_rate}：{audio_path}")
    values = torch.as_tensor(data).float()
    if values.ndim == 2:
        values = values.mean(dim=1)
    if data.dtype.kind in {"i", "u"}:
        info = np.iinfo(data.dtype)
        values = values / float(max(abs(info.min), abs(info.max)))
    rms, nonsilent = [], []
    for start, end in timestamps.tolist():
        left = max(0, int(round(start * sample_rate)))
        right = min(values.numel(), int(round(end * sample_rate)))
        window = values[left:right]
        if window.numel() == 0:
            rms.append(0.0)
            nonsilent.append(0.0)
        else:
            rms.append(float(torch.sqrt((window * window).mean()).item()))
            nonsilent.append(float((window.abs() > 1e-4).float().mean().item()))
    return torch.tensor(rms, dtype=torch.float32), torch.tensor(nonsilent, dtype=torch.float32)


def neighbor_delta(values: torch.Tensor, index: int) -> torch.Tensor:
    prev_idx = max(0, index - 1)
    next_idx = min(values.shape[0] - 1, index + 1)
    return torch.cat([values[index], values[index] - values[prev_idx], values[next_idx] - values[index]], dim=-1)


def context_indices(index: int, length: int, radius: int) -> list[int]:
    return [max(0, min(length - 1, index + delta)) for delta in range(-radius, radius + 1)]


def context_with_deltas(values: torch.Tensor, index: int, radius: int) -> torch.Tensor:
    indices = context_indices(index, values.shape[0], radius)
    context = values[indices]
    deltas = torch.zeros_like(context)
    if context.shape[0] > 1:
        step = context[1:] - context[:-1]
        deltas[1:] = step
        deltas[:-1] = deltas[:-1] + step
    return torch.cat([context.reshape(-1), deltas.reshape(-1)], dim=0)


def scalar_context(cache: dict[str, Any], audio_index: int, video_index: int, radius: int) -> torch.Tensor:
    audio_indices = context_indices(audio_index, cache["audio"].shape[0], radius)
    video_indices = context_indices(video_index, cache["clip"].shape[0], radius)
    values = torch.stack(
        [
            cache["rms"][audio_indices],
            cache["nonsilent"][audio_indices],
            cache["audio_change"][audio_indices],
            cache["energy_change"][audio_indices],
            cache["clip_change"][video_indices],
            cache["rgb_change"][video_indices],
        ],
        dim=1,
    )
    return values.reshape(-1)


def build_row_cache(clip_row: dict[str, Any], rgb_row: dict[str, Any]) -> dict[str, Any]:
    audio, audio_ts = load_feature(Path(clip_row["audio_feature_path"]), "audio_embedding")
    clip, clip_ts = load_feature(Path(clip_row["video_feature_path"]), "video_features")
    rgb, rgb_ts = load_feature(Path(rgb_row["video_feature_path"]), "video_features")
    if not (audio.shape[0] == clip.shape[0] == rgb.shape[0]):
        raise ValueError(f"窗口数不一致：{clip_row['youtube_id']}")
    if not torch.allclose(audio_ts, clip_ts, atol=1e-5, rtol=0.0) or not torch.allclose(audio_ts, rgb_ts, atol=1e-5, rtol=0.0):
        raise ValueError(f"时间戳不一致：{clip_row['youtube_id']}")
    rms, nonsilent = window_audio_stats(Path(rgb_row["audio_path"]), audio_ts)
    audio_change = diff_norm(audio)
    clip_change = diff_norm(clip)
    rgb_change = diff_norm(rgb)
    energy_change = diff_norm(rms.unsqueeze(-1))
    combined = zscore(audio_change) + zscore(clip_change) + zscore(rgb_change) + zscore(energy_change)
    return {
        "youtube_id": str(clip_row["youtube_id"]),
        "label": str(clip_row["label"]),
        "split": str(clip_row.get("split", "train")),
        "audio": l2_normalize(audio),
        "clip": l2_normalize(clip),
        "rgb": l2_normalize(rgb),
        "timestamps": audio_ts,
        "rms": rms,
        "nonsilent": nonsilent,
        "audio_change": audio_change,
        "clip_change": clip_change,
        "rgb_change": rgb_change,
        "energy_change": energy_change,
        "combined_change": combined,
    }


def zscore(values: torch.Tensor) -> torch.Tensor:
    std = values.std(unbiased=False)
    if float(std.item()) <= 1e-8:
        return torch.zeros_like(values)
    return (values - values.mean()) / std


def select_change_windows(cache: dict[str, Any], top_k: int, min_change_quantile: float) -> list[tuple[int, str]]:
    scores = cache["combined_change"]
    valid = list(range(1, int(scores.numel()) - 1))
    if not valid:
        return []
    threshold = torch.quantile(scores[valid], min_change_quantile).item()
    ranked = sorted(valid, key=lambda idx: float(scores[idx].item()), reverse=True)
    selected = []
    for idx in ranked:
        if float(scores[idx].item()) < threshold and selected:
            continue
        boundary_type = max(
            [
                ("audio_change", float(cache["audio_change"][idx].item())),
                ("clip_change", float(cache["clip_change"][idx].item())),
                ("rgb_change", float(cache["rgb_change"][idx].item())),
                ("energy_change", float(cache["energy_change"][idx].item())),
            ],
            key=lambda item: item[1],
        )[0]
        selected.append((idx, boundary_type))
        if len(selected) >= top_k:
            break
    return selected


def candidate_change_margin(cache: dict[str, Any], candidate_windows: list[int]) -> float:
    values = cache["audio_change"][candidate_windows] + cache["energy_change"][candidate_windows]
    return float((values.max() - values.min()).item())


def build_records(
    caches: list[dict[str, Any]],
    split_ids: set[str],
    top_k: int,
    min_change_quantile: float,
    min_candidate_change_margin: float,
) -> list[OffsetRecord]:
    records: list[OffsetRecord] = []
    for cache in caches:
        if cache["youtube_id"] not in split_ids:
            continue
        for window_idx, boundary_type in select_change_windows(cache, top_k, min_change_quantile):
            per_condition: list[OffsetRecord] = []
            for condition in CONDITIONS:
                base_shift = CONDITION_TO_BASE_SHIFT[condition]
                base = window_idx + base_shift
                candidate_windows = [base - 1, base, base + 1]
                if min(candidate_windows) < 0 or max(candidate_windows) >= cache["audio"].shape[0]:
                    per_condition = []
                    break
                change_margin = candidate_change_margin(cache, candidate_windows)
                if change_margin < min_candidate_change_margin:
                    per_condition = []
                    break
                target_index = CONDITION_TO_TARGET[condition]
                per_condition.append(
                    OffsetRecord(
                        youtube_id=cache["youtube_id"],
                        label=cache["label"],
                        split=cache["split"],
                        condition=condition,
                        video_window=window_idx,
                        audio_candidate_windows=candidate_windows,
                        correct_offset=OFFSETS[target_index],
                        target_index=target_index,
                        event_boundary_type=boundary_type,
                        audio_change_strength=float(cache["audio_change"][window_idx].item()),
                        clip_change_strength=float(cache["clip_change"][window_idx].item()),
                        rgb_change_strength=float(cache["rgb_change"][window_idx].item()),
                        energy_change_strength=float(cache["energy_change"][window_idx].item()),
                        candidate_change_margin=change_margin,
                    )
                )
            if len(per_condition) == len(CONDITIONS):
                records.extend(per_condition)
    return records


def balance_records_by_target(records: list[OffsetRecord], seed: int) -> list[OffsetRecord]:
    rng = random.Random(seed)
    by_target: dict[int, list[OffsetRecord]] = defaultdict(list)
    for record in records:
        by_target[record.target_index].append(record)
    if set(by_target) != {0, 1, 2}:
        raise ValueError(f"offset 类别不完整：{sorted(by_target)}")
    count = min(len(items) for items in by_target.values())
    balanced: list[OffsetRecord] = []
    for target in sorted(by_target):
        items = list(by_target[target])
        rng.shuffle(items)
        balanced.extend(items[:count])
    rng.shuffle(balanced)
    return balanced


def make_tensor_dataset(
    records: list[OffsetRecord],
    cache_by_id: dict[str, dict[str, Any]],
    context_radius: int,
    scalar_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[OffsetRecord], tuple[torch.Tensor, torch.Tensor]]:
    audio_rows, video_rows, scalar_rows, targets = [], [], [], []
    kept = []
    for record in records:
        cache = cache_by_id[record.youtube_id]
        video_idx = record.video_window
        video_context = torch.cat(
            [
                context_with_deltas(cache["clip"], video_idx, context_radius),
                context_with_deltas(cache["rgb"], video_idx, context_radius),
            ],
            dim=-1,
        )
        candidate_audio, candidate_scalars = [], []
        for cand_idx in record.audio_candidate_windows:
            candidate_audio.append(context_with_deltas(cache["audio"], cand_idx, context_radius))
            candidate_scalars.append(scalar_context(cache, cand_idx, video_idx, context_radius))
        audio_rows.append(torch.stack(candidate_audio))
        video_rows.append(video_context)
        scalar_rows.append(torch.stack(candidate_scalars))
        targets.append(record.target_index)
        kept.append(record)
    if not kept:
        raise ValueError("没有可训练 offset 样本")
    audio = torch.stack(audio_rows)
    video = torch.stack(video_rows)
    scalars = torch.stack(scalar_rows)
    if scalar_stats is None:
        scalar_mean = scalars.reshape(-1, scalars.shape[-1]).mean(dim=0)
        scalar_std = scalars.reshape(-1, scalars.shape[-1]).std(dim=0, unbiased=False).clamp_min(1e-6)
    else:
        scalar_mean, scalar_std = scalar_stats
    scalars = (scalars - scalar_mean) / scalar_std
    if not torch.isfinite(audio).all() or not torch.isfinite(video).all() or not torch.isfinite(scalars).all():
        raise ValueError("offset 特征包含 NaN/Inf")
    return audio, video, scalars, torch.tensor(targets, dtype=torch.long), kept, (scalar_mean, scalar_std)


def train_model(
    audio: torch.Tensor,
    video: torch.Tensor,
    scalars: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
    lr: float,
    hidden_dim: int,
    seed: int,
    batch_size: int,
) -> tuple[OffsetScorer, list[dict[str, Any]]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = OffsetScorer(audio_dim=audio.shape[-1], video_dim=video.shape[-1], scalar_dim=scalars.shape[-1], hidden_dim=hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    for step in range(1, steps + 1):
        if batch_size >= targets.numel():
            batch_indices = torch.arange(targets.numel())
        else:
            batch_indices = torch.tensor(rng.sample(range(targets.numel()), batch_size), dtype=torch.long)
        batch_audio = audio[batch_indices]
        batch_video = video[batch_indices]
        batch_scalars = scalars[batch_indices]
        batch_targets = targets[batch_indices]
        optimizer.zero_grad(set_to_none=True)
        scores = model(batch_audio, batch_video, batch_scalars)
        loss = F.cross_entropy(scores, batch_targets)
        loss.backward()
        grad_norm_sq = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm_sq += float(param.grad.detach().float().pow(2).sum().item())
        optimizer.step()
        with torch.no_grad():
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "accuracy": float((scores.argmax(dim=-1) == batch_targets).float().mean().item()),
                    "batch_size": int(batch_targets.numel()),
                    "score_std": float(scores.std(unbiased=False).item()),
                    "grad_norm": math.sqrt(grad_norm_sq),
                    "finite": bool(torch.isfinite(scores).all().item()),
                }
            )
    return model, history


@torch.no_grad()
def evaluate(model: OffsetScorer, audio: torch.Tensor, video: torch.Tensor, scalars: torch.Tensor, targets: torch.Tensor, records: list[OffsetRecord]) -> dict[str, Any]:
    scores = model(audio, video, scalars)
    probs = torch.softmax(scores, dim=-1)
    pred = scores.argmax(dim=-1)
    sorted_scores = scores.sort(dim=-1, descending=True).values
    margins = sorted_scores[:, 0] - sorted_scores[:, 1]
    correct = pred.eq(targets)
    by_condition: dict[str, list[float]] = defaultdict(list)
    by_label: dict[str, list[float]] = defaultdict(list)
    by_boundary: dict[str, list[float]] = defaultdict(list)
    pred_counts = Counter()
    for idx, record in enumerate(records):
        value = float(correct[idx].item())
        by_condition[record.condition].append(value)
        by_label[record.label].append(value)
        by_boundary[record.event_boundary_type].append(value)
        pred_counts[str(OFFSETS[int(pred[idx].item())])] += 1
    mean = lambda xs: float(torch.tensor(xs, dtype=torch.float32).mean().item()) if xs else None
    return {
        "sample_count": len(records),
        "accuracy": float(correct.float().mean().item()),
        "rank_first": float(correct.float().mean().item()),
        "margin_mean": float(margins.mean().item()),
        "confidence_mean": float(probs.max(dim=-1).values.mean().item()),
        "condition_accuracy": {key: mean(values) for key, values in sorted(by_condition.items())},
        "label_accuracy": {key: mean(values) for key, values in sorted(by_label.items(), key=lambda item: (int(item[0]) if item[0].isdigit() else 9999, item[0]))},
        "boundary_type_accuracy": {key: mean(values) for key, values in sorted(by_boundary.items())},
        "prediction_distribution": dict(pred_counts),
        "score_mean": [float(x) for x in scores.mean(dim=0).tolist()],
        "score_std": float(scores.std(unbiased=False).item()),
        "all_scores_finite": bool(torch.isfinite(scores).all().item()),
        "collapsed_scores": float(scores.std(unbiased=False).item()) <= 1e-6,
    }


def save_checkpoint(path: Path, model: OffsetScorer, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" not in payload:
        raise ValueError(f"checkpoint 保存失败：{path}")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# AVE_HF 时间同步 offset scorer 报告",
        "",
        "本实验独立训练轻量 offset scorer，只判断 -0.5s/0/+0.5s 三候选；未接 Gate、动态窗口或正式 M4 推理。",
        "",
        "## 数据",
        f"- 固定基线提交：`e5b5df3`",
        f"- train/val split：来自 `{summary['config']['split_summary']}`",
        f"- 候选窗口来源：BEATs/CLIP/RGB/能量变化峰值；本地 manifest 缺少可靠 AVE 事件起止字段，因此未使用官方边界监督。",
        f"- 上下文窗口数：{summary['config']['context_window_count']}",
        f"- 候选变化强度最小差距：{summary['config']['min_candidate_change_margin']}",
        f"- train 样本数：{summary['manifest_summary']['train_count']}",
        f"- val 样本数：{summary['manifest_summary']['val_count']}",
        f"- offset 分布：`{summary['manifest_summary']['target_offset_counts']}`",
        "",
        "## 结果",
    ]
    for run in summary["runs"]:
        val = run["validation"]
        lines.extend(
            [
                f"### {run['name']}",
                f"- steps：{run['steps']}",
                f"- final loss：{run['history'][-1]['loss']}",
                f"- train accuracy：{run['training']['accuracy']}",
                f"- val accuracy：{val['accuracy']}",
                f"- val margin mean：{val['margin_mean']}",
                f"- val confidence mean：{val['confidence_mean']}",
                f"- val condition accuracy：`{val['condition_accuracy']}`",
                f"- val prediction distribution：`{val['prediction_distribution']}`",
                f"- 无 NaN/Inf：{val['all_scores_finite']}",
                f"- 分数塌缩：{val['collapsed_scores']}",
                "",
            ]
        )
    lines.extend(["## 结论", f"- 状态：{summary['decision']}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    clip_rows = load_jsonl(Path(args.clip_manifest).resolve())
    rgb_rows = {str(row["youtube_id"]): row for row in load_jsonl(Path(args.rgb_manifest).resolve())}
    split_summary = load_json(Path(args.split_summary).resolve())
    caches = []
    for row in clip_rows:
        youtube_id = str(row["youtube_id"])
        if youtube_id not in rgb_rows:
            raise ValueError(f"缺少 RGB 特征行：{youtube_id}")
        caches.append(build_row_cache(row, rgb_rows[youtube_id]))
    cache_by_id = {cache["youtube_id"]: cache for cache in caches}
    train_ids = set(split_summary["split"]["train_ids"])
    val_ids = set(split_summary["split"]["val_ids"])
    train_records = build_records(
        caches,
        train_ids,
        top_k=args.top_k_windows,
        min_change_quantile=args.min_change_quantile,
        min_candidate_change_margin=args.min_candidate_change_margin,
    )
    val_records = build_records(
        caches,
        val_ids,
        top_k=args.top_k_windows,
        min_change_quantile=args.min_change_quantile,
        min_candidate_change_margin=args.min_candidate_change_margin,
    )
    train_records = balance_records_by_target(train_records, seed=args.seed)
    val_records = balance_records_by_target(val_records, seed=args.seed)
    write_jsonl(output_root / "temporal_offset_train_manifest.jsonl", train_records)
    write_jsonl(output_root / "temporal_offset_val_manifest.jsonl", val_records)

    train_audio, train_video, train_scalars, train_targets, train_records, scalar_stats = make_tensor_dataset(
        train_records, cache_by_id, context_radius=args.context_radius
    )
    val_audio, val_video, val_scalars, val_targets, val_records, _ = make_tensor_dataset(
        val_records, cache_by_id, context_radius=args.context_radius, scalar_stats=scalar_stats
    )
    epoch_steps = math.ceil(train_targets.numel() / args.batch_equivalent_size)
    runs = []
    for name, steps in [("2step", 2), ("overfit_small", args.overfit_steps), ("20step", 20), ("one_epoch", epoch_steps)]:
        if name == "overfit_small":
            subset = min(args.overfit_samples, train_targets.numel())
            audio, video, scalars, targets = train_audio[:subset], train_video[:subset], train_scalars[:subset], train_targets[:subset]
            records = train_records[:subset]
        else:
            audio, video, scalars, targets = train_audio, train_video, train_scalars, train_targets
            records = train_records
        train_batch_size = subset if name == "overfit_small" else args.batch_equivalent_size
        model, history = train_model(
            audio,
            video,
            scalars,
            targets,
            steps=steps,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            seed=args.seed,
            batch_size=train_batch_size,
        )
        train_eval = evaluate(model, audio, video, scalars, targets, records)
        val_eval = evaluate(model, val_audio, val_video, val_scalars, val_targets, val_records)
        save_checkpoint(output_root / f"temporal_offset_scorer_{name}.pt", model, {"stage": name, "steps": steps})
        runs.append({"name": name, "steps": steps, "history": history, "training": train_eval, "validation": val_eval})
    final = runs[-1]["validation"]
    balanced = final["prediction_distribution"]
    decision = (
        "temporal_offset_scorer_passed"
        if final["accuracy"] >= args.pass_accuracy
        and final["margin_mean"] > 0
        and len(balanced) >= 2
        and final["all_scores_finite"]
        and not final["collapsed_scores"]
        else "temporal_offset_scorer_not_passed"
    )
    manifest_summary = {
        "train_count": len(train_records),
        "val_count": len(val_records),
        "target_offset_counts": dict(Counter(str(record.correct_offset) for record in train_records + val_records)),
        "condition_counts": dict(Counter(record.condition for record in train_records + val_records)),
        "boundary_type_counts": dict(Counter(record.event_boundary_type for record in train_records + val_records)),
        "train_label_counts": dict(Counter(record.label for record in train_records)),
        "val_label_counts": dict(Counter(record.label for record in val_records)),
    }
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "clip_manifest": str(Path(args.clip_manifest).resolve()),
            "rgb_manifest": str(Path(args.rgb_manifest).resolve()),
            "split_summary": str(Path(args.split_summary).resolve()),
            "output_root": str(output_root),
            "top_k_windows": args.top_k_windows,
            "min_change_quantile": args.min_change_quantile,
            "min_candidate_change_margin": args.min_candidate_change_margin,
            "context_radius": args.context_radius,
            "context_window_count": args.context_radius * 2 + 1,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "seed": args.seed,
            "offsets": list(OFFSETS),
            "gate_or_dynamic_window_modified": False,
        },
        "manifest_summary": manifest_summary,
        "runs": runs,
        "decision": decision,
    }
    write_json(output_root / "temporal_offset_scorer_summary.json", summary)
    write_report(output_root / "temporal_offset_scorer_report.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-manifest", default=str(DEFAULT_CLIP_MANIFEST))
    parser.add_argument("--rgb-manifest", default=str(DEFAULT_RGB_MANIFEST))
    parser.add_argument("--split-summary", default=str(DEFAULT_SPLIT_SUMMARY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--top-k-windows", type=int, default=4)
    parser.add_argument("--min-change-quantile", type=float, default=0.6)
    parser.add_argument("--min-candidate-change-margin", type=float, default=0.5)
    parser.add_argument("--context-radius", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--overfit-steps", type=int, default=50)
    parser.add_argument("--overfit-samples", type=int, default=96)
    parser.add_argument("--batch-equivalent-size", type=int, default=32)
    parser.add_argument("--pass-accuracy", type=float, default=0.65)
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    final = summary["runs"][-1]["validation"]
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "accuracy": final["accuracy"],
                "margin": final["margin_mean"],
                "report": str(Path(summary["config"]["output_root"]) / "temporal_offset_scorer_report.md"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
