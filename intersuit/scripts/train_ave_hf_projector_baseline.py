#!/usr/bin/env python
"""诊断 AVE_HF RGB 视频窗口特征并训练基础 audio/video projector。

本脚本只消费离线 BEATs 音频特征和离线 RGB 视频窗口特征。它不会修改
Gate、动态窗口、视频关注权重、Fusion 或 M4 主体。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"
DEFAULT_FEATURE_MANIFEST = INTERSUIT_ROOT / "harness/artifacts/ave_hf_300_window_features/ave_hf_window_feature_manifest.jsonl"
DEFAULT_OUTPUT_ROOT = INTERSUIT_ROOT / "harness/artifacts/ave_hf_projector_baseline"


@dataclass(frozen=True)
class WindowPair:
    dataset: str
    youtube_id: str
    label: str | None
    split: str
    window_index: int
    window_start: float
    window_end: float
    pair_type: str
    offset: float | None
    target: int
    audio_feature_path: str
    video_feature_path: str
    audio_window_index: int | None
    video_window_index: int
    negative_source_youtube_id: str | None = None


class AVProjector(nn.Module):
    def __init__(self, input_dim: int = 768, project_dim: int = 128) -> None:
        super().__init__()
        self.audio_proj = nn.Linear(input_dim, project_dim, bias=False)
        self.video_proj = nn.Linear(input_dim, project_dim, bias=False)

    def forward(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        audio_z = F.normalize(self.audio_proj(audio.float()), dim=-1, eps=1e-6)
        video_z = F.normalize(self.video_proj(video.float()), dim=-1, eps=1e-6)
        return (audio_z * video_z).sum(dim=-1)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"manifest 为空：{path}")
    return rows


def load_feature(path: Path, key: str) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    values = payload[key].float()
    timestamps = payload["timestamps"].float()
    if values.ndim != 2 or timestamps.shape != (values.shape[0], 2):
        raise ValueError(f"特征/时间戳形状非法：{path}")
    if not torch.isfinite(values).all() or not torch.isfinite(timestamps).all():
        raise ValueError(f"特征包含 NaN/Inf：{path}")
    return values, timestamps


def load_video_entries(manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = []
    for row in manifest_rows:
        video, timestamps = load_feature(Path(row["video_feature_path"]), "video_features")
        entries.append({"row": row, "video": video, "timestamps": timestamps})
    return entries


def diagnose_video_features(manifest_rows: list[dict[str, Any]], max_pairs: int = 5000, seed: int = 20260717) -> dict[str, Any]:
    rng = random.Random(seed)
    entries = load_video_entries(manifest_rows)
    all_video = torch.cat([entry["video"] for entry in entries], dim=0)
    norms = all_video.norm(dim=-1)
    global_std = all_video.std(dim=0, unbiased=False)

    adjacent_distances = []
    adjacent_cosines = []
    for entry in entries:
        video = entry["video"]
        if video.shape[0] >= 2:
            adjacent_distances.extend((video[1:] - video[:-1]).norm(dim=-1).tolist())
            adjacent_cosines.extend(F.cosine_similarity(video[1:], video[:-1], dim=-1).tolist())

    by_video = {entry["row"]["youtube_id"]: entry for entry in entries}
    ids = sorted(by_video)
    diff_distances = []
    diff_cosines = []
    for _ in range(min(max_pairs, len(ids) * 20)):
        a, b = rng.sample(ids, 2)
        va = by_video[a]["video"]
        vb = by_video[b]["video"]
        ia = rng.randrange(va.shape[0])
        ib = rng.randrange(vb.shape[0])
        diff = va[ia] - vb[ib]
        diff_distances.append(float(diff.norm().item()))
        diff_cosines.append(float(F.cosine_similarity(va[ia].unsqueeze(0), vb[ib].unsqueeze(0), dim=-1).item()))

    label_vectors: dict[str, list[torch.Tensor]] = defaultdict(list)
    for entry in entries:
        label_vectors[str(entry["row"].get("label"))].append(entry["video"].mean(dim=0))
    centroids = {label: torch.stack(values).mean(dim=0) for label, values in label_vectors.items()}
    centroid_pairs = []
    labels = sorted(centroids)
    for i, label_a in enumerate(labels):
        for label_b in labels[i + 1 :]:
            centroid_pairs.append(float((centroids[label_a] - centroids[label_b]).norm().item()))

    adjacent_mean = _mean(adjacent_distances)
    diff_mean = _mean(diff_distances)
    adjacent_cos_mean = _mean(adjacent_cosines)
    diff_cos_mean = _mean(diff_cosines)
    collapse_reasons = []
    if float(norms.mean().item()) <= 1e-6:
        collapse_reasons.append("near_zero_norm")
    if float(global_std.mean().item()) <= 1e-5:
        collapse_reasons.append("near_zero_global_std")
    if diff_cos_mean is not None and diff_cos_mean >= 0.999:
        collapse_reasons.append("different_videos_highly_identical")
    if diff_mean is not None and diff_mean <= 1e-4:
        collapse_reasons.append("different_videos_near_zero_distance")
    continuity_ok = adjacent_mean is not None and diff_mean is not None and adjacent_mean < diff_mean
    distinguish_ok = not collapse_reasons and diff_mean is not None and diff_mean > 1e-4
    meets_minimum = bool(continuity_ok and distinguish_ok)
    return {
        "sample_count": len(entries),
        "window_count": int(all_video.shape[0]),
        "feature_dim": int(all_video.shape[-1]),
        "feature_mean": float(all_video.mean().item()),
        "feature_std": float(all_video.std(unbiased=False).item()),
        "per_dim_std_mean": float(global_std.mean().item()),
        "norm_mean": float(norms.mean().item()),
        "norm_std": float(norms.std(unbiased=False).item()),
        "adjacent_window_distance_mean": adjacent_mean,
        "different_video_window_distance_mean": diff_mean,
        "adjacent_window_cosine_mean": adjacent_cos_mean,
        "different_video_window_cosine_mean": diff_cos_mean,
        "event_label_count": len(label_vectors),
        "event_centroid_distance_mean": _mean(centroid_pairs),
        "label_counts": dict(Counter(str(row.get("label")) for row in manifest_rows)),
        "collapse_reasons": collapse_reasons,
        "continuity_ok": bool(continuity_ok),
        "distinguishability_ok": bool(distinguish_ok),
        "meets_minimum_training_gate": meets_minimum,
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(torch.tensor(values, dtype=torch.float32).mean().item())


def build_pairs(manifest_rows: list[dict[str, Any]], seed: int = 20260717) -> list[WindowPair]:
    rng = random.Random(seed)
    rows_by_id = {row["youtube_id"]: row for row in manifest_rows}
    ids = sorted(rows_by_id)
    payload_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for row in manifest_rows:
        audio, audio_ts = load_feature(Path(row["audio_feature_path"]), "audio_embedding")
        video, video_ts = load_feature(Path(row["video_feature_path"]), "video_features")
        if audio.shape[0] != video.shape[0] or not torch.allclose(audio_ts, video_ts, atol=1e-5, rtol=0.0):
            raise ValueError(f"音视频窗口不一致：{row['youtube_id']}")
        payload_cache[row["youtube_id"]] = (audio, video, audio_ts, video_ts)

    pairs: list[WindowPair] = []
    for youtube_id in ids:
        row = rows_by_id[youtube_id]
        audio, _, timestamps, _ = payload_cache[youtube_id]
        window_count = int(audio.shape[0])
        wrong_candidates = [item for item in ids if item != youtube_id]
        for index in range(window_count):
            start = float(timestamps[index, 0].item())
            end = float(timestamps[index, 1].item())
            base = {
                "dataset": "AVE_HF",
                "youtube_id": youtube_id,
                "label": row.get("label"),
                "split": row.get("split", "train"),
                "window_index": index,
                "window_start": start,
                "window_end": end,
                "audio_feature_path": row["audio_feature_path"],
                "video_feature_path": row["video_feature_path"],
                "video_window_index": index,
            }
            pairs.append(WindowPair(**base, pair_type="positive", offset=0.0, target=1, audio_window_index=index))
            if index > 0:
                pairs.append(WindowPair(**base, pair_type="shifted_negative", offset=-0.5, target=0, audio_window_index=index - 1))
            if index + 1 < window_count:
                pairs.append(WindowPair(**base, pair_type="shifted_negative", offset=0.5, target=0, audio_window_index=index + 1))
            wrong_id = rng.choice(wrong_candidates)
            wrong_audio, _, _, _ = payload_cache[wrong_id]
            wrong_index = min(index, int(wrong_audio.shape[0]) - 1)
            wrong_base = dict(base)
            wrong_base["audio_feature_path"] = rows_by_id[wrong_id]["audio_feature_path"]
            pairs.append(
                WindowPair(
                    **wrong_base,
                    pair_type="wrong_audio_negative",
                    offset=None,
                    target=0,
                    audio_window_index=wrong_index,
                    negative_source_youtube_id=wrong_id,
                )
            )
            pairs.append(WindowPair(**base, pair_type="silence_negative", offset=None, target=0, audio_window_index=None))
    return pairs


def split_videos(manifest_rows: list[dict[str, Any]], train_ratio: float = 0.8, seed: int = 20260717) -> dict[str, Any]:
    rng = random.Random(seed)
    by_label: dict[str, list[str]] = defaultdict(list)
    for row in manifest_rows:
        by_label[str(row.get("label"))].append(row["youtube_id"])
    train_ids: set[str] = set()
    val_ids: set[str] = set()
    for label, ids in sorted(by_label.items()):
        unique_ids = sorted(set(ids))
        rng.shuffle(unique_ids)
        if len(unique_ids) == 1:
            train_ids.add(unique_ids[0])
            continue
        train_count = max(1, min(len(unique_ids) - 1, round(len(unique_ids) * train_ratio)))
        train_ids.update(unique_ids[:train_count])
        val_ids.update(unique_ids[train_count:])
    if train_ids & val_ids:
        raise ValueError("train/val youtube_id 泄漏")
    return {
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "label_split_counts": {
            label: {
                "train": sum(1 for item in ids if item in train_ids),
                "val": sum(1 for item in ids if item in val_ids),
                "total": len(set(ids)),
            }
            for label, ids in sorted(by_label.items())
        },
        "labels_missing_val": sorted(label for label, ids in by_label.items() if not any(item in val_ids for item in ids)),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_pair_jsonl(path: Path, pairs: list[WindowPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")


def load_pair_tensor(pair: WindowPair | dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(pair, dict):
        pair = WindowPair(**pair)
    video, _ = load_feature(Path(pair.video_feature_path), "video_features")
    video_vec = video[pair.video_window_index]
    if pair.pair_type == "silence_negative":
        audio_vec = torch.zeros_like(video_vec)
    else:
        audio, _ = load_feature(Path(pair.audio_feature_path), "audio_embedding")
        if pair.audio_window_index is None:
            raise ValueError("非静音 pair 缺少 audio_window_index")
        audio_vec = audio[pair.audio_window_index]
    label = torch.tensor(float(pair.target), dtype=torch.float32)
    return audio_vec, video_vec, label


def collect_pair_tensors(pairs: list[WindowPair]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    audio_rows, video_rows, labels = [], [], []
    for pair in pairs:
        audio, video, label = load_pair_tensor(pair)
        audio_rows.append(audio)
        video_rows.append(video)
        labels.append(label)
    return torch.stack(audio_rows), torch.stack(video_rows), torch.stack(labels)


def train_projector(
    train_pairs: list[WindowPair],
    output_root: Path,
    steps: int,
    project_dim: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    audio, video, labels = collect_pair_tensors(train_pairs)
    model = AVProjector(input_dim=audio.shape[-1], project_dim=project_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
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
                    "positive_logit_mean": float(pos.mean().item()),
                    "negative_logit_mean": float(neg.mean().item()),
                }
            )
    checkpoint = output_root / f"projector_checkpoint_{steps}step.pt"
    torch.save(
        {
            "audio_proj.weight": model.audio_proj.weight.detach().cpu(),
            "video_proj.weight": model.video_proj.weight.detach().cpu(),
            "metadata": {"steps": steps, "project_dim": project_dim, "lr": lr, "seed": seed, "input_dim": int(audio.shape[-1])},
        },
        checkpoint,
    )
    reloaded = load_projector(checkpoint)
    with torch.no_grad():
        reload_diff = float((model(audio, video) - reloaded(audio, video)).abs().max().item())
    return {
        "steps": steps,
        "checkpoint_path": str(checkpoint),
        "history": history,
        "reload_max_abs_similarity_diff": reload_diff,
        "checkpoint_reload_passed": reload_diff <= 1e-6,
        "frozen_components": {
            "BEATs": "frozen_precomputed_audio_features_only",
            "video_feature_extractor": "frozen_precomputed_rgb_statistics_only",
            "M4": "not_loaded_frozen_no_grad",
            "Gate": "not_loaded_frozen_no_grad",
            "Fusion": "not_loaded_frozen_no_grad",
            "trainable_parameter_names": ["audio_proj.weight", "video_proj.weight"],
        },
    }


def load_projector(path: Path) -> AVProjector:
    state = torch.load(path, map_location="cpu", weights_only=True)
    audio_weight = state["audio_proj.weight"].float()
    video_weight = state["video_proj.weight"].float()
    model = AVProjector(input_dim=audio_weight.shape[1], project_dim=audio_weight.shape[0])
    with torch.no_grad():
        model.audio_proj.weight.copy_(audio_weight)
        model.video_proj.weight.copy_(video_weight)
    model.eval()
    return model


@torch.no_grad()
def evaluate_pairs(model: AVProjector, pairs: list[WindowPair]) -> dict[str, Any]:
    rows = []
    by_anchor: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_label_margin: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        audio, video, _ = load_pair_tensor(pair)
        score = float(model(audio.unsqueeze(0), video.unsqueeze(0)).item())
        rows.append({"youtube_id": pair.youtube_id, "label": pair.label, "window_index": pair.window_index, "pair_type": pair.pair_type, "score": score})
        by_anchor[(pair.youtube_id, pair.window_index)][pair.pair_type].append(score)
    criteria = Counter()
    margins = []
    for (youtube_id, window_index), scores in by_anchor.items():
        if "positive" not in scores:
            continue
        pos = scores["positive"][0]
        negative_scores = [value for key, values in scores.items() if key != "positive" for value in values]
        if not negative_scores:
            continue
        best_neg = max(negative_scores)
        margin = pos - best_neg
        margins.append(margin)
        label = next(row["label"] for row in rows if row["youtube_id"] == youtube_id and row["window_index"] == window_index)
        by_label_margin[str(label)].append(margin)
        if margin > 0:
            criteria["positive_ranked_first"] += 1
        for key in ("wrong_audio_negative", "silence_negative", "shifted_negative"):
            if key in scores and pos > max(scores[key]):
                criteria[f"positive_gt_{key}"] += 1
    by_type: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_type[row["pair_type"]].append(row["score"])
    anchor_count = len(margins)
    return {
        "anchor_count": anchor_count,
        "pair_count": len(pairs),
        "pair_type_mean_scores": {key: _mean(values) for key, values in sorted(by_type.items())},
        "original_ranked_first_ratio": criteria["positive_ranked_first"] / anchor_count if anchor_count else 0.0,
        "criteria_counts": dict(criteria),
        "margin_mean": _mean(margins),
        "margin_min": min(margins) if margins else None,
        "label_margin_mean": {key: _mean(values) for key, values in sorted(by_label_margin.items())},
        "all_scores_finite": all(torch.isfinite(torch.tensor(row["score"])).item() for row in rows),
    }


def summarize_pairs(pairs: list[WindowPair], split: dict[str, Any]) -> dict[str, Any]:
    by_split = {
        "train": [pair for pair in pairs if pair.youtube_id in set(split["train_ids"])],
        "val": [pair for pair in pairs if pair.youtube_id in set(split["val_ids"])],
    }
    return {
        "total_pair_count": len(pairs),
        "pair_type_counts": dict(Counter(pair.pair_type for pair in pairs)),
        "train_pair_count": len(by_split["train"]),
        "val_pair_count": len(by_split["val"]),
        "train_pair_type_counts": dict(Counter(pair.pair_type for pair in by_split["train"])),
        "val_pair_type_counts": dict(Counter(pair.pair_type for pair in by_split["val"])),
        "split": split,
    }


def write_report(path: Path, video_diag: dict[str, Any], pair_summary: dict[str, Any], reports: list[dict[str, Any]]) -> None:
    lines = [
        "# AVE_HF 基础 projector 诊断与训练报告",
        "",
        "本轮只使用离线 BEATs 音频特征和 RGB 统计视频特征训练基础 projector，不修改 Gate、动态窗口、视频关注权重、Fusion 或 M4 主体。",
        "",
        "## 视频特征诊断",
        f"- 最低训练门槛：{video_diag['meets_minimum_training_gate']}",
        f"- collapse 原因：`{video_diag['collapse_reasons']}`",
        f"- 特征均值/标准差：{video_diag['feature_mean']:.6f} / {video_diag['feature_std']:.6f}",
        f"- 范数均值/标准差：{video_diag['norm_mean']:.6f} / {video_diag['norm_std']:.6f}",
        f"- 相邻窗口距离均值：{video_diag['adjacent_window_distance_mean']}",
        f"- 不同视频窗口距离均值：{video_diag['different_video_window_distance_mean']}",
        f"- 相邻窗口余弦均值：{video_diag['adjacent_window_cosine_mean']}",
        f"- 不同视频窗口余弦均值：{video_diag['different_video_window_cosine_mean']}",
        f"- 事件类别数：{video_diag['event_label_count']}",
        "",
        "## 配对与划分",
        f"- 总 pair 数：{pair_summary['total_pair_count']}",
        f"- pair 类型：`{pair_summary['pair_type_counts']}`",
        f"- train/val 视频数：{pair_summary['split']['train_count']} / {pair_summary['split']['val_count']}",
        f"- 缺少 val 的稀有类别：`{pair_summary['split']['labels_missing_val']}`",
        "",
        "## 训练与验证",
    ]
    for report in reports:
        val = report["validation"]
        lines.extend(
            [
                f"- {report['steps']} step: reload={report['checkpoint_reload_passed']}, "
                f"val_rank1={val['original_ranked_first_ratio']:.4f}, "
                f"margin_mean={val['margin_mean']}, scores=`{val['pair_type_mean_scores']}`",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = Path(args.feature_manifest).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(manifest)
    video_diag = diagnose_video_features(rows, seed=args.seed)
    write_json(output_root / "video_feature_diagnostics.json", video_diag)
    if not video_diag["meets_minimum_training_gate"]:
        summary = {"status": "stopped_video_features_failed", "video_diagnostics": video_diag}
        write_json(output_root / "run_summary.json", summary)
        return summary

    pairs = build_pairs(rows, seed=args.seed)
    split = split_videos(rows, train_ratio=args.train_ratio, seed=args.seed)
    train_ids = set(split["train_ids"])
    val_ids = set(split["val_ids"])
    train_pairs = [pair for pair in pairs if pair.youtube_id in train_ids]
    val_pairs = [pair for pair in pairs if pair.youtube_id in val_ids]
    pair_summary = summarize_pairs(pairs, split)
    write_pair_jsonl(output_root / "projector_window_pairs.jsonl", pairs)
    write_pair_jsonl(output_root / "projector_window_pairs_train.jsonl", train_pairs)
    write_pair_jsonl(output_root / "projector_window_pairs_val.jsonl", val_pairs)
    write_json(output_root / "projector_pair_split_summary.json", pair_summary)

    reports = []
    for steps in args.steps:
        train_report = train_projector(train_pairs, output_root, steps=steps, project_dim=args.project_dim, lr=args.lr, seed=args.seed)
        model = load_projector(Path(train_report["checkpoint_path"]))
        train_eval = evaluate_pairs(model, train_pairs)
        val_eval = evaluate_pairs(model, val_pairs)
        report = {**train_report, "train_evaluation": train_eval, "validation": val_eval}
        write_json(output_root / f"projector_{steps}step_eval.json", report)
        reports.append(report)

    summary = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "video_diagnostics": video_diag,
        "pair_summary": pair_summary,
        "training_reports": reports,
        "gate_or_dynamic_window_modified": False,
    }
    write_json(output_root / "run_summary.json", summary)
    write_report(output_root / "projector_baseline_report.md", video_diag, pair_summary, reports)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", default=str(DEFAULT_FEATURE_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--steps", type=int, nargs="+", default=[2, 20])
    parser.add_argument("--project-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps({"status": summary["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
