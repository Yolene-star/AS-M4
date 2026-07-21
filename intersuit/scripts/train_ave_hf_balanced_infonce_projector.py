#!/usr/bin/env python
"""训练 AVE_HF 类别均衡对称 InfoNCE projector。

本脚本固定上一轮 M4 CLIP 视频特征实验作为对照，不覆盖旧结果目录。它只改
训练采样和损失：类别均衡 mini-batch、同类非对角线屏蔽、对称 InfoNCE。
不加入 shifted negative，不加入 silence cosine negative，不修改 Gate、
动态窗口、视频关注权重或 M4 主体。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"
DEFAULT_FEATURE_MANIFEST = INTERSUIT_ROOT / "harness/artifacts/ave_hf_300_clip_window_features/ave_hf_clip_window_feature_manifest.jsonl"
DEFAULT_BASELINE_SUMMARY = INTERSUIT_ROOT / "harness/artifacts/ave_hf_semantic_projector_clip300/semantic_projector_run_summary.json"
DEFAULT_OUTPUT_ROOT = INTERSUIT_ROOT / "harness/artifacts/ave_hf_balanced_infonce_projector_clip300"
DEFAULT_CORE_LABELS = ("4", "8", "10", "11", "12", "13", "15", "17", "19", "20", "22", "25", "26", "27")
DEFAULT_HELDOUT_LABELS = ("1", "3", "7", "14")
DEFAULT_LOW_VAL_LABELS = ("0", "18")


def import_semantic_module():
    path = INTERSUIT_ROOT / "scripts/train_ave_hf_semantic_projector.py"
    spec = importlib.util.spec_from_file_location("ave_hf_semantic_projector_for_balanced_infonce", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


semantic = import_semantic_module()


class AVProjector(nn.Module):
    """Keep the baseline projector structure unchanged for this ablation."""

    def __init__(self, audio_input_dim: int = 768, video_input_dim: int = 1024, project_dim: int = 128) -> None:
        super().__init__()
        self.audio_proj = nn.Linear(audio_input_dim, project_dim, bias=False)
        self.video_proj = nn.Linear(video_input_dim, project_dim, bias=False)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(0.07)))

    def encode(self, audio: torch.Tensor, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        audio_z = F.normalize(self.audio_proj(audio.float()), dim=-1, eps=1e-6)
        video_z = F.normalize(self.video_proj(video.float()), dim=-1, eps=1e-6)
        return audio_z, video_z

    def similarity_matrix(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        audio_z, video_z = self.encode(audio, video)
        return audio_z @ video_z.T

    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(0.01, 0.5)


@dataclass(frozen=True)
class WindowItem:
    youtube_id: str
    label: str
    window_index: int
    audio: torch.Tensor
    video: torch.Tensor


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


def load_feature_pair(row: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    audio, audio_ts = semantic.load_feature(Path(row["audio_feature_path"]), "audio_embedding")
    video, video_ts = semantic.load_feature(Path(row["video_feature_path"]), "video_features")
    if audio.shape[0] != video.shape[0] or not torch.allclose(audio_ts, video_ts, atol=1e-5, rtol=0.0):
        raise ValueError(f"音视频窗口不一致：{row['youtube_id']}")
    return audio.float(), video.float()


def build_window_items(rows: list[dict[str, Any]], ids: set[str], labels: set[str]) -> list[WindowItem]:
    items = []
    for row in rows:
        youtube_id = str(row["youtube_id"])
        label = str(row["label"])
        if youtube_id not in ids or label not in labels:
            continue
        audio, video = load_feature_pair(row)
        for window_index in range(audio.shape[0]):
            items.append(WindowItem(youtube_id=youtube_id, label=label, window_index=window_index, audio=audio[window_index], video=video[window_index]))
    if not items:
        raise ValueError("没有可用窗口样本")
    return items


def group_items(items: list[WindowItem]) -> dict[str, dict[str, list[WindowItem]]]:
    grouped: dict[str, dict[str, list[WindowItem]]] = defaultdict(lambda: defaultdict(list))
    for item in items:
        grouped[item.label][item.youtube_id].append(item)
    return grouped


def sample_balanced_batch(
    grouped: dict[str, dict[str, list[WindowItem]]],
    labels_per_batch: int,
    videos_per_label: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str], list[int]]:
    eligible = [label for label, by_video in grouped.items() if len(by_video) >= videos_per_label]
    if len(eligible) < labels_per_batch:
        raise ValueError(f"可采样类别不足：需要 {labels_per_batch}，实际 {len(eligible)}")
    labels = rng.sample(sorted(eligible), labels_per_batch)
    batch_items: list[WindowItem] = []
    for label in labels:
        video_ids = rng.sample(sorted(grouped[label]), videos_per_label)
        for video_id in video_ids:
            batch_items.append(rng.choice(grouped[label][video_id]))
    audio = torch.stack([item.audio for item in batch_items])
    video = torch.stack([item.video for item in batch_items])
    return audio, video, [item.label for item in batch_items], [item.youtube_id for item in batch_items], [item.window_index for item in batch_items]


def same_label_non_diagonal_mask(labels: list[str]) -> torch.Tensor:
    same = torch.tensor([[left == right for right in labels] for left in labels], dtype=torch.bool)
    diagonal = torch.eye(len(labels), dtype=torch.bool)
    return same & ~diagonal


def masked_symmetric_infonce_loss(similarity: torch.Tensor, labels: list[str], temperature: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    logits = similarity / temperature
    mask = same_label_non_diagonal_mask(labels).to(logits.device)
    masked_logits = logits.masked_fill(mask, -1e9)
    targets = torch.arange(logits.shape[0], device=logits.device)
    audio_to_video = F.cross_entropy(masked_logits, targets)
    video_to_audio = F.cross_entropy(masked_logits.T, targets)
    loss = 0.5 * (audio_to_video + video_to_audio)
    with torch.no_grad():
        finite_logits = masked_logits[masked_logits > -1e8]
        stats = {
            "audio_to_video_loss": float(audio_to_video.item()),
            "video_to_audio_loss": float(video_to_audio.item()),
            "logits_min": float(finite_logits.min().item()) if finite_logits.numel() else None,
            "logits_max": float(finite_logits.max().item()) if finite_logits.numel() else None,
            "masked_same_label_entries": int(mask.sum().item()),
        }
    return loss, stats


def train_steps(
    train_items: list[WindowItem],
    steps: int,
    labels_per_batch: int,
    videos_per_label: int,
    project_dim: int,
    lr: float,
    seed: int,
) -> tuple[AVProjector, list[dict[str, Any]]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    grouped = group_items(train_items)
    audio_dim = train_items[0].audio.numel()
    video_dim = train_items[0].video.numel()
    model = AVProjector(audio_input_dim=audio_dim, video_input_dim=video_dim, project_dim=project_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    for step in range(1, steps + 1):
        audio, video, labels, video_ids, window_indices = sample_balanced_batch(grouped, labels_per_batch, videos_per_label, rng)
        optimizer.zero_grad(set_to_none=True)
        sim = model.similarity_matrix(audio, video)
        loss, loss_stats = masked_symmetric_infonce_loss(sim, labels, model.temperature())
        loss.backward()
        grad_norm_sq = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm_sq += float(param.grad.detach().float().pow(2).sum().item())
        optimizer.step()
        with torch.no_grad():
            audio_z, video_z = model.encode(audio, video)
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "temperature": float(model.temperature().item()),
                    "grad_norm": math.sqrt(grad_norm_sq),
                    "label_counts": dict(Counter(labels)),
                    "unique_video_count": len(set(video_ids)),
                    "batch_size": len(labels),
                    "audio_embedding_std": float(audio_z.std(unbiased=False).item()),
                    "video_embedding_std": float(video_z.std(unbiased=False).item()),
                    "finite": bool(torch.isfinite(sim).all().item()),
                    **loss_stats,
                }
            )
    return model, history


@torch.no_grad()
def evaluate_retrieval(model: AVProjector, items: list[WindowItem]) -> dict[str, Any]:
    audio = torch.stack([item.audio for item in items])
    video = torch.stack([item.video for item in items])
    labels = [item.label for item in items]
    sim = model.similarity_matrix(audio, video)
    label_equal = torch.tensor([[left == right for right in labels] for left in labels], dtype=torch.bool)
    diagonal = torch.eye(len(items), dtype=torch.bool)
    cross_label = ~label_equal

    def recall_at_k(matrix: torch.Tensor, k: int) -> float:
        topk = matrix.topk(k=min(k, matrix.shape[1]), dim=1).indices
        hits = []
        for row_idx in range(matrix.shape[0]):
            hits.append(bool(label_equal[row_idx, topk[row_idx]].any().item()))
        return sum(hits) / len(hits) if hits else 0.0

    a2v_r1 = recall_at_k(sim, 1)
    a2v_r5 = recall_at_k(sim, 5)
    v2a_r1 = recall_at_k(sim.T, 1)
    v2a_r5 = recall_at_k(sim.T, 5)

    exact_scores = sim.diag()
    cross_label_logits = sim.masked_fill(~cross_label, -1e9)
    strongest_cross = cross_label_logits.max(dim=1).values
    margins = exact_scores - strongest_cross
    rank_first = margins > 0
    by_label: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for idx, label in enumerate(labels):
        by_label[label]["margin"].append(float(margins[idx].item()))
        by_label[label]["rank_first"].append(float(rank_first[idx].item()))
        by_label[label]["positive_score"].append(float(exact_scores[idx].item()))
        by_label[label]["cross_negative_score"].append(float(strongest_cross[idx].item()))

    def mean(values: list[float]) -> float | None:
        return float(torch.tensor(values, dtype=torch.float32).mean().item()) if values else None

    label_stats = {
        label: {
            "window_count": len(values["margin"]),
            "rank_first_ratio": mean(values["rank_first"]),
            "margin_mean": mean(values["margin"]),
            "positive_score_mean": mean(values["positive_score"]),
            "strongest_cross_label_negative_mean": mean(values["cross_negative_score"]),
        }
        for label, values in sorted(by_label.items(), key=lambda item: (int(item[0]) if item[0].isdigit() else 9999, item[0]))
    }
    all_scores = sim.reshape(-1)
    return {
        "sample_count": len(items),
        "audio_to_video_class_recall_at_1": a2v_r1,
        "audio_to_video_class_recall_at_5": a2v_r5,
        "video_to_audio_class_recall_at_1": v2a_r1,
        "video_to_audio_class_recall_at_5": v2a_r5,
        "exact_pair_cross_label_rank_first": float(rank_first.float().mean().item()),
        "exact_pair_cross_label_margin_mean": float(margins.mean().item()),
        "exact_pair_cross_label_margin_min": float(margins.min().item()),
        "positive_score_mean": float(exact_scores.mean().item()),
        "strongest_cross_label_negative_mean": float(strongest_cross.mean().item()),
        "label_stats": label_stats,
        "all_scores_finite": bool(torch.isfinite(all_scores).all().item()),
        "score_std": float(all_scores.std(unbiased=False).item()),
        "collapsed_scores": float(all_scores.std(unbiased=False).item()) <= 1e-6,
    }


def save_checkpoint(path: Path, model: AVProjector, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "audio_proj.weight": model.audio_proj.weight.detach().cpu(),
            "video_proj.weight": model.video_proj.weight.detach().cpu(),
            "log_temperature": model.log_temperature.detach().cpu(),
            "metadata": metadata,
        },
        path,
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["audio_proj.weight"].shape == model.audio_proj.weight.shape
    assert state["video_proj.weight"].shape == model.video_proj.weight.shape


def load_projector_checkpoint(path: Path) -> AVProjector:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = AVProjector(
        audio_input_dim=state["audio_proj.weight"].shape[1],
        video_input_dim=state["video_proj.weight"].shape[1],
        project_dim=state["audio_proj.weight"].shape[0],
    )
    with torch.no_grad():
        model.audio_proj.weight.copy_(state["audio_proj.weight"])
        model.video_proj.weight.copy_(state["video_proj.weight"])
        model.log_temperature.copy_(state["log_temperature"])
    model.eval()
    return model


def write_report(path: Path, summary: dict[str, Any]) -> None:
    baseline = summary["fixed_baseline"]
    strict = summary["fixed_baseline_strict_metrics"]
    lines = [
        "# AVE_HF 类别均衡对称 InfoNCE projector 报告",
        "",
        "本实验固定上一轮 clip300 full-batch 结果为只读对照，旧结果目录和 checkpoint 未覆盖。本轮只改类别均衡 mini-batch 和同类屏蔽对称 InfoNCE，projector 结构保持线性无 bias。",
        "",
        "## 固定对照",
        f"- 对照 checkpoint：`{baseline['checkpoint']}`",
        f"- 旧采样负样本验证 rank-first：{baseline['validation_rank_first']}",
        f"- 旧采样负样本验证 margin：{baseline['validation_margin_mean']}",
        f"- 新严格核心类 exact rank-first：{strict['validation']['exact_pair_cross_label_rank_first']}",
        f"- 新严格核心类 margin：{strict['validation']['exact_pair_cross_label_margin_mean']}",
        f"- 新严格核心类 A->V / V->A class Recall@1：{strict['validation']['audio_to_video_class_recall_at_1']} / {strict['validation']['video_to_audio_class_recall_at_1']}",
        "",
        "## 本轮配置",
        f"- 核心类别：`{summary['config']['core_labels']}`",
        f"- 单独报告类别：无验证 `{summary['config']['heldout_no_val_labels']}`，验证过少 `{summary['config']['low_val_labels']}`",
        f"- batch：{summary['config']['labels_per_batch']} 类 x {summary['config']['videos_per_label']} 视频 = {summary['config']['batch_size']}",
        f"- projector：audio 768->128 / video 1024->128，bias=False，投影后 L2 normalize",
        f"- loss：同类非对角线屏蔽的对称 InfoNCE",
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
                f"- temperature：{run['history'][-1]['temperature']}",
                f"- 验证 exact rank-first：{val['exact_pair_cross_label_rank_first']}",
                f"- 验证 margin mean：{val['exact_pair_cross_label_margin_mean']}",
                f"- 验证正样本均分：{val['positive_score_mean']}",
                f"- 验证最强跨类别负样本均分：{val['strongest_cross_label_negative_mean']}",
                f"- A->V / V->A class Recall@1：{val['audio_to_video_class_recall_at_1']} / {val['video_to_audio_class_recall_at_1']}",
                f"- A->V / V->A class Recall@5：{val['audio_to_video_class_recall_at_5']} / {val['video_to_audio_class_recall_at_5']}",
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
    rows = load_jsonl(Path(args.feature_manifest).resolve())
    baseline = load_json(Path(args.baseline_summary).resolve())
    split = baseline["split"]
    core_labels = set(args.core_labels.split(","))
    train_items = build_window_items(rows, set(split["train_ids"]), core_labels)
    val_items = build_window_items(rows, set(split["val_ids"]), core_labels)
    baseline_checkpoint = Path(args.baseline_summary).resolve().parent / "semantic_projector_20step.pt"
    baseline_model = load_projector_checkpoint(baseline_checkpoint)
    baseline_strict = {"training": evaluate_retrieval(baseline_model, train_items), "validation": evaluate_retrieval(baseline_model, val_items)}
    epoch_steps = math.ceil(len(train_items) / (args.labels_per_batch * args.videos_per_label))
    run_steps = [2, 20, epoch_steps]
    runs = []
    for steps in run_steps:
        model, history = train_steps(
            train_items=train_items,
            steps=steps,
            labels_per_batch=args.labels_per_batch,
            videos_per_label=args.videos_per_label,
            project_dim=args.project_dim,
            lr=args.lr,
            seed=args.seed,
        )
        train_eval = evaluate_retrieval(model, train_items)
        val_eval = evaluate_retrieval(model, val_items)
        run_name = f"{steps}step" if steps != epoch_steps else "one_epoch"
        save_checkpoint(output_root / f"balanced_infonce_projector_{run_name}.pt", model, {"stage": run_name, "steps": steps})
        runs.append({"name": run_name, "steps": steps, "history": history, "training": train_eval, "validation": val_eval})

    final_val = runs[-1]["validation"]
    decision = (
        "balanced_infonce_ready_for_seed_check"
        if final_val["exact_pair_cross_label_rank_first"] >= args.pass_rank_first
        and final_val["exact_pair_cross_label_margin_mean"] >= args.pass_margin
        and final_val["positive_score_mean"] > final_val["strongest_cross_label_negative_mean"]
        else "balanced_infonce_not_passed"
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fixed_baseline": {
            "summary": str(Path(args.baseline_summary).resolve()),
            "checkpoint": str(Path(args.baseline_summary).resolve().parent / "semantic_projector_20step.pt"),
            "validation_rank_first": baseline["validation"]["evaluation"]["rank_first_ratio"],
            "validation_margin_mean": baseline["validation"]["evaluation"]["margin_mean"],
        },
        "fixed_baseline_strict_metrics": baseline_strict,
        "config": {
            "feature_manifest": str(Path(args.feature_manifest).resolve()),
            "core_labels": sorted(core_labels, key=lambda x: int(x)),
            "heldout_no_val_labels": list(DEFAULT_HELDOUT_LABELS),
            "low_val_labels": list(DEFAULT_LOW_VAL_LABELS),
            "labels_per_batch": args.labels_per_batch,
            "videos_per_label": args.videos_per_label,
            "batch_size": args.labels_per_batch * args.videos_per_label,
            "project_dim": args.project_dim,
            "lr": args.lr,
            "seed": args.seed,
            "epoch_steps": epoch_steps,
            "train_window_count": len(train_items),
            "val_window_count": len(val_items),
            "projector_structure_changed": False,
            "gate_or_dynamic_window_modified": False,
        },
        "split_label_counts": split["label_split_counts"],
        "runs": runs,
        "decision": decision,
    }
    write_json(output_root / "balanced_infonce_run_summary.json", summary)
    write_report(output_root / "balanced_infonce_report.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", default=str(DEFAULT_FEATURE_MANIFEST))
    parser.add_argument("--baseline-summary", default=str(DEFAULT_BASELINE_SUMMARY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--core-labels", default=",".join(DEFAULT_CORE_LABELS))
    parser.add_argument("--labels-per-batch", type=int, default=8)
    parser.add_argument("--videos-per-label", type=int, default=4)
    parser.add_argument("--project-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--pass-rank-first", type=float, default=0.65)
    parser.add_argument("--pass-margin", type=float, default=0.02)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    final_val = summary["runs"][-1]["validation"]
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "rank_first": final_val["exact_pair_cross_label_rank_first"],
                "margin": final_val["exact_pair_cross_label_margin_mean"],
                "report": str(Path(args.output_root).resolve() / "balanced_infonce_report.md"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
