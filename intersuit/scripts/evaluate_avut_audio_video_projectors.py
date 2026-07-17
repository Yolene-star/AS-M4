#!/usr/bin/env python
"""评估 AVUT audio/video projector 的训练集表现和按视频留一泛化。

本脚本只读取离线窗口特征、pair manifest 和 projector checkpoint。
它不会修改 Gate、视频关注权重、M4 主体或正式融合路径。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_avut_audio_video_projectors import (  # noqa: E402
    AVProjector,
    collect_training_tensors,
    load_pair_tensors,
    read_pair_manifest,
    resolve_repo_path,
)


DEFAULT_PAIR_MANIFEST = INTERSUIT_ROOT / "harness/artifacts/avut_projector_training_real/projector_pair_manifest.jsonl"
DEFAULT_CHECKPOINT = INTERSUIT_ROOT / "harness/artifacts/avut_projector_training_real/projector_checkpoint_20step.pt"
DEFAULT_OUTPUT_ROOT = INTERSUIT_ROOT / "harness/artifacts/avut_projector_eval"
CONDITION_ORDER = ("original", "silence", "wrong_audio", "shift_plus_0_5", "shift_minus_0_5")


def load_projector_checkpoint(path: Path) -> AVProjector:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "audio_proj.weight" not in state or "video_proj.weight" not in state:
        raise ValueError(f"projector checkpoint lacks audio/video projection weights: {path}")
    audio_weight = state["audio_proj.weight"].float()
    video_weight = state["video_proj.weight"].float()
    if audio_weight.ndim != 2 or video_weight.ndim != 2:
        raise ValueError("projector weights must be rank-2 tensors")
    if audio_weight.shape != video_weight.shape:
        raise ValueError(f"audio/video projector weight shape mismatch: {audio_weight.shape} vs {video_weight.shape}")
    model = AVProjector(input_dim=audio_weight.shape[1], project_dim=audio_weight.shape[0])
    with torch.no_grad():
        model.audio_proj.weight.copy_(audio_weight)
        model.video_proj.weight.copy_(video_weight)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


@torch.no_grad()
def score_record(model: AVProjector, record: dict[str, Any], temperature: float = 0.07) -> dict[str, Any]:
    audio, video, labels = load_pair_tensors(record)
    similarity = model(audio, video)
    logits = similarity / float(temperature)
    probabilities = torch.sigmoid(logits)
    floating_values = (similarity, logits, probabilities)
    finite = all(torch.isfinite(value).all().item() for value in floating_values)
    saturated = ((probabilities <= 1e-3) | (probabilities >= 1.0 - 1e-3)).float().mean()
    return {
        "sample_id": record["sample_id"],
        "condition": record["audio_condition"],
        "label": int(record["label"]),
        "window_count": int(similarity.numel()),
        "mean_similarity": float(similarity.mean().item()),
        "mean_logit": float(logits.mean().item()),
        "mean_probability": float(probabilities.mean().item()),
        "min_probability": float(probabilities.min().item()),
        "max_probability": float(probabilities.max().item()),
        "saturation_fraction": float(saturated.item()),
        "finite": bool(finite),
    }


def summarize_scores(score_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[float]] = defaultdict(list)
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in score_rows:
        by_condition[row["condition"]].append(row["mean_logit"])
        by_sample[row["sample_id"]][row["condition"]] = row

    sample_summaries = []
    criteria_counts = {
        "original_ranked_first": 0,
        "original_gt_wrong_audio": 0,
        "original_gt_silence": 0,
        "original_gt_shift_plus_0_5": 0,
        "original_gt_shift_minus_0_5": 0,
        "positive_margin": 0,
    }
    for sample_id, conditions in sorted(by_sample.items()):
        missing = [condition for condition in CONDITION_ORDER if condition not in conditions]
        if missing:
            raise ValueError(f"sample {sample_id} missing conditions: {missing}")
        condition_scores = {condition: conditions[condition]["mean_logit"] for condition in CONDITION_ORDER}
        ranked = sorted(condition_scores.items(), key=lambda item: item[1], reverse=True)
        original_score = condition_scores["original"]
        best_negative = max(score for condition, score in condition_scores.items() if condition != "original")
        margin = original_score - best_negative
        if ranked[0][0] == "original":
            criteria_counts["original_ranked_first"] += 1
        for condition in ("wrong_audio", "silence", "shift_plus_0_5", "shift_minus_0_5"):
            if original_score > condition_scores[condition]:
                criteria_counts[f"original_gt_{condition}"] += 1
        if margin > 0:
            criteria_counts["positive_margin"] += 1
        sample_summaries.append(
            {
                "sample_id": sample_id,
                "condition_scores": condition_scores,
                "ranked_conditions": ranked,
                "best_condition": ranked[0][0],
                "original_rank": 1 + [condition for condition, _ in ranked].index("original"),
                "original_margin": float(margin),
            }
        )

    condition_means = {
        condition: float(torch.tensor(values, dtype=torch.float32).mean().item())
        for condition, values in sorted(by_condition.items())
    }
    all_finite = all(row["finite"] for row in score_rows)
    saturation_mean = float(torch.tensor([row["saturation_fraction"] for row in score_rows]).mean().item())
    return {
        "condition_mean_logits": condition_means,
        "sample_summaries": sample_summaries,
        "criteria_counts": criteria_counts,
        "sample_count": len(sample_summaries),
        "all_scores_finite": bool(all_finite),
        "mean_saturation_fraction": saturation_mean,
        "max_saturation_fraction": max(row["saturation_fraction"] for row in score_rows) if score_rows else 0.0,
        "score_rows": score_rows,
    }


def evaluate_checkpoint(pair_manifest: Path, checkpoint: Path, temperature: float = 0.07) -> dict[str, Any]:
    model = load_projector_checkpoint(checkpoint)
    rows = read_pair_manifest(pair_manifest)
    score_rows = [score_record(model, row, temperature=temperature) for row in rows]
    summary = summarize_scores(score_rows)
    summary.update(
        {
            "checkpoint": str(checkpoint),
            "pair_manifest": str(pair_manifest),
            "projector_frozen": all(not param.requires_grad for param in model.parameters()),
            "projector_shapes": {
                "audio_proj": list(model.audio_proj.weight.shape),
                "video_proj": list(model.video_proj.weight.shape),
            },
        }
    )
    return summary


def train_subset_model(
    rows: list[dict[str, Any]],
    input_dim: int,
    project_dim: int,
    steps: int,
    lr: float,
    seed: int,
) -> tuple[AVProjector, list[dict[str, Any]]]:
    torch.manual_seed(seed)
    audio, video, labels = collect_training_tensors(rows)
    model = AVProjector(input_dim=input_dim, project_dim=project_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(audio, video) / 0.07
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        history.append({"step": step, "loss": float(loss.item())})
    return model, history


def leave_one_video_out(
    pair_manifest: Path,
    output_root: Path,
    steps: int = 20,
    project_dim: int = 128,
    lr: float = 1e-3,
    seed: int = 20260717,
    temperature: float = 0.07,
) -> dict[str, Any]:
    rows = read_pair_manifest(pair_manifest)
    sample_ids = sorted({row["sample_id"] for row in rows})
    if len(sample_ids) < 2:
        raise ValueError("leave-one-video-out requires at least two samples")
    first_audio, _, _ = load_pair_tensors(rows[0])
    input_dim = int(first_audio.shape[-1])
    folds = []
    fold_root = output_root / "leave_one_out"
    fold_root.mkdir(parents=True, exist_ok=True)
    for fold_index, test_sample_id in enumerate(sample_ids):
        train_rows = [row for row in rows if row["sample_id"] != test_sample_id]
        test_rows = [row for row in rows if row["sample_id"] == test_sample_id]
        train_sample_ids = sorted({row["sample_id"] for row in train_rows})
        if test_sample_id in train_sample_ids:
            raise ValueError("test sample leaked into training set")
        model, history = train_subset_model(
            train_rows,
            input_dim=input_dim,
            project_dim=project_dim,
            steps=steps,
            lr=lr,
            seed=seed + fold_index,
        )
        score_rows = [score_record(model, row, temperature=temperature) for row in test_rows]
        summary = summarize_scores(score_rows)
        checkpoint_path = fold_root / f"fold_{fold_index}_{test_sample_id}_projector.pt"
        torch.save(
            {
                "audio_proj.weight": model.audio_proj.weight.detach().cpu(),
                "video_proj.weight": model.video_proj.weight.detach().cpu(),
                "metadata": {
                    "fold_index": fold_index,
                    "test_sample_id": test_sample_id,
                    "train_sample_ids": train_sample_ids,
                    "steps": steps,
                    "seed": seed + fold_index,
                },
            },
            checkpoint_path,
        )
        folds.append(
            {
                "fold_index": fold_index,
                "test_sample_id": test_sample_id,
                "train_sample_ids": train_sample_ids,
                "history": history,
                "checkpoint_path": str(checkpoint_path),
                "summary": summary,
                "no_video_leakage": test_sample_id not in train_sample_ids,
            }
        )

    sample_count = len(folds)
    aggregate_counts: dict[str, int] = defaultdict(int)
    margins = []
    for fold in folds:
        for key, value in fold["summary"]["criteria_counts"].items():
            aggregate_counts[key] += int(value)
        margins.extend(item["original_margin"] for item in fold["summary"]["sample_summaries"])
    return {
        "fold_count": len(folds),
        "steps": int(steps),
        "folds": folds,
        "aggregate_criteria_counts": dict(sorted(aggregate_counts.items())),
        "aggregate_criteria_rates": {
            key: float(value / sample_count) for key, value in sorted(aggregate_counts.items())
        },
        "mean_original_margin": float(torch.tensor(margins, dtype=torch.float32).mean().item()) if margins else None,
        "all_scores_finite": all(fold["summary"]["all_scores_finite"] for fold in folds),
        "no_video_leakage": all(fold["no_video_leakage"] for fold in folds),
    }


def write_markdown_report(path: Path, train_eval: dict[str, Any], loocv: dict[str, Any]) -> None:
    lines = [
        "# AVUT projector 训练集评估与5折留一验证",
        "",
        "## 训练集 checkpoint 评估",
        "",
        f"- checkpoint：{train_eval['checkpoint']}",
        f"- pair manifest：{train_eval['pair_manifest']}",
        f"- projector 冻结：{train_eval['projector_frozen']}",
        f"- 分数有限：{train_eval['all_scores_finite']}",
        f"- 平均饱和比例：{train_eval['mean_saturation_fraction']:.6f}",
        f"- 条件平均 logit：{train_eval['condition_mean_logits']}",
        f"- 正确音频排第一：{train_eval['criteria_counts']['original_ranked_first']}/{train_eval['sample_count']}",
        f"- 正 margin：{train_eval['criteria_counts']['positive_margin']}/{train_eval['sample_count']}",
        "",
        "## 5折按视频留一",
        "",
        f"- fold 数：{loocv['fold_count']}",
        f"- 无视频泄漏：{loocv['no_video_leakage']}",
        f"- 分数有限：{loocv['all_scores_finite']}",
        f"- 平均 original margin：{loocv['mean_original_margin']}",
        f"- 条件通过率：{loocv['aggregate_criteria_rates']}",
        "",
        "## 每条样本训练集排序",
        "",
    ]
    for item in train_eval["sample_summaries"]:
        lines.append(
            f"- {item['sample_id']}: best={item['best_condition']}, "
            f"rank(original)={item['original_rank']}, margin={item['original_margin']:.6f}"
        )
    lines.extend(["", "## 每折留一排序", ""])
    for fold in loocv["folds"]:
        sample = fold["summary"]["sample_summaries"][0]
        lines.append(
            f"- fold {fold['fold_index']} test={fold['test_sample_id']}: "
            f"best={sample['best_condition']}, rank(original)={sample['original_rank']}, "
            f"margin={sample['original_margin']:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pair_manifest = resolve_repo_path(args.pair_manifest)
    checkpoint = resolve_repo_path(args.projector_checkpoint)
    train_eval = evaluate_checkpoint(pair_manifest, checkpoint, temperature=args.temperature)
    loocv = leave_one_video_out(
        pair_manifest,
        output_root=output_root,
        steps=args.loocv_steps,
        project_dim=args.project_dim,
        lr=args.lr,
        seed=args.seed,
        temperature=args.temperature,
    )
    result = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_eval": train_eval,
        "leave_one_video_out": loocv,
        "route_hint": decide_route(train_eval, loocv),
    }
    (output_root / "projector_eval_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(output_root / "projector_eval_report.md", train_eval, loocv)
    return result


def decide_route(train_eval: dict[str, Any], loocv: dict[str, Any]) -> str:
    train_ok = (
        train_eval["criteria_counts"]["original_gt_wrong_audio"] == train_eval["sample_count"]
        and train_eval["criteria_counts"]["original_gt_silence"] == train_eval["sample_count"]
        and train_eval["criteria_counts"]["original_gt_shift_plus_0_5"] == train_eval["sample_count"]
        and train_eval["criteria_counts"]["original_gt_shift_minus_0_5"] == train_eval["sample_count"]
    )
    loocv_ok = (
        loocv["aggregate_criteria_counts"].get("original_gt_wrong_audio", 0) == loocv["fold_count"]
        and loocv["aggregate_criteria_counts"].get("original_gt_silence", 0) == loocv["fold_count"]
        and loocv["aggregate_criteria_counts"].get("original_gt_shift_plus_0_5", 0) == loocv["fold_count"]
        and loocv["aggregate_criteria_counts"].get("original_gt_shift_minus_0_5", 0) == loocv["fold_count"]
    )
    if train_ok and loocv_ok:
        return "A_train_and_leave_one_out_effective"
    if train_ok:
        return "B_train_effective_leave_one_out_failed_expand_avut"
    return "C_train_failed_debug_labels_timestamps_pooling_normalization_loss"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", default=str(DEFAULT_PAIR_MANIFEST))
    parser.add_argument("--projector-checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--loocv-steps", type=int, default=20)
    parser.add_argument("--project-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--temperature", type=float, default=0.07)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        result = run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
