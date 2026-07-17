#!/usr/bin/env python
"""训练 AVE_HF 语义音画匹配 projector。

本脚本只处理任务A：判断声音和画面是否属于同类事件。它不会使用静音作为
余弦负样本，也不会把 ±0.5s 相邻窗口当作时间错位负样本。时间同步任务留给
后续单独处理。
"""

from __future__ import annotations

import argparse
import importlib.util
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
DEFAULT_OLD_PAIR_MANIFEST = INTERSUIT_ROOT / "harness/artifacts/ave_hf_projector_baseline/projector_window_pairs.jsonl"
DEFAULT_OUTPUT_ROOT = INTERSUIT_ROOT / "harness/artifacts/ave_hf_semantic_projector"


def import_baseline_module():
    path = INTERSUIT_ROOT / "scripts/train_ave_hf_projector_baseline.py"
    spec = importlib.util.spec_from_file_location("ave_hf_projector_baseline_for_semantic", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


baseline = import_baseline_module()


@dataclass(frozen=True)
class SemanticPair:
    dataset: str
    youtube_id: str
    label: str
    window_index: int
    window_start: float
    window_end: float
    pair_type: str
    target: int
    audio_feature_path: str
    video_feature_path: str
    audio_window_index: int
    video_window_index: int
    source_youtube_id: str
    source_label: str
    target_label: str
    negative_kind: str | None = None


class AVProjector(nn.Module):
    def __init__(self, input_dim: int = 768, project_dim: int = 128) -> None:
        super().__init__()
        self.audio_proj = nn.Linear(input_dim, project_dim, bias=False)
        self.video_proj = nn.Linear(input_dim, project_dim, bias=False)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(0.07)))

    def encode(self, audio: torch.Tensor, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        audio_z = F.normalize(self.audio_proj(audio.float()), dim=-1, eps=1e-6)
        video_z = F.normalize(self.video_proj(video.float()), dim=-1, eps=1e-6)
        return audio_z, video_z

    def similarity(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        audio_z, video_z = self.encode(audio, video)
        return (audio_z * video_z).sum(dim=-1)

    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(0.01, 0.5)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"manifest 为空：{path}")
    return rows


def audit_old_pairs(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False}
    rows = load_jsonl(path)
    same_label_wrong = 0
    wrong_total = 0
    shifted_total = 0
    shifted_suspect_false_negative = 0
    labels_by_id = {row["youtube_id"]: str(row.get("label")) for row in rows if row.get("pair_type") == "positive"}
    for row in rows:
        if row.get("pair_type") == "wrong_audio_negative":
            wrong_total += 1
            source = row.get("negative_source_youtube_id")
            if source is not None and labels_by_id.get(source) == str(row.get("label")):
                same_label_wrong += 1
        if row.get("pair_type") == "shifted_negative":
            shifted_total += 1
            shifted_suspect_false_negative += 1
    return {
        "available": True,
        "old_pair_count": len(rows),
        "old_wrong_audio_count": wrong_total,
        "old_wrong_audio_same_label_count": same_label_wrong,
        "old_wrong_audio_same_label_ratio": same_label_wrong / wrong_total if wrong_total else 0.0,
        "old_shifted_negative_count": shifted_total,
        "old_shifted_suspect_false_negative_count": shifted_suspect_false_negative,
        "old_silence_negative_count": sum(1 for row in rows if row.get("pair_type") == "silence_negative"),
        "decision": "exclude_same_label_wrong_shifted_and_silence_from_semantic_training",
    }


def load_feature(path: Path, key: str) -> tuple[torch.Tensor, torch.Tensor]:
    return baseline.load_feature(path, key)


def build_semantic_pairs(rows: list[dict[str, Any]], negatives_per_positive: int, seed: int) -> list[SemanticPair]:
    rng = random.Random(seed)
    rows_by_id = {row["youtube_id"]: row for row in rows}
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row["label"])].append(row)
    labels = sorted(by_label)
    payload_cache = {}
    for row in rows:
        audio, audio_ts = load_feature(Path(row["audio_feature_path"]), "audio_embedding")
        video, video_ts = load_feature(Path(row["video_feature_path"]), "video_features")
        if audio.shape[0] != video.shape[0] or not torch.allclose(audio_ts, video_ts, atol=1e-5, rtol=0.0):
            raise ValueError(f"音视频窗口不一致：{row['youtube_id']}")
        payload_cache[row["youtube_id"]] = (audio, video, audio_ts)

    pairs: list[SemanticPair] = []
    removed_same_label_wrong = 0
    for row in rows:
        youtube_id = row["youtube_id"]
        label = str(row["label"])
        audio, _, timestamps = payload_cache[youtube_id]
        neg_pool = [candidate for other_label in labels if other_label != label for candidate in by_label[other_label]]
        if not neg_pool:
            continue
        for window_index in range(audio.shape[0]):
            start = float(timestamps[window_index, 0].item())
            end = float(timestamps[window_index, 1].item())
            common = {
                "dataset": "AVE_HF",
                "youtube_id": youtube_id,
                "label": label,
                "window_index": window_index,
                "window_start": start,
                "window_end": end,
                "video_feature_path": row["video_feature_path"],
                "video_window_index": window_index,
                "target_label": label,
            }
            pairs.append(
                SemanticPair(
                    **common,
                    pair_type="semantic_positive",
                    target=1,
                    audio_feature_path=row["audio_feature_path"],
                    audio_window_index=window_index,
                    source_youtube_id=youtube_id,
                    source_label=label,
                )
            )
            for _ in range(negatives_per_positive):
                neg = rng.choice(neg_pool)
                neg_label = str(neg["label"])
                if neg_label == label:
                    removed_same_label_wrong += 1
                    continue
                neg_audio, _, _ = payload_cache[neg["youtube_id"]]
                neg_index = min(window_index, int(neg_audio.shape[0]) - 1)
                pairs.append(
                    SemanticPair(
                        **common,
                        pair_type="cross_label_negative",
                        target=0,
                        audio_feature_path=neg["audio_feature_path"],
                        audio_window_index=neg_index,
                        source_youtube_id=neg["youtube_id"],
                        source_label=neg_label,
                        negative_kind="different_label",
                    )
                )
    setattr(build_semantic_pairs, "last_removed_same_label_wrong", removed_same_label_wrong)
    return pairs


def split_videos(rows: list[dict[str, Any]], train_ratio: float, seed: int) -> dict[str, Any]:
    return baseline.split_videos(rows, train_ratio=train_ratio, seed=seed)


def select_overfit_rows(rows: list[dict[str, Any]], class_count: int, videos_per_class: int) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row["label"])].append(row)
    eligible = [(label, items) for label, items in sorted(by_label.items()) if len(items) >= videos_per_class]
    selected = []
    for _, items in eligible[:class_count]:
        selected.extend(sorted(items, key=lambda row: row["youtube_id"])[:videos_per_class])
    if len({str(row["label"]) for row in selected}) < class_count:
        raise ValueError("可用于最小过拟合的类别不足")
    return selected


def load_pair_tensor(pair: SemanticPair | dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(pair, dict):
        pair = SemanticPair(**pair)
    audio, _ = load_feature(Path(pair.audio_feature_path), "audio_embedding")
    video, _ = load_feature(Path(pair.video_feature_path), "video_features")
    return audio[pair.audio_window_index], video[pair.video_window_index], torch.tensor(float(pair.target))


def collect_tensors(pairs: list[SemanticPair]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    audio_rows, video_rows, labels = [], [], []
    for pair in pairs:
        audio, video, label = load_pair_tensor(pair)
        audio_rows.append(audio)
        video_rows.append(video)
        labels.append(label)
    return torch.stack(audio_rows), torch.stack(video_rows), torch.stack(labels)


def train_model(
    pairs: list[SemanticPair],
    steps: int,
    project_dim: int,
    lr: float,
    margin: float,
    seed: int,
) -> tuple[AVProjector, list[dict[str, Any]]]:
    torch.manual_seed(seed)
    audio, video, labels = collect_tensors(pairs)
    model = AVProjector(input_dim=audio.shape[-1], project_dim=project_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        sim = model.similarity(audio, video)
        logits = sim / model.temperature()
        contrastive = F.binary_cross_entropy_with_logits(logits, labels)
        pos = sim[labels > 0.5]
        neg = sim[labels < 0.5]
        if pos.numel() and neg.numel():
            ranking = F.relu(margin - pos.mean() + neg.mean())
        else:
            ranking = torch.zeros((), dtype=sim.dtype)
        loss = contrastive + ranking
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "contrastive_loss": float(contrastive.item()),
                    "ranking_loss": float(ranking.item()),
                    "positive_similarity_mean": float(pos.mean().item()) if pos.numel() else None,
                    "negative_similarity_mean": float(neg.mean().item()) if neg.numel() else None,
                    "temperature": float(model.temperature().item()),
                }
            )
    return model, history


@torch.no_grad()
def evaluate_semantic(model: AVProjector, pairs: list[SemanticPair]) -> dict[str, Any]:
    by_anchor: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_label_margin: dict[str, list[float]] = defaultdict(list)
    by_type: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        audio, video, _ = load_pair_tensor(pair)
        score = float(model.similarity(audio.unsqueeze(0), video.unsqueeze(0)).item())
        by_anchor[(pair.youtube_id, pair.window_index)][pair.pair_type].append(score)
        by_type[pair.pair_type].append(score)
    rank_first = 0
    margins = []
    for (youtube_id, window_index), scores in by_anchor.items():
        if "semantic_positive" not in scores or "cross_label_negative" not in scores:
            continue
        pos = scores["semantic_positive"][0]
        best_neg = max(scores["cross_label_negative"])
        margin = pos - best_neg
        margins.append(margin)
        if margin > 0:
            rank_first += 1
        label = next(pair.label for pair in pairs if pair.youtube_id == youtube_id and pair.window_index == window_index)
        by_label_margin[label].append(margin)
    anchor_count = len(margins)
    all_scores = [value for values in by_type.values() for value in values]
    return {
        "anchor_count": anchor_count,
        "pair_count": len(pairs),
        "rank_first_ratio": rank_first / anchor_count if anchor_count else 0.0,
        "margin_mean": _mean(margins),
        "margin_min": min(margins) if margins else None,
        "pair_type_mean_scores": {key: _mean(values) for key, values in sorted(by_type.items())},
        "label_margin_mean": {key: _mean(values) for key, values in sorted(by_label_margin.items())},
        "all_scores_finite": all(torch.isfinite(torch.tensor(all_scores)).tolist()) if all_scores else True,
        "score_std": float(torch.tensor(all_scores).std(unbiased=False).item()) if all_scores else 0.0,
        "collapsed_scores": float(torch.tensor(all_scores).std(unbiased=False).item()) <= 1e-6 if all_scores else True,
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(torch.tensor(values, dtype=torch.float32).mean().item())


def save_checkpoint(path: Path, model: AVProjector, metadata: dict[str, Any]) -> bool:
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
    reloaded = AVProjector(input_dim=state["audio_proj.weight"].shape[1], project_dim=state["audio_proj.weight"].shape[0])
    with torch.no_grad():
        reloaded.audio_proj.weight.copy_(state["audio_proj.weight"])
        reloaded.video_proj.weight.copy_(state["video_proj.weight"])
        reloaded.log_temperature.copy_(state["log_temperature"])
    return True


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[SemanticPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def summarize_pairs(pairs: list[SemanticPair]) -> dict[str, Any]:
    return {
        "pair_count": len(pairs),
        "pair_type_counts": dict(Counter(pair.pair_type for pair in pairs)),
        "label_counts": dict(Counter(pair.label for pair in pairs if pair.pair_type == "semantic_positive")),
        "source_target_label_pairs": dict(Counter(f"{pair.source_label}->{pair.target_label}" for pair in pairs if pair.target == 0)),
        "contains_shifted_negative": any(pair.pair_type == "shifted_negative" for pair in pairs),
        "contains_silence_negative": any(pair.pair_type == "silence_negative" for pair in pairs),
        "same_label_negative_count": sum(1 for pair in pairs if pair.target == 0 and pair.source_label == pair.target_label),
        "removed_same_label_wrong_count": getattr(build_semantic_pairs, "last_removed_same_label_wrong", 0),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    overfit = summary["overfit"]
    validation = summary["validation"]
    lines = [
        "# AVE_HF 语义 projector 训练报告",
        "",
        "本轮只训练任务A：音画语义匹配。未使用静音余弦负样本，未使用 ±0.5s 时间错位负样本。",
        "",
        "## 假负样本审计",
        f"- 旧 pair 审计：`{summary['old_pair_audit']}`",
        f"- 新语义 pair：`{summary['semantic_pair_summary']}`",
        "",
        "## 最小过拟合",
        f"- 类别/视频：{summary['overfit_config']}",
        f"- rank-first：{overfit['evaluation']['rank_first_ratio']}",
        f"- margin mean：{overfit['evaluation']['margin_mean']}",
        f"- 分数：`{overfit['evaluation']['pair_type_mean_scores']}`",
        "",
        "## 独立验证",
        f"- train/val 视频数：{summary['split']['train_count']} / {summary['split']['val_count']}",
        f"- rank-first：{validation['evaluation']['rank_first_ratio']}",
        f"- margin mean：{validation['evaluation']['margin_mean']}",
        f"- 分数：`{validation['evaluation']['pair_type_mean_scores']}`",
        f"- NaN/Inf：{validation['evaluation']['all_scores_finite']}",
        f"- 分数塌缩：{validation['evaluation']['collapsed_scores']}",
        "",
        "## 结论",
        f"- 状态：{summary['decision']}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(Path(args.feature_manifest).resolve())
    old_audit = audit_old_pairs(Path(args.old_pair_manifest).resolve())

    semantic_pairs = build_semantic_pairs(rows, negatives_per_positive=args.negatives_per_positive, seed=args.seed)
    pair_summary = summarize_pairs(semantic_pairs)
    write_jsonl(output_root / "semantic_pairs.jsonl", semantic_pairs)
    write_json(output_root / "semantic_pair_summary.json", pair_summary)
    write_json(output_root / "old_pair_false_negative_audit.json", old_audit)

    overfit_rows = select_overfit_rows(rows, class_count=args.overfit_classes, videos_per_class=args.overfit_videos_per_class)
    overfit_pairs = build_semantic_pairs(overfit_rows, negatives_per_positive=args.negatives_per_positive, seed=args.seed)
    overfit_model, overfit_history = train_model(
        overfit_pairs,
        steps=args.overfit_steps,
        project_dim=args.project_dim,
        lr=args.lr,
        margin=args.margin,
        seed=args.seed,
    )
    overfit_eval = evaluate_semantic(overfit_model, overfit_pairs)
    save_checkpoint(output_root / f"semantic_overfit_projector_{args.overfit_steps}step.pt", overfit_model, {"stage": "overfit"})

    split = split_videos(rows, train_ratio=args.train_ratio, seed=args.seed)
    train_ids = set(split["train_ids"])
    val_ids = set(split["val_ids"])
    train_pairs = [pair for pair in semantic_pairs if pair.youtube_id in train_ids]
    val_pairs = [pair for pair in semantic_pairs if pair.youtube_id in val_ids]
    write_jsonl(output_root / "semantic_pairs_train.jsonl", train_pairs)
    write_jsonl(output_root / "semantic_pairs_val.jsonl", val_pairs)
    model, history = train_model(
        train_pairs,
        steps=args.steps,
        project_dim=args.project_dim,
        lr=args.lr,
        margin=args.margin,
        seed=args.seed,
    )
    train_eval = evaluate_semantic(model, train_pairs)
    val_eval = evaluate_semantic(model, val_pairs)
    save_checkpoint(output_root / f"semantic_projector_{args.steps}step.pt", model, {"stage": "validation"})

    decision = "semantic_validation_passed" if val_eval["rank_first_ratio"] >= args.pass_rank_first and (val_eval["margin_mean"] or -1.0) > 0 else "semantic_validation_not_passed"
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "old_pair_audit": old_audit,
        "semantic_pair_summary": pair_summary,
        "overfit_config": {"classes": args.overfit_classes, "videos_per_class": args.overfit_videos_per_class, "steps": args.overfit_steps},
        "overfit": {"history": overfit_history, "evaluation": overfit_eval},
        "split": split,
        "training": {"history": history, "evaluation": train_eval},
        "validation": {"evaluation": val_eval},
        "decision": decision,
        "gate_or_dynamic_window_modified": False,
    }
    write_json(output_root / "semantic_projector_run_summary.json", summary)
    write_report(output_root / "semantic_projector_report.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", default=str(DEFAULT_FEATURE_MANIFEST))
    parser.add_argument("--old-pair-manifest", default=str(DEFAULT_OLD_PAIR_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    parser.add_argument("--overfit-classes", type=int, default=4)
    parser.add_argument("--overfit-videos-per-class", type=int, default=4)
    parser.add_argument("--overfit-steps", type=int, default=100)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--project-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--pass-rank-first", type=float, default=0.6)
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps({"decision": summary["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
