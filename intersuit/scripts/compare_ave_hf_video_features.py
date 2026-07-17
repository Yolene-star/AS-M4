#!/usr/bin/env python
"""对比 AVE_HF 两组窗口级视频特征的基础区分能力。

本脚本只读取预计算特征，不训练 projector，不修改 Gate、动态窗口或正式
M4 推理路径。它用于比较 RGB 统计特征和 M4 CLIP 视觉塔特征是否具备
更好的事件语义区分趋势。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"
DEFAULT_RGB_MANIFEST = INTERSUIT_ROOT / "harness/artifacts/ave_hf_300_window_features/ave_hf_window_feature_manifest.jsonl"
DEFAULT_CLIP_MANIFEST = INTERSUIT_ROOT / "harness/artifacts/ave_hf_300_clip_window_features/ave_hf_clip_window_feature_manifest.jsonl"
DEFAULT_OUTPUT_ROOT = INTERSUIT_ROOT / "harness/artifacts/ave_hf_video_feature_comparison"


def import_baseline_module():
    path = INTERSUIT_ROOT / "scripts/train_ave_hf_projector_baseline.py"
    spec = importlib.util.spec_from_file_location("ave_hf_projector_baseline_for_video_compare", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


baseline = import_baseline_module()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"manifest 为空：{path}")
    return rows


def load_video(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    return baseline.load_feature(path, "video_features")


def select_rows(rows: list[dict[str, Any]], samples_per_label: int | None) -> list[dict[str, Any]]:
    if samples_per_label is None:
        return rows
    counts: Counter[str] = Counter()
    selected = []
    for row in rows:
        label = str(row.get("label"))
        if counts[label] < samples_per_label:
            selected.append(row)
            counts[label] += 1
    return selected


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(torch.tensor(values, dtype=torch.float32).mean().item())


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    return float(torch.tensor(values, dtype=torch.float32).std(unbiased=False).item())


def diagnose_manifest(rows: list[dict[str, Any]], seed: int, max_pairs: int) -> dict[str, Any]:
    rng = random.Random(seed)
    payloads = []
    for row in rows:
        video, timestamps = load_video(Path(row["video_feature_path"]))
        payloads.append({"row": row, "video": video, "timestamps": timestamps})
    all_video = torch.cat([item["video"] for item in payloads], dim=0)
    norms = all_video.norm(dim=-1)
    per_dim_std = all_video.std(dim=0, unbiased=False)

    adjacent_distances = []
    adjacent_cosines = []
    for item in payloads:
        video = item["video"]
        if video.shape[0] >= 2:
            adjacent_distances.extend((video[1:] - video[:-1]).norm(dim=-1).tolist())
            adjacent_cosines.extend(F.cosine_similarity(video[1:], video[:-1], dim=-1).tolist())

    ids = list(range(len(payloads)))
    diff_video_distances = []
    diff_video_cosines = []
    same_label_distances = []
    diff_label_distances = []
    same_label_cosines = []
    diff_label_cosines = []
    video_means = [item["video"].mean(dim=0) for item in payloads]
    labels = [str(item["row"].get("label")) for item in payloads]
    for _ in range(min(max_pairs, max(1, len(ids) * 20))):
        left, right = rng.sample(ids, 2)
        va = payloads[left]["video"]
        vb = payloads[right]["video"]
        ia = rng.randrange(va.shape[0])
        ib = rng.randrange(vb.shape[0])
        diff_video_distances.append(float((va[ia] - vb[ib]).norm().item()))
        diff_video_cosines.append(float(F.cosine_similarity(va[ia].unsqueeze(0), vb[ib].unsqueeze(0), dim=-1).item()))

        dist = float((video_means[left] - video_means[right]).norm().item())
        cos = float(F.cosine_similarity(video_means[left].unsqueeze(0), video_means[right].unsqueeze(0), dim=-1).item())
        if labels[left] == labels[right]:
            same_label_distances.append(dist)
            same_label_cosines.append(cos)
        else:
            diff_label_distances.append(dist)
            diff_label_cosines.append(cos)

    by_label: dict[str, list[torch.Tensor]] = defaultdict(list)
    for item, video_mean in zip(payloads, video_means):
        by_label[str(item["row"].get("label"))].append(video_mean)
    centroids = {label: torch.stack(values).mean(dim=0) for label, values in by_label.items()}
    centroid_distances = []
    for idx, left in enumerate(sorted(centroids)):
        for right in sorted(centroids)[idx + 1 :]:
            centroid_distances.append(float((centroids[left] - centroids[right]).norm().item()))

    adjacent_mean = _mean(adjacent_distances)
    diff_mean = _mean(diff_video_distances)
    same_label_distance_mean = _mean(same_label_distances)
    diff_label_distance_mean = _mean(diff_label_distances)
    same_label_cosine_mean = _mean(same_label_cosines)
    diff_label_cosine_mean = _mean(diff_label_cosines)
    collapse_reasons = []
    if float(norms.mean().item()) <= 1e-6:
        collapse_reasons.append("near_zero_norm")
    if float(per_dim_std.mean().item()) <= 1e-5:
        collapse_reasons.append("near_zero_global_std")
    if diff_mean is not None and diff_mean <= 1e-4:
        collapse_reasons.append("different_videos_near_zero_distance")
    if _mean(diff_video_cosines) is not None and _mean(diff_video_cosines) >= 0.999:
        collapse_reasons.append("different_videos_highly_identical")
    return {
        "sample_count": len(rows),
        "window_count": int(all_video.shape[0]),
        "feature_dim": int(all_video.shape[-1]),
        "feature_mean": float(all_video.mean().item()),
        "feature_std": float(all_video.std(unbiased=False).item()),
        "per_dim_std_mean": float(per_dim_std.mean().item()),
        "norm_mean": float(norms.mean().item()),
        "norm_std": float(norms.std(unbiased=False).item()),
        "adjacent_window_distance_mean": adjacent_mean,
        "adjacent_window_cosine_mean": _mean(adjacent_cosines),
        "different_video_window_distance_mean": diff_mean,
        "different_video_window_cosine_mean": _mean(diff_video_cosines),
        "same_label_video_distance_mean": same_label_distance_mean,
        "different_label_video_distance_mean": diff_label_distance_mean,
        "same_label_video_cosine_mean": same_label_cosine_mean,
        "different_label_video_cosine_mean": diff_label_cosine_mean,
        "event_centroid_distance_mean": _mean(centroid_distances),
        "label_counts": dict(Counter(labels)),
        "collapse_reasons": collapse_reasons,
        "all_finite": bool(torch.isfinite(all_video).all().item()),
        "continuity_ok": bool(adjacent_mean is not None and diff_mean is not None and adjacent_mean < diff_mean),
        "label_separation_trend": bool(
            same_label_distance_mean is not None
            and diff_label_distance_mean is not None
            and same_label_distance_mean < diff_label_distance_mean
        ),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# AVE_HF 视频特征对比诊断",
        "",
        "本报告只比较预计算窗口级视频特征，不训练 projector，不修改 Gate、动态窗口或 M4 正式推理路径。",
        "",
    ]
    for name, diag in summary["diagnostics"].items():
        lines.extend(
            [
                f"## {name}",
                f"- 样本/窗口：{diag['sample_count']} / {diag['window_count']}",
                f"- 特征维度：{diag['feature_dim']}",
                f"- 均值/标准差：{diag['feature_mean']} / {diag['feature_std']}",
                f"- 范数均值：{diag['norm_mean']}",
                f"- 相邻窗口距离/余弦：{diag['adjacent_window_distance_mean']} / {diag['adjacent_window_cosine_mean']}",
                f"- 不同视频距离/余弦：{diag['different_video_window_distance_mean']} / {diag['different_video_window_cosine_mean']}",
                f"- 同 label 视频距离/余弦：{diag['same_label_video_distance_mean']} / {diag['same_label_video_cosine_mean']}",
                f"- 异 label 视频距离/余弦：{diag['different_label_video_distance_mean']} / {diag['different_label_video_cosine_mean']}",
                f"- 类别中心距离均值：{diag['event_centroid_distance_mean']}",
                f"- 连续性检查：{diag['continuity_ok']}",
                f"- label 分离趋势：{diag['label_separation_trend']}",
                f"- NaN/Inf：{not diag['all_finite']}",
                f"- 塌缩原因：`{diag['collapse_reasons']}`",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifests = {"rgb_statistics": Path(args.rgb_manifest).resolve(), "m4_clip_vision_tower": Path(args.clip_manifest).resolve()}
    diagnostics = {}
    for name, path in manifests.items():
        rows = select_rows(load_jsonl(path), args.samples_per_label)
        diagnostics[name] = diagnose_manifest(rows, seed=args.seed, max_pairs=args.max_pairs)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "samples_per_label": args.samples_per_label,
        "manifests": {key: str(value) for key, value in manifests.items()},
        "diagnostics": diagnostics,
        "gate_or_dynamic_window_modified": False,
    }
    output_root = Path(args.output_root).resolve()
    write_json(output_root / "video_feature_comparison_summary.json", summary)
    write_report(output_root / "video_feature_comparison_report.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-manifest", default=str(DEFAULT_RGB_MANIFEST))
    parser.add_argument("--clip-manifest", default=str(DEFAULT_CLIP_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--samples-per-label", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--max-pairs", type=int, default=5000)
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps({"ok": True, "summary": str(Path(summary["manifests"]["m4_clip_vision_tower"]).name)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
