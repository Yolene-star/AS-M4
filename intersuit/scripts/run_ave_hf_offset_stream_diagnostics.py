#!/usr/bin/env python
"""在真实 AVE 开发序列上运行冻结 offset scorer 诊断旁路。

默认只使用已经参与策略选择的开发集，不读取独立测试集；输出三候选分数、
raw/suggested offset、margin、接受标志及相邻窗口跳变统计。不会移动窗口、
不会接 Gate，也不会写回模型输入。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from intersuit.model.streaming_av.audio_event_aligner import (
    FrozenOffsetScorerInputs,
    FrozenTemporalOffsetScorer,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"
DEFAULT_BUNDLE = INTERSUIT_ROOT / (
    "harness/artifacts/ave_hf_temporal_offset_zero125_centerpeak_expanded_frozen/"
    "seed_20260719/temporal_offset_scorer_runtime_bundle.pt"
)
DEFAULT_DEV = INTERSUIT_ROOT / "harness/artifacts/ave_hf_selective_1200_split/dev_manifest.jsonl"
DEFAULT_CLIP = INTERSUIT_ROOT / (
    "harness/artifacts/ave_hf_selective_1200_clip_window_features/"
    "ave_hf_clip_window_feature_manifest.jsonl"
)
DEFAULT_RGB = INTERSUIT_ROOT / (
    "harness/artifacts/ave_hf_selective_1200_window_features/"
    "ave_hf_window_feature_manifest.jsonl"
)
DEFAULT_OUTPUT = INTERSUIT_ROOT / "harness/artifacts/ave_hf_offset_stream_diagnostics_dev"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_feature(path: str | Path, key: str) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    values = payload[key].float()
    timestamps = payload["timestamps"].float()
    if values.ndim != 2 or timestamps.shape != (values.shape[0], 2):
        raise ValueError(f"特征或时间戳形状非法：{path}")
    if not torch.isfinite(values).all() or not torch.isfinite(timestamps).all():
        raise ValueError(f"特征包含 NaN/Inf：{path}")
    return values, timestamps


def window_audio_stats(audio_path: str | Path, timestamps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from scipy.io import wavfile

    rate, data = wavfile.read(audio_path)
    if rate != 16000:
        raise ValueError(f"诊断音频采样率必须为 16000：{audio_path}")
    values = torch.as_tensor(data).float()
    if values.ndim == 2:
        values = values.mean(dim=1)
    if data.dtype.kind in {"i", "u"}:
        info = torch.iinfo(torch.as_tensor(data).dtype)
        values = values / float(max(abs(info.min), abs(info.max)))
    rms, nonsilent = [], []
    for start, end in timestamps.tolist():
        left = max(0, int(round(start * rate)))
        right = min(values.numel(), int(round(end * rate)))
        window = values[left:right]
        rms.append(float(torch.sqrt((window * window).mean()).item()) if window.numel() else 0.0)
        nonsilent.append(float((window.abs() > 1e-4).float().mean().item()) if window.numel() else 0.0)
    return torch.tensor(rms), torch.tensor(nonsilent)


def jump_rate(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return sum(left != right for left, right in zip(values, values[1:])) / (len(values) - 1)


def offset_distribution(values: list[float]) -> dict[str, int]:
    return dict(sorted(Counter(f"{value:.1f}" for value in values).items()))


def run(args: argparse.Namespace) -> dict[str, Any]:
    dev_rows = load_jsonl(Path(args.dev_manifest).resolve())
    dev_ids = {str(row["youtube_id"]) for row in dev_rows}
    clip_rows = {
        str(row["youtube_id"]): row
        for row in load_jsonl(Path(args.clip_manifest).resolve())
        if str(row["youtube_id"]) in dev_ids
    }
    rgb_rows = {
        str(row["youtube_id"]): row
        for row in load_jsonl(Path(args.rgb_manifest).resolve())
        if str(row["youtube_id"]) in dev_ids
    }
    if set(clip_rows) != dev_ids or set(rgb_rows) != dev_ids:
        raise ValueError("开发集与冻结三路特征清单不一致")
    scorer = FrozenTemporalOffsetScorer(
        Path(args.bundle).resolve(),
        margin_threshold=float(args.margin_threshold),
    ).eval()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sequence_rows = []
    all_raw, all_suggested, all_accepted, all_margins = [], [], [], []
    for youtube_id in sorted(dev_ids):
        clip_row = clip_rows[youtube_id]
        rgb_row = rgb_rows[youtube_id]
        audio, audio_ts = load_feature(clip_row["audio_feature_path"], "audio_embedding")
        clip, clip_ts = load_feature(clip_row["video_feature_path"], "video_features")
        rgb, rgb_ts = load_feature(rgb_row["video_feature_path"], "video_features")
        if not torch.allclose(audio_ts, clip_ts, atol=1e-5, rtol=0.0) or not torch.allclose(
            audio_ts, rgb_ts, atol=1e-5, rtol=0.0
        ):
            raise ValueError(f"{youtube_id} 三路时间戳不一致")
        rms, nonsilent = window_audio_stats(rgb_row["audio_path"], audio_ts)
        output = scorer(
            FrozenOffsetScorerInputs(
                audio.unsqueeze(0),
                clip.unsqueeze(0),
                rgb.unsqueeze(0),
                rms.unsqueeze(0),
                nonsilent.unsqueeze(0),
            )
        )
        raw = [float(value) for value in output.best_offset[0].tolist()]
        suggested = [float(value) for value in output.suggested_offset[0].tolist()]
        accepted = [bool(value) for value in output.accepted[0].tolist()]
        margins = [float(value) for value in output.margin[0].tolist()]
        scores = [[float(value) for value in row] for row in output.candidate_scores[0].tolist()]
        sequence_rows.append(
            {
                "youtube_id": youtube_id,
                "label": clip_row.get("label"),
                "window_count": len(raw),
                "candidate_offsets": [-0.5, 0.0, 0.5],
                "candidate_scores": scores,
                "best_offset": raw,
                "margin": margins,
                "accepted": accepted,
                "suggested_offset": suggested,
                "raw_jump_rate": jump_rate(raw),
                "suggested_jump_rate": jump_rate(suggested),
                "raw_offset_distribution": offset_distribution(raw),
                "suggested_offset_distribution": offset_distribution(suggested),
            }
        )
        all_raw.extend(raw)
        all_suggested.extend(suggested)
        all_accepted.extend(accepted)
        all_margins.extend(margins)
    with (output_root / "stream_offset_diagnostics.jsonl").open("w", encoding="utf-8") as handle:
        for row in sequence_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "diagnostic_only": True,
        "moves_audio_window": False,
        "feeds_gate": False,
        "video_count": len(sequence_rows),
        "window_count": len(all_raw),
        "margin_threshold": float(args.margin_threshold),
        "accepted_count": sum(all_accepted),
        "accepted_rate": mean(all_accepted),
        "margin_mean": mean(all_margins),
        "raw_offset_distribution": offset_distribution(all_raw),
        "suggested_offset_distribution": offset_distribution(all_suggested),
        "mean_video_raw_jump_rate": mean(row["raw_jump_rate"] for row in sequence_rows),
        "mean_video_suggested_jump_rate": mean(row["suggested_jump_rate"] for row in sequence_rows),
        "videos_with_suggested_jump_rate_above_0.5": sum(
            row["suggested_jump_rate"] > 0.5 for row in sequence_rows
        ),
        "all_scores_finite": all(
            all(torch.isfinite(torch.tensor(scores)).all().item() for scores in row["candidate_scores"])
            for row in sequence_rows
        ),
        "source_split": "development_only",
        "test_set_read": False,
        "paths": {
            "sequences": str(output_root / "stream_offset_diagnostics.jsonl"),
            "summary": str(output_root / "stream_offset_diagnostics_summary.json"),
            "report": str(output_root / "stream_offset_diagnostics_report.md"),
        },
    }
    (output_root / "stream_offset_diagnostics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(output_root / "stream_offset_diagnostics_report.md", summary)
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# AVE 冻结 offset scorer 真实流式序列诊断",
        "",
        "本报告只使用开发集序列；不读取独立测试集，不移动窗口，不接 Gate。",
        "",
        f"- 视频数：{summary['video_count']}",
        f"- 窗口数：{summary['window_count']}",
        f"- margin 阈值：{summary['margin_threshold']}",
        f"- 接受率：{summary['accepted_rate']:.4f}",
        f"- raw offset 分布：`{summary['raw_offset_distribution']}`",
        f"- 最终建议 offset 分布：`{summary['suggested_offset_distribution']}`",
        f"- 视频内 raw 相邻跳变率均值：{summary['mean_video_raw_jump_rate']:.4f}",
        f"- 视频内建议 offset 相邻跳变率均值：{summary['mean_video_suggested_jump_rate']:.4f}",
        f"- 建议跳变率 > 0.5 的视频：{summary['videos_with_suggested_jump_rate_above_0.5']}",
        f"- 分数均有限：{summary['all_scores_finite']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--dev-manifest", default=str(DEFAULT_DEV))
    parser.add_argument("--clip-manifest", default=str(DEFAULT_CLIP))
    parser.add_argument("--rgb-manifest", default=str(DEFAULT_RGB))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--margin-threshold", type=float, default=0.15)
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps({"ok": True, **summary["paths"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
