#!/usr/bin/env python
"""比较冻结 offset scorer 的三种因果诊断稳定策略。

真实连续流只用于跳变、持续长度、孤立修正和延迟统计；冻结开发记录用于
平滑前后高置信准确率。脚本不读取独立测试集，不训练 scorer，不移动窗口，
不接 Gate，也不修改融合输出。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from dataclasses import fields
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

from intersuit.model.streaming_av.audio_event_aligner import (
    FrozenOffsetScorerInputs,
    FrozenTemporalOffsetScorer,
)
from intersuit.model.streaming_av.offset_stabilizer import stabilize_offset_scores


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"
DEFAULT_STREAMS = INTERSUIT_ROOT / (
    "harness/artifacts/ave_hf_offset_stream_diagnostics_dev/stream_offset_diagnostics.jsonl"
)
DEFAULT_DEV_PREDICTIONS = INTERSUIT_ROOT / (
    "harness/artifacts/ave_hf_selective_correction_acceptance/dev_predictions.jsonl"
)
DEFAULT_STRATEGY_LOCK = INTERSUIT_ROOT / (
    "harness/artifacts/ave_hf_selective_correction_acceptance/strategy_lock.json"
)
DEFAULT_BUNDLE = INTERSUIT_ROOT / (
    "harness/artifacts/ave_hf_temporal_offset_zero125_centerpeak_expanded_frozen/"
    "seed_20260719/temporal_offset_scorer_runtime_bundle.pt"
)
DEFAULT_FROZEN_ROOT = INTERSUIT_ROOT / (
    "harness/artifacts/ave_hf_temporal_offset_zero125_centerpeak_expanded_frozen"
)
DEFAULT_CLIP = INTERSUIT_ROOT / (
    "harness/artifacts/ave_hf_selective_1200_clip_window_features/"
    "ave_hf_clip_window_feature_manifest.jsonl"
)
DEFAULT_RGB = INTERSUIT_ROOT / (
    "harness/artifacts/ave_hf_selective_1200_window_features/"
    "ave_hf_window_feature_manifest.jsonl"
)
DEFAULT_OUTPUT = INTERSUIT_ROOT / "harness/artifacts/ave_hf_offset_temporal_stability_dev"
STRATEGIES = ("none", "consecutive", "hysteresis", "moving_average")
OFFSETS = (-0.5, 0.0, 0.5)
HOP_SECONDS = 0.5


def _load_stream_helpers():
    path = INTERSUIT_ROOT / "scripts/run_ave_hf_offset_stream_diagnostics.py"
    spec = importlib.util.spec_from_file_location("offset_stream_diagnostic_helpers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


stream_helpers = _load_stream_helpers()


def _load_selective_helpers():
    path = INTERSUIT_ROOT / "scripts/evaluate_ave_hf_selective_correction.py"
    spec = importlib.util.spec_from_file_location("offset_stability_selective_helpers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


selective_helpers = _load_selective_helpers()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def strategy_kwargs(name: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "strategy": name,
        "margin_threshold": args.margin_threshold,
        "consecutive_windows": args.consecutive_windows,
        "hold_margin": args.hold_margin,
        "switch_margin": args.switch_margin,
        "moving_average_windows": args.moving_average_windows,
    }


def tensor_output(scores: list[list[float]], name: str, args: argparse.Namespace):
    return stabilize_offset_scores(
        torch.tensor(scores, dtype=torch.float32).unsqueeze(0),
        **strategy_kwargs(name, args),
    )


def jump_rate(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return sum(left != right for left, right in zip(values, values[1:])) / (len(values) - 1)


def nonzero_runs(values: list[float]) -> list[tuple[int, int, float]]:
    runs: list[tuple[int, int, float]] = []
    start = 0
    while start < len(values):
        value = values[start]
        end = start + 1
        while end < len(values) and values[end] == value:
            end += 1
        if value != 0.0:
            runs.append((start, end, value))
        start = end
    return runs


def onset_delays(raw: list[float], stable: list[float]) -> tuple[list[int], int]:
    delays: list[int] = []
    missed = 0
    for start, end, value in nonzero_runs(raw):
        match = next((index for index in range(start, end) if stable[index] == value), None)
        if match is None:
            missed += 1
        else:
            delays.append(match - start)
    return delays, missed


def stream_metrics(
    stream_rows: list[dict[str, Any]],
    strategy: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    all_accepted: list[bool] = []
    all_suggested: list[float] = []
    run_lengths: list[int] = []
    delays: list[int] = []
    missed_onsets = 0
    for source in stream_rows:
        output = tensor_output(source["candidate_scores"], strategy, args)
        suggested = [float(value) for value in output.suggested_offset[0].tolist()]
        accepted = [bool(value) for value in output.accepted[0].tolist()]
        source_raw = [float(value) for value in source["suggested_offset"]]
        runs = nonzero_runs(suggested)
        row_delays, row_missed = onset_delays(source_raw, suggested)
        rows.append(
            {
                "youtube_id": source["youtube_id"],
                "strategy": strategy,
                "stable_candidate_scores": output.candidate_scores[0].tolist(),
                "stable_best_offset": output.best_offset[0].tolist(),
                "stable_margin": output.margin[0].tolist(),
                "stable_accepted": accepted,
                "stable_suggested_offset": suggested,
                "stable_jump_rate": jump_rate(suggested),
                "nonzero_run_lengths": [end - start for start, end, _ in runs],
                "isolated_nonzero_count": sum(end - start == 1 for start, end, _ in runs),
                "matched_onset_delay_windows": row_delays,
                "missed_raw_nonzero_onsets": row_missed,
            }
        )
        all_accepted.extend(accepted)
        all_suggested.extend(suggested)
        run_lengths.extend(end - start for start, end, _ in runs)
        delays.extend(row_delays)
        missed_onsets += row_missed
    nonzero_count = sum(value != 0.0 for value in all_suggested)
    isolated_windows = sum(length == 1 for length in run_lengths)
    return {
        "strategy": strategy,
        "video_count": len(rows),
        "window_count": len(all_suggested),
        "mean_video_jump_rate": mean(row["stable_jump_rate"] for row in rows),
        "accepted_rate": mean(all_accepted),
        "nonzero_suggestion_rate": nonzero_count / len(all_suggested),
        "nonzero_run_count": len(run_lengths),
        "nonzero_run_length_mean_windows": mean(run_lengths) if run_lengths else 0.0,
        "nonzero_run_length_median_windows": median(run_lengths) if run_lengths else 0.0,
        "nonzero_run_length_max_windows": max(run_lengths, default=0),
        "isolated_nonzero_window_ratio": isolated_windows / nonzero_count if nonzero_count else 0.0,
        "decision_delay_mean_windows": mean(delays) if delays else None,
        "decision_delay_max_windows": max(delays, default=None),
        "decision_delay_mean_seconds": mean(delays) * HOP_SECONDS if delays else None,
        "decision_delay_max_seconds": max(delays) * HOP_SECONDS if delays else None,
        "matched_raw_nonzero_onsets": len(delays),
        "missed_raw_nonzero_onsets": missed_onsets,
        "suggested_offset_distribution": dict(
            sorted(Counter(f"{value:.1f}" for value in all_suggested).items())
        ),
    }, rows


def shift_stream(values: torch.Tensor, shift: int) -> torch.Tensor:
    indices = torch.arange(values.shape[0]) + int(shift)
    indices = indices.clamp(0, values.shape[0] - 1)
    return values[indices]


def shifted_scores(
    youtube_id: str,
    condition: str,
    clip_row: dict[str, Any],
    rgb_row: dict[str, Any],
    scorer: FrozenTemporalOffsetScorer,
) -> list[list[float]]:
    audio, audio_ts = stream_helpers.load_feature(clip_row["audio_feature_path"], "audio_embedding")
    clip, clip_ts = stream_helpers.load_feature(clip_row["video_feature_path"], "video_features")
    rgb, rgb_ts = stream_helpers.load_feature(rgb_row["video_feature_path"], "video_features")
    if not torch.allclose(audio_ts, clip_ts, atol=1e-5, rtol=0.0) or not torch.allclose(
        audio_ts, rgb_ts, atol=1e-5, rtol=0.0
    ):
        raise ValueError(f"{youtube_id} 三路时间戳不一致")
    rms, nonsilent = stream_helpers.window_audio_stats(rgb_row["audio_path"], audio_ts)
    shift = {"original": 0, "shift_plus_0.5": 1, "shift_minus_0.5": -1}[condition]
    output = scorer(
        FrozenOffsetScorerInputs(
            shift_stream(audio, shift).unsqueeze(0),
            clip.unsqueeze(0),
            rgb.unsqueeze(0),
            shift_stream(rms, shift).unsqueeze(0),
            shift_stream(nonsilent, shift).unsqueeze(0),
        )
    )
    return output.candidate_scores[0].tolist()


def build_labeled_score_sequences(
    records: list[dict[str, Any]],
    stream_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[tuple[str, str], list[list[float]]]:
    original = {str(row["youtube_id"]): row["candidate_scores"] for row in stream_rows}
    needed = {(str(row["youtube_id"]), str(row["condition"])) for row in records}
    clip_rows = {
        str(row["youtube_id"]): row
        for row in load_jsonl(Path(args.clip_manifest).resolve())
        if str(row["youtube_id"]) in {key[0] for key in needed}
    }
    rgb_rows = {
        str(row["youtube_id"]): row
        for row in load_jsonl(Path(args.rgb_manifest).resolve())
        if str(row["youtube_id"]) in {key[0] for key in needed}
    }
    scorer = FrozenTemporalOffsetScorer(
        Path(args.bundle).resolve(),
        margin_threshold=args.margin_threshold,
    ).eval()
    result: dict[tuple[str, str], list[list[float]]] = {}
    for index, (youtube_id, condition) in enumerate(sorted(needed), start=1):
        if condition == "original":
            result[(youtube_id, condition)] = original[youtube_id]
        else:
            result[(youtube_id, condition)] = shifted_scores(
                youtube_id,
                condition,
                clip_rows[youtube_id],
                rgb_rows[youtube_id],
                scorer,
            )
        if index % 100 == 0:
            print(f"[accuracy] 已准备 {index}/{len(needed)} 条视频/条件序列", flush=True)
    return result


@torch.no_grad()
def exact_labeled_scores(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, list[float]]:
    """按原验收的记录级构造精确复算 seed=20260719 分数，不读取测试集。"""

    frozen_root = Path(args.frozen_root).resolve()
    frozen_summary = json.loads(
        (frozen_root / "frozen_seed_summary.json").read_text(encoding="utf-8")
    )
    config = frozen_summary["config"]
    record_fields = {field.name for field in fields(selective_helpers.temporal.OffsetRecord)}
    typed_records = [
        selective_helpers.temporal.OffsetRecord(
            **{key: value for key, value in row.items() if key in record_fields}
        )
        for row in records
    ]
    eval_ids = {record.youtube_id for record in typed_records}
    eval_caches = selective_helpers.build_filtered_caches(
        Path(args.clip_manifest).resolve(),
        Path(args.rgb_manifest).resolve(),
        eval_ids,
    )
    train_records = selective_helpers.load_frozen_train_records(
        frozen_root,
        "20260719",
    )
    train_ids = {record.youtube_id for record in train_records}
    train_caches = selective_helpers.build_filtered_caches(
        (REPO_ROOT / config["clip_manifest"]).resolve(),
        (REPO_ROOT / config["rgb_manifest"]).resolve(),
        train_ids,
    )
    _, _, _, _, _, scalar_stats = selective_helpers.temporal.make_tensor_dataset(
        train_records,
        train_caches,
        context_radius=int(config["context_radius"]),
    )
    audio, video, scalars, _, kept, _ = selective_helpers.temporal.make_tensor_dataset(
        typed_records,
        eval_caches,
        context_radius=int(config["context_radius"]),
        scalar_stats=scalar_stats,
    )
    checkpoint = torch.load(
        frozen_root / "seed_20260719/temporal_offset_scorer_one_epoch.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = selective_helpers.temporal.OffsetScorer(
        audio_dim=audio.shape[-1],
        video_dim=video.shape[-1],
        scalar_dim=scalars.shape[-1],
        hidden_dim=int(config["hidden_dim"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    scores = model(audio, video, scalars)
    return {
        selective_helpers.record_key(record): scores[index].tolist()
        for index, record in enumerate(kept)
    }


def patch_exact_labeled_points(
    records: list[dict[str, Any]],
    sequences: dict[tuple[str, str], list[list[float]]],
    exact_scores: dict[str, list[float]],
) -> None:
    for record in records:
        key = (str(record["youtube_id"]), str(record["condition"]))
        sequences[key][int(record["video_window"])] = exact_scores[record["record_key"]]


def labeled_metrics(
    records: list[dict[str, Any]],
    score_sequences: dict[tuple[str, str], list[list[float]]],
    strategy: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_cache = {
        key: tensor_output(scores, strategy, args)
        for key, scores in score_sequences.items()
    }
    accepted, correct = [], []
    by_condition: dict[str, list[bool]] = defaultdict(list)
    decisions = []
    for record in records:
        key = (str(record["youtube_id"]), str(record["condition"]))
        step = int(record["video_window"])
        output = output_cache[key]
        is_accepted = bool(output.accepted[0, step].item())
        predicted = OFFSETS.index(float(output.suggested_offset[0, step].item()))
        is_correct = predicted == int(record["target_index"])
        accepted.append(is_accepted)
        correct.append(is_correct)
        if is_accepted:
            by_condition[str(record["condition"])].append(is_correct)
        decisions.append(
            {
                "record_key": record["record_key"],
                "strategy": strategy,
                "accepted": is_accepted,
                "predicted_offset": OFFSETS[predicted],
                "target_offset": OFFSETS[int(record["target_index"])],
                "correct": is_correct,
            }
        )
    accepted_count = sum(accepted)
    return {
        "sample_count": len(records),
        "accepted_count": accepted_count,
        "coverage": accepted_count / len(records),
        "accepted_accuracy": (
            sum(is_correct for is_correct, keep in zip(correct, accepted) if keep) / accepted_count
            if accepted_count
            else None
        ),
        "accepted_condition_accuracy": {
            key: mean(values) for key, values in sorted(by_condition.items())
        },
        "decisions": decisions,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# 冻结 offset scorer 时序稳定策略开发集分析",
        "",
        "仅使用开发集；不读取独立测试集，不重新训练 scorer，不移动窗口，不接 Gate，不修改融合输出。",
        "",
        "## 固定参数",
        "",
        f"- 原始 margin：`{payload['parameters']['margin_threshold']}`",
        f"- 连续一致：`N={payload['parameters']['consecutive_windows']}`",
        f"- 滞回：进入 `{payload['parameters']['margin_threshold']}`，保持 `{payload['parameters']['hold_margin']}`，切换 `{payload['parameters']['switch_margin']}`",
        f"- 滑动平均：`{payload['parameters']['moving_average_windows']}` 个因果窗口",
        f"- hop：`{HOP_SECONDS}s`",
        "",
        "## 真实流稳定性",
        "",
        "| 策略 | 跳变率 | 接受率 | 非零率 | 平均持续(窗) | 单窗孤立比例 | 平均/最大延迟(s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in STRATEGIES:
        row = payload["strategies"][name]["stream"]
        delay_mean = row["decision_delay_mean_seconds"]
        delay_max = row["decision_delay_max_seconds"]
        delay = "N/A" if delay_mean is None else f"{delay_mean:.3f}/{delay_max:.3f}"
        lines.append(
            f"| `{name}` | {row['mean_video_jump_rate']:.4f} | {row['accepted_rate']:.4f} | "
            f"{row['nonzero_suggestion_rate']:.4f} | {row['nonzero_run_length_mean_windows']:.2f} | "
            f"{row['isolated_nonzero_window_ratio']:.4f} | {delay} |"
        )
    lines.extend(
        [
            "",
            "## 冻结开发标签准确率",
            "",
            "| 策略 | 覆盖率 | 高置信准确率 | 相对原 scorer |",
            "|---|---:|---:|---:|",
        ]
    )
    raw_accuracy = payload["strategies"]["none"]["labeled"]["accepted_accuracy"]
    for name in STRATEGIES:
        row = payload["strategies"][name]["labeled"]
        delta = row["accepted_accuracy"] - raw_accuracy if row["accepted_accuracy"] is not None else None
        lines.append(
            f"| `{name}` | {row['coverage']:.4f} | {row['accepted_accuracy']:.4f} | {delta:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## 自动判定",
            "",
            f"- 推荐策略：`{payload['recommendation']['strategy']}`",
            f"- 达到本轮目标：{payload['recommendation']['meets_targets']}",
            f"- 说明：{payload['recommendation']['reason']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    stream_rows = load_jsonl(Path(args.stream_sequences).resolve())
    records = load_jsonl(Path(args.dev_predictions).resolve())
    lock = json.loads(Path(args.strategy_lock).resolve().read_text(encoding="utf-8"))
    if lock["selected_strategy"]["seed"] != "20260719" or float(
        lock["selected_strategy"]["threshold"]
    ) != args.margin_threshold:
        raise ValueError("开发策略锁与冻结 seed=20260719 / margin=0.15 不一致")
    score_sequences = build_labeled_score_sequences(records, stream_rows, args)
    exact_scores = exact_labeled_scores(records, args)
    patch_exact_labeled_points(records, score_sequences, exact_scores)
    payload: dict[str, Any] = {
        "diagnostic_only": True,
        "moves_audio_window": False,
        "feeds_gate": False,
        "modifies_fusion": False,
        "retrained_scorer": False,
        "source_split": "development_only",
        "test_set_read": False,
        "parameters": {
            "margin_threshold": args.margin_threshold,
            "consecutive_windows": args.consecutive_windows,
            "hold_margin": args.hold_margin,
            "switch_margin": args.switch_margin,
            "moving_average_windows": args.moving_average_windows,
            "hop_seconds": HOP_SECONDS,
        },
        "strategies": {},
    }
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sequence_path = output_root / "temporal_stability_sequences.jsonl"
    decision_path = output_root / "temporal_stability_labeled_decisions.jsonl"
    sequence_rows, decision_rows = [], []
    for name in STRATEGIES:
        stream, per_video = stream_metrics(stream_rows, name, args)
        labeled = labeled_metrics(records, score_sequences, name, args)
        payload["strategies"][name] = {
            "stream": stream,
            "labeled": {key: value for key, value in labeled.items() if key != "decisions"},
        }
        sequence_rows.extend(per_video)
        decision_rows.extend(labeled["decisions"])
        print(
            f"[{name}] jump={stream['mean_video_jump_rate']:.4f} "
            f"isolated={stream['isolated_nonzero_window_ratio']:.4f} "
            f"coverage={labeled['coverage']:.4f} acc={labeled['accepted_accuracy']:.4f}",
            flush=True,
        )
    locked = lock["selected_strategy"]
    reproduced = payload["strategies"]["none"]["labeled"]
    if reproduced["accepted_count"] != int(locked["accepted_count"]) or abs(
        reproduced["accepted_accuracy"] - float(locked["accepted_accuracy"])
    ) > 1e-6:
        raise RuntimeError(
            "稳定策略分析未精确复现冻结开发基线："
            f"got={reproduced}, locked={locked['accepted_count']}/"
            f"{locked['accepted_accuracy']}"
        )
    raw = payload["strategies"]["none"]
    candidates = []
    for name in STRATEGIES[1:]:
        row = payload["strategies"][name]
        stable = row["stream"]
        labeled = row["labeled"]
        accuracy_drop = raw["labeled"]["accepted_accuracy"] - labeled["accepted_accuracy"]
        delay = stable["decision_delay_max_seconds"]
        meets = bool(
            stable["mean_video_jump_rate"] < raw["stream"]["mean_video_jump_rate"] * 0.8
            and stable["isolated_nonzero_window_ratio"]
            < raw["stream"]["isolated_nonzero_window_ratio"] * 0.5
            and labeled["coverage"] >= 0.10
            and accuracy_drop <= 0.03
            and (delay is None or delay <= 1.0)
        )
        candidates.append((meets, -stable["mean_video_jump_rate"], labeled["accepted_accuracy"], name))
    chosen = max(candidates)
    any_passed = any(row[0] for row in candidates)
    payload["recommendation"] = {
        "strategy": chosen[-1] if any_passed else "none",
        "best_experimental_strategy": chosen[-1],
        "meets_targets": bool(any_passed),
        "reason": (
            "优先满足跳变、孤立修正、覆盖率、准确率下降不超过3个百分点和最大延迟不超过1s；"
            "满足者中优先选择跳变率更低、准确率更高的策略。若全部失败，保持 none。"
        ),
    }
    for path, rows in ((sequence_path, sequence_rows), (decision_path, decision_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = output_root / "temporal_stability_summary.json"
    report_path = output_root / "temporal_stability_report.md"
    payload["paths"] = {
        "sequences": str(sequence_path),
        "labeled_decisions": str(decision_path),
        "summary": str(summary_path),
        "report": str(report_path),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-sequences", default=str(DEFAULT_STREAMS))
    parser.add_argument("--dev-predictions", default=str(DEFAULT_DEV_PREDICTIONS))
    parser.add_argument("--strategy-lock", default=str(DEFAULT_STRATEGY_LOCK))
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--frozen-root", default=str(DEFAULT_FROZEN_ROOT))
    parser.add_argument("--clip-manifest", default=str(DEFAULT_CLIP))
    parser.add_argument("--rgb-manifest", default=str(DEFAULT_RGB))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--margin-threshold", type=float, default=0.15)
    parser.add_argument("--consecutive-windows", type=int, default=2)
    parser.add_argument("--hold-margin", type=float, default=0.10)
    parser.add_argument("--switch-margin", type=float, default=0.30)
    parser.add_argument("--moving-average-windows", type=int, default=3)
    return parser


def main() -> None:
    payload = run(build_parser().parse_args())
    print(json.dumps({"ok": True, **payload["paths"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
