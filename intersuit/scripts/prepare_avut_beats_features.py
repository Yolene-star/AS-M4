#!/usr/bin/env python
"""为固定 5 条 AVUT 样本离线提取 BEATs 音频事件窗口特征。

本脚本只生成和校验 precomputed 音频特征，不修改 Gate、融合、动态对齐
或任何训练流程。BEATs checkpoint 必须由用户显式准备到本地；脚本不会
自动联网下载权重。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import torch
import torchaudio.compliance.kaldi as ta_kaldi


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"
DEFAULT_MANIFEST = INTERSUIT_ROOT / "inputs/texts/avut/avut_audio_smoke.json"
DEFAULT_BEATS_CODE_ROOT = REPO_ROOT / "third_party/OmniMMI/baselines/videollama2/model"
DEFAULT_OUTPUT_ROOT = INTERSUIT_ROOT / "harness/artifacts/avut_beats_features"
CONDITIONS = ("original", "silence", "wrong_audio", "shift_plus_0_5", "shift_minus_0_5")


class WindowEncoder(Protocol):
    encoder_name: str
    checkpoint_name: str
    checkpoint_sha256: str
    embedding_dim: int | None

    def encode_windows(self, windows: torch.Tensor) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class AudioCondition:
    name: str
    source_sample_id: str
    waveform: torch.Tensor
    shift_seconds: float = 0.0


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


def load_manifest(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("AVUT manifest must be a JSON list")
    rows = data[:limit] if limit is not None else data
    if not rows:
        raise ValueError("AVUT manifest is empty")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_local_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} not found: {path}. Automatic downloads are disabled; "
            "prepare the pretrained BEATs checkpoint locally and pass --beats-checkpoint."
        )
    return path


class BEATsWindowEncoder:
    """Frozen BEATs encoder loaded from local source code and checkpoint."""

    encoder_name = "BEATs"

    def __init__(
        self,
        checkpoint_path: Path,
        beats_code_root: Path = DEFAULT_BEATS_CODE_ROOT,
        device: str = "cpu",
    ) -> None:
        checkpoint_path = require_local_file(checkpoint_path, "BEATs checkpoint")
        beats_code_root = require_local_file(beats_code_root / "beats/BEATs.py", "BEATs source").parents[1]
        sys.path.insert(0, str(beats_code_root))
        beats_module = importlib.import_module("beats.BEATs")
        BEATsConfig = getattr(beats_module, "BEATsConfig")
        BEATs = getattr(beats_module, "BEATs")

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError("BEATs checkpoint must be a dictionary containing model weights")
        cfg = checkpoint.get("cfg") or checkpoint.get("config") or {}
        state = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
        if not isinstance(state, dict):
            raise ValueError("BEATs checkpoint does not contain a model/state_dict mapping")

        self.device = torch.device(device)
        self.model = BEATs(BEATsConfig(cfg)).to(self.device)
        self.model.load_state_dict(_strip_module_prefix(state), strict=False)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.checkpoint_path = checkpoint_path
        self.checkpoint_name = checkpoint_path.name
        self.checkpoint_sha256 = sha256_file(checkpoint_path)
        self.embedding_dim = int(getattr(self.model.cfg, "encoder_embed_dim", 0)) or None

    @torch.no_grad()
    def encode_windows(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 2:
            raise ValueError(f"Expected windows shaped [T,S], got {tuple(windows.shape)}")
        fbanks = torch.stack([waveform_to_fbank(window) for window in windows.float()], dim=0).to(self.device)
        features, _, _ = self.model.extract_features(fbanks, padding_mask=None, feature_only=True)
        embeddings = features.float().mean(dim=1).cpu()
        if not torch.isfinite(embeddings).all():
            raise ValueError("BEATs produced NaN/Inf embeddings")
        self.embedding_dim = int(embeddings.shape[-1])
        return embeddings


class WaveformStatsWindowEncoder:
    """CPU-only deterministic fake encoder used only by tests."""

    encoder_name = "test_waveform_stats"
    checkpoint_name = "none"
    checkpoint_sha256 = "none"
    embedding_dim = 4

    def encode_windows(self, windows: torch.Tensor) -> torch.Tensor:
        values = windows.float()
        mean = values.mean(dim=-1)
        rms = values.square().mean(dim=-1).sqrt()
        peak = values.abs().amax(dim=-1)
        zero_cross = ((values[:, 1:] * values[:, :-1]) < 0).float().mean(dim=-1)
        return torch.stack([mean, rms, peak, zero_cross], dim=-1)


def _strip_module_prefix(state: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            cleaned[key.removeprefix("module.")] = value
    return cleaned


def waveform_to_fbank(waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    mono = waveform.float().flatten().unsqueeze(0)
    return ta_kaldi.fbank(
        mono * (2**15),
        num_mel_bins=128,
        sample_frequency=sample_rate,
        frame_length=25,
        frame_shift=10,
    )


def shift_waveform(waveform: torch.Tensor, sample_rate: int, shift_seconds: float) -> torch.Tensor:
    shift_samples = int(round(float(shift_seconds) * sample_rate))
    if shift_samples == 0:
        return waveform.clone()
    result = torch.zeros_like(waveform)
    length = int(waveform.shape[-1])
    amount = min(abs(shift_samples), length)
    if shift_samples > 0 and amount < length:
        result[..., amount:] = waveform[..., : length - amount]
    elif shift_samples < 0 and amount < length:
        result[..., : length - amount] = waveform[..., amount:]
    return result


def load_sample_waveforms(samples: list[dict[str, Any]]) -> dict[str, tuple[torch.Tensor, int, Path]]:
    from intersuit.streaming.audio_stream import load_scene_audio

    loaded: dict[str, tuple[torch.Tensor, int, Path]] = {}
    for sample in samples:
        sample_id = str(sample.get("id"))
        path = resolve_repo_path(str(sample.get("scene_audio_path") or sample.get("video_path")))
        sample_rate = int(sample.get("scene_audio_sample_rate") or 16000)
        waveform, rate = load_scene_audio(path, sample_rate=sample_rate, mono=True)
        if waveform.numel() == 0:
            raise ValueError(f"Decoded empty audio for {sample_id}: {path}")
        if not torch.isfinite(waveform).all():
            raise ValueError(f"Decoded audio contains NaN/Inf for {sample_id}: {path}")
        loaded[sample_id] = (waveform, rate, path)
    return loaded


def build_conditions(
    sample: dict[str, Any],
    sample_index: int,
    samples: list[dict[str, Any]],
    waveforms: dict[str, tuple[torch.Tensor, int, Path]],
) -> list[AudioCondition]:
    sample_id = str(sample.get("id"))
    waveform, sample_rate, _ = waveforms[sample_id]
    wrong_sample = samples[(sample_index + 1) % len(samples)]
    wrong_id = str(wrong_sample.get("id"))
    wrong_waveform, _, _ = waveforms[wrong_id]
    return [
        AudioCondition("original", sample_id, waveform),
        AudioCondition("silence", sample_id, torch.zeros_like(waveform)),
        AudioCondition("wrong_audio", wrong_id, _match_length(wrong_waveform, waveform.shape[-1])),
        AudioCondition("shift_plus_0_5", sample_id, shift_waveform(waveform, sample_rate, 0.5), 0.5),
        AudioCondition("shift_minus_0_5", sample_id, shift_waveform(waveform, sample_rate, -0.5), -0.5),
    ]


def _match_length(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    if waveform.shape[-1] == target_length:
        return waveform.clone()
    if waveform.shape[-1] > target_length:
        return waveform[..., :target_length].clone()
    result = torch.zeros(target_length, dtype=waveform.dtype)
    result[: waveform.shape[-1]] = waveform
    return result


def window_condition(
    waveform: torch.Tensor,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from intersuit.streaming.audio_stream import split_audio_windows, stack_audio_windows

    windows = split_audio_windows(waveform, sample_rate=sample_rate, window_sec=window_sec, hop_sec=hop_sec)
    samples, timestamps = stack_audio_windows(windows)
    validate_windows(samples, timestamps)
    return samples, timestamps


def validate_windows(windows: torch.Tensor, timestamps: torch.Tensor) -> None:
    if windows.ndim != 2:
        raise ValueError(f"audio windows must be [T,S], got {tuple(windows.shape)}")
    if timestamps.shape != (windows.shape[0], 2):
        raise ValueError(f"timestamps shape {tuple(timestamps.shape)} does not match window count {windows.shape[0]}")
    if windows.shape[0] == 0:
        raise ValueError("missing audio windows")
    if not torch.isfinite(windows).all() or not torch.isfinite(timestamps).all():
        raise ValueError("audio windows/timestamps contain NaN or Inf")
    if (timestamps[:, 0] > timestamps[:, 1]).any():
        raise ValueError("timestamps must satisfy start <= end")


def validate_feature_file(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"sample_id", "condition", "timestamps", "audio_embedding", "metadata"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"precomputed feature file {path} is missing keys: {sorted(missing)}")
    embeddings = payload["audio_embedding"].float()
    timestamps = payload["timestamps"].float()
    if embeddings.ndim != 2:
        raise ValueError(f"audio_embedding must be [T,D], got {tuple(embeddings.shape)}")
    if timestamps.shape != (embeddings.shape[0], 2):
        raise ValueError("timestamp/window count mismatch in precomputed feature file")
    if not torch.isfinite(embeddings).all() or not torch.isfinite(timestamps).all():
        raise ValueError("precomputed feature file contains NaN or Inf")
    return {
        "window_count": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_norm_mean": float(embeddings.norm(dim=-1).mean().item()),
    }


def save_feature_file(
    output_dir: Path,
    sample: dict[str, Any],
    condition: AudioCondition,
    windows: torch.Tensor,
    timestamps: torch.Tensor,
    embeddings: torch.Tensor,
    encoder: WindowEncoder,
    media_path: Path,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
) -> Path:
    sample_id = str(sample.get("id"))
    condition_dir = output_dir / sample_id
    condition_dir.mkdir(parents=True, exist_ok=True)
    path = condition_dir / f"{condition.name}.pt"
    metadata = {
        "sample_id": sample_id,
        "condition": condition.name,
        "source_sample_id": condition.source_sample_id,
        "source_audio_path": str(media_path),
        "encoder_name": encoder.encoder_name,
        "checkpoint_name": encoder.checkpoint_name,
        "checkpoint_sha256": encoder.checkpoint_sha256,
        "sample_rate": int(sample_rate),
        "window_sec": float(window_sec),
        "hop_sec": float(hop_sec),
        "window_count": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "shift_seconds": float(condition.shift_seconds),
    }
    torch.save(
        {
            "sample_id": sample_id,
            "condition": condition.name,
            "timestamps": timestamps.cpu(),
            "audio_embedding": embeddings.cpu(),
            "metadata": metadata,
        },
        path,
    )
    validate_feature_file(path)
    return path


def write_window_jsonl(feature_files: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for path in feature_files:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            metadata = payload["metadata"]
            timestamps = payload["timestamps"].float()
            embeddings = payload["audio_embedding"].float()
            for idx in range(embeddings.shape[0]):
                row = {
                    "sample_id": payload["sample_id"],
                    "condition": payload["condition"],
                    "window_index": idx,
                    "window_start": float(timestamps[idx, 0].item()),
                    "window_end": float(timestamps[idx, 1].item()),
                    "audio_embedding": embeddings[idx].tolist(),
                    "encoder_name": metadata["encoder_name"],
                    "checkpoint_name": metadata["checkpoint_name"],
                    "sample_rate": metadata["sample_rate"],
                    "embedding_dim": metadata["embedding_dim"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_features(feature_files: list[Path]) -> dict[str, Any]:
    rows = []
    dims = set()
    for path in feature_files:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        embeddings = payload["audio_embedding"].float()
        timestamps = payload["timestamps"].float()
        metadata = payload["metadata"]
        dims.add(int(embeddings.shape[1]))
        rows.append(
            {
                "sample_id": payload["sample_id"],
                "condition": payload["condition"],
                "source_sample_id": metadata["source_sample_id"],
                "window_count": int(embeddings.shape[0]),
                "embedding_dim": int(embeddings.shape[1]),
                "timestamp_start": float(timestamps[0, 0].item()),
                "timestamp_end": float(timestamps[-1, 1].item()),
                "norm_mean": float(embeddings.norm(dim=-1).mean().item()),
                "norm_std": float(embeddings.norm(dim=-1).std(unbiased=False).item()),
            }
        )
    by_sample: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_sample.setdefault(row["sample_id"], {})[row["condition"]] = row
    comparisons = []
    for sample_id, conditions in by_sample.items():
        original_path = _find_feature_file(feature_files, sample_id, "original")
        silence_path = _find_feature_file(feature_files, sample_id, "silence")
        plus_path = _find_feature_file(feature_files, sample_id, "shift_plus_0_5")
        minus_path = _find_feature_file(feature_files, sample_id, "shift_minus_0_5")
        wrong_path = _find_feature_file(feature_files, sample_id, "wrong_audio")
        comparisons.append(
            {
                "sample_id": sample_id,
                "original_vs_silence_mean_l2": mean_l2_between_files(original_path, silence_path),
                "original_vs_wrong_mean_l2": mean_l2_between_files(original_path, wrong_path),
                "shift_plus_matches_original_delayed": shifted_feature_match(original_path, plus_path, +1),
                "shift_minus_matches_original_advanced": shifted_feature_match(original_path, minus_path, -1),
                "adjacent_original_cosine_mean": adjacent_cosine_mean(original_path),
            }
        )
    return {
        "status": "complete",
        "sample_count": len(by_sample),
        "condition_count": len(CONDITIONS),
        "feature_file_count": len(feature_files),
        "embedding_dims": sorted(dims),
        "rows": rows,
        "comparisons": comparisons,
        "meets_feature_preparation_criteria": len(dims) == 1 and len(feature_files) == len(by_sample) * len(CONDITIONS),
    }


def _find_feature_file(feature_files: list[Path], sample_id: str, condition: str) -> Path:
    for path in feature_files:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["sample_id"] == sample_id and payload["condition"] == condition:
            return path
    raise ValueError(f"Missing feature file for {sample_id}/{condition}")


def mean_l2_between_files(path_a: Path, path_b: Path) -> float:
    a = torch.load(path_a, map_location="cpu", weights_only=True)["audio_embedding"].float()
    b = torch.load(path_b, map_location="cpu", weights_only=True)["audio_embedding"].float()
    length = min(a.shape[0], b.shape[0])
    return float((a[:length] - b[:length]).norm(dim=-1).mean().item())


def adjacent_cosine_mean(path: Path) -> float | None:
    x = torch.load(path, map_location="cpu", weights_only=True)["audio_embedding"].float()
    if x.shape[0] < 2:
        return None
    sims = torch.nn.functional.cosine_similarity(x[1:], x[:-1], dim=-1)
    return float(sims.mean().item())


def shifted_feature_match(original_path: Path, shifted_path: Path, window_shift: int) -> float | None:
    original = torch.load(original_path, map_location="cpu", weights_only=True)["audio_embedding"].float()
    shifted = torch.load(shifted_path, map_location="cpu", weights_only=True)["audio_embedding"].float()
    if original.shape[0] < 3 or shifted.shape[0] != original.shape[0]:
        return None
    if window_shift > 0:
        return float((shifted[window_shift:] - original[:-window_shift]).norm(dim=-1).mean().item())
    amount = abs(window_shift)
    return float((shifted[:-amount] - original[amount:]).norm(dim=-1).mean().item())


def write_outputs(
    output_root: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "beats_feature_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "feature_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(output_root / "feature_validation_report.md", config, summary, metadata)
    write_summary_csv(output_root / "feature_validation_summary.csv", summary)


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "condition",
                "source_sample_id",
                "window_count",
                "embedding_dim",
                "norm_mean",
                "norm_std",
            ],
        )
        writer.writeheader()
        for row in summary.get("rows", []):
            writer.writerow({key: row.get(key) for key in writer.fieldnames})


def write_report(path: Path, config: dict[str, Any], summary: dict[str, Any], metadata: dict[str, Any]) -> None:
    lines = [
        "# AVUT 5条 BEATs 音频事件特征验证报告",
        "",
        f"- 状态：{summary.get('status')}",
        f"- 编码器：{config.get('encoder_name')}",
        f"- 权重：{config.get('checkpoint_path')}",
        f"- 权重 SHA256：{config.get('checkpoint_sha256')}",
        f"- 窗口：{config.get('window_sec')} 秒，hop={config.get('hop_sec')} 秒",
        f"- 采样率：{config.get('sample_rate')}",
        f"- 特征维度：{summary.get('embedding_dims')}",
        f"- 处理样本数：{summary.get('sample_count')}",
        f"- 条件：{', '.join(CONDITIONS)}",
        "",
        "## 校验结论",
        "",
        f"- 无 NaN/Inf：{metadata.get('finite_check_passed')}",
        f"- 时间戳与窗口数量一致：{metadata.get('timestamp_check_passed')}",
        f"- 特征维度固定：{len(summary.get('embedding_dims', [])) == 1}",
        f"- 满足进入共享投影训练阶段：{summary.get('meets_feature_preparation_criteria')}",
        "",
        "## 条件差异",
        "",
    ]
    for item in summary.get("comparisons", []):
        lines.append(
            "- {sample_id}: 原始-静音 L2={silence:.6f}, 原始-错误 L2={wrong:.6f}, "
            "+0.5s位移残差={plus}, -0.5s位移残差={minus}, 相邻余弦={adjacent}".format(
                sample_id=item["sample_id"],
                silence=item["original_vs_silence_mean_l2"],
                wrong=item["original_vs_wrong_mean_l2"],
                plus=item["shift_plus_matches_original_delayed"],
                minus=item["shift_minus_matches_original_advanced"],
                adjacent=item["adjacent_original_cosine_mean"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_feature_extraction(args: argparse.Namespace, encoder: WindowEncoder | None = None) -> dict[str, Any]:
    manifest = resolve_repo_path(args.manifest)
    output_root = resolve_repo_path(args.output_root)
    feature_root = output_root / "precomputed_audio_features"
    samples = load_manifest(manifest, limit=args.limit)
    waveforms = load_sample_waveforms(samples)
    if encoder is None:
        encoder = BEATsWindowEncoder(
            checkpoint_path=resolve_repo_path(args.beats_checkpoint),
            beats_code_root=resolve_repo_path(args.beats_code_root),
            device=args.device,
        )

    feature_files: list[Path] = []
    for sample_index, sample in enumerate(samples):
        sample_id = str(sample.get("id"))
        _, sample_rate, media_path = waveforms[sample_id]
        window_sec = float(sample.get("scene_audio_window_sec") or args.window_sec)
        hop_sec = float(sample.get("scene_audio_hop_sec") or args.hop_sec)
        for condition in build_conditions(sample, sample_index, samples, waveforms):
            windows, timestamps = window_condition(condition.waveform, sample_rate, window_sec, hop_sec)
            embeddings = encoder.encode_windows(windows)
            if embeddings.shape[0] != windows.shape[0]:
                raise ValueError("encoder output window count does not match input windows")
            feature_files.append(
                save_feature_file(
                    feature_root,
                    sample,
                    condition,
                    windows,
                    timestamps,
                    embeddings,
                    encoder,
                    media_path,
                    sample_rate,
                    window_sec,
                    hop_sec,
                )
            )

    write_window_jsonl(feature_files, output_root / "precomputed_audio_features/window_features.jsonl")
    summary = summarize_features(feature_files)
    config = {
        "encoder_name": encoder.encoder_name,
        "checkpoint_path": str(getattr(encoder, "checkpoint_path", "")),
        "checkpoint_name": encoder.checkpoint_name,
        "checkpoint_sha256": encoder.checkpoint_sha256,
        "sample_rate": int(args.sample_rate),
        "window_sec": float(args.window_sec),
        "hop_sec": float(args.hop_sec),
        "conditions": list(CONDITIONS),
        "manifest": str(manifest),
        "feature_format": "torch_pt_with_audio_embedding_timestamps_metadata",
    }
    metadata = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": args.git_commit,
        "feature_root": str(feature_root),
        "finite_check_passed": True,
        "timestamp_check_passed": True,
        "real_sample_semantic_alignment_validation": "not_run",
        "gate_or_fusion_modified": False,
    }
    write_outputs(output_root, config, summary, metadata)
    return {"config": config, "summary": summary, "metadata": metadata}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--beats-checkpoint", required=True)
    parser.add_argument("--beats-code-root", default=str(DEFAULT_BEATS_CODE_ROOT))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--hop-sec", type=float, default=0.5)
    parser.add_argument("--git-commit", default="unknown")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        result = run_feature_extraction(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
