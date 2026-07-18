#!/usr/bin/env python
"""为 AVUT 离线软窗口实验生成冻结 offset scorer 三路特征清单。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"
DEFAULT_QA = INTERSUIT_ROOT / "inputs/texts/avut/avut_audio_smoke.json"
DEFAULT_AUDIO_ROOT = INTERSUIT_ROOT / "harness/artifacts/avut_beats_features_real/precomputed_audio_features"
DEFAULT_RGB_ROOT = INTERSUIT_ROOT / "harness/artifacts/avut_projector_training_real/video_window_features"
DEFAULT_OUTPUT = INTERSUIT_ROOT / "harness/artifacts/video_window_weighting_avut_smoke/offset_inputs"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> dict[str, str]:
    qa_path = Path(args.qa_manifest).resolve()
    audio_root = Path(args.audio_root).resolve()
    rgb_root = Path(args.rgb_root).resolve()
    output_root = Path(args.output_root).resolve()
    rows = json.loads(qa_path.read_text(encoding="utf-8"))
    clip_sources, dev_rows, rgb_rows = [], [], []
    for qa in rows:
        sample_id = str(qa["id"])
        video_path = Path(str(qa["video_path"]))
        if not video_path.is_absolute():
            video_path = INTERSUIT_ROOT / video_path
        audio_path = audio_root / sample_id / "original.pt"
        rgb_path = rgb_root / f"{sample_id}.pt"
        audio_payload = torch.load(audio_path, map_location="cpu", weights_only=True)
        rgb_payload = torch.load(rgb_path, map_location="cpu", weights_only=True)
        timestamps = audio_payload["timestamps"].float()
        if not torch.allclose(
            timestamps,
            rgb_payload["timestamps"].float(),
            atol=1e-5,
            rtol=0.0,
        ):
            raise ValueError(f"{sample_id} 的 BEATs/RGB 时间戳不一致")
        common = {
            "youtube_id": sample_id,
            "sample_id": sample_id,
            "split": "smoke",
            "label": qa.get("answer"),
        }
        clip_sources.append(
            {
                **common,
                "audio_feature_path": str(audio_path),
                "video_feature_path": str(rgb_path),
                "video_path": str(video_path),
            }
        )
        dev_rows.append(common)
        rgb_rows.append(
            {
                **common,
                "audio_feature_path": str(audio_path),
                "video_feature_path": str(rgb_path),
                "audio_path": str(video_path),
                "video_path": str(video_path),
            }
        )
    paths = {
        "clip_source_manifest": str(output_root / "clip_source_manifest.jsonl"),
        "dev_manifest": str(output_root / "dev_manifest.jsonl"),
        "rgb_manifest": str(output_root / "rgb_manifest.jsonl"),
    }
    write_jsonl(Path(paths["clip_source_manifest"]), clip_sources)
    write_jsonl(Path(paths["dev_manifest"]), dev_rows)
    write_jsonl(Path(paths["rgb_manifest"]), rgb_rows)
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "sample_count": len(rows),
                "diagnostic_only": True,
                "test_set_read": False,
                "paths": paths,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-manifest", default=str(DEFAULT_QA))
    parser.add_argument("--audio-root", default=str(DEFAULT_AUDIO_ROOT))
    parser.add_argument("--rgb-root", default=str(DEFAULT_RGB_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
