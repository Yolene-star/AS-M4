#!/usr/bin/env python
"""离线提取 M4 视觉塔、projector 与时空池化之后的视频特征。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from intersuit.harness.runners.run_predictions_from_plan import (
    _load_video_tensor,
    load_model_once,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"
DEFAULT_MODEL = INTERSUIT_ROOT / "checkpoints/M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    qa_path = Path(args.qa_manifest).resolve()
    output_root = Path(args.output_root).resolve()
    rows = json.loads(qa_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("QA manifest 必须是非空 JSON list")
    _, model, image_processor, _ = load_model_once(
        str(Path(args.model_path).resolve()),
        args.device,
        model_name_override=args.model_name_override,
    )
    output_rows = []
    shapes = {}
    for qa in rows:
        sample_id = str(qa["id"])
        output_path = output_root / "features" / f"{sample_id}.pt"
        if args.resume and output_path.is_file():
            payload = torch.load(output_path, map_location="cpu", weights_only=True)
            features = payload.get("features") if isinstance(payload, dict) else payload
            if not isinstance(features, torch.Tensor):
                raise ValueError(f"{sample_id} 的已有 M4 视频特征非法")
        else:
            frames = _load_video_tensor(
                str(qa["video_path"]),
                image_processor,
                model,
                args.device,
                max_frames=qa.get("video_max_frames"),
            )
            features = model.encode_multimodals(
                frames,
                [0],
                [int(frames.shape[0])],
            )[0]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "features": features.detach().cpu(),
                    "metadata": {
                        "sample_id": sample_id,
                        "source_video": qa["video_path"],
                        "model_path": str(Path(args.model_path).resolve()),
                        "feature_stage": "vision_tower_mm_projector_spatial_pool",
                        "formal_runtime_modified": False,
                    },
                },
                output_path,
            )
        if features.ndim != 3 or not torch.isfinite(features).all():
            raise ValueError(f"{sample_id} 的 M4 视频特征非法")
        row = dict(qa)
        row["video_features"] = str(output_path)
        row.pop("video_path", None)
        output_rows.append(row)
        shapes[sample_id] = list(features.shape)
    manifest_path = output_root / "m4_projected_video_feature_manifest.json"
    write_json(manifest_path, output_rows)
    summary = {
        "sample_count": len(output_rows),
        "model_path": str(Path(args.model_path).resolve()),
        "device": args.device,
        "feature_shapes": shapes,
        "all_finite": True,
        "trains_m4": False,
        "formal_runtime_modified": False,
        "manifest": str(manifest_path),
    }
    write_json(output_root / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--model-name-override", default="LongVA-Qwen2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true", help="校验并复用已有特征")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
