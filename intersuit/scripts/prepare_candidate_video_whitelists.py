#!/usr/bin/env python
"""从 projector 候选 manifest 生成 LLP/AVE 小批量视频 ID 白名单。"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"
DEFAULT_CANDIDATES = INTERSUIT_ROOT / "harness/artifacts/audio_visual_event_manifests/projector_positive_candidates.jsonl"
DEFAULT_OUTPUT_ROOT = INTERSUIT_ROOT / "harness/artifacts/audio_visual_event_download_prep"


def infer_video_id(dataset: str, sample_id: str) -> str:
    if dataset == "LLP":
        parts = sample_id.rsplit("_", 2)
        if len(parts) == 3 and parts[-1].replace(".", "", 1).isdigit() and parts[-2].replace(".", "", 1).isdigit():
            return parts[0]
        return sample_id[:11]
    return sample_id


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(Path(args.candidates).resolve())
    by_dataset: dict[str, OrderedDict[str, None]] = {"LLP": OrderedDict(), "AVE": OrderedDict()}
    for row in rows:
        dataset = str(row.get("dataset"))
        if dataset not in by_dataset:
            continue
        by_dataset[dataset].setdefault(infer_video_id(dataset, str(row.get("sample_id"))), None)

    output_root = Path(args.output_root).resolve()
    llp_ids = list(by_dataset["LLP"].keys())[: args.llp_limit]
    ave_ids = list(by_dataset["AVE"].keys())[: args.ave_limit]
    paths = {
        "llp_whitelist": str(output_root / "llp_candidate_video_ids.txt"),
        "ave_candidate_video_ids": str(output_root / "ave_candidate_video_ids.txt"),
        "summary": str(output_root / "candidate_video_whitelist_summary.json"),
    }
    write_lines(Path(paths["llp_whitelist"]), llp_ids)
    write_lines(Path(paths["ave_candidate_video_ids"]), ave_ids)
    summary = {
        "candidates": str(Path(args.candidates).resolve()),
        "unique_counts": {dataset: len(values) for dataset, values in by_dataset.items()},
        "selected_counts": {"LLP": len(llp_ids), "AVE": len(ave_ids)},
        "requested_limits": {"LLP": args.llp_limit, "AVE": args.ave_limit},
        "LLP": llp_ids,
        "AVE": ave_ids,
        "paths": paths,
    }
    Path(paths["summary"]).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--llp-limit", type=int, default=20)
    parser.add_argument("--ave-limit", type=int, default=20)
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps({"ok": True, "summary": summary["paths"]["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
