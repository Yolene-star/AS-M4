"""CPU tests for AVE_HF video feature comparison helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "compare_ave_hf_video_features.py"
SPEC = importlib.util.spec_from_file_location("compare_ave_hf_video_features", SCRIPT_PATH)
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)


def _write_video(path: Path, values: torch.Tensor) -> None:
    timestamps = torch.stack([torch.arange(values.shape[0]).float(), torch.arange(values.shape[0]).float() + 1.0], dim=1)
    torch.save({"video_features": values.float(), "timestamps": timestamps}, path)


def test_diagnose_manifest_reports_label_separation(tmp_path):
    rows = []
    for idx, (label, base) in enumerate([("a", 0.0), ("a", 0.2), ("b", 5.0), ("b", 5.2)]):
        path = tmp_path / f"v{idx}.pt"
        values = torch.tensor([[base, 0.0], [base + 0.05, 0.0]])
        _write_video(path, values)
        rows.append({"youtube_id": f"v{idx}", "label": label, "video_feature_path": str(path)})

    diag = compare.diagnose_manifest(rows, seed=1, max_pairs=20)

    assert diag["all_finite"]
    assert diag["continuity_ok"]
    assert diag["label_separation_trend"]
    assert not diag["collapse_reasons"]
