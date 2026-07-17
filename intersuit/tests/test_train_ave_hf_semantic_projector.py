"""CPU tests for AVE_HF semantic projector helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "train_ave_hf_semantic_projector.py"
SPEC = importlib.util.spec_from_file_location("train_ave_hf_semantic_projector", SCRIPT_PATH)
semantic = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = semantic
SPEC.loader.exec_module(semantic)


def _write_feature(path: Path, key: str, values: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = torch.stack([torch.arange(values.shape[0]).float(), torch.arange(values.shape[0]).float() + 1.0], dim=1)
    torch.save({key: values.float(), "timestamps": timestamps}, path)


def _make_rows(tmp_path: Path) -> list[dict]:
    rows = []
    for index, label in enumerate(["a", "a", "b", "b"]):
        youtube_id = f"v{index}"
        audio_path = tmp_path / f"{youtube_id}_audio.pt"
        video_path = tmp_path / f"{youtube_id}_video.pt"
        values = torch.eye(4)[:2] + index
        _write_feature(audio_path, "audio_embedding", values)
        _write_feature(video_path, "video_features", values)
        rows.append({"youtube_id": youtube_id, "label": label, "audio_feature_path": str(audio_path), "video_feature_path": str(video_path)})
    return rows


def test_build_semantic_pairs_uses_only_cross_label_negatives(tmp_path):
    rows = _make_rows(tmp_path)

    pairs = semantic.build_semantic_pairs(rows, negatives_per_positive=1, seed=1)

    assert {pair.pair_type for pair in pairs} == {"semantic_positive", "cross_label_negative"}
    assert not any(pair.source_label == pair.target_label and pair.target == 0 for pair in pairs)
    assert not any(pair.pair_type == "silence_negative" for pair in pairs)
    assert not any(pair.pair_type == "shifted_negative" for pair in pairs)


def test_audit_old_pairs_counts_same_label_wrong_and_shifted(tmp_path):
    path = tmp_path / "old.jsonl"
    rows = [
        {"youtube_id": "a", "label": "x", "pair_type": "positive"},
        {"youtube_id": "b", "label": "x", "pair_type": "positive"},
        {"youtube_id": "a", "label": "x", "pair_type": "wrong_audio_negative", "negative_source_youtube_id": "b"},
        {"youtube_id": "a", "label": "x", "pair_type": "shifted_negative"},
        {"youtube_id": "a", "label": "x", "pair_type": "silence_negative"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    audit = semantic.audit_old_pairs(path)

    assert audit["old_wrong_audio_same_label_count"] == 1
    assert audit["old_shifted_suspect_false_negative_count"] == 1
    assert audit["old_silence_negative_count"] == 1
