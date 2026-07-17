"""CPU tests for LLP/AVE/AVQA unified event manifest preparation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "prepare_audio_visual_event_manifests.py"
SPEC = importlib.util.spec_from_file_location("prepare_audio_visual_event_manifests", SCRIPT_PATH)
manifests = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = manifests
SPEC.loader.exec_module(manifests)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_h5_order(path: Path, values: list[int]) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("order", data=values)


def _make_llp(root: Path) -> None:
    data = root / "LLP" / "data"
    _write(data / "AVVP_train.csv", "filename\tevent_labels\ntrain_video_0_10\tDog\n")
    _write(data / "AVVP_val_pd.csv", "filename\tevent_labels\nsample_a_0_10\tDog,Speech\n")
    _write(data / "AVVP_test_pd.csv", "filename\tevent_labels\nsample_b_0_10\tSpeech\n")
    _write(
        data / "AVVP_eval_audio.csv",
        "filename,onset,offset,event_labels\n"
        "sample_a_0_10,0,2,Dog\n"
        "sample_a_0_10,4,5,Speech\n"
        "sample_b_0_10,1,3,Speech\n",
    )
    _write(
        data / "AVVP_eval_visual.csv",
        "filename,onset,offset,event_labels\n"
        "sample_a_0_10,0,2,Dog\n"
        "sample_a_0_10,4,5,Car\n"
        "sample_b_0_10,7,8,Speech\n",
    )


def _make_ave(root: Path) -> None:
    data = root / "AVE" / "data"
    _write(
        data / "Annotations.txt",
        "Dog&ave_train&good&0&2\n"
        "Car&ave_val&good&2&4\n"
        "Speech&ave_test&good&0&0\n",
    )
    _write_h5_order(data / "train_order.h5", [0])
    _write_h5_order(data / "val_order.h5", [1])
    _write_h5_order(data / "test_order.h5", [2])


def test_llp_dense_records_classify_audio_visual_audio_only_visual_only_and_background(tmp_path):
    _make_llp(tmp_path)

    records, summary = manifests.build_llp_records(tmp_path / "LLP")

    dog_records = [row for row in records if row.sample_id == "sample_a_0_10" and row.event_label == "Dog"]
    assert any(row.modality_role == "audio_visual" for row in dog_records)
    assert any(row.modality_role == "audio_only" and row.event_label == "Speech" for row in records)
    assert any(row.modality_role == "visual_only" and row.event_label == "Car" for row in records)
    assert any(row.modality_role == "background" for row in records)
    assert summary["weak_split_counts"] == {"train": 1, "val": 1, "test": 1}
    assert summary["role_counts"]["audio_visual"] > 0


def test_ave_records_expand_event_intervals_and_skip_empty_intervals(tmp_path):
    pytest.importorskip("h5py")
    _make_ave(tmp_path)

    records, summary = manifests.build_ave_records(tmp_path / "AVE")

    assert {row.sample_id for row in records} == {"ave_train", "ave_val"}
    assert all(row.modality_role == "audio_visual" for row in records)
    assert {row.split for row in records} == {"train", "val"}
    assert summary["skipped"]["empty_or_invalid_interval"] == 1
    assert summary["split_counts"] == {"train": 1, "val": 1, "test": 1}


def test_ave_split_rejects_out_of_range_index(tmp_path):
    pytest.importorskip("h5py")
    _make_ave(tmp_path)
    _write_h5_order(tmp_path / "AVE" / "data" / "test_order.h5", [99])

    with pytest.raises(ValueError, match="越界索引"):
        manifests.build_ave_records(tmp_path / "AVE")


def test_avqa_summary_skips_missing_files(tmp_path):
    summary = manifests.summarize_avqa(tmp_path / "AVQA", per_role_limit=2)

    assert summary["available"] is False
    assert "train_qa.json" in summary["missing_files"][0]


def test_run_writes_unified_manifests_and_projector_candidates(tmp_path):
    pytest.importorskip("h5py")
    _make_llp(tmp_path / "datasets")
    _make_ave(tmp_path / "datasets")
    output_root = tmp_path / "out"
    args = argparse.Namespace(
        dataset_root=str(tmp_path / "datasets"),
        llp_root=None,
        ave_root=None,
        avqa_root=None,
        output_root=str(output_root),
        llp_audio_visual_limit=3,
        ave_video_limit=2,
        avqa_per_role_limit=2,
        media_root=[],
        media_check_limit=5,
        window_sec=1.0,
        hop_sec=0.5,
    )

    summary = manifests.run(args)

    unified = output_root / "unified_audio_visual_manifest.jsonl"
    candidates = output_root / "projector_positive_candidates.jsonl"
    assert unified.is_file()
    assert candidates.is_file()
    candidate_rows = [json.loads(line) for line in candidates.read_text(encoding="utf-8").splitlines()]
    assert len(candidate_rows) <= 5
    assert all(row["modality_role"] == "audio_visual" for row in candidate_rows)
    assert summary["projector_candidate_counts"]["LLP_audio_visual"] == 3
    assert summary["media_validation"]["enabled"] is False
    assert (output_root / "annotation_report.md").is_file()
