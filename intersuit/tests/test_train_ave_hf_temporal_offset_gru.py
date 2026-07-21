"""连续 AVE offset GRU 训练脚本的 CPU 辅助测试。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_ave_hf_temporal_offset_gru.py"
SPEC = importlib.util.spec_from_file_location("train_ave_hf_temporal_offset_gru", SCRIPT)
trainer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)


def _record(length: int, syncable: float, target: int):
    return {
        "candidate_logits": torch.randn(length, 3),
        "candidate_features": torch.randn(length, 3, 128),
        "evidence": torch.randn(length, 8),
        "offset_targets": torch.full((length,), target, dtype=torch.long),
        "sync_targets": torch.full((length,), syncable),
    }


def test_split_is_disjoint_and_deterministic():
    ids = [f"video-{index}" for index in range(20)]

    first = trainer.split_youtube_ids(ids, dev_ratio=0.2, seed=7)
    second = trainer.split_youtube_ids(ids, dev_ratio=0.2, seed=7)

    assert first == second
    assert set(first[0]).isdisjoint(first[1])
    assert len(first[1]) == 4


def test_frozen_test_manifest_is_rejected_before_read():
    with pytest.raises(ValueError, match="冻结 test manifest"):
        trainer.assert_development_source(trainer.FROZEN_TEST)


def test_collate_and_three_losses_are_finite():
    records = [_record(5, 1.0, 1), _record(9, 0.0, 1)]
    batch = trainer.collate(records)
    model = trainer.TemporalOffsetGRUDiagnostic()

    loss, metrics = trainer.compute_loss(
        model,
        batch,
        emd_weight=0.25,
        sync_weight=1.0,
    )

    assert batch["mask"].sum().item() == 14
    assert torch.isfinite(loss)
    assert set(metrics) == {"loss", "offset_ce", "offset_emd", "sync_bce"}
