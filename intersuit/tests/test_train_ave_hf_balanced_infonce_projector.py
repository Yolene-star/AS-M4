"""CPU tests for balanced InfoNCE AVE_HF projector helpers."""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "train_ave_hf_balanced_infonce_projector.py"
SPEC = importlib.util.spec_from_file_location("train_ave_hf_balanced_infonce_projector", SCRIPT_PATH)
balanced = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = balanced
SPEC.loader.exec_module(balanced)


def _make_items() -> list:
    items = []
    for label in ["a", "b", "c"]:
        for video_idx in range(3):
            for window_idx in range(2):
                value = float(len(items) + 1)
                items.append(
                    balanced.WindowItem(
                        youtube_id=f"{label}_v{video_idx}",
                        label=label,
                        window_index=window_idx,
                        audio=torch.ones(4) * value,
                        video=torch.ones(5) * value,
                    )
                )
    return items


def test_same_label_mask_masks_only_non_diagonal_same_label_entries():
    mask = balanced.same_label_non_diagonal_mask(["a", "a", "b", "c", "c"])

    assert mask.tolist() == [
        [False, True, False, False, False],
        [True, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, True],
        [False, False, False, True, False],
    ]


def test_balanced_sampler_uses_equal_labels_and_distinct_videos():
    grouped = balanced.group_items(_make_items())

    _, _, labels, video_ids, _ = balanced.sample_balanced_batch(grouped, labels_per_batch=2, videos_per_label=3, rng=random.Random(1))

    counts = {label: labels.count(label) for label in set(labels)}
    assert sorted(counts.values()) == [3, 3]
    for label in counts:
        ids = [video_id for item_label, video_id in zip(labels, video_ids) if item_label == label]
        assert len(ids) == len(set(ids))


def test_masked_symmetric_infonce_loss_is_finite_with_same_label_duplicates():
    similarity = torch.eye(4)
    loss, stats = balanced.masked_symmetric_infonce_loss(similarity, ["a", "a", "b", "c"], torch.tensor(0.07))

    assert torch.isfinite(loss)
    assert stats["masked_same_label_entries"] == 2
