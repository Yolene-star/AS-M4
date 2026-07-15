"""CPU harness for AS-M4 dataset collator scene-audio fields."""

from __future__ import annotations

import torch
import types
import sys
import importlib.machinery

deepspeed_stub = types.ModuleType("deepspeed")
deepspeed_stub.__spec__ = importlib.machinery.ModuleSpec("deepspeed", loader=None)
sys.modules.setdefault("deepspeed", deepspeed_stub)
trainer_stub = types.ModuleType("intersuit.train.llava_trainer")
trainer_stub.__spec__ = importlib.machinery.ModuleSpec("intersuit.train.llava_trainer", loader=None)
trainer_stub.LLaVATrainer = object
sys.modules.setdefault("intersuit.train.llava_trainer", trainer_stub)
from intersuit.train.train import DataCollatorForSupervisedDataset


class DummyTokenizer:
    padding_side = "right"
    pad_token_id = 0
    model_max_length = 16


def _base_instance(token_offset=0):
    input_ids = torch.tensor([1 + token_offset, 2 + token_offset], dtype=torch.long)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "labels": labels}


def test_scene_audio_mixed_batch_pads_and_masks():
    collator = DataCollatorForSupervisedDataset(tokenizer=DummyTokenizer())
    a = _base_instance()
    a["scene_audio"] = torch.ones(3, 4)
    a["scene_audio_timestamps"] = torch.tensor([[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]])
    b = _base_instance(2)
    c = _base_instance(4)
    c["scene_audio"] = torch.ones(1, 4) * 2
    c["scene_audio_timestamps"] = torch.tensor([[0.0, 1.0]])

    batch = collator([a, b, c])

    assert batch["scene_audios"].shape == (3, 3, 4)
    assert batch["scene_audio_mask"].tolist() == [
        [True, True, True],
        [False, False, False],
        [True, False, False],
    ]
    assert batch["scene_audio_timestamps"].shape == (3, 3, 2)
    assert torch.all(batch["scene_audios"][1] == 0)


def test_old_speech_collate_path_still_stacks():
    collator = DataCollatorForSupervisedDataset(tokenizer=DummyTokenizer())
    a = _base_instance()
    b = _base_instance(2)
    a["speech"] = [(torch.ones(5, 80), torch.LongTensor([5]))]
    b["speech"] = [(torch.ones(5, 80) * 2, torch.LongTensor([5]))]

    batch = collator([a, b])

    assert batch["speeches"].shape == (2, 5, 80)
    assert batch["speech_lengths"].shape == (2, 1)
    assert "scene_audios" not in batch


def test_precomputed_video_features_collate_as_video_feature_modality():
    collator = DataCollatorForSupervisedDataset(tokenizer=DummyTokenizer())
    a = _base_instance()
    b = _base_instance(2)
    a["image"] = [(torch.ones(2, 3, 4), (0, 0), "video_feature")]
    b["image"] = [(torch.ones(2, 3, 4) * 2, (0, 0), "video_feature")]

    batch = collator([a, b])

    assert batch["images"].shape == (2, 2, 3, 4)
    assert batch["modalities"] == ["video_feature", "video_feature"]


if __name__ == "__main__":
    test_scene_audio_mixed_batch_pads_and_masks()
    test_old_speech_collate_path_still_stacks()
    test_precomputed_video_features_collate_as_video_feature_modality()
    print("dataset_collator harness passed")
