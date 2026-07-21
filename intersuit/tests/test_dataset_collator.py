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
import intersuit.train.train as train_module
from intersuit.train.train import (
    DataArguments,
    DataCollatorForSupervisedDataset,
    ModelArguments,
    TrainingArguments,
)


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


def test_scene_audio_negative_batch_pads_and_masks():
    collator = DataCollatorForSupervisedDataset(tokenizer=DummyTokenizer())
    a = _base_instance()
    a["scene_audio_negative"] = torch.ones(2, 4)
    a["scene_audio_negative_timestamps"] = torch.tensor([[0.0, 1.0], [0.5, 1.5]])
    b = _base_instance(2)

    batch = collator([a, b])

    assert batch["scene_audio_negatives"].shape == (2, 2, 4)
    assert batch["scene_audio_negative_mask"].tolist() == [[True, True], [False, False]]


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


def test_scene_audio_path_uses_media_windows_and_timestamps(monkeypatch):
    expected_audio = torch.ones(2, 8)
    expected_timestamps = torch.tensor([[0.0, 1.0], [0.5, 1.5]])

    def fake_load(path, folder, sample_rate, window_sec, hop_sec):
        assert path == "sample.mp4"
        assert folder == "audio"
        assert sample_rate == 16000
        assert window_sec == 1.0
        assert hop_sec == 0.5
        return expected_audio, expected_timestamps

    monkeypatch.setattr(train_module, "_load_scene_audio_media", fake_load)
    dataset = object.__new__(train_module.LazySupervisedDataset)
    dataset.data_args = types.SimpleNamespace(
        scene_audio_folder="audio",
        scene_audio_feature_folder=None,
        scene_audio_data_sample_rate=16000,
    )

    result = dataset._load_scene_audio_fields(
        {
            "scene_audio_path": "sample.mp4",
            "scene_audio_sample_rate": 16000,
            "scene_audio_window_sec": 1.0,
            "scene_audio_hop_sec": 0.5,
        }
    )

    assert torch.equal(result["scene_audio"], expected_audio)
    assert torch.equal(result["scene_audio_timestamps"], expected_timestamps)


def test_training_argument_parser_has_unique_scene_audio_options():
    parser = train_module.transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--scene_audio_sample_rate" in option_strings
    assert "--scene_audio_data_sample_rate" in option_strings


if __name__ == "__main__":
    test_scene_audio_mixed_batch_pads_and_masks()
    test_old_speech_collate_path_still_stacks()
    test_precomputed_video_features_collate_as_video_feature_modality()
    print("dataset_collator harness passed")
