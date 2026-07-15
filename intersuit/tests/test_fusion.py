"""CPU harness for AS-M4 gated audio-video fusion."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "intersuit" / "model" / "streaming_av" / "fusion.py"
SPEC = importlib.util.spec_from_file_location("as_m4_fusion", MODULE_PATH)
fusion = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fusion
SPEC.loader.exec_module(fusion)

GatedAVFusion = fusion.GatedAVFusion


def test_fusion_preserves_video_token_shape():
    module = GatedAVFusion(hidden_size=4)
    video = torch.randn(2, 3, 5, 4)
    audio = torch.randn(2, 3, 4)
    gate = torch.ones(2, 3)

    fused = module(video, audio, gate)

    assert fused.shape == video.shape


def test_gate_zero_exactly_recovers_video_tokens():
    module = GatedAVFusion(hidden_size=4)
    video = torch.randn(1, 2, 3, 4)
    audio = torch.randn(1, 2, 4)
    gate = torch.zeros(1, 2)

    fused = module(video, audio, gate)

    assert torch.allclose(fused, video)


def test_gate_one_changes_nonzero_audio_tokens():
    module = GatedAVFusion(hidden_size=4)
    video = torch.zeros(1, 2, 3, 4)
    audio = torch.ones(1, 2, 4)
    gate = torch.ones(1, 2)

    fused = module(video, audio, gate)

    assert fused.shape == video.shape
    assert torch.all(fused != 0)


def test_token_level_gate_broadcasts():
    module = GatedAVFusion(hidden_size=4)
    video = torch.zeros(1, 1, 2, 4)
    audio = torch.ones(1, 1, 4)
    gate = torch.tensor([[[1.0, 0.0]]])

    fused = module(video, audio, gate)

    assert torch.all(fused[0, 0, 0] != 0)
    assert torch.all(fused[0, 0, 1] == 0)


if __name__ == "__main__":
    test_fusion_preserves_video_token_shape()
    test_gate_zero_exactly_recovers_video_tokens()
    test_gate_one_changes_nonzero_audio_tokens()
    test_token_level_gate_broadcasts()
    print("fusion harness passed")

