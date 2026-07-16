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
    module = GatedAVFusion(hidden_size=4, fusion_init="identity")
    video = torch.zeros(1, 2, 3, 4)
    audio = torch.ones(1, 2, 4)
    gate = torch.ones(1, 2)

    fused = module(video, audio, gate)

    assert fused.shape == video.shape
    assert torch.all(fused != 0)


def test_zero_init_gate_one_recovers_video_tokens():
    module = GatedAVFusion(hidden_size=4, fusion_init="zero")
    video = torch.randn(1, 2, 3, 4)
    audio = torch.randn(1, 2, 4)
    gate = torch.ones(1, 2)

    fused = module(video, audio, gate)

    assert torch.allclose(fused, video)


def test_zero_init_projector_receives_gradients():
    module = GatedAVFusion(hidden_size=4, fusion_init="zero")
    video = torch.zeros(1, 2, 3, 4)
    audio = torch.randn(1, 2, 4)
    gate = torch.ones(1, 2)

    loss = module(video, audio, gate).sum()
    loss.backward()

    assert module.audio_projector.weight.grad is not None
    assert torch.any(module.audio_projector.weight.grad != 0)


def test_gate_zero_masks_nan_audio_delta():
    module = GatedAVFusion(hidden_size=4, fusion_init="identity")
    video = torch.randn(1, 2, 3, 4)
    audio = torch.full((1, 2, 4), float("nan"))
    gate = torch.zeros(1, 2)

    fused = module(video, audio, gate)

    assert torch.allclose(fused, video)
    assert torch.isfinite(fused).all()


def test_token_level_gate_broadcasts():
    module = GatedAVFusion(hidden_size=4, fusion_init="identity")
    video = torch.zeros(1, 1, 2, 4)
    audio = torch.ones(1, 1, 4)
    gate = torch.tensor([[[1.0, 0.0]]])

    fused = module(video, audio, gate)

    assert torch.all(fused[0, 0, 0] != 0)
    assert torch.all(fused[0, 0, 1] == 0)


def test_debug_residual_scale_defaults_to_one_and_scales_delta():
    module = GatedAVFusion(hidden_size=4, fusion_init="identity")
    video = torch.zeros(1, 1, 2, 4)
    audio = torch.ones(1, 1, 4)
    gate = torch.ones(1, 1)

    default = module(video, audio, gate)
    explicit_one = module(video, audio, gate, residual_scale=1.0)
    half = module(video, audio, gate, residual_scale=0.5)
    zero = module(video, audio, gate, residual_scale=0.0)

    assert torch.equal(default, explicit_one)
    assert torch.allclose(half, default * 0.5)
    assert torch.equal(zero, video)


if __name__ == "__main__":
    test_fusion_preserves_video_token_shape()
    test_gate_zero_exactly_recovers_video_tokens()
    test_gate_one_changes_nonzero_audio_tokens()
    test_zero_init_gate_one_recovers_video_tokens()
    test_zero_init_projector_receives_gradients()
    test_gate_zero_masks_nan_audio_delta()
    test_token_level_gate_broadcasts()
    print("fusion harness passed")
