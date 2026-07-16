"""CPU harness for the minimal AS-M4 streaming AV fusion path."""

from __future__ import annotations

import torch

from intersuit.model.llava_arch import LlavaMetaForCausalLM
from intersuit.model.scene_audio_encoder.scene_audio_encoder import SceneAudioEncoderOutput
from intersuit.model.streaming_av.builder import build_streaming_av_module


class DummyConfig:
    hidden_size = 4
    num_audio_events = 2
    audio_quality_dim = 1
    streaming_av_align_dim = 4
    max_av_offset_sec = 2.0
    av_similarity_chunk_size = 1
    as_m4_fusion_init = "zero"
    as_m4_gate_logit_bias = -5.0


class DummyStreamingModel(LlavaMetaForCausalLM):
    def __init__(self, fusion_init: str = "zero"):
        self.config = DummyConfig()
        self.config.as_m4_fusion_init = fusion_init
        self.device = torch.device("cpu")
        self.streaming_av_module = build_streaming_av_module(self.config)

    def get_model(self):
        return self

    def get_streaming_av_module(self):
        return self.streaming_av_module


def _scene_output():
    features = torch.ones(1, 2, 4) * 0.5
    mask = torch.ones(1, 2, dtype=torch.bool)
    return SceneAudioEncoderOutput(features=features, mask=mask)


def test_streaming_av_fusion_changes_video_features_when_gate_enabled():
    model = DummyStreamingModel(fusion_init="identity")
    video = torch.zeros(2, 3, 4)

    fused = model.fuse_scene_audio_into_image_features(
        [video],
        ["video"],
        _scene_output(),
        scene_audio_timestamps=torch.tensor([[[0.0, 1.0], [1.0, 2.0]]]),
        frame_timestamps=torch.tensor([[0.5, 1.5]]),
        lookahead_sec=0.0,
    )[0]

    assert fused.shape == video.shape
    assert not torch.allclose(fused, video)


def test_force_gate_zero_recovers_video_features():
    model = DummyStreamingModel()
    video = torch.randn(2, 3, 4)

    fused = model.fuse_scene_audio_into_image_features(
        [video],
        ["video"],
        _scene_output(),
        scene_audio_timestamps=torch.tensor([[[0.0, 1.0], [1.0, 2.0]]]),
        frame_timestamps=torch.tensor([[0.5, 1.5]]),
        force_audio_gate=0.0,
    )[0]

    assert torch.allclose(fused, video)


def test_debug_residual_scale_zero_recovers_video_with_live_gate():
    model = DummyStreamingModel(fusion_init="identity")
    video = torch.randn(2, 3, 4)

    fused = model.fuse_scene_audio_into_image_features(
        [video],
        ["video"],
        _scene_output(),
        scene_audio_timestamps=torch.tensor([[[0.0, 1.0], [1.0, 2.0]]]),
        frame_timestamps=torch.tensor([[0.5, 1.5]]),
        audio_residual_scale=0.0,
    )[0]

    diagnostics = model._last_streaming_av_diagnostics[0]
    assert torch.equal(fused, video)
    assert diagnostics["audio_residual_scale"] == 0.0
    assert diagnostics["delta_norm"].item() == 0.0
    assert diagnostics["delta_to_video_ratio"].item() == 0.0


def test_gate_zero_remains_exactly_zero_with_cap():
    model = DummyStreamingModel(fusion_init="identity")
    video = torch.randn(2, 3, 4)

    fused = model.fuse_scene_audio_into_image_features(
        [video],
        ["video"],
        _scene_output(),
        scene_audio_timestamps=torch.tensor([[[0.0, 1.0], [1.0, 2.0]]]),
        frame_timestamps=torch.tensor([[0.5, 1.5]]),
        force_audio_gate=0.0,
        audio_delta_ratio_cap=0.03,
    )[0]

    diagnostics = model._last_streaming_av_diagnostics[0]
    assert torch.equal(fused, video)
    assert diagnostics["gate_mean"].item() == 0.0
    assert diagnostics["raw_delta_norm"].item() == 0.0
    assert diagnostics["delta_norm"].item() == 0.0
    assert diagnostics["delta_to_video_ratio"].item() == 0.0


def test_non_video_features_pass_through_unchanged():
    model = DummyStreamingModel()
    image = torch.randn(1, 3, 4)

    fused = model.fuse_scene_audio_into_image_features(
        [image],
        ["image"],
        _scene_output(),
    )[0]

    assert torch.allclose(fused, image)


def test_streaming_av_modules_receive_gradients_from_fused_video_loss():
    model = DummyStreamingModel(fusion_init="identity")
    video = torch.zeros(2, 3, 4, requires_grad=True)

    fused = model.fuse_scene_audio_into_image_features(
        [video],
        ["video"],
        _scene_output(),
        scene_audio_timestamps=torch.tensor([[[0.0, 1.0], [1.0, 2.0]]]),
        frame_timestamps=torch.tensor([[0.5, 1.5]]),
    )[0]
    loss = fused.square().mean()
    loss.backward()

    grad_names = [
        name
        for name, param in model.streaming_av_module.named_parameters()
        if param.requires_grad and param.grad is not None and torch.isfinite(param.grad).all()
    ]
    assert any("fusion" in name for name in grad_names)
    assert any("confidence_gate" in name for name in grad_names)


if __name__ == "__main__":
    test_streaming_av_fusion_changes_video_features_when_gate_enabled()
    test_force_gate_zero_recovers_video_features()
    test_non_video_features_pass_through_unchanged()
    test_streaming_av_modules_receive_gradients_from_fused_video_loss()
    print("streaming_av_integration harness passed")
