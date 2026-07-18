"""CPU harness for the minimal AS-M4 streaming AV fusion path."""

from __future__ import annotations

from functools import wraps

import pytest
import torch

from intersuit.model.llava_arch import LlavaMetaForCausalLM
from intersuit.model.scene_audio_encoder.scene_audio_encoder import SceneAudioEncoderOutput
from intersuit.model.streaming_av.builder import build_streaming_av_module
from intersuit.model.streaming_av.audio_event_aligner import FrozenOffsetScorerInputs
from intersuit.model.streaming_av.confidence_gate import compute_audio_signal_features


def preserve_torch_rng(test_fn):
    @wraps(test_fn)
    def wrapped():
        with torch.random.fork_rng(devices=[]):
            return test_fn()

    return wrapped


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
    def __init__(
        self,
        fusion_init: str = "zero",
        gate_v1: bool = False,
        event_aligner_v1: bool = False,
        offset_scorer_bundle: str | None = None,
    ):
        self.config = DummyConfig()
        self.config.as_m4_fusion_init = fusion_init
        self.config.enable_audio_confidence_gate_v1 = gate_v1
        self.config.enable_audio_event_aligner_v1 = event_aligner_v1
        self.config.enable_audio_event_offset_scorer = offset_scorer_bundle is not None
        self.config.audio_event_offset_scorer_bundle_path = offset_scorer_bundle
        self.config.audio_event_offset_scorer_margin_threshold = 0.15
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


def test_enabled_frozen_offset_scorer_requires_bundle_path():
    config = DummyConfig()
    config.enable_audio_event_offset_scorer = True
    config.audio_event_offset_scorer_bundle_path = None

    with pytest.raises(ValueError, match="bundle_path is required"):
        build_streaming_av_module(config)


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


def test_audio_event_aligner_disabled_is_exact_no_op():
    model = DummyStreamingModel(fusion_init="identity", event_aligner_v1=False)
    video = torch.randn(2, 3, 4)
    kwargs = {
        "scene_audio_timestamps": torch.tensor([[[0.0, 1.0], [1.0, 2.0]]]),
        "frame_timestamps": torch.tensor([[0.5, 1.5]]),
    }

    baseline = model.fuse_scene_audio_into_image_features(
        [video],
        ["video"],
        _scene_output(),
        **kwargs,
    )[0]
    with_unused_windows = model.fuse_scene_audio_into_image_features(
        [video],
        ["video"],
        _scene_output(),
        scene_audio_windows=torch.ones(1, 2, 160) * 0.1,
        **kwargs,
    )[0]

    assert torch.equal(with_unused_windows, baseline)
    assert "audio_event_aligner_v1_enabled" not in model._last_streaming_av_diagnostics[0]


def _write_offset_bundle(path):
    radius = 1
    audio_dim, clip_dim, rgb_dim = 2, 3, 2
    context_count = 3
    scalar_dim = context_count * 6 + 7
    pair_dim = context_count * audio_dim * 2 + context_count * (clip_dim + rgb_dim) * 2 + scalar_dim
    hidden_dim = 4
    first_weight = torch.zeros(hidden_dim, pair_dim)
    scalar_start = context_count * audio_dim * 2 + context_count * (clip_dim + rgb_dim) * 2
    first_weight[0, scalar_start + scalar_dim - 7] = 1.0
    torch.save(
        {
            "format_version": 1,
            "state_dict": {
                "net.0.weight": first_weight,
                "net.0.bias": torch.zeros(hidden_dim),
                "net.2.weight": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                "net.2.bias": torch.zeros(1),
            },
            "scalar_mean": torch.zeros(scalar_dim),
            "scalar_std": torch.ones(scalar_dim),
            "metadata": {
                "audio_dim": audio_dim,
                "clip_dim": clip_dim,
                "rgb_dim": rgb_dim,
                "context_radius": radius,
                "scalar_dim": scalar_dim,
                "hidden_dim": hidden_dim,
                "candidate_offsets": [-0.5, 0.0, 0.5],
            },
        },
        path,
    )


def test_frozen_offset_diagnostic_is_exactly_non_invasive(tmp_path):
    bundle = tmp_path / "offset_bundle.pt"
    _write_offset_bundle(bundle)
    model = DummyStreamingModel(
        fusion_init="identity",
        event_aligner_v1=True,
        offset_scorer_bundle=str(bundle),
    )
    video = torch.randn(2, 3, 4)
    kwargs = {
        "scene_audio_timestamps": torch.tensor([[[0.0, 1.0], [1.0, 2.0]]]),
        "frame_timestamps": torch.tensor([[0.5, 1.5]]),
        "scene_audio_windows": torch.ones(1, 2, 160) * 0.1,
    }
    baseline = model.fuse_scene_audio_into_image_features(
        [video],
        ["video"],
        _scene_output(),
        **kwargs,
    )[0]
    scorer_inputs = FrozenOffsetScorerInputs(
        audio_features=torch.randn(1, 5, 2),
        clip_features=torch.randn(1, 5, 3),
        rgb_features=torch.randn(1, 5, 2),
        audio_rms=torch.linspace(0.1, 0.5, 5).unsqueeze(0),
        non_silent_ratio=torch.ones(1, 5),
    )
    with_diagnostics = model.fuse_scene_audio_into_image_features(
        [video],
        ["video"],
        _scene_output(),
        frozen_offset_scorer_inputs=scorer_inputs,
        **kwargs,
    )[0]

    assert torch.equal(with_diagnostics, baseline)
    diagnostics = model._last_streaming_av_diagnostics[0]
    assert diagnostics["offset_scorer_available"].all()
    assert diagnostics["offset_scorer_accepted"].any()


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


@preserve_torch_rng
def test_gate_v1_integration_emits_signal_question_and_offset_diagnostics():
    model = DummyStreamingModel(fusion_init="identity", gate_v1=True)
    video = torch.ones(2, 3, 4) * 0.5
    waveforms = torch.ones(1, 2, 160) * 0.1

    fused = model.fuse_scene_audio_into_image_features(
        [video],
        ["video"],
        _scene_output(),
        scene_audio_timestamps=torch.tensor([[[0.0, 1.0], [1.0, 2.0]]]),
        frame_timestamps=torch.tensor([[0.5, 1.5]]),
        question_features=torch.ones(1, 4) * 0.5,
        scene_audio_signal_features=compute_audio_signal_features(waveforms),
    )[0]

    diagnostics = model._last_streaming_av_diagnostics[0]
    assert fused.shape == video.shape
    assert diagnostics["gate_v1_enabled"] is True
    for key in (
        "audio_rms",
        "audio_loudness_dbfs",
        "silence_ratio",
        "audio_input_norm",
        "question_audio_similarity",
        "offset_confidence",
        "offset_score",
        "v1_quality_factor",
        "v1_relevance_factor",
    ):
        assert key in diagnostics
        assert torch.isfinite(diagnostics[key]).all()


@preserve_torch_rng
def test_gate_v1_integration_closes_silent_audio():
    model = DummyStreamingModel(fusion_init="identity", gate_v1=True)
    video = torch.randn(2, 3, 4)
    silent_output = SceneAudioEncoderOutput(
        features=torch.zeros(1, 2, 4),
        mask=torch.ones(1, 2, dtype=torch.bool),
    )

    fused = model.fuse_scene_audio_into_image_features(
        [video],
        ["video"],
        silent_output,
        scene_audio_timestamps=torch.tensor([[[0.0, 1.0], [1.0, 2.0]]]),
        frame_timestamps=torch.tensor([[0.5, 1.5]]),
        question_features=torch.ones(1, 4),
        scene_audio_signal_features=compute_audio_signal_features(torch.zeros(1, 2, 160)),
    )[0]

    diagnostics = model._last_streaming_av_diagnostics[0]
    assert diagnostics["gate_max"].item() == 0.0
    assert torch.equal(fused, video)


@preserve_torch_rng
def test_e7_gate_zero_stays_exact_with_alignment_diagnostics():
    model = DummyStreamingModel(fusion_init="identity", gate_v1=True, event_aligner_v1=True)
    video = torch.randn(2, 3, 4)

    fused = model.fuse_scene_audio_into_image_features(
        [video],
        ["video"],
        _scene_output(),
        scene_audio_timestamps=torch.tensor([[[0.0, 1.0], [1.0, 2.0]]]),
        frame_timestamps=torch.tensor([[0.5, 1.5]]),
        force_audio_gate=0.0,
        audio_delta_ratio_cap=0.03,
        question_features=torch.ones(1, 4),
        scene_audio_signal_features=compute_audio_signal_features(torch.ones(1, 2, 160) * 0.1),
        scene_audio_windows=torch.ones(1, 2, 160) * 0.1,
    )[0]

    diagnostics = model._last_streaming_av_diagnostics[0]
    assert torch.equal(fused, video)
    assert diagnostics["gate_mean"].item() == 0.0
    assert diagnostics["delta_norm"].item() == 0.0
    assert diagnostics["audio_event_aligner_v1_enabled"] is True
    for key in (
        "event_strength",
        "candidate_scores",
        "best_offset",
        "best_alignment_score",
        "alignment_margin",
        "alignment_confidence",
        "offset_scorer_candidate_scores",
        "offset_scorer_best_offset",
        "offset_scorer_margin",
        "offset_scorer_suggested_offset",
    ):
        assert torch.isfinite(diagnostics[key]).all()
    assert not diagnostics["offset_scorer_available"].any()
    assert not diagnostics["offset_scorer_accepted"].any()


if __name__ == "__main__":
    test_streaming_av_fusion_changes_video_features_when_gate_enabled()
    test_force_gate_zero_recovers_video_features()
    test_non_video_features_pass_through_unchanged()
    test_streaming_av_modules_receive_gradients_from_fused_video_loss()
    test_gate_v1_integration_emits_signal_question_and_offset_diagnostics()
    test_gate_v1_integration_closes_silent_audio()
    print("streaming_av_integration harness passed")
