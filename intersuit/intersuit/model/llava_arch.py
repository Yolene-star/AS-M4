#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

from .speech_encoder.builder import build_speech_encoder
from .speech_projector.builder import  build_speech_projector
from .scene_audio_encoder.builder import build_scene_audio_encoder
from .scene_audio_encoder.scene_audio_encoder import SceneAudioEncoderOutput
from .streaming_av.builder import build_streaming_av_module
from .streaming_av.audio_event_aligner import FrozenOffsetScorerInputs, compute_audio_event_features
from .streaming_av.confidence_gate import AudioSignalFeatures, compute_audio_signal_features
from .streaming_av.fusion import apply_audio_delta_ratio_cap

from intersuit.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from intersuit.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN
from intersuit.mm_utils import get_anyres_image_grid_shape
from intersuit.utils import rank0_print, lengths_to_padding_mask
import random


def _select_frozen_offset_scorer_inputs(inputs, index):
    if inputs is None:
        return None
    if isinstance(inputs, (list, tuple)) and not isinstance(inputs, FrozenOffsetScorerInputs):
        if index >= len(inputs):
            return None
        return inputs[index]
    if not isinstance(inputs, FrozenOffsetScorerInputs):
        raise ValueError("frozen_offset_scorer_inputs must be FrozenOffsetScorerInputs or a per-sample list")
    if inputs.audio_features.shape[0] == 1:
        return inputs if index == 0 else None
    return FrozenOffsetScorerInputs(*(value[index : index + 1] for value in inputs))


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))
        
        # speech encoder     
        if hasattr(config, "speech_encoder"):
            self.speech_encoder = build_speech_encoder(config)
            self.speech_projector = build_speech_projector(config)

        # scene audio encoder for AS-M4. This is intentionally separate from
        # the existing speech/query-speech path.
        if hasattr(config, "scene_audio_encoder_type") or hasattr(config, "scene_audio_encoder"):
            self.scene_audio_encoder = build_scene_audio_encoder(config)
            config.scene_audio_selector_dim = int(
                getattr(self.scene_audio_encoder, "encoder_dim", None)
                or getattr(config, "scene_audio_hidden_size", None)
                or config.hidden_size
            )
            self.streaming_av_module = build_streaming_av_module(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type
        
        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")

    # speech encoder
    def get_speech_encoder(self):
        speech_encoder = getattr(self, 'speech_encoder', None)
        if type(speech_encoder) is list:
            speech_encoder = speech_encoder[0]
        return speech_encoder

    def get_scene_audio_encoder(self):
        scene_audio_encoder = getattr(self, "scene_audio_encoder", None)
        if type(scene_audio_encoder) is list:
            scene_audio_encoder = scene_audio_encoder[0]
        return scene_audio_encoder

    def get_streaming_av_module(self):
        streaming_av_module = getattr(self, "streaming_av_module", None)
        if type(streaming_av_module) is list:
            streaming_av_module = streaming_av_module[0]
        return streaming_av_module
    
    def initialize_speech_modules(self, model_args, fsdp=None):
        self.config.speech_encoder = getattr(model_args, "speech_encoder", None)
        self.config.speech_encoder_type = getattr(model_args, "speech_encoder_type", None)
        self.config.speech_projector_type = getattr(model_args, 'speech_projector_type', 'linear')
        self.config.speech_encoder_ds_rate = getattr(model_args, 'speech_encoder_ds_rate', 5)
        self.config.speech_encoder_hidden_size = getattr(model_args, 'speech_encoder_hidden_size', 1280)

        if self.get_speech_encoder() is None:
            speech_encoder = build_speech_encoder(self.config)
            if fsdp is not None and len(fsdp) > 0:
                self.speech_encoder = [speech_encoder]
            else:
                self.speech_encoder = speech_encoder

        if getattr(self, 'speech_projector', None) is None:
            self.speech_projector = build_speech_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.speech_projector.parameters():
                p.requires_grad = True

        if model_args.pretrain_speech_projector is not None:
            pretrain_speech_projector_weights = torch.load(model_args.pretrain_speech_projector, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.speech_projector.load_state_dict(get_w(pretrain_speech_projector_weights, 'speech_projector'))

    def initialize_scene_audio_modules(self, model_args, fsdp=None):
        self.config.scene_audio_encoder_type = getattr(model_args, "scene_audio_encoder_type", "dummy")
        self.config.scene_audio_hidden_size = getattr(model_args, "scene_audio_hidden_size", self.config.hidden_size)
        self.config.scene_audio_precomputed_dim = getattr(model_args, "scene_audio_precomputed_dim", None)
        self.config.scene_audio_precomputed_shared_space = getattr(
            model_args, "scene_audio_precomputed_shared_space", False
        )
        self.config.scene_audio_torchaudio_bundle = getattr(model_args, "scene_audio_torchaudio_bundle", "WAV2VEC2_BASE")
        self.config.scene_audio_torchaudio_weight_path = (
            getattr(model_args, "scene_audio_torchaudio_weight_path", None) or None
        )
        self.config.scene_audio_sample_rate = getattr(model_args, "scene_audio_sample_rate", 16000)
        self.config.scene_audio_beats_checkpoint = getattr(
            model_args,
            "scene_audio_beats_checkpoint",
            "intersuit/checkpoints/BEATs_iter3_plus_AS2M.pt",
        )
        self.config.scene_audio_beats_code_root = getattr(
            model_args,
            "scene_audio_beats_code_root",
            "third_party/OmniMMI/baselines/videollama2/model",
        )
        self.config.scene_audio_beats_checkpoint_sha256 = (
            getattr(model_args, "scene_audio_beats_checkpoint_sha256", None) or None
        )
        self.config.scene_audio_window_mode = getattr(model_args, "scene_audio_window_mode", "fixed")
        self.config.scene_audio_selector_dim = getattr(model_args, "scene_audio_selector_dim", None)
        self.config.dynamic_window_scales_sec = getattr(model_args, "dynamic_window_scales_sec", "1.0,2.0,4.0")
        self.config.dynamic_window_top_k = getattr(model_args, "dynamic_window_top_k", 16)
        self.config.dynamic_window_nms_iou = getattr(model_args, "dynamic_window_nms_iou", 0.6)
        self.config.dynamic_window_ema_beta = getattr(model_args, "dynamic_window_ema_beta", 0.9)
        self.config.dynamic_window_start_scale = getattr(model_args, "dynamic_window_start_scale", 1.0)
        self.config.dynamic_window_hold_scale = getattr(model_args, "dynamic_window_hold_scale", 0.25)
        self.config.dynamic_window_min_score = getattr(model_args, "dynamic_window_min_score", 0.05)
        self.config.dynamic_window_causal = getattr(model_args, "dynamic_window_causal", True)
        self.config.num_audio_events = getattr(model_args, "num_audio_events", 25)
        self.config.audio_quality_dim = getattr(model_args, "audio_quality_dim", 1)
        self.config.streaming_av_align_dim = getattr(model_args, "streaming_av_align_dim", self.config.hidden_size)
        self.config.max_av_offset_sec = getattr(model_args, "max_av_offset_sec", 1.5)
        self.config.av_similarity_chunk_size = getattr(model_args, "av_similarity_chunk_size", None)
        self.config.force_audio_gate = getattr(model_args, "force_audio_gate", None)
        self.config.audio_delta_ratio_cap = getattr(model_args, "audio_delta_ratio_cap", 0.0)
        self.config.enable_scene_audio = getattr(model_args, "enable_scene_audio", True)
        self.config.as_m4_fusion_init = getattr(model_args, "as_m4_fusion_init", "zero")
        self.config.as_m4_gate_logit_bias = getattr(model_args, "as_m4_gate_logit_bias", -5.0)
        self.config.as_m4_fusion_mode = getattr(
            model_args,
            "as_m4_fusion_mode",
            "aligned_gated",
        )
        self.config.as_m4_simple_audio_gate = getattr(
            model_args,
            "as_m4_simple_audio_gate",
            1.0,
        )
        self.config.as_m4_inference_simple_audio_gate = getattr(
            model_args,
            "as_m4_inference_simple_audio_gate",
            None,
        )
        self.config.enable_audio_confidence_gate_v1 = getattr(
            model_args, "enable_audio_confidence_gate_v1", False
        )
        self.config.audio_gate_silence_threshold = getattr(
            model_args, "audio_gate_silence_threshold", 1e-4
        )
        self.config.audio_gate_rms_reference = getattr(model_args, "audio_gate_rms_reference", 0.05)
        self.config.enable_audio_event_aligner_v1 = getattr(
            model_args, "enable_audio_event_aligner_v1", False
        )
        self.config.audio_event_local_offset_sec = getattr(model_args, "audio_event_local_offset_sec", 0.5)
        self.config.audio_event_silence_threshold = getattr(
            model_args, "audio_event_silence_threshold", 1e-4
        )
        self.config.audio_event_rms_reference = getattr(model_args, "audio_event_rms_reference", 0.05)
        self.config.audio_event_strength_weight = getattr(model_args, "audio_event_strength_weight", 0.05)
        self.config.audio_event_align_dim = getattr(
            model_args, "audio_event_align_dim", self.config.streaming_av_align_dim
        )
        self.config.audio_event_semantic_feature_mode = getattr(
            model_args, "audio_event_semantic_feature_mode", "disabled"
        )
        self.config.audio_event_projector_checkpoint_path = (
            getattr(model_args, "audio_event_projector_checkpoint_path", None) or None
        )
        self.config.enable_audio_event_offset_scorer = getattr(
            model_args, "enable_audio_event_offset_scorer", False
        )
        self.config.audio_event_offset_scorer_bundle_path = (
            getattr(model_args, "audio_event_offset_scorer_bundle_path", None) or None
        )
        self.config.audio_event_offset_scorer_margin_threshold = getattr(
            model_args, "audio_event_offset_scorer_margin_threshold", 0.15
        )
        self.config.audio_event_offset_scorer_stabilization_strategy = getattr(
            model_args, "audio_event_offset_scorer_stabilization_strategy", "none"
        )
        self.config.audio_event_offset_scorer_consecutive_windows = getattr(
            model_args, "audio_event_offset_scorer_consecutive_windows", 2
        )
        self.config.audio_event_offset_scorer_hold_margin = getattr(
            model_args, "audio_event_offset_scorer_hold_margin", 0.10
        )
        self.config.audio_event_offset_scorer_switch_margin = getattr(
            model_args, "audio_event_offset_scorer_switch_margin", 0.30
        )
        self.config.audio_event_offset_scorer_moving_average_windows = getattr(
            model_args, "audio_event_offset_scorer_moving_average_windows", 3
        )
        self.config.enable_temporal_offset_gru_diagnostic = getattr(
            model_args, "enable_temporal_offset_gru_diagnostic", False
        )
        self.config.temporal_offset_gru_checkpoint_path = (
            getattr(model_args, "temporal_offset_gru_checkpoint_path", None) or None
        )

        if self.get_scene_audio_encoder() is None:
            scene_audio_encoder = build_scene_audio_encoder(self.config)
            self.config.scene_audio_selector_dim = int(
                getattr(scene_audio_encoder, "encoder_dim", self.config.scene_audio_selector_dim or self.config.hidden_size)
            )
            streaming_av_module = build_streaming_av_module(self.config)
            if fsdp is not None and len(fsdp) > 0:
                self.scene_audio_encoder = [scene_audio_encoder]
                self.streaming_av_module = [streaming_av_module]
            else:
                self.scene_audio_encoder = scene_audio_encoder
                self.streaming_av_module = streaming_av_module
    
    
def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


def _timestamps_to_centers(timestamps, batch_idx, steps, device, dtype):
    if timestamps is None:
        return torch.arange(steps, device=device, dtype=dtype).unsqueeze(0)
    if isinstance(timestamps, (list, tuple)):
        if batch_idx >= len(timestamps) or timestamps[batch_idx] is None:
            return torch.arange(steps, device=device, dtype=dtype).unsqueeze(0)
        ts = timestamps[batch_idx]
    else:
        ts = timestamps
    if not torch.is_tensor(ts):
        ts = torch.as_tensor(ts, device=device, dtype=dtype)
    else:
        ts = ts.to(device=device, dtype=dtype)
    if ts.ndim == 3:
        ts = ts[batch_idx]
    if ts.ndim == 2 and ts.shape[-1] == 2:
        centers = ts[:, :2].mean(dim=-1)
    elif ts.ndim == 1:
        centers = ts
    else:
        raise ValueError(f"Unexpected timestamp shape: {tuple(ts.shape)}")
    if centers.numel() < steps:
        pad = torch.arange(centers.numel(), steps, device=device, dtype=dtype)
        centers = torch.cat([centers, pad], dim=0)
    return centers[:steps].unsqueeze(0)


def _linearly_align_audio_windows(audio, audio_mask, num_frames):
    """Map fixed audio windows to video frames without learned alignment."""

    if audio.ndim != 3 or audio.shape[0] != 1:
        raise ValueError("Simple BEATs alignment expects audio shaped [1,T,H]")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    valid = audio[0][audio_mask[0].to(dtype=torch.bool)]
    if valid.shape[0] == 0:
        return torch.zeros(
            1,
            num_frames,
            audio.shape[-1],
            device=audio.device,
            dtype=audio.dtype,
        )
    if valid.shape[0] == 1:
        return valid.unsqueeze(0).expand(1, num_frames, -1)
    aligned = nn.functional.interpolate(
        valid.transpose(0, 1).unsqueeze(0).float(),
        size=num_frames,
        mode="linear",
        align_corners=False,
    )
    return aligned.squeeze(0).transpose(0, 1).unsqueeze(0).to(dtype=audio.dtype)


def _pool_question_text_features(embed_tokens, input_ids, attention_mask=None):
    """Mean-pool valid prompt token embeddings without embedding MM sentinels."""

    valid = input_ids.ge(0)
    if attention_mask is not None:
        valid = valid & attention_mask.to(device=input_ids.device, dtype=torch.bool)
    safe_ids = input_ids.masked_fill(~valid, 0)
    token_features = embed_tokens(safe_ids)
    weights = valid.to(device=token_features.device, dtype=token_features.dtype).unsqueeze(-1)
    pooled = (token_features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    return torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_2dPool(self, image_feature):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        #image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        # image_features = self.get_2dPool(image_features)
        image_features = self.get_model().vision_resampler(image_features, images=images)
        return image_features

    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []

        for idx, feat in enumerate(per_videos_or_images_features):
            feat = self.get_model().mm_projector(feat)
            # Post pooling
            if idx in video_idx_in_batch:
                feat = self.get_2dPool(feat)
            all_videos_or_images_features.append(feat)
        return all_videos_or_images_features
    
    # speech proposer
    def get_speech_encoder(self):
        return self.get_model().get_speech_encoder()
    
    def get_speech_projector(self):
        return self.get_model().speech_projector

    def get_scene_audio_encoder(self):
        return self.get_model().get_scene_audio_encoder()

    def get_streaming_av_module(self):
        return self.get_model().get_streaming_av_module()

    def encode_scene_audio(self, scene_audios, scene_audio_mask=None, scene_audio_timestamps=None):
        scene_audio_encoder = self.get_scene_audio_encoder()
        if scene_audio_encoder is None:
            raise ValueError("Scene audio encoder is not initialized. Call initialize_scene_audio_modules first.")
        window_mode = str(getattr(self.config, "scene_audio_window_mode", "fixed")).lower()
        if window_mode not in {"fixed", "dynamic"}:
            raise ValueError(f"encode_scene_audio does not support window mode: {window_mode}")
        if window_mode == "dynamic":
            streaming_av_module = self.get_streaming_av_module()
            if streaming_av_module is None:
                raise ValueError("Dynamic scene audio requires streaming_av_module")
            if scene_audio_timestamps is None:
                raise ValueError("Dynamic scene audio requires scene_audio_timestamps")
            if hasattr(scene_audio_encoder, "encode_raw") and hasattr(scene_audio_encoder, "project"):
                raw_output = scene_audio_encoder.encode_raw(scene_audios, sample_mask=scene_audio_mask)
                selected = streaming_av_module.dynamic_window_selector(
                    raw_output.features,
                    scene_audio_timestamps,
                    audio_mask=raw_output.mask,
                    audio_windows=scene_audios,
                )
                projected = scene_audio_encoder.project(selected.selected_features)
            else:
                fixed_output = scene_audio_encoder(
                    scene_audios,
                    sample_mask=scene_audio_mask,
                    timestamps=scene_audio_timestamps,
                )
                selected = streaming_av_module.dynamic_window_selector(
                    fixed_output.features,
                    scene_audio_timestamps,
                    audio_mask=fixed_output.mask,
                    audio_windows=scene_audios,
                )
                projected = selected.selected_features
            self._last_dynamic_window_selector_output = selected
            return SceneAudioEncoderOutput(
                features=projected.masked_fill(~selected.selection_mask.unsqueeze(-1), 0.0),
                mask=selected.selection_mask,
                timestamps=selected.selected_timestamps,
                feature_kind="dynamic_window_selected",
            )
        output = scene_audio_encoder(
            scene_audios,
            sample_mask=scene_audio_mask,
            timestamps=scene_audio_timestamps,
        )
        return output

    def fuse_scene_audio_into_image_features(
        self,
        image_features,
        modalities,
        scene_audio_output,
        scene_audio_timestamps=None,
        frame_timestamps=None,
        lookahead_sec=0.0,
        force_audio_gate=None,
        audio_residual_scale=1.0,
        audio_delta_ratio_cap=0.0,
        question_features=None,
        scene_audio_signal_features=None,
        scene_audio_windows=None,
        frozen_offset_scorer_inputs=None,
    ):
        """Fuse scene audio into per-frame video features before flattening.

        ``image_features`` is the list returned by ``encode_multimodals`` where
        video entries are shaped ``[frames, tokens, hidden]``. Non-video entries
        pass through unchanged.
        """

        streaming_av_module = self.get_streaming_av_module()
        if streaming_av_module is None or scene_audio_output is None:
            return image_features

        fused_features = []
        diagnostics = []
        for idx, feature in enumerate(image_features):
            if idx >= len(modalities) or modalities[idx] not in {"video", "video_feature"} or feature.ndim != 3:
                fused_features.append(feature)
                diagnostics.append(None)
                continue
            if idx >= scene_audio_output.features.shape[0]:
                fused_features.append(feature)
                diagnostics.append(None)
                continue

            audio = scene_audio_output.features[idx : idx + 1].to(device=feature.device, dtype=feature.dtype)
            audio_mask = scene_audio_output.mask[idx : idx + 1].to(device=feature.device)
            video = feature.unsqueeze(0)
            num_audio = audio.shape[1]
            num_frames = video.shape[1]

            fusion_mode = str(
                getattr(self.config, "as_m4_fusion_mode", "aligned_gated")
            )
            if fusion_mode == "beats_simple_residual":
                frame_audio = _linearly_align_audio_windows(
                    audio,
                    audio_mask,
                    num_frames,
                )
                default_gate = float(getattr(self.config, "as_m4_simple_audio_gate", 1.0))
                inference_gate = getattr(
                    self.config,
                    "as_m4_inference_simple_audio_gate",
                    None,
                )
                if not getattr(self, "training", False) and inference_gate is not None:
                    default_gate = float(inference_gate)
                gate_value = float(force_audio_gate) if force_audio_gate is not None else default_gate
                if not math.isfinite(gate_value) or not 0.0 <= gate_value <= 1.0:
                    raise ValueError("as_m4_simple_audio_gate must be finite and in [0,1]")
                gate = torch.full(
                    (1, num_frames),
                    gate_value,
                    device=feature.device,
                    dtype=feature.dtype,
                )
                signal_gate = None
                if scene_audio_signal_features is not None:
                    audio_rms = scene_audio_signal_features.rms[idx : idx + 1].to(
                        device=feature.device,
                        dtype=feature.dtype,
                    )
                    non_silent = audio_rms.gt(
                        float(getattr(self.config, "audio_gate_silence_threshold", 1e-4))
                    )
                    signal_gate = _linearly_align_audio_windows(
                        non_silent.to(dtype=feature.dtype).unsqueeze(-1),
                        audio_mask,
                        num_frames,
                    ).squeeze(-1).clamp(0.0, 1.0)
                    gate = gate * signal_gate
                audio_delta = streaming_av_module.fusion.audio_delta(frame_audio)
                raw_delta = (
                    float(audio_residual_scale)
                    * gate.view(1, num_frames, 1, 1)
                    * audio_delta.unsqueeze(2)
                )
                capped_delta, cap_diagnostics = apply_audio_delta_ratio_cap(
                    video,
                    raw_delta,
                    ratio_cap=audio_delta_ratio_cap,
                )
                fused_features.append((video + capped_delta).squeeze(0))
                diagnostics.append(
                    {
                        "fusion_mode": fusion_mode,
                        "dynamic_alignment_enabled": False,
                        "learned_gate_enabled": False,
                        "gate": gate.detach(),
                        "gate_mean": gate.detach().float().mean(),
                        "gate_max": gate.detach().float().max(),
                        "gate_min": gate.detach().float().min(),
                        "signal_gate_mean": (
                            signal_gate.detach().float().mean()
                            if signal_gate is not None
                            else None
                        ),
                        "video_norm": cap_diagnostics["video_norm"][0],
                        "audio_norm": frame_audio.detach().float().norm(),
                        "delta_norm": cap_diagnostics["capped_delta_norm"][0],
                        "delta_to_video_ratio": cap_diagnostics[
                            "capped_delta_to_video_ratio"
                        ][0],
                        "raw_delta_norm": cap_diagnostics["raw_delta_norm"][0],
                        "raw_delta_to_video_ratio": cap_diagnostics[
                            "raw_delta_to_video_ratio"
                        ][0],
                        "audio_delta_cap": float(audio_delta_ratio_cap),
                        "audio_delta_applied_scale": cap_diagnostics[
                            "audio_delta_applied_scale"
                        ][0],
                        "capped_delta_norm": cap_diagnostics[
                            "capped_delta_norm"
                        ][0],
                        "capped_delta_to_video_ratio": cap_diagnostics[
                            "capped_delta_to_video_ratio"
                        ][0],
                        "audio_residual_scale": float(audio_residual_scale),
                    }
                )
                continue
            if fusion_mode != "aligned_gated":
                raise ValueError(f"Unknown AS-M4 fusion mode: {fusion_mode}")

            audio_times = _timestamps_to_centers(
                scene_audio_timestamps,
                idx,
                num_audio,
                feature.device,
                feature.dtype,
            )
            video_times = _timestamps_to_centers(
                frame_timestamps,
                idx,
                num_frames,
                feature.device,
                feature.dtype,
            )

            event_output = streaming_av_module.event_detector(audio, mask=audio_mask)
            align_output = streaming_av_module.temporal_aligner(
                audio,
                video,
                audio_times,
                video_times,
                audio_mask=audio_mask,
                lookahead_sec=lookahead_sec,
            )

            weights_t = align_output.alignment_weights.transpose(1, 2).to(dtype=audio.dtype)
            weights_t = torch.nan_to_num(weights_t, nan=0.0, posinf=0.0, neginf=0.0)
            denom = weights_t.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            frame_audio = torch.matmul(weights_t, audio) / denom
            frame_audio = torch.nan_to_num(frame_audio, nan=0.0, posinf=0.0, neginf=0.0)
            video_summary = video.mean(dim=2)
            local_alignment = None
            event_features = None
            if bool(getattr(self.config, "enable_audio_event_aligner_v1", False)) and scene_audio_windows is not None:
                event_features = compute_audio_event_features(
                    scene_audio_windows[idx : idx + 1].to(device=feature.device),
                    audio_features=audio,
                    sample_mask=audio_mask,
                    silence_threshold=float(getattr(self.config, "audio_event_silence_threshold", 1e-4)),
                    rms_reference=float(getattr(self.config, "audio_event_rms_reference", 0.05)),
                )
                local_alignment = streaming_av_module.audio_event_aligner(
                    audio,
                    video,
                    audio_times,
                    video_times,
                    event_features,
                    audio_mask=audio_mask,
                    frozen_offset_inputs=_select_frozen_offset_scorer_inputs(
                        frozen_offset_scorer_inputs,
                        idx,
                    ),
                )
            frame_confidence = align_output.offset_confidence.to(dtype=audio.dtype).unsqueeze(1).expand(-1, num_frames, -1)
            frame_confidence = (weights_t * frame_confidence).sum(dim=-1) / denom.squeeze(-1)
            frame_confidence = torch.nan_to_num(frame_confidence, nan=0.0, posinf=1.0, neginf=0.0)
            frame_eventness = event_output.eventness.to(dtype=audio.dtype).unsqueeze(1).expand(-1, num_frames, -1)
            frame_eventness = (weights_t * frame_eventness).sum(dim=-1, keepdim=True) / denom
            frame_eventness = torch.nan_to_num(frame_eventness, nan=0.0, posinf=1.0, neginf=0.0)

            frame_question = None
            frame_signal = None
            frame_offset = None
            if streaming_av_module.confidence_gate.enable_v1:
                if question_features is not None and idx < question_features.shape[0]:
                    frame_question = question_features[idx : idx + 1]
                if scene_audio_signal_features is not None:
                    def align_scalar(values):
                        sample_values = values[idx : idx + 1].to(device=feature.device, dtype=audio.dtype)
                        return (weights_t * sample_values.unsqueeze(1)).sum(dim=-1) / denom.squeeze(-1)

                    frame_signal = AudioSignalFeatures(
                        rms=align_scalar(scene_audio_signal_features.rms),
                        loudness_dbfs=align_scalar(scene_audio_signal_features.loudness_dbfs),
                        silence_ratio=align_scalar(scene_audio_signal_features.silence_ratio),
                        norm=align_scalar(scene_audio_signal_features.norm),
                    )
                frame_offset = (weights_t * align_output.offset_sec.to(dtype=audio.dtype).unsqueeze(1)).sum(
                    dim=-1
                ) / denom.squeeze(-1)

            gate_output = streaming_av_module.confidence_gate(
                frame_audio,
                video_summary,
                question_feature=frame_question,
                quality_features=frame_eventness,
                alignment_confidence=frame_confidence,
                signal_features=frame_signal,
                offset_sec=frame_offset,
            )
            gate = gate_output.gate
            if force_audio_gate is not None:
                gate = torch.zeros_like(gate) + float(force_audio_gate)
            gate = torch.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0)

            audio_delta = streaming_av_module.fusion.audio_delta(frame_audio)
            gated_delta = float(audio_residual_scale) * gate.to(dtype=audio_delta.dtype).view(audio_delta.shape[0], audio_delta.shape[1], 1, 1) * audio_delta.unsqueeze(2)
            capped_delta, cap_diagnostics = apply_audio_delta_ratio_cap(
                video,
                gated_delta,
                ratio_cap=audio_delta_ratio_cap,
            )
            video_norm = cap_diagnostics["video_norm"][0]
            audio_norm = frame_audio.detach().float().norm()
            raw_delta_norm = cap_diagnostics["raw_delta_norm"][0]
            raw_delta_to_video_ratio = cap_diagnostics["raw_delta_to_video_ratio"][0]
            audio_delta_applied_scale = cap_diagnostics["audio_delta_applied_scale"][0]
            capped_delta_norm = cap_diagnostics["capped_delta_norm"][0]
            capped_delta_to_video_ratio = cap_diagnostics["capped_delta_to_video_ratio"][0]
            fused = video + capped_delta
            fused_features.append(fused.squeeze(0))
            diagnostics.append(
                {
                    "eventness": event_output.eventness.detach(),
                    "offset_sec": align_output.offset_sec.detach(),
                    "gate": gate.detach(),
                    "quality_gate": gate_output.quality.detach(),
                    "relevance_gate": gate_output.relevance.detach(),
                    "gate_mean": gate.detach().float().mean(),
                    "gate_max": gate.detach().float().max(),
                    "gate_min": gate.detach().float().min(),
                    "video_norm": video_norm,
                    "audio_norm": audio_norm,
                    # Backward-compatible fields explicitly represent the
                    # residual after the optional ratio cap.
                    "delta_norm": capped_delta_norm,
                    "delta_to_video_ratio": capped_delta_to_video_ratio,
                    "raw_delta_norm": raw_delta_norm,
                    "raw_delta_to_video_ratio": raw_delta_to_video_ratio,
                    "audio_delta_cap": float(audio_delta_ratio_cap),
                    "audio_delta_applied_scale": audio_delta_applied_scale,
                    "capped_delta_norm": capped_delta_norm,
                    "capped_delta_to_video_ratio": capped_delta_to_video_ratio,
                    "audio_residual_scale": float(audio_residual_scale),
                }
            )
            if local_alignment is not None and event_features is not None:
                diagnostics[-1].update(
                    {
                        "audio_event_aligner_v1_enabled": True,
                        "event_strength": event_features.event_strength.detach(),
                        "is_silent_window": event_features.is_silent_window.detach(),
                        "audio_rms": event_features.audio_rms.detach(),
                        "audio_peak": event_features.audio_peak.detach(),
                        "candidate_offsets": local_alignment.candidate_offsets.detach(),
                        "candidate_valid": local_alignment.candidate_valid.detach(),
                        "semantic_similarity": local_alignment.semantic_similarity.detach(),
                        "video_event_strength": local_alignment.video_event_strength.detach(),
                        "candidate_scores": local_alignment.candidate_scores.detach(),
                        "best_offset": local_alignment.best_offset.detach(),
                        "best_alignment_score": local_alignment.best_alignment_score.detach(),
                        "second_best_alignment_score": local_alignment.second_best_alignment_score.detach(),
                        "alignment_margin": local_alignment.alignment_margin.detach(),
                        "alignment_confidence": local_alignment.alignment_confidence.detach(),
                        "offset_scorer_candidate_scores": local_alignment.offset_scorer_candidate_scores.detach(),
                        "offset_scorer_best_offset": local_alignment.offset_scorer_best_offset.detach(),
                        "offset_scorer_margin": local_alignment.offset_scorer_margin.detach(),
                        "offset_scorer_accepted": local_alignment.offset_scorer_accepted.detach(),
                        "offset_scorer_suggested_offset": local_alignment.offset_scorer_suggested_offset.detach(),
                        "offset_scorer_available": local_alignment.offset_scorer_available.detach(),
                        "offset_scorer_stable_candidate_scores": local_alignment.offset_scorer_stable_candidate_scores.detach(),
                        "offset_scorer_stable_best_offset": local_alignment.offset_scorer_stable_best_offset.detach(),
                        "offset_scorer_stable_margin": local_alignment.offset_scorer_stable_margin.detach(),
                        "offset_scorer_stable_accepted": local_alignment.offset_scorer_stable_accepted.detach(),
                        "offset_scorer_stable_suggested_offset": local_alignment.offset_scorer_stable_suggested_offset.detach(),
                        "offset_scorer_stable_delay_windows": local_alignment.offset_scorer_stable_delay_windows.detach(),
                        "temporal_offset_gru_enabled": local_alignment.temporal_offset_available.any().detach(),
                        "temporal_offset_logits": local_alignment.temporal_offset_logits.detach(),
                        "temporal_offset_sync_prob": local_alignment.temporal_offset_sync_prob.detach(),
                        "temporal_offset_predicted_offset": local_alignment.temporal_offset_predicted_offset.detach(),
                        "temporal_offset_accepted": local_alignment.temporal_offset_accepted.detach(),
                        "temporal_offset_suggested_offset": local_alignment.temporal_offset_suggested_offset.detach(),
                        "temporal_offset_available": local_alignment.temporal_offset_available.detach(),
                    }
                )
            if streaming_av_module.confidence_gate.enable_v1:
                gate_v1_diagnostics = streaming_av_module.confidence_gate.last_v1_diagnostics
                diagnostics[-1].update(
                    {
                        "gate_v1_enabled": True,
                        "audio_rms": gate_v1_diagnostics["audio_rms"].detach(),
                        "audio_loudness_dbfs": gate_v1_diagnostics["audio_loudness_dbfs"].detach(),
                        "silence_ratio": gate_v1_diagnostics["silence_ratio"].detach(),
                        "audio_input_norm": gate_v1_diagnostics["audio_norm"].detach(),
                        "question_audio_similarity": gate_v1_diagnostics["question_similarity"].detach(),
                        "offset_confidence": frame_confidence.detach(),
                        "offset_score": gate_v1_diagnostics["offset_score"].detach(),
                        "v1_quality_factor": gate_v1_diagnostics["v1_quality_factor"].detach(),
                        "v1_relevance_factor": gate_v1_diagnostics["v1_relevance_factor"].detach(),
                    }
                )
        self._last_streaming_av_diagnostics = diagnostics
        return fused_features

    def encode_speech(self, speech, speech_lengths):
        speech_encoder_type = self.config.speech_encoder_type
        speech_encoder = self.get_speech_encoder()
        
        if "whisper" in speech_encoder_type.lower():
        #     if ".pt" in self.config.speech_encoder:
        #         # whisper
        #         encoder_outs = speech_encoder(speech.permute(0, 2, 1))
        #     else:
        #         # transformers @ huggingface
        #         encoder_outs = speech_encoder(speech.permute(0, 2, 1)).last_hidden_state
            encoder_outs = speech_encoder(speech.permute(0, 2, 1))
            # encoder_outs = speech_encoder(speech.permute(0, 2, 1)).last_hidden_state
            speech_lengths = (speech_lengths + 1) // 2
        else:
            raise ValueError(f'Unknown speech encoder: {speech_encoder}')
        speech_projector_type = self.config.speech_projector_type
        speech_projector = self.get_speech_projector()
        if speech_projector_type == "linear":
            encoder_outs = speech_projector(encoder_outs)
            speech_lengths = speech_lengths // speech_projector.k
        else:
            raise ValueError(f'Unknown speech projector: {speech_projector_type}')
        speech_features = [encoder_outs[i, :speech_lengths[i]] for i in range(len(encoder_outs))]
        return speech_features

    def _pack_image_features_into_inputs(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        image_features,
        modalities,
    ):
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        new_labels = None if _labels is None else new_labels_padded
        attention_mask = None if _attention_mask is None else attention_mask.to(dtype=_attention_mask.dtype)
        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    
    
    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        # print(images)
        
        has_video_feature = any(modality == "video_feature" for modality in modalities)
        if type(images) is list or images.ndim == 5 or (has_video_feature and images.ndim == 4):
            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] in {"video", "video_feature"}:
                    video_idx_in_batch.append(_)

            if has_video_feature:
                if torch.is_tensor(images):
                    image_features = [images[idx] for idx in range(images.shape[0])]
                else:
                    image_features = []
                    for image in images:
                        if image.ndim == 4 and image.shape[0] == 1:
                            image_features.append(image[0])
                        elif image.ndim == 3:
                            image_features.append(image)
                        else:
                            image_features.append(image.squeeze(0) if image.ndim == 4 else image)
            else:
                if type(images) is list:
                    images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

                images_list = []
                for image in images:
                    if image.ndim == 4:
                        images_list.append(image)
                    else:
                        images_list.append(image.unsqueeze(0))

                concat_images = torch.cat([image for image in images_list], dim=0)
                split_sizes = [image.shape[0] for image in images_list]

                image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]
            
            elif mm_patch_merge_type== "unires":
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # rank0_print(f"Initial feature size : {image_feature.shape}")
                    if image_idx in video_idx_in_batch:  # video operations
                        image_feature = image_feature.flatten(0, 1)
                    elif image_feature.shape[0] > 1:
                        # base image feature is never used in unires
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        # rank0_print(f"Before pool : {image_feature.shape}")
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if hasattr(self.get_vision_tower(), "image_size"):
                            vision_tower_image_size = self.get_vision_tower().image_size
                        else:
                            raise ValueError("vision_tower_image_size is not found in the vision tower.")
                        num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        # Assume 2*2 patches
                        # After this, [2,2, 24,24, 4096]
                        kernel_size = mm_patch_merge_type.split("avgpool")[-1].split("x")[-1]
                        kernel_size = 2
                        image_feature = image_feature.view(num_patch_height * num_patch_width, height, width, -1) # [4, 24, 24, 4096]
                        image_feature = image_feature.permute(0, 3, 1, 2).contiguous() # [4, 4096, 24, 24]
                        image_feature = nn.functional.avg_pool2d(image_feature, kernel_size) # [4, 4096, 12, 12]
                        image_feature = image_feature.flatten(2, 3) # [4, 4096, 144]
                        image_feature = image_feature.permute(0, 2, 1).contiguous() # [4, 144, 4096]
                        image_feature = image_feature.flatten(0, 1) # [576, 4096]
                        # rank0_print(f"After pool : {image_feature.shape}")
                    else:
                        # for text only data, there is a placeholder image feature that is actually never used. 
                        image_feature = image_feature[0]
                        # rank0_print(f"After here : {image_feature.shape}")
                    new_image_features.append(image_feature) # npt * nfr, dim
                #     print("*"*20)
                #     print(image_feature.shape)
                #     print("*"*20)
                # raise ValueError("debug")

                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            
            
            print("this is pretraining")
            
            # # pretraining
            # image_features = self.encode_images(images) # nfr, npt, dim
            # image_features = [image_features.flatten(0,1)]
            
            error_message = """
            Something is wrong with the input shape. Most likely, you did not wrap the video input in a list:
            This is correct:
                model.generate(input_ids, images=[video_tensor],  modalities=["video"], **gen_kwargs)
            This is wrong:
                model.generate(input_ids, images=video_tensor,  modalities=["video"], **gen_kwargs)
            """
            raise ValueError(error_message)
            # image_features = self.encode_images(images)
                
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        
        
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                # print("*"*20)
                # print(cur_new_labels[-1].shape)
                # print("*"*20)
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        
        

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def prepare_inputs_labels_for_streaming_av(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        modalities=["image"],
        image_sizes=None,
        speeches=None,
        speech_lengths=None,
        scene_audios=None,
        scene_audio_mask=None,
        scene_audio_timestamps=None,
        frame_timestamps=None,
        **kwargs,
    ):
        enable_scene_audio = getattr(self.config, "enable_scene_audio", False)
        window_mode = str(getattr(self.config, "scene_audio_window_mode", "fixed")).lower()
        if not enable_scene_audio or window_mode == "disabled" or scene_audios is None:
            if speeches is not None:
                return self.prepare_inputs_labels_for_multimodal_av(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    modalities,
                    image_sizes,
                    speeches,
                    speech_lengths,
                )
            return self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                modalities,
                image_sizes,
            )

        if speeches is not None:
            raise NotImplementedError("streaming scene_audio + query speech packing is handled in a later AS-M4 phase")

        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        has_video_feature = any(modality == "video_feature" for modality in modalities)
        valid_images_shape = (
            type(images) is list
            or images.ndim == 5
            or (has_video_feature and images.ndim == 4)
        )
        if not valid_images_shape:
            raise ValueError("Streaming AV path expects images as a list, 5D video/image batch, or 4D video_feature batch")

        if type(images) is list:
            images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

        video_idx_in_batch = []
        for idx, modality in enumerate(modalities):
            if modality in {"video", "video_feature"}:
                video_idx_in_batch.append(idx)

        if any(modality == "video_feature" for modality in modalities):
            if torch.is_tensor(images):
                image_features = [images[idx] for idx in range(images.shape[0])]
            else:
                image_features = [image for image in images]
        else:
            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)

        enable_gate_v1 = bool(getattr(self.config, "enable_audio_confidence_gate_v1", False))
        simple_residual = str(
            getattr(self.config, "as_m4_fusion_mode", "aligned_gated")
        ) == "beats_simple_residual"
        question_features = None
        scene_audio_signal_features = None
        if enable_gate_v1:
            question_features = _pool_question_text_features(
                self.get_model().embed_tokens,
                input_ids,
                attention_mask,
            )
        if enable_gate_v1 or simple_residual:
            scene_audio_signal_features = compute_audio_signal_features(
                scene_audios,
                sample_mask=scene_audio_mask,
                silence_threshold=float(getattr(self.config, "audio_gate_silence_threshold", 1e-4)),
            )

        scene_audio_output = self.encode_scene_audio(
            scene_audios,
            scene_audio_mask=scene_audio_mask,
            scene_audio_timestamps=scene_audio_timestamps,
        )
        self._last_scene_audio_output = scene_audio_output
        if window_mode == "dynamic" and scene_audio_signal_features is not None:
            selector_output = self._last_dynamic_window_selector_output
            weights = selector_output.source_weights.to(device=scene_audio_signal_features.rms.device)
            scene_audio_signal_features = AudioSignalFeatures(
                rms=torch.einsum("bkt,bt->bk", weights, scene_audio_signal_features.rms),
                loudness_dbfs=torch.einsum("bkt,bt->bk", weights, scene_audio_signal_features.loudness_dbfs),
                silence_ratio=torch.einsum("bkt,bt->bk", weights, scene_audio_signal_features.silence_ratio),
                norm=torch.einsum("bkt,bt->bk", weights, scene_audio_signal_features.norm),
            )
        image_features = self.fuse_scene_audio_into_image_features(
            image_features,
            modalities,
            scene_audio_output,
            scene_audio_timestamps=scene_audio_output.timestamps,
            frame_timestamps=frame_timestamps,
            lookahead_sec=getattr(self.config, "streaming_av_lookahead_sec", 0.0),
            force_audio_gate=getattr(self.config, "force_audio_gate", None),
            audio_residual_scale=getattr(self.config, "debug_audio_residual_scale", 1.0),
            audio_delta_ratio_cap=getattr(self.config, "audio_delta_ratio_cap", 0.0),
            question_features=question_features,
            scene_audio_signal_features=scene_audio_signal_features,
            scene_audio_windows=(
                scene_audios
                if window_mode != "dynamic" and bool(getattr(self.config, "enable_audio_event_aligner_v1", False))
                else None
            ),
            frozen_offset_scorer_inputs=getattr(
                self,
                "_audio_event_offset_diagnostic_inputs",
                None,
            ) if window_mode != "dynamic" else None,
        )
        if window_mode == "dynamic":
            selector_output = self._last_dynamic_window_selector_output
            selector_diagnostics = self._last_streaming_av_diagnostics
            for idx, diagnostic in enumerate(selector_diagnostics):
                if diagnostic is None or idx >= selector_output.selection_mask.shape[0]:
                    continue
                selected_mask = selector_output.selection_mask[idx]
                diagnostic.update(
                    {
                        "dynamic_window_enabled": True,
                        "dynamic_window_selected_ratio": selected_mask.float().mean(),
                        "dynamic_window_selected_count": selected_mask.sum(),
                        "dynamic_window_selection_scores": selector_output.selection_scores[idx].detach(),
                        "dynamic_window_selection_mask": selected_mask.detach(),
                        "dynamic_window_timestamps": selector_output.selected_timestamps[idx].detach(),
                        "dynamic_window_micro_scores": selector_output.micro_scores[idx].detach(),
                        "dynamic_window_thresholds": selector_output.dynamic_thresholds[idx].detach(),
                    }
                )

        mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
        if mm_patch_merge_type == "flat":
            image_features = [x.flatten(0, 1) for x in image_features]
        elif mm_patch_merge_type == "unires":
            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                if image_idx in video_idx_in_batch:
                    image_feature = image_feature.flatten(0, 1)
                elif image_feature.shape[0] > 1:
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]
                    height = width = self.get_vision_tower().num_patches_per_side
                    assert height * width == base_image_feature.shape[0]
                    if hasattr(self.get_vision_tower(), "image_size"):
                        vision_tower_image_size = self.get_vision_tower().image_size
                    else:
                        raise ValueError("vision_tower_image_size is not found in the vision tower.")
                    num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                    image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                    kernel_size = 2
                    image_feature = image_feature.view(num_patch_height * num_patch_width, height, width, -1)
                    image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
                    image_feature = nn.functional.avg_pool2d(image_feature, kernel_size)
                    image_feature = image_feature.flatten(2, 3)
                    image_feature = image_feature.permute(0, 2, 1).contiguous()
                    image_feature = image_feature.flatten(0, 1)
                else:
                    image_feature = image_feature[0]
                new_image_features.append(image_feature)
            image_features = new_image_features
        else:
            raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")

        return self._pack_image_features_into_inputs(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            image_features,
            modalities,
        )

    def prepare_inputs_labels_for_multimodal_av(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None, speeches=None, speech_lengths=None):
        
        
        # preprocess <image>
        vision_tower = self.get_vision_tower()
        if images is None and speeches is None:
            if vision_tower is None or images is None or input_ids.shape[1] == 1:
                return input_ids, position_ids, attention_mask, past_key_values, None, labels
        if images is not None:
            if type(images) is list or images.ndim == 5:
                if type(images) is list:
                    images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

                video_idx_in_batch = []
                for _ in range(len(modalities)):
                    if modalities[_] == "video":
                        video_idx_in_batch.append(_)

                images_list = []
                for image in images:
                    if image.ndim == 4:
                        images_list.append(image)
                    else:
                        images_list.append(image.unsqueeze(0))

                concat_images = torch.cat([image for image in images_list], dim=0)
                split_sizes = [image.shape[0] for image in images_list]

                image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
                # image_features = torch.split(image_features, split_sizes, dim=0)
                mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
                image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")

                if mm_patch_merge_type == "flat":
                    image_features = [x.flatten(0, 1) for x in image_features]
                
                elif mm_patch_merge_type== "unires":
                    new_image_features = []
                    for image_idx, image_feature in enumerate(image_features):
                        # rank0_print(f"Initial feature size : {image_feature.shape}")
                        if image_idx in video_idx_in_batch:  # video operations
                            image_feature = image_feature.flatten(0, 1)
                        elif image_feature.shape[0] > 1:
                            # base image feature is never used in unires
                            base_image_feature = image_feature[0]
                            image_feature = image_feature[1:]
                            # rank0_print(f"Before pool : {image_feature.shape}")
                            height = width = self.get_vision_tower().num_patches_per_side
                            assert height * width == base_image_feature.shape[0]
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            # Assume 2*2 patches
                            # After this, [2,2, 24,24, 4096]
                            kernel_size = mm_patch_merge_type.split("avgpool")[-1].split("x")[-1]
                            kernel_size = 2
                            image_feature = image_feature.view(num_patch_height * num_patch_width, height, width, -1) # [4, 24, 24, 4096]
                            image_feature = image_feature.permute(0, 3, 1, 2).contiguous() # [4, 4096, 24, 24]
                            image_feature = nn.functional.avg_pool2d(image_feature, kernel_size) # [4, 4096, 12, 12]
                            image_feature = image_feature.flatten(2, 3) # [4, 4096, 144]
                            image_feature = image_feature.permute(0, 2, 1).contiguous() # [4, 144, 4096]
                            image_feature = image_feature.flatten(0, 1) # [576, 4096]
                            # rank0_print(f"After pool : {image_feature.shape}")
                        else:
                            # for text only data, there is a placeholder image feature that is actually never used. 
                            image_feature = image_feature[0]
                            # rank0_print(f"After here : {image_feature.shape}")
                        new_image_features.append(image_feature) # npt * nfr, dim
                    #     print("*"*20)
                    #     print(image_feature.shape)
                    #     print("*"*20)
                    # raise ValueError("debug")

                    image_features = new_image_features
                else:
                    raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
            else:
                
                
                print("this is pretraining")
                
                # # pretraining
                # image_features = self.encode_images(images) # nfr, npt, dim
                # image_features = [image_features.flatten(0,1)]
                
                error_message = """
                Something is wrong with the input shape. Most likely, you did not wrap the video input in a list:
                This is correct:
                    model.generate(input_ids, images=[video_tensor],  modalities=["video"], **gen_kwargs)
                This is wrong:
                    model.generate(input_ids, images=video_tensor,  modalities=["video"], **gen_kwargs)
                """
                raise ValueError(error_message)
                # image_features = self.encode_images(images)
                    
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        
        # preprocess <speech>
        speech_features = self.encode_speech(speeches, speech_lengths)
        speech_features = [speech_feature.to(dtype=image_features[0].dtype) for speech_feature in speech_features]
        # print("debug*"*30)
        # print(image_features[0].shape)
        # print(len(image_features))
        # # print(speeches.shape)
        # # print(speech_features[0].shape)
        # # print(speech_lengths)
        # print(speech_features[0].shape)
        # print(len(speech_features))
        # print("debug*"*30)
        
        # print("inputs*"*20)
        # print(input_ids)
        # print(labels)
        # print("inputs*"*20)
        
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        
        
        # replace <image> with image tensor, replace <speech> with speech tensor
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_speeches = (cur_input_ids == SPEECH_TOKEN_INDEX).sum()
            if num_images == 0 and num_speeches == 0:
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                continue
            
            # Identify token indices
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            speech_token_indices = torch.where(cur_input_ids == SPEECH_TOKEN_INDEX)[0].tolist()

            # Merge and sort the indices
            token_indices = sorted(image_token_indices + speech_token_indices)

            cur_input_ids_no_special = []
            cur_labels = labels[batch_idx]
            cur_labels_no_special = []
            prev_index = -1

            for index in token_indices + [cur_input_ids.shape[0]]:
                cur_input_ids_no_special.append(cur_input_ids[prev_index + 1: index])
                cur_labels_no_special.append(cur_labels[prev_index + 1: index])
                prev_index = index

            split_sizes = [x.shape[0] for x in cur_labels_no_special]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_no_special))
            cur_input_embeds_no_special = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            cur_image_idx = 0
            cur_speech_idx = 0

            for i, index in enumerate(token_indices):
                cur_new_input_embeds.append(cur_input_embeds_no_special[i])
                cur_new_labels.append(cur_labels_no_special[i])

                if index in image_token_indices:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                
                elif index in speech_token_indices:
                    cur_speech_features = speech_features[cur_speech_idx]
                    cur_speech_idx += 1
                    cur_new_input_embeds.append(cur_speech_features)
                    cur_new_labels.append(torch.full((cur_speech_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds.append(cur_input_embeds_no_special[-1])
            cur_new_labels.append(cur_labels_no_special[-1])

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        
        # truncate then padding
        
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        
        

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
