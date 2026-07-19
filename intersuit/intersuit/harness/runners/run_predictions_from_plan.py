#!/usr/bin/env python
"""Generate prediction JSONL files from an AS-M4 attribution matrix plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest_lock(exp: dict[str, Any], manifest: Path) -> None:
    expected = str(exp.get("manifest_sha256") or "").strip().lower()
    if not expected:
        return
    actual = sha256_file(manifest)
    if actual != expected:
        raise RuntimeError(
            f"{exp['id']} manifest SHA256 不匹配：expected={expected}, actual={actual}"
        )


def iter_qa_samples(manifest: Path, limit: int | None = None) -> list[dict[str, Any]]:
    data = load_json(manifest)
    if not isinstance(data, list):
        raise TypeError("manifest must be a JSON list")
    rows: list[dict[str, Any]] = []
    for sample in data:
        conversations = sample.get("conversations") or []
        for idx in range(0, len(conversations) - 1, 2):
            human = conversations[idx]
            gpt = conversations[idx + 1]
            if human.get("from") != "human" or gpt.get("from") != "gpt":
                continue
            rows.append(
                {
                    "id": f"{sample.get('id', 'sample')}_turn{idx // 2}",
                    "sample_id": sample.get("id"),
                    "question": human.get("value", ""),
                    "raw_question": sample.get("question") or str(human.get("value", "")).replace("<image>", "").strip(),
                    "choices": sample.get("choices"),
                    "answer": gpt.get("value", ""),
                    "generation_mode": sample.get("generation_mode", "generate"),
                    "context": sample.get("context"),
                    "new_query": sample.get("new_query"),
                    "new_query_pos": sample.get("new_query_pos", 20),
                    "video_path": sample.get("video_path"),
                    "video_max_frames": sample.get("video_max_frames"),
                    "video_features": sample.get("video_features"),
                    "scene_audio": sample.get("scene_audio"),
                    "scene_audio_path": sample.get("scene_audio_path"),
                    "scene_audio_sample_rate": sample.get("scene_audio_sample_rate", 16000),
                    "scene_audio_window_sec": sample.get("scene_audio_window_sec", 1.0),
                    "scene_audio_hop_sec": sample.get("scene_audio_hop_sec", 0.5),
                    "scene_audio_timestamps": sample.get("scene_audio_timestamps"),
                    "frame_timestamps": sample.get("frame_timestamps"),
                    "accept_contains": sample.get("accept_contains"),
                    "accept_regex": sample.get("accept_regex"),
                }
            )
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def format_choice_query(question: str, choices: Any) -> tuple[str, str]:
    """Format benchmark choices without inventing or reordering option labels."""

    if not isinstance(choices, dict) or not choices:
        return question.strip(), ""
    formatted_choices = "\n".join(f"{label}. {str(value).strip()}" for label, value in choices.items())
    query = (
        f"Question: {question.strip()}\n\nOptions:\n{formatted_choices}\n\n"
        "Answer with only A, B, C, or D."
    )
    return query, formatted_choices


def extract_generated_token_ids(output_ids: torch.Tensor, input_ids: torch.Tensor, generation_mode: str) -> torch.Tensor:
    """Accept both generated-only and full-sequence generation APIs."""

    if output_ids.ndim != 2:
        raise ValueError(f"generated ids must be rank 2, got shape={tuple(output_ids.shape)}")
    if generation_mode == "parallel":
        return output_ids
    input_length = int(input_ids.shape[1])
    if output_ids.shape[1] >= input_length and torch.equal(output_ids[:, :input_length], input_ids):
        return output_ids[:, input_length:]
    return output_ids


def decode_generated_tokens(tokenizer: Any, new_token_ids: torch.Tensor) -> tuple[str, dict[str, Any]]:
    raw = tokenizer.batch_decode(new_token_ids, skip_special_tokens=False)[0] if new_token_ids.shape[1] else ""
    clean = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0] if new_token_ids.shape[1] else ""
    ids = new_token_ids[0].detach().cpu().tolist() if new_token_ids.shape[1] else []
    first_id = ids[0] if ids else None
    first_token = tokenizer.convert_ids_to_tokens(first_id) if first_id is not None else None
    return clean.strip(), {
        "new_token_count": len(ids),
        "first_new_token_id": first_id,
        "first_new_token": first_token,
        "first_token_is_eos": first_id == tokenizer.eos_token_id if first_id is not None else False,
        "decode_skip_special_tokens_false": raw,
        "decode_skip_special_tokens_true": clean,
        "first_20_new_token_ids": ids[:20],
        "last_20_new_token_ids": ids[-20:],
    }


def _find_token_subsequence(sequence: torch.Tensor, subsequence: torch.Tensor) -> tuple[int | None, int | None]:
    values = sequence.detach().cpu().tolist()
    needle = subsequence.detach().cpu().tolist()
    if not needle:
        return None, None
    for start in range(len(values) - len(needle) + 1):
        if values[start : start + len(needle)] == needle:
            return start, start + len(needle)
    return None, None


def prediction_correct(prediction: str, qa: dict[str, Any]) -> bool:
    normalized_prediction = normalize_text(prediction)
    choices = qa.get("choices")
    if isinstance(choices, dict) and re.fullmatch(r"[a-d]", normalized_prediction):
        label = normalized_prediction.upper()
        selected = choices.get(label)
        return normalize_text(selected) == normalize_text(qa["answer"]) or label == str(qa["answer"]).strip().upper()
    contains = qa.get("accept_contains")
    contains_match = False
    if contains:
        if isinstance(contains, str):
            contains = [contains]
        contains_match = any(normalize_text(item) in normalized_prediction for item in contains)
    patterns = qa.get("accept_regex")
    regex_match = False
    if patterns:
        if isinstance(patterns, str):
            patterns = [patterns]
        regex_match = any(re.search(str(pattern), prediction or "", flags=re.IGNORECASE) for pattern in patterns)
    if contains or patterns:
        return contains_match or regex_match
    return normalized_prediction == normalize_text(qa["answer"])


def jsonable_diagnostics(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        detached = value.detach().float().cpu()
        if detached.numel() == 1:
            return float(detached.item())
        return detached.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable_diagnostics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_diagnostics(item) for item in value]
    return value


def first_diagnostic_scalar(diagnostics: Any, key: str) -> float | None:
    def flatten_numbers(value: Any) -> list[float]:
        if isinstance(value, (int, float)):
            number = float(value)
            return [number] if math.isfinite(number) else []
        if isinstance(value, list):
            values: list[float] = []
            for item in value:
                values.extend(flatten_numbers(item))
            return values
        return []

    if not isinstance(diagnostics, list):
        return None
    for item in diagnostics:
        if isinstance(item, dict) and item.get(key) is not None:
            values = flatten_numbers(item[key])
            if values:
                return sum(values) / len(values)
            return None
    return None


def stable_bucket(value: Any) -> int:
    text = str(value or "")
    numbers = re.findall(r"\d+", text)
    if numbers:
        return int(numbers[-1])
    return sum(ord(ch) for ch in text)


def oracle_prediction(answer: str, exp_id: str, qa_id: Any = "") -> str:
    """Deterministic CPU-only backend for pipeline smoke tests."""

    # Keep E2/E6 stronger and E3/E5 weaker so the scorer can exercise the
    # attribution inequalities without launching a model. E0/E1/E7 are partial
    # and identical, which preserves the "AS-M4 muted ~= original M4" check.
    if exp_id in {"E3", "E5"}:
        return "__wrong__"
    if exp_id in {"E0", "E1", "E7"} and stable_bucket(qa_id) % 2 == 1:
        return "__wrong__"
    return answer


def _audio_to_tensor(audio: Any) -> torch.Tensor | None:
    if audio is None:
        return None
    value = torch.as_tensor(audio, dtype=torch.float32)
    if value.numel() == 0:
        return None
    return value


def apply_audio_condition(
    qa: dict[str, Any],
    exp: dict[str, Any],
    audio_pool: list[dict[str, Any]],
    sample_index: int,
) -> dict[str, Any]:
    """Return a copy of ``qa`` with the experiment's audio condition applied."""

    result = dict(qa)
    condition = str(exp.get("audio_condition") or "correct")
    audio = _audio_to_tensor(result.get("scene_audio"))
    timestamps = result.get("scene_audio_timestamps")

    if condition in {"none", "muted"}:
        result["scene_audio"] = None
        result["scene_audio_timestamps"] = None
        return result

    if condition == "mismatched":
        if audio_pool:
            source = audio_pool[(sample_index + 1) % len(audio_pool)]
            result["scene_audio"] = source.get("scene_audio")
            result["scene_audio_timestamps"] = source.get("scene_audio_timestamps")
            if source.get("scene_audio_path"):
                result["scene_audio_path"] = source["scene_audio_path"]
                for key in (
                    "scene_audio_sample_rate",
                    "scene_audio_window_sec",
                    "scene_audio_hop_sec",
                ):
                    if source.get(key) is not None:
                        result[key] = source[key]
        return result

    if audio is None:
        return result

    if condition == "noisy":
        noise = torch.linspace(-0.05, 0.05, steps=audio.numel(), dtype=audio.dtype).reshape_as(audio)
        result["scene_audio"] = (audio + noise).tolist()
        return result

    if condition == "shifted":
        if audio.ndim >= 2 and audio.shape[0] > 1:
            result["scene_audio"] = torch.roll(audio, shifts=1, dims=0).tolist()
            if timestamps is not None:
                ts = torch.as_tensor(timestamps, dtype=torch.float32)
                if ts.ndim >= 2 and ts.shape[0] == audio.shape[0]:
                    result["scene_audio_timestamps"] = torch.roll(ts, shifts=1, dims=0).tolist()
        return result

    return result


def load_model_once(model_path: str, device: str, model_name_override: str | None = None):
    from intersuit.mm_utils import get_model_name_from_path
    from intersuit.model.builder import load_pretrained_model

    model_path_abs = Path(model_path)
    if not model_path_abs.is_absolute():
        model_path_abs = (REPO_ROOT / model_path_abs).resolve()
    model_path_str = str(model_path_abs)
    model_name = model_name_override or get_model_name_from_path(model_path_str)
    cwd = Path.cwd()
    try:
        os.chdir(INTERSUIT_ROOT)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path_str,
            None,
            model_name,
            device_map=device,
            multimodal=True,
            attn_implementation="eager",
            overwrite_config={
                "mm_spatial_pool_stride": 2,
                "mm_spatial_pool_mode": "average",
            },
        )
    finally:
        os.chdir(cwd)
    model.eval()
    return tokenizer, model, image_processor, context_len


def maybe_replace_scene_audio_encoder(model: Any, env: dict[str, str]) -> tuple[Any, Any]:
    encoder_type = env.get("AS_M4_SCENE_AUDIO_ENCODER_TYPE")
    if not encoder_type:
        return None, None
    model_body = model.get_model() if hasattr(model, "get_model") else model
    previous_encoder = getattr(model_body, "scene_audio_encoder", None)
    previous_type = getattr(model.config, "scene_audio_encoder_type", None)
    model.config.scene_audio_encoder_type = encoder_type
    model.config.scene_audio_torchaudio_bundle = env.get("AS_M4_SCENE_AUDIO_TORCHAUDIO_BUNDLE", "WAV2VEC2_BASE")
    model.config.scene_audio_torchaudio_weight_path = env.get("AS_M4_SCENE_AUDIO_TORCHAUDIO_WEIGHT_PATH")
    model.config.scene_audio_sample_rate = int(env.get("AS_M4_SCENE_AUDIO_SAMPLE_RATE", "16000"))
    model.config.scene_audio_precomputed_shared_space = env.get(
        "AS_M4_SCENE_AUDIO_PRECOMPUTED_SHARED_SPACE", "0"
    ) in {"1", "true", "True", "yes"}

    from intersuit.model.scene_audio_encoder.builder import build_scene_audio_encoder

    device = next(model.parameters()).device
    replacement = build_scene_audio_encoder(model.config).to(device=device)
    replacement.eval()
    setattr(model_body, "scene_audio_encoder", replacement)
    return previous_encoder, previous_type


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidates = [
        (INTERSUIT_ROOT / path).resolve(),
        (REPO_ROOT / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_video_tensor(video_path: str, image_processor: Any, model: Any, device: str, max_frames: int | None = None) -> torch.Tensor:
    from intersuit.vid_utils import load_video

    path = _resolve_repo_path(video_path)
    max_frames = int(max_frames or getattr(model.config, "as_m4_harness_max_frames", 32))
    frames = load_video(str(path), num_frames=max_frames, max_frames=max_frames)
    tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    model_dtype = next(model.parameters()).dtype
    return tensor.to(device=device, dtype=model_dtype)


def _apply_debug_waveform_condition(
    waveform: torch.Tensor,
    sample_rate: int,
    condition: str = "original",
    shift_seconds: float = 0.0,
) -> torch.Tensor:
    """Apply debug-only audio perturbations without changing tensor length."""

    if condition == "original" or condition == "wrong_audio":
        return waveform
    if condition == "silence":
        return torch.zeros_like(waveform)
    if condition != "shift":
        raise ValueError(f"Unknown AS-M4 debug audio condition: {condition}")
    shift_samples = int(round(float(shift_seconds) * sample_rate))
    if shift_samples == 0:
        return waveform.clone()
    result = torch.zeros_like(waveform)
    length = int(waveform.shape[-1])
    amount = min(abs(shift_samples), length)
    if shift_samples > 0 and amount < length:
        result[..., amount:] = waveform[..., : length - amount]
    elif shift_samples < 0 and amount < length:
        result[..., : length - amount] = waveform[..., amount:]
    return result


def _load_scene_audio_tensor(
    qa: dict[str, Any],
    debug_condition: str = "original",
    debug_source_path: str | None = None,
    debug_shift_seconds: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]] | tuple[None, None, None]:
    audio_path = debug_source_path or qa.get("scene_audio_path")
    if not audio_path:
        return None, None, None
    from intersuit.streaming.audio_stream import load_scene_audio, split_audio_windows, stack_audio_windows

    path = _resolve_repo_path(str(audio_path))
    waveform, sample_rate = load_scene_audio(path, sample_rate=int(qa.get("scene_audio_sample_rate") or 16000), mono=True)
    original_num_samples = int(waveform.shape[-1])
    waveform = _apply_debug_waveform_condition(
        waveform,
        sample_rate,
        condition=debug_condition,
        shift_seconds=debug_shift_seconds,
    )
    if int(waveform.shape[-1]) != original_num_samples:
        raise ValueError("Debug audio perturbation changed waveform length")
    windows = split_audio_windows(
        waveform,
        sample_rate,
        window_sec=float(qa.get("scene_audio_window_sec") or 1.0),
        hop_sec=float(qa.get("scene_audio_hop_sec") or 0.5),
    )
    samples, timestamps = stack_audio_windows(windows)
    if samples.numel() == 0:
        raise ValueError(f"Scene audio decoded to zero samples: {path}")
    if not torch.isfinite(samples).all() or not torch.isfinite(timestamps).all():
        raise ValueError(f"Scene audio contains NaN/Inf: {path}")
    if samples.shape[0] != timestamps.shape[0]:
        raise ValueError("Scene audio window/timestamp count mismatch")
    metadata = {
        "condition": debug_condition,
        "source_path": str(path),
        "shift_seconds": float(debug_shift_seconds),
        "waveform_num_samples": original_num_samples,
        "window_count": int(samples.shape[0]),
        "window_num_samples": int(samples.shape[-1]),
        "input_window_norm": float(samples.float().norm().item()),
    }
    return samples, timestamps, metadata


def _first_token_logits_debug(scores: torch.Tensor, tokenizer: Any) -> dict[str, Any]:
    logits = scores[0].detach().float().cpu()
    probabilities = torch.softmax(logits, dim=-1)
    _, top_ids = torch.topk(logits, k=min(10, logits.numel()))

    def record(token_id: int) -> dict[str, Any]:
        token_id = int(token_id)
        logit = float(logits[token_id].item())
        return {
            "token_id": token_id,
            "token_text": tokenizer.convert_ids_to_tokens(token_id),
            "logit": logit,
            "probability": float(probabilities[token_id].item()),
            "rank": int((logits > logit).sum().item()) + 1,
        }

    top10 = [record(int(token_id)) for token_id in top_ids.tolist()]
    focus: dict[str, Any] = {}
    for label in ("A", "B", "C", "D"):
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Expected {label!r} to encode to one token, got {encoded}")
        focus[label] = record(encoded[0])
    focus["EOS"] = record(tokenizer.eos_token_id)
    return {"top10": top10, "focus_tokens": focus}


def model_prediction(
    qa: dict[str, Any],
    exp: dict[str, Any],
    model_cache: dict[str, Any],
    feature_root: Path,
    device: str,
    max_new_tokens: int,
    model_name_override: str | None = None,
    dump_diagnostics: bool = False,
) -> tuple[str, Any, dict[str, Any]]:
    from intersuit.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from intersuit.conversation import conv_templates
    from intersuit.mm_utils import tokenizer_image_token

    model_path = exp["model_path"]
    if model_path not in model_cache:
        model_cache[model_path] = load_model_once(model_path, device, model_name_override=model_name_override)
    tokenizer, model, image_processor, _context_len = model_cache[model_path]

    raw_question = str(qa.get("raw_question") or qa["question"]).replace(DEFAULT_IMAGE_TOKEN, "").strip()
    final_query, formatted_choices = format_choice_query(raw_question, qa.get("choices"))
    question = f"{DEFAULT_IMAGE_TOKEN}\n{final_query}" if qa.get("choices") else str(qa["question"])
    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    assistant_marker = conv.roles[1]
    prompt_debug = {
        "raw_question": raw_question,
        "raw_choices": qa.get("choices"),
        "formatted_choices": formatted_choices,
        "final_query": final_query,
        "full_conversation_prompt": prompt,
        "conversation_template": "qwen_1_5",
        "video_placeholder_positions": [match.start() for match in re.finditer(re.escape(DEFAULT_IMAGE_TOKEN), prompt)],
        "assistant_role_marker_positions": [match.start() for match in re.finditer(re.escape(assistant_marker), prompt)],
        "prompt_ends_at_assistant_boundary": prompt.endswith(assistant_marker + "\n"),
    }
    token_debug: dict[str, Any] = {
        "generation_mode": str(qa.get("generation_mode") or "generate"),
        "input_ids_shape": list(input_ids.shape),
        "input_ids_length": int(input_ids.shape[1]),
        "attention_mask_shape": list(attention_mask.shape),
        "attention_mask_sum": int(attention_mask.sum().item()),
        "image_token_count": int((input_ids == IMAGE_TOKEN_INDEX).sum().item()),
        "image_token_positions": torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].detach().cpu().tolist(),
        "max_new_tokens": int(max_new_tokens),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
    }
    query_token_ids = tokenizer.encode(final_query, add_special_tokens=False, return_tensors="pt").to(device)
    query_start, query_end = _find_token_subsequence(input_ids[0], query_token_ids[0])
    assistant_token_ids = tokenizer.encode(assistant_marker, add_special_tokens=False, return_tensors="pt").to(device)
    assistant_start, _assistant_end = _find_token_subsequence(input_ids[0], assistant_token_ids[0])
    token_debug.update(
        {
            "query_start_position": query_start,
            "query_end_position": query_end,
            "assistant_start_position": assistant_start,
            "computed_prefix_length": None,
        }
    )

    model_dtype = next(model.parameters()).dtype
    modalities = ["video_feature"]
    images: Any
    using_video_feature = qa.get("video_features") is not None
    if qa.get("video_path"):
        max_frames = int(qa["video_max_frames"]) if qa.get("video_max_frames") is not None else None
        images = [_load_video_tensor(str(qa["video_path"]), image_processor, model, device, max_frames=max_frames)]
        modalities = ["video"]
        using_video_feature = False
    else:
        video_features = torch.load(feature_root / str(qa["video_features"]), map_location="cpu")
        if isinstance(video_features, dict):
            video_features = video_features.get("features", video_features.get("video_features"))
        if video_features is None:
            raise ValueError(f"Missing video features for sample {qa['id']}")
        images = video_features.unsqueeze(0).to(device=device, dtype=model_dtype)

    scene_audios = None
    scene_audio_timestamps = None
    audio_input_debug = None
    if exp.get("env", {}).get("AS_M4_ENABLE_SCENE_AUDIO") == "1" and qa.get("scene_audio") is not None:
        scene_audios = torch.as_tensor(qa["scene_audio"], dtype=model_dtype, device=device).unsqueeze(0)
        if qa.get("scene_audio_timestamps") is not None:
            scene_audio_timestamps = torch.as_tensor(qa["scene_audio_timestamps"], dtype=torch.float32, device=device).unsqueeze(0)
    elif exp.get("env", {}).get("AS_M4_ENABLE_SCENE_AUDIO") == "1" and qa.get("scene_audio_path"):
        debug_env = exp.get("env", {})
        loaded_audio, loaded_timestamps, audio_input_debug = _load_scene_audio_tensor(
            qa,
            debug_condition=str(debug_env.get("AS_M4_DEBUG_AUDIO_CONDITION") or "original"),
            debug_source_path=debug_env.get("AS_M4_DEBUG_AUDIO_SOURCE_PATH"),
            debug_shift_seconds=float(debug_env.get("AS_M4_DEBUG_AUDIO_SHIFT_SECONDS") or 0.0),
        )
        if loaded_audio is not None:
            scene_audios = loaded_audio.to(device=device, dtype=model_dtype).unsqueeze(0)
            scene_audio_timestamps = loaded_timestamps.to(device=device, dtype=torch.float32).unsqueeze(0)
    elif using_video_feature and getattr(model.config, "enable_scene_audio", False):
        # Precomputed video_feature inputs are only packed by the streaming AV
        # path. For muted/behavior rollback experiments, feed a tiny zero audio
        # window and force gate=0 so the path is exercised without audio effect.
        scene_audios = torch.zeros((1, 1, 16), dtype=model_dtype, device=device)
        scene_audio_timestamps = torch.zeros((1, 1, 2), dtype=torch.float32, device=device)

    force_audio_gate = exp.get("env", {}).get("AS_M4_FORCE_AUDIO_GATE")
    enable_scene_audio = exp.get("env", {}).get("AS_M4_ENABLE_SCENE_AUDIO")
    previous_force_audio_gate = getattr(model.config, "force_audio_gate", None)
    previous_enable_scene_audio = getattr(model.config, "enable_scene_audio", None)
    previous_residual_scale = getattr(model.config, "debug_audio_residual_scale", 1.0)
    previous_delta_ratio_cap = getattr(model.config, "audio_delta_ratio_cap", 0.0)
    previous_gate_v1 = getattr(model.config, "enable_audio_confidence_gate_v1", False)
    previous_event_aligner_v1 = getattr(model.config, "enable_audio_event_aligner_v1", False)
    previous_scene_audio_encoder = None
    previous_scene_audio_encoder_type = None
    if force_audio_gate is not None:
        model.config.force_audio_gate = float(force_audio_gate)
    if enable_scene_audio is not None:
        model.config.enable_scene_audio = str(enable_scene_audio) in {"1", "true", "True"}
    residual_scale = exp.get("env", {}).get("AS_M4_DEBUG_AUDIO_RESIDUAL_SCALE")
    active_residual_scale = float(residual_scale) if residual_scale is not None else 1.0
    model.config.debug_audio_residual_scale = active_residual_scale
    delta_ratio_cap = exp.get("env", {}).get("AS_M4_AUDIO_DELTA_RATIO_CAP")
    active_delta_ratio_cap = float(delta_ratio_cap) if delta_ratio_cap is not None else 0.0
    model.config.audio_delta_ratio_cap = active_delta_ratio_cap
    gate_v1 = exp.get("env", {}).get("AS_M4_ENABLE_AUDIO_CONFIDENCE_GATE_V1")
    active_gate_v1 = str(gate_v1 or "0") in {"1", "true", "True"}
    model.config.enable_audio_confidence_gate_v1 = active_gate_v1
    event_aligner_v1 = exp.get("env", {}).get("AS_M4_ENABLE_AUDIO_EVENT_ALIGNER_V1")
    active_event_aligner_v1 = str(event_aligner_v1 or "0") in {"1", "true", "True"}
    model.config.enable_audio_event_aligner_v1 = active_event_aligner_v1
    streaming_av_module = getattr(model.get_model(), "streaming_av_module", None)
    if isinstance(streaming_av_module, list):
        streaming_av_module = streaming_av_module[0]
    previous_module_gate_v1 = None
    if streaming_av_module is not None and hasattr(streaming_av_module, "confidence_gate"):
        previous_module_gate_v1 = streaming_av_module.confidence_gate.enable_v1
        streaming_av_module.confidence_gate.enable_v1 = active_gate_v1
    if exp.get("env", {}).get("AS_M4_SCENE_AUDIO_ENCODER_TYPE"):
        previous_scene_audio_encoder, previous_scene_audio_encoder_type = maybe_replace_scene_audio_encoder(
            model,
            exp.get("env", {}),
        )
    frame_timestamps_arg = None
    if using_video_feature and qa.get("frame_timestamps") is not None:
        frame_timestamps_arg = torch.as_tensor(qa.get("frame_timestamps") or [], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.inference_mode():
        first_token_logits = None
        try:
            if str(qa.get("generation_mode") or "generate") == "parallel":
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                context = str(qa.get("context") or "Can you describe the video?")
                if DEFAULT_IMAGE_TOKEN not in context:
                    context = DEFAULT_IMAGE_TOKEN + "\n" + context
                base_conv = conv_templates["qwen_1_5"].copy()
                base_conv.append_message(base_conv.roles[0], context)
                base_conv.append_message(base_conv.roles[1], None)
                base_ids = tokenizer_image_token(base_conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
                base_attention = base_ids.ne(pad_token_id).to(device)

                new_query_str = str(qa.get("new_query") or question.replace(DEFAULT_IMAGE_TOKEN, "").strip())
                if qa.get("choices"):
                    new_query_str = final_query
                new_conv = conv_templates["qwen_1_5"].copy()
                new_conv.append_message(new_conv.roles[0], new_query_str)
                new_conv.append_message(new_conv.roles[1], None)
                new_query_ids = tokenizer_image_token(new_conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
                query_token_ids = tokenizer.encode(new_query_str, add_special_tokens=False, return_tensors="pt").to(device)
                query_start, query_end = _find_token_subsequence(new_query_ids[0], query_token_ids[0])
                token_debug.update(
                    {
                        "base_input_ids_shape": list(base_ids.shape),
                        "base_input_ids_length": int(base_ids.shape[1]),
                        "base_attention_mask_shape": list(base_attention.shape),
                        "base_attention_mask_sum": int(base_attention.sum().item()),
                        "query_start_position": query_start,
                        "query_end_position": query_end,
                        "assistant_start_position": int(new_query_ids.shape[1]),
                        "new_query_pos": int(qa.get("new_query_pos") or 20),
                    }
                )

                output_ids = model.generate_parallel(
                    base_ids,
                    attention_mask=base_attention,
                    pad_token_id=pad_token_id,
                    images=images,
                    modalities=modalities,
                    scene_audios=scene_audios,
                    scene_audio_timestamps=scene_audio_timestamps,
                    frame_timestamps=frame_timestamps_arg,
                    use_cache=True,
                    new_query=new_query_ids,
                    new_query_str=new_query_str,
                    new_query_pos=int(qa.get("new_query_pos") or 20),
                    query_str=context,
                    tokenizer=tokenizer,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=max_new_tokens,
                )
                new_token_ids = extract_generated_token_ids(output_ids, base_ids, "parallel")
            else:
                capture_logits = str(exp.get("env", {}).get("AS_M4_DEBUG_CAPTURE_FIRST_TOKEN_LOGITS") or "0") in {"1", "true", "True"}
                generation_output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    modalities=modalities,
                    scene_audios=scene_audios,
                    scene_audio_timestamps=scene_audio_timestamps,
                    frame_timestamps=frame_timestamps_arg,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    return_dict_in_generate=capture_logits,
                    output_scores=capture_logits,
                )
                if capture_logits:
                    output_ids = generation_output.sequences
                    if generation_output.scores:
                        first_token_logits = _first_token_logits_debug(generation_output.scores[0], tokenizer)
                else:
                    output_ids = generation_output
                new_token_ids = extract_generated_token_ids(output_ids, input_ids, "generate")

        finally:
            model.config.force_audio_gate = previous_force_audio_gate
            if previous_enable_scene_audio is not None:
                model.config.enable_scene_audio = previous_enable_scene_audio
            model.config.debug_audio_residual_scale = previous_residual_scale
            model.config.audio_delta_ratio_cap = previous_delta_ratio_cap
            model.config.enable_audio_confidence_gate_v1 = previous_gate_v1
            model.config.enable_audio_event_aligner_v1 = previous_event_aligner_v1
            if previous_scene_audio_encoder is not None:
                model_body = model.get_model() if hasattr(model, "get_model") else model
                setattr(model_body, "scene_audio_encoder", previous_scene_audio_encoder)
                model.config.scene_audio_encoder_type = previous_scene_audio_encoder_type
            if previous_module_gate_v1 is not None:
                streaming_av_module.confidence_gate.enable_v1 = previous_module_gate_v1
    diagnostics = None
    if dump_diagnostics:
        diagnostics = jsonable_diagnostics(getattr(model, "_last_streaming_av_diagnostics", None))
    prediction, decoded_debug = decode_generated_tokens(tokenizer, new_token_ids)
    output_values = output_ids[0].detach().cpu().tolist()
    token_debug.update(decoded_debug)
    token_debug.update(jsonable_diagnostics(getattr(model, "_last_generation_debug", {})) or {})
    token_debug.update(
        {
            "generated_ids_shape": list(output_ids.shape),
            "full_generated_token_count": int(output_ids.shape[1]),
            "first_20_full_generated_ids": output_values[:20],
        }
    )
    token_debug["audio_input"] = audio_input_debug
    token_debug["audio_residual_scale"] = active_residual_scale
    token_debug["audio_delta_ratio_cap"] = active_delta_ratio_cap
    token_debug["audio_confidence_gate_v1"] = active_gate_v1
    token_debug["audio_event_aligner_v1"] = active_event_aligner_v1
    token_debug["first_token_logits"] = first_token_logits
    return prediction, diagnostics, {"prompt": prompt_debug, "tokens": token_debug}


def run_predictions(
    plan_path: Path,
    backend: str,
    limit: int | None,
    feature_root: Path,
    device: str,
    max_new_tokens: int,
    experiments: set[str] | None = None,
    model_name_override: str | None = None,
    dry_run: bool = False,
    dump_diagnostics: bool = False,
    debug_dir: Path | None = None,
) -> dict[str, Any]:
    plan = read_jsonl(plan_path)
    feature_root = feature_root.resolve()
    model_cache: dict[str, Any] = {}
    outputs: list[str] = []
    num_selected = 0
    prompt_debug_records: list[dict[str, Any]] = []
    token_debug_records: list[dict[str, Any]] = []
    for exp in plan:
        exp_id = str(exp["id"])
        if experiments is not None and exp_id not in experiments:
            continue
        num_selected += 1
        output_jsonl = Path(str(exp["output_jsonl"]))
        outputs.append(str(output_jsonl))
        manifest = Path(str(exp["manifest"]))
        verify_manifest_lock(exp, manifest)
        if dry_run:
            continue
        all_samples = iter_qa_samples(manifest, limit=None)
        samples = all_samples[:limit] if limit is not None else all_samples
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in (exp.get("env") or {}).items()})
        rows = []
        for sample_index, qa in enumerate(samples):
            qa = apply_audio_condition(qa, exp, all_samples, sample_index)
            if backend == "oracle":
                prediction = oracle_prediction(str(qa["answer"]), str(exp["id"]), qa.get("id"))
                diagnostics = None
                generation_debug = None
            elif backend == "model":
                os.environ.update(env)
                prediction, diagnostics, generation_debug = model_prediction(
                    qa,
                    exp,
                    model_cache,
                    feature_root,
                    device,
                    max_new_tokens,
                    model_name_override=model_name_override,
                    dump_diagnostics=dump_diagnostics,
                )
            else:
                raise ValueError(f"Unknown backend: {backend}")
            correct = prediction_correct(prediction, qa)
            row = {
                "id": qa["id"],
                "sample_id": qa["sample_id"],
                "experiment_id": exp_id,
                "question": qa["question"],
                "answer": qa["answer"],
                "prediction": prediction,
                "correct": correct,
                "audio_condition": exp.get("audio_condition"),
                "alignment": exp.get("alignment"),
                "gate_ablation": exp.get("gate_ablation"),
            }
            if dump_diagnostics:
                row["as_m4_diagnostics"] = diagnostics
                row["gate"] = first_diagnostic_scalar(diagnostics, "gate_mean")
                row["quality_gate"] = first_diagnostic_scalar(diagnostics, "quality_gate")
                row["relevance_gate"] = first_diagnostic_scalar(diagnostics, "relevance_gate")
                row["delta_to_video_ratio"] = first_diagnostic_scalar(diagnostics, "delta_to_video_ratio")
                row["raw_delta_to_video_ratio"] = first_diagnostic_scalar(diagnostics, "raw_delta_to_video_ratio")
                row["audio_delta_applied_scale"] = first_diagnostic_scalar(diagnostics, "audio_delta_applied_scale")
                row["capped_delta_to_video_ratio"] = first_diagnostic_scalar(diagnostics, "capped_delta_to_video_ratio")
                row["audio_norm"] = first_diagnostic_scalar(diagnostics, "audio_norm")
                row["generation_debug"] = generation_debug
                if generation_debug is not None:
                    prompt_debug_records.append({"case": exp_id, "sample_id": qa["sample_id"], **generation_debug["prompt"]})
                    token_debug_records.append({"case": exp_id, "sample_id": qa["sample_id"], **generation_debug["tokens"]})
            rows.append(row)
        with output_jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if debug_dir is not None and dump_diagnostics and not dry_run:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "prompt_debug.json").write_text(json.dumps(prompt_debug_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (debug_dir / "token_debug.json").write_text(json.dumps(token_debug_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "pass",
        "backend": backend,
        "dry_run": dry_run,
        "outputs": outputs,
        "num_plan_experiments": len(plan),
        "num_selected_experiments": num_selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prediction JSONL files for AS-M4 E0-E7 plan.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--backend", choices=["oracle", "model"], default="oracle")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--feature_root", default="intersuit/inputs/features/as_m4_smoke")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--experiments", default="", help="Comma-separated experiment ids to run, e.g. E1,E2.")
    parser.add_argument("--model_name_override", default="", help="Optional model name passed to load_pretrained_model, e.g. LongVA-7B-Qwen2.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--dump_diagnostics", action="store_true", help="Write AS-M4 fusion diagnostics into prediction rows.")
    parser.add_argument("--debug_dir", default="", help="Optional directory for full prompt_debug.json and token_debug.json.")
    args = parser.parse_args()
    experiments = {item.strip() for item in args.experiments.split(",") if item.strip()} or None
    model_name_override = args.model_name_override.strip() or None

    result = run_predictions(
        Path(args.plan),
        backend=args.backend,
        limit=args.limit,
        feature_root=Path(args.feature_root),
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        experiments=experiments,
        model_name_override=model_name_override,
        dry_run=args.dry_run,
        dump_diagnostics=args.dump_diagnostics,
        debug_dir=Path(args.debug_dir) if args.debug_dir else None,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
