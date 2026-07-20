#!/usr/bin/env python
"""诊断 AS-M4 checkpoint 加载后的运行时状态差异。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERSUIT_ROOT = REPO_ROOT / "intersuit"


DEFAULT_CONFIG_KEYS = [
    "_name_or_path",
    "architectures",
    "model_type",
    "torch_dtype",
    "tie_word_embeddings",
    "use_cache",
    "tokenizer_model_max_length",
    "max_position_embeddings",
    "mm_tunable_parts",
    "enable_scene_audio",
    "mm_patch_merge_type",
    "mm_spatial_pool_stride",
    "mm_spatial_pool_mode",
    "image_aspect_ratio",
    "as_m4_fusion_init",
    "as_m4_gate_logit_bias",
    "as_m4_inference_simple_audio_gate",
]


def resolve_path(path_value: str) -> Path:
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


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def tensor_digest(tensor: torch.Tensor) -> str | None:
    if tensor.is_meta:
        return None
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    try:
        raw = value.numpy().tobytes()
    except TypeError:
        raw = value.view(torch.int16).numpy().tobytes()
    digest.update(raw)
    return digest.hexdigest()


def tensor_summary(name: str, tensor: torch.Tensor, hash_pattern: re.Pattern[str] | None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "is_meta": bool(tensor.is_meta),
        "requires_grad": bool(getattr(tensor, "requires_grad", False)),
    }
    if not tensor.is_meta:
        summary["numel"] = int(tensor.numel())
        if tensor.is_floating_point():
            finite = torch.isfinite(tensor.detach())
            summary["finite_all"] = bool(finite.all().item())
            summary["nan_count"] = int(torch.isnan(tensor.detach()).sum().item())
    if hash_pattern is not None and hash_pattern.search(name):
        summary["digest"] = tensor_digest(tensor)
    return summary


def summarize_tokenizer(tokenizer: Any) -> dict[str, Any]:
    return {
        "length": len(tokenizer),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "special_tokens_map": tokenizer.special_tokens_map,
        "additional_special_tokens": list(getattr(tokenizer, "additional_special_tokens", []) or []),
    }


def summarize_config(config: Any) -> dict[str, Any]:
    return {key: jsonable(getattr(config, key, None)) for key in DEFAULT_CONFIG_KEYS}


def summarize_modules(model: Any) -> dict[str, Any]:
    vision_tower = model.get_vision_tower() if hasattr(model, "get_vision_tower") else None
    streaming = model.get_streaming_av_module() if hasattr(model, "get_streaming_av_module") else None
    return {
        "model_class": model.__class__.__name__,
        "base_model_class": model.get_model().__class__.__name__ if hasattr(model, "get_model") else None,
        "vision_tower_class": vision_tower.__class__.__name__ if vision_tower is not None else None,
        "vision_tower_loaded": bool(getattr(vision_tower, "is_loaded", False)) if vision_tower is not None else None,
        "streaming_av_module_class": streaming.__class__.__name__ if streaming is not None else None,
        "has_streaming_av_module": streaming is not None,
    }


def load_and_summarize(
    model_path: str,
    model_name: str,
    device: str,
    hash_regex: str | None,
    state_regex: str | None,
) -> dict[str, Any]:
    from intersuit.model.builder import load_pretrained_model

    model_path_abs = resolve_path(model_path)
    hash_pattern = re.compile(hash_regex) if hash_regex else None
    state_pattern = re.compile(state_regex) if state_regex else None
    warning_messages: list[str] = []

    cwd = Path.cwd()
    try:
        os.chdir(INTERSUIT_ROOT)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tokenizer, model, _image_processor, context_len = load_pretrained_model(
                str(model_path_abs),
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
            warning_messages = [str(item.message) for item in caught]
    finally:
        os.chdir(cwd)

    model.eval()
    state: dict[str, Any] = {}
    with torch.inference_mode():
        for key, value in model.state_dict().items():
            if state_pattern is not None and state_pattern.search(key) is None:
                continue
            state[key] = tensor_summary(key, value, hash_pattern)

    summary = {
        "model_path": str(model_path_abs),
        "context_len": context_len,
        "warnings": warning_messages,
        "tokenizer": summarize_tokenizer(tokenizer),
        "config": summarize_config(model.config),
        "modules": summarize_modules(model),
        "state": state,
    }

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return jsonable(summary)


def diff_dict(left: dict[str, Any], right: dict[str, Any]) -> dict[str, dict[str, Any]]:
    diff: dict[str, dict[str, Any]] = {}
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            diff[key] = {"baseline": left.get(key), "as_m4": right.get(key)}
    return diff


def diff_state(left: dict[str, Any], right: dict[str, Any], max_items: int) -> dict[str, Any]:
    baseline_keys = set(left)
    as_keys = set(right)
    common = sorted(baseline_keys & as_keys)
    changed = []
    for key in common:
        if left[key] != right[key]:
            changed.append({"key": key, "baseline": left[key], "as_m4": right[key]})
            if len(changed) >= max_items:
                break
    return {
        "baseline_only": sorted(baseline_keys - as_keys)[:max_items],
        "as_m4_only": sorted(as_keys - baseline_keys)[:max_items],
        "changed_count_at_least": len(changed),
        "changed_sample": changed,
        "common_count": len(common),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="比较 baseline 与 AS-M4 checkpoint 加载后的运行时状态。")
    parser.add_argument("--baseline", default="checkpoints/M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze")
    parser.add_argument("--as_m4", default="checkpoints/AS-M4-12kbase-smoke-vfeat-asmodules-zero-gatebias-2step")
    parser.add_argument("--model_name", default="LongVA-Qwen2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="intersuit/harness/artifacts/as_m4_runtime_debug.json")
    parser.add_argument(
        "--state_regex",
        default=r"(embed_tokens|lm_head|mm_projector|vision_tower|streaming_av_module|rotary|norm)",
        help="只汇总匹配的 state_dict key；设为空字符串可汇总全部 key。",
    )
    parser.add_argument(
        "--hash_regex",
        default=r"(embed_tokens\.weight|lm_head\.weight|mm_projector|vision_tower.*(patch_embedding|post_layernorm|layers\.0)|streaming_av_module)",
        help="只对匹配的 key 计算 digest；设为空字符串则不计算。",
    )
    parser.add_argument("--max_diff_items", type=int, default=80)
    args = parser.parse_args()

    state_regex = args.state_regex or None
    hash_regex = args.hash_regex or None

    baseline = load_and_summarize(args.baseline, args.model_name, args.device, hash_regex, state_regex)
    as_m4 = load_and_summarize(args.as_m4, args.model_name, args.device, hash_regex, state_regex)
    report = {
        "baseline": baseline,
        "as_m4": as_m4,
        "diff": {
            "config": diff_dict(baseline["config"], as_m4["config"]),
            "tokenizer": diff_dict(baseline["tokenizer"], as_m4["tokenizer"]),
            "modules": diff_dict(baseline["modules"], as_m4["modules"]),
            "state": diff_state(baseline["state"], as_m4["state"], args.max_diff_items),
            "warning_count": {
                "baseline": len(baseline["warnings"]),
                "as_m4": len(as_m4["warnings"]),
            },
        },
    }

    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output), "diff": report["diff"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
