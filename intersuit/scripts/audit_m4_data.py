#!/usr/bin/env python
"""审计 M4-IT 本地数据是否接近官方发布数据。

这个脚本默认只做本地离线检查；如果安装了 huggingface_hub 且允许联网，
可以加 --check_hf_manifest 对 Hugging Face 仓库文件清单做额外核验。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} 不是 list 格式")
    return data


def rel_exists(root: Path, name: str) -> bool:
    return (root / name).exists()


def collect_resource_refs(samples: list[dict[str, Any]]) -> tuple[Counter[str], Counter[str]]:
    images: Counter[str] = Counter()
    audios: Counter[str] = Counter()
    for item in samples:
        for key in ("image", "images"):
            value = item.get(key)
            if isinstance(value, str):
                images[value] += 1
            elif isinstance(value, list):
                for entry in value:
                    if isinstance(entry, str):
                        images[entry] += 1

        has_explicit_audio = False
        for key in ("speech", "audio", "audio_file", "speech_file", "audios"):
            value = item.get(key)
            if isinstance(value, str):
                has_explicit_audio = True
                audios[value] += 1
            elif isinstance(value, list):
                for entry in value:
                    if isinstance(entry, str):
                        has_explicit_audio = True
                        audios[entry] += 1

        item_id = item.get("id")
        conversations = item.get("conversations")
        if not has_explicit_audio and isinstance(item_id, str) and isinstance(conversations, list):
            human_idx = 0
            for turn in conversations:
                if not isinstance(turn, dict):
                    continue
                value = str(turn.get("value", ""))
                if turn.get("from") == "human":
                    if "<speech>" in value or "<audio>" in value:
                        audios[f"{item_id}_{human_idx}.wav"] += 1
                    human_idx += 1
    return images, audios


def count_files(root: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not root.exists():
        return counts
    for path in root.rglob("*"):
        if path.is_file():
            suffix = path.suffix.lower()
            if suffix in IMAGE_EXTS:
                counts["image_files"] += 1
            elif suffix in AUDIO_EXTS:
                counts["audio_files"] += 1
            else:
                counts["other_files"] += 1
    return counts


def hf_manifest(repo_id: str, repo_type: str) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover - 依赖环境可选
        return {"ok": False, "error": f"无法导入 huggingface_hub: {exc}"}

    try:
        files = HfApi().list_repo_files(repo_id=repo_id, repo_type=repo_type)
    except Exception as exc:  # pragma: no cover - 依赖环境/网络可选
        return {"ok": False, "error": f"无法读取 Hugging Face 清单: {exc}"}

    suffix_counts: Counter[str] = Counter(Path(name).suffix.lower() or "<no_suffix>" for name in files)
    return {
        "ok": True,
        "repo_id": repo_id,
        "repo_type": repo_type,
        "file_count": len(files),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "sample_files": files[:30],
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    text_path = Path(args.text_json).resolve()
    audio_text_path = Path(args.audio_json).resolve() if args.audio_json else None
    image_root = Path(args.image_root).resolve()
    speech_root = Path(args.speech_root).resolve() if args.speech_root else image_root

    samples = load_json(text_path)
    audio_samples = load_json(audio_text_path) if audio_text_path and audio_text_path.exists() else []
    image_refs, audio_refs_from_text = collect_resource_refs(samples)
    audio_image_refs, audio_refs = collect_resource_refs(audio_samples)
    image_refs.update(audio_image_refs)
    audio_refs.update(audio_refs_from_text)

    missing_images = sorted(name for name in image_refs if not rel_exists(image_root, name))
    missing_audio = sorted(name for name in audio_refs if not rel_exists(speech_root, name) and not rel_exists(image_root, name))
    duplicate_ids = [key for key, count in Counter(str(item.get("id")) for item in samples).items() if count > 1]

    report: dict[str, Any] = {
        "status": "pass",
        "text_json": str(text_path),
        "audio_json": str(audio_text_path) if audio_text_path else None,
        "image_root": str(image_root),
        "speech_root": str(speech_root),
        "expected_m4_it_count": args.expected_count,
        "sample_count": len(samples),
        "audio_sample_count": len(audio_samples) if audio_samples else None,
        "unique_image_refs": len(image_refs),
        "unique_audio_refs": len(audio_refs),
        "missing_image_count": len(missing_images),
        "missing_audio_count": len(missing_audio),
        "duplicate_id_count": len(duplicate_ids),
        "local_file_counts": dict(count_files(image_root)),
        "missing_images_head": missing_images[:50],
        "missing_audio_head": missing_audio[:50],
        "duplicate_ids_head": duplicate_ids[:50],
        "notes_zh": [],
    }

    if len(samples) != args.expected_count:
        report["status"] = "fail"
        report["notes_zh"].append(f"样本数不是预期的 {args.expected_count} 条。")
    if missing_images:
        report["status"] = "fail"
        report["notes_zh"].append("存在 JSON 引用但本地缺失的图像文件。")
    if missing_audio:
        report["status"] = "fail"
        report["notes_zh"].append("存在 JSON 引用但本地缺失的音频文件。")
    if duplicate_ids:
        report["status"] = "warn" if report["status"] == "pass" else report["status"]
        report["notes_zh"].append("存在重复 id，需要确认是否为官方数据预期。")
    if not audio_refs:
        report["status"] = "warn" if report["status"] == "pass" else report["status"]
        report["notes_zh"].append("音频 JSON 未显式引用 speech/audio 字段；如果音频由 id_轮次.wav 隐式对应，需要用训练加载器再做一次端到端校验。")

    if args.check_hf_manifest:
        report["hf_manifest"] = hf_manifest(args.hf_repo, args.hf_repo_type)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 M4-IT 本地数据完整性。")
    parser.add_argument("--text_json", default="intersuit/inputs/texts/m4-it-qwen.json")
    parser.add_argument("--audio_json", default="intersuit/inputs/texts/m4-it-qwen-audio.json")
    parser.add_argument("--image_root", default="intersuit/inputs/images/llava-next")
    parser.add_argument("--speech_root", default="")
    parser.add_argument("--expected_count", type=int, default=9963)
    parser.add_argument("--output_json", default="intersuit/train_logs/m4_data_audit.json")
    parser.add_argument("--check_hf_manifest", action="store_true")
    parser.add_argument("--hf_repo", default="ColorfulAI/M4-IT")
    parser.add_argument("--hf_repo_type", default="dataset")
    args = parser.parse_args()

    report = build_report(args)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] in {"pass", "warn"} else 1)


if __name__ == "__main__":
    main()
