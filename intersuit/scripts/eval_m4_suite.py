#!/usr/bin/env python
"""统一运行 M4 demo/评测用例，并输出 JSONL 结果。

这个入口优先复用 local_demo 下已经跑通过的官方交互链路。纯文本/单图
仍可通过 --case-file 自定义，但正式质量结论应优先看 video/audio/benchmark。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_CASES = [
    {"id": "video_turntaking_valid", "kind": "demo", "mode": "turntaking-valid", "expect_contains": ["two", "2"]},
    {"id": "video_turntaking_interrupt", "kind": "demo", "mode": "turntaking-interrupt"},
    {"id": "video_turntaking_noise", "kind": "demo", "mode": "turntaking-noise"},
    {"id": "video_proactive", "kind": "demo", "mode": "proactive", "expect_contains": ["REQUIREMENT MEET"]},
    {"id": "audio_baseline_tts", "kind": "demo", "mode": "baseline-audio-tts"},
    {"id": "audio_turntaking_tts", "kind": "demo", "mode": "turntaking-audio-tts"},
]


def load_cases(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return DEFAULT_CASES
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("--case-file 必须是 JSON list")
    return data


def run_command(command: list[str], cwd: Path, env: dict[str, str], timeout: int) -> dict[str, Any]:
    start = time.time()
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return {
        "returncode": proc.returncode,
        "runtime_sec": round(time.time() - start, 3),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def judge(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    text = (result.get("stdout") or "") + "\n" + (result.get("stderr") or "")
    passed = result.get("returncode") == 0
    reasons: list[str] = []
    if not passed:
        reasons.append("命令退出码非 0")
    if case.get("expect_nonempty", True) and not text.strip():
        passed = False
        reasons.append("输出为空")
    contains = case.get("expect_contains") or []
    if contains and not any(token in text for token in contains):
        passed = False
        reasons.append(f"没有命中期望关键词: {contains}")
    return {"passed": passed, "reasons_zh": reasons}


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", args.cuda_visible_devices)
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_SHM_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("HF_HOME", str(args.repo_root / ".cache/huggingface"))
    env.setdefault("TRITON_CACHE_DIR", str(args.repo_root / ".cache/triton"))
    env.setdefault("MPLCONFIGDIR", str(args.repo_root / ".cache/matplotlib"))
    env.setdefault("PYTHONPATH", str(args.repo_root / "intersuit"))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["MODEL_PATH"] = args.model_path
    return env


def command_for_case(case: dict[str, Any], args: argparse.Namespace) -> list[str]:
    kind = case.get("kind", "demo")
    if kind == "demo":
        return ["bash", "scripts/run_demos_stable.sh", str(case["mode"])]
    if kind == "sanity":
        cmd = [
            args.python_bin,
            "scripts/eval_m4_sanity.py",
            "--model_path",
            args.model_path,
            "--prompt",
            str(case["prompt"]),
            "--max_new_tokens",
            str(case.get("max_new_tokens", 128)),
        ]
        if case.get("image_path"):
            cmd += ["--image_path", str(case["image_path"])]
        return cmd
    raise ValueError(f"未知 case kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 M4 demo/评测用例并输出 JSONL。")
    parser.add_argument("--model_path", default="checkpoints/M4-LongVA-7B-Qwen2")
    parser.add_argument("--case_file")
    parser.add_argument("--output_jsonl", default="train_logs/m4_eval_suite.jsonl")
    parser.add_argument("--cuda_visible_devices", default="0")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--python_bin", default="/home/yjm/miniconda3/envs/M4/bin/python")
    parser.add_argument("--repo_root", type=Path, default=Path("/home/yjm/M4-main"))
    args = parser.parse_args()

    intersuit_root = args.repo_root / "intersuit"
    output = intersuit_root / args.output_jsonl
    output.parent.mkdir(parents=True, exist_ok=True)
    env = build_env(args)
    cases = load_cases(args.case_file)

    with output.open("w", encoding="utf-8") as f:
        for case in cases:
            command = command_for_case(case, args)
            record: dict[str, Any] = {
                "id": case.get("id"),
                "kind": case.get("kind", "demo"),
                "mode": case.get("mode"),
                "model_path": args.model_path,
                "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
                "command": command,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            try:
                result = run_command(command, intersuit_root, env, args.timeout)
            except subprocess.TimeoutExpired as exc:
                result = {
                    "returncode": -1,
                    "runtime_sec": args.timeout,
                    "stdout": exc.stdout or "",
                    "stderr": (exc.stderr or "") + "\n命令超时。",
                }
            record.update(result)
            record.update(judge(case, result))
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps({k: record[k] for k in ("id", "passed", "runtime_sec", "returncode")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
