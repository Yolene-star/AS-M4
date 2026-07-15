#!/usr/bin/env python
"""M4 复现训练前检查。

检查目标：GPU 可见性、稳定四卡是否空闲、功率限制是否为已验证策略、
以及数据审计报告是否存在。脚本不修改系统状态。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def parse_csv(text: str) -> list[dict[str, str]]:
    lines = [line for line in text.splitlines() if line.strip()]
    reader = csv.DictReader(lines)
    rows: list[dict[str, str]] = []
    for row in reader:
        rows.append({str(key).strip(): str(value).strip() for key, value in row.items()})
    return rows


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute() or path.exists():
        return path
    script_root = Path(__file__).resolve().parents[1]
    script_relative = script_root / path
    if script_relative.exists():
        return script_relative
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 M4 复现训练前置条件。")
    parser.add_argument("--stable_gpus", default="0,2,3,4")
    parser.add_argument(
        "--allowed_power_limits",
        default="300.00 W,450.00 W",
        help="逗号分隔的允许功率限制；300W 是保守默认，450W 已验证可用。",
    )
    parser.add_argument("--max_idle_memory_mib", type=int, default=500)
    parser.add_argument("--audit_json", default="train_logs/m4_data_audit_generated_audio.json")
    args = parser.parse_args()

    stable = {gpu.strip() for gpu in args.stable_gpus.split(",") if gpu.strip()}
    allowed_power_limits = {
        value.strip()
        for value in args.allowed_power_limits.split(",")
        if value.strip()
    }
    result = {
        "status": "pass",
        "stable_gpus": sorted(stable),
        "allowed_power_limits": sorted(allowed_power_limits),
        "checks": [],
    }

    smi = run([
        "nvidia-smi",
        "--query-gpu=index,pci.bus_id,power.limit,memory.used,memory.total,utilization.gpu",
        "--format=csv",
    ])
    if smi.returncode != 0:
        result["status"] = "fail"
        result["checks"].append({"name": "nvidia_smi", "passed": False, "stderr": smi.stderr.strip()})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    rows = parse_csv(smi.stdout)
    gpu_failures = []
    for row in rows:
        idx = row["index"].strip()
        if idx not in stable:
            continue
        power = row["power.limit [W]"].strip()
        used = int(row["memory.used [MiB]"].split()[0])
        if power not in allowed_power_limits:
            gpu_failures.append(
                f"GPU {idx} 功率限制是 {power}，不在允许列表 {sorted(allowed_power_limits)} 中"
            )
        if used > args.max_idle_memory_mib:
            gpu_failures.append(f"GPU {idx} 显存占用 {used} MiB，超过空闲阈值 {args.max_idle_memory_mib} MiB")

    result["checks"].append({
        "name": "stable_gpu_idle_and_power",
        "passed": not gpu_failures,
        "failures_zh": gpu_failures,
        "gpu_rows": rows,
    })
    if gpu_failures:
        result["status"] = "fail"

    audit_path = resolve_path(args.audit_json)
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit_ok = audit.get("missing_image_count") == 0 and audit.get("missing_audio_count") == 0
        result["checks"].append({
            "name": "data_audit",
            "passed": audit_ok,
            "audit_status": audit.get("status"),
            "missing_image_count": audit.get("missing_image_count"),
            "missing_audio_count": audit.get("missing_audio_count"),
        })
        if not audit_ok:
            result["status"] = "fail"
    else:
        result["status"] = "fail"
        result["checks"].append({"name": "data_audit", "passed": False, "reason_zh": "没有找到数据审计报告"})

    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
