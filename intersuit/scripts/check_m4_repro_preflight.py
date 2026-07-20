#!/usr/bin/env python
"""M4 复现训练前检查。

检查目标：GPU 可见性、稳定四卡是否空闲、功率限制是否为已验证策略、
以及数据审计报告是否存在。脚本不修改系统状态。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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


def validate_data_audit(audit: dict) -> tuple[bool, dict]:
    """Validate either the legacy M4 audit or the Stage 2 manifest gate."""

    if "missing_image_count" in audit or "missing_audio_count" in audit:
        details = {
            "audit_format": "legacy_m4",
            "audit_status": audit.get("status"),
            "missing_image_count": audit.get("missing_image_count"),
            "missing_audio_count": audit.get("missing_audio_count"),
        }
        passed = (
            audit.get("missing_image_count") == 0
            and audit.get("missing_audio_count") == 0
        )
        return passed, details

    if "scene_audio_path_valid_rate" in audit or "video_audio_decode_rate" in audit:
        details = {
            "audit_format": "stage2_gate",
            "audit_status": audit.get("status"),
            "error_count": audit.get("error_count"),
            "full_decode": audit.get("full_decode"),
            "scene_audio_path_valid_rate": audit.get("scene_audio_path_valid_rate"),
            "video_audio_decode_rate": audit.get("video_audio_decode_rate"),
        }
        passed = (
            audit.get("status") == "PASS"
            and audit.get("error_count") == 0
            and audit.get("full_decode") is True
            and audit.get("scene_audio_path_valid_rate") == 1.0
            and audit.get("video_audio_decode_rate") == 1.0
        )
        return passed, details

    return False, {
        "audit_format": "unknown",
        "audit_status": audit.get("status"),
        "reason_zh": "数据审计格式无法识别",
    }


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
    parser.add_argument(
        "--expected_nccl_socket_ifname",
        default="",
        help="如提供，则要求当前 NCCL_SOCKET_IFNAME 与该值一致；AS-M4 单机四卡默认建议 lo。",
    )
    parser.add_argument(
        "--allow_existing_train_processes",
        action="store_true",
        help="允许已有 train_mem.py 训练进程存在。默认不允许，用于避免 timeout 残留 rank 污染后续 NCCL。",
    )
    parser.add_argument(
        "--train_process_pattern",
        default="intersuit/train/train_mem.py",
        help="用于检测残留训练进程的命令行子串。",
    )
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

    expected_ifname = args.expected_nccl_socket_ifname.strip()
    actual_ifname = os.environ.get("NCCL_SOCKET_IFNAME", "").strip()
    if expected_ifname:
        passed = actual_ifname == expected_ifname
        result["checks"].append(
            {
                "name": "nccl_socket_ifname",
                "passed": passed,
                "expected": expected_ifname,
                "actual": actual_ifname,
                "reason_zh": "" if passed else "NCCL_SOCKET_IFNAME 与预期接口不一致，可能导致单机训练走到错误网卡。",
            }
        )
        if not passed:
            result["status"] = "fail"
    else:
        result["checks"].append(
            {
                "name": "nccl_socket_ifname",
                "passed": True,
                "expected": "",
                "actual": actual_ifname,
                "reason_zh": "未要求固定 NCCL_SOCKET_IFNAME。",
            }
        )

    ps = run(["ps", "-eo", "pid=,ppid=,stat=,cmd="])
    if ps.returncode == 0:
        current_pid = os.getpid()
        stale_processes = []
        for line in ps.stdout.splitlines():
            parts = line.strip().split(None, 3)
            if len(parts) < 4:
                continue
            pid_text, ppid_text, stat, cmd = parts
            try:
                pid = int(pid_text)
                ppid = int(ppid_text)
            except ValueError:
                continue
            if pid == current_pid:
                continue
            if args.train_process_pattern in cmd:
                stale_processes.append({"pid": pid, "ppid": ppid, "stat": stat, "cmd": cmd})
        passed = args.allow_existing_train_processes or not stale_processes
        result["checks"].append(
            {
                "name": "existing_train_processes",
                "passed": passed,
                "pattern": args.train_process_pattern,
                "allow_existing": args.allow_existing_train_processes,
                "processes": stale_processes,
            }
        )
        if not passed:
            result["status"] = "fail"
    else:
        result["checks"].append(
            {
                "name": "existing_train_processes",
                "passed": False,
                "stderr": ps.stderr.strip(),
            }
        )
        result["status"] = "fail"

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
        audit_ok, audit_details = validate_data_audit(audit)
        result["checks"].append(
            {
                "name": "data_audit",
                "passed": audit_ok,
                **audit_details,
            }
        )
        if not audit_ok:
            result["status"] = "fail"
    else:
        result["status"] = "fail"
        result["checks"].append({"name": "data_audit", "passed": False, "reason_zh": "没有找到数据审计报告"})

    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
