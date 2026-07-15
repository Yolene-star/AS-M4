#!/usr/bin/env python3
"""下载 OmniMMI 指定子任务缺失的视频或 clips。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import time
from pathlib import Path


TASK_FILES = {
    "ap": "action_prediction.json",
    "sg": "dynamic_state_grounding.json",
    "md": "multiturn_dependency_reasoning.json",
    "si": "speaker_identification.json",
    "pa": "proactive_alerting.json",
    "pt": "proactive_turntaking.json",
}


def collect_assets(root: Path, tasks: list[str], kind: str) -> list[str]:
    assets: set[str] = set()
    for task in tasks:
        data = json.loads((root / TASK_FILES[task]).read_text())
        if kind == "videos":
            assets.update(sample["video"] for sample in data)
        elif kind == "clips":
            if task not in {"sg", "md"}:
                continue
            for sample in data:
                base = sample["video"].split(".mp4")[0]
                for qa in sample["qa"]:
                    _, end = map(float, qa["timestamp"].split("--"))
                    assets.add(f"{base}_{int(end)}.mp4")
        else:
            raise ValueError(f"unsupported kind: {kind}")
    return sorted(assets)


def download_one(url: str, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    for attempt in range(1, 6):
        cmd = [
            "curl",
            "-L",
            "--http1.1",
            "--silent",
            "--show-error",
            "--retry",
            "10",
            "--retry-delay",
            "2",
            "--retry-max-time",
            "900",
            "--connect-timeout",
            "30",
            "-C",
            "-",
            "-o",
            str(tmp),
            url,
        ]
        proc = subprocess.run(cmd)
        if proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(out)
            return True
        if attempt < 5:
            time.sleep(3 * attempt)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 OmniMMI 缺失资源")
    parser.add_argument("--root", default="/home/yjm/M4-main/third_party/OmniMMI/omnimmi")
    parser.add_argument("--tasks", nargs="+", required=True, choices=sorted(TASK_FILES))
    parser.add_argument("--kind", choices=["videos", "clips"], required=True)
    parser.add_argument("--jobs", type=int, default=1, help="并发下载数")
    args = parser.parse_args()

    root = Path(args.root)
    target_dir = root / args.kind
    # ColorfulAI/OmniMMI 会重定向到 bigai-nlco/OmniMMI；直接使用目标仓库可减少一次跳转。
    base_url = f"https://huggingface.co/datasets/bigai-nlco/OmniMMI/resolve/main/{args.kind}"
    assets = collect_assets(root, args.tasks, args.kind)

    missing = [
        asset
        for asset in assets
        if not (target_dir / asset).exists() or (target_dir / asset).stat().st_size == 0
    ]
    print(
        f"任务={','.join(args.tasks)} 类型={args.kind} 总数={len(assets)} 缺失={len(missing)}",
        flush=True,
    )

    def run_one(item: tuple[int, str]) -> tuple[str, bool]:
        idx, asset = item
        out = target_dir / asset
        print(f"[{idx}/{len(missing)}] 下载 {asset}", flush=True)
        ok = download_one(f"{base_url}/{asset}", out)
        if not ok:
            print(f"失败：{asset}", flush=True)
        return asset, ok

    failed: list[str] = []
    if args.jobs <= 1:
        for item in enumerate(missing, 1):
            asset, ok = run_one(item)
            if not ok:
                failed.append(asset)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(run_one, item) for item in enumerate(missing, 1)]
            for future in as_completed(futures):
                asset, ok = future.result()
                if not ok:
                    failed.append(asset)

    print(f"完成：失败数量={len(failed)}", flush=True)
    if failed:
        print("\n".join(failed[:100]), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
