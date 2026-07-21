#!/usr/bin/env python
"""把启动器 gate-only 结果纳入阶段 2 最终报告。"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from new_dataset_common import load_json, sha256_file, write_json


LAUNCHER_PASS_MARKERS = (
    "训练 manifest 内容门禁：PASS",
    "AS-M4 BEATs 启动器内容门禁：PASS（仅检查，不启动训练）",
)


def finalize(
    gate_report: Path,
    train_audit: Path,
    launcher_log: Path,
    split_report: Path,
    build_summary: Path,
    dedup_report: Path,
) -> dict[str, Any]:
    gate = load_json(gate_report)
    audit = load_json(train_audit)
    build = load_json(build_summary)
    dedup = load_json(dedup_report)
    log = launcher_log.read_text(encoding="utf-8")
    if gate.get("status") != "PASS" or int(gate.get("error_count", -1)) != 0:
        raise ValueError("阶段 2 全量门禁不是 PASS")
    if not gate.get("full_decode") or gate.get("video_audio_decode_rate") != 1.0:
        raise ValueError("阶段 2 未完成 100% 全量媒体解码")
    if gate.get("scene_audio_path_valid_rate") != 1.0:
        raise ValueError("阶段 2 scene_audio_path 有效率不是 100%")
    if audit.get("status") != "PASS" or int(audit.get("error_count", -1)) != 0:
        raise ValueError("训练 manifest audit 不是 PASS")
    if build.get("status") != "PASS" or dedup.get("status") != "PASS":
        raise ValueError("阶段 2 构建或内容去重报告不是 PASS")
    for split in ("train", "dev", "reserve"):
        if build["manifest_sha256"].get(split) != gate["manifest_sha256"][split]["sha256"]:
            raise ValueError(f"{split} manifest 在构建与门禁之间 SHA256 不一致")
    if int(dedup.get("post_dedup_physical_media_count", 0)) < 1350:
        raise ValueError("内容去重后的物理媒体不足 1350")
    missing = [marker for marker in LAUNCHER_PASS_MARKERS if marker not in log]
    if missing:
        raise ValueError(f"启动器日志缺少 PASS 标记：{missing}")
    gate["launcher_content_gate"] = {
        "status": "PASS",
        "gate_only": True,
        "training_started": False,
        "log_path": str(launcher_log.resolve()),
        "log_sha256": sha256_file(launcher_log),
    }
    gate["build_evidence"] = {
        "status": "PASS",
        "build_summary_path": str(build_summary.resolve()),
        "build_summary_sha256": sha256_file(build_summary),
        "deduplication_report_path": str(dedup_report.resolve()),
        "deduplication_report_sha256": sha256_file(dedup_report),
        "pre_dedup_physical_media_count": dedup["pre_dedup_physical_media_count"],
        "post_dedup_physical_media_count": dedup["post_dedup_physical_media_count"],
        "dropped_group_count": dedup["dropped_group_count"],
        "input_sha256": build["input_sha256"],
    }
    gate["status"] = "PASS"
    gate["training_started"] = False
    write_json(gate_report, gate)
    counts = gate["split_physical_media_count"]
    qa_counts = gate["split_qa_count"]
    lines = [
        "# 阶段 2 正式训练集划分与门禁报告",
        "",
        "## 结论",
        "",
        "**PASS**",
        "",
        f"- 内容级去重：{dedup['pre_dedup_physical_media_count']} → "
        f"{dedup['post_dedup_physical_media_count']}，"
        f"丢弃物理组 {dedup['dropped_group_count']}",
        f"- 物理媒体 train/dev/reserve：{counts['train']}/{counts['dev']}/{counts['reserve']}",
        f"- QA train/dev/reserve：{qa_counts['train']}/{qa_counts['dev']}/{qa_counts['reserve']}",
        "- 三集合 video ID、YouTube ID、派生媒体 ID、物理组和媒体 SHA256 重叠：0",
        "- 冻结 300、历史 21、其他 dev/test 重叠：0",
        "- scene_audio_path 有效率：100%",
        "- 视频与音轨完整解码率：100%",
        "- sample_id 重复：0",
        "- Dataset 加载：PASS",
        "- DataCollator 组批：PASS",
        "- 启动器内容门禁：PASS（gate-only，未启动训练）",
        "",
        "## 分层结果",
        "",
    ]
    for split in ("train", "dev", "reserve"):
        tasks = gate["task_distribution"][split]
        sources = gate["source_distribution"][split]
        lines.extend([
            f"### {split}",
            "",
            f"- 任务 QA：audio={tasks.get('audio', 0)}，visual={tasks.get('visual', 0)}，"
            f"audio_visual={tasks.get('audio_visual', 0)}",
            "- 来源 QA：" + "，".join(f"{key}={value}" for key, value in sorted(sources.items())),
            "",
        ])
    lines.extend([
        "所有划分均在 `physical_media_id` 组级完成；同源 flip、裁剪和 suffix "
        "派生媒体没有跨集合。阶段 2 只生成正式 manifest 和门禁证据，阶段 3 训练尚未开始。",
    ])
    split_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--train-audit", type=Path, required=True)
    parser.add_argument("--launcher-log", type=Path, required=True)
    parser.add_argument("--split-report", type=Path, required=True)
    parser.add_argument("--build-summary", type=Path, required=True)
    parser.add_argument("--dedup-report", type=Path, required=True)
    args = parser.parse_args()
    report = finalize(
        args.gate_report.resolve(),
        args.train_audit.resolve(),
        args.launcher_log.resolve(),
        args.split_report.resolve(),
        args.build_summary.resolve(),
        args.dedup_report.resolve(),
    )
    print(f"阶段 2 最终门禁：{report['status']}")


if __name__ == "__main__":
    main()
