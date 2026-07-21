"""三类冻结音视频评测集构建逻辑的 CPU 测试。"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/build_frozen_av_task_eval.py"
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("build_frozen_av_task_eval", SCRIPT_PATH)
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def test_balanced_unique_keeps_video_ids_isolated():
    rows = [
        {"id": f"a{index}", "youtube_id": f"video{index}", "task_type": "A"}
        for index in range(4)
    ] + [
        {"id": f"b{index}", "youtube_id": f"other{index}", "task_type": "B"}
        for index in range(4)
    ]

    selected = builder.balanced_unique(rows, 4, "task_type", {"video0"})

    assert len(selected) == 4
    assert len({row["youtube_id"] for row in selected}) == 4
    assert "video0" not in {row["youtube_id"] for row in selected}
    assert {row["task_type"] for row in selected} == {"A", "B"}


def test_time_alignment_sensitive_regex():
    assert builder.TIME_RE.search("When does the audio say hello?")
    assert builder.TIME_RE.search("At which period is the word first heard?")
    assert not builder.TIME_RE.search("Which brand of headphones is featured in the audio?")


def test_refuses_category_size_outside_frozen_range(tmp_path):
    args = argparse.Namespace(per_category=99)
    with pytest.raises(ValueError, match="100～200"):
        builder.run(args)
