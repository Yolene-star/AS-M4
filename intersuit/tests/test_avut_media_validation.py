import json
import subprocess
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from validate_avut_media import validate_video  # noqa: E402


def test_video_missing(tmp_path):
    result = validate_video(tmp_path / "missing.mp4")
    assert result["valid"] is False
    assert result["video_exists"] is False


def test_video_without_audio(tmp_path):
    path = tmp_path / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=16x16:rate=1:duration=1", "-an", "-y", str(path)],
        check=True,
    )
    result = validate_video(path)
    assert result["has_video_stream"] is True
    assert result["has_audio_stream"] is False
    assert result["valid"] is False
