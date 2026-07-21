from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_m4_repro_preflight.py"
SPEC = importlib.util.spec_from_file_location("check_m4_repro_preflight", SCRIPT)
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def test_legacy_audit_requires_zero_missing_assets():
    passed, details = preflight.validate_data_audit(
        {"status": "pass", "missing_image_count": 0, "missing_audio_count": 0}
    )
    assert passed is True
    assert details["audit_format"] == "legacy_m4"

    passed, _ = preflight.validate_data_audit(
        {"status": "fail", "missing_image_count": 1, "missing_audio_count": 0}
    )
    assert passed is False


def test_stage2_gate_requires_complete_valid_media():
    audit = {
        "status": "PASS",
        "error_count": 0,
        "full_decode": True,
        "scene_audio_path_valid_rate": 1.0,
        "video_audio_decode_rate": 1.0,
    }
    passed, details = preflight.validate_data_audit(audit)
    assert passed is True
    assert details["audit_format"] == "stage2_gate"

    for key, bad_value in (
        ("status", "FAIL"),
        ("error_count", 1),
        ("full_decode", False),
        ("scene_audio_path_valid_rate", 0.99),
        ("video_audio_decode_rate", 0.99),
    ):
        invalid = dict(audit)
        invalid[key] = bad_value
        passed, _ = preflight.validate_data_audit(invalid)
        assert passed is False


def test_unknown_audit_format_is_rejected():
    passed, details = preflight.validate_data_audit({"status": "PASS"})
    assert passed is False
    assert details["audit_format"] == "unknown"
