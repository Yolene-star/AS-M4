"""CPU tests for AVUT projector evaluation and leave-one-video-out validation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_avut_audio_video_projectors.py"
TRAIN_SPEC = importlib.util.spec_from_file_location("train_avut_audio_video_projectors", TRAIN_SCRIPT)
train_projectors = importlib.util.module_from_spec(TRAIN_SPEC)
sys.modules[TRAIN_SPEC.name] = train_projectors
TRAIN_SPEC.loader.exec_module(train_projectors)

EVAL_SCRIPT = ROOT / "scripts" / "evaluate_avut_audio_video_projectors.py"
EVAL_SPEC = importlib.util.spec_from_file_location("evaluate_avut_audio_video_projectors", EVAL_SCRIPT)
evaluate = importlib.util.module_from_spec(EVAL_SPEC)
sys.modules[EVAL_SPEC.name] = evaluate
EVAL_SPEC.loader.exec_module(evaluate)


def _write_feature(path: Path, sample_id: str, condition: str, features: torch.Tensor, timestamps: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sample_id": sample_id,
            "condition": condition,
            "timestamps": timestamps,
            "audio_embedding": features,
            "metadata": {"source_sample_id": sample_id},
        },
        path,
    )


def _make_pair_manifest(tmp_path: Path) -> Path:
    rows = []
    timestamps = torch.tensor([[0.0, 1.0], [0.5, 1.5]], dtype=torch.float32)
    for sample_idx, sample_id in enumerate(("s0", "s1")):
        video_path = tmp_path / "video" / f"{sample_id}.pt"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video = torch.eye(4)[:2] + sample_idx
        torch.save({"video_features": video, "timestamps": timestamps}, video_path)
        for condition in evaluate.CONDITION_ORDER:
            audio_path = tmp_path / "audio" / sample_id / f"{condition}.pt"
            audio = video.clone() if condition == "original" else torch.zeros_like(video)
            _write_feature(audio_path, sample_id, condition, audio, timestamps)
            rows.append(
                {
                    "sample_id": sample_id,
                    "pair_type": "positive" if condition == "original" else condition,
                    "audio_condition": condition,
                    "label": 1 if condition == "original" else 0,
                    "audio_feature_path": str(audio_path),
                    "video_feature_path": str(video_path),
                    "window_count": 2,
                    "embedding_dim": 4,
                    "timestamp_start": 0.0,
                    "timestamp_end": 1.5,
                }
            )
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return manifest


def test_summarize_scores_identifies_original_rank_and_margin():
    rows = [
        {"sample_id": "s0", "condition": "original", "mean_logit": 3.0, "finite": True, "saturation_fraction": 0.0},
        {"sample_id": "s0", "condition": "silence", "mean_logit": -1.0, "finite": True, "saturation_fraction": 0.0},
        {"sample_id": "s0", "condition": "wrong_audio", "mean_logit": 0.5, "finite": True, "saturation_fraction": 0.0},
        {"sample_id": "s0", "condition": "shift_plus_0_5", "mean_logit": 1.0, "finite": True, "saturation_fraction": 0.0},
        {"sample_id": "s0", "condition": "shift_minus_0_5", "mean_logit": 2.0, "finite": True, "saturation_fraction": 0.0},
    ]

    summary = evaluate.summarize_scores(rows)

    assert summary["criteria_counts"]["original_ranked_first"] == 1
    assert summary["criteria_counts"]["original_gt_wrong_audio"] == 1
    assert summary["sample_summaries"][0]["original_margin"] == 1.0
    assert summary["all_scores_finite"] is True


def test_evaluate_checkpoint_and_leave_one_out(tmp_path):
    manifest = _make_pair_manifest(tmp_path)
    model = train_projectors.AVProjector(input_dim=4, project_dim=2)
    checkpoint = tmp_path / "projector.pt"
    torch.save(
        {
            "audio_proj.weight": model.audio_proj.weight.detach(),
            "video_proj.weight": model.video_proj.weight.detach(),
        },
        checkpoint,
    )

    train_eval = evaluate.evaluate_checkpoint(manifest, checkpoint)
    loocv = evaluate.leave_one_video_out(
        manifest,
        output_root=tmp_path / "eval",
        steps=2,
        project_dim=2,
        lr=1e-3,
        seed=1,
    )

    assert train_eval["projector_frozen"] is True
    assert train_eval["projector_shapes"]["audio_proj"] == [2, 4]
    assert loocv["fold_count"] == 2
    assert loocv["no_video_leakage"] is True
    assert loocv["all_scores_finite"] is True
    assert all(Path(fold["checkpoint_path"]).is_file() for fold in loocv["folds"])


def test_decide_route_reports_train_failure():
    train_eval = {"criteria_counts": {"original_gt_wrong_audio": 0, "original_gt_silence": 0, "original_gt_shift_plus_0_5": 0, "original_gt_shift_minus_0_5": 0}, "sample_count": 1}
    loocv = {"aggregate_criteria_counts": {}, "fold_count": 1}

    assert evaluate.decide_route(train_eval, loocv).startswith("C_train_failed")
