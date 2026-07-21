#!/usr/bin/env python
"""Build and validate an AS-M4 E0-E7 attribution matrix.

This runner is intentionally a planning harness first. It does not launch GPU
inference by default. Its job is to make sure the attribution experiments use
the same dataset split, scorer, output root and explicit AS-M4 rollback/audio
conditions before any expensive run starts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_IDS = ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7")


DEFAULT_EXPERIMENTS: list[dict[str, Any]] = [
    {
        "id": "E0",
        "description": "Original M4 baseline.",
        "model_key": "baseline_m4",
        "audio_condition": "none",
        "rollback_mode": "weights12k",
        "force_audio_gate": None,
        "alignment": "off",
        "gate_ablation": "none",
    },
    {
        "id": "E1",
        "description": "AS-M4 with silent or disabled scene audio.",
        "model_key": "as_m4",
        "audio_condition": "muted",
        "rollback_mode": "behavior",
        "force_audio_gate": "0",
        "alignment": "off",
        "gate_ablation": "none",
    },
    {
        "id": "E2",
        "description": "AS-M4 with correct synchronized scene audio.",
        "model_key": "as_m4",
        "audio_condition": "correct",
        "rollback_mode": "none",
        "force_audio_gate": None,
        "alignment": "on",
        "gate_ablation": "none",
    },
    {
        "id": "E3",
        "description": "AS-M4 with cross-video mismatched audio.",
        "model_key": "as_m4",
        "audio_condition": "mismatched",
        "rollback_mode": "none",
        "force_audio_gate": None,
        "alignment": "on",
        "gate_ablation": "none",
    },
    {
        "id": "E4",
        "description": "AS-M4 with noisy correct audio.",
        "model_key": "as_m4",
        "audio_condition": "noisy",
        "rollback_mode": "none",
        "force_audio_gate": None,
        "alignment": "on",
        "gate_ablation": "none",
    },
    {
        "id": "E5",
        "description": "AS-M4 with shifted audio and alignment disabled.",
        "model_key": "as_m4",
        "audio_condition": "shifted",
        "rollback_mode": "none",
        "force_audio_gate": None,
        "alignment": "off",
        "gate_ablation": "none",
    },
    {
        "id": "E6",
        "description": "AS-M4 with shifted audio and alignment enabled.",
        "model_key": "as_m4",
        "audio_condition": "shifted",
        "rollback_mode": "none",
        "force_audio_gate": None,
        "alignment": "on",
        "gate_ablation": "none",
    },
    {
        "id": "E7",
        "description": "AS-M4 gate ablation.",
        "model_key": "as_m4",
        "audio_condition": "correct",
        "rollback_mode": "gate0",
        "force_audio_gate": "0",
        "alignment": "on",
        "gate_ablation": "all",
    },
]


@dataclass(frozen=True)
class MatrixConfig:
    dataset_name: str
    split: str
    manifest: str
    manifest_sha256: str | None
    scorer: str
    baseline_model: str
    as_m4_model: str
    output_root: str
    experiments: list[dict[str, Any]]
    experiment_ids: tuple[str, ...]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError("config must be a JSON object")
    return data


def parse_config(data: dict[str, Any]) -> MatrixConfig:
    dataset = data.get("dataset") or {}
    models = data.get("models") or {}
    output_root = data.get("output_root")
    experiments = data.get("experiments") or DEFAULT_EXPERIMENTS
    experiment_ids = tuple(str(item) for item in (data.get("experiment_ids") or REQUIRED_IDS))

    required_dataset = ("name", "split", "manifest", "scorer")
    missing_dataset = [key for key in required_dataset if not dataset.get(key)]
    if missing_dataset:
        raise ValueError(f"dataset missing required keys: {missing_dataset}")
    missing_models = [key for key in ("baseline_m4", "as_m4") if not models.get(key)]
    if missing_models:
        raise ValueError(f"models missing required keys: {missing_models}")
    if not output_root:
        raise ValueError("output_root is required")
    if not isinstance(experiments, list):
        raise TypeError("experiments must be a list")

    return MatrixConfig(
        dataset_name=str(dataset["name"]),
        split=str(dataset["split"]),
        manifest=str(dataset["manifest"]),
        manifest_sha256=str(dataset["manifest_sha256"]) if dataset.get("manifest_sha256") else None,
        scorer=str(dataset["scorer"]),
        baseline_model=str(models["baseline_m4"]),
        as_m4_model=str(models["as_m4"]),
        output_root=str(output_root),
        experiments=experiments,
        experiment_ids=experiment_ids,
    )


def validate_matrix(config: MatrixConfig, strict_paths: bool = False) -> list[str]:
    errors: list[str] = []
    ids = [str(exp.get("id", "")) for exp in config.experiments]
    if ids != list(config.experiment_ids):
        errors.append(f"experiments must be exactly {list(config.experiment_ids)}, got {ids}")

    allowed_audio = {"none", "muted", "correct", "mismatched", "noisy", "shifted"}
    allowed_rollback = {"none", "behavior", "gate0", "weights12k", "weights32k"}
    allowed_alignment = {"on", "off"}
    allowed_gate_ablation = {"none", "quality", "relevance", "all"}
    for exp in config.experiments:
        exp_id = exp.get("id")
        if exp.get("audio_condition") not in allowed_audio:
            errors.append(f"{exp_id}: invalid audio_condition={exp.get('audio_condition')}")
        if exp.get("rollback_mode") not in allowed_rollback:
            errors.append(f"{exp_id}: invalid rollback_mode={exp.get('rollback_mode')}")
        if exp.get("alignment") not in allowed_alignment:
            errors.append(f"{exp_id}: invalid alignment={exp.get('alignment')}")
        if exp.get("gate_ablation") not in allowed_gate_ablation:
            errors.append(f"{exp_id}: invalid gate_ablation={exp.get('gate_ablation')}")
        if not exp.get("model_key"):
            errors.append(f"{exp_id}: model_key is required")

    e5 = next((exp for exp in config.experiments if exp.get("id") == "E5"), None)
    e6 = next((exp for exp in config.experiments if exp.get("id") == "E6"), None)
    if e5 and e6:
        if e5.get("audio_condition") != e6.get("audio_condition"):
            errors.append("E5 and E6 must use the same shifted audio condition")
        if e5.get("alignment") != "off" or e6.get("alignment") != "on":
            errors.append("E5 must disable alignment and E6 must enable alignment")

    if strict_paths:
        for label, value in (
            ("dataset.manifest", config.manifest),
            ("models.baseline_m4", config.baseline_model),
            ("models.as_m4", config.as_m4_model),
        ):
            if not Path(value).exists():
                errors.append(f"{label} path does not exist: {value}")
    return errors


def build_plan(config: MatrixConfig) -> list[dict[str, Any]]:
    model_paths = {
        "baseline_m4": config.baseline_model,
        "as_m4": config.as_m4_model,
    }
    records: list[dict[str, Any]] = []
    for exp in config.experiments:
        exp_id = str(exp["id"])
        model_key = str(exp["model_key"])
        output_jsonl = str(Path(config.output_root) / f"{exp_id.lower()}_predictions.jsonl")
        env = {
            "AS_M4_ROLLBACK_MODE": str(exp.get("rollback_mode") or "none"),
            "AS_M4_ENABLE_SCENE_AUDIO": "0" if exp.get("rollback_mode") in {"behavior", "weights12k", "weights32k"} else "1",
            "AS_M4_FUSION_INIT": str(exp.get("fusion_init") or "zero"),
            "AS_M4_GATE_LOGIT_BIAS": str(exp.get("gate_logit_bias") if exp.get("gate_logit_bias") is not None else -5.0),
        }
        for key, value in (exp.get("env") or {}).items():
            env.setdefault(str(key), str(value))
        if exp.get("force_audio_gate") is not None:
            env["AS_M4_FORCE_AUDIO_GATE"] = str(exp["force_audio_gate"])
        records.append(
            {
                "id": exp_id,
                "description": exp.get("description", ""),
                "dataset": config.dataset_name,
                "split": config.split,
                "manifest": config.manifest,
                "manifest_sha256": config.manifest_sha256,
                "scorer": config.scorer,
                "model_key": model_key,
                "model_path": model_paths.get(model_key, model_key),
                "audio_condition": exp.get("audio_condition"),
                "alignment": exp.get("alignment"),
                "gate_ablation": exp.get("gate_ablation"),
                "env": env,
                "output_jsonl": output_jsonl,
            }
        )
    return records


def write_outputs(plan: list[dict[str, Any]], output_dir: Path, validation_errors: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "matrix_plan.jsonl"
    with plan_path.open("w", encoding="utf-8") as f:
        for record in plan:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "status": "pass" if not validation_errors else "fail",
        "num_experiments": len(plan),
        "experiment_ids": [record["id"] for record in plan],
        "validation_errors": validation_errors,
        "plan_path": str(plan_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and materialize an AS-M4 E0-E7 attribution matrix.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default="intersuit/harness/artifacts/as_m4_e0_e7")
    parser.add_argument("--strict_paths", action="store_true", help="Require manifest and model paths to exist.")
    args = parser.parse_args()

    config = parse_config(load_config(Path(args.config)))
    errors = validate_matrix(config, strict_paths=args.strict_paths)
    plan = build_plan(config)
    write_outputs(plan, Path(args.output_dir), errors)
    print(json.dumps({"status": "pass" if not errors else "fail", "errors": errors, "num_experiments": len(plan)}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
