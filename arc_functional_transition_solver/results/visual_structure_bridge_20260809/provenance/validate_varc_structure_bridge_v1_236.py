"""Validate and freeze the query-blind VARC run before any gold transfer."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from collections.abc import Mapping
from pathlib import Path


ROOT = Path("/home/wangmeiqi/codex_runs/arc_visual_posterior_bridge_20260809")
VARC = Path(
    "/home/wangmeiqi/codex_runs/arc_visual_provider_probe_20260807/external/VARC"
)
VALIDATOR_SOURCE = ROOT / "validator_src"
sys.path.insert(0, str(VALIDATOR_SOURCE))

from afts_arc.experiment_safety import atomic_write_json, canonical_sha256  # noqa: E402
from afts_arc.visual_provider_gate import (  # noqa: E402
    COHORT_SCHEMA,
    _file_sha256,
    _load_json_object,
    _load_provider_predictions,
    _validate_provider_predictions,
)


MANIFEST_PATH = ROOT / "cohort_v1" / "manifest.json"
BLIND_ROOT = ROOT / "cohort_v1" / "blind_data" / "ARC-AGI-2"
RUN_ROOT = ROOT / "runs" / "varc_structure_bridge_v1"
PREDICTION_ROOT = VARC / "outputs" / "query_blind_structure_v1_attempt_0"
OUTPUT_PATH = ROOT / "provider_raw_validation_structure_v1.json"
CHECKPOINT = VARC / "saves" / "offline_train_ViT" / "checkpoint_best.pt"
RUNNER = ROOT / "run_varc_structure_bridge_task_v1_236.sh"
LAUNCHER = ROOT / "launch_varc_structure_bridge_v1_236.sh"


def _sentinel_audit() -> dict[str, int]:
    provider_files = tuple(
        sorted((BLIND_ROOT / "data" / "evaluation").glob("*.json"))
    ) + tuple(
        sorted((BLIND_ROOT / "data" / "eval_color_permute_ttt_9").glob("*/*.json"))
    )
    mismatch_count = 0
    query_count = 0
    for path in provider_files:
        payload = _load_json_object(path)
        tests = payload["test"]
        if not isinstance(tests, list):
            raise TypeError(f"test field is not a list: {path}")
        for query in tests:
            if not isinstance(query, Mapping):
                raise TypeError(f"query is not an object: {path}")
            query_count += 1
            mismatch_count += query["output"] != query["input"]
    return {
        "provider_json_file_count": len(provider_files),
        "sentinel_query_count": query_count,
        "sentinel_mismatch_count": mismatch_count,
    }


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"validation artifact already exists: {OUTPUT_PATH}")
    manifest = _load_json_object(MANIFEST_PATH)
    if manifest["schema"] != COHORT_SCHEMA:
        raise ValueError("cohort schema mismatch")
    task_ids = tuple(record["task_id"] for record in manifest["tasks"])
    expected_predictions = {
        f"{task_id}_predictions.json" for task_id in task_ids
    }
    actual_predictions = {
        path.name for path in PREDICTION_ROOT.glob("*_predictions.json")
    }
    if actual_predictions != expected_predictions:
        raise ValueError("prediction file set differs from the frozen cohort")

    statuses = []
    for task_id in task_ids:
        status = _load_json_object(RUN_ROOT / "tasks" / task_id / "status.json")
        if status["task_id"] != task_id:
            raise ValueError(f"status task ID mismatch: {task_id}")
        statuses.append(status)
    if any(status["exit_code"] != 0 for status in statuses):
        raise ValueError("one or more provider tasks failed")
    for gpu_id in (0, 1):
        complete = _load_json_object(RUN_ROOT / f"gpu{gpu_id}" / "complete.json")
        if complete != {"gpu_id": gpu_id, "complete": True}:
            raise ValueError(f"worker completion record mismatch: GPU {gpu_id}")

    audit = _sentinel_audit()
    if audit["provider_json_file_count"] != 624:
        raise ValueError("unexpected query-blind provider file count")
    if audit["sentinel_mismatch_count"] != 0:
        raise ValueError("query-blind sentinel mismatch")
    if any(path.name == "gold_data" for path in ROOT.rglob("gold_data")):
        raise ValueError("gold_data path is present in the provider root")

    raw_predictions, prediction_files = _load_provider_predictions(
        task_ids, (PREDICTION_ROOT,)
    )
    _, prediction_validation = _validate_provider_predictions(
        raw_predictions,
        invalid_candidate_policy="reject",
    )
    executed_source = {
        name: _file_sha256(VARC / name)
        for name in (
            "src/ARC_ViT.py",
            "src/ARC_loader.py",
            "test_time_train_ARC.py",
        )
    }
    packages = {
        name: importlib.metadata.version(name)
        for name in ("diffusers", "einops", "timm", "torch", "torchvision")
    }
    content = {
        "schema": "afts.visual-provider-raw-validation/v1",
        "cohort_id": manifest["cohort_id"],
        "cohort_manifest_sha256": _file_sha256(MANIFEST_PATH),
        "gold_accessed": False,
        "predictions": {
            "file_count": len(prediction_files),
            "files": prediction_files,
            **prediction_validation,
        },
        "query_blind_audit": audit,
        "run": {
            "task_count": len(statuses),
            "failure_count": 0,
            "gpu_seconds": sum(status["elapsed_seconds"] for status in statuses),
            "max_task_seconds": max(status["elapsed_seconds"] for status in statuses),
            "tasks": statuses,
        },
        "runtime": {
            "packages": packages,
            "platform": platform.platform(),
            "python": sys.version,
        },
        "source": {
            "checkpoint_sha256": _file_sha256(CHECKPOINT),
            "executed_source_sha256": executed_source,
            "launcher_sha256": _file_sha256(LAUNCHER),
            "runner_sha256": _file_sha256(RUNNER),
            "varc_commit": "bd478ecf362e6499a988b05f33223e5c5fc6a6be",
        },
    }
    artifact = {"validation_id": canonical_sha256(content), **content}
    atomic_write_json(OUTPUT_PATH, artifact)
    print(
        json.dumps(
            {
                "validation_id": artifact["validation_id"],
                "task_count": len(statuses),
                "raw_candidate_count": prediction_validation[
                    "raw_candidate_count"
                ],
                "rejected_candidate_count": prediction_validation[
                    "rejected_candidate_count"
                ],
                "sentinel_mismatch_count": audit["sentinel_mismatch_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
