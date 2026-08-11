"""Validate a query-blind VARC cohort run and close its native-cost receipt."""

from __future__ import annotations

from collections.abc import Mapping

from afts_arc.experiment_safety import canonical_sha256
from afts_arc.varc_blind import VARC_BLIND_SCHEMA
from afts_arc.varc_diagnostic_compat import validate_varc_diagnostic_compatibility


VARC_RUN_RECEIPT_SCHEMA = "afts.varc-query-blind-run-receipt/v2"
VARC_TTT_CONFIG: dict[str, object] = {
    "architecture": "vit",
    "batch_size": 8,
    "depth": 10,
    "embed_dim": 512,
    "epoch_argument": 100,
    "executed_epoch_count": 101,
    "image_size": 64,
    "learning_rate": 0.0003,
    "lr_scheduler": "cosine",
    "num_attempts": 10,
    "num_colors": 12,
    "num_heads": 8,
    "patch_size": 2,
    "per_task_timeout_seconds": 2400,
    "resume_skip_task_token": True,
    "seed": 42,
    "ttt_num_each": 1,
    "weight_decay": 0.0,
}


def _object(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field} keys must be strings")
    return value


def build_varc_run_receipt(
    *,
    blind_manifest: Mapping[str, object],
    task_statuses: Mapping[str, object],
    prediction_files: Mapping[str, object],
    source: Mapping[str, object],
    runtime: Mapping[str, object],
    gpu_ids: tuple[int, ...],
    compatibility_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Close a run receipt without reading query solutions."""

    if blind_manifest["schema"] != VARC_BLIND_SCHEMA:
        raise ValueError("unsupported VARC blind-manifest schema")
    records = blind_manifest["tasks"]
    if not isinstance(records, list) or not records:
        raise ValueError("blind manifest task records are malformed")
    task_ids = tuple(record["task_id"] for record in records)
    compatibility_task_ids = validate_varc_diagnostic_compatibility(
        compatibility_manifest
    )
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("blind manifest contains duplicate task IDs")
    if set(task_ids) != set(task_statuses) or set(task_ids) != set(prediction_files):
        raise ValueError("manifest, status, and prediction task sets differ")
    if set(task_ids) != set(compatibility_task_ids):
        raise ValueError("blind manifest and diagnostic task sets differ")
    if (
        compatibility_manifest["blind_cohort_id"]
        != blind_manifest["blind_cohort_id"]
    ):
        raise ValueError("diagnostic overlay belongs to a different blind cohort")
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("gpu_ids must be non-empty and unique")
    if any(
        isinstance(gpu_id, bool) or not isinstance(gpu_id, int) for gpu_id in gpu_ids
    ):
        raise TypeError("gpu_ids must be integers")
    if not source or not runtime:
        raise ValueError("source and runtime receipts must not be empty")
    status_rows: list[dict[str, object]] = []
    gpu_seconds = 0
    max_task_seconds = 0
    for task_id in sorted(task_ids):
        status = _object(task_statuses[task_id], field=f"status[{task_id}]")
        if set(status) != {
            "elapsed_seconds",
            "end_epoch",
            "exit_code",
            "gpu_id",
            "start_epoch",
            "task_id",
        }:
            raise ValueError(f"status {task_id} has missing or unknown fields")
        if status["task_id"] != task_id:
            raise ValueError(f"status task_id differs for {task_id}")
        for field in (
            "elapsed_seconds",
            "end_epoch",
            "exit_code",
            "gpu_id",
            "start_epoch",
        ):
            if isinstance(status[field], bool) or not isinstance(status[field], int):
                raise TypeError(f"status {task_id}.{field} must be an integer")
        if status["exit_code"] != 0:
            raise ValueError(f"VARC task failed: {task_id}")
        if status["gpu_id"] not in gpu_ids:
            raise ValueError(f"status uses an undeclared GPU: {task_id}")
        if status["end_epoch"] - status["start_epoch"] != status["elapsed_seconds"]:
            raise ValueError(f"status elapsed time is inconsistent: {task_id}")
        if status["elapsed_seconds"] < 0:
            raise ValueError(f"status elapsed time is negative: {task_id}")
        prediction = _object(
            prediction_files[task_id], field=f"prediction_files[{task_id}]"
        )
        if set(prediction) != {"bytes", "sha256"}:
            raise ValueError(
                f"prediction receipt has missing or unknown fields: {task_id}"
            )
        if (
            isinstance(prediction["bytes"], bool)
            or not isinstance(prediction["bytes"], int)
            or prediction["bytes"] <= 0
        ):
            raise ValueError(f"prediction artifact is empty or malformed: {task_id}")
        if not isinstance(prediction["sha256"], str) or not prediction["sha256"]:
            raise ValueError(f"prediction SHA-256 is malformed: {task_id}")
        elapsed = int(status["elapsed_seconds"])
        gpu_seconds += elapsed
        max_task_seconds = max(max_task_seconds, elapsed)
        status_rows.append(
            {
                **dict(status),
                "prediction_bytes": prediction["bytes"],
                "prediction_sha256": prediction["sha256"],
            }
        )
    content: dict[str, object] = {
        "blind_cohort_id": blind_manifest["blind_cohort_id"],
        "compute": {
            "failure_count": 0,
            "gpu_ids": list(gpu_ids),
            "gpu_seconds": gpu_seconds,
            "max_task_seconds": max_task_seconds,
            "task_count": len(task_ids),
        },
        "controller_training_started": False,
        "query_blind_protocol": {
            "candidate_logits_consume": [
                "attention_mask",
                "inputs",
                "task_ids",
            ],
            "provider_visible_test_output": blind_manifest[
                "provider_visible_test_output"
            ],
            "query_gold_read": False,
            "test_output_role": "input-derived sentinel for loader shape only",
            "training_examples": "demonstrations only",
            "candidate_path_uses_original_task_and_variant_names": True,
            "diagnostic_alias_role": (
                "post-candidate upstream sentinel diagnostic only"
            ),
            "single_gpu_diagnostic_alias_serialization": True,
            "upstream_logged_score": "sentinel identity diagnostic; not ARC accuracy",
        },
        "runtime": dict(runtime),
        "runtime_diagnostic_compatibility_id": compatibility_manifest[
            "compatibility_id"
        ],
        "schema": VARC_RUN_RECEIPT_SCHEMA,
        "scientific_scope": "static visual candidate distribution",
        "source": dict(source),
        "task_statuses": status_rows,
        "test_time_configuration": dict(VARC_TTT_CONFIG),
    }
    return {"receipt_id": canonical_sha256(content), **content}
