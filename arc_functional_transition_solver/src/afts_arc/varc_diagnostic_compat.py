"""Isolate VARC's underscore-sensitive diagnostics from candidate computation."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from afts_arc.experiment_safety import canonical_sha256, file_sha256
from afts_arc.varc_blind import VARC_BLIND_SCHEMA


VARC_DIAGNOSTIC_COMPAT_SCHEMA = "afts.varc-diagnostic-compatibility/v1"
VARC_DIAGNOSTIC_ALIAS_RECORD_SCHEMA = "afts.varc-diagnostic-alias-record/v1"
_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9_-]{1,200}$")


def _object(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field} keys must be strings")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be an array")
    return value


def validate_varc_diagnostic_compatibility(
    compatibility_manifest: Mapping[str, object],
) -> tuple[str, ...]:
    if set(compatibility_manifest) != {
        "blind_cohort_id",
        "compatibility_id",
        "schema",
        "tasks",
    }:
        raise ValueError("diagnostic compatibility has missing or unknown fields")
    if compatibility_manifest["schema"] != VARC_DIAGNOSTIC_COMPAT_SCHEMA:
        raise ValueError("unsupported VARC diagnostic compatibility schema")
    body = dict(compatibility_manifest)
    declared_id = body.pop("compatibility_id")
    if declared_id != canonical_sha256(body):
        raise ValueError("diagnostic compatibility ID does not match content")
    task_ids: list[str] = []
    for raw_record in _sequence(
        compatibility_manifest["tasks"], field="compatibility tasks"
    ):
        record = _object(raw_record, field="compatibility task")
        if set(record) != {
            "augmentation_count",
            "augmentation_set_sha256",
            "evaluation_sha256",
            "task_id",
        }:
            raise ValueError("compatibility task has missing or unknown fields")
        task_id = record["task_id"]
        if (
            not isinstance(task_id, str)
            or _SAFE_TASK_ID.fullmatch(task_id) is None
            or task_id in task_ids
        ):
            raise ValueError("compatibility task ID is invalid or duplicated")
        augmentation_count = record["augmentation_count"]
        if (
            isinstance(augmentation_count, bool)
            or not isinstance(augmentation_count, int)
            or augmentation_count <= 0
        ):
            raise ValueError("compatibility augmentation count must be positive")
        for field in ("augmentation_set_sha256", "evaluation_sha256"):
            digest = record[field]
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"compatibility {field} must be a SHA-256 digest")
        task_ids.append(task_id)
    if not task_ids:
        raise ValueError("diagnostic compatibility contains no tasks")
    return tuple(task_ids)


def build_varc_diagnostic_overlay(
    *, blind_data_root: Path, runtime_data_root: Path
) -> dict[str, object]:
    """Mirror exact task names and bytes into a new runtime path."""

    blind_data_root = blind_data_root.resolve()
    runtime_data_root = runtime_data_root.resolve()
    if runtime_data_root.exists():
        raise FileExistsError(f"refusing to replace runtime data: {runtime_data_root}")
    blind_manifest = _object(
        json.loads((blind_data_root / "manifest.json").read_text(encoding="utf-8")),
        field="blind manifest",
    )
    if blind_manifest["schema"] != VARC_BLIND_SCHEMA:
        raise ValueError("unsupported VARC blind-manifest schema")
    records = _sequence(blind_manifest["tasks"], field="blind tasks")
    task_ids = sorted(
        str(_object(record, field="blind task")["task_id"]) for record in records
    )
    evaluation_source = blind_data_root / "data" / "evaluation"
    augmentation_source = blind_data_root / "data" / "eval_color_permute_ttt_9"
    if task_ids != sorted(path.stem for path in evaluation_source.glob("*.json")):
        raise ValueError("blind manifest and evaluation task sets differ")
    if task_ids != sorted(
        path.name for path in augmentation_source.iterdir() if path.is_dir()
    ):
        raise ValueError("blind manifest and augmentation task sets differ")

    evaluation_runtime = runtime_data_root / "data" / "evaluation"
    augmentation_runtime = runtime_data_root / "data" / "eval_color_permute_ttt_9"
    evaluation_runtime.mkdir(parents=True)
    augmentation_runtime.mkdir()
    compatibility_tasks: list[dict[str, object]] = []
    for task_id in task_ids:
        evaluation_file = evaluation_source / f"{task_id}.json"
        (evaluation_runtime / evaluation_file.name).symlink_to(
            evaluation_file.resolve()
        )
        source_directory = augmentation_source / task_id
        (augmentation_runtime / task_id).symlink_to(
            source_directory.resolve(), target_is_directory=True
        )
        augmentation_hashes = [
            {"name": path.name, "sha256": file_sha256(path)}
            for path in sorted(source_directory.glob("*.json"))
        ]
        if not augmentation_hashes:
            raise ValueError("compatibility task contains no augmentations")
        compatibility_tasks.append(
            {
                "augmentation_count": len(augmentation_hashes),
                "augmentation_set_sha256": canonical_sha256(augmentation_hashes),
                "evaluation_sha256": file_sha256(evaluation_file),
                "task_id": task_id,
            }
        )
    body: dict[str, object] = {
        "blind_cohort_id": blind_manifest["blind_cohort_id"],
        "schema": VARC_DIAGNOSTIC_COMPAT_SCHEMA,
        "tasks": compatibility_tasks,
    }
    result = {"compatibility_id": canonical_sha256(body), **body}
    validate_varc_diagnostic_compatibility(result)
    return result


def prepare_varc_diagnostic_alias(
    *, runtime_data_root: Path, task_id: str, archive_dir: Path
) -> dict[str, object]:
    """Point only the post-candidate diagnostic name at the current task."""

    runtime_data_root = runtime_data_root.resolve()
    archive_dir = archive_dir.resolve()
    if _SAFE_TASK_ID.fullmatch(task_id) is None:
        raise ValueError("unsafe VARC task ID")
    diagnostic_task_name = task_id.split("_", 1)[0]
    evaluation_dir = runtime_data_root / "data" / "evaluation"
    source_file = evaluation_dir / f"{task_id}.json"
    if not source_file.is_file():
        raise FileNotFoundError(f"runtime evaluation task is missing: {source_file}")
    alias_file = evaluation_dir / f"{diagnostic_task_name}.json"
    replaced_previous_alias = False
    if alias_file != source_file:
        if alias_file.exists() or alias_file.is_symlink():
            archive_dir.mkdir(parents=True, exist_ok=True)
            archived_alias = archive_dir / f"{diagnostic_task_name}.json"
            if archived_alias.exists() or archived_alias.is_symlink():
                raise FileExistsError(
                    f"refusing to replace diagnostic alias archive: {archived_alias}"
                )
            alias_file.replace(archived_alias)
            replaced_previous_alias = True
        alias_file.symlink_to(source_file.resolve())
    body: dict[str, object] = {
        "diagnostic_task_name": diagnostic_task_name,
        "evaluation_sha256": file_sha256(source_file),
        "replaced_previous_alias": replaced_previous_alias,
        "schema": VARC_DIAGNOSTIC_ALIAS_RECORD_SCHEMA,
        "task_id": task_id,
    }
    return {"record_id": canonical_sha256(body), **body}
