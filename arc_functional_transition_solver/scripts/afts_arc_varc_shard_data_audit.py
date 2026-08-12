"""Verify that isolated VARC shards preserve full-cohort task bytes exactly."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


def _object(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be an array")
    return value


def _load(path: Path) -> Mapping[str, object]:
    return _object(json.loads(path.read_text(encoding="utf-8")), field=str(path))


def _task_ids(manifest: Mapping[str, object]) -> list[str]:
    records = _sequence(manifest["tasks"], field="manifest tasks")
    task_ids = [str(_object(record, field="manifest task")["task_id"]) for record in records]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("blind manifest contains duplicate task IDs")
    return sorted(task_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--full-blind-root", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--shard-root-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    project_root = arguments.project_root.resolve()
    sys.path.insert(0, str(project_root / "src"))
    from afts_arc.experiment_safety import (  # noqa: PLC0415
        atomic_write_json,
        canonical_sha256,
        file_sha256,
    )

    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace artifact: {output}")
    full_root = arguments.full_blind_root.resolve()
    full_manifest_path = full_root / "manifest.json"
    full_manifest = _load(full_manifest_path)
    full_task_ids = _task_ids(full_manifest)
    assignment_path = arguments.assignment.resolve()
    assignment = _load(assignment_path)
    if assignment["query_gold_read"] is not False:
        raise ValueError("shard assignment is not query-gold-free")
    if assignment["task_ids"] != full_task_ids:
        raise ValueError("assignment and full blind task universes differ")

    observed_task_ids: list[str] = []
    total_augmentation_files = 0
    shard_audits: list[dict[str, object]] = []
    shard_records = _sequence(assignment["shards"], field="assignment shards")
    for raw_shard in shard_records:
        shard = _object(raw_shard, field="assignment shard")
        shard_index = int(shard["shard_index"])
        shard_root = Path(f"{arguments.shard_root_prefix}{shard_index}_blind_v1").resolve()
        shard_manifest_path = shard_root / "manifest.json"
        shard_manifest = _load(shard_manifest_path)
        shard_task_ids = _task_ids(shard_manifest)
        expected_task_ids = list(shard["task_ids"])
        if shard_task_ids != expected_task_ids:
            raise ValueError(f"shard {shard_index} task universe differs")
        if shard_manifest["source_challenges_sha256"] != shard["challenge_file_sha256"]:
            raise ValueError(f"shard {shard_index} challenge identity differs")

        shard_augmentation_count = 0
        for task_id in shard_task_ids:
            full_evaluation = full_root / "data" / "evaluation" / f"{task_id}.json"
            shard_evaluation = shard_root / "data" / "evaluation" / f"{task_id}.json"
            if file_sha256(full_evaluation) != file_sha256(shard_evaluation):
                raise ValueError(f"evaluation bytes differ for {task_id}")
            full_augmentation = (
                full_root / "data" / "eval_color_permute_ttt_9" / task_id
            )
            shard_augmentation = (
                shard_root / "data" / "eval_color_permute_ttt_9" / task_id
            )
            full_files = sorted(path.name for path in full_augmentation.glob("*.json"))
            shard_files = sorted(path.name for path in shard_augmentation.glob("*.json"))
            if not full_files or shard_files != full_files:
                raise ValueError(f"augmentation file set differs for {task_id}")
            for filename in full_files:
                if file_sha256(full_augmentation / filename) != file_sha256(
                    shard_augmentation / filename
                ):
                    raise ValueError(f"augmentation bytes differ for {task_id}/{filename}")
            shard_augmentation_count += len(full_files)
        observed_task_ids.extend(shard_task_ids)
        total_augmentation_files += shard_augmentation_count
        shard_audits.append(
            {
                "blind_cohort_id": shard_manifest["blind_cohort_id"],
                "manifest_sha256": file_sha256(shard_manifest_path),
                "shard_index": shard_index,
                "task_count": len(shard_task_ids),
                "augmentation_file_count": shard_augmentation_count,
            }
        )
    if sorted(observed_task_ids) != full_task_ids:
        raise ValueError("shards do not cover the full task universe")
    if len(set(observed_task_ids)) != len(observed_task_ids):
        raise ValueError("shards overlap")

    body: dict[str, object] = {
        "assignment_id": assignment["assignment_id"],
        "assignment_sha256": file_sha256(assignment_path),
        "full_blind_cohort_id": full_manifest["blind_cohort_id"],
        "full_manifest_sha256": file_sha256(full_manifest_path),
        "query_gold_read": False,
        "schema": "afts.varc-shard-data-audit/v1",
        "shards": sorted(shard_audits, key=lambda row: int(row["shard_index"])),
        "status": "clean",
        "task_count": len(full_task_ids),
        "total_augmentation_file_count": total_augmentation_files,
    }
    result = {"audit_id": canonical_sha256(body), **body}
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "audit_id": result["audit_id"],
                "status": result["status"],
                "task_count": result["task_count"],
                "total_augmentation_file_count": result[
                    "total_augmentation_file_count"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
