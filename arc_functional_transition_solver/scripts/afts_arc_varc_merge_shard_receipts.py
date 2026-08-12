"""Close one auditable VARC receipt over isolated single-GPU shard receipts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


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


def _load_object(path: Path) -> Mapping[str, object]:
    return _object(json.loads(path.read_text(encoding="utf-8")), field=str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--full-blind-manifest", type=Path, required=True)
    parser.add_argument("--full-compatibility", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--orchestration-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--shard-receipt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    project_root = arguments.project_root.resolve()
    sys.path.insert(0, str(project_root / "src"))
    from afts_arc.experiment_safety import (  # noqa: PLC0415
        atomic_write_json,
        canonical_sha256,
        file_sha256,
    )
    from afts_arc.varc_run import (  # noqa: PLC0415
        VARC_RUN_RECEIPT_SCHEMA,
        VARC_TTT_CONFIG,
        build_varc_run_receipt,
    )

    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace artifact: {output}")
    checkpoint = arguments.checkpoint.resolve()
    if file_sha256(checkpoint) != arguments.expected_checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 differs from the frozen contract")
    full_manifest_path = arguments.full_blind_manifest.resolve()
    full_manifest = _load_object(full_manifest_path)
    full_records = _sequence(full_manifest["tasks"], field="full manifest tasks")
    full_task_ids = {str(_object(row, field="full task")["task_id"]) for row in full_records}
    if len(full_task_ids) != len(full_records):
        raise ValueError("full blind manifest has duplicate task IDs")

    prediction_root = arguments.prediction_root.resolve()
    statuses: dict[str, object] = {}
    predictions: dict[str, object] = {}
    declared_gpu_ids: list[int] = []
    shard_sources: list[dict[str, object]] = []
    common_packages: object | None = None
    common_platform: object | None = None
    common_python: object | None = None
    common_executed_source: object | None = None

    for receipt_path_argument in arguments.shard_receipt:
        receipt_path = receipt_path_argument.resolve()
        receipt = _load_object(receipt_path)
        body = dict(receipt)
        declared_receipt_id = body.pop("receipt_id")
        if declared_receipt_id != canonical_sha256(body):
            raise ValueError(f"shard receipt ID mismatch: {receipt_path}")
        if receipt["schema"] != VARC_RUN_RECEIPT_SCHEMA:
            raise ValueError("unsupported shard receipt schema")
        if receipt["controller_training_started"] is not False:
            raise ValueError("a shard reports controller training")
        protocol = _object(receipt["query_blind_protocol"], field="query protocol")
        if protocol["query_gold_read"] is not False:
            raise ValueError("a shard reports query-gold access")
        if receipt["test_time_configuration"] != VARC_TTT_CONFIG:
            raise ValueError("a shard changes the frozen VARC configuration")
        source = _object(receipt["source"], field="shard source")
        if source["source_commit"] != arguments.expected_source_commit:
            raise ValueError("a shard source commit differs from the contract")
        if source["checkpoint_sha256"] != arguments.expected_checkpoint_sha256:
            raise ValueError("a shard checkpoint differs from the contract")
        executed_source = source["executed_source_sha256"]
        if common_executed_source is None:
            common_executed_source = executed_source
        elif executed_source != common_executed_source:
            raise ValueError("executed source hashes differ across shards")
        runtime = _object(receipt["runtime"], field="shard runtime")
        for field, value in (
            ("packages", runtime["packages"]),
            ("platform", runtime["platform"]),
            ("python", runtime["python"]),
        ):
            common_value = {
                "packages": common_packages,
                "platform": common_platform,
                "python": common_python,
            }[field]
            if common_value is not None and value != common_value:
                raise ValueError(f"runtime {field} differs across shards")
        common_packages = runtime["packages"]
        common_platform = runtime["platform"]
        common_python = runtime["python"]
        compute = _object(receipt["compute"], field="shard compute")
        gpu_ids = _sequence(compute["gpu_ids"], field="shard GPU IDs")
        if len(gpu_ids) != 1:
            raise ValueError("each VARC shard must use exactly one GPU")
        gpu_id = gpu_ids[0]
        if isinstance(gpu_id, bool) or not isinstance(gpu_id, int):
            raise TypeError("shard GPU ID must be an integer")
        if gpu_id in declared_gpu_ids:
            raise ValueError("two simultaneous shards declare the same GPU")
        declared_gpu_ids.append(gpu_id)

        shard_task_ids: list[str] = []
        for raw_status in _sequence(receipt["task_statuses"], field="task statuses"):
            receipt_status = dict(_object(raw_status, field="task status"))
            prediction_bytes = receipt_status.pop("prediction_bytes")
            prediction_sha256 = receipt_status.pop("prediction_sha256")
            task_id = str(receipt_status["task_id"])
            if task_id in statuses:
                raise ValueError(f"task appears in multiple shards: {task_id}")
            prediction_path = prediction_root / f"{task_id}_predictions.json"
            if prediction_path.stat().st_size != prediction_bytes:
                raise ValueError(f"prediction byte count differs for {task_id}")
            if file_sha256(prediction_path) != prediction_sha256:
                raise ValueError(f"prediction SHA-256 differs for {task_id}")
            statuses[task_id] = receipt_status
            predictions[task_id] = {
                "bytes": prediction_bytes,
                "sha256": prediction_sha256,
            }
            shard_task_ids.append(task_id)
        if compute["task_count"] != len(shard_task_ids):
            raise ValueError("shard receipt task count is inconsistent")
        shard_sources.append(
            {
                "blind_cohort_id": receipt["blind_cohort_id"],
                "gpu_id": gpu_id,
                "receipt_id": declared_receipt_id,
                "receipt_sha256": file_sha256(receipt_path),
                "task_count": len(shard_task_ids),
                "task_ids_sha256": canonical_sha256(sorted(shard_task_ids)),
            }
        )

    if set(statuses) != full_task_ids:
        missing = sorted(full_task_ids - set(statuses))
        extra = sorted(set(statuses) - full_task_ids)
        raise ValueError(f"shards are not an exact full-task cover: missing={missing}, extra={extra}")
    if common_packages is None or common_platform is None or common_python is None:
        raise ValueError("no shard runtime was supplied")
    if common_executed_source is None:
        raise ValueError("no executed source manifest was supplied")

    compatibility_path = arguments.full_compatibility.resolve()
    orchestration_path = arguments.orchestration_manifest.resolve()
    source_receipt = {
        "checkpoint_sha256": arguments.expected_checkpoint_sha256,
        "executed_source_sha256": common_executed_source,
        "full_blind_manifest_sha256": file_sha256(full_manifest_path),
        "full_diagnostic_compatibility_sha256": file_sha256(compatibility_path),
        "merge_script_sha256": file_sha256(Path(__file__).resolve()),
        "orchestration_manifest_sha256": file_sha256(orchestration_path),
        "shards": sorted(shard_sources, key=lambda row: int(row["gpu_id"])),
        "source_commit": arguments.expected_source_commit,
    }
    runtime_receipt = {
        "execution_mode": "isolated-single-gpu-shards/v1",
        "packages": common_packages,
        "platform": common_platform,
        "python": common_python,
        "shard_count": len(shard_sources),
    }
    merged = build_varc_run_receipt(
        blind_manifest=full_manifest,
        task_statuses=statuses,
        prediction_files=predictions,
        source=source_receipt,
        runtime=runtime_receipt,
        gpu_ids=tuple(sorted(declared_gpu_ids)),
        compatibility_manifest=_load_object(compatibility_path),
    )
    atomic_write_json(output, merged)
    print(
        json.dumps(
            {
                "gpu_seconds": merged["compute"]["gpu_seconds"],
                "output": str(output),
                "receipt_id": merged["receipt_id"],
                "task_count": merged["compute"]["task_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
