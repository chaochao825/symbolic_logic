"""Freeze a query-gold-free, deterministic task partition for isolated VARC runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


def _load_object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain an object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{path} keys must be strings")
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenges", type=Path, required=True)
    parser.add_argument("--source-cohort-id", required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    challenges_path = arguments.challenges.resolve()
    output_dir = arguments.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace shard commitment: {output_dir}")
    if arguments.shard_count <= 0:
        raise ValueError("shard-count must be positive")
    challenges = _load_object(challenges_path)
    task_ids = sorted(challenges)
    if not task_ids:
        raise ValueError("challenge set is empty")
    if arguments.shard_count > len(task_ids):
        raise ValueError("shard-count exceeds task count")

    shard_records: list[dict[str, object]] = []
    assigned: list[str] = []
    for shard_index in range(arguments.shard_count):
        shard_task_ids = task_ids[shard_index :: arguments.shard_count]
        shard_challenges = {
            task_id: challenges[task_id] for task_id in shard_task_ids
        }
        shard_path = output_dir / f"shard_{shard_index}_challenges.json"
        _write_json(shard_path, shard_challenges)
        assigned.extend(shard_task_ids)
        shard_records.append(
            {
                "challenge_file": shard_path.name,
                "challenge_file_sha256": _file_sha256(shard_path),
                "shard_index": shard_index,
                "task_count": len(shard_task_ids),
                "task_ids": shard_task_ids,
            }
        )
    if sorted(assigned) != task_ids or len(set(assigned)) != len(task_ids):
        raise AssertionError("shard partition is not an exact disjoint cover")

    body: dict[str, object] = {
        "assignment_rule": "sorted-task-id-round-robin/v1",
        "query_gold_read": False,
        "schema": "afts.varc-isolated-shard-assignment/v1",
        "shard_count": arguments.shard_count,
        "shards": shard_records,
        "source_challenges_sha256": _file_sha256(challenges_path),
        "source_cohort_id": arguments.source_cohort_id,
        "task_count": len(task_ids),
        "task_ids": task_ids,
    }
    assignment = {"assignment_id": _canonical_sha256(body), **body}
    _write_json(output_dir / "assignment.json", assignment)
    print(
        json.dumps(
            {
                "assignment_id": assignment["assignment_id"],
                "shard_sizes": [record["task_count"] for record in shard_records],
                "task_count": len(task_ids),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
