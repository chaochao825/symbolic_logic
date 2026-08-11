#!/usr/bin/env python3
"""Build compact, payload-free summaries for Object–Program Workspace runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


SUMMARY_SCHEMA = "afts.object-program-workspace-public-summary/v1"
ARTIFACT_SCHEMA = "afts.object-program-workspace-artifact-sha256/v1"
GENERATED_NAMES = frozenset({"README.md", "artifact_sha256.json", "summary.json"})


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_replay_pair(directory: Path, stem: str) -> tuple[dict[str, object], bytes]:
    first_path = directory / f"{stem}_a.json"
    second_path = directory / f"{stem}_b.json"
    first = first_path.read_bytes()
    second = second_path.read_bytes()
    if first != second:
        raise ValueError(f"{stem} replay mismatch: {first_path} != {second_path}")
    return json.loads(first), first


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _artifact_manifest(directory: Path) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name in GENERATED_NAMES:
            continue
        payload = path.read_bytes()
        entries.append(
            {
                "bytes": len(payload),
                "path": path.name,
                "sha256": _sha256_bytes(payload),
            }
        )
    return {"entries": entries, "schema": ARTIFACT_SCHEMA}


def build(result_directory: Path) -> dict[str, object]:
    freeze, freeze_bytes = _read_replay_pair(result_directory, "candidate_freeze")
    result, result_bytes = _read_replay_pair(result_directory, "result")

    if freeze["freeze_id"] != result["candidate_freeze_id"]:
        raise ValueError("result references a different candidate freeze")
    if freeze["scientific_lane"] != result["scientific_lane"]:
        raise ValueError("freeze/result scientific-lane mismatch")
    if freeze["task_count"] != result["task_count"]:
        raise ValueError("freeze/result task-count mismatch")

    freeze_tasks = freeze["tasks"]
    result_tasks = result["tasks"]
    if not isinstance(freeze_tasks, list) or not isinstance(result_tasks, list):
        raise TypeError("freeze and result tasks must be lists")

    reason_counts = Counter(task["selection_reason"] for task in freeze_tasks)
    program_only_count = sum(
        task["program_opportunity"] and not task["frontier_opportunity"]
        for task in result_tasks
    )

    artifact_manifest = _artifact_manifest(result_directory)
    artifact_path = result_directory / "artifact_sha256.json"
    _write_json(artifact_path, artifact_manifest)
    artifact_payload = artifact_path.read_bytes()

    summary: dict[str, object] = {
        "action_interventions": result["action_interventions"],
        "artifact_manifest": {
            "entry_count": len(artifact_manifest["entries"]),
            "path": artifact_path.name,
            "sha256": _sha256_bytes(artifact_payload),
        },
        "candidate_freeze_id": freeze["freeze_id"],
        "controller_training_started": result["controller_training_started"],
        "controlled_semantic_gate": result["controlled_semantic_gate"],
        "development_screen": result["development_screen"],
        "dsl_version": freeze["dsl_version"],
        "output_frontier_opportunity_count": result["opportunity_count"],
        "program_only_opportunity_count": program_only_count,
        "program_opportunity_count": result["program_opportunity_count"],
        "public_evaluation_read": result["public_evaluation_read"],
        "query_gold_read_after_freeze": result["query_gold_read_after_freeze"],
        "replay": {
            "candidate_freeze_a_b_byte_identical": True,
            "candidate_freeze_file_sha256": _sha256_bytes(freeze_bytes),
            "result_a_b_byte_identical": True,
            "result_file_sha256": _sha256_bytes(result_bytes),
        },
        "result_id": result["result_id"],
        "schema": SUMMARY_SCHEMA,
        "scientific_lane": result["scientific_lane"],
        "selection_reason_counts": dict(sorted(reason_counts.items())),
        "strict_task_coverage": result["strict_task_coverage"],
        "task_count": result["task_count"],
        "unique_recovery_over_base_prefix": result[
            "unique_recovery_over_base_prefix"
        ],
        "visual_unique_over_all_controls": result["visual_unique_over_all_controls"],
    }
    _write_json(result_directory / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build(args.result_dir)
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
