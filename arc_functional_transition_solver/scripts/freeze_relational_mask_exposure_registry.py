"""Freeze the exposure snapshot for the ARC-2 relational-mask gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from afts_arc.experiment_safety import atomic_write_json, canonical_sha256  # noqa: E402
from afts_arc.visual_provider_gate import (  # noqa: E402
    EXPOSURE_REGISTRY_SCHEMA,
    discover_previously_exposed_task_ids,
)


EXPECTED_SCAN_ID = "459b1667edbfb6d9af5aa180f1a402b8309890991b6fa22bcf32decd758a1440"


def _validate_scan_identity(scan: Mapping[str, object]) -> None:
    content = {key: value for key, value in scan.items() if key != "scan_id"}
    if scan["scan_id"] != canonical_sha256(content):
        raise ValueError("fresh ARC-GEN scan content ID mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fresh_arcgen_scan", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    if args.output_path.exists():
        raise FileExistsError(f"exposure registry already exists: {args.output_path}")
    scan = json.loads(args.fresh_arcgen_scan.read_text(encoding="utf-8"))
    if not isinstance(scan, dict):
        raise TypeError("fresh ARC-GEN scan must be an object")
    _validate_scan_identity(scan)
    if scan["scan_id"] != EXPECTED_SCAN_ID:
        raise ValueError("fresh ARC-GEN scan ID mismatch")
    if scan["query_gold_read"] is not False:
        raise ValueError("fresh ARC-GEN scan is not query blind")
    scan_ids = {row["family_id"] for row in scan["families"]}
    if len(scan_ids) != scan["scanned_family_count"]:
        raise ValueError("fresh ARC-GEN scan contains duplicate family IDs")
    authored = discover_previously_exposed_task_ids(PROJECT_ROOT)
    task_ids = sorted(authored | scan_ids)
    content = {
        "schema": EXPOSURE_REGISTRY_SCHEMA,
        "sources": [
            {
                "kind": "authored_repository_snapshot",
                "project_root": str(PROJECT_ROOT),
                "task_count": len(authored),
            },
            {
                "kind": "fresh_arcgen_feasibility_scan",
                "scan_id": scan["scan_id"],
                "task_count": len(scan_ids),
            },
        ],
        "task_ids": task_ids,
    }
    registry = {"registry_id": canonical_sha256(content), **content}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_path, registry)
    print(
        json.dumps(
            {
                "registry_id": registry["registry_id"],
                "task_count": len(task_ids),
                "authored_task_count": len(authored),
                "fresh_scan_task_count": len(scan_ids),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
