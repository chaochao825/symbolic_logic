"""Validate and consolidate committed DiffLogic-ARC run directories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = ROOT / "results" / "runs"

SELECTED_FIELDS = (
    "run",
    "suite",
    "cohort",
    "source_commit",
    "task_id",
    "variant",
    "seed",
    "status",
    "deployment_eligible",
    "strict_hard_task_exact",
    "selected_by_demo",
    "hard_demo_task_exact",
    "hard_demo_cell_accuracy",
    "hard_task_exact",
    "hard_cell_accuracy",
    "soft_task_exact",
    "hardening_task_drop",
    "selected_horizon",
    "state_bits_per_cell",
    "parameter_count",
    "parameter_storage_bits",
    "hard_fixed_width_circuit_bits",
    "active_non_passthrough_gates",
    "hard_gates_per_cell_step",
    "test_dynamic_gate_evaluations",
    "dense_macs_per_cell_step",
    "test_dense_macs",
    "test_state_bit_updates",
    "hard_export_equivalent",
    "elapsed_seconds",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _number(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def validate_run(directory: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("experiment") != "arc_difflogic_v1":
        raise ValueError(f"{directory}: unexpected experiment")
    if metadata.get("mode") == "full" and (
        metadata.get("source_status_start") or metadata.get("source_status_end")
    ):
        raise ValueError(f"{directory}: full run did not remain clean")
    for name, record in metadata.get("artifacts", {}).items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"{directory}: missing or truncated {name}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"{directory}: hash mismatch for {name}")
    selected = _read_csv(directory / "arc_difflogic_selected_results.csv")
    if len(selected) != int(metadata["selected_rows"]):
        raise ValueError(f"{directory}: selected-row count mismatch")
    return metadata, selected


def discover_runs(root: Path) -> list[Path]:
    runs = []
    for metadata_path in sorted(root.glob("*/metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("experiment") == "arc_difflogic_v1":
            runs.append(metadata_path.parent)
    return runs


def consolidate(directories: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_rows: list[dict[str, Any]] = []
    for directory in directories:
        metadata, rows = validate_run(directory)
        for row in rows:
            deployment_eligible = _number(row, "deployment_eligible")
            if not math.isfinite(deployment_eligible):
                deployment_eligible = _number(row, "hard_demo_task_exact")
            hard_exact = _number(row, "hard_task_exact")
            selected_rows.append(
                {
                    **row,
                    "run": directory.name,
                    "suite": metadata["suite"],
                    "cohort": metadata["cohort"],
                    "source_commit": metadata["source_commit"],
                    "deployment_eligible": deployment_eligible,
                    "strict_hard_task_exact": (
                        hard_exact * deployment_eligible
                        if math.isfinite(hard_exact) and math.isfinite(deployment_eligible)
                        else math.nan
                    ),
                }
            )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in selected_rows:
        key = (str(row["suite"]), str(row["cohort"]), str(row["variant"]))
        grouped.setdefault(key, []).append(row)
    aggregate = []
    for (suite, cohort, variant), rows in sorted(grouped.items()):
        exact = [_number(row, "hard_task_exact") for row in rows]
        strict_exact = [_number(row, "strict_hard_task_exact") for row in rows]
        demo = [_number(row, "hard_demo_task_exact") for row in rows]
        eligible = [_number(row, "deployment_eligible") for row in rows]
        finite_eligible = [value for value in eligible if math.isfinite(value)]
        aggregate.append(
            {
                "suite": suite,
                "cohort": cohort,
                "variant": variant,
                "tasks": len(rows),
                "hard_exact_tasks": int(sum(value for value in exact if math.isfinite(value))),
                "hard_exact_rate": sum(value for value in exact if math.isfinite(value)) / len(rows),
                "strict_hard_exact_tasks": int(sum(value for value in strict_exact if math.isfinite(value))),
                "strict_hard_exact_rate": sum(value for value in strict_exact if math.isfinite(value)) / len(rows),
                "hard_demo_exact_tasks": int(sum(value for value in demo if math.isfinite(value))),
                "hard_demo_exact_rate": sum(value for value in demo if math.isfinite(value)) / len(rows),
                "deployment_eligible_tasks": int(sum(finite_eligible)) if finite_eligible else "",
            }
        )
    return selected_rows, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--selected-output", type=Path, default=ROOT / "results" / "arc_difflogic_v1_selected.csv")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "results" / "arc_difflogic_v1_summary.csv")
    parser.add_argument("--metadata-output", type=Path, default=ROOT / "results" / "arc_difflogic_v1_metadata.json")
    args = parser.parse_args()
    directories = discover_runs(args.runs_root)
    if not directories:
        parser.error("no DiffLogic-ARC runs found")
    selected, aggregate = consolidate(directories)
    _write_csv(args.selected_output, selected, SELECTED_FIELDS)
    _write_csv(
        args.summary_output,
        aggregate,
        (
            "suite",
            "cohort",
            "variant",
            "tasks",
            "hard_exact_tasks",
            "hard_exact_rate",
            "strict_hard_exact_tasks",
            "strict_hard_exact_rate",
            "hard_demo_exact_tasks",
            "hard_demo_exact_rate",
            "deployment_eligible_tasks",
        ),
    )
    aggregate_metadata = {
        "experiment": "arc_difflogic_v1_aggregate",
        "runs": [directory.name for directory in directories],
        "run_count": len(directories),
        "selected_rows": len(selected),
        "aggregate_rows": len(aggregate),
        "source_commits": sorted({str(row["source_commit"]) for row in selected}),
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (args.selected_output, args.summary_output)
        },
        "strict_metric": "hard_demo_task_exact * hard_task_exact",
        "claim_boundary": "ARC cohorts are public-training mechanism probes; ARC-AGI-2 evaluation is not rerun; software timing is not PPA.",
    }
    args.metadata_output.write_text(
        json.dumps(aggregate_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"validated {len(directories)} runs and {len(selected)} selected rows")


if __name__ == "__main__":
    main()
