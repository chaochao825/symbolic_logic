"""CLI for the frozen one-recolor natural-utility audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from afts_arc.experiment_safety import atomic_write_json, file_sha256  # noqa: E402
from afts_arc.recolor_topology_gate import (  # noqa: E402
    SCIENTIFIC_LANES,
    freeze_recolor_topology_opportunities,
    score_recolor_topology_freeze,
)


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _freeze(args: argparse.Namespace) -> dict[str, object]:
    challenges = _read_object(args.challenges)
    parent_freeze = _read_object(args.parent_freeze)
    source_paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "afts_arc" / "executable_workspace.py",
        PROJECT_ROOT / "src" / "afts_arc" / "recolor_topology_gate.py",
        args.protocol.resolve(),
    )
    source_files = {
        str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/"):
        file_sha256(path)
        for path in source_paths
    }
    freeze = freeze_recolor_topology_opportunities(
        challenges=challenges,
        parent_freeze=parent_freeze,
        parent_freeze_sha256=file_sha256(args.parent_freeze),
        cohort_id=args.cohort_id,
        scientific_lane=args.scientific_lane,
        source_files=source_files,
        max_first_stage_trials=args.max_first_stage_trials,
        max_parents=args.max_parents,
        reserved_native_trials=args.reserved_native_trials,
        max_workers=args.workers,
    )
    atomic_write_json(args.output, freeze)
    return {
        "freeze_id": freeze["freeze_id"],
        "task_count": freeze["task_count"],
        "reachable_parent_count": freeze["reachable_parent_count"],
        "reachable_task_count": freeze["reachable_task_count"],
        "demo_exact_candidate_task_count": freeze[
            "demo_exact_candidate_task_count"
        ],
        "novel_frontier_task_count": freeze["novel_frontier_task_count"],
        "output": str(args.output),
    }


def _score(args: argparse.Namespace) -> dict[str, object]:
    freeze = _read_object(args.freeze)
    solutions = _read_object(args.solutions)
    result = score_recolor_topology_freeze(
        freeze=freeze,
        solutions=solutions,
        solution_source_sha256=file_sha256(args.solutions),
    )
    atomic_write_json(args.output, result)
    return {
        "result_id": result["result_id"],
        "task_count": result["task_count"],
        "baseline_union_exact": result["baseline_union_exact"],
        "recolor_exact": result["recolor_exact"],
        "novel_recolor_exact": result["novel_recolor_exact"],
        "unique_recovery_over_incumbent_and_cold": result[
            "unique_recovery_over_incumbent_and_cold"
        ],
        "outcome": result["outcome"],
        "output": str(args.output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze")
    freeze.add_argument("--challenges", type=Path, required=True)
    freeze.add_argument("--parent-freeze", type=Path, required=True)
    freeze.add_argument("--cohort-id", required=True)
    freeze.add_argument(
        "--scientific-lane",
        choices=SCIENTIFIC_LANES,
        required=True,
    )
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--max-first-stage-trials", type=int, default=2_000)
    freeze.add_argument("--max-parents", type=int, default=4)
    freeze.add_argument("--reserved-native-trials", type=int, default=2_048)
    freeze.add_argument("--workers", type=int, default=1)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.set_defaults(handler=_freeze)

    score = commands.add_parser("score")
    score.add_argument("--freeze", type=Path, required=True)
    score.add_argument("--solutions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(handler=_score)

    args = parser.parse_args()
    summary = args.handler(args)
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
