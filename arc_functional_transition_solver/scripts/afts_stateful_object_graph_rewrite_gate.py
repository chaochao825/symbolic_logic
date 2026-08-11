"""CLI for the preregistered stateful Object-Graph Rewrite v3 gate."""

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
from afts_arc.stateful_object_graph_rewrite_gate import (  # noqa: E402
    freeze_stateful_object_graph_rewrite_opportunities,
    score_stateful_object_graph_rewrite_freeze,
)


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _freeze(args: argparse.Namespace) -> dict[str, object]:
    challenges = _read_object(args.challenges)
    incumbent = _read_object(args.incumbent_freeze)
    source_paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "afts_arc" / "stateful_scene.py",
        PROJECT_ROOT / "src" / "afts_arc" / "stateful_object_graph_rewrite.py",
        PROJECT_ROOT / "src" / "afts_arc" / "stateful_object_graph_rewrite_gate.py",
        args.protocol.resolve(),
    )
    source_files = {
        str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace(
            "\\", "/"
        ): file_sha256(path)
        for path in source_paths
    }
    freeze = freeze_stateful_object_graph_rewrite_opportunities(
        challenges=challenges,
        incumbent_freeze=incumbent,
        cohort_id=args.cohort_id,
        scientific_lane=args.scientific_lane,
        source_files=source_files,
        max_first_stage_trials=args.max_first_stage_trials,
        max_parents=args.max_parents,
        max_rewrite_trials=args.max_rewrite_trials,
        max_candidates=args.max_candidates,
    )
    atomic_write_json(args.output, freeze)
    return {
        "freeze_id": freeze["freeze_id"],
        "task_count": freeze["task_count"],
        "opportunity_count": freeze["opportunity_count"],
        "novel_frontier_count": freeze["novel_frontier_count"],
        "output": str(args.output),
    }


def _score(args: argparse.Namespace) -> dict[str, object]:
    freeze = _read_object(args.freeze)
    solutions = _read_object(args.solutions)
    result = score_stateful_object_graph_rewrite_freeze(
        freeze=freeze,
        solutions=solutions,
        solution_source_sha256=file_sha256(args.solutions),
    )
    atomic_write_json(args.output, result)
    return {
        "result_id": result["result_id"],
        "task_count": result["task_count"],
        "metrics": result["metrics"],
        "output": str(args.output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze")
    freeze.add_argument("--challenges", type=Path, required=True)
    freeze.add_argument("--incumbent-freeze", type=Path, required=True)
    freeze.add_argument("--cohort-id", required=True)
    freeze.add_argument(
        "--scientific-lane",
        choices=(
            "controlled_semantic",
            "outcome_exposed_development",
            "prospective_reserve",
        ),
        required=True,
    )
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--max-first-stage-trials", type=int, default=2_000)
    freeze.add_argument("--max-parents", type=int, default=4)
    freeze.add_argument("--max-rewrite-trials", type=int, default=512)
    freeze.add_argument("--max-candidates", type=int, default=32)
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
