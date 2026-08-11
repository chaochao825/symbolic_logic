"""CLI for the preregistered counterfactual transition v4 gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from afts_arc.counterfactual_transition import (  # noqa: E402
    COUNTERFACTUAL_STRATEGIES,
)
from afts_arc.counterfactual_transition_gate import (  # noqa: E402
    freeze_counterfactual_transition_opportunities,
    score_counterfactual_transition_freeze,
)
from afts_arc.experiment_safety import atomic_write_json, file_sha256  # noqa: E402


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _freeze(args: argparse.Namespace) -> dict[str, object]:
    challenges = _read_object(args.challenges)
    v3_freeze = _read_object(args.v3_freeze)
    source_paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "afts_arc" / "stateful_scene.py",
        PROJECT_ROOT / "src" / "afts_arc" / "counterfactual_transition.py",
        PROJECT_ROOT / "src" / "afts_arc" / "counterfactual_transition_gate.py",
        args.protocol.resolve(),
    )
    source_files = {
        str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/"):
        file_sha256(path)
        for path in source_paths
    }
    freeze = freeze_counterfactual_transition_opportunities(
        challenges=challenges,
        v3_freeze=v3_freeze,
        cohort_id=args.cohort_id,
        scientific_lane=args.scientific_lane,
        source_files=source_files,
        strategies=tuple(args.strategies),
        max_first_stage_trials=args.max_first_stage_trials,
        max_parents=args.max_parents,
        max_transition_trials=args.max_transition_trials,
        max_candidates=args.max_candidates,
        max_workers=args.workers,
    )
    atomic_write_json(args.output, freeze)
    return {
        "freeze_id": freeze["freeze_id"],
        "task_count": freeze["task_count"],
        "opportunity_counts": freeze["opportunity_counts"],
        "novel_frontier_counts": freeze["novel_frontier_counts"],
        "output": str(args.output),
    }


def _score(args: argparse.Namespace) -> dict[str, object]:
    freeze = _read_object(args.freeze)
    solutions = _read_object(args.solutions)
    result = score_counterfactual_transition_freeze(
        freeze=freeze,
        solutions=solutions,
        solution_source_sha256=file_sha256(args.solutions),
    )
    atomic_write_json(args.output, result)
    return {
        "result_id": result["result_id"],
        "task_count": result["task_count"],
        "incumbent_exact": result["incumbent_exact"],
        "frozen_cold_exact": result["frozen_cold_exact"],
        "arm_metrics": result["arm_metrics"],
        "three_arm_oracle_union_exact": result["three_arm_oracle_union_exact"],
        "output": str(args.output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze")
    freeze.add_argument("--challenges", type=Path, required=True)
    freeze.add_argument("--v3-freeze", type=Path, required=True)
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
    freeze.add_argument(
        "--strategies",
        nargs="+",
        choices=COUNTERFACTUAL_STRATEGIES,
        default=COUNTERFACTUAL_STRATEGIES,
    )
    freeze.add_argument("--max-first-stage-trials", type=int, default=2_000)
    freeze.add_argument("--max-parents", type=int, default=4)
    freeze.add_argument("--max-transition-trials", type=int, default=512)
    freeze.add_argument("--max-candidates", type=int, default=32)
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
