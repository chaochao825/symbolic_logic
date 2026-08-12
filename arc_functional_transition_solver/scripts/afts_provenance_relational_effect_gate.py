"""Freeze and score the sequential provenance-relational-effect experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from afts_arc.experiment_safety import (  # noqa: E402
    atomic_write_json,
    file_sha256,
)
from afts_arc.relational_effect_gate import (  # noqa: E402
    audit_relational_effect_sensor,
    evaluate_relational_effect_interventions,
    freeze_relational_effect_opportunities,
    score_relational_effect_freeze,
)


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _source_files(args: argparse.Namespace) -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "afts_arc" / "stateful_scene.py",
        PROJECT_ROOT / "src" / "afts_arc" / "relational_effect.py",
        PROJECT_ROOT / "src" / "afts_arc" / "relational_effect_gate.py",
        args.protocol.resolve(),
    )
    return {
        str(path.relative_to(PROJECT_ROOT.resolve())).replace("\\", "/"):
            file_sha256(path)
        for path in paths
    }


def _freeze(args: argparse.Namespace) -> dict[str, object]:
    challenges = _read_object(args.challenges)
    seal = _read_object(args.seal)
    freeze = freeze_relational_effect_opportunities(
        challenges=challenges,
        seal=seal,
        seal_sha256=file_sha256(args.seal),
        source_files=_source_files(args),
        max_first_stage_trials=args.max_first_stage_trials,
        max_exact_programs=args.max_exact_programs,
        max_parents=args.max_parents,
        reserved_native_trials=args.reserved_native_trials,
        max_workers=args.workers,
    )
    atomic_write_json(args.output, freeze)
    return {
        "freeze_id": freeze["freeze_id"],
        "task_count": freeze["task_count"],
        "sensor": freeze["sensor"],
        "frontier_evaluated": freeze["frontier_evaluated"],
        "opportunity_count": freeze["opportunity_count"],
        "opportunity_family_count": freeze["opportunity_family_count"],
        "frontier_gate_passed": freeze["frontier_gate_passed"],
        "query_gold_read": freeze["query_gold_read"],
        "controller_training_started": freeze["controller_training_started"],
    }


def _score(args: argparse.Namespace) -> dict[str, object]:
    freeze = _read_object(args.freeze)
    solutions = _read_object(args.solutions)
    result = score_relational_effect_freeze(
        freeze=freeze,
        solutions=solutions,
        solution_source_sha256=file_sha256(args.solutions),
    )
    atomic_write_json(args.output, result)
    return {
        "result_id": result["result_id"],
        "metrics": result["metrics"],
        "unique_recovery_family_count": result["unique_recovery_family_count"],
        "natural_utility_gate_passed": result["natural_utility_gate_passed"],
        "outcome": result["outcome"],
        "controller_training_started": result["controller_training_started"],
    }


def _audit(args: argparse.Namespace) -> dict[str, object]:
    freeze = _read_object(args.freeze)
    audit = audit_relational_effect_sensor(freeze)
    atomic_write_json(args.output, audit)
    return {
        "audit_id": audit["audit_id"],
        "task_count": audit["task_count"],
        "terminal_reason_counts": audit["terminal_reason_counts"],
        "parent_failure_counts": audit["parent_failure_counts"],
        "lodo_fold_outcome_counts": audit["lodo_fold_outcome_counts"],
        "source_backed_recall": audit["source_backed_recall"],
        "entity_backed_recall": audit["entity_backed_recall"],
        "query_gold_read": audit["query_gold_read"],
    }


def _intervene(args: argparse.Namespace) -> dict[str, object]:
    intervention = evaluate_relational_effect_interventions(
        challenges=_read_object(args.challenges),
        freeze=_read_object(args.freeze),
        result=_read_object(args.result),
        solutions=_read_object(args.solutions),
    )
    atomic_write_json(args.output, intervention)
    return {
        "intervention_id": intervention["intervention_id"],
        "counts": intervention["counts"],
        "effects": intervention["effects"],
        "mechanism_gate_passed": intervention["mechanism_gate_passed"],
        "controller_training_started": intervention["controller_training_started"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--challenges", type=Path, required=True)
    freeze.add_argument("--seal", type=Path, required=True)
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--max-first-stage-trials", type=int, default=2_000)
    freeze.add_argument("--max-exact-programs", type=int, default=32)
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

    audit = commands.add_parser("audit")
    audit.add_argument("--freeze", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.set_defaults(handler=_audit)

    intervene = commands.add_parser("intervene")
    intervene.add_argument("--challenges", type=Path, required=True)
    intervene.add_argument("--freeze", type=Path, required=True)
    intervene.add_argument("--result", type=Path, required=True)
    intervene.add_argument("--solutions", type=Path, required=True)
    intervene.add_argument("--output", type=Path, required=True)
    intervene.set_defaults(handler=_intervene)
    args = parser.parse_args()
    print(json.dumps(args.handler(args), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
