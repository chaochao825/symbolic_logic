"""Freeze and score the NVARC/TRM anchor candidate pool."""

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

from afts_arc.experiment_safety import atomic_write_json, file_sha256  # noqa: E402
from afts_arc.nvarc_anchor import (  # noqa: E402
    freeze_submission,
    freeze_submission_with_rejections,
    score_candidate_freeze,
    score_candidate_freeze_gate,
)


def _load_object(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"JSON artifact must be an object: {path}")
    return payload


def _write_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace artifact: {path}")
    atomic_write_json(path, payload)


def _freeze(arguments: argparse.Namespace) -> None:
    challenges = Path(arguments.challenges).resolve()
    submission = Path(arguments.submission).resolve()
    output = Path(arguments.output).resolve()
    freeze_builder = (
        freeze_submission_with_rejections
        if arguments.record_invalid_attempts
        else freeze_submission
    )
    payload = freeze_builder(
        challenges=_load_object(challenges),
        submission=_load_object(submission),
        cohort_id=arguments.cohort_id,
        anchor_run_id=arguments.anchor_run_id,
        source_files={
            "challenges": file_sha256(challenges),
            "submission": file_sha256(submission),
        },
    )
    _write_new(output, payload)
    print(json.dumps({"freeze_id": payload["freeze_id"], "output": str(output)}))


def _score(arguments: argparse.Namespace) -> None:
    candidate_freeze = Path(arguments.candidate_freeze).resolve()
    solutions = Path(arguments.solutions).resolve()
    output = Path(arguments.output).resolve()
    payload = score_candidate_freeze(
        freeze=_load_object(candidate_freeze),
        solutions=_load_object(solutions),
        solution_source_sha256=file_sha256(solutions),
    )
    _write_new(output, payload)
    print(json.dumps({"result_id": payload["result_id"], "output": str(output)}))


def _score_gate(arguments: argparse.Namespace) -> None:
    candidate_freeze = Path(arguments.candidate_freeze).resolve()
    solutions = Path(arguments.solutions).resolve()
    output = Path(arguments.output).resolve()
    payload = score_candidate_freeze_gate(
        freeze=_load_object(candidate_freeze),
        solutions=_load_object(solutions),
        solution_source_sha256=file_sha256(solutions),
    )
    _write_new(output, payload)
    print(
        json.dumps(
            {"gate_summary_id": payload["gate_summary_id"], "output": str(output)}
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--challenges", required=True)
    freeze.add_argument("--submission", required=True)
    freeze.add_argument("--cohort-id", required=True)
    freeze.add_argument("--anchor-run-id", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--record-invalid-attempts", action="store_true")
    freeze.set_defaults(handler=_freeze)

    score = subparsers.add_parser("score")
    score.add_argument("--candidate-freeze", required=True)
    score.add_argument("--solutions", required=True)
    score.add_argument("--output", required=True)
    score.set_defaults(handler=_score)

    score_gate = subparsers.add_parser("score-gate")
    score_gate.add_argument("--candidate-freeze", required=True)
    score_gate.add_argument("--solutions", required=True)
    score_gate.add_argument("--output", required=True)
    score_gate.set_defaults(handler=_score_gate)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
