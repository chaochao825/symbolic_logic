"""Freeze or score the Object–Program Workspace v1 development gate."""

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
from afts_arc.object_program_workspace_gate import (  # noqa: E402
    freeze_object_program_workspace_gate,
    score_object_program_workspace_gate,
)


def _load_object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain an object")
    return value


def _write_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace artifact: {path}")
    atomic_write_json(path, payload)


def _additional_source_hashes(values: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    reserved = {"challenges", "protocol", "visual_freeze"}
    for value in values:
        if "=" not in value:
            raise ValueError("additional source must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        if not name or not raw_path or name in reserved or name in hashes:
            raise ValueError("additional source name or path is invalid")
        hashes[name] = file_sha256(Path(raw_path).resolve())
    return dict(sorted(hashes.items()))


def _freeze(arguments: argparse.Namespace) -> None:
    challenges = arguments.challenges.resolve()
    visual = arguments.visual_freeze.resolve()
    protocol = arguments.protocol.resolve()
    output = arguments.output.resolve()
    payload = freeze_object_program_workspace_gate(
        challenges=_load_object(challenges),
        visual_freeze=_load_object(visual),
        cohort_id=arguments.cohort_id,
        scientific_lane=arguments.scientific_lane,
        source_files={
            "challenges": file_sha256(challenges),
            "protocol": file_sha256(protocol),
            "visual_freeze": file_sha256(visual),
            **_additional_source_hashes(arguments.additional_source),
        },
    )
    _write_new(output, payload)
    print(json.dumps({"freeze_id": payload["freeze_id"], "output": str(output)}))


def _score(arguments: argparse.Namespace) -> None:
    candidate_freeze = arguments.candidate_freeze.resolve()
    solutions = arguments.solutions.resolve()
    output = arguments.output.resolve()
    payload = score_object_program_workspace_gate(
        freeze=_load_object(candidate_freeze),
        solutions=_load_object(solutions),
        solution_source_sha256=file_sha256(solutions),
    )
    _write_new(output, payload)
    print(json.dumps({"output": str(output), "result_id": payload["result_id"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--challenges", type=Path, required=True)
    freeze.add_argument("--visual-freeze", type=Path, required=True)
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--cohort-id", required=True)
    freeze.add_argument(
        "--scientific-lane",
        choices=("controlled_semantic", "outcome_exposed_development"),
        required=True,
    )
    freeze.add_argument("--additional-source", action="append", default=[])
    freeze.add_argument("--output", type=Path, required=True)
    freeze.set_defaults(handler=_freeze)

    score = subparsers.add_parser("score")
    score.add_argument("--candidate-freeze", type=Path, required=True)
    score.add_argument("--solutions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(handler=_score)

    arguments = parser.parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
