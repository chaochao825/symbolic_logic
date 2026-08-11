"""Freeze or score the visual relational-transducer allocation gate."""

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
from afts_arc.relational_transducer_gate import (  # noqa: E402
    audit_relational_transducer_allocation_ablation,
    freeze_relational_transducer_gate,
    score_relational_transducer_allocation_ablation,
    score_relational_transducer_gate,
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
    reserved = {
        "challenges",
        "cost_amendment",
        "protocol",
        "recursive_freeze",
        "visual_freeze",
    }
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
    recursive = arguments.recursive_freeze.resolve()
    protocol = arguments.protocol.resolve()
    cost_amendment = arguments.cost_amendment.resolve()
    output = arguments.output.resolve()
    payload = freeze_relational_transducer_gate(
        challenges=_load_object(challenges),
        visual_freeze=_load_object(visual),
        recursive_freeze=_load_object(recursive),
        cohort_id=arguments.cohort_id,
        source_files={
            "challenges": file_sha256(challenges),
            "cost_amendment": file_sha256(cost_amendment),
            "protocol": file_sha256(protocol),
            "recursive_freeze": file_sha256(recursive),
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
    payload = score_relational_transducer_gate(
        freeze=_load_object(candidate_freeze),
        solutions=_load_object(solutions),
        solution_source_sha256=file_sha256(solutions),
    )
    _write_new(output, payload)
    print(json.dumps({"output": str(output), "result_id": payload["result_id"]}))


def _ablate(arguments: argparse.Namespace) -> None:
    candidate_freeze = arguments.candidate_freeze.resolve()
    protocol = arguments.protocol.resolve()
    output = arguments.output.resolve()
    payload = audit_relational_transducer_allocation_ablation(
        freeze=_load_object(candidate_freeze),
        protocol_sha256=file_sha256(protocol),
    )
    _write_new(output, payload)
    print(json.dumps({"ablation_id": payload["ablation_id"], "output": str(output)}))


def _score_ablation(arguments: argparse.Namespace) -> None:
    candidate_freeze = arguments.candidate_freeze.resolve()
    allocation_ablation = arguments.allocation_ablation.resolve()
    solutions = arguments.solutions.resolve()
    output = arguments.output.resolve()
    payload = score_relational_transducer_allocation_ablation(
        freeze=_load_object(candidate_freeze),
        ablation=_load_object(allocation_ablation),
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
    freeze.add_argument("--recursive-freeze", type=Path, required=True)
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--cost-amendment", type=Path, required=True)
    freeze.add_argument("--cohort-id", required=True)
    freeze.add_argument("--additional-source", action="append", default=[])
    freeze.add_argument("--output", type=Path, required=True)
    freeze.set_defaults(handler=_freeze)

    score = subparsers.add_parser("score")
    score.add_argument("--candidate-freeze", type=Path, required=True)
    score.add_argument("--solutions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(handler=_score)

    ablate = subparsers.add_parser("ablate-allocation")
    ablate.add_argument("--candidate-freeze", type=Path, required=True)
    ablate.add_argument("--protocol", type=Path, required=True)
    ablate.add_argument("--output", type=Path, required=True)
    ablate.set_defaults(handler=_ablate)

    score_ablation = subparsers.add_parser("score-allocation-ablation")
    score_ablation.add_argument("--candidate-freeze", type=Path, required=True)
    score_ablation.add_argument("--allocation-ablation", type=Path, required=True)
    score_ablation.add_argument("--solutions", type=Path, required=True)
    score_ablation.add_argument("--output", type=Path, required=True)
    score_ablation.set_defaults(handler=_score_ablation)
    arguments = parser.parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
