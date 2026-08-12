"""Freeze and score a heterogeneous, content-addressed hypothesis population."""

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
from afts_arc.hypothesis_population import (  # noqa: E402
    freeze_hypothesis_population,
    score_hypothesis_population,
    summarize_hypothesis_population_result,
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


def _provider_arguments(
    rows: list[list[str]], receipts: list[list[str]]
) -> tuple[
    dict[str, Mapping[str, object]],
    dict[str, str],
    dict[str, str],
    dict[str, str],
]:
    provider_paths: dict[str, Path] = {}
    provider_families: dict[str, str] = {}
    for name, family, raw_path in rows:
        if name in provider_paths:
            raise ValueError(f"duplicate provider: {name}")
        provider_paths[name] = Path(raw_path).resolve()
        provider_families[name] = family
    receipt_paths: dict[str, Path] = {}
    for name, raw_path in receipts:
        if name in receipt_paths:
            raise ValueError(f"duplicate provider receipt: {name}")
        receipt_paths[name] = Path(raw_path).resolve()
    if set(receipt_paths) != set(provider_paths):
        raise ValueError("provider receipt set differs from candidate-freeze set")
    freezes = {name: _load_object(path) for name, path in provider_paths.items()}
    freeze_hashes = {name: file_sha256(path) for name, path in provider_paths.items()}
    receipt_hashes = {name: file_sha256(path) for name, path in receipt_paths.items()}
    return freezes, provider_families, freeze_hashes, receipt_hashes


def _freeze(arguments: argparse.Namespace) -> None:
    freezes, families, freeze_hashes, receipt_hashes = _provider_arguments(
        arguments.provider, arguments.receipt
    )
    output = arguments.output.resolve()
    payload = freeze_hypothesis_population(
        cohort_id=arguments.cohort_id,
        provider_freezes=freezes,
        provider_families=families,
        provider_freeze_sha256s=freeze_hashes,
        provider_receipt_sha256s=receipt_hashes,
    )
    _write_new(output, payload)
    print(
        json.dumps(
            {
                "aggregate": payload["aggregate"],
                "output": str(output),
                "population_id": payload["population_id"],
            }
        )
    )


def _score(arguments: argparse.Namespace) -> None:
    population_path = arguments.population.resolve()
    solutions_path = arguments.solutions.resolve()
    output = arguments.output.resolve()
    payload = score_hypothesis_population(
        population=_load_object(population_path),
        solutions=_load_object(solutions_path),
        solution_source_sha256=file_sha256(solutions_path),
    )
    _write_new(output, payload)
    response: dict[str, object] = {
        "metrics": payload["metrics"],
        "output": str(output),
        "result_id": payload["result_id"],
    }
    if arguments.summary_output is not None:
        summary_output = arguments.summary_output.resolve()
        summary = summarize_hypothesis_population_result(payload)
        _write_new(summary_output, summary)
        response["summary_id"] = summary["summary_id"]
        response["summary_output"] = str(summary_output)
    print(json.dumps(response))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--cohort-id", required=True)
    freeze.add_argument(
        "--provider",
        action="append",
        nargs=3,
        metavar=("NAME", "FAMILY", "CANDIDATE_FREEZE"),
        required=True,
    )
    freeze.add_argument(
        "--receipt",
        action="append",
        nargs=2,
        metavar=("NAME", "COST_RECEIPT"),
        required=True,
    )
    freeze.add_argument("--output", type=Path, required=True)
    freeze.set_defaults(handler=_freeze)

    score = subparsers.add_parser("score")
    score.add_argument("--population", type=Path, required=True)
    score.add_argument("--solutions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--summary-output", type=Path)
    score.set_defaults(handler=_score)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
