"""Freeze and score anchor-only functional recruitment orders."""

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
from afts_arc.functional_recruitment import (  # noqa: E402
    freeze_functional_recruitment_plan,
    score_functional_recruitment_plan,
    summarize_functional_recruitment_result,
)


def _load_object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain an object")
    return value


def _write_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace artifact: {path}")
    atomic_write_json(path, value)


def _freeze(arguments: argparse.Namespace) -> None:
    anchor_freeze_path = arguments.anchor_freeze.resolve()
    anchor_receipt_path = arguments.anchor_receipt.resolve()
    protocol_path = arguments.protocol.resolve()
    output = arguments.output.resolve()
    source_files = {
        "anchor_candidate_freeze": file_sha256(anchor_freeze_path),
        "anchor_cost_receipt": file_sha256(anchor_receipt_path),
        "protocol": file_sha256(protocol_path),
    }
    task_universe = None
    if arguments.task_universe is not None:
        raw_task_universe = json.loads(
            arguments.task_universe.resolve().read_text(encoding="utf-8")
        )
        if not isinstance(raw_task_universe, list):
            raise TypeError("task-universe artifact must contain an array")
        task_universe = raw_task_universe
        source_files["task_universe_manifest"] = file_sha256(
            arguments.task_universe.resolve()
        )
    payload = freeze_functional_recruitment_plan(
        anchor_freeze=_load_object(anchor_freeze_path),
        cohort_id=arguments.cohort_id,
        anchor_provider_name=arguments.anchor_provider_name,
        recruited_provider_name=arguments.recruited_provider_name,
        budget_percentages=arguments.budget_percent,
        random_seed_count=arguments.random_seed_count,
        source_files=source_files,
        task_universe=task_universe,
    )
    _write_new(output, payload)
    print(
        json.dumps(
            {
                "budget_task_counts": payload["budget_task_counts"],
                "output": str(output),
                "plan_id": payload["plan_id"],
            }
        )
    )


def _score(arguments: argparse.Namespace) -> None:
    plan_path = arguments.plan.resolve()
    population_result_path = arguments.population_result.resolve()
    recruited_receipt_path = arguments.recruited_provider_receipt.resolve()
    output = arguments.output.resolve()
    payload = score_functional_recruitment_plan(
        plan=_load_object(plan_path),
        population_result=_load_object(population_result_path),
        recruited_provider_run_receipt=_load_object(recruited_receipt_path),
        recruited_provider_run_receipt_sha256=file_sha256(recruited_receipt_path),
    )
    _write_new(output, payload)
    response: dict[str, object] = {
        "budgets": payload["budgets"],
        "development_gate": payload["development_gate"],
        "output": str(output),
        "result_id": payload["result_id"],
    }
    if arguments.summary_output is not None:
        summary_output = arguments.summary_output.resolve()
        summary = summarize_functional_recruitment_result(payload)
        _write_new(summary_output, summary)
        response["summary_id"] = summary["summary_id"]
        response["summary_output"] = str(summary_output)
    print(json.dumps(response))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--anchor-freeze", type=Path, required=True)
    freeze.add_argument("--anchor-receipt", type=Path, required=True)
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--task-universe", type=Path)
    freeze.add_argument("--cohort-id", required=True)
    freeze.add_argument("--anchor-provider-name", required=True)
    freeze.add_argument("--recruited-provider-name", required=True)
    freeze.add_argument(
        "--budget-percent", action="append", type=int, required=True
    )
    freeze.add_argument("--random-seed-count", type=int, default=256)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.set_defaults(handler=_freeze)

    score = subparsers.add_parser("score")
    score.add_argument("--plan", type=Path, required=True)
    score.add_argument("--population-result", type=Path, required=True)
    score.add_argument("--recruited-provider-receipt", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--summary-output", type=Path)
    score.set_defaults(handler=_score)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
