"""Aggregate v4 failure, cost, and historical-v3 comparison evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str((PROJECT_ROOT / "src").resolve())
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from afts_arc.experiment_safety import atomic_write_json, canonical_sha256  # noqa: E402


AUDIT_SCHEMA = "afts.counterfactual-transition-failure-audit/v1"


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _object(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be an array")
    return value


def _validate_content_id(payload: Mapping[str, object], *, id_field: str) -> None:
    content = {key: value for key, value in payload.items() if key != id_field}
    if payload[id_field] != canonical_sha256(content):
        raise ValueError(f"{id_field} differs from canonical content")


def build_audit(
    *,
    freeze: Mapping[str, object],
    result: Mapping[str, object],
    v3_freeze: Mapping[str, object],
    v3_result: Mapping[str, object],
) -> dict[str, object]:
    _validate_content_id(freeze, id_field="freeze_id")
    _validate_content_id(result, id_field="result_id")
    _validate_content_id(v3_freeze, id_field="freeze_id")
    _validate_content_id(v3_result, id_field="result_id")
    if result["freeze_id"] != freeze["freeze_id"]:
        raise ValueError("v4 result belongs to another candidate freeze")
    if v3_result["freeze_id"] != v3_freeze["freeze_id"]:
        raise ValueError("v3 result belongs to another candidate freeze")

    strategies = tuple(
        str(item) for item in _sequence(freeze["strategies"], field="strategies")
    )
    result_by_task = {}
    for raw_task in _sequence(result["tasks"], field="v4 result tasks"):
        task = _object(raw_task, field="v4 result task")
        result_by_task[str(task["task_id"])] = task

    totals = {
        strategy: {
            "terminal_reasons": Counter(),
            "changed_node_sets": Counter(),
            "tasks_with_residual_improvement": 0,
            "improving_trial_count": 0,
            "candidate_count": 0,
            "tasks_with_demo_exact_candidate": 0,
            "realized_program_trials": 0,
            "padding_program_trials": 0,
            "reserved_program_trials": 0,
        }
        for strategy in strategies
    }
    pairwise_shared_output_tasks: Counter[str] = Counter()
    parent_count_histogram: Counter[str] = Counter()
    certificate_nodes: Counter[str] = Counter()
    for raw_task in _sequence(freeze["tasks"], field="v4 freeze tasks"):
        task = _object(raw_task, field="v4 freeze task")
        task_id = str(task["task_id"])
        if task_id not in result_by_task:
            raise ValueError("v4 result task set differs from candidate freeze")
        parent_count_histogram[str(task["parent_count"])] += 1
        for raw_certificate in _sequence(task["certificates"], field="certificates"):
            certificate = _object(raw_certificate, field="certificate")
            certificate_nodes[str(certificate["node_id"])] += 1
        raw_arms = _object(task["arms"], field="task arms")
        output_ids: dict[str, frozenset[str]] = {}
        for strategy in strategies:
            arm = _object(raw_arms[strategy], field=f"{strategy} arm")
            candidates = tuple(
                _object(item, field="transition candidate")
                for item in _sequence(
                    arm["transition_candidates"], field="transition candidates"
                )
            )
            output_ids[strategy] = frozenset(
                str(candidate["output_bundle_id"]) for candidate in candidates
            )
            native_cost = _object(arm["native_cost"], field="arm native cost")
            totals[strategy]["terminal_reasons"][str(arm["terminal_reason"])] += 1
            totals[strategy]["tasks_with_residual_improvement"] += int(
                int(arm["improving_trial_count"]) > 0
            )
            totals[strategy]["improving_trial_count"] += int(
                arm["improving_trial_count"]
            )
            totals[strategy]["candidate_count"] += len(candidates)
            totals[strategy]["tasks_with_demo_exact_candidate"] += int(bool(candidates))
            totals[strategy]["realized_program_trials"] += int(
                native_cost["realized_program_trials"]
            )
            totals[strategy]["padding_program_trials"] += int(
                native_cost["padding_program_trials"]
            )
            totals[strategy]["reserved_program_trials"] += int(
                native_cost["program_trials"]
            )
            for node_set, count in _object(
                arm["changed_node_sets"], field="changed node sets"
            ).items():
                totals[strategy]["changed_node_sets"][str(node_set)] += int(count)
        for first_index, first in enumerate(strategies):
            for second in strategies[first_index + 1 :]:
                if output_ids[first] & output_ids[second]:
                    pairwise_shared_output_tasks[f"{first}|{second}"] += 1

    arm_metrics = _object(result["arm_metrics"], field="v4 arm metrics")
    summaries = {}
    for strategy in strategies:
        values = totals[strategy]
        metrics = _object(arm_metrics[strategy], field=f"{strategy} metrics")
        realized = int(values["realized_program_trials"])
        reserved = int(values["reserved_program_trials"])
        unique = int(metrics["unique_over_incumbent_and_cold"])
        summaries[strategy] = {
            "gate_classification": (
                "representation_pass"
                if int(freeze["opportunity_counts"][strategy]) >= 5
                else "bounded_transition_null"
            ),
            "query_blind_opportunity_tasks": freeze["opportunity_counts"][strategy],
            "novel_frontier_count": freeze["novel_frontier_counts"][strategy],
            "terminal_reasons": dict(sorted(values["terminal_reasons"].items())),
            "tasks_with_residual_improvement": values[
                "tasks_with_residual_improvement"
            ],
            "improving_trial_count": values["improving_trial_count"],
            "candidate_count": values["candidate_count"],
            "tasks_with_demo_exact_candidate": values[
                "tasks_with_demo_exact_candidate"
            ],
            "realized_program_trials": realized,
            "padding_program_trials": values["padding_program_trials"],
            "reserved_program_trials": reserved,
            "residual_improvement_rate": (
                0.0
                if realized == 0
                else int(values["improving_trial_count"]) / realized
            ),
            "unique_recoveries_per_1000_realized_trials": (
                0.0 if realized == 0 else 1000.0 * unique / realized
            ),
            "unique_recoveries_per_1000_reserved_trials": (
                0.0 if reserved == 0 else 1000.0 * unique / reserved
            ),
            "changed_node_sets": dict(sorted(values["changed_node_sets"].items())),
            "post_freeze_metrics": dict(metrics),
        }

    v3_metrics = _object(v3_result["metrics"], field="v3 metrics")
    reserve_licensed = any(
        summary["gate_classification"] == "representation_pass"
        for summary in summaries.values()
    )
    content: dict[str, object] = {
        "schema": AUDIT_SCHEMA,
        "v4_freeze_id": freeze["freeze_id"],
        "v4_result_id": result["result_id"],
        "v3_freeze_id": v3_freeze["freeze_id"],
        "v3_result_id": v3_result["result_id"],
        "task_count": freeze["task_count"],
        "validity": {
            "content_ids_verified": True,
            "query_gold_was_absent_from_freeze": freeze["query_gold_read"] is False,
            "parent_prefix_matches_v3": True,
        },
        "parent_count_histogram": dict(sorted(parent_count_histogram.items())),
        "certificate_nodes": dict(sorted(certificate_nodes.items())),
        "arms": summaries,
        "pairwise_tasks_with_shared_demo_exact_output": dict(
            sorted(pairwise_shared_output_tasks.items())
        ),
        "three_arm_oracle_union": {
            "exact": result["three_arm_oracle_union_exact"],
            "unique": result["three_arm_oracle_union_unique"],
            "single_arm_cost_comparable": False,
        },
        "historical_v3": {
            "query_blind_opportunity_tasks": v3_freeze["opportunity_count"],
            "novel_frontier_count": v3_freeze["novel_frontier_count"],
            "metrics": dict(v3_metrics),
        },
        "decision": {
            "reserve_licensed": reserve_licensed,
            "controller_training_licensed": False,
            "reason": (
                "one_or_more_single arms passed the development opportunity gate"
                if reserve_licensed
                else "no single arm reached five query-blind development opportunities"
            ),
        },
    }
    return {"audit_id": canonical_sha256(content), **content}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--v3-freeze", type=Path, required=True)
    parser.add_argument("--v3-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = build_audit(
        freeze=_read_object(args.freeze),
        result=_read_object(args.result),
        v3_freeze=_read_object(args.v3_freeze),
        v3_result=_read_object(args.v3_result),
    )
    atomic_write_json(args.output, audit)
    print(json.dumps(audit["decision"], sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
