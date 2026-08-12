"""Score frozen recruitment and random orders under the same native GPU budget."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


def _load_object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be an array")
    return value


def _prefix_within_cost(
    order: Sequence[str], *, costs: Mapping[str, int], budget: int
) -> tuple[list[str], int]:
    selected: list[str] = []
    consumed = 0
    for task_id in order:
        next_cost = costs[task_id]
        if consumed + next_cost > budget:
            break
        selected.append(task_id)
        consumed += next_cost
    return selected, consumed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--population-result", type=Path, required=True)
    parser.add_argument("--recruited-provider-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    project_root = arguments.project_root.resolve()
    sys.path.insert(0, str(project_root / "src"))
    from afts_arc.experiment_safety import (  # noqa: PLC0415
        atomic_write_json,
        canonical_sha256,
        file_sha256,
    )
    from afts_arc.functional_recruitment import (  # noqa: PLC0415
        _validated_gpu_seconds,
        _validated_plan,
        _validated_population_result,
    )

    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace artifact: {output}")
    plan_path = arguments.plan.resolve()
    result_path = arguments.population_result.resolve()
    receipt_path = arguments.recruited_provider_receipt.resolve()
    plan = _load_object(plan_path)
    population_result = _load_object(result_path)
    receipt = _load_object(receipt_path)
    priority_order, budgets = _validated_plan(plan)
    population_tasks = _validated_population_result(population_result)
    task_ids = set(priority_order)
    if set(population_tasks) != task_ids:
        raise ValueError("plan and population result task sets differ")
    costs = _validated_gpu_seconds(receipt, expected_task_ids=task_ids)
    anchor_name = str(plan["anchor_provider_name"])
    marginal_task_ids = {
        task_id
        for task_id in priority_order
        if population_tasks[task_id]["union_hit"] is True
        and population_tasks[task_id]["provider_hits"][anchor_name] is False
    }
    primary_count = budgets["30"]
    fixed_selected = list(priority_order[:primary_count])
    fixed_budget = sum(costs[task_id] for task_id in fixed_selected)
    fixed_recoveries = len(set(fixed_selected) & marginal_task_ids)

    random_rows: list[dict[str, object]] = []
    for raw_order in _sequence(plan["random_priority_orders"], field="random orders"):
        order = raw_order["task_priority_order"]
        if not isinstance(order, list):
            raise TypeError("random task priority order must be an array")
        selected, consumed = _prefix_within_cost(
            order, costs=costs, budget=fixed_budget
        )
        random_rows.append(
            {
                "consumed_gpu_seconds": consumed,
                "recovery_count": len(set(selected) & marginal_task_ids),
                "seed": raw_order["seed"],
                "selected_task_count": len(selected),
            }
        )
    recovery_counts = sorted(int(row["recovery_count"]) for row in random_rows)
    selected_counts = sorted(int(row["selected_task_count"]) for row in random_rows)
    middle = len(recovery_counts) // 2
    body: dict[str, object] = {
        "comparison": {
            "fixed_strictly_exceeds_random_median_high": (
                fixed_recoveries > recovery_counts[middle]
            ),
            "random_maximum_recoveries": max(recovery_counts),
            "random_mean_recoveries": {
                "denominator": len(recovery_counts),
                "numerator": sum(recovery_counts),
            },
            "random_median_high_recoveries": recovery_counts[middle],
            "random_median_low_recoveries": recovery_counts[
                (len(recovery_counts) - 1) // 2
            ],
            "random_selected_task_count_range": [
                min(selected_counts),
                max(selected_counts),
            ],
        },
        "fixed_policy": {
            "activation_percentage": 30,
            "gpu_seconds": fixed_budget,
            "recovery_count": fixed_recoveries,
            "selected_task_count": primary_count,
        },
        "marginal_task_count": len(marginal_task_ids),
        "plan_id": plan["plan_id"],
        "population_result_id": population_result["result_id"],
        "query_gold_read": True,
        "random_orders": random_rows,
        "recruited_provider_receipt_sha256": file_sha256(receipt_path),
        "schema": "afts.functional-recruitment-equal-native-cost/v1",
        "source_files": {
            "plan": file_sha256(plan_path),
            "population_result": file_sha256(result_path),
            "scorer": file_sha256(Path(__file__).resolve()),
        },
        "truncation_rule": "longest random-order prefix with cumulative task GPU seconds not exceeding the fixed 30-task policy cost",
    }
    artifact = {"comparison_id": canonical_sha256(body), **body}
    atomic_write_json(output, artifact)
    print(
        json.dumps(
            {
                "comparison": artifact["comparison"],
                "comparison_id": artifact["comparison_id"],
                "fixed_policy": artifact["fixed_policy"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
