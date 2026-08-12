"""Query-gold-blind recruitment plans for an expensive ARC provider.

The first policy is intentionally non-learned.  It orders tasks by exact,
integer-valued disagreement statistics from a frozen cheap-anchor population.
The plan is content-addressed before visual candidates or solutions are read;
post-hoc scoring only measures how much independently frozen provider complement
that order would have recovered at fixed activation counts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction

from afts_arc.experiment_safety import canonical_sha256
from afts_arc.grid import as_grid
from afts_arc.hypothesis_population import HYPOTHESIS_POPULATION_RESULT_SCHEMA
from afts_arc.nvarc_anchor import _validated_candidates as _validated_nvarc_candidates
from afts_arc.varc_run import VARC_RUN_RECEIPT_SCHEMA


FUNCTIONAL_RECRUITMENT_PLAN_SCHEMA = "afts.functional-recruitment-plan/v1"
FUNCTIONAL_RECRUITMENT_PLAN_WITH_ABSTENTIONS_SCHEMA = (
    "afts.functional-recruitment-plan/v2"
)
FUNCTIONAL_RECRUITMENT_RESULT_SCHEMA = "afts.functional-recruitment-result/v1"
FUNCTIONAL_RECRUITMENT_SUMMARY_SCHEMA = "afts.functional-recruitment-summary/v1"
POLICY_NAME = "anchor-disagreement-lexicographic/v1"
POLICY_WITH_ABSTENTIONS_NAME = "anchor-abstention-then-disagreement/v2"
RANDOM_POLICY_NAME = "content-hash-uniform-order/v1"


def _object(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field} keys must be strings")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be an array")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field} must be a non-empty string")
    return value


def _sha256(value: object, *, field: str) -> str:
    digest = _string(value, field=field)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _budget_counts(task_count: int, budget_percentages: Sequence[int]) -> dict[str, int]:
    if not budget_percentages:
        raise ValueError("at least one recruitment budget is required")
    normalized = tuple(budget_percentages)
    if any(type(value) is not int or not 1 <= value <= 100 for value in normalized):
        raise ValueError("budget percentages must be integers in [1, 100]")
    if normalized != tuple(sorted(set(normalized))):
        raise ValueError("budget percentages must be sorted and unique")
    return {
        str(percentage): (task_count * percentage + 99) // 100
        for percentage in normalized
    }


def _task_diagnostics(raw_queries: Sequence[object]) -> dict[str, int]:
    minimum_top_support = 10
    total_non_top_support = 0
    total_output_shape_count = 0
    total_unique_candidate_count = 0
    for raw_query in raw_queries:
        query = _object(raw_query, field="anchor query")
        candidates = _sequence(query["candidates"], field="anchor candidates")
        supports = []
        shapes = set()
        for raw_candidate in candidates:
            candidate = _object(raw_candidate, field="anchor candidate")
            ranks = _sequence(candidate["ranks"], field="anchor candidate ranks")
            supports.append(len(ranks))
            output = as_grid(candidate["output"])
            shapes.add((len(output), len(output[0])))
        top_support = max(supports) if supports else 0
        minimum_top_support = min(minimum_top_support, top_support)
        total_non_top_support += 10 - top_support
        total_output_shape_count += len(shapes)
        total_unique_candidate_count += len(candidates)
    return {
        "minimum_top_support": minimum_top_support,
        "query_count": len(raw_queries),
        "total_non_top_support": total_non_top_support,
        "total_output_shape_count": total_output_shape_count,
        "total_unique_candidate_count": total_unique_candidate_count,
    }


def _uncertainty_key(row: Mapping[str, object]) -> tuple[object, ...]:
    query_count = int(row["query_count"])
    return (
        int(row["minimum_top_support"]),
        -Fraction(int(row["total_non_top_support"]), query_count),
        -Fraction(int(row["total_unique_candidate_count"]), query_count),
        -Fraction(int(row["total_output_shape_count"]), query_count),
        str(row["task_id"]),
    )


def _uncertainty_key_with_abstentions(
    row: Mapping[str, object],
) -> tuple[object, ...]:
    if row["anchor_abstained"] is True:
        return (0, str(row["task_id"]))
    return (1, *_uncertainty_key(row))


def freeze_functional_recruitment_plan(
    *,
    anchor_freeze: Mapping[str, object],
    cohort_id: str,
    anchor_provider_name: str,
    recruited_provider_name: str,
    budget_percentages: Sequence[int],
    random_seed_count: int,
    source_files: Mapping[str, str],
    task_universe: Sequence[str] | None = None,
) -> dict[str, object]:
    """Freeze one anchor-only priority order and deterministic random controls."""

    cohort_id = _string(cohort_id, field="cohort_id")
    anchor_provider_name = _string(anchor_provider_name, field="anchor_provider_name")
    recruited_provider_name = _string(
        recruited_provider_name, field="recruited_provider_name"
    )
    if anchor_provider_name == recruited_provider_name:
        raise ValueError("anchor and recruited provider names must differ")
    if type(random_seed_count) is not int or not 1 <= random_seed_count <= 1024:
        raise ValueError("random_seed_count must be an integer in [1, 1024]")
    expected_source_fields = {
        "anchor_candidate_freeze",
        "anchor_cost_receipt",
        "protocol",
    }
    if task_universe is not None:
        expected_source_fields.add("task_universe_manifest")
    if set(source_files) != expected_source_fields:
        raise ValueError("recruitment plan source_files fields differ")
    normalized_sources = {
        name: _sha256(source_files[name], field=f"source_files[{name}]")
        for name in sorted(source_files)
    }
    if anchor_freeze["query_gold_read"] is not False:
        raise ValueError("anchor freeze is not query-gold-free")
    if anchor_freeze["controller_training_started"] is not False:
        raise ValueError("anchor freeze was produced after controller training")
    if anchor_freeze["public_evaluation_read"] is not False:
        raise ValueError("anchor freeze read public evaluation labels")
    anchor_cohort_id, tasks = _validated_nvarc_candidates(anchor_freeze)
    if anchor_cohort_id != cohort_id:
        raise ValueError("anchor freeze cohort differs from recruitment cohort")

    if task_universe is None:
        task_ids = tuple(sorted(tasks))
    else:
        task_ids = tuple(
            _string(task_id, field="task_universe task_id")
            for task_id in task_universe
        )
        if task_ids != tuple(sorted(set(task_ids))):
            raise ValueError("task_universe must be sorted and unique")
        if not set(tasks) <= set(task_ids):
            raise ValueError("anchor freeze contains tasks outside task_universe")

    diagnostic_rows = []
    for task_id in task_ids:
        if task_id in tasks:
            diagnostics = _task_diagnostics(tasks[task_id])
            if task_universe is None:
                diagnostic_rows.append({"task_id": task_id, **diagnostics})
            else:
                diagnostic_rows.append(
                    {"anchor_abstained": False, "task_id": task_id, **diagnostics}
                )
        else:
            diagnostic_rows.append(
                {
                    "anchor_abstained": True,
                    "minimum_top_support": 0,
                    "query_count": 0,
                    "task_id": task_id,
                    "total_non_top_support": 0,
                    "total_output_shape_count": 0,
                    "total_unique_candidate_count": 0,
                }
            )
    task_count = len(diagnostic_rows)
    if not task_count:
        raise ValueError("anchor freeze contains no tasks")
    budgets = _budget_counts(task_count, budget_percentages)
    priority_key = (
        _uncertainty_key
        if task_universe is None
        else _uncertainty_key_with_abstentions
    )
    priority_order = [
        str(row["task_id"]) for row in sorted(diagnostic_rows, key=priority_key)
    ]
    priority_rank = {task_id: rank for rank, task_id in enumerate(priority_order, start=1)}
    diagnostics_with_rank = [
        {**row, "priority_rank": priority_rank[str(row["task_id"])]}
        for row in diagnostic_rows
    ]
    random_orders = []
    for seed in range(random_seed_count):
        order = sorted(
            task_ids,
            key=lambda task_id: canonical_sha256(
                {"policy": RANDOM_POLICY_NAME, "seed": seed, "task_id": task_id}
            ),
        )
        random_orders.append({"seed": seed, "task_priority_order": order})

    has_task_universe = task_universe is not None
    body: dict[str, object] = {
        "anchor_candidate_freeze_id": anchor_freeze["freeze_id"],
        "anchor_provider_name": anchor_provider_name,
        "budget_task_counts": budgets,
        "cohort_id": cohort_id,
        "controller_training_started": False,
        "diagnostics": diagnostics_with_rank,
        "policy_name": (
            POLICY_WITH_ABSTENTIONS_NAME if has_task_universe else POLICY_NAME
        ),
        "priority_order": priority_order,
        "public_evaluation_read": False,
        "query_gold_read": False,
        "random_policy_name": RANDOM_POLICY_NAME,
        "random_priority_orders": random_orders,
        "recruited_provider_candidates_read": False,
        "recruited_provider_name": recruited_provider_name,
        "schema": (
            FUNCTIONAL_RECRUITMENT_PLAN_WITH_ABSTENTIONS_SCHEMA
            if has_task_universe
            else FUNCTIONAL_RECRUITMENT_PLAN_SCHEMA
        ),
        "source_files": normalized_sources,
        "task_count": task_count,
    }
    if has_task_universe:
        body["anchor_abstention_count"] = sum(
            int(bool(row["anchor_abstained"])) for row in diagnostic_rows
        )
    return {"plan_id": canonical_sha256(body), **body}


def _validated_plan(plan: Mapping[str, object]) -> tuple[tuple[str, ...], Mapping[str, int]]:
    body = dict(plan)
    declared_id = body.pop("plan_id")
    if declared_id != canonical_sha256(body):
        raise ValueError("plan_id does not match canonical recruitment plan content")
    if plan["schema"] not in {
        FUNCTIONAL_RECRUITMENT_PLAN_SCHEMA,
        FUNCTIONAL_RECRUITMENT_PLAN_WITH_ABSTENTIONS_SCHEMA,
    }:
        raise ValueError("unsupported functional-recruitment plan schema")
    for field in (
        "controller_training_started",
        "public_evaluation_read",
        "query_gold_read",
        "recruited_provider_candidates_read",
    ):
        if plan[field] is not False:
            raise ValueError(f"recruitment plan violates {field} boundary")
    order = tuple(
        _string(task_id, field="priority task_id")
        for task_id in _sequence(plan["priority_order"], field="priority_order")
    )
    if len(order) != len(set(order)) or plan["task_count"] != len(order):
        raise ValueError("recruitment priority order is inconsistent")
    raw_budgets = _object(plan["budget_task_counts"], field="budget_task_counts")
    budgets: dict[str, int] = {}
    previous = 0
    for percentage in sorted(raw_budgets, key=int):
        count = raw_budgets[percentage]
        if type(count) is not int or not previous <= count <= len(order):
            raise ValueError("recruitment budget task count is invalid")
        budgets[percentage] = count
        previous = count
    random_orders = _sequence(plan["random_priority_orders"], field="random orders")
    expected_seeds = list(range(len(random_orders)))
    observed_seeds = []
    for raw_random in random_orders:
        random_row = _object(raw_random, field="random order")
        observed_seeds.append(random_row["seed"])
        random_order = _sequence(random_row["task_priority_order"], field="random priority")
        if set(random_order) != set(order) or len(random_order) != len(order):
            raise ValueError("random priority order differs from task set")
    if observed_seeds != expected_seeds:
        raise ValueError("random priority seeds are not contiguous")
    return order, budgets


def _validated_population_result(
    result: Mapping[str, object],
) -> Mapping[str, Mapping[str, object]]:
    body = dict(result)
    declared_id = body.pop("result_id")
    if declared_id != canonical_sha256(body):
        raise ValueError("population result_id does not match result content")
    if result["schema"] != HYPOTHESIS_POPULATION_RESULT_SCHEMA:
        raise ValueError("unsupported hypothesis-population result schema")
    tasks: dict[str, Mapping[str, object]] = {}
    for raw_task in _sequence(result["tasks"], field="population result tasks"):
        task = _object(raw_task, field="population result task")
        task_id = _string(task["task_id"], field="task_id")
        if task_id in tasks:
            raise ValueError("population result task_id is duplicated")
        tasks[task_id] = task
    return tasks


def _validated_gpu_seconds(
    receipt: Mapping[str, object], *, expected_task_ids: set[str]
) -> dict[str, int]:
    body = dict(receipt)
    declared_id = body.pop("receipt_id")
    if declared_id != canonical_sha256(body):
        raise ValueError("recruited provider receipt_id does not match content")
    if receipt["schema"] != VARC_RUN_RECEIPT_SCHEMA:
        raise ValueError("unsupported recruited-provider run receipt schema")
    if receipt["controller_training_started"] is not False:
        raise ValueError("recruited provider receipt follows controller training")
    query_blind = _object(receipt["query_blind_protocol"], field="query_blind_protocol")
    if query_blind["query_gold_read"] is not False:
        raise ValueError("recruited provider run was not query-gold-blind")
    seconds: dict[str, int] = {}
    for raw_status in _sequence(receipt["task_statuses"], field="task statuses"):
        status = _object(raw_status, field="task status")
        task_id = _string(status["task_id"], field="status task_id")
        elapsed = status["elapsed_seconds"]
        if type(elapsed) is not int or elapsed < 0 or task_id in seconds:
            raise ValueError("task elapsed seconds are invalid or duplicated")
        seconds[task_id] = elapsed
    if set(seconds) != expected_task_ids:
        raise ValueError("recruited provider receipt task set differs from plan")
    return seconds


def _recovery_count(
    order: Sequence[str], count: int, marginal_task_ids: set[str]
) -> tuple[int, list[str]]:
    selected = list(order[:count])
    recovered = sorted(set(selected) & marginal_task_ids)
    return len(recovered), recovered


def score_functional_recruitment_plan(
    *,
    plan: Mapping[str, object],
    population_result: Mapping[str, object],
    recruited_provider_run_receipt: Mapping[str, object],
    recruited_provider_run_receipt_sha256: str,
) -> dict[str, object]:
    """Score a pre-frozen task order against independently frozen complement."""

    recruited_provider_run_receipt_sha256 = _sha256(
        recruited_provider_run_receipt_sha256,
        field="recruited_provider_run_receipt_sha256",
    )
    priority_order, budgets = _validated_plan(plan)
    population_tasks = _validated_population_result(population_result)
    task_ids = set(priority_order)
    if set(population_tasks) != task_ids:
        raise ValueError("recruitment plan and population result task sets differ")
    _string(population_result["population_id"], field="population_id")
    anchor_name = _string(plan["anchor_provider_name"], field="anchor provider")
    recruited_name = _string(
        plan["recruited_provider_name"], field="recruited provider"
    )
    anchor_coverage = 0
    full_union_coverage = 0
    marginal_task_ids: set[str] = set()
    for task_id in priority_order:
        task = population_tasks[task_id]
        provider_hits = _object(task["provider_hits"], field="provider hits")
        if anchor_name not in provider_hits or recruited_name not in provider_hits:
            raise ValueError("population result lacks a recruitment provider")
        anchor_hit = provider_hits[anchor_name]
        union_hit = task["union_hit"]
        if not isinstance(anchor_hit, bool) or not isinstance(union_hit, bool):
            raise TypeError("population task hits must be boolean")
        anchor_coverage += int(anchor_hit)
        full_union_coverage += int(union_hit)
        if union_hit and not anchor_hit:
            marginal_task_ids.add(task_id)
    gpu_seconds = _validated_gpu_seconds(
        recruited_provider_run_receipt, expected_task_ids=task_ids
    )
    all_provider_gpu_seconds = sum(gpu_seconds.values())

    random_orders = _sequence(plan["random_priority_orders"], field="random orders")
    budget_rows = []
    for percentage in budgets:
        count = budgets[percentage]
        recoveries, recovered_ids = _recovery_count(
            priority_order, count, marginal_task_ids
        )
        random_recoveries = []
        for raw_random in random_orders:
            random_row = _object(raw_random, field="random order")
            random_order = _sequence(
                random_row["task_priority_order"], field="random priority"
            )
            random_count, _ = _recovery_count(random_order, count, marginal_task_ids)
            random_recoveries.append(random_count)
        sorted_random = sorted(random_recoveries)
        middle = len(sorted_random) // 2
        median_low = sorted_random[(len(sorted_random) - 1) // 2]
        median_high = sorted_random[middle]
        selected = priority_order[:count]
        observed_seconds = sum(gpu_seconds[task_id] for task_id in selected)
        budget_rows.append(
            {
                "activation_percentage": int(percentage),
                "anchor_plus_recruitment_strict_coverage": anchor_coverage
                + recoveries,
                "incremental_recoveries": recoveries,
                "observed_recruited_gpu_seconds": observed_seconds,
                "observed_recruited_gpu_seconds_fraction": {
                    "denominator": all_provider_gpu_seconds,
                    "numerator": observed_seconds,
                },
                "random_recovery_distribution": {
                    "maximum": max(random_recoveries),
                    "mean": {
                        "denominator": len(random_recoveries),
                        "numerator": sum(random_recoveries),
                    },
                    "median_high": median_high,
                    "median_low": median_low,
                    "minimum": min(random_recoveries),
                    "recovery_counts": random_recoveries,
                },
                "recovered_marginal_task_ids": recovered_ids,
                "selected_task_count": count,
                "target_recall": {
                    "denominator": len(marginal_task_ids),
                    "numerator": recoveries,
                },
            }
        )

    primary = next(
        (row for row in budget_rows if row["activation_percentage"] == 30), None
    )
    gate_applicable = primary is not None and len(marginal_task_ids) >= 2
    gate_passed = False
    if primary is not None and gate_applicable:
        distribution = _object(
            primary["random_recovery_distribution"], field="random distribution"
        )
        recovery = int(primary["incremental_recoveries"])
        gate_passed = (
            recovery >= 2
            and recovery * 2 >= len(marginal_task_ids)
            and recovery > int(distribution["median_high"])
        )
    content: dict[str, object] = {
        "schema": FUNCTIONAL_RECRUITMENT_RESULT_SCHEMA,
        "plan_id": plan["plan_id"],
        "population_result_id": population_result["result_id"],
        "recruited_provider_run_receipt_sha256": recruited_provider_run_receipt_sha256,
        "query_gold_read": True,
        "anchor_strict_coverage": anchor_coverage,
        "full_population_union_strict_coverage": full_union_coverage,
        "marginal_task_count": len(marginal_task_ids),
        "marginal_task_ids": sorted(marginal_task_ids),
        "all_provider_gpu_seconds": all_provider_gpu_seconds,
        "budgets": budget_rows,
        "development_gate": {
            "applicable": gate_applicable,
            "minimum_30_percent_recoveries": 2,
            "passed": gate_passed,
            "requires_half_marginal_recall": True,
            "requires_exceeding_random_median_high": True,
        },
    }
    return {"result_id": canonical_sha256(content), **content}


def summarize_functional_recruitment_result(
    result: Mapping[str, object],
) -> dict[str, object]:
    """Remove task identities and random seed traces for compact publication."""

    if result["schema"] != FUNCTIONAL_RECRUITMENT_RESULT_SCHEMA:
        raise ValueError("unsupported functional-recruitment result schema")
    body = dict(result)
    declared_id = body.pop("result_id")
    if declared_id != canonical_sha256(body):
        raise ValueError("functional-recruitment result_id does not match content")
    compact_budgets = []
    for raw_budget in _sequence(result["budgets"], field="budgets"):
        budget = dict(_object(raw_budget, field="budget"))
        budget.pop("recovered_marginal_task_ids")
        random_distribution = dict(
            _object(budget["random_recovery_distribution"], field="random distribution")
        )
        random_distribution.pop("recovery_counts")
        budget["random_recovery_distribution"] = random_distribution
        compact_budgets.append(budget)
    content: dict[str, object] = {
        "schema": FUNCTIONAL_RECRUITMENT_SUMMARY_SCHEMA,
        "full_result_id_commitment": declared_id,
        "plan_id": result["plan_id"],
        "population_result_id": result["population_result_id"],
        "query_gold_read": True,
        "task_level_outcomes_exposed": False,
        "anchor_strict_coverage": result["anchor_strict_coverage"],
        "full_population_union_strict_coverage": result[
            "full_population_union_strict_coverage"
        ],
        "marginal_task_count": result["marginal_task_count"],
        "all_provider_gpu_seconds": result["all_provider_gpu_seconds"],
        "budgets": compact_budgets,
        "development_gate": result["development_gate"],
    }
    return {"summary_id": canonical_sha256(content), **content}
