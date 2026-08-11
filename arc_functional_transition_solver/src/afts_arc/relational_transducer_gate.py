"""Freeze and score a budget-matched visual relational-transducer gate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from afts_arc.blind import BlindTask
from afts_arc.experiment_safety import canonical_sha256
from afts_arc.grid import Grid, as_grid, grid_to_lists
from afts_arc.relational_transducer import (
    STRUCTURAL_HYPOTHESIS_COUNT,
    allocation_count,
    build_visual_transducer_certificate,
    cold_allocation_key,
    execute_relational_transducer,
    synthesize_relational_transducers,
    visual_allocation_key,
)
from afts_arc.task import ARCPair


TRANSDUCER_GATE_FREEZE_SCHEMA = "afts.visual-relational-transducer-freeze/v1"
TRANSDUCER_GATE_RESULT_SCHEMA = "afts.visual-relational-transducer-result/v1"
TRANSDUCER_ALLOCATION_ABLATION_SCHEMA = (
    "afts.visual-relational-transducer-allocation-ablation/v1"
)
TRANSDUCER_ALLOCATION_OUTCOME_SCHEMA = (
    "afts.visual-relational-transducer-allocation-outcome/v1"
)
ARM_NAMES = ("visual_typed", "cold_restart", "static_family")


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


def _blind_task(raw_task: Mapping[str, object]) -> BlindTask:
    train = _sequence(raw_task["train"], field="challenge train")
    test = _sequence(raw_task["test"], field="challenge test")
    pairs = []
    for raw_pair in train:
        pair = _object(raw_pair, field="challenge demonstration")
        if set(pair) != {"input", "output"}:
            raise ValueError("challenge demonstration fields differ")
        pairs.append(ARCPair(as_grid(pair["input"]), as_grid(pair["output"])))
    queries = []
    for raw_query in test:
        query = _object(raw_query, field="challenge query")
        if set(query) != {"input"}:
            raise ValueError("challenge query must contain input only")
        queries.append(as_grid(query["input"]))
    return BlindTask.from_observations(train=tuple(pairs), test_inputs=tuple(queries))


def _provider_task_map(freeze: Mapping[str, object], *, provider: str) -> dict[str, object]:
    if freeze["query_gold_read"] is not False:
        raise ValueError(f"{provider} candidate freeze is not query-gold-free")
    tasks: dict[str, object] = {}
    for raw_task in _sequence(freeze["tasks"], field=f"{provider}.tasks"):
        task = _object(raw_task, field=f"{provider} task")
        task_id = task["task_id"]
        if not isinstance(task_id, str) or not task_id or task_id in tasks:
            raise ValueError(f"{provider} task ID is invalid or duplicated")
        tasks[task_id] = task
    return tasks


def _visual_query(raw_query: object) -> tuple[tuple[tuple[Grid, int], ...], set[str]]:
    query = _object(raw_query, field="visual query")
    weighted = []
    output_ids = set()
    for raw_candidate in _sequence(query["candidates"], field="visual candidates"):
        candidate = _object(raw_candidate, field="visual candidate")
        output = as_grid(candidate["output"])
        weight = candidate["sample_count"]
        if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
            raise ValueError("visual sample count must be a positive integer")
        weighted.append((output, weight))
        output_ids.add(canonical_sha256(grid_to_lists(output)))
    if not weighted:
        raise ValueError("visual query has no valid posterior candidates")
    return tuple(weighted), output_ids


def _recursive_query(raw_query: object) -> set[str]:
    query = _object(raw_query, field="recursive query")
    outputs = set()
    for raw_candidate in _sequence(query["candidates"], field="recursive candidates"):
        candidate = _object(raw_candidate, field="recursive candidate")
        output = as_grid(candidate["output"])
        outputs.add(canonical_sha256(grid_to_lists(output)))
    return outputs


def _native_cost(
    *, selected_task_count: int, maximum_demo_count: int, maximum_query_count: int
) -> dict[str, int]:
    return {
        "selected_task_count": selected_task_count,
        "charged_program_trials": selected_task_count * STRUCTURAL_HYPOTHESIS_COUNT,
        "charged_demo_executions": selected_task_count
        * STRUCTURAL_HYPOTHESIS_COUNT
        * maximum_demo_count,
        "charged_query_executions": selected_task_count
        * STRUCTURAL_HYPOTHESIS_COUNT
        * maximum_query_count,
    }


def freeze_relational_transducer_gate(
    *,
    challenges: Mapping[str, object],
    visual_freeze: Mapping[str, object],
    recursive_freeze: Mapping[str, object],
    cohort_id: str,
    source_files: Mapping[str, str],
) -> dict[str, object]:
    """Freeze query-blind certificates, allocations, programs, and outputs."""

    if not cohort_id:
        raise ValueError("cohort_id must not be empty")
    if not source_files or not all(source_files[name] for name in source_files):
        raise ValueError("transducer source hashes must not be empty")
    task_ids = sorted(challenges)
    if not task_ids:
        raise ValueError("transducer gate requires tasks")
    visual_tasks = _provider_task_map(visual_freeze, provider="visual")
    recursive_tasks = _provider_task_map(recursive_freeze, provider="recursive")
    for provider, tasks in (("visual", visual_tasks), ("recursive", recursive_tasks)):
        missing = sorted(set(task_ids) - set(tasks))
        if missing:
            raise ValueError(f"{provider} freeze is missing cohort tasks: {missing}")

    blind_tasks: dict[str, BlindTask] = {}
    base_outputs: dict[str, tuple[tuple[str, ...], ...]] = {}
    certificates = []
    maximum_demo_count = 0
    maximum_query_count = 0
    for task_id in task_ids:
        raw_task = _object(challenges[task_id], field=f"challenge[{task_id}]")
        if set(raw_task) != {"train", "test"}:
            raise ValueError("challenge task fields differ")
        task = _blind_task(raw_task)
        blind_tasks[task_id] = task
        maximum_demo_count = max(maximum_demo_count, len(task.train))
        maximum_query_count = max(maximum_query_count, len(task.test_inputs))
        visual_task = _object(visual_tasks[task_id], field="visual task")
        recursive_task = _object(recursive_tasks[task_id], field="recursive task")
        visual_queries = _sequence(visual_task["queries"], field="visual task queries")
        recursive_queries = _sequence(
            recursive_task["queries"], field="recursive task queries"
        )
        if not (
            len(visual_queries) == len(recursive_queries) == len(task.test_inputs)
        ):
            raise ValueError("provider and challenge query counts differ")
        posterior_queries = []
        task_base_outputs = []
        visual_unique_count = 0
        recursive_unique_count = 0
        overlap_count = 0
        for visual_query, recursive_query in zip(
            visual_queries, recursive_queries, strict=True
        ):
            weighted, visual_ids = _visual_query(visual_query)
            recursive_ids = _recursive_query(recursive_query)
            posterior_queries.append(weighted)
            task_base_outputs.append(tuple(sorted(visual_ids | recursive_ids)))
            visual_unique_count += len(visual_ids)
            recursive_unique_count += len(recursive_ids)
            overlap_count += len(visual_ids & recursive_ids)
        base_outputs[task_id] = tuple(task_base_outputs)
        certificates.append(
            build_visual_transducer_certificate(
                task_id=task_id,
                query_inputs=task.test_inputs,
                posterior_queries=tuple(posterior_queries),
                visual_unique_candidate_count=visual_unique_count,
                recursive_unique_candidate_count=recursive_unique_count,
                exact_pool_overlap_count=overlap_count,
            )
        )

    selected_count = allocation_count(len(task_ids))
    visual_selected = tuple(
        certificate["task_id"]
        for certificate in sorted(certificates, key=visual_allocation_key)[
            :selected_count
        ]
    )
    cold_selected = tuple(sorted(task_ids, key=cold_allocation_key)[:selected_count])

    task_rows = []
    candidate_ids_by_task: dict[str, tuple[str, ...]] = {}
    for task_id in task_ids:
        task = blind_tasks[task_id]
        by_signature: dict[tuple[str, ...], dict[str, object]] = {}
        for program in synthesize_relational_transducers(task):
            outputs = tuple(
                execute_relational_transducer(program, query)
                for query in task.test_inputs
            )
            if any(output is None for output in outputs):
                continue
            normalized_outputs = tuple(output for output in outputs if output is not None)
            output_ids = tuple(
                canonical_sha256(grid_to_lists(output)) for output in normalized_outputs
            )
            novel_query_indices = [
                query_index
                for query_index, output_id in enumerate(output_ids)
                if output_id not in base_outputs[task_id][query_index]
            ]
            record_content: dict[str, object] = {
                "task_id": task_id,
                "program_id": program.program_id,
                "program": program.to_json_dict(),
                "demo_exact": True,
                "query_outputs": [
                    grid_to_lists(output) for output in normalized_outputs
                ],
                "query_output_sha256s": list(output_ids),
                "novel_query_indices": novel_query_indices,
            }
            record = {
                "candidate_id": canonical_sha256(record_content),
                **record_content,
            }
            existing = by_signature.get(output_ids)
            if existing is None or record["program_id"] < existing["program_id"]:
                by_signature[output_ids] = record
        candidates = tuple(
            sorted(by_signature.values(), key=lambda record: record["program_id"])
        )
        novel_candidate_ids = tuple(
            record["candidate_id"]
            for record in candidates
            if record["novel_query_indices"]
        )
        candidate_ids_by_task[task_id] = novel_candidate_ids[:2]
        task_rows.append(
            {
                "task_id": task_id,
                "base_query_output_sha256s": [
                    list(values) for values in base_outputs[task_id]
                ],
                "candidate_count": len(candidates),
                "novel_candidate_count": len(novel_candidate_ids),
                "frontier_changed": bool(novel_candidate_ids),
                "candidates": list(candidates),
            }
        )

    arm_selected = {
        "visual_typed": visual_selected,
        "cold_restart": cold_selected,
        "static_family": tuple(task_ids),
    }
    arms = {}
    for arm_name in ARM_NAMES:
        selected = arm_selected[arm_name]
        selected_ids = set(selected)
        arms[arm_name] = {
            "selected_task_ids": list(selected),
            "selected_candidate_ids_by_task": {
                task_id: list(candidate_ids_by_task[task_id])
                for task_id in sorted(selected_ids)
            },
            "frontier_changed_task_count": sum(
                bool(candidate_ids_by_task[task_id]) for task_id in selected_ids
            ),
            "novel_frontier_count": sum(
                len(candidate_ids_by_task[task_id]) for task_id in selected_ids
            ),
            "native_cost": _native_cost(
                selected_task_count=len(selected),
                maximum_demo_count=maximum_demo_count,
                maximum_query_count=maximum_query_count,
            ),
        }
    if arms["visual_typed"]["native_cost"] != arms["cold_restart"]["native_cost"]:
        raise AssertionError("visual and cold native costs differ")

    content: dict[str, object] = {
        "schema": TRANSDUCER_GATE_FREEZE_SCHEMA,
        "cohort_id": cohort_id,
        "query_gold_read": False,
        "controller_training_started": False,
        "allocation_fraction": {
            "numerator": 3,
            "denominator": 10,
            "selected_task_count": selected_count,
        },
        "structural_hypothesis_count": STRUCTURAL_HYPOTHESIS_COUNT,
        "padding_contract": {
            "maximum_demo_count": maximum_demo_count,
            "maximum_query_count": maximum_query_count,
        },
        "source_files": dict(sorted(source_files.items())),
        "provider_freeze_ids": {
            "recursive": recursive_freeze["freeze_id"],
            "visual": visual_freeze["freeze_id"],
        },
        "certificates": sorted(certificates, key=lambda row: row["task_id"]),
        "arms": arms,
        "tasks": task_rows,
    }
    return {"freeze_id": canonical_sha256(content), **content}


def _validated_freeze(freeze: Mapping[str, object]) -> None:
    body = dict(freeze)
    declared = body.pop("freeze_id")
    if declared != canonical_sha256(body):
        raise ValueError("transducer gate freeze ID does not match content")
    if freeze["schema"] != TRANSDUCER_GATE_FREEZE_SCHEMA:
        raise ValueError("unsupported transducer gate freeze schema")
    if freeze["query_gold_read"] is not False:
        raise ValueError("transducer gate freeze is not query-gold-free")


def audit_relational_transducer_allocation_ablation(
    *, freeze: Mapping[str, object], protocol_sha256: str
) -> dict[str, object]:
    """Ablate visual-posterior allocation without reading query solutions."""

    _validated_freeze(freeze)
    if not protocol_sha256:
        raise ValueError("allocation-ablation protocol identity must not be empty")
    certificates = [
        _object(value, field="freeze certificate")
        for value in _sequence(freeze["certificates"], field="freeze certificates")
    ]
    task_ids = sorted(str(certificate["task_id"]) for certificate in certificates)
    if len(task_ids) != len(set(task_ids)) or not task_ids:
        raise ValueError("freeze certificates contain invalid task IDs")
    allocation = _object(freeze["allocation_fraction"], field="allocation fraction")
    selected_count = allocation["selected_task_count"]
    if (
        isinstance(selected_count, bool)
        or not isinstance(selected_count, int)
        or not 0 < selected_count <= len(task_ids)
    ):
        raise ValueError("freeze selected-task count is invalid")
    arms = _object(freeze["arms"], field="freeze arms")
    visual_arm = _object(arms["visual_typed"], field="visual arm")
    observed = tuple(
        str(task_id)
        for task_id in _sequence(
            visual_arm["selected_task_ids"], field="visual selected task IDs"
        )
    )
    if len(observed) != selected_count or len(set(observed)) != selected_count:
        raise ValueError("visual selected task IDs are invalid")

    def posterior_cleared_key(certificate: Mapping[str, object]) -> tuple[object, ...]:
        score = _object(certificate["score"], field="certificate score")
        return (
            -score["total_unique_candidate_count"],
            score["exact_pool_overlap_count"],
            certificate["task_id"],
        )

    posterior_cleared = tuple(
        str(certificate["task_id"])
        for certificate in sorted(certificates, key=posterior_cleared_key)[
            :selected_count
        ]
    )
    all_features_cleared = tuple(task_ids[:selected_count])

    def comparison(alternative: tuple[str, ...]) -> dict[str, object]:
        observed_set = set(observed)
        alternative_set = set(alternative)
        retained = len(observed_set & alternative_set)
        return {
            "selected_set_changed": observed_set != alternative_set,
            "retained_task_count": retained,
            "replaced_task_count": selected_count - retained,
            "symmetric_difference_task_count": len(
                observed_set.symmetric_difference(alternative_set)
            ),
        }

    posterior_comparison = comparison(posterior_cleared)
    full_comparison = comparison(all_features_cleared)
    content: dict[str, object] = {
        "schema": TRANSDUCER_ALLOCATION_ABLATION_SCHEMA,
        "candidate_freeze_id": freeze["freeze_id"],
        "protocol_sha256": protocol_sha256,
        "query_gold_read": False,
        "task_count": len(task_ids),
        "selected_task_count": selected_count,
        "observed_visual_selected_task_ids": list(observed),
        "posterior_cleared": {
            "retained_features": [
                "total_unique_candidate_count",
                "exact_pool_overlap_count",
                "task_id",
            ],
            "selected_task_ids": list(posterior_cleared),
            **posterior_comparison,
        },
        "all_certificate_features_cleared": {
            "retained_features": ["task_id"],
            "selected_task_ids": list(all_features_cleared),
            **full_comparison,
        },
        "primary_gate_passed": posterior_comparison["selected_set_changed"],
    }
    return {"ablation_id": canonical_sha256(content), **content}


def score_relational_transducer_allocation_ablation(
    *,
    freeze: Mapping[str, object],
    ablation: Mapping[str, object],
    solutions: Mapping[str, object],
    solution_source_sha256: str,
) -> dict[str, object]:
    """Score frozen allocation interventions after their query-blind audit."""

    _validated_freeze(freeze)
    ablation_body = dict(ablation)
    declared_ablation_id = ablation_body.pop("ablation_id")
    if declared_ablation_id != canonical_sha256(ablation_body):
        raise ValueError("allocation-ablation ID does not match content")
    if ablation["schema"] != TRANSDUCER_ALLOCATION_ABLATION_SCHEMA:
        raise ValueError("unsupported allocation-ablation schema")
    if ablation["query_gold_read"] is not False:
        raise ValueError("allocation ablation is not query-gold-free")
    if ablation["candidate_freeze_id"] != freeze["freeze_id"]:
        raise ValueError("allocation ablation refers to a different candidate freeze")
    if not solution_source_sha256:
        raise ValueError("solution source identity must not be empty")

    task_rows = {
        _object(row, field="freeze task")["task_id"]: _object(
            row, field="freeze task"
        )
        for row in _sequence(freeze["tasks"], field="freeze tasks")
    }
    if set(task_rows) != set(solutions):
        raise ValueError("freeze and solution task sets differ")
    arms = _object(freeze["arms"], field="freeze arms")
    static_arm = _object(arms["static_family"], field="static-family arm")
    static_candidates = _object(
        static_arm["selected_candidate_ids_by_task"],
        field="static-family selected candidates",
    )
    visual_arm = _object(arms["visual_typed"], field="visual arm")
    visual_selected = tuple(
        str(task_id)
        for task_id in _sequence(
            visual_arm["selected_task_ids"], field="visual selected task IDs"
        )
    )

    posterior_cleared = _object(
        ablation["posterior_cleared"], field="posterior-cleared allocation"
    )
    all_features_cleared = _object(
        ablation["all_certificate_features_cleared"],
        field="all-features-cleared allocation",
    )
    allocation_ids = {
        "observed_visual": tuple(
            str(task_id)
            for task_id in _sequence(
                ablation["observed_visual_selected_task_ids"],
                field="observed visual selected task IDs",
            )
        ),
        "posterior_cleared": tuple(
            str(task_id)
            for task_id in _sequence(
                posterior_cleared["selected_task_ids"],
                field="posterior-cleared selected task IDs",
            )
        ),
        "all_features_cleared": tuple(
            str(task_id)
            for task_id in _sequence(
                all_features_cleared["selected_task_ids"],
                field="all-features-cleared selected task IDs",
            )
        ),
    }
    task_count = ablation["task_count"]
    selected_count = ablation["selected_task_count"]
    if (
        isinstance(task_count, bool)
        or not isinstance(task_count, int)
        or task_count != len(task_rows)
    ):
        raise ValueError("allocation ablation has an invalid task count")
    if (
        isinstance(selected_count, bool)
        or not isinstance(selected_count, int)
        or not 0 < selected_count <= task_count
    ):
        raise ValueError("allocation ablation has an invalid selected-task count")
    if allocation_ids["observed_visual"] != visual_selected:
        raise ValueError("ablation observed allocation differs from visual arm")
    for name, selected in allocation_ids.items():
        if len(selected) != selected_count or len(set(selected)) != selected_count:
            raise ValueError(f"{name} allocation has an invalid task count")
        if not set(selected) <= set(task_rows):
            raise ValueError(f"{name} allocation contains an unknown task")
    posterior_changed = set(allocation_ids["observed_visual"]) != set(
        allocation_ids["posterior_cleared"]
    )
    if posterior_cleared["selected_set_changed"] is not posterior_changed:
        raise ValueError("posterior-cleared comparison is inconsistent")
    if ablation["primary_gate_passed"] is not posterior_changed:
        raise ValueError("allocation-ablation primary gate is inconsistent")

    candidate_by_id: dict[str, Mapping[str, object]] = {}
    for task in task_rows.values():
        for raw_candidate in _sequence(task["candidates"], field="task candidates"):
            candidate = _object(raw_candidate, field="task candidate")
            candidate_id = str(candidate["candidate_id"])
            if candidate_id in candidate_by_id:
                raise ValueError("candidate ID is duplicated")
            candidate_by_id[candidate_id] = candidate

    recovery_counts = {name: 0 for name in allocation_ids}
    scored_tasks = []
    for task_id in sorted(task_rows):
        task = task_rows[task_id]
        gold_ids = tuple(
            canonical_sha256(grid_to_lists(as_grid(solution)))
            for solution in _sequence(
                solutions[task_id], field=f"solutions[{task_id}]"
            )
        )
        base_queries = _sequence(
            task["base_query_output_sha256s"], field="base query outputs"
        )
        if len(base_queries) != len(gold_ids):
            raise ValueError("base output and solution query counts differ")
        base_hit = all(
            gold_id in _sequence(outputs, field="base query output IDs")
            for gold_id, outputs in zip(gold_ids, base_queries, strict=True)
        )
        task_allocations = {}
        for name, selected in allocation_ids.items():
            selected_candidate_ids = (
                _sequence(
                    static_candidates[task_id], field="static selected candidate IDs"
                )
                if task_id in selected
                else ()
            )
            added_by_query = [set() for _ in gold_ids]
            for candidate_id in selected_candidate_ids:
                candidate = candidate_by_id[str(candidate_id)]
                if candidate["task_id"] != task_id or candidate["demo_exact"] is not True:
                    raise ValueError("selected transducer candidate is inconsistent")
                output_ids = _sequence(
                    candidate["query_output_sha256s"], field="candidate query outputs"
                )
                if len(output_ids) != len(gold_ids):
                    raise ValueError("candidate and solution query counts differ")
                for query_index, output_id in enumerate(output_ids):
                    added_by_query[query_index].add(output_id)
            combined_hit = all(
                gold_id in _sequence(base_outputs, field="base outputs")
                or gold_id in added_by_query[query_index]
                for query_index, (gold_id, base_outputs) in enumerate(
                    zip(gold_ids, base_queries, strict=True)
                )
            )
            recovery = combined_hit and not base_hit
            recovery_counts[name] += int(recovery)
            task_allocations[name] = {
                "selected": task_id in selected,
                "recovery": recovery,
            }
        scored_tasks.append(
            {
                "task_id": task_id,
                "base_union_hit": base_hit,
                "allocations": task_allocations,
            }
        )

    posterior_effect = (
        recovery_counts["observed_visual"] - recovery_counts["posterior_cleared"]
    )
    content: dict[str, object] = {
        "schema": TRANSDUCER_ALLOCATION_OUTCOME_SCHEMA,
        "candidate_freeze_id": freeze["freeze_id"],
        "allocation_ablation_id": declared_ablation_id,
        "solution_source_sha256": solution_source_sha256,
        "query_gold_read": True,
        "task_count": len(task_rows),
        "selected_task_count": selected_count,
        "native_cost_per_allocation": visual_arm["native_cost"],
        "unique_recoveries": recovery_counts,
        "observed_minus_posterior_cleared_recoveries": posterior_effect,
        "posterior_performance_attribution_supported": posterior_effect > 0,
        "tasks": scored_tasks,
    }
    return {"result_id": canonical_sha256(content), **content}


def score_relational_transducer_gate(
    *,
    freeze: Mapping[str, object],
    solutions: Mapping[str, object],
    solution_source_sha256: str,
) -> dict[str, object]:
    """Score only after the complete query-blind gate freeze exists."""

    _validated_freeze(freeze)
    if not solution_source_sha256:
        raise ValueError("solution source identity must not be empty")
    task_rows_raw = _sequence(freeze["tasks"], field="freeze tasks")
    task_rows = {
        _object(row, field="freeze task")["task_id"]: _object(
            row, field="freeze task"
        )
        for row in task_rows_raw
    }
    if set(task_rows) != set(solutions):
        raise ValueError("freeze and solution task sets differ")

    candidate_by_id: dict[str, Mapping[str, object]] = {}
    for task in task_rows.values():
        for raw_candidate in _sequence(task["candidates"], field="task candidates"):
            candidate = _object(raw_candidate, field="task candidate")
            candidate_by_id[str(candidate["candidate_id"])] = candidate

    arms = _object(freeze["arms"], field="freeze arms")
    arm_counts = {name: 0 for name in ARM_NAMES}
    base_count = 0
    scored_tasks = []
    for task_id in sorted(task_rows):
        task = task_rows[task_id]
        raw_solutions = _sequence(solutions[task_id], field=f"solutions[{task_id}]")
        gold_ids = tuple(
            canonical_sha256(grid_to_lists(as_grid(solution)))
            for solution in raw_solutions
        )
        base_queries = _sequence(
            task["base_query_output_sha256s"], field="base query outputs"
        )
        if len(base_queries) != len(gold_ids):
            raise ValueError("base output and solution query counts differ")
        base_hit = all(
            gold_id in _sequence(outputs, field="base query output IDs")
            for gold_id, outputs in zip(gold_ids, base_queries, strict=True)
        )
        base_count += int(base_hit)
        task_arm_rows = {}
        for arm_name in ARM_NAMES:
            arm = _object(arms[arm_name], field=f"arm {arm_name}")
            selected_by_task = _object(
                arm["selected_candidate_ids_by_task"],
                field=f"arm {arm_name} selected candidates",
            )
            selected_ids = (
                _sequence(selected_by_task[task_id], field="selected candidate IDs")
                if task_id in selected_by_task
                else ()
            )
            added_by_query = [set() for _ in gold_ids]
            for candidate_id in selected_ids:
                candidate = candidate_by_id[str(candidate_id)]
                if candidate["task_id"] != task_id or candidate["demo_exact"] is not True:
                    raise ValueError("selected transducer candidate is inconsistent")
                output_ids = _sequence(
                    candidate["query_output_sha256s"], field="candidate query outputs"
                )
                if len(output_ids) != len(gold_ids):
                    raise ValueError("candidate and solution query counts differ")
                for query_index, output_id in enumerate(output_ids):
                    added_by_query[query_index].add(output_id)
            combined_hit = all(
                gold_id in _sequence(base_outputs, field="base outputs")
                or gold_id in added_by_query[query_index]
                for query_index, (gold_id, base_outputs) in enumerate(
                    zip(gold_ids, base_queries, strict=True)
                )
            )
            recovery = combined_hit and not base_hit
            arm_counts[arm_name] += int(recovery)
            task_arm_rows[arm_name] = {
                "combined_hit": combined_hit,
                "recovery": recovery,
                "selected_candidate_ids": list(selected_ids),
            }
        scored_tasks.append(
            {"task_id": task_id, "base_union_hit": base_hit, "arms": task_arm_rows}
        )

    task_count = len(task_rows)
    confirm_gate_applicable = task_count == 100
    visual_recoveries = arm_counts["visual_typed"]
    cold_recoveries = arm_counts["cold_restart"]
    content: dict[str, object] = {
        "schema": TRANSDUCER_GATE_RESULT_SCHEMA,
        "candidate_freeze_id": freeze["freeze_id"],
        "solution_source_sha256": solution_source_sha256,
        "query_gold_read": True,
        "task_count": task_count,
        "base_union_strict_coverage": base_count,
        "unique_recoveries": arm_counts,
        "visual_minus_cold_recoveries": visual_recoveries - cold_recoveries,
        "confirm_gate": {
            "applicable": confirm_gate_applicable,
            "minimum_visual_unique_recoveries": 5,
            "visual_exceeds_cold": visual_recoveries > cold_recoveries,
            "passed": confirm_gate_applicable
            and visual_recoveries >= 5
            and visual_recoveries > cold_recoveries,
        },
        "tasks": scored_tasks,
    }
    return {"result_id": canonical_sha256(content), **content}
