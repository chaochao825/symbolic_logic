"""Freeze and score the Object–Program Workspace v1 development gate."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from afts_arc.blind import BlindTask
from afts_arc.experiment_safety import canonical_sha256
from afts_arc.grid import Grid, as_grid, grid_to_lists
from afts_arc.object_program_workspace import (
    OBJECT_PROGRAM_WORKSPACE_DSL_VERSION,
    ObjectRematchCertificate,
    ObjectRematchProgram,
    RankedObjectRematchEdit,
    base_prefix_programs,
    compile_object_rematch_certificate,
    enumerate_object_rematch_programs,
    execute_object_rematch,
    object_rematch_near_miss_quality,
    object_rematch_variants,
    rank_object_rematch_variants,
    score_object_rematch_program,
)
from afts_arc.task import ARCPair


OBJECT_PROGRAM_WORKSPACE_FREEZE_SCHEMA = "afts.object-program-workspace-freeze/v1"
OBJECT_PROGRAM_WORKSPACE_RESULT_SCHEMA = "afts.object-program-workspace-result/v1"
OBJECT_PROGRAM_WORKSPACE_PROVIDER_VERSION = "afts-object-program-workspace/v1"
ARM_NAMES = (
    "visual_typed",
    "residual_cleared",
    "posterior_shuffled",
    "cold_restart",
)
PROGRAM_TRIAL_BUDGET = 4
COLD_RESTART_SEED = "object-program-workspace-cold-v1-20260811"
POSTERIOR_SHUFFLE_SEED = "object-program-workspace-shuffle-v1-20260811"
SCIENTIFIC_LANES = ("controlled_semantic", "outcome_exposed_development")


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


def _visual_task_map(freeze: Mapping[str, object]) -> dict[str, object]:
    if freeze["query_gold_read"] is not False:
        raise ValueError("visual candidate freeze is not query-gold-free")
    tasks: dict[str, object] = {}
    for raw_task in _sequence(freeze["tasks"], field="visual tasks"):
        task = _object(raw_task, field="visual task")
        task_id = task["task_id"]
        if not isinstance(task_id, str) or not task_id or task_id in tasks:
            raise ValueError("visual task ID is invalid or duplicated")
        tasks[task_id] = task
    return tasks


def _visual_queries(raw_task: object) -> tuple[tuple[tuple[Grid, int], ...], ...]:
    task = _object(raw_task, field="visual task")
    queries = []
    for raw_query in _sequence(task["queries"], field="visual queries"):
        query = _object(raw_query, field="visual query")
        weighted = []
        for raw_candidate in _sequence(
            query["candidates"], field="visual candidates"
        ):
            candidate = _object(raw_candidate, field="visual candidate")
            weight = candidate["sample_count"]
            if type(weight) is not int or weight <= 0:
                raise ValueError("visual sample count must be positive")
            weighted.append((as_grid(candidate["output"]), weight))
        if not weighted:
            raise ValueError("visual query has no complete-grid candidates")
        queries.append(tuple(weighted))
    return tuple(queries)


def _program_output_record(
    *,
    task_id: str,
    program: ObjectRematchProgram,
    outputs: Sequence[Grid],
    source: str,
) -> dict[str, object]:
    content: dict[str, object] = {
        "provider_version": OBJECT_PROGRAM_WORKSPACE_PROVIDER_VERSION,
        "task_id": task_id,
        "source": source,
        "program_id": program.program_id,
        "program": program.to_json_dict(),
        "query_outputs": [grid_to_lists(output) for output in outputs],
    }
    return {"candidate_id": canonical_sha256(content), **content}


def _query_outputs(
    program: ObjectRematchProgram,
    task: BlindTask,
) -> tuple[Grid, ...] | None:
    outputs = []
    for query in task.test_inputs:
        execution = execute_object_rematch(program, query)
        if not execution.ok or execution.output is None:
            return None
        outputs.append(execution.output)
    return tuple(outputs)


def _deduplicate_candidates(
    candidates: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    by_output: dict[str, Mapping[str, object]] = {}
    for candidate in candidates:
        output_id = canonical_sha256(candidate["query_outputs"])
        if output_id not in by_output:
            by_output[output_id] = candidate
    return tuple(by_output.values())


def _score_rows(
    programs: Sequence[ObjectRematchProgram],
    task: BlindTask,
) -> dict[str, object]:
    return {
        program.program_id: score_object_rematch_program(program, task)
        for program in programs
    }


def _eligible_parent(
    *,
    task_id: str,
    task: BlindTask,
    base_programs: Sequence[ObjectRematchProgram],
    full_programs: Sequence[ObjectRematchProgram],
    scores: Mapping[str, object],
) -> tuple[
    ObjectRematchProgram | None,
    ObjectRematchCertificate | None,
    tuple[ObjectRematchProgram, ...],
    str,
]:
    exact_relational = tuple(
        program
        for program in full_programs
        if program.correspond.selector not in {
            candidate.correspond.selector for candidate in base_programs
        }
        and scores[program.program_id].all_demo_exact
    )
    if not exact_relational:
        return None, None, (), "no_demo_exact_relational_child"
    exact_by_context = {
        (program.parse, program.effect) for program in exact_relational
    }
    parents = []
    for program in base_programs:
        if (program.parse, program.effect) not in exact_by_context:
            continue
        quality = object_rematch_near_miss_quality(program, task)
        if quality.eligible:
            parents.append((scores[program.program_id], program))
    if not parents:
        return None, None, exact_relational, "no_quality_base_parent"
    parent_score, parent = min(
        parents,
        key=lambda item: (
            item[0].mismatch_count,
            item[1].description_bits,
            item[1].program_id,
        ),
    )
    if parent_score.all_demo_exact:
        raise AssertionError("eligible near-miss parent cannot be exact")
    certificate = compile_object_rematch_certificate(
        task_id=task_id,
        task=task,
        parent=parent,
        mode="residual_only",
        posterior_views=(),
    )
    variants = object_rematch_variants(
        parent,
        candidate_programs=full_programs,
        existing_program_ids=tuple(program.program_id for program in base_programs),
    )
    exact_variants = tuple(
        edit.program
        for edit in variants
        if scores[edit.program.program_id].all_demo_exact
    )
    if not exact_variants:
        return None, None, exact_relational, "no_single_slot_exact_child"
    return parent, certificate, exact_variants, "eligible"


def _aggregate_ranked_variants(
    *,
    task: BlindTask,
    parent: ObjectRematchProgram,
    certificate: ObjectRematchCertificate,
    full_programs: Sequence[ObjectRematchProgram],
    base_program_ids: Sequence[str],
    posterior_queries: Sequence[Sequence[tuple[Grid, int]]],
    use_posterior: bool,
) -> tuple[RankedObjectRematchEdit, ...]:
    if len(posterior_queries) != len(task.test_inputs):
        raise ValueError("posterior query count differs from blind task")
    by_program: dict[str, list[RankedObjectRematchEdit]] = {}
    for query_index, posterior in enumerate(posterior_queries):
        ranked = rank_object_rematch_variants(
            task=task,
            parent=parent,
            certificate=certificate,
            candidate_programs=full_programs,
            existing_program_ids=base_program_ids,
            query_index=query_index,
            query_posterior=posterior,
            use_posterior=use_posterior,
        )
        for row in ranked:
            by_program.setdefault(row.edit.program.program_id, []).append(row)
    aggregated = []
    for program_id in sorted(by_program):
        rows = by_program[program_id]
        if len(rows) != len(task.test_inputs):
            raise ValueError("ranked variant does not cover every query")
        first = rows[0]
        outputs = tuple(row.query_output for row in rows)
        query_outputs = tuple(output for output in outputs if output is not None)
        aggregated.append(
            RankedObjectRematchEdit(
                first.edit,
                first.demo_exact,
                first.demo_mismatch_count,
                sum(row.posterior_assignment_weight for row in rows),
                sum(row.posterior_total_weight for row in rows),
                query_outputs[0] if len(query_outputs) == 1 else None,
            )
        )
    return tuple(
        sorted(
            aggregated,
            key=lambda item: (
                not item.demo_exact,
                item.demo_mismatch_count,
                -item.posterior_assignment_weight if use_posterior else 0,
                item.edit.program.description_bits,
                item.edit.program.program_id,
            ),
        )
    )


def _typed_arm(
    *,
    task_id: str,
    task: BlindTask,
    parent: ObjectRematchProgram,
    certificate: ObjectRematchCertificate,
    full_programs: Sequence[ObjectRematchProgram],
    base_program_ids: Sequence[str],
    base_output_ids: Sequence[str],
    posterior_queries: Sequence[Sequence[tuple[Grid, int]]],
    use_posterior: bool,
    source: str,
) -> dict[str, object]:
    ranked = _aggregate_ranked_variants(
        task=task,
        parent=parent,
        certificate=certificate,
        full_programs=full_programs,
        base_program_ids=base_program_ids,
        posterior_queries=posterior_queries,
        use_posterior=use_posterior,
    )
    trials = ranked[:PROGRAM_TRIAL_BUDGET]
    candidates = []
    trial_rows = []
    for trial in trials:
        outputs = _query_outputs(trial.edit.program, task)
        trial_rows.append(
            {
                "program_id": trial.edit.program.program_id,
                "demo_exact": trial.demo_exact,
                "demo_mismatch_count": trial.demo_mismatch_count,
                "posterior_assignment_weight": trial.posterior_assignment_weight,
                "posterior_total_weight": trial.posterior_total_weight,
            }
        )
        if trial.demo_exact and outputs is not None:
            candidates.append(
                _program_output_record(
                    task_id=task_id,
                    program=trial.edit.program,
                    outputs=outputs,
                    source=source,
                )
            )
    padding_count = PROGRAM_TRIAL_BUDGET - len(trials)
    deduplicated = _deduplicate_candidates(candidates)
    base_outputs = frozenset(base_output_ids)
    novel_candidate_count = sum(
        canonical_sha256(candidate["query_outputs"]) not in base_outputs
        for candidate in deduplicated
    )
    return {
        "action": "object_rematch",
        "affected_slots": list(certificate.affected_slots),
        "selected_program_ids": [row["program_id"] for row in trial_rows],
        "trial_rows": trial_rows,
        "padding_count": padding_count,
        "novel_program_count": len(trials),
        "program_frontier_changed": bool(trials),
        "novel_frontier_count": novel_candidate_count,
        "frontier_changed": novel_candidate_count > 0,
        "candidates": list(deduplicated),
        "native_cost": {
            "program_trials": PROGRAM_TRIAL_BUDGET,
            "demo_executions": PROGRAM_TRIAL_BUDGET * len(task.train),
            "query_executions": PROGRAM_TRIAL_BUDGET * len(task.test_inputs),
        },
    }


def _cold_arm(
    *,
    task_id: str,
    task: BlindTask,
    parent: ObjectRematchProgram,
    full_programs: Sequence[ObjectRematchProgram],
    base_program_ids: Sequence[str],
    base_output_ids: Sequence[str],
) -> dict[str, object]:
    base = frozenset(base_program_ids)
    novel = tuple(program for program in full_programs if program.program_id not in base)
    ordered = tuple(
        sorted(
            novel,
            key=lambda program: (
                hashlib.sha256(
                    f"{COLD_RESTART_SEED}\0{task_id}\0{program.program_id}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
                program.program_id,
            ),
        )
    )
    trials = ordered[:PROGRAM_TRIAL_BUDGET]
    candidates = []
    trial_rows = []
    for program in trials:
        score = score_object_rematch_program(program, task)
        outputs = _query_outputs(program, task)
        trial_rows.append(
            {
                "program_id": program.program_id,
                "demo_exact": score.all_demo_exact,
                "demo_mismatch_count": score.mismatch_count,
                "posterior_assignment_weight": 0,
                "posterior_total_weight": 0,
            }
        )
        if score.all_demo_exact and outputs is not None:
            candidates.append(
                _program_output_record(
                    task_id=task_id,
                    program=program,
                    outputs=outputs,
                    source="cold_restart",
                )
            )
    padding_count = PROGRAM_TRIAL_BUDGET - len(trials)
    deduplicated = _deduplicate_candidates(candidates)
    base_outputs = frozenset(base_output_ids)
    novel_candidate_count = sum(
        canonical_sha256(candidate["query_outputs"]) not in base_outputs
        for candidate in deduplicated
    )
    return {
        "action": "cold_resynthesis",
        "affected_slots": [],
        "selected_program_ids": [row["program_id"] for row in trial_rows],
        "trial_rows": trial_rows,
        "padding_count": padding_count,
        "novel_program_count": len(trials),
        "program_frontier_changed": bool(trials),
        "novel_frontier_count": novel_candidate_count,
        "frontier_changed": novel_candidate_count > 0,
        "candidates": list(deduplicated),
        "native_cost": {
            "program_trials": PROGRAM_TRIAL_BUDGET,
            "demo_executions": PROGRAM_TRIAL_BUDGET * len(task.train),
            "query_executions": PROGRAM_TRIAL_BUDGET * len(task.test_inputs),
        },
        "padding_program_id": parent.program_id,
    }


def _static_candidates(
    *,
    task_id: str,
    task: BlindTask,
    programs: Sequence[ObjectRematchProgram],
    scores: Mapping[str, object],
    source: str,
) -> tuple[Mapping[str, object], ...]:
    candidates = []
    for program in programs:
        if not scores[program.program_id].all_demo_exact:
            continue
        outputs = _query_outputs(program, task)
        if outputs is None:
            continue
        candidates.append(
            _program_output_record(
                task_id=task_id,
                program=program,
                outputs=outputs,
                source=source,
            )
        )
    return _deduplicate_candidates(candidates)


def freeze_object_program_workspace_gate(
    *,
    challenges: Mapping[str, object],
    visual_freeze: Mapping[str, object],
    cohort_id: str,
    scientific_lane: str,
    source_files: Mapping[str, str],
) -> dict[str, object]:
    """Build a deterministic, query-gold-free candidate and action freeze."""

    if not challenges:
        raise ValueError("object-program workspace cohort must not be empty")
    if scientific_lane not in SCIENTIFIC_LANES:
        raise ValueError("unknown object-program workspace scientific lane")
    visual_tasks = _visual_task_map(visual_freeze)
    task_ids = tuple(sorted(challenges))
    if set(task_ids) != set(visual_tasks):
        raise ValueError("challenge and visual task sets differ")
    shuffle_order = tuple(
        sorted(
            task_ids,
            key=lambda task_id: (
                hashlib.sha256(
                    f"{POSTERIOR_SHUFFLE_SEED}\0{task_id}".encode("utf-8")
                ).hexdigest(),
                task_id,
            ),
        )
    )
    shuffled_source = {
        task_id: shuffle_order[(index + 1) % len(shuffle_order)]
        for index, task_id in enumerate(shuffle_order)
    }
    frozen_tasks = []
    program_opportunity_count = 0
    opportunity_count = 0
    for task_id in task_ids:
        raw_task = _object(challenges[task_id], field=f"challenge {task_id}")
        task = _blind_task(raw_task)
        posterior_queries = _visual_queries(visual_tasks[task_id])
        if len(posterior_queries) != len(task.test_inputs):
            raise ValueError("visual query count differs from challenge")
        full_programs = enumerate_object_rematch_programs(task)
        base_programs = base_prefix_programs(full_programs)
        base_ids = tuple(program.program_id for program in base_programs)
        scores = _score_rows(full_programs, task)
        parent, certificate, exact_children, reason = _eligible_parent(
            task_id=task_id,
            task=task,
            base_programs=base_programs,
            full_programs=full_programs,
            scores=scores,
        )
        base_candidates = _static_candidates(
            task_id=task_id,
            task=task,
            programs=base_programs,
            scores=scores,
            source="base_prefix",
        )
        full_candidates = _static_candidates(
            task_id=task_id,
            task=task,
            programs=full_programs,
            scores=scores,
            source="complete_grammar",
        )
        base_output_ids = tuple(
            canonical_sha256(candidate["query_outputs"])
            for candidate in base_candidates
        )
        exact_child_output_ids = set()
        for child in exact_children:
            outputs = _query_outputs(child, task)
            if outputs is not None:
                exact_child_output_ids.add(
                    canonical_sha256([grid_to_lists(output) for output in outputs])
                )
        frontier_opportunity = bool(
            parent is not None
            and exact_child_output_ids.difference(base_output_ids)
        )
        task_row: dict[str, object] = {
            "task_id": task_id,
            "blind_task_id": canonical_sha256(raw_task),
            "demo_count": len(task.train),
            "query_count": len(task.test_inputs),
            "grammar_program_count": len(full_programs),
            "base_program_count": len(base_programs),
            "base_program_ids_sha256": canonical_sha256(list(base_ids)),
            "base_candidates": list(base_candidates),
            "complete_grammar_candidates": list(full_candidates),
            "program_opportunity": parent is not None,
            "frontier_opportunity": frontier_opportunity,
            "selection_reason": reason,
            "demo_exact_relational_child_count": len(exact_children),
        }
        if parent is None or certificate is None:
            task_row["parent"] = None
            task_row["certificate"] = None
            task_row["shared_proposal_cost"] = {
                "candidate_correspondences_evaluated": 0,
                "demo_assignment_comparisons": 0,
                "query_assignment_comparisons": 0,
            }
            task_row["arms"] = None
            frozen_tasks.append(task_row)
            continue
        program_opportunity_count += 1
        opportunity_count += int(frontier_opportunity)
        variants = object_rematch_variants(
            parent,
            candidate_programs=full_programs,
            existing_program_ids=base_ids,
        )
        source_task_id = shuffled_source[task_id]
        shuffled_queries = _visual_queries(visual_tasks[source_task_id])
        aligned_shuffled = tuple(
            shuffled_queries[index % len(shuffled_queries)]
            for index in range(len(task.test_inputs))
        )
        arms = {
            "visual_typed": _typed_arm(
                task_id=task_id,
                task=task,
                parent=parent,
                certificate=certificate,
                full_programs=full_programs,
                base_program_ids=base_ids,
                base_output_ids=base_output_ids,
                posterior_queries=posterior_queries,
                use_posterior=True,
                source="visual_typed",
            ),
            "residual_cleared": _typed_arm(
                task_id=task_id,
                task=task,
                parent=parent,
                certificate=certificate,
                full_programs=full_programs,
                base_program_ids=base_ids,
                base_output_ids=base_output_ids,
                posterior_queries=posterior_queries,
                use_posterior=False,
                source="residual_cleared",
            ),
            "posterior_shuffled": _typed_arm(
                task_id=task_id,
                task=task,
                parent=parent,
                certificate=certificate,
                full_programs=full_programs,
                base_program_ids=base_ids,
                base_output_ids=base_output_ids,
                posterior_queries=aligned_shuffled,
                use_posterior=True,
                source="posterior_shuffled",
            ),
            "cold_restart": _cold_arm(
                task_id=task_id,
                task=task,
                parent=parent,
                full_programs=full_programs,
                base_program_ids=base_ids,
                base_output_ids=base_output_ids,
            ),
        }
        costs = [arms[arm]["native_cost"] for arm in ARM_NAMES]
        if any(cost != costs[0] for cost in costs[1:]):
            raise AssertionError("object-program workspace arm costs differ")
        for arm in ARM_NAMES:
            arm_row = arms[arm]
            if arm_row["frontier_changed"] != (
                arm_row["novel_frontier_count"] > 0
            ):
                raise AssertionError("frontier novelty guard does not close")
        task_row["parent"] = {
            "program_id": parent.program_id,
            "program": parent.to_json_dict(),
            "near_miss_quality": object_rematch_near_miss_quality(
                parent, task
            ).to_json_dict(),
        }
        task_row["certificate"] = certificate.to_json_dict()
        task_row["posterior_shuffle_source_task_id"] = source_task_id
        task_row["shared_proposal_cost"] = {
            "candidate_correspondences_evaluated": len(variants),
            "demo_assignment_comparisons": len(variants) * len(task.train),
            "query_assignment_comparisons": (
                len(variants) * len(task.test_inputs) * 3
            ),
        }
        task_row["arms"] = arms
        frozen_tasks.append(task_row)
    content: dict[str, object] = {
        "schema": OBJECT_PROGRAM_WORKSPACE_FREEZE_SCHEMA,
        "provider_version": OBJECT_PROGRAM_WORKSPACE_PROVIDER_VERSION,
        "dsl_version": OBJECT_PROGRAM_WORKSPACE_DSL_VERSION,
        "cohort_id": cohort_id,
        "scientific_lane": scientific_lane,
        "visual_freeze_id": visual_freeze["freeze_id"],
        "query_gold_read": False,
        "public_evaluation_read": False,
        "controller_training_started": False,
        "program_trial_budget": PROGRAM_TRIAL_BUDGET,
        "arm_names": list(ARM_NAMES),
        "source_files": dict(sorted(source_files.items())),
        "task_count": len(frozen_tasks),
        "program_opportunity_count": program_opportunity_count,
        "opportunity_count": opportunity_count,
        "tasks": frozen_tasks,
    }
    return {"freeze_id": canonical_sha256(content), **content}


def _task_exact(
    candidates: Sequence[object],
    gold_outputs: Sequence[Grid],
    *,
    limit: int | None,
) -> bool:
    selected = candidates if limit is None else candidates[:limit]
    for raw_candidate in selected:
        candidate = _object(raw_candidate, field="candidate")
        outputs = tuple(
            as_grid(output)
            for output in _sequence(
                candidate["query_outputs"], field="candidate query outputs"
            )
        )
        if outputs == tuple(gold_outputs):
            return True
    return False


def score_object_program_workspace_gate(
    *,
    freeze: Mapping[str, object],
    solutions: Mapping[str, object],
    solution_source_sha256: str,
) -> dict[str, object]:
    """Open query gold only after a content-addressed candidate freeze exists."""

    freeze_content = {
        key: value for key, value in freeze.items() if key != "freeze_id"
    }
    if freeze["freeze_id"] != canonical_sha256(freeze_content):
        raise ValueError("object-program workspace freeze ID differs")
    if freeze["schema"] != OBJECT_PROGRAM_WORKSPACE_FREEZE_SCHEMA:
        raise ValueError("unsupported object-program workspace freeze schema")
    if freeze["query_gold_read"] is not False:
        raise ValueError("candidate freeze is not query-gold-free")
    tasks = _sequence(freeze["tasks"], field="frozen tasks")
    if set(solutions) != {
        _object(task, field="frozen task")["task_id"] for task in tasks
    }:
        raise ValueError("solution and frozen task sets differ")
    counts = {
        "base_prefix": 0,
        "complete_grammar": 0,
        **{arm: 0 for arm in ARM_NAMES},
    }
    unique = {arm: 0 for arm in ARM_NAMES}
    visual_unique_over_controls = 0
    visual_residual_action_changes = 0
    visual_shuffle_action_changes = 0
    scored_tasks = []
    for raw_task in tasks:
        task = _object(raw_task, field="frozen task")
        task_id = task["task_id"]
        gold_outputs = tuple(
            as_grid(output)
            for output in _sequence(solutions[task_id], field=f"solution {task_id}")
        )
        if len(gold_outputs) != task["query_count"]:
            raise ValueError("solution query count differs from frozen task")
        base_exact = _task_exact(
            _sequence(task["base_candidates"], field="base candidates"),
            gold_outputs,
            limit=None,
        )
        full_exact = _task_exact(
            _sequence(
                task["complete_grammar_candidates"],
                field="complete grammar candidates",
            ),
            gold_outputs,
            limit=None,
        )
        counts["base_prefix"] += int(base_exact)
        counts["complete_grammar"] += int(full_exact)
        arm_exact = {arm: False for arm in ARM_NAMES}
        if task["program_opportunity"]:
            arms = _object(task["arms"], field="task arms")
            for arm in ARM_NAMES:
                arm_row = _object(arms[arm], field=f"arm {arm}")
                arm_exact[arm] = _task_exact(
                    _sequence(arm_row["candidates"], field=f"{arm} candidates"),
                    gold_outputs,
                    limit=2,
                )
                counts[arm] += int(arm_exact[arm])
                unique[arm] += int(arm_exact[arm] and not base_exact)
            visual_ids = arms["visual_typed"]["selected_program_ids"]
            residual_ids = arms["residual_cleared"]["selected_program_ids"]
            shuffled_ids = arms["posterior_shuffled"]["selected_program_ids"]
            visual_residual_action_changes += int(visual_ids != residual_ids)
            visual_shuffle_action_changes += int(visual_ids != shuffled_ids)
            visual_unique_over_controls += int(
                arm_exact["visual_typed"]
                and not base_exact
                and not arm_exact["residual_cleared"]
                and not arm_exact["posterior_shuffled"]
                and not arm_exact["cold_restart"]
            )
        scored_tasks.append(
            {
                "task_id": task_id,
                "program_opportunity": task["program_opportunity"],
                "frontier_opportunity": task["frontier_opportunity"],
                "base_prefix_exact": base_exact,
                "complete_grammar_exact": full_exact,
                "arm_exact": arm_exact,
            }
        )
    opportunity_count = freeze["opportunity_count"]
    program_opportunity_count = freeze["program_opportunity_count"]
    scientific_lane = freeze["scientific_lane"]
    if scientific_lane not in SCIENTIFIC_LANES:
        raise ValueError("frozen object-program workspace lane is invalid")
    control_applicable = scientific_lane == "controlled_semantic"
    control_recovery_rate = (
        counts["visual_typed"] / program_opportunity_count
        if program_opportunity_count
        else 0.0
    )
    control_gate_passed = (
        control_applicable
        and program_opportunity_count == freeze["task_count"]
        and control_recovery_rate >= 0.8
        and visual_residual_action_changes > 0
        and visual_shuffle_action_changes > 0
    )
    screening_applicable = (
        scientific_lane == "outcome_exposed_development"
        and opportunity_count >= 5
    )
    content: dict[str, object] = {
        "schema": OBJECT_PROGRAM_WORKSPACE_RESULT_SCHEMA,
        "candidate_freeze_id": freeze["freeze_id"],
        "solution_source_sha256": solution_source_sha256,
        "query_gold_read_after_freeze": True,
        "public_evaluation_read": False,
        "controller_training_started": False,
        "scientific_lane": scientific_lane,
        "task_count": freeze["task_count"],
        "program_opportunity_count": program_opportunity_count,
        "opportunity_count": opportunity_count,
        "strict_task_coverage": counts,
        "unique_recovery_over_base_prefix": unique,
        "visual_unique_over_all_controls": visual_unique_over_controls,
        "action_interventions": {
            "visual_vs_residual_selected_set_changes": (
                visual_residual_action_changes
            ),
            "visual_vs_shuffled_selected_set_changes": visual_shuffle_action_changes,
        },
        "controlled_semantic_gate": {
            "applicable": control_applicable,
            "minimum_typed_recovery_rate": 0.8,
            "typed_recovery_rate": control_recovery_rate,
            "all_tasks_are_program_opportunities": program_opportunity_count
            == freeze["task_count"],
            "action_intervention_observed": (
                visual_residual_action_changes > 0
                and visual_shuffle_action_changes > 0
            ),
            "passed": control_gate_passed,
            "licenses_generalization_claim": False,
        },
        "development_screen": {
            "applicable": screening_applicable,
            "minimum_opportunity_count": 5,
            "opportunity_gate_passed": screening_applicable,
            "prospective_confirmation_authorized": (
                screening_applicable
                and unique["visual_typed"] > unique["cold_restart"]
                and unique["visual_typed"] > 0
            ),
        },
        "tasks": scored_tasks,
    }
    return {"result_id": canonical_sha256(content), **content}
