"""Query-blind v3 freeze and post-freeze scoring for stateful node rewrites."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from .experiment_safety import canonical_sha256
from .grid import Grid, as_grid, grid_key, grid_to_lists
from .hybrid.object_code import (
    ObjectCodeProgram,
    enumerate_object_code_programs,
    execute_object_code_program,
    object_code_program_id,
    synthesize_object_code_programs,
)
from .object_graph_rewrite_gate import blind_task_from_challenge
from .stateful_object_graph_rewrite import (
    STATEFUL_REWRITE_DSL_VERSION,
    STATEFUL_REWRITE_PROVIDER_VERSION,
    StatefulRewriteCandidate,
    synthesize_stateful_object_graph_rewrites,
)


STATEFUL_REWRITE_FREEZE_SCHEMA = "afts.stateful-object-graph-rewrite-freeze/v3"
STATEFUL_REWRITE_RESULT_SCHEMA = "afts.stateful-object-graph-rewrite-result/v3"
SCIENTIFIC_LANES = (
    "controlled_semantic",
    "outcome_exposed_development",
    "prospective_reserve",
)
COLD_RESTART_SEED = "stateful-object-graph-rewrite-cold-v3-20260811"


def _object(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{field} keys must be strings")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be an array")
    return value


def _query_outputs(
    program: ObjectCodeProgram,
    inputs: Sequence[Grid],
) -> tuple[Grid, ...] | None:
    outputs = []
    for grid in inputs:
        execution = execute_object_code_program(program, grid)
        if not execution.ok or execution.output is None:
            return None
        outputs.append(execution.output)
    return tuple(outputs)


def _single_stage_row(
    *,
    program: ObjectCodeProgram,
    outputs: Sequence[Grid],
    source: str,
) -> dict[str, object]:
    output_lists = [grid_to_lists(output) for output in outputs]
    output_bundle_id = canonical_sha256([grid_key(output) for output in outputs])
    identity = {
        "source": source,
        "program_id": object_code_program_id(program),
        "output_bundle_id": output_bundle_id,
    }
    return {
        "candidate_id": canonical_sha256(identity),
        **identity,
        "query_outputs": output_lists,
        "description_bits": program.description_bits,
    }


def _candidate_row(candidate: StatefulRewriteCandidate) -> dict[str, object]:
    identity = {
        "provider_version": STATEFUL_REWRITE_PROVIDER_VERSION,
        "rewrite_id": candidate.rewrite.rewrite_id,
        "output_bundle_id": candidate.output_bundle_id,
    }
    return {
        "candidate_id": canonical_sha256(identity),
        **identity,
        "program_id": object_code_program_id(candidate.rewrite.program),
        "rewrite": candidate.rewrite.to_json_dict(),
        "certificate": candidate.certificate.to_json_dict(),
        "parent_trace_ids": list(candidate.parent_trace_ids),
        "query_outputs": [grid_to_lists(output) for output in candidate.query_outputs],
        "description_bits": candidate.rewrite.program.description_bits,
    }


def _incumbent_tasks(
    incumbent_freeze: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    if incumbent_freeze["query_gold_read"] is not False:
        raise ValueError("incumbent candidate freeze is not query-blind")
    content = {
        key: value for key, value in incumbent_freeze.items() if key != "freeze_id"
    }
    if incumbent_freeze["freeze_id"] != canonical_sha256(content):
        raise ValueError("incumbent freeze ID differs from canonical content")
    tasks = {}
    for raw_task in _sequence(incumbent_freeze["tasks"], field="incumbent tasks"):
        task = _object(raw_task, field="incumbent task")
        task_id = task["task_id"]
        if not isinstance(task_id, str) or not task_id:
            raise TypeError("incumbent task ID must be a non-empty string")
        if task_id in tasks:
            raise ValueError("incumbent freeze repeats a task ID")
        tasks[task_id] = task
    return tasks


def _incumbent_rows(task: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    rows = tuple(
        _object(item, field="incumbent baseline candidate")
        for item in _sequence(
            task["baseline_candidates"], field="incumbent baseline candidates"
        )
    ) + tuple(
        _object(item, field="incumbent composition candidate")
        for item in _sequence(
            task["composition_candidates"],
            field="incumbent composition candidates",
        )
    )
    by_output: dict[str, Mapping[str, object]] = {}
    for row in rows:
        output_id = row["output_bundle_id"]
        if not isinstance(output_id, str) or not output_id:
            raise TypeError("incumbent output bundle ID must be a non-empty string")
        incumbent = by_output.get(output_id)
        if incumbent is None or str(row["candidate_id"]) < str(
            incumbent["candidate_id"]
        ):
            by_output[output_id] = row
    return tuple(by_output[key] for key in sorted(by_output))


def _cold_candidates(
    task: object,
    *,
    max_first_stage_trials: int,
    reserved_trials: int,
    max_candidates: int,
) -> tuple[tuple[dict[str, object], ...], int, int]:
    grammar = enumerate_object_code_programs(task)
    remaining = grammar[max_first_stage_trials:]
    ordered = tuple(
        sorted(
            remaining,
            key=lambda program: (
                hashlib.sha256(
                    f"{COLD_RESTART_SEED}\0{object_code_program_id(program)}".encode(
                        "ascii"
                    )
                ).hexdigest(),
                object_code_program_id(program),
            ),
        )
    )[:reserved_trials]
    result = synthesize_object_code_programs(
        task,
        max_program_trials=reserved_trials,
        max_exact_programs=max_candidates,
        max_near_misses=0,
        programs=ordered,
    )
    rows = []
    for score in result.exact_scores:
        outputs = _query_outputs(score.program, task.test_inputs)
        if outputs is not None:
            rows.append(
                _single_stage_row(
                    program=score.program,
                    outputs=outputs,
                    source="equal_cost_cold_restart_v3",
                )
            )
    by_output = {row["output_bundle_id"]: row for row in rows}
    return (
        tuple(by_output[key] for key in sorted(by_output)),
        result.program_trial_count,
        reserved_trials - result.program_trial_count,
    )


def freeze_stateful_object_graph_rewrite_opportunities(
    *,
    challenges: Mapping[str, object],
    incumbent_freeze: Mapping[str, object],
    cohort_id: str,
    scientific_lane: str,
    source_files: Mapping[str, str],
    max_first_stage_trials: int = 2_000,
    max_parents: int = 4,
    max_rewrite_trials: int = 512,
    max_candidates: int = 32,
) -> dict[str, object]:
    """Freeze v3 candidates while keeping the v2 frontier immutable."""

    if not challenges:
        raise ValueError("stateful rewrite cohort must not be empty")
    if scientific_lane not in SCIENTIFIC_LANES:
        raise ValueError("unknown stateful rewrite scientific lane")
    incumbent_by_task = _incumbent_tasks(incumbent_freeze)
    if set(incumbent_by_task) != set(challenges):
        raise ValueError("incumbent and v3 challenge task sets differ")

    tasks = []
    opportunity_count = 0
    total_novel_outputs = 0
    for task_id in sorted(challenges):
        raw_task = _object(challenges[task_id], field=f"challenge {task_id}")
        task = blind_task_from_challenge(raw_task)
        incumbent_task = incumbent_by_task[task_id]
        if incumbent_task["blind_content_sha256"] != task.blind_content_sha256:
            raise ValueError("incumbent and v3 blind task identities differ")
        synthesis = synthesize_stateful_object_graph_rewrites(
            task,
            max_first_stage_trials=max_first_stage_trials,
            max_parents=max_parents,
            max_rewrite_trials=max_rewrite_trials,
            max_candidates=max_candidates,
        )
        incumbent_rows = _incumbent_rows(incumbent_task)
        incumbent_by_output = {row["output_bundle_id"]: row for row in incumbent_rows}
        fresh_baseline_ids = frozenset(synthesis.baseline_output_bundle_ids)
        if not fresh_baseline_ids <= frozenset(incumbent_by_output):
            raise ValueError(
                "v3 first-stage baseline drifted beyond the frozen incumbent"
            )

        candidate_rows = tuple(_candidate_row(item) for item in synthesis.candidates)
        novel_rows = tuple(
            row
            for row in candidate_rows
            if row["output_bundle_id"] not in incumbent_by_output
        )
        reserved_trials = max_parents * max_rewrite_trials
        cold_rows, cold_trials, cold_padding = _cold_candidates(
            task,
            max_first_stage_trials=max_first_stage_trials,
            reserved_trials=reserved_trials,
            max_candidates=max_candidates,
        )
        cold_novel_rows = tuple(
            row
            for row in cold_rows
            if row["output_bundle_id"] not in incumbent_by_output
        )
        opportunity = bool(novel_rows)
        opportunity_count += int(opportunity)
        total_novel_outputs += len(novel_rows)
        if opportunity:
            terminal_reason = "novel_frontier_opportunity"
        elif synthesis.parent_count == 0:
            terminal_reason = "no_stateful_scene_parent"
        elif synthesis.rewrite_program_trials == 0:
            terminal_reason = "diagnosed_node_has_no_legal_rewrite"
        else:
            terminal_reason = "legal_rewrites_but_no_demo_exact_candidate"
        tasks.append(
            {
                "task_id": task_id,
                "blind_task_id": task.task_id,
                "blind_content_sha256": task.blind_content_sha256,
                "demo_count": len(task.train),
                "query_count": len(task.test_inputs),
                "parent_count": synthesis.parent_count,
                "terminal_reason": terminal_reason,
                "certificates": [
                    item.to_json_dict() for item in synthesis.certificates
                ],
                "incumbent_candidates": list(incumbent_rows),
                "rewrite_candidates": list(candidate_rows),
                "novel_rewrite_candidate_ids": [
                    row["candidate_id"] for row in novel_rows
                ],
                "cold_candidates": list(cold_rows),
                "novel_cold_candidate_ids": [
                    row["candidate_id"] for row in cold_novel_rows
                ],
                "frontier_opportunity": opportunity,
                "novel_frontier_count": len(novel_rows),
                "cold_novel_frontier_count": len(cold_novel_rows),
                "trace_accounting": {
                    "demo_node_executions": dict(synthesis.demo_node_executions),
                    "query_node_executions": dict(synthesis.query_node_executions),
                    "reused_demo_nodes": dict(synthesis.reused_demo_nodes),
                },
                "native_cost": {
                    "shared_first_stage": {
                        "program_trials": synthesis.first_stage_program_trials,
                        "demo_executions": synthesis.first_stage_program_trials
                        * len(task.train),
                    },
                    "rewrite_arm": {
                        "program_trials": reserved_trials,
                        "realized_program_trials": synthesis.rewrite_program_trials,
                        "padding_program_trials": synthesis.rewrite_padding_trials,
                        "reserved_demo_executions": reserved_trials * len(task.train),
                        "reserved_query_executions": reserved_trials
                        * len(task.test_inputs),
                    },
                    "cold_arm": {
                        "program_trials": reserved_trials,
                        "realized_program_trials": cold_trials,
                        "padding_program_trials": cold_padding,
                        "reserved_demo_executions": reserved_trials * len(task.train),
                        "reserved_query_executions": reserved_trials
                        * len(task.test_inputs),
                    },
                },
            }
        )
    content: dict[str, object] = {
        "schema": STATEFUL_REWRITE_FREEZE_SCHEMA,
        "provider_version": STATEFUL_REWRITE_PROVIDER_VERSION,
        "dsl_version": STATEFUL_REWRITE_DSL_VERSION,
        "cohort_id": cohort_id,
        "scientific_lane": scientific_lane,
        "incumbent_freeze_id": incumbent_freeze["freeze_id"],
        "query_gold_read": False,
        "public_evaluation_read": False,
        "controller_training_started": False,
        "dependency_replay_required": True,
        "search_contract": {
            "max_first_stage_trials": max_first_stage_trials,
            "max_parents": max_parents,
            "max_rewrite_trials": max_rewrite_trials,
            "max_candidates": max_candidates,
        },
        "source_files": dict(sorted(source_files.items())),
        "task_count": len(tasks),
        "opportunity_count": opportunity_count,
        "novel_frontier_count": total_novel_outputs,
        "tasks": tasks,
    }
    return {"freeze_id": canonical_sha256(content), **content}


def _candidate_exact(candidate: Mapping[str, object], gold: Sequence[Grid]) -> bool:
    outputs = tuple(
        as_grid(item)
        for item in _sequence(candidate["query_outputs"], field="candidate outputs")
    )
    return outputs == tuple(gold)


def score_stateful_object_graph_rewrite_freeze(
    *,
    freeze: Mapping[str, object],
    solutions: Mapping[str, object],
    solution_source_sha256: str,
) -> dict[str, object]:
    """Open query labels only after the v3 candidate freeze is immutable."""

    content = {key: value for key, value in freeze.items() if key != "freeze_id"}
    if freeze["freeze_id"] != canonical_sha256(content):
        raise ValueError("stateful rewrite freeze ID differs")
    if freeze["query_gold_read"] is not False:
        raise ValueError("stateful rewrite freeze is not query-gold-free")
    rows = []
    counts = {
        "incumbent_exact": 0,
        "rewrite_exact": 0,
        "rewrite_pass_at_2": 0,
        "cold_exact": 0,
        "rewrite_unique_over_incumbent": 0,
        "rewrite_unique_over_incumbent_and_cold": 0,
    }
    for raw_task in _sequence(freeze["tasks"], field="freeze tasks"):
        task = _object(raw_task, field="freeze task")
        task_id = task["task_id"]
        if task_id not in solutions:
            raise ValueError("solution task set does not cover the v3 freeze")
        gold = tuple(
            as_grid(item)
            for item in _sequence(solutions[task_id], field="solution outputs")
        )
        incumbent = tuple(
            _object(item, field="incumbent candidate")
            for item in _sequence(
                task["incumbent_candidates"], field="incumbent candidates"
            )
        )
        rewrite = tuple(
            _object(item, field="rewrite candidate")
            for item in _sequence(
                task["rewrite_candidates"], field="rewrite candidates"
            )
        )
        cold = tuple(
            _object(item, field="cold candidate")
            for item in _sequence(task["cold_candidates"], field="cold candidates")
        )
        incumbent_exact = any(_candidate_exact(item, gold) for item in incumbent)
        rewrite_exact = any(_candidate_exact(item, gold) for item in rewrite)
        rewrite_pass_at_2 = any(_candidate_exact(item, gold) for item in rewrite[:2])
        cold_exact = any(_candidate_exact(item, gold) for item in cold)
        unique_incumbent = rewrite_exact and not incumbent_exact
        unique_all = unique_incumbent and not cold_exact
        task_metrics = {
            "task_id": task_id,
            "incumbent_exact": incumbent_exact,
            "rewrite_exact": rewrite_exact,
            "rewrite_pass_at_2": rewrite_pass_at_2,
            "cold_exact": cold_exact,
            "rewrite_unique_over_incumbent": unique_incumbent,
            "rewrite_unique_over_incumbent_and_cold": unique_all,
        }
        rows.append(task_metrics)
        for name in counts:
            counts[name] += int(task_metrics[name])
    result_content: dict[str, object] = {
        "schema": STATEFUL_REWRITE_RESULT_SCHEMA,
        "freeze_id": freeze["freeze_id"],
        "solution_source_sha256": solution_source_sha256,
        "task_count": freeze["task_count"],
        "metrics": counts,
        "tasks": rows,
    }
    return {"result_id": canonical_sha256(result_content), **result_content}
