"""Query-blind opportunity freeze and post-freeze scoring for rewrite v2."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from .blind import BlindTask
from .experiment_safety import canonical_sha256
from .grid import Grid, as_grid, grid_key, grid_to_lists
from .hybrid.object_code import (
    ObjectCodeProgram,
    enumerate_object_code_programs,
    execute_object_code_program,
    object_code_program_id,
    synthesize_object_code_programs,
)
from .object_graph_rewrite import (
    OBJECT_GRAPH_REWRITE_DSL_VERSION,
    OBJECT_GRAPH_REWRITE_PROVIDER_VERSION,
    ObjectGraphRewriteCandidate,
    synthesize_object_graph_rewrites,
)
from .object_program_workspace import (
    execute_object_rematch,
    synthesize_exact_object_rematch_programs,
)
from .task import ARCPair


OBJECT_GRAPH_REWRITE_FREEZE_SCHEMA = "afts.object-graph-rewrite-freeze/v2"
OBJECT_GRAPH_REWRITE_RESULT_SCHEMA = "afts.object-graph-rewrite-result/v2"
SCIENTIFIC_LANES = (
    "controlled_semantic",
    "outcome_exposed_development",
    "prospective_reserve",
)
COLD_RESTART_SEED = "object-graph-rewrite-cold-v2-20260811"


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


def blind_task_from_challenge(raw_task: Mapping[str, object]) -> BlindTask:
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


def _object_code_query_outputs(
    program: ObjectCodeProgram, task: BlindTask
) -> tuple[Grid, ...] | None:
    outputs = []
    for query in task.test_inputs:
        execution = execute_object_code_program(program, query)
        if not execution.ok or execution.output is None:
            return None
        outputs.append(execution.output)
    return tuple(outputs)


def _v1_query_outputs(task: BlindTask) -> tuple[tuple[Grid, ...], ...]:
    bundles = []
    for program in synthesize_exact_object_rematch_programs(task):
        outputs = []
        for query in task.test_inputs:
            execution = execute_object_rematch(program, query)
            if not execution.ok or execution.output is None:
                break
            outputs.append(execution.output)
        if len(outputs) == len(task.test_inputs):
            bundles.append(tuple(outputs))
    return tuple(bundles)


def _candidate_row(candidate: ObjectGraphRewriteCandidate) -> dict[str, object]:
    return {
        "candidate_id": canonical_sha256(
            {
                "provider_version": OBJECT_GRAPH_REWRITE_PROVIDER_VERSION,
                "program_id": candidate.program.program_id,
                "output_bundle_id": candidate.output_bundle_id,
            }
        ),
        "program_id": candidate.program.program_id,
        "program": candidate.program.to_json_dict(),
        "certificate": candidate.certificate.to_json_dict(),
        "output_bundle_id": candidate.output_bundle_id,
        "query_outputs": [grid_to_lists(output) for output in candidate.query_outputs],
        "description_bits": candidate.program.description_bits,
    }


def _single_stage_row(
    *, program: ObjectCodeProgram, outputs: Sequence[Grid], source: str
) -> dict[str, object]:
    output_lists = [grid_to_lists(output) for output in outputs]
    bundle_id = canonical_sha256([grid_key(output) for output in outputs])
    content = {
        "source": source,
        "program_id": object_code_program_id(program),
        "output_bundle_id": bundle_id,
    }
    return {
        "candidate_id": canonical_sha256(content),
        **content,
        "query_outputs": output_lists,
        "description_bits": program.description_bits,
    }


def _cold_candidates(
    task: BlindTask,
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
        outputs = _object_code_query_outputs(score.program, task)
        if outputs is not None:
            rows.append(
                _single_stage_row(
                    program=score.program,
                    outputs=outputs,
                    source="equal_cost_cold_restart",
                )
            )
    by_output = {row["output_bundle_id"]: row for row in rows}
    return (
        tuple(by_output[key] for key in sorted(by_output)),
        result.program_trial_count,
        reserved_trials - result.program_trial_count,
    )


def freeze_object_graph_rewrite_opportunities(
    *,
    challenges: Mapping[str, object],
    cohort_id: str,
    scientific_lane: str,
    source_files: Mapping[str, str],
    max_first_stage_trials: int = 2_000,
    max_parents: int = 4,
    max_second_stage_trials: int = 512,
    max_candidates: int = 32,
) -> dict[str, object]:
    """Freeze candidates and novelty without reading query gold."""

    if not challenges:
        raise ValueError("object-graph rewrite cohort must not be empty")
    if scientific_lane not in SCIENTIFIC_LANES:
        raise ValueError("unknown object-graph rewrite scientific lane")
    cohort_id = str(cohort_id)
    tasks = []
    opportunity_count = 0
    total_novel_outputs = 0
    for task_id in sorted(challenges):
        raw_task = _object(challenges[task_id], field=f"challenge {task_id}")
        task = blind_task_from_challenge(raw_task)
        synthesis = synthesize_object_graph_rewrites(
            task,
            max_first_stage_trials=max_first_stage_trials,
            max_parents=max_parents,
            max_second_stage_trials=max_second_stage_trials,
            max_candidates=max_candidates,
        )
        baseline_rows = []
        for program in synthesis.baseline_programs:
            outputs = _object_code_query_outputs(program, task)
            if outputs is None:
                raise AssertionError("demo-exact baseline failed on query input")
            baseline_rows.append(
                _single_stage_row(
                    program=program,
                    outputs=outputs,
                    source="one_stage_object_code",
                )
            )
        for outputs in _v1_query_outputs(task):
            output_lists = [grid_to_lists(output) for output in outputs]
            bundle_id = canonical_sha256([grid_key(output) for output in outputs])
            baseline_rows.append(
                {
                    "candidate_id": canonical_sha256(
                        {"source": "object_program_workspace_v1", "outputs": output_lists}
                    ),
                    "source": "object_program_workspace_v1",
                    "program_id": None,
                    "output_bundle_id": bundle_id,
                    "query_outputs": output_lists,
                    "description_bits": None,
                }
            )
        baseline_by_output = {
            row["output_bundle_id"]: row for row in baseline_rows
        }
        candidate_rows = tuple(_candidate_row(item) for item in synthesis.candidates)
        novel_rows = tuple(
            row
            for row in candidate_rows
            if row["output_bundle_id"] not in baseline_by_output
        )
        reserved_trials = max_parents * max_second_stage_trials
        cold_rows, cold_trials, cold_padding = _cold_candidates(
            task,
            max_first_stage_trials=max_first_stage_trials,
            reserved_trials=reserved_trials,
            max_candidates=max_candidates,
        )
        cold_novel_rows = tuple(
            row
            for row in cold_rows
            if row["output_bundle_id"] not in baseline_by_output
        )
        opportunity = bool(novel_rows)
        opportunity_count += int(opportunity)
        total_novel_outputs += len(novel_rows)
        tasks.append(
            {
                "task_id": task_id,
                "blind_task_id": task.task_id,
                "blind_content_sha256": task.blind_content_sha256,
                "demo_count": len(task.train),
                "query_count": len(task.test_inputs),
                "parent_count": synthesis.parent_count,
                "certificates": [
                    item.to_json_dict() for item in synthesis.parent_certificates
                ],
                "baseline_candidates": [
                    baseline_by_output[key] for key in sorted(baseline_by_output)
                ],
                "composition_candidates": list(candidate_rows),
                "novel_composition_candidate_ids": [
                    row["candidate_id"] for row in novel_rows
                ],
                "cold_candidates": list(cold_rows),
                "novel_cold_candidate_ids": [
                    row["candidate_id"] for row in cold_novel_rows
                ],
                "frontier_opportunity": opportunity,
                "novel_frontier_count": len(novel_rows),
                "cold_novel_frontier_count": len(cold_novel_rows),
                "native_cost": {
                    "shared_first_stage": {
                        "program_trials": synthesis.first_stage_program_trials,
                        "demo_executions": (
                            synthesis.first_stage_program_trials * len(task.train)
                        ),
                    },
                    "composition_arm": {
                        "program_trials": reserved_trials,
                        "realized_program_trials": synthesis.second_stage_program_trials,
                        "padding_program_trials": synthesis.second_stage_padding_trials,
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
        "schema": OBJECT_GRAPH_REWRITE_FREEZE_SCHEMA,
        "provider_version": OBJECT_GRAPH_REWRITE_PROVIDER_VERSION,
        "dsl_version": OBJECT_GRAPH_REWRITE_DSL_VERSION,
        "cohort_id": cohort_id,
        "scientific_lane": scientific_lane,
        "query_gold_read": False,
        "public_evaluation_read": False,
        "controller_training_started": False,
        "search_contract": {
            "max_first_stage_trials": max_first_stage_trials,
            "max_parents": max_parents,
            "max_second_stage_trials": max_second_stage_trials,
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


def score_object_graph_rewrite_freeze(
    *,
    freeze: Mapping[str, object],
    solutions: Mapping[str, object],
    solution_source_sha256: str,
) -> dict[str, object]:
    """Score a previously content-addressed freeze without changing its order."""

    content = {key: value for key, value in freeze.items() if key != "freeze_id"}
    if freeze["freeze_id"] != canonical_sha256(content):
        raise ValueError("object-graph rewrite freeze ID differs")
    if freeze["query_gold_read"] is not False:
        raise ValueError("candidate freeze is not query-gold-free")
    rows = []
    counts = {
        "baseline_exact": 0,
        "composition_exact": 0,
        "composition_pass_at_2": 0,
        "cold_exact": 0,
        "composition_unique_over_baseline": 0,
        "composition_unique_over_baseline_and_cold": 0,
    }
    for raw_task in _sequence(freeze["tasks"], field="freeze tasks"):
        task = _object(raw_task, field="freeze task")
        task_id = task["task_id"]
        if task_id not in solutions:
            raise ValueError("solution task set does not cover the freeze")
        gold = tuple(
            as_grid(item)
            for item in _sequence(solutions[task_id], field="solution outputs")
        )
        baseline = tuple(
            _object(item, field="baseline candidate")
            for item in _sequence(
                task["baseline_candidates"], field="baseline candidates"
            )
        )
        composition = tuple(
            _object(item, field="composition candidate")
            for item in _sequence(
                task["composition_candidates"], field="composition candidates"
            )
        )
        cold = tuple(
            _object(item, field="cold candidate")
            for item in _sequence(task["cold_candidates"], field="cold candidates")
        )
        baseline_exact = any(_candidate_exact(item, gold) for item in baseline)
        composition_exact = any(
            _candidate_exact(item, gold) for item in composition
        )
        composition_pass_at_2 = any(
            _candidate_exact(item, gold) for item in composition[:2]
        )
        cold_exact = any(_candidate_exact(item, gold) for item in cold)
        unique_baseline = composition_exact and not baseline_exact
        unique_all = unique_baseline and not cold_exact
        task_metrics = {
            "task_id": task_id,
            "baseline_exact": baseline_exact,
            "composition_exact": composition_exact,
            "composition_pass_at_2": composition_pass_at_2,
            "cold_exact": cold_exact,
            "composition_unique_over_baseline": unique_baseline,
            "composition_unique_over_baseline_and_cold": unique_all,
        }
        rows.append(task_metrics)
        for name in counts:
            counts[name] += int(task_metrics[name])
    result_content: dict[str, object] = {
        "schema": OBJECT_GRAPH_REWRITE_RESULT_SCHEMA,
        "freeze_id": freeze["freeze_id"],
        "solution_source_sha256": solution_source_sha256,
        "task_count": freeze["task_count"],
        "metrics": counts,
        "tasks": rows,
    }
    return {"result_id": canonical_sha256(result_content), **result_content}
