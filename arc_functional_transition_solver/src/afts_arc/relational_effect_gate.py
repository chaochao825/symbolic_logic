"""Sequential query-blind gate for provenance-aligned relational effects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from .arc_tgi_reserve import ARC_TGI_RESERVE_SEAL_SCHEMA
from .anchor_mask import AnchorRasterizedDeltaNode, execute_anchor_rasterized_delta
from .blind import BlindTask
from .experiment_safety import canonical_sha256
from .grid import Grid, as_grid, grid_key, grid_to_lists
from .hybrid.object_code import (
    ObjectCodeProgram,
    ObjectCodeProgramScore,
    _score_program,
    enumerate_object_code_programs,
    execute_object_code_program,
    object_code_program_from_json,
    object_code_program_id,
)
from .hybrid.scene_graph import SceneGraph, ScenePipelineProgram
from .object_graph_rewrite_gate import blind_task_from_challenge
from .relational_effect import (
    RELATIONAL_EFFECT_PROGRAMS_PER_DELTA,
    RELATIONAL_EFFECT_PROVIDER_VERSION,
    RelationalEffectNode,
    candidate_output_bundle,
    compile_relational_failure_core,
    execute_relational_effect,
    execute_scene_with_cell_provenance,
    lesion_entity_links,
    leave_one_demo_out_relational_effects,
)
from .stateful_object_graph_rewrite_gate import _cold_candidates


RELATIONAL_EFFECT_FREEZE_SCHEMA = "afts.provenance-relational-effect-freeze/v1"
RELATIONAL_EFFECT_RESULT_SCHEMA = "afts.provenance-relational-effect-result/v1"
RELATIONAL_EFFECT_INTERVENTION_SCHEMA = (
    "afts.provenance-relational-effect-intervention/v1"
)
RELATIONAL_EFFECT_FAILURE_AUDIT_SCHEMA = (
    "afts.provenance-relational-effect-failure-audit/v1"
)
SCIENTIFIC_LANE = "prospective_reserve"
MINIMUM_LODO_TASKS = 5
MINIMUM_LODO_FAMILIES = 3
MINIMUM_LODO_ADVANTAGE = 3
MINIMUM_UNIQUE_LODO_FAMILIES = 2
MINIMUM_FRONTIER_TASKS = 5
MINIMUM_FRONTIER_FAMILIES = 3
MINIMUM_UNIQUE_RECOVERIES = 5
MINIMUM_RECOVERY_FAMILIES = 3


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


def _bundle_id(outputs: Sequence[Grid]) -> str:
    return canonical_sha256([grid_key(output) for output in outputs])


def _program_sort_key(program: ObjectCodeProgram) -> tuple[object, ...]:
    return program.description_bits, object_code_program_id(program)


def _parent_sort_key(score: ObjectCodeProgramScore) -> tuple[object, ...]:
    return (
        -score.exact_demo_count,
        -score.shape_match_count,
        score.mismatch_count,
        -score.agreement,
        score.program.description_bits,
        object_code_program_id(score.program),
    )


@dataclass(frozen=True, slots=True)
class QueryBlindFirstStage:
    exact_programs: tuple[ObjectCodeProgram, ...]
    parent_programs: tuple[ScenePipelineProgram, ...]
    program_trials: int
    demo_executions: int


def query_blind_first_stage(
    task: BlindTask,
    *,
    max_program_trials: int,
    max_exact_programs: int,
    max_parents: int,
) -> QueryBlindFirstStage:
    """Select exact programs and near-miss parents without executing a query."""

    if not isinstance(task, BlindTask):
        raise TypeError("query-blind first stage requires BlindTask")
    for name, value in (
        ("max_program_trials", max_program_trials),
        ("max_exact_programs", max_exact_programs),
        ("max_parents", max_parents),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    programs = enumerate_object_code_programs(task)[:max_program_trials]
    scene_cache: dict[tuple[object, ...], SceneGraph] = {}
    scores = tuple(_score_program(program, task, scene_cache) for program in programs)
    exact = tuple(
        sorted(
            (score.program for score in scores if score.all_demo_exact),
            key=_program_sort_key,
        )[:max_exact_programs]
    )
    parents = tuple(
        score.program
        for score in sorted(
            (
                score
                for score in scores
                if isinstance(score.program, ScenePipelineProgram)
                and score.execution_valid
                and not score.all_demo_exact
            ),
            key=_parent_sort_key,
        )[:max_parents]
    )
    if any(not isinstance(program, ScenePipelineProgram) for program in parents):
        raise AssertionError("query-blind parent selection lost its scene type")
    return QueryBlindFirstStage(
        exact,
        parents,
        len(programs),
        len(programs) * len(task.train),
    )


def _validated_seal(
    *,
    seal: Mapping[str, object],
    challenges: Mapping[str, object],
) -> tuple[str, dict[str, str]]:
    content = {key: value for key, value in seal.items() if key != "seal_id"}
    if seal["seal_id"] != canonical_sha256(content):
        raise ValueError("reserve seal ID differs from canonical content")
    if seal["schema"] != ARC_TGI_RESERVE_SEAL_SCHEMA:
        raise ValueError("unsupported reserve seal schema")
    if seal["query_gold_written"] is not False or seal["witness_written"] is not False:
        raise ValueError("reserve oracle or witness was opened before the gate")
    if seal["challenge_content_sha256"] != canonical_sha256(challenges):
        raise ValueError("challenge content differs from the reserve seal")
    family_by_task = {}
    for raw in _sequence(seal["tasks"], field="sealed tasks"):
        row = _object(raw, field="sealed task")
        task_id = row["task_id"]
        family_id = row["family_id"]
        if not isinstance(task_id, str) or not isinstance(family_id, str):
            raise TypeError("sealed task and family IDs must be strings")
        family_by_task[task_id] = family_id
    if set(family_by_task) != set(challenges):
        raise ValueError("sealed and challenge task sets differ")
    cohort_id = seal["cohort_id"]
    if not isinstance(cohort_id, str) or not cohort_id:
        raise TypeError("reserve cohort ID must be a non-empty string")
    return cohort_id, family_by_task


@dataclass(frozen=True, slots=True)
class _SensorJob:
    task_id: str
    family_id: str
    raw_task: Mapping[str, object]
    max_first_stage_trials: int
    max_exact_programs: int
    max_parents: int


def _sensor_task(job: _SensorJob) -> dict[str, object]:
    task = blind_task_from_challenge(job.raw_task)
    first_stage = query_blind_first_stage(
        task,
        max_program_trials=job.max_first_stage_trials,
        max_exact_programs=job.max_exact_programs,
        max_parents=job.max_parents,
    )
    inputs = tuple(pair.input for pair in task.train)
    targets = tuple(pair.output for pair in task.train)
    if any(target is None for target in targets):
        raise ValueError("blind demonstration lost its output")
    parent_rows = []
    for parent in first_stage.parent_programs:
        executions = tuple(
            execute_scene_with_cell_provenance(parent, input_grid)
            for input_grid in inputs
        )
        core = compile_relational_failure_core(
            blind_task_id=task.task_id,
            executions=executions,
            targets=targets,  # type: ignore[arg-type]
        )
        lodo = leave_one_demo_out_relational_effects(
            blind_task_id=task.task_id,
            program=parent,
            inputs=inputs,
            targets=targets,  # type: ignore[arg-type]
        )
        parent_rows.append(
            {
                "parent_program_id": object_code_program_id(parent),
                "parent_program": parent.to_json_dict(),
                "core": core.to_json_dict(),
                "lodo": lodo,
            }
        )
    method_strict = any(
        row["lodo"]["strict_all_folds_exact"] is True for row in parent_rows
    )
    baseline_fields = {
        "identity": "identity_strict_all_folds_exact",
        "global": "global_recolor_strict_all_folds_exact",
        "component": "component_recolor_strict_all_folds_exact",
    }
    baselines = {
        name: any(row["lodo"][field] is True for row in parent_rows)
        for name, field in baseline_fields.items()
    }
    return {
        "task_id": job.task_id,
        "family_id": job.family_id,
        "blind_task_id": task.task_id,
        "blind_content_sha256": task.blind_content_sha256,
        "demo_count": len(task.train),
        "query_count": len(task.test_inputs),
        "first_stage": {
            "program_trials": first_stage.program_trials,
            "demo_executions": first_stage.demo_executions,
            "query_executions": 0,
            "exact_programs": [
                program.to_json_dict() for program in first_stage.exact_programs
            ],
        },
        "parent_count": len(parent_rows),
        "parents": parent_rows,
        "strict_lodo": method_strict,
        "baseline_strict_lodo": baselines,
    }


def _sensor_summary(tasks: Sequence[Mapping[str, object]]) -> dict[str, object]:
    method_tasks = {str(task["task_id"]) for task in tasks if task["strict_lodo"] is True}
    method_families = {
        str(task["family_id"]) for task in tasks if task["strict_lodo"] is True
    }
    baseline_task_sets = {
        name: {
            str(task["task_id"])
            for task in tasks
            if _object(
                task["baseline_strict_lodo"], field="baseline strict LODO"
            )[name]
            is True
        }
        for name in ("identity", "global", "component")
    }
    strongest_name = min(
        baseline_task_sets,
        key=lambda name: (-len(baseline_task_sets[name]), name),
    )
    any_baseline = set().union(*baseline_task_sets.values())
    unique_tasks = method_tasks - any_baseline
    unique_families = {
        str(task["family_id"])
        for task in tasks
        if task["task_id"] in unique_tasks
    }
    passed = (
        len(method_tasks) >= MINIMUM_LODO_TASKS
        and len(method_families) >= MINIMUM_LODO_FAMILIES
        and len(method_tasks) - len(baseline_task_sets[strongest_name])
        >= MINIMUM_LODO_ADVANTAGE
        and len(unique_families) >= MINIMUM_UNIQUE_LODO_FAMILIES
    )
    return {
        "method_task_count": len(method_tasks),
        "method_family_count": len(method_families),
        "baseline_task_counts": {
            name: len(values) for name, values in baseline_task_sets.items()
        },
        "strongest_baseline": strongest_name,
        "strongest_baseline_task_count": len(baseline_task_sets[strongest_name]),
        "advantage_over_strongest_baseline": (
            len(method_tasks) - len(baseline_task_sets[strongest_name])
        ),
        "unique_over_all_baselines_task_count": len(unique_tasks),
        "unique_over_all_baselines_family_count": len(unique_families),
        "thresholds": {
            "minimum_method_tasks": MINIMUM_LODO_TASKS,
            "minimum_method_families": MINIMUM_LODO_FAMILIES,
            "minimum_advantage": MINIMUM_LODO_ADVANTAGE,
            "minimum_unique_families": MINIMUM_UNIQUE_LODO_FAMILIES,
        },
        "passed": passed,
    }


def _execute_program_queries(
    program: ObjectCodeProgram,
    inputs: Sequence[Grid],
) -> tuple[Grid, ...] | None:
    outputs = []
    for input_grid in inputs:
        execution = execute_object_code_program(program, input_grid)
        if not execution.ok or execution.output is None:
            return None
        outputs.append(execution.output)
    return tuple(outputs)


def _baseline_row(
    *,
    program: ObjectCodeProgram,
    outputs: Sequence[Grid],
    source: str,
) -> dict[str, object]:
    identity = {
        "source": source,
        "program_id": object_code_program_id(program),
        "output_bundle_id": _bundle_id(outputs),
    }
    return {
        "candidate_id": canonical_sha256(identity),
        **identity,
        "query_outputs": [grid_to_lists(output) for output in outputs],
        "description_bits": program.description_bits,
    }


@dataclass(frozen=True, slots=True)
class _FrontierJob:
    sensor_task: Mapping[str, object]
    raw_task: Mapping[str, object]
    max_first_stage_trials: int
    reserved_native_trials: int
    max_exact_programs: int


def _frontier_task(job: _FrontierJob) -> dict[str, object]:
    task = blind_task_from_challenge(job.raw_task)
    if task.blind_content_sha256 != job.sensor_task["blind_content_sha256"]:
        raise ValueError("sensor and frontier blind identities differ")
    incumbent_rows = []
    for raw_program in _sequence(
        _object(job.sensor_task["first_stage"], field="first stage")["exact_programs"],
        field="exact programs",
    ):
        program = object_code_program_from_json(raw_program)
        outputs = _execute_program_queries(program, task.test_inputs)
        if outputs is not None:
            incumbent_rows.append(
                _baseline_row(
                    program=program,
                    outputs=outputs,
                    source="query_blind_first_stage_exact",
                )
            )
    incumbent_by_output = {
        str(row["output_bundle_id"]): row for row in incumbent_rows
    }
    cold_rows, cold_trials, cold_padding = _cold_candidates(
        task,
        max_first_stage_trials=job.max_first_stage_trials,
        reserved_trials=job.reserved_native_trials,
        max_candidates=job.max_exact_programs,
    )
    baseline_ids = frozenset(incumbent_by_output) | frozenset(
        str(row["output_bundle_id"]) for row in cold_rows
    )

    targets = tuple(pair.output for pair in task.train)
    if any(target is None for target in targets):
        raise ValueError("blind demonstration lost its output")
    candidate_rows = []
    evaluated_parents = 0
    for raw_parent in _sequence(job.sensor_task["parents"], field="sensor parents"):
        parent_row = _object(raw_parent, field="sensor parent")
        parent = ScenePipelineProgram.from_json_dict(parent_row["parent_program"])
        demo_executions = tuple(
            execute_scene_with_cell_provenance(parent, pair.input)
            for pair in task.train
        )
        core = compile_relational_failure_core(
            blind_task_id=task.task_id,
            executions=demo_executions,
            targets=targets,  # type: ignore[arg-type]
        )
        if core.core_id != _object(parent_row["core"], field="sensor core")["core_id"]:
            raise ValueError("frontier replay changed the failure core")
        evaluated_parents += 1
        query_executions = tuple(
            execute_scene_with_cell_provenance(parent, query)
            for query in task.test_inputs
        )
        bundles = candidate_output_bundle(core, query_executions)
        candidates_by_bundle: dict[str, RelationalEffectNode] = {}
        for candidate in core.candidates:
            outputs = tuple(
                execute_relational_effect(candidate, execution)
                for execution in query_executions
            ) if all(execution.exact for execution in query_executions) else ()
            if outputs:
                output_id = _bundle_id(outputs)
                incumbent = candidates_by_bundle.get(output_id)
                if incumbent is None or candidate.sort_key < incumbent.sort_key:
                    candidates_by_bundle[output_id] = candidate
        if set(candidates_by_bundle) != {_bundle_id(bundle) for bundle in bundles}:
            raise ValueError("candidate bundle replay differs from core execution")
        for bundle in bundles:
            output_id = _bundle_id(bundle)
            candidate = candidates_by_bundle[output_id]
            identity = {
                "provider_version": RELATIONAL_EFFECT_PROVIDER_VERSION,
                "parent_program_id": object_code_program_id(parent),
                "core_id": core.core_id,
                "effect_node_id": candidate.node_id,
                "output_bundle_id": output_id,
            }
            candidate_rows.append(
                {
                    "candidate_id": canonical_sha256(identity),
                    **identity,
                    "parent_program": parent.to_json_dict(),
                    "effect_node": candidate.to_json_dict(),
                    "query_outputs": [grid_to_lists(output) for output in bundle],
                    "globally_novel": output_id not in baseline_ids,
                    "description_bits": parent.description_bits + 16,
                }
            )

    ordered = sorted(
        candidate_rows,
        key=lambda row: (
            row["description_bits"],
            row["parent_program_id"],
            row["candidate_id"],
        ),
    )
    by_output: dict[str, dict[str, object]] = {}
    for row in ordered:
        output_id = str(row["output_bundle_id"])
        if output_id not in by_output:
            by_output[output_id] = row
    candidates = tuple(by_output.values())
    novel = tuple(row for row in candidates if row["globally_novel"] is True)
    realized_effect_trials = evaluated_parents * RELATIONAL_EFFECT_PROGRAMS_PER_DELTA
    if realized_effect_trials > job.reserved_native_trials:
        raise ValueError("relational effect domain exceeds its native reservation")
    return {
        "task_id": job.sensor_task["task_id"],
        "family_id": job.sensor_task["family_id"],
        "incumbent_candidates": list(incumbent_by_output.values()),
        "cold_candidates": list(cold_rows),
        "relational_effect_candidates": list(candidates),
        "novel_candidates": list(novel),
        "novel_frontier_count": len(novel),
        "frontier_opportunity": bool(novel),
        "cost": {
            "effect_programs_per_parent": RELATIONAL_EFFECT_PROGRAMS_PER_DELTA,
            "realized_effect_trials": realized_effect_trials,
            "effect_padding_trials": job.reserved_native_trials - realized_effect_trials,
            "reserved_effect_trials": job.reserved_native_trials,
            "cold_realized_trials": cold_trials,
            "cold_padding_trials": cold_padding,
            "reserved_cold_trials": job.reserved_native_trials,
            "reserved_effect_demo_executions": (
                job.reserved_native_trials * int(job.sensor_task["demo_count"])
            ),
            "reserved_effect_query_executions": (
                job.reserved_native_trials * int(job.sensor_task["query_count"])
            ),
        },
    }


def freeze_relational_effect_opportunities(
    *,
    challenges: Mapping[str, object],
    seal: Mapping[str, object],
    seal_sha256: str,
    source_files: Mapping[str, str],
    max_first_stage_trials: int = 2_000,
    max_exact_programs: int = 32,
    max_parents: int = 4,
    reserved_native_trials: int = 2_048,
    max_workers: int = 1,
) -> dict[str, object]:
    """Run G1, and only on success execute the query-blind G2 frontier."""

    if not challenges:
        raise ValueError("relational effect cohort must not be empty")
    if type(max_workers) is not int or max_workers < 1:
        raise ValueError("worker count must be positive")
    if max_parents * RELATIONAL_EFFECT_PROGRAMS_PER_DELTA > reserved_native_trials:
        raise ValueError("effect language exceeds the native reservation")
    cohort_id, family_by_task = _validated_seal(seal=seal, challenges=challenges)
    sensor_jobs = tuple(
        _SensorJob(
            task_id,
            family_by_task[task_id],
            _object(challenges[task_id], field=f"challenge {task_id}"),
            max_first_stage_trials,
            max_exact_programs,
            max_parents,
        )
        for task_id in sorted(challenges)
    )
    if max_workers == 1:
        sensor_tasks = tuple(_sensor_task(job) for job in sensor_jobs)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            sensor_tasks = tuple(executor.map(_sensor_task, sensor_jobs))
    sensor = _sensor_summary(sensor_tasks)

    frontier_tasks: tuple[dict[str, object], ...] = ()
    if sensor["passed"] is True:
        frontier_jobs = tuple(
            _FrontierJob(
                sensor_task,
                _object(
                    challenges[str(sensor_task["task_id"])],
                    field=f"challenge {sensor_task['task_id']}",
                ),
                max_first_stage_trials,
                reserved_native_trials,
                max_exact_programs,
            )
            for sensor_task in sensor_tasks
        )
        if max_workers == 1:
            frontier_tasks = tuple(_frontier_task(job) for job in frontier_jobs)
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                frontier_tasks = tuple(executor.map(_frontier_task, frontier_jobs))
    opportunity_tasks = {
        str(task["task_id"])
        for task in frontier_tasks
        if task["frontier_opportunity"] is True
    }
    opportunity_families = {
        str(task["family_id"])
        for task in frontier_tasks
        if task["frontier_opportunity"] is True
    }
    frontier_gate_passed = (
        len(opportunity_tasks) >= MINIMUM_FRONTIER_TASKS
        and len(opportunity_families) >= MINIMUM_FRONTIER_FAMILIES
    )
    frontier_by_task = {str(task["task_id"]): task for task in frontier_tasks}
    tasks = [
        {
            **dict(sensor_task),
            "frontier": frontier_by_task.get(str(sensor_task["task_id"])),
        }
        for sensor_task in sensor_tasks
    ]
    content: dict[str, object] = {
        "schema": RELATIONAL_EFFECT_FREEZE_SCHEMA,
        "provider_version": RELATIONAL_EFFECT_PROVIDER_VERSION,
        "cohort_id": cohort_id,
        "scientific_lane": SCIENTIFIC_LANE,
        "seal_id": seal["seal_id"],
        "seal_sha256": seal_sha256,
        "query_gold_read": False,
        "public_evaluation_read": False,
        "controller_training_started": False,
        "execution_workers": max_workers,
        "source_files": dict(sorted(source_files.items())),
        "search_contract": {
            "max_first_stage_trials": max_first_stage_trials,
            "max_exact_programs": max_exact_programs,
            "max_parents": max_parents,
            "effect_programs_per_delta": RELATIONAL_EFFECT_PROGRAMS_PER_DELTA,
            "reserved_native_trials": reserved_native_trials,
            "equal_cost_control": "cold_candidates",
        },
        "task_count": len(tasks),
        "family_count": len(set(family_by_task.values())),
        "sensor": sensor,
        "frontier_evaluated": sensor["passed"],
        "opportunity_count": len(opportunity_tasks),
        "opportunity_family_count": len(opportunity_families),
        "frontier_gate_passed": frontier_gate_passed,
        "frontier_thresholds": {
            "minimum_tasks": MINIMUM_FRONTIER_TASKS,
            "minimum_families": MINIMUM_FRONTIER_FAMILIES,
        },
        "tasks": tasks,
    }
    return {"freeze_id": canonical_sha256(content), **content}


def _candidate_outputs(candidate: Mapping[str, object]) -> tuple[Grid, ...]:
    outputs = tuple(
        as_grid(item)
        for item in _sequence(candidate["query_outputs"], field="candidate outputs")
    )
    if candidate["output_bundle_id"] != _bundle_id(outputs):
        raise ValueError("candidate output bundle ID differs from its content")
    return outputs


def score_relational_effect_freeze(
    *,
    freeze: Mapping[str, object],
    solutions: Mapping[str, object],
    solution_source_sha256: str,
) -> dict[str, object]:
    content = {key: value for key, value in freeze.items() if key != "freeze_id"}
    if freeze["freeze_id"] != canonical_sha256(content):
        raise ValueError("relational effect freeze ID differs")
    if freeze["query_gold_read"] is not False:
        raise ValueError("relational effect freeze is not query-blind")
    if freeze["frontier_gate_passed"] is not True:
        raise PermissionError("frontier gate did not authorize oracle scoring")
    task_rows = tuple(
        _object(item, field="freeze task")
        for item in _sequence(freeze["tasks"], field="freeze tasks")
    )
    if {str(task["task_id"]) for task in task_rows} != set(solutions):
        raise ValueError("solution and freeze task sets differ")
    counts = {
        "incumbent_exact": 0,
        "cold_exact": 0,
        "cold_unique_over_incumbent": 0,
        "relational_effect_exact": 0,
        "relational_effect_pass_at_2": 0,
        "unique_recovery_over_incumbent_and_cold": 0,
    }
    recovered_families = set()
    scored_tasks = []
    for task in task_rows:
        task_id = str(task["task_id"])
        gold = tuple(
            as_grid(item)
            for item in _sequence(solutions[task_id], field="solution outputs")
        )
        frontier = _object(task["frontier"], field="task frontier")
        incumbent = tuple(
            _object(item, field="incumbent candidate")
            for item in _sequence(
                frontier["incumbent_candidates"], field="incumbent candidates"
            )
        )
        cold = tuple(
            _object(item, field="cold candidate")
            for item in _sequence(frontier["cold_candidates"], field="cold candidates")
        )
        novel = tuple(
            _object(item, field="novel relational candidate")
            for item in _sequence(frontier["novel_candidates"], field="novel candidates")
        )
        incumbent_exact = any(_candidate_outputs(row) == gold for row in incumbent)
        cold_exact = any(_candidate_outputs(row) == gold for row in cold)
        effect_exact = any(_candidate_outputs(row) == gold for row in novel)
        effect_pass_at_2 = any(_candidate_outputs(row) == gold for row in novel[:2])
        cold_unique = cold_exact and not incumbent_exact
        unique_recovery = effect_exact and not incumbent_exact and not cold_exact
        values = {
            "incumbent_exact": incumbent_exact,
            "cold_exact": cold_exact,
            "cold_unique_over_incumbent": cold_unique,
            "relational_effect_exact": effect_exact,
            "relational_effect_pass_at_2": effect_pass_at_2,
            "unique_recovery_over_incumbent_and_cold": unique_recovery,
        }
        for name, value in values.items():
            counts[name] += int(value)
        if unique_recovery:
            recovered_families.add(str(task["family_id"]))
        scored_tasks.append(
            {"task_id": task_id, "family_id": task["family_id"], **values}
        )
    gate_passed = (
        counts["unique_recovery_over_incumbent_and_cold"]
        >= MINIMUM_UNIQUE_RECOVERIES
        and len(recovered_families) >= MINIMUM_RECOVERY_FAMILIES
        and counts["unique_recovery_over_incumbent_and_cold"]
        > counts["cold_unique_over_incumbent"]
    )
    if gate_passed:
        outcome = "promote_to_residual_intervention"
    elif counts["unique_recovery_over_incumbent_and_cold"] > 0:
        outcome = "boundary_natural_utility"
    else:
        outcome = "natural_utility_null"
    result_content: dict[str, object] = {
        "schema": RELATIONAL_EFFECT_RESULT_SCHEMA,
        "freeze_id": freeze["freeze_id"],
        "solution_source_sha256": solution_source_sha256,
        "task_count": freeze["task_count"],
        "family_count": freeze["family_count"],
        "metrics": counts,
        "unique_recovery_family_count": len(recovered_families),
        "natural_utility_thresholds": {
            "minimum_unique_recoveries": MINIMUM_UNIQUE_RECOVERIES,
            "minimum_recovery_families": MINIMUM_RECOVERY_FAMILIES,
            "must_exceed_cold_unique": True,
        },
        "natural_utility_gate_passed": gate_passed,
        "controller_training_started": False,
        "outcome": outcome,
        "tasks": scored_tasks,
    }
    return {"result_id": canonical_sha256(result_content), **result_content}


def audit_relational_effect_sensor(
    freeze: Mapping[str, object],
) -> dict[str, object]:
    """Build a query-gold-free, mutually exclusive G1 failure matrix."""

    content = {key: value for key, value in freeze.items() if key != "freeze_id"}
    if freeze["freeze_id"] != canonical_sha256(content):
        raise ValueError("relational effect freeze ID differs")
    if freeze["query_gold_read"] is not False:
        raise ValueError("sensor audit requires a query-blind freeze")
    tasks = tuple(
        _object(item, field="freeze task")
        for item in _sequence(freeze["tasks"], field="freeze tasks")
    )
    parent_failure_counts: dict[str, int] = {}
    fold_outcome_counts: dict[str, int] = {}
    terminal_counts: dict[str, int] = {}
    family_terminal_counts: dict[str, dict[str, int]] = {}
    changed_cell_count = 0
    source_backed_cell_count = 0
    entity_backed_cell_count = 0
    single_delta_demo_count = 0
    task_rows = []
    for task in tasks:
        parents = tuple(
            _object(item, field="sensor parent")
            for item in _sequence(task["parents"], field="sensor parents")
        )
        failure_kinds = []
        identified_parent_count = 0
        strict_parent_count = 0
        for parent in parents:
            core = _object(parent["core"], field="failure core")
            failure_kind = core["failure_kind"]
            key = "identified" if core["status"] == "identified" else str(failure_kind)
            parent_failure_counts[key] = parent_failure_counts.get(key, 0) + 1
            failure_kinds.append(key)
            identified_parent_count += int(core["status"] == "identified")
            lodo = _object(parent["lodo"], field="parent LODO")
            strict_parent_count += int(lodo["strict_all_folds_exact"] is True)
            for raw_fold in _sequence(lodo["folds"], field="LODO folds"):
                fold = _object(raw_fold, field="LODO fold")
                if fold["exact"] is True:
                    fold_key = "unanimous_exact"
                elif fold["identified"] is True and fold["unanimous"] is True:
                    fold_key = "unanimous_wrong"
                elif fold["identified"] is True:
                    fold_key = "ambiguous_prediction"
                else:
                    fold_key = f"unidentified:{fold['failure_kind']}"
                fold_outcome_counts[fold_key] = fold_outcome_counts.get(fold_key, 0) + 1
            for raw_summary in _sequence(
                core["effect_summaries"], field="effect summaries"
            ):
                summary = _object(raw_summary, field="effect summary")
                if summary["status"] != "single_delta":
                    continue
                single_delta_demo_count += 1
                changed_cell_count += len(
                    _sequence(summary["changed_cells"], field="changed cells")
                )
                source_backed_cell_count += len(
                    _sequence(
                        summary["source_backed_cells"], field="source-backed cells"
                    )
                )
                entity_backed_cell_count += len(
                    _sequence(
                        summary["entity_backed_cells"], field="entity-backed cells"
                    )
                )

        baselines = _object(task["baseline_strict_lodo"], field="baseline LODO")
        any_baseline = any(baselines[name] is True for name in baselines)
        if not parents:
            terminal = "no_scene_parent"
        elif task["strict_lodo"] is True and any_baseline:
            terminal = "strict_but_baseline_redundant"
        elif task["strict_lodo"] is True:
            terminal = "strict_unique_sensor"
        elif identified_parent_count:
            terminal = "identified_but_not_lodo_generalizing"
        elif "relation_effect_language_unreachable" in failure_kinds:
            terminal = "relation_effect_language_unreachable"
        elif "cross_demo_delta_inconsistent" in failure_kinds:
            terminal = "cross_demo_delta_inconsistent"
        elif "delta_semantics_inconsistent" in failure_kinds:
            terminal = "delta_semantics_inconsistent"
        elif "shape_incompatible" in failure_kinds:
            terminal = "shape_incompatible"
        elif "provenance_unsupported" in failure_kinds:
            terminal = "provenance_unsupported"
        else:
            terminal = "unclassified_sensor_failure"
        terminal_counts[terminal] = terminal_counts.get(terminal, 0) + 1
        family_id = str(task["family_id"])
        family_counts = family_terminal_counts.setdefault(family_id, {})
        family_counts[terminal] = family_counts.get(terminal, 0) + 1
        task_rows.append(
            {
                "task_id": task["task_id"],
                "family_id": family_id,
                "terminal_reason": terminal,
                "parent_count": len(parents),
                "identified_parent_count": identified_parent_count,
                "strict_parent_count": strict_parent_count,
                "baseline_strict_lodo": dict(baselines),
                "parent_failure_kinds": sorted(set(failure_kinds)),
            }
        )
    audit_content: dict[str, object] = {
        "schema": RELATIONAL_EFFECT_FAILURE_AUDIT_SCHEMA,
        "freeze_id": freeze["freeze_id"],
        "query_gold_read": False,
        "controller_training_started": False,
        "task_count": len(tasks),
        "family_count": len(family_terminal_counts),
        "terminal_reason_counts": dict(sorted(terminal_counts.items())),
        "parent_failure_counts": dict(sorted(parent_failure_counts.items())),
        "lodo_fold_outcome_counts": dict(sorted(fold_outcome_counts.items())),
        "single_delta_demo_count": single_delta_demo_count,
        "changed_cell_count": changed_cell_count,
        "source_backed_cell_count": source_backed_cell_count,
        "entity_backed_cell_count": entity_backed_cell_count,
        "source_backed_recall": (
            source_backed_cell_count / changed_cell_count
            if changed_cell_count
            else 0.0
        ),
        "entity_backed_recall": (
            entity_backed_cell_count / changed_cell_count
            if changed_cell_count
            else 0.0
        ),
        "family_terminal_reason_counts": {
            family_id: dict(sorted(counts.items()))
            for family_id, counts in sorted(family_terminal_counts.items())
        },
        "tasks": task_rows,
    }
    return {"audit_id": canonical_sha256(audit_content), **audit_content}


def _intervened_effect_outputs(
    *,
    task: BlindTask,
    candidate: Mapping[str, object],
    node: RelationalEffectNode,
    lesion_entities: bool,
) -> tuple[Grid, ...] | None:
    parent = ScenePipelineProgram.from_json_dict(candidate["parent_program"])
    outputs = []
    for query in task.test_inputs:
        execution = execute_scene_with_cell_provenance(parent, query)
        if not execution.exact:
            return None
        if lesion_entities:
            execution = lesion_entity_links(execution)
        outputs.append(execute_relational_effect(node, execution))
    return tuple(outputs)


def _raw_alignment_outputs(
    *,
    task: BlindTask,
    candidate: Mapping[str, object],
) -> tuple[Grid, ...] | None:
    parent = ScenePipelineProgram.from_json_dict(candidate["parent_program"])
    node = RelationalEffectNode.from_json_dict(candidate["effect_node"])
    old_node = AnchorRasterizedDeltaNode(
        node.anchor_color,
        node.effect_kind,
        node.effect_parameter,
        node.source_color,
        node.target_color,
    )
    outputs = []
    for query in task.test_inputs:
        execution = execute_scene_with_cell_provenance(parent, query)
        if not execution.exact or execution.output is None:
            return None
        if (len(query), len(query[0])) != (
            len(execution.output),
            len(execution.output[0]),
        ):
            return None
        outputs.append(
            execute_anchor_rasterized_delta(
                old_node,
                query,
                execution.output,
                execution.parent_execution.parse_state.scene,
            )
        )
    return tuple(outputs)


def evaluate_relational_effect_interventions(
    *,
    challenges: Mapping[str, object],
    freeze: Mapping[str, object],
    result: Mapping[str, object],
    solutions: Mapping[str, object],
) -> dict[str, object]:
    """Run G4 only after the natural-utility result has opened the gate."""

    freeze_content = {
        key: value for key, value in freeze.items() if key != "freeze_id"
    }
    if freeze["freeze_id"] != canonical_sha256(freeze_content):
        raise ValueError("intervention freeze ID differs")
    result_content = {
        key: value for key, value in result.items() if key != "result_id"
    }
    if result["result_id"] != canonical_sha256(result_content):
        raise ValueError("intervention result ID differs")
    if result["freeze_id"] != freeze["freeze_id"]:
        raise ValueError("intervention result belongs to another freeze")
    if result["natural_utility_gate_passed"] is not True:
        raise PermissionError("natural utility did not authorize residual intervention")
    freeze_tasks = {
        str(task["task_id"]): _object(task, field="freeze task")
        for task in _sequence(freeze["tasks"], field="freeze tasks")
    }
    recovered = tuple(
        sorted(
            str(task["task_id"])
            for task in _sequence(result["tasks"], field="result tasks")
            if _object(task, field="result task")[
                "unique_recovery_over_incumbent_and_cold"
            ]
            is True
        )
    )
    if set(freeze_tasks) != set(challenges) or set(freeze_tasks) != set(solutions):
        raise ValueError("intervention task artifacts differ")
    if len(recovered) < MINIMUM_UNIQUE_RECOVERIES:
        raise ValueError("natural utility result contradicts its recovery threshold")

    rows = []
    counts = {
        "intact": 0,
        "cleared_core": 0,
        "family_shuffled_core": 0,
        "raw_alignment_bridge_lesion": 0,
        "entity_link_lesion": 0,
    }
    changed_shuffled_actions = 0
    for index, task_id in enumerate(recovered):
        task = blind_task_from_challenge(
            _object(challenges[task_id], field=f"challenge {task_id}")
        )
        gold = tuple(
            as_grid(item)
            for item in _sequence(solutions[task_id], field="solution outputs")
        )
        frontier = _object(freeze_tasks[task_id]["frontier"], field="task frontier")
        candidates = tuple(
            _object(item, field="novel candidate")
            for item in _sequence(frontier["novel_candidates"], field="novel candidates")
        )
        intact = any(_candidate_outputs(candidate) == gold for candidate in candidates)

        donor_id = recovered[(index + 1) % len(recovered)]
        donor_frontier = _object(
            freeze_tasks[donor_id]["frontier"], field="donor frontier"
        )
        donor_candidates = tuple(
            _object(item, field="donor candidate")
            for item in _sequence(
                donor_frontier["novel_candidates"], field="donor novel candidates"
            )
        )
        donor_nodes = tuple(
            RelationalEffectNode.from_json_dict(candidate["effect_node"])
            for candidate in donor_candidates
        )
        own_node_ids = {
            str(candidate["effect_node_id"]) for candidate in candidates
        }
        changed_shuffled_actions += int(
            any(node.node_id not in own_node_ids for node in donor_nodes)
        )
        shuffled = any(
            outputs == gold
            for candidate in candidates
            for node in donor_nodes
            for outputs in (
                _intervened_effect_outputs(
                    task=task,
                    candidate=candidate,
                    node=node,
                    lesion_entities=False,
                ),
            )
            if outputs is not None
        )
        raw_alignment = any(
            outputs == gold
            for candidate in candidates
            for outputs in (
                _raw_alignment_outputs(task=task, candidate=candidate),
            )
            if outputs is not None
        )
        entity_lesion = any(
            outputs == gold
            for candidate in candidates
            for outputs in (
                _intervened_effect_outputs(
                    task=task,
                    candidate=candidate,
                    node=RelationalEffectNode.from_json_dict(
                        candidate["effect_node"]
                    ),
                    lesion_entities=True,
                ),
            )
            if outputs is not None
        )
        values = {
            "intact": intact,
            "cleared_core": False,
            "family_shuffled_core": shuffled,
            "raw_alignment_bridge_lesion": raw_alignment,
            "entity_link_lesion": entity_lesion,
        }
        for name, value in values.items():
            counts[name] += int(value)
        rows.append(
            {
                "task_id": task_id,
                "family_id": freeze_tasks[task_id]["family_id"],
                "donor_task_id": donor_id,
                **values,
            }
        )
    effects = {
        "cleared_core_loss": counts["intact"] - counts["cleared_core"],
        "family_shuffled_core_loss": (
            counts["intact"] - counts["family_shuffled_core"]
        ),
        "raw_alignment_bridge_loss": (
            counts["intact"] - counts["raw_alignment_bridge_lesion"]
        ),
        "entity_link_lesion_loss": (
            counts["intact"] - counts["entity_link_lesion"]
        ),
    }
    mechanism_passed = (
        effects["cleared_core_loss"] >= 3
        and effects["family_shuffled_core_loss"] >= 3
        and effects["raw_alignment_bridge_loss"] >= 3
        and changed_shuffled_actions >= 3
    )
    intervention_content: dict[str, object] = {
        "schema": RELATIONAL_EFFECT_INTERVENTION_SCHEMA,
        "freeze_id": freeze["freeze_id"],
        "result_id": result["result_id"],
        "query_gold_read": True,
        "controller_training_started": False,
        "recovered_task_count": len(recovered),
        "counts": counts,
        "effects": effects,
        "changed_shuffled_action_task_count": changed_shuffled_actions,
        "mechanism_gate_passed": mechanism_passed,
        "tasks": rows,
    }
    return {
        "intervention_id": canonical_sha256(intervention_content),
        **intervention_content,
    }
