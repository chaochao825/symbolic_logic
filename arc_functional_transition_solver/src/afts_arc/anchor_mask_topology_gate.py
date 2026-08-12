"""Query-blind natural-utility gate for one anchor-rasterized delta node."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from .anchor_mask import (
    ANCHOR_MASK_PROGRAMS_PER_DELTA,
    AnchorRasterizedDeltaNode,
)
from .anchor_mask_reachability import (
    analyze_anchor_mask_reachability,
    insert_anchor_mask_hole,
    leave_one_demo_out_anchor_masks,
)
from .blind import BlindTask
from .executable_workspace import (
    TypedProgramSketch,
    abstract_execute,
    execute_typed_sketch,
    fill_obligation,
)
from .experiment_safety import canonical_sha256
from .grid import Grid, as_grid, grid_key, grid_to_lists
from .hybrid.object_code import object_code_program_id
from .hybrid.scene_graph import ScenePipelineProgram
from .object_graph_rewrite_gate import blind_task_from_challenge
from .recolor_topology_gate import reconstruct_frozen_scene_parents


ANCHOR_MASK_PROVIDER_VERSION = "afts-anchor-rasterized-delta/v0.3"
ANCHOR_MASK_FREEZE_SCHEMA = "afts.anchor-mask-topology-freeze/v1"
ANCHOR_MASK_RESULT_SCHEMA = "afts.anchor-mask-topology-result/v1"
ANCHOR_MASK_PROMOTION_RECOVERIES = 3
SCIENTIFIC_LANES = (
    "controlled_semantic",
    "outcome_exposed_development",
    "prospective_reserve",
)


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


def _candidate_outputs(candidate: Mapping[str, object], *, field: str) -> tuple[Grid, ...]:
    outputs = tuple(
        as_grid(item)
        for item in _sequence(candidate["query_outputs"], field=f"{field} outputs")
    )
    output_bundle_id = candidate["output_bundle_id"]
    if not isinstance(output_bundle_id, str) or not output_bundle_id:
        raise TypeError(f"{field} output bundle ID must be a non-empty string")
    if output_bundle_id != _bundle_id(outputs):
        raise ValueError(f"{field} output bundle ID differs from its outputs")
    return outputs


def _baseline_output_ids(parent_task: Mapping[str, object]) -> frozenset[str]:
    rows = tuple(
        _object(item, field="incumbent candidate")
        for item in _sequence(
            parent_task["incumbent_candidates"], field="incumbent candidates"
        )
    ) + tuple(
        _object(item, field="cold candidate")
        for item in _sequence(
            parent_task["frozen_cold_candidates"], field="cold candidates"
        )
    )
    output_ids = []
    for index, row in enumerate(rows):
        _candidate_outputs(row, field=f"baseline candidate {index}")
        output_ids.append(str(row["output_bundle_id"]))
    return frozenset(output_ids)


def _validated_parent_tasks(
    parent_freeze: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    if parent_freeze["query_gold_read"] is not False:
        raise ValueError("parent freeze is not query-blind")
    content = {key: value for key, value in parent_freeze.items() if key != "freeze_id"}
    if parent_freeze["freeze_id"] != canonical_sha256(content):
        raise ValueError("parent freeze ID differs from canonical content")
    tasks: dict[str, Mapping[str, object]] = {}
    for raw_task in _sequence(parent_freeze["tasks"], field="parent tasks"):
        task = _object(raw_task, field="parent task")
        task_id = task["task_id"]
        if not isinstance(task_id, str) or not task_id:
            raise TypeError("parent task ID must be a non-empty string")
        if task_id in tasks:
            raise ValueError("parent freeze repeats a task ID")
        tasks[task_id] = task
    if parent_freeze["task_count"] != len(tasks):
        raise ValueError("parent freeze task count differs")
    return tasks


def _candidate_row(
    *,
    parent: ScenePipelineProgram,
    parent_certificate: Mapping[str, object],
    completed_sketch: TypedProgramSketch,
    mask_node: AnchorRasterizedDeltaNode,
    reachability_id: str,
    insertion_obligation_id: str,
    fill_obligation_id: str,
    query_outputs: tuple[Grid, ...],
    globally_novel: bool,
) -> dict[str, object]:
    parent_program_id = object_code_program_id(parent)
    output_bundle_id = _bundle_id(query_outputs)
    identity = {
        "provider_version": ANCHOR_MASK_PROVIDER_VERSION,
        "parent_program_id": parent_program_id,
        "completed_sketch_id": completed_sketch.sketch_id,
        "mask_node": mask_node.to_json_dict(),
        "output_bundle_id": output_bundle_id,
    }
    return {
        "candidate_id": canonical_sha256(identity),
        **identity,
        "parent_certificate_id": parent_certificate["certificate_id"],
        "reachability_id": reachability_id,
        "insertion_obligation_id": insertion_obligation_id,
        "fill_obligation_id": fill_obligation_id,
        "completed_sketch": completed_sketch.to_json_dict(),
        "query_outputs": [grid_to_lists(output) for output in query_outputs],
        "description_bits": parent.description_bits + 16,
        "globally_novel": globally_novel,
    }


def evaluate_anchor_mask_parents(
    *,
    task: BlindTask,
    parents: Sequence[ScenePipelineProgram],
    parent_certificates: Sequence[Mapping[str, object]],
    baseline_output_bundle_ids: frozenset[str],
    reserved_native_trials: int,
) -> dict[str, object]:
    """Apply the frozen mask version space to reconstructed scene parents."""

    if not isinstance(task, BlindTask):
        raise TypeError("anchor-mask parent evaluation accepts BlindTask only")
    if type(reserved_native_trials) is not int or reserved_native_trials < 1:
        raise ValueError("reserved native trials must be a positive integer")
    parent_tuple = tuple(parents)
    certificate_tuple = tuple(parent_certificates)
    if len(parent_tuple) != len(certificate_tuple):
        raise ValueError("anchor-mask parents and certificates must align")
    if any(not isinstance(parent, ScenePipelineProgram) for parent in parent_tuple):
        raise TypeError("anchor-mask parents must be scene-pipeline programs")

    realized_native_trials = len(parent_tuple) * ANCHOR_MASK_PROGRAMS_PER_DELTA
    if realized_native_trials > reserved_native_trials:
        raise ValueError("anchor-mask domain exceeds the native trial reservation")

    parent_rows = []
    candidates = []
    for parent_index, (parent, certificate) in enumerate(
        zip(parent_tuple, certificate_tuple, strict=True)
    ):
        parent_program_id = object_code_program_id(parent)
        if certificate["parent_program_id"] != parent_program_id:
            raise ValueError("reconstructed mask parent differs from its certificate")
        base_sketch = TypedProgramSketch.from_scene_pipeline(parent)
        reachability = analyze_anchor_mask_reachability(base_sketch, task)
        lodo = leave_one_demo_out_anchor_masks(base_sketch, task)
        parent_candidate_ids = []
        if reachability.status == "reachable":
            insertion = reachability.obligations[0]
            inserted = insert_anchor_mask_hole(
                base_sketch,
                insertion,
                hole_id=f"anchor-mask-hole-{parent_index}",
            )
            incomplete = abstract_execute(inserted, task)
            if incomplete.status != "incomplete" or len(incomplete.obligations) != 1:
                raise ValueError("inserted anchor-mask sketch has an invalid fill obligation")
            fill = incomplete.obligations[0]
            for mask_node in reachability.candidates:
                completed = fill_obligation(inserted, fill, mask_node)
                verification = abstract_execute(completed, task)
                if not verification.exact:
                    raise ValueError("reachable anchor-mask candidate failed demo replay")
                outputs = []
                for query in task.test_inputs:
                    execution = execute_typed_sketch(completed, query)
                    if not execution.ok or execution.output is None:
                        raise ValueError("demo-exact anchor-mask candidate failed query replay")
                    outputs.append(execution.output)
                output_tuple = tuple(outputs)
                globally_novel = _bundle_id(output_tuple) not in baseline_output_bundle_ids
                row = _candidate_row(
                    parent=parent,
                    parent_certificate=certificate,
                    completed_sketch=completed,
                    mask_node=mask_node,
                    reachability_id=reachability.result_id,
                    insertion_obligation_id=insertion.obligation_id,
                    fill_obligation_id=fill.obligation_id,
                    query_outputs=output_tuple,
                    globally_novel=globally_novel,
                )
                candidates.append(row)
                parent_candidate_ids.append(row["candidate_id"])
        parent_rows.append(
            {
                "parent_program_id": parent_program_id,
                "certificate_id": certificate["certificate_id"],
                "base_sketch_id": base_sketch.sketch_id,
                "reachability": reachability.to_json_dict(),
                "lodo": lodo,
                "candidate_ids": parent_candidate_ids,
                "native_program_trials": ANCHOR_MASK_PROGRAMS_PER_DELTA,
                "reserved_lodo_audit_trials": (
                    len(task.train) * ANCHOR_MASK_PROGRAMS_PER_DELTA
                ),
            }
        )

    ordered = sorted(
        candidates,
        key=lambda row: (
            row["description_bits"],
            row["parent_program_id"],
            row["candidate_id"],
        ),
    )
    by_output: dict[str, dict[str, object]] = {}
    for row in ordered:
        output_bundle_id = str(row["output_bundle_id"])
        if output_bundle_id not in by_output:
            by_output[output_bundle_id] = row
    frontier = tuple(by_output.values())
    novel = tuple(row for row in frontier if row["globally_novel"] is True)
    return {
        "parent_count": len(parent_tuple),
        "reachable_parent_count": sum(
            row["reachability"]["status"] == "reachable" for row in parent_rows
        ),
        "strict_lodo_parent_count": sum(
            row["lodo"]["strict_all_folds_exact"] is True for row in parent_rows
        ),
        "parent_evaluations": parent_rows,
        "demo_exact_candidates": list(frontier),
        "novel_candidates": list(novel),
        "cost": {
            "domain_programs_per_parent": ANCHOR_MASK_PROGRAMS_PER_DELTA,
            "realized_program_trials": realized_native_trials,
            "padding_program_trials": reserved_native_trials - realized_native_trials,
            "reserved_program_trials": reserved_native_trials,
            "reserved_demo_executions": reserved_native_trials * len(task.train),
            "reserved_query_executions": reserved_native_trials * len(task.test_inputs),
            "separate_lodo_audit_trials": (
                len(parent_tuple)
                * len(task.train)
                * ANCHOR_MASK_PROGRAMS_PER_DELTA
            ),
        },
    }


@dataclass(frozen=True, slots=True)
class _FreezeTaskJob:
    task_id: str
    raw_task: Mapping[str, object]
    parent_task: Mapping[str, object]
    max_first_stage_trials: int
    max_parents: int
    reserved_native_trials: int


def _freeze_task(job: _FreezeTaskJob) -> dict[str, object]:
    task = blind_task_from_challenge(job.raw_task)
    if job.parent_task["blind_content_sha256"] != task.blind_content_sha256:
        raise ValueError("anchor-mask task differs from its frozen blind identity")
    certificates = tuple(
        _object(item, field="parent certificate")
        for item in _sequence(job.parent_task["certificates"], field="certificates")
    )
    reconstructed = reconstruct_frozen_scene_parents(
        task,
        max_first_stage_trials=job.max_first_stage_trials,
        max_parents=job.max_parents,
    )
    expected_parent_ids = tuple(str(item["parent_program_id"]) for item in certificates)
    observed_parent_ids = tuple(
        object_code_program_id(program) for program in reconstructed.programs
    )
    if observed_parent_ids != expected_parent_ids:
        raise ValueError("reconstructed anchor-mask parent sequence drifted from v4")
    if job.parent_task["parent_count"] != len(reconstructed.programs):
        raise ValueError("reconstructed anchor-mask parent count drifted from v4")
    shared = _object(job.parent_task["shared_first_stage"], field="shared first stage")
    if shared["program_trials"] != reconstructed.first_stage_program_trials:
        raise ValueError("reconstructed first-stage program cost drifted from v4")
    if shared["demo_executions"] != reconstructed.first_stage_demo_executions:
        raise ValueError("reconstructed first-stage demo cost drifted from v4")

    evaluation = evaluate_anchor_mask_parents(
        task=task,
        parents=reconstructed.programs,
        parent_certificates=certificates,
        baseline_output_bundle_ids=_baseline_output_ids(job.parent_task),
        reserved_native_trials=job.reserved_native_trials,
    )
    return {
        "task_id": job.task_id,
        "blind_task_id": task.task_id,
        "blind_content_sha256": task.blind_content_sha256,
        "demo_count": len(task.train),
        "query_count": len(task.test_inputs),
        "incumbent_candidates": job.parent_task["incumbent_candidates"],
        "frozen_cold_candidates": job.parent_task["frozen_cold_candidates"],
        "shared_first_stage": {
            "program_trials": reconstructed.first_stage_program_trials,
            "demo_executions": reconstructed.first_stage_demo_executions,
        },
        **evaluation,
    }


def freeze_anchor_mask_opportunities(
    *,
    challenges: Mapping[str, object],
    parent_freeze: Mapping[str, object],
    parent_freeze_sha256: str,
    cohort_id: str,
    scientific_lane: str,
    source_files: Mapping[str, str],
    max_first_stage_trials: int = 2_000,
    max_parents: int = 4,
    reserved_native_trials: int = 2_048,
    max_workers: int = 1,
) -> dict[str, object]:
    """Freeze mask candidates before development query solutions are opened."""

    if not challenges:
        raise ValueError("anchor-mask cohort must not be empty")
    if scientific_lane not in SCIENTIFIC_LANES:
        raise ValueError("unknown anchor-mask scientific lane")
    if type(max_workers) is not int or max_workers < 1:
        raise ValueError("anchor-mask worker count must be positive")
    parent_tasks = _validated_parent_tasks(parent_freeze)
    if parent_freeze["cohort_id"] != cohort_id:
        raise ValueError("anchor-mask cohort ID differs from the parent freeze")
    if set(parent_tasks) != set(challenges):
        raise ValueError("anchor-mask challenge and parent task sets differ")
    parent_contract = _object(parent_freeze["search_contract"], field="parent search")
    if parent_contract["max_first_stage_trials"] != max_first_stage_trials:
        raise ValueError("anchor-mask first-stage bound differs from parent freeze")
    if parent_contract["max_parents"] != max_parents:
        raise ValueError("anchor-mask parent bound differs from parent freeze")
    if parent_contract["reserved_trials_per_arm"] != reserved_native_trials:
        raise ValueError("anchor-mask reservation differs from frozen cold arm")
    if max_parents * ANCHOR_MASK_PROGRAMS_PER_DELTA > reserved_native_trials:
        raise ValueError("anchor-mask domain does not fit the frozen reservation")

    jobs = tuple(
        _FreezeTaskJob(
            task_id,
            _object(challenges[task_id], field=f"challenge {task_id}"),
            parent_tasks[task_id],
            max_first_stage_trials,
            max_parents,
            reserved_native_trials,
        )
        for task_id in sorted(challenges)
    )
    if max_workers == 1:
        tasks = tuple(_freeze_task(job) for job in jobs)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            tasks = tuple(executor.map(_freeze_task, jobs))

    content: dict[str, object] = {
        "schema": ANCHOR_MASK_FREEZE_SCHEMA,
        "provider_version": ANCHOR_MASK_PROVIDER_VERSION,
        "cohort_id": cohort_id,
        "scientific_lane": scientific_lane,
        "parent_freeze_id": parent_freeze["freeze_id"],
        "parent_freeze_sha256": parent_freeze_sha256,
        "query_gold_read": False,
        "public_evaluation_read": False,
        "controller_training_started": False,
        "execution_workers": max_workers,
        "source_files": dict(sorted(source_files.items())),
        "search_contract": {
            "max_first_stage_trials": max_first_stage_trials,
            "max_parents": max_parents,
            "mask_programs_per_delta": ANCHOR_MASK_PROGRAMS_PER_DELTA,
            "max_realized_trials": (
                max_parents * ANCHOR_MASK_PROGRAMS_PER_DELTA
            ),
            "reserved_native_trials": reserved_native_trials,
            "equal_cost_control": "parent_freeze.frozen_cold_candidates",
            "lodo_is_audit_only": True,
        },
        "task_count": len(tasks),
        "reachable_parent_count": sum(
            int(task["reachable_parent_count"]) for task in tasks
        ),
        "reachable_task_count": sum(
            int(task["reachable_parent_count"] > 0) for task in tasks
        ),
        "strict_lodo_task_count": sum(
            int(task["strict_lodo_parent_count"] > 0) for task in tasks
        ),
        "demo_exact_candidate_task_count": sum(
            int(bool(task["demo_exact_candidates"])) for task in tasks
        ),
        "novel_frontier_task_count": sum(
            int(bool(task["novel_candidates"])) for task in tasks
        ),
        "tasks": list(tasks),
    }
    return {"freeze_id": canonical_sha256(content), **content}


def _candidate_exact(candidate: Mapping[str, object], gold: Sequence[Grid]) -> bool:
    return _candidate_outputs(candidate, field="scored candidate") == tuple(gold)


def score_anchor_mask_freeze(
    *,
    freeze: Mapping[str, object],
    solutions: Mapping[str, object],
    solution_source_sha256: str,
) -> dict[str, object]:
    """Score one immutable query-blind mask candidate freeze."""

    content = {key: value for key, value in freeze.items() if key != "freeze_id"}
    if freeze["freeze_id"] != canonical_sha256(content):
        raise ValueError("anchor-mask freeze ID differs")
    if freeze["query_gold_read"] is not False:
        raise ValueError("anchor-mask freeze is not query-blind")
    task_rows = tuple(
        _object(item, field="anchor-mask freeze task")
        for item in _sequence(freeze["tasks"], field="anchor-mask freeze tasks")
    )
    task_ids = tuple(str(task["task_id"]) for task in task_rows)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("anchor-mask freeze repeats a task ID")
    if set(task_ids) != set(solutions):
        raise ValueError("solution and anchor-mask task sets differ")

    counters = {
        "incumbent_exact": 0,
        "cold_exact": 0,
        "baseline_union_exact": 0,
        "mask_exact": 0,
        "novel_mask_exact": 0,
        "unique_recovery_over_incumbent_and_cold": 0,
        "mask_pass_at_2": 0,
    }
    scored_tasks = []
    for task in task_rows:
        task_id = str(task["task_id"])
        gold = tuple(
            as_grid(item)
            for item in _sequence(solutions[task_id], field="solution outputs")
        )
        incumbent = tuple(
            _object(item, field="incumbent candidate")
            for item in _sequence(task["incumbent_candidates"], field="incumbent candidates")
        )
        cold = tuple(
            _object(item, field="cold candidate")
            for item in _sequence(task["frozen_cold_candidates"], field="cold candidates")
        )
        candidates = tuple(
            _object(item, field="mask candidate")
            for item in _sequence(task["demo_exact_candidates"], field="mask candidates")
        )
        novel = tuple(
            _object(item, field="novel mask candidate")
            for item in _sequence(task["novel_candidates"], field="novel mask candidates")
        )
        incumbent_exact = any(_candidate_exact(item, gold) for item in incumbent)
        cold_exact = any(_candidate_exact(item, gold) for item in cold)
        baseline_exact = incumbent_exact or cold_exact
        mask_exact = any(_candidate_exact(item, gold) for item in candidates)
        novel_exact = any(_candidate_exact(item, gold) for item in novel)
        unique_recovery = novel_exact and not baseline_exact
        pass_at_2 = any(_candidate_exact(item, gold) for item in novel[:2])
        values = {
            "incumbent_exact": incumbent_exact,
            "cold_exact": cold_exact,
            "baseline_union_exact": baseline_exact,
            "mask_exact": mask_exact,
            "novel_mask_exact": novel_exact,
            "unique_recovery_over_incumbent_and_cold": unique_recovery,
            "mask_pass_at_2": pass_at_2,
        }
        for name, value in values.items():
            counters[name] += int(value)
        scored_tasks.append({"task_id": task_id, **values})

    unique_recoveries = counters["unique_recovery_over_incumbent_and_cold"]
    if freeze["strict_lodo_task_count"] < 3:
        outcome = "sensor_null"
    elif unique_recoveries >= ANCHOR_MASK_PROMOTION_RECOVERIES:
        outcome = "promote_for_prospective_confirmation"
    elif unique_recoveries > 0:
        outcome = "boundary_natural_utility"
    elif freeze["novel_frontier_task_count"] > 0:
        outcome = "natural_utility_null"
    else:
        outcome = "boundary_no_novel_frontier"

    result_content: dict[str, object] = {
        "schema": ANCHOR_MASK_RESULT_SCHEMA,
        "freeze_id": freeze["freeze_id"],
        "solution_source_sha256": solution_source_sha256,
        "scientific_lane": freeze["scientific_lane"],
        "task_count": freeze["task_count"],
        "reachable_task_count": freeze["reachable_task_count"],
        "strict_lodo_task_count": freeze["strict_lodo_task_count"],
        "novel_frontier_task_count": freeze["novel_frontier_task_count"],
        **counters,
        "promotion_threshold": ANCHOR_MASK_PROMOTION_RECOVERIES,
        "outcome": outcome,
        "tasks": scored_tasks,
    }
    return {"result_id": canonical_sha256(result_content), **result_content}
