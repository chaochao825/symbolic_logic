"""Demo-only counterfactual audit for a closed stateful-rewrite gate.

The audit never emits query candidates and never reads query labels. It asks a
narrow failure-analysis question: would another single typed node have reached
an exact demonstration program, or is the complete one-node neighborhood still
insufficient? The result diagnoses the compiler/action boundary without
reopening the solver gate.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from .blind import BlindTask
from .experiment_safety import canonical_sha256
from .hybrid.object_code import enumerate_object_code_programs, object_code_program_id
from .hybrid.scene_graph import ScenePipelineProgram, enumerate_scene_pipeline_programs
from .object_graph_rewrite_gate import blind_task_from_challenge
from .residual import compare_grids
from .stateful_object_graph_rewrite import (
    StatefulNodeFailureCertificate,
    enumerate_stateful_scene_rewrites,
    execute_stateful_scene_rewrite,
)
from .stateful_scene import STATEFUL_NODE_ORDER, execute_stateful_scene_pipeline


STATEFUL_REWRITE_FAILURE_AUDIT_SCHEMA = "afts.stateful-rewrite-failure-audit/v1"


def _object(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be an array")
    return value


def _certificate_for_node(
    *,
    task: BlindTask,
    parent: ScenePipelineProgram,
    source: StatefulNodeFailureCertificate,
    node_id: str,
) -> StatefulNodeFailureCertificate:
    return StatefulNodeFailureCertificate.create(
        task=task,
        parent=parent,
        diagnosis=(
            source.diagnosis
            if node_id == source.node_id
            else f"audit_counterfactual_node:{node_id}"
        ),
        node_id=node_id,
        mismatch_counts=source.mismatch_counts,
        shape_match=source.shape_match,
        parent_execution_ids=source.parent_execution_ids,
        evidence={
            "audit_only": True,
            "source_certificate_id": source.certificate_id,
        },
    )


def audit_parent_node_frontiers(
    *,
    task: BlindTask,
    parent: ScenePipelineProgram,
    certificate: StatefulNodeFailureCertificate,
    candidate_programs: Sequence[ScenePipelineProgram],
    max_trials_per_node: int,
) -> dict[str, object]:
    """Compare every legal one-node neighborhood on demonstrations only."""

    if type(max_trials_per_node) is not int or max_trials_per_node < 1:
        raise ValueError("audit node-trial bound must be positive")
    demo_inputs = tuple(pair.input for pair in task.train)
    parent_executions = tuple(
        execute_stateful_scene_pipeline(parent, grid) for grid in demo_inputs
    )
    if any(not execution.ok for execution in parent_executions):
        raise ValueError("failure audit requires an execution-valid parent")
    if tuple(item.execution_id for item in parent_executions) != (
        certificate.parent_execution_ids
    ):
        raise ValueError("reconstructed parent trace IDs differ from the freeze")

    node_rows = []
    for node_id in STATEFUL_NODE_ORDER:
        injected = _certificate_for_node(
            task=task,
            parent=parent,
            source=certificate,
            node_id=node_id,
        )
        rewrites = enumerate_stateful_scene_rewrites(
            parent,
            injected,
            candidate_programs=candidate_programs,
            max_trials=max_trials_per_node,
        )
        valid_count = 0
        best_mismatch_count: int | None = None
        best_description_bits: int | None = None
        best_program_id: str | None = None
        exact_program_ids = []
        for rewrite in rewrites:
            executions = tuple(
                execute_stateful_scene_rewrite(
                    rewrite,
                    pair.input,
                    parent_execution=parent_execution,
                )
                for pair, parent_execution in zip(
                    task.train, parent_executions, strict=True
                )
            )
            if any(
                not execution.ok or execution.output is None for execution in executions
            ):
                continue
            valid_count += 1
            mismatch_count = sum(
                compare_grids(
                    execution.output,
                    pair.output,
                    pair_index=index,
                    invalid_code=None,
                ).mismatch_count
                for index, (execution, pair) in enumerate(
                    zip(executions, task.train, strict=True)
                )
            )
            program_id = object_code_program_id(rewrite.program)
            if best_mismatch_count is None or (
                mismatch_count,
                rewrite.program.description_bits,
                program_id,
            ) < (
                best_mismatch_count,
                best_description_bits,
                best_program_id,
            ):
                best_mismatch_count = mismatch_count
                best_description_bits = rewrite.program.description_bits
                best_program_id = program_id
            if mismatch_count == 0:
                exact_program_ids.append(program_id)
        node_rows.append(
            {
                "node_id": node_id,
                "selected_by_compiler": node_id == certificate.node_id,
                "trial_count": len(rewrites),
                "valid_count": valid_count,
                "best_mismatch_count": best_mismatch_count,
                "best_program_id": best_program_id,
                "demo_exact_program_ids": sorted(set(exact_program_ids)),
            }
        )
    selected_row = next(
        row for row in node_rows if row["node_id"] == certificate.node_id
    )
    exact_nodes = tuple(
        row["node_id"] for row in node_rows if row["demo_exact_program_ids"]
    )
    available_best = tuple(
        row["best_mismatch_count"]
        for row in node_rows
        if row["best_mismatch_count"] is not None
    )
    return {
        "parent_program_id": object_code_program_id(parent),
        "source_certificate_id": certificate.certificate_id,
        "selected_node_id": certificate.node_id,
        "parent_mismatch_count": sum(certificate.mismatch_counts),
        "selected_node_exact": bool(selected_row["demo_exact_program_ids"]),
        "any_node_exact": bool(exact_nodes),
        "exact_node_ids": list(exact_nodes),
        "best_any_node_mismatch_count": min(available_best) if available_best else None,
        "nodes": node_rows,
    }


def audit_stateful_rewrite_freeze(
    *,
    challenges: Mapping[str, object],
    freeze: Mapping[str, object],
    source_files: Mapping[str, str],
    max_trials_per_node: int = 512,
) -> dict[str, object]:
    """Reconstruct frozen parents and audit every one-node demo frontier."""

    freeze_content = {key: value for key, value in freeze.items() if key != "freeze_id"}
    if freeze["freeze_id"] != canonical_sha256(freeze_content):
        raise ValueError("stateful rewrite freeze ID differs")
    if freeze["query_gold_read"] is not False:
        raise ValueError("failure audit requires a query-blind freeze")
    raw_tasks = {
        task["task_id"]: task
        for task in (
            _object(item, field="freeze task")
            for item in _sequence(freeze["tasks"], field="freeze tasks")
        )
    }
    if set(raw_tasks) != set(challenges):
        raise ValueError("failure audit task sets differ")
    first_stage_trials = freeze["search_contract"]["max_first_stage_trials"]
    if type(first_stage_trials) is not int or first_stage_trials < 1:
        raise ValueError("freeze first-stage bound differs")

    task_rows = []
    aggregate = Counter()
    for task_id in sorted(challenges):
        task = blind_task_from_challenge(
            _object(challenges[task_id], field=f"challenge {task_id}")
        )
        frozen_task = raw_tasks[task_id]
        if frozen_task["blind_content_sha256"] != task.blind_content_sha256:
            raise ValueError("failure audit blind task identity differs")
        prefix = enumerate_object_code_programs(task)[:first_stage_trials]
        parent_by_id = {
            object_code_program_id(program): program
            for program in prefix
            if isinstance(program, ScenePipelineProgram)
        }
        grammar = enumerate_scene_pipeline_programs(task)
        parent_rows = []
        for raw_certificate in _sequence(
            frozen_task["certificates"], field="failure certificates"
        ):
            value = _object(raw_certificate, field="failure certificate")
            parent_id = value["parent_program_id"]
            if parent_id not in parent_by_id:
                raise ValueError("frozen parent is absent from its first-stage prefix")
            parent = parent_by_id[parent_id]
            certificate = StatefulNodeFailureCertificate.create(
                task=task,
                parent=parent,
                diagnosis=value["diagnosis"],
                node_id=value["node_id"],
                mismatch_counts=_sequence(
                    value["mismatch_counts"], field="mismatch counts"
                ),
                shape_match=_sequence(value["shape_match"], field="shape flags"),
                parent_execution_ids=_sequence(
                    value["parent_execution_ids"], field="parent execution IDs"
                ),
                evidence=_object(value["evidence"], field="certificate evidence"),
            )
            if certificate.certificate_id != value["certificate_id"]:
                raise ValueError("reconstructed failure certificate ID differs")
            row = audit_parent_node_frontiers(
                task=task,
                parent=parent,
                certificate=certificate,
                candidate_programs=grammar,
                max_trials_per_node=max_trials_per_node,
            )
            parent_rows.append(row)
        selected_exact = any(row["selected_node_exact"] for row in parent_rows)
        any_exact = any(row["any_node_exact"] for row in parent_rows)
        aggregate["task_count"] += 1
        aggregate["parent_count"] += len(parent_rows)
        aggregate["selected_node_exact_task_count"] += int(selected_exact)
        aggregate["alternative_node_exact_task_count"] += int(
            any_exact and not selected_exact
        )
        aggregate["any_node_exact_task_count"] += int(any_exact)
        aggregate["trial_count"] += sum(
            node["trial_count"] for row in parent_rows for node in row["nodes"]
        )
        task_rows.append(
            {
                "task_id": task_id,
                "selected_node_exact": selected_exact,
                "alternative_node_exact": any_exact and not selected_exact,
                "any_node_exact": any_exact,
                "parents": parent_rows,
            }
        )
    content: dict[str, object] = {
        "schema": STATEFUL_REWRITE_FAILURE_AUDIT_SCHEMA,
        "freeze_id": freeze["freeze_id"],
        "query_gold_read": False,
        "query_candidates_emitted": False,
        "controller_training_started": False,
        "max_trials_per_node": max_trials_per_node,
        "source_files": dict(sorted(source_files.items())),
        "metrics": dict(sorted(aggregate.items())),
        "tasks": task_rows,
    }
    return {"audit_id": canonical_sha256(content), **content}
