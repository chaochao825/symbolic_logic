from __future__ import annotations

import pytest

from afts_arc.cognitive_workspace import (
    CapabilityOption,
    CapabilityPlan,
    CognitiveWorkspaceState,
    FailureSignal,
    GoalFrame,
    MemoryRecord,
    plan_capability_path,
    retrievable_memories,
)
from afts_arc.hybrid.metareasoning import NativeCostVector


def _cost(trials: int) -> NativeCostVector:
    return NativeCostVector.from_mapping({"program_trials": trials})


def _failure(kind: str, target: str) -> FailureSignal:
    return FailureSignal.create(
        certificate_type=kind,
        current_representation="object_code_program",
        target_representation=target,
        affected_slots=("stages[1]",),
        evidence_ids=(f"evidence-{kind}",),
    )


def _option(kind: str, target: str, operator: str) -> CapabilityOption:
    return CapabilityOption.create(
        actor="object_graph_rewrite",
        operator=operator,
        input_representation="object_code_program",
        output_representation=target,
        accepted_certificate_types=(kind,),
        reservation=_cost(2),
        expected_frontier_gain=1,
        expected_hypothesis_reduction=2,
    )


def test_workspace_round_trip_preserves_content_address_and_budget() -> None:
    state = CognitiveWorkspaceState.empty(
        task_id="task-a",
        family_id="family-a",
        budget_limit=_cost(4),
    )
    root = GoalFrame.create(
        objective="solve task",
        success_condition="demo exact and novel query output",
    )
    state = state.push_goal(root)
    failure = _failure("residual_ast_hole", "object_program_composition")
    option = _option(
        "residual_ast_hole", "object_program_composition", "fill_ast_hole"
    )
    state = state.broadcast_failure(failure).register_option(option)
    state = state.apply_option_result(
        certificate_id=failure.certificate_id,
        option_id=option.option_id,
        produced_candidate_ids=("candidate-a", "candidate-b"),
        novel_candidate_ids=("candidate-b",),
        success=True,
    )

    rebuilt = CognitiveWorkspaceState.from_json_dict(state.to_json_dict())

    assert rebuilt == state
    assert rebuilt.state_id == state.state_id
    assert rebuilt.episodes[0].frontier_changed
    assert rebuilt.budget.used == _cost(2)


def test_memory_retrieval_enforces_task_family_and_outcome_boundaries() -> None:
    local = MemoryRecord.create(
        record_type="execution_trace",
        scope="task_local_demo",
        payload={"trace": "demo-only"},
        source_task_id="task-a",
        source_family_id="family-a",
    )
    reusable = MemoryRecord.create(
        record_type="validated_schema",
        scope="family_disjoint_validated",
        payload={"program": "schema-a"},
        source_task_id="task-b",
        source_family_id="family-b",
        validation_evidence_ids=("demo-replay-b",),
    )
    exposed = MemoryRecord.create(
        record_type="posthoc_diagnosis",
        scope="outcome_exposed_diagnostic",
        payload={"failure": "known-after-score"},
        source_task_id="task-a",
        source_family_id="family-a",
        outcome_exposed=True,
    )
    sealed = MemoryRecord.create(
        record_type="query_solution",
        scope="sealed_query_oracle",
        payload={"answer": "sealed"},
        source_task_id="task-a",
        source_family_id="family-a",
        outcome_exposed=True,
        contains_query_oracle=True,
    )
    records = (local, reusable, exposed, sealed)

    prospective = retrievable_memories(
        records,
        target_task_id="task-a",
        target_family_id="family-a",
        lane="prospective",
    )
    diagnostic = retrievable_memories(
        records,
        target_task_id="task-a",
        target_family_id="family-a",
        lane="diagnostic",
    )

    assert {item.memory_id for item in prospective} == {
        local.memory_id,
        reusable.memory_id,
    }
    assert {item.memory_id for item in diagnostic} == {
        local.memory_id,
        reusable.memory_id,
        exposed.memory_id,
    }
    assert sealed.memory_id not in {item.memory_id for item in diagnostic}


def test_certificate_intervention_changes_legal_option_selectively() -> None:
    hole = _failure("residual_ast_hole", "object_program_composition")
    canvas = _failure("canvas_error", "canvas_hypothesis")
    fill = _option(
        "residual_ast_hole", "object_program_composition", "fill_ast_hole"
    )
    reinfer = _option("canvas_error", "canvas_hypothesis", "canvas_reinfer")
    state = CognitiveWorkspaceState.empty(
        task_id="task-a",
        family_id="family-a",
        budget_limit=_cost(4),
    )
    state = state.register_option(fill).register_option(reinfer)
    state = state.broadcast_failure(hole).broadcast_failure(canvas)

    assert state.select_option(hole.certificate_id).operator == "fill_ast_hole"
    assert state.select_option(canvas.certificate_id).operator == "canvas_reinfer"


def test_budget_and_novelty_guards_fail_closed() -> None:
    failure = _failure("residual_ast_hole", "object_program_composition")
    option = _option(
        "residual_ast_hole", "object_program_composition", "fill_ast_hole"
    )
    state = CognitiveWorkspaceState.empty(
        task_id="task-a",
        family_id="family-a",
        budget_limit=_cost(2),
    )
    state = state.register_option(option).broadcast_failure(failure)
    state = state.apply_option_result(
        certificate_id=failure.certificate_id,
        option_id=option.option_id,
        produced_candidate_ids=("candidate-a",),
        novel_candidate_ids=(),
        success=False,
    )
    assert not state.episodes[0].frontier_changed
    with pytest.raises(ValueError, match="budget-feasible"):
        state.select_option(failure.certificate_id)

    with pytest.raises(ValueError, match="subset"):
        CognitiveWorkspaceState.empty(
            task_id="task-a",
            family_id="family-a",
            budget_limit=_cost(2),
        ).register_option(option).broadcast_failure(failure).apply_option_result(
            certificate_id=failure.certificate_id,
            option_id=option.option_id,
            produced_candidate_ids=("candidate-a",),
            novel_candidate_ids=("candidate-b",),
            success=True,
        )


def test_bounded_plan_has_selective_bridge_lesion_and_cost_effects() -> None:
    failure = FailureSignal.create(
        certificate_type="perceptual_object_mismatch",
        current_representation="visual_posterior",
        target_representation="object_code_program",
        affected_slots=("parse.objects",),
        evidence_ids=("posterior-disagreement",),
    )
    parse = CapabilityOption.create(
        actor="object_parser",
        operator="parse_objects",
        input_representation="visual_posterior",
        output_representation="object_code_program",
        accepted_certificate_types=("perceptual_object_mismatch",),
        reservation=_cost(1),
        expected_frontier_gain=0,
        expected_hypothesis_reduction=4,
    )
    bridge = CapabilityOption.create(
        actor="object_graph_rewrite",
        operator="fill_ast_hole",
        input_representation="object_code_program",
        output_representation="object_program_composition",
        accepted_certificate_types=("residual_ast_hole",),
        reservation=_cost(1),
        expected_frontier_gain=1,
        expected_hypothesis_reduction=2,
    )
    state = CognitiveWorkspaceState.empty(
        task_id="task-a",
        family_id="family-a",
        budget_limit=_cost(2),
    )
    state = state.broadcast_failure(failure).register_option(parse).register_option(bridge)

    plan = plan_capability_path(
        state,
        certificate_id=failure.certificate_id,
        goal_representation="object_program_composition",
    )
    assert plan.option_ids == (parse.option_id, bridge.option_id)
    assert CapabilityPlan.from_json_dict(plan.to_json_dict()) == plan
    assert plan.reserved_cost == _cost(2)

    lesioned = CognitiveWorkspaceState.empty(
        task_id="task-a",
        family_id="family-a",
        budget_limit=_cost(2),
    ).broadcast_failure(failure).register_option(parse)
    with pytest.raises(ValueError, match="no legal"):
        plan_capability_path(
            lesioned,
            certificate_id=failure.certificate_id,
            goal_representation="object_program_composition",
        )

    underfunded = CognitiveWorkspaceState.empty(
        task_id="task-a",
        family_id="family-a",
        budget_limit=_cost(1),
    ).broadcast_failure(failure).register_option(parse).register_option(bridge)
    with pytest.raises(ValueError, match="budget-feasible"):
        plan_capability_path(
            underfunded,
            certificate_id=failure.certificate_id,
            goal_representation="object_program_composition",
        )
