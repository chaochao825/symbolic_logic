from __future__ import annotations

import pytest

from afts_arc.blind import BlindTask
from afts_arc.experiment_safety import canonical_sha256
from afts_arc.grid import as_grid
from afts_arc.hybrid.object_code import object_code_program_id
from afts_arc.hybrid.scene_graph import (
    CanvasNode,
    CorrespondObjectsNode,
    ObjectOperationNode,
    ParseObjectsNode,
    RenderObjectsNode,
    ScenePipelineProgram,
    SelectObjectsNode,
)
from afts_arc.recolor_topology_gate import (
    RECOLOR_TOPOLOGY_NATIVE_DOMAIN_SIZE,
    evaluate_recolor_parents,
    score_recolor_topology_freeze,
)
from afts_arc.task import ARCPair


def _crop_program() -> ScenePipelineProgram:
    return ScenePipelineProgram(
        ParseObjectsNode(0, 4, "monochrome_components"),
        CorrespondObjectsNode(),
        SelectObjectsNode("smallest_area"),
        ObjectOperationNode("crop"),
        CanvasNode("bbox", 0),
        RenderObjectsNode("source_crop"),
    )


def _task(target: list[list[int]]) -> BlindTask:
    source = as_grid(
        [
            [0, 0, 0, 0],
            [0, 4, 4, 0],
            [0, 4, 4, 0],
            [0, 0, 0, 0],
        ]
    )
    return BlindTask.from_observations(
        train=(ARCPair(source, as_grid(target)),),
        test_inputs=(source,),
    )


def _certificate(program: ScenePipelineProgram) -> dict[str, object]:
    return {
        "certificate_id": "certificate-0",
        "parent_program_id": object_code_program_id(program),
    }


def test_natural_parent_recolor_creates_typed_demo_exact_novel_frontier() -> None:
    program = _crop_program()
    result = evaluate_recolor_parents(
        task=_task([[5, 5], [5, 5]]),
        parents=(program,),
        parent_certificates=(_certificate(program),),
        baseline_output_bundle_ids=frozenset(),
        reserved_native_trials=2_048,
    )

    assert result["parent_count"] == 1
    assert result["reachable_parent_count"] == 1
    assert len(result["demo_exact_candidates"]) == 1
    assert len(result["novel_candidates"]) == 1
    candidate = result["novel_candidates"][0]
    assert candidate["recolor_node"] == {
        "op": "recolor_grid",
        "source_color": 4,
        "target_color": 5,
    }
    assert candidate["query_outputs"] == [[[5, 5], [5, 5]]]
    assert result["cost"] == {
        "domain_programs_per_parent": RECOLOR_TOPOLOGY_NATIVE_DOMAIN_SIZE,
        "realized_program_trials": 90,
        "padding_program_trials": 1_958,
        "reserved_program_trials": 2_048,
        "reserved_demo_executions": 2_048,
        "reserved_query_executions": 2_048,
    }


def test_global_duplicate_is_not_reported_as_frontier_changing() -> None:
    program = _crop_program()
    first = evaluate_recolor_parents(
        task=_task([[5, 5], [5, 5]]),
        parents=(program,),
        parent_certificates=(_certificate(program),),
        baseline_output_bundle_ids=frozenset(),
        reserved_native_trials=2_048,
    )
    output_bundle_id = first["demo_exact_candidates"][0]["output_bundle_id"]
    duplicate = evaluate_recolor_parents(
        task=_task([[5, 5], [5, 5]]),
        parents=(program,),
        parent_certificates=(_certificate(program),),
        baseline_output_bundle_ids=frozenset((output_bundle_id,)),
        reserved_native_trials=2_048,
    )

    assert len(duplicate["demo_exact_candidates"]) == 1
    assert duplicate["novel_candidates"] == []
    assert duplicate["demo_exact_candidates"][0]["globally_novel"] is False


def test_unreachable_parent_emits_no_candidate() -> None:
    program = _crop_program()
    result = evaluate_recolor_parents(
        task=_task([[5, 6], [5, 6]]),
        parents=(program,),
        parent_certificates=(_certificate(program),),
        baseline_output_bundle_ids=frozenset(),
        reserved_native_trials=2_048,
    )

    assert result["reachable_parent_count"] == 0
    assert result["demo_exact_candidates"] == []
    assert result["novel_candidates"] == []
    assert result["parent_evaluations"][0]["reachability"]["status"] == "unreachable"


def test_parent_certificate_and_budget_are_hard_guards() -> None:
    program = _crop_program()
    certificate = _certificate(program)
    certificate["parent_program_id"] = "another-parent"
    with pytest.raises(ValueError, match="differs from its certificate"):
        evaluate_recolor_parents(
            task=_task([[5, 5], [5, 5]]),
            parents=(program,),
            parent_certificates=(certificate,),
            baseline_output_bundle_ids=frozenset(),
            reserved_native_trials=2_048,
        )
    with pytest.raises(ValueError, match="exceeds"):
        evaluate_recolor_parents(
            task=_task([[5, 5], [5, 5]]),
            parents=(program,),
            parent_certificates=(_certificate(program),),
            baseline_output_bundle_ids=frozenset(),
            reserved_native_trials=89,
        )


def test_postfreeze_score_classifies_one_unique_recovery_as_boundary() -> None:
    program = _crop_program()
    evaluation = evaluate_recolor_parents(
        task=_task([[5, 5], [5, 5]]),
        parents=(program,),
        parent_certificates=(_certificate(program),),
        baseline_output_bundle_ids=frozenset(),
        reserved_native_trials=2_048,
    )
    task_row = {
        "task_id": "task-0",
        "incumbent_candidates": [],
        "frozen_cold_candidates": [],
        **evaluation,
    }
    content = {
        "schema": "afts.recolor-topology-freeze/v1",
        "query_gold_read": False,
        "scientific_lane": "outcome_exposed_development",
        "search_contract": {"reserved_native_trials": 2_048},
        "task_count": 1,
        "tasks": [task_row],
    }
    freeze = {"freeze_id": canonical_sha256(content), **content}

    result = score_recolor_topology_freeze(
        freeze=freeze,
        solutions={"task-0": [[[5, 5], [5, 5]]]},
        solution_source_sha256="solution-sha",
    )

    assert result["baseline_union_exact"] == 0
    assert result["recolor_exact"] == 1
    assert result["novel_recolor_exact"] == 1
    assert result["unique_recovery_over_incumbent_and_cold"] == 1
    assert result["outcome"] == "boundary_natural_utility"


def test_postfreeze_score_rejects_content_tampering() -> None:
    freeze = {
        "freeze_id": "wrong",
        "query_gold_read": False,
        "scientific_lane": "outcome_exposed_development",
        "search_contract": {"reserved_native_trials": 2_048},
        "task_count": 0,
        "tasks": [],
    }
    with pytest.raises(ValueError, match="freeze ID differs"):
        score_recolor_topology_freeze(
            freeze=freeze,
            solutions={},
            solution_source_sha256="solution-sha",
        )
