from __future__ import annotations

import pytest

from afts_arc.anchor_mask import ANCHOR_MASK_PROGRAMS_PER_DELTA
from afts_arc.anchor_mask_topology_gate import (
    evaluate_anchor_mask_parents,
    score_anchor_mask_freeze,
)
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
from afts_arc.task import ARCPair


def _program() -> ScenePipelineProgram:
    return ScenePipelineProgram(
        ParseObjectsNode(0, 4, "monochrome_components"),
        CorrespondObjectsNode(),
        SelectObjectsNode("all"),
        ObjectOperationNode("crop"),
        CanvasNode("bbox", 0),
        RenderObjectsNode("source_crop"),
    )


def _pair(anchor_column: int) -> ARCPair:
    source = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    source[0][anchor_column] = 2
    source[2][anchor_column] = 2
    output = [row[:] for row in source]
    output[1][anchor_column] = 5
    return ARCPair(as_grid(source), as_grid(output))


def _task() -> BlindTask:
    return BlindTask.from_observations(
        train=(_pair(0), _pair(1)),
        test_inputs=(_pair(2).input,),
    )


def _certificate(program: ScenePipelineProgram) -> dict[str, object]:
    return {
        "certificate_id": "certificate-0",
        "parent_program_id": object_code_program_id(program),
    }


def test_natural_parent_mask_creates_lodo_exact_novel_frontier() -> None:
    program = _program()
    result = evaluate_anchor_mask_parents(
        task=_task(),
        parents=(program,),
        parent_certificates=(_certificate(program),),
        baseline_output_bundle_ids=frozenset(),
        reserved_native_trials=2_048,
    )

    assert result["parent_count"] == 1
    assert result["reachable_parent_count"] == 1
    assert result["strict_lodo_parent_count"] == 1
    assert result["demo_exact_candidates"]
    assert result["novel_candidates"]
    candidate = next(
        item
        for item in result["novel_candidates"]
        if item["query_outputs"]
        == [[[1, 1, 2], [1, 1, 5], [1, 1, 2]]]
    )
    assert candidate["query_outputs"] == [
        [[1, 1, 2], [1, 1, 5], [1, 1, 2]]
    ]
    assert result["cost"]["domain_programs_per_parent"] == (
        ANCHOR_MASK_PROGRAMS_PER_DELTA
    )
    assert result["cost"]["realized_program_trials"] == 120
    assert result["cost"]["padding_program_trials"] == 1_928


def test_budget_and_parent_certificate_are_hard_guards() -> None:
    program = _program()
    certificate = _certificate(program)
    certificate["parent_program_id"] = "another-parent"
    with pytest.raises(ValueError, match="differs from its certificate"):
        evaluate_anchor_mask_parents(
            task=_task(),
            parents=(program,),
            parent_certificates=(certificate,),
            baseline_output_bundle_ids=frozenset(),
            reserved_native_trials=2_048,
        )
    with pytest.raises(ValueError, match="exceeds"):
        evaluate_anchor_mask_parents(
            task=_task(),
            parents=(program,) * 18,
            parent_certificates=(_certificate(program),) * 18,
            baseline_output_bundle_ids=frozenset(),
            reserved_native_trials=2_048,
        )


def test_postfreeze_score_classifies_unique_recovery() -> None:
    program = _program()
    evaluation = evaluate_anchor_mask_parents(
        task=_task(),
        parents=(program,),
        parent_certificates=(_certificate(program),),
        baseline_output_bundle_ids=frozenset(),
        reserved_native_trials=2_048,
    )
    content = {
        "schema": "afts.anchor-mask-topology-freeze/v1",
        "query_gold_read": False,
        "scientific_lane": "outcome_exposed_development",
        "task_count": 1,
        "reachable_task_count": 1,
        "strict_lodo_task_count": 3,
        "novel_frontier_task_count": 1,
        "tasks": [
            {
                "task_id": "task-0",
                "incumbent_candidates": [],
                "frozen_cold_candidates": [],
                **evaluation,
            }
        ],
    }
    freeze = {"freeze_id": canonical_sha256(content), **content}
    result = score_anchor_mask_freeze(
        freeze=freeze,
        solutions={"task-0": [[[1, 1, 2], [1, 1, 5], [1, 1, 2]]]},
        solution_source_sha256="solution-sha",
    )

    assert result["mask_exact"] == 1
    assert result["novel_mask_exact"] == 1
    assert result["unique_recovery_over_incumbent_and_cold"] == 1
    assert result["outcome"] == "boundary_natural_utility"


def test_postfreeze_score_rejects_tampering() -> None:
    with pytest.raises(ValueError, match="freeze ID differs"):
        score_anchor_mask_freeze(
            freeze={
                "freeze_id": "wrong",
                "query_gold_read": False,
                "tasks": [],
            },
            solutions={},
            solution_source_sha256="solution-sha",
        )
