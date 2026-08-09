from __future__ import annotations

import pytest

from afts_arc.blind import BlindTask
from afts_arc.grid import as_grid
from afts_arc.relational_delta import (
    DeltaParseNode,
    DeltaRenderNode,
    DeltaTargetNode,
    NaryRelationNode,
    RelationalDeltaProgram,
    RoleAssignmentNode,
    TypedDeltaMaskNode,
    compile_relational_delta_failure_certificate,
    enumerate_relational_delta_programs,
    execute_relational_delta,
    relational_delta_near_miss_quality,
    relational_delta_program_id,
    score_relational_delta_program,
    single_slot_relational_delta_variants,
)
from afts_arc.task import ARCPair


def _blind(
    source: list[list[int]],
    target: list[list[int]],
    query: list[list[int]] | None = None,
) -> BlindTask:
    return BlindTask.from_observations(
        train=(ARCPair(as_grid(source), as_grid(target)),),
        test_inputs=(as_grid(source if query is None else query),),
    )


def _program(
    *,
    relation: str,
    target_kind: str,
    target_selector: str,
    operation: str,
    erase: str,
    connectivity: int = 4,
    actor_role: str = "minority_foreground",
    support_role: str = "majority_foreground",
    color_policy: str = "actor",
    palette: tuple[int, ...] = (),
) -> RelationalDeltaProgram:
    return RelationalDeltaProgram(
        DeltaParseNode(connectivity),
        RoleAssignmentNode(actor_role, support_role),
        NaryRelationNode(relation),
        DeltaTargetNode(target_kind, target_selector),
        TypedDeltaMaskNode(erase),
        DeltaRenderNode(operation, color_policy, palette),
    )


def _ring_source() -> list[list[int]]:
    return [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 8, 8, 8, 8, 8, 0],
        [0, 8, 0, 0, 0, 8, 0],
        [0, 8, 0, 0, 0, 8, 0],
        [0, 8, 0, 0, 0, 8, 0],
        [0, 8, 8, 8, 8, 8, 0],
        [0, 0, 0, 0, 0, 0, 5],
    ]


def _filled_ring_target() -> list[list[int]]:
    return [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 8, 8, 8, 8, 8, 0],
        [0, 8, 5, 5, 5, 8, 0],
        [0, 8, 5, 5, 5, 8, 0],
        [0, 8, 5, 5, 5, 8, 0],
        [0, 8, 8, 8, 8, 8, 0],
        [0, 0, 0, 0, 0, 0, 5],
    ]


def test_add_uses_a_coordinate_target_and_preserves_the_actor() -> None:
    source = _ring_source()
    expected = [row[:] for row in source]
    expected[3][3] = 5
    program = _program(
        relation="support_bbox_centers",
        target_kind="coordinate",
        target_selector="all",
        operation="add",
        erase="none",
    )
    execution = execute_relational_delta(program, as_grid(source))
    assert execution.ok
    assert execution.output == as_grid(expected)
    assert execution.erase_mask == ()
    assert execution.write_mask == ((3, 3),)
    assert execution.topology_valid


def test_fill_relation_region_writes_the_complete_enclosed_region() -> None:
    source = _ring_source()
    program = _program(
        relation="support_enclosed_regions",
        target_kind="region",
        target_selector="all",
        operation="fill_relation_region",
        erase="none",
    )
    execution = execute_relational_delta(program, as_grid(source))
    assert execution.ok
    assert execution.output == as_grid(_filled_ring_target())
    assert len(execution.write_mask) == 9
    assert execution.topology_valid


def test_erase_add_matches_actor_area_to_a_complete_row_interior() -> None:
    source = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 8, 0, 8, 0, 0, 0, 0],
        [0, 0, 8, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 5, 0],
    ]
    expected = [row[:] for row in source]
    expected[1][2] = 5
    expected[4][6] = 0
    program = _program(
        relation="actor_area_to_row_interiors",
        target_kind="region",
        target_selector="unique_actor_area_match",
        operation="erase_add",
        erase="actor_cells",
        connectivity=8,
    )
    execution = execute_relational_delta(program, as_grid(source))
    assert execution.ok
    assert execution.output == as_grid(expected)
    assert execution.erase_mask == ((4, 6),)
    assert execution.write_mask == ((1, 2),)
    assert execution.topology_valid


def _recolor_task() -> tuple[BlindTask, RelationalDeltaProgram, RelationalDeltaProgram]:
    source = [
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        [4, 4, 4, 1, 1, 4, 4, 4, 4, 4],
        [4, 4, 4, 1, 1, 4, 4, 4, 4, 4],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        [4, 4, 1, 1, 1, 4, 4, 4, 4, 4],
        [4, 4, 1, 4, 4, 4, 4, 4, 4, 4],
        [4, 4, 1, 1, 1, 4, 4, 4, 4, 4],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    ]
    target = [
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        [2, 2, 4, 4, 4, 4, 4, 4, 4, 4],
        [2, 2, 4, 4, 4, 4, 4, 4, 4, 4],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        [4, 4, 4, 4, 4, 4, 4, 3, 3, 3],
        [4, 4, 4, 4, 4, 4, 4, 3, 4, 4],
        [4, 4, 4, 4, 4, 4, 4, 3, 3, 3],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    ]
    correct = _program(
        relation="component_to_horizontal_border",
        target_kind="component",
        target_selector="all",
        operation="recolor_component",
        erase="actor_cells",
        actor_role="only_foreground",
        support_role="canvas",
        color_policy="target_side_palette",
        palette=(2, 3),
    )
    swapped = _program(
        relation="component_to_horizontal_border",
        target_kind="component",
        target_selector="all",
        operation="recolor_component",
        erase="actor_cells",
        actor_role="only_foreground",
        support_role="canvas",
        color_policy="target_side_palette",
        palette=(3, 2),
    )
    return _blind(source, target), correct, swapped


def test_recolor_component_uses_topology_role_and_preserves_shapes() -> None:
    task, correct, _ = _recolor_task()
    execution = execute_relational_delta(correct, task.train[0].input)
    assert execution.ok
    assert execution.output == task.train[0].output
    assert tuple(item.side for item in execution.correspondences) == ("left", "right")
    assert len(execution.erase_mask) == len(execution.write_mask) == 11
    assert execution.topology_valid


def test_overlapping_component_targets_are_rejected_as_topology_invalid() -> None:
    source = as_grid(
        [
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 1, 1, 4, 4, 1, 1, 4, 4],
            [4, 4, 1, 1, 4, 4, 1, 1, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        ]
    )
    _, program, _ = _recolor_task()
    execution = execute_relational_delta(program, source)
    assert not execution.ok
    assert execution.reason == "topology_violation"
    assert execution.output is None


def test_program_round_trip_is_strict_and_content_addressed() -> None:
    _, program, _ = _recolor_task()
    reconstructed = RelationalDeltaProgram.from_json_dict(program.to_json_dict())
    assert reconstructed == program
    assert relational_delta_program_id(reconstructed) == relational_delta_program_id(
        program
    )
    malformed = program.to_json_dict()
    malformed["unexpected"] = True
    with pytest.raises(ValueError, match="missing or unknown"):
        RelationalDeltaProgram.from_json_dict(malformed)


def test_near_miss_requires_identity_gain_precision_recall_and_topology() -> None:
    task, _, swapped = _recolor_task()
    quality = relational_delta_near_miss_quality(swapped, task)
    assert quality.eligible
    assert quality.parent_mismatch_count < quality.identity_mismatch_count
    assert quality.delta_precision == 1.0
    assert quality.delta_recall == 1.0
    assert quality.topology_valid

    source = _ring_source()
    fill_task = _blind(source, _filled_ring_target())
    center_only = _program(
        relation="support_bbox_centers",
        target_kind="coordinate",
        target_selector="all",
        operation="add",
        erase="none",
    )
    rejected = relational_delta_near_miss_quality(center_only, fill_task)
    assert not rejected.eligible
    assert rejected.parent_mismatch_count < rejected.identity_mismatch_count
    assert rejected.delta_precision == 1.0
    assert rejected.delta_recall == pytest.approx(1 / 9)
    assert rejected.reason == "delta_recall_below_threshold"

    canvas_task = _blind([[0, 5], [0, 0]], [[5]])
    canvas_rejected = relational_delta_near_miss_quality(center_only, canvas_task)
    assert not canvas_rejected.eligible
    assert not canvas_rejected.canvas_compatible
    assert canvas_rejected.reason == "canvas_incompatible"


def test_failure_certificate_reaches_a_novel_single_slot_palette_edit() -> None:
    task, correct, swapped = _recolor_task()
    certificate = compile_relational_delta_failure_certificate(
        task_id=task.task_id,
        task=task,
        parent=swapped,
    )
    assert certificate.diagnosis == "render"
    assert certificate.action == "edit_render"
    assert certificate.evidence["query_gold_read"] is False
    edits = single_slot_relational_delta_variants(
        swapped,
        allowed_slots=certificate.affected_slots,
        candidate_programs=(swapped, correct),
        existing_program_ids=(relational_delta_program_id(swapped),),
    )
    assert len(edits) == 1
    assert edits[0].slot == "render.palette"
    assert edits[0].program == correct
    assert relational_delta_program_id(edits[0].program) != relational_delta_program_id(
        swapped
    )
    assert score_relational_delta_program(edits[0].program, task).all_demo_exact


def test_enumeration_is_query_blind_bounded_and_reaches_recolor_fixture() -> None:
    task, correct, _ = _recolor_task()
    programs = enumerate_relational_delta_programs(task)
    assert len(programs) <= 256
    assert relational_delta_program_id(correct) in {
        relational_delta_program_id(program) for program in programs
    }
    assert any(
        score_relational_delta_program(program, task).all_demo_exact
        for program in programs
    )
