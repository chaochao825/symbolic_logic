from __future__ import annotations

from afts_arc.blind import BlindTask
from afts_arc.grid import as_grid
from afts_arc.hybrid.object_code import object_code_program_id
from afts_arc.hybrid.scene_graph import (
    CanvasNode,
    CorrespondObjectsNode,
    ExecutionTraceNode,
    ObjectOperationNode,
    ParseObjectsNode,
    RenderObjectsNode,
    ScenePipelineExecution,
    ScenePipelineProgram,
    SelectObjectsNode,
)
from afts_arc.task import ARCPair
from afts_arc.visual_trace_repair import (
    compile_visual_trace_certificate,
    query_posterior_rank_key,
    relational_difference_groups,
    single_slot_scene_variants,
)


def _program(*, role: str = "all", transform: str = "identity") -> ScenePipelineProgram:
    return ScenePipelineProgram(
        ParseObjectsNode(0, 4, "monochrome_components"),
        CorrespondObjectsNode(),
        SelectObjectsNode(role),
        ObjectOperationNode("arrange", transform, "row", 0),
        CanvasNode("tight", 0),
        RenderObjectsNode("objects"),
    )


def _execution(output: list[list[int]]) -> ScenePipelineExecution:
    trace = tuple(
        ExecutionTraceNode.create(node, node, "ok")
        for node in ("parse", "correspond", "select", "operate", "canvas", "render")
    )
    return ScenePipelineExecution("ok", as_grid(output), None, 1, 1, trace)


def _blind_task() -> BlindTask:
    pair = ARCPair(as_grid([[0, 1]]), as_grid([[0, 2]]))
    return BlindTask.from_observations(
        train=(pair,),
        test_inputs=(as_grid([[0, 1]]),),
    )


def test_relational_difference_preserves_complete_grid_groups() -> None:
    program = _program()
    parent = as_grid([[0, 1]])
    recolored = as_grid([[0, 2]])
    resized = as_grid([[0, 1], [0, 0]])

    assert relational_difference_groups(program, parent, parent) == ()
    assert "palette" in relational_difference_groups(program, parent, recolored)
    assert "canvas" in relational_difference_groups(program, parent, resized)


def test_lodo_support_changes_certificate_from_bridge_lesion() -> None:
    program = _program()
    parent = _execution([[0, 1]])
    gold = as_grid([[0, 2], [0, 0]])
    lodo = tuple(as_grid([[0, 2]]) for _ in range(5))
    query = tuple(as_grid([[0, 2]]) for _ in range(5))

    visual = compile_visual_trace_certificate(
        task_id="fixture",
        parent_program=program,
        demo_executions=(parent,),
        demo_gold_outputs=(gold,),
        lodo_samples=(lodo,),
        query_execution=parent,
        query_samples=query,
        mode="visual_lodo",
    )
    lesion = compile_visual_trace_certificate(
        task_id="fixture",
        parent_program=program,
        demo_executions=(parent,),
        demo_gold_outputs=(gold,),
        lodo_samples=(lodo,),
        query_execution=parent,
        query_samples=query,
        mode="bridge_lesion",
    )

    assert visual.diagnosis_group == "palette"
    assert lesion.diagnosis_group == "canvas"
    assert visual.certificate_id != lesion.certificate_id
    assert visual.evidence["complete_grid_posterior_only"] is True


def test_single_slot_frontier_rejects_multi_slot_and_existing_edits(
    monkeypatch,
) -> None:
    parent = _program()
    role_edit = _program(role="largest_area")
    transform_edit = _program(transform="rotate90")
    two_slot_edit = _program(role="largest_area", transform="rotate90")
    monkeypatch.setattr(
        "afts_arc.visual_trace_repair.enumerate_scene_pipeline_programs",
        lambda task: (parent, role_edit, transform_edit, two_slot_edit),
    )

    edits = single_slot_scene_variants(
        _blind_task(),
        parent,
        allowed_slots=("ast.select.role", "ast.operate.transform"),
    )
    assert tuple(edit.slot for edit in edits) == (
        "ast.operate.transform",
        "ast.select.role",
    )
    assert all(edit.program != two_slot_edit for edit in edits)

    retained = single_slot_scene_variants(
        _blind_task(),
        parent,
        allowed_slots=("ast.select.role", "ast.operate.transform"),
        existing_program_ids=(object_code_program_id(transform_edit),),
    )
    assert tuple(edit.slot for edit in retained) == ("ast.select.role",)


def test_query_ranking_uses_whole_grid_support_before_description_length() -> None:
    program = _program()
    supported = as_grid([[0, 2]])
    unsupported = as_grid([[0, 1]])
    samples = (supported, supported, unsupported)

    assert query_posterior_rank_key(program, supported, samples) < (
        query_posterior_rank_key(program, unsupported, samples)
    )
