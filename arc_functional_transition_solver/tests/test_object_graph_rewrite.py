from __future__ import annotations

from afts_arc.blind import BlindTask
from afts_arc.grid import as_grid
from afts_arc.hybrid.object_code import (
    D4LabelCompletionProgram,
    execute_object_code_program,
)
from afts_arc.hybrid.scene_graph import (
    CanvasNode,
    CorrespondObjectsNode,
    ObjectOperationNode,
    ParseObjectsNode,
    RenderObjectsNode,
    ScenePipelineProgram,
    SelectObjectsNode,
)
from afts_arc.object_graph_rewrite import (
    ObjectGraphRewriteProgram,
    execute_object_graph_rewrite,
    make_object_graph_rewrite_hypothesis,
    synthesize_object_graph_rewrites,
)
from afts_arc.task import ARCPair


def _controlled_composition() -> tuple[
    BlindTask,
    D4LabelCompletionProgram,
    ScenePipelineProgram,
]:
    source = [[0 for _ in range(12)] for _ in range(7)]
    for row, column in (
        (1, 1),
        (2, 1),
        (3, 1),
        (3, 2),
        (1, 7),
        (2, 7),
        (3, 7),
        (3, 8),
    ):
        source[row][column] = 4
    source[1][2] = 1
    source[4][1] = 3
    source[1][8] = 1
    first = D4LabelCompletionProgram(0, 4, 4, 1)
    second = ScenePipelineProgram(
        ParseObjectsNode(0, 4, "foreground_components"),
        CorrespondObjectsNode(),
        SelectObjectsNode("rightmost"),
        ObjectOperationNode("crop"),
        CanvasNode("bbox", 0),
        RenderObjectsNode("source_crop"),
    )
    intermediate = execute_object_code_program(first, as_grid(source))
    assert intermediate.output is not None
    target = execute_object_code_program(second, intermediate.output)
    assert target.output is not None
    task = BlindTask.from_observations(
        train=(ARCPair(as_grid(source), target.output),),
        test_inputs=(as_grid(source),),
    )
    return task, first, second


def test_trace_conditioned_hole_fill_creates_replayable_novel_output() -> None:
    task, first, second = _controlled_composition()
    result = synthesize_object_graph_rewrites(
        task,
        max_first_stage_trials=1,
        max_parents=1,
        max_second_stage_trials=1,
        max_candidates=4,
        first_stage_programs=(first,),
        second_stage_programs=(second,),
    )

    assert result.first_stage_program_trials == 1
    assert result.second_stage_program_trials == 1
    assert result.second_stage_padding_trials == 0
    assert len(result.parent_certificates) == 1
    assert len(result.candidates) == 1
    assert len(result.novel_candidates) == 1
    assert result.parent_certificates[0].affected_slots == ("stages[1]",)
    assert (
        result.parent_certificates[0].to_failure_signal().certificate_type
        == "canvas_then_ast_hole"
    )

    candidate = result.novel_candidates[0]
    serialized = candidate.program.to_json_dict()
    reconstructed = ObjectGraphRewriteProgram.from_json_dict(serialized)
    assert reconstructed == candidate.program
    assert execute_object_graph_rewrite(
        reconstructed, task.test_inputs[0]
    ).output == candidate.query_outputs[0]

    hypothesis = make_object_graph_rewrite_hypothesis(candidate)
    assert hypothesis.hard_verifier is not None
    assert hypothesis.hard_verifier(task)
    assert hypothesis.spec["filled_holes"] == ["stages[1]"]


def test_program_content_id_changes_when_stage_order_changes() -> None:
    _, first, second = _controlled_composition()
    program = ObjectGraphRewriteProgram(first, second)
    reconstructed = ObjectGraphRewriteProgram.from_json_dict(program.to_json_dict())

    assert program.program_id == reconstructed.program_id
    assert program.functional_trace[0] == "workspace:load_executable_parent"
