from __future__ import annotations

import pytest

from afts_arc.blind import BlindTask
from afts_arc.grid import Grid, as_grid
from afts_arc.hybrid.scene_graph import (
    CanvasNode,
    CorrespondObjectsNode,
    ObjectOperationNode,
    ParseObjectsNode,
    RenderObjectsNode,
    ScenePipelineProgram,
    SelectObjectsNode,
    execute_scene_pipeline,
)
from afts_arc.stateful_object_graph_rewrite import (
    StatefulNodeFailureCertificate,
    StatefulSceneRewrite,
    compile_stateful_node_failure,
    enumerate_stateful_scene_rewrites,
    execute_stateful_scene_rewrite,
    synthesize_stateful_object_graph_rewrites,
)
from afts_arc.stateful_scene import execute_stateful_scene_pipeline
from afts_arc.task import ARCPair


def _grid() -> Grid:
    return as_grid(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 0, 3, 3, 0],
            [0, 2, 2, 0, 3, 3, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )


def _program(role: str, canvas: CanvasNode | None = None) -> ScenePipelineProgram:
    return ScenePipelineProgram(
        ParseObjectsNode(0, 4, "monochrome_components"),
        CorrespondObjectsNode(),
        SelectObjectsNode(role),
        ObjectOperationNode("crop"),
        CanvasNode("bbox", 0) if canvas is None else canvas,
        RenderObjectsNode("source_crop"),
    )


def _task(parent: ScenePipelineProgram, child: ScenePipelineProgram) -> BlindTask:
    grid = _grid()
    gold = execute_scene_pipeline(child, grid).output
    assert gold is not None
    assert execute_scene_pipeline(parent, grid).output != gold
    return BlindTask.from_observations(
        train=(ARCPair(grid, gold),),
        test_inputs=(grid,),
    )


def test_demo_residual_compiles_to_assignment_and_one_legal_rewrite() -> None:
    parent = _program("leftmost")
    child = _program("rightmost")
    task = _task(parent, child)
    parent_execution = execute_stateful_scene_pipeline(parent, task.train[0].input)
    certificate = compile_stateful_node_failure(task, parent, (parent_execution,))
    rewrites = enumerate_stateful_scene_rewrites(
        parent,
        certificate,
        candidate_programs=(parent, child),
        max_trials=4,
    )

    assert certificate.node_id == "assignment"
    assert certificate.affected_subtree == (
        "assignment",
        "operate",
        "canvas",
        "render",
    )
    assert len(rewrites) == 1
    rewrite = rewrites[0]
    assert StatefulSceneRewrite.from_json_dict(rewrite.to_json_dict()) == rewrite
    replay = execute_stateful_scene_rewrite(
        rewrite,
        task.train[0].input,
        parent_execution=parent_execution,
    )
    assert replay.output == task.train[0].output
    assert replay.reused_node_ids == ("parse",)


def test_canvas_failure_replays_only_canvas_and_render() -> None:
    parent = _program("leftmost", CanvasNode("fixed", 0, height=3, width=3))
    child = _program("leftmost")
    task = _task(parent, child)
    parent_execution = execute_stateful_scene_pipeline(parent, task.train[0].input)
    certificate = compile_stateful_node_failure(task, parent, (parent_execution,))
    rewrite = enumerate_stateful_scene_rewrites(
        parent,
        certificate,
        candidate_programs=(child,),
        max_trials=1,
    )[0]
    replay = execute_stateful_scene_rewrite(
        rewrite,
        task.train[0].input,
        parent_execution=parent_execution,
    )

    assert certificate.node_id == "canvas"
    assert replay.reused_node_ids == ("parse", "assignment", "operate")
    assert replay.executed_node_ids == ("canvas", "render")
    assert replay.output == task.train[0].output


def test_synthesis_emits_demo_exact_novel_candidate_with_trace_contract() -> None:
    parent = _program("leftmost")
    child = _program("rightmost")
    task = _task(parent, child)
    result = synthesize_stateful_object_graph_rewrites(
        task,
        max_first_stage_trials=1,
        max_parents=1,
        max_rewrite_trials=1,
        max_candidates=2,
        first_stage_programs=(parent,),
        rewrite_programs=(child,),
    )

    assert result.parent_count == 1
    assert result.rewrite_program_trials == 1
    assert result.rewrite_padding_trials == 0
    assert len(result.candidates) == 1
    assert len(result.novel_candidates) == 1
    assert result.demo_node_executions == (
        ("assignment", 1),
        ("canvas", 1),
        ("operate", 1),
        ("render", 1),
    )
    assert result.reused_demo_nodes == (("parse", 1),)
    assert result.candidates[0].query_outputs == (task.train[0].output,)


def test_bridge_lesion_and_wrong_node_injection_are_rejected() -> None:
    parent = _program("leftmost")
    child = _program("rightmost")
    task = _task(parent, child)
    parent_execution = execute_stateful_scene_pipeline(parent, task.train[0].input)
    certificate = compile_stateful_node_failure(task, parent, (parent_execution,))
    rewrite = enumerate_stateful_scene_rewrites(
        parent,
        certificate,
        candidate_programs=(child,),
        max_trials=1,
    )[0]

    with pytest.raises(ValueError, match="executable parent trace"):
        execute_stateful_scene_rewrite(
            rewrite,
            task.train[0].input,
            parent_execution=None,
        )

    injected = StatefulNodeFailureCertificate.create(
        task=task,
        parent=parent,
        diagnosis="injected_operation_failure",
        node_id="operate",
        mismatch_counts=(4,),
        shape_match=(True,),
        parent_execution_ids=(parent_execution.execution_id,),
        evidence={"injected": True},
    )
    with pytest.raises(ValueError, match="diagnosed node"):
        StatefulSceneRewrite.create(
            parent=parent,
            certificate=injected,
            program=child,
        )
