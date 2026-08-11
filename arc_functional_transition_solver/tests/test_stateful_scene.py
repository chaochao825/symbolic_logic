from __future__ import annotations

import pytest

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
from afts_arc.stateful_scene import execute_stateful_scene_pipeline


def _two_object_grid() -> Grid:
    return as_grid(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 0, 3, 3, 0],
            [0, 2, 2, 0, 3, 3, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )


def _crop_program(
    role: str,
    *,
    canvas: CanvasNode | None = None,
) -> ScenePipelineProgram:
    return ScenePipelineProgram(
        ParseObjectsNode(0, 4, "monochrome_components"),
        CorrespondObjectsNode(),
        SelectObjectsNode(role),
        ObjectOperationNode("crop"),
        CanvasNode("bbox", 0) if canvas is None else canvas,
        RenderObjectsNode("source_crop"),
    )


def test_assignment_rewrite_reuses_persistent_parse_and_executes_suffix() -> None:
    grid = _two_object_grid()
    parent = _crop_program("leftmost")
    child = _crop_program("rightmost")
    parent_execution = execute_stateful_scene_pipeline(parent, grid)
    replay = execute_stateful_scene_pipeline(
        child,
        grid,
        prior=parent_execution,
        changed_node="assignment",
    )

    assert replay.ok
    assert replay.output == execute_scene_pipeline(child, grid).output
    assert replay.parse_state.state_id == parent_execution.parse_state.state_id
    assert replay.parse_state.objects == parent_execution.parse_state.objects
    assert replay.parse_state.relations == parent_execution.parse_state.relations
    assert replay.reused_node_ids == ("parse",)
    assert replay.executed_node_ids == (
        "assignment",
        "operate",
        "canvas",
        "render",
    )
    assert replay.assignment_state is not None
    assert parent_execution.assignment_state is not None
    assert (
        replay.assignment_state.selected_entity_ids
        != parent_execution.assignment_state.selected_entity_ids
    )


def test_canvas_rewrite_reuses_parse_assignment_and_operation() -> None:
    grid = _two_object_grid()
    parent = _crop_program("leftmost", canvas=CanvasNode("fixed", 0, height=3, width=3))
    child = _crop_program("leftmost")
    parent_execution = execute_stateful_scene_pipeline(parent, grid)
    replay = execute_stateful_scene_pipeline(
        child,
        grid,
        prior=parent_execution,
        changed_node="canvas",
    )

    assert replay.ok
    assert replay.output == execute_scene_pipeline(child, grid).output
    assert replay.reused_node_ids == ("parse", "assignment", "operate")
    assert replay.executed_node_ids == ("canvas", "render")
    assert replay.operation_state == parent_execution.operation_state


def test_reparse_records_split_merge_lineage_without_reusing_old_parse() -> None:
    grid = as_grid(
        [
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 0],
        ]
    )

    def program(connectivity: int) -> ScenePipelineProgram:
        return ScenePipelineProgram(
            ParseObjectsNode(0, connectivity, "monochrome_components"),
            CorrespondObjectsNode(),
            SelectObjectsNode("all"),
            ObjectOperationNode("arrange", axis="row"),
            CanvasNode("tight", 0),
            RenderObjectsNode("objects"),
        )

    parent = program(4)
    child = program(8)
    parent_execution = execute_stateful_scene_pipeline(parent, grid)
    replay = execute_stateful_scene_pipeline(
        child,
        grid,
        prior=parent_execution,
        changed_node="parse",
    )

    assert parent_execution.ok
    assert replay.ok
    assert len(parent_execution.parse_state.objects) == 2
    assert len(replay.parse_state.objects) == 1
    assert set(replay.parse_state.objects[0].ancestor_entity_ids) == {
        item.entity_id for item in parent_execution.parse_state.objects
    }
    assert replay.reused_node_ids == ()
    assert replay.executed_node_ids == (
        "parse",
        "assignment",
        "operate",
        "canvas",
        "render",
    )


def test_typed_replay_rejects_multi_node_or_wrong_input_changes() -> None:
    grid = _two_object_grid()
    parent = _crop_program("leftmost")
    parent_execution = execute_stateful_scene_pipeline(parent, grid)
    multi_node = ScenePipelineProgram(
        parent.parse,
        parent.correspond,
        SelectObjectsNode("rightmost"),
        parent.operate,
        CanvasNode("fixed", 0, height=2, width=2),
        parent.render,
    )

    with pytest.raises(ValueError, match="exactly its declared"):
        execute_stateful_scene_pipeline(
            multi_node,
            grid,
            prior=parent_execution,
            changed_node="assignment",
        )
    with pytest.raises(ValueError, match="another input grid"):
        execute_stateful_scene_pipeline(
            _crop_program("rightmost"),
            as_grid([[0, 2], [0, 0]]),
            prior=parent_execution,
            changed_node="assignment",
        )
