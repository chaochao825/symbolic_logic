from __future__ import annotations

import pytest

from afts_arc.blind import BlindTask
from afts_arc.counterfactual_transition import (
    CounterfactualTransition,
    execute_counterfactual_transition,
    synthesize_counterfactual_transition_arms,
)
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
from afts_arc.stateful_object_graph_rewrite import compile_stateful_node_failure
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


def _program(role: str, canvas: CanvasNode) -> ScenePipelineProgram:
    return ScenePipelineProgram(
        ParseObjectsNode(0, 4, "monochrome_components"),
        CorrespondObjectsNode(),
        SelectObjectsNode(role),
        ObjectOperationNode("crop"),
        canvas,
        RenderObjectsNode("source_crop"),
    )


def _fixture() -> tuple[
    BlindTask,
    ScenePipelineProgram,
    ScenePipelineProgram,
    ScenePipelineProgram,
    ScenePipelineProgram,
]:
    fixed = CanvasNode("fixed", 0, height=3, width=3)
    bbox = CanvasNode("bbox", 0)
    parent = _program("leftmost", fixed)
    assignment_only = _program("rightmost", fixed)
    canvas_only = _program("leftmost", bbox)
    child = _program("rightmost", bbox)
    grid = _grid()
    gold = execute_scene_pipeline(child, grid).output
    assert gold is not None
    task = BlindTask.from_observations(
        train=(ARCPair(grid, gold),),
        test_inputs=(grid,),
    )
    return task, parent, assignment_only, canvas_only, child


def test_two_node_transition_round_trip_and_bridge_contract() -> None:
    task, parent, _, _, child = _fixture()
    parent_execution = execute_stateful_scene_pipeline(parent, task.train[0].input)
    certificate = compile_stateful_node_failure(task, parent, (parent_execution,))
    transition = CounterfactualTransition.create(
        strategy="typed_semantic_bundles",
        parent=parent,
        certificate=certificate,
        program=child,
    )

    assert transition.changed_nodes == ("assignment", "canvas")
    assert CounterfactualTransition.from_json_dict(
        transition.to_json_dict()
    ) == transition
    replay = execute_counterfactual_transition(
        transition,
        task.train[0].input,
        parent_execution=parent_execution,
    )
    assert replay.output == task.train[0].output
    assert replay.reused_node_ids == ("parse",)
    assert replay.executed_node_ids == (
        "assignment",
        "operate",
        "canvas",
        "render",
    )
    with pytest.raises(ValueError, match="requires its parent trace"):
        execute_counterfactual_transition(
            transition,
            task.train[0].input,
            parent_execution=None,
        )


def test_all_frozen_arms_recover_a_controlled_two_node_failure() -> None:
    task, parent, assignment_only, canvas_only, child = _fixture()
    result = synthesize_counterfactual_transition_arms(
        task,
        max_first_stage_trials=1,
        max_parents=1,
        max_transition_trials=12,
        max_candidates=4,
        first_stage_programs=(parent,),
        transition_programs=(assignment_only, canvas_only, child),
    )

    assert result.parent_count == 1
    assert result.first_stage_program_trials == 1
    assert tuple(arm.strategy for arm in result.arms) == (
        "distance_tiered",
        "typed_semantic_bundles",
        "residual_beam",
    )
    for arm in result.arms:
        assert arm.program_trials <= 12
        assert arm.program_trials + arm.padding_trials == 12
        assert len(arm.candidates) == 1
        assert arm.candidates[0].query_outputs == (task.train[0].output,)
        assert arm.candidates[0].transition.changed_nodes == (
            "assignment",
            "canvas",
        )
    assert result.arms[2].improving_trial_count >= 2
