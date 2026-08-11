from __future__ import annotations

from afts_arc.blind import BlindTask
from afts_arc.grid import as_grid
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
    compile_stateful_node_failure,
)
from afts_arc.stateful_rewrite_failure_audit import audit_parent_node_frontiers
from afts_arc.stateful_scene import execute_stateful_scene_pipeline
from afts_arc.task import ARCPair


def _program(role: str) -> ScenePipelineProgram:
    return ScenePipelineProgram(
        ParseObjectsNode(0, 4, "monochrome_components"),
        CorrespondObjectsNode(),
        SelectObjectsNode(role),
        ObjectOperationNode("crop"),
        CanvasNode("bbox", 0),
        RenderObjectsNode("source_crop"),
    )


def _fixture() -> tuple[BlindTask, ScenePipelineProgram, ScenePipelineProgram]:
    grid = as_grid(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 0, 3, 3, 0],
            [0, 2, 2, 0, 3, 3, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    parent = _program("leftmost")
    child = _program("rightmost")
    gold = execute_scene_pipeline(child, grid).output
    assert gold is not None
    task = BlindTask.from_observations(
        train=(ARCPair(grid, gold),),
        test_inputs=(grid,),
    )
    return task, parent, child


def test_audit_distinguishes_selected_node_from_action_language_failure() -> None:
    task, parent, child = _fixture()
    parent_execution = execute_stateful_scene_pipeline(parent, task.train[0].input)
    selected = compile_stateful_node_failure(task, parent, (parent_execution,))
    selected_audit = audit_parent_node_frontiers(
        task=task,
        parent=parent,
        certificate=selected,
        candidate_programs=(parent, child),
        max_trials_per_node=4,
    )

    assert selected_audit["selected_node_id"] == "assignment"
    assert selected_audit["selected_node_exact"] is True
    assert selected_audit["any_node_exact"] is True
    assert selected_audit["exact_node_ids"] == ["assignment"]

    wrong = StatefulNodeFailureCertificate.create(
        task=task,
        parent=parent,
        diagnosis="injected_wrong_node",
        node_id="operate",
        mismatch_counts=selected.mismatch_counts,
        shape_match=selected.shape_match,
        parent_execution_ids=selected.parent_execution_ids,
        evidence={"injected": True},
    )
    wrong_audit = audit_parent_node_frontiers(
        task=task,
        parent=parent,
        certificate=wrong,
        candidate_programs=(parent, child),
        max_trials_per_node=4,
    )

    assert wrong_audit["selected_node_exact"] is False
    assert wrong_audit["any_node_exact"] is True
    assert wrong_audit["exact_node_ids"] == ["assignment"]
