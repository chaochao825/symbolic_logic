from __future__ import annotations

import pytest

from afts_arc.anchor_mask import AnchorRasterizedDeltaNode, execute_anchor_rasterized_delta
from afts_arc.grid import Grid, as_grid
from afts_arc.hybrid.scene_graph import (
    SCENE_D4_TRANSFORMS,
    CanvasNode,
    CorrespondObjectsNode,
    ObjectOperationNode,
    ParseObjectsNode,
    RenderObjectsNode,
    ScenePipelineProgram,
    SelectObjectsNode,
    extract_scene_graph,
)
from afts_arc.relational_effect import (
    RELATIONAL_EFFECT_PROGRAMS_PER_DELTA,
    EffectSummary,
    RelationalEffectNode,
    compile_relational_failure_core,
    execute_relational_effect,
    execute_scene_with_cell_provenance,
    leave_one_demo_out_relational_effects,
    lesion_entity_links,
    provenance_anchor_coordinates,
    relational_effect_domain,
)


def _crop_program(
    *,
    background: int = 0,
    transform: str = "identity",
    render_mode: str = "source_crop",
) -> ScenePipelineProgram:
    return ScenePipelineProgram(
        ParseObjectsNode(background, 4, "foreground_components"),
        CorrespondObjectsNode(),
        SelectObjectsNode("all"),
        ObjectOperationNode("crop", transform=transform),
        CanvasNode("bbox", background),
        RenderObjectsNode(render_mode),
    )


def _background_anchor_demo(height: int, width: int) -> tuple[Grid, Grid]:
    source = [[0 for _ in range(width)] for _ in range(height)]
    source[0][0] = 9
    source[0][width - 1] = 9
    source[0][width // 2] = 2
    target = [row[:] for row in source]
    for row in range(height):
        target[row][0] = 9
        target[row][width - 1] = 9
    return as_grid(source), as_grid(target)


def test_exact_crop_provenance_covers_every_d4_rendered_cell() -> None:
    grid = as_grid([[1, 2, 3], [4, 5, 6]])

    for transform in SCENE_D4_TRANSFORMS:
        execution = execute_scene_with_cell_provenance(
            _crop_program(transform=transform),
            grid,
        )

        assert execution.exact
        assert execution.output is not None
        assert len(execution.cells) == len(grid) * len(grid[0])
        assert {
            item.source_coordinate for item in execution.cells
        } == {
            (row, column)
            for row in range(len(grid))
            for column in range(len(grid[0]))
        }
        for item in execution.cells:
            assert item.source_coordinate is not None
            source_row, source_column = item.source_coordinate
            output_row, output_column = item.output_coordinate
            assert item.source_color == grid[source_row][source_column]
            assert (
                execution.output[output_row][output_column]
                == grid[source_row][source_column]
            )


def test_source_background_keeps_coordinate_without_claiming_an_entity() -> None:
    source, target = _background_anchor_demo(4, 5)
    program = _crop_program(background=9)
    execution = execute_scene_with_cell_provenance(program, source)
    node = RelationalEffectNode(9, "axis_project", "column", 0, 9)

    assert execution.exact
    background_anchors = tuple(
        item
        for item in execution.cells
        if item.source_coordinate in {(0, 0), (0, 4)}
    )
    assert all(item.origin_kind == "source_background" for item in background_anchors)
    assert all(item.entity_ids == () for item in background_anchors)
    assert provenance_anchor_coordinates(node, execution) == ((0, 0), (0, 4))
    assert execute_relational_effect(node, execution) == target

    old_scene = extract_scene_graph(
        source,
        background=9,
        connectivity=4,
        grouping="foreground_components",
    )
    old_node = AnchorRasterizedDeltaNode(9, "axis_project", "column", 0, 9)
    assert execute_anchor_rasterized_delta(
        old_node,
        source,
        execution.output,
        old_scene,
    ) == execution.output


def test_selected_only_background_is_explicitly_generated_not_false_source() -> None:
    ring = as_grid([[2, 2, 2], [2, 0, 2], [2, 2, 2]])
    execution = execute_scene_with_cell_provenance(
        _crop_program(render_mode="selected_only"),
        ring,
    )
    by_coordinate = {item.output_coordinate: item for item in execution.cells}

    assert execution.exact
    assert by_coordinate[(1, 1)].origin_kind == "render_background"
    assert by_coordinate[(1, 1)].source_coordinate is None
    assert by_coordinate[(1, 1)].source_color is None
    assert by_coordinate[(0, 0)].origin_kind == "source_entity"
    assert by_coordinate[(0, 0)].entity_ids


def test_failure_core_is_content_addressed_demo_exact_and_strict_lodo() -> None:
    program = _crop_program(background=9)
    demos = tuple(_background_anchor_demo(height, width) for height, width in ((3, 4), (4, 5), (5, 6)))
    inputs = tuple(item[0] for item in demos)
    targets = tuple(item[1] for item in demos)
    executions = tuple(
        execute_scene_with_cell_provenance(program, input_grid)
        for input_grid in inputs
    )

    core = compile_relational_failure_core(
        blind_task_id="blind_background_anchor",
        executions=executions,
        targets=targets,
    )
    replay = compile_relational_failure_core(
        blind_task_id="blind_background_anchor",
        executions=executions,
        targets=targets,
    )
    audit = leave_one_demo_out_relational_effects(
        blind_task_id="blind_background_anchor",
        program=program,
        inputs=inputs,
        targets=targets,
    )

    assert core.identified
    assert core.core_id == replay.core_id
    assert core.to_json_dict() == replay.to_json_dict()
    assert all(
        execute_relational_effect(candidate, execution) == target
        for candidate in core.candidates
        for execution, target in zip(executions, targets)
    )
    assert audit["strict_all_folds_exact"] is True
    assert audit["identity_strict_all_folds_exact"] is False
    assert audit["global_recolor_strict_all_folds_exact"] is False
    assert audit["component_recolor_strict_all_folds_exact"] is False
    assert audit["persistent_entity_strict_all_folds_exact"] is False


def test_base_exact_demo_constrains_instead_of_invalidating_one_effect() -> None:
    program = _crop_program(background=9)
    changed = tuple(
        _background_anchor_demo(height, width)
        for height, width in ((3, 4), (4, 5))
    )
    identity = as_grid([[0, 0, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    inputs = tuple(item[0] for item in changed) + (identity,)
    targets = tuple(item[1] for item in changed) + (identity,)
    executions = tuple(
        execute_scene_with_cell_provenance(program, input_grid)
        for input_grid in inputs
    )

    core = compile_relational_failure_core(
        blind_task_id="blind_conditional_noop",
        executions=executions,
        targets=targets,
    )
    audit = leave_one_demo_out_relational_effects(
        blind_task_id="blind_conditional_noop",
        program=program,
        inputs=inputs,
        targets=targets,
    )

    assert [summary.status for summary in core.effect_summaries] == [
        "single_delta",
        "single_delta",
        "base_exact",
    ]
    assert core.identified
    assert audit["strict_all_folds_exact"] is True


def test_effect_summary_separates_source_and_entity_coverage() -> None:
    source, target = _background_anchor_demo(4, 5)
    execution = execute_scene_with_cell_provenance(
        _crop_program(background=9),
        source,
    )
    summary = EffectSummary.create(execution, target)

    assert summary.status == "single_delta"
    assert summary.delta_pairs == ((0, 9),)
    assert len(summary.changed_cells) == 6
    assert summary.source_backed_cells == summary.changed_cells
    assert summary.entity_backed_cells == summary.changed_cells
    assert summary.affected_entity_ids


def test_entity_lesion_preserves_source_cells_but_removes_semantic_links() -> None:
    source, target = _background_anchor_demo(4, 5)
    execution = execute_scene_with_cell_provenance(
        _crop_program(background=9),
        source,
    )
    lesioned = lesion_entity_links(execution)
    node = RelationalEffectNode(9, "axis_project", "column", 0, 9)

    assert lesioned.exact
    assert lesioned.output == execution.output
    assert lesioned.provenance_execution_id != execution.provenance_execution_id
    assert all(not item.entity_ids and not item.relation_ids for item in lesioned.cells)
    assert execute_relational_effect(node, lesioned) == target


def test_frozen_domain_and_serialization_match_the_old_twelve_effects() -> None:
    domain = relational_effect_domain(0, 5)
    node = RelationalEffectNode(2, "stencil", "east", 0, 5)

    assert RELATIONAL_EFFECT_PROGRAMS_PER_DELTA == 120
    assert len(domain) == 120
    assert len(set(domain)) == 120
    assert RelationalEffectNode.from_json_dict(node.to_json_dict()) == node
    with pytest.raises(ValueError, match="outside the frozen"):
        RelationalEffectNode(2, "axis_project", "cross", 0, 5)
    with pytest.raises(ValueError, match="missing or unknown"):
        RelationalEffectNode.from_json_dict({**node.to_json_dict(), "extra": True})


def test_unsupported_parent_is_typed_and_never_silently_approximated() -> None:
    grid = as_grid([[2, 0], [0, 0]])
    program = ScenePipelineProgram(
        ParseObjectsNode(0, 4, "monochrome_components"),
        CorrespondObjectsNode(),
        SelectObjectsNode("all"),
        ObjectOperationNode("copy", axis="row", repeat_rule="scene_objects"),
        CanvasNode("tight", 0),
        RenderObjectsNode("objects"),
    )

    execution = execute_scene_with_cell_provenance(program, grid)

    assert execution.status == "unsupported"
    assert execution.reason == "provenance_unsupported_operator"
    assert execution.cells == ()
    with pytest.raises(ValueError, match="exact cell provenance"):
        provenance_anchor_coordinates(
            RelationalEffectNode(2, "stencil", "east", 0, 5),
            execution,
        )
