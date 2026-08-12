from __future__ import annotations

from itertools import product

import pytest

from afts_arc.anchor_mask import (
    ANCHOR_MASK_NODE_ID,
    ANCHOR_MASK_NODE_TYPE,
    anchor_mask_domain,
    execute_anchor_rasterized_delta,
)
from afts_arc.anchor_mask_reachability import (
    abstract_anchor_mask_domain,
    analyze_anchor_mask_reachability,
    insert_anchor_mask_hole,
    leave_one_demo_out_anchor_masks,
)
from afts_arc.blind import BlindTask
from afts_arc.executable_workspace import (
    ProofObligation,
    TypedProgramSketch,
    abstract_execute,
    execute_typed_sketch,
    fill_obligation,
)
from afts_arc.grid import as_grid
from afts_arc.hybrid.scene_graph import (
    CanvasNode,
    CorrespondObjectsNode,
    ObjectOperationNode,
    ParseObjectsNode,
    RenderObjectsNode,
    ScenePipelineProgram,
    SelectObjectsNode,
    extract_scene_graph,
)
from afts_arc.task import ARCPair


def _identity_crop_program() -> ScenePipelineProgram:
    return ScenePipelineProgram(
        ParseObjectsNode(0, 4, "monochrome_components"),
        CorrespondObjectsNode(),
        SelectObjectsNode("all"),
        ObjectOperationNode("crop"),
        CanvasNode("bbox", 0),
        RenderObjectsNode("source_crop"),
    )


def _pair(anchor_column: int) -> ARCPair:
    source = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    source[0][anchor_column] = 2
    source[2][anchor_column] = 2
    output = [row[:] for row in source]
    output[1][anchor_column] = 5
    return ARCPair(as_grid(source), as_grid(output))


def _task(*, query_column: int = 2) -> BlindTask:
    return BlindTask.from_observations(
        train=(_pair(0), _pair(1)),
        test_inputs=(_pair(query_column).input,),
    )


def test_exact_version_space_inserts_cross_representation_node_and_replays() -> None:
    base = TypedProgramSketch.from_scene_pipeline(_identity_crop_program())
    task = _task()
    reachability = analyze_anchor_mask_reachability(base, task)

    assert reachability.status == "reachable"
    assert any(
        node.anchor_color == 2
        and node.mask_kind == "axis_project"
        and node.mask_parameter == "column"
        and node.source_color == 1
        and node.target_color == 5
        for node in reachability.candidates
    )
    inserted = insert_anchor_mask_hole(
        base,
        reachability.obligations[0],
        hole_id="mask-hole",
    )
    assert inserted.topology == "scene_pipeline_anchor_mask"
    assert inserted.nodes[-1].node_id == ANCHOR_MASK_NODE_ID
    assert inserted.nodes[-1].node_type == ANCHOR_MASK_NODE_TYPE
    assert inserted.nodes[-1].input_ids == ("render", "$input", "parse")

    incomplete = abstract_execute(inserted, task)
    assert incomplete.status == "incomplete"
    chosen = next(
        node
        for node in reachability.candidates
        if node.anchor_color == 2
        and node.mask_kind == "axis_project"
        and node.mask_parameter == "column"
    )
    complete = fill_obligation(
        inserted,
        incomplete.obligations[0],
        chosen,
    )
    assert abstract_execute(complete, task).exact
    query = execute_typed_sketch(complete, task.test_inputs[0])
    assert query.ok
    assert query.output == _pair(2).output
    reconstructed = TypedProgramSketch.from_json_dict(complete.to_json_dict())
    assert reconstructed == complete
    assert reconstructed.sketch_id == complete.sketch_id


def test_lodo_predicts_every_fold_and_beats_simple_recolor_baselines() -> None:
    base = TypedProgramSketch.from_scene_pipeline(_identity_crop_program())
    lodo = leave_one_demo_out_anchor_masks(base, _task())

    assert lodo["predictable_fold_count"] == 2
    assert lodo["exact_fold_count"] == 2
    assert lodo["strict_all_folds_exact"] is True
    assert lodo["identity_exact_fold_count"] == 0
    assert lodo["global_recolor_exact_fold_count"] == 0
    assert lodo["component_recolor_exact_fold_count"] == 0


def test_mask_language_rejects_multi_delta_and_canvas_mismatch() -> None:
    base = TypedProgramSketch.from_scene_pipeline(_identity_crop_program())
    multi = ARCPair(
        _pair(0).input,
        as_grid([[2, 1, 1], [5, 3, 1], [2, 1, 1]]),
    )
    multi_task = BlindTask.from_observations(
        train=(multi,),
        test_inputs=(_pair(0).input,),
    )
    multi_result = analyze_anchor_mask_reachability(base, multi_task)

    assert multi_result.status == "unreachable"
    assert dict(multi_result.obligations[0].evidence)["failure_kind"] == (
        "multi_delta_unreachable"
    )

    crop_source = as_grid([[0, 0, 0], [0, 2, 0], [0, 0, 0]])
    crop_target = as_grid([[5]])
    canvas_task = BlindTask.from_observations(
        train=(ARCPair(crop_source, crop_target),),
        test_inputs=(crop_source,),
    )
    canvas_result = analyze_anchor_mask_reachability(base, canvas_task)
    assert canvas_result.status == "unreachable"


def test_abstract_domain_matches_exhaustive_concrete_execution() -> None:
    input_grid = as_grid([[2, 0], [0, 2]])
    scene = extract_scene_graph(
        input_grid,
        background=0,
        connectivity=4,
        grouping="monochrome_components",
    )
    palette = range(2)
    for parent_values in product(palette, repeat=4):
        parent = as_grid([parent_values[:2], parent_values[2:]])
        for expected_values in product(palette, repeat=4):
            expected = as_grid([expected_values[:2], expected_values[2:]])
            abstract = abstract_anchor_mask_domain(
                input_grid=input_grid,
                parent_grid=parent,
                expected_grid=expected,
                scene=scene,
                source_color=0,
                target_color=1,
            )
            concrete = tuple(
                node
                for node in anchor_mask_domain(0, 1)
                if execute_anchor_rasterized_delta(
                    node,
                    input_grid,
                    parent,
                    scene,
                )
                == expected
            )
            assert abstract == concrete


def test_query_input_never_changes_demo_version_space() -> None:
    base = TypedProgramSketch.from_scene_pipeline(_identity_crop_program())
    first = analyze_anchor_mask_reachability(base, _task(query_column=0))
    second = analyze_anchor_mask_reachability(base, _task(query_column=2))

    assert first.candidates == second.candidates
    assert first.obligations == second.obligations
    with pytest.raises(ValueError, match="another node type"):
        source = first.obligations[0]
        evidence = tuple(
            (key, "recolor_grid" if key == "required_node_type" else value)
            for key, value in source.evidence
        )
        insert_anchor_mask_hole(
            base,
            ProofObligation(
                source.obligation_type,
                source.target_node_ids,
                source.target_node_types,
                source.demo_indices,
                evidence,
                source.hole_id,
            ),
            hole_id="wrong",
        )
