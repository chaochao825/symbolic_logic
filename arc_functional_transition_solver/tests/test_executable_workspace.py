from __future__ import annotations

from itertools import product

import pytest

from afts_arc.blind import BlindTask
from afts_arc.executable_workspace import (
    INPUT_GRID_NODE_ID,
    RECOLOR_NODE_DOMAIN,
    ProofObligation,
    RecolorGridNode,
    TypedHole,
    TypedProgramSketch,
    TypedSketchNode,
    abstract_recolor_domain,
    abstract_execute,
    analyze_recolor_reachability,
    execute_recolor_grid,
    execute_typed_sketch,
    fill_obligation,
    insert_recolor_hole,
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
)
from afts_arc.task import ARCPair, ARCTask


def _crop_program(
    *,
    correspond: CorrespondObjectsNode | None = None,
    role: str = "smallest_area",
) -> ScenePipelineProgram:
    return ScenePipelineProgram(
        ParseObjectsNode(0, 4, "monochrome_components"),
        CorrespondObjectsNode() if correspond is None else correspond,
        SelectObjectsNode(role),
        ObjectOperationNode("crop"),
        CanvasNode("bbox", 0),
        RenderObjectsNode("source_crop"),
    )


def _blind(
    source: list[list[int]],
    target: list[list[int]],
    *,
    query: list[list[int]] | None = None,
) -> BlindTask:
    return BlindTask.from_observations(
        train=(ARCPair(as_grid(source), as_grid(target)),),
        test_inputs=(as_grid(source if query is None else query),),
    )


def _blind_many(
    demonstrations: tuple[tuple[list[list[int]], list[list[int]]], ...],
    *,
    query: list[list[int]],
) -> BlindTask:
    return BlindTask.from_observations(
        train=tuple(
            ARCPair(as_grid(source), as_grid(target))
            for source, target in demonstrations
        ),
        test_inputs=(as_grid(query),),
    )


def _square_source() -> list[list[int]]:
    return [
        [0, 0, 0, 0],
        [0, 4, 4, 0],
        [0, 4, 4, 0],
        [0, 0, 0, 0],
    ]


def test_complete_sketch_round_trips_and_reuses_exact_legacy_semantics() -> None:
    program = _crop_program()
    sketch = TypedProgramSketch.from_scene_pipeline(program)
    reconstructed = TypedProgramSketch.from_json_dict(sketch.to_json_dict())

    assert reconstructed == sketch
    assert reconstructed.sketch_id == sketch.sketch_id
    assert reconstructed.materialize_scene_pipeline() == program

    result = abstract_execute(
        reconstructed,
        _blind(_square_source(), [[4, 4], [4, 4]]),
    )
    assert result.exact
    assert result.status == "exact"
    assert result.obligations == ()
    assert result.demo_executions[0].pixel_mismatch_count == 0
    assert result.result_id == abstract_execute(
        reconstructed,
        _blind(_square_source(), [[4, 4], [4, 4]]),
    ).result_id


def test_fill_hole_obligation_closes_the_minimal_execution_loop() -> None:
    program = _crop_program()
    hole = TypedHole(
        "selector-hole",
        "select",
        (("role_family", "extremal"),),
    )
    sketch = TypedProgramSketch.from_scene_pipeline(program, holes=(hole,))
    task = _blind(_square_source(), [[4, 4], [4, 4]])

    incomplete = abstract_execute(sketch, task)
    assert incomplete.status == "incomplete"
    assert incomplete.demo_executions == ()
    assert len(incomplete.obligations) == 1
    obligation = incomplete.obligations[0]
    assert obligation.obligation_type == "fill_typed_hole"
    assert obligation.target_node_types == ("select",)
    assert obligation.legal_actions == ("fill_typed_hole",)
    assert obligation.hole_id == "selector-hole"

    filled = fill_obligation(sketch, obligation, SelectObjectsNode("smallest_area"))
    exact = abstract_execute(filled, task)
    assert filled.complete
    assert exact.exact
    assert exact.obligations == ()


def test_fill_hole_rejects_wrong_node_type_and_non_fill_obligation() -> None:
    program = _crop_program()
    sketch = TypedProgramSketch.from_scene_pipeline(
        program,
        holes=(TypedHole("selector-hole", "select"),),
    )
    task = _blind(_square_source(), [[4, 4], [4, 4], [0, 0]])
    fill = abstract_execute(sketch, task).obligations[0]

    with pytest.raises(TypeError, match="does not match"):
        fill_obligation(sketch, fill, CanvasNode("bbox", 0))
    with pytest.raises(ValueError, match="signature roles require"):
        fill_obligation(sketch, fill, SelectObjectsNode("unique_signature"))
    another_sketch = TypedProgramSketch.from_scene_pipeline(
        _crop_program(
            correspond=CorrespondObjectsNode(
                "equivalence",
                ("shape",),
                True,
            ),
            role="unique_signature",
        ),
        holes=(TypedHole("selector-hole", "select"),),
    )
    with pytest.raises(ValueError, match="another parent sketch"):
        fill_obligation(
            another_sketch,
            fill,
            SelectObjectsNode("largest_area"),
        )

    complete = TypedProgramSketch.from_scene_pipeline(program)
    canvas = abstract_execute(complete, task).obligations[0]
    assert canvas.obligation_type == "canvas_mismatch"
    with pytest.raises(ValueError, match="only a fill-hole"):
        fill_obligation(sketch, canvas, SelectObjectsNode("smallest_area"))


def test_type_graph_rejects_dependency_type_mismatch_before_execution() -> None:
    parse = TypedSketchNode(
        "parse",
        "parse",
        (INPUT_GRID_NODE_ID,),
        ParseObjectsNode(0, 4, "monochrome_components"),
    )
    correspond = TypedSketchNode(
        "correspond",
        "correspond",
        (INPUT_GRID_NODE_ID,),
        CorrespondObjectsNode(),
    )

    with pytest.raises(TypeError, match="requires inputs"):
        TypedProgramSketch((parse, correspond), "correspond")


def test_complete_shape_failure_compiles_to_canvas_reinference() -> None:
    result = abstract_execute(
        TypedProgramSketch.from_scene_pipeline(_crop_program()),
        _blind(_square_source(), [[4, 4], [4, 4], [0, 0]]),
    )

    assert result.status == "incompatible"
    assert result.obligations[0].obligation_type == "canvas_mismatch"
    assert result.obligations[0].target_node_types == ("canvas",)
    assert result.obligations[0].legal_actions == ("canvas_reinfer",)
    assert result.demo_executions[0].shape_matches is False
    assert result.demo_executions[0].pixel_mismatch_count is None


def test_same_canvas_failures_preserve_joint_support_distinction() -> None:
    sketch = TypedProgramSketch.from_scene_pipeline(_crop_program())
    production = abstract_execute(
        sketch,
        _blind(_square_source(), [[4, 0], [4, 4]]),
    )
    render = abstract_execute(
        sketch,
        _blind(_square_source(), [[2, 2], [2, 2]]),
    )

    assert production.obligations[0].obligation_type == "production_mismatch"
    assert production.obligations[0].target_node_types == (
        "correspond",
        "select",
        "operate",
    )
    assert "insert_typed_node" in production.obligations[0].legal_actions
    assert render.obligations[0].obligation_type == "render_mismatch"
    assert render.obligations[0].target_node_types == ("render",)
    assert production.demo_executions[0].pixel_mismatch_count == 1
    assert render.demo_executions[0].pixel_mismatch_count == 4


def test_trace_invalidity_compiles_to_parse_or_selection_obligation() -> None:
    empty = abstract_execute(
        TypedProgramSketch.from_scene_pipeline(_crop_program()),
        _blind([[0, 0], [0, 0]], [[1]]),
    )
    assert empty.obligations[0].obligation_type == "reparse"
    assert empty.obligations[0].target_node_types == ("parse",)

    equivalence = CorrespondObjectsNode(
        "equivalence",
        ("shape", "size"),
        True,
    )
    source = [
        [1, 0, 1],
        [0, 0, 0],
    ]
    no_unique = abstract_execute(
        TypedProgramSketch.from_scene_pipeline(
            _crop_program(correspond=equivalence, role="unique_signature")
        ),
        _blind(source, [[1]]),
    )
    assert no_unique.obligations[0].obligation_type == "selection_conflict"
    assert no_unique.obligations[0].target_node_types == ("correspond", "select")


def test_diagnosis_is_query_blind_and_rejects_oracle_bearing_task() -> None:
    sketch = TypedProgramSketch.from_scene_pipeline(
        _crop_program(),
        holes=(TypedHole("selector-hole", "select"),),
    )
    first = abstract_execute(
        sketch,
        _blind(_square_source(), [[4, 4], [4, 4]], query=[[1]]),
    )
    second = abstract_execute(
        sketch,
        _blind(_square_source(), [[4, 4], [4, 4]], query=[[9, 9, 9]]),
    )
    assert first.blind_task_id != second.blind_task_id
    assert first.obligations == second.obligations
    assert first.obligations[0].obligation_id == second.obligations[0].obligation_id

    oracle_task = ARCTask(
        "oracle-bearing",
        (ARCPair(as_grid(_square_source()), as_grid([[4, 4], [4, 4]])),),
        (ARCPair(as_grid([[1]]), as_grid([[1]])),),
        "oracle-bearing.json",
        "0" * 64,
    )
    with pytest.raises(TypeError, match="oracle-free BlindTask"):
        abstract_execute(sketch, oracle_task)  # type: ignore[arg-type]


def test_strict_sketch_schema_rejects_unknown_fields() -> None:
    sketch = TypedProgramSketch.from_scene_pipeline(_crop_program())
    payload = sketch.to_json_dict()
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="missing or unknown"):
        TypedProgramSketch.from_json_dict(payload)


def test_recolor_reachability_inserts_one_node_and_closes_exactly() -> None:
    second_source = [
        [0, 4, 0],
        [0, 4, 4],
        [0, 0, 0],
    ]
    exact_source = [
        [0, 0, 0, 0],
        [0, 5, 5, 0],
        [0, 5, 5, 0],
        [0, 0, 0, 0],
    ]
    task = _blind_many(
        (
            (_square_source(), [[2, 2], [2, 2]]),
            (second_source, [[2, 0], [2, 2]]),
            (exact_source, [[5, 5], [5, 5]]),
        ),
        query=[[0, 4], [4, 4]],
    )
    base = TypedProgramSketch.from_scene_pipeline(_crop_program())

    reachability = analyze_recolor_reachability(base, task)
    assert reachability.status == "reachable"
    assert reachability.candidates == (RecolorGridNode(4, 2),)
    assert reachability.novel_frontier_count == 1
    assert reachability.obligations[0].obligation_type == "insert_typed_node"
    assert reachability.obligations[0].legal_actions == ("insert_typed_node",)
    assert reachability.result_id == analyze_recolor_reachability(
        base, task
    ).result_id

    inserted = insert_recolor_hole(
        base,
        reachability.obligations[0],
        hole_id="post-recolor-hole",
    )
    incomplete = abstract_execute(inserted, task)
    assert inserted.topology == "scene_pipeline_recolor"
    assert incomplete.status == "incomplete"
    assert incomplete.obligations[0].target_node_types == ("recolor_grid",)
    with pytest.raises(ValueError, match="output must be the appended node"):
        TypedProgramSketch(inserted.nodes, "render")
    wrong_edge = TypedSketchNode(
        inserted.nodes[-1].node_id,
        inserted.nodes[-1].node_type,
        (INPUT_GRID_NODE_ID,),
        inserted.nodes[-1].binding,
    )
    with pytest.raises(ValueError, match="extend the render output edge"):
        TypedProgramSketch(inserted.nodes[:-1] + (wrong_edge,), wrong_edge.node_id)

    completed = fill_obligation(
        inserted,
        incomplete.obligations[0],
        reachability.candidates[0],
    )
    exact = abstract_execute(completed, task)
    query_execution = execute_typed_sketch(completed, task.test_inputs[0])
    reconstructed = TypedProgramSketch.from_json_dict(completed.to_json_dict())

    assert exact.exact
    assert exact.obligations == ()
    assert exact.demo_executions[0].node_trace[-1].operator == "recolor_grid"
    assert query_execution.ok
    assert query_execution.output == as_grid([[0, 2], [2, 2]])
    assert reconstructed == completed
    assert reconstructed.sketch_id == completed.sketch_id
    with pytest.raises(ValueError, match="legacy materialization"):
        completed.materialize_scene_pipeline()


def test_recolor_reachability_rejects_shape_and_two_rewrite_failures() -> None:
    base = TypedProgramSketch.from_scene_pipeline(_crop_program())
    shape = analyze_recolor_reachability(
        base,
        _blind(_square_source(), [[2, 2], [2, 2], [2, 2]]),
    )
    two_rewrites = analyze_recolor_reachability(
        base,
        _blind(_square_source(), [[2, 1], [2, 2]]),
    )
    invalid_base = analyze_recolor_reachability(
        base,
        _blind([[0, 0], [0, 0]], [[2]]),
    )

    assert shape.status == "unreachable"
    assert shape.novel_frontier_count == 0
    assert dict(shape.obligations[0].evidence)["failure_kind"] == "shape_unreachable"
    assert shape.obligations[0].legal_actions == ("expand_representation",)
    assert two_rewrites.status == "unreachable"
    assert dict(two_rewrites.obligations[0].evidence)["failure_kind"] == (
        "single_recolor_unreachable"
    )
    assert invalid_base.status == "unreachable"
    assert dict(invalid_base.obligations[0].evidence)["failure_kind"] == (
        "base_execution_invalid"
    )


def test_recolor_domain_requires_one_mapping_shared_by_every_demo() -> None:
    base = TypedProgramSketch.from_scene_pipeline(_crop_program())
    task = _blind_many(
        (
            (_square_source(), [[2, 2], [2, 2]]),
            (_square_source(), [[3, 3], [3, 3]]),
        ),
        query=_square_source(),
    )
    reachability = analyze_recolor_reachability(base, task)

    assert all(demo.candidates for demo in reachability.demos)
    assert reachability.status == "unreachable"
    assert reachability.candidates == ()
    assert dict(reachability.obligations[0].evidence)["failure_kind"] == (
        "cross_demo_inconsistent"
    )


def test_recolor_base_exact_and_illegal_insertions_never_claim_frontier() -> None:
    base = TypedProgramSketch.from_scene_pipeline(_crop_program())
    exact_task = _blind(_square_source(), [[4, 4], [4, 4]])
    exact = analyze_recolor_reachability(base, exact_task)

    assert exact.status == "base_exact"
    assert exact.novel_frontier_count == 0
    assert exact.candidates == ()
    assert exact.obligations == ()

    wrong_obligation = abstract_execute(
        base,
        _blind(_square_source(), [[4, 4], [4, 4], [0, 0]]),
    ).obligations[0]
    with pytest.raises(ValueError, match="insert-node obligation"):
        insert_recolor_hole(base, wrong_obligation, hole_id="illegal-hole")

    zero_frontier = ProofObligation(
        "insert_typed_node",
        ("render",),
        ("render",),
        (0,),
        (
            ("novel_frontier_count", 0),
            ("required_node_type", "recolor_grid"),
        ),
    )
    with pytest.raises(ValueError, match="no novel frontier"):
        insert_recolor_hole(base, zero_frontier, hole_id="zero-hole")

    reachable = analyze_recolor_reachability(
        base,
        _blind(_square_source(), [[2, 2], [2, 2]]),
    )
    another_parent = TypedProgramSketch.from_scene_pipeline(
        _crop_program(role="largest_area")
    )
    with pytest.raises(ValueError, match="another parent sketch"):
        insert_recolor_hole(
            another_parent,
            reachable.obligations[0],
            hole_id="cross-parent-hole",
        )


def test_recolor_reachability_is_query_blind() -> None:
    base = TypedProgramSketch.from_scene_pipeline(_crop_program())
    first = analyze_recolor_reachability(
        base,
        _blind(_square_source(), [[2, 2], [2, 2]], query=[[1]]),
    )
    second = analyze_recolor_reachability(
        base,
        _blind(_square_source(), [[2, 2], [2, 2]], query=[[9, 9, 9]]),
    )

    assert first.blind_task_id != second.blind_task_id
    assert first.candidates == second.candidates
    assert first.obligations == second.obligations
    assert first.obligations[0].obligation_id == second.obligations[0].obligation_id


def test_recolor_abstract_domain_matches_exhaustive_concrete_execution() -> None:
    palette = range(3)
    for predicted_values in product(palette, repeat=2):
        predicted = as_grid([list(predicted_values)])
        for expected_values in product(palette, repeat=2):
            expected = as_grid([list(expected_values)])
            abstract = abstract_recolor_domain(predicted, expected)
            concrete = tuple(
                node
                for node in RECOLOR_NODE_DOMAIN
                if execute_recolor_grid(node, predicted) == expected
            )
            assert abstract == concrete
    for source_color in range(10):
        target_color = (source_color + 1) % 10
        assert abstract_recolor_domain(
            as_grid([[source_color]]),
            as_grid([[target_color]]),
        ) == (RecolorGridNode(source_color, target_color),)


def test_recolor_node_is_strict_content_addressed_and_shape_preserving() -> None:
    node = RecolorGridNode(4, 2)
    reconstructed = RecolorGridNode.from_json_dict(node.to_json_dict())
    source = as_grid([[4, 0], [1, 4]])

    assert reconstructed == node
    assert reconstructed.node_id == node.node_id
    assert execute_recolor_grid(node, source) == as_grid([[2, 0], [1, 2]])
    with pytest.raises(ValueError, match="must differ"):
        RecolorGridNode(4, 4)
    with pytest.raises(ValueError, match="missing or unknown"):
        RecolorGridNode.from_json_dict(
            {
                "op": "recolor_grid",
                "source_color": 4,
                "target_color": 2,
                "unknown": True,
            }
        )
