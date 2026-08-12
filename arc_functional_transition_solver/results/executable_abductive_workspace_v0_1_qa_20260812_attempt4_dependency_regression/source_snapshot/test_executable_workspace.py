from __future__ import annotations

import pytest

from afts_arc.blind import BlindTask
from afts_arc.executable_workspace import (
    INPUT_GRID_NODE_ID,
    TypedHole,
    TypedProgramSketch,
    TypedSketchNode,
    abstract_execute,
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
