from __future__ import annotations

import pytest

from afts_arc.blind import BlindTask
from afts_arc.control_legend import (
    ControlLegendProgram,
    LegendParseNode,
    LegendPayloadNode,
    LegendRenderNode,
    OrderedColorCorrespondenceNode,
    control_legend_program_id,
    enumerate_control_legend_programs,
    execute_control_legend,
    parse_control_legend,
    synthesize_control_legend_programs,
)
from afts_arc.grid import Grid, as_grid, transpose
from afts_arc.task import ARCPair


def _program(
    operation: str,
    *,
    pair_axis: str = "horizontal",
    pair_direction: str = "forward",
    sequence_direction: str = "forward",
) -> ControlLegendProgram:
    rewrite = operation == "ordered_rewrite_payload"
    return ControlLegendProgram(
        LegendParseNode(pair_axis, pair_direction, sequence_direction),
        OrderedColorCorrespondenceNode(),
        LegendPayloadNode(
            "non_control_cells" if rewrite else "source_color_components",
            None if rewrite else 4,
        ),
        LegendRenderNode(operation, None if rewrite else 1),
    )


def _rewrite_fixture() -> tuple[Grid, Grid]:
    source = as_grid(
        [
            [1, 4, 0, 0, 1],
            [4, 6, 0, 0, 4],
            [6, 7, 0, 0, 3],
            [0, 0, 0, 0, 0],
        ]
    )
    target = as_grid(
        [
            [1, 4, 0, 0, 7],
            [4, 6, 0, 0, 7],
            [6, 7, 0, 0, 3],
            [0, 0, 0, 0, 0],
        ]
    )
    return source, target


def _blind(source: Grid, target: Grid, query: Grid | None = None) -> BlindTask:
    return BlindTask.from_observations(
        train=(ARCPair(source, target),),
        test_inputs=(source if query is None else query,),
    )


def test_ordered_rewrite_applies_rules_sequentially_and_protects_control() -> None:
    source, target = _rewrite_fixture()
    execution = execute_control_legend(_program("ordered_rewrite_payload"), source)
    assert execution.ok
    assert execution.output == target
    assert execution.ordered_rules == ((1, 4), (4, 6), (6, 7))
    assert execution.control_cells == (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    )
    assert execution.write_mask == ((0, 4), (1, 4))
    assert all(cell not in execution.write_mask for cell in execution.control_cells)


def test_fill_exterior_bbox_preserves_enclosed_holes() -> None:
    source = as_grid(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 3, 3, 3, 3, 3, 0, 0],
            [0, 0, 0, 3, 0, 0, 0, 3, 0, 0],
            [0, 0, 0, 3, 0, 0, 0, 3, 0, 0],
            [0, 0, 0, 3, 0, 0, 0, 3, 0, 0],
            [0, 0, 0, 3, 3, 3, 3, 3, 0, 0],
            [3, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 2, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    execution = execute_control_legend(
        _program("fill_exterior_bbox_background"), source
    )
    assert execution.ok
    assert execution.payload_component_count == 1
    assert execution.output is not None
    assert execution.output[0][2:9] == (1, 1, 1, 1, 1, 1, 1)
    assert execution.output[6][2:9] == (1, 1, 1, 1, 1, 1, 1)
    assert tuple(execution.output[row][2] for row in range(7)) == (1,) * 7
    assert tuple(execution.output[row][8] for row in range(7)) == (1,) * 7
    assert execution.output[3][5] == 0
    assert execution.output[6][0:2] == (3, 1)
    assert execution.output[7][0:2] == (3, 1)
    assert execution.output[9][0:2] == (4, 2)


def test_repeated_pairs_collapse_without_losing_control_cells() -> None:
    source = as_grid(
        [
            [3, 1, 0, 0, 3],
            [3, 1, 0, 0, 3],
            [0, 0, 0, 0, 0],
            [4, 2, 0, 0, 4],
        ]
    )
    parsed = parse_control_legend(
        LegendParseNode("horizontal", "forward", "forward"), source
    )
    assert parsed.ordered_rules == ((3, 1), (4, 2))
    assert parsed.raw_pair_count == 3
    assert len(parsed.control_cells) == 6


def test_vertical_lane_is_supported_by_the_same_typed_program() -> None:
    source, target = _rewrite_fixture()
    execution = execute_control_legend(
        _program("ordered_rewrite_payload", pair_axis="vertical"),
        transpose(source),
    )
    assert execution.ok
    assert execution.output == transpose(target)
    assert execution.ordered_rules == ((1, 4), (4, 6), (6, 7))


def test_ambiguous_control_lanes_are_rejected() -> None:
    source = as_grid(
        [
            [1, 2, 0, 0, 0, 0, 0, 0],
            [2, 3, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 4, 5, 0],
            [0, 0, 0, 0, 0, 5, 6, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    execution = execute_control_legend(_program("ordered_rewrite_payload"), source)
    assert not execution.ok
    assert execution.reason == "ambiguous_control_lanes"
    assert execution.output is None
    with pytest.raises(ValueError, match="ambiguous_control_lanes"):
        parse_control_legend(
            LegendParseNode("horizontal", "forward", "forward"), source
        )


def test_program_round_trip_is_strict_and_content_addressed() -> None:
    program = _program("fill_exterior_bbox_background")
    reconstructed = ControlLegendProgram.from_json_dict(program.to_json_dict())
    assert reconstructed == program
    assert control_legend_program_id(reconstructed) == control_legend_program_id(
        program
    )
    malformed = program.to_json_dict()
    malformed["unexpected"] = True
    with pytest.raises(ValueError, match="missing or unknown"):
        ControlLegendProgram.from_json_dict(malformed)


def test_enumeration_is_bounded_and_query_output_free() -> None:
    source, target = _rewrite_fixture()
    first = _blind(source, target, as_grid([[0, 1], [0, 0]]))
    second = _blind(source, target, as_grid([[7, 0, 0]]))
    first_programs = enumerate_control_legend_programs(first)
    second_programs = enumerate_control_legend_programs(second)
    assert len(first_programs) == len(second_programs) == 16
    assert tuple(map(control_legend_program_id, first_programs)) == tuple(
        map(control_legend_program_id, second_programs)
    )


def test_demo_exact_synthesis_rejects_near_miss_render() -> None:
    source, target = _rewrite_fixture()
    exact = synthesize_control_legend_programs(_blind(source, target))
    assert exact
    assert all(
        score.program.render.operation == "ordered_rewrite_payload" for score in exact
    )
    assert all(score.all_demo_exact for score in exact)
