from __future__ import annotations

from afts_arc.blind import BlindTask
from afts_arc.grid import as_grid
from afts_arc.relational_mask import (
    RelationalMaskProgram,
    compile_relational_mask_failure_certificate,
    enumerate_relational_mask_programs,
    execute_relational_mask,
    relational_mask_program_id,
    single_slot_relational_mask_variants,
)
from afts_arc.task import ARCPair


def _radial_grid() -> tuple[list[list[int]], list[list[int]]]:
    grid = [[8 for _ in range(7)] for _ in range(7)]
    for row, column in (
        (0, 3),
        (1, 3),
        (5, 3),
        (6, 3),
        (3, 0),
        (3, 1),
        (3, 5),
        (3, 6),
    ):
        grid[row][column] = 1
    output = [row[:] for row in grid]
    output[3][3] = 4
    return grid, output


def _blind_task() -> BlindTask:
    grid, output = _radial_grid()
    return BlindTask.from_observations(
        train=(ARCPair(as_grid(grid), as_grid(output)),),
        test_inputs=(as_grid(grid),),
    )


def _program(*, radii: tuple[int, ...] = (2, 3)) -> RelationalMaskProgram:
    return RelationalMaskProgram(8, 1, 4, "cardinal4", radii, "copy")


def test_relational_mask_marks_jointly_supported_center() -> None:
    grid, output = _radial_grid()

    execution = execute_relational_mask(_program(), as_grid(grid))

    assert execution.ok
    assert execution.output == as_grid(output)
    assert execution.mask == ((3, 3),)
    assert tuple(node.node_id for node in execution.node_trace) == (
        "parse",
        "relate",
        "mask",
        "render",
    )


def test_relational_mask_serialization_and_id_are_replayable() -> None:
    program = _program()
    reconstructed = RelationalMaskProgram.from_json_dict(program.to_json_dict())

    assert reconstructed == program
    assert relational_mask_program_id(reconstructed) == relational_mask_program_id(
        program
    )


def test_query_blind_enumeration_contains_radial_rule() -> None:
    programs = enumerate_relational_mask_programs(_blind_task())

    assert _program() in programs


def test_single_slot_variant_changes_only_existing_radius_slot() -> None:
    parent = _program(radii=(1, 2))
    target = _program()

    edits = single_slot_relational_mask_variants(
        _blind_task(),
        parent,
        allowed_slots=("relation.radii",),
        candidate_programs=(parent, target),
    )

    assert edits == (
        type(edits[0])("relation.radii", target),
    )


def test_relational_mask_reports_empty_relation_as_invalid() -> None:
    grid = as_grid([[8, 1], [1, 8]])

    execution = execute_relational_mask(_program(), grid)

    assert not execution.ok
    assert execution.reason == "no_relational_centers"
    assert execution.output is None


def test_failure_certificate_selects_existing_canvas_slot() -> None:
    task = _blind_task()
    parent = RelationalMaskProgram(8, 1, 4, "cardinal4", (2, 3), "blank")

    certificate = compile_relational_mask_failure_certificate(
        task_id="fixture",
        task=task,
        parent=parent,
    )

    assert certificate.diagnosis == "render_canvas"
    assert certificate.affected_slots == ("render.canvas_mode",)
    assert certificate.evidence["query_gold_read"] is False


def test_failure_certificate_separates_palette_from_geometry() -> None:
    task = _blind_task()
    palette = RelationalMaskProgram(8, 1, 2, "cardinal4", (2, 3), "copy")
    geometry = RelationalMaskProgram(8, 1, 4, "diagonal4", (2, 3), "copy")

    palette_certificate = compile_relational_mask_failure_certificate(
        task_id="palette",
        task=task,
        parent=palette,
    )
    geometry_certificate = compile_relational_mask_failure_certificate(
        task_id="geometry",
        task=task,
        parent=geometry,
    )

    assert palette_certificate.diagnosis == "render_palette"
    assert palette_certificate.affected_slots == ("render.output_color",)
    assert geometry_certificate.diagnosis == "relation_geometry"
    assert "relation.radii" in geometry_certificate.affected_slots
