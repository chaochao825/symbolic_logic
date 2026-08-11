from __future__ import annotations

from afts_arc.blind import BlindTask
from afts_arc.experiment_safety import canonical_sha256
from afts_arc.grid import as_grid
from afts_arc.relational_transducer import (
    STRUCTURAL_HYPOTHESIS_COUNT,
    RelationalTransducerProgram,
    allocation_count,
    build_visual_transducer_certificate,
    execute_relational_transducer,
    synthesize_relational_transducers,
    transform_coordinates,
)
from afts_arc.task import ARCPair


def _task(pairs: list[tuple[list[list[int]], list[list[int]]]]) -> BlindTask:
    return BlindTask.from_observations(
        train=tuple(ARCPair(as_grid(source), as_grid(target)) for source, target in pairs),
        test_inputs=(as_grid([[0]]),),
    )


def test_fixed_grammar_has_exactly_twenty_four_structural_hypotheses() -> None:
    assert STRUCTURAL_HYPOTHESIS_COUNT == 24


def test_synthesizes_static_palette_homomorphism_without_fixed_colors() -> None:
    task = _task(
        [
            (
                [[0, 8, 9], [4, 0, 9]],
                [[0, 6, 9], [6, 0, 9]],
            ),
            (
                [[4, 8, 0], [9, 0, 9]],
                [[6, 6, 0], [9, 0, 9]],
            ),
        ]
    )

    programs = synthesize_relational_transducers(task)
    program = next(
        item
        for item in programs
        if item.coordinate_mode == "identity"
        and item.color_mode == "global_homomorphism"
    )

    assert execute_relational_transducer(program, as_grid([[8, 4, 9]])) == as_grid(
        [[6, 6, 9]]
    )
    assert dict(program.color_mapping) == {0: 0, 4: 6, 8: 6, 9: 9}


def test_synthesizes_row_count_and_mirror_palette_programs() -> None:
    count_task = _task(
        [
            (
                [[3, 0, 3, 0], [7, 7, 7, 0]],
                [[4, 0, 4, 0], [5, 5, 5, 0]],
                ),
                (
                    [[8, 8, 8, 8], [9, 0, 0, 0], [0, 0, 0, 0]],
                    [[6, 6, 6, 6], [3, 0, 0, 0], [0, 0, 0, 0]],
                ),
        ]
    )
    count_program = next(
        item
        for item in synthesize_relational_transducers(count_task)
        if item.coordinate_mode == "identity"
        and item.color_mode == "row_foreground_count_offset"
    )
    assert count_program.count_offset == 2
    assert execute_relational_transducer(
        count_program, as_grid([[5, 5, 5, 0], [2, 2, 0, 0]])
    ) == as_grid([[5, 5, 5, 0], [4, 4, 0, 0]])

    mirror_task = _task(
        [
            (
                [[8, 6, 0, 0], [0, 8, 0, 0]],
                [[0, 0, 2, 4], [0, 0, 4, 0]],
            ),
            (
                [[6, 0, 0, 0], [8, 8, 0, 0]],
                [[0, 0, 0, 2], [0, 0, 4, 4]],
            ),
        ]
    )
    mirror = next(
        item
        for item in synthesize_relational_transducers(mirror_task)
        if item.coordinate_mode == "reflect_vertical_axis"
        and item.color_mode == "global_homomorphism"
    )
    assert execute_relational_transducer(mirror, as_grid([[8, 6, 0, 0]])) == as_grid(
        [[0, 0, 2, 4]]
    )


def test_program_round_trip_d4_and_visual_certificate_are_content_addressed() -> None:
    program = RelationalTransducerProgram(
        coordinate_mode="reflect_vertical_axis",
        color_mode="global_homomorphism",
        color_mapping=((0, 0), (2, 5)),
    )
    assert RelationalTransducerProgram.from_json_dict(program.to_json_dict()) == program
    grid = as_grid([[1, 2, 3], [4, 5, 6]])
    assert transform_coordinates(grid, "rotate_90_clockwise") == as_grid(
        [[4, 1], [5, 2], [6, 3]]
    )

    certificate = build_visual_transducer_certificate(
        task_id="task",
        query_inputs=(as_grid([[0, 2], [0, 0]]),),
        posterior_queries=(
            (
                (as_grid([[0, 5], [0, 0]]), 3),
                (as_grid([[5, 5], [0, 0]]), 1),
            ),
        ),
        visual_unique_candidate_count=2,
        recursive_unique_candidate_count=3,
        exact_pool_overlap_count=1,
    )
    assert certificate["score"] == {
        "minimum_preserving_numerator": 3,
        "minimum_preserving_denominator": 4,
        "total_unique_candidate_count": 5,
        "exact_pool_overlap_count": 1,
    }
    assert certificate["certificate_id"] == canonical_sha256(
        {
            key: value
            for key, value in certificate.items()
            if key != "certificate_id"
        }
    )
    assert allocation_count(49) == 15
    assert allocation_count(100) == 30
