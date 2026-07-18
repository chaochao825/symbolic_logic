from __future__ import annotations

import unittest
from unittest.mock import patch

from afts_arc.blind import BlindTask
from afts_arc.grid import as_grid, rotate180, rotate90
from afts_arc.dsl import ExecutionOutcome, InvalidCode
from afts_arc.search import (
    BBoxContactBound,
    BBoxContactProposalResult,
    PanelLatticePeriodicBound,
    PanelLatticePeriodicProposalResult,
    SearchConfig,
    bbox_contact_instruction_proposals,
    bbox_contact_options_after_cap,
    dsl_candidates,
    instruction_proposals,
    panel_instruction_proposals,
    panel_lattice_periodic_instruction_proposals,
    panel_lattice_periodic_options_after_cap,
    panel_sequence_d4_instruction_proposals,
    search_programs,
)
from afts_arc.task import ARCPair, ARCTask


def _arc_task(
    task_id: str,
    train: tuple[tuple[list[list[int]], list[list[int]]], ...],
    test_input: list[list[int]],
    test_output: list[list[int]],
) -> ARCTask:
    return ARCTask(
        task_id=task_id,
        train=tuple(ARCPair(as_grid(source), as_grid(target)) for source, target in train),
        test=(ARCPair(as_grid(test_input), as_grid(test_output)),),
        source_path="fixture",
        source_sha256=(task_id.encode().hex() + "0" * 64)[:64],
    )


def _panel_scene(separator: int) -> list[list[int]]:
    return [
        [1, 0, separator, 3, 4],
        [0, 2, separator, 0, 0],
        [separator, separator, separator, separator, separator],
        [5, 0, separator, 7, 8],
        [6, 0, separator, 0, 0],
    ]


def _panel_sequence_scene(seed: list[list[int]], separator: int) -> list[list[int]]:
    empty = [[0, 0, 0] for _ in range(3)]
    panels = [seed, empty, empty]
    return [
        [cell for index, panel in enumerate(panels) for cell in (([separator] if index else []) + panel[row])]
        for row in range(3)
    ]


def _panel_sequence_expected(seed: list[list[int]], separator: int) -> list[list[int]]:
    panels = [as_grid(seed), rotate90(as_grid(seed)), rotate180(as_grid(seed))]
    return [
        [cell for index, panel in enumerate(panels) for cell in (([separator] if index else []) + list(panel[row]))]
        for row in range(3)
    ]


def _panel_lattice_case(
    *,
    row_heights: tuple[int, ...],
    column_widths: tuple[int, ...],
    seed: list[list[int]],
    source: tuple[int, int],
    separator: int,
    row_period: int = 2,
    column_period: int = 2,
) -> tuple[list[list[int]], list[list[int]]]:
    height = sum(row_heights) + len(row_heights) - 1
    width = sum(column_widths) + len(column_widths) - 1
    values = [[0 for _ in range(width)] for _ in range(height)]
    row_starts: list[int] = []
    cursor = 0
    for index, panel_height in enumerate(row_heights):
        if index:
            values[cursor] = [separator for _ in range(width)]
            cursor += 1
        row_starts.append(cursor)
        cursor += panel_height
    column_starts: list[int] = []
    cursor = 0
    for index, panel_width in enumerate(column_widths):
        if index:
            for row in range(height):
                values[row][cursor] = separator
            cursor += 1
        column_starts.append(cursor)
        cursor += panel_width

    source_row, source_column = source
    source_top = row_starts[source_row]
    source_left = column_starts[source_column]
    for row in range(row_heights[source_row]):
        for column in range(column_widths[source_column]):
            values[source_top + row][source_left + column] = seed[row][column]

    expected = [list(row) for row in values]
    for panel_row, panel_height in enumerate(row_heights):
        for panel_column, panel_width in enumerate(column_widths):
            if (
                (panel_row - source_row) % row_period
                or (panel_column - source_column) % column_period
            ):
                continue
            top = row_starts[panel_row]
            left = column_starts[panel_column]
            for row in range(panel_height):
                for column in range(panel_width):
                    expected[top + row][left + column] = seed[row][column]
    return values, expected


def _bbox_contact_scene(
    markers: tuple[tuple[str, int, int, int], ...],
    *,
    anchor_height: int = 3,
    anchor_width: int = 4,
) -> tuple[list[list[int]], list[list[int]]]:
    top = 4
    left = 4
    values = [[0 for _ in range(13)] for _ in range(13)]
    for row in range(top, top + anchor_height):
        for column in range(left, left + anchor_width):
            values[row][column] = 8
    expected = [list(row) for row in values]
    for side, offset, gap, color in markers:
        if side == "left":
            marker = (top + offset, left - gap - 1)
            contact = (top + offset, left)
        elif side == "right":
            marker = (top + offset, left + anchor_width + gap)
            contact = (top + offset, left + anchor_width - 1)
        elif side == "top":
            marker = (top - gap - 1, left + offset)
            contact = (top, left + offset)
        elif side == "bottom":
            marker = (top + anchor_height + gap, left + offset)
            contact = (top + anchor_height - 1, left + offset)
        else:
            raise ValueError(f"unknown bbox-contact side: {side}")
        values[marker[0]][marker[1]] = color
        expected[marker[0]][marker[1]] = color
        expected[contact[0]][contact[1]] = color
    return values, expected


class SearchTests(unittest.TestCase):
    def test_internal_executor_error_fails_search_instead_of_becoming_candidate(self) -> None:
        task = _arc_task("internal", (([[0]], [[1]]),), [[2]], [[3]])
        blind = BlindTask.from_task(task)
        internal = ExecutionOutcome(
            status="invalid",
            output=None,
            invalid_code=InvalidCode.INTERNAL_ERROR,
            failing_instruction_index=0,
            semantic_trace=(),
        )
        with (
            patch("afts_arc.search.execute_program", return_value=internal),
            self.assertRaisesRegex(RuntimeError, "internal error"),
        ):
            search_programs(blind, config=SearchConfig(max_depth=1))

    def test_finds_geometric_program_and_query_output(self) -> None:
        train_input = [[1, 2, 3], [4, 5, 6]]
        train_output = [list(row) for row in rotate90(as_grid(train_input))]
        test_input = [[7, 8], [9, 1], [2, 3]]
        test_output = [list(row) for row in rotate90(as_grid(test_input))]
        task = _arc_task("rotate", ((train_input, train_output),), test_input, test_output)
        blind = BlindTask.from_task(task)
        result = search_programs(blind, config=SearchConfig(max_depth=1))
        self.assertTrue(any(program.functional_trace == ("rotate90",) for program in result.exact_programs))
        self.assertTrue(any(candidate.output == task.test[0].output for candidate in dsl_candidates(blind, search_result=result)))

    def test_finds_composed_rotate_and_recolor_without_query_oracle(self) -> None:
        train_input = [[1, 2, 0], [0, 1, 2]]
        train_output = [[0, 7], [7, 2], [2, 0]]
        test_input = [[1, 0], [2, 1]]
        expected = ((2, 7), (7, 0))
        first = _arc_task(
            "compose",
            ((train_input, train_output),),
            test_input,
            [list(row) for row in expected],
        )
        altered_oracle = _arc_task(
            "compose",
            ((train_input, train_output),),
            test_input,
            [[9]],
        )
        blind = BlindTask.from_task(first)
        changed_blind = BlindTask.from_task(altered_oracle)
        self.assertEqual(blind, changed_blind)
        result = search_programs(blind, config=SearchConfig(max_depth=2, beam_width=64))
        changed_result = search_programs(changed_blind, config=result.config)
        self.assertEqual(result, changed_result)
        outputs = {candidate.output for candidate in dsl_candidates(blind, search_result=result)}
        self.assertIn(expected, outputs)

    def test_crop_non_background_is_found_as_demo_exact_program(self) -> None:
        task = _arc_task(
            "crop",
            (
                (
                    [[0, 0, 0, 0], [0, 3, 3, 0], [0, 3, 3, 0]],
                    [[3, 3], [3, 3]],
                ),
            ),
            [[0, 4, 4], [0, 4, 4], [0, 0, 0]],
            [[4, 4], [4, 4]],
        )
        blind = BlindTask.from_task(task)
        result = search_programs(blind, config=SearchConfig(max_depth=1))
        self.assertTrue(
            any(program.functional_trace == ("crop_non_background",) for program in result.exact_programs)
        )
        self.assertTrue(all(evaluation.all_demo_exact for evaluation in result.exact_evaluations))

    def test_demo_shape_ratio_proposes_and_finds_scale_without_query_oracle(self) -> None:
        def scale(values: list[list[int]]) -> list[list[int]]:
            return [
                [cell for cell in row for _ in range(2)]
                for row in values
                for _ in range(2)
            ]

        train = (
            ([[1, 2], [3, 4]], scale([[1, 2], [3, 4]])),
            ([[5, 6, 7]], scale([[5, 6, 7]])),
        )
        test_input = [[8, 9], [1, 2]]
        expected = scale(test_input)
        task = _arc_task("scale", train, test_input, expected)
        altered = _arc_task("scale", train, test_input, [[9]])
        blind = BlindTask.from_task(task)
        changed_blind = BlindTask.from_task(altered)
        self.assertEqual(blind, changed_blind)

        options = instruction_proposals(blind)
        option_names = [item.op for item in options[:64]]
        self.assertIn("scale_pixels", option_names)
        self.assertIn("tile_grid", option_names)
        result = search_programs(
            blind,
            config=SearchConfig(max_depth=1, max_instruction_options=64),
        )
        changed_result = search_programs(changed_blind, config=result.config)
        self.assertEqual(result, changed_result)
        self.assertEqual(
            (result.shape_proposals[0].row_factor, result.shape_proposals[0].column_factor),
            (2, 2),
        )
        self.assertTrue(
            any(program.functional_trace == ("scale_pixels",) for program in result.exact_programs)
        )
        self.assertIn(
            as_grid(expected),
            {candidate.output for candidate in dsl_candidates(blind, search_result=result)},
        )

    def test_demo_shape_ratio_finds_whole_grid_horizontal_tile(self) -> None:
        train_input = [[1, 2], [3, 4]]
        train_output = [[1, 2, 1, 2], [3, 4, 3, 4]]
        test_input = [[5, 6, 7]]
        expected = [[5, 6, 7, 5, 6, 7]]
        task = _arc_task(
            "tile",
            ((train_input, train_output),),
            test_input,
            expected,
        )
        blind = BlindTask.from_task(task)
        result = search_programs(blind, config=SearchConfig(max_depth=1))
        self.assertTrue(
            any(program.functional_trace == ("tile_grid",) for program in result.exact_programs)
        )
        self.assertIn(
            as_grid(expected),
            {candidate.output for candidate in dsl_candidates(blind, search_result=result)},
        )

    def test_depth_two_shape_overflow_does_not_abort_search(self) -> None:
        train_input = [[1] for _ in range(4)]
        train_output = [[1] for _ in range(8)]
        test_input = [[2] for _ in range(10)]
        test_output = [[2] for _ in range(20)]
        task = _arc_task(
            "shape-overflow",
            ((train_input, train_output),),
            test_input,
            test_output,
        )
        result = search_programs(
            BlindTask.from_task(task),
            config=SearchConfig(max_depth=2, beam_width=64),
        )
        self.assertTrue(result.exact_programs)
        self.assertGreater(result.expansions, len(result.instruction_options))

    def test_panel_overlay_is_proposed_early_and_is_query_oracle_blind(self) -> None:
        expected = [[1, 4], [6, 2]]
        train = (
            (_panel_scene(9), expected),
            (_panel_scene(5), expected),
        )
        task = _arc_task("panel", train, _panel_scene(8), expected)
        altered = _arc_task("panel", train, _panel_scene(8), [[9]])
        blind = BlindTask.from_task(task)
        changed_blind = BlindTask.from_task(altered)
        self.assertEqual(blind, changed_blind)
        self.assertEqual(
            [item.to_json_dict() for item in panel_instruction_proposals(blind)],
            [{"op": "overlay_panel_grid", "arguments": {"background": 0}}],
        )
        option_names = [item.op for item in instruction_proposals(blind)[:64]]
        self.assertIn("overlay_panel_grid", option_names)
        result = search_programs(blind, config=SearchConfig(max_depth=1))
        self.assertEqual(result, search_programs(changed_blind, config=result.config))
        self.assertTrue(
            any(
                program.functional_trace == ("overlay_panel_grid",)
                for program in result.exact_programs
            )
        )
        self.assertIn(
            as_grid(expected),
            {candidate.output for candidate in dsl_candidates(blind, search_result=result)},
        )

    def test_panel_overlay_composes_with_d4(self) -> None:
        expected = [[6, 1], [2, 4]]
        task = _arc_task(
            "panel-rotate",
            (
                (_panel_scene(9), expected),
                (_panel_scene(5), expected),
            ),
            _panel_scene(8),
            expected,
        )
        result = search_programs(
            BlindTask.from_task(task),
            config=SearchConfig(max_depth=2, beam_width=64),
        )
        self.assertTrue(
            any(
                program.functional_trace == ("overlay_panel_grid", "rotate90")
                for program in result.exact_programs
            )
        )

    def test_panel_sequence_d4_enumerates_all_steps_and_is_query_oracle_blind(
        self,
    ) -> None:
        seed_a = [[6, 3, 3], [6, 3, 3], [6, 3, 2]]
        seed_b = [[1, 1, 2], [4, 1, 1], [4, 4, 1]]
        seed_q = [[3, 3, 9], [9, 9, 9], [2, 9, 9]]
        train = (
            (_panel_sequence_scene(seed_a, 5), _panel_sequence_expected(seed_a, 5)),
            (_panel_sequence_scene(seed_b, 7), _panel_sequence_expected(seed_b, 7)),
        )
        test_input = _panel_sequence_scene(seed_q, 8)
        expected = _panel_sequence_expected(seed_q, 8)
        task = _arc_task("panel-sequence", train, test_input, expected)
        altered = _arc_task("panel-sequence", train, test_input, [[9]])
        blind = BlindTask.from_task(task)
        changed_blind = BlindTask.from_task(altered)
        self.assertEqual(blind, changed_blind)
        proposals = panel_sequence_d4_instruction_proposals(blind)
        self.assertEqual(len(proposals), 8)
        self.assertEqual(
            {item.arguments["step"] for item in proposals},
            {
                "identity",
                "rotate90",
                "rotate180",
                "rotate270",
                "flip_horizontal",
                "flip_vertical",
                "transpose",
                "anti_transpose",
            },
        )
        option_names = [item.op for item in instruction_proposals(blind)[:64]]
        self.assertEqual(option_names.count("broadcast_panel_sequence_d4"), 8)
        result = search_programs(blind, config=SearchConfig(max_depth=1))
        self.assertEqual(result, search_programs(changed_blind, config=result.config))
        self.assertEqual(result.panel_sequence_d4_trial_count, 8)
        self.assertEqual(result.panel_sequence_d4_demo_execution_count, 16)
        self.assertEqual(
            result.instruction_option_count_post_cap,
            len(result.instruction_options),
        )
        self.assertEqual(
            result.instruction_option_truncation_count,
            result.instruction_option_count_pre_cap
            - result.instruction_option_count_post_cap,
        )
        self.assertTrue(
            any(
                program.instructions[0].op == "broadcast_panel_sequence_d4"
                and program.instructions[0].arguments["step"] == "rotate90"
                for program in result.exact_programs
            )
        )
        self.assertIn(
            as_grid(expected),
            {candidate.output for candidate in dsl_candidates(blind, search_result=result)},
        )
        capped = search_programs(
            blind,
            config=SearchConfig(max_depth=1, max_instruction_options=10),
        )
        self.assertGreater(capped.instruction_option_count_pre_cap, 10)
        self.assertEqual(capped.instruction_option_count_post_cap, 10)
        self.assertEqual(
            capped.instruction_option_truncation_count,
            capped.instruction_option_count_pre_cap - 10,
        )
        self.assertEqual(
            sum(
                item.op == "broadcast_panel_sequence_d4"
                for item in capped.instruction_options
            ),
            2,
        )

    def test_panel_lattice_periodic_bounds_proposals_and_cap_are_blind(self) -> None:
        seed_a = [[1, 2, 3], [4, 5, 6]]
        seed_b = [[2, 4, 6], [1, 3, 5]]
        seed_q = [[7, 2, 4], [5, 1, 6]]
        train_a = _panel_lattice_case(
            row_heights=(2, 2, 1),
            column_widths=(3, 3, 2),
            seed=seed_a,
            source=(0, 0),
            separator=5,
        )
        train_b = _panel_lattice_case(
            row_heights=(2, 2, 2, 1),
            column_widths=(3, 3, 3, 2),
            seed=seed_b,
            source=(2, 2),
            separator=7,
        )
        query = _panel_lattice_case(
            row_heights=(2, 2, 2, 1),
            column_widths=(3, 3, 2),
            seed=seed_q,
            source=(0, 1),
            separator=8,
        )
        task = _arc_task(
            "panel-periodic",
            ((train_a[0], train_a[1]), (train_b[0], train_b[1])),
            query[0],
            query[1],
        )
        altered = _arc_task(
            "panel-periodic",
            ((train_a[0], train_a[1]), (train_b[0], train_b[1])),
            query[0],
            [[9]],
        )
        blind = BlindTask.from_task(task)
        changed_blind = BlindTask.from_task(altered)
        self.assertEqual(blind, changed_blind)

        proposals = panel_lattice_periodic_instruction_proposals(blind)
        self.assertEqual(len(proposals), 9)
        self.assertEqual(
            [
                (
                    item.arguments["row_period"],
                    item.arguments["column_period"],
                )
                for item in proposals
            ],
            [(row, column) for row in range(1, 4) for column in range(1, 4)],
        )
        result = search_programs(blind, config=SearchConfig(max_depth=1))
        self.assertEqual(result, search_programs(changed_blind, config=result.config))
        self.assertEqual(
            result.panel_lattice_periodic_period_bounds,
            (
                PanelLatticePeriodicBound(
                    background=0,
                    max_row_period=3,
                    max_column_period=3,
                ),
            ),
        )
        self.assertEqual(result.panel_lattice_periodic_structural_check_count, 2)
        self.assertEqual(result.panel_lattice_periodic_trial_count, 9)
        self.assertEqual(result.panel_lattice_periodic_demo_execution_count, 18)
        with self.assertRaisesRegex(ValueError, "both be zero or positive"):
            PanelLatticePeriodicProposalResult(
                instructions=result.panel_lattice_periodic_instruction_proposals,
                period_bounds=result.panel_lattice_periodic_period_bounds,
                structural_check_count=2,
                trial_count=9,
                demo_execution_count=0,
            )
        self.assertEqual(
            panel_lattice_periodic_options_after_cap(
                result.panel_lattice_periodic_instruction_proposals,
                result.instruction_options,
            ),
            9,
        )
        self.assertTrue(
            any(
                program.functional_trace
                == ("broadcast_panel_lattice_periodic",)
                and program.instructions[0].arguments["row_period"] == 2
                and program.instructions[0].arguments["column_period"] == 2
                for program in result.exact_programs
            )
        )
        self.assertIn(
            as_grid(query[1]),
            {
                candidate.output
                for candidate in dsl_candidates(blind, search_result=result)
            },
        )

        capped = search_programs(
            blind,
            config=SearchConfig(max_depth=1, max_instruction_options=13),
        )
        self.assertGreater(capped.instruction_option_count_pre_cap, 13)
        self.assertEqual(capped.instruction_option_count_post_cap, 13)
        self.assertEqual(
            panel_lattice_periodic_options_after_cap(
                capped.panel_lattice_periodic_instruction_proposals,
                capped.instruction_options,
            ),
            5,
        )

    def test_bbox_contact_bounds_costs_cap_and_proposals_are_blind(self) -> None:
        train_a = _bbox_contact_scene(
            (
                ("left", 1, 1, 3),
                ("right", 0, 2, 4),
                ("top", 2, 1, 6),
                ("bottom", 1, 0, 7),
            )
        )
        train_b = _bbox_contact_scene(
            (
                ("left", 0, 2, 2),
                ("top", 1, 0, 5),
                ("bottom", 2, 1, 9),
            )
        )
        query = _bbox_contact_scene(
            (
                ("left", 2, 0, 4),
                ("right", 1, 1, 6),
                ("top", 2, 2, 3),
                ("bottom", 1, 1, 7),
            )
        )
        task = _arc_task(
            "bbox-contact",
            ((train_a[0], train_a[1]), (train_b[0], train_b[1])),
            query[0],
            query[1],
        )
        altered_oracle = _arc_task(
            "bbox-contact",
            ((train_a[0], train_a[1]), (train_b[0], train_b[1])),
            query[0],
            [[9]],
        )
        blind = BlindTask.from_task(task)
        changed_blind = BlindTask.from_task(altered_oracle)
        self.assertEqual(blind, changed_blind)
        self.assertEqual(
            [item.to_json_dict() for item in bbox_contact_instruction_proposals(blind)],
            [{"op": "paint_bbox_contacts", "arguments": {"background": 0}}],
        )
        changed_query_input = BlindTask.from_observations(
            train=blind.train,
            test_inputs=(as_grid([[0, 0], [0, 0]]),),
        )
        self.assertEqual(
            bbox_contact_instruction_proposals(blind),
            bbox_contact_instruction_proposals(changed_query_input),
        )
        result = search_programs(
            blind,
            config=SearchConfig(max_depth=1, max_instruction_options=64),
        )
        self.assertEqual(result, search_programs(changed_blind, config=result.config))
        self.assertEqual(
            result.bbox_contact_bounds,
            (
                BBoxContactBound(
                    background=0,
                    max_object_count=5,
                    max_anchor_candidate_count=1,
                    max_relation_count=4,
                ),
                BBoxContactBound(
                    background=8,
                    max_object_count=5,
                    max_anchor_candidate_count=0,
                    max_relation_count=0,
                ),
            ),
        )
        self.assertEqual(result.bbox_contact_structural_check_count, 4)
        self.assertEqual(result.bbox_contact_anchor_candidate_count, 2)
        self.assertEqual(result.bbox_contact_relation_check_count, 7)
        self.assertEqual(result.bbox_contact_admissible_binding_count, 1)
        self.assertEqual(result.bbox_contact_action_trial_count, 1)
        self.assertEqual(result.bbox_contact_demo_execution_count, 2)
        self.assertEqual(
            bbox_contact_options_after_cap(
                result.bbox_contact_instruction_proposals,
                result.instruction_options,
            ),
            1,
        )
        self.assertTrue(
            any(
                program.functional_trace == ("paint_bbox_contacts",)
                for program in result.exact_programs
            )
        )
        self.assertIn(
            as_grid(query[1]),
            {
                candidate.output
                for candidate in dsl_candidates(blind, search_result=result)
            },
        )
        capped = search_programs(
            blind,
            config=SearchConfig(max_depth=1, max_instruction_options=9),
        )
        self.assertEqual(
            bbox_contact_options_after_cap(
                capped.bbox_contact_instruction_proposals,
                capped.instruction_options,
            ),
            1,
        )
        same_shape_wrong_outputs = BlindTask.from_task(
            _arc_task(
                "bbox-contact-wrong-train",
                (
                    (train_a[0], [[0 for _ in row] for row in train_a[1]]),
                    (train_b[0], [[0 for _ in row] for row in train_b[1]]),
                ),
                query[0],
                query[1],
            )
        )
        changed_pixels = search_programs(
            same_shape_wrong_outputs,
            config=SearchConfig(max_depth=1, max_instruction_options=64),
        )
        self.assertEqual(
            result.bbox_contact_instruction_proposals,
            changed_pixels.bbox_contact_instruction_proposals,
        )
        self.assertEqual(result.bbox_contact_bounds, changed_pixels.bbox_contact_bounds)
        self.assertEqual(
            result.bbox_contact_structural_check_count,
            changed_pixels.bbox_contact_structural_check_count,
        )
        self.assertEqual(
            result.bbox_contact_relation_check_count,
            changed_pixels.bbox_contact_relation_check_count,
        )
        with self.assertRaisesRegex(ValueError, "demo executions"):
            BBoxContactProposalResult(
                instructions=result.bbox_contact_instruction_proposals,
                bounds=result.bbox_contact_bounds,
                structural_check_count=4,
                anchor_candidate_count=2,
                relation_check_count=7,
                admissible_binding_count=1,
                action_trial_count=1,
                demo_execution_count=0,
            )


if __name__ == "__main__":
    unittest.main()
