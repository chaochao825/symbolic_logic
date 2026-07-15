from __future__ import annotations

import unittest

from afts_arc.dsl import (
    Instruction,
    InvalidCode,
    Program,
    execute_program,
    primitive_registry,
)
from afts_arc.grid import as_grid


class DslTests(unittest.TestCase):
    def test_registry_is_versioned_deterministic_and_typed(self) -> None:
        names = tuple(spec.name for spec in primitive_registry())
        self.assertEqual(names[0], "identity")
        self.assertEqual(len(names), len(set(names)))
        with self.assertRaisesRegex(ValueError, "ARC color"):
            Instruction.create("recolor", old=0, new=10)

    def test_geometry_and_recolor_compose_deterministically(self) -> None:
        program = Program.create(
            (
                Instruction.create("rotate90"),
                Instruction.create("recolor", old=1, new=7),
            )
        )
        outcome = execute_program(program, as_grid([[1, 2, 3], [4, 5, 6]]))
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.output, ((4, 7), (5, 2), (6, 3)))
        self.assertEqual(outcome, execute_program(program, as_grid([[1, 2, 3], [4, 5, 6]])))

    def test_color_map_canonicalization_stabilizes_program_id(self) -> None:
        first = Program.create(
            (Instruction.create("map_colors", pairs=[(2, 8), (1, 7)]),)
        )
        second = Program.create(
            (Instruction.create("map_colors", pairs=[(1, 7), (2, 8)]),)
        )
        self.assertEqual(first, second)

    def test_pixel_scaling_and_grid_tiling_are_distinct_typed_operations(self) -> None:
        grid = as_grid([[1, 2], [3, 4]])
        scale = Program.create(
            (
                Instruction.create(
                    "scale_pixels", row_factor=2, column_factor=2
                ),
            )
        )
        tile = Program.create(
            (
                Instruction.create(
                    "tile_grid", row_repeats=1, column_repeats=2
                ),
            )
        )
        self.assertEqual(
            execute_program(scale, grid).output,
            (
                (1, 1, 2, 2),
                (1, 1, 2, 2),
                (3, 3, 4, 4),
                (3, 3, 4, 4),
            ),
        )
        self.assertEqual(
            execute_program(tile, grid).output,
            ((1, 2, 1, 2), (3, 4, 3, 4)),
        )
        for invalid in (False, 0, -1, 31):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "integer"
            ):
                Instruction.create(
                    "scale_pixels", row_factor=invalid, column_factor=1
                )

    def test_shape_overflow_is_semantic_invalid_at_the_failing_instruction(self) -> None:
        one_step = Program.create(
            (
                Instruction.create(
                    "scale_pixels", row_factor=2, column_factor=1
                ),
            )
        )
        overflow = execute_program(one_step, as_grid([[1] for _ in range(16)]))
        self.assertEqual(overflow.invalid_code, InvalidCode.OUTPUT_SHAPE_EXCEEDS_LIMIT)
        self.assertEqual(overflow.failing_instruction_index, 0)

        two_steps = Program.create(
            (
                Instruction.create(
                    "scale_pixels", row_factor=2, column_factor=1
                ),
                Instruction.create(
                    "scale_pixels", row_factor=2, column_factor=1
                ),
            )
        )
        second_overflow = execute_program(two_steps, as_grid([[1] for _ in range(8)]))
        self.assertEqual(
            second_overflow.invalid_code, InvalidCode.OUTPUT_SHAPE_EXCEEDS_LIMIT
        )
        self.assertEqual(second_overflow.failing_instruction_index, 1)

    def test_panel_overlay_executes_and_maps_normal_invalid_codes(self) -> None:
        program = Program.create(
            (Instruction.create("overlay_panel_grid", background=0),)
        )
        grid = as_grid(
            [
                [1, 0, 9, 3, 4],
                [0, 2, 9, 0, 0],
                [9, 9, 9, 9, 9],
                [5, 0, 9, 7, 8],
                [6, 0, 9, 0, 0],
            ]
        )
        self.assertEqual(execute_program(program, grid).output, ((1, 4), (6, 2)))

        ragged = [[0 for _ in range(7)] for _ in range(7)]
        for index in (2, 5):
            ragged[index] = [8 for _ in range(7)]
            for row in range(7):
                ragged[row][index] = 8
        incompatible = execute_program(program, as_grid(ragged))
        self.assertEqual(
            incompatible.invalid_code, InvalidCode.INCOMPATIBLE_PANEL_SHAPES
        )
        self.assertEqual(incompatible.failing_instruction_index, 0)

        empty = execute_program(program, as_grid([[1, 2], [3, 4]]))
        self.assertEqual(empty.invalid_code, InvalidCode.EMPTY_SELECTION)

    def test_panel_sequence_d4_is_typed_and_maps_selection_invalids(self) -> None:
        program = Program.create(
            (
                Instruction.create(
                    "broadcast_panel_sequence_d4",
                    background=0,
                    step="rotate90",
                ),
            )
        )
        grid = as_grid(
            [
                [6, 3, 3, 5, 0, 0, 0, 5, 0, 0, 0],
                [6, 3, 3, 5, 0, 0, 0, 5, 0, 0, 0],
                [6, 3, 2, 5, 0, 0, 0, 5, 0, 0, 0],
            ]
        )
        outcome = execute_program(program, grid)
        self.assertTrue(outcome.ok)
        self.assertEqual(
            outcome.output,
            as_grid(
                [
                    [6, 3, 3, 5, 6, 6, 6, 5, 2, 3, 6],
                    [6, 3, 3, 5, 3, 3, 3, 5, 3, 3, 6],
                    [6, 3, 2, 5, 2, 3, 3, 5, 3, 3, 6],
                ]
            ),
        )
        tied = execute_program(
            program,
            as_grid([[1, 0, 9, 2, 0], [0, 0, 9, 0, 0]]),
        )
        self.assertEqual(tied.invalid_code, InvalidCode.NON_UNIQUE_SELECTION)
        with self.assertRaisesRegex(ValueError, "D4"):
            Instruction.create(
                "broadcast_panel_sequence_d4",
                background=0,
                step="rotate45",
            )

    def test_panel_lattice_periodic_is_typed_and_maps_invalids(self) -> None:
        program = Program.create(
            (
                Instruction.create(
                    "broadcast_panel_lattice_periodic",
                    background=0,
                    row_period=2,
                    column_period=2,
                ),
            )
        )
        grid = as_grid(
            [
                [1, 2, 3, 9, 0, 0, 0, 9, 0, 0],
                [4, 5, 6, 9, 0, 0, 0, 9, 0, 0],
                [9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
                [0, 0, 0, 9, 0, 0, 0, 9, 0, 0],
                [0, 0, 0, 9, 0, 0, 0, 9, 0, 0],
                [9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
                [0, 0, 0, 9, 0, 0, 0, 9, 0, 0],
            ]
        )
        outcome = execute_program(program, grid)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.output[0][-2:], (1, 2))
        self.assertEqual(outcome.output[-1][:3], (1, 2, 3))
        self.assertEqual(outcome.output[-1][-2:], (1, 2))

        tied = as_grid(
            [
                [1, 0, 9, 2, 0],
                [0, 0, 9, 0, 0],
                [9, 9, 9, 9, 9],
                [0, 0, 9, 0, 0],
                [0, 0, 9, 0, 0],
            ]
        )
        invalid = execute_program(program, tied)
        self.assertEqual(invalid.invalid_code, InvalidCode.NON_UNIQUE_SELECTION)
        for bad_period in (False, 0, -1, 31):
            with self.subTest(bad_period=bad_period), self.assertRaisesRegex(
                ValueError, "integer"
            ):
                Instruction.create(
                    "broadcast_panel_lattice_periodic",
                    background=0,
                    row_period=bad_period,
                    column_period=1,
                )

    def test_bbox_contact_renderer_is_typed_and_maps_relation_invalids(self) -> None:
        program = Program.create(
            (Instruction.create("paint_bbox_contacts", background=0),)
        )
        grid = as_grid(
            [
                [0, 0, 0, 0, 4, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 8, 8, 8, 0, 0],
                [2, 0, 0, 8, 8, 8, 0, 0],
                [0, 0, 0, 8, 8, 8, 0, 6],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        outcome = execute_program(program, grid)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.output[0][4], 4)
        self.assertEqual(outcome.output[2][4], 4)
        self.assertEqual(outcome.output[3][0], 2)
        self.assertEqual(outcome.output[3][3], 2)
        self.assertEqual(outcome.output[4][7], 6)
        self.assertEqual(outcome.output[4][5], 6)

        collision = execute_program(
            program,
            as_grid(
                [
                    [0, 0, 2, 0, 0],
                    [0, 0, 0, 0, 0],
                    [3, 0, 8, 8, 0],
                    [0, 0, 8, 8, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
        )
        self.assertEqual(collision.invalid_code, InvalidCode.TARGET_COLLISION)
        with self.assertRaisesRegex(ValueError, "ARC color"):
            Instruction.create("paint_bbox_contacts", background=10)

    def test_empty_and_nonunique_object_selection_return_invalid(self) -> None:
        instruction = Instruction.create(
            "crop_largest_object",
            background=0,
            connectivity=4,
            color_mode="single_color",
        )
        program = Program.create((instruction,))
        empty = execute_program(program, as_grid([[0, 0]]))
        tied = execute_program(program, as_grid([[1, 0, 1]]))
        self.assertEqual(empty.invalid_code, InvalidCode.EMPTY_SELECTION)
        self.assertEqual(tied.invalid_code, InvalidCode.NON_UNIQUE_SELECTION)

    def test_crop_and_keep_largest_have_explicit_shapes(self) -> None:
        grid = as_grid([[0, 2, 2, 0], [0, 2, 2, 0], [3, 0, 0, 0]])
        arguments = dict(background=0, connectivity=4, color_mode="single_color")
        crop = Program.create((Instruction.create("crop_largest_object", **arguments),))
        keep = Program.create((Instruction.create("keep_largest_object", **arguments),))
        self.assertEqual(execute_program(crop, grid).output, ((2, 2), (2, 2)))
        self.assertEqual(
            execute_program(keep, grid).output,
            ((0, 2, 2, 0), (0, 2, 2, 0), (0, 0, 0, 0)),
        )

    def test_program_json_round_trip_rejects_wrong_version_or_id(self) -> None:
        program = Program.create((Instruction.create("rotate90"),))
        self.assertEqual(Program.from_json_dict(program.to_json_dict()), program)
        wrong_version = program.to_json_dict()
        wrong_version["dsl_semantics_version"] = "future"
        with self.assertRaisesRegex(ValueError, "semantics"):
            Program.from_json_dict(wrong_version)
        forged = program.to_json_dict()
        forged["program_id"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "program_id"):
            Program.from_json_dict(forged)


if __name__ == "__main__":
    unittest.main()
