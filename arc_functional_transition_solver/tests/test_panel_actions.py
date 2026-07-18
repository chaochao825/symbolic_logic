from __future__ import annotations

import unittest

from afts_arc.grid import as_grid, rotate90, rotate270
from afts_arc.panel import parse_panels
from afts_arc.panel_actions import (
    D4Step,
    PanelLatticePeriodicCode,
    PanelSequenceD4Code,
    apply_d4_power,
    broadcast_panel_lattice_periodic,
    broadcast_panel_sequence_d4,
    panel_lattice_period_bounds,
    reassemble_panels,
)


def _horizontal_sequence(
    panels: list[list[list[int]]], *, separator: int = 9
) -> list[list[int]]:
    height = len(panels[0])
    values: list[list[int]] = []
    for row in range(height):
        combined: list[int] = []
        for index, panel in enumerate(panels):
            if index:
                combined.append(separator)
            combined.extend(panel[row])
        values.append(combined)
    return values


def _periodic_lattice(
    *,
    row_heights: tuple[int, ...],
    column_widths: tuple[int, ...],
    seed: list[list[int]],
    source: tuple[int, int],
    separator: int = 9,
    background: int = 0,
    extra_sources: tuple[tuple[int, int], ...] = (),
) -> list[list[int]]:
    height = sum(row_heights) + len(row_heights) - 1
    width = sum(column_widths) + len(column_widths) - 1
    values = [[background for _ in range(width)] for _ in range(height)]
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

    for panel_row, panel_column in (source, *extra_sources):
        top = row_starts[panel_row]
        left = column_starts[panel_column]
        for row in range(row_heights[panel_row]):
            for column in range(column_widths[panel_column]):
                values[top + row][left + column] = seed[row][column]
    return values


class PanelActionTests(unittest.TestCase):
    def test_d4_powers_support_all_elements_and_negative_exponents(self) -> None:
        grid = as_grid([[1, 2, 3], [4, 5, 6]])
        orders = {
            D4Step.IDENTITY: 1,
            D4Step.ROTATE90: 4,
            D4Step.ROTATE180: 2,
            D4Step.ROTATE270: 4,
            D4Step.FLIP_HORIZONTAL: 2,
            D4Step.FLIP_VERTICAL: 2,
            D4Step.TRANSPOSE: 2,
            D4Step.ANTI_TRANSPOSE: 2,
        }
        for step, order in orders.items():
            with self.subTest(step=step):
                self.assertEqual(apply_d4_power(grid, step=step, exponent=0), grid)
                self.assertEqual(
                    apply_d4_power(grid, step=step, exponent=order), grid
                )
                self.assertEqual(
                    apply_d4_power(grid, step=step, exponent=-1),
                    apply_d4_power(grid, step=step, exponent=order - 1),
                )
        self.assertEqual(
            apply_d4_power(grid, step=D4Step.ROTATE90, exponent=-1),
            rotate270(grid),
        )
        self.assertEqual(
            apply_d4_power(grid, step=D4Step.ROTATE270, exponent=-1),
            rotate90(grid),
        )

    def test_horizontal_interior_seed_uses_relative_positive_and_negative_powers(self) -> None:
        empty = [[0, 0, 0] for _ in range(3)]
        seed = [[1, 2, 0], [0, 3, 4], [5, 0, 6]]
        grid = as_grid(_horizontal_sequence([empty, seed, empty]))
        output, invalid = broadcast_panel_sequence_d4(
            grid, background=0, step=D4Step.ROTATE90
        )
        self.assertIsNone(invalid)
        self.assertEqual(
            output,
            as_grid(
                [
                    [0, 4, 6, 9, 1, 2, 0, 9, 5, 0, 1],
                    [2, 3, 0, 9, 0, 3, 4, 9, 0, 3, 2],
                    [1, 0, 5, 9, 5, 0, 6, 9, 6, 4, 0],
                ]
            ),
        )
    def test_arc2_8e5a5113_first_demonstration_is_exact(self) -> None:
        grid = as_grid(
            [
                [6, 3, 3, 5, 0, 0, 0, 5, 0, 0, 0],
                [6, 3, 3, 5, 0, 0, 0, 5, 0, 0, 0],
                [6, 3, 2, 5, 0, 0, 0, 5, 0, 0, 0],
            ]
        )
        expected = as_grid(
            [
                [6, 3, 3, 5, 6, 6, 6, 5, 2, 3, 6],
                [6, 3, 3, 5, 3, 3, 3, 5, 3, 3, 6],
                [6, 3, 2, 5, 2, 3, 3, 5, 3, 3, 6],
            ]
        )
        output, invalid = broadcast_panel_sequence_d4(
            grid, background=0, step=D4Step.ROTATE90
        )
        self.assertIsNone(invalid)
        self.assertEqual(output, expected)

    def test_empty_multiple_seed_ragged_cross_and_rectangular_rotation_are_invalid(
        self,
    ) -> None:
        empty = [[0, 0], [0, 0]]
        one = [[1, 0], [0, 0]]
        two = [[0, 2], [0, 0]]

        output, invalid = broadcast_panel_sequence_d4(
            as_grid(_horizontal_sequence([empty, empty])),
            background=0,
            step=D4Step.IDENTITY,
        )
        self.assertIsNone(output)
        self.assertEqual(invalid, PanelSequenceD4Code.EMPTY_SELECTION)

        output, invalid = broadcast_panel_sequence_d4(
            as_grid(_horizontal_sequence([one, two])),
            background=0,
            step=D4Step.IDENTITY,
        )
        self.assertIsNone(output)
        self.assertEqual(invalid, PanelSequenceD4Code.NON_UNIQUE_SELECTION)

        ragged = as_grid([[1, 0, 9, 0, 0, 9, 0], [0, 0, 9, 0, 0, 9, 0]])
        output, invalid = broadcast_panel_sequence_d4(
            ragged, background=0, step=D4Step.IDENTITY
        )
        self.assertIsNone(output)
        self.assertEqual(invalid, PanelSequenceD4Code.INCOMPATIBLE_PANEL_SHAPES)

        ragged_multiple_seed = as_grid(
            [[1, 0, 9, 2, 0, 9, 0], [0, 0, 9, 0, 0, 9, 0]]
        )
        output, invalid = broadcast_panel_sequence_d4(
            ragged_multiple_seed,
            background=0,
            step=D4Step.IDENTITY,
        )
        self.assertIsNone(output)
        self.assertEqual(invalid, PanelSequenceD4Code.NON_UNIQUE_SELECTION)

        cross = as_grid(
            [[1, 0, 9, 0, 0], [0, 0, 9, 0, 0], [9, 9, 9, 9, 9]]
        )
        output, invalid = broadcast_panel_sequence_d4(
            cross, background=0, step=D4Step.IDENTITY
        )
        self.assertIsNone(output)
        self.assertEqual(invalid, PanelSequenceD4Code.EMPTY_SELECTION)

        rectangular = as_grid(
            _horizontal_sequence([[[1, 0, 2], [0, 3, 0]], [[0, 0, 0], [0, 0, 0]]])
        )
        for step in (
            D4Step.IDENTITY,
            D4Step.ROTATE180,
            D4Step.FLIP_HORIZONTAL,
            D4Step.FLIP_VERTICAL,
        ):
            with self.subTest(rectangular_valid=step):
                output, invalid = broadcast_panel_sequence_d4(
                    rectangular, background=0, step=step
                )
                self.assertIsNone(invalid)
                self.assertIsNotNone(output)
        for step in (
            D4Step.ROTATE90,
            D4Step.ROTATE270,
            D4Step.TRANSPOSE,
            D4Step.ANTI_TRANSPOSE,
        ):
            with self.subTest(rectangular_invalid=step):
                output, invalid = broadcast_panel_sequence_d4(
                    rectangular, background=0, step=step
                )
                self.assertIsNone(output)
                self.assertEqual(
                    invalid,
                    PanelSequenceD4Code.INCOMPATIBLE_PANEL_SHAPES,
                )

    def test_reassembly_rejects_unbound_or_wrong_shape_replacements(self) -> None:
        empty = [[0, 0], [0, 0]]
        seed = [[1, 0], [0, 2]]
        grid = as_grid(_horizontal_sequence([seed, empty]))
        hypothesis = next(
            item
            for item in parse_panels(grid).hypotheses
            if item.separator_color == 9
        )
        with self.assertRaisesRegex(ValueError, "replacement count"):
            reassemble_panels(grid, hypothesis=hypothesis, replacements=(as_grid(seed),))
        with self.assertRaisesRegex(ValueError, "shape"):
            reassemble_panels(
                grid,
                hypothesis=hypothesis,
                replacements=(as_grid([[1]]), as_grid(empty)),
            )
        other = as_grid([[1, 9, 0]])
        with self.assertRaisesRegex(ValueError, "bound"):
            reassemble_panels(
                other,
                hypothesis=hypothesis,
                replacements=(as_grid(seed), as_grid(empty)),
            )

    def test_periodic_lattice_broadcast_clips_bottom_and_right_suffixes(self) -> None:
        seed = [[1, 2, 3], [4, 5, 6]]
        grid = as_grid(
            _periodic_lattice(
                row_heights=(2, 2, 1),
                column_widths=(3, 3, 2),
                seed=seed,
                source=(0, 0),
            )
        )
        self.assertEqual(panel_lattice_period_bounds(grid, background=0), (2, 2))
        output, invalid = broadcast_panel_lattice_periodic(
            grid,
            background=0,
            row_period=2,
            column_period=2,
        )
        self.assertIsNone(invalid)
        self.assertEqual(
            output,
            as_grid(
                [
                    [1, 2, 3, 9, 0, 0, 0, 9, 1, 2],
                    [4, 5, 6, 9, 0, 0, 0, 9, 4, 5],
                    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
                    [0, 0, 0, 9, 0, 0, 0, 9, 0, 0],
                    [0, 0, 0, 9, 0, 0, 0, 9, 0, 0],
                    [9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
                    [1, 2, 3, 9, 0, 0, 0, 9, 1, 2],
                ]
            ),
        )
        no_repeat, invalid = broadcast_panel_lattice_periodic(
            grid,
            background=0,
            row_period=30,
            column_period=30,
        )
        self.assertIsNone(invalid)
        self.assertEqual(no_repeat, grid)

    def test_periodic_lattice_rejects_ambiguous_and_non_suffix_geometry(self) -> None:
        seed = [[1, 2], [3, 4]]
        multiple = as_grid(
            _periodic_lattice(
                row_heights=(2, 2),
                column_widths=(2, 2),
                seed=seed,
                source=(0, 0),
                extra_sources=((1, 1),),
            )
        )
        output, invalid = broadcast_panel_lattice_periodic(
            multiple,
            background=0,
            row_period=1,
            column_period=1,
        )
        self.assertIsNone(output)
        self.assertEqual(invalid, PanelLatticePeriodicCode.NON_UNIQUE_SELECTION)

        for row_heights in ((1, 2, 2), (2, 1, 2)):
            with self.subTest(row_heights=row_heights):
                grid = as_grid(
                    _periodic_lattice(
                        row_heights=row_heights,
                        column_widths=(2, 2),
                        seed=seed,
                        source=(1, 0),
                    )
                )
                output, invalid = broadcast_panel_lattice_periodic(
                    grid,
                    background=0,
                    row_period=1,
                    column_period=1,
                )
                self.assertIsNone(output)
                self.assertEqual(
                    invalid,
                    PanelLatticePeriodicCode.INCOMPATIBLE_PANEL_SHAPES,
                )

        for column_widths in ((1, 2, 2), (2, 1, 2)):
            with self.subTest(column_widths=column_widths):
                grid = as_grid(
                    _periodic_lattice(
                        row_heights=(2, 2),
                        column_widths=column_widths,
                        seed=seed,
                        source=(0, 1),
                    )
                )
                output, invalid = broadcast_panel_lattice_periodic(
                    grid,
                    background=0,
                    row_period=1,
                    column_period=1,
                )
                self.assertIsNone(output)
                self.assertEqual(
                    invalid,
                    PanelLatticePeriodicCode.INCOMPATIBLE_PANEL_SHAPES,
                )

        partial_source = as_grid(
            _periodic_lattice(
                row_heights=(2, 1),
                column_widths=(2, 2),
                seed=seed,
                source=(1, 0),
            )
        )
        output, invalid = broadcast_panel_lattice_periodic(
            partial_source,
            background=0,
            row_period=1,
            column_period=1,
        )
        self.assertIsNone(output)
        self.assertEqual(
            invalid,
            PanelLatticePeriodicCode.INCOMPATIBLE_PANEL_SHAPES,
        )

        partial_column_source = as_grid(
            _periodic_lattice(
                row_heights=(2, 2),
                column_widths=(2, 1),
                seed=seed,
                source=(0, 1),
            )
        )
        output, invalid = broadcast_panel_lattice_periodic(
            partial_column_source,
            background=0,
            row_period=1,
            column_period=1,
        )
        self.assertIsNone(output)
        self.assertEqual(
            invalid,
            PanelLatticePeriodicCode.INCOMPATIBLE_PANEL_SHAPES,
        )

        empty = as_grid(
            _periodic_lattice(
                row_heights=(2, 2),
                column_widths=(2, 2),
                seed=[[0, 0], [0, 0]],
                source=(0, 0),
            )
        )
        output, invalid = broadcast_panel_lattice_periodic(
            empty,
            background=0,
            row_period=1,
            column_period=1,
        )
        self.assertIsNone(output)
        self.assertEqual(invalid, PanelLatticePeriodicCode.EMPTY_SELECTION)

        for invalid_period in (False, 0, -1, 31):
            with self.subTest(invalid_period=invalid_period), self.assertRaisesRegex(
                ValueError, "integer"
            ):
                broadcast_panel_lattice_periodic(
                    multiple,
                    background=0,
                    row_period=invalid_period,
                    column_period=1,
                )


if __name__ == "__main__":
    unittest.main()
