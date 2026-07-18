from __future__ import annotations

import unittest

from afts_arc.contact_actions import BBoxContactCode, paint_bbox_contacts
from afts_arc.grid import as_grid


class ContactActionTests(unittest.TestCase):
    def test_paints_only_contacts_and_preserves_markers_paths_and_shape(self) -> None:
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
        output, invalid = paint_bbox_contacts(grid, background=0)
        self.assertIsNone(invalid)
        self.assertIsNotNone(output)
        self.assertEqual((len(output), len(output[0])), (len(grid), len(grid[0])))
        changed = {
            (row, column)
            for row in range(len(grid))
            for column in range(len(grid[0]))
            if output[row][column] != grid[row][column]
        }
        self.assertEqual(changed, {(2, 4), (3, 3), (4, 5)})
        self.assertEqual(output[0][4], 4)
        self.assertEqual(output[3][0], 2)
        self.assertEqual(output[4][7], 6)
        self.assertEqual(output[1][4], 0)
        self.assertEqual(output[3][1:3], (0, 0))

    def test_complete_hypothesis_executes_despite_invalid_competing_anchor(self) -> None:
        grid = as_grid(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 8, 8, 0, 3, 0, 7, 7, 0],
                [0, 8, 8, 0, 0, 0, 7, 7, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 4, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        output, invalid = paint_bbox_contacts(grid, background=0)
        self.assertIsNone(invalid)
        self.assertIsNotNone(output)
        self.assertEqual(output[2][2], 3)
        self.assertEqual(output[3][1], 4)
        self.assertEqual(output[2][6], 7)

    def test_invalid_precedence_is_fail_closed(self) -> None:
        cases = (
            (
                as_grid(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 8, 8, 0, 3, 0, 7, 7, 0],
                        [0, 8, 8, 0, 0, 0, 7, 7, 0],
                    ]
                ),
                BBoxContactCode.NON_UNIQUE_SELECTION,
            ),
            (
                as_grid(
                    [
                        [0, 0, 2, 0, 0],
                        [0, 0, 0, 0, 0],
                        [3, 0, 8, 8, 0],
                        [0, 0, 8, 8, 0],
                        [0, 0, 0, 0, 4],
                    ]
                ),
                BBoxContactCode.TARGET_COLLISION,
            ),
            (
                as_grid(
                    [
                        [0, 0, 2, 0, 0],
                        [0, 0, 3, 0, 0],
                        [0, 0, 0, 0, 4],
                        [0, 8, 8, 8, 0],
                        [0, 8, 8, 8, 0],
                    ]
                ),
                BBoxContactCode.OCCLUDED_RAY,
            ),
            (
                as_grid(
                    [
                        [0, 0, 2, 0, 0],
                        [0, 0, 0, 8, 8],
                        [0, 0, 0, 8, 8],
                    ]
                ),
                BBoxContactCode.INCOMPATIBLE_RELATION_GEOMETRY,
            ),
            (as_grid([[0, 1, 0]]), BBoxContactCode.EMPTY_SELECTION),
        )
        for grid, expected in cases:
            with self.subTest(expected=expected):
                output, invalid = paint_bbox_contacts(grid, background=0)
                self.assertIsNone(output)
                self.assertEqual(invalid, expected)

    def test_wrong_but_typed_background_fails_closed(self) -> None:
        grid = as_grid(
            [
                [0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 8, 8, 0, 0],
                [0, 8, 8, 0, 0],
            ]
        )
        output, invalid = paint_bbox_contacts(grid, background=8)
        self.assertIsNone(output)
        self.assertIsNotNone(invalid)


if __name__ == "__main__":
    unittest.main()
