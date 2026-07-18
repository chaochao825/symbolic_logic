from __future__ import annotations

import unittest

from afts_arc.grid import (
    GridValidationError,
    as_grid,
    dihedral_variants,
    flip_horizontal,
    flip_vertical,
    rotate90,
    rotate180,
    rotate270,
    transpose,
)


class GridTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = as_grid([[1, 2, 3], [4, 5, 6]])

    def test_as_grid_converts_to_immutable_tuple(self) -> None:
        self.assertEqual(self.grid, ((1, 2, 3), (4, 5, 6)))

    def test_rejects_ragged_grid(self) -> None:
        with self.assertRaises(GridValidationError):
            as_grid([[1, 2], [3]])

    def test_rejects_boolean_and_out_of_range_cells(self) -> None:
        for invalid in ([[True]], [[-1]], [[10]], [[1.0]]):
            with self.subTest(invalid=invalid), self.assertRaises(GridValidationError):
                as_grid(invalid)

    def test_rejects_empty_or_oversized_grid(self) -> None:
        with self.assertRaises(GridValidationError):
            as_grid([])
        with self.assertRaises(GridValidationError):
            as_grid([[0] * 31])

    def test_d4_transform_orientation(self) -> None:
        self.assertEqual(rotate90(self.grid), ((4, 1), (5, 2), (6, 3)))
        self.assertEqual(rotate180(self.grid), ((6, 5, 4), (3, 2, 1)))
        self.assertEqual(rotate270(self.grid), ((3, 6), (2, 5), (1, 4)))
        self.assertEqual(flip_horizontal(self.grid), ((3, 2, 1), (6, 5, 4)))
        self.assertEqual(flip_vertical(self.grid), ((4, 5, 6), (1, 2, 3)))
        self.assertEqual(transpose(self.grid), ((1, 4), (2, 5), (3, 6)))

    def test_dihedral_variants_deduplicate_symmetric_inputs(self) -> None:
        variants = dihedral_variants(as_grid([[1, 1], [1, 1]]))
        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0][0], "identity")

    def test_non_square_grid_has_all_eight_oriented_d4_variants(self) -> None:
        variants = dict(dihedral_variants(self.grid))
        self.assertEqual(len(variants), 8)
        self.assertEqual(
            variants["anti_transpose"],
            ((6, 3), (5, 2), (4, 1)),
        )
        self.assertEqual(set(variants), {
            "identity",
            "rotate90",
            "rotate180",
            "rotate270",
            "flip_horizontal",
            "flip_vertical",
            "transpose",
            "anti_transpose",
        })


if __name__ == "__main__":
    unittest.main()
