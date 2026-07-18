from __future__ import annotations

import unittest

from afts_arc.grid import as_grid
from afts_arc.residual import compare_grids


class ResidualTests(unittest.TestCase):
    def test_exact_one_cell_and_shape_mismatch_are_distinct(self) -> None:
        target = as_grid([[1, 2]])
        exact = compare_grids(target, target, pair_index=0)
        one_cell = compare_grids(as_grid([[1, 3]]), target, pair_index=0)
        shape = compare_grids(as_grid([[1], [2]]), target, pair_index=0)
        invalid = compare_grids(None, target, pair_index=0, invalid_code="empty")
        self.assertTrue(exact.exact)
        self.assertEqual(one_cell.mismatch_cells, ((0, 1),))
        self.assertFalse(shape.shape_match)
        self.assertEqual(shape.mismatch_cells, ())
        self.assertFalse(invalid.execution_valid)
        self.assertEqual(invalid.invalid_code, "empty")


if __name__ == "__main__":
    unittest.main()
