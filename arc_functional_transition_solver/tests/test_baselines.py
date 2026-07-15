from __future__ import annotations

import unittest

from afts_arc.baselines import dihedral_candidates
from afts_arc.grid import as_grid
from afts_arc.task import ARCPair, ARCTask


class BaselineTests(unittest.TestCase):
    def test_d4_generator_is_deterministic_and_does_not_need_output(self) -> None:
        task = ARCTask(
            task_id="t",
            train=(ARCPair(as_grid([[0]]), as_grid([[1]])),),
            test=(ARCPair(as_grid([[1, 2], [3, 4]]), None),),
            source_path="fixture",
            source_sha256="0" * 64,
        )
        first = dihedral_candidates(task)
        second = dihedral_candidates(task)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(first[0].functional_trace, ("identity",))


if __name__ == "__main__":
    unittest.main()
