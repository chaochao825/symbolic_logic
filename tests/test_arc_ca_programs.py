"""Bounded categorical CA program-search regressions."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc_ca_programs import LocalCAProgram, select_demo_program  # noqa: E402
from arc_data import ArcExample  # noqa: E402


class ArcCAProgramTests(unittest.TestCase):
    def test_program_api_rejects_fractional_colors(self) -> None:
        with self.assertRaises(ValueError):
            LocalCAProgram("identity", (), 4, 0).run(np.asarray([[0.5]]))

    def test_copy_wire_generalizes_unseen_colors(self) -> None:
        source = np.asarray([[0, 1, 0], [0, 2, 0], [0, 3, 0]], dtype=np.uint8)
        target = source.copy()
        target[1:] = source[:-1]
        selected = select_demo_program((ArcExample(source, target),))
        self.assertEqual(selected.status, "selected")
        self.assertEqual(selected.program.kind, "copy_offset")
        unseen = np.asarray([[4, 5], [6, 7]], dtype=np.uint8)
        predicted, _, _ = selected.program.run(unseen)
        self.assertTrue(np.array_equal(predicted[1], unseen[0]))

    def test_iterative_propagation_is_selected_from_demonstrations(self) -> None:
        source = np.zeros((7, 7), dtype=np.uint8)
        source[3, 3] = 2
        program = LocalCAProgram("propagate", (2, 0, 2, 4, 3), 21, 35)
        target, _, _ = program.run(source)
        selected = select_demo_program((ArcExample(source, target),))
        self.assertEqual(selected.status, "selected")
        predicted, _, _ = selected.program.run(source)
        self.assertTrue(np.array_equal(predicted, target))

    def test_integer_horizon_executes_declared_updates_without_hidden_halt_test(self) -> None:
        stable = np.zeros((3, 3), dtype=np.uint8)
        program = LocalCAProgram("propagate", (2, 0, 2, 4, 8), 21, 35)
        terminal, steps, reason = program.run(stable)
        self.assertTrue(np.array_equal(terminal, stable))
        self.assertEqual(steps, 8)
        self.assertEqual(reason, "max_steps")

    def test_program_library_rejects_multiple_changed_color_pairs(self) -> None:
        source = np.asarray([[0, 1]], dtype=np.uint8)
        target = np.asarray([[2, 3]], dtype=np.uint8)
        selected = select_demo_program((ArcExample(source, target),))
        self.assertEqual(selected.status, "no_exact_program")


if __name__ == "__main__":
    unittest.main()
