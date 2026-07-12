"""Regression tests for the pixel-to-symbol-to-solver gridworld model."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurosymbolic_gridworld import (  # noqa: E402
    LearnedBinaryGate,
    _grid_reachable,
    extract_patches,
    make_dataset,
)


class EndToEndGridworldTests(unittest.TestCase):
    def test_generated_dataset_is_balanced_and_semantically_exact(self) -> None:
        dataset = make_dataset(20, 8, 4, 17, "clean")
        self.assertEqual(int(dataset.task_labels.sum()), 10)
        for labels, source, target, expected in zip(
            dataset.cell_labels,
            dataset.source_indices,
            dataset.target_indices,
            dataset.task_labels,
        ):
            self.assertEqual(_grid_reachable(labels, int(source), int(target)), bool(expected))

    def test_patch_extraction_preserves_all_cells(self) -> None:
        dataset = make_dataset(3, 8, 4, 21, "clean")
        patches = extract_patches(dataset.images, 8, 4)
        self.assertEqual(patches.shape, (3 * 8 * 8, 4 * 4 * 3))

    def test_learned_binary_gate_hardens_to_and(self) -> None:
        a = np.asarray([0.0, 0.0, 1.0, 1.0] * 64)
        b = np.asarray([0.0, 1.0, 0.0, 1.0] * 64)
        y = (a.astype(bool) & b.astype(bool)).astype(np.uint8)
        gate = LearnedBinaryGate(seed=4).fit(a, b, y, steps=500)
        self.assertEqual(gate.selected_op(), "AND")
        self.assertTrue(np.array_equal(gate.predict_hard(a, b).astype(np.uint8), y))


if __name__ == "__main__":
    unittest.main(verbosity=2)

