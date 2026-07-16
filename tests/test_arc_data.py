"""Strict ARC loader and label-separation regressions."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc_data import load_arc_split, parse_arc_task, split_digest, task_id_digest  # noqa: E402


class ArcDataTests(unittest.TestCase):
    def test_solver_problem_does_not_contain_test_outputs(self) -> None:
        document = {
            "train": [{"input": [[0, 1]], "output": [[1, 0]]}],
            "test": [{"input": [[2, 3]], "output": [[3, 2]]}],
        }
        problem, labels = parse_arc_task(document, "fixture")
        self.assertEqual(problem.task_id, "fixture")
        self.assertFalse(hasattr(problem, "test_outputs"))
        self.assertTrue(np.array_equal(problem.test_inputs[0], np.asarray([[2, 3]])))
        self.assertTrue(np.array_equal(labels.test_outputs[0], np.asarray([[3, 2]])))
        self.assertFalse(problem.test_inputs[0].flags.writeable)

    def test_missing_test_output_is_kept_as_unscored_label(self) -> None:
        document = {
            "train": [{"input": [[0]], "output": [[0]]}],
            "test": [{"input": [[1]]}],
        }
        _, labels = parse_arc_task(document, "hidden")
        self.assertIsNone(labels.test_outputs[0])

    def test_rejects_ragged_noninteger_and_invalid_color_grids(self) -> None:
        bad_grids = ([[0], [1, 2]], [[0.5]], [[10]])
        for grid in bad_grids:
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                parse_arc_task(
                    {"train": [{"input": grid, "output": [[0]]}], "test": [{"input": [[0]]}]},
                    "bad",
                )

    def test_split_loading_is_sorted_limited_and_content_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "data" / "training"
            split.mkdir(parents=True)
            payload = {"train": [{"input": [[0]], "output": [[0]]}], "test": [{"input": [[0]], "output": [[0]]}]}
            for name in ("b", "a"):
                (split / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_arc_split(root, "training", limit=1)
            self.assertEqual(loaded[0][0].task_id, "a")
            first = split_digest(path for _, _, path in loaded)
            loaded_again = load_arc_split(root, "training", task_ids=["a"])
            self.assertEqual(first, split_digest(path for _, _, path in loaded_again))
            self.assertEqual(
                task_id_digest(path for _, _, path in loaded),
                task_id_digest(path for _, _, path in loaded_again),
            )
            self.assertNotEqual(
                task_id_digest(path for _, _, path in load_arc_split(root, "training")),
                task_id_digest(path for _, _, path in loaded_again),
            )


if __name__ == "__main__":
    unittest.main()
