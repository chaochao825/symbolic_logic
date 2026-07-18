from __future__ import annotations

import unittest

from afts_arc.grid import as_grid
from afts_arc.scoring import score_dataset, score_task
from afts_arc.task import ARCPair, ARCTask


def _task(task_id: str = "t") -> ARCTask:
    return ARCTask(
        task_id=task_id,
        train=(ARCPair(as_grid([[0]]), as_grid([[1]])),),
        test=(
            ARCPair(as_grid([[2]]), as_grid([[3]])),
            ARCPair(as_grid([[4]]), as_grid([[5]])),
        ),
        source_path="fixture",
        source_sha256="0" * 64,
    )


class ScoringTests(unittest.TestCase):
    def test_second_attempt_counts_for_pass_at_2(self) -> None:
        score = score_task(
            _task(),
            {
                0: [as_grid([[9]]), as_grid([[3]])],
                1: [as_grid([[5]])],
            },
        )
        self.assertEqual(score.official_pass_at_1, 0.5)
        self.assertEqual(score.official_pass_at_2, 1.0)
        self.assertFalse(score.strict_task_solved_at_1)
        self.assertTrue(score.strict_task_solved_at_2)

    def test_official_score_is_fraction_of_test_pairs(self) -> None:
        score = score_task(_task(), {0: [as_grid([[3]])], 1: [as_grid([[0]])]})
        self.assertEqual(score.official_pass_at_1, 0.5)
        self.assertEqual(score.official_pass_at_2, 0.5)
        self.assertFalse(score.strict_task_solved_at_2)

    def test_attempts_after_two_are_ignored(self) -> None:
        score = score_task(
            _task(),
            {
                0: [as_grid([[0]]), as_grid([[1]]), as_grid([[3]])],
                1: [as_grid([[5]])],
            },
        )
        self.assertEqual(score.official_pass_at_2, 0.5)
        self.assertEqual(score.pair_scores[0].attempts_available, 3)
        self.assertEqual(score.pair_scores[0].attempts_considered, 2)

    def test_wrong_shape_and_one_cell_mismatch_are_not_partial_credit(self) -> None:
        score = score_task(
            _task(),
            {0: [as_grid([[3, 3]])], 1: [as_grid([[4]])]},
        )
        self.assertEqual(score.official_pass_at_2, 0.0)

    def test_dataset_score_averages_task_fractions(self) -> None:
        first = ARCTask(
            task_id="a",
            train=(ARCPair(as_grid([[0]]), as_grid([[1]])),),
            test=(ARCPair(as_grid([[2]]), as_grid([[3]])),),
            source_path="fixture",
            source_sha256="0" * 64,
        )
        second = ARCTask(
            task_id="b",
            train=(ARCPair(as_grid([[0]]), as_grid([[1]])),),
            test=(
                ARCPair(as_grid([[2]]), as_grid([[3]])),
                ARCPair(as_grid([[4]]), as_grid([[5]])),
                ARCPair(as_grid([[6]]), as_grid([[7]])),
            ),
            source_path="fixture",
            source_sha256="0" * 64,
        )
        score = score_dataset(
            (first, second),
            {
                "a": {0: [as_grid([[3]])]},
                "b": {},
            },
        )
        self.assertEqual(score.official_pass_at_2, 0.5)
        self.assertEqual(score.strict_task_solved_rate_at_2, 0.5)

    def test_rejects_string_or_out_of_range_test_indices(self) -> None:
        with self.assertRaises(TypeError):
            score_task(_task(), {"0": [as_grid([[3]])]})  # type: ignore[dict-item]
        with self.assertRaises(ValueError):
            score_task(_task(), {2: [as_grid([[3]])]})

    def test_hidden_reference_cannot_be_scored(self) -> None:
        task = ARCTask(
            task_id="hidden",
            train=(ARCPair(as_grid([[0]]), as_grid([[1]])),),
            test=(ARCPair(as_grid([[2]]), None),),
            source_path="fixture",
            source_sha256="0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "no reference output"):
            score_task(task, {0: [as_grid([[3]])]})

    def test_dataset_rejects_unknown_submission_or_duplicate_task_id(self) -> None:
        task = _task("good")
        with self.assertRaisesRegex(ValueError, "unknown task IDs"):
            score_dataset((task,), {"goood": {0: [as_grid([[3]])]}})
        with self.assertRaisesRegex(ValueError, "duplicate task IDs"):
            score_dataset((task, task), {"good": {}})


if __name__ == "__main__":
    unittest.main()
