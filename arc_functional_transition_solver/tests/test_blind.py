from __future__ import annotations

import unittest

from afts_arc.blind import BlindTask
from afts_arc.grid import as_grid
from afts_arc.task import ARCPair, ARCTask


def _task(test_output: int) -> ARCTask:
    return ARCTask(
        task_id="blind",
        train=(ARCPair(as_grid([[0]]), as_grid([[1]])),),
        test=(ARCPair(as_grid([[2]]), as_grid([[test_output]])),),
        source_path="fixture",
        source_sha256="0" * 64,
    )


class BlindTaskTests(unittest.TestCase):
    def test_query_output_cannot_change_blind_task_or_serialization(self) -> None:
        first = BlindTask.from_task(_task(3))
        second = BlindTask.from_task(_task(9))
        self.assertEqual(first, second)
        serialized = first.to_json_dict()
        self.assertNotIn("output", serialized["test"][0])  # type: ignore[index]
        self.assertEqual(len(first.blind_content_sha256), 64)
        self.assertNotIn("source_path", serialized)
        self.assertNotIn("source_sha256", serialized)

    def test_ordered_identity_and_unordered_semantic_fingerprint_are_distinct(self) -> None:
        first_pair = ARCPair(as_grid([[0]]), as_grid([[1]]))
        second_pair = ARCPair(as_grid([[2]]), as_grid([[3]]))
        ordered = BlindTask.from_observations(
            train=(first_pair, second_pair), test_inputs=(as_grid([[4]]), as_grid([[5]]))
        )
        permuted = BlindTask.from_observations(
            train=(second_pair, first_pair), test_inputs=(as_grid([[5]]), as_grid([[4]]))
        )
        self.assertNotEqual(ordered.task_id, permuted.task_id)
        self.assertNotEqual(ordered.blind_content_sha256, permuted.blind_content_sha256)
        self.assertEqual(
            ordered.semantic_fingerprint_sha256,
            permuted.semantic_fingerprint_sha256,
        )

    def test_strict_round_trip_rejects_query_output_field(self) -> None:
        blind = BlindTask.from_task(_task(3))
        self.assertEqual(BlindTask.from_json_dict(blind.to_json_dict()), blind)
        poisoned = blind.to_json_dict()
        poisoned["test"][0]["output"] = [[9]]  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "no output"):
            BlindTask.from_json_dict(poisoned)


if __name__ == "__main__":
    unittest.main()
