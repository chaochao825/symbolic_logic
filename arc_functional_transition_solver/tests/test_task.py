from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from afts_arc.scoring import score_task
from afts_arc.task import ARCPair, ARCTask, load_task, task_semantic_fingerprint


class TaskLoadingTests(unittest.TestCase):
    def _write(self, root: Path, payload: object) -> Path:
        path = root / "task42.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_loads_public_task_and_records_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                Path(directory),
                {
                    "train": [{"input": [[0]], "output": [[1]]}],
                    "test": [{"input": [[2]], "output": [[3]]}],
                },
            )
            task = load_task(path)
            self.assertEqual(task.task_id, "task42")
            self.assertEqual(task.train[0].output, ((1,),))
            self.assertEqual(task.test[0].output, ((3,),))
            self.assertEqual(len(task.source_sha256), 64)

    def test_allows_hidden_test_output_to_be_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                Path(directory),
                {
                    "train": [{"input": [[0]], "output": [[1]]}],
                    "test": [{"input": [[2]]}],
                },
            )
            task = load_task(path)
            self.assertIsNone(task.test[0].output)

    def test_requires_training_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                Path(directory),
                {"train": [{"input": [[0]]}], "test": [{"input": [[2]]}]},
            )
            with self.assertRaisesRegex(ValueError, "missing output"):
                load_task(path)

    def test_rejects_misspelled_or_unknown_pair_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                Path(directory),
                {
                    "train": [{"input": [[0]], "output": [[1]]}],
                    "test": [{"input": [[2]], "outpt": [[3]]}],
                },
            )
            with self.assertRaisesRegex(ValueError, "unknown fields"):
                load_task(path)

    def test_accepts_matching_legacy_name_and_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "name": "task42",
                "train": [{"input": [[0]], "output": [[1]]}],
                "test": [{"input": [[2]], "output": [[3]]}],
            }
            self.assertEqual(load_task(self._write(root, payload)).task_id, "task42")
            payload["name"] = "another-task"
            with self.assertRaisesRegex(ValueError, "must exactly match"):
                load_task(self._write(root, payload))

    def test_rejects_duplicate_json_object_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(
                '{"train":[{"input":[[0]],"input":[[9]],"output":[[1]]}],'
                '"test":[{"input":[[2]]}]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
                load_task(path)

    def test_public_constructors_restore_grid_and_sequence_invariants(self) -> None:
        train_pair = ARCPair([[0]], [[1]])  # type: ignore[arg-type]
        test_pair = ARCPair([[2]], [[3]])  # type: ignore[arg-type]
        task = ARCTask(
            task_id="direct",
            train=[train_pair],  # type: ignore[arg-type]
            test=[test_pair],  # type: ignore[arg-type]
            source_path="fixture",
            source_sha256="A" * 64,
        )
        self.assertIsInstance(task.train, tuple)
        self.assertIsInstance(task.test[0].output, tuple)
        self.assertEqual(task.source_sha256, "a" * 64)
        self.assertEqual(score_task(task, {0: [[[3]]]}).official_pass_at_1, 1.0)
        with self.assertRaisesRegex(TypeError, "task_id"):
            ARCTask(
                task_id=["bad"],  # type: ignore[arg-type]
                train=(train_pair,),
                test=(test_pair,),
                source_path="fixture",
                source_sha256="0" * 64,
            )

    def test_semantic_fingerprint_ignores_pair_order_but_preserves_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = self._write(
                root,
                {
                    "train": [
                        {"input": [[0]], "output": [[1]]},
                        {"input": [[2]], "output": [[3]]},
                    ],
                    "test": [
                        {"input": [[4]], "output": [[5]]},
                        {"input": [[6]], "output": [[7]]},
                    ],
                },
            )
            first = load_task(first_path)
            second_path = root / "reordered.json"
            second_path.write_text(
                json.dumps(
                    {
                        "train": [
                            {"input": [[2]], "output": [[3]]},
                            {"input": [[0]], "output": [[1]]},
                        ],
                        "test": [
                            {"input": [[6]], "output": [[7]]},
                            {"input": [[4]], "output": [[5]]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            reordered = load_task(second_path)
            self.assertEqual(
                task_semantic_fingerprint(first),
                task_semantic_fingerprint(reordered),
            )

            role_swapped_path = root / "role-swapped.json"
            role_swapped_path.write_text(
                json.dumps(
                    {
                        "train": [
                            {"input": [[4]], "output": [[5]]},
                            {"input": [[6]], "output": [[7]]},
                        ],
                        "test": [
                            {"input": [[0]], "output": [[1]]},
                            {"input": [[2]], "output": [[3]]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.assertNotEqual(
                task_semantic_fingerprint(first),
                task_semantic_fingerprint(load_task(role_swapped_path)),
            )


if __name__ == "__main__":
    unittest.main()
