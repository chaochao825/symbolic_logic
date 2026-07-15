from __future__ import annotations

import json
import hashlib
import math
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from afts_arc.manifest import (
    audit_task_directory,
    capture_runtime_source,
    publish_evidence_bundle,
    snapshot_task_directory,
    write_json_new,
    verify_source_snapshot_zip,
)


class ManifestTests(unittest.TestCase):
    def test_audit_validates_and_fingerprints_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.json").write_text(
                json.dumps(
                    {
                        "train": [{"input": [[0]], "output": [[1]]}],
                        "test": [
                            {"input": [[2]], "output": [[3]]},
                            {"input": [[4]], "output": [[5]]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            audit = audit_task_directory(
                root, dataset_name="fixture", split_name="test", validate=True
            )
            self.assertEqual(audit.file_count, 1)
            self.assertEqual(audit.total_train_pairs, 1)
            self.assertEqual(audit.total_test_pairs, 2)
            self.assertEqual(audit.multi_test_task_count, 1)
            self.assertEqual(audit.test_outputs_present, 2)
            self.assertEqual(len(audit.file_manifest_sha256), 64)

    def test_evidence_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "evidence.json"
            write_json_new(target, {"x": 1})
            with self.assertRaises(FileExistsError):
                write_json_new(target, {"x": 2})

    def test_json_serialization_failure_leaves_no_partial_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "evidence.json"
            for invalid in ({"x": math.nan}, {"x": object()}):
                with self.subTest(invalid=invalid), self.assertRaises(
                    (TypeError, ValueError)
                ):
                    write_json_new(target, invalid)
                self.assertFalse(target.exists())

    def test_max_tasks_snapshots_only_the_selected_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.json").write_text(
                json.dumps(
                    {
                        "train": [{"input": [[0]], "output": [[1]]}],
                        "test": [{"input": [[2]], "output": [[3]]}],
                    }
                ),
                encoding="utf-8",
            )
            (root / "z.json").write_text("not-json", encoding="utf-8")
            snapshot = snapshot_task_directory(
                root,
                dataset_name="fixture",
                split_name="training",
                max_tasks=1,
            )
            self.assertEqual(snapshot.selected_file_names, ("a.json",))
            self.assertEqual(snapshot.audit.available_file_count, 2)
            self.assertEqual(snapshot.audit.file_count, 1)
            self.assertEqual(tuple(task.task_id for task in snapshot.tasks), ("a",))

    def test_atomic_bundle_records_hashes_rows_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run"
            artifacts = {
                "summary.json": b'{"ok":true}\n',
                "tasks.jsonl": b'{"id":1}\n{"id":2}\n',
                "source_snapshot.zip": b"zip-bytes",
            }
            manifest = publish_evidence_bundle(
                target,
                artifacts=artifacts,
                run_id="run-123",
            )
            self.assertEqual(manifest["bundle_status"], "complete")
            self.assertEqual(manifest["artifacts"]["tasks.jsonl"]["rows"], 2)
            for name, content in artifacts.items():
                self.assertEqual((target / name).read_bytes(), content)
                self.assertEqual(
                    manifest["artifacts"][name]["sha256"],
                    hashlib.sha256(content).hexdigest(),
                )
            on_disk = json.loads((target / "artifact_manifest.json").read_text("utf-8"))
            self.assertEqual(on_disk, manifest)
            with self.assertRaises(FileExistsError):
                publish_evidence_bundle(target, artifacts=artifacts, run_id="other")

    def test_failed_bundle_never_publishes_final_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for index, unsafe in enumerate(
                (
                    "../escape",
                    "/escape",
                    "C:escape.bin",
                    "file:stream",
                    "a\\b",
                    "NUL",
                    "aux.txt",
                    "dir/COM1",
                )
            ):
                target = Path(directory) / f"run-{index}"
                with self.subTest(unsafe=unsafe), self.assertRaisesRegex(
                    RuntimeError, "preserved at"
                ):
                    publish_evidence_bundle(
                        target,
                        artifacts={unsafe: b"bad"},
                        run_id="run-123",
                    )
                self.assertFalse(target.exists())

    def test_runtime_source_fingerprint_and_snapshot_cover_executing_package(self) -> None:
        capture = capture_runtime_source()
        fingerprint = capture.fingerprint_sha256
        self.assertEqual(len(fingerprint), 64)
        self.assertNotEqual(fingerprint, hashlib.sha256(b"").hexdigest())
        snapshot = capture.snapshot_zip
        with zipfile.ZipFile(BytesIO(snapshot)) as archive:
            names = set(archive.namelist())
            self.assertIn("afts_arc/cli.py", names)
            self.assertIn("afts_arc/manifest.py", names)
            digest = hashlib.sha256()
            for name in sorted(names):
                content = archive.read(name)
                encoded_name = name.encode("utf-8")
                digest.update(len(encoded_name).to_bytes(8, "big"))
                digest.update(encoded_name)
                digest.update(len(content).to_bytes(8, "big"))
                digest.update(content)
            self.assertEqual(digest.hexdigest(), fingerprint)
        verified = verify_source_snapshot_zip(
            snapshot, expected_fingerprint_sha256=fingerprint
        )
        self.assertIn("afts_arc/_source_bootstrap.py", verified)
        with self.assertRaisesRegex(ValueError, "ZIP"):
            verify_source_snapshot_zip(
                b"not-a-zip", expected_fingerprint_sha256=fingerprint
            )

    def test_runtime_capture_reads_materials_once(self) -> None:
        materials = {"afts_arc/example.py": b"print('fixed')\n"}
        with patch(
            "afts_arc.manifest.runtime_materials", return_value=materials
        ) as material_reader:
            capture = capture_runtime_source()
        material_reader.assert_called_once_with()
        with zipfile.ZipFile(BytesIO(capture.snapshot_zip)) as archive:
            self.assertEqual(archive.read("afts_arc/example.py"), materials["afts_arc/example.py"])

if __name__ == "__main__":
    unittest.main()
