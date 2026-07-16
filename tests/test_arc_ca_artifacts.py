"""Fast runner/artifact contract checks for the ARC-CA extension."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_arc_ca import FROZEN_SPLITS, _complete_receipt, _reserve_receipt  # noqa: E402


class ArcCAArtifactTests(unittest.TestCase):
    def test_receipt_ledger_is_exclusive_and_append_only_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = root / "receipt.jsonl"
            output = root / "out"
            output.mkdir()
            (output / "metadata.json").write_text('{"ok":true}\n', encoding="utf-8")
            _reserve_receipt(ledger, {"status": "reserved_before_scoring"})
            with self.assertRaises(FileExistsError):
                _reserve_receipt(ledger, {"status": "duplicate"})
            _complete_receipt(ledger, output, "2026-07-16T00:00:00+00:00")
            records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["status"] for record in records], ["reserved_before_scoring", "completed"])
            self.assertEqual(len(records[1]["metadata_sha256"]), 64)

    def test_synthetic_runner_writes_nonempty_hashed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "src" / "run_arc_ca.py"),
                    "--suite",
                    "synthetic",
                    "--mode",
                    "smoke",
                    "--benchmark-repeats",
                    "1",
                    "--output-dir",
                    str(output),
                ],
                check=True,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["row_counts"]["arc_ca_synthetic_results.csv"], 13)
            for filename, artifact in metadata["artifacts"].items():
                self.assertGreater(artifact["bytes"], 0, filename)
                self.assertEqual(len(artifact["sha256"]), 64)
                self.assertEqual(
                    hashlib.sha256((output / filename).read_bytes()).hexdigest(),
                    artifact["sha256"],
                )
            with (output / "arc_ca_synthetic_results.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            by_task = {row["task"]: row for row in rows}
            self.assertEqual(by_task["global_majority_recolor"]["status"], "no_deterministic_local_rule")
            self.assertEqual(by_task["resize_crop"]["status"], "unsupported_shape_change")
            self.assertEqual(by_task["task_context_switch"]["level"], "L5")
            self.assertEqual(by_task["random_global_output"]["preferred_description_route"], "raw_global_labels")
            self.assertEqual(by_task["random_binary_lut5"]["preferred_description_route"], "raw_lut")

    def test_manifest_split_digests_match_runner_constants(self) -> None:
        manifest = json.loads((ROOT / "third_party" / "arc_agi_manifest.json").read_text(encoding="utf-8"))
        for (dataset, split), expected in FROZEN_SPLITS.items():
            declared = manifest["datasets"][dataset]["splits"][split]
            self.assertEqual(declared["content_digest"], expected["split_digest"])
            self.assertEqual(declared["task_id_digest"], expected["task_id_digest"])
            self.assertEqual(manifest["datasets"][dataset][f"{split}_tasks"], expected["count"])

    def test_arc2_evaluation_cannot_bypass_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "src" / "run_arc_ca.py"),
                    "--suite", "arc",
                    "--dataset-root", directory,
                    "--dataset-name", "ARC-AGI-2",
                    "--split", "evaluation",
                    "--mode", "full",
                    "--output-dir", str(Path(directory) / "out"),
                ],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("requires --public-evaluation-receipt", completed.stderr)

    def test_receipt_rejects_partial_task_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "src" / "run_arc_ca.py"),
                    "--suite", "arc",
                    "--dataset-root", directory,
                    "--dataset-name", "ARC-AGI-2",
                    "--split", "evaluation",
                    "--mode", "full",
                    "--task-limit", "1",
                    "--public-evaluation-receipt",
                    "--receipt-ledger", str(Path(directory) / "ledger.jsonl"),
                    "--output-dir", str(Path(directory) / "out"),
                ],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("forbids a task limit", completed.stderr)


if __name__ == "__main__":
    unittest.main()
