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
    FROZEN_COMMIT = "888573d15de3c865e339677df8d2e804e999ff88"
    COMMITTED_RUNS = {
        "arc_ca_v1_arc2_training_full_210_888573d": (1000, 13, 253, 7, 253),
        "arc_ca_v1_arc2_evaluation_full_210_888573d": (120, 0, 18, 0, 18),
        "arc_ca_v1_arc1_training_full_210_888573d": (400, 12, 114, 6, 114),
        "arc_ca_v1_arc1_evaluation_full_210_888573d": (400, 1, 79, 1, 79),
    }

    @staticmethod
    def _csv_rows(path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_committed_full_runs_are_hash_valid_and_keep_full_denominators(self) -> None:
        runs = ROOT / "results" / "runs"
        compiled_arc_rows = 0
        compiled_equivalent = 0
        for name, expected in self.COMMITTED_RUNS.items():
            with self.subTest(run=name):
                directory = runs / name
                metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["source_git_at_start"]["commit"], self.FROZEN_COMMIT)
                self.assertFalse(metadata["source_git_at_start"]["dirty"])
                self.assertFalse(metadata["source_git_at_end"]["dirty"])
                self.assertEqual(metadata["configuration"]["compile_policy"], "all_selected_rules")
                for filename, artifact in metadata["artifacts"].items():
                    path = directory / filename
                    self.assertEqual(path.stat().st_size, artifact["bytes"])
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"])
                for relative, source in metadata["sources"].items():
                    path = ROOT / relative
                    self.assertEqual(path.stat().st_size, source["bytes"])
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), source["sha256"])

                tasks = self._csv_rows(directory / "arc_ca_task_results.csv")
                programs = self._csv_rows(directory / "arc_ca_program_results.csv")
                codecs = self._csv_rows(directory / "arc_ca_codec_results.csv")
                task_count, exact_count, selected_count, program_exact, arc_codec_count = expected
                self.assertEqual(metadata["dataset"]["task_count"], task_count)
                self.assertEqual(len(tasks), task_count)
                self.assertEqual(sum(float(row["task_exact"]) == 1.0 for row in tasks), exact_count)
                self.assertEqual(sum(row["selection_status"] == "selected" for row in tasks), selected_count)
                self.assertEqual(sum(float(row["task_exact"]) == 1.0 for row in programs), program_exact)
                arc_codecs = [row for row in codecs if row.get("dataset")]
                self.assertEqual(len(arc_codecs), arc_codec_count)
                compiled_arc_rows += len(arc_codecs)
                compiled_equivalent += sum(float(row["compiled_equivalent"]) == 1.0 for row in arc_codecs)
                self.assertEqual(
                    sum(1 for line in (directory / "arc_ca_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line),
                    task_count,
                )
        self.assertEqual(compiled_arc_rows, 464)
        self.assertEqual(compiled_equivalent, 464)

    def test_public_evaluation_receipt_binds_completed_metadata(self) -> None:
        runs = ROOT / "results" / "runs"
        directory = runs / "arc_ca_v1_arc2_evaluation_full_210_888573d"
        ledger = runs / "arc_ca_v1_arc2_evaluation_full_210_888573d.receipt.jsonl"
        records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([record["status"] for record in records], ["reserved_before_scoring", "completed"])
        self.assertEqual(records[0]["source_commit"], self.FROZEN_COMMIT)
        self.assertEqual(records[0]["task_id_digest"], FROZEN_SPLITS[("ARC-AGI-2", "evaluation")]["task_id_digest"])
        self.assertEqual(
            records[1]["metadata_sha256"],
            hashlib.sha256((directory / "metadata.json").read_bytes()).hexdigest(),
        )

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
