from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from summarize_arc_difflogic_results import consolidate, validate_run  # noqa: E402


SOURCE_COMMIT = "4f54aa45d1e6328d96352028375718b7506d926a"
CONFIRM_RECEIPT_SHA256 = "0817bad7b056fc3ca39a9802c54637499373a8b31c5f9ce598b83eee0bd90e26"
RUN_ROOT = ROOT / "results" / "runs"
EXPECTED_RUNS = {
    "diffarc_formal_confirm_4f54aa4",
    "diffarc_formal_dev_core_4f54aa4",
    "diffarc_formal_probe_conflict_4f54aa4",
    "diffarc_formal_probe_positive_4f54aa4",
    "diffarc_formal_probe_representation_4f54aa4",
    "diffarc_formal_probe_shape_4f54aa4",
    "diffarc_formal_syn_context_4f54aa4",
    "diffarc_formal_syn_local_not_4f54aa4",
    "diffarc_formal_syn_object_4f54aa4",
    "diffarc_formal_syn_recurrence_4f54aa4",
    "diffarc_formal_syn_shape_4f54aa4",
}


def _rows(run: str, name: str) -> list[dict[str, str]]:
    with (RUN_ROOT / run / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _strict(row: dict[str, str]) -> float:
    eligible = row.get("deployment_eligible", "")
    if eligible == "":
        eligible = row.get("hard_demo_task_exact", "0")
    return float(row.get("hard_task_exact", "0") or 0) * float(eligible or 0)


class ArcDiffLogicArtifactTests(unittest.TestCase):
    def test_all_formal_runs_are_hash_valid_clean_and_source_bound(self) -> None:
        directories = [RUN_ROOT / name for name in sorted(EXPECTED_RUNS)]
        self.assertTrue(all(path.is_dir() for path in directories))
        for directory in directories:
            with self.subTest(run=directory.name):
                metadata, selected = validate_run(directory)
                self.assertEqual(metadata["mode"], "full")
                self.assertEqual(metadata["source_commit"], SOURCE_COMMIT)
                self.assertEqual(metadata["source_status_start"], "")
                self.assertEqual(metadata["source_status_end"], "")
                self.assertGreater(len(selected), 0)
                self.assertNotEqual(metadata.get("cohort"), "evaluation")
                for relative, record in metadata["source_files"].items():
                    payload = subprocess.run(
                        ("git", "show", f"{SOURCE_COMMIT}:{relative}"),
                        cwd=ROOT,
                        check=True,
                        stdout=subprocess.PIPE,
                    ).stdout
                    self.assertEqual(len(payload), int(record["bytes"]))
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), record["sha256"])

    def test_every_trained_difflogic_circuit_has_independent_export_equivalence(self) -> None:
        checked = 0
        for run in EXPECTED_RUNS:
            for row in _rows(run, "arc_difflogic_seed_results.csv"):
                if row.get("model_kind") == "difflogic" and row.get("status") == "trained":
                    checked += 1
                    self.assertEqual(float(row["hard_export_equivalent"]), 1.0)
                    self.assertTrue(row["hard_export_sha256"])
        self.assertGreater(checked, 100)

    def test_confirmatory_receipt_and_strict_result_are_immutable(self) -> None:
        run = "diffarc_formal_confirm_4f54aa4"
        metadata = json.loads((RUN_ROOT / run / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["confirmatory_config_sha256"], CONFIRM_RECEIPT_SHA256)
        receipt = metadata["confirmatory_config"]
        self.assertEqual(receipt["source_commit"], SOURCE_COMMIT)
        self.assertEqual(receipt["variants"], ["dl1", "dl1_wide"])
        self.assertEqual(receipt["seeds"], [0, 1, 2])
        self.assertEqual(len(receipt["task_ids"]), 12)
        self.assertIn("hard_demo_task_exact", receipt["analysis_plan"]["primary_metric"])

        rows = _rows(run, "arc_difflogic_selected_results.csv")
        by_variant = {}
        for row in rows:
            by_variant.setdefault(row["variant"], []).append(row)
        self.assertEqual(sum(_strict(row) for row in by_variant["dl1"]), 0.0)
        self.assertEqual(sum(_strict(row) for row in by_variant["dl1_wide"]), 0.0)
        self.assertEqual(sum(_strict(row) for row in by_variant["sparse"]), 0.0)
        self.assertEqual(sum(_strict(row) for row in by_variant["d4_sparse"]), 1.0)
        self.assertEqual(sum(_strict(row) for row in by_variant["d4_bgpad_sparse"]), 1.0)
        exact_d4 = {row["task_id"] for row in by_variant["d4_sparse"] if _strict(row) == 1.0}
        exact_padded = {row["task_id"] for row in by_variant["d4_bgpad_sparse"] if _strict(row) == 1.0}
        self.assertEqual(exact_d4, {"b60334d2"})
        self.assertEqual(exact_padded, exact_d4)

    def test_synthetic_mechanism_and_real_probe_boundaries(self) -> None:
        local = [
            row
            for row in _rows("diffarc_formal_syn_local_not_4f54aa4", "arc_difflogic_seed_results.csv")
            if row["variant"] == "dl1"
        ]
        self.assertEqual(len(local), 3)
        self.assertTrue(all(float(row["hard_demo_task_exact"]) == 1.0 for row in local))
        self.assertTrue(all(float(row["hard_task_exact"]) == 1.0 for row in local))

        recurrence = _rows("diffarc_formal_syn_recurrence_4f54aa4", "arc_difflogic_selected_results.csv")
        shift_dl1 = next(row for row in recurrence if row["task_id"] == "syn_hidden_shift2" and row["variant"] == "dl1")
        shift_dlr = next(row for row in recurrence if row["task_id"] == "syn_hidden_shift2" and row["variant"] == "dlr2")
        self.assertEqual(_strict(shift_dl1), 0.0)
        self.assertEqual(_strict(shift_dlr), 1.0)
        self.assertEqual(int(shift_dlr["selected_horizon"]), 2)
        self.assertEqual(int(float(shift_dlr["test_dynamic_gate_evaluations"])), 14040)

        positive = _rows("diffarc_formal_probe_positive_4f54aa4", "arc_difflogic_selected_results.csv")
        self.assertEqual(sum(_strict(row) for row in positive if row["variant"] == "sparse"), 4.0)
        self.assertEqual(sum(_strict(row) for row in positive if row["variant"] == "dl1"), 1.0)
        for run, variants in (
            ("diffarc_formal_probe_representation_4f54aa4", {"dlo", "dlf", "mlp"}),
            ("diffarc_formal_probe_conflict_4f54aa4", {"dlr", "dlo", "mlp"}),
            ("diffarc_formal_probe_shape_4f54aa4", {"dlf", "mlp"}),
        ):
            rows = _rows(run, "arc_difflogic_selected_results.csv")
            self.assertEqual(sum(_strict(row) for row in rows if row["variant"] in variants), 0.0)

    def test_consolidation_keeps_raw_hits_separate_from_deployable_hits(self) -> None:
        selected, aggregate = consolidate(RUN_ROOT / name for name in sorted(EXPECTED_RUNS))
        self.assertEqual(len(selected), 239)
        dev_dl1 = next(
            row
            for row in aggregate
            if row["suite"] == "arc" and row["cohort"] == "dev_low_mdl" and row["variant"] == "dl1"
        )
        self.assertEqual(dev_dl1["hard_exact_tasks"], 1)
        self.assertEqual(dev_dl1["strict_hard_exact_tasks"], 0)
        metadata = json.loads((ROOT / "results" / "arc_difflogic_v1_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["run_count"], 11)
        self.assertEqual(metadata["selected_rows"], 239)
        self.assertEqual(metadata["source_commits"], [SOURCE_COMMIT])
        for name, record in metadata["artifacts"].items():
            path = ROOT / "results" / name
            self.assertEqual(path.stat().st_size, int(record["bytes"]))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])


if __name__ == "__main__":
    unittest.main()
