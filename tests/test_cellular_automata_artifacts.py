"""Integrity and claim-boundary checks for committed CA experiment artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))

from run_official_difflogic_gol import (  # noqa: E402
    NOTEBOOK_COMMIT,
    NOTEBOOK_SHA256,
    OFFICIAL_STEP_CELL_INDEX,
    OFFICIAL_STEP_CELL_SHA256,
    official_step,
)
from cellular_automata import game_of_life_next_from_patches  # noqa: E402


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_integer(value: str) -> int:
    """Parse integer-valued CSV fields that pandas may emit with .0."""
    return int(float(value))


class CellularAutomataArtifactTests(unittest.TestCase):
    def test_local_official_step_matches_all_512_gol_transitions(self) -> None:
        class IdentityJax:
            @staticmethod
            def jit(function):
                return function

        codes = np.arange(512, dtype=np.uint16)
        patches = ((codes[:, None] >> np.arange(9, dtype=np.uint16)) & 1).astype(np.uint8).reshape(-1, 3, 3)
        step = official_step(np, IdentityJax)
        predicted = np.asarray([step(patch)[1, 1] for patch in patches], dtype=np.uint8)
        self.assertTrue(np.array_equal(predicted, game_of_life_next_from_patches(patches)))

    def test_official_notebook_manifest_matches_training_harness(self) -> None:
        manifest = json.loads((ROOT / "third_party" / "difflogic_ca_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["notebook_source_commit"], NOTEBOOK_COMMIT)
        self.assertEqual(manifest["notebook_sha256"], NOTEBOOK_SHA256)
        self.assertEqual(manifest["notebook_git_blob"], "85dd4c1272fc01acf4ef9a70e7a0e80cddbb07a6")
        self.assertEqual(manifest["notebook_bytes"], 201717)
        self.assertEqual(manifest["notebook_gol_step_cell"], OFFICIAL_STEP_CELL_INDEX)
        self.assertEqual(manifest["notebook_gol_step_cell_sha256"], OFFICIAL_STEP_CELL_SHA256)

    def test_metadata_hashes_every_declared_artifact(self) -> None:
        metadata = json.loads((RESULTS / "cellular_automata_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["mode"], "full")
        self.assertEqual(metadata["suite"], "all")
        for filename, record in metadata["artifacts"].items():
            path = RESULTS / filename
            self.assertTrue(path.is_file(), filename)
            self.assertGreater(path.stat().st_size, 0, filename)
            self.assertEqual(path.stat().st_size, record["bytes"], filename)
            self.assertEqual(sha256(path), record["sha256"], filename)

    def test_full_run_started_clean_and_source_hashes_are_current(self) -> None:
        metadata = json.loads((RESULTS / "cellular_automata_metadata.json").read_text(encoding="utf-8"))
        self.assertFalse(metadata["started_from_git"]["dirty"])
        self.assertTrue(metadata["started_from_git"]["commit"])
        for relative_path, record in metadata["sources"].items():
            path = ROOT / relative_path
            self.assertTrue(path.is_file(), relative_path)
            self.assertEqual(path.stat().st_size, record["bytes"], relative_path)
            self.assertEqual(sha256(path), record["sha256"], relative_path)

    def test_all_eca_catalog_entries_are_exact(self) -> None:
        rows = [row for row in read_csv("cellular_automata_local_rule_results.csv") if row["task"] == "eca_catalog"]
        self.assertEqual(len(rows), 256)
        self.assertTrue(all(float(row["local_accuracy"]) == 1.0 for row in rows))
        self.assertLessEqual(max(int(float(row["gate_count"])) for row in rows), 4)

    def test_official_visible_semantics_are_exact_and_training_is_not_claimed(self) -> None:
        rows = read_csv("cellular_automata_official_results.csv")
        local = [row for row in rows if row["condition"] == "all_512_local_transitions"]
        self.assertEqual(len(local), 1)
        self.assertEqual(float(local[0]["local_accuracy"]), 1.0)
        self.assertTrue(all(float(row["channel0_accuracy"]) == 1.0 for row in rows))
        self.assertTrue(all(row["training_reproduction"] == "not_claimed" for row in rows))
        damage = [row for row in rows if row["condition"] == "persistent_damage_40_then_release_40"]
        self.assertEqual(len(damage), 3)
        self.assertTrue(all(csv_integer(row["steps"]) == 80 for row in damage))
        self.assertTrue(all(csv_integer(row["damage_steps"]) == 40 for row in damage))
        self.assertTrue(all(csv_integer(row["release_steps"]) == 40 for row in damage))
        self.assertTrue(all(csv_integer(row["damage_side"]) == 20 for row in damage))
        attempt = json.loads((RESULTS / "difflogic_ca_training_attempt.json").read_text(encoding="utf-8"))
        self.assertEqual(attempt["status"], "blocked_before_training")
        self.assertIsNone(attempt["measured_training_result"])

    def test_global_task_controls_expose_horizon_and_rule_quality(self) -> None:
        rows = read_csv("cellular_automata_task_results.csv")
        path_full = [row for row in rows if row["method"] == "BooleanWavefrontFull"]
        path_fixed = [row for row in rows if row["method"] == "FixedK16Wavefront"]
        self.assertTrue(path_full and path_fixed)
        self.assertTrue(all(float(row["accuracy"]) == 1.0 for row in path_full))
        self.assertLess(sum(float(row["accuracy"]) for row in path_fixed) / len(path_fixed), 1.0)
        density = [row for row in rows if row["task"] == "density_classification" and int(row["width"]) == 149]
        by_method = {}
        for method in {row["method"] for row in density}:
            values = [float(row["accuracy"]) for row in density if row["method"] == method]
            by_method[method] = sum(values) / len(values)
        self.assertGreater(by_method["particle"], by_method["majority"])
        sync = [row for row in rows if row["task"] == "global_synchronization" and int(row["width"]) == 149]
        self.assertTrue(all(csv_integer(row["steps"]) == 2 * csv_integer(row["width"]) + 1 for row in sync))
        self.assertTrue(
            all(csv_integer(row["decision_horizon"]) == 2 * csv_integer(row["width"]) for row in sync)
        )
        self.assertTrue(all(csv_integer(row["validation_steps"]) == 1 for row in sync))
        sync_means = {
            method: sum(float(row["accuracy"]) for row in sync if row["method"] == method)
            / sum(row["method"] == method for row in sync)
            for method in {row["method"] for row in sync}
        }
        self.assertGreater(sync_means["phi_sync"], sync_means["naive_oscillator"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
