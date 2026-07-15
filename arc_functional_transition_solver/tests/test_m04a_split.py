from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tempfile
import unittest
import warnings
import zipfile
from dataclasses import replace
from pathlib import Path

from afts_arc.m04a_split_privileged import (
    PrivilegedSplitSpec,
    _arc2_initial_folds,
    audit_rearc_archive_bytes,
    compile_privileged_split,
)
from afts_arc.task import parse_task_bytes


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise AssertionError(completed.stderr or completed.stdout)
    return completed.stdout.strip()


def _commit_repo(repo: Path, *, origin: str) -> str:
    _git(repo, "init")
    _git(repo, "config", "user.email", "m04a-fixture@example.invalid")
    _git(repo, "config", "user.name", "M04a Fixture")
    _git(repo, "remote", "add", "origin", origin)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    return _git(repo, "rev-parse", "HEAD")


def _grid(height: int, width: int, seed: int) -> list[list[int]]:
    return [[(seed + row * 3 + column * 5) % 10 for column in range(width)] for row in range(height)]


def _task_bytes(index: int) -> bytes:
    height = index + 1
    payload = {
        "train": [
            {
                "input": _grid(height, 1, index),
                "output": _grid(height, 2, index + 1),
            },
            {
                "input": _grid(height, 1, index + 2),
                "output": _grid(height, 2, index + 4),
            },
        ],
        "test": [
            {
                "input": _grid(height, 1, index + 6),
                "output": _grid(height, 2, index + 8),
            }
        ],
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _example(height: int, seed: int) -> dict[str, object]:
    return {
        "input": _grid(height, 1, seed),
        "output": _grid(height, 2, seed + 1),
    }


def _zip_bytes(materials: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in materials:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    return buffer.getvalue()


def _member_bytes(examples: list[dict[str, object]]) -> bytes:
    return json.dumps(examples, separators=(",", ":"), sort_keys=True).encode("utf-8")


class RearcArchiveAuditTests(unittest.TestCase):
    def _spec(self, *, examples: int = 1) -> PrivilegedSplitSpec:
        return PrivilegedSplitSpec(
            arc1_origin="https://example.invalid/arc1.git",
            arc1_commit="1" * 40,
            arc2_origin="https://example.invalid/arc2.git",
            arc2_commit="2" * 40,
            rearc_origin="https://example.invalid/rearc.git",
            rearc_commit="3" * 40,
            rearc_archive_sha256=None,
            rearc_archive_bytes=None,
            expected_zip_entry_count=None,
            arc1_task_count=1,
            arc2_task_count=1,
            rearc_examples_per_task=examples,
            holdout_count=0,
            validation_count=0,
            smoke_count=0,
            max_grid_side=3,
            expected_exact_semantic_quarantine_count=None,
            expected_aliases=None,
            expected_initial_rearc_counts=None,
            expected_validation_quarantine_ids=None,
            expected_training_orbit_quarantine_ids=None,
            expected_dimension_quarantine_ids=None,
            expected_final_counts=None,
        )

    def test_archive_schema_and_dimension_audit(self) -> None:
        examples = [_example(2, 0), _example(4, 1)]
        archive = _zip_bytes(
            [("re_arc/tasks/00000000.json", _member_bytes(examples))]
        )
        audit = audit_rearc_archive_bytes(
            archive,
            expected_parent_ids=("00000000",),
            spec=self._spec(examples=2),
        )
        self.assertEqual(audit.member_ids, ("00000000",))
        self.assertEqual(audit.member_audits[0].oversized_example_count, 1)
        self.assertEqual(audit.member_audits[0].oversized_grid_count, 2)
        self.assertEqual(audit.member_audits[0].maximum_height, 4)

    def test_archive_rejects_unsafe_duplicate_unknown_and_bad_json(self) -> None:
        valid = _member_bytes([_example(1, 0)])
        cases: list[tuple[str, bytes, str]] = []
        cases.append(
            (
                "path escape",
                _zip_bytes(
                    [
                        ("re_arc/tasks/00000000.json", valid),
                        ("../escape.json", b"{}"),
                    ]
                ),
                "unsafe",
            )
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            duplicate = _zip_bytes(
                [
                    ("re_arc/tasks/00000000.json", valid),
                    ("re_arc/tasks/00000000.json", valid),
                ]
            )
        cases.append(("duplicate member", duplicate, "duplicate"))
        cases.append(
            (
                "unknown member",
                _zip_bytes(
                    [
                        ("re_arc/tasks/00000000.json", valid),
                        ("notes.txt", b"unexpected"),
                    ]
                ),
                "unexpected",
            )
        )
        duplicate_key = (
            b'[{"input":[[0]],"input":[[1]],"output":[[0]]}]'
        )
        cases.append(
            (
                "duplicate JSON key",
                _zip_bytes(
                    [("re_arc/tasks/00000000.json", duplicate_key)]
                ),
                "duplicate JSON key",
            )
        )
        wrong_count = _member_bytes([_example(1, 0), _example(1, 1)])
        cases.append(
            (
                "wrong example count",
                _zip_bytes(
                    [("re_arc/tasks/00000000.json", wrong_count)]
                ),
                "exactly 1 examples",
            )
        )
        for label, archive, pattern in cases:
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, pattern):
                audit_rearc_archive_bytes(
                    archive,
                    expected_parent_ids=("00000000",),
                    spec=self._spec(),
                )


class PrivilegedSplitCompilerTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path, PrivilegedSplitSpec, dict[str, str]]:
        arc1 = root / "arc1"
        arc2 = root / "arc2"
        rearc = root / "rearc"
        for repo in (arc1, arc2, rearc):
            repo.mkdir()
        (arc1 / "data" / "training").mkdir(parents=True)
        (arc2 / "data" / "training").mkdir(parents=True)

        task_bytes: dict[str, bytes] = {}
        parsed = {}
        for index in range(8):
            task_id = f"{index:08x}"
            content = _task_bytes(index)
            task_bytes[task_id] = content
            parsed[task_id] = parse_task_bytes(
                content,
                source_path=arc2 / "data" / "training" / f"{task_id}.json",
                task_id=task_id,
            )
            (arc2 / "data" / "training" / f"{task_id}.json").write_bytes(content)
            (arc1 / "data" / "training" / f"{task_id}.json").write_bytes(content)

        provisional = PrivilegedSplitSpec(
            arc1_origin="https://example.invalid/arc1.git",
            arc1_commit="1" * 40,
            arc2_origin="https://example.invalid/arc2.git",
            arc2_commit="2" * 40,
            rearc_origin="https://example.invalid/rearc.git",
            rearc_commit="3" * 40,
            rearc_archive_sha256=None,
            rearc_archive_bytes=None,
            expected_zip_entry_count=8,
            arc1_task_count=8,
            arc2_task_count=8,
            rearc_examples_per_task=4,
            holdout_count=1,
            validation_count=2,
            smoke_count=1,
            max_grid_side=8,
            expected_exact_semantic_quarantine_count=0,
            expected_aliases=(),
            expected_initial_rearc_counts=None,
            expected_validation_quarantine_ids=None,
            expected_training_orbit_quarantine_ids=None,
            expected_dimension_quarantine_ids=None,
            expected_final_counts=None,
        )
        folds, _ = _arc2_initial_folds(parsed, spec=provisional)
        by_fold = {
            fold: sorted(task_id for task_id, assigned in folds.items() if assigned == fold)
            for fold in ("holdout", "validation", "smoke", "train")
        }
        protected = by_fold["holdout"][0]
        validation_bad, validation_good = by_fold["validation"]
        training_bad, training_dimension, *training_good = by_fold["train"]

        defaults = {
            task_id: [_example(index + 1, 20 + index + offset) for offset in range(4)]
            for index, task_id in enumerate(sorted(task_bytes))
        }
        defaults[validation_bad][0] = dict(defaults[protected][0])
        defaults[training_bad][0] = dict(defaults[validation_good][0])
        defaults[training_dimension][0] = _example(9, 91)
        archive = _zip_bytes(
            [
                (f"re_arc/tasks/{task_id}.json", _member_bytes(defaults[task_id]))
                for task_id in sorted(defaults)
            ]
        )
        (rearc / "re_arc.zip").write_bytes(archive)

        origins = {
            "arc1": "https://example.invalid/arc1.git",
            "arc2": "https://example.invalid/arc2.git",
            "rearc": "https://example.invalid/rearc.git",
        }
        commits = {
            "arc1": _commit_repo(arc1, origin=origins["arc1"]),
            "arc2": _commit_repo(arc2, origin=origins["arc2"]),
            "rearc": _commit_repo(rearc, origin=origins["rearc"]),
        }
        expected_initial = tuple(
            sorted((fold, len(ids)) for fold, ids in by_fold.items() if ids)
        )
        expected_final = tuple(
            sorted(
                {
                    "arc2_train": len(training_good),
                    "arc2_validation": 1,
                    "rearc_train": len(training_good),
                    "rearc_validation": 1,
                }.items()
            )
        )
        spec = replace(
            provisional,
            arc1_commit=commits["arc1"],
            arc2_commit=commits["arc2"],
            rearc_commit=commits["rearc"],
            rearc_archive_sha256=hashlib.sha256(archive).hexdigest(),
            rearc_archive_bytes=len(archive),
            expected_initial_rearc_counts=expected_initial,
            expected_validation_quarantine_ids=(validation_bad,),
            expected_training_orbit_quarantine_ids=(training_bad,),
            expected_dimension_quarantine_ids=(training_dimension,),
            expected_final_counts=expected_final,
        )
        identities = {
            "protected": protected,
            "validation_bad": validation_bad,
            "validation_good": validation_good,
            "training_bad": training_bad,
            "training_dimension": training_dimension,
            "training_good_0": training_good[0],
            "training_good_1": training_good[1],
        }
        return arc1, arc2, rearc, spec, identities

    def test_compiler_separates_sealed_evidence_and_cropped_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arc1, arc2, rearc, spec, ids = self._fixture(root)
            sealed = root / "sealed"
            sanitized = root / "sanitized"
            result = compile_privileged_split(
                arc1_repo=arc1,
                arc2_repo=arc2,
                rearc_repo=rearc,
                sealed_output_dir=sealed,
                sanitized_output_dir=sanitized,
                spec=spec,
            )
            self.assertEqual(result["status"], "published_privileged_and_sanitized_split_bundles")
            self.assertTrue((sealed / "collisions.jsonl").is_file())
            self.assertFalse((sanitized / "collisions.jsonl").exists())
            manifest = json.loads((sanitized / "data_split_manifest.json").read_text("utf-8"))
            self.assertTrue(manifest["protected_payloads_absent"])
            self.assertEqual(manifest["usable_fold_counts"]["arc2_train"], 2)
            self.assertEqual(manifest["usable_fold_counts"]["rearc_validation"], 1)
            self.assertEqual(
                manifest["training_quarantine_ids"],
                sorted((ids["training_bad"], ids["training_dimension"])),
            )
            sanitized_text = (sanitized / "data_split_manifest.json").read_text("utf-8")
            self.assertNotIn(ids["protected"], sanitized_text)
            protected_hash = hashlib.sha256(
                (arc2 / "data" / "training" / f"{ids['protected']}.json").read_bytes()
            ).hexdigest()
            self.assertNotIn(protected_hash, sanitized_text)

            expected_names = {
                "arc2_train.zip": {
                    f"arc2/train/{ids['training_good_0']}.json",
                    f"arc2/train/{ids['training_good_1']}.json",
                },
                "arc2_validation.zip": {
                    f"arc2/validation/{ids['validation_good']}.json"
                },
                "rearc_train.zip": {
                    f"rearc/train/{ids['training_good_0']}.json",
                    f"rearc/train/{ids['training_good_1']}.json",
                },
                "rearc_validation.zip": {
                    f"rearc/validation/{ids['validation_good']}.json"
                },
            }
            for artifact, names in expected_names.items():
                with self.subTest(artifact=artifact), zipfile.ZipFile(sanitized / artifact) as archive:
                    self.assertEqual(set(archive.namelist()), names)
                    for info in archive.infolist():
                        self.assertEqual(info.date_time, (1980, 1, 1, 0, 0, 0))
                        self.assertEqual(info.compress_type, zipfile.ZIP_DEFLATED)
                        self.assertEqual((info.external_attr >> 16) & 0o777, 0o644)
                        self.assertNotIn(ids["protected"], info.filename)

            with self.assertRaises(FileExistsError):
                compile_privileged_split(
                    arc1_repo=arc1,
                    arc2_repo=arc2,
                    rearc_repo=rearc,
                    sealed_output_dir=sealed,
                    sanitized_output_dir=root / "other-sanitized",
                    spec=spec,
                )

    def test_compiler_rejects_overlapping_outputs_and_wrong_pin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arc1, arc2, rearc, spec, _ = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "non-overlapping"):
                compile_privileged_split(
                    arc1_repo=arc1,
                    arc2_repo=arc2,
                    rearc_repo=rearc,
                    sealed_output_dir=root / "output",
                    sanitized_output_dir=root / "output" / "sanitized",
                    spec=spec,
                )
            with self.assertRaisesRegex(ValueError, "commit mismatch"):
                compile_privileged_split(
                    arc1_repo=arc1,
                    arc2_repo=arc2,
                    rearc_repo=rearc,
                    sealed_output_dir=root / "sealed-wrong",
                    sanitized_output_dir=root / "sanitized-wrong",
                    spec=replace(spec, arc1_commit="0" * 40),
                )
            self.assertFalse((root / "sealed-wrong").exists())
            self.assertFalse((root / "sanitized-wrong").exists())


if __name__ == "__main__":
    unittest.main()
