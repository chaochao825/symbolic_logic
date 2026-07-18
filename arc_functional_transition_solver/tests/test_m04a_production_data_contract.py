from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

from afts_arc import m04a_production_data_contract as contract


def _publication_marks_missing_paths_manifest_only(
    repository: Path, paths: tuple[Path, ...]
) -> bool:
    publication_manifest = repository / "PUBLICATION_MANIFEST.json"
    if not publication_manifest.is_file() or not paths:
        return False
    manifest = json.loads(publication_manifest.read_text(encoding="utf-8"))
    entries = {entry.get("path"): entry for entry in manifest.get("entries", ())}
    return all(
        entries.get(path.relative_to(repository).as_posix(), {}).get("disposition")
        == "manifest_only"
        for path in paths
    )


class M04AProductionDataContractTests(unittest.TestCase):
    @unittest.skipUnless(
        os.name == "posix" and hasattr(os, "memfd_create"),
        "sealed production-data bindings require Linux memfd",
    )
    def test_sealed_descriptor_binding_revalidates_without_paths(self) -> None:
        import fcntl

        content = b"sealed-data"
        target_fd = 710
        try:
            os.fstat(target_fd)
        except OSError:
            pass
        else:
            self.skipTest("fixed descriptor selected for the test is occupied")
        source_fd = os.memfd_create(
            "afts-m04a-data-fixture",
            os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
        )
        try:
            os.write(source_fd, content)
            required_seals = (
                fcntl.F_SEAL_SEAL
                | fcntl.F_SEAL_SHRINK
                | fcntl.F_SEAL_GROW
                | fcntl.F_SEAL_WRITE
            )
            fcntl.fcntl(source_fd, fcntl.F_ADD_SEALS, required_seals)
            os.dup2(source_fd, target_fd, inheritable=True)
            tiny_files = MappingProxyType(
                {
                    "sanitized_split/artifact_manifest.json": (
                        len(content),
                        hashlib.sha256(content).hexdigest(),
                    )
                }
            )
            with (
                tempfile.TemporaryDirectory() as temporary_directory,
                patch.object(contract, "PRODUCTION_DATA_FILES", tiny_files),
                patch.object(contract, "PRODUCTION_DATA_FD_BASE", target_fd),
            ):
                binding = contract.validate_sealed_production_data_snapshot(
                    temporary_directory
                )
                self.assertTrue(binding.is_sealed_snapshot)
                self.assertEqual(
                    binding.artifact_source(
                        "sanitized_split/artifact_manifest.json"
                    ),
                    target_fd,
                )
                self.assertIs(
                    contract.revalidate_production_data_binding(binding), binding
                )
                with self.assertRaises(OSError):
                    os.pwrite(target_fd, b"x", 0)
        finally:
            try:
                os.close(target_fd)
            except OSError:
                pass
            os.close(source_fd)

    def test_frozen_manifest_is_canonical_and_matches_repository_data(self) -> None:
        payload = contract.production_data_manifest_payload()
        semantic = dict(payload)
        closure_id = semantic.pop("closure_id")
        self.assertEqual(closure_id, contract.PRODUCTION_DATA_CLOSURE_ID)
        self.assertEqual(
            closure_id,
            contract.canonical_sha256(semantic),
        )
        self.assertEqual(payload["file_count"], 14)
        self.assertEqual(
            payload["total_bytes"],
            sum(row[0] for row in contract.PRODUCTION_DATA_FILES.values()),
        )
        repository = Path(__file__).resolve().parents[1]
        root = repository / "results" / "m04a_global_source_v0_1"
        if not root.is_dir():
            self.skipTest("canonical production data fixtures are not present")
        missing_paths = tuple(
            root.joinpath(*relative.split("/"))
            for relative in contract.PRODUCTION_DATA_FILES
            if not root.joinpath(*relative.split("/")).is_file()
        )
        if _publication_marks_missing_paths_manifest_only(repository, missing_paths):
            self.skipTest(
                "canonical production-data payloads are manifest-only in this publication"
            )
        for relative, (expected_bytes, expected_sha256) in (
            contract.PRODUCTION_DATA_FILES.items()
        ):
            path = root.joinpath(*relative.split("/"))
            self.assertEqual(path.stat().st_size, expected_bytes, relative)
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                expected_sha256,
                relative,
            )

    def test_tiny_closed_world_rejects_extra_or_changed_file(self) -> None:
        content = b"frozen-data"
        tiny_files = MappingProxyType(
            {
                "sanitized_split/artifact_manifest.json": (
                    len(content),
                    hashlib.sha256(content).hexdigest(),
                )
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for name in (
                contract.SANITIZED_DIRECTORY,
                contract.TRAINING_CACHE_DIRECTORY,
                contract.VALIDATION_CACHE_DIRECTORY,
            ):
                (root / name).mkdir()
            target = root / "sanitized_split" / "artifact_manifest.json"
            target.write_bytes(content)
            with patch.object(contract, "PRODUCTION_DATA_FILES", tiny_files):
                binding = contract.validate_production_data_root(root)
                self.assertEqual(binding.root, root)

                (root / "rearc_train_cache" / "extra.bin").write_bytes(b"extra")
                with self.assertRaisesRegex(ValueError, "closed world"):
                    contract.validate_production_data_root(root)

                (root / "rearc_train_cache" / "extra.bin").unlink()
                target.write_bytes(b"changed-data")
                with self.assertRaisesRegex(ValueError, "byte count"):
                    contract.validate_production_data_root(root)


if __name__ == "__main__":
    unittest.main()
