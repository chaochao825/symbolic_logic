from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from afts_arc.m04a_production_data import load_indexed_m04a_data
from afts_arc.m04a_rearc_index import build_rearc_mmap_cache
from tests.test_m04a_data import _sanitized_fixture


class M04aProductionDataTests(unittest.TestCase):
    @unittest.skipUnless(
        os.name == "posix" and Path("/proc/self/fd").is_dir(),
        "descriptor-backed production data requires Linux /proc",
    )
    def test_descriptor_sources_are_the_only_bytes_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sanitized = _sanitized_fixture(root / "sanitized")
            sanitized_sha = hashlib.sha256(
                (sanitized / "artifact_manifest.json").read_bytes()
            ).hexdigest()
            train_result = build_rearc_mmap_cache(
                sanitized_split_dir=sanitized,
                fold="train",
                output_dir=root / "train-cache",
                expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                allow_nonproduction_fixture=True,
            )
            validation_result = build_rearc_mmap_cache(
                sanitized_split_dir=sanitized,
                fold="validation",
                output_dir=root / "validation-cache",
                expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                allow_nonproduction_fixture=True,
            )
            descriptors: list[int] = []

            def sources(path: Path) -> dict[str, int]:
                result: dict[str, int] = {}
                for artifact in path.iterdir():
                    descriptor = os.open(artifact, os.O_RDONLY)
                    descriptors.append(descriptor)
                    result[artifact.name] = descriptor
                return result

            try:
                sanitized_sources = sources(sanitized)
                training_sources = sources(root / "train-cache")
                validation_sources = sources(root / "validation-cache")
                with load_indexed_m04a_data(
                    sanitized_bundle_dir=sanitized,
                    expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                    training_rearc_cache_dir=root / "train-cache",
                    expected_training_cache_artifact_manifest_sha256=train_result[
                        "artifact_manifest_sha256"
                    ],
                    validation_rearc_cache_dir=root / "validation-cache",
                    expected_validation_cache_artifact_manifest_sha256=(
                        validation_result["artifact_manifest_sha256"]
                    ),
                    allow_nonproduction_fixture=True,
                    sanitized_artifact_sources=sanitized_sources,
                    training_cache_artifact_sources=training_sources,
                    validation_cache_artifact_sources=validation_sources,
                ) as indexed:
                    self.assertEqual(
                        indexed.training.rearc.example(
                            "cccccccc", 999
                        ).input_grid,
                        ((9,),),
                    )
            finally:
                for descriptor in descriptors:
                    os.close(descriptor)

    def test_fixture_binding_replaces_both_zip_sources_with_mmaps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sanitized = _sanitized_fixture(root / "sanitized")
            sanitized_sha = hashlib.sha256(
                (sanitized / "artifact_manifest.json").read_bytes()
            ).hexdigest()
            train_result = build_rearc_mmap_cache(
                sanitized_split_dir=sanitized,
                fold="train",
                output_dir=root / "train-cache",
                expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                allow_nonproduction_fixture=True,
            )
            validation_result = build_rearc_mmap_cache(
                sanitized_split_dir=sanitized,
                fold="validation",
                output_dir=root / "validation-cache",
                expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                allow_nonproduction_fixture=True,
            )
            with load_indexed_m04a_data(
                sanitized_bundle_dir=sanitized,
                expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                training_rearc_cache_dir=root / "train-cache",
                expected_training_cache_artifact_manifest_sha256=train_result[
                    "artifact_manifest_sha256"
                ],
                validation_rearc_cache_dir=root / "validation-cache",
                expected_validation_cache_artifact_manifest_sha256=validation_result[
                    "artifact_manifest_sha256"
                ],
                allow_nonproduction_fixture=True,
            ) as indexed:
                self.assertEqual(indexed.training.rearc_parent_ids, ("cccccccc",))
                self.assertEqual(indexed.validation.rearc_parent_ids, ("dddddddd",))
                self.assertEqual(
                    indexed.training.rearc.example("cccccccc", 999).input_grid,
                    ((9,),),
                )
            with self.assertRaises(RuntimeError):
                indexed.training.rearc.example("cccccccc", 0)

    def test_swapped_cache_folds_are_rejected_and_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sanitized = _sanitized_fixture(root / "sanitized")
            sanitized_sha = hashlib.sha256(
                (sanitized / "artifact_manifest.json").read_bytes()
            ).hexdigest()
            train_result = build_rearc_mmap_cache(
                sanitized_split_dir=sanitized,
                fold="train",
                output_dir=root / "train-cache",
                expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                allow_nonproduction_fixture=True,
            )
            validation_result = build_rearc_mmap_cache(
                sanitized_split_dir=sanitized,
                fold="validation",
                output_dir=root / "validation-cache",
                expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                allow_nonproduction_fixture=True,
            )
            with self.assertRaises(ValueError):
                load_indexed_m04a_data(
                    sanitized_bundle_dir=sanitized,
                    expected_sanitized_artifact_manifest_sha256=sanitized_sha,
                    training_rearc_cache_dir=root / "validation-cache",
                    expected_training_cache_artifact_manifest_sha256=validation_result[
                        "artifact_manifest_sha256"
                    ],
                    validation_rearc_cache_dir=root / "train-cache",
                    expected_validation_cache_artifact_manifest_sha256=train_result[
                        "artifact_manifest_sha256"
                    ],
                    allow_nonproduction_fixture=True,
                )


if __name__ == "__main__":
    unittest.main()
