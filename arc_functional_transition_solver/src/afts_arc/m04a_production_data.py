"""Production M04a data binding with mandatory mmap-backed ReARC sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .m04a_data import (
    ArtifactSource,
    M04ATrainingData,
    SanitizedM04ABundle,
    load_sanitized_m04a_data,
)
from .m04a_rearc_index import MMapReARCExampleSource


@dataclass(slots=True)
class IndexedM04AData:
    sanitized_bundle: SanitizedM04ABundle
    training: M04ATrainingData
    validation: M04ATrainingData
    training_rearc: MMapReARCExampleSource
    validation_rearc: MMapReARCExampleSource

    def close(self) -> None:
        self.training_rearc.close()
        self.validation_rearc.close()

    def __enter__(self) -> "IndexedM04AData":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def load_indexed_m04a_data(
    *,
    sanitized_bundle_dir: str | Path,
    expected_sanitized_artifact_manifest_sha256: str,
    training_rearc_cache_dir: str | Path,
    expected_training_cache_artifact_manifest_sha256: str,
    validation_rearc_cache_dir: str | Path,
    expected_validation_cache_artifact_manifest_sha256: str,
    allow_nonproduction_fixture: bool = False,
    sanitized_artifact_sources: Mapping[str, ArtifactSource] | None = None,
    training_cache_artifact_sources: Mapping[str, ArtifactSource] | None = None,
    validation_cache_artifact_sources: Mapping[str, ArtifactSource] | None = None,
) -> IndexedM04AData:
    """Bind ARC2 shards to two externally committed, read-only ReARC mmaps."""

    if type(allow_nonproduction_fixture) is not bool:
        raise TypeError("allow_nonproduction_fixture must be bool")
    sanitized = load_sanitized_m04a_data(
        sanitized_bundle_dir,
        require_production_counts=not allow_nonproduction_fixture,
        expected_artifact_manifest_sha256=(
            expected_sanitized_artifact_manifest_sha256
        ),
        artifact_sources=sanitized_artifact_sources,
    )
    training_source: MMapReARCExampleSource | None = None
    validation_source: MMapReARCExampleSource | None = None
    try:
        training_source = MMapReARCExampleSource(
            training_rearc_cache_dir,
            expected_artifact_manifest_sha256=(
                expected_training_cache_artifact_manifest_sha256
            ),
            expected_sanitized_artifact_manifest_sha256=(
                expected_sanitized_artifact_manifest_sha256
            ),
            allow_nonproduction_fixture=allow_nonproduction_fixture,
            artifact_sources=training_cache_artifact_sources,
        )
        validation_source = MMapReARCExampleSource(
            validation_rearc_cache_dir,
            expected_artifact_manifest_sha256=(
                expected_validation_cache_artifact_manifest_sha256
            ),
            expected_sanitized_artifact_manifest_sha256=(
                expected_sanitized_artifact_manifest_sha256
            ),
            allow_nonproduction_fixture=allow_nonproduction_fixture,
            artifact_sources=validation_cache_artifact_sources,
        )
        if training_source.fold != "train" or validation_source.fold != "validation":
            raise ValueError("ReARC mmap cache folds are swapped or invalid")
        training = M04ATrainingData(
            arc2_parent_ids=sanitized.training.arc2_parent_ids,
            arc2_parents=sanitized.training.arc2_parents,
            rearc_parent_ids=training_source.parent_ids,
            rearc=training_source,
        )
        validation = M04ATrainingData(
            arc2_parent_ids=sanitized.validation.arc2_parent_ids,
            arc2_parents=sanitized.validation.arc2_parents,
            rearc_parent_ids=validation_source.parent_ids,
            rearc=validation_source,
        )
        return IndexedM04AData(
            sanitized_bundle=sanitized,
            training=training,
            validation=validation,
            training_rearc=training_source,
            validation_rearc=validation_source,
        )
    except Exception:
        if training_source is not None:
            training_source.close()
        if validation_source is not None:
            validation_source.close()
        raise


__all__ = ["IndexedM04AData", "load_indexed_m04a_data"]
