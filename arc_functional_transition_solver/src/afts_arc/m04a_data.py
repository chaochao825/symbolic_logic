"""Oracle-isolated M04a episode construction and validation manifests.

This module is deliberately PyTorch-free.  It consumes only sanitized parent
objects or a narrow ReARC example source and turns the frozen counter-based data
contract into immutable episodes.  Neural code must not reinterpret sampling,
augmentation, or masking rules.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import stat
import zipfile
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from contextlib import contextmanager
from typing import BinaryIO, Iterator, Protocol

from .authority import DATASET_COMMITS, DATASET_ORIGINS
from .grid import Grid, as_grid, grid_to_lists
from .m04a_contract import (
    DATA_FOLDS_SEMANTICS_VERSION,
    GRADIENT_ACCUMULATION,
    MAX_GRID_SIDE,
    VALIDATION_SEMANTICS_VERSION,
    apply_indexed_d4,
    canonical_sha256,
    episode_seed,
    validation_episode_seed,
)
from .task import parse_task_bytes


ARC2_SOURCE = "arc2"
REARC_SOURCE = "rearc"
VALIDATION_MASK_FRACTIONS: tuple[tuple[int, int, str], ...] = (
    (15, 100, "0.15"),
    (35, 100, "0.35"),
    (65, 100, "0.65"),
    (1, 1, "1.0"),
)
SANITIZED_MANIFEST_SCHEMA = "afts.m04a-sanitized-split/v1"
_SANITIZED_FILES = frozenset(
    {
        "arc2_train.zip",
        "arc2_validation.zip",
        "rearc_train.zip",
        "rearc_validation.zip",
        "data_split_manifest.json",
        "artifact_manifest.json",
    }
)
_PRODUCTION_COUNTS = {
    "arc2_train": 662,
    "arc2_validation": 145,
    "rearc_train": 262,
    "rearc_validation": 51,
}
_PRODUCTION_SOURCE_COMMITMENTS = {
    "arc1": {
        "origin": DATASET_ORIGINS["ARC-AGI-1"],
        "commit": DATASET_COMMITS["ARC-AGI-1"],
    },
    "arc2": {
        "origin": DATASET_ORIGINS["ARC-AGI-2"],
        "commit": DATASET_COMMITS["ARC-AGI-2"],
    },
    "rearc": {
        "origin": "https://github.com/michaelhodel/re-arc.git",
        "commit": "e5b7f1d06362a76f9d3b8c25154ff1fafca897ce",
        "archive_sha256": "4a7c309499f450eb47c2117fa06f3c97fe741cf6e7912579c0b5b8cc759e9ac4",
        "archive_bytes": 50_913_541,
    },
}


def _strict_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field} must be a non-empty string")
    return value


def _grid(value: object, *, field: str) -> Grid:
    try:
        return as_grid(value, max_height=MAX_GRID_SIDE, max_width=MAX_GRID_SIDE)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class M04AExample:
    input_grid: Grid
    output_grid: Grid

    @classmethod
    def create(cls, input_grid: object, output_grid: object) -> "M04AExample":
        return cls(
            input_grid=_grid(input_grid, field="example input"),
            output_grid=_grid(output_grid, field="example output"),
        )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "input_grid", _grid(self.input_grid, field="example input")
        )
        object.__setattr__(
            self, "output_grid", _grid(self.output_grid, field="example output")
        )


@dataclass(frozen=True, slots=True)
class ARC2Parent:
    parent_id: str
    train: tuple[M04AExample, ...]
    test: tuple[M04AExample, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parent_id", _strict_id(self.parent_id, field="parent_id"))
        train = tuple(self.train)
        test = tuple(self.test)
        if not train or not test:
            raise ValueError("ARC2 parent requires non-empty train and test examples")
        if len(train) > 10:
            raise ValueError("M04a does not support more than ten demonstrations")
        if any(not isinstance(example, M04AExample) for example in (*train, *test)):
            raise TypeError("ARC2 parent examples must be M04AExample values")
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "test", test)


class ReARCExampleSource(Protocol):
    """Narrow training-safe interface; it cannot expose original/protected sources."""

    @property
    def parent_ids(self) -> tuple[str, ...]: ...

    def example(self, parent_id: str, example_index: int) -> M04AExample: ...

    def semantic_parent_id(self, parent_id: str) -> str: ...


def _strict_json_bytes(content: bytes, *, label: str) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant in {label}: {value}")

    try:
        return json.loads(
            content.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {label}: {exc}") from exc


ArtifactSource = str | Path | int


def _open_artifact_binary(source: ArtifactSource) -> BinaryIO:
    if type(source) is int:
        descriptor = os.open(
            f"/proc/self/fd/{source}",
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        return os.fdopen(descriptor, "rb")
    return Path(source).open("rb")


@contextmanager
def _open_zip(source: ArtifactSource) -> Iterator[zipfile.ZipFile]:
    with _open_artifact_binary(source) as handle:
        with zipfile.ZipFile(handle, mode="r") as archive:
            yield archive


def _read_artifact_bytes(source: ArtifactSource) -> bytes:
    with _open_artifact_binary(source) as handle:
        return handle.read()


def _artifact_size(source: ArtifactSource) -> int:
    if type(source) is int:
        return os.fstat(source).st_size
    return Path(source).stat().st_size


def _artifact_label(source: ArtifactSource) -> Path:
    if type(source) is int:
        return Path(f"sealed-production-data-fd-{source}")
    return Path(source)


def _file_sha256(path: ArtifactSource) -> str:
    digest = hashlib.sha256()
    with _open_artifact_binary(path) as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sanitized_zip_info(info: zipfile.ZipInfo, *, expected_path: str) -> None:
    path = PurePosixPath(info.filename)
    if (
        info.filename != expected_path
        or path.is_absolute()
        or path.as_posix() != info.filename
        or any(part in {"", ".", ".."} for part in path.parts)
        or info.is_dir()
        or stat.S_ISLNK(info.external_attr >> 16)
        or info.flag_bits & 0x1
        or info.compress_type != zipfile.ZIP_DEFLATED
        or info.date_time != (1980, 1, 1, 0, 0, 0)
        or (info.external_attr >> 16) != 0o100644
    ):
        raise ValueError(f"unsafe or noncanonical sanitized member: {info.filename!r}")


def _parse_rearc_member(content: bytes, *, label: str) -> tuple[M04AExample, ...]:
    payload = _strict_json_bytes(content, label=label)
    if not isinstance(payload, list) or len(payload) != 1000:
        raise ValueError(f"{label} must contain exactly 1,000 examples")
    result: list[M04AExample] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict) or set(row) != {"input", "output"}:
            raise ValueError(f"{label}[{index}] must contain exactly input and output")
        result.append(M04AExample.create(row["input"], row["output"]))
    return tuple(result)


class SanitizedReARCZip:
    """Strict, bounded-cache reader for one physically cropped ReARC shard."""

    def __init__(
        self,
        archive_path: ArtifactSource,
        *,
        source_parent_ids: Sequence[str],
        member_rows: Sequence[Mapping[str, object]],
        source: str,
        fold: str,
        cache_parents: int = 2,
    ) -> None:
        if type(archive_path) is int:
            if archive_path < 3:
                raise ValueError("sealed archive descriptor must be >= 3")
            self._archive_source: ArtifactSource = archive_path
        else:
            self._archive_source = Path(archive_path).expanduser().resolve()
        ids = tuple(source_parent_ids)
        if not ids or len(ids) != len(set(ids)) or tuple(sorted(ids)) != ids:
            raise ValueError("ReARC source parent IDs must be sorted and unique")
        if type(cache_parents) is not int or cache_parents <= 0:
            raise ValueError("cache_parents must be positive")
        expected_prefix = f"{source}/{fold}/"
        rows = tuple(member_rows)
        if len(rows) != len(ids):
            raise ValueError("ReARC member rows do not close against allowlist")
        self._members: dict[str, tuple[str, str, int]] = {}
        self._semantic_ids: dict[str, str] = {}
        for parent_id, row in zip(ids, rows, strict=True):
            if not isinstance(row, Mapping):
                raise TypeError("ReARC member row must be a mapping")
            expected_path = f"{expected_prefix}{parent_id}.json"
            if row.get("path") != expected_path or row.get("source_parent_id") != parent_id:
                raise ValueError("ReARC member row identity mismatch")
            semantic_parent_id = row.get("semantic_parent_id")
            if not isinstance(semantic_parent_id, str) or not re.fullmatch(
                r"[0-9a-f]{8}", semantic_parent_id
            ):
                raise ValueError("ReARC semantic parent ID must be lowercase 8-hex")
            digest = row.get("sha256")
            byte_count = row.get("bytes")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("ReARC member digest must be lowercase SHA-256")
            if type(byte_count) is not int or byte_count <= 0:
                raise ValueError("ReARC member byte count must be positive")
            self._members[parent_id] = (expected_path, digest, byte_count)
            self._semantic_ids[parent_id] = semantic_parent_id
        with _open_zip(self._archive_source) as archive:
            infos = archive.infolist()
            if len(infos) != len(ids):
                raise ValueError("ReARC shard has an unexpected member count")
            info_by_name = {info.filename: info for info in infos}
            if len(info_by_name) != len(infos):
                raise ValueError("ReARC shard contains duplicate members")
            for member_path, _, byte_count in self._members.values():
                info = info_by_name.get(member_path)
                if info is None:
                    raise ValueError(f"ReARC shard is missing {member_path}")
                _validate_sanitized_zip_info(info, expected_path=member_path)
                if info.file_size != byte_count:
                    raise ValueError("ReARC member byte count differs from manifest")
        self._parent_ids = ids
        self._cache_limit = cache_parents
        self._cache: OrderedDict[str, tuple[M04AExample, ...]] = OrderedDict()

    @property
    def parent_ids(self) -> tuple[str, ...]:
        return self._parent_ids

    def _parent(self, parent_id: str) -> tuple[M04AExample, ...]:
        if parent_id not in self._members:
            raise KeyError(parent_id)
        cached = self._cache.pop(parent_id, None)
        if cached is not None:
            self._cache[parent_id] = cached
            return cached
        member_path, expected_sha, expected_bytes = self._members[parent_id]
        with _open_zip(self._archive_source) as archive:
            content = archive.read(member_path)
        if len(content) != expected_bytes or hashlib.sha256(content).hexdigest() != expected_sha:
            raise ValueError("ReARC member changed or failed its content commitment")
        examples = _parse_rearc_member(content, label=member_path)
        self._cache[parent_id] = examples
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return examples

    def example(self, parent_id: str, example_index: int) -> M04AExample:
        if type(example_index) is not int or not 0 <= example_index < 1000:
            raise IndexError("ReARC example index must be in 0..999")
        return self._parent(parent_id)[example_index]

    def semantic_parent_id(self, parent_id: str) -> str:
        try:
            return self._semantic_ids[parent_id]
        except KeyError as exc:
            raise KeyError(parent_id) from exc


@dataclass(frozen=True, slots=True)
class SanitizedM04ABundle:
    root: str
    manifest: Mapping[str, object]
    manifest_sha256: str
    artifact_manifest_sha256: str
    training: "M04ATrainingData"
    validation: "M04ATrainingData"


def _load_arc2_shard(
    path: ArtifactSource,
    *,
    fold: str,
    ids: Sequence[str],
    member_rows: Sequence[Mapping[str, object]],
) -> dict[str, ARC2Parent]:
    expected_ids = tuple(ids)
    rows = tuple(member_rows)
    if len(rows) != len(expected_ids):
        raise ValueError("ARC2 member rows do not close against allowlist")
    result: dict[str, ARC2Parent] = {}
    with _open_zip(path) as archive:
        infos = archive.infolist()
        if len(infos) != len(expected_ids):
            raise ValueError("ARC2 shard has an unexpected member count")
        info_by_name = {info.filename: info for info in infos}
        if len(info_by_name) != len(infos):
            raise ValueError("ARC2 shard contains duplicate members")
        for parent_id, row in zip(expected_ids, rows, strict=True):
            member_path = f"arc2/{fold}/{parent_id}.json"
            if (
                row.get("path") != member_path
                or row.get("source_parent_id") != parent_id
                or row.get("semantic_parent_id") != parent_id
            ):
                raise ValueError("ARC2 member row identity mismatch")
            info = info_by_name.get(member_path)
            if info is None:
                raise ValueError(f"ARC2 shard is missing {member_path}")
            _validate_sanitized_zip_info(info, expected_path=member_path)
            content = archive.read(info)
            digest = row.get("sha256")
            byte_count = row.get("bytes")
            if (
                not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or type(byte_count) is not int
                or byte_count != len(content)
                or hashlib.sha256(content).hexdigest() != digest
            ):
                raise ValueError("ARC2 member commitment mismatch")
            task = parse_task_bytes(
                content,
                source_path=_artifact_label(path) / member_path,
                task_id=parent_id,
            )
            if any(pair.output is None for pair in task.test):
                raise ValueError("sanitized ARC2 tasks must retain public test outputs")
            result[parent_id] = ARC2Parent(
                parent_id=parent_id,
                train=tuple(
                    M04AExample.create(pair.input, pair.output) for pair in task.train
                ),
                test=tuple(
                    M04AExample.create(pair.input, pair.output) for pair in task.test
                ),
            )
    return result


def load_sanitized_m04a_data(
    bundle_dir: str | Path,
    *,
    require_production_counts: bool = True,
    expected_artifact_manifest_sha256: str | None = None,
    artifact_sources: Mapping[str, ArtifactSource] | None = None,
) -> SanitizedM04ABundle:
    """Load only the physically cropped bundle; no original-data path is accepted."""

    unresolved = Path(bundle_dir).expanduser()
    if artifact_sources is None:
        if unresolved.is_symlink():
            raise ValueError("sanitized bundle root may not be a symlink")
        root = unresolved.resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        visible = {path.name for path in root.iterdir()}
        if visible != _SANITIZED_FILES or any(
            path.is_symlink() for path in root.iterdir()
        ):
            raise ValueError(
                "sanitized bundle has missing, extra, or symlinked artifacts"
            )
        sources: dict[str, ArtifactSource] = {
            name: root / name for name in _SANITIZED_FILES
        }
    else:
        if set(artifact_sources) != _SANITIZED_FILES:
            raise ValueError("sealed sanitized artifact sources are not closed-world")
        root = Path(os.path.abspath(os.fspath(unresolved)))
        sources = dict(artifact_sources)
        if any(type(source) is bool for source in sources.values()):
            raise TypeError("sealed sanitized artifact source is invalid")
    artifact_bytes = _read_artifact_bytes(sources["artifact_manifest.json"])
    artifact_manifest_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    if expected_artifact_manifest_sha256 is None:
        if require_production_counts:
            raise ValueError("production load requires the external artifact-manifest SHA-256")
    elif (
        not re.fullmatch(r"[0-9a-f]{64}", expected_artifact_manifest_sha256)
        or artifact_manifest_sha256 != expected_artifact_manifest_sha256
    ):
        raise ValueError("sanitized artifact-manifest external commitment mismatch")
    artifact = _strict_json_bytes(artifact_bytes, label="artifact_manifest.json")
    if (
        not isinstance(artifact, dict)
        or set(artifact) != {"schema_version", "bundle_status", "run_id", "artifacts"}
        or artifact.get("schema_version") != 1
    ):
        raise ValueError("invalid sanitized artifact manifest")
    if artifact.get("bundle_status") != "complete" or not isinstance(
        artifact.get("artifacts"), dict
    ):
        raise ValueError("sanitized artifact manifest is incomplete")
    artifact_rows = artifact["artifacts"]
    expected_artifacts = _SANITIZED_FILES - {"artifact_manifest.json"}
    if set(artifact_rows) != expected_artifacts:
        raise ValueError("sanitized artifact manifest is not closed-world")
    for name in sorted(expected_artifacts):
        row = artifact_rows[name]
        path = sources[name]
        if not isinstance(row, dict):
            raise TypeError("artifact metadata row must be an object")
        if row.get("bytes") != _artifact_size(path) or row.get(
            "sha256"
        ) != _file_sha256(path):
            raise ValueError(f"sanitized artifact commitment mismatch: {name}")

    split_bytes = _read_artifact_bytes(sources["data_split_manifest.json"])
    manifest = _strict_json_bytes(split_bytes, label="data_split_manifest.json")
    expected_manifest_keys = {
        "schema",
        "data_folds_semantics_version",
        "sealed_artifact_manifest_sha256",
        "source_commitments",
        "usable_fold_counts",
        "ordered_allowlists",
        "semantic_aliases_in_usable_shards",
        "validation_quarantine_ids",
        "training_quarantine_ids",
        "quarantine_reasons",
        "shards",
        "protected_payloads_absent",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected_manifest_keys
        or manifest.get("schema") != SANITIZED_MANIFEST_SCHEMA
    ):
        raise ValueError("sanitized split manifest schema mismatch")
    if manifest.get("protected_payloads_absent") is not True:
        raise ValueError("sanitized manifest does not attest protected-payload absence")
    if manifest.get("data_folds_semantics_version") != DATA_FOLDS_SEMANTICS_VERSION:
        raise ValueError("sanitized data-fold semantics version mismatch")
    sealed_commitment = manifest.get("sealed_artifact_manifest_sha256")
    if not isinstance(sealed_commitment, str) or re.fullmatch(
        r"[0-9a-f]{64}", sealed_commitment
    ) is None:
        raise ValueError("sanitized sealed-bundle commitment is invalid")
    forbidden_keys = {"collisions", "collision_rows", "protected_orbit_ids", "output_hashes"}

    def check_keys(value: object) -> None:
        if isinstance(value, dict):
            overlap = forbidden_keys & set(value)
            if overlap:
                raise ValueError(f"protected payload field in sanitized manifest: {sorted(overlap)}")
            for child in value.values():
                check_keys(child)
        elif isinstance(value, list):
            for child in value:
                check_keys(child)

    check_keys(manifest)
    counts = manifest.get("usable_fold_counts")
    allowlists = manifest.get("ordered_allowlists")
    shards = manifest.get("shards")
    if not isinstance(counts, dict) or not isinstance(allowlists, dict) or not isinstance(shards, dict):
        raise ValueError("sanitized split manifest lacks counts, allowlists, or shards")
    expected_count_keys = set(_PRODUCTION_COUNTS)
    if set(counts) != expected_count_keys or set(allowlists) != expected_count_keys:
        raise ValueError("sanitized count or allowlist keys are not closed-world")
    if set(shards) != {
        "arc2_train.zip",
        "arc2_validation.zip",
        "rearc_train.zip",
        "rearc_validation.zip",
    }:
        raise ValueError("sanitized shard manifest keys are not closed-world")
    if require_production_counts and counts != _PRODUCTION_COUNTS:
        raise ValueError("sanitized bundle does not match frozen production counts")
    if require_production_counts and manifest.get("source_commitments") != (
        _PRODUCTION_SOURCE_COMMITMENTS
    ):
        raise ValueError("sanitized bundle source commitments are not frozen production inputs")

    alias_rows = manifest.get("semantic_aliases_in_usable_shards")
    validation_quarantine = manifest.get("validation_quarantine_ids")
    training_quarantine = manifest.get("training_quarantine_ids")
    quarantine_reasons = manifest.get("quarantine_reasons")
    if (
        not isinstance(alias_rows, list)
        or not isinstance(validation_quarantine, list)
        or not isinstance(training_quarantine, list)
        or not isinstance(quarantine_reasons, dict)
    ):
        raise ValueError("sanitized semantic-alias or quarantine closure is malformed")

    def ordered_ids(values: list[object], *, field: str) -> tuple[str, ...]:
        if any(
            not isinstance(item, str) or re.fullmatch(r"[0-9a-f]{8}", item) is None
            for item in values
        ):
            raise ValueError(f"{field} must contain lowercase 8-hex IDs")
        ids = tuple(str(item) for item in values)
        if (
            ids != tuple(sorted(set(ids)))
        ):
            raise ValueError(f"{field} must be sorted unique lowercase 8-hex IDs")
        return ids

    validation_quarantine_ids = ordered_ids(
        validation_quarantine, field="validation_quarantine_ids"
    )
    training_quarantine_ids = ordered_ids(
        training_quarantine, field="training_quarantine_ids"
    )
    quarantine_union = set(validation_quarantine_ids) | set(training_quarantine_ids)
    if set(quarantine_reasons) != quarantine_union:
        raise ValueError("quarantine reasons do not close against quarantine IDs")
    if any(
        not isinstance(reasons, list)
        or not reasons
        or reasons != sorted(set(reasons))
        or any(not isinstance(reason, str) or not reason for reason in reasons)
        for reasons in quarantine_reasons.values()
    ):
        raise ValueError("quarantine reason lists must be non-empty sorted strings")

    alias_map: dict[str, str] = {}
    for index, row in enumerate(alias_rows):
        if not isinstance(row, dict) or set(row) != {
            "rearc_parent_id",
            "arc2_parent_id",
        }:
            raise ValueError(f"semantic alias row {index} has invalid fields")
        source_id = row["rearc_parent_id"]
        target_id = row["arc2_parent_id"]
        if (
            not isinstance(source_id, str)
            or re.fullmatch(r"[0-9a-f]{8}", source_id) is None
            or not isinstance(target_id, str)
            or re.fullmatch(r"[0-9a-f]{8}", target_id) is None
            or source_id == target_id
            or source_id in alias_map
        ):
            raise ValueError("semantic aliases must be unique nonidentity 8-hex mappings")
        alias_map[source_id] = target_id

    views: dict[str, tuple[str, tuple[str, ...], dict[str, object]]] = {}
    for key, artifact_name in (
        ("arc2_train", "arc2_train.zip"),
        ("arc2_validation", "arc2_validation.zip"),
        ("rearc_train", "rearc_train.zip"),
        ("rearc_validation", "rearc_validation.zip"),
    ):
        ids_value = allowlists.get(key)
        shard = shards.get(artifact_name)
        if (
            not isinstance(ids_value, list)
            or any(not isinstance(item, str) for item in ids_value)
            or tuple(ids_value) != tuple(sorted(set(ids_value)))
            or not isinstance(shard, dict)
            or shard.get("ordered_source_parent_ids") != ids_value
            or shard.get("member_count") != len(ids_value)
            or counts.get(key) != len(ids_value)
            or shard.get("zip_sha256") != _file_sha256(sources[artifact_name])
            or shard.get("zip_bytes") != _artifact_size(sources[artifact_name])
        ):
            raise ValueError(f"sanitized shard closure mismatch: {key}")
        members = shard.get("members")
        if not isinstance(members, list):
            raise ValueError(f"sanitized shard lacks member rows: {key}")
        views[key] = (artifact_name, tuple(ids_value), {"manifest": shard, "members": members})

    arc_train_name, arc_train_ids, arc_train_meta = views["arc2_train"]
    arc_val_name, arc_val_ids, arc_val_meta = views["arc2_validation"]
    rearc_train_name, rearc_train_ids, rearc_train_meta = views["rearc_train"]
    rearc_val_name, rearc_val_ids, rearc_val_meta = views["rearc_validation"]
    usable_rearc_ids = set(rearc_train_ids) | set(rearc_val_ids)
    if set(alias_map) - usable_rearc_ids:
        raise ValueError("semantic alias source is absent from usable ReARC shards")
    observed_aliases: dict[str, str] = {}
    for parent_ids, metadata in (
        (rearc_train_ids, rearc_train_meta),
        (rearc_val_ids, rearc_val_meta),
    ):
        rows = metadata["members"]
        if not isinstance(rows, list):
            raise ValueError("ReARC member metadata must be a list")
        for source_id, row in zip(parent_ids, rows, strict=True):
            if not isinstance(row, dict):
                raise TypeError("ReARC member metadata row must be an object")
            semantic_id = row.get("semantic_parent_id")
            expected_semantic_id = alias_map.get(source_id, source_id)
            if semantic_id != expected_semantic_id:
                raise ValueError("ReARC member semantic parent differs from alias closure")
            if semantic_id != source_id:
                observed_aliases[source_id] = semantic_id
            if semantic_id in quarantine_union:
                raise ValueError("usable ReARC member maps to a quarantined semantic parent")
    if observed_aliases != alias_map:
        raise ValueError("semantic alias table does not close bidirectionally from shards")
    if (set(arc_train_ids) | set(arc_val_ids)) & quarantine_union:
        raise ValueError("usable ARC2 allowlist contains a quarantined parent")
    arc_train = _load_arc2_shard(
        sources[arc_train_name],
        fold="train",
        ids=arc_train_ids,
        member_rows=arc_train_meta["members"],  # type: ignore[arg-type]
    )
    arc_validation = _load_arc2_shard(
        sources[arc_val_name],
        fold="validation",
        ids=arc_val_ids,
        member_rows=arc_val_meta["members"],  # type: ignore[arg-type]
    )
    rearc_train = SanitizedReARCZip(
        sources[rearc_train_name],
        source_parent_ids=rearc_train_ids,
        member_rows=rearc_train_meta["members"],  # type: ignore[arg-type]
        source="rearc",
        fold="train",
    )
    rearc_validation = SanitizedReARCZip(
        sources[rearc_val_name],
        source_parent_ids=rearc_val_ids,
        member_rows=rearc_val_meta["members"],  # type: ignore[arg-type]
        source="rearc",
        fold="validation",
    )
    return SanitizedM04ABundle(
        root=str(root),
        manifest=manifest,
        manifest_sha256=hashlib.sha256(split_bytes).hexdigest(),
        artifact_manifest_sha256=artifact_manifest_sha256,
        training=M04ATrainingData(
            arc2_parent_ids=arc_train_ids,
            arc2_parents=arc_train,
            rearc_parent_ids=rearc_train_ids,
            rearc=rearc_train,
        ),
        validation=M04ATrainingData(
            arc2_parent_ids=arc_val_ids,
            arc2_parents=arc_validation,
            rearc_parent_ids=rearc_val_ids,
            rearc=rearc_validation,
        ),
    )


@dataclass(frozen=True, slots=True)
class M04ATrainingData:
    arc2_parent_ids: tuple[str, ...]
    arc2_parents: Mapping[str, ARC2Parent]
    rearc_parent_ids: tuple[str, ...]
    rearc: ReARCExampleSource

    def __post_init__(self) -> None:
        arc_ids = tuple(self.arc2_parent_ids)
        if not arc_ids or len(arc_ids) != len(set(arc_ids)):
            raise ValueError("ordered ARC2 parent IDs must be non-empty and unique")
        if tuple(sorted(self.arc2_parents)) != tuple(sorted(arc_ids)):
            raise ValueError("ARC2 parent mapping does not close against its ordered IDs")
        for parent_id in arc_ids:
            _strict_id(parent_id, field="ARC2 parent ID")
            if self.arc2_parents[parent_id].parent_id != parent_id:
                raise ValueError("ARC2 parent mapping key/record mismatch")
        rearc_ids = tuple(self.rearc_parent_ids)
        if not rearc_ids or len(rearc_ids) != len(set(rearc_ids)):
            raise ValueError("ordered ReARC parent IDs must be non-empty and unique")
        if tuple(self.rearc.parent_ids) != rearc_ids:
            raise ValueError("ReARC source does not close against its ordered IDs")
        object.__setattr__(self, "arc2_parent_ids", arc_ids)
        object.__setattr__(self, "rearc_parent_ids", rearc_ids)


@dataclass(frozen=True, slots=True)
class M04AEpisode:
    source: str
    parent_id: str
    semantic_parent_id: str
    target_descriptor: str
    demonstration_descriptors: tuple[str, ...]
    demonstrations: tuple[M04AExample, ...]
    query_input: Grid
    target_output: Grid
    d4_index: int
    color_permutation: tuple[int, ...]
    seed_u64: int
    masked_linear_indices: tuple[int, ...]
    corruption_kind: str
    optimizer_step: int | None = None
    microbatch_slot: int | None = None
    validation_view_index: int | None = None
    validation_mask_fraction: str | None = None

    def __post_init__(self) -> None:
        if self.source not in {ARC2_SOURCE, REARC_SOURCE}:
            raise ValueError("episode source must be arc2 or rearc")
        object.__setattr__(self, "parent_id", _strict_id(self.parent_id, field="parent_id"))
        object.__setattr__(
            self,
            "semantic_parent_id",
            _strict_id(self.semantic_parent_id, field="semantic_parent_id"),
        )
        object.__setattr__(
            self,
            "target_descriptor",
            _strict_id(self.target_descriptor, field="target_descriptor"),
        )
        descriptors = tuple(self.demonstration_descriptors)
        demos = tuple(self.demonstrations)
        if not demos or len(demos) > 10 or len(descriptors) != len(demos):
            raise ValueError("episode requires 1-10 demonstrations with descriptors")
        if any(not isinstance(example, M04AExample) for example in demos):
            raise TypeError("episode demonstrations must be M04AExample values")
        if any(not isinstance(item, str) or not item for item in descriptors):
            raise TypeError("demonstration descriptors must be non-empty strings")
        query = _grid(self.query_input, field="query input")
        target = _grid(self.target_output, field="target output")
        if type(self.d4_index) is not int or not 0 <= self.d4_index < 8:
            raise ValueError("d4_index must be in 0..7")
        permutation = tuple(self.color_permutation)
        if permutation != tuple(permutation) or sorted(permutation) != list(range(10)):
            raise ValueError("color_permutation must be a bijection of colors 0..9")
        if type(self.seed_u64) is not int or not 0 <= self.seed_u64 < 2**64:
            raise ValueError("seed_u64 must be an unsigned 64-bit integer")
        cell_count = len(target) * len(target[0])
        masked = tuple(self.masked_linear_indices)
        if not masked or masked != tuple(sorted(set(masked))):
            raise ValueError("masked positions must be a non-empty sorted unique tuple")
        if masked[0] < 0 or masked[-1] >= cell_count:
            raise ValueError("masked position falls outside target")
        if self.corruption_kind not in {"single_cell", "all_mask", "partial", "validation"}:
            raise ValueError("unknown corruption_kind")
        is_training = self.optimizer_step is not None or self.microbatch_slot is not None
        is_validation = self.validation_view_index is not None
        if is_training == is_validation:
            raise ValueError("episode must be exactly one of training or validation")
        if is_training:
            if type(self.optimizer_step) is not int or type(self.microbatch_slot) is not int:
                raise TypeError("training episode needs integer step and microbatch slot")
            if not 0 <= self.microbatch_slot < GRADIENT_ACCUMULATION:
                raise ValueError("microbatch_slot is outside the accumulation window")
            if self.seed_u64 != episode_seed(self.optimizer_step, self.microbatch_slot):
                raise ValueError("training episode seed does not match its counter address")
            if self.validation_mask_fraction is not None:
                raise ValueError("training episode cannot carry validation fraction")
            if self.corruption_kind == "single_cell" and not (
                cell_count == 1 and masked == (0,)
            ):
                raise ValueError("single-cell corruption requires the sole target cell")
            if self.corruption_kind == "all_mask" and len(masked) != cell_count:
                raise ValueError("all-mask corruption must mask the complete target")
            if self.corruption_kind == "partial" and not 1 <= len(masked) < cell_count:
                raise ValueError("partial corruption must leave at least one true cell")
        else:
            if type(self.validation_view_index) is not int or not 0 <= self.validation_view_index < 4:
                raise ValueError("validation_view_index must be in 0..3")
            expected_fraction = VALIDATION_MASK_FRACTIONS[self.validation_view_index][2]
            if self.validation_mask_fraction != expected_fraction:
                raise ValueError("invalid validation mask fraction")
            if self.corruption_kind != "validation":
                raise ValueError("validation episode must use validation corruption kind")
            if self.seed_u64 != validation_episode_seed(
                self.parent_id, self.target_descriptor, self.validation_view_index
            ):
                raise ValueError("validation episode seed does not match its identity")
            expected_mask = _validation_mask(
                parent_id=self.parent_id,
                target_descriptor=self.target_descriptor,
                view_index=self.validation_view_index,
                height=len(target),
                width=len(target[0]),
            )
            if masked != expected_mask:
                raise ValueError("validation mask coordinates do not match their hash rank")
        object.__setattr__(self, "demonstration_descriptors", descriptors)
        object.__setattr__(self, "demonstrations", demos)
        object.__setattr__(self, "query_input", query)
        object.__setattr__(self, "target_output", target)
        object.__setattr__(self, "color_permutation", permutation)
        object.__setattr__(self, "masked_linear_indices", masked)

    def to_json_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": "afts-grid-cmlm-episode-row/v0.1",
            "source": self.source,
            "parent_id": self.parent_id,
            "semantic_parent_id": self.semantic_parent_id,
            "target_descriptor": self.target_descriptor,
            "demonstration_descriptors": list(self.demonstration_descriptors),
            "demonstrations": [
                {
                    "input": grid_to_lists(example.input_grid),
                    "output": grid_to_lists(example.output_grid),
                }
                for example in self.demonstrations
            ],
            "query_input": grid_to_lists(self.query_input),
            "target_output": grid_to_lists(self.target_output),
            "d4_index": self.d4_index,
            "color_permutation": list(self.color_permutation),
            "seed_u64": self.seed_u64,
            "masked_linear_indices": list(self.masked_linear_indices),
            "corruption_kind": self.corruption_kind,
            "optimizer_step": self.optimizer_step,
            "microbatch_slot": self.microbatch_slot,
            "validation_view_index": self.validation_view_index,
            "validation_mask_fraction": self.validation_mask_fraction,
        }
        payload["episode_sha256"] = canonical_sha256(payload)
        return payload


def _transform_grid(grid: Grid, d4_index: int, permutation: Sequence[int]) -> Grid:
    transformed = apply_indexed_d4(grid, d4_index)
    return tuple(tuple(permutation[color] for color in row) for row in transformed)


def _transform_example(
    example: M04AExample, d4_index: int, permutation: Sequence[int]
) -> M04AExample:
    return M04AExample.create(
        _transform_grid(example.input_grid, d4_index, permutation),
        _transform_grid(example.output_grid, d4_index, permutation),
    )


def _augment_episode(
    *,
    rng: random.Random,
    demonstrations: Sequence[tuple[str, M04AExample]],
    query_input: Grid,
    target_output: Grid,
) -> tuple[
    tuple[str, ...],
    tuple[M04AExample, ...],
    Grid,
    Grid,
    int,
    tuple[int, ...],
]:
    d4_index = rng.randrange(8)
    permutation_list = list(range(10))
    rng.shuffle(permutation_list)
    permutation = tuple(permutation_list)
    demo_order = list(range(len(demonstrations)))
    rng.shuffle(demo_order)
    ordered = tuple(demonstrations[index] for index in demo_order)
    descriptors = tuple(item[0] for item in ordered)
    transformed_demos = tuple(
        _transform_example(item[1], d4_index, permutation) for item in ordered
    )
    return (
        descriptors,
        transformed_demos,
        _transform_grid(query_input, d4_index, permutation),
        _transform_grid(target_output, d4_index, permutation),
        d4_index,
        permutation,
    )


def _draw_training_mask(
    rng: random.Random, cell_count: int
) -> tuple[tuple[int, ...], str]:
    if type(cell_count) is not int or cell_count <= 0:
        raise ValueError("cell_count must be positive")
    if cell_count == 1:
        return (0,), "single_cell"
    if rng.random() < 0.1:
        return tuple(range(cell_count)), "all_mask"
    while True:
        probability = rng.random()
        masked = tuple(
            index for index in range(cell_count) if rng.random() < probability
        )
        if 1 <= len(masked) < cell_count:
            return masked, "partial"


def _arc_target(
    parent: ARC2Parent, target_index: int
) -> tuple[str, tuple[tuple[str, M04AExample], ...], M04AExample]:
    target_count = len(parent.test) + (len(parent.train) if len(parent.train) > 1 else 0)
    if not 0 <= target_index < target_count:
        raise IndexError("ARC target index is outside the eligible list")
    if target_index < len(parent.test):
        target = parent.test[target_index]
        demos = tuple(
            (f"train:{index}", example) for index, example in enumerate(parent.train)
        )
        return f"test:{target_index}", demos, target
    train_index = target_index - len(parent.test)
    target = parent.train[train_index]
    demos = tuple(
        (f"train:{index}", example)
        for index, example in enumerate(parent.train)
        if index != train_index
    )
    return f"train:{train_index}", demos, target


def build_training_episode(
    data: M04ATrainingData, optimizer_step: int, microbatch_slot: int
) -> M04AEpisode:
    """Build exactly one counter-addressed training episode."""

    seed = episode_seed(optimizer_step, microbatch_slot)
    rng = random.Random(seed)
    if microbatch_slot % 2 == 0:
        source = ARC2_SOURCE
        parent_id = data.arc2_parent_ids[rng.randrange(len(data.arc2_parent_ids))]
        parent = data.arc2_parents[parent_id]
        target_count = len(parent.test) + (
            len(parent.train) if len(parent.train) > 1 else 0
        )
        descriptor, demos, target = _arc_target(parent, rng.randrange(target_count))
    else:
        source = REARC_SOURCE
        parent_id = data.rearc_parent_ids[rng.randrange(len(data.rearc_parent_ids))]
        indices = rng.sample(range(1000), 4)
        demos = tuple(
            (f"rearc:{index}", data.rearc.example(parent_id, index))
            for index in indices[:3]
        )
        target_index = indices[3]
        descriptor = f"rearc:{target_index}"
        target = data.rearc.example(parent_id, target_index)
    (
        demo_descriptors,
        transformed_demos,
        query_input,
        target_output,
        d4_index,
        permutation,
    ) = _augment_episode(
        rng=rng,
        demonstrations=demos,
        query_input=target.input_grid,
        target_output=target.output_grid,
    )
    cell_count = len(target_output) * len(target_output[0])
    masked, kind = _draw_training_mask(rng, cell_count)
    return M04AEpisode(
        source=source,
        parent_id=parent_id,
        semantic_parent_id=(
            parent_id if source == ARC2_SOURCE else data.rearc.semantic_parent_id(parent_id)
        ),
        target_descriptor=descriptor,
        demonstration_descriptors=demo_descriptors,
        demonstrations=transformed_demos,
        query_input=query_input,
        target_output=target_output,
        d4_index=d4_index,
        color_permutation=permutation,
        seed_u64=seed,
        masked_linear_indices=masked,
        corruption_kind=kind,
        optimizer_step=optimizer_step,
        microbatch_slot=microbatch_slot,
    )


def _validation_identity(parent_id: str, target_descriptor: str, view_index: int) -> bytes:
    return (
        f"{VALIDATION_SEMANTICS_VERSION}\0{parent_id}\0"
        f"{target_descriptor}\0{view_index}"
    ).encode("utf-8")


def _rearc_validation_indices(parent_id: str, view_index: int) -> tuple[int, int, int, int]:
    prefix = f"{VALIDATION_SEMANTICS_VERSION}\0{parent_id}\0{view_index}\0".encode(
        "utf-8"
    )
    ranked = sorted(
        range(1000),
        key=lambda index: (hashlib.sha256(prefix + str(index).encode("ascii")).digest(), index),
    )
    selected = tuple(ranked[:4])
    if len(selected) != 4:
        raise RuntimeError("ReARC validation ranking did not return four examples")
    return selected[0], selected[1], selected[2], selected[3]


def _validation_mask(
    *,
    parent_id: str,
    target_descriptor: str,
    view_index: int,
    height: int,
    width: int,
) -> tuple[int, ...]:
    numerator, denominator, _ = VALIDATION_MASK_FRACTIONS[view_index]
    count = (height * width * numerator + denominator - 1) // denominator
    identity = _validation_identity(parent_id, target_descriptor, view_index)
    ranked = sorted(
        range(height * width),
        key=lambda linear: (
            hashlib.sha256(
                identity
                + b"\0"
                + str(linear // width).encode("ascii")
                + b"\0"
                + str(linear % width).encode("ascii")
            ).digest(),
            linear,
        ),
    )
    return tuple(sorted(ranked[:count]))


def _validation_episode(
    *,
    source: str,
    parent_id: str,
    semantic_parent_id: str,
    descriptor: str,
    demos: Sequence[tuple[str, M04AExample]],
    target: M04AExample,
    view_index: int,
) -> M04AEpisode:
    seed = validation_episode_seed(parent_id, descriptor, view_index)
    rng = random.Random(seed)
    (
        demo_descriptors,
        transformed_demos,
        query_input,
        target_output,
        d4_index,
        permutation,
    ) = _augment_episode(
        rng=rng,
        demonstrations=demos,
        query_input=target.input_grid,
        target_output=target.output_grid,
    )
    masked = _validation_mask(
        parent_id=parent_id,
        target_descriptor=descriptor,
        view_index=view_index,
        height=len(target_output),
        width=len(target_output[0]),
    )
    return M04AEpisode(
        source=source,
        parent_id=parent_id,
        semantic_parent_id=semantic_parent_id,
        target_descriptor=descriptor,
        demonstration_descriptors=demo_descriptors,
        demonstrations=transformed_demos,
        query_input=query_input,
        target_output=target_output,
        d4_index=d4_index,
        color_permutation=permutation,
        seed_u64=seed,
        masked_linear_indices=masked,
        corruption_kind="validation",
        validation_view_index=view_index,
        validation_mask_fraction=VALIDATION_MASK_FRACTIONS[view_index][2],
    )


def build_validation_episodes(data: M04ATrainingData) -> tuple[M04AEpisode, ...]:
    """Materialize the literal single-episode validation order before training."""

    episodes: list[M04AEpisode] = []
    for parent_id in data.arc2_parent_ids:
        parent = data.arc2_parents[parent_id]
        target_count = len(parent.test) + (
            len(parent.train) if len(parent.train) > 1 else 0
        )
        for target_index in range(target_count):
            descriptor, demos, target = _arc_target(parent, target_index)
            for view_index in range(4):
                episodes.append(
                    _validation_episode(
                        source=ARC2_SOURCE,
                        parent_id=parent_id,
                        semantic_parent_id=parent_id,
                        descriptor=descriptor,
                        demos=demos,
                        target=target,
                        view_index=view_index,
                    )
                )
    for parent_id in data.rearc_parent_ids:
        for view_index in range(4):
            indices = _rearc_validation_indices(parent_id, view_index)
            demos = tuple(
                (f"rearc:{index}", data.rearc.example(parent_id, index))
                for index in indices[:3]
            )
            target_index = indices[3]
            descriptor = f"rearc:{target_index}"
            episodes.append(
                _validation_episode(
                    source=REARC_SOURCE,
                    parent_id=parent_id,
                    semantic_parent_id=data.rearc.semantic_parent_id(parent_id),
                    descriptor=descriptor,
                    demos=demos,
                    target=data.rearc.example(parent_id, target_index),
                    view_index=view_index,
                )
            )
    return tuple(episodes)
