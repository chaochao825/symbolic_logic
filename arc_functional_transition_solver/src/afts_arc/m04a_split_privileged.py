"""Privileged, oracle-aware M04a split compiler.

The compiler is deliberately isolated from the training loader.  It reads the
complete pinned parents and ReARC archive, performs the frozen contamination
audits, and publishes two physically separate evidence roots:

* a sealed root containing protected hashes and collision rows; and
* a sanitized root containing only four allow-listed data shards and a manifest.

The module uses only the Python standard library and other pure-standard-library
``afts_arc`` modules.  It never overwrites an output directory.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import stat
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .authority import DATASET_COMMITS, DATASET_ORIGINS
from .blind import BlindTask
from .m04a_contract import (
    ARC2_SPLIT_SALT,
    DATA_FOLDS_SEMANTICS_VERSION,
    MAX_GRID_SIDE,
    PUBLIC_SMOKE_SELECTION_SALT,
    SAMPLE_ORBIT_SEMANTICS_VERSION,
    TASK_ORBIT_SEMANTICS_VERSION,
    canonical_json_bytes,
    canonical_sha256,
    rearc_parent_split_bucket,
    sample_orbit_id,
    task_orbit_id,
)
from .manifest import (
    GitRepoState,
    git_blob_bytes,
    git_repo_state,
    publish_evidence_bundle,
    serialize_json,
    serialize_jsonl,
    snapshot_task_directory,
)
from .task import ARCPair, ARCTask, task_semantic_fingerprint


PRIVILEGED_MANIFEST_SCHEMA = "afts.m04a-privileged-split/v1"
SANITIZED_MANIFEST_SCHEMA = "afts.m04a-sanitized-split/v1"
COLLISION_ROW_SCHEMA = "afts.m04a-split-collision/v1"
SOURCE_ROW_SCHEMA = "afts.m04a-source-file/v1"
SPLIT_ROW_SCHEMA = "afts.m04a-parent-split-row/v1"
REARC_ORIGIN = "https://github.com/michaelhodel/re-arc.git"
REARC_COMMIT = "e5b7f1d06362a76f9d3b8c25154ff1fafca897ce"
REARC_ARCHIVE_SHA256 = (
    "4a7c309499f450eb47c2117fa06f3c97fe741cf6e7912579c0b5b8cc759e9ac4"
)
REARC_ARCHIVE_BYTES = 50_913_541

FROZEN_VALIDATION_QUARANTINE_IDS = (
    "3af2c5a8",
    "4c4377d9",
    "67a3c6ac",
    "74dd1130",
    "c9e6f938",
)
FROZEN_TRAINING_ORBIT_QUARANTINE_IDS = (
    "4258a5f9",
    "44f52bb0",
    "46442a0e",
    "46f33fce",
    "6150a2bd",
    "62c24649",
    "67e8384a",
    "6d0aefbc",
    "6fa7a44f",
    "7fe24cdd",
    "85c4e7cd",
    "99b1bc43",
    "ac0a08a4",
    "b91ae062",
    "cce03e0d",
    "ed36ccf7",
    "f5b8619d",
)
FROZEN_DIMENSION_QUARANTINE_IDS = ("e26a3af2",)
FROZEN_TRAINING_QUARANTINE_IDS = tuple(
    sorted((*FROZEN_TRAINING_ORBIT_QUARANTINE_IDS, *FROZEN_DIMENSION_QUARANTINE_IDS))
)

_TASK_MEMBER_RE = re.compile(r"re_arc/tasks/([0-9a-f]{8})\.json")
_RESOURCE_MEMBER_RE = re.compile(r"__MACOSX/re_arc/tasks/\._[0-9a-f]{8}\.json")
_ARC_TASK_NAME_RE = re.compile(r"[0-9a-f]{8}\.json")
_ALLOWED_REARC_METADATA = {
    "re_arc/",
    "re_arc/tasks/",
    "re_arc/metadata.json",
    "__MACOSX/._re_arc",
    "__MACOSX/re_arc/._tasks",
    "__MACOSX/re_arc/._metadata.json",
}


@dataclass(frozen=True, slots=True)
class PrivilegedSplitSpec:
    """Frozen source and closure expectations, with small-fixture overrides."""

    arc1_origin: str = DATASET_ORIGINS["ARC-AGI-1"]
    arc1_commit: str = DATASET_COMMITS["ARC-AGI-1"]
    arc2_origin: str = DATASET_ORIGINS["ARC-AGI-2"]
    arc2_commit: str = DATASET_COMMITS["ARC-AGI-2"]
    rearc_origin: str = REARC_ORIGIN
    rearc_commit: str = REARC_COMMIT
    rearc_archive_sha256: str | None = REARC_ARCHIVE_SHA256
    rearc_archive_bytes: int | None = REARC_ARCHIVE_BYTES
    expected_zip_entry_count: int | None = 806
    arc1_task_count: int = 400
    arc2_task_count: int = 1000
    rearc_examples_per_task: int = 1000
    holdout_count: int = 150
    validation_count: int = 150
    smoke_count: int = 20
    max_grid_side: int = MAX_GRID_SIDE
    max_rearc_member_bytes: int = 16 * 1024 * 1024
    max_rearc_uncompressed_bytes: int = 800 * 1024 * 1024
    expected_exact_semantic_quarantine_count: int | None = 0
    expected_aliases: tuple[tuple[str, str], ...] | None = (
        ("40853293", "070dd51e"),
    )
    expected_initial_rearc_counts: tuple[tuple[str, int], ...] | None = (
        ("holdout", 52),
        ("internal", 2),
        ("smoke", 10),
        ("train", 280),
        ("validation", 56),
    )
    expected_validation_quarantine_ids: tuple[str, ...] | None = (
        FROZEN_VALIDATION_QUARANTINE_IDS
    )
    expected_training_orbit_quarantine_ids: tuple[str, ...] | None = (
        FROZEN_TRAINING_ORBIT_QUARANTINE_IDS
    )
    expected_dimension_quarantine_ids: tuple[str, ...] | None = (
        FROZEN_DIMENSION_QUARANTINE_IDS
    )
    expected_final_counts: tuple[tuple[str, int], ...] | None = (
        ("arc2_train", 662),
        ("arc2_validation", 145),
        ("rearc_train", 262),
        ("rearc_validation", 51),
    )

    def __post_init__(self) -> None:
        for field in (
            "arc1_origin",
            "arc1_commit",
            "arc2_origin",
            "arc2_commit",
            "rearc_origin",
            "rearc_commit",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise TypeError(f"{field} must be a non-empty string")
        for field in ("arc1_commit", "arc2_commit", "rearc_commit"):
            if not re.fullmatch(r"[0-9a-f]{40}", getattr(self, field)):
                raise ValueError(f"{field} must be lowercase 40-character hexadecimal")
        if self.rearc_archive_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.rearc_archive_sha256
        ):
            raise ValueError("rearc_archive_sha256 must be lowercase SHA-256 or None")
        for field in (
            "arc1_task_count",
            "arc2_task_count",
            "rearc_examples_per_task",
            "max_grid_side",
            "max_rearc_member_bytes",
            "max_rearc_uncompressed_bytes",
        ):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        for field in ("holdout_count", "validation_count", "smoke_count"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if self.holdout_count + self.validation_count + self.smoke_count > self.arc2_task_count:
            raise ValueError("reserved ARC2 folds exceed arc2_task_count")


@dataclass(frozen=True, slots=True)
class RearcMemberAudit:
    parent_id: str
    member_name: str
    sha256: str
    byte_count: int
    example_count: int
    maximum_height: int
    maximum_width: int
    oversized_example_count: int
    oversized_grid_count: int


@dataclass(frozen=True, slots=True)
class RearcArchiveAudit:
    archive_sha256: str
    archive_bytes: int
    zip_entry_count: int
    member_ids: tuple[str, ...]
    member_audits: tuple[RearcMemberAudit, ...]


@dataclass(frozen=True, slots=True)
class _RearcParent:
    source_parent_id: str
    semantic_parent_id: str
    initial_fold: str
    inheritance: str
    alias_target_id: str | None


@dataclass(frozen=True, slots=True)
class _SampleRef:
    source: str
    source_parent_id: str
    semantic_parent_id: str
    fold: str
    role: str
    index: int

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _TaskRef:
    source: str
    source_parent_id: str
    semantic_parent_id: str
    fold: str

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def _strict_json(content: bytes, *, label: str) -> Any:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
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
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {label}: {exc}") from exc


def _validate_grid(value: object, *, label: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty grid")
    rows: list[tuple[int, ...]] = []
    width: int | None = None
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or not row:
            raise ValueError(f"{label}[{row_index}] must be a non-empty row")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"{label} must be rectangular")
        normalized: list[int] = []
        for column_index, color in enumerate(row):
            if type(color) is not int or not 0 <= color <= 9:
                raise ValueError(
                    f"{label}[{row_index}][{column_index}] must be integer color 0-9"
                )
            normalized.append(color)
        rows.append(tuple(normalized))
    return tuple(rows)


def _parse_rearc_examples(
    content: bytes, *, member_name: str, expected_count: int
) -> tuple[tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]], ...]:
    payload = _strict_json(content, label=member_name)
    if not isinstance(payload, list) or len(payload) != expected_count:
        found = len(payload) if isinstance(payload, list) else type(payload).__name__
        raise ValueError(
            f"{member_name} must contain exactly {expected_count} examples, found {found}"
        )
    result = []
    for index, example in enumerate(payload):
        if not isinstance(example, dict) or set(example) != {"input", "output"}:
            raise ValueError(
                f"{member_name}[{index}] must contain exactly input and output"
            )
        input_grid = _validate_grid(example["input"], label=f"{member_name}[{index}].input")
        output_grid = _validate_grid(
            example["output"], label=f"{member_name}[{index}].output"
        )
        result.append((input_grid, output_grid))
    return tuple(result)


def _validate_zip_name(info: zipfile.ZipInfo) -> None:
    name = info.filename
    if not name or "\\" in name or ":" in name or any(ord(char) < 32 for char in name):
        raise ValueError(f"unsafe ReARC ZIP member: {name!r}")
    trimmed = name[:-1] if info.is_dir() and name.endswith("/") else name
    path = PurePosixPath(trimmed)
    if (
        not trimmed
        or path.is_absolute()
        or path.as_posix() != trimmed
        or any(part in {"", ".", ".."} for part in path.parts)
        or stat.S_ISLNK(info.external_attr >> 16)
        or info.flag_bits & 0x1
        or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
    ):
        raise ValueError(f"unsafe ReARC ZIP member: {name!r}")


def audit_rearc_archive_bytes(
    archive_bytes: bytes,
    *,
    expected_parent_ids: Sequence[str],
    spec: PrivilegedSplitSpec,
) -> RearcArchiveAudit:
    """Validate the complete ReARC archive without retaining decompressed members."""

    if not isinstance(archive_bytes, bytes):
        raise TypeError("archive_bytes must be bytes")
    actual_sha = hashlib.sha256(archive_bytes).hexdigest()
    if spec.rearc_archive_sha256 is not None and actual_sha != spec.rearc_archive_sha256:
        raise ValueError("ReARC archive SHA-256 mismatch")
    if spec.rearc_archive_bytes is not None and len(archive_bytes) != spec.rearc_archive_bytes:
        raise ValueError("ReARC archive byte count mismatch")
    expected_ids = tuple(sorted(expected_parent_ids))
    if len(expected_ids) != spec.arc1_task_count or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("expected ReARC parent IDs do not close to ARC1 task count")
    if any(not re.fullmatch(r"[0-9a-f]{8}", parent_id) for parent_id in expected_ids):
        raise ValueError("expected ReARC parent IDs must be lowercase 8-hex")

    audits: list[RearcMemberAudit] = []
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), mode="r") as archive:
            infos = archive.infolist()
            if spec.expected_zip_entry_count is not None and len(infos) != spec.expected_zip_entry_count:
                raise ValueError("ReARC ZIP entry count mismatch")
            names: set[str] = set()
            accepted: dict[str, zipfile.ZipInfo] = {}
            total_uncompressed = 0
            for info in infos:
                _validate_zip_name(info)
                if info.filename in names:
                    raise ValueError(f"duplicate ReARC ZIP member: {info.filename}")
                names.add(info.filename)
                total_uncompressed += info.file_size
                if info.file_size > spec.max_rearc_member_bytes:
                    raise ValueError(f"ReARC ZIP member exceeds size limit: {info.filename}")
                if total_uncompressed > spec.max_rearc_uncompressed_bytes:
                    raise ValueError("ReARC ZIP exceeds total uncompressed size limit")
                match = _TASK_MEMBER_RE.fullmatch(info.filename)
                if match is not None:
                    if info.is_dir():
                        raise ValueError(f"ReARC task member is a directory: {info.filename}")
                    parent_id = match.group(1)
                    if parent_id in accepted:
                        raise ValueError(f"duplicate ReARC parent member: {parent_id}")
                    accepted[parent_id] = info
                    continue
                allowed_metadata = (
                    info.filename in _ALLOWED_REARC_METADATA
                    or _RESOURCE_MEMBER_RE.fullmatch(info.filename) is not None
                )
                if not allowed_metadata:
                    raise ValueError(f"unexpected ReARC ZIP member: {info.filename}")
            if tuple(sorted(accepted)) != expected_ids:
                missing = sorted(set(expected_ids) - set(accepted))
                extra = sorted(set(accepted) - set(expected_ids))
                raise ValueError(
                    f"ReARC task member IDs mismatch: missing={missing}, extra={extra}"
                )
            for parent_id in expected_ids:
                info = accepted[parent_id]
                content = archive.read(info)
                examples = _parse_rearc_examples(
                    content,
                    member_name=info.filename,
                    expected_count=spec.rearc_examples_per_task,
                )
                max_height = max(max(len(inp), len(out)) for inp, out in examples)
                max_width = max(max(len(inp[0]), len(out[0])) for inp, out in examples)
                oversized_examples = sum(
                    max(len(inp), len(inp[0]), len(out), len(out[0])) > spec.max_grid_side
                    for inp, out in examples
                )
                oversized_grids = sum(
                    int(max(len(inp), len(inp[0])) > spec.max_grid_side)
                    + int(max(len(out), len(out[0])) > spec.max_grid_side)
                    for inp, out in examples
                )
                audits.append(
                    RearcMemberAudit(
                        parent_id=parent_id,
                        member_name=info.filename,
                        sha256=hashlib.sha256(content).hexdigest(),
                        byte_count=len(content),
                        example_count=len(examples),
                        maximum_height=max_height,
                        maximum_width=max_width,
                        oversized_example_count=oversized_examples,
                        oversized_grid_count=oversized_grids,
                    )
                )
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        raise ValueError(f"invalid ReARC ZIP: {exc}") from exc
    return RearcArchiveAudit(
        archive_sha256=actual_sha,
        archive_bytes=len(archive_bytes),
        zip_entry_count=len(infos),
        member_ids=expected_ids,
        member_audits=tuple(audits),
    )


def _content_id(label: str, payload: object) -> str:
    return hashlib.sha256(label.encode("ascii") + b"\0" + canonical_json_bytes(payload)).hexdigest()


def _arc2_initial_folds(
    tasks: Mapping[str, ARCTask], *, spec: PrivilegedSplitSpec
) -> tuple[dict[str, str], tuple[str, ...]]:
    groups: defaultdict[str, list[tuple[BlindTask, ARCTask]]] = defaultdict(list)
    for task in tasks.values():
        if any(pair.output is None for pair in task.test):
            raise ValueError(f"ARC2 public training task lacks test output: {task.task_id}")
        blind = BlindTask.from_task(task)
        groups[blind.semantic_fingerprint_sha256].append((blind, task))
    duplicate_ids = tuple(
        sorted(item[1].task_id for group in groups.values() if len(group) > 1 for item in group)
    )
    if (
        spec.expected_exact_semantic_quarantine_count is not None
        and len(duplicate_ids) != spec.expected_exact_semantic_quarantine_count
    ):
        raise ValueError("ARC2 exact-semantic quarantine count mismatch")
    eligible = [group[0] for group in groups.values() if len(group) == 1]
    required = spec.holdout_count + spec.validation_count + spec.smoke_count
    if len(eligible) < required:
        raise ValueError("not enough singleton ARC2 tasks for frozen folds")

    def split_key(item: tuple[BlindTask, ARCTask]) -> tuple[str, str]:
        return (
            _content_id(
                ARC2_SPLIT_SALT,
                {"blind_content_sha256": item[0].blind_content_sha256},
            ),
            item[0].task_id,
        )

    assigned = sorted(eligible, key=split_key)
    holdout = assigned[: spec.holdout_count]
    validation = assigned[spec.holdout_count : spec.holdout_count + spec.validation_count]
    development = assigned[spec.holdout_count + spec.validation_count :]

    def smoke_key(item: tuple[BlindTask, ARCTask]) -> tuple[str, str]:
        return (
            _content_id(
                PUBLIC_SMOKE_SELECTION_SALT,
                {"blind_content_sha256": item[0].blind_content_sha256},
            ),
            item[0].task_id,
        )

    smoke_ids = {
        item[1].task_id
        for item in sorted(development, key=smoke_key)[: spec.smoke_count]
    }
    folds = {item[1].task_id: "holdout" for item in holdout}
    folds.update({item[1].task_id: "validation" for item in validation})
    folds.update(
        {
            item[1].task_id: ("smoke" if item[1].task_id in smoke_ids else "train")
            for item in development
        }
    )
    return folds, duplicate_ids


def _load_pinned_tasks(
    repo_dir: str | Path,
    *,
    origin: str,
    commit: str,
    dataset_name: str,
    expected_count: int,
) -> tuple[GitRepoState, dict[str, ARCTask], dict[str, bytes]]:
    root = Path(repo_dir).expanduser().resolve()
    state = git_repo_state(
        root,
        expected_origin=origin,
        expected_commit=commit,
        require_clean=True,
    )
    snapshot = snapshot_task_directory(
        root / "data" / "training",
        dataset_name=dataset_name,
        split_name="public_training",
        repo_dir=root,
        expected_origin=origin,
        expected_commit=commit,
    )
    if len(snapshot.tasks) != expected_count:
        raise ValueError(f"{dataset_name} training task count mismatch")
    if any(_ARC_TASK_NAME_RE.fullmatch(name) is None for name in snapshot.selected_file_names):
        raise ValueError(f"{dataset_name} task filenames must be lowercase 8-hex JSON")
    tasks = {task.task_id: task for task in snapshot.tasks}
    raw = {
        name[:-5]: git_blob_bytes(state, f"data/training/{name}")
        for name in snapshot.selected_file_names
    }
    return state, tasks, raw


def _rearc_parents(
    *,
    arc1_tasks: Mapping[str, ARCTask],
    arc2_tasks: Mapping[str, ARCTask],
    arc2_folds: Mapping[str, str],
) -> tuple[dict[str, _RearcParent], tuple[tuple[str, str], ...]]:
    semantic_to_arc2: defaultdict[str, list[str]] = defaultdict(list)
    for task in arc2_tasks.values():
        semantic_to_arc2[task_semantic_fingerprint(task, include_test_outputs=True)].append(
            task.task_id
        )
    records: dict[str, _RearcParent] = {}
    aliases: list[tuple[str, str]] = []
    for parent_id in sorted(arc1_tasks):
        alias_target: str | None = None
        if parent_id in arc2_folds:
            semantic_parent = parent_id
            fold = arc2_folds[parent_id]
            inheritance = "exact_parent_id"
        elif parent_id in arc2_tasks:
            raise ValueError(f"ReARC parent maps to quarantined ARC2 duplicate: {parent_id}")
        else:
            fingerprint = task_semantic_fingerprint(
                arc1_tasks[parent_id], include_test_outputs=True
            )
            matches = semantic_to_arc2.get(fingerprint, [])
            if len(matches) > 1:
                raise ValueError(f"ambiguous ARC2 semantic aliases for ReARC parent {parent_id}")
            if matches:
                alias_target = matches[0]
                if alias_target not in arc2_folds:
                    raise ValueError(f"ReARC alias targets quarantined ARC2 task: {parent_id}")
                semantic_parent = alias_target
                fold = arc2_folds[alias_target]
                inheritance = "full_semantic_alias"
                aliases.append((parent_id, alias_target))
            else:
                semantic_parent = parent_id
                bucket = rearc_parent_split_bucket(parent_id)
                fold = "train" if bucket < 80 else "validation" if bucket < 90 else "internal"
                inheritance = f"external_hash_bucket_{bucket}"
        records[parent_id] = _RearcParent(
            source_parent_id=parent_id,
            semantic_parent_id=semantic_parent,
            initial_fold=fold,
            inheritance=inheritance,
            alias_target_id=alias_target,
        )
    return records, tuple(sorted(aliases))


def _task_pairs(task: ARCTask) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    def payload(pair: ARCPair) -> dict[str, object]:
        if pair.output is None:
            raise ValueError(f"task orbit requires output for task {task.task_id}")
        return {"input": pair.input, "output": pair.output}

    return [payload(pair) for pair in task.train], [payload(pair) for pair in task.test]


def _task_sources(
    *,
    fold: str,
    arc1_tasks: Mapping[str, ARCTask],
    arc2_tasks: Mapping[str, ARCTask],
    arc2_folds: Mapping[str, str],
    rearc_parents: Mapping[str, _RearcParent],
) -> Iterable[tuple[_TaskRef, ARCTask]]:
    for task_id in sorted(task_id for task_id, item_fold in arc2_folds.items() if item_fold == fold):
        yield _TaskRef("arc2", task_id, task_id, fold), arc2_tasks[task_id]
    for source_id in sorted(
        source_id for source_id, record in rearc_parents.items() if record.initial_fold == fold
    ):
        record = rearc_parents[source_id]
        yield (
            _TaskRef("arc1_parent", source_id, record.semantic_parent_id, fold),
            arc1_tasks[source_id],
        )


def _original_sample_sources(
    *,
    fold: str,
    arc2_tasks: Mapping[str, ARCTask],
    arc2_folds: Mapping[str, str],
) -> Iterable[tuple[_SampleRef, tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]]:
    for task_id in sorted(task_id for task_id, item_fold in arc2_folds.items() if item_fold == fold):
        task = arc2_tasks[task_id]
        for role, pairs in (("train", task.train), ("test", task.test)):
            for index, pair in enumerate(pairs):
                if pair.output is None:
                    raise ValueError(f"sample audit requires output for ARC2 task {task_id}")
                yield (
                    _SampleRef("arc2", task_id, task_id, fold, role, index),
                    pair.input,
                    pair.output,
                )


def _iter_rearc_samples(
    archive: zipfile.ZipFile,
    *,
    fold: str,
    rearc_parents: Mapping[str, _RearcParent],
    spec: PrivilegedSplitSpec,
) -> Iterable[tuple[_SampleRef, tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]]:
    for source_id in sorted(
        source_id for source_id, record in rearc_parents.items() if record.initial_fold == fold
    ):
        record = rearc_parents[source_id]
        member_name = f"re_arc/tasks/{source_id}.json"
        examples = _parse_rearc_examples(
            archive.read(member_name),
            member_name=member_name,
            expected_count=spec.rearc_examples_per_task,
        )
        for index, (input_grid, output_grid) in enumerate(examples):
            yield (
                _SampleRef(
                    "rearc", source_id, record.semantic_parent_id, fold, "example", index
                ),
                input_grid,
                output_grid,
            )


def _collision_row(
    *,
    kind: str,
    orbit_id: str,
    consumer_fold: str,
    source: _SampleRef | _TaskRef,
    target: _SampleRef | _TaskRef,
) -> dict[str, object]:
    row: dict[str, object] = {
        "schema": COLLISION_ROW_SCHEMA,
        "kind": kind,
        "orbit_semantics_version": (
            SAMPLE_ORBIT_SEMANTICS_VERSION if kind == "sample_orbit" else TASK_ORBIT_SEMANTICS_VERSION
        ),
        "orbit_id": orbit_id,
        "consumer_fold": consumer_fold,
        "source": source.to_json_dict(),
        "target": target.to_json_dict(),
    }
    row["collision_id"] = canonical_sha256(row)
    return row


def _collect_orbit_quarantines(
    *,
    archive_bytes: bytes,
    arc1_tasks: Mapping[str, ARCTask],
    arc2_tasks: Mapping[str, ARCTask],
    arc2_folds: Mapping[str, str],
    rearc_parents: Mapping[str, _RearcParent],
    spec: PrivilegedSplitSpec,
) -> tuple[set[str], set[str], list[dict[str, object]]]:
    collisions: dict[str, dict[str, object]] = {}
    validation_quarantine: set[str] = set()
    training_quarantine: set[str] = set()

    protected_task_orbits: defaultdict[str, list[_TaskRef]] = defaultdict(list)
    validation_task_orbits: defaultdict[str, list[_TaskRef]] = defaultdict(list)
    for fold, target in (
        ("holdout", protected_task_orbits),
        ("smoke", protected_task_orbits),
        ("validation", validation_task_orbits),
    ):
        for ref, task in _task_sources(
            fold=fold,
            arc1_tasks=arc1_tasks,
            arc2_tasks=arc2_tasks,
            arc2_folds=arc2_folds,
            rearc_parents=rearc_parents,
        ):
            train, test = _task_pairs(task)
            target[task_orbit_id(train, test)].append(ref)

    def scan_tasks(
        fold: str,
        targets: Mapping[str, list[_TaskRef]],
        quarantine: set[str],
    ) -> None:
        for ref, task in _task_sources(
            fold=fold,
            arc1_tasks=arc1_tasks,
            arc2_tasks=arc2_tasks,
            arc2_folds=arc2_folds,
            rearc_parents=rearc_parents,
        ):
            train, test = _task_pairs(task)
            orbit = task_orbit_id(train, test)
            for target_ref in targets.get(orbit, []):
                quarantine.add(ref.semantic_parent_id)
                row = _collision_row(
                    kind="task_orbit",
                    orbit_id=orbit,
                    consumer_fold=fold,
                    source=ref,
                    target=target_ref,
                )
                collisions[str(row["collision_id"])] = row

    scan_tasks("validation", protected_task_orbits, validation_quarantine)
    combined_task_targets: defaultdict[str, list[_TaskRef]] = defaultdict(list)
    for mapping in (protected_task_orbits, validation_task_orbits):
        for orbit, refs in mapping.items():
            combined_task_targets[orbit].extend(refs)
    scan_tasks("train", combined_task_targets, training_quarantine)

    protected_sample_orbits: defaultdict[str, list[_SampleRef]] = defaultdict(list)
    validation_sample_orbits: defaultdict[str, list[_SampleRef]] = defaultdict(list)
    with zipfile.ZipFile(io.BytesIO(archive_bytes), mode="r") as archive:
        for fold, target in (
            ("holdout", protected_sample_orbits),
            ("smoke", protected_sample_orbits),
            ("validation", validation_sample_orbits),
        ):
            for ref, input_grid, output_grid in _original_sample_sources(
                fold=fold, arc2_tasks=arc2_tasks, arc2_folds=arc2_folds
            ):
                target[sample_orbit_id(input_grid, output_grid)].append(ref)
            for ref, input_grid, output_grid in _iter_rearc_samples(
                archive,
                fold=fold,
                rearc_parents=rearc_parents,
                spec=spec,
            ):
                target[sample_orbit_id(input_grid, output_grid)].append(ref)

        def scan_samples(
            fold: str,
            targets: Mapping[str, list[_SampleRef]],
            quarantine: set[str],
        ) -> None:
            streams = (
                _original_sample_sources(
                    fold=fold, arc2_tasks=arc2_tasks, arc2_folds=arc2_folds
                ),
                _iter_rearc_samples(
                    archive,
                    fold=fold,
                    rearc_parents=rearc_parents,
                    spec=spec,
                ),
            )
            for stream in streams:
                for ref, input_grid, output_grid in stream:
                    orbit = sample_orbit_id(input_grid, output_grid)
                    for target_ref in targets.get(orbit, []):
                        quarantine.add(ref.semantic_parent_id)
                        row = _collision_row(
                            kind="sample_orbit",
                            orbit_id=orbit,
                            consumer_fold=fold,
                            source=ref,
                            target=target_ref,
                        )
                        collisions[str(row["collision_id"])] = row

        scan_samples("validation", protected_sample_orbits, validation_quarantine)
        combined_sample_targets: defaultdict[str, list[_SampleRef]] = defaultdict(list)
        for mapping in (protected_sample_orbits, validation_sample_orbits):
            for orbit, refs in mapping.items():
                combined_sample_targets[orbit].extend(refs)
        scan_samples("train", combined_sample_targets, training_quarantine)

    return (
        validation_quarantine,
        training_quarantine,
        sorted(collisions.values(), key=lambda row: str(row["collision_id"])),
    )


def _named_bytes_sha256(materials: Iterable[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for name, content in sorted(materials):
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _deterministic_zip(materials: Mapping[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name, content in sorted(materials.items()):
            if not re.fullmatch(
                r"(?:arc2|rearc)/(?:train|validation)/[0-9a-f]{8}\.json", name
            ):
                raise ValueError(f"unsafe sanitized shard member: {name}")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return buffer.getvalue()


def _shard(
    *,
    source: str,
    fold: str,
    ids: Sequence[str],
    raw_by_id: Mapping[str, bytes],
    semantic_by_source_id: Mapping[str, str],
) -> tuple[bytes, dict[str, object]]:
    ordered_ids = tuple(sorted(ids))
    materials = {f"{source}/{fold}/{source_id}.json": raw_by_id[source_id] for source_id in ordered_ids}
    members = [
        {
            "path": name,
            "source_parent_id": name.rsplit("/", 1)[1][:-5],
            "semantic_parent_id": semantic_by_source_id[name.rsplit("/", 1)[1][:-5]],
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
        for name, content in sorted(materials.items())
    ]
    archive_bytes = _deterministic_zip(materials)
    return archive_bytes, {
        "source": source,
        "fold": fold,
        "member_prefix": f"{source}/{fold}/",
        "compression": "ZIP_DEFLATED_level_9",
        "zip_member_timestamp": "1980-01-01T00:00:00",
        "zip_member_mode": "0100644",
        "member_count": len(members),
        "ordered_source_parent_ids": list(ordered_ids),
        "member_aggregate_sha256": _named_bytes_sha256(materials.items()),
        "members": members,
        "zip_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "zip_bytes": len(archive_bytes),
    }


def _artifact_manifest_payload(artifacts: Mapping[str, bytes], *, run_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "bundle_status": "complete",
        "run_id": run_id,
        "artifacts": {
            name: {
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "rows": content.count(b"\n") if name.endswith(".jsonl") else None,
            }
            for name, content in sorted(artifacts.items())
        },
    }


def _require_separate_new_outputs(
    sealed_output_dir: str | Path,
    sanitized_output_dir: str | Path,
    *,
    source_roots: Sequence[Path],
) -> tuple[Path, Path]:
    sealed = Path(sealed_output_dir).expanduser().resolve()
    sanitized = Path(sanitized_output_dir).expanduser().resolve()
    if sealed == sanitized or sealed.is_relative_to(sanitized) or sanitized.is_relative_to(sealed):
        raise ValueError("sealed and sanitized output roots must be non-overlapping")
    if sealed.exists() or sanitized.exists():
        raise FileExistsError("refusing to overwrite split compiler output root")
    for output in (sealed, sanitized):
        for source in source_roots:
            if output == source or output.is_relative_to(source):
                raise ValueError("split compiler outputs must not be inside source repositories")
    return sealed, sanitized


def _counts(records: Iterable[_RearcParent]) -> dict[str, int]:
    result: defaultdict[str, int] = defaultdict(int)
    for record in records:
        result[record.initial_fold] += 1
    return dict(sorted(result.items()))


def _check_expected(actual: object, expected: object, *, label: str) -> None:
    if expected is not None and actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, found {actual!r}")


def compile_privileged_split(
    *,
    arc1_repo: str | Path,
    arc2_repo: str | Path,
    rearc_repo: str | Path,
    sealed_output_dir: str | Path,
    sanitized_output_dir: str | Path,
    spec: PrivilegedSplitSpec | None = None,
) -> dict[str, object]:
    """Compile sealed audit evidence and physically cropped sanitized shards."""

    selected_spec = spec or PrivilegedSplitSpec()
    source_roots = tuple(
        Path(path).expanduser().resolve() for path in (arc1_repo, arc2_repo, rearc_repo)
    )
    sealed_target, sanitized_target = _require_separate_new_outputs(
        sealed_output_dir,
        sanitized_output_dir,
        source_roots=source_roots,
    )

    arc1_state, arc1_tasks, arc1_raw = _load_pinned_tasks(
        source_roots[0],
        origin=selected_spec.arc1_origin,
        commit=selected_spec.arc1_commit,
        dataset_name="ARC-AGI-1",
        expected_count=selected_spec.arc1_task_count,
    )
    arc2_state, arc2_tasks, arc2_raw = _load_pinned_tasks(
        source_roots[1],
        origin=selected_spec.arc2_origin,
        commit=selected_spec.arc2_commit,
        dataset_name="ARC-AGI-2",
        expected_count=selected_spec.arc2_task_count,
    )
    rearc_state = git_repo_state(
        source_roots[2],
        expected_origin=selected_spec.rearc_origin,
        expected_commit=selected_spec.rearc_commit,
        require_clean=True,
    )
    rearc_archive_bytes = git_blob_bytes(rearc_state, "re_arc.zip")
    archive_audit = audit_rearc_archive_bytes(
        rearc_archive_bytes,
        expected_parent_ids=tuple(arc1_tasks),
        spec=selected_spec,
    )

    arc2_folds, duplicate_arc2_ids = _arc2_initial_folds(arc2_tasks, spec=selected_spec)
    rearc_records, aliases = _rearc_parents(
        arc1_tasks=arc1_tasks,
        arc2_tasks=arc2_tasks,
        arc2_folds=arc2_folds,
    )
    _check_expected(
        aliases,
        selected_spec.expected_aliases,
        label="ReARC semantic aliases",
    )
    initial_rearc_counts = tuple(sorted(_counts(rearc_records.values()).items()))
    _check_expected(
        initial_rearc_counts,
        selected_spec.expected_initial_rearc_counts,
        label="initial ReARC fold counts",
    )

    validation_orbit_quarantine, training_orbit_quarantine, collision_rows = (
        _collect_orbit_quarantines(
            archive_bytes=rearc_archive_bytes,
            arc1_tasks=arc1_tasks,
            arc2_tasks=arc2_tasks,
            arc2_folds=arc2_folds,
            rearc_parents=rearc_records,
            spec=selected_spec,
        )
    )
    dimension_source_ids = {
        audit.parent_id for audit in archive_audit.member_audits if audit.oversized_example_count
    }
    dimension_semantic_ids = {
        rearc_records[source_id].semantic_parent_id for source_id in dimension_source_ids
    }
    validation_dimension_quarantine = {
        parent_id
        for parent_id in dimension_semantic_ids
        if any(
            record.semantic_parent_id == parent_id and record.initial_fold == "validation"
            for record in rearc_records.values()
        )
    }
    training_dimension_quarantine = {
        parent_id
        for parent_id in dimension_semantic_ids
        if any(
            record.semantic_parent_id == parent_id and record.initial_fold == "train"
            for record in rearc_records.values()
        )
    }
    validation_quarantine = validation_orbit_quarantine | validation_dimension_quarantine
    training_quarantine = training_orbit_quarantine | training_dimension_quarantine

    _check_expected(
        tuple(sorted(validation_orbit_quarantine)),
        selected_spec.expected_validation_quarantine_ids,
        label="validation orbit quarantine IDs",
    )
    _check_expected(
        tuple(sorted(training_orbit_quarantine)),
        selected_spec.expected_training_orbit_quarantine_ids,
        label="training orbit quarantine IDs",
    )
    _check_expected(
        tuple(sorted(dimension_semantic_ids)),
        selected_spec.expected_dimension_quarantine_ids,
        label="dimension quarantine IDs",
    )

    arc2_train_ids = tuple(
        sorted(
            task_id
            for task_id, fold in arc2_folds.items()
            if fold == "train" and task_id not in training_quarantine
        )
    )
    arc2_validation_ids = tuple(
        sorted(
            task_id
            for task_id, fold in arc2_folds.items()
            if fold == "validation" and task_id not in validation_quarantine
        )
    )
    rearc_train_ids = tuple(
        sorted(
            source_id
            for source_id, record in rearc_records.items()
            if record.initial_fold == "train"
            and record.semantic_parent_id not in training_quarantine
        )
    )
    rearc_validation_ids = tuple(
        sorted(
            source_id
            for source_id, record in rearc_records.items()
            if record.initial_fold == "validation"
            and record.semantic_parent_id not in validation_quarantine
        )
    )
    final_counts = {
        "arc2_train": len(arc2_train_ids),
        "arc2_validation": len(arc2_validation_ids),
        "rearc_train": len(rearc_train_ids),
        "rearc_validation": len(rearc_validation_ids),
    }
    _check_expected(
        tuple(sorted(final_counts.items())),
        selected_spec.expected_final_counts,
        label="final sanitized fold counts",
    )

    rearc_raw: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(rearc_archive_bytes), mode="r") as archive:
        for source_id in (*rearc_train_ids, *rearc_validation_ids):
            rearc_raw[source_id] = archive.read(f"re_arc/tasks/{source_id}.json")
    arc2_semantic = {task_id: task_id for task_id in arc2_tasks}
    rearc_semantic = {
        source_id: record.semantic_parent_id for source_id, record in rearc_records.items()
    }
    shard_inputs = (
        ("arc2_train.zip", "arc2", "train", arc2_train_ids, arc2_raw, arc2_semantic),
        (
            "arc2_validation.zip",
            "arc2",
            "validation",
            arc2_validation_ids,
            arc2_raw,
            arc2_semantic,
        ),
        ("rearc_train.zip", "rearc", "train", rearc_train_ids, rearc_raw, rearc_semantic),
        (
            "rearc_validation.zip",
            "rearc",
            "validation",
            rearc_validation_ids,
            rearc_raw,
            rearc_semantic,
        ),
    )
    shard_artifacts: dict[str, bytes] = {}
    shard_manifests: dict[str, dict[str, object]] = {}
    for artifact_name, source, fold, ids, raw, semantic in shard_inputs:
        shard_bytes, shard_manifest = _shard(
            source=source,
            fold=fold,
            ids=ids,
            raw_by_id=raw,
            semantic_by_source_id=semantic,
        )
        shard_artifacts[artifact_name] = shard_bytes
        shard_manifests[artifact_name] = shard_manifest

    source_rows: list[dict[str, object]] = []
    for source, raw in (("arc1", arc1_raw), ("arc2", arc2_raw)):
        for parent_id, content in sorted(raw.items()):
            source_rows.append(
                {
                    "schema": SOURCE_ROW_SCHEMA,
                    "source": source,
                    "parent_id": parent_id,
                    "path": f"data/training/{parent_id}.json",
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "bytes": len(content),
                }
            )
    for audit in archive_audit.member_audits:
        source_rows.append(
            {
                "schema": SOURCE_ROW_SCHEMA,
                "source": "rearc",
                **asdict(audit),
            }
        )

    split_rows: list[dict[str, object]] = []
    for task_id, fold in sorted(arc2_folds.items()):
        quarantine = (
            "training" if task_id in training_quarantine else "validation" if task_id in validation_quarantine else None
        )
        split_rows.append(
            {
                "schema": SPLIT_ROW_SCHEMA,
                "source": "arc2",
                "source_parent_id": task_id,
                "semantic_parent_id": task_id,
                "initial_fold": fold,
                "final_status": f"quarantined_{quarantine}" if quarantine else fold,
                "inheritance": "authoritative_arc2_split",
            }
        )
    for source_id, record in sorted(rearc_records.items()):
        quarantine = (
            "training"
            if record.semantic_parent_id in training_quarantine
            else "validation"
            if record.semantic_parent_id in validation_quarantine
            else None
        )
        split_rows.append(
            {
                "schema": SPLIT_ROW_SCHEMA,
                "source": "rearc",
                **asdict(record),
                "final_status": f"quarantined_{quarantine}" if quarantine else record.initial_fold,
            }
        )

    privileged_manifest = {
        "schema": PRIVILEGED_MANIFEST_SCHEMA,
        "data_folds_semantics_version": DATA_FOLDS_SEMANTICS_VERSION,
        "task_orbit_semantics_version": TASK_ORBIT_SEMANTICS_VERSION,
        "sample_orbit_semantics_version": SAMPLE_ORBIT_SEMANTICS_VERSION,
        "source_states": {
            "arc1": asdict(arc1_state),
            "arc2": asdict(arc2_state),
            "rearc": asdict(rearc_state),
        },
        "rearc_archive": {
            "sha256": archive_audit.archive_sha256,
            "bytes": archive_audit.archive_bytes,
            "zip_entry_count": archive_audit.zip_entry_count,
            "accepted_member_count": len(archive_audit.member_ids),
        },
        "arc2_exact_semantic_quarantine_ids": list(duplicate_arc2_ids),
        "semantic_aliases": [
            {"rearc_parent_id": source_id, "arc2_parent_id": target_id}
            for source_id, target_id in aliases
        ],
        "initial_counts": {
            "arc2": dict(
                sorted(
                    (fold, sum(value == fold for value in arc2_folds.values()))
                    for fold in ("train", "validation", "holdout", "smoke")
                )
            ),
            "rearc": dict(initial_rearc_counts),
        },
        "validation_quarantine_ids": sorted(validation_quarantine),
        "training_orbit_quarantine_ids": sorted(training_orbit_quarantine),
        "dimension_quarantine_ids": sorted(dimension_semantic_ids),
        "training_quarantine_ids": sorted(training_quarantine),
        "final_counts": final_counts,
        "collision_row_count": len(collision_rows),
        "source_row_count": len(source_rows),
        "split_row_count": len(split_rows),
    }
    sealed_artifacts = {
        "privileged_manifest.json": serialize_json(privileged_manifest),
        "collisions.jsonl": serialize_jsonl(collision_rows),
        "source_files.jsonl": serialize_jsonl(source_rows),
        "split_rows.jsonl": serialize_jsonl(split_rows),
    }
    sealed_run_id = _content_id(
        "afts-m04a-privileged-split-bundle/v1",
        {name: hashlib.sha256(content).hexdigest() for name, content in sealed_artifacts.items()},
    )[:20]
    expected_sealed_artifact_manifest = _artifact_manifest_payload(
        sealed_artifacts, run_id=sealed_run_id
    )
    sealed_commitment_sha256 = hashlib.sha256(
        serialize_json(expected_sealed_artifact_manifest)
    ).hexdigest()

    sanitized_manifest = {
        "schema": SANITIZED_MANIFEST_SCHEMA,
        "data_folds_semantics_version": DATA_FOLDS_SEMANTICS_VERSION,
        "sealed_artifact_manifest_sha256": sealed_commitment_sha256,
        "source_commitments": {
            "arc1": {"origin": arc1_state.origin, "commit": arc1_state.commit},
            "arc2": {"origin": arc2_state.origin, "commit": arc2_state.commit},
            "rearc": {
                "origin": rearc_state.origin,
                "commit": rearc_state.commit,
                "archive_sha256": archive_audit.archive_sha256,
                "archive_bytes": archive_audit.archive_bytes,
            },
        },
        "usable_fold_counts": final_counts,
        "ordered_allowlists": {
            "arc2_train": list(arc2_train_ids),
            "arc2_validation": list(arc2_validation_ids),
            "rearc_train": list(rearc_train_ids),
            "rearc_validation": list(rearc_validation_ids),
        },
        "semantic_aliases_in_usable_shards": [
            {"rearc_parent_id": source_id, "arc2_parent_id": target_id}
            for source_id, target_id in aliases
            if source_id in {*rearc_train_ids, *rearc_validation_ids}
        ],
        "validation_quarantine_ids": sorted(validation_quarantine),
        "training_quarantine_ids": sorted(training_quarantine),
        "quarantine_reasons": {
            parent_id: sorted(
                reason
                for reason, members in (
                    ("sample_or_task_orbit", validation_orbit_quarantine | training_orbit_quarantine),
                    ("grid_dimension", dimension_semantic_ids),
                )
                if parent_id in members
            )
            for parent_id in sorted(validation_quarantine | training_quarantine)
        },
        "shards": shard_manifests,
        "protected_payloads_absent": True,
    }
    sanitized_artifacts = {
        **shard_artifacts,
        "data_split_manifest.json": serialize_json(sanitized_manifest),
    }
    sanitized_run_id = _content_id(
        "afts-m04a-sanitized-split-bundle/v1",
        {name: hashlib.sha256(content).hexdigest() for name, content in sanitized_artifacts.items()},
    )[:20]

    sealed_manifest = publish_evidence_bundle(
        sealed_target,
        artifacts=sealed_artifacts,
        run_id=sealed_run_id,
    )
    if sealed_manifest != expected_sealed_artifact_manifest:
        raise RuntimeError("sealed artifact manifest differs from precomputed commitment")
    on_disk_sealed_commitment = hashlib.sha256(
        (sealed_target / "artifact_manifest.json").read_bytes()
    ).hexdigest()
    if on_disk_sealed_commitment != sealed_commitment_sha256:
        raise RuntimeError("sealed artifact manifest commitment mismatch")
    sanitized_manifest_result = publish_evidence_bundle(
        sanitized_target,
        artifacts=sanitized_artifacts,
        run_id=sanitized_run_id,
    )
    return {
        "status": "published_privileged_and_sanitized_split_bundles",
        "sealed_output_dir": str(sealed_target),
        "sanitized_output_dir": str(sanitized_target),
        "sealed_artifact_manifest_sha256": sealed_commitment_sha256,
        "sealed_artifact_manifest": sealed_manifest,
        "sanitized_artifact_manifest": sanitized_manifest_result,
        "final_counts": final_counts,
    }


__all__ = [
    "COLLISION_ROW_SCHEMA",
    "FROZEN_DIMENSION_QUARANTINE_IDS",
    "FROZEN_TRAINING_ORBIT_QUARANTINE_IDS",
    "FROZEN_TRAINING_QUARANTINE_IDS",
    "FROZEN_VALIDATION_QUARANTINE_IDS",
    "PRIVILEGED_MANIFEST_SCHEMA",
    "PrivilegedSplitSpec",
    "REARC_ARCHIVE_BYTES",
    "REARC_ARCHIVE_SHA256",
    "REARC_COMMIT",
    "REARC_ORIGIN",
    "RearcArchiveAudit",
    "RearcMemberAudit",
    "SANITIZED_MANIFEST_SCHEMA",
    "audit_rearc_archive_bytes",
    "compile_privileged_split",
]
