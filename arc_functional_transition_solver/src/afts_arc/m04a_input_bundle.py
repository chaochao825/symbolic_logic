"""Materialize the immutable, closed-world input bundle for M04a.

The module is intentionally PyTorch-free.  Every copied input is read through an
externally supplied SHA-256 commitment.  Every JSON file generated here is a
deterministic projection of the pure M04a contract or of the committed sanitized
and validation bundles.  The launch plan lives in ``control/`` and is therefore
never part of the training-visible input directory.
"""

from __future__ import annotations

import ast
import ctypes
import errno
import hashlib
import io
import json
import os
import re
import stat
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from .m04a_contract import (
    ATTENTION_HEADS,
    DATA_FOLDS_SEMANTICS_VERSION,
    DECODER_FORWARD_LEDGER_SCHEMA_VERSION,
    DECODER_LAYER_COUNT,
    DENOISING_STEPS,
    D_MODEL,
    DROPOUT_PROBABILITY,
    ENCODER_FORWARD_LEDGER_SCHEMA_VERSION,
    ENCODER_LAYER_COUNT,
    EPISODE_SEMANTICS_VERSION,
    FFN_WIDTH,
    GRADIENT_ACCUMULATION,
    INFERENCE_MICROBATCH,
    LANE_ROW_SCHEMA_VERSION,
    LANE_TRACE_SCHEMA_VERSION,
    LAYER_NORM_EPSILON,
    MASK_SCHEDULE_TABLE_SHA256,
    MAX_ACCEPTED_SHAPES,
    MODEL_PARAMETER_COUNT,
    MODEL_SEMANTICS_VERSION,
    MODEL_STAGE,
    OUTPUT_COLOR_COUNT,
    OPTIMIZER_UPDATES,
    PAIR_COST_SCHEMA_VERSION,
    PAIR_SLOT_EMBEDDING_COUNT,
    POOL_SETUP_COST_SCHEMA_VERSION,
    POSITION_EMBEDDING_COUNT,
    REARC_PARENT_SPLIT_SEMANTICS_VERSION,
    ROLE_EMBEDDING_COUNT,
    SAMPLE_ORBIT_SEMANTICS_VERSION,
    SAMPLER_SEMANTICS_VERSION,
    TASK_ORBIT_SEMANTICS_VERSION,
    TOKEN_EMBEDDING_COUNT,
    TRAINING_COST_LEDGER_SCHEMA_VERSION,
    TRAINING_SEED,
    VALIDATION_SEMANTICS_VERSION,
    apply_indexed_d4,
    canonical_sha256,
    decoder_batch_calls,
    episode_seed,
    lane_plan,
    lane_seed,
    mask_count_trace,
    masked_token_predictions,
    model_parameter_closure,
    rearc_parent_split_bucket,
    shape_lane_allocations,
    validation_episode_seed,
)
from .m04a_launch_plan import (
    EXPECTED_INPUT_ARTIFACT_PATHS,
    LaunchPlanArtifact,
    build_launch_plan,
    canonical_launch_plan_bytes,
    read_launch_plan_artifact,
)
from .m04a_train_contract import (
    CHECKPOINT_SELECTION_SCHEMA_VERSION,
    TRAINING_CONFIG_SCHEMA_VERSION,
    WARMUP_UPDATES,
    learning_rate_for_update,
    training_config_payload,
    training_config_sha256,
)
from .m04a_validation_manifest import (
    VALIDATION_EPISODE_JSONL,
    VALIDATION_EPISODE_ROW_SCHEMA_VERSION,
    VALIDATION_EPISODE_SUMMARY_JSON,
    VALIDATION_EPISODE_SUMMARY_SCHEMA_VERSION,
    ValidationEpisodeManifest,
    build_validation_manifest_commitment,
    read_validation_episode_manifest,
)
from .authority import DATASET_COMMITS, DATASET_ORIGINS
from .manifest import serialize_json, serialize_jsonl, verify_source_snapshot_zip


INPUT_BUNDLE_SCHEMA_VERSION = "afts-m04a-input-bundle/v0.1"
SANITIZED_SHARD_MANIFEST_SCHEMA_VERSION = (
    "afts-m04a-sanitized-shard-input-manifest/v0.1"
)
MODEL_CONFIG_SCHEMA_VERSION = "afts-grid-cmlm-model-config/v0.1"
PARAMETER_COUNT_SCHEMA_VERSION = "afts-grid-cmlm-parameter-count/v0.1"
ORDERED_FOLD_IDS_SCHEMA_VERSION = "afts-m04a-ordered-fold-ids/v0.1"
QUARANTINE_IDS_SCHEMA_VERSION = "afts-m04a-quarantine-parent-ids/v0.1"
SCHEMA_CONFIG_MANIFEST_SCHEMA_VERSION = "afts-m04a-schema-config-manifest/v0.1"
SEED_POLICY_SCHEMA_VERSION = "afts-m04a-seed-policy/v0.1"
SHORT_REPLAY_FIXTURE_SCHEMA_VERSION = "afts-m04a-short-exact-replay/v0.1"

BUNDLE_MANIFEST_NAME = "bundle_manifest.json"
VISIBLE_DIRECTORY_NAME = "visible"
CONTROL_DIRECTORY_NAME = "control"
LAUNCH_PLAN_NAME = "launch_plan.json"
CONDA_EXPLICIT_NAME = "conda-explicit.txt"
PYTHON_RUNTIME_LOCK_NAME = "python-runtime-lock.json"
SANITIZED_SHARD_NAMES = (
    "arc2_train.zip",
    "arc2_validation.zip",
    "rearc_train.zip",
    "rearc_validation.zip",
)
SANITIZED_SOURCE_NAMES = frozenset(
    {"artifact_manifest.json", "data_split_manifest.json", *SANITIZED_SHARD_NAMES}
)

MAX_SINGLE_INPUT_BYTES = 512 * 1024 * 1024
MAX_TOTAL_SOURCE_BYTES = 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_SANITIZED_UNCOMPRESSED_BYTES = 16 * 1024 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_HEX_PARENT_ID = re.compile(r"[0-9a-f]{8}\Z")
_REPARSE_POINT_ATTRIBUTE = 0x400
_SANITIZED_READER_TOKEN = object()
_INPUT_BUNDLE_READER_TOKEN = object()
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


def _sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TypeError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _strict_json_bytes(content: bytes, *, label: str) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant in {label}: {value}")

    try:
        return json.loads(
            content.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid strict UTF-8 JSON in {label}") from exc


def _canonical_json_payload(content: bytes, *, label: str) -> dict[str, Any]:
    payload = _strict_json_bytes(content, label=label)
    if type(payload) is not dict:
        raise TypeError(f"{label} must be a JSON object")
    if serialize_json(payload) != content:
        raise ValueError(f"{label} is not canonically serialized")
    return payload


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _is_link_or_reparse(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE
    )


def _safe_read(
    path: str | Path,
    *,
    expected_sha256: str,
    label: str,
    max_bytes: int = MAX_SINGLE_INPUT_BYTES,
) -> bytes:
    """Use the hardened evidence reader (openat/no-follow/O_NONBLOCK)."""

    from .m04a_evidence import _read_regular_snapshot

    return _read_regular_snapshot(
        _absolute(path),
        label=label,
        expected_sha256=_sha256(expected_sha256, field_name=f"{label} SHA-256"),
        max_bytes=max_bytes,
    )


def _enumerate_flat_directory(
    root: Path, *, expected_names: set[str], label: str
) -> None:
    if not os.path.lexists(root) or _is_link_or_reparse(root):
        raise ValueError(f"{label} root is absent, linked, or a reparse point")
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError(f"{label} root must be a real directory")
    before = root.lstat()
    names: set[str] = set()
    with os.scandir(root) as entries:
        for entry in entries:
            path = root / entry.name
            if entry.is_symlink() or _is_link_or_reparse(path):
                raise ValueError(f"{label} contains a symlink or reparse point")
            if not entry.is_file(follow_symlinks=False):
                raise ValueError(f"{label} must contain only regular files")
            names.add(entry.name)
    after = root.lstat()
    if not os.path.samestat(before, after):
        raise RuntimeError(f"{label} directory identity changed during enumeration")
    if names != expected_names:
        raise ValueError(
            f"{label} is not closed-world: missing={sorted(expected_names - names)}, "
            f"extra={sorted(names - expected_names)}"
        )


def _metadata(content: bytes) -> dict[str, object]:
    return {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def _forbidden_payload_fields(value: object) -> None:
    forbidden = {
        "collisions",
        "collision_rows",
        "protected_orbit_ids",
        "output_hashes",
        "holdout_ids",
        "smoke_ids",
        "sealed_payload",
        "sealed_artifacts",
        "privileged_manifest",
    }
    if isinstance(value, dict):
        overlap = forbidden & set(value)
        if overlap:
            raise ValueError(
                f"sanitized input exposes protected fields: {sorted(overlap)}"
            )
        for child in value.values():
            _forbidden_payload_fields(child)
    elif isinstance(value, list):
        for child in value:
            _forbidden_payload_fields(child)


@dataclass(frozen=True, slots=True)
class CommittedSanitizedInput:
    root: Path
    outer_bytes: bytes = field(repr=False)
    data_split_bytes: bytes = field(repr=False)
    artifact_manifest_sha256: str
    _reader_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._reader_token is not _SANITIZED_READER_TOKEN:
            raise TypeError(
                "CommittedSanitizedInput must come from its external reader"
            )
        if (
            hashlib.sha256(self.outer_bytes).hexdigest()
            != self.artifact_manifest_sha256
        ):
            raise ValueError("sanitized outer immutable snapshot mismatch")

    @property
    def outer(self) -> dict[str, Any]:
        return _canonical_json_payload(
            self.outer_bytes, label="immutable sanitized artifact_manifest.json"
        )

    @property
    def data_split(self) -> dict[str, Any]:
        return _canonical_json_payload(
            self.data_split_bytes, label="immutable data_split_manifest.json"
        )

    @property
    def shard_metadata(self) -> dict[str, dict[str, object]]:
        payload = self.data_split.get("shards")
        if type(payload) is not dict:
            raise ValueError("immutable data split lacks shard metadata")
        return {name: dict(row) for name, row in payload.items()}


def _ordered_parent_ids(value: object, *, label: str) -> tuple[str, ...]:
    if type(value) is not list or any(
        not isinstance(item, str) or _HEX_PARENT_ID.fullmatch(item) is None
        for item in value
    ):
        raise ValueError(f"{label} must be a list of lowercase 8-hex parent IDs")
    result = tuple(value)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{label} must be sorted and unique")
    return result


def read_committed_sanitized_input(
    directory: str | Path,
    *,
    expected_artifact_manifest_sha256: str,
    require_production_counts: bool = True,
) -> CommittedSanitizedInput:
    """Read and close a sanitized split using only its external outer commitment."""

    if type(require_production_counts) is not bool:
        raise TypeError("require_production_counts must be bool")
    root = _absolute(directory)
    _enumerate_flat_directory(
        root, expected_names=set(SANITIZED_SOURCE_NAMES), label="sanitized split"
    )
    expected_outer_sha = _sha256(
        expected_artifact_manifest_sha256,
        field_name="expected sanitized artifact-manifest SHA-256",
    )
    outer_bytes = _safe_read(
        root / "artifact_manifest.json",
        expected_sha256=expected_outer_sha,
        label="sanitized artifact_manifest.json",
        max_bytes=MAX_MANIFEST_BYTES,
    )
    outer = _canonical_json_payload(
        outer_bytes, label="sanitized artifact_manifest.json"
    )
    if set(outer) != {"schema_version", "bundle_status", "run_id", "artifacts"}:
        raise ValueError("sanitized outer manifest fields are not exact")
    if outer["schema_version"] != 1 or outer["bundle_status"] != "complete":
        raise ValueError("sanitized outer manifest is not complete schema v1")
    rows = outer["artifacts"]
    expected_payload_names = SANITIZED_SOURCE_NAMES - {"artifact_manifest.json"}
    if type(rows) is not dict or set(rows) != expected_payload_names:
        raise ValueError("sanitized outer artifact set is not closed-world")

    snapshots: dict[str, bytes] = {}
    total = 0
    for name in sorted(expected_payload_names):
        row = rows[name]
        if type(row) is not dict or set(row) != {"sha256", "bytes", "rows"}:
            raise ValueError(f"sanitized outer metadata is malformed for {name}")
        digest = _sha256(row["sha256"], field_name=f"sanitized {name} SHA-256")
        byte_count = row["bytes"]
        if type(byte_count) is not int or byte_count <= 0:
            raise ValueError(f"sanitized {name} byte count must be positive")
        snapshot = _safe_read(
            root / name,
            expected_sha256=digest,
            label=f"sanitized {name}",
        )
        if len(snapshot) != byte_count:
            raise ValueError(f"sanitized {name} byte count differs from outer manifest")
        expected_rows = snapshot.count(b"\n") if name.endswith(".jsonl") else None
        if row["rows"] != expected_rows:
            raise ValueError(f"sanitized {name} row count differs from outer manifest")
        total += len(snapshot)
        if total > MAX_TOTAL_SOURCE_BYTES:
            raise ValueError("sanitized source exceeds the total byte limit")
        snapshots[name] = snapshot

    split_bytes = snapshots["data_split_manifest.json"]
    split = _canonical_json_payload(split_bytes, label="data_split_manifest.json")
    expected_split_fields = {
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
        set(split) != expected_split_fields
        or split["schema"] != "afts.m04a-sanitized-split/v1"
    ):
        raise ValueError("sanitized data-split manifest schema is not exact")
    if split["data_folds_semantics_version"] != DATA_FOLDS_SEMANTICS_VERSION:
        raise ValueError("sanitized data-fold semantics mismatch")
    if split["protected_payloads_absent"] is not True:
        raise ValueError("sanitized manifest does not attest protected payload absence")
    _sha256(
        split["sealed_artifact_manifest_sha256"],
        field_name="sealed artifact-manifest commitment",
    )
    _forbidden_payload_fields(split)

    allowlists = split["ordered_allowlists"]
    counts = split["usable_fold_counts"]
    expected_folds = {
        "arc2_train",
        "arc2_validation",
        "rearc_train",
        "rearc_validation",
    }
    if type(allowlists) is not dict or set(allowlists) != expected_folds:
        raise ValueError("sanitized ordered allowlists are not closed-world")
    if type(counts) is not dict or set(counts) != expected_folds:
        raise ValueError("sanitized fold counts are not closed-world")
    normalized_allowlists = {
        name: _ordered_parent_ids(allowlists[name], label=f"ordered_allowlists.{name}")
        for name in sorted(expected_folds)
    }
    for name, ids in normalized_allowlists.items():
        if type(counts[name]) is not int or counts[name] != len(ids):
            raise ValueError(f"sanitized count does not close for {name}")
    if require_production_counts and counts != {
        "arc2_train": 662,
        "arc2_validation": 145,
        "rearc_train": 262,
        "rearc_validation": 51,
    }:
        raise ValueError("sanitized source does not have frozen production counts")
    if (
        require_production_counts
        and split["source_commitments"] != _PRODUCTION_SOURCE_COMMITMENTS
    ):
        raise ValueError(
            "sanitized source commitments are not the frozen production inputs"
        )

    validation_quarantine = _ordered_parent_ids(
        split["validation_quarantine_ids"], label="validation_quarantine_ids"
    )
    training_quarantine = _ordered_parent_ids(
        split["training_quarantine_ids"], label="training_quarantine_ids"
    )
    reasons = split["quarantine_reasons"]
    if type(reasons) is not dict or set(reasons) != set(validation_quarantine) | set(
        training_quarantine
    ):
        raise ValueError("quarantine reasons do not close over quarantine IDs")
    if any(
        type(value) is not list
        or not value
        or value != sorted(set(value))
        or any(not isinstance(reason, str) or not reason for reason in value)
        for value in reasons.values()
    ):
        raise ValueError("quarantine reasons must be nonempty sorted string lists")
    quarantine_union = set(validation_quarantine) | set(training_quarantine)
    usable_arc2 = set(normalized_allowlists["arc2_train"]) | set(
        normalized_allowlists["arc2_validation"]
    )
    usable_rearc = set(normalized_allowlists["rearc_train"]) | set(
        normalized_allowlists["rearc_validation"]
    )
    if usable_arc2 & quarantine_union:
        raise ValueError("usable ARC2 allowlists contain quarantined semantic parents")
    aliases = split["semantic_aliases_in_usable_shards"]
    if type(aliases) is not list:
        raise ValueError("sanitized semantic aliases must be a JSON array")
    alias_map: dict[str, str] = {}
    for row in aliases:
        if (
            type(row) is not dict
            or set(row) != {"rearc_parent_id", "arc2_parent_id"}
            or not isinstance(row["rearc_parent_id"], str)
            or _HEX_PARENT_ID.fullmatch(row["rearc_parent_id"]) is None
            or not isinstance(row["arc2_parent_id"], str)
            or _HEX_PARENT_ID.fullmatch(row["arc2_parent_id"]) is None
            or row["rearc_parent_id"] == row["arc2_parent_id"]
            or row["rearc_parent_id"] in alias_map
            or row["rearc_parent_id"] not in usable_rearc
            or row["arc2_parent_id"] not in usable_arc2
        ):
            raise ValueError("sanitized semantic alias row is malformed or out of fold")
        alias_map[row["rearc_parent_id"]] = row["arc2_parent_id"]

    shard_manifests = split["shards"]
    if type(shard_manifests) is not dict or set(shard_manifests) != set(
        SANITIZED_SHARD_NAMES
    ):
        raise ValueError("sanitized shard manifest set is not closed-world")
    observed_aliases: dict[str, str] = {}
    total_uncompressed = 0
    for name in SANITIZED_SHARD_NAMES:
        shard = shard_manifests[name]
        if type(shard) is not dict:
            raise TypeError(f"sanitized shard metadata must be an object: {name}")
        outer_row = rows[name]
        if (
            shard.get("zip_sha256") != outer_row["sha256"]
            or shard.get("zip_bytes") != outer_row["bytes"]
        ):
            raise ValueError(f"sanitized shard differs from outer commitment: {name}")
        fold_key = name.removesuffix(".zip")
        ids = normalized_allowlists[fold_key]
        if shard.get("ordered_source_parent_ids") != list(ids) or shard.get(
            "member_count"
        ) != len(ids):
            raise ValueError(f"sanitized shard allowlist closure mismatch: {name}")
        members = shard.get("members")
        if type(members) is not list or len(members) != len(ids):
            raise ValueError(f"sanitized shard member rows do not close: {name}")
        source, fold = fold_key.split("_", 1)
        expected_member_paths: list[str] = []
        for parent_id, member in zip(ids, members, strict=True):
            expected_path = f"{source}/{fold}/{parent_id}.json"
            if (
                type(member) is not dict
                or set(member)
                != {
                    "path",
                    "source_parent_id",
                    "semantic_parent_id",
                    "sha256",
                    "bytes",
                }
                or member["path"] != expected_path
                or member["source_parent_id"] != parent_id
                or not isinstance(member["semantic_parent_id"], str)
                or _HEX_PARENT_ID.fullmatch(member["semantic_parent_id"]) is None
                or type(member["bytes"]) is not int
                or member["bytes"] <= 0
            ):
                raise ValueError(f"sanitized shard member row is malformed: {name}")
            _sha256(member["sha256"], field_name=f"sanitized member {expected_path}")
            expected_semantic = alias_map.get(parent_id, parent_id)
            if member["semantic_parent_id"] != expected_semantic:
                raise ValueError(
                    "sanitized member semantic parent differs from alias closure"
                )
            if member["semantic_parent_id"] in quarantine_union:
                raise ValueError(
                    "usable sanitized member maps to a quarantined semantic parent"
                )
            if expected_semantic != parent_id:
                observed_aliases[parent_id] = expected_semantic
            expected_member_paths.append(expected_path)
        try:
            with zipfile.ZipFile(io.BytesIO(snapshots[name]), mode="r") as archive:
                infos = archive.infolist()
                if len(infos) != len(ids) or len(
                    {info.filename for info in infos}
                ) != len(infos):
                    raise ValueError(f"sanitized shard member set is malformed: {name}")
                if [info.filename for info in infos] != expected_member_paths:
                    raise ValueError(
                        f"sanitized shard member order or names drifted: {name}"
                    )
                if any(
                    info.is_dir()
                    or stat.S_ISLNK(info.external_attr >> 16)
                    or info.flag_bits & 0x1
                    or info.compress_type != zipfile.ZIP_DEFLATED
                    or info.date_time != (1980, 1, 1, 0, 0, 0)
                    or (info.external_attr >> 16) != 0o100644
                    or info.file_size != member["bytes"]
                    for info, member in zip(infos, members, strict=True)
                ):
                    raise ValueError(f"sanitized shard contains unsafe members: {name}")
                aggregate = hashlib.sha256()
                for info, member in zip(infos, members, strict=True):
                    encoded_name = info.filename.encode("utf-8")
                    aggregate.update(len(encoded_name).to_bytes(8, "big"))
                    aggregate.update(encoded_name)
                    aggregate.update(int(member["bytes"]).to_bytes(8, "big"))
                    member_digest = hashlib.sha256()
                    observed_bytes = 0
                    with archive.open(info, "r") as handle:
                        while chunk := handle.read(1024 * 1024):
                            observed_bytes += len(chunk)
                            total_uncompressed += len(chunk)
                            if total_uncompressed > MAX_SANITIZED_UNCOMPRESSED_BYTES:
                                raise ValueError(
                                    "sanitized shards exceed uncompressed byte limit"
                                )
                            member_digest.update(chunk)
                            aggregate.update(chunk)
                    if (
                        observed_bytes != member["bytes"]
                        or member_digest.hexdigest() != member["sha256"]
                    ):
                        raise ValueError(
                            f"sanitized shard member bytes differ from manifest: {info.filename}"
                        )
                if aggregate.hexdigest() != shard.get("member_aggregate_sha256"):
                    raise ValueError(
                        f"sanitized shard member aggregate differs: {name}"
                    )
        except (zipfile.BadZipFile, OSError) as exc:
            raise ValueError(f"invalid sanitized shard ZIP: {name}") from exc

    if observed_aliases != alias_map:
        raise ValueError("sanitized semantic aliases do not close bidirectionally")

    _enumerate_flat_directory(
        root, expected_names=set(SANITIZED_SOURCE_NAMES), label="sanitized split"
    )
    return CommittedSanitizedInput(
        root=root,
        outer_bytes=outer_bytes,
        data_split_bytes=split_bytes,
        artifact_manifest_sha256=expected_outer_sha,
        _reader_token=_SANITIZED_READER_TOKEN,
    )


def _model_config_payload() -> dict[str, object]:
    return {
        "model_semantics_version": MODEL_SEMANTICS_VERSION,
        "model_stage": MODEL_STAGE,
        "d_model": D_MODEL,
        "attention_heads": ATTENTION_HEADS,
        "ffn_width": FFN_WIDTH,
        "encoder_layers": ENCODER_LAYER_COUNT,
        "decoder_layers": DECODER_LAYER_COUNT,
        "dropout": DROPOUT_PROBABILITY,
        "layer_norm_epsilon": LAYER_NORM_EPSILON,
        "token_embeddings": TOKEN_EMBEDDING_COUNT,
        "position_embeddings": POSITION_EMBEDDING_COUNT,
        "role_embeddings": ROLE_EMBEDDING_COUNT,
        "pair_slot_embeddings": PAIR_SLOT_EMBEDDING_COUNT,
        "output_colors": OUTPUT_COLOR_COUNT,
        "parameter_count": MODEL_PARAMETER_COUNT,
        "initialization_seed": TRAINING_SEED,
        "target_has_grid_cls": False,
        "embedding_combination": "direct_unscaled_sum",
        "norm_first": True,
        "causal_decoder": False,
    }


def _parameter_count_payload() -> dict[str, object]:
    semantic: dict[str, object] = {
        "schema": PARAMETER_COUNT_SCHEMA_VERSION,
        "model_semantics_version": MODEL_SEMANTICS_VERSION,
        "parameter_count": MODEL_PARAMETER_COUNT,
        "parameter_closure": asdict(model_parameter_closure()),
    }
    return {**semantic, "closure_id": canonical_sha256(semantic)}


def _schema_config_payload() -> dict[str, object]:
    semantic: dict[str, object] = {
        "schema": SCHEMA_CONFIG_MANIFEST_SCHEMA_VERSION,
        "training_config": training_config_payload(),
        "frozen_schema_versions": {
            "checkpoint_selection": CHECKPOINT_SELECTION_SCHEMA_VERSION,
            "decoder_forward_ledger": DECODER_FORWARD_LEDGER_SCHEMA_VERSION,
            "encoder_forward_ledger": ENCODER_FORWARD_LEDGER_SCHEMA_VERSION,
            "lane_row": LANE_ROW_SCHEMA_VERSION,
            "lane_trace": LANE_TRACE_SCHEMA_VERSION,
            "model_config_contract": MODEL_CONFIG_SCHEMA_VERSION,
            "pair_cost": PAIR_COST_SCHEMA_VERSION,
            "pool_setup_cost": POOL_SETUP_COST_SCHEMA_VERSION,
            "rearc_parent_split": REARC_PARENT_SPLIT_SEMANTICS_VERSION,
            "sample_orbit": SAMPLE_ORBIT_SEMANTICS_VERSION,
            "task_orbit": TASK_ORBIT_SEMANTICS_VERSION,
            "training_config": TRAINING_CONFIG_SCHEMA_VERSION,
            "training_cost_ledger": TRAINING_COST_LEDGER_SCHEMA_VERSION,
            "validation_episode_row": VALIDATION_EPISODE_ROW_SCHEMA_VERSION,
            "validation_episode_summary": VALIDATION_EPISODE_SUMMARY_SCHEMA_VERSION,
        },
    }
    return {**semantic, "manifest_id": canonical_sha256(semantic)}


def _seed_policy_payload() -> dict[str, object]:
    semantic: dict[str, object] = {
        "schema": SEED_POLICY_SCHEMA_VERSION,
        "training_seed": TRAINING_SEED,
        "training_episode_seed": (
            "uint64_le(SHA256(episode_semantics+'\\0'+training_seed+'\\0'+"
            "global_microbatch)[0:8])"
        ),
        "validation_episode_seed": (
            "uint64_le(SHA256(validation_semantics+'\\0'+parent_id+'\\0'+"
            "target_descriptor+'\\0'+view_index)[0:8])"
        ),
        "generation_lane_seed": (
            "uint64_le(SHA256(sampler_semantics+'\\0'+blind_task_id+'\\0'+"
            "test_index+'\\0'+shape_proposal_id+'\\0'+local_lane)[0:8])"
        ),
        "episode_semantics_version": EPISODE_SEMANTICS_VERSION,
        "validation_semantics_version": VALIDATION_SEMANTICS_VERSION,
        "sampler_semantics_version": SAMPLER_SEMANTICS_VERSION,
    }
    return {**semantic, "policy_id": canonical_sha256(semantic)}


def build_short_exact_replay_fixture() -> dict[str, object]:
    """Build a compact fixture whose every result comes from a pure contract helper."""

    grid = ((0, 1, 2), (3, 4, 5))
    lane_rows = [asdict(row) for row in lane_plan(3)]
    semantic: dict[str, object] = {
        "schema": SHORT_REPLAY_FIXTURE_SCHEMA_VERSION,
        "model_semantics_version": MODEL_SEMANTICS_VERSION,
        "sampler_semantics_version": SAMPLER_SEMANTICS_VERSION,
        "parameter_count": MODEL_PARAMETER_COUNT,
        "episode_seed_cases": [
            {
                "optimizer_step": step,
                "microbatch_slot": slot,
                "seed_u64": episode_seed(step, slot),
            }
            for step, slot in (
                (0, 0),
                (0, GRADIENT_ACCUMULATION - 1),
                (OPTIMIZER_UPDATES - 1, 0),
                (OPTIMIZER_UPDATES - 1, GRADIENT_ACCUMULATION - 1),
            )
        ],
        "validation_seed_cases": [
            {
                "parent_id": parent,
                "target_descriptor": target,
                "view_index": view,
                "seed_u64": validation_episode_seed(parent, target, view),
            }
            for parent, target, view in (
                ("aaaaaaaa", "test:0", 0),
                ("bbbbbbbb", "rearc:17", 3),
            )
        ],
        "lane_seed_cases": [
            {
                "blind_task_id": task,
                "test_index": test_index,
                "shape_proposal_id": proposal,
                "local_lane": lane,
                "seed_u64": lane_seed(task, test_index, proposal, lane),
            }
            for task, test_index, proposal, lane in (
                ("fixture-task", 0, "shape-0", 0),
                ("fixture-task", 1, "shape-2", 63),
            )
        ],
        "rearc_split_cases": [
            {"parent_id": parent, "bucket": rearc_parent_split_bucket(parent)}
            for parent in ("40853293", "e26a3af2")
        ],
        "d4_case": {
            "input": [list(row) for row in grid],
            "ordered_outputs": [
                [list(row) for row in apply_indexed_d4(grid, index)]
                for index in range(8)
            ],
        },
        "mask_schedule_table_sha256": MASK_SCHEDULE_TABLE_SHA256,
        "mask_cases": [
            {
                "cell_count": count,
                "trace": list(mask_count_trace(count)),
                "masked_token_predictions": masked_token_predictions(count),
            }
            for count in (1, 2, 37, 900)
        ],
        "three_shape_lane_case": {
            "allocations": list(shape_lane_allocations(3)),
            "decoder_batch_calls": decoder_batch_calls(3),
            "lane_rows": lane_rows,
            "lane_rows_sha256": canonical_sha256(lane_rows),
            "denoising_steps": DENOISING_STEPS,
            "inference_microbatch": INFERENCE_MICROBATCH,
            "maximum_shape_count": MAX_ACCEPTED_SHAPES,
        },
        "learning_rate_hex_cases": [
            {
                "update": update,
                "learning_rate_hex": learning_rate_for_update(update).hex(),
            }
            for update in (
                1,
                WARMUP_UPDATES,
                WARMUP_UPDATES + 1,
                OPTIMIZER_UPDATES // 2,
                OPTIMIZER_UPDATES,
            )
        ],
    }
    return {**semantic, "fixture_id": canonical_sha256(semantic)}


def validate_short_exact_replay_fixture(payload: object) -> dict[str, object]:
    if type(payload) is not dict or payload != build_short_exact_replay_fixture():
        raise ValueError(
            "short exact replay fixture does not replay from the frozen contract"
        )
    return dict(payload)


def _ordered_fold_payload(sanitized: CommittedSanitizedInput) -> dict[str, object]:
    semantic: dict[str, object] = {
        "schema": ORDERED_FOLD_IDS_SCHEMA_VERSION,
        "data_folds_semantics_version": DATA_FOLDS_SEMANTICS_VERSION,
        "sanitized_artifact_manifest_sha256": sanitized.artifact_manifest_sha256,
        "ordered_usable_fold_ids": dict(sanitized.data_split["ordered_allowlists"]),
        "usable_fold_counts": dict(sanitized.data_split["usable_fold_counts"]),
    }
    return {**semantic, "manifest_id": canonical_sha256(semantic)}


def _quarantine_payload(sanitized: CommittedSanitizedInput) -> dict[str, object]:
    semantic: dict[str, object] = {
        "schema": QUARANTINE_IDS_SCHEMA_VERSION,
        "data_folds_semantics_version": DATA_FOLDS_SEMANTICS_VERSION,
        "sanitized_artifact_manifest_sha256": sanitized.artifact_manifest_sha256,
        "validation_parent_ids": list(
            sanitized.data_split["validation_quarantine_ids"]
        ),
        "training_parent_ids": list(sanitized.data_split["training_quarantine_ids"]),
        "reasons": dict(sanitized.data_split["quarantine_reasons"]),
        "contains_protected_payloads": False,
    }
    return {**semantic, "manifest_id": canonical_sha256(semantic)}


def _sanitized_shard_payload(sanitized: CommittedSanitizedInput) -> dict[str, object]:
    outer = dict(sanitized.outer)
    outer_rows = outer["artifacts"]
    shards: dict[str, object] = {}
    for name in SANITIZED_SHARD_NAMES:
        split_row = sanitized.shard_metadata[name]
        shards[name] = {
            "sha256": outer_rows[name]["sha256"],
            "bytes": outer_rows[name]["bytes"],
            "member_count": split_row["member_count"],
            "ordered_source_parent_ids_sha256": canonical_sha256(
                split_row["ordered_source_parent_ids"]
            ),
        }
    semantic: dict[str, object] = {
        "schema": SANITIZED_SHARD_MANIFEST_SCHEMA_VERSION,
        "sanitized_outer_artifact_manifest_sha256": sanitized.artifact_manifest_sha256,
        "sanitized_outer_artifact_manifest": outer,
        "data_split_manifest_sha256": hashlib.sha256(
            sanitized.data_split_bytes
        ).hexdigest(),
        "data_split_manifest_bytes": len(sanitized.data_split_bytes),
        "data_split_manifest_semantic_sha256": canonical_sha256(
            dict(sanitized.data_split)
        ),
        "shards": shards,
        "protected_payloads_absent": True,
    }
    return {**semantic, "manifest_id": canonical_sha256(semantic)}


def _validate_frozen_contract(content: bytes) -> None:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("frozen contract must be UTF-8") from exc
    required = (
        "# M04a global masked-grid candidate-source contract",
        MODEL_SEMANTICS_VERSION,
        SAMPLER_SEMANTICS_VERSION,
        "8,733,706",
        "20260711",
    )
    if (
        "\r" in text
        or not text.endswith("\n")
        or any(value not in text for value in required)
    ):
        raise ValueError("frozen contract is not the complete canonical M04a contract")


def _validate_launcher(content: bytes) -> None:
    try:
        text = content.decode("utf-8")
        tree = ast.parse(text, filename="remote_launcher.py")
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ValueError("remote launcher is not valid UTF-8 Python") from exc
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    if "\r" in text or not text.endswith("\n") or "main" not in names:
        raise ValueError("remote launcher is not canonical production source")
    for anchor in ("os.execve", "CUDA_VISIBLE_DEVICES", "flock", "launch_plan"):
        if anchor not in text:
            raise ValueError(f"remote launcher lacks production anchor: {anchor}")


def _validate_conda_explicit(content: bytes) -> None:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("conda explicit artifact must be UTF-8") from exc
    if not text or "\x00" in text or "\r" in text or not text.endswith("\n"):
        raise ValueError("conda explicit artifact must be canonical LF text")
    lines = text.splitlines()
    if lines.count("@EXPLICIT") != 1 or any(len(line) > 16_384 for line in lines):
        raise ValueError("conda explicit artifact must contain one @EXPLICIT marker")


def _validate_opaque_runtime_lock(content: bytes) -> dict[str, Any]:
    payload = _canonical_json_payload(content, label=PYTHON_RUNTIME_LOCK_NAME)
    if payload.get("schema") != "afts-m04a-python-runtime-lock/v0.1":
        raise ValueError("unsupported python runtime-lock schema")
    from .m04a_python_runtime_lock import validate_python_runtime_lock_payload

    payload = validate_python_runtime_lock_payload(payload)
    if serialize_json(payload) != content:
        raise ValueError("python runtime lock is not canonical after validation")
    return payload


def _read_committed_runtime_lock(path: Path, *, expected_sha256: str) -> bytes:
    """Require the dedicated runtime-lock reader at the external input boundary."""

    from .m04a_python_runtime_lock import read_python_runtime_lock_artifact

    artifact = read_python_runtime_lock_artifact(
        path, expected_artifact_sha256=expected_sha256
    )
    _validate_opaque_runtime_lock(artifact.snapshot)
    return artifact.snapshot


@dataclass(frozen=True, slots=True)
class M04AInputBundleSources:
    sanitized_bundle_dir: Path
    sanitized_artifact_manifest_sha256: str
    validation_manifest_dir: Path
    validation_artifact_manifest_sha256: str
    runtime_source_zip_path: Path
    runtime_source_zip_sha256: str
    runtime_source_fingerprint_sha256: str
    test_snapshot_zip_path: Path
    test_snapshot_zip_sha256: str
    test_source_fingerprint_sha256: str
    launcher_path: Path
    launcher_sha256: str
    frozen_contract_path: Path
    frozen_contract_sha256: str
    conda_explicit_path: Path
    conda_explicit_sha256: str
    python_runtime_lock_path: Path
    python_runtime_lock_sha256: str
    run_id: str
    remote_project_root: str
    attempt_nonce: str
    ordered_import_roots: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "sanitized_artifact_manifest_sha256",
            "validation_artifact_manifest_sha256",
            "runtime_source_zip_sha256",
            "runtime_source_fingerprint_sha256",
            "test_snapshot_zip_sha256",
            "test_source_fingerprint_sha256",
            "launcher_sha256",
            "frozen_contract_sha256",
            "conda_explicit_sha256",
            "python_runtime_lock_sha256",
            "attempt_nonce",
        ):
            _sha256(getattr(self, field_name), field_name=field_name)
        if not isinstance(self.run_id, str) or _SAFE_ID.fullmatch(self.run_id) is None:
            raise ValueError("run_id must be one safe component")
        if (
            type(self.ordered_import_roots) is not tuple
            or not self.ordered_import_roots
        ):
            raise TypeError("ordered_import_roots must be a nonempty tuple")
        for field_name in (
            "sanitized_bundle_dir",
            "validation_manifest_dir",
            "runtime_source_zip_path",
            "test_snapshot_zip_path",
            "launcher_path",
            "frozen_contract_path",
            "conda_explicit_path",
            "python_runtime_lock_path",
        ):
            object.__setattr__(self, field_name, _absolute(getattr(self, field_name)))


@dataclass(frozen=True, slots=True)
class _SourceReference:
    path: Path
    sha256: str
    label: str
    max_bytes: int = MAX_SINGLE_INPUT_BYTES


def _validation_source_artifacts(
    sources: M04AInputBundleSources,
) -> tuple[ValidationEpisodeManifest, dict[str, bytes], tuple[_SourceReference, ...]]:
    validation = read_validation_episode_manifest(
        sources.validation_manifest_dir,
        expected_artifact_manifest_sha256=sources.validation_artifact_manifest_sha256,
    )
    root = sources.validation_manifest_dir
    outer = _safe_read(
        root / "artifact_manifest.json",
        expected_sha256=sources.validation_artifact_manifest_sha256,
        label="validation outer manifest",
        max_bytes=MAX_MANIFEST_BYTES,
    )
    jsonl_sha = str(validation.summary["jsonl_sha256"])
    summary = serialize_json(validation.summary)
    summary_sha = hashlib.sha256(summary).hexdigest()
    jsonl = _safe_read(
        root / VALIDATION_EPISODE_JSONL,
        expected_sha256=jsonl_sha,
        label="validation episode JSONL",
    )
    summary_disk = _safe_read(
        root / VALIDATION_EPISODE_SUMMARY_JSON,
        expected_sha256=summary_sha,
        label="validation episode summary",
        max_bytes=MAX_MANIFEST_BYTES,
    )
    if jsonl != validation.jsonl_bytes or summary_disk != summary:
        raise RuntimeError("validation source changed after its committed reader")
    refs = (
        _SourceReference(
            root / "artifact_manifest.json",
            sources.validation_artifact_manifest_sha256,
            "validation outer manifest",
            MAX_MANIFEST_BYTES,
        ),
        _SourceReference(
            root / VALIDATION_EPISODE_JSONL, jsonl_sha, "validation JSONL"
        ),
        _SourceReference(
            root / VALIDATION_EPISODE_SUMMARY_JSON,
            summary_sha,
            "validation summary",
            MAX_MANIFEST_BYTES,
        ),
    )
    return (
        validation,
        {
            "validation_episode_outer_manifest.json": outer,
            VALIDATION_EPISODE_JSONL: jsonl,
            VALIDATION_EPISODE_SUMMARY_JSON: summary,
        },
        refs,
    )


def _assert_sources_unchanged(references: Sequence[_SourceReference]) -> None:
    for reference in references:
        _safe_read(
            reference.path,
            expected_sha256=reference.sha256,
            label=f"final drift check: {reference.label}",
            max_bytes=reference.max_bytes,
        )


def _source_commitments(
    sources: M04AInputBundleSources,
    *,
    validation: ValidationEpisodeManifest,
) -> dict[str, object]:
    return {
        "sanitized_outer_artifact_manifest_sha256": sources.sanitized_artifact_manifest_sha256,
        "validation_outer_artifact_manifest_sha256": sources.validation_artifact_manifest_sha256,
        "validation_jsonl_sha256": validation.summary["jsonl_sha256"],
        "validation_summary_id": validation.summary["summary_id"],
        "runtime_source_zip_sha256": sources.runtime_source_zip_sha256,
        "runtime_source_fingerprint_sha256": sources.runtime_source_fingerprint_sha256,
        "test_snapshot_zip_sha256": sources.test_snapshot_zip_sha256,
        "test_source_fingerprint_sha256": sources.test_source_fingerprint_sha256,
        "launcher_sha256": sources.launcher_sha256,
        "frozen_contract_sha256": sources.frozen_contract_sha256,
        "conda_explicit_sha256": sources.conda_explicit_sha256,
        "python_runtime_lock_sha256": sources.python_runtime_lock_sha256,
    }


def _write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOINHERIT", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        view = memoryview(content)
        while offset < len(content):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError(f"short publication write for {path.name}")
            offset += written
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size != len(content):
            raise OSError(f"published metadata differs for {path.name}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _paths_overlap(first: Path, second: Path) -> bool:
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically activate a staged directory without replacement on Linux."""

    if os.name == "posix":
        renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
        if renameat2 is None:
            raise OSError(errno.ENOSYS, "renameat2(RENAME_NOREPLACE) is required")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(target),
            1,
        )
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(
                    error, "publication target already exists", target
                )
            raise OSError(error, os.strerror(error), target)
        return
    if os.name == "nt":
        os.rename(source, target)
        return
    raise OSError(errno.ENOSYS, "exclusive directory rename is unsupported")


def _bundle_manifest_payload(
    *,
    visible: Mapping[str, bytes],
    launch_plan_bytes: bytes,
    launch_plan_id: str,
    sources: M04AInputBundleSources,
    validation: ValidationEpisodeManifest,
) -> dict[str, object]:
    semantic: dict[str, object] = {
        "schema": INPUT_BUNDLE_SCHEMA_VERSION,
        "status": "complete_immutable_m04a_input_bundle",
        "run_id": sources.run_id,
        "visible_layout": "flat_external_directory",
        "control_layout": "disjoint_external_directory",
        "source_commitments": _source_commitments(sources, validation=validation),
        "visible_artifacts": {
            name: _metadata(content) for name, content in sorted(visible.items())
        },
        "control_artifacts": {LAUNCH_PLAN_NAME: _metadata(launch_plan_bytes)},
        "launch_plan_id": launch_plan_id,
        "launch_plan_sha256": hashlib.sha256(launch_plan_bytes).hexdigest(),
    }
    return {**semantic, "bundle_id": canonical_sha256(semantic)}


@dataclass(frozen=True, slots=True)
class M04AInputBundle:
    visible_root: Path
    control_root: Path
    manifest_sha256: str
    manifest_bytes: bytes = field(repr=False)
    launch_plan: LaunchPlanArtifact
    visible_artifacts: Mapping[str, bytes] = field(repr=False)
    _reader_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._reader_token is not _INPUT_BUNDLE_READER_TOKEN:
            raise TypeError("M04AInputBundle must come from its external reader")
        if not isinstance(self.visible_artifacts, MappingProxyType):
            raise TypeError("visible artifacts must be immutable")

    @property
    def manifest(self) -> dict[str, Any]:
        return _canonical_json_payload(self.manifest_bytes, label=BUNDLE_MANIFEST_NAME)

    @property
    def launch_plan_path(self) -> Path:
        return self.control_root / LAUNCH_PLAN_NAME


def _validate_internal_derivations(
    visible: Mapping[str, bytes], *, manifest: Mapping[str, Any]
) -> None:
    expected_exact = {
        "model_config.json": _model_config_payload(),
        "parameter_count.json": _parameter_count_payload(),
        "schema_config_manifest.json": _schema_config_payload(),
        "seed_policy.json": _seed_policy_payload(),
        "short_exact_replay_fixture.json": build_short_exact_replay_fixture(),
    }
    for name, expected in expected_exact.items():
        actual = _canonical_json_payload(visible[name], label=name)
        if actual != expected:
            raise ValueError(f"{name} does not replay from the frozen pure contract")

    _canonical_json_payload(
        visible["data_split_manifest.json"], label="data_split_manifest.json"
    )
    sanitized_sha = manifest["source_commitments"][
        "sanitized_outer_artifact_manifest_sha256"
    ]
    synthetic = object.__new__(CommittedSanitizedInput)
    object.__setattr__(synthetic, "root", Path("."))
    object.__setattr__(synthetic, "outer_bytes", b"")
    object.__setattr__(
        synthetic, "data_split_bytes", visible["data_split_manifest.json"]
    )
    shard_payload = _canonical_json_payload(
        visible["sanitized_shard_manifest.json"], label="sanitized_shard_manifest.json"
    )
    outer = shard_payload.get("sanitized_outer_artifact_manifest")
    if type(outer) is not dict:
        raise ValueError("sanitized shard manifest lacks the full outer manifest")
    outer_bytes = serialize_json(outer)
    if hashlib.sha256(outer_bytes).hexdigest() != sanitized_sha:
        raise ValueError(
            "embedded sanitized outer manifest differs from its commitment"
        )
    if (
        set(outer) != {"schema_version", "bundle_status", "run_id", "artifacts"}
        or outer["schema_version"] != 1
        or outer["bundle_status"] != "complete"
        or type(outer["artifacts"]) is not dict
        or set(outer["artifacts"])
        != {"data_split_manifest.json", *SANITIZED_SHARD_NAMES}
    ):
        raise ValueError("embedded sanitized outer manifest schema drifted")
    split_bytes = visible["data_split_manifest.json"]
    split_row = outer["artifacts"]["data_split_manifest.json"]
    if split_row != {
        "sha256": hashlib.sha256(split_bytes).hexdigest(),
        "bytes": len(split_bytes),
        "rows": None,
    }:
        raise ValueError(
            "copied data split differs from embedded sanitized outer manifest"
        )
    object.__setattr__(synthetic, "outer_bytes", outer_bytes)
    object.__setattr__(synthetic, "artifact_manifest_sha256", sanitized_sha)
    object.__setattr__(synthetic, "_reader_token", _SANITIZED_READER_TOKEN)
    if _sanitized_shard_payload(synthetic) != shard_payload:
        raise ValueError(
            "sanitized shard manifest does not replay from copied upstream metadata"
        )
    if _ordered_fold_payload(synthetic) != _canonical_json_payload(
        visible["ordered_fold_ids.json"], label="ordered_fold_ids.json"
    ):
        raise ValueError("ordered fold IDs do not replay from the data split")
    if _quarantine_payload(synthetic) != _canonical_json_payload(
        visible["quarantine_parent_ids.json"], label="quarantine_parent_ids.json"
    ):
        raise ValueError("quarantine IDs do not replay from the data split")

    runtime_materials = verify_source_snapshot_zip(
        visible["reviewed_runtime_source.zip"],
        expected_fingerprint_sha256=manifest["source_commitments"][
            "runtime_source_fingerprint_sha256"
        ],
    )
    if (
        runtime_materials.get("scripts/afts_arc_m04a.py")
        != visible["remote_launcher.py"]
    ):
        raise ValueError("remote launcher differs from the reviewed runtime snapshot")
    from .m04a_evidence import verify_test_source_snapshot_zip

    verify_test_source_snapshot_zip(
        visible["reviewed_test_snapshot.zip"],
        expected_fingerprint_sha256=manifest["source_commitments"][
            "test_source_fingerprint_sha256"
        ],
    )
    _validate_launcher(visible["remote_launcher.py"])
    _validate_frozen_contract(visible["frozen_contract.md"])
    _validate_conda_explicit(visible[CONDA_EXPLICIT_NAME])
    _validate_opaque_runtime_lock(visible[PYTHON_RUNTIME_LOCK_NAME])

    from .m04a_validation_manifest import (
        _strict_jsonl_bytes,
        _validate_summary,
        validate_validation_episode_rows,
    )

    jsonl_bytes = visible[VALIDATION_EPISODE_JSONL]
    rows = _strict_jsonl_bytes(jsonl_bytes)
    validate_validation_episode_rows(list(rows))
    if serialize_jsonl(rows) != jsonl_bytes:
        raise ValueError("visible validation JSONL is not canonical")
    summary_bytes = visible[VALIDATION_EPISODE_SUMMARY_JSON]
    summary = _canonical_json_payload(
        summary_bytes, label=VALIDATION_EPISODE_SUMMARY_JSON
    )
    _validate_summary(summary, rows=rows, jsonl_bytes=jsonl_bytes)
    outer_bytes = visible["validation_episode_outer_manifest.json"]
    outer = _canonical_json_payload(
        outer_bytes, label="validation_episode_outer_manifest.json"
    )
    if (
        set(outer) != {"schema_version", "bundle_status", "run_id", "artifacts"}
        or outer["schema_version"] != 1
        or outer["bundle_status"] != "complete"
        or outer["run_id"] != summary["summary_id"]
    ):
        raise ValueError("visible validation outer manifest identity drifted")
    expected_validation_names = {
        VALIDATION_EPISODE_JSONL,
        VALIDATION_EPISODE_SUMMARY_JSON,
    }
    if (
        type(outer["artifacts"]) is not dict
        or set(outer["artifacts"]) != expected_validation_names
    ):
        raise ValueError("visible validation outer artifact set drifted")
    for name, content in (
        (VALIDATION_EPISODE_JSONL, jsonl_bytes),
        (VALIDATION_EPISODE_SUMMARY_JSON, summary_bytes),
    ):
        expected_row = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "rows": content.count(b"\n") if name.endswith(".jsonl") else None,
        }
        if outer["artifacts"][name] != expected_row:
            raise ValueError(f"visible validation outer metadata differs for {name}")
    commitments = manifest["source_commitments"]
    if (
        hashlib.sha256(outer_bytes).hexdigest()
        != commitments["validation_outer_artifact_manifest_sha256"]
        or hashlib.sha256(jsonl_bytes).hexdigest()
        != commitments["validation_jsonl_sha256"]
        or summary["summary_id"] != commitments["validation_summary_id"]
    ):
        raise ValueError("visible validation copies differ from upstream commitments")


def read_m04a_input_bundle(
    visible_directory: str | Path,
    *,
    control_directory: str | Path,
    expected_manifest_sha256: str,
) -> M04AInputBundle:
    visible_root = _absolute(visible_directory)
    control_root = _absolute(control_directory)
    for path, label in (
        (visible_root, "visible input root"),
        (control_root, "external control root"),
    ):
        if (
            not os.path.lexists(path)
            or _is_link_or_reparse(path)
            or not stat.S_ISDIR(path.lstat().st_mode)
        ):
            raise ValueError(f"{label} must be a real directory")
    if _paths_overlap(visible_root, control_root):
        raise ValueError(
            "launch plan and bundle manifest must be outside visible input"
        )

    expected_manifest = _sha256(
        expected_manifest_sha256, field_name="input bundle manifest SHA-256"
    )
    manifest_bytes = _safe_read(
        control_root / BUNDLE_MANIFEST_NAME,
        expected_sha256=expected_manifest,
        label=BUNDLE_MANIFEST_NAME,
        max_bytes=MAX_MANIFEST_BYTES,
    )
    manifest = _canonical_json_payload(manifest_bytes, label=BUNDLE_MANIFEST_NAME)
    expected_fields = {
        "schema",
        "status",
        "run_id",
        "visible_layout",
        "control_layout",
        "source_commitments",
        "visible_artifacts",
        "control_artifacts",
        "launch_plan_id",
        "launch_plan_sha256",
        "bundle_id",
    }
    if (
        set(manifest) != expected_fields
        or manifest["schema"] != INPUT_BUNDLE_SCHEMA_VERSION
    ):
        raise ValueError("input bundle manifest schema is not exact")
    if (
        manifest["visible_layout"] != "flat_external_directory"
        or manifest["control_layout"] != "disjoint_external_directory"
    ):
        raise ValueError("input bundle layout declarations are unsupported")
    if manifest["status"] != "complete_immutable_m04a_input_bundle":
        raise ValueError("input bundle manifest status is not complete")
    semantic = dict(manifest)
    claimed_id = semantic.pop("bundle_id")
    if claimed_id != canonical_sha256(semantic):
        raise ValueError("input bundle bundle_id does not close")

    visible_rows = manifest["visible_artifacts"]
    expected_visible_names = set(EXPECTED_INPUT_ARTIFACT_PATHS) | {CONDA_EXPLICIT_NAME}
    if type(visible_rows) is not dict or set(visible_rows) != expected_visible_names:
        raise ValueError(
            "input bundle visible manifest does not match the frozen artifact set"
        )
    _enumerate_flat_directory(
        visible_root,
        expected_names=expected_visible_names,
        label="M04a visible input",
    )
    visible: dict[str, bytes] = {}
    for name in sorted(expected_visible_names):
        row = visible_rows[name]
        if type(row) is not dict or set(row) != {"sha256", "bytes"}:
            raise ValueError(f"visible artifact metadata is malformed: {name}")
        content = _safe_read(
            visible_root / name,
            expected_sha256=row["sha256"],
            label=f"visible artifact {name}",
        )
        if len(content) != row["bytes"]:
            raise ValueError(f"visible artifact byte count mismatch: {name}")
        visible[name] = content

    control_rows = manifest["control_artifacts"]
    if type(control_rows) is not dict or set(control_rows) != {LAUNCH_PLAN_NAME}:
        raise ValueError("input bundle control manifest is not closed-world")
    _enumerate_flat_directory(
        control_root,
        expected_names={LAUNCH_PLAN_NAME, BUNDLE_MANIFEST_NAME},
        label="M04a control input",
    )
    plan_sha = _sha256(manifest["launch_plan_sha256"], field_name="launch_plan_sha256")
    plan_row = control_rows[LAUNCH_PLAN_NAME]
    if type(plan_row) is not dict or set(plan_row) != {"sha256", "bytes"}:
        raise ValueError("control launch-plan metadata mismatch")
    plan = read_launch_plan_artifact(
        control_root / LAUNCH_PLAN_NAME,
        expected_artifact_sha256=plan_sha,
    )
    if plan_row != {"sha256": plan_sha, "bytes": len(plan.snapshot)}:
        raise ValueError("control launch-plan metadata mismatch")
    payload = plan.payload
    if (
        payload["launch_plan_id"] != manifest["launch_plan_id"]
        or payload["run_id"] != manifest["run_id"]
    ):
        raise ValueError("bundle identity differs from the launch plan")
    actual_plan_inputs = {
        name: hashlib.sha256(visible[name]).hexdigest()
        for name in EXPECTED_INPUT_ARTIFACT_PATHS
    }
    if payload["expected_input_artifacts"] != actual_plan_inputs:
        raise ValueError("visible artifacts differ from launch-plan commitments")
    if (
        payload["conda_explicit_sha256"]
        != hashlib.sha256(visible[CONDA_EXPLICIT_NAME]).hexdigest()
    ):
        raise ValueError("visible conda explicit artifact differs from launch plan")
    commitments = manifest["source_commitments"]
    expected_commitment_fields = {
        "sanitized_outer_artifact_manifest_sha256",
        "validation_outer_artifact_manifest_sha256",
        "validation_jsonl_sha256",
        "validation_summary_id",
        "runtime_source_zip_sha256",
        "runtime_source_fingerprint_sha256",
        "test_snapshot_zip_sha256",
        "test_source_fingerprint_sha256",
        "launcher_sha256",
        "frozen_contract_sha256",
        "conda_explicit_sha256",
        "python_runtime_lock_sha256",
    }
    if type(commitments) is not dict or set(commitments) != expected_commitment_fields:
        raise ValueError("source commitments do not match the exact schema")
    for field_name in expected_commitment_fields:
        _sha256(commitments[field_name], field_name=f"source_commitments.{field_name}")
    direct_commitments = {
        "reviewed_runtime_source.zip": "runtime_source_zip_sha256",
        "reviewed_test_snapshot.zip": "test_snapshot_zip_sha256",
        "remote_launcher.py": "launcher_sha256",
        "frozen_contract.md": "frozen_contract_sha256",
        CONDA_EXPLICIT_NAME: "conda_explicit_sha256",
        PYTHON_RUNTIME_LOCK_NAME: "python_runtime_lock_sha256",
    }
    for name, field_name in direct_commitments.items():
        if hashlib.sha256(visible[name]).hexdigest() != commitments[field_name]:
            raise ValueError(f"visible {name} differs from its upstream commitment")
    _validate_internal_derivations(visible, manifest=manifest)
    return M04AInputBundle(
        visible_root=visible_root,
        control_root=control_root,
        manifest_sha256=expected_manifest,
        manifest_bytes=manifest_bytes,
        launch_plan=plan,
        visible_artifacts=MappingProxyType(visible),
        _reader_token=_INPUT_BUNDLE_READER_TOKEN,
    )


def materialize_m04a_input_bundle(
    output_directory: str | Path,
    *,
    control_output_directory: str | Path,
    sources: M04AInputBundleSources,
    allow_nonproduction_fixture: bool = False,
) -> M04AInputBundle:
    """Build, atomically publish, and independently re-read one immutable bundle."""

    if type(sources) is not M04AInputBundleSources:
        raise TypeError("sources must be an exact M04AInputBundleSources")
    if type(allow_nonproduction_fixture) is not bool:
        raise TypeError("allow_nonproduction_fixture must be bool")
    if PYTHON_RUNTIME_LOCK_NAME not in EXPECTED_INPUT_ARTIFACT_PATHS:
        raise RuntimeError("launch plan has not integrated python-runtime-lock.json")

    target = _absolute(output_directory)
    control_target = _absolute(control_output_directory)
    if _paths_overlap(target, control_target):
        raise ValueError("control output must be outside the visible input root")
    for path, label in ((target, "visible input"), (control_target, "control")):
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite M04a {label}: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if _is_link_or_reparse(path.parent):
            raise ValueError(f"{label} parent may not be a symlink or reparse point")

    sanitized = read_committed_sanitized_input(
        sources.sanitized_bundle_dir,
        expected_artifact_manifest_sha256=sources.sanitized_artifact_manifest_sha256,
        require_production_counts=not allow_nonproduction_fixture,
    )
    validation, validation_artifacts, validation_refs = _validation_source_artifacts(
        sources
    )

    direct_specs = (
        (
            sources.runtime_source_zip_path,
            sources.runtime_source_zip_sha256,
            "reviewed runtime source",
            MAX_SINGLE_INPUT_BYTES,
        ),
        (
            sources.test_snapshot_zip_path,
            sources.test_snapshot_zip_sha256,
            "reviewed test snapshot",
            MAX_SINGLE_INPUT_BYTES,
        ),
        (
            sources.launcher_path,
            sources.launcher_sha256,
            "remote launcher",
            4 * 1024 * 1024,
        ),
        (
            sources.frozen_contract_path,
            sources.frozen_contract_sha256,
            "frozen contract",
            4 * 1024 * 1024,
        ),
        (
            sources.conda_explicit_path,
            sources.conda_explicit_sha256,
            "conda explicit",
            64 * 1024 * 1024,
        ),
    )
    direct: dict[str, bytes] = {}
    direct_names = (
        "reviewed_runtime_source.zip",
        "reviewed_test_snapshot.zip",
        "remote_launcher.py",
        "frozen_contract.md",
        CONDA_EXPLICIT_NAME,
    )
    direct_refs: list[_SourceReference] = []
    for name, (path, digest, label, limit) in zip(
        direct_names, direct_specs, strict=True
    ):
        direct[name] = _safe_read(
            path, expected_sha256=digest, label=label, max_bytes=limit
        )
        direct_refs.append(_SourceReference(path, digest, label, limit))
    direct[PYTHON_RUNTIME_LOCK_NAME] = _read_committed_runtime_lock(
        sources.python_runtime_lock_path,
        expected_sha256=sources.python_runtime_lock_sha256,
    )
    direct_refs.append(
        _SourceReference(
            sources.python_runtime_lock_path,
            sources.python_runtime_lock_sha256,
            "python runtime lock",
            32 * 1024 * 1024,
        )
    )
    runtime_materials = verify_source_snapshot_zip(
        direct["reviewed_runtime_source.zip"],
        expected_fingerprint_sha256=sources.runtime_source_fingerprint_sha256,
    )
    from .m04a_evidence import verify_test_source_snapshot_zip

    verify_test_source_snapshot_zip(
        direct["reviewed_test_snapshot.zip"],
        expected_fingerprint_sha256=sources.test_source_fingerprint_sha256,
    )
    if (
        runtime_materials.get("scripts/afts_arc_m04a.py")
        != direct["remote_launcher.py"]
    ):
        raise ValueError("launcher differs from reviewed runtime source bytes")
    _validate_launcher(direct["remote_launcher.py"])
    _validate_frozen_contract(direct["frozen_contract.md"])
    _validate_conda_explicit(direct[CONDA_EXPLICIT_NAME])
    _validate_opaque_runtime_lock(direct[PYTHON_RUNTIME_LOCK_NAME])

    visible: dict[str, bytes] = {
        "data_split_manifest.json": sanitized.data_split_bytes,
        "frozen_contract.md": direct["frozen_contract.md"],
        "model_config.json": serialize_json(_model_config_payload()),
        "ordered_fold_ids.json": serialize_json(_ordered_fold_payload(sanitized)),
        "parameter_count.json": serialize_json(_parameter_count_payload()),
        "quarantine_parent_ids.json": serialize_json(_quarantine_payload(sanitized)),
        "remote_launcher.py": direct["remote_launcher.py"],
        "reviewed_runtime_source.zip": direct["reviewed_runtime_source.zip"],
        "reviewed_test_snapshot.zip": direct["reviewed_test_snapshot.zip"],
        "sanitized_shard_manifest.json": serialize_json(
            _sanitized_shard_payload(sanitized)
        ),
        "schema_config_manifest.json": serialize_json(_schema_config_payload()),
        "seed_policy.json": serialize_json(_seed_policy_payload()),
        "short_exact_replay_fixture.json": serialize_json(
            build_short_exact_replay_fixture()
        ),
        **validation_artifacts,
        PYTHON_RUNTIME_LOCK_NAME: direct[PYTHON_RUNTIME_LOCK_NAME],
        CONDA_EXPLICIT_NAME: direct[CONDA_EXPLICIT_NAME],
    }
    if set(visible) != set(EXPECTED_INPUT_ARTIFACT_PATHS) | {CONDA_EXPLICIT_NAME}:
        raise RuntimeError(
            "materialized visible inputs do not match the frozen launch-plan set"
        )

    validation_commitment = build_validation_manifest_commitment(
        validation,
        expected_artifact_manifest_sha256=sources.validation_artifact_manifest_sha256,
        expected_jsonl_sha256=str(validation.summary["jsonl_sha256"]),
    )
    expected_inputs = {
        name: hashlib.sha256(visible[name]).hexdigest()
        for name in EXPECTED_INPUT_ARTIFACT_PATHS
    }
    run_root = f"{sources.remote_project_root}/runs/{sources.run_id}"
    plan_payload = build_launch_plan(
        attempt_nonce=sources.attempt_nonce,
        run_id=sources.run_id,
        remote_project_root=sources.remote_project_root,
        run_root=run_root,
        expected_input_artifacts=expected_inputs,
        runtime_source_fingerprint_sha256=sources.runtime_source_fingerprint_sha256,
        test_source_fingerprint_sha256=sources.test_source_fingerprint_sha256,
        training_config_sha256=training_config_sha256(),
        validation_manifest_commitment=validation_commitment.to_json_dict(),
        conda_explicit_sha256=sources.conda_explicit_sha256,
        ordered_import_roots=sources.ordered_import_roots,
    )
    plan_bytes = canonical_launch_plan_bytes(plan_payload)
    bundle_manifest = _bundle_manifest_payload(
        visible=visible,
        launch_plan_bytes=plan_bytes,
        launch_plan_id=plan_payload["launch_plan_id"],
        sources=sources,
        validation=validation,
    )
    bundle_manifest_bytes = serialize_json(bundle_manifest)
    bundle_manifest_sha = hashlib.sha256(bundle_manifest_bytes).hexdigest()

    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    control_staging = control_target.parent / (
        f".{control_target.name}.staging-{uuid.uuid4().hex}"
    )
    staging.mkdir(mode=0o700, exist_ok=False)
    control_staging.mkdir(mode=0o700, exist_ok=False)
    try:
        for name, content in sorted(visible.items()):
            relative = PurePosixPath(name)
            if len(relative.parts) != 1 or relative.as_posix() != name:
                raise ValueError(f"visible artifact name is not flat and safe: {name}")
            _write_new(staging / name, content)
        _write_new(control_staging / LAUNCH_PLAN_NAME, plan_bytes)
        _write_new(control_staging / BUNDLE_MANIFEST_NAME, bundle_manifest_bytes)

        source_refs = [
            _SourceReference(
                sources.sanitized_bundle_dir / "artifact_manifest.json",
                sources.sanitized_artifact_manifest_sha256,
                "sanitized outer manifest",
                MAX_MANIFEST_BYTES,
            ),
            _SourceReference(
                sources.sanitized_bundle_dir / "data_split_manifest.json",
                hashlib.sha256(sanitized.data_split_bytes).hexdigest(),
                "sanitized data split",
                MAX_MANIFEST_BYTES,
            ),
            *(
                _SourceReference(
                    sources.sanitized_bundle_dir / name,
                    str(sanitized.outer["artifacts"][name]["sha256"]),
                    f"sanitized shard {name}",
                )
                for name in SANITIZED_SHARD_NAMES
            ),
            *validation_refs,
            *direct_refs,
        ]
        _assert_sources_unchanged(source_refs)
        _enumerate_flat_directory(
            sources.sanitized_bundle_dir,
            expected_names=set(SANITIZED_SOURCE_NAMES),
            label="final sanitized split drift check",
        )
        _enumerate_flat_directory(
            sources.validation_manifest_dir,
            expected_names={
                "artifact_manifest.json",
                VALIDATION_EPISODE_JSONL,
                VALIDATION_EPISODE_SUMMARY_JSON,
            },
            label="final validation manifest drift check",
        )
        if os.path.lexists(target) or os.path.lexists(control_target):
            raise FileExistsError(
                "an M04a input publication target appeared during staging"
            )
        # The flat visible root is the activation point.  Publish the inert
        # external control material first, then atomically activate visible
        # inputs with RENAME_NOREPLACE; a late failure can leave only an inert
        # control directory, never a runnable visible root without its plan.
        _rename_directory_noreplace(control_staging, control_target)
        try:
            _rename_directory_noreplace(staging, target)
        except BaseException as exc:
            raise RuntimeError(
                "external control was published but visible activation failed; "
                f"preserved visible staging at {staging}: {exc}"
            ) from exc
    except BaseException as exc:
        raise RuntimeError(
            "M04a input staging failed; any existing staging/publication is preserved: "
            f"visible={staging}, control={control_staging}: {exc}"
        ) from exc

    return read_m04a_input_bundle(
        target,
        control_directory=control_target,
        expected_manifest_sha256=bundle_manifest_sha,
    )


def replay_validate_m04a_input_bundle(
    bundle: M04AInputBundle,
    *,
    sources: M04AInputBundleSources,
    allow_nonproduction_fixture: bool = False,
) -> dict[str, object]:
    """Re-read all upstream sources and prove the published derivations again."""

    if (
        type(bundle) is not M04AInputBundle
        or bundle._reader_token is not _INPUT_BUNDLE_READER_TOKEN
    ):
        raise TypeError("bundle must be reader-produced")
    sanitized = read_committed_sanitized_input(
        sources.sanitized_bundle_dir,
        expected_artifact_manifest_sha256=sources.sanitized_artifact_manifest_sha256,
        require_production_counts=not allow_nonproduction_fixture,
    )
    validation, validation_files, _ = _validation_source_artifacts(sources)
    expected_upstream_copies = {
        "data_split_manifest.json": sanitized.data_split_bytes,
        "sanitized_shard_manifest.json": serialize_json(
            _sanitized_shard_payload(sanitized)
        ),
        "ordered_fold_ids.json": serialize_json(_ordered_fold_payload(sanitized)),
        "quarantine_parent_ids.json": serialize_json(_quarantine_payload(sanitized)),
        **validation_files,
    }
    for name, expected in expected_upstream_copies.items():
        if bundle.visible_artifacts[name] != expected:
            raise ValueError(f"published {name} does not replay from upstream bytes")
    direct_specs = {
        "reviewed_runtime_source.zip": (
            sources.runtime_source_zip_path,
            sources.runtime_source_zip_sha256,
            MAX_SINGLE_INPUT_BYTES,
        ),
        "reviewed_test_snapshot.zip": (
            sources.test_snapshot_zip_path,
            sources.test_snapshot_zip_sha256,
            MAX_SINGLE_INPUT_BYTES,
        ),
        "remote_launcher.py": (
            sources.launcher_path,
            sources.launcher_sha256,
            4 * 1024 * 1024,
        ),
        "frozen_contract.md": (
            sources.frozen_contract_path,
            sources.frozen_contract_sha256,
            4 * 1024 * 1024,
        ),
        CONDA_EXPLICIT_NAME: (
            sources.conda_explicit_path,
            sources.conda_explicit_sha256,
            64 * 1024 * 1024,
        ),
    }
    for name, (path, digest, maximum) in direct_specs.items():
        expected = _safe_read(
            path,
            expected_sha256=digest,
            label=f"replay upstream {name}",
            max_bytes=maximum,
        )
        if bundle.visible_artifacts[name] != expected:
            raise ValueError(f"published {name} differs from upstream committed bytes")
    runtime_lock = _read_committed_runtime_lock(
        sources.python_runtime_lock_path,
        expected_sha256=sources.python_runtime_lock_sha256,
    )
    if bundle.visible_artifacts[PYTHON_RUNTIME_LOCK_NAME] != runtime_lock:
        raise ValueError("published Python runtime lock differs from its typed reader")
    manifest = bundle.manifest
    if manifest["source_commitments"] != _source_commitments(
        sources, validation=validation
    ):
        raise ValueError(
            "bundle source commitments differ from supplied upstream sources"
        )
    return {
        "status": "VERIFIED",
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": bundle.manifest_sha256,
        "launch_plan_sha256": bundle.launch_plan.artifact_sha256,
        "visible_artifact_count": len(bundle.visible_artifacts),
    }


__all__ = [
    "BUNDLE_MANIFEST_NAME",
    "CONDA_EXPLICIT_NAME",
    "CONTROL_DIRECTORY_NAME",
    "INPUT_BUNDLE_SCHEMA_VERSION",
    "M04AInputBundle",
    "M04AInputBundleSources",
    "PYTHON_RUNTIME_LOCK_NAME",
    "VISIBLE_DIRECTORY_NAME",
    "build_short_exact_replay_fixture",
    "materialize_m04a_input_bundle",
    "read_committed_sanitized_input",
    "read_m04a_input_bundle",
    "replay_validate_m04a_input_bundle",
    "validate_short_exact_replay_fixture",
]
