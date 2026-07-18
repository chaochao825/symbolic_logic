"""Closed-world evidence schemas and validators for the frozen M04a source.

The module is deliberately pure Python.  It defines the oracle-free blind-input
shape sidecar, trace and accounting ledgers, and file-backed artifact publication
without importing PyTorch.  Neural execution code is expected to emit these rows;
this module only validates and content-addresses them.
"""

from __future__ import annotations

import hashlib
import importlib
import errno
import io
import json
import math
import os
import re
import stat
import struct
import uuid
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path, PurePosixPath
from types import MappingProxyType, ModuleType
from typing import Any

from .blind import BlindTask
from ._source_bootstrap import named_bytes_fingerprint
from .grid import as_grid, grid_key, grid_to_lists
from .manifest import serialize_json, serialize_jsonl, verify_source_snapshot_zip
from .m04a_production_data_contract import PRODUCTION_DATA_CLOSURE_ID


# The fixed development population and its already-published blind DSL-v0.6 pool.
FIXED_BLIND_CASE_SET_ID = (
    "93fe4870982313fb840d27bc4082e079f41d8b58d4f679cea15b5e5b1381b177"
)
FIXED_BLIND_TASKS_SHA256 = (
    "e80dacf901852e878d49bc2f2fe124792460a2bbfe4553adc10c2e36aea7f290"
)
FIXED_BLIND_ARTIFACT_MANIFEST_SHA256 = (
    "7df2afbe088ad866f329ab654d90dd1678a0e985adee7af138fee692e66aab3b"
)
FIXED_DSL_POOL_CONTENT_ID = (
    "e2b5ea2084fdf1ac072a6e15c7d13024dd8c63a019961b4c1c4b9fb0386ad49d"
)
FIXED_DSL_POOL_SPEC_ID = (
    "6772be7c69a768b15d8e9c8f7b788400acf9feba311de97a613c4968fc08f232"
)
FIXED_DSL_POOL_ARTIFACT_MANIFEST_SHA256 = (
    "3a39d4703955a5359166b4a0ffdcf73689d62e7c2468b8b7525d2428c570ca7e"
)
FIXED_DSL_SEMANTICS_VERSION = "afts-grid-dsl/v0.6"
FIXED_BLIND_TASK_COUNT = 20
FIXED_BLIND_TEST_PAIR_COUNT = 21
FIXED_ZERO_CANDIDATE_TASK_COUNT = 12
FIXED_BLIND_SHAPE_SIDECAR_ID = (
    "eb9d22761d45e453d0e912b1e82c092e1d0a1f626a56e2e9c119c2812957cf5c"
)
FIXED_BLIND_SHAPE_ROWS_SHA256 = (
    "4a0184d8d952a09bf3059628a6ee2f24cce46c3f501a3871abd945a6b96d57f0"
)

BLIND_INPUT_ROW_SCHEMA_VERSION = "afts-m04a-blind-input/v0.1"
BLIND_SHAPE_SIDECAR_SCHEMA_VERSION = "afts-m04a-blind-shape-sidecar/v0.1"
FILE_BUNDLE_SCHEMA_VERSION = "afts-m04a-file-bundle/v0.1"
POOL_COST_SUMMARY_SCHEMA_VERSION = "afts-m04a-pool-cost-summary/v0.1"
POOL_REPLAY_RECEIPT_SCHEMA_VERSION = "afts-m04a-pool-replay-receipt/v0.1"
TRAINING_COST_SUMMARY_SCHEMA_VERSION = "afts-training-cost-summary/v0.2"

# Training-artifact schemas are duplicated here deliberately: this module is the
# pure-Python, no-Torch verifier used at publication time.  Neural producers use
# the same literal versions from m04a_preflight/m04a_train_contract.
PREFLIGHT_FIXTURE_SCHEMA_VERSION = "afts-grid-cmlm-preflight-fixture/v0.1"
PREFLIGHT_INFERENCE_SCHEMA_VERSION = "afts-grid-cmlm-preflight-inference/v0.1"
PREFLIGHT_PROJECTION_SCHEMA_VERSION = "afts-grid-cmlm-preflight-projection/v0.4"
PREFLIGHT_REPORT_SCHEMA_VERSION = "afts-grid-cmlm-preflight-report/v0.4"
PREFLIGHT_VALIDATION_COMMITMENT_SCHEMA_VERSION = (
    "afts-m04a-validation-manifest-commitment/v0.1"
)
PREFLIGHT_FRESH_RECONSTRUCTION_PROBE_SCHEMA_VERSION = (
    "afts-grid-cmlm-preflight-fresh-reconstruction-probe/v0.1"
)
PREFLIGHT_CHECKPOINT_COST_PROBE_SCHEMA_VERSION = (
    "afts-grid-cmlm-preflight-checkpoint-cost-probe/v0.1"
)
PREFLIGHT_FINAL_SELECTION_PROBE_SCHEMA_VERSION = (
    "afts-grid-cmlm-preflight-final-selection-probe/v0.1"
)
PREFLIGHT_LOCK_HANDSHAKE_PROBE_SCHEMA_VERSION = (
    "afts-grid-cmlm-preflight-lock-handshake-probe/v0.1"
)
LOCK_HANDSHAKE_ARTIFACT_SCHEMA_VERSION = "afts-m04a-held-gpu-lock-handshake/v0.2"
CAMPAIGN_OVERHEAD_PROBE_SCHEMA_VERSION = (
    "afts-grid-cmlm-campaign-overhead-cost-probe/v0.1"
)
CAMPAIGN_OVERHEAD_PROJECTION_SCHEMA_VERSION = (
    "afts-grid-cmlm-campaign-overhead-projection/v0.1"
)
PREFLIGHT_SELECTED_METRIC_SUMMARY_SCHEMA_VERSION = (
    "afts-grid-cmlm-preflight-selected-metric-summary/v0.1"
)
CHECKPOINT_MANIFEST_SCHEMA_VERSION = "afts-grid-cmlm-checkpoint-manifest/v0.1"
VALIDATION_METRIC_SCHEMA_VERSION = "afts-grid-cmlm-validation-metric/v0.2"
SELECTED_CHECKPOINT_MANIFEST_SCHEMA_VERSION = "afts-grid-cmlm-checkpoint-selection/v0.2"
TRAINING_LOCK_INTERVAL_SCHEMA_VERSION = "afts-grid-cmlm-lock-interval/v0.1"
OPTIMIZER_SCHEDULE_STATE_SCHEMA_VERSION = (
    "afts-m04a-optimizer-schedule-state/v0.1"
)

PREFLIGHT_UPDATES = 100
PREFLIGHT_MICROBATCHES_PER_UPDATE = 16
PREFLIGHT_GRID_SIDE = 30
PREFLIGHT_TARGET_CELL_COUNT = PREFLIGHT_GRID_SIDE * PREFLIGHT_GRID_SIDE
PREFLIGHT_MASKED_TOKEN_PREDICTIONS_PER_UPDATE = (
    PREFLIGHT_MICROBATCHES_PER_UPDATE * PREFLIGHT_TARGET_CELL_COUNT
)
PREFLIGHT_LANES = 8
PREFLIGHT_DENOISING_STEPS = 12
OPTIMIZER_UPDATES = 20_000
VALIDATION_INTERVAL = 2_000
VALIDATION_PASS_COUNT = OPTIMIZER_UPDATES // VALIDATION_INTERVAL
CHECKPOINT_SELECTION_RULE = "minimum_parent_grouped_masked_cell_ce_then_earlier_step"
VALIDATION_AGGREGATION_SEMANTICS = "sequential_round_to_fp32_views_targets_parents/v0.1"
OVERHEAD_PROJECTION_METHOD = (
    "measured_lock_handshake_plus_measured_fresh_reconstruction_plus_"
    f"{VALIDATION_PASS_COUNT}_measured_checkpoint_writes_plus_"
    f"{VALIDATION_PASS_COUNT}_verified_weights_only_loads_plus_measured_"
    f"{VALIDATION_PASS_COUNT}_metric_final_selection"
)
PREFLIGHT_PROJECTION_METHOD = (
    "measured_lock_handshake_plus_handshake_to_preflight_gap_plus_preflight_"
    "endpoint_plus_measured_fresh_"
    f"reconstruction_plus_ceil_max_context_update_rate_x{OPTIMIZER_UPDATES}"
    "_plus_ceil_"
    f"microbatch_rate_x{VALIDATION_PASS_COUNT}_validation_manifests_plus_"
    f"{VALIDATION_PASS_COUNT}_measured_checkpoint_roundtrips_plus_measured_"
    f"{VALIDATION_PASS_COUNT}_metric_final_selection_plus_5min_"
    "post_preflight_validation_margin"
)
SELECTION_TIE_INDICES = (
    VALIDATION_PASS_COUNT - 3,
    VALIDATION_PASS_COUNT - 1,
)

GPU_HOUR_BUDGET_NS = 24 * 60 * 60 * 1_000_000_000
MAX_HANDSHAKE_TO_PREFLIGHT_NS = 10 * 60 * 1_000_000_000
PREFLIGHT_POST_VALIDATION_MARGIN_NS = 5 * 60 * 1_000_000_000

# These sets are intentionally exact.  A complete production bundle is evidence,
# not a bag of convenient output files: omitting any row ledger or parent
# commitment makes the corresponding result incomplete.
TRAINING_ARTIFACT_FILES = frozenset(
    {
        "checkpoint_manifests.jsonl",
        "data_split_manifest.json",
        "environment_manifest.json",
        "frozen_contract.md",
        "model_config.json",
        "optimizer_schedule_state.json",
        "ordered_fold_ids.json",
        "parameter_count.json",
        "python-runtime-lock.json",
        "launch_plan.json",
        "lock_handshake_artifact.json",
        "preflight.json",
        "preflight_diagnostic_checkpoint.pt",
        "preflight_overhead_cost_report.json",
        "quarantine_parent_ids.json",
        "remote_launcher.py",
        "resume_manifests.jsonl",
        "reviewed_runtime_source.zip",
        "reviewed_test_snapshot.zip",
        "sanitized_shard_manifest.json",
        "schema_config_manifest.json",
        "seed_policy.json",
        "selected_checkpoint.pt",
        "selected_checkpoint_manifest.json",
        "short_exact_replay_fixture.json",
        "training_cost_ledger.jsonl",
        "training_cost_summary.json",
        "validation_metrics.jsonl",
        "validation_episode_manifest.jsonl",
        "validation_episode_manifest_summary.json",
        "validation_episode_outer_manifest.json",
    }
)

POOL_ARTIFACT_FILES = frozenset(
    {
        "batch_forward_ledger.jsonl",
        "blind_artifact_manifest.json",
        "blind_input_shape_manifest.json",
        "blind_input_shape_rows.jsonl",
        "blind_tasks.jsonl",
        "candidate_rows.jsonl",
        "encoder_forward_ledger.jsonl",
        "environment_manifest.json",
        "lane_traces.jsonl",
        "pair_cost_ledger.jsonl",
        "pool_cost_summary.json",
        "pool_setup_cost.json",
        "pool_summary.json",
        "replay_verification_receipt.json",
        "sampler_config.json",
        "selected_checkpoint_manifest.json",
        "source_snapshot.zip",
        "training_artifact_manifest.json",
    }
)

TRAINING_ARTIFACT_MANIFEST_SCHEMA_VERSION = "afts-m04a-training-artifact-manifest/v0.4"
POOL_ARTIFACT_MANIFEST_SCHEMA_VERSION = "afts-m04a-pool-artifact-manifest/v0.3"
EVALUATION_ARTIFACT_MANIFEST_SCHEMA_VERSION = (
    "afts-m04a-evaluation-artifact-manifest/v0.3"
)

NO_SHAPE_PROPOSAL = "NO_SHAPE_PROPOSAL"
SHAPE_READY = "SHAPE_READY"
MASK_SENTINEL = "MASK"

_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GPU_UUID_PATTERN = re.compile(
    r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
_BOOT_ID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
_REPARSE_POINT_ATTRIBUTE = 0x400
_MAX_ARTIFACT_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_SINGLE_BUNDLE_ARTIFACT_BYTES = 512 * 1024 * 1024
_MAX_TOTAL_BUNDLE_ARTIFACT_BYTES = 1024 * 1024 * 1024
_MAX_BUNDLE_ARTIFACT_COUNT = 4096
_VERIFIED_BUNDLE_TOKEN = object()
_CONTRACT_MODULE: ModuleType | None = None
_TRAIN_CONTRACT_MODULE: ModuleType | None = None

_WINDOWS_FILE_SHARE_READ = 0x00000001
_WINDOWS_FILE_SHARE_WRITE = 0x00000002
_WINDOWS_FILE_SHARE_DELETE = 0x00000004
_WINDOWS_OPEN_EXISTING = 3
_WINDOWS_OPEN_REPARSE_POINT = 0x00200000
_WINDOWS_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000

_FALLBACK_CONTRACT: dict[str, Any] = {
    "MODEL_SEMANTICS_VERSION": "afts-grid-cmlm/v0.1",
    "MODEL_STAGE": "M04a_global_masked_grid_generation",
    "SAMPLER_SEMANTICS_VERSION": "afts-maskgit-cosine/v0.1",
    "OUTPUT_SHAPE_SEMANTICS_VERSION": "afts-output-shape/v0.1",
    "TRAINING_COST_LEDGER_SCHEMA_VERSION": "afts-training-cost-ledger/v0.1",
    "LANE_ROW_SCHEMA_VERSION": "afts-m04a-lane/v0.1",
    "LANE_TRACE_SCHEMA_VERSION": "afts-maskgit-trace/v0.1",
    "ENCODER_FORWARD_LEDGER_SCHEMA_VERSION": "afts-encoder-forward-ledger/v0.1",
    "DECODER_FORWARD_LEDGER_SCHEMA_VERSION": "afts-decoder-forward-ledger/v0.1",
    "PAIR_COST_SCHEMA_VERSION": "afts-m04a-pair-cost/v0.1",
    "POOL_SETUP_COST_SCHEMA_VERSION": "afts-m04a-pool-setup-cost/v0.1",
    "MODEL_PARAMETER_COUNT": 8_733_706,
    "GRADIENT_ACCUMULATION": PREFLIGHT_MICROBATCHES_PER_UPDATE,
    "OPTIMIZER_UPDATES": OPTIMIZER_UPDATES,
    "VALIDATION_INTERVAL": VALIDATION_INTERVAL,
    "VALIDATION_PASS_COUNT": VALIDATION_PASS_COUNT,
    "DENOISING_STEPS": 12,
    "MAX_ACCEPTED_SHAPES": 4,
    "MAX_GRID_SIDE": 30,
    "INFERENCE_MICROBATCH": 8,
}


def _contract() -> ModuleType | None:
    """Import the pure semantic contract only when one of its values is needed."""

    global _CONTRACT_MODULE
    if _CONTRACT_MODULE is None:
        try:
            _CONTRACT_MODULE = importlib.import_module(".m04a_contract", __package__)
        except ModuleNotFoundError as exc:
            if exc.name not in {
                "afts_arc.m04a_contract",
                f"{__package__}.m04a_contract",
            }:
                raise
    return _CONTRACT_MODULE


def _contract_value(name: str) -> Any:
    module = _contract()
    if module is not None and hasattr(module, name):
        return getattr(module, name)
    return _FALLBACK_CONTRACT[name]


def _train_contract() -> ModuleType:
    """Load only the pure optimizer contract; importing this must not load Torch."""

    global _TRAIN_CONTRACT_MODULE
    if _TRAIN_CONTRACT_MODULE is None:
        _TRAIN_CONTRACT_MODULE = importlib.import_module(
            ".m04a_train_contract", __package__
        )
    return _TRAIN_CONTRACT_MODULE


def _canonical_json_bytes(payload: object) -> bytes:
    module = _contract()
    if module is not None and hasattr(module, "canonical_json_bytes"):
        return module.canonical_json_bytes(payload)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _strict_int(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(f"{field} must be an integer >= {minimum}")
    return value


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field} must be a non-empty string")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise TypeError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _finite_number(
    value: object,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{field} must be <= {maximum}")
    return normalized


def _exact_fields(payload: object, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object")
    if set(payload) != expected:
        raise ValueError(
            f"{label} fields must be {sorted(expected)}, found {sorted(payload)}"
        )
    return payload


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is not allowed: {key!r}")
        result[key] = value
    return result


def _parse_json_bytes(content: bytes, *, label: str) -> Any:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8 JSON") from exc
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc


def _read_json(path: Path) -> Any:
    return _parse_json_bytes(path.read_bytes(), label=str(path))


def _parse_jsonl_bytes(content: bytes, *, label: str) -> tuple[dict[str, Any], ...]:
    if content and not content.endswith(b"\n"):
        raise ValueError(f"JSONL artifact must end with a newline: {label}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line:
            raise ValueError(f"blank JSONL row at {label}:{line_number}")
        parsed = _parse_json_bytes(line, label=f"{label}:{line_number}")
        if not isinstance(parsed, dict):
            raise TypeError(f"JSONL row must be an object: {label}:{line_number}")
        rows.append(parsed)
    return tuple(rows)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    return _parse_jsonl_bytes(path.read_bytes(), label=str(path))


_FORBIDDEN_BLIND_KEYS = {
    "answer",
    "answers",
    "ground_truth",
    "ground_truths",
    "oracle_grid",
    "oracle_output",
    "oracle_outputs",
    "oracle_shape",
    "oracle_shapes",
    "query_output",
    "query_outputs",
    "query_output_shape",
    "query_output_shapes",
    "test_output",
    "test_outputs",
}


def _looks_like_oracle_path(value: str) -> bool:
    if "/" not in value and "\\" not in value:
        return False
    components = re.split(r"[/\\]+", value.lower())
    return any("oracle" in component for component in components)


def assert_oracle_free_payload(payload: object, *, location: str = "payload") -> None:
    """Reject privileged query-output fields and paths in a blind/pool payload."""

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if not isinstance(key, str):
                raise TypeError(f"{location} contains a non-string object key")
            lowered = key.lower()
            compact = re.sub(r"[^a-z0-9]+", "", lowered)
            child = f"{location}.{key}"
            if lowered == "oracle_inputs_available_to_pool":
                if value is not False:
                    raise ValueError(f"{child} must be false")
            elif (
                lowered in _FORBIDDEN_BLIND_KEYS
                or compact in {"answer", "answers"}
                or any(
                    marker in compact
                    for marker in ("oracle", "queryoutput", "testoutput", "groundtruth")
                )
            ):
                raise ValueError(f"oracle/query-output field is forbidden at {child}")
            assert_oracle_free_payload(value, location=child)
        return
    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            assert_oracle_free_payload(value, location=f"{location}[{index}]")
        return
    if isinstance(payload, str) and _looks_like_oracle_path(payload):
        raise ValueError(f"oracle path is forbidden at {location}")


def _is_windows_reserved_component(part: str) -> bool:
    if not part or part[-1] in {" ", "."}:
        return True
    stem = part.split(".", 1)[0].rstrip(" ").upper()
    return stem in {"CON", "PRN", "AUX", "NUL"} or bool(
        re.fullmatch(r"(?:COM|LPT)[1-9]", stem)
    )


def _safe_artifact_name(name: object, *, allow_manifest: bool = False) -> str:
    if not isinstance(name, str):
        raise TypeError("artifact name must be a string")
    relative = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or ":" in name
        or any(ord(character) < 32 or character in '<>"|?*' for character in name)
        or relative.is_absolute()
        or relative.as_posix() != name
        or any(part in {"", ".", ".."} for part in relative.parts)
        or any(_is_windows_reserved_component(part) for part in relative.parts)
    ):
        raise ValueError(f"unsafe artifact path: {name!r}")
    if not allow_manifest and name == "artifact_manifest.json":
        raise ValueError("artifact_manifest.json is reserved")
    return name


def _file_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    rows = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            rows += chunk.count(b"\n")
    return {
        "sha256": digest.hexdigest(),
        "bytes": size,
        "rows": rows if path.name.endswith(".jsonl") else None,
    }


def _path_has_reparse_point(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE
    )


def _assert_existing_nonsymlink_chain(path: Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        if not os.path.lexists(current):
            raise ValueError(f"{label} ancestor does not exist")
        if _path_has_reparse_point(current):
            raise ValueError(f"{label} cannot traverse a symlink or reparse point")


def _windows_component_share_mode(*, is_leaf: bool) -> int:
    """Keep directory identities immovable while a Windows path is traversed."""

    share_mode = _WINDOWS_FILE_SHARE_READ | _WINDOWS_FILE_SHARE_WRITE
    if is_leaf:
        # The regular-file snapshot is already bound to its handle, so allowing a
        # leaf rename preserves normal read-only interoperability.  Directory
        # handles deliberately omit FILE_SHARE_DELETE: otherwise an attacker can
        # rename-swap the verified root between opening its handle and its leaf.
        share_mode |= _WINDOWS_FILE_SHARE_DELETE
    return share_mode


def _create_windows_component_handle(
    create_file: Callable[..., Any],
    path: Path,
    *,
    desired_access: int,
    is_leaf: bool,
) -> Any:
    """Call CreateFileW with the share policy used by all verified path opens."""

    return create_file(
        os.fspath(path),
        desired_access,
        _windows_component_share_mode(is_leaf=is_leaf),
        None,
        _WINDOWS_OPEN_EXISTING,
        _WINDOWS_OPEN_REPARSE_POINT | _WINDOWS_FILE_FLAG_BACKUP_SEMANTICS,
        None,
    )


def _open_regular_nofollow(
    path: Path,
    *,
    label: str,
    canonical_path_out: list[Path] | None = None,
    expected_parent_identity: tuple[int, int] | None = None,
    parent_identity_out: list[tuple[int, int]] | None = None,
) -> int:
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if os.name == "posix":
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            current_fd = os.open(absolute.anchor, directory_flags)
            try:
                for component in absolute.parts[1:-1]:
                    next_fd = os.open(component, directory_flags, dir_fd=current_fd)
                    if not stat.S_ISDIR(os.fstat(next_fd).st_mode):
                        os.close(next_fd)
                        raise ValueError(f"{label} ancestor is not a real directory")
                    os.close(current_fd)
                    current_fd = next_fd
                parent_stat = os.fstat(current_fd)
                parent_identity = (int(parent_stat.st_dev), int(parent_stat.st_ino))
                if (
                    expected_parent_identity is not None
                    and parent_identity != expected_parent_identity
                ):
                    raise ValueError(f"{label} parent directory identity changed")
                descriptor = os.open(absolute.parts[-1], file_flags, dir_fd=current_fd)
                if canonical_path_out is not None:
                    canonical_path_out.append(absolute)
                if parent_identity_out is not None:
                    parent_identity_out.append(parent_identity)
                return descriptor
            finally:
                os.close(current_fd)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ValueError(
                    f"{label} cannot traverse a symlink or reparse point"
                ) from exc
            raise

    if os.name == "nt":
        import ctypes
        import msvcrt
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create_file.restype = wintypes.HANDLE
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL

        class _FileTime(ctypes.Structure):
            _fields_ = (("low", wintypes.DWORD), ("high", wintypes.DWORD))

        class _ByHandleFileInformation(ctypes.Structure):
            _fields_ = (
                ("attributes", wintypes.DWORD),
                ("creation_time", _FileTime),
                ("access_time", _FileTime),
                ("write_time", _FileTime),
                ("volume_serial", wintypes.DWORD),
                ("size_high", wintypes.DWORD),
                ("size_low", wintypes.DWORD),
                ("link_count", wintypes.DWORD),
                ("file_index_high", wintypes.DWORD),
                ("file_index_low", wintypes.DWORD),
            )

        get_file_information = kernel32.GetFileInformationByHandle
        get_file_information.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_ByHandleFileInformation),
        ]
        get_file_information.restype = wintypes.BOOL
        get_final_path = kernel32.GetFinalPathNameByHandleW
        get_final_path.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        get_final_path.restype = ctypes.c_uint32

        def canonical_handle_path(handle: int) -> str:
            size = get_final_path(handle, None, 0, 0)
            if size == 0:
                raise OSError(
                    ctypes.get_last_error(),
                    f"cannot resolve final handle path for {label}",
                )
            buffer = ctypes.create_unicode_buffer(size + 1)
            written = get_final_path(handle, buffer, len(buffer), 0)
            if written == 0 or written >= len(buffer):
                raise OSError(
                    ctypes.get_last_error(),
                    f"cannot read final handle path for {label}",
                )
            result = buffer.value
            if result.startswith("\\\\?\\UNC\\"):
                result = "\\\\" + result[8:]
            elif result.startswith("\\\\?\\"):
                result = result[4:]
            return os.path.normpath(result)

        file_read_attributes = 0x00000080
        generic_read = 0x80000000
        directory_attribute = 0x00000010
        reparse_attribute = 0x00000400
        invalid_handle = ctypes.c_void_p(-1).value
        held_directories: list[int] = []
        leaf_handle: int | None = None
        previous_canonical: str | None = None
        last_directory_identity: tuple[int, int] | None = None
        current = Path(absolute.anchor)
        paths = [current]
        for component in absolute.parts[1:]:
            current = current / component
            paths.append(current)
        try:
            for index, component_path in enumerate(paths):
                is_leaf = index == len(paths) - 1
                handle = _create_windows_component_handle(
                    create_file,
                    component_path,
                    desired_access=(generic_read if is_leaf else file_read_attributes),
                    is_leaf=is_leaf,
                )
                if handle == invalid_handle:
                    raise OSError(
                        ctypes.get_last_error(),
                        f"cannot open path component for {label}",
                    )
                information = _ByHandleFileInformation()
                if not get_file_information(handle, ctypes.byref(information)):
                    error = ctypes.get_last_error()
                    close_handle(handle)
                    raise OSError(error, f"cannot inspect path component for {label}")
                if information.attributes & reparse_attribute:
                    close_handle(handle)
                    raise ValueError(
                        f"{label} cannot traverse a symlink or reparse point"
                    )
                if is_leaf == bool(information.attributes & directory_attribute):
                    close_handle(handle)
                    raise ValueError(f"{label} path component has the wrong file type")
                current_canonical = canonical_handle_path(handle)
                if previous_canonical is not None and os.path.normcase(
                    os.path.normpath(os.path.dirname(current_canonical))
                ) != os.path.normcase(os.path.normpath(previous_canonical)):
                    close_handle(handle)
                    raise ValueError(
                        f"{label} handle escaped through a reparse ancestor"
                    )
                previous_canonical = current_canonical
                if is_leaf:
                    leaf_handle = int(handle)
                else:
                    held_directories.append(int(handle))
                    last_directory_identity = (
                        int(information.volume_serial),
                        (int(information.file_index_high) << 32)
                        | int(information.file_index_low),
                    )
            if leaf_handle is None:
                raise ValueError(f"{label} does not identify a regular file")
            if last_directory_identity is None:
                raise ValueError(f"{label} has no bound parent directory")
            if (
                expected_parent_identity is not None
                and last_directory_identity != expected_parent_identity
            ):
                raise ValueError(f"{label} parent directory identity changed")
            descriptor = msvcrt.open_osfhandle(
                leaf_handle, os.O_RDONLY | getattr(os, "O_BINARY", 0)
            )
            leaf_handle = None
            if canonical_path_out is not None:
                canonical_path_out.append(Path(previous_canonical))
            if parent_identity_out is not None:
                parent_identity_out.append(last_directory_identity)
            return descriptor
        finally:
            if leaf_handle is not None:
                close_handle(leaf_handle)
            for handle in reversed(held_directories):
                close_handle(handle)

    _assert_existing_nonsymlink_chain(absolute, label=label)
    parent_stat = absolute.parent.stat(follow_symlinks=False)
    parent_identity = (int(parent_stat.st_dev), int(parent_stat.st_ino))
    if (
        expected_parent_identity is not None
        and parent_identity != expected_parent_identity
    ):
        raise ValueError(f"{label} parent directory identity changed")
    descriptor = os.open(absolute, file_flags)
    if canonical_path_out is not None:
        canonical_path_out.append(absolute)
    if parent_identity_out is not None:
        parent_identity_out.append(parent_identity)
    return descriptor


def _read_regular_snapshot(
    path: Path,
    *,
    label: str,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
    expected_rows: int | None = None,
    max_bytes: int | None = None,
    canonical_path_out: list[Path] | None = None,
    expected_parent_identity: tuple[int, int] | None = None,
    parent_identity_out: list[tuple[int, int]] | None = None,
) -> bytes:
    if expected_bytes is not None:
        byte_count = _strict_int(expected_bytes, field=f"{label}.bytes")
    else:
        byte_count = None
    if max_bytes is not None:
        max_count = _strict_int(max_bytes, field=f"{label}.max_bytes", minimum=1)
    elif byte_count is not None:
        max_count = byte_count
    else:
        raise ValueError("unbounded regular-file snapshots are forbidden")
    descriptor = _open_regular_nofollow(
        path,
        label=label,
        canonical_path_out=canonical_path_out,
        expected_parent_identity=expected_parent_identity,
        parent_identity_out=parent_identity_out,
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 0
            or before.st_size > max_count
            or (byte_count is not None and before.st_size != byte_count)
        ):
            raise ValueError(
                f"{label} metadata mismatch: not the expected bounded regular file"
            )
        chunks: list[bytes] = []
        remaining = max_count + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (
            not os.path.samestat(before, after)
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise RuntimeError(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    snapshot = b"".join(chunks)
    if len(snapshot) != before.st_size:
        raise ValueError(f"{label} byte count changed while reading")
    if expected_sha256 is not None and hashlib.sha256(snapshot).hexdigest() != _sha256(
        expected_sha256, field=f"{label}.sha256"
    ):
        raise ValueError(f"{label} metadata mismatch: SHA-256 differs")
    if expected_rows is not None and snapshot.count(b"\n") != _strict_int(
        expected_rows, field=f"{label}.rows"
    ):
        raise ValueError(f"{label} metadata mismatch: row count differs")
    return snapshot


@dataclass(frozen=True, slots=True)
class VerifiedBundle:
    root: Path
    artifact_manifest_bytes: bytes
    _root_identity: tuple[int, int] = dataclass_field(repr=False, compare=False)
    _metadata: Mapping[str, tuple[str, int, int | None]] = dataclass_field(
        repr=False, compare=False
    )
    _reader_token: object = dataclass_field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._reader_token is not _VERIFIED_BUNDLE_TOKEN:
            raise TypeError("VerifiedBundle must come from read_closed_world_bundle")
        if not isinstance(self._metadata, MappingProxyType):
            raise TypeError("VerifiedBundle metadata must be deeply immutable")

    @property
    def artifact_manifest(self) -> dict[str, Any]:
        payload = _parse_json_bytes(
            self.artifact_manifest_bytes,
            label="immutable bundle artifact_manifest.json",
        )
        if not isinstance(payload, dict):
            raise TypeError("immutable artifact manifest is not a JSON object")
        return payload

    @property
    def artifacts(self) -> dict[str, dict[str, Any]]:
        return {
            name: {"sha256": row[0], "bytes": row[1], "rows": row[2]}
            for name, row in self._metadata.items()
        }

    def path(self, name: str) -> Path:
        if name not in self._metadata:
            raise KeyError(f"artifact is not declared: {name}")
        return self.root.joinpath(*PurePosixPath(name).parts)

    def read_bytes(self, name: str) -> bytes:
        path = self.path(name)
        sha256, byte_count, row_count = self._metadata[name]
        return _read_regular_snapshot(
            path,
            label=f"bundle artifact {name}",
            expected_bytes=byte_count,
            expected_sha256=sha256,
            expected_rows=row_count if name.endswith(".jsonl") else None,
            max_bytes=_MAX_SINGLE_BUNDLE_ARTIFACT_BYTES,
            expected_parent_identity=self._root_identity,
        )

    def read_json(self, name: str) -> Any:
        return _parse_json_bytes(self.read_bytes(name), label=f"bundle artifact {name}")

    def read_jsonl(self, name: str) -> tuple[dict[str, Any], ...]:
        return _parse_jsonl_bytes(
            self.read_bytes(name), label=f"bundle artifact {name}"
        )


def _list_flat_directory_bound(
    root: Path, *, expected_identity: tuple[int, int]
) -> tuple[set[str], set[str]]:
    """Enumerate one flat bundle through a handle bound to its verified root."""

    files: set[str] = set()
    directories: set[str] = set()
    if os.name == "posix":
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(root, flags)
        try:
            info = os.fstat(descriptor)
            if (int(info.st_dev), int(info.st_ino)) != expected_identity:
                raise ValueError("bundle root directory identity changed")
            for name in os.listdir(descriptor):
                child = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if stat.S_ISLNK(child.st_mode):
                    raise ValueError("bundle root contains a symlink")
                if stat.S_ISDIR(child.st_mode):
                    directories.add(name)
                elif stat.S_ISREG(child.st_mode):
                    files.add(name)
                else:
                    raise ValueError(
                        "bundle root contains an entry that is not a regular file "
                        "or directory"
                    )
        finally:
            os.close(descriptor)
        return files, directories

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create_file.restype = wintypes.HANDLE
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL

        class _FileTime(ctypes.Structure):
            _fields_ = (("low", wintypes.DWORD), ("high", wintypes.DWORD))

        class _ByHandleFileInformation(ctypes.Structure):
            _fields_ = (
                ("attributes", wintypes.DWORD),
                ("creation_time", _FileTime),
                ("access_time", _FileTime),
                ("write_time", _FileTime),
                ("volume_serial", wintypes.DWORD),
                ("size_high", wintypes.DWORD),
                ("size_low", wintypes.DWORD),
                ("link_count", wintypes.DWORD),
                ("file_index_high", wintypes.DWORD),
                ("file_index_low", wintypes.DWORD),
            )

        class _FileIdBothDirectoryInfo(ctypes.Structure):
            _fields_ = (
                ("next_entry_offset", wintypes.DWORD),
                ("file_index", wintypes.DWORD),
                ("creation_time", ctypes.c_longlong),
                ("last_access_time", ctypes.c_longlong),
                ("last_write_time", ctypes.c_longlong),
                ("change_time", ctypes.c_longlong),
                ("end_of_file", ctypes.c_longlong),
                ("allocation_size", ctypes.c_longlong),
                ("file_attributes", wintypes.DWORD),
                ("file_name_length", wintypes.DWORD),
                ("ea_size", wintypes.DWORD),
                ("short_name_length", ctypes.c_ubyte),
                ("short_name", wintypes.WCHAR * 12),
                ("file_id", ctypes.c_longlong),
                ("file_name", wintypes.WCHAR * 1),
            )

        get_basic = kernel32.GetFileInformationByHandle
        get_basic.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_ByHandleFileInformation),
        ]
        get_basic.restype = wintypes.BOOL
        get_extended = kernel32.GetFileInformationByHandleEx
        get_extended.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        get_extended.restype = wintypes.BOOL
        handle = _create_windows_component_handle(
            create_file,
            root,
            desired_access=0x00000001 | 0x00000080,
            is_leaf=False,
        )
        if handle == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_last_error(), "cannot open verified bundle root")
        try:
            basic = _ByHandleFileInformation()
            if not get_basic(handle, ctypes.byref(basic)):
                raise OSError(
                    ctypes.get_last_error(), "cannot inspect verified bundle root"
                )
            identity = (
                int(basic.volume_serial),
                (int(basic.file_index_high) << 32) | int(basic.file_index_low),
            )
            if identity != expected_identity:
                raise ValueError("bundle root directory identity changed")
            if basic.attributes & _REPARSE_POINT_ATTRIBUTE:
                raise ValueError("bundle root became a reparse point")
            buffer = ctypes.create_string_buffer(256 * 1024)
            while True:
                if not get_extended(handle, 10, buffer, len(buffer)):
                    error = ctypes.get_last_error()
                    if error == 18:
                        break
                    raise OSError(error, "cannot enumerate verified bundle root")
                offset = 0
                while True:
                    entry = _FileIdBothDirectoryInfo.from_buffer(buffer, offset)
                    name = ctypes.wstring_at(
                        ctypes.addressof(buffer)
                        + offset
                        + _FileIdBothDirectoryInfo.file_name.offset,
                        entry.file_name_length // ctypes.sizeof(wintypes.WCHAR),
                    )
                    if name not in {".", ".."}:
                        if entry.file_attributes & _REPARSE_POINT_ATTRIBUTE:
                            raise ValueError("bundle root contains a reparse point")
                        if entry.file_attributes & 0x00000010:
                            directories.add(name)
                        else:
                            files.add(name)
                    if entry.next_entry_offset == 0:
                        break
                    offset += entry.next_entry_offset
        finally:
            close_handle(handle)
        return files, directories

    before = root.stat(follow_symlinks=False)
    for entry in os.scandir(root):
        if entry.is_symlink():
            raise ValueError("bundle root contains a symlink")
        if entry.is_dir(follow_symlinks=False):
            directories.add(entry.name)
        elif entry.is_file(follow_symlinks=False):
            files.add(entry.name)
        else:
            raise ValueError(
                "bundle root contains an entry that is not a regular file or directory"
            )
    after = root.stat(follow_symlinks=False)
    if not os.path.samestat(before, after):
        raise RuntimeError("bundle root changed while enumerating")
    return files, directories


def read_closed_world_bundle(directory: str | Path) -> VerifiedBundle:
    """Verify a complete bundle, including absence of symlinks and extra files."""

    requested_root = Path(directory).expanduser()
    lexical_root = Path(os.path.abspath(os.fspath(requested_root)))
    manifest_path = lexical_root / "artifact_manifest.json"
    opened_manifest_paths: list[Path] = []
    opened_root_identities: list[tuple[int, int]] = []
    manifest_bytes = _read_regular_snapshot(
        manifest_path,
        label="bundle artifact_manifest.json",
        max_bytes=_MAX_ARTIFACT_MANIFEST_BYTES,
        canonical_path_out=opened_manifest_paths,
        parent_identity_out=opened_root_identities,
    )
    if len(opened_manifest_paths) != 1 or len(opened_root_identities) != 1:
        raise RuntimeError("bundle manifest open did not return one bound path")
    root = opened_manifest_paths[0].parent
    root_identity = opened_root_identities[0]
    manifest = _parse_json_bytes(manifest_bytes, label="bundle artifact_manifest.json")
    if not isinstance(manifest, dict):
        raise TypeError("artifact manifest must be a JSON object")
    if manifest.get("schema") == FILE_BUNDLE_SCHEMA_VERSION:
        _exact_fields(
            manifest,
            {
                "schema",
                "bundle_status",
                "artifact_kind",
                "semantic_manifest",
                "bundle_id",
                "artifacts",
            },
            label="M04a file-bundle manifest",
        )
        if manifest["bundle_status"] != "complete":
            raise ValueError("bundle_status must be complete")
        _nonempty_string(manifest["artifact_kind"], field="artifact_kind")
        _safe_artifact_name(manifest["semantic_manifest"])
        _sha256(manifest["bundle_id"], field="bundle_id")
    else:
        _exact_fields(
            manifest,
            {"schema_version", "bundle_status", "run_id", "artifacts"},
            label="legacy evidence artifact manifest",
        )
        if manifest["schema_version"] != 1 or manifest["bundle_status"] != "complete":
            raise ValueError("unsupported or incomplete legacy evidence bundle")
        _nonempty_string(manifest["run_id"], field="run_id")

    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("artifact manifest must declare at least one artifact")
    if len(artifacts) > _MAX_BUNDLE_ARTIFACT_COUNT:
        raise ValueError("artifact manifest exceeds the artifact-count limit")
    normalized: dict[str, dict[str, Any]] = {}
    total_artifact_bytes = 0
    expected_files = {"artifact_manifest.json"}
    for raw_name, raw_metadata in artifacts.items():
        name = _safe_artifact_name(raw_name)
        if len(PurePosixPath(name).parts) != 1:
            raise ValueError("closed-world bundles require a flat artifact namespace")
        metadata = _exact_fields(
            raw_metadata, {"sha256", "bytes", "rows"}, label=f"metadata for {name}"
        )
        _sha256(metadata["sha256"], field=f"{name}.sha256")
        artifact_bytes = _strict_int(metadata["bytes"], field=f"{name}.bytes")
        if artifact_bytes > _MAX_SINGLE_BUNDLE_ARTIFACT_BYTES:
            raise ValueError(
                f"bundle artifact exceeds the single-file byte limit: {name}"
            )
        total_artifact_bytes += artifact_bytes
        if total_artifact_bytes > _MAX_TOTAL_BUNDLE_ARTIFACT_BYTES:
            raise ValueError("bundle artifacts exceed the total byte limit")
        if name.endswith(".jsonl"):
            _strict_int(metadata["rows"], field=f"{name}.rows")
        elif metadata["rows"] is not None:
            raise ValueError(f"{name}.rows must be null for a non-JSONL artifact")
        normalized[name] = dict(metadata)
        expected_files.add(name)

    actual_files, actual_directories = _list_flat_directory_bound(
        root, expected_identity=root_identity
    )
    if actual_files != expected_files:
        raise ValueError(
            "closed-world artifact set mismatch: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )
    if actual_directories:
        raise ValueError(
            f"closed-world directory set mismatch: extra={sorted(actual_directories)}"
        )
    bundle = VerifiedBundle(
        root=root,
        artifact_manifest_bytes=manifest_bytes,
        _root_identity=root_identity,
        _metadata=MappingProxyType(
            {
                name: (
                    str(metadata["sha256"]),
                    int(metadata["bytes"]),
                    None if metadata["rows"] is None else int(metadata["rows"]),
                )
                for name, metadata in normalized.items()
            }
        ),
        _reader_token=_VERIFIED_BUNDLE_TOKEN,
    )
    for name in normalized:
        bundle.read_bytes(name)
    return bundle


def _artifact_manifest_sha256(bundle: VerifiedBundle) -> str:
    return hashlib.sha256(bundle.artifact_manifest_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class ShapeSelection:
    pre_dedup_count: int
    post_dedup_count: int
    truncation_count: int
    accepted_shapes: tuple[dict[str, Any], ...]
    status: str


def _normalize_shape_proposal(
    payload: object, *, expected_test_count: int
) -> dict[str, Any]:
    proposal = _exact_fields(
        payload,
        {
            "shape_proposer_semantics_version",
            "proposal_id",
            "basis",
            "row_factor",
            "column_factor",
            "query_shapes",
        },
        label="blind output-shape proposal",
    )
    version = _contract_value("OUTPUT_SHAPE_SEMANTICS_VERSION")
    if proposal["shape_proposer_semantics_version"] != version:
        raise ValueError("shape proposal uses an unsupported semantics version")
    if proposal["basis"] != "input":
        raise ValueError("M04a v0.1 accepts only input-basis shape proposals")
    row_factor = _strict_int(proposal["row_factor"], field="row_factor", minimum=1)
    column_factor = _strict_int(
        proposal["column_factor"], field="column_factor", minimum=1
    )
    max_side = _contract_value("MAX_GRID_SIDE")
    if row_factor > max_side or column_factor > max_side:
        raise ValueError("shape proposal factor exceeds ARC bounds")
    raw_shapes = proposal["query_shapes"]
    if not isinstance(raw_shapes, list) or len(raw_shapes) != expected_test_count:
        raise ValueError("shape proposal must contain one query shape per test pair")
    query_shapes: list[list[int]] = []
    for index, raw_shape in enumerate(raw_shapes):
        if not isinstance(raw_shape, list) or len(raw_shape) != 2:
            raise TypeError(f"query_shapes[{index}] must be [height, width]")
        height = _strict_int(raw_shape[0], field=f"query_shapes[{index}][0]", minimum=1)
        width = _strict_int(raw_shape[1], field=f"query_shapes[{index}][1]", minimum=1)
        if height > max_side or width > max_side:
            raise ValueError(f"query_shapes[{index}] exceeds ARC bounds")
        query_shapes.append([height, width])
    identity_payload = {
        "shape_proposer_semantics_version": version,
        "basis": "input",
        "row_factor": row_factor,
        "column_factor": column_factor,
        "query_shapes": query_shapes,
    }
    proposal_id = _sha256(proposal["proposal_id"], field="proposal_id")
    if proposal_id != _canonical_sha256(identity_payload):
        raise ValueError("proposal_id does not match canonical blind shape content")
    return {**identity_payload, "proposal_id": proposal_id}


def select_blind_shapes(
    proposals: object, *, test_index: int, expected_test_count: int
) -> ShapeSelection:
    """Project, order-dedupe, and cap the M03a proposals for one test pair."""

    test_index = _strict_int(test_index, field="test_index")
    expected_test_count = _strict_int(
        expected_test_count, field="expected_test_count", minimum=1
    )
    if test_index >= expected_test_count:
        raise ValueError("test_index is outside the declared test-pair population")
    if not isinstance(proposals, list):
        raise TypeError("shape_proposals must be a list")
    normalized = [
        _normalize_shape_proposal(item, expected_test_count=expected_test_count)
        for item in proposals
    ]
    distinct: list[dict[str, Any]] = []
    seen_shapes: set[tuple[int, int]] = set()
    for proposal in normalized:
        height, width = proposal["query_shapes"][test_index]
        shape = (height, width)
        if shape in seen_shapes:
            continue
        seen_shapes.add(shape)
        distinct.append(
            {
                "shape_order": len(distinct),
                "shape_proposal_id": proposal["proposal_id"],
                "proposed_height": height,
                "proposed_width": width,
            }
        )
    maximum = _contract_value("MAX_ACCEPTED_SHAPES")
    accepted = tuple(distinct[:maximum])
    return ShapeSelection(
        pre_dedup_count=len(normalized),
        post_dedup_count=len(distinct),
        truncation_count=max(0, len(distinct) - maximum),
        accepted_shapes=accepted,
        status=SHAPE_READY if accepted else NO_SHAPE_PROPOSAL,
    )


_BLIND_INPUT_ROW_FIELDS = {
    "schema",
    "row_id",
    "blind_task_id",
    "blind_content_sha256",
    "test_index",
    "test_input",
    "test_input_sha256",
    "dsl_candidate_count",
    "candidate_count_zero_subgroup",
    "shape_proposals_pre_dedup",
    "shape_proposals_post_dedup",
    "accepted_shape_count",
    "shape_proposal_truncation_count",
    "accepted_shapes",
    "status",
}

_BLIND_SIDECAR_MANIFEST_FIELDS = {
    "schema",
    "sidecar_id",
    "case_set_id",
    "blind_artifact_manifest_sha256",
    "blind_tasks_sha256",
    "dsl_pool_artifact_manifest_sha256",
    "dsl_pool_content_id",
    "dsl_pool_spec_id",
    "dsl_semantics_version",
    "shape_proposer_semantics_version",
    "task_count",
    "test_pair_count",
    "candidate_count_zero_task_count",
    "candidate_count_zero_test_pair_count",
    "shape_supported_test_pair_count",
    "no_shape_test_pair_count",
    "rows_sha256",
}


def _blind_row_semantic_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "row_id"}


def validate_blind_input_row(row: object) -> dict[str, Any]:
    payload = _exact_fields(row, _BLIND_INPUT_ROW_FIELDS, label="M04a blind-input row")
    assert_oracle_free_payload(payload, location="blind_input_row")
    if payload["schema"] != BLIND_INPUT_ROW_SCHEMA_VERSION:
        raise ValueError("unsupported M04a blind-input row schema")
    _nonempty_string(payload["blind_task_id"], field="blind_task_id")
    blind_sha = _sha256(payload["blind_content_sha256"], field="blind_content_sha256")
    if payload["blind_task_id"] != f"blind_{blind_sha}":
        raise ValueError("blind_task_id does not match blind_content_sha256")
    _strict_int(payload["test_index"], field="test_index")
    test_input = as_grid(payload["test_input"])
    expected_input_sha = _canonical_sha256(grid_to_lists(test_input))
    if payload["test_input_sha256"] != expected_input_sha:
        raise ValueError("test_input_sha256 does not match test_input")
    candidate_count = _strict_int(
        payload["dsl_candidate_count"], field="dsl_candidate_count"
    )
    if type(payload["candidate_count_zero_subgroup"]) is not bool:
        raise TypeError("candidate_count_zero_subgroup must be boolean")
    if payload["candidate_count_zero_subgroup"] != (candidate_count == 0):
        raise ValueError(
            "zero-candidate subgroup must derive only from candidate count == 0"
        )
    pre = _strict_int(
        payload["shape_proposals_pre_dedup"], field="shape_proposals_pre_dedup"
    )
    post = _strict_int(
        payload["shape_proposals_post_dedup"], field="shape_proposals_post_dedup"
    )
    accepted_count = _strict_int(
        payload["accepted_shape_count"], field="accepted_shape_count"
    )
    truncated = _strict_int(
        payload["shape_proposal_truncation_count"],
        field="shape_proposal_truncation_count",
    )
    if post > pre or accepted_count != min(
        post, _contract_value("MAX_ACCEPTED_SHAPES")
    ):
        raise ValueError("shape proposal counts do not close")
    if truncated != post - accepted_count:
        raise ValueError("shape proposal truncation count does not close")
    shapes = payload["accepted_shapes"]
    if not isinstance(shapes, list) or len(shapes) != accepted_count:
        raise ValueError("accepted_shapes length does not match accepted_shape_count")
    seen: set[tuple[int, int]] = set()
    for order, shape in enumerate(shapes):
        entry = _exact_fields(
            shape,
            {"shape_order", "shape_proposal_id", "proposed_height", "proposed_width"},
            label=f"accepted_shapes[{order}]",
        )
        if entry["shape_order"] != order:
            raise ValueError("accepted shape_order must be consecutive")
        _sha256(entry["shape_proposal_id"], field="shape_proposal_id")
        height = _strict_int(
            entry["proposed_height"], field="proposed_height", minimum=1
        )
        width = _strict_int(entry["proposed_width"], field="proposed_width", minimum=1)
        if height > 30 or width > 30 or (height, width) in seen:
            raise ValueError("accepted shapes must be distinct and within ARC bounds")
        seen.add((height, width))
    expected_status = SHAPE_READY if accepted_count else NO_SHAPE_PROPOSAL
    if payload["status"] != expected_status:
        raise ValueError("blind-input status does not match accepted shapes")
    expected_id = _canonical_sha256(_blind_row_semantic_payload(payload))
    if payload["row_id"] != expected_id:
        raise ValueError("blind-input row_id does not match semantic content")
    return payload


@dataclass(frozen=True, slots=True)
class BlindInputShapeSidecar:
    manifest: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    fixture_mode: bool = False


def _sidecar_semantic_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "sidecar_id"}


def validate_blind_input_shape_sidecar(
    manifest: object, rows: object, *, fixture_mode: bool = False
) -> BlindInputShapeSidecar:
    if type(fixture_mode) is not bool:
        raise TypeError("fixture_mode must be boolean")
    payload = _exact_fields(
        manifest, _BLIND_SIDECAR_MANIFEST_FIELDS, label="blind shape sidecar manifest"
    )
    assert_oracle_free_payload(payload, location="blind_shape_manifest")
    if payload["schema"] != BLIND_SHAPE_SIDECAR_SCHEMA_VERSION:
        raise ValueError("unsupported blind shape sidecar schema")
    for field in (
        "case_set_id",
        "blind_artifact_manifest_sha256",
        "blind_tasks_sha256",
        "dsl_pool_artifact_manifest_sha256",
        "dsl_pool_content_id",
        "dsl_pool_spec_id",
        "rows_sha256",
        "sidecar_id",
    ):
        _sha256(payload[field], field=field)
    if payload["dsl_semantics_version"] != FIXED_DSL_SEMANTICS_VERSION:
        raise ValueError("blind shape sidecar must bind DSL v0.6")
    if payload["shape_proposer_semantics_version"] != _contract_value(
        "OUTPUT_SHAPE_SEMANTICS_VERSION"
    ):
        raise ValueError("blind shape sidecar has the wrong shape semantics")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("blind shape sidecar rows must be an ordered sequence")
    normalized_rows = tuple(validate_blind_input_row(row) for row in rows)
    row_bytes = serialize_jsonl(normalized_rows)
    if hashlib.sha256(row_bytes).hexdigest() != payload["rows_sha256"]:
        raise ValueError("blind shape sidecar rows_sha256 mismatch")
    ordered_pairs = [
        (row["blind_task_id"], row["test_index"]) for row in normalized_rows
    ]
    if len(set(ordered_pairs)) != len(ordered_pairs):
        raise ValueError("blind shape sidecar contains duplicate task/test pairs")
    if ordered_pairs != sorted(ordered_pairs, key=lambda item: (item[0], item[1])):
        raise ValueError(
            "blind shape sidecar rows are not in canonical task/test order"
        )
    task_ids = {task_id for task_id, _ in ordered_pairs}
    counts = {
        "task_count": len(task_ids),
        "test_pair_count": len(normalized_rows),
        "candidate_count_zero_task_count": len(
            {
                row["blind_task_id"]
                for row in normalized_rows
                if row["candidate_count_zero_subgroup"]
            }
        ),
        "candidate_count_zero_test_pair_count": sum(
            row["candidate_count_zero_subgroup"] for row in normalized_rows
        ),
        "shape_supported_test_pair_count": sum(
            row["accepted_shape_count"] > 0 for row in normalized_rows
        ),
        "no_shape_test_pair_count": sum(
            row["accepted_shape_count"] == 0 for row in normalized_rows
        ),
    }
    for field, expected in counts.items():
        if payload[field] != expected:
            raise ValueError(f"{field} does not close from blind-input rows")
    if payload["sidecar_id"] != _canonical_sha256(_sidecar_semantic_payload(payload)):
        raise ValueError("sidecar_id does not match semantic manifest content")
    if not fixture_mode:
        fixed_fields = {
            "case_set_id": FIXED_BLIND_CASE_SET_ID,
            "blind_artifact_manifest_sha256": FIXED_BLIND_ARTIFACT_MANIFEST_SHA256,
            "blind_tasks_sha256": FIXED_BLIND_TASKS_SHA256,
            "dsl_pool_artifact_manifest_sha256": FIXED_DSL_POOL_ARTIFACT_MANIFEST_SHA256,
            "dsl_pool_content_id": FIXED_DSL_POOL_CONTENT_ID,
            "dsl_pool_spec_id": FIXED_DSL_POOL_SPEC_ID,
            "task_count": FIXED_BLIND_TASK_COUNT,
            "test_pair_count": FIXED_BLIND_TEST_PAIR_COUNT,
            "candidate_count_zero_task_count": FIXED_ZERO_CANDIDATE_TASK_COUNT,
            "candidate_count_zero_test_pair_count": 13,
            "shape_supported_test_pair_count": 16,
            "no_shape_test_pair_count": 5,
            "rows_sha256": FIXED_BLIND_SHAPE_ROWS_SHA256,
            "sidecar_id": FIXED_BLIND_SHAPE_SIDECAR_ID,
        }
        for field, expected in fixed_fields.items():
            if payload[field] != expected:
                raise ValueError(
                    f"production blind shape sidecar {field} is not the frozen fixed-20 value"
                )
    return BlindInputShapeSidecar(
        manifest=payload, rows=normalized_rows, fixture_mode=fixture_mode
    )


def build_m04a_blind_input_sidecar(
    blind_bundle_dir: str | Path, dsl_pool_dir: str | Path
) -> BlindInputShapeSidecar:
    """Build the only primary M04a input population from the two frozen parents."""

    blind_bundle = read_closed_world_bundle(blind_bundle_dir)
    dsl_bundle = read_closed_world_bundle(dsl_pool_dir)
    if _artifact_manifest_sha256(blind_bundle) != FIXED_BLIND_ARTIFACT_MANIFEST_SHA256:
        raise ValueError("blind bundle is not the frozen fixed-20 artifact")
    if _artifact_manifest_sha256(dsl_bundle) != FIXED_DSL_POOL_ARTIFACT_MANIFEST_SHA256:
        raise ValueError("DSL bundle is not the frozen v0.6 pool artifact")
    required_blind = {"blind_manifest.json", "blind_tasks.jsonl", "source_snapshot.zip"}
    if set(blind_bundle.artifacts) != required_blind:
        raise ValueError("blind bundle artifact set is not the frozen closed world")
    required_dsl = {
        "candidates.jsonl",
        "config.json",
        "exact_programs.jsonl",
        "panel_parses.jsonl",
        "parses.jsonl",
        "pool_manifest.json",
        "pool_summary.json",
        "relation_parses.jsonl",
        "search.jsonl",
        "source_snapshot.zip",
        "tasks.jsonl",
    }
    if set(dsl_bundle.artifacts) != required_dsl:
        raise ValueError("DSL pool artifact set is not the frozen closed world")

    blind_manifest = blind_bundle.read_json("blind_manifest.json")
    pool_manifest = dsl_bundle.read_json("pool_manifest.json")
    pool_config = dsl_bundle.read_json("config.json")
    pool_summary = dsl_bundle.read_json("pool_summary.json")
    task_rows = blind_bundle.read_jsonl("blind_tasks.jsonl")
    dsl_task_rows = dsl_bundle.read_jsonl("tasks.jsonl")
    search_rows = dsl_bundle.read_jsonl("search.jsonl")
    # The already-pinned E01a builder manifest records the privileged sibling's
    # command-line destination as provenance.  Its exact byte hash is trusted above;
    # no such path or field is admitted into the blind task/search material below.
    for label, payload in (
        ("pool_manifest", pool_manifest),
        ("pool_config", pool_config),
        ("pool_summary", pool_summary),
        ("blind_tasks", task_rows),
        ("dsl_tasks", dsl_task_rows),
        ("search", search_rows),
    ):
        assert_oracle_free_payload(payload, location=label)

    if (
        not isinstance(blind_manifest, dict)
        or blind_manifest.get("schema") != "afts.e01a-blind-manifest/v1"
    ):
        raise ValueError("unsupported fixed blind manifest")
    if blind_manifest.get("case_set_id") != FIXED_BLIND_CASE_SET_ID:
        raise ValueError("blind case_set_id does not match the frozen population")
    if blind_manifest.get("blind_tasks_sha256") != FIXED_BLIND_TASKS_SHA256:
        raise ValueError("blind manifest does not bind the frozen blind tasks")
    if blind_manifest.get("task_count") != FIXED_BLIND_TASK_COUNT:
        raise ValueError("blind manifest must contain all 20 fixed tasks")
    if (
        not isinstance(pool_manifest, dict)
        or pool_manifest.get("schema") != "afts.e01a-symbolic-pool/v6"
    ):
        raise ValueError("M04a requires the frozen DSL pool schema v6")
    if pool_manifest.get("case_set_id") != FIXED_BLIND_CASE_SET_ID:
        raise ValueError("DSL pool and blind population case_set_id mismatch")
    if pool_manifest.get("pool_content_id") != FIXED_DSL_POOL_CONTENT_ID:
        raise ValueError("DSL pool content ID is not frozen v0.6")
    if pool_manifest.get("pool_spec_id") != FIXED_DSL_POOL_SPEC_ID:
        raise ValueError("DSL pool spec ID is not frozen v0.6")
    if pool_manifest.get("config") != pool_config:
        raise ValueError("pool_manifest config does not match config.json")
    if pool_config.get("dsl_semantics_version") != FIXED_DSL_SEMANTICS_VERSION:
        raise ValueError("M04a blind shapes must come from DSL v0.6")
    if pool_config.get("oracle_inputs_available_to_pool") is not False:
        raise ValueError("DSL pool was not produced behind the blind boundary")
    parent = pool_manifest.get("parent_blind_bundle")
    if parent != {
        "artifact_manifest_sha256": FIXED_BLIND_ARTIFACT_MANIFEST_SHA256,
        "blind_tasks_sha256": FIXED_BLIND_TASKS_SHA256,
    }:
        raise ValueError("DSL pool does not bind the exact frozen blind parent")
    if dsl_bundle.read_bytes("tasks.jsonl") != blind_bundle.read_bytes(
        "blind_tasks.jsonl"
    ):
        raise ValueError("DSL pool blind-task snapshot differs from blind parent")
    if not isinstance(pool_summary, dict) or any(
        pool_summary.get(name) != expected
        for name, expected in {
            "schema_version": 6,
            "case_set_id": FIXED_BLIND_CASE_SET_ID,
            "pool_content_id": FIXED_DSL_POOL_CONTENT_ID,
            "pool_spec_id": FIXED_DSL_POOL_SPEC_ID,
            "task_count": FIXED_BLIND_TASK_COUNT,
        }.items()
    ):
        raise ValueError("DSL pool summary does not close to the frozen pool")
    if len(task_rows) != FIXED_BLIND_TASK_COUNT or len(search_rows) != len(task_rows):
        raise ValueError("M04a requires every one of the 20 fixed task/search rows")
    tasks = tuple(BlindTask.from_json_dict(row) for row in task_rows)
    if dsl_task_rows != task_rows:
        raise ValueError("DSL task rows are not byte-order equivalent to blind tasks")
    if [row.get("task_id") for row in search_rows] != [task.task_id for task in tasks]:
        raise ValueError("DSL search rows do not preserve blind-task order")

    rows: list[dict[str, Any]] = []
    zero_candidate_tasks = 0
    for task, search in zip(tasks, search_rows):
        if not isinstance(search, dict):
            raise TypeError("DSL search row must be a JSON object")
        candidate_count = _strict_int(
            search.get("candidate_records"), field=f"{task.task_id}.candidate_records"
        )
        if candidate_count == 0:
            zero_candidate_tasks += 1
        proposals = search.get("shape_proposals")
        if not isinstance(proposals, list):
            raise TypeError("DSL search shape_proposals must be a list")
        for test_index, test_input in enumerate(task.test_inputs):
            selection = select_blind_shapes(
                proposals,
                test_index=test_index,
                expected_test_count=len(task.test_inputs),
            )
            row: dict[str, Any] = {
                "schema": BLIND_INPUT_ROW_SCHEMA_VERSION,
                "blind_task_id": task.task_id,
                "blind_content_sha256": task.blind_content_sha256,
                "test_index": test_index,
                "test_input": grid_to_lists(test_input),
                "test_input_sha256": _canonical_sha256(grid_to_lists(test_input)),
                "dsl_candidate_count": candidate_count,
                "candidate_count_zero_subgroup": candidate_count == 0,
                "shape_proposals_pre_dedup": selection.pre_dedup_count,
                "shape_proposals_post_dedup": selection.post_dedup_count,
                "accepted_shape_count": len(selection.accepted_shapes),
                "shape_proposal_truncation_count": selection.truncation_count,
                "accepted_shapes": list(selection.accepted_shapes),
                "status": selection.status,
            }
            row["row_id"] = _canonical_sha256(row)
            rows.append(validate_blind_input_row(row))
    if zero_candidate_tasks != FIXED_ZERO_CANDIDATE_TASK_COUNT:
        raise ValueError("zero-candidate task subgroup does not close to 12 tasks")
    if len(rows) != FIXED_BLIND_TEST_PAIR_COUNT:
        raise ValueError("M04a blind sidecar must contain all 21 fixed test pairs")
    rows_tuple = tuple(rows)
    rows_sha = hashlib.sha256(serialize_jsonl(rows_tuple)).hexdigest()
    manifest: dict[str, Any] = {
        "schema": BLIND_SHAPE_SIDECAR_SCHEMA_VERSION,
        "case_set_id": FIXED_BLIND_CASE_SET_ID,
        "blind_artifact_manifest_sha256": FIXED_BLIND_ARTIFACT_MANIFEST_SHA256,
        "blind_tasks_sha256": FIXED_BLIND_TASKS_SHA256,
        "dsl_pool_artifact_manifest_sha256": FIXED_DSL_POOL_ARTIFACT_MANIFEST_SHA256,
        "dsl_pool_content_id": FIXED_DSL_POOL_CONTENT_ID,
        "dsl_pool_spec_id": FIXED_DSL_POOL_SPEC_ID,
        "dsl_semantics_version": FIXED_DSL_SEMANTICS_VERSION,
        "shape_proposer_semantics_version": _contract_value(
            "OUTPUT_SHAPE_SEMANTICS_VERSION"
        ),
        "task_count": len(tasks),
        "test_pair_count": len(rows_tuple),
        "candidate_count_zero_task_count": zero_candidate_tasks,
        "candidate_count_zero_test_pair_count": sum(
            row["candidate_count_zero_subgroup"] for row in rows_tuple
        ),
        "shape_supported_test_pair_count": sum(
            row["accepted_shape_count"] > 0 for row in rows_tuple
        ),
        "no_shape_test_pair_count": sum(
            row["accepted_shape_count"] == 0 for row in rows_tuple
        ),
        "rows_sha256": rows_sha,
    }
    manifest["sidecar_id"] = _canonical_sha256(manifest)
    return validate_blind_input_shape_sidecar(manifest, rows_tuple)


def _write_bytes_new(path: Path, content: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
    return _file_metadata(path)


def _new_staging_directory(target: Path) -> Path:
    if target.exists():
        raise FileExistsError(f"refusing to overwrite artifact directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(exist_ok=False)
    return staging


def _outer_bundle_manifest(
    *,
    artifact_kind: str,
    semantic_manifest: str,
    bundle_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": FILE_BUNDLE_SCHEMA_VERSION,
        "bundle_status": "complete",
        "artifact_kind": artifact_kind,
        "semantic_manifest": semantic_manifest,
        "bundle_id": bundle_id,
        "artifacts": {
            name: dict(metadata) for name, metadata in sorted(artifacts.items())
        },
    }


def publish_blind_input_shape_sidecar(
    output_dir: str | Path,
    *,
    blind_bundle_dir: str | Path,
    dsl_pool_dir: str | Path,
) -> BlindInputShapeSidecar:
    """Exclusively and atomically publish the fixed blind shape sidecar."""

    sidecar = build_m04a_blind_input_sidecar(blind_bundle_dir, dsl_pool_dir)
    target = Path(output_dir).expanduser().resolve()
    staging = _new_staging_directory(target)
    try:
        artifacts = {
            "blind_input_shape_manifest.json": _write_bytes_new(
                staging / "blind_input_shape_manifest.json",
                serialize_json(sidecar.manifest),
            ),
            "blind_input_shape_rows.jsonl": _write_bytes_new(
                staging / "blind_input_shape_rows.jsonl",
                serialize_jsonl(sidecar.rows),
            ),
        }
        outer = _outer_bundle_manifest(
            artifact_kind="blind_input",
            semantic_manifest="blind_input_shape_manifest.json",
            bundle_id=sidecar.manifest["sidecar_id"],
            artifacts=artifacts,
        )
        _write_bytes_new(staging / "artifact_manifest.json", serialize_json(outer))
        read_blind_input_shape_sidecar(staging)
        staging.rename(target)
    except Exception as exc:
        raise RuntimeError(
            f"blind sidecar staging failed; preserved at {staging}: {exc}"
        ) from exc
    return sidecar


def read_blind_input_shape_sidecar(
    directory: str | Path, *, fixture_mode: bool = False
) -> BlindInputShapeSidecar:
    bundle = read_closed_world_bundle(directory)
    manifest = bundle.artifact_manifest
    if manifest.get("schema") != FILE_BUNDLE_SCHEMA_VERSION:
        raise ValueError("blind shape sidecar must use the M04a file-bundle schema")
    if manifest["artifact_kind"] != "blind_input" or manifest["semantic_manifest"] != (
        "blind_input_shape_manifest.json"
    ):
        raise ValueError("bundle is not an M04a blind-input sidecar")
    if set(bundle.artifacts) != {
        "blind_input_shape_manifest.json",
        "blind_input_shape_rows.jsonl",
    }:
        raise ValueError("blind-input sidecar contains an unexpected artifact")
    sidecar = validate_blind_input_shape_sidecar(
        bundle.read_json("blind_input_shape_manifest.json"),
        bundle.read_jsonl("blind_input_shape_rows.jsonl"),
        fixture_mode=fixture_mode,
    )
    if manifest["bundle_id"] != sidecar.manifest["sidecar_id"]:
        raise ValueError("outer bundle_id does not match blind sidecar_id")
    return sidecar


_PREFLIGHT_TRAINING_FIELDS = {
    "updates",
    "microbatches_per_update",
    "total_microbatches",
    "masked_tokens_per_microbatch",
    "masked_token_predictions",
    "encoder_forward_calls",
    "decoder_forward_calls",
    "backward_calls",
    "ledger_row_count",
    "ledger_rows_sha256",
    "total_update_wall_ns",
    "total_update_cuda_event_ns",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "learning_rate_hex",
    "mean_masked_cell_ce_hex",
    "gradient_norm_before_clip_hex",
}

_PREFLIGHT_INFERENCE_FIELDS = {
    "schema",
    "inference_id",
    "lane_count",
    "denoising_steps",
    "encoder_batch_calls",
    "decoder_batch_calls",
    "sample_equivalent_forward_calls",
    "masked_token_predictions",
    "padded_encoder_batch_shape",
    "unpadded_memory_length",
    "cache_dtype",
    "decoder_batch_sizes",
    "mask_count_trace",
    "lane_trace_sha256",
    "final_output_keys",
    "unique_outputs",
    "h2d_ns",
    "d2h_ns",
    "encoder_gpu_ns",
    "encoder_wall_time_ns",
    "decoder_gpu_ns",
    "decoder_wall_time_ns",
    "cpu_sampling_ns",
    "hashing_ns",
    "inference_wall_time_ns",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
}

_PREFLIGHT_REPORT_FIELDS = {
    "schema",
    "report_id",
    "status",
    "fixture",
    "runtime",
    "model",
    "training",
    "inference",
    "validation_manifest_commitment",
    "overhead_cost_probe_id",
    "lock_handshake_artifact_sha256",
    "preflight_started_perf_counter_ns",
    "fresh_reconstruction_probe",
    "checkpoint_cost_probe",
    "final_selection_probe",
    "lock_handshake_probe",
    "budget_projection",
    "dataset_file_reads",
    "training_checkpoint_writes",
    "diagnostic_checkpoint_writes",
    "fallback_used",
    "rng_state_restored",
    "ledger_rows_sha256",
}


def _mask_count_trace_900() -> tuple[int, ...]:
    module = _contract()
    if module is not None and hasattr(module, "mask_count_trace"):
        return tuple(module.mask_count_trace(PREFLIGHT_TARGET_CELL_COUNT))
    return tuple(
        [PREFLIGHT_TARGET_CELL_COUNT]
        + [
            (
                0
                if step == PREFLIGHT_DENOISING_STEPS
                else (
                    (PREFLIGHT_TARGET_CELL_COUNT + 1) // 2
                    if step == 8
                    else math.ceil(
                        PREFLIGHT_TARGET_CELL_COUNT * math.cos(math.pi * step / 24)
                    )
                )
            )
            for step in range(1, PREFLIGHT_DENOISING_STEPS + 1)
        ]
    )


def _canonical_nonnegative_float_hex(value: object, *, field: str) -> float:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a canonical float.hex string")
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a canonical float.hex string") from exc
    if (
        not math.isfinite(parsed)
        or parsed < 0.0
        or (parsed == 0.0 and math.copysign(1.0, parsed) < 0.0)
        or value != parsed.hex()
    ):
        raise ValueError(f"{field} must be canonical, finite, and non-negative")
    return parsed


def _fp32(value: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError as exc:
        raise ValueError("FP32 validation metric overflowed") from exc


def _canonical_nonnegative_fp32_hex(value: object, *, field: str) -> float:
    parsed = _canonical_nonnegative_float_hex(value, field=field)
    if _fp32(parsed) != parsed:
        raise ValueError(f"{field} must encode an exact finite non-negative FP32 value")
    return parsed


def sequential_fp32_mean(values: Sequence[float]) -> float:
    """Mean in literal order, rounding every input/add/divide to IEEE FP32."""

    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or not values
    ):
        raise TypeError("values must be a non-empty ordered sequence")
    accumulator = _fp32(0.0)
    for index, value in enumerate(values):
        normalized = _finite_number(value, field=f"values[{index}]", minimum=0.0)
        accumulator = _fp32(accumulator + _fp32(normalized))
    return _fp32(accumulator / len(values))


def _preflight_fixture_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": PREFLIGHT_FIXTURE_SCHEMA_VERSION,
        "grid_side": PREFLIGHT_GRID_SIDE,
        "target_cell_count": PREFLIGHT_TARGET_CELL_COUNT,
        "demonstration_count": 10,
        "encoder_grid_count": 21,
        "encoder_grid_token_count": 901,
        "expected_memory_length": 18_921,
        "demonstrations": [
            {
                "pair_index": index,
                "input_constant_color": index,
                "output_constant_color": (index + 1) % 10,
                "height": PREFLIGHT_GRID_SIDE,
                "width": PREFLIGHT_GRID_SIDE,
            }
            for index in range(10)
        ],
        "query_constant_color": 0,
        "target_constant_color": 1,
        "training_mask_policy": "all_900_cells_every_microbatch",
        "training_updates": PREFLIGHT_UPDATES,
        "microbatches_per_update": PREFLIGHT_MICROBATCHES_PER_UPDATE,
        "inference_lanes": PREFLIGHT_LANES,
        "inference_steps": PREFLIGHT_DENOISING_STEPS,
    }
    payload["fixture_id"] = _canonical_sha256(payload)
    return payload


def _validate_preflight_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("preflight rows must be an ordered sequence")
    normalized = tuple(validate_training_cost_ledger_row(row) for row in rows)
    if len(normalized) != PREFLIGHT_UPDATES:
        raise ValueError("preflight must bind exactly 100 update rows")
    expected_counts = {
        "optimizer_updates": 1,
        "microbatches": PREFLIGHT_MICROBATCHES_PER_UPDATE,
        "arc2_episodes": 0,
        "rearc_episodes": 0,
        "encoder_forward_calls": PREFLIGHT_MICROBATCHES_PER_UPDATE,
        "decoder_forward_calls": PREFLIGHT_MICROBATCHES_PER_UPDATE,
        "backward_calls": PREFLIGHT_MICROBATCHES_PER_UPDATE,
        "validation_episode_calls": 0,
        "validation_encoder_forward_calls": 0,
        "validation_decoder_forward_calls": 0,
        "checkpoint_writes": 0,
        "masked_token_predictions": PREFLIGHT_MASKED_TOKEN_PREDICTIONS_PER_UPDATE,
    }
    for index, row in enumerate(normalized):
        if (
            row["phase"] != "preflight_update"
            or row["event_index"] != index
            or row["optimizer_step"] != index + 1
            or any(
                row[field] != expected for field, expected in expected_counts.items()
            )
            or row["checkpoint_io_bytes"] != 0
            or row["checkpoint_io_ns"] != 0
        ):
            raise ValueError(
                "preflight update ledger does not close exactly 100x16x900"
            )
    return normalized


def _validate_preflight_inference(summary: object) -> dict[str, Any]:
    payload = _exact_fields(
        summary,
        _PREFLIGHT_INFERENCE_FIELDS,
        label="M04a preflight inference summary",
    )
    mask_trace = _mask_count_trace_900()
    expected = {
        "schema": PREFLIGHT_INFERENCE_SCHEMA_VERSION,
        "lane_count": PREFLIGHT_LANES,
        "denoising_steps": PREFLIGHT_DENOISING_STEPS,
        "encoder_batch_calls": 1,
        "decoder_batch_calls": PREFLIGHT_DENOISING_STEPS,
        "sample_equivalent_forward_calls": (
            PREFLIGHT_LANES * PREFLIGHT_DENOISING_STEPS
        ),
        "masked_token_predictions": PREFLIGHT_LANES * sum(mask_trace[:-1]),
        "padded_encoder_batch_shape": [21, 901],
        "unpadded_memory_length": 18_921,
        "cache_dtype": "bfloat16",
        "decoder_batch_sizes": [PREFLIGHT_LANES] * PREFLIGHT_DENOISING_STEPS,
        "mask_count_trace": list(mask_trace),
    }
    if any(payload[field] != value for field, value in expected.items()):
        raise ValueError(
            "preflight inference does not close exactly eight lanes x 12 steps"
        )
    trace_ids = payload["lane_trace_sha256"]
    output_keys = payload["final_output_keys"]
    if (
        not isinstance(trace_ids, list)
        or len(trace_ids) != PREFLIGHT_LANES
        or any(
            not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None
            for value in trace_ids
        )
        or len(set(trace_ids)) != PREFLIGHT_LANES
    ):
        raise ValueError("preflight must bind eight distinct lane-trace SHA-256 values")
    if (
        not isinstance(output_keys, list)
        or len(output_keys) != PREFLIGHT_LANES
        or any(
            not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None
            for value in output_keys
        )
    ):
        raise ValueError("preflight must bind eight final grid keys")
    if payload["unique_outputs"] != len(set(output_keys)):
        raise ValueError("preflight unique-output count does not close")
    measurement_fields = {
        "h2d_ns",
        "d2h_ns",
        "encoder_gpu_ns",
        "encoder_wall_time_ns",
        "decoder_gpu_ns",
        "decoder_wall_time_ns",
        "cpu_sampling_ns",
        "hashing_ns",
        "inference_wall_time_ns",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
    }
    for field in measurement_fields:
        _strict_int(payload[field], field=f"inference.{field}")
    if payload["inference_wall_time_ns"] <= 0:
        raise ValueError("preflight inference endpoint must have positive wall time")
    if payload["encoder_gpu_ns"] > payload["encoder_wall_time_ns"]:
        raise ValueError("preflight encoder CUDA time exceeds wall time")
    if payload["decoder_gpu_ns"] > payload["decoder_wall_time_ns"]:
        raise ValueError("preflight decoder CUDA time exceeds wall time")
    sequential = sum(
        payload[field]
        for field in (
            "h2d_ns",
            "d2h_ns",
            "encoder_wall_time_ns",
            "decoder_wall_time_ns",
            "cpu_sampling_ns",
            "hashing_ns",
        )
    )
    if sequential > payload["inference_wall_time_ns"]:
        raise ValueError("preflight inference endpoint omits a sequential component")
    if payload["cuda_peak_allocated_bytes"] > payload["cuda_peak_reserved_bytes"]:
        raise ValueError("preflight allocated CUDA peak exceeds reserved peak")
    expected_id = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "inference_id"}
    )
    if payload["inference_id"] != expected_id:
        raise ValueError("preflight inference_id mismatch")
    return dict(payload)


def _validate_preflight_validation_commitment(
    commitment: object,
    *,
    expected_outer_manifest_sha256: str,
    expected_jsonl_sha256: str,
    expected_summary_id: str,
    expected_row_count: int,
) -> dict[str, Any]:
    payload = _exact_fields(
        commitment,
        {
            "schema",
            "commitment_id",
            "outer_artifact_manifest_sha256",
            "jsonl_sha256",
            "summary_id",
            "row_count",
        },
        label="M04a preflight validation commitment",
    )
    if payload["schema"] != PREFLIGHT_VALIDATION_COMMITMENT_SCHEMA_VERSION:
        raise ValueError("unsupported preflight validation commitment schema")
    _sha256(payload["commitment_id"], field="validation_commitment.commitment_id")
    _sha256(
        payload["outer_artifact_manifest_sha256"],
        field="validation_commitment.outer_artifact_manifest_sha256",
    )
    _sha256(payload["jsonl_sha256"], field="validation_commitment.jsonl_sha256")
    _sha256(payload["summary_id"], field="validation_commitment.summary_id")
    _strict_int(
        payload["row_count"], field="validation_commitment.row_count", minimum=1
    )
    outer = _sha256(
        expected_outer_manifest_sha256,
        field="expected_validation_outer_manifest_sha256",
    )
    jsonl = _sha256(expected_jsonl_sha256, field="expected_validation_jsonl_sha256")
    summary_id = _sha256(expected_summary_id, field="expected_validation_summary_id")
    row_count = _strict_int(
        expected_row_count, field="expected_validation_row_count", minimum=1
    )
    if (
        payload["outer_artifact_manifest_sha256"] != outer
        or payload["jsonl_sha256"] != jsonl
        or payload["summary_id"] != summary_id
        or payload["row_count"] != row_count
    ):
        raise ValueError(
            "preflight validation commitment differs from external evidence"
        )
    expected_id = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "commitment_id"}
    )
    if payload["commitment_id"] != expected_id:
        raise ValueError("preflight validation commitment_id mismatch")
    return dict(payload)


def _validate_fresh_reconstruction_probe(
    probe: object, *, expected_run_id: str
) -> dict[str, Any]:
    payload = _exact_fields(
        probe,
        {
            "schema",
            "probe_id",
            "run_id",
            "model_construct_wall_ns",
            "cuda_transfer_wall_ns",
            "optimizer_construct_wall_ns",
            "total_wall_ns",
            "state_discarded",
        },
        label="M04a fresh-reconstruction probe",
    )
    if payload["schema"] != PREFLIGHT_FRESH_RECONSTRUCTION_PROBE_SCHEMA_VERSION:
        raise ValueError("unsupported fresh-reconstruction probe schema")
    if payload["run_id"] != expected_run_id:
        raise ValueError("fresh-reconstruction probe belongs to another run")
    components = []
    for field in (
        "model_construct_wall_ns",
        "cuda_transfer_wall_ns",
        "optimizer_construct_wall_ns",
    ):
        components.append(_strict_int(payload[field], field=f"fresh_probe.{field}"))
    total = _strict_int(
        payload["total_wall_ns"], field="fresh_probe.total_wall_ns", minimum=1
    )
    if total != sum(components):
        raise ValueError("fresh-reconstruction probe total does not close")
    if payload["state_discarded"] is not True:
        raise ValueError("fresh-reconstruction probe state must be discarded")
    expected_id = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "probe_id"}
    )
    if payload["probe_id"] != expected_id:
        raise ValueError("fresh-reconstruction probe_id mismatch")
    return dict(payload)


def _validate_checkpoint_cost_probe(
    probe: object, *, expected_run_id: str
) -> dict[str, Any]:
    payload = _exact_fields(
        probe,
        {
            "schema",
            "probe_id",
            "probe_scope",
            "run_id",
            "checkpoint_sha256",
            "checkpoint_bytes",
            "write_wall_ns",
            "weights_only_load_wall_ns",
            "roundtrip_wall_ns",
            "weights_only_load",
            "artifact_disposition",
            "selectable_checkpoint_created",
        },
        label="M04a checkpoint-cost probe",
    )
    if payload["schema"] != PREFLIGHT_CHECKPOINT_COST_PROBE_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint-cost probe schema")
    if payload["probe_scope"] != "content_addressed_independent_probe":
        raise ValueError("checkpoint-cost probe scope is unsupported")
    if payload["run_id"] != expected_run_id:
        raise ValueError("checkpoint-cost probe belongs to another run")
    _sha256(payload["checkpoint_sha256"], field="probe.checkpoint_sha256")
    _strict_int(payload["checkpoint_bytes"], field="probe.checkpoint_bytes", minimum=1)
    write = _strict_int(
        payload["write_wall_ns"], field="probe.write_wall_ns", minimum=1
    )
    load = _strict_int(
        payload["weights_only_load_wall_ns"],
        field="probe.weights_only_load_wall_ns",
        minimum=1,
    )
    if payload["roundtrip_wall_ns"] != write + load:
        raise ValueError("checkpoint-cost probe roundtrip does not close")
    if (
        payload["weights_only_load"] is not True
        or payload["artifact_disposition"] != "retained_nonselectable_diagnostic"
        or payload["selectable_checkpoint_created"] is not False
    ):
        raise ValueError("checkpoint-cost probe is not isolated scratch evidence")
    expected_id = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "probe_id"}
    )
    if payload["probe_id"] != expected_id:
        raise ValueError("checkpoint-cost probe_id mismatch")
    return dict(payload)


def _validate_final_selection_probe(
    probe: object, *, expected_run_id: str
) -> dict[str, Any]:
    payload = _exact_fields(
        probe,
        {
            "schema",
            "probe_id",
            "run_id",
            "candidate_count",
            "selection_rule",
            "selected_checkpoint_index",
            "selection_wall_ns",
        },
        label="M04a final-selection probe",
    )
    if payload["schema"] != PREFLIGHT_FINAL_SELECTION_PROBE_SCHEMA_VERSION:
        raise ValueError("unsupported final-selection probe schema")
    if payload["run_id"] != expected_run_id:
        raise ValueError("final-selection probe belongs to another run")
    if (
        payload["candidate_count"] != VALIDATION_PASS_COUNT
        or payload["selection_rule"] != CHECKPOINT_SELECTION_RULE
    ):
        raise ValueError(
            "final-selection probe did not execute the frozen "
            f"{VALIDATION_PASS_COUNT}-way rule"
        )
    selected_index = _strict_int(
        payload["selected_checkpoint_index"],
        field="selection_probe.selected_checkpoint_index",
        minimum=1,
    )
    if selected_index > VALIDATION_PASS_COUNT:
        raise ValueError(
            "final-selection probe selected index must be in "
            f"1..{VALIDATION_PASS_COUNT}"
        )
    _strict_int(
        payload["selection_wall_ns"],
        field="selection_probe.selection_wall_ns",
        minimum=1,
    )
    expected_id = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "probe_id"}
    )
    if payload["probe_id"] != expected_id:
        raise ValueError("final-selection probe_id mismatch")
    return dict(payload)


def _validate_lock_handshake_probe(
    probe: object, *, expected_run_id: str
) -> dict[str, Any]:
    payload = _exact_fields(
        probe,
        {
            "schema",
            "probe_id",
            "run_id",
            "source",
            "acquisition_started_perf_counter_ns",
            "handshake_completed_perf_counter_ns",
            "wall_ns",
        },
        label="M04a lock-handshake probe",
    )
    if payload["schema"] != PREFLIGHT_LOCK_HANDSHAKE_PROBE_SCHEMA_VERSION:
        raise ValueError("unsupported lock-handshake probe schema")
    if payload["run_id"] != expected_run_id:
        raise ValueError("lock-handshake probe belongs to another run")
    if payload["source"] != "verified_launcher_lock_handshake":
        raise ValueError("lock-handshake probe must originate in the verified launcher")
    start = _strict_int(
        payload["acquisition_started_perf_counter_ns"],
        field="lock_probe.acquisition_started_perf_counter_ns",
    )
    end = _strict_int(
        payload["handshake_completed_perf_counter_ns"],
        field="lock_probe.handshake_completed_perf_counter_ns",
    )
    if end <= start or payload["wall_ns"] != end - start:
        raise ValueError("lock-handshake probe wall time does not close from endpoints")
    expected_id = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "probe_id"}
    )
    if payload["probe_id"] != expected_id:
        raise ValueError("lock-handshake probe_id mismatch")
    return dict(payload)


_LOCK_HANDSHAKE_ARTIFACT_FIELDS = {
    "schema",
    "handshake_id",
    "run_id",
    "gpu_uuid",
    "lock_path",
    "lock_st_dev",
    "lock_st_ino",
    "lock_holder_pid",
    "lock_holder_start_ticks",
    "inherited_lock_fd",
    "attempt_nonce",
    "boot_id",
    "acquisition_started_perf_counter_ns",
    "eligibility_rechecked_perf_counter_ns",
    "handshake_completed_perf_counter_ns",
    "wall_ns",
    "launcher_sha256",
    "launch_plan_sha256",
    "source",
}


def validate_lock_handshake_artifact(
    artifact: object,
    *,
    artifact_bytes: bytes,
    expected_artifact_sha256: str,
    expected_run_id: str,
    expected_gpu_uuid: str,
    expected_launcher_sha256: str,
    expected_launch_plan_sha256: str,
    expected_remote_project_root: str,
    expected_attempt_nonce: str,
) -> dict[str, Any]:
    """Validate the complete canonical launcher handshake without opening its FD."""

    payload = _exact_fields(
        artifact,
        _LOCK_HANDSHAKE_ARTIFACT_FIELDS,
        label="M04a lock-handshake artifact",
    )
    if type(artifact_bytes) is not bytes or serialize_json(payload) != artifact_bytes:
        raise ValueError("lock-handshake artifact bytes are not canonical")
    expected_artifact = _sha256(
        expected_artifact_sha256, field="expected_lock_handshake_artifact_sha256"
    )
    if hashlib.sha256(artifact_bytes).hexdigest() != expected_artifact:
        raise ValueError("lock-handshake artifact SHA-256 mismatch")
    if payload["schema"] != LOCK_HANDSHAKE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported lock-handshake artifact schema")
    run_id = _nonempty_string(expected_run_id, field="expected_run_id")
    gpu_uuid = _nonempty_string(expected_gpu_uuid, field="expected_gpu_uuid")
    if _GPU_UUID_PATTERN.fullmatch(gpu_uuid) is None:
        raise ValueError("expected_gpu_uuid must be a full NVIDIA GPU UUID")
    remote_root_text = _nonempty_string(
        expected_remote_project_root, field="expected_remote_project_root"
    )
    remote_root = PurePosixPath(remote_root_text)
    if (
        not remote_root.is_absolute()
        or ".." in remote_root.parts
        or remote_root.as_posix() != remote_root_text
    ):
        raise ValueError("expected_remote_project_root must be absolute lexical POSIX")
    expected = {
        "run_id": run_id,
        "gpu_uuid": gpu_uuid,
        "lock_path": (remote_root / "locks" / f"gpu-{gpu_uuid}.lock").as_posix(),
        "launcher_sha256": _sha256(
            expected_launcher_sha256, field="expected_launcher_sha256"
        ),
        "launch_plan_sha256": _sha256(
            expected_launch_plan_sha256, field="expected_launch_plan_sha256"
        ),
        "attempt_nonce": _sha256(
            expected_attempt_nonce, field="expected_attempt_nonce"
        ),
        "source": "reviewed_launcher_inherited_posix_flock",
    }
    if any(payload[field] != value for field, value in expected.items()):
        raise ValueError("lock-handshake artifact differs from launch commitments")
    for field, minimum in (
        ("lock_st_dev", 0),
        ("lock_st_ino", 1),
        ("lock_holder_pid", 1),
        ("lock_holder_start_ticks", 1),
        ("inherited_lock_fd", 3),
    ):
        _strict_int(payload[field], field=f"lock_artifact.{field}", minimum=minimum)
    _sha256(payload["attempt_nonce"], field="lock_artifact.attempt_nonce")
    if (
        not isinstance(payload["boot_id"], str)
        or _BOOT_ID_PATTERN.fullmatch(payload["boot_id"]) is None
    ):
        raise ValueError("lock-handshake boot_id is malformed")
    started = _strict_int(
        payload["acquisition_started_perf_counter_ns"],
        field="lock_artifact.acquisition_started_perf_counter_ns",
    )
    rechecked = _strict_int(
        payload["eligibility_rechecked_perf_counter_ns"],
        field="lock_artifact.eligibility_rechecked_perf_counter_ns",
        minimum=1,
    )
    completed = _strict_int(
        payload["handshake_completed_perf_counter_ns"],
        field="lock_artifact.handshake_completed_perf_counter_ns",
        minimum=1,
    )
    wall = _strict_int(payload["wall_ns"], field="lock_artifact.wall_ns", minimum=1)
    if not started < rechecked <= completed or wall != completed - started:
        raise ValueError("lock-handshake artifact endpoints do not close")
    expected_id = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "handshake_id"}
    )
    if payload["handshake_id"] != expected_id:
        raise ValueError("lock-handshake handshake_id mismatch")
    return dict(payload)


def _lock_probe_from_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    child: dict[str, Any] = {
        "schema": PREFLIGHT_LOCK_HANDSHAKE_PROBE_SCHEMA_VERSION,
        "run_id": artifact["run_id"],
        "acquisition_started_perf_counter_ns": artifact[
            "acquisition_started_perf_counter_ns"
        ],
        "handshake_completed_perf_counter_ns": artifact[
            "handshake_completed_perf_counter_ns"
        ],
        "wall_ns": artifact["wall_ns"],
        "source": "verified_launcher_lock_handshake",
    }
    child["probe_id"] = _canonical_sha256(child)
    return child


_CAMPAIGN_OVERHEAD_PROJECTION_FIELDS = {
    "schema",
    "projection_id",
    "run_id",
    "projection_method",
    "fixed_checkpoint_multiplier",
    "lock_handshake_probe_id",
    "measured_lock_setup_handshake_wall_ns",
    "fresh_reconstruction_probe_id",
    "measured_fresh_reconstruction_wall_ns",
    "checkpoint_cost_probe_id",
    "measured_checkpoint_write_wall_ns",
    "measured_checkpoint_weights_only_load_wall_ns",
    "measured_checkpoint_roundtrip_wall_ns",
    "final_selection_probe_id",
    "measured_final_selection_wall_ns",
    "projected_checkpoint_writes",
    "projected_checkpoint_loads",
    "projected_checkpoint_write_wall_ns",
    "projected_checkpoint_weights_only_load_wall_ns",
    "projected_checkpoint_roundtrip_wall_ns",
    "projected_fresh_setup_wall_ns",
    "projected_final_selection_wall_ns",
    "projected_lock_setup_handshake_wall_ns",
    "projected_campaign_overhead_wall_ns",
}


def _validate_campaign_overhead_projection(
    projection: object, *, expected_run_id: str
) -> dict[str, Any]:
    payload = _exact_fields(
        projection,
        _CAMPAIGN_OVERHEAD_PROJECTION_FIELDS,
        label="M04a campaign-overhead projection",
    )
    if (
        payload["schema"] != CAMPAIGN_OVERHEAD_PROJECTION_SCHEMA_VERSION
        or payload["run_id"] != expected_run_id
        or payload["projection_method"] != OVERHEAD_PROJECTION_METHOD
        or payload["fixed_checkpoint_multiplier"] != VALIDATION_PASS_COUNT
    ):
        raise ValueError("campaign-overhead projection identity drifted")
    for field in (
        "lock_handshake_probe_id",
        "fresh_reconstruction_probe_id",
        "checkpoint_cost_probe_id",
        "final_selection_probe_id",
    ):
        _sha256(payload[field], field=f"overhead_projection.{field}")
    handshake = _strict_int(
        payload["measured_lock_setup_handshake_wall_ns"],
        field="overhead_projection.measured_lock_setup_handshake_wall_ns",
        minimum=1,
    )
    fresh = _strict_int(
        payload["measured_fresh_reconstruction_wall_ns"],
        field="overhead_projection.measured_fresh_reconstruction_wall_ns",
        minimum=1,
    )
    write = _strict_int(
        payload["measured_checkpoint_write_wall_ns"],
        field="overhead_projection.measured_checkpoint_write_wall_ns",
        minimum=1,
    )
    load = _strict_int(
        payload["measured_checkpoint_weights_only_load_wall_ns"],
        field="overhead_projection.measured_checkpoint_weights_only_load_wall_ns",
        minimum=1,
    )
    roundtrip = _strict_int(
        payload["measured_checkpoint_roundtrip_wall_ns"],
        field="overhead_projection.measured_checkpoint_roundtrip_wall_ns",
        minimum=1,
    )
    selection = _strict_int(
        payload["measured_final_selection_wall_ns"],
        field="overhead_projection.measured_final_selection_wall_ns",
        minimum=1,
    )
    if roundtrip != write + load:
        raise ValueError("campaign-overhead measured roundtrip does not close")
    multiplier = VALIDATION_PASS_COUNT
    expected = {
        "projected_checkpoint_writes": multiplier,
        "projected_checkpoint_loads": multiplier,
        "projected_checkpoint_write_wall_ns": multiplier * write,
        "projected_checkpoint_weights_only_load_wall_ns": multiplier * load,
        "projected_checkpoint_roundtrip_wall_ns": multiplier * roundtrip,
        "projected_fresh_setup_wall_ns": fresh,
        "projected_final_selection_wall_ns": selection,
        "projected_lock_setup_handshake_wall_ns": handshake,
        "projected_campaign_overhead_wall_ns": (
            handshake + fresh + multiplier * roundtrip + selection
        ),
    }
    if any(payload[field] != value for field, value in expected.items()):
        raise ValueError("campaign-overhead projection does not close")
    expected_id = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "projection_id"}
    )
    if payload["projection_id"] != expected_id:
        raise ValueError("campaign-overhead projection_id mismatch")
    return dict(payload)


def _selection_input_payloads(diagnostic_sha256: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(1, VALIDATION_PASS_COUNT + 1):
        parent_ce = (
            0.25 if index in SELECTION_TIE_INDICES else 1.0 + index / 100.0
        )
        rows.append(
            {
                "checkpoint_index": index,
                "optimizer_step": index * VALIDATION_INTERVAL,
                "parent_grouped_ce_hex": parent_ce.hex(),
                "checkpoint_sha256": hashlib.sha256(
                    f"{diagnostic_sha256}:{index}".encode("ascii")
                ).hexdigest(),
            }
        )
    return rows


_CAMPAIGN_OVERHEAD_REPORT_FIELDS = {
    "schema",
    "probe_id",
    "status",
    "run_id",
    "runtime_mode",
    "optimizer_step",
    "source_optimizer_state",
    "config_sha256",
    "runtime_source_sha256",
    "test_source_sha256",
    "lock_handshake_probe",
    "fresh_reconstruction_probe",
    "checkpoint_cost_probe",
    "final_selection_probe",
    "selection_input_sha256",
    "selected_metric_summary",
    "overhead_projection",
    "checkpoint_artifact_filename",
    "checkpoint_artifact_status",
    "dataset_file_reads",
    "fallback_used",
}


def validate_campaign_overhead_cost_probe_report(
    report: object,
    *,
    expected_run_id: str,
    expected_config_sha256: str,
    expected_runtime_source_sha256: str,
    expected_test_source_sha256: str,
    diagnostic_checkpoint_snapshot: bytes,
) -> dict[str, Any]:
    """Pure replay of the production CUDA overhead report and diagnostic bytes."""

    payload = _exact_fields(
        report,
        _CAMPAIGN_OVERHEAD_REPORT_FIELDS,
        label="M04a campaign-overhead cost probe",
    )
    run_id = _nonempty_string(expected_run_id, field="expected_run_id")
    if (
        payload["schema"] != CAMPAIGN_OVERHEAD_PROBE_SCHEMA_VERSION
        or payload["status"] != "PASS"
        or payload["run_id"] != run_id
        or payload["runtime_mode"] != "frozen_cuda"
        or payload["optimizer_step"] != PREFLIGHT_UPDATES
        or payload["source_optimizer_state"] != "fully_initialized_step_100"
    ):
        raise ValueError("campaign-overhead production identity drifted")
    expected_sources = {
        "config_sha256": expected_config_sha256,
        "runtime_source_sha256": expected_runtime_source_sha256,
        "test_source_sha256": expected_test_source_sha256,
    }
    for field, expected in expected_sources.items():
        if payload[field] != _sha256(expected, field=f"expected_{field}"):
            raise ValueError(f"campaign-overhead {field} differs from commitment")
    if payload["config_sha256"] != _train_contract().training_config_sha256():
        raise ValueError("campaign-overhead training config drifted")
    handshake = _validate_lock_handshake_probe(
        payload["lock_handshake_probe"], expected_run_id=run_id
    )
    fresh = _validate_fresh_reconstruction_probe(
        payload["fresh_reconstruction_probe"], expected_run_id=run_id
    )
    checkpoint = _validate_checkpoint_cost_probe(
        payload["checkpoint_cost_probe"], expected_run_id=run_id
    )
    selection = _validate_final_selection_probe(
        payload["final_selection_probe"], expected_run_id=run_id
    )
    selection_rows = _selection_input_payloads(checkpoint["checkpoint_sha256"])
    selection_input = _canonical_sha256(selection_rows)
    if payload["selection_input_sha256"] != selection_input:
        raise ValueError(
            "campaign-overhead selection input does not bind "
            f"{VALIDATION_PASS_COUNT} rows"
        )
    metrics = tuple(
        _train_contract().CheckpointMetric(
            optimizer_step=int(row["optimizer_step"]),
            parent_grouped_ce=float.fromhex(str(row["parent_grouped_ce_hex"])),
            checkpoint_sha256=str(row["checkpoint_sha256"]),
        )
        for row in selection_rows
    )
    selected_metric = _train_contract().select_checkpoint(metrics)
    selected_index = metrics.index(selected_metric) + 1
    if (
        selection["selected_checkpoint_index"] != selected_index
        or selected_index != SELECTION_TIE_INDICES[0]
    ):
        raise ValueError("campaign-overhead selection violates exact argmin tie-break")
    selected_summary = _exact_fields(
        payload["selected_metric_summary"],
        {
            "schema",
            "summary_id",
            "selection_input_sha256",
            "final_selection_probe_id",
            "checkpoint_index",
            "optimizer_step",
            "parent_grouped_ce_hex",
            "checkpoint_sha256",
        },
        label="M04a selected-metric summary",
    )
    expected_summary = {
        "schema": PREFLIGHT_SELECTED_METRIC_SUMMARY_SCHEMA_VERSION,
        "selection_input_sha256": selection_input,
        "final_selection_probe_id": selection["probe_id"],
        **selection_rows[selected_index - 1],
    }
    if any(
        selected_summary[field] != value for field, value in expected_summary.items()
    ):
        raise ValueError("campaign-overhead selected summary drifted")
    if selected_summary["summary_id"] != _canonical_sha256(expected_summary):
        raise ValueError("campaign-overhead selected summary_id mismatch")
    projection = _validate_campaign_overhead_projection(
        payload["overhead_projection"], expected_run_id=run_id
    )
    child_bindings = {
        "lock_handshake_probe_id": handshake["probe_id"],
        "fresh_reconstruction_probe_id": fresh["probe_id"],
        "checkpoint_cost_probe_id": checkpoint["probe_id"],
        "final_selection_probe_id": selection["probe_id"],
        "measured_lock_setup_handshake_wall_ns": handshake["wall_ns"],
        "measured_fresh_reconstruction_wall_ns": fresh["total_wall_ns"],
        "measured_checkpoint_write_wall_ns": checkpoint["write_wall_ns"],
        "measured_checkpoint_weights_only_load_wall_ns": checkpoint[
            "weights_only_load_wall_ns"
        ],
        "measured_checkpoint_roundtrip_wall_ns": checkpoint["roundtrip_wall_ns"],
        "measured_final_selection_wall_ns": selection["selection_wall_ns"],
    }
    if any(projection[field] != value for field, value in child_bindings.items()):
        raise ValueError("campaign-overhead projection differs from child probes")
    expected_filename = (
        f"NOT_TRAINING_CHECKPOINT.diagnostic.{checkpoint['checkpoint_sha256']}.pt"
    )
    if (
        payload["checkpoint_artifact_filename"] != expected_filename
        or payload["checkpoint_artifact_status"] != "NOT_TRAINING_CHECKPOINT/diagnostic"
        or payload["dataset_file_reads"] != 0
        or payload["fallback_used"] is not False
    ):
        raise ValueError("campaign-overhead diagnostic disposition drifted")
    if type(diagnostic_checkpoint_snapshot) is not bytes:
        raise TypeError("diagnostic checkpoint snapshot must be immutable bytes")
    if (
        len(diagnostic_checkpoint_snapshot) != checkpoint["checkpoint_bytes"]
        or hashlib.sha256(diagnostic_checkpoint_snapshot).hexdigest()
        != checkpoint["checkpoint_sha256"]
    ):
        raise ValueError("diagnostic checkpoint bytes differ from the cost probe")
    expected_id = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "probe_id"}
    )
    if payload["probe_id"] != expected_id:
        raise ValueError("campaign-overhead probe_id mismatch")
    return dict(payload)


_ENVIRONMENT_FIELDS = {
    "schema",
    "python",
    "python_implementation",
    "python_version",
    "python_executable",
    "python_executable_sha256",
    "python_executable_bytes",
    "platform",
    "hostname",
    "torch",
    "torch_cuda",
    "cudnn",
    "nvidia_driver",
    "cuda_visible_devices",
    "gpu_uuid",
    "run_id",
    "cublas_workspace_config",
    "gpu",
    "deterministic_algorithms",
    "cudnn_benchmark",
    "tf32_matmul",
    "tf32_cudnn",
    "flash_sdp",
    "mem_efficient_sdp",
    "math_sdp",
    "launcher_path",
    "launcher_sha256",
    "ordered_sys_path",
    "ordered_import_roots",
    "conda_explicit_path",
    "conda_explicit_sha256",
    "visible_root",
    "runtime_source_sha256",
    "test_source_sha256",
    "visible_files",
    "python_runtime_lock_sha256",
    "python_runtime_lock_id",
}


def _runtime_lock_python_identity(runtime_lock: object) -> dict[str, Any]:
    runtime_lock_module = importlib.import_module(
        ".m04a_python_runtime_lock", __package__
    )
    artifact_type = runtime_lock_module.PythonRuntimeLockArtifact
    if type(runtime_lock) is artifact_type:
        runtime_payload = runtime_lock_module._fresh_payload_from_artifact(
            runtime_lock,
            expected_artifact_sha256=runtime_lock.artifact_sha256,
        )
    elif isinstance(runtime_lock, Mapping):
        runtime_payload = runtime_lock
    else:
        raise TypeError(
            "python_runtime_lock must be a parsed payload or typed artifact"
        )
    python = runtime_payload.get("python")
    if not isinstance(python, Mapping):
        raise ValueError("python runtime lock is missing its Python identity")
    return {
        "runtime_lock_id": _sha256(
            runtime_payload.get("runtime_lock_id"),
            field="python_runtime_lock.runtime_lock_id",
        ),
        "python_implementation": _nonempty_string(
            python.get("implementation"),
            field="python_runtime_lock.python.implementation",
        ),
        "python_version": _nonempty_string(
            python.get("version"), field="python_runtime_lock.python.version"
        ),
        "python_executable": _nonempty_string(
            python.get("executable_lexical_path"),
            field="python_runtime_lock.python.executable_lexical_path",
        ),
        "python_executable_sha256": _sha256(
            python.get("executable_sha256"),
            field="python_runtime_lock.python.executable_sha256",
        ),
        "python_executable_bytes": _strict_int(
            python.get("executable_bytes"),
            field="python_runtime_lock.python.executable_bytes",
            minimum=1,
        ),
    }


_PYTHON_RUNTIME_IDENTITY_FIELDS = {
    "implementation",
    "version",
    "executable",
    "executable_sha256",
    "executable_bytes",
}


def python_runtime_identity_sha256(identity: object) -> str:
    """Content-address the exact Python identity propagated through M04a lineage."""

    payload = _exact_fields(
        identity,
        _PYTHON_RUNTIME_IDENTITY_FIELDS,
        label="Python runtime identity",
    )
    _nonempty_string(payload["implementation"], field="identity.implementation")
    _nonempty_string(payload["version"], field="identity.version")
    executable = _nonempty_string(payload["executable"], field="identity.executable")
    executable_path = PurePosixPath(executable)
    if (
        not executable_path.is_absolute()
        or executable_path.as_posix() != executable
        or any(part in {"", ".", ".."} for part in executable_path.parts[1:])
    ):
        raise ValueError("identity.executable must be one lexical POSIX path")
    _sha256(payload["executable_sha256"], field="identity.executable_sha256")
    _strict_int(
        payload["executable_bytes"], field="identity.executable_bytes", minimum=1
    )
    return _canonical_sha256(payload)


def _runtime_lock_identity_payload(runtime_lock: object) -> dict[str, Any]:
    identity = _runtime_lock_python_identity(runtime_lock)
    return {
        "implementation": identity["python_implementation"],
        "version": identity["python_version"],
        "executable": identity["python_executable"],
        "executable_sha256": identity["python_executable_sha256"],
        "executable_bytes": identity["python_executable_bytes"],
    }


def _environment_identity_payload(environment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "implementation": environment["python_implementation"],
        "version": environment["python_version"],
        "executable": environment["python_executable"],
        "executable_sha256": environment["python_executable_sha256"],
        "executable_bytes": environment["python_executable_bytes"],
    }


def _validate_environment_manifest_shape(manifest: object) -> dict[str, Any]:
    payload = _exact_fields(
        manifest, _ENVIRONMENT_FIELDS, label="M04a environment manifest"
    )
    if payload["schema"] != "afts-m04a-environment/v0.3":
        raise ValueError("unsupported M04a environment schema")
    for field in (
        "python",
        "python_implementation",
        "python_version",
        "python_executable",
        "platform",
        "hostname",
        "torch",
        "torch_cuda",
        "nvidia_driver",
        "cuda_visible_devices",
        "gpu_uuid",
        "run_id",
        "launcher_path",
        "conda_explicit_path",
        "visible_root",
    ):
        _nonempty_string(payload[field], field=f"environment.{field}")
    if (
        payload["python_implementation"] != "CPython"
        or payload["python_version"] != "3.10.20"
        or payload["python"].split(maxsplit=1)[0] != payload["python_version"]
    ):
        raise ValueError("environment requires exact CPython 3.10.20")
    executable = PurePosixPath(payload["python_executable"])
    if (
        not executable.is_absolute()
        or executable.as_posix() != payload["python_executable"]
        or any(part in {"", ".", ".."} for part in executable.parts[1:])
    ):
        raise ValueError("environment Python executable path is not lexical POSIX")
    _sha256(
        payload["python_executable_sha256"],
        field="environment.python_executable_sha256",
    )
    _strict_int(
        payload["python_executable_bytes"],
        field="environment.python_executable_bytes",
        minimum=1,
    )
    if (
        payload["torch"] != "2.10.0+cu128"
        or payload["torch_cuda"] != "12.8"
        or payload["cudnn"] != 91_002
    ):
        raise ValueError("environment frozen PyTorch/CUDA/cuDNN versions drifted")
    if (
        _GPU_UUID_PATTERN.fullmatch(payload["gpu_uuid"]) is None
        or payload["cuda_visible_devices"] != payload["gpu_uuid"]
    ):
        raise ValueError("environment GPU UUID binding is malformed")
    for field in (
        "launcher_sha256",
        "conda_explicit_sha256",
        "runtime_source_sha256",
        "test_source_sha256",
        "python_runtime_lock_sha256",
        "python_runtime_lock_id",
    ):
        _sha256(payload[field], field=f"environment.{field}")
    fixed_controls = {
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "tf32_matmul": False,
        "tf32_cudnn": False,
        "flash_sdp": False,
        "mem_efficient_sdp": False,
        "math_sdp": True,
    }
    if any(payload[field] != value for field, value in fixed_controls.items()):
        raise ValueError("environment deterministic runtime controls drifted")
    ordered_sys_path = payload["ordered_sys_path"]
    if (
        not isinstance(ordered_sys_path, list)
        or not ordered_sys_path
        or any(not isinstance(path, str) or not path for path in ordered_sys_path)
    ):
        raise ValueError("environment ordered_sys_path is malformed")
    ordered_import_roots = payload["ordered_import_roots"]
    if (
        not isinstance(ordered_import_roots, list)
        or not ordered_import_roots
        or any(
            not isinstance(path, str)
            or not PurePosixPath(path).is_absolute()
            or PurePosixPath(path).as_posix() != path
            for path in ordered_import_roots
        )
    ):
        raise ValueError("environment ordered_import_roots are malformed")
    gpu = _exact_fields(
        payload["gpu"], {"name", "total_memory", "major", "minor"}, label="GPU"
    )
    _nonempty_string(gpu["name"], field="environment.gpu.name")
    _strict_int(gpu["total_memory"], field="environment.gpu.total_memory", minimum=1)
    _strict_int(gpu["major"], field="environment.gpu.major", minimum=1)
    _strict_int(gpu["minor"], field="environment.gpu.minor")
    visible_files = payload["visible_files"]
    if not isinstance(visible_files, dict) or not visible_files:
        raise ValueError("environment visible-file inventory must be nonempty")
    for name, digest in visible_files.items():
        if not isinstance(name, str):
            raise TypeError("environment visible-file names must be strings")
        relative = PurePosixPath(name)
        if (
            not name
            or "\\" in name
            or relative.is_absolute()
            or relative.as_posix() != name
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError("environment visible-file path is unsafe")
        _sha256(digest, field=f"environment.visible_files[{name!r}]")
    if (
        visible_files.get("python-runtime-lock.json")
        != payload["python_runtime_lock_sha256"]
        or visible_files.get("remote_launcher.py") != payload["launcher_sha256"]
        or visible_files.get("conda-explicit.txt") != payload["conda_explicit_sha256"]
    ):
        raise ValueError("environment visible files do not bind runtime commitments")
    visible_root = PurePosixPath(payload["visible_root"])
    conda_path = PurePosixPath(payload["conda_explicit_path"])
    launcher_path = PurePosixPath(payload["launcher_path"])
    if (
        not visible_root.is_absolute()
        or visible_root.as_posix() != payload["visible_root"]
        or conda_path != visible_root / "conda-explicit.txt"
        or launcher_path != visible_root / "remote_launcher.py"
    ):
        raise ValueError("environment visible-root paths differ from the frozen layout")
    return dict(payload)


def validate_environment_manifest_artifact(
    manifest: object,
    *,
    launch_plan: Mapping[str, Any],
    expected_gpu_uuid: str,
    expected_launcher_sha256: str,
    python_runtime_lock: object,
) -> dict[str, Any]:
    """Purely bind the CUDA environment record to launch-plan commitments."""

    payload = _validate_environment_manifest_shape(manifest)
    runtime_identity = _runtime_lock_python_identity(python_runtime_lock)
    if (
        not isinstance(expected_gpu_uuid, str)
        or _GPU_UUID_PATTERN.fullmatch(expected_gpu_uuid) is None
    ):
        raise ValueError("expected_gpu_uuid must be a canonical full NVIDIA GPU UUID")
    expected = {
        "run_id": launch_plan["run_id"],
        "gpu_uuid": expected_gpu_uuid,
        "cuda_visible_devices": expected_gpu_uuid,
        "launcher_sha256": _sha256(
            expected_launcher_sha256, field="expected_launcher_sha256"
        ),
        "ordered_import_roots": launch_plan["ordered_import_roots"],
        "conda_explicit_sha256": launch_plan["conda_explicit_sha256"],
        "runtime_source_sha256": launch_plan["runtime_source_fingerprint_sha256"],
        "test_source_sha256": launch_plan["test_source_fingerprint_sha256"],
        "python_runtime_lock_sha256": launch_plan["expected_input_artifacts"][
            "python-runtime-lock.json"
        ],
        "python_runtime_lock_id": runtime_identity["runtime_lock_id"],
        "python_implementation": runtime_identity["python_implementation"],
        "python_version": runtime_identity["python_version"],
        "python_executable": runtime_identity["python_executable"],
        "python_executable_sha256": runtime_identity["python_executable_sha256"],
        "python_executable_bytes": runtime_identity["python_executable_bytes"],
    }
    if any(payload[field] != value for field, value in expected.items()):
        raise ValueError("environment manifest differs from launch/runtime commitments")
    ordered_sys_path = payload["ordered_sys_path"]
    expected_bound_roots = [
        f"/proc/self/fd/{200 + index}"
        for index in range(len(launch_plan["ordered_import_roots"]))
    ]
    if (
        not isinstance(ordered_sys_path, list)
        or len(ordered_sys_path) < len(launch_plan["ordered_import_roots"])
        or ordered_sys_path[-len(launch_plan["ordered_import_roots"]) :]
        != expected_bound_roots
        or any(not isinstance(path, str) for path in ordered_sys_path)
    ):
        raise ValueError("environment ordered_sys_path lost isolated import roots")
    visible_files = payload["visible_files"]
    expected_visible_files = {
        **launch_plan["expected_input_artifacts"],
        "conda-explicit.txt": launch_plan["conda_explicit_sha256"],
    }
    if visible_files != expected_visible_files:
        raise ValueError(
            "environment inventory differs from the launch-plan closed world"
        )
    return dict(payload)


def _validate_preflight_projection(
    projection: object,
    *,
    training_wall_time_ns: int,
    validation_episodes_per_pass: int,
    lock_handshake_probe_id: str,
    lock_setup_handshake_wall_ns: int,
    handshake_completed_perf_counter_ns: int,
    preflight_started_perf_counter_ns: int,
    fresh_reconstruction_probe_id: str,
    fresh_reconstruction_wall_ns: int,
    checkpoint_cost_probe_id: str,
    checkpoint_roundtrip_wall_ns: int,
    final_selection_probe_id: str,
    final_selection_wall_ns: int,
) -> dict[str, Any]:
    fields = {
        "schema",
        "projection_id",
        "projection_method",
        "measured_preflight_training_wall_ns",
        "measured_preflight_total_wall_ns",
        "mean_training_update_wall_ns",
        "mean_training_microbatch_wall_ns",
        "lock_handshake_probe_id",
        "measured_lock_setup_handshake_wall_ns",
        "measured_handshake_to_preflight_start_ns",
        "fresh_reconstruction_probe_id",
        "measured_fresh_reconstruction_wall_ns",
        "checkpoint_cost_probe_id",
        "measured_checkpoint_roundtrip_wall_ns",
        "final_selection_probe_id",
        "measured_final_selection_wall_ns",
        "projected_primary_updates",
        "projected_primary_wall_ns",
        "projected_validation_passes",
        "validation_episodes_per_pass",
        "projected_validation_episode_calls",
        "projected_validation_wall_ns",
        "projected_checkpoint_writes",
        "projected_checkpoint_wall_ns",
        "projected_final_selection_wall_ns",
        "post_preflight_validation_margin_ns",
        "projected_campaign_wall_ns",
        "budget_limit_ns",
        "budget_remaining_ns",
        "budget_status",
    }
    payload = _exact_fields(projection, fields, label="M04a preflight projection")
    total_wall = _strict_int(
        payload["measured_preflight_total_wall_ns"],
        field="projection.measured_preflight_total_wall_ns",
        minimum=1,
    )
    validation_count = _strict_int(
        validation_episodes_per_pass,
        field="validation_episodes_per_pass",
        minimum=1,
    )
    mean_update = (training_wall_time_ns + PREFLIGHT_UPDATES - 1) // PREFLIGHT_UPDATES
    microbatch_count = PREFLIGHT_UPDATES * PREFLIGHT_MICROBATCHES_PER_UPDATE
    mean_microbatch = (training_wall_time_ns + microbatch_count - 1) // microbatch_count
    primary_wall = mean_update * OPTIMIZER_UPDATES
    validation_calls = VALIDATION_PASS_COUNT * validation_count
    validation_wall = mean_microbatch * validation_calls
    handshake_wall = _strict_int(
        lock_setup_handshake_wall_ns,
        field="lock_setup_handshake_wall_ns",
        minimum=1,
    )
    handshake_completed = _strict_int(
        handshake_completed_perf_counter_ns,
        field="handshake_completed_perf_counter_ns",
        minimum=1,
    )
    preflight_started = _strict_int(
        preflight_started_perf_counter_ns,
        field="preflight_started_perf_counter_ns",
        minimum=1,
    )
    if preflight_started < handshake_completed:
        raise ValueError("preflight started before the lock handshake completed")
    handshake_to_preflight = preflight_started - handshake_completed
    if handshake_to_preflight > MAX_HANDSHAKE_TO_PREFLIGHT_NS:
        raise ValueError("lock handshake is too old for this preflight attempt")
    fresh_setup_wall = _strict_int(
        fresh_reconstruction_wall_ns,
        field="fresh_reconstruction_wall_ns",
        minimum=1,
    )
    checkpoint_roundtrip = _strict_int(
        checkpoint_roundtrip_wall_ns,
        field="checkpoint_roundtrip_wall_ns",
        minimum=1,
    )
    final_selection_wall = _strict_int(
        final_selection_wall_ns,
        field="final_selection_wall_ns",
        minimum=1,
    )
    checkpoint_wall = VALIDATION_PASS_COUNT * checkpoint_roundtrip
    projected = (
        handshake_wall
        + handshake_to_preflight
        + total_wall
        + fresh_setup_wall
        + primary_wall
        + validation_wall
        + checkpoint_wall
        + final_selection_wall
        + PREFLIGHT_POST_VALIDATION_MARGIN_NS
    )
    expected = {
        "schema": PREFLIGHT_PROJECTION_SCHEMA_VERSION,
        "projection_method": PREFLIGHT_PROJECTION_METHOD,
        "measured_preflight_training_wall_ns": training_wall_time_ns,
        "measured_preflight_total_wall_ns": total_wall,
        "mean_training_update_wall_ns": mean_update,
        "mean_training_microbatch_wall_ns": mean_microbatch,
        "lock_handshake_probe_id": _sha256(
            lock_handshake_probe_id, field="lock_handshake_probe_id"
        ),
        "measured_lock_setup_handshake_wall_ns": handshake_wall,
        "measured_handshake_to_preflight_start_ns": handshake_to_preflight,
        "fresh_reconstruction_probe_id": _sha256(
            fresh_reconstruction_probe_id,
            field="fresh_reconstruction_probe_id",
        ),
        "measured_fresh_reconstruction_wall_ns": fresh_setup_wall,
        "checkpoint_cost_probe_id": _sha256(
            checkpoint_cost_probe_id, field="checkpoint_cost_probe_id"
        ),
        "measured_checkpoint_roundtrip_wall_ns": checkpoint_roundtrip,
        "final_selection_probe_id": _sha256(
            final_selection_probe_id, field="final_selection_probe_id"
        ),
        "measured_final_selection_wall_ns": final_selection_wall,
        "projected_primary_updates": OPTIMIZER_UPDATES,
        "projected_primary_wall_ns": primary_wall,
        "projected_validation_passes": VALIDATION_PASS_COUNT,
        "validation_episodes_per_pass": validation_count,
        "projected_validation_episode_calls": validation_calls,
        "projected_validation_wall_ns": validation_wall,
        "projected_checkpoint_writes": VALIDATION_PASS_COUNT,
        "projected_checkpoint_wall_ns": checkpoint_wall,
        "projected_final_selection_wall_ns": final_selection_wall,
        "post_preflight_validation_margin_ns": PREFLIGHT_POST_VALIDATION_MARGIN_NS,
        "projected_campaign_wall_ns": projected,
        "budget_limit_ns": GPU_HOUR_BUDGET_NS,
        "budget_remaining_ns": GPU_HOUR_BUDGET_NS - projected,
        "budget_status": "WITHIN_BUDGET",
    }
    if total_wall < training_wall_time_ns or projected > GPU_HOUR_BUDGET_NS:
        raise ValueError("preflight projection is inconsistent or exceeds 24 GPU-hours")
    if any(payload[field] != value for field, value in expected.items()):
        raise ValueError("preflight budget projection does not close")
    expected_id = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "projection_id"}
    )
    if payload["projection_id"] != expected_id:
        raise ValueError("preflight projection_id mismatch")
    return dict(payload)


def validate_preflight_artifact(
    report: object,
    preflight_rows: Sequence[Mapping[str, Any]],
    *,
    validation_episodes_per_pass: int,
    expected_validation_outer_manifest_sha256: str,
    expected_validation_jsonl_sha256: str,
    expected_validation_summary_id: str,
    overhead_cost_probe_report: Mapping[str, Any],
    diagnostic_checkpoint_snapshot: bytes,
    lock_handshake_artifact: Mapping[str, Any],
    lock_handshake_artifact_bytes: bytes,
    expected_lock_handshake_artifact_sha256: str,
    expected_config_sha256: str,
    expected_runtime_source_sha256: str,
    expected_test_source_sha256: str,
    expected_launcher_sha256: str,
    expected_launch_plan_sha256: str,
    expected_remote_project_root: str,
    expected_attempt_nonce: str,
) -> dict[str, Any]:
    """Validate ``preflight.json`` against its literal first 100 cost rows."""

    payload = _exact_fields(
        report, _PREFLIGHT_REPORT_FIELDS, label="M04a preflight report"
    )
    rows = _validate_preflight_rows(preflight_rows)
    if (
        payload["schema"] != PREFLIGHT_REPORT_SCHEMA_VERSION
        or payload["status"] != "PASS"
    ):
        raise ValueError("training requires a successful frozen preflight report")
    if payload["fixture"] != _preflight_fixture_payload():
        raise ValueError("preflight constant-grid fixture drifted")
    if (
        payload["dataset_file_reads"] != 0
        or payload["training_checkpoint_writes"] != 0
        or payload["diagnostic_checkpoint_writes"] != 1
        or payload["fallback_used"] is not False
        or payload["rng_state_restored"] is not True
    ):
        raise ValueError("preflight isolation/disposition closure failed")
    ledger_hash = _canonical_sha256(list(rows))
    if payload["ledger_rows_sha256"] != ledger_hash:
        raise ValueError("preflight report does not bind its exact 100 ledger rows")

    runtime = _exact_fields(
        payload["runtime"],
        {"run_id", "gpu_uuid", "logical_device_index"},
        label="preflight runtime identity",
    )
    _nonempty_string(runtime["run_id"], field="runtime.run_id")
    _nonempty_string(runtime["gpu_uuid"], field="runtime.gpu_uuid")
    if runtime["logical_device_index"] != 0:
        raise ValueError("preflight must use logical CUDA device zero")

    model = _exact_fields(
        payload["model"],
        {
            "model_semantics_version",
            "parameter_count",
            "training_config_sha256",
            "optimizer",
            "precision",
            "grad_scaler",
            "disposition",
            "primary_requires_fresh_reconstruction",
        },
        label="preflight model record",
    )
    expected_model = {
        "model_semantics_version": _contract_value("MODEL_SEMANTICS_VERSION"),
        "parameter_count": _contract_value("MODEL_PARAMETER_COUNT"),
        "training_config_sha256": _train_contract().training_config_sha256(),
        "optimizer": "one_group_adamw_full_weight_decay",
        "precision": "bf16_autocast_forward_fp32_loss_gradients_state",
        "grad_scaler": False,
        "disposition": "discarded_without_checkpoint",
        "primary_requires_fresh_reconstruction": True,
    }
    if model != expected_model:
        raise ValueError("preflight model/disposition record drifted")
    if model["training_config_sha256"] != _sha256(
        expected_config_sha256, field="expected_config_sha256"
    ):
        raise ValueError("preflight model differs from external config commitment")

    training = _exact_fields(
        payload["training"],
        _PREFLIGHT_TRAINING_FIELDS,
        label="preflight training summary",
    )
    expected_training = {
        "updates": PREFLIGHT_UPDATES,
        "microbatches_per_update": PREFLIGHT_MICROBATCHES_PER_UPDATE,
        "total_microbatches": PREFLIGHT_UPDATES * PREFLIGHT_MICROBATCHES_PER_UPDATE,
        "masked_tokens_per_microbatch": PREFLIGHT_TARGET_CELL_COUNT,
        "masked_token_predictions": PREFLIGHT_UPDATES
        * PREFLIGHT_MASKED_TOKEN_PREDICTIONS_PER_UPDATE,
        "encoder_forward_calls": PREFLIGHT_UPDATES * PREFLIGHT_MICROBATCHES_PER_UPDATE,
        "decoder_forward_calls": PREFLIGHT_UPDATES * PREFLIGHT_MICROBATCHES_PER_UPDATE,
        "backward_calls": PREFLIGHT_UPDATES * PREFLIGHT_MICROBATCHES_PER_UPDATE,
        "ledger_row_count": PREFLIGHT_UPDATES,
        "ledger_rows_sha256": ledger_hash,
        "total_update_wall_ns": sum(row["wall_time_ns"] for row in rows),
        "total_update_cuda_event_ns": sum(row["cuda_event_ns"] for row in rows),
        "peak_allocated_bytes": max(row["cuda_peak_allocated_bytes"] for row in rows),
        "peak_reserved_bytes": max(row["cuda_peak_reserved_bytes"] for row in rows),
    }
    if any(training[field] != value for field, value in expected_training.items()):
        raise ValueError("preflight training summary does not close from 100 rows")
    for field in (
        "learning_rate_hex",
        "mean_masked_cell_ce_hex",
        "gradient_norm_before_clip_hex",
    ):
        values = training[field]
        if not isinstance(values, list) or len(values) != PREFLIGHT_UPDATES:
            raise ValueError(f"preflight {field} must contain exactly 100 values")
        for index, value in enumerate(values):
            _canonical_nonnegative_float_hex(value, field=f"training.{field}[{index}]")
    expected_lr = [
        _train_contract().learning_rate_for_update(step).hex()
        for step in range(1, PREFLIGHT_UPDATES + 1)
    ]
    if training["learning_rate_hex"] != expected_lr:
        raise ValueError("preflight learning-rate trace drifted")

    inference = _validate_preflight_inference(payload["inference"])
    _validate_preflight_validation_commitment(
        payload["validation_manifest_commitment"],
        expected_outer_manifest_sha256=expected_validation_outer_manifest_sha256,
        expected_jsonl_sha256=expected_validation_jsonl_sha256,
        expected_summary_id=expected_validation_summary_id,
        expected_row_count=validation_episodes_per_pass,
    )
    full_lock = validate_lock_handshake_artifact(
        lock_handshake_artifact,
        artifact_bytes=lock_handshake_artifact_bytes,
        expected_artifact_sha256=expected_lock_handshake_artifact_sha256,
        expected_run_id=runtime["run_id"],
        expected_gpu_uuid=runtime["gpu_uuid"],
        expected_launcher_sha256=expected_launcher_sha256,
        expected_launch_plan_sha256=expected_launch_plan_sha256,
        expected_remote_project_root=expected_remote_project_root,
        expected_attempt_nonce=expected_attempt_nonce,
    )
    full_cost = validate_campaign_overhead_cost_probe_report(
        overhead_cost_probe_report,
        expected_run_id=runtime["run_id"],
        expected_config_sha256=expected_config_sha256,
        expected_runtime_source_sha256=expected_runtime_source_sha256,
        expected_test_source_sha256=expected_test_source_sha256,
        diagnostic_checkpoint_snapshot=diagnostic_checkpoint_snapshot,
    )
    derived_lock_probe = _lock_probe_from_artifact(full_lock)
    if (
        payload["lock_handshake_artifact_sha256"]
        != _sha256(
            expected_lock_handshake_artifact_sha256,
            field="expected_lock_handshake_artifact_sha256",
        )
        or payload["overhead_cost_probe_id"] != full_cost["probe_id"]
        or payload["lock_handshake_probe"] != derived_lock_probe
        or full_cost["lock_handshake_probe"] != derived_lock_probe
    ):
        raise ValueError("preflight report differs from full lock/cost parents")
    preflight_started = _strict_int(
        payload["preflight_started_perf_counter_ns"],
        field="preflight_started_perf_counter_ns",
        minimum=1,
    )
    fresh_probe = _validate_fresh_reconstruction_probe(
        payload["fresh_reconstruction_probe"], expected_run_id=runtime["run_id"]
    )
    checkpoint_probe = _validate_checkpoint_cost_probe(
        payload["checkpoint_cost_probe"], expected_run_id=runtime["run_id"]
    )
    selection_probe = _validate_final_selection_probe(
        payload["final_selection_probe"], expected_run_id=runtime["run_id"]
    )
    handshake_probe = _validate_lock_handshake_probe(
        payload["lock_handshake_probe"], expected_run_id=runtime["run_id"]
    )
    child_probes = {
        "fresh_reconstruction_probe": fresh_probe,
        "checkpoint_cost_probe": checkpoint_probe,
        "final_selection_probe": selection_probe,
        "lock_handshake_probe": handshake_probe,
    }
    if any(full_cost[name] != probe for name, probe in child_probes.items()):
        raise ValueError("preflight child probes differ from the full cost report")
    projection = _validate_preflight_projection(
        payload["budget_projection"],
        training_wall_time_ns=training["total_update_wall_ns"],
        validation_episodes_per_pass=validation_episodes_per_pass,
        lock_handshake_probe_id=handshake_probe["probe_id"],
        lock_setup_handshake_wall_ns=handshake_probe["wall_ns"],
        handshake_completed_perf_counter_ns=handshake_probe[
            "handshake_completed_perf_counter_ns"
        ],
        preflight_started_perf_counter_ns=preflight_started,
        fresh_reconstruction_probe_id=fresh_probe["probe_id"],
        fresh_reconstruction_wall_ns=fresh_probe["total_wall_ns"],
        checkpoint_cost_probe_id=checkpoint_probe["probe_id"],
        checkpoint_roundtrip_wall_ns=checkpoint_probe["roundtrip_wall_ns"],
        final_selection_probe_id=selection_probe["probe_id"],
        final_selection_wall_ns=selection_probe["selection_wall_ns"],
    )
    measured_preflight_components = (
        training["total_update_wall_ns"]
        + inference["inference_wall_time_ns"]
        + fresh_probe["total_wall_ns"]
        + checkpoint_probe["roundtrip_wall_ns"]
        + selection_probe["selection_wall_ns"]
    )
    if projection["measured_preflight_total_wall_ns"] < measured_preflight_components:
        raise ValueError("preflight endpoint omits a required measured cost probe")
    expected_report_id = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "report_id"}
    )
    if payload["report_id"] != expected_report_id:
        raise ValueError("preflight report_id mismatch")
    return dict(payload)


_VALIDATION_PARENT_METRIC_FIELDS = {
    "semantic_parent_id",
    "target_count",
    "mean_masked_cell_ce_hex",
}

_VALIDATION_EPISODE_METRIC_FIELDS = {
    "ledger_event_id",
    "row_ordinal",
    "episode_sha256",
    "semantic_parent_id",
    "target_group_id",
    "masked_token_predictions",
    "masked_cell_ce_hex",
}

_VALIDATION_METRIC_FIELDS = {
    "schema",
    "metric_id",
    "validation_pass_index",
    "optimizer_step",
    "validation_episode_outer_manifest_sha256",
    "validation_episode_manifest_sha256",
    "validation_episode_count",
    "validation_event_ids_sha256",
    "episode_metrics",
    "checkpoint_sha256",
    "aggregation_semantics",
    "semantic_parent_count",
    "target_group_count",
    "parent_metrics",
    "parent_grouped_masked_cell_ce_hex",
}

_CHECKPOINT_MANIFEST_FIELDS = {
    "schema",
    "manifest_id",
    "checkpoint_schema",
    "checkpoint_index",
    "optimizer_step",
    "checkpoint_sha256",
    "checkpoint_bytes",
    "checkpoint_event_id",
    "validation_metric_id",
    "training_config_sha256",
}

_SELECTED_CHECKPOINT_FIELDS = {
    "schema",
    "selection_id",
    "selection_rule",
    "candidate_count",
    "selected_checkpoint_index",
    "optimizer_step",
    "parent_grouped_masked_cell_ce_hex",
    "validation_metric_id",
    "checkpoint_manifest_id",
    "checkpoint_sha256",
    "checkpoint_bytes",
    "checkpoint_manifests_sha256",
    "validation_metrics_sha256",
    "training_cost_ledger_sha256",
    "validation_episode_outer_manifest_sha256",
    "validation_episode_manifest_sha256",
    "training_config_sha256",
    "lock_interval_id",
    "selection_completed_perf_counter_ns",
    "budget_status",
}


def _validate_validation_episode_metrics(
    rows: object,
) -> list[dict[str, Any]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise TypeError("episode_metrics must be a non-empty ordered sequence")
    normalized: list[dict[str, Any]] = []
    closed_targets: set[str] = set()
    previous_target: str | None = None
    target_parent: dict[str, str] = {}
    for ordinal, raw in enumerate(rows):
        row = _exact_fields(
            raw,
            _VALIDATION_EPISODE_METRIC_FIELDS,
            label=f"validation episode_metrics[{ordinal}]",
        )
        row_ordinal = _strict_int(
            row["row_ordinal"],
            field=f"episode_metrics[{ordinal}].row_ordinal",
        )
        if row_ordinal != ordinal:
            raise ValueError(
                "validation episode metrics must follow manifest row order"
            )
        _sha256(
            row["ledger_event_id"], field=f"episode_metrics[{ordinal}].ledger_event_id"
        )
        _sha256(
            row["episode_sha256"], field=f"episode_metrics[{ordinal}].episode_sha256"
        )
        parent_id = _nonempty_string(
            row["semantic_parent_id"],
            field=f"episode_metrics[{ordinal}].semantic_parent_id",
        )
        target_id = _nonempty_string(
            row["target_group_id"],
            field=f"episode_metrics[{ordinal}].target_group_id",
        )
        _strict_int(
            row["masked_token_predictions"],
            field=f"episode_metrics[{ordinal}].masked_token_predictions",
            minimum=1,
        )
        _canonical_nonnegative_fp32_hex(
            row["masked_cell_ce_hex"],
            field=f"episode_metrics[{ordinal}].masked_cell_ce_hex",
        )
        if target_id != previous_target:
            if target_id in closed_targets:
                raise ValueError(
                    "validation target groups must form literal contiguous runs"
                )
            if previous_target is not None:
                closed_targets.add(previous_target)
            previous_target = target_id
        previous_parent = target_parent.setdefault(target_id, parent_id)
        if previous_parent != parent_id:
            raise ValueError(
                "one validation target group cannot cross semantic parents"
            )
        normalized.append(dict(row))
    return normalized


def _replay_validation_hierarchy(
    episode_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int, float]:
    target_order: list[str] = []
    target_parent: dict[str, str] = {}
    target_losses: dict[str, list[float]] = {}
    for row in episode_rows:
        target_id = str(row["target_group_id"])
        if target_id not in target_losses:
            target_order.append(target_id)
            target_parent[target_id] = str(row["semantic_parent_id"])
            target_losses[target_id] = []
        target_losses[target_id].append(
            _canonical_nonnegative_fp32_hex(
                row["masked_cell_ce_hex"], field="masked_cell_ce_hex"
            )
        )
    parent_order: list[str] = []
    parent_targets: dict[str, list[float]] = {}
    for target_id in target_order:
        target_mean = sequential_fp32_mean(target_losses[target_id])
        parent_id = target_parent[target_id]
        if parent_id not in parent_targets:
            parent_order.append(parent_id)
            parent_targets[parent_id] = []
        parent_targets[parent_id].append(target_mean)
    parent_metrics: list[dict[str, Any]] = []
    parent_means: list[float] = []
    for parent_id in parent_order:
        parent_mean = sequential_fp32_mean(parent_targets[parent_id])
        parent_means.append(parent_mean)
        parent_metrics.append(
            {
                "semantic_parent_id": parent_id,
                "target_count": len(parent_targets[parent_id]),
                "mean_masked_cell_ce_hex": parent_mean.hex(),
            }
        )
    return parent_metrics, len(target_order), sequential_fp32_mean(parent_means)


def _validate_validation_parent_metrics(
    rows: object,
) -> tuple[list[dict[str, Any]], int]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise TypeError("parent_metrics must be a non-empty ordered sequence")
    normalized: list[dict[str, Any]] = []
    parent_ids: set[str] = set()
    target_count = 0
    for index, raw in enumerate(rows):
        row = _exact_fields(
            raw,
            _VALIDATION_PARENT_METRIC_FIELDS,
            label=f"validation parent_metrics[{index}]",
        )
        parent_id = _nonempty_string(
            row["semantic_parent_id"],
            field=f"parent_metrics[{index}].semantic_parent_id",
        )
        if parent_id in parent_ids:
            raise ValueError("validation parent metrics contain a duplicate parent")
        parent_ids.add(parent_id)
        count = _strict_int(
            row["target_count"],
            field=f"parent_metrics[{index}].target_count",
            minimum=1,
        )
        _canonical_nonnegative_fp32_hex(
            row["mean_masked_cell_ce_hex"],
            field=f"parent_metrics[{index}].mean_masked_cell_ce_hex",
        )
        target_count += count
        normalized.append(dict(row))
    return normalized, target_count


def validate_validation_metric_row(row: object) -> dict[str, Any]:
    payload = _exact_fields(
        row, _VALIDATION_METRIC_FIELDS, label="M04a validation metric row"
    )
    if payload["schema"] != VALIDATION_METRIC_SCHEMA_VERSION:
        raise ValueError("unsupported validation metric schema")
    pass_index = _strict_int(
        payload["validation_pass_index"], field="validation_pass_index", minimum=1
    )
    if pass_index > VALIDATION_PASS_COUNT:
        raise ValueError(
            f"validation_pass_index must be in 1..{VALIDATION_PASS_COUNT}"
        )
    optimizer_step = _strict_int(
        payload["optimizer_step"], field="optimizer_step", minimum=1
    )
    if optimizer_step != pass_index * VALIDATION_INTERVAL:
        raise ValueError("validation metric optimizer step does not match pass index")
    for field in (
        "validation_episode_outer_manifest_sha256",
        "validation_episode_manifest_sha256",
        "validation_event_ids_sha256",
        "checkpoint_sha256",
    ):
        _sha256(payload[field], field=field)
    episode_count = _strict_int(
        payload["validation_episode_count"],
        field="validation_episode_count",
        minimum=1,
    )
    if payload["aggregation_semantics"] != VALIDATION_AGGREGATION_SEMANTICS:
        raise ValueError("validation aggregation semantics drifted")
    episode_metrics = _validate_validation_episode_metrics(payload["episode_metrics"])
    if episode_count != len(episode_metrics):
        raise ValueError("validation episode count does not close from episode metrics")
    event_ids = [row["ledger_event_id"] for row in episode_metrics]
    if payload["validation_event_ids_sha256"] != _canonical_sha256(event_ids):
        raise ValueError("validation event hash does not close from episode metrics")
    parents, target_count = _validate_validation_parent_metrics(
        payload["parent_metrics"]
    )
    replayed_parents, replayed_target_count, replayed_overall = (
        _replay_validation_hierarchy(episode_metrics)
    )
    if parents != replayed_parents:
        raise ValueError("validation parent metrics do not replay from episode metrics")
    semantic_parent_count = _strict_int(
        payload["semantic_parent_count"], field="semantic_parent_count", minimum=1
    )
    if semantic_parent_count != len(replayed_parents):
        raise ValueError("validation semantic-parent count does not close")
    target_group_count = _strict_int(
        payload["target_group_count"], field="target_group_count", minimum=1
    )
    if target_count != replayed_target_count or target_group_count != target_count:
        raise ValueError("validation target-group count does not close")
    reported_parent_grouped = _canonical_nonnegative_fp32_hex(
        payload["parent_grouped_masked_cell_ce_hex"],
        field="parent_grouped_masked_cell_ce_hex",
    )
    if reported_parent_grouped != replayed_overall:
        raise ValueError(
            "parent-grouped CE does not replay from sequential FP32 episode metrics"
        )
    _sha256(payload["metric_id"], field="metric_id")
    expected_id = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "metric_id"}
    )
    if payload["metric_id"] != expected_id:
        raise ValueError("validation metric_id mismatch")
    return dict(payload)


def make_validation_metric_row(
    *,
    validation_pass_index: int,
    validation_episode_outer_manifest_sha256: str,
    validation_episode_manifest_sha256: str,
    episode_metrics: Sequence[Mapping[str, Any]],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    pass_index = _strict_int(
        validation_pass_index, field="validation_pass_index", minimum=1
    )
    normalized_episodes = _validate_validation_episode_metrics(episode_metrics)
    event_ids = [row["ledger_event_id"] for row in normalized_episodes]
    normalized_parents, target_count, parent_grouped_ce = _replay_validation_hierarchy(
        normalized_episodes
    )
    row: dict[str, Any] = {
        "schema": VALIDATION_METRIC_SCHEMA_VERSION,
        "validation_pass_index": pass_index,
        "optimizer_step": pass_index * VALIDATION_INTERVAL,
        "validation_episode_outer_manifest_sha256": _sha256(
            validation_episode_outer_manifest_sha256,
            field="validation_episode_outer_manifest_sha256",
        ),
        "validation_episode_manifest_sha256": _sha256(
            validation_episode_manifest_sha256,
            field="validation_episode_manifest_sha256",
        ),
        "validation_episode_count": len(normalized_episodes),
        "validation_event_ids_sha256": _canonical_sha256(event_ids),
        "episode_metrics": normalized_episodes,
        "checkpoint_sha256": _sha256(checkpoint_sha256, field="checkpoint_sha256"),
        "aggregation_semantics": VALIDATION_AGGREGATION_SEMANTICS,
        "semantic_parent_count": len(normalized_parents),
        "target_group_count": target_count,
        "parent_metrics": normalized_parents,
        "parent_grouped_masked_cell_ce_hex": parent_grouped_ce.hex(),
    }
    row["metric_id"] = _canonical_sha256(row)
    return validate_validation_metric_row(row)


def validate_checkpoint_manifest_row(row: object) -> dict[str, Any]:
    payload = _exact_fields(
        row, _CHECKPOINT_MANIFEST_FIELDS, label="M04a checkpoint manifest row"
    )
    if payload["schema"] != CHECKPOINT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint manifest schema")
    if payload["checkpoint_schema"] != "afts-grid-cmlm-checkpoint/v0.1":
        raise ValueError("checkpoint payload schema drifted")
    checkpoint_index = _strict_int(
        payload["checkpoint_index"], field="checkpoint_index", minimum=1
    )
    if checkpoint_index > VALIDATION_PASS_COUNT:
        raise ValueError(f"checkpoint_index must be in 1..{VALIDATION_PASS_COUNT}")
    optimizer_step = _strict_int(
        payload["optimizer_step"], field="optimizer_step", minimum=1
    )
    if optimizer_step != checkpoint_index * VALIDATION_INTERVAL:
        raise ValueError("checkpoint optimizer step does not match index")
    for field in (
        "checkpoint_sha256",
        "checkpoint_event_id",
        "validation_metric_id",
        "training_config_sha256",
        "manifest_id",
    ):
        _sha256(payload[field], field=field)
    _strict_int(payload["checkpoint_bytes"], field="checkpoint_bytes", minimum=1)
    if payload["training_config_sha256"] != _train_contract().training_config_sha256():
        raise ValueError("checkpoint manifest training config drifted")
    expected_id = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_id"}
    )
    if payload["manifest_id"] != expected_id:
        raise ValueError("checkpoint manifest_id mismatch")
    return dict(payload)


def make_checkpoint_manifest_row(
    *,
    checkpoint_index: int,
    checkpoint_sha256: str,
    checkpoint_bytes: int,
    checkpoint_event_id: str,
    validation_metric_id: str,
) -> dict[str, Any]:
    index = _strict_int(checkpoint_index, field="checkpoint_index", minimum=1)
    row: dict[str, Any] = {
        "schema": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "checkpoint_schema": "afts-grid-cmlm-checkpoint/v0.1",
        "checkpoint_index": index,
        "optimizer_step": index * VALIDATION_INTERVAL,
        "checkpoint_sha256": _sha256(checkpoint_sha256, field="checkpoint_sha256"),
        "checkpoint_bytes": _strict_int(
            checkpoint_bytes, field="checkpoint_bytes", minimum=1
        ),
        "checkpoint_event_id": _sha256(
            checkpoint_event_id, field="checkpoint_event_id"
        ),
        "validation_metric_id": _sha256(
            validation_metric_id, field="validation_metric_id"
        ),
        "training_config_sha256": _train_contract().training_config_sha256(),
    }
    row["manifest_id"] = _canonical_sha256(row)
    return validate_checkpoint_manifest_row(row)


def validate_selected_checkpoint_manifest(row: object) -> dict[str, Any]:
    payload = _exact_fields(
        row,
        _SELECTED_CHECKPOINT_FIELDS,
        label="M04a selected-checkpoint manifest",
    )
    if payload["schema"] != SELECTED_CHECKPOINT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported selected-checkpoint schema")
    if payload["selection_rule"] != CHECKPOINT_SELECTION_RULE:
        raise ValueError("selected-checkpoint rule drifted")
    candidate_count = _strict_int(
        payload["candidate_count"], field="candidate_count", minimum=1
    )
    if candidate_count != VALIDATION_PASS_COUNT:
        raise ValueError(
            f"selection must consider all {VALIDATION_PASS_COUNT} checkpoints"
        )
    index = _strict_int(
        payload["selected_checkpoint_index"],
        field="selected_checkpoint_index",
        minimum=1,
    )
    optimizer_step = _strict_int(
        payload["optimizer_step"], field="optimizer_step", minimum=1
    )
    if index > VALIDATION_PASS_COUNT or optimizer_step != index * VALIDATION_INTERVAL:
        raise ValueError("selected checkpoint coordinates drifted")
    _canonical_nonnegative_fp32_hex(
        payload["parent_grouped_masked_cell_ce_hex"],
        field="parent_grouped_masked_cell_ce_hex",
    )
    for field in _SELECTED_CHECKPOINT_FIELDS - {
        "schema",
        "selection_rule",
        "candidate_count",
        "selected_checkpoint_index",
        "optimizer_step",
        "parent_grouped_masked_cell_ce_hex",
        "checkpoint_bytes",
        "selection_completed_perf_counter_ns",
        "budget_status",
    }:
        _sha256(payload[field], field=field)
    _strict_int(payload["checkpoint_bytes"], field="checkpoint_bytes", minimum=1)
    _strict_int(
        payload["selection_completed_perf_counter_ns"],
        field="selection_completed_perf_counter_ns",
        minimum=1,
    )
    if payload["training_config_sha256"] != _train_contract().training_config_sha256():
        raise ValueError("selected checkpoint training config drifted")
    if payload["budget_status"] != "WITHIN_BUDGET":
        raise ValueError("selected checkpoint cannot be published over budget")
    expected_id = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "selection_id"}
    )
    if payload["selection_id"] != expected_id:
        raise ValueError("selected-checkpoint selection_id mismatch")
    return dict(payload)


def make_selected_checkpoint_manifest(
    *,
    checkpoint_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    checkpoint_manifests_sha256: str,
    validation_metrics_sha256: str,
    training_cost_ledger_sha256: str,
    validation_episode_outer_manifest_sha256: str,
    validation_episode_manifest_sha256: str,
    lock_interval_id: str,
    selection_completed_perf_counter_ns: int,
) -> dict[str, Any]:
    checkpoints = tuple(
        validate_checkpoint_manifest_row(row) for row in checkpoint_rows
    )
    metrics = tuple(validate_validation_metric_row(row) for row in validation_rows)
    if (
        len(checkpoints) != VALIDATION_PASS_COUNT
        or len(metrics) != VALIDATION_PASS_COUNT
    ):
        raise ValueError(
            "checkpoint selection requires exactly "
            f"{VALIDATION_PASS_COUNT} candidates"
        )
    expected_indices = tuple(range(1, VALIDATION_PASS_COUNT + 1))
    if (
        tuple(row["checkpoint_index"] for row in checkpoints) != expected_indices
        or tuple(row["validation_pass_index"] for row in metrics) != expected_indices
    ):
        raise ValueError(
            "checkpoint selection candidates must be ordered "
            f"1..{VALIDATION_PASS_COUNT}"
        )
    chosen = min(
        metrics,
        key=lambda metric: (
            _canonical_nonnegative_fp32_hex(
                metric["parent_grouped_masked_cell_ce_hex"],
                field="parent_grouped_masked_cell_ce_hex",
            ),
            metric["optimizer_step"],
        ),
    )
    index = chosen["validation_pass_index"]
    checkpoint = checkpoints[index - 1]
    if checkpoint["validation_metric_id"] != chosen["metric_id"]:
        raise ValueError("selected checkpoint candidate does not bind its metric")
    row: dict[str, Any] = {
        "schema": SELECTED_CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "selection_rule": CHECKPOINT_SELECTION_RULE,
        "candidate_count": VALIDATION_PASS_COUNT,
        "selected_checkpoint_index": index,
        "optimizer_step": chosen["optimizer_step"],
        "parent_grouped_masked_cell_ce_hex": chosen[
            "parent_grouped_masked_cell_ce_hex"
        ],
        "validation_metric_id": chosen["metric_id"],
        "checkpoint_manifest_id": checkpoint["manifest_id"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "checkpoint_bytes": checkpoint["checkpoint_bytes"],
        "checkpoint_manifests_sha256": _sha256(
            checkpoint_manifests_sha256, field="checkpoint_manifests_sha256"
        ),
        "validation_metrics_sha256": _sha256(
            validation_metrics_sha256, field="validation_metrics_sha256"
        ),
        "training_cost_ledger_sha256": _sha256(
            training_cost_ledger_sha256, field="training_cost_ledger_sha256"
        ),
        "validation_episode_outer_manifest_sha256": _sha256(
            validation_episode_outer_manifest_sha256,
            field="validation_episode_outer_manifest_sha256",
        ),
        "validation_episode_manifest_sha256": _sha256(
            validation_episode_manifest_sha256,
            field="validation_episode_manifest_sha256",
        ),
        "training_config_sha256": _train_contract().training_config_sha256(),
        "lock_interval_id": _sha256(lock_interval_id, field="lock_interval_id"),
        "selection_completed_perf_counter_ns": _strict_int(
            selection_completed_perf_counter_ns,
            field="selection_completed_perf_counter_ns",
            minimum=1,
        ),
        "budget_status": "WITHIN_BUDGET",
    }
    row["selection_id"] = _canonical_sha256(row)
    return validate_selected_checkpoint_manifest(row)


def validate_training_checkpoint_artifacts(
    checkpoint_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    selected_manifest: object,
    *,
    training_ledger_rows: Sequence[Mapping[str, Any]],
    checkpoint_manifests_sha256: str,
    validation_metrics_sha256: str,
    training_cost_ledger_sha256: str,
    validation_episode_outer_manifest_sha256: str,
    validation_episode_manifest_sha256: str,
    validation_episode_rows: Sequence[Mapping[str, Any]],
    selected_checkpoint_sha256: str,
    selected_checkpoint_bytes: int,
    lock_interval_id: str,
    lock_selection_completed_perf_counter_ns: int,
) -> dict[str, Any]:
    """Close every checkpoint row and metric under the frozen min/tie rule."""

    if isinstance(checkpoint_rows, (str, bytes)) or not isinstance(
        checkpoint_rows, Sequence
    ):
        raise TypeError("checkpoint_rows must be an ordered sequence")
    if isinstance(validation_rows, (str, bytes)) or not isinstance(
        validation_rows, Sequence
    ):
        raise TypeError("validation_rows must be an ordered sequence")
    checkpoints = tuple(
        validate_checkpoint_manifest_row(row) for row in checkpoint_rows
    )
    metrics = tuple(validate_validation_metric_row(row) for row in validation_rows)
    expected_indices = tuple(range(1, VALIDATION_PASS_COUNT + 1))
    if tuple(row["checkpoint_index"] for row in checkpoints) != expected_indices:
        raise ValueError(
            "checkpoint manifests must be literal ordered rows "
            f"1..{VALIDATION_PASS_COUNT}"
        )
    if tuple(row["validation_pass_index"] for row in metrics) != expected_indices:
        raise ValueError(
            "validation metrics must be literal ordered rows "
            f"1..{VALIDATION_PASS_COUNT}"
        )

    ledger = tuple(
        validate_training_cost_ledger_row(row) for row in training_ledger_rows
    )
    checkpoint_events = tuple(
        row for row in ledger if row["phase"] == "checkpoint_operation"
    )
    validation_events = tuple(
        row for row in ledger if row["phase"] == "validation_episode"
    )
    if tuple(row["checkpoint_index"] for row in checkpoint_events) != expected_indices:
        raise ValueError(
            "checkpoint manifests do not have "
            f"{VALIDATION_PASS_COUNT} ordered ledger operations"
        )
    if (
        isinstance(validation_episode_rows, (str, bytes))
        or not isinstance(validation_episode_rows, Sequence)
        or not validation_episode_rows
    ):
        raise TypeError("validation_episode_rows must be a non-empty ordered sequence")
    manifest_rows = tuple(validation_episode_rows)
    episodes_per_pass = len(manifest_rows)
    expected_validation_coordinates = tuple(
        (pass_index, episode_index)
        for pass_index in range(1, VALIDATION_PASS_COUNT + 1)
        for episode_index in range(episodes_per_pass)
    )
    actual_validation_coordinates = tuple(
        (row["validation_pass_index"], row["episode_index"])
        for row in validation_events
    )
    if actual_validation_coordinates != expected_validation_coordinates:
        raise ValueError(
            "validation ledger must contain exactly "
            f"{VALIDATION_PASS_COUNT} ordered passes of literal episodes"
        )
    validation_manifest_sha = _sha256(
        validation_episode_manifest_sha256,
        field="validation_episode_manifest_sha256",
    )
    validation_outer_sha = _sha256(
        validation_episode_outer_manifest_sha256,
        field="validation_episode_outer_manifest_sha256",
    )
    for index, (checkpoint, metric, event) in enumerate(
        zip(checkpoints, metrics, checkpoint_events, strict=True), start=1
    ):
        if (
            checkpoint["optimizer_step"] != metric["optimizer_step"]
            or checkpoint["checkpoint_sha256"] != metric["checkpoint_sha256"]
            or checkpoint["validation_metric_id"] != metric["metric_id"]
            or checkpoint["checkpoint_event_id"] != event["event_id"]
            or checkpoint["checkpoint_bytes"] != event["checkpoint_io_bytes"]
        ):
            raise ValueError("checkpoint/metric/ledger cross-reference mismatch")
        pass_event_rows = [
            row for row in validation_events if row["validation_pass_index"] == index
        ]
        pass_events = [row["event_id"] for row in pass_event_rows]
        if (
            metric["validation_episode_count"] != episodes_per_pass
            or len(pass_events) != episodes_per_pass
            or metric["validation_event_ids_sha256"] != _canonical_sha256(pass_events)
            or metric["validation_episode_manifest_sha256"] != validation_manifest_sha
            or metric["validation_episode_outer_manifest_sha256"]
            != validation_outer_sha
        ):
            raise ValueError(
                "validation metric does not bind its literal episode events"
            )
        episode_metrics = metric["episode_metrics"]
        for ordinal, (episode_metric, manifest_row, ledger_event) in enumerate(
            zip(episode_metrics, manifest_rows, pass_event_rows, strict=True)
        ):
            manifest_ordinal = _strict_int(
                manifest_row.get("row_ordinal"),
                field=f"validation_episode_rows[{ordinal}].row_ordinal",
            )
            masked_indices = manifest_row.get("masked_linear_indices")
            if not isinstance(masked_indices, list) or not masked_indices:
                raise ValueError("validation manifest row has no literal fixed mask")
            for masked_index, value in enumerate(masked_indices):
                _strict_int(
                    value,
                    field=(
                        f"validation_episode_rows[{ordinal}]."
                        f"masked_linear_indices[{masked_index}]"
                    ),
                )
            if (
                manifest_ordinal != ordinal
                or episode_metric["row_ordinal"] != ordinal
                or episode_metric["ledger_event_id"] != ledger_event["event_id"]
                or ledger_event["episode_index"] != ordinal
                or episode_metric["episode_sha256"]
                != manifest_row.get("episode_sha256")
                or episode_metric["semantic_parent_id"]
                != manifest_row.get("semantic_parent_id")
                or episode_metric["target_group_id"]
                != manifest_row.get("target_group_id")
                or episode_metric["masked_token_predictions"] != len(masked_indices)
                or ledger_event["masked_token_predictions"] != len(masked_indices)
            ):
                raise ValueError(
                    "validation episode metric differs from manifest row or ledger event"
                )

    selected = validate_selected_checkpoint_manifest(selected_manifest)
    expected_selected = make_selected_checkpoint_manifest(
        checkpoint_rows=checkpoints,
        validation_rows=metrics,
        checkpoint_manifests_sha256=checkpoint_manifests_sha256,
        validation_metrics_sha256=validation_metrics_sha256,
        training_cost_ledger_sha256=training_cost_ledger_sha256,
        validation_episode_outer_manifest_sha256=validation_outer_sha,
        validation_episode_manifest_sha256=validation_manifest_sha,
        lock_interval_id=lock_interval_id,
        selection_completed_perf_counter_ns=(lock_selection_completed_perf_counter_ns),
    )
    if selected != expected_selected:
        raise ValueError(
            "selected checkpoint is not the minimum parent-grouped CE/tie-earlier row"
        )
    if selected["checkpoint_sha256"] != _sha256(
        selected_checkpoint_sha256, field="selected_checkpoint_sha256"
    ) or selected["checkpoint_bytes"] != _strict_int(
        selected_checkpoint_bytes, field="selected_checkpoint_bytes", minimum=1
    ):
        raise ValueError("selected-checkpoint manifest does not bind selected weights")
    return selected


_ARTIFACT_KIND_TO_SCHEMA = {
    "training": TRAINING_ARTIFACT_MANIFEST_SCHEMA_VERSION,
    "pool": POOL_ARTIFACT_MANIFEST_SCHEMA_VERSION,
    "evaluation": EVALUATION_ARTIFACT_MANIFEST_SCHEMA_VERSION,
}

_ARTIFACT_KIND_TO_MANIFEST_NAME = {
    "training": "training_artifact_manifest.json",
    "pool": "pool_artifact_manifest.json",
    "evaluation": "evaluation_artifact_manifest.json",
}

TRAINING_LINEAGE_FIELDS = frozenset(
    {
        "reviewed_runtime_source_sha256",
        "reviewed_test_snapshot_sha256",
        "remote_launcher_sha256",
        "launch_plan_sha256",
        "lock_handshake_artifact_sha256",
        "preflight_overhead_cost_report_sha256",
        "preflight_diagnostic_checkpoint_sha256",
        "frozen_contract_sha256",
        "sanitized_shard_manifest_sha256",
        "data_split_manifest_sha256",
        "validation_episode_outer_manifest_sha256",
        "validation_episode_manifest_sha256",
        "validation_episode_summary_sha256",
        "validation_episode_summary_id",
        "environment_manifest_sha256",
        "model_config_sha256",
        "training_cost_ledger_sha256",
        "selected_checkpoint_manifest_sha256",
        "checkpoint_sha256",
        "python_runtime_lock_sha256",
        "python_runtime_lock_id",
        "python_runtime_identity_sha256",
    }
)

POOL_LINEAGE_FIELDS = frozenset(
    {
        "training_artifact_id",
        "training_artifact_manifest_sha256",
        "selected_checkpoint_manifest_sha256",
        "checkpoint_sha256",
        "blind_artifact_manifest_sha256",
        "blind_tasks_sha256",
        "blind_shape_sidecar_id",
        "source_snapshot_sha256",
        "environment_manifest_sha256",
        "sampler_config_sha256",
        "dsl_pool_content_id",
        "python_runtime_lock_sha256",
        "python_runtime_lock_id",
        "python_runtime_identity_sha256",
    }
)

EVALUATION_LINEAGE_FIELDS = frozenset(
    {
        "pool_artifact_id",
        "pool_artifact_manifest_sha256",
        "oracle_artifact_manifest_sha256",
        "dsl_pool_content_id",
        "python_runtime_lock_sha256",
        "python_runtime_lock_id",
        "python_runtime_identity_sha256",
    }
)

_LINEAGE_FIELDS = {
    "training": TRAINING_LINEAGE_FIELDS,
    "pool": POOL_LINEAGE_FIELDS,
    "evaluation": EVALUATION_LINEAGE_FIELDS,
}

_SEMANTIC_ARTIFACT_FIELDS = {
    "schema",
    "artifact_kind",
    "semantic_id",
    "lineage",
    "files",
    "closure",
    "timing",
}

TRAINING_ARTIFACT_CLOSURE_FIELDS = frozenset(
    {
        "status",
        "preflight_optimizer_updates",
        "preflight_microbatches",
        "preflight_encoder_forward_calls",
        "preflight_decoder_forward_calls",
        "preflight_backward_calls",
        "preflight_inference_lanes",
        "preflight_inference_steps",
        "preflight_inference_encoder_batch_calls",
        "preflight_inference_decoder_batch_calls",
        "preflight_inference_sample_equivalent_forward_calls",
        "primary_optimizer_updates",
        "primary_microbatches",
        "primary_arc2_episodes",
        "primary_rearc_episodes",
        "primary_encoder_forward_calls",
        "primary_decoder_forward_calls",
        "primary_backward_calls",
        "validation_passes",
        "validation_episode_calls",
        "validation_encoder_forward_calls",
        "validation_decoder_forward_calls",
        "checkpoint_writes",
        "resume_segments",
        "masked_token_predictions",
        "selected_checkpoint_complete",
        "budget_status",
    }
)

POOL_ARTIFACT_CLOSURE_FIELDS = frozenset(
    {
        "status",
        "task_count",
        "test_pair_count",
        "candidate_count_zero_task_count",
        "shape_supported_test_pair_count",
        "no_shape_test_pair_count",
        "raw_lanes",
        "lane_trace_rows",
        "candidate_rows",
        "encoder_batch_calls",
        "decoder_batch_calls",
        "total_actual_batch_calls",
        "sample_equivalent_forward_calls",
        "masked_token_predictions",
        "format_valid",
        "format_invalid",
        "unique_outputs",
        "duplicate_outputs",
        "pair_cost_rows",
        "replay_verified_trace_count",
        "replay_verified_decoder_call_count",
        "replay_verified_prediction_count",
        "bf16_cache_pair_count",
    }
)

TRAINING_ARTIFACT_TIMING_FIELDS = frozenset(
    {
        "preflight_wall_time_ns",
        "primary_training_wall_time_ns",
        "validation_wall_time_ns",
        "checkpoint_wall_time_ns",
        "resume_setup_wall_time_ns",
        "gpu_lock_wall_time_ns",
        "budget_limit_ns",
        "budget_remaining_ns",
    }
)

_TRAINING_LOCK_INTERVAL_FIELDS = {
    "schema",
    "interval_id",
    "clock",
    "segments",
    "lock_segment_count",
    "gpu_lock_wall_time_ns",
    "budget_limit_ns",
    "budget_remaining_ns",
    "budget_status",
}

_TRAINING_LOCK_SEGMENT_FIELDS = {
    "segment_index",
    "start_endpoint",
    "end_endpoint",
    "start_perf_counter_ns",
    "end_perf_counter_ns",
    "wall_time_ns",
}


def validate_training_lock_interval(
    interval: object, *, expected_resume_segments: int | None = None
) -> dict[str, Any]:
    """Validate measured lock endpoints, including fresh and resumed segments."""

    payload = _exact_fields(
        interval,
        _TRAINING_LOCK_INTERVAL_FIELDS,
        label="M04a training lock interval",
    )
    if payload["schema"] != TRAINING_LOCK_INTERVAL_SCHEMA_VERSION:
        raise ValueError("unsupported training lock-interval schema")
    if payload["clock"] != "time.perf_counter_ns":
        raise ValueError("training lock endpoints must use time.perf_counter_ns")
    segments = payload["segments"]
    if (
        isinstance(segments, (str, bytes))
        or not isinstance(segments, list)
        or not segments
    ):
        raise TypeError("training lock interval must contain a non-empty segment list")
    normalized_segments: list[dict[str, Any]] = []
    total = 0
    for offset, raw in enumerate(segments, start=1):
        segment = _exact_fields(
            raw,
            _TRAINING_LOCK_SEGMENT_FIELDS,
            label=f"training lock segment[{offset - 1}]",
        )
        if segment["segment_index"] != offset:
            raise ValueError(
                "training lock segment indices must be consecutive one-based"
            )
        expected_start = (
            "preflight_lock_handshake" if offset == 1 else "resume_lock_handshake"
        )
        expected_end = (
            "final_checkpoint_selection" if offset == len(segments) else "lock_release"
        )
        if (
            segment["start_endpoint"] != expected_start
            or segment["end_endpoint"] != expected_end
        ):
            raise ValueError(
                "training lock endpoint roles do not cover handshake-to-selection"
            )
        start = _strict_int(
            segment["start_perf_counter_ns"],
            field=f"segments[{offset - 1}].start_perf_counter_ns",
        )
        end = _strict_int(
            segment["end_perf_counter_ns"],
            field=f"segments[{offset - 1}].end_perf_counter_ns",
        )
        wall = _strict_int(
            segment["wall_time_ns"],
            field=f"segments[{offset - 1}].wall_time_ns",
            minimum=1,
        )
        if end <= start or wall != end - start:
            raise ValueError(
                "training lock segment wall time does not close from endpoints"
            )
        total += wall
        normalized_segments.append(dict(segment))
    if payload["lock_segment_count"] != len(normalized_segments):
        raise ValueError("training lock segment count does not close")
    if expected_resume_segments is not None:
        resumes = _strict_int(
            expected_resume_segments, field="expected_resume_segments"
        )
        if len(normalized_segments) != resumes + 1:
            raise ValueError(
                "training lock segments do not close from resume manifests"
            )
    if payload["gpu_lock_wall_time_ns"] != total:
        raise ValueError("GPU-lock wall time does not close from measured endpoints")
    if payload["budget_limit_ns"] != GPU_HOUR_BUDGET_NS:
        raise ValueError("training lock interval does not bind the 24 GPU-hour cap")
    if total > GPU_HOUR_BUDGET_NS:
        raise ValueError("training lock endpoints exceed the 24 GPU-hour cap")
    if payload["budget_remaining_ns"] != GPU_HOUR_BUDGET_NS - total:
        raise ValueError("training lock budget remainder does not close")
    if payload["budget_status"] != "WITHIN_BUDGET":
        raise ValueError("complete training lock interval must be within budget")
    _sha256(payload["interval_id"], field="interval_id")
    expected_id = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "interval_id"}
    )
    if payload["interval_id"] != expected_id:
        raise ValueError("training lock interval_id mismatch")
    normalized = dict(payload)
    normalized["segments"] = normalized_segments
    return normalized


def make_training_lock_interval(
    endpoint_pairs: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Build endpoint evidence from one fresh lock interval plus optional resumes."""

    if (
        isinstance(endpoint_pairs, (str, bytes))
        or not isinstance(endpoint_pairs, Sequence)
        or not endpoint_pairs
    ):
        raise TypeError("endpoint_pairs must be a non-empty ordered sequence")
    segments: list[dict[str, Any]] = []
    for index, pair in enumerate(endpoint_pairs, start=1):
        if type(pair) is not tuple or len(pair) != 2:
            raise TypeError("each endpoint pair must be a (start_ns, end_ns) tuple")
        start = _strict_int(pair[0], field=f"endpoint_pairs[{index - 1}].start")
        end = _strict_int(pair[1], field=f"endpoint_pairs[{index - 1}].end")
        if end <= start:
            raise ValueError("training lock endpoint must advance")
        segments.append(
            {
                "segment_index": index,
                "start_endpoint": (
                    "preflight_lock_handshake"
                    if index == 1
                    else "resume_lock_handshake"
                ),
                "end_endpoint": (
                    "final_checkpoint_selection"
                    if index == len(endpoint_pairs)
                    else "lock_release"
                ),
                "start_perf_counter_ns": start,
                "end_perf_counter_ns": end,
                "wall_time_ns": end - start,
            }
        )
    total = sum(segment["wall_time_ns"] for segment in segments)
    payload: dict[str, Any] = {
        "schema": TRAINING_LOCK_INTERVAL_SCHEMA_VERSION,
        "clock": "time.perf_counter_ns",
        "segments": segments,
        "lock_segment_count": len(segments),
        "gpu_lock_wall_time_ns": total,
        "budget_limit_ns": GPU_HOUR_BUDGET_NS,
        "budget_remaining_ns": GPU_HOUR_BUDGET_NS - total,
        "budget_status": "WITHIN_BUDGET",
    }
    payload["interval_id"] = _canonical_sha256(payload)
    return validate_training_lock_interval(payload)


def _validate_training_lock_timeline_coverage(
    *,
    lock_interval: Mapping[str, Any],
    preflight_started_perf_counter_ns: int,
    phase_timing: Mapping[str, Any],
    preflight_endpoint_wall_time_ns: int,
) -> None:
    """Close the acquisition-to-preflight gap and every measured phase in the lock."""

    interval = validate_training_lock_interval(lock_interval)
    preflight_started = _strict_int(
        preflight_started_perf_counter_ns,
        field="preflight_started_perf_counter_ns",
        minimum=1,
    )
    preflight_endpoint = _strict_int(
        preflight_endpoint_wall_time_ns,
        field="preflight_endpoint_wall_time_ns",
        minimum=1,
    )
    segments = interval["segments"]
    acquisition_started = segments[0]["start_perf_counter_ns"]
    if preflight_started < acquisition_started:
        raise ValueError("preflight starts before the measured GPU-lock interval")
    phase_fields = (
        "preflight_wall_time_ns",
        "primary_training_wall_time_ns",
        "validation_wall_time_ns",
        "checkpoint_wall_time_ns",
        "resume_setup_wall_time_ns",
    )
    phase_total = sum(
        _strict_int(phase_timing.get(field), field=f"phase_timing.{field}")
        for field in phase_fields
    )
    acquisition_to_preflight = preflight_started - acquisition_started
    if acquisition_to_preflight + phase_total > interval["gpu_lock_wall_time_ns"]:
        raise ValueError(
            "GPU-lock endpoints omit the acquisition-to-preflight gap or measured phases"
        )
    if preflight_started + preflight_endpoint > segments[0]["end_perf_counter_ns"]:
        raise ValueError("first GPU-lock segment ends before the preflight endpoint")


POOL_ARTIFACT_TIMING_FIELDS = frozenset(
    {
        "setup_wall_time_ns",
        "sequential_pair_wall_time_ns",
        "finalization_wall_time_ns",
        "pool_wall_time_ns",
        "encoder_gpu_ns",
        "decoder_gpu_ns",
        "total_gpu_ns",
    }
)


def _normalize_payload_file_metadata(files: object) -> list[dict[str, Any]]:
    if isinstance(files, (str, bytes)) or not isinstance(files, Sequence):
        raise TypeError("files must be an ordered sequence of metadata rows")
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(files):
        row = _exact_fields(raw, {"path", "sha256", "bytes"}, label=f"files[{index}]")
        name = _safe_artifact_name(row["path"])
        if name in names:
            raise ValueError(f"duplicate artifact path: {name}")
        names.add(name)
        normalized.append(
            {
                "path": name,
                "sha256": _sha256(row["sha256"], field=f"files[{index}].sha256"),
                "bytes": _strict_int(row["bytes"], field=f"files[{index}].bytes"),
            }
        )
    if not normalized:
        raise ValueError("artifact manifest must bind at least one payload file")
    if [row["path"] for row in normalized] != sorted(names):
        raise ValueError("artifact file metadata must be sorted by path")
    return normalized


def _normalize_timing(timing: object) -> dict[str, int]:
    if not isinstance(timing, dict):
        raise TypeError("timing must be a flat JSON object")
    normalized: dict[str, int] = {}
    for key, value in timing.items():
        _nonempty_string(key, field="timing key")
        normalized[key] = _strict_int(value, field=f"timing.{key}")
    return normalized


def _validate_training_artifact_closure(value: object) -> dict[str, Any]:
    closure = _exact_fields(
        value,
        set(TRAINING_ARTIFACT_CLOSURE_FIELDS),
        label="training artifact closure",
    )
    if closure["status"] != "COMPLETE":
        raise ValueError("training artifact status must be COMPLETE")
    if closure["budget_status"] != "WITHIN_BUDGET":
        raise ValueError("a complete training artifact must be within budget")
    if closure["selected_checkpoint_complete"] is not True:
        raise ValueError("a complete training artifact requires a selected checkpoint")
    fixed = {
        "preflight_optimizer_updates": 100,
        "preflight_microbatches": 1_600,
        "preflight_encoder_forward_calls": 1_600,
        "preflight_decoder_forward_calls": 1_600,
        "preflight_backward_calls": 1_600,
        "preflight_inference_lanes": 8,
        "preflight_inference_steps": 12,
        "preflight_inference_encoder_batch_calls": 1,
        "preflight_inference_decoder_batch_calls": 12,
        "preflight_inference_sample_equivalent_forward_calls": 96,
        "primary_optimizer_updates": OPTIMIZER_UPDATES,
        "primary_microbatches": (
            OPTIMIZER_UPDATES * PREFLIGHT_MICROBATCHES_PER_UPDATE
        ),
        "primary_arc2_episodes": (
            OPTIMIZER_UPDATES * PREFLIGHT_MICROBATCHES_PER_UPDATE // 2
        ),
        "primary_rearc_episodes": (
            OPTIMIZER_UPDATES * PREFLIGHT_MICROBATCHES_PER_UPDATE // 2
        ),
        "primary_encoder_forward_calls": (
            OPTIMIZER_UPDATES * PREFLIGHT_MICROBATCHES_PER_UPDATE
        ),
        "primary_decoder_forward_calls": (
            OPTIMIZER_UPDATES * PREFLIGHT_MICROBATCHES_PER_UPDATE
        ),
        "primary_backward_calls": (
            OPTIMIZER_UPDATES * PREFLIGHT_MICROBATCHES_PER_UPDATE
        ),
        "validation_passes": VALIDATION_PASS_COUNT,
        "checkpoint_writes": VALIDATION_PASS_COUNT,
    }
    for field in TRAINING_ARTIFACT_CLOSURE_FIELDS - {
        "status",
        "budget_status",
        "selected_checkpoint_complete",
    }:
        _strict_int(closure[field], field=f"closure.{field}")
    for field, expected in fixed.items():
        if closure[field] != expected:
            raise ValueError(f"training artifact closure {field} must be {expected}")
    validation_calls = closure["validation_episode_calls"]
    if validation_calls <= 0 or any(
        closure[field] != validation_calls
        for field in (
            "validation_encoder_forward_calls",
            "validation_decoder_forward_calls",
        )
    ):
        raise ValueError(
            "training validation calls must close from validation episodes"
        )
    if closure["masked_token_predictions"] <= 0:
        raise ValueError("training masked-token closure must be positive")
    assert_oracle_free_payload(closure, location="training.closure")
    return dict(closure)


def _validate_pool_artifact_closure(value: object) -> dict[str, Any]:
    closure = _exact_fields(
        value, set(POOL_ARTIFACT_CLOSURE_FIELDS), label="pool artifact closure"
    )
    if closure["status"] != "COMPLETE":
        raise ValueError("pool artifact status must be COMPLETE")
    for field in POOL_ARTIFACT_CLOSURE_FIELDS - {"status"}:
        _strict_int(closure[field], field=f"closure.{field}")
    fixed = {
        "task_count": FIXED_BLIND_TASK_COUNT,
        "test_pair_count": FIXED_BLIND_TEST_PAIR_COUNT,
        "candidate_count_zero_task_count": FIXED_ZERO_CANDIDATE_TASK_COUNT,
        "shape_supported_test_pair_count": 16,
        "no_shape_test_pair_count": 5,
        "pair_cost_rows": FIXED_BLIND_TEST_PAIR_COUNT,
    }
    for field, expected in fixed.items():
        if closure[field] != expected:
            raise ValueError(f"pool artifact closure {field} must be {expected}")
    raw_lanes = closure["raw_lanes"]
    expected_raw = closure["shape_supported_test_pair_count"] * 64
    if raw_lanes != expected_raw:
        raise ValueError("pool raw lanes do not close from shape-supported pairs")
    if (
        closure["lane_trace_rows"] != raw_lanes
        or closure["candidate_rows"] != raw_lanes
    ):
        raise ValueError("pool trace/candidate rows do not close to raw lanes")
    if closure["sample_equivalent_forward_calls"] != raw_lanes * _contract_value(
        "DENOISING_STEPS"
    ):
        raise ValueError("pool sample-equivalent calls do not close")
    if closure["encoder_batch_calls"] != closure["shape_supported_test_pair_count"]:
        raise ValueError("pool encoder calls do not close from supported pairs")
    if closure["total_actual_batch_calls"] != (
        closure["encoder_batch_calls"] + closure["decoder_batch_calls"]
    ):
        raise ValueError("pool actual batch calls do not close")
    minimum_decoder = closure["shape_supported_test_pair_count"] * 96
    maximum_decoder = closure["shape_supported_test_pair_count"] * 108
    if not minimum_decoder <= closure["decoder_batch_calls"] <= maximum_decoder:
        raise ValueError("pool decoder calls are outside the frozen allocation bounds")
    if closure["format_valid"] + closure["format_invalid"] != raw_lanes:
        raise ValueError("pool format counts do not close")
    if (
        closure["unique_outputs"] + closure["duplicate_outputs"]
        != closure["format_valid"]
    ):
        raise ValueError("pool deduplication counts do not close")
    if closure["replay_verified_trace_count"] != raw_lanes:
        raise ValueError("complete pool must replay-verify every trace")
    if closure["replay_verified_decoder_call_count"] != closure["decoder_batch_calls"]:
        raise ValueError("complete pool must replay-verify every decoder call")
    if (
        closure["replay_verified_prediction_count"]
        != closure["masked_token_predictions"]
    ):
        raise ValueError("complete pool must replay-verify every prediction")
    if closure["bf16_cache_pair_count"] != closure["shape_supported_test_pair_count"]:
        raise ValueError("complete pool must verify one BF16 cache per supported pair")
    assert_oracle_free_payload(closure, location="pool.closure")
    return dict(closure)


def _validate_artifact_timing(kind: str, value: object) -> dict[str, int]:
    timing = _normalize_timing(value)
    required = {
        "training": TRAINING_ARTIFACT_TIMING_FIELDS,
        "pool": POOL_ARTIFACT_TIMING_FIELDS,
    }.get(kind)
    if required is None:
        return timing
    if set(timing) != required:
        raise ValueError(f"{kind} timing fields must be {sorted(required)}")
    if kind == "training":
        phase_total = sum(
            timing[field]
            for field in (
                "preflight_wall_time_ns",
                "primary_training_wall_time_ns",
                "validation_wall_time_ns",
                "checkpoint_wall_time_ns",
                "resume_setup_wall_time_ns",
            )
        )
        lock_total = timing["gpu_lock_wall_time_ns"]
        if phase_total > lock_total:
            raise ValueError(
                "training phase rows exceed the measured GPU-lock endpoint"
            )
        if timing["budget_limit_ns"] != GPU_HOUR_BUDGET_NS:
            raise ValueError("training timing must bind the frozen 24 GPU-hour cap")
        if lock_total > GPU_HOUR_BUDGET_NS:
            raise ValueError("complete training artifact exceeds the 24 GPU-hour cap")
        if timing["budget_remaining_ns"] != GPU_HOUR_BUDGET_NS - lock_total:
            raise ValueError("training budget remaining time does not close")
    else:
        if timing["pool_wall_time_ns"] != (
            timing["setup_wall_time_ns"]
            + timing["sequential_pair_wall_time_ns"]
            + timing["finalization_wall_time_ns"]
        ):
            raise ValueError("pool endpoint wall time does not close")
        if (
            timing["total_gpu_ns"]
            != timing["encoder_gpu_ns"] + timing["decoder_gpu_ns"]
        ):
            raise ValueError("pool GPU time does not close")
    return timing


def _validate_json_value(value: object, *, label: str) -> None:
    try:
        _canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite canonical JSON") from exc


def validate_optimizer_schedule_state(value: object) -> dict[str, Any]:
    """Bind final optimizer coordinates to the frozen production-data closure."""

    payload = _exact_fields(
        value,
        {
            "schema",
            "optimizer_step",
            "learning_rate_hex",
            "checkpoint_count",
            "production_data_closure_id",
        },
        label="M04a optimizer/schedule state",
    )
    expected = {
        "schema": OPTIMIZER_SCHEDULE_STATE_SCHEMA_VERSION,
        "optimizer_step": OPTIMIZER_UPDATES,
        "learning_rate_hex": _train_contract()
        .learning_rate_for_update(OPTIMIZER_UPDATES)
        .hex(),
        "checkpoint_count": VALIDATION_PASS_COUNT,
        "production_data_closure_id": PRODUCTION_DATA_CLOSURE_ID,
    }
    if payload != expected:
        raise ValueError(
            "optimizer/schedule state differs from the frozen campaign/data closure"
        )
    return dict(payload)


def _semantic_artifact_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: manifest[key]
        for key in ("schema", "artifact_kind", "lineage", "files", "closure")
    }


def validate_artifact_manifest(
    manifest: object, *, expected_kind: str | None = None
) -> dict[str, Any]:
    payload = _exact_fields(
        manifest, _SEMANTIC_ARTIFACT_FIELDS, label="M04a semantic artifact manifest"
    )
    kind = payload["artifact_kind"]
    if kind not in _ARTIFACT_KIND_TO_SCHEMA:
        raise ValueError("artifact_kind must be training, pool, or evaluation")
    if expected_kind is not None and kind != expected_kind:
        raise ValueError(f"expected {expected_kind} artifact, found {kind}")
    if payload["schema"] != _ARTIFACT_KIND_TO_SCHEMA[kind]:
        raise ValueError("semantic artifact schema does not match artifact_kind")
    lineage = payload["lineage"]
    if not isinstance(lineage, dict) or set(lineage) != _LINEAGE_FIELDS[kind]:
        found = sorted(lineage) if isinstance(lineage, dict) else type(lineage).__name__
        raise ValueError(
            f"{kind} lineage fields must be {sorted(_LINEAGE_FIELDS[kind])}, found {found}"
        )
    for field, value in lineage.items():
        _sha256(value, field=f"lineage.{field}")
    files = _normalize_payload_file_metadata(payload["files"])
    required_files = {
        "training": TRAINING_ARTIFACT_FILES,
        "pool": POOL_ARTIFACT_FILES,
    }.get(kind)
    if required_files is not None and {row["path"] for row in files} != required_files:
        found = {row["path"] for row in files}
        raise ValueError(
            f"{kind} artifact file set mismatch: "
            f"missing={sorted(required_files - found)}, extra={sorted(found - required_files)}"
        )
    if kind == "evaluation" and "pool_artifact_manifest.json" not in {
        row["path"] for row in files
    }:
        raise ValueError("evaluation artifact must embed pool_artifact_manifest.json")
    if not isinstance(payload["closure"], dict):
        raise TypeError("closure must be a JSON object")
    _validate_json_value(payload["closure"], label="closure")
    if kind == "training":
        closure = _validate_training_artifact_closure(payload["closure"])
    elif kind == "pool":
        closure = _validate_pool_artifact_closure(payload["closure"])
    else:
        closure = dict(payload["closure"])
    timing = _validate_artifact_timing(kind, payload["timing"])
    if kind in {"training", "pool"}:
        assert_oracle_free_payload(lineage, location=f"{kind}.lineage")
        assert_oracle_free_payload(closure, location=f"{kind}.closure")
        for file_row in files:
            if "oracle" in file_row["path"].lower():
                raise ValueError(f"oracle-named file is forbidden in {kind} bundle")
    normalized = {
        "schema": payload["schema"],
        "artifact_kind": kind,
        "semantic_id": payload["semantic_id"],
        "lineage": dict(lineage),
        "files": files,
        "closure": closure,
        "timing": timing,
    }
    expected_id = _canonical_sha256(_semantic_artifact_payload(normalized))
    if payload["semantic_id"] != expected_id:
        raise ValueError("semantic_id does not match manifest content excluding timing")
    return normalized


def build_artifact_manifest(
    artifact_kind: str,
    *,
    lineage: Mapping[str, str],
    files: Sequence[Mapping[str, Any]],
    closure: Mapping[str, Any],
    timing: Mapping[str, int],
) -> dict[str, Any]:
    if artifact_kind not in _ARTIFACT_KIND_TO_SCHEMA:
        raise ValueError("artifact_kind must be training, pool, or evaluation")
    manifest: dict[str, Any] = {
        "schema": _ARTIFACT_KIND_TO_SCHEMA[artifact_kind],
        "artifact_kind": artifact_kind,
        "lineage": dict(lineage),
        "files": [dict(row) for row in files],
        "closure": dict(closure),
        "timing": dict(timing),
    }
    manifest["semantic_id"] = _canonical_sha256(_semantic_artifact_payload(manifest))
    return validate_artifact_manifest(manifest, expected_kind=artifact_kind)


def build_training_artifact_manifest(**kwargs: Any) -> dict[str, Any]:
    return build_artifact_manifest("training", **kwargs)


def build_pool_artifact_manifest(**kwargs: Any) -> dict[str, Any]:
    return build_artifact_manifest("pool", **kwargs)


def build_evaluation_artifact_manifest(**kwargs: Any) -> dict[str, Any]:
    return build_artifact_manifest("evaluation", **kwargs)


def _copy_file_new(source: Path, destination: Path) -> dict[str, Any]:
    if source.is_symlink() or not source.is_file():
        raise ValueError(
            f"artifact source must be a regular non-symlink file: {source}"
        )
    before = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    rows = 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        while True:
            chunk = reader.read(1024 * 1024)
            if not chunk:
                break
            writer.write(chunk)
            digest.update(chunk)
            size += len(chunk)
            rows += chunk.count(b"\n")
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"artifact source changed while copying: {source}")
    return {
        "sha256": digest.hexdigest(),
        "bytes": size,
        "rows": rows if destination.name.endswith(".jsonl") else None,
    }


def _scan_json_file_for_oracle(path: Path) -> None:
    if path.name.endswith(".json"):
        assert_oracle_free_payload(_read_json(path), location=path.name)
    elif path.name.endswith(".jsonl"):
        assert_oracle_free_payload(_read_jsonl(path), location=path.name)


def _scan_json_bytes_for_oracle(name: str, content: bytes) -> None:
    if name.endswith(".json"):
        assert_oracle_free_payload(
            _parse_json_bytes(content, label=name), location=name
        )
    elif name.endswith(".jsonl"):
        assert_oracle_free_payload(
            _parse_jsonl_bytes(content, label=name), location=name
        )


@dataclass(frozen=True, slots=True)
class M04aArtifactBundle:
    root: Path
    manifest: dict[str, Any]
    outer_manifest: dict[str, Any]


def publish_m04a_artifact_bundle(
    output_dir: str | Path,
    *,
    artifact_kind: str,
    lineage: Mapping[str, str],
    artifact_files: Mapping[str, str | Path],
    closure: Mapping[str, Any],
    timing: Mapping[str, int],
) -> M04aArtifactBundle:
    """Chunk-copy files into a sibling staging dir and publish with one rename."""

    if artifact_kind not in _ARTIFACT_KIND_TO_SCHEMA:
        raise ValueError("artifact_kind must be training, pool, or evaluation")
    target = Path(output_dir).expanduser().resolve()
    staging = _new_staging_directory(target)
    semantic_name = _ARTIFACT_KIND_TO_MANIFEST_NAME[artifact_kind]
    try:
        metadata: dict[str, dict[str, Any]] = {}
        for raw_name, raw_source in sorted(artifact_files.items()):
            name = _safe_artifact_name(raw_name)
            if name == semantic_name:
                raise ValueError(f"{semantic_name} is reserved")
            if artifact_kind in {"training", "pool"} and "oracle" in name.lower():
                raise ValueError(
                    f"oracle-named artifact is forbidden in {artifact_kind}"
                )
            requested_source = Path(raw_source).expanduser()
            if requested_source.is_symlink():
                raise ValueError(
                    f"artifact source symlink is forbidden: {requested_source}"
                )
            source = requested_source.resolve()
            destination = staging.joinpath(*PurePosixPath(name).parts)
            metadata[name] = _copy_file_new(source, destination)
            if artifact_kind in {"training", "pool"}:
                _scan_json_file_for_oracle(destination)
        file_rows = [
            {"path": name, "sha256": row["sha256"], "bytes": row["bytes"]}
            for name, row in sorted(metadata.items())
        ]
        semantic = build_artifact_manifest(
            artifact_kind,
            lineage=lineage,
            files=file_rows,
            closure=closure,
            timing=timing,
        )
        metadata[semantic_name] = _write_bytes_new(
            staging / semantic_name, serialize_json(semantic)
        )
        outer = _outer_bundle_manifest(
            artifact_kind=artifact_kind,
            semantic_manifest=semantic_name,
            bundle_id=semantic["semantic_id"],
            artifacts=metadata,
        )
        _write_bytes_new(staging / "artifact_manifest.json", serialize_json(outer))
        read_m04a_artifact_bundle(staging, expected_kind=artifact_kind)
        result = M04aArtifactBundle(
            root=target,
            manifest=semantic,
            outer_manifest=outer,
        )
        staging.rename(target)
    except Exception as exc:
        raise RuntimeError(
            f"M04a artifact staging failed; preserved at {staging}: {exc}"
        ) from exc
    return result


def read_m04a_artifact_bundle(
    directory: str | Path, *, expected_kind: str | None = None
) -> M04aArtifactBundle:
    bundle = read_closed_world_bundle(directory)
    outer = bundle.artifact_manifest
    if outer.get("schema") != FILE_BUNDLE_SCHEMA_VERSION:
        raise ValueError("M04a artifact requires the M04a file-bundle schema")
    kind = outer["artifact_kind"]
    if expected_kind is not None and kind != expected_kind:
        raise ValueError(f"expected {expected_kind} artifact, found {kind}")
    if kind not in _ARTIFACT_KIND_TO_MANIFEST_NAME:
        raise ValueError("unsupported M04a artifact kind")
    semantic_name = _ARTIFACT_KIND_TO_MANIFEST_NAME[kind]
    if outer["semantic_manifest"] != semantic_name:
        raise ValueError("outer manifest points at the wrong semantic manifest")
    semantic = validate_artifact_manifest(
        bundle.read_json(semantic_name), expected_kind=kind
    )
    if outer["bundle_id"] != semantic["semantic_id"]:
        raise ValueError("outer bundle_id does not match semantic_id")
    expected_payload_metadata = {
        row["path"]: {"sha256": row["sha256"], "bytes": row["bytes"]}
        for row in semantic["files"]
    }
    actual_payload_names = set(bundle.artifacts) - {semantic_name}
    if actual_payload_names != set(expected_payload_metadata):
        raise ValueError("semantic manifest file set does not match outer bundle")
    for name, expected in expected_payload_metadata.items():
        actual = bundle.artifacts[name]
        if (
            actual["sha256"] != expected["sha256"]
            or actual["bytes"] != expected["bytes"]
        ):
            raise ValueError(f"semantic file metadata mismatch: {name}")
        if kind in {"training", "pool"}:
            _scan_json_bytes_for_oracle(name, bundle.read_bytes(name))
    if kind == "training":
        _validate_complete_training_bundle(bundle, semantic)
    elif kind == "pool":
        _validate_complete_pool_bundle(bundle, semantic)
    elif kind == "evaluation":
        _validate_complete_evaluation_bundle(bundle, semantic)
    return M04aArtifactBundle(root=bundle.root, manifest=semantic, outer_manifest=outer)


def read_committed_training_artifact_bundle(
    directory: str | Path, *, expected_launch_plan_sha256: str
) -> M04aArtifactBundle:
    """Upgrade structural training evidence using its prelaunch external file SHA."""

    expected = _sha256(expected_launch_plan_sha256, field="expected_launch_plan_sha256")
    artifact = read_m04a_artifact_bundle(directory, expected_kind="training")
    if artifact.manifest["lineage"]["launch_plan_sha256"] != expected:
        raise ValueError(
            "training bundle differs from the external launch-plan SHA-256"
        )
    return artifact


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _values_for_key(value: object, key: str) -> list[object]:
    values: list[object] = []
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            if child_key == key:
                values.append(child_value)
            values.extend(_values_for_key(child_value, key))
    elif isinstance(value, list):
        for child_value in value:
            values.extend(_values_for_key(child_value, key))
    return values


def _validate_embedded_validation_manifest(
    bundle: VerifiedBundle, lineage: Mapping[str, Any]
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Replay the copied external validation outer/summary/JSONL commitment."""

    jsonl_name = "validation_episode_manifest.jsonl"
    summary_name = "validation_episode_manifest_summary.json"
    outer_name = "validation_episode_outer_manifest.json"
    jsonl_bytes = bundle.read_bytes(jsonl_name)
    summary_bytes = bundle.read_bytes(summary_name)
    outer_bytes = bundle.read_bytes(outer_name)
    rows = _parse_jsonl_bytes(jsonl_bytes, label=jsonl_name)
    validation_module = importlib.import_module(
        ".m04a_validation_manifest", __package__
    )
    episodes = validation_module.validate_validation_episode_rows(rows)
    materialized = validation_module.build_validation_episode_artifacts(episodes)
    if materialized.rows != rows or materialized.jsonl_bytes != jsonl_bytes:
        raise ValueError(
            "embedded validation JSONL is not its canonical semantic replay"
        )
    summary = _parse_json_bytes(summary_bytes, label=summary_name)
    if summary != materialized.summary or serialize_json(summary) != summary_bytes:
        raise ValueError("embedded validation summary does not close over JSONL bytes")

    outer = _exact_fields(
        _parse_json_bytes(outer_bytes, label=outer_name),
        {"schema_version", "bundle_status", "run_id", "artifacts"},
        label="embedded validation outer manifest",
    )
    if (
        outer["schema_version"] != 1
        or outer["bundle_status"] != "complete"
        or outer["run_id"] != summary["summary_id"]
    ):
        raise ValueError("embedded validation outer manifest identity drifted")
    artifacts = outer["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "validation_episode_manifest.jsonl",
        "validation_episode_manifest_summary.json",
    }:
        raise ValueError("embedded validation outer artifact set drifted")
    copied_bytes = {
        "validation_episode_manifest.jsonl": jsonl_bytes,
        "validation_episode_manifest_summary.json": summary_bytes,
    }
    for name, content in copied_bytes.items():
        metadata = _exact_fields(
            artifacts[name],
            {"sha256", "bytes", "rows"},
            label=f"embedded validation metadata {name}",
        )
        expected_metadata = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "rows": content.count(b"\n") if name.endswith(".jsonl") else None,
        }
        if metadata != expected_metadata:
            raise ValueError(
                "embedded validation outer metadata does not match payload"
            )
    if (
        lineage["validation_episode_outer_manifest_sha256"]
        != hashlib.sha256(outer_bytes).hexdigest()
        or lineage["validation_episode_manifest_sha256"]
        != hashlib.sha256(jsonl_bytes).hexdigest()
        or lineage["validation_episode_summary_sha256"]
        != hashlib.sha256(summary_bytes).hexdigest()
        or lineage["validation_episode_summary_id"] != summary["summary_id"]
    ):
        raise ValueError("training lineage differs from embedded validation commitment")
    return rows, dict(summary)


def verify_test_source_snapshot_zip(
    snapshot_zip: bytes, *, expected_fingerprint_sha256: str
) -> dict[str, bytes]:
    """Bound and replay the reviewed tests/**/*.py named-byte fingerprint."""

    if type(snapshot_zip) is not bytes:
        raise TypeError("test source snapshot must be immutable bytes")
    expected = _sha256(
        expected_fingerprint_sha256, field="expected_test_source_fingerprint_sha256"
    )
    materials: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(snapshot_zip), mode="r") as archive:
            infos = archive.infolist()
            if not infos or len(infos) > 4096:
                raise ValueError("test source snapshot has an invalid entry count")
            total_size = 0
            for info in infos:
                name = info.filename
                relative = PurePosixPath(name)
                if (
                    not name
                    or "\\" in name
                    or ":" in name
                    or relative.is_absolute()
                    or relative.as_posix() != name
                    or any(part in {"", ".", ".."} for part in relative.parts)
                    or info.is_dir()
                    or stat.S_ISLNK(info.external_attr >> 16)
                    or info.flag_bits & 0x1
                    or info.compress_type
                    not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    or len(relative.parts) < 2
                    or relative.parts[0] != "tests"
                    or not name.endswith(".py")
                ):
                    raise ValueError(f"unsafe test source snapshot entry: {name!r}")
                if name in materials:
                    raise ValueError(f"duplicate test source snapshot entry: {name}")
                total_size += info.file_size
                if info.file_size > 16 * 1024 * 1024 or total_size > 64 * 1024 * 1024:
                    raise ValueError("test source snapshot exceeds size limits")
                content = archive.read(info)
                if len(content) != info.file_size:
                    raise ValueError("test source snapshot entry size drifted")
                materials[name] = content
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        raise ValueError(f"invalid test source snapshot ZIP: {exc}") from exc
    if named_bytes_fingerprint(materials.items()) != expected:
        raise ValueError("test snapshot bytes do not match the committed fingerprint")
    return materials


def _validate_complete_training_bundle(
    bundle: VerifiedBundle, semantic: Mapping[str, Any]
) -> None:
    lineage = semantic["lineage"]
    direct_hashes = {
        "reviewed_runtime_source.zip": "reviewed_runtime_source_sha256",
        "reviewed_test_snapshot.zip": "reviewed_test_snapshot_sha256",
        "remote_launcher.py": "remote_launcher_sha256",
        "launch_plan.json": "launch_plan_sha256",
        "lock_handshake_artifact.json": "lock_handshake_artifact_sha256",
        "preflight_overhead_cost_report.json": (
            "preflight_overhead_cost_report_sha256"
        ),
        "preflight_diagnostic_checkpoint.pt": (
            "preflight_diagnostic_checkpoint_sha256"
        ),
        "frozen_contract.md": "frozen_contract_sha256",
        "sanitized_shard_manifest.json": "sanitized_shard_manifest_sha256",
        "data_split_manifest.json": "data_split_manifest_sha256",
        "validation_episode_outer_manifest.json": (
            "validation_episode_outer_manifest_sha256"
        ),
        "validation_episode_manifest.jsonl": "validation_episode_manifest_sha256",
        "validation_episode_manifest_summary.json": (
            "validation_episode_summary_sha256"
        ),
        "environment_manifest.json": "environment_manifest_sha256",
        "model_config.json": "model_config_sha256",
        "training_cost_ledger.jsonl": "training_cost_ledger_sha256",
        "selected_checkpoint_manifest.json": "selected_checkpoint_manifest_sha256",
        "selected_checkpoint.pt": "checkpoint_sha256",
        "python-runtime-lock.json": "python_runtime_lock_sha256",
    }
    for filename, lineage_field in direct_hashes.items():
        if (
            hashlib.sha256(bundle.read_bytes(filename)).hexdigest()
            != lineage[lineage_field]
        ):
            raise ValueError(
                f"training payload {filename} does not match lineage.{lineage_field}"
            )

    optimizer_state_bytes = bundle.read_bytes("optimizer_schedule_state.json")
    optimizer_state = validate_optimizer_schedule_state(
        bundle.read_json("optimizer_schedule_state.json")
    )
    if serialize_json(optimizer_state) != optimizer_state_bytes:
        raise ValueError("optimizer/schedule state is not canonically serialized")

    launch_plan_bytes = bundle.read_bytes("launch_plan.json")
    launch_plan_module = importlib.import_module(".m04a_launch_plan", __package__)
    launch_plan = launch_plan_module.validate_launch_plan_payload(
        bundle.read_json("launch_plan.json")
    )
    if launch_plan_module.canonical_launch_plan_bytes(launch_plan) != launch_plan_bytes:
        raise ValueError("launch plan is not its canonical committed byte payload")
    for filename, expected_sha256 in launch_plan["expected_input_artifacts"].items():
        if hashlib.sha256(bundle.read_bytes(filename)).hexdigest() != expected_sha256:
            raise ValueError(f"launch plan input commitment differs for {filename}")
    runtime_snapshot = bundle.read_bytes("reviewed_runtime_source.zip")
    test_snapshot = bundle.read_bytes("reviewed_test_snapshot.zip")
    verify_source_snapshot_zip(
        runtime_snapshot,
        expected_fingerprint_sha256=launch_plan["runtime_source_fingerprint_sha256"],
    )
    verify_test_source_snapshot_zip(
        test_snapshot,
        expected_fingerprint_sha256=launch_plan["test_source_fingerprint_sha256"],
    )
    if (
        launch_plan["training_config_sha256"]
        != _train_contract().training_config_sha256()
    ):
        raise ValueError("launch plan does not bind the frozen training config")
    runtime_lock_module = importlib.import_module(
        ".m04a_python_runtime_lock", __package__
    )
    runtime_lock_bytes = bundle.read_bytes("python-runtime-lock.json")
    runtime_lock = runtime_lock_module.validate_python_runtime_lock_payload(
        bundle.read_json("python-runtime-lock.json")
    )
    if (
        runtime_lock_module.canonical_python_runtime_lock_bytes(runtime_lock)
        != runtime_lock_bytes
        or hashlib.sha256(runtime_lock_bytes).hexdigest()
        != lineage["python_runtime_lock_sha256"]
        or runtime_lock["runtime_lock_id"] != lineage["python_runtime_lock_id"]
        or lineage["python_runtime_lock_sha256"]
        != launch_plan["expected_input_artifacts"]["python-runtime-lock.json"]
        or launch_plan["ordered_import_roots"].count(
            runtime_lock["environment"]["site_packages_path"]
        )
        != 1
    ):
        raise ValueError("training runtime lock differs from plan/lineage commitments")

    preflight_report = bundle.read_json("preflight.json")
    preflight_runtime = _exact_fields(
        preflight_report.get("runtime") if isinstance(preflight_report, dict) else None,
        {"run_id", "gpu_uuid", "logical_device_index"},
        label="preflight runtime identity",
    )
    if preflight_runtime["run_id"] != launch_plan["run_id"]:
        raise ValueError("preflight runtime belongs to another launch plan")
    environment_bytes = bundle.read_bytes("environment_manifest.json")
    environment = bundle.read_json("environment_manifest.json")
    if serialize_json(environment) != environment_bytes:
        raise ValueError("environment manifest is not canonically serialized")
    validate_environment_manifest_artifact(
        environment,
        launch_plan=launch_plan,
        expected_gpu_uuid=preflight_runtime["gpu_uuid"],
        expected_launcher_sha256=lineage["remote_launcher_sha256"],
        python_runtime_lock=runtime_lock,
    )
    locked_python_identity = python_runtime_identity_sha256(
        _runtime_lock_identity_payload(runtime_lock)
    )
    environment_python_identity = python_runtime_identity_sha256(
        _environment_identity_payload(environment)
    )
    if (
        locked_python_identity != environment_python_identity
        or lineage["python_runtime_identity_sha256"] != locked_python_identity
    ):
        raise ValueError(
            "training Python runtime identity differs from lock/environment lineage"
        )
    lock_artifact_bytes = bundle.read_bytes("lock_handshake_artifact.json")
    lock_artifact = bundle.read_json("lock_handshake_artifact.json")
    overhead_cost_report_bytes = bundle.read_bytes(
        "preflight_overhead_cost_report.json"
    )
    overhead_cost_report = bundle.read_json("preflight_overhead_cost_report.json")
    if serialize_json(overhead_cost_report) != overhead_cost_report_bytes:
        raise ValueError("overhead cost report is not canonically serialized")
    diagnostic_checkpoint_snapshot = bundle.read_bytes(
        "preflight_diagnostic_checkpoint.pt"
    )

    ledger_rows = bundle.read_jsonl("training_cost_ledger.jsonl")
    validation_manifest_rows, validation_manifest_summary = (
        _validate_embedded_validation_manifest(bundle, lineage)
    )
    validation_episodes_per_pass = len(validation_manifest_rows)
    if validation_episodes_per_pass <= 0:
        raise ValueError("fixed validation episode manifest must be nonempty")
    if launch_plan["validation_manifest_commitment"] != {
        "schema": PREFLIGHT_VALIDATION_COMMITMENT_SCHEMA_VERSION,
        "commitment_id": launch_plan["validation_manifest_commitment"]["commitment_id"],
        "outer_artifact_manifest_sha256": lineage[
            "validation_episode_outer_manifest_sha256"
        ],
        "jsonl_sha256": lineage["validation_episode_manifest_sha256"],
        "summary_id": validation_manifest_summary["summary_id"],
        "row_count": validation_episodes_per_pass,
    }:
        raise ValueError(
            "launch plan validation commitment differs from embedded evidence"
        )
    preflight = validate_preflight_artifact(
        preflight_report,
        ledger_rows[:PREFLIGHT_UPDATES],
        validation_episodes_per_pass=validation_episodes_per_pass,
        expected_validation_outer_manifest_sha256=lineage[
            "validation_episode_outer_manifest_sha256"
        ],
        expected_validation_jsonl_sha256=lineage["validation_episode_manifest_sha256"],
        expected_validation_summary_id=validation_manifest_summary["summary_id"],
        overhead_cost_probe_report=overhead_cost_report,
        diagnostic_checkpoint_snapshot=diagnostic_checkpoint_snapshot,
        lock_handshake_artifact=lock_artifact,
        lock_handshake_artifact_bytes=lock_artifact_bytes,
        expected_lock_handshake_artifact_sha256=lineage[
            "lock_handshake_artifact_sha256"
        ],
        expected_config_sha256=launch_plan["training_config_sha256"],
        expected_runtime_source_sha256=launch_plan["runtime_source_fingerprint_sha256"],
        expected_test_source_sha256=launch_plan["test_source_fingerprint_sha256"],
        expected_launcher_sha256=lineage["remote_launcher_sha256"],
        expected_launch_plan_sha256=lineage["launch_plan_sha256"],
        expected_remote_project_root=launch_plan["remote_project_root"],
        expected_attempt_nonce=launch_plan["attempt_nonce"],
    )
    if semantic["closure"]["resume_segments"] != 0:
        raise ValueError(
            "M04a v0.4 training evidence forbids resume until per-segment locks exist"
        )
    resume_rows = [row for row in ledger_rows if row.get("phase") == "resume_segment"]
    resume_wall_time_ns = sum(
        _strict_int(row.get("wall_time_ns"), field="resume.wall_time_ns")
        for row in resume_rows
    )
    recorded_resume_setup = semantic["timing"]["resume_setup_wall_time_ns"]
    if recorded_resume_setup < resume_wall_time_ns:
        raise ValueError("training timing omits a resume segment")
    summary = validate_training_cost_summary(
        bundle.read_json("training_cost_summary.json")
    )
    lock_segments = summary["lock_interval"]["segments"]
    if (
        lock_segments[0]["start_perf_counter_ns"]
        != lock_artifact["acquisition_started_perf_counter_ns"]
    ):
        raise ValueError("training lock interval does not start at the full handshake")
    if summary["ledger_sha256"] != lineage["training_cost_ledger_sha256"]:
        raise ValueError("training summary does not bind the event ledger")
    if (
        summary["closure"] != semantic["closure"]
        or summary["timing"] != semantic["timing"]
    ):
        raise ValueError("training summary differs from semantic artifact closure")
    ledger_closure = validate_training_cost_ledger(
        ledger_rows,
        validation_episodes_per_pass=validation_episodes_per_pass,
        fresh_setup_wall_time_ns=recorded_resume_setup - resume_wall_time_ns,
        expected_masked_token_predictions=semantic["closure"][
            "masked_token_predictions"
        ],
        lock_interval=summary["lock_interval"],
        preflight_endpoint_wall_time_ns=preflight["budget_projection"][
            "measured_preflight_total_wall_ns"
        ],
    )
    _validate_training_lock_timeline_coverage(
        lock_interval=summary["lock_interval"],
        preflight_started_perf_counter_ns=preflight[
            "preflight_started_perf_counter_ns"
        ],
        phase_timing=ledger_closure,
        preflight_endpoint_wall_time_ns=preflight["budget_projection"][
            "measured_preflight_total_wall_ns"
        ],
    )
    closure_bindings = {
        "preflight_optimizer_updates": ledger_closure["preflight_optimizer_updates"],
        "preflight_microbatches": ledger_closure["preflight_microbatches"],
        "preflight_encoder_forward_calls": ledger_closure[
            "preflight_encoder_forward_calls"
        ],
        "preflight_decoder_forward_calls": ledger_closure[
            "preflight_decoder_forward_calls"
        ],
        "preflight_backward_calls": ledger_closure["preflight_backward_calls"],
        "primary_optimizer_updates": ledger_closure["optimizer_updates"],
        "primary_microbatches": ledger_closure["microbatches"],
        "primary_arc2_episodes": ledger_closure["arc2_episodes"],
        "primary_rearc_episodes": ledger_closure["rearc_episodes"],
        "primary_encoder_forward_calls": ledger_closure["encoder_forward_calls"],
        "primary_decoder_forward_calls": ledger_closure["decoder_forward_calls"],
        "primary_backward_calls": ledger_closure["backward_calls"],
        "validation_passes": ledger_closure["validation_passes"],
        "validation_episode_calls": ledger_closure["validation_episode_calls"],
        "validation_encoder_forward_calls": ledger_closure[
            "validation_encoder_forward_calls"
        ],
        "validation_decoder_forward_calls": ledger_closure[
            "validation_decoder_forward_calls"
        ],
        "checkpoint_writes": ledger_closure["checkpoint_writes"],
        "resume_segments": ledger_closure["resume_segments"],
        "masked_token_predictions": ledger_closure["masked_token_predictions"],
    }
    if any(
        semantic["closure"][field] != expected
        for field, expected in closure_bindings.items()
    ):
        raise ValueError("training semantic closure differs from the event ledger")
    for field in TRAINING_ARTIFACT_TIMING_FIELDS:
        if semantic["timing"][field] != ledger_closure[field]:
            raise ValueError("training phase timing differs from the event ledger")

    preflight_inference = preflight["inference"]
    inference_bindings = {
        "preflight_inference_lanes": preflight_inference["lane_count"],
        "preflight_inference_steps": preflight_inference["denoising_steps"],
        "preflight_inference_encoder_batch_calls": preflight_inference[
            "encoder_batch_calls"
        ],
        "preflight_inference_decoder_batch_calls": preflight_inference[
            "decoder_batch_calls"
        ],
        "preflight_inference_sample_equivalent_forward_calls": preflight_inference[
            "sample_equivalent_forward_calls"
        ],
    }
    if any(
        semantic["closure"][field] != value
        for field, value in inference_bindings.items()
    ):
        raise ValueError("training closure differs from exact preflight inference")

    checkpoint_rows = bundle.read_jsonl("checkpoint_manifests.jsonl")
    validation_rows = bundle.read_jsonl("validation_metrics.jsonl")
    selected = bundle.read_json("selected_checkpoint_manifest.json")
    diagnostic_sha256 = lineage["preflight_diagnostic_checkpoint_sha256"]
    real_checkpoint_sha256s = {row.get("checkpoint_sha256") for row in checkpoint_rows}
    if (
        diagnostic_sha256 == lineage["checkpoint_sha256"]
        or diagnostic_sha256 in real_checkpoint_sha256s
    ):
        raise ValueError("diagnostic checkpoint was promoted into the selectable set")
    validate_training_checkpoint_artifacts(
        checkpoint_rows,
        validation_rows,
        selected,
        training_ledger_rows=ledger_rows,
        checkpoint_manifests_sha256=hashlib.sha256(
            bundle.read_bytes("checkpoint_manifests.jsonl")
        ).hexdigest(),
        validation_metrics_sha256=hashlib.sha256(
            bundle.read_bytes("validation_metrics.jsonl")
        ).hexdigest(),
        training_cost_ledger_sha256=lineage["training_cost_ledger_sha256"],
        validation_episode_outer_manifest_sha256=lineage[
            "validation_episode_outer_manifest_sha256"
        ],
        validation_episode_manifest_sha256=lineage[
            "validation_episode_manifest_sha256"
        ],
        validation_episode_rows=validation_manifest_rows,
        selected_checkpoint_sha256=lineage["checkpoint_sha256"],
        selected_checkpoint_bytes=bundle.artifacts["selected_checkpoint.pt"]["bytes"],
        lock_interval_id=summary["lock_interval"]["interval_id"],
        lock_selection_completed_perf_counter_ns=lock_segments[-1][
            "end_perf_counter_ns"
        ],
    )
    if (
        len(bundle.read_jsonl("resume_manifests.jsonl"))
        != semantic["closure"]["resume_segments"]
    ):
        raise ValueError("resume manifest rows do not close from the training ledger")


def _validate_pool_parent_lineage(
    parent: Mapping[str, Any], lineage: Mapping[str, Any]
) -> None:
    if parent.get("semantic_id") != lineage["training_artifact_id"]:
        raise ValueError("pool training parent semantic ID mismatch")
    parent_lineage = parent.get("lineage")
    if not isinstance(parent_lineage, dict) or any(
        parent_lineage.get(field) != lineage[field]
        for field in (
            "selected_checkpoint_manifest_sha256",
            "checkpoint_sha256",
            "environment_manifest_sha256",
            "python_runtime_lock_sha256",
            "python_runtime_lock_id",
            "python_runtime_identity_sha256",
        )
    ):
        raise ValueError("pool checkpoint/runtime differs from its training parent")


def _validate_pool_environment_runtime(
    environment: object,
    *,
    lineage: Mapping[str, Any],
    training_parent: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _validate_environment_manifest_shape(environment)
    parent_lineage = training_parent.get("lineage")
    if not isinstance(parent_lineage, Mapping):
        raise ValueError("pool training parent lineage is malformed")
    expected_sha256 = lineage["python_runtime_lock_sha256"]
    expected_id = lineage["python_runtime_lock_id"]
    expected_identity = lineage["python_runtime_identity_sha256"]
    environment_identity = python_runtime_identity_sha256(
        _environment_identity_payload(payload)
    )
    if (
        payload["python_runtime_lock_sha256"] != expected_sha256
        or payload["python_runtime_lock_id"] != expected_id
        or environment_identity != expected_identity
        or parent_lineage.get("python_runtime_lock_sha256") != expected_sha256
        or parent_lineage.get("python_runtime_lock_id") != expected_id
        or parent_lineage.get("python_runtime_identity_sha256") != expected_identity
    ):
        raise ValueError(
            "pool environment runtime lock differs from pool/training lineage"
        )
    return payload


def _validate_complete_pool_bundle(
    bundle: VerifiedBundle, semantic: Mapping[str, Any]
) -> None:
    lineage = semantic["lineage"]
    direct_hashes = {
        "training_artifact_manifest.json": "training_artifact_manifest_sha256",
        "selected_checkpoint_manifest.json": "selected_checkpoint_manifest_sha256",
        "blind_artifact_manifest.json": "blind_artifact_manifest_sha256",
        "blind_tasks.jsonl": "blind_tasks_sha256",
        "source_snapshot.zip": "source_snapshot_sha256",
        "environment_manifest.json": "environment_manifest_sha256",
        "sampler_config.json": "sampler_config_sha256",
    }
    for filename, lineage_field in direct_hashes.items():
        if (
            hashlib.sha256(bundle.read_bytes(filename)).hexdigest()
            != lineage[lineage_field]
        ):
            raise ValueError(
                f"pool payload {filename} does not match lineage.{lineage_field}"
            )
    if (
        lineage["blind_artifact_manifest_sha256"]
        != FIXED_BLIND_ARTIFACT_MANIFEST_SHA256
    ):
        raise ValueError("pool does not bind the frozen blind artifact")
    if lineage["blind_tasks_sha256"] != FIXED_BLIND_TASKS_SHA256:
        raise ValueError("pool does not bind the frozen blind task bytes")
    if lineage["dsl_pool_content_id"] != FIXED_DSL_POOL_CONTENT_ID:
        raise ValueError("pool does not bind the frozen DSL v0.6 content ID")
    training_parent = validate_artifact_manifest(
        bundle.read_json("training_artifact_manifest.json"),
        expected_kind="training",
    )
    _validate_pool_parent_lineage(training_parent, lineage)
    environment_bytes = bundle.read_bytes("environment_manifest.json")
    environment = bundle.read_json("environment_manifest.json")
    if serialize_json(environment) != environment_bytes:
        raise ValueError("pool environment manifest is not canonically serialized")
    _validate_pool_environment_runtime(
        environment,
        lineage=lineage,
        training_parent=training_parent,
    )
    selected_checkpoint = bundle.read_json("selected_checkpoint_manifest.json")
    selected_values = _values_for_key(selected_checkpoint, "checkpoint_sha256")
    if not selected_values or set(selected_values) != {lineage["checkpoint_sha256"]}:
        raise ValueError("pool selected-checkpoint manifest binds another checkpoint")

    sidecar = validate_blind_input_shape_sidecar(
        bundle.read_json("blind_input_shape_manifest.json"),
        bundle.read_jsonl("blind_input_shape_rows.jsonl"),
    )
    if sidecar.manifest["sidecar_id"] != lineage["blind_shape_sidecar_id"]:
        raise ValueError("pool lineage blind sidecar ID mismatch")
    receipt = validate_pool_replay_receipt(
        bundle.read_json("replay_verification_receipt.json")
    )
    if any(
        receipt[field] != expected
        for field, expected in {
            "checkpoint_sha256": lineage["checkpoint_sha256"],
            "blind_shape_sidecar_id": sidecar.manifest["sidecar_id"],
            "blind_shape_rows_sha256": sidecar.manifest["rows_sha256"],
            "case_set_id": sidecar.manifest["case_set_id"],
        }.items()
    ):
        raise ValueError("pool replay receipt parent binding mismatch")

    traces = bundle.read_jsonl("lane_traces.jsonl")
    lanes = bundle.read_jsonl("candidate_rows.jsonl")
    encoders = bundle.read_jsonl("encoder_forward_ledger.jsonl")
    decoders = bundle.read_jsonl("batch_forward_ledger.jsonl")
    pairs = bundle.read_jsonl("pair_cost_ledger.jsonl")
    expected_pairs = [(row["blind_task_id"], row["test_index"]) for row in sidecar.rows]
    if [
        (row.get("blind_task_id"), row.get("test_index")) for row in pairs
    ] != expected_pairs:
        raise ValueError("pool pair-cost rows differ from frozen sidecar order")
    pair_order_payload = [
        {
            "blind_task_id": row["blind_task_id"],
            "test_index": row["test_index"],
            "row_id": row["row_id"],
        }
        for row in sidecar.rows
    ]
    receipt_bindings = {
        "verifier_source_sha256": lineage["source_snapshot_sha256"],
        "pair_order_sha256": _canonical_sha256(pair_order_payload),
        "pair_cost_ids_sha256": _canonical_sha256(
            [row.get("pair_cost_id") for row in pairs]
        ),
        "trace_sha256s_sha256": _canonical_sha256(
            [lane_trace_sha256(row) for row in traces]
        ),
        "lane_ids_sha256": _canonical_sha256([row.get("lane_id") for row in lanes]),
        "encoder_call_ids_sha256": _canonical_sha256(
            [row.get("call_id") for row in encoders]
        ),
        "decoder_call_ids_sha256": _canonical_sha256(
            [row.get("call_id") for row in decoders]
        ),
        "pair_count": len(pairs),
        "trace_count": len(traces),
        "lane_count": len(lanes),
        "encoder_call_count": len(encoders),
        "decoder_call_count": len(decoders),
    }
    if any(receipt[field] != expected for field, expected in receipt_bindings.items()):
        raise ValueError("pool replay receipt does not bind the published row ledgers")

    normalized_pairs: list[dict[str, Any]] = []
    for sidecar_row, raw_pair in zip(sidecar.rows, pairs):
        pair = validate_pair_cost_row(raw_pair)
        if pair["accepted_shapes"] != sidecar_row["accepted_shapes"]:
            raise ValueError("published pair shapes differ from frozen sidecar")
        task_id, test_index = pair["blind_task_id"], pair["test_index"]
        pair_lanes = _rows_for_pair(lanes, task_id, test_index)
        pair_traces = _rows_for_pair(traces, task_id, test_index)
        pair_encoders = _rows_for_pair(encoders, task_id, test_index)
        pair_decoders = _rows_for_pair(decoders, task_id, test_index)
        validate_pair_evidence(
            pair,
            lane_rows=pair_lanes,
            traces=pair_traces,
            encoder_rows=pair_encoders,
            decoder_rows=pair_decoders,
        )
        if any(
            row["checkpoint_sha256"] != lineage["checkpoint_sha256"]
            for row in pair_lanes
        ):
            raise ValueError("published lane checkpoint differs from pool lineage")
        if pair_encoders:
            encoder = pair_encoders[0]
            expected_cache = {
                "encoder_call_id": encoder["call_id"],
                "cache_key": _canonical_sha256(
                    {
                        "checkpoint_sha256": lineage["checkpoint_sha256"],
                        "blind_task_id": task_id,
                        "test_index": test_index,
                        "memory_length": encoder["unpadded_memory_length"],
                        "dtype": "bfloat16",
                    }
                ),
                "memory_length": encoder["unpadded_memory_length"],
                "dtype": "bfloat16",
            }
            if encoder["cache_dtype"] != "bfloat16" or any(
                row["cached_memory_descriptor"] != expected_cache
                for row in pair_decoders
            ):
                raise ValueError(
                    "published decoder calls do not share one bound BF16 cache"
                )
        normalized_pairs.append(pair)

    setup = validate_pool_setup_cost_row(bundle.read_json("pool_setup_cost.json"))
    if setup["checkpoint_sha256"] != lineage["checkpoint_sha256"]:
        raise ValueError("pool setup checkpoint differs from pool lineage")
    pool_summary = validate_pool_cost_summary(
        bundle.read_json("pool_cost_summary.json"),
        setup_row=setup,
        pair_rows=normalized_pairs,
    )
    if any(
        semantic["timing"][field] != pool_summary[field]
        for field in POOL_ARTIFACT_TIMING_FIELDS
    ):
        raise ValueError("pool semantic timing differs from the cost summary")
    masked_predictions = sum(
        validate_lane_row(lane, trace=trace)["masked_token_predictions"]
        for lane, trace in zip(lanes, traces)
    )
    closure = semantic["closure"]
    derived = {
        "raw_lanes": sum(pair["raw_lanes"] for pair in normalized_pairs),
        "lane_trace_rows": len(traces),
        "candidate_rows": len(lanes),
        "encoder_batch_calls": len(encoders),
        "decoder_batch_calls": len(decoders),
        "total_actual_batch_calls": len(encoders) + len(decoders),
        "sample_equivalent_forward_calls": sum(
            pair["sample_equivalent_forward_calls"] for pair in normalized_pairs
        ),
        "masked_token_predictions": masked_predictions,
        "format_valid": sum(pair["format_valid"] for pair in normalized_pairs),
        "format_invalid": sum(pair["format_invalid"] for pair in normalized_pairs),
        "unique_outputs": sum(pair["unique_outputs"] for pair in normalized_pairs),
        "duplicate_outputs": sum(
            pair["duplicate_outputs"] for pair in normalized_pairs
        ),
        "pair_cost_rows": len(normalized_pairs),
        "replay_verified_trace_count": receipt["trace_count"],
        "replay_verified_decoder_call_count": receipt["decoder_call_count"],
        "replay_verified_prediction_count": receipt["masked_token_predictions"],
        "bf16_cache_pair_count": len(encoders),
    }
    if any(closure[field] != expected for field, expected in derived.items()):
        raise ValueError("pool semantic closure does not close from published evidence")


def _validate_complete_evaluation_bundle(
    bundle: VerifiedBundle, semantic: Mapping[str, Any]
) -> None:
    lineage = semantic["lineage"]
    parent_bytes = bundle.read_bytes("pool_artifact_manifest.json")
    if (
        hashlib.sha256(parent_bytes).hexdigest()
        != lineage["pool_artifact_manifest_sha256"]
    ):
        raise ValueError("evaluation pool-parent bytes differ from lineage")
    parent = validate_artifact_manifest(
        bundle.read_json("pool_artifact_manifest.json"), expected_kind="pool"
    )
    parent_lineage = parent["lineage"]
    if (
        parent["semantic_id"] != lineage["pool_artifact_id"]
        or parent_lineage["dsl_pool_content_id"] != lineage["dsl_pool_content_id"]
        or parent_lineage["python_runtime_lock_sha256"]
        != lineage["python_runtime_lock_sha256"]
        or parent_lineage["python_runtime_lock_id"] != lineage["python_runtime_lock_id"]
        or parent_lineage["python_runtime_identity_sha256"]
        != lineage["python_runtime_identity_sha256"]
    ):
        raise ValueError("evaluation runtime/pool lineage differs from its parent")


def publish_training_artifact_bundle(
    output_dir: str | Path, **kwargs: Any
) -> M04aArtifactBundle:
    return publish_m04a_artifact_bundle(output_dir, artifact_kind="training", **kwargs)


def publish_pool_artifact_bundle(
    output_dir: str | Path, **kwargs: Any
) -> M04aArtifactBundle:
    return publish_m04a_artifact_bundle(output_dir, artifact_kind="pool", **kwargs)


def publish_evaluation_artifact_bundle(
    output_dir: str | Path, **kwargs: Any
) -> M04aArtifactBundle:
    return publish_m04a_artifact_bundle(
        output_dir, artifact_kind="evaluation", **kwargs
    )


def _mask_count_trace(cell_count: int) -> tuple[int, ...]:
    module = _contract()
    if module is not None and hasattr(module, "mask_count_trace"):
        return module.mask_count_trace(cell_count)
    cell_count = _strict_int(cell_count, field="cell_count", minimum=1)
    steps = _contract_value("DENOISING_STEPS")
    counts = [cell_count]
    for step in range(1, steps + 1):
        if step == steps:
            count = 0
        elif step == 8:
            count = (cell_count + 1) // 2
        else:
            count = math.ceil(cell_count * math.cos(math.pi * step / 24))
        counts.append(count)
    return tuple(counts)


def _lane_seed(
    blind_task_id: str, test_index: int, shape_proposal_id: str, local_lane: int
) -> int:
    module = _contract()
    if module is not None and hasattr(module, "lane_seed"):
        return module.lane_seed(
            blind_task_id, test_index, shape_proposal_id, local_lane
        )
    payload = (
        f"{_contract_value('SAMPLER_SEMANTICS_VERSION')}\0{blind_task_id}\0"
        f"{test_index}\0{shape_proposal_id}\0{local_lane}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _shape_lane_allocations(count: int) -> tuple[int, ...]:
    module = _contract()
    if module is not None and hasattr(module, "shape_lane_allocations"):
        return module.shape_lane_allocations(count)
    allocations = {0: (), 1: (64,), 2: (32, 32), 3: (22, 21, 21), 4: (16, 16, 16, 16)}
    count = _strict_int(count, field="accepted_shape_count")
    if count not in allocations:
        raise ValueError("accepted_shape_count must be in 0..4")
    return allocations[count]


def _decoder_batch_calls(count: int) -> int:
    module = _contract()
    if module is not None and hasattr(module, "decoder_batch_calls"):
        return module.decoder_batch_calls(count)
    return _contract_value("DENOISING_STEPS") * sum(
        math.ceil(value / _contract_value("INFERENCE_MICROBATCH"))
        for value in _shape_lane_allocations(count)
    )


def _canonical_float_hex(value: object, *, field: str) -> float:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field} must be a float.hex() string")
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field} is not a float.hex() string") from exc
    if not math.isfinite(parsed) or parsed.hex() != value:
        raise ValueError(f"{field} must be canonical and finite")
    return parsed


def _probability_matches_log(probability: float, log_probability: float) -> bool:
    reconstructed = math.exp(log_probability)
    if probability == 0.0:
        return reconstructed == 0.0
    return math.isclose(
        probability,
        reconstructed,
        rel_tol=2e-14,
        abs_tol=float.fromhex("0x0.0000000000001p-1022"),
    )


_TRACE_FIELDS = {
    "schema",
    "blind_task_id",
    "test_index",
    "shape_order",
    "shape_proposal_id",
    "proposed_height",
    "proposed_width",
    "global_lane",
    "local_lane",
    "greedy",
    "seed_u64",
    "initial_masked_indices",
    "steps",
    "final_grid",
    "output_key",
}

_TRACE_STEP_FIELDS = {
    "step",
    "masked_indices_at_entry",
    "logits_float64_le_sha256",
    "predictions",
    "remasked_indices",
    "state_after_step",
}

_TRACE_PREDICTION_FIELDS = {
    "linear_index",
    "row",
    "column",
    "chosen_color",
    "probability_hex",
    "log_probability_hex",
    "uniform_hex",
    "provisional_remasked",
}


@dataclass(frozen=True, slots=True)
class LaneTraceSummary:
    mask_count_trace: tuple[int, ...]
    rng_draw_count: int
    masked_token_predictions: int
    final_grid: tuple[tuple[int, ...], ...]
    output_key: str
    mean_log_probability: float
    minimum_probability: float


def lane_trace_sha256(trace: Mapping[str, Any]) -> str:
    if not isinstance(trace, Mapping):
        raise TypeError("trace must be a mapping")
    if trace.get("schema") != _contract_value("LANE_TRACE_SCHEMA_VERSION"):
        raise ValueError("trace schema does not match the frozen lane-trace schema")
    return _canonical_sha256(dict(trace))


def validate_lane_trace(trace: object) -> LaneTraceSummary:
    payload = _exact_fields(trace, _TRACE_FIELDS, label="M04a lane trace")
    if payload["schema"] != _contract_value("LANE_TRACE_SCHEMA_VERSION"):
        raise ValueError("unsupported M04a lane trace schema")
    task_id = _nonempty_string(payload["blind_task_id"], field="blind_task_id")
    test_index = _strict_int(payload["test_index"], field="test_index")
    _strict_int(payload["shape_order"], field="shape_order")
    proposal_id = _sha256(payload["shape_proposal_id"], field="shape_proposal_id")
    height = _strict_int(payload["proposed_height"], field="proposed_height", minimum=1)
    width = _strict_int(payload["proposed_width"], field="proposed_width", minimum=1)
    if height > 30 or width > 30:
        raise ValueError("proposed trace shape exceeds ARC bounds")
    _strict_int(payload["global_lane"], field="global_lane")
    local_lane = _strict_int(payload["local_lane"], field="local_lane")
    if type(payload["greedy"]) is not bool:
        raise TypeError("greedy must be boolean")
    if payload["greedy"] != (local_lane == 0):
        raise ValueError("only local lane 0 may be greedy")
    seed = _strict_int(payload["seed_u64"], field="seed_u64")
    if seed >= 1 << 64:
        raise ValueError("seed_u64 must fit in uint64")
    if seed != _lane_seed(task_id, test_index, proposal_id, local_lane):
        raise ValueError("seed_u64 does not match the frozen lane seed")

    cell_count = height * width
    schedule = _mask_count_trace(cell_count)
    expected_initial = list(range(cell_count))
    if payload["initial_masked_indices"] != expected_initial:
        raise ValueError("trace must begin with every row-major cell masked")
    steps = payload["steps"]
    denoising_steps = _contract_value("DENOISING_STEPS")
    if not isinstance(steps, list) or len(steps) != denoising_steps:
        raise ValueError(f"lane trace must contain exactly {denoising_steps} steps")

    state: list[int | str] = [MASK_SENTINEL] * cell_count
    final_probabilities: dict[int, tuple[float, float]] = {}
    rng_draw_count = 0
    masked_prediction_count = 0
    for step_index, raw_step in enumerate(steps, start=1):
        step = _exact_fields(
            raw_step, _TRACE_STEP_FIELDS, label=f"trace step {step_index}"
        )
        if step["step"] != step_index:
            raise ValueError("trace step indices must be consecutive and one-based")
        entry = step["masked_indices_at_entry"]
        expected_entry = [
            index for index, value in enumerate(state) if value == MASK_SENTINEL
        ]
        if entry != expected_entry or len(entry) != schedule[step_index - 1]:
            raise ValueError(f"masked entry state does not close at step {step_index}")
        _sha256(
            step["logits_float64_le_sha256"],
            field=f"steps[{step_index - 1}].logits_float64_le_sha256",
        )
        predictions = step["predictions"]
        if not isinstance(predictions, list) or len(predictions) != len(entry):
            raise ValueError("one prediction is required for every masked entry cell")
        parsed_predictions: list[tuple[int, int, float, float, bool]] = []
        for prediction_index, raw_prediction in enumerate(predictions):
            prediction = _exact_fields(
                raw_prediction,
                _TRACE_PREDICTION_FIELDS,
                label=f"step {step_index} prediction {prediction_index}",
            )
            linear_index = _strict_int(
                prediction["linear_index"], field="prediction.linear_index"
            )
            if linear_index != entry[prediction_index]:
                raise ValueError("predictions must preserve masked row-major order")
            if prediction["row"] != linear_index // width or prediction["column"] != (
                linear_index % width
            ):
                raise ValueError("prediction row/column does not match linear index")
            color = _strict_int(
                prediction["chosen_color"], field="prediction.chosen_color"
            )
            if color > 9:
                raise ValueError("prediction color must be in 0..9")
            probability = _canonical_float_hex(
                prediction["probability_hex"], field="prediction.probability_hex"
            )
            log_probability = _canonical_float_hex(
                prediction["log_probability_hex"],
                field="prediction.log_probability_hex",
            )
            if not 0.0 <= probability <= 1.0 or log_probability > 0.0:
                raise ValueError(
                    "prediction probability/log probability is out of range"
                )
            if not _probability_matches_log(probability, log_probability):
                raise ValueError(
                    "prediction probability does not match its natural log probability"
                )
            uniform = prediction["uniform_hex"]
            if payload["greedy"]:
                if uniform is not None:
                    raise ValueError("greedy traces must not consume RNG draws")
            else:
                parsed_uniform = _canonical_float_hex(
                    uniform, field="prediction.uniform_hex"
                )
                if not 0.0 <= parsed_uniform < 1.0:
                    raise ValueError("sampled uniform must be in [0, 1)")
                rng_draw_count += 1
            if type(prediction["provisional_remasked"]) is not bool:
                raise TypeError("provisional_remasked must be boolean")
            parsed_predictions.append(
                (
                    linear_index,
                    color,
                    probability,
                    log_probability,
                    prediction["provisional_remasked"],
                )
            )
        masked_prediction_count += len(entry)
        remasked = step["remasked_indices"]
        if not isinstance(remasked, list) or any(
            type(value) is not int for value in remasked
        ):
            raise TypeError("remasked_indices must be a list of integer indices")
        flagged = [item[0] for item in parsed_predictions if item[4]]
        if remasked != sorted(flagged):
            raise ValueError("remasked_indices must match provisional prediction flags")
        retained_count = schedule[step_index]
        confidence_selected = sorted(
            parsed_predictions, key=lambda item: (item[2], item[0])
        )[:retained_count]
        if remasked != sorted(item[0] for item in confidence_selected):
            raise ValueError(
                "re-mask set is not the frozen lowest-confidence selection"
            )
        next_state = list(state)
        for (
            linear_index,
            color,
            probability,
            log_probability,
            is_remasked,
        ) in parsed_predictions:
            if is_remasked:
                next_state[linear_index] = MASK_SENTINEL
            else:
                next_state[linear_index] = color
                final_probabilities[linear_index] = (probability, log_probability)
        raw_state = step["state_after_step"]
        if raw_state != next_state:
            raise ValueError(f"state_after_step does not close at step {step_index}")
        state = next_state

    if (
        any(value == MASK_SENTINEL for value in state)
        or len(final_probabilities) != cell_count
    ):
        raise ValueError("final trace step must commit every target cell")
    final_grid = as_grid(
        [
            [int(state[row * width + column]) for column in range(width)]
            for row in range(height)
        ]
    )
    declared_grid = as_grid(payload["final_grid"])
    if declared_grid != final_grid:
        raise ValueError("trace final_grid does not match reconstructed final state")
    output_key = grid_key(final_grid)
    if payload["output_key"] != output_key:
        raise ValueError("trace output_key does not match final_grid")
    ordered_probabilities = [final_probabilities[index] for index in range(cell_count)]
    mean_log_probability = sum(value[1] for value in ordered_probabilities) / cell_count
    minimum_probability = min(value[0] for value in ordered_probabilities)
    if payload["greedy"] and rng_draw_count != 0:
        raise AssertionError("greedy trace unexpectedly consumed RNG")
    if not payload["greedy"] and rng_draw_count != masked_prediction_count:
        raise ValueError("sampled trace RNG draw count does not close")
    return LaneTraceSummary(
        mask_count_trace=schedule,
        rng_draw_count=rng_draw_count,
        masked_token_predictions=masked_prediction_count,
        final_grid=final_grid,
        output_key=output_key,
        mean_log_probability=mean_log_probability,
        minimum_probability=minimum_probability,
    )


def _float64_sequence_sha256(values: Sequence[float]) -> str:
    module = _contract()
    if module is not None and hasattr(module, "float64_sequence_sha256"):
        return module.float64_sequence_sha256(values)
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<d", value))
    return digest.hexdigest()


def _normalize_replayed_logits(
    value: object, *, expected_rows: int, label: str
) -> list[list[float]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be an ordered sequence of logit rows")
    if len(value) != expected_rows:
        raise ValueError(f"{label} must contain exactly {expected_rows} rows")
    normalized: list[list[float]] = []
    for row_index, raw_row in enumerate(value):
        if isinstance(raw_row, (str, bytes)) or not isinstance(raw_row, Sequence):
            raise TypeError(f"{label}[{row_index}] must be a sequence")
        if len(raw_row) != 10:
            raise ValueError(f"{label}[{row_index}] must contain ten color logits")
        row = [
            _finite_number(item, field=f"{label}[{row_index}][{color}]")
            for color, item in enumerate(raw_row)
        ]
        normalized.append(row)
    return normalized


def _float64_softmax(logits: Sequence[float]) -> tuple[list[float], list[float]]:
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    denominator = sum(exponentials)
    log_denominator = math.log(denominator)
    probabilities = [value / denominator for value in exponentials]
    log_probabilities = [value - maximum - log_denominator for value in logits]
    return probabilities, log_probabilities


def _probability_close(actual: float, expected: float) -> bool:
    return actual == expected or math.isclose(
        actual,
        expected,
        rel_tol=2e-14,
        abs_tol=float.fromhex("0x0.0000000000001p-1022"),
    )


def _inverse_cdf_float64(probabilities: Sequence[float], uniform: float) -> int:
    cumulative = 0.0
    for color, probability in enumerate(probabilities):
        cumulative += probability
        if cumulative > uniform:
            return color
    return 9


def validate_lane_trace_replay(
    trace: object,
    *,
    replayed_logits_by_step: Sequence[object],
    replayed_uniforms: Sequence[object],
) -> LaneTraceSummary:
    """Replay one trace from checkpoint logits and seed-derived CPU uniforms.

    The immutable v0.1 trace deliberately stores only a SHA-256 of raw float64
    logits.  Therefore this stronger validator accepts logits from a bound neural
    replay and uniforms from a reviewed ``torch.Generator(device='cpu')`` replay.
    It checks the raw little-endian digest, float64 softmax/log-softmax, greedy or
    inverse-CDF choice, and every stored probability.  The pool verifier below is
    the production entry point and requires both replay callbacks.
    """

    summary = validate_lane_trace(trace)
    assert isinstance(trace, dict)
    if isinstance(replayed_logits_by_step, (str, bytes)) or not isinstance(
        replayed_logits_by_step, Sequence
    ):
        raise TypeError("replayed_logits_by_step must be an ordered sequence")
    if len(replayed_logits_by_step) != _contract_value("DENOISING_STEPS"):
        raise ValueError("replayed logits must contain all 12 steps")
    if isinstance(replayed_uniforms, (str, bytes)) or not isinstance(
        replayed_uniforms, Sequence
    ):
        raise TypeError("replayed_uniforms must be an ordered sequence")
    if len(replayed_uniforms) != summary.rng_draw_count:
        raise ValueError("seeded CPU RNG replay draw count does not close")

    uniform_offset = 0
    for step_index, (step, raw_logits) in enumerate(
        zip(trace["steps"], replayed_logits_by_step), start=1
    ):
        entry = step["masked_indices_at_entry"]
        logits = _normalize_replayed_logits(
            raw_logits,
            expected_rows=len(entry),
            label=f"replayed_logits_by_step[{step_index - 1}]",
        )
        flattened = [value for row in logits for value in row]
        if _float64_sequence_sha256(flattened) != step["logits_float64_le_sha256"]:
            raise ValueError(f"replayed logits hash mismatch at step {step_index}")
        replayed_confidences: list[tuple[float, int]] = []
        for prediction, cell_logits in zip(step["predictions"], logits):
            probabilities, log_probabilities = _float64_softmax(cell_logits)
            chosen = prediction["chosen_color"]
            stored_probability = float.fromhex(prediction["probability_hex"])
            stored_log_probability = float.fromhex(prediction["log_probability_hex"])
            if not _probability_close(stored_probability, probabilities[chosen]):
                raise ValueError(
                    f"stored probability is not replayed float64 softmax at step {step_index}"
                )
            if not _probability_close(
                stored_log_probability, log_probabilities[chosen]
            ):
                raise ValueError(
                    f"stored log probability is not replayed float64 log-softmax at step {step_index}"
                )
            if trace["greedy"]:
                maximum = max(probabilities)
                expected_color = next(
                    color
                    for color, probability in enumerate(probabilities)
                    if probability == maximum
                )
            else:
                replayed_uniform = _finite_number(
                    replayed_uniforms[uniform_offset],
                    field=f"replayed_uniforms[{uniform_offset}]",
                    minimum=0.0,
                    maximum=1.0,
                )
                if replayed_uniform >= 1.0:
                    raise ValueError("replayed CPU uniform must be in [0, 1)")
                stored_uniform = float.fromhex(prediction["uniform_hex"])
                if stored_uniform.hex() != replayed_uniform.hex():
                    raise ValueError(
                        "stored uniform differs from seeded CPU RNG replay"
                    )
                expected_color = _inverse_cdf_float64(probabilities, replayed_uniform)
                uniform_offset += 1
            if chosen != expected_color:
                method = "greedy" if trace["greedy"] else "inverse-CDF"
                raise ValueError(
                    f"chosen color does not match replayed {method} choice at step {step_index}"
                )
            replayed_confidences.append(
                (probabilities[chosen], prediction["linear_index"])
            )
        retained_count = summary.mask_count_trace[step_index]
        replayed_remasked = sorted(
            linear_index
            for _, linear_index in sorted(
                replayed_confidences, key=lambda item: (item[0], item[1])
            )[:retained_count]
        )
        if step["remasked_indices"] != replayed_remasked:
            raise ValueError(
                f"re-mask set differs from replayed logits at step {step_index}"
            )
    if uniform_offset != len(replayed_uniforms):
        raise AssertionError(
            "seeded CPU RNG replay did not consume every supplied draw"
        )
    return summary


_LANE_FIELDS = {
    "schema",
    "lane_id",
    "blind_task_id",
    "test_index",
    "shape_order",
    "shape_proposal_id",
    "proposed_height",
    "proposed_width",
    "checkpoint_sha256",
    "model_semantics_version",
    "sampler_semantics_version",
    "global_lane",
    "local_lane",
    "greedy",
    "seed_u64",
    "temperature",
    "denoising_steps",
    "mask_count_trace",
    "trace_sha256",
    "output_grid",
    "output_key",
    "duplicate_class",
    "mean_log_probability",
    "minimum_probability",
    "format_status",
    "failure_code",
    "sample_equivalent_forward_calls",
    "batch_forward_call_ids",
    "masked_token_predictions",
    "cpu_sampling_ns",
    "lane_wall_time_ns",
}

_LANE_TIMING_FIELDS = {"cpu_sampling_ns", "lane_wall_time_ns"}


def lane_semantic_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key != "lane_id" and key not in _LANE_TIMING_FIELDS
    }


def make_lane_row(
    trace: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    batch_forward_call_ids: Sequence[str],
    duplicate_class: str | None = None,
    cpu_sampling_ns: int = 0,
    lane_wall_time_ns: int = 0,
) -> dict[str, Any]:
    summary = validate_lane_trace(trace)
    row: dict[str, Any] = {
        "schema": _contract_value("LANE_ROW_SCHEMA_VERSION"),
        "blind_task_id": trace["blind_task_id"],
        "test_index": trace["test_index"],
        "shape_order": trace["shape_order"],
        "shape_proposal_id": trace["shape_proposal_id"],
        "proposed_height": trace["proposed_height"],
        "proposed_width": trace["proposed_width"],
        "checkpoint_sha256": checkpoint_sha256,
        "model_semantics_version": _contract_value("MODEL_SEMANTICS_VERSION"),
        "sampler_semantics_version": _contract_value("SAMPLER_SEMANTICS_VERSION"),
        "global_lane": trace["global_lane"],
        "local_lane": trace["local_lane"],
        "greedy": trace["greedy"],
        "seed_u64": trace["seed_u64"],
        "temperature": None if trace["greedy"] else 1.0,
        "denoising_steps": _contract_value("DENOISING_STEPS"),
        "mask_count_trace": list(summary.mask_count_trace),
        "trace_sha256": lane_trace_sha256(trace),
        "output_grid": grid_to_lists(summary.final_grid),
        "output_key": summary.output_key,
        "duplicate_class": duplicate_class or summary.output_key,
        "mean_log_probability": summary.mean_log_probability,
        "minimum_probability": summary.minimum_probability,
        "format_status": "VALID",
        "failure_code": None,
        "sample_equivalent_forward_calls": _contract_value("DENOISING_STEPS"),
        "batch_forward_call_ids": list(batch_forward_call_ids),
        "masked_token_predictions": summary.masked_token_predictions,
        "cpu_sampling_ns": cpu_sampling_ns,
        "lane_wall_time_ns": lane_wall_time_ns,
    }
    row["lane_id"] = _canonical_sha256(lane_semantic_payload(row))
    return validate_lane_row(row, trace=trace)


def validate_lane_row(
    row: object, *, trace: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    payload = _exact_fields(row, _LANE_FIELDS, label="M04a lane row")
    if payload["schema"] != _contract_value("LANE_ROW_SCHEMA_VERSION"):
        raise ValueError("unsupported M04a lane row schema")
    _nonempty_string(payload["blind_task_id"], field="blind_task_id")
    _strict_int(payload["test_index"], field="test_index")
    _strict_int(payload["shape_order"], field="shape_order")
    _sha256(payload["shape_proposal_id"], field="shape_proposal_id")
    height = _strict_int(payload["proposed_height"], field="proposed_height", minimum=1)
    width = _strict_int(payload["proposed_width"], field="proposed_width", minimum=1)
    if height > 30 or width > 30:
        raise ValueError("lane proposed shape exceeds ARC bounds")
    _sha256(payload["checkpoint_sha256"], field="checkpoint_sha256")
    if payload["model_semantics_version"] != _contract_value("MODEL_SEMANTICS_VERSION"):
        raise ValueError("lane row model semantics mismatch")
    if payload["sampler_semantics_version"] != _contract_value(
        "SAMPLER_SEMANTICS_VERSION"
    ):
        raise ValueError("lane row sampler semantics mismatch")
    _strict_int(payload["global_lane"], field="global_lane")
    local_lane = _strict_int(payload["local_lane"], field="local_lane")
    if type(payload["greedy"]) is not bool:
        raise TypeError("greedy must be boolean")
    seed = _strict_int(payload["seed_u64"], field="seed_u64")
    if seed != _lane_seed(
        payload["blind_task_id"],
        payload["test_index"],
        payload["shape_proposal_id"],
        local_lane,
    ):
        raise ValueError("lane seed does not match frozen seed policy")
    if payload["greedy"]:
        if payload["temperature"] is not None:
            raise ValueError("greedy lane temperature must be null")
    elif _finite_number(payload["temperature"], field="temperature") != 1.0:
        raise ValueError("sampled lane temperature must be 1.0")
    if payload["denoising_steps"] != _contract_value("DENOISING_STEPS"):
        raise ValueError("lane denoising_steps does not match frozen sampler")
    expected_trace = list(_mask_count_trace(height * width))
    if payload["mask_count_trace"] != expected_trace:
        raise ValueError("lane mask_count_trace does not match frozen schedule")
    _sha256(payload["trace_sha256"], field="trace_sha256")
    output = as_grid(payload["output_grid"])
    if len(output) != height or len(output[0]) != width:
        raise ValueError("lane output_grid does not match proposed shape")
    if payload["output_key"] != grid_key(output):
        raise ValueError("lane output_key does not match output_grid")
    _nonempty_string(payload["duplicate_class"], field="duplicate_class")
    mean_log = _finite_number(
        payload["mean_log_probability"], field="mean_log_probability"
    )
    minimum = _finite_number(
        payload["minimum_probability"],
        field="minimum_probability",
        minimum=0.0,
        maximum=1.0,
    )
    if mean_log > 0.0:
        raise ValueError("mean_log_probability must be <= 0")
    if payload["format_status"] != "VALID" or payload["failure_code"] is not None:
        raise ValueError("a completed trace-backed M04a lane must be VALID")
    if payload["sample_equivalent_forward_calls"] != _contract_value("DENOISING_STEPS"):
        raise ValueError("lane sample-equivalent forward calls must equal 12")
    calls = payload["batch_forward_call_ids"]
    if (
        not isinstance(calls, list)
        or len(calls) != _contract_value("DENOISING_STEPS")
        or len(set(calls)) != len(calls)
    ):
        raise ValueError("lane must bind one distinct decoder batch call per step")
    for index, call_id in enumerate(calls):
        _sha256(call_id, field=f"batch_forward_call_ids[{index}]")
    expected_predictions = sum(expected_trace[: _contract_value("DENOISING_STEPS")])
    if payload["masked_token_predictions"] != expected_predictions:
        raise ValueError("lane masked-token prediction count does not close")
    for field in _LANE_TIMING_FIELDS:
        _strict_int(payload[field], field=field)
    if trace is not None:
        summary = validate_lane_trace(trace)
        linked = {
            "blind_task_id": trace["blind_task_id"],
            "test_index": trace["test_index"],
            "shape_order": trace["shape_order"],
            "shape_proposal_id": trace["shape_proposal_id"],
            "proposed_height": trace["proposed_height"],
            "proposed_width": trace["proposed_width"],
            "global_lane": trace["global_lane"],
            "local_lane": trace["local_lane"],
            "greedy": trace["greedy"],
            "seed_u64": trace["seed_u64"],
        }
        if any(payload[field] != value for field, value in linked.items()):
            raise ValueError("lane identity fields do not match linked trace")
        if payload["trace_sha256"] != lane_trace_sha256(trace):
            raise ValueError("lane trace_sha256 does not match linked trace")
        if output != summary.final_grid or payload["output_key"] != summary.output_key:
            raise ValueError("lane output does not match linked trace")
        if (
            mean_log != summary.mean_log_probability
            or minimum != summary.minimum_probability
        ):
            raise ValueError("lane probability summaries do not close from trace")
        if payload["masked_token_predictions"] != summary.masked_token_predictions:
            raise ValueError("lane masked-token count does not close from trace")
    expected_id = _canonical_sha256(lane_semantic_payload(payload))
    if payload["lane_id"] != expected_id:
        raise ValueError("lane_id does not match semantic content excluding timing")
    return payload


_CALL_MEASUREMENT_FIELDS = {
    "gpu_forward_ns",
    "wall_time_ns",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
}


def _semantic_row_payload(
    row: Mapping[str, Any], *, identity_field: str, measurement_fields: set[str]
) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key != identity_field and key not in measurement_fields
    }


def _validate_measurements(row: Mapping[str, Any], fields: Iterable[str]) -> None:
    for field in fields:
        _strict_int(row[field], field=field)
    if (
        "cuda_peak_allocated_bytes" in row
        and "cuda_peak_reserved_bytes" in row
        and row["cuda_peak_reserved_bytes"] < row["cuda_peak_allocated_bytes"]
    ):
        raise ValueError("CUDA peak reserved bytes cannot be below allocated bytes")


_GRID_DESCRIPTOR_FIELDS = {"role", "pair_index", "height", "width", "grid_key"}


def _validate_grid_descriptors(
    value: object, *, query_test_index: int
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value or len(value) % 2 != 1:
        raise ValueError("ordered_grid_descriptors must contain 2*D+1 rows")
    normalized: list[dict[str, Any]] = []
    demo_count = (len(value) - 1) // 2
    expected_roles: list[tuple[str, int]] = []
    for pair_index in range(demo_count):
        expected_roles.extend((("demo_input", pair_index), ("demo_output", pair_index)))
    expected_roles.append(("query_input", query_test_index))
    for index, raw in enumerate(value):
        row = _exact_fields(
            raw, _GRID_DESCRIPTOR_FIELDS, label=f"ordered_grid_descriptors[{index}]"
        )
        if (row["role"], row["pair_index"]) != expected_roles[index]:
            raise ValueError(
                "grid descriptors do not follow demo input/output then query order"
            )
        height = _strict_int(row["height"], field="descriptor.height", minimum=1)
        width = _strict_int(row["width"], field="descriptor.width", minimum=1)
        if height > 30 or width > 30:
            raise ValueError("grid descriptor exceeds ARC bounds")
        _sha256(row["grid_key"], field="descriptor.grid_key")
        normalized.append(dict(row))
    return normalized


_ENCODER_FIELDS = {
    "schema",
    "call_id",
    "blind_task_id",
    "test_index",
    "ordered_grid_descriptors",
    "padded_batch_shape",
    "unpadded_memory_length",
    "cache_dtype",
    "gpu_forward_ns",
    "wall_time_ns",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
}


def make_encoder_forward_row(
    *,
    blind_task_id: str,
    test_index: int,
    ordered_grid_descriptors: Sequence[Mapping[str, Any]],
    padded_batch_shape: Sequence[int],
    unpadded_memory_length: int,
    cache_dtype: str,
    gpu_forward_ns: int,
    wall_time_ns: int,
    cuda_peak_allocated_bytes: int,
    cuda_peak_reserved_bytes: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema": _contract_value("ENCODER_FORWARD_LEDGER_SCHEMA_VERSION"),
        "blind_task_id": blind_task_id,
        "test_index": test_index,
        "ordered_grid_descriptors": [dict(item) for item in ordered_grid_descriptors],
        "padded_batch_shape": list(padded_batch_shape),
        "unpadded_memory_length": unpadded_memory_length,
        "cache_dtype": cache_dtype,
        "gpu_forward_ns": gpu_forward_ns,
        "wall_time_ns": wall_time_ns,
        "cuda_peak_allocated_bytes": cuda_peak_allocated_bytes,
        "cuda_peak_reserved_bytes": cuda_peak_reserved_bytes,
    }
    row["call_id"] = _canonical_sha256(
        _semantic_row_payload(
            row, identity_field="call_id", measurement_fields=_CALL_MEASUREMENT_FIELDS
        )
    )
    return validate_encoder_forward_row(row)


def validate_encoder_forward_row(row: object) -> dict[str, Any]:
    payload = _exact_fields(row, _ENCODER_FIELDS, label="M04a encoder call row")
    if payload["schema"] != _contract_value("ENCODER_FORWARD_LEDGER_SCHEMA_VERSION"):
        raise ValueError("unsupported encoder forward ledger schema")
    _nonempty_string(payload["blind_task_id"], field="blind_task_id")
    _strict_int(payload["test_index"], field="test_index")
    _validate_grid_descriptors(
        payload["ordered_grid_descriptors"], query_test_index=payload["test_index"]
    )
    shape = payload["padded_batch_shape"]
    if not isinstance(shape, list) or not shape:
        raise TypeError("padded_batch_shape must be a non-empty integer list")
    for index, dimension in enumerate(shape):
        _strict_int(dimension, field=f"padded_batch_shape[{index}]", minimum=1)
    _strict_int(
        payload["unpadded_memory_length"], field="unpadded_memory_length", minimum=1
    )
    if payload["cache_dtype"] not in {"float16", "bfloat16", "float32"}:
        raise ValueError("unsupported encoder cache dtype")
    _validate_measurements(payload, _CALL_MEASUREMENT_FIELDS)
    expected = _canonical_sha256(
        _semantic_row_payload(
            payload,
            identity_field="call_id",
            measurement_fields=_CALL_MEASUREMENT_FIELDS,
        )
    )
    if payload["call_id"] != expected:
        raise ValueError("encoder call_id does not match semantic content")
    return payload


_CACHE_DESCRIPTOR_FIELDS = {"encoder_call_id", "cache_key", "memory_length", "dtype"}


def _validate_cache_descriptor(value: object) -> dict[str, Any]:
    descriptor = _exact_fields(
        value, _CACHE_DESCRIPTOR_FIELDS, label="cached-memory descriptor"
    )
    _sha256(descriptor["encoder_call_id"], field="cache.encoder_call_id")
    _sha256(descriptor["cache_key"], field="cache.cache_key")
    _strict_int(descriptor["memory_length"], field="cache.memory_length", minimum=1)
    if descriptor["dtype"] not in {"float16", "bfloat16", "float32"}:
        raise ValueError("unsupported cached-memory dtype")
    return descriptor


_DECODER_FIELDS = {
    "schema",
    "call_id",
    "blind_task_id",
    "test_index",
    "shape_order",
    "shape_proposal_id",
    "step",
    "batch_index_within_shape",
    "global_lanes",
    "batch_size",
    "cached_memory_descriptor",
    "gpu_forward_ns",
    "wall_time_ns",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
}


def make_decoder_forward_row(
    *,
    blind_task_id: str,
    test_index: int,
    shape_order: int,
    shape_proposal_id: str,
    step: int,
    batch_index_within_shape: int,
    global_lanes: Sequence[int],
    cached_memory_descriptor: Mapping[str, Any],
    gpu_forward_ns: int,
    wall_time_ns: int,
    cuda_peak_allocated_bytes: int,
    cuda_peak_reserved_bytes: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema": _contract_value("DECODER_FORWARD_LEDGER_SCHEMA_VERSION"),
        "blind_task_id": blind_task_id,
        "test_index": test_index,
        "shape_order": shape_order,
        "shape_proposal_id": shape_proposal_id,
        "step": step,
        "batch_index_within_shape": batch_index_within_shape,
        "global_lanes": list(global_lanes),
        "batch_size": len(global_lanes),
        "cached_memory_descriptor": dict(cached_memory_descriptor),
        "gpu_forward_ns": gpu_forward_ns,
        "wall_time_ns": wall_time_ns,
        "cuda_peak_allocated_bytes": cuda_peak_allocated_bytes,
        "cuda_peak_reserved_bytes": cuda_peak_reserved_bytes,
    }
    row["call_id"] = _canonical_sha256(
        _semantic_row_payload(
            row, identity_field="call_id", measurement_fields=_CALL_MEASUREMENT_FIELDS
        )
    )
    return validate_decoder_forward_row(row)


def validate_decoder_forward_row(row: object) -> dict[str, Any]:
    payload = _exact_fields(row, _DECODER_FIELDS, label="M04a decoder call row")
    if payload["schema"] != _contract_value("DECODER_FORWARD_LEDGER_SCHEMA_VERSION"):
        raise ValueError("unsupported decoder forward ledger schema")
    _nonempty_string(payload["blind_task_id"], field="blind_task_id")
    _strict_int(payload["test_index"], field="test_index")
    _strict_int(payload["shape_order"], field="shape_order")
    _sha256(payload["shape_proposal_id"], field="shape_proposal_id")
    step = _strict_int(payload["step"], field="step", minimum=1)
    if step > _contract_value("DENOISING_STEPS"):
        raise ValueError("decoder step exceeds frozen schedule")
    _strict_int(payload["batch_index_within_shape"], field="batch_index_within_shape")
    lanes = payload["global_lanes"]
    if (
        not isinstance(lanes, list)
        or not lanes
        or len(lanes) > _contract_value("INFERENCE_MICROBATCH")
        or any(type(lane) is not int or lane < 0 for lane in lanes)
        or lanes != list(range(lanes[0], lanes[0] + len(lanes)))
    ):
        raise ValueError(
            "decoder global_lanes must be a consecutive batch of at most 8"
        )
    if payload["batch_size"] != len(lanes):
        raise ValueError("decoder batch_size does not match global_lanes")
    _validate_cache_descriptor(payload["cached_memory_descriptor"])
    _validate_measurements(payload, _CALL_MEASUREMENT_FIELDS)
    expected = _canonical_sha256(
        _semantic_row_payload(
            payload,
            identity_field="call_id",
            measurement_fields=_CALL_MEASUREMENT_FIELDS,
        )
    )
    if payload["call_id"] != expected:
        raise ValueError("decoder call_id does not match semantic content")
    return payload


def expected_pair_cost_payload(shape_cell_counts: Sequence[int]) -> dict[str, Any]:
    if isinstance(shape_cell_counts, (str, bytes)) or not isinstance(
        shape_cell_counts, Sequence
    ):
        raise TypeError("shape_cell_counts must be an ordered sequence")
    normalized = [
        _strict_int(value, field=f"shape_cell_counts[{index}]", minimum=1)
        for index, value in enumerate(shape_cell_counts)
    ]
    if len(normalized) > _contract_value("MAX_ACCEPTED_SHAPES"):
        raise ValueError("at most four accepted shapes are allowed")
    if any(value > 900 for value in normalized):
        raise ValueError("shape cell count exceeds 30x30")
    allocations = _shape_lane_allocations(len(normalized))
    traces = [_mask_count_trace(value) for value in normalized]
    predictions = [sum(trace[: _contract_value("DENOISING_STEPS")]) for trace in traces]
    raw_lanes = sum(allocations)
    encoder_calls = 1 if normalized else 0
    decoder_calls = _decoder_batch_calls(len(normalized))
    return {
        "accepted_shape_count": len(normalized),
        "shape_cell_counts": normalized,
        "shape_lane_allocations": list(allocations),
        "raw_lanes": raw_lanes,
        "sample_equivalent_forward_calls": raw_lanes
        * _contract_value("DENOISING_STEPS"),
        "encoder_batch_calls": encoder_calls,
        "decoder_batch_calls": decoder_calls,
        "total_actual_batch_calls": encoder_calls + decoder_calls,
        "mask_count_traces_per_shape": [list(trace) for trace in traces],
        "masked_token_predictions_per_shape_lane": predictions,
        "masked_token_predictions_total": sum(
            allocation * prediction
            for allocation, prediction in zip(allocations, predictions)
        ),
    }


_PAIR_MEASUREMENT_FIELDS = {
    "pair_wall_time_ns",
    "h2d_ns",
    "d2h_ns",
    "encoder_gpu_ns",
    "decoder_gpu_ns",
    "cpu_sampling_ns",
    "hashing_ns",
    "serialization_ns",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
}

_PAIR_FIELDS = {
    "schema",
    "pair_cost_id",
    "blind_task_id",
    "test_index",
    "accepted_shapes",
    "accepted_shape_count",
    "shape_lane_allocations",
    "raw_lanes",
    "sample_equivalent_forward_calls",
    "encoder_batch_calls",
    "decoder_batch_calls",
    "total_actual_batch_calls",
    "mask_trace_table_sha256",
    "masked_token_predictions_total",
    "format_valid",
    "format_invalid",
    "unique_outputs",
    "duplicate_outputs",
    "candidate_rows",
    "lane_ids",
    "encoder_call_ids",
    "decoder_call_ids",
    "status",
    "failure_code",
    "pair_wall_time_ns",
    "h2d_ns",
    "d2h_ns",
    "encoder_gpu_ns",
    "decoder_gpu_ns",
    "cpu_sampling_ns",
    "hashing_ns",
    "serialization_ns",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
}


def _normalize_accepted_shapes(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > _contract_value(
        "MAX_ACCEPTED_SHAPES"
    ):
        raise TypeError("accepted_shapes must be a list of at most four shapes")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for index, raw in enumerate(value):
        shape = _exact_fields(
            raw,
            {"shape_order", "shape_proposal_id", "proposed_height", "proposed_width"},
            label=f"accepted_shapes[{index}]",
        )
        if shape["shape_order"] != index:
            raise ValueError("shape_order must be consecutive")
        _sha256(shape["shape_proposal_id"], field="shape_proposal_id")
        height = _strict_int(
            shape["proposed_height"], field="proposed_height", minimum=1
        )
        width = _strict_int(shape["proposed_width"], field="proposed_width", minimum=1)
        if height > 30 or width > 30 or (height, width) in seen:
            raise ValueError("accepted shapes must be distinct and within ARC bounds")
        seen.add((height, width))
        normalized.append(dict(shape))
    return normalized


def _validate_sha_list(value: object, *, field: str, length: int) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{field} must contain {length} distinct SHA-256 IDs")
    for index, item in enumerate(value):
        _sha256(item, field=f"{field}[{index}]")
    return value


def pair_cost_semantic_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return _semantic_row_payload(
        row,
        identity_field="pair_cost_id",
        measurement_fields=_PAIR_MEASUREMENT_FIELDS,
    )


def make_pair_cost_row(
    *,
    blind_task_id: str,
    test_index: int,
    accepted_shapes: Sequence[Mapping[str, Any]],
    lane_ids: Sequence[str],
    encoder_call_ids: Sequence[str],
    decoder_call_ids: Sequence[str],
    format_invalid: int = 0,
    unique_outputs: int | None = None,
    pair_wall_time_ns: int = 0,
    h2d_ns: int = 0,
    d2h_ns: int = 0,
    encoder_gpu_ns: int = 0,
    decoder_gpu_ns: int = 0,
    cpu_sampling_ns: int = 0,
    hashing_ns: int = 0,
    serialization_ns: int = 0,
    cuda_peak_allocated_bytes: int = 0,
    cuda_peak_reserved_bytes: int = 0,
) -> dict[str, Any]:
    shapes = _normalize_accepted_shapes([dict(shape) for shape in accepted_shapes])
    shape_cells = [
        shape["proposed_height"] * shape["proposed_width"] for shape in shapes
    ]
    expected = expected_pair_cost_payload(shape_cells)
    format_invalid = _strict_int(format_invalid, field="format_invalid")
    format_valid = expected["raw_lanes"] - format_invalid
    if format_valid < 0:
        raise ValueError("format_invalid exceeds raw lanes")
    if unique_outputs is None:
        unique_outputs = format_valid
    unique_outputs = _strict_int(unique_outputs, field="unique_outputs")
    if unique_outputs > format_valid:
        raise ValueError("unique_outputs exceeds format-valid lanes")
    trace_table_sha = _canonical_sha256(
        {
            "shape_cell_counts": expected["shape_cell_counts"],
            "shape_lane_allocations": expected["shape_lane_allocations"],
            "mask_count_traces_per_shape": expected["mask_count_traces_per_shape"],
        }
    )
    row: dict[str, Any] = {
        "schema": _contract_value("PAIR_COST_SCHEMA_VERSION"),
        "blind_task_id": blind_task_id,
        "test_index": test_index,
        "accepted_shapes": shapes,
        "accepted_shape_count": expected["accepted_shape_count"],
        "shape_lane_allocations": expected["shape_lane_allocations"],
        "raw_lanes": expected["raw_lanes"],
        "sample_equivalent_forward_calls": expected["sample_equivalent_forward_calls"],
        "encoder_batch_calls": expected["encoder_batch_calls"],
        "decoder_batch_calls": expected["decoder_batch_calls"],
        "total_actual_batch_calls": expected["total_actual_batch_calls"],
        "mask_trace_table_sha256": trace_table_sha,
        "masked_token_predictions_total": expected["masked_token_predictions_total"],
        "format_valid": format_valid,
        "format_invalid": format_invalid,
        "unique_outputs": unique_outputs,
        "duplicate_outputs": format_valid - unique_outputs,
        "candidate_rows": expected["raw_lanes"],
        "lane_ids": list(lane_ids),
        "encoder_call_ids": list(encoder_call_ids),
        "decoder_call_ids": list(decoder_call_ids),
        "status": SHAPE_READY if shapes else NO_SHAPE_PROPOSAL,
        "failure_code": None if shapes else NO_SHAPE_PROPOSAL,
        "pair_wall_time_ns": pair_wall_time_ns,
        "h2d_ns": h2d_ns,
        "d2h_ns": d2h_ns,
        "encoder_gpu_ns": encoder_gpu_ns,
        "decoder_gpu_ns": decoder_gpu_ns,
        "cpu_sampling_ns": cpu_sampling_ns,
        "hashing_ns": hashing_ns,
        "serialization_ns": serialization_ns,
        "cuda_peak_allocated_bytes": cuda_peak_allocated_bytes,
        "cuda_peak_reserved_bytes": cuda_peak_reserved_bytes,
    }
    row["pair_cost_id"] = _canonical_sha256(pair_cost_semantic_payload(row))
    return validate_pair_cost_row(row)


def validate_pair_cost_row(row: object) -> dict[str, Any]:
    payload = _exact_fields(row, _PAIR_FIELDS, label="M04a pair-cost row")
    if payload["schema"] != _contract_value("PAIR_COST_SCHEMA_VERSION"):
        raise ValueError("unsupported M04a pair-cost schema")
    _nonempty_string(payload["blind_task_id"], field="blind_task_id")
    _strict_int(payload["test_index"], field="test_index")
    shapes = _normalize_accepted_shapes(payload["accepted_shapes"])
    expected = expected_pair_cost_payload(
        [shape["proposed_height"] * shape["proposed_width"] for shape in shapes]
    )
    scalar_fields = (
        "accepted_shape_count",
        "shape_lane_allocations",
        "raw_lanes",
        "sample_equivalent_forward_calls",
        "encoder_batch_calls",
        "decoder_batch_calls",
        "total_actual_batch_calls",
        "masked_token_predictions_total",
    )
    for field in scalar_fields:
        if payload[field] != expected[field]:
            raise ValueError(f"pair-cost {field} does not close")
    expected_trace_sha = _canonical_sha256(
        {
            "shape_cell_counts": expected["shape_cell_counts"],
            "shape_lane_allocations": expected["shape_lane_allocations"],
            "mask_count_traces_per_shape": expected["mask_count_traces_per_shape"],
        }
    )
    if payload["mask_trace_table_sha256"] != expected_trace_sha:
        raise ValueError("pair-cost mask trace table does not close")
    for field in (
        "format_valid",
        "format_invalid",
        "unique_outputs",
        "duplicate_outputs",
        "candidate_rows",
    ):
        _strict_int(payload[field], field=field)
    if payload["format_valid"] + payload["format_invalid"] != expected["raw_lanes"]:
        raise ValueError("format validity counts do not close to raw lanes")
    if (
        payload["unique_outputs"] + payload["duplicate_outputs"]
        != payload["format_valid"]
    ):
        raise ValueError("deduplication counts do not close to format-valid lanes")
    if payload["candidate_rows"] != expected["raw_lanes"]:
        raise ValueError("candidate row count does not close to raw lanes")
    _validate_sha_list(
        payload["lane_ids"], field="lane_ids", length=expected["raw_lanes"]
    )
    _validate_sha_list(
        payload["encoder_call_ids"],
        field="encoder_call_ids",
        length=expected["encoder_batch_calls"],
    )
    _validate_sha_list(
        payload["decoder_call_ids"],
        field="decoder_call_ids",
        length=expected["decoder_batch_calls"],
    )
    if shapes:
        if payload["status"] != SHAPE_READY or payload["failure_code"] is not None:
            raise ValueError("shape-supported pair must complete with SHAPE_READY")
    elif (
        payload["status"] != NO_SHAPE_PROPOSAL
        or payload["failure_code"] != NO_SHAPE_PROPOSAL
    ):
        raise ValueError("K=0 pair must record NO_SHAPE_PROPOSAL")
    _validate_measurements(payload, _PAIR_MEASUREMENT_FIELDS)
    for component in _PAIR_MEASUREMENT_FIELDS - {
        "pair_wall_time_ns",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
    }:
        if payload[component] > payload["pair_wall_time_ns"]:
            raise ValueError(f"{component} exceeds pair endpoint wall time")
    expected_id = _canonical_sha256(pair_cost_semantic_payload(payload))
    if payload["pair_cost_id"] != expected_id:
        raise ValueError("pair_cost_id does not match semantic content")
    return payload


_SETUP_MEASUREMENT_FIELDS = {
    "model_construction_ns",
    "checkpoint_load_ns",
    "warmup_ns",
    "setup_wall_time_ns",
    "gpu_warmup_ns",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
}

_SETUP_FIELDS = {
    "schema",
    "setup_id",
    "model_semantics_version",
    "sampler_semantics_version",
    "checkpoint_sha256",
    "warmup_descriptor",
    *_SETUP_MEASUREMENT_FIELDS,
}


def pool_setup_semantic_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return _semantic_row_payload(
        row,
        identity_field="setup_id",
        measurement_fields=_SETUP_MEASUREMENT_FIELDS,
    )


def make_pool_setup_cost_row(
    *,
    checkpoint_sha256: str,
    warmup_descriptor: Mapping[str, Any],
    model_construction_ns: int,
    checkpoint_load_ns: int,
    warmup_ns: int,
    setup_wall_time_ns: int,
    gpu_warmup_ns: int,
    cuda_peak_allocated_bytes: int,
    cuda_peak_reserved_bytes: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema": _contract_value("POOL_SETUP_COST_SCHEMA_VERSION"),
        "model_semantics_version": _contract_value("MODEL_SEMANTICS_VERSION"),
        "sampler_semantics_version": _contract_value("SAMPLER_SEMANTICS_VERSION"),
        "checkpoint_sha256": checkpoint_sha256,
        "warmup_descriptor": dict(warmup_descriptor),
        "model_construction_ns": model_construction_ns,
        "checkpoint_load_ns": checkpoint_load_ns,
        "warmup_ns": warmup_ns,
        "setup_wall_time_ns": setup_wall_time_ns,
        "gpu_warmup_ns": gpu_warmup_ns,
        "cuda_peak_allocated_bytes": cuda_peak_allocated_bytes,
        "cuda_peak_reserved_bytes": cuda_peak_reserved_bytes,
    }
    row["setup_id"] = _canonical_sha256(pool_setup_semantic_payload(row))
    return validate_pool_setup_cost_row(row)


def validate_pool_setup_cost_row(row: object) -> dict[str, Any]:
    payload = _exact_fields(row, _SETUP_FIELDS, label="M04a pool setup-cost row")
    if payload["schema"] != _contract_value("POOL_SETUP_COST_SCHEMA_VERSION"):
        raise ValueError("unsupported pool setup-cost schema")
    if payload["model_semantics_version"] != _contract_value("MODEL_SEMANTICS_VERSION"):
        raise ValueError("pool setup model semantics mismatch")
    if payload["sampler_semantics_version"] != _contract_value(
        "SAMPLER_SEMANTICS_VERSION"
    ):
        raise ValueError("pool setup sampler semantics mismatch")
    _sha256(payload["checkpoint_sha256"], field="checkpoint_sha256")
    if not isinstance(payload["warmup_descriptor"], dict):
        raise TypeError("warmup_descriptor must be a JSON object")
    _validate_json_value(payload["warmup_descriptor"], label="warmup_descriptor")
    _validate_measurements(payload, _SETUP_MEASUREMENT_FIELDS)
    nonoverlap = (
        payload["model_construction_ns"]
        + payload["checkpoint_load_ns"]
        + payload["warmup_ns"]
    )
    if payload["setup_wall_time_ns"] != nonoverlap:
        raise ValueError(
            "setup wall time must close from construction, load, and warmup"
        )
    if payload["gpu_warmup_ns"] > payload["warmup_ns"]:
        raise ValueError("GPU warmup time exceeds warmup wall time")
    expected = _canonical_sha256(pool_setup_semantic_payload(payload))
    if payload["setup_id"] != expected:
        raise ValueError("setup_id does not match semantic content")
    return payload


def validate_pair_evidence(
    pair_row: object,
    *,
    lane_rows: Sequence[Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]],
    encoder_rows: Sequence[Mapping[str, Any]],
    decoder_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Close one pair across lanes, traces, calls, counts, and measured GPU time."""

    pair = validate_pair_cost_row(pair_row)
    if len(lane_rows) != len(traces):
        raise ValueError("one trace is required for every lane row")
    lanes = tuple(
        validate_lane_row(row, trace=trace) for row, trace in zip(lane_rows, traces)
    )
    encoders = tuple(validate_encoder_forward_row(row) for row in encoder_rows)
    decoders = tuple(validate_decoder_forward_row(row) for row in decoder_rows)
    if [lane["lane_id"] for lane in lanes] != pair["lane_ids"]:
        raise ValueError("pair lane_ids do not preserve canonical lane row order")
    if [row["call_id"] for row in encoders] != pair["encoder_call_ids"]:
        raise ValueError("pair encoder_call_ids do not match encoder ledger")
    if [row["call_id"] for row in decoders] != pair["decoder_call_ids"]:
        raise ValueError("pair decoder_call_ids do not match decoder ledger")
    task_id = pair["blind_task_id"]
    test_index = pair["test_index"]
    for row in (*lanes, *encoders, *decoders):
        if row["blind_task_id"] != task_id or row["test_index"] != test_index:
            raise ValueError("pair evidence contains a row from another task/test pair")

    allocations = pair["shape_lane_allocations"]
    expected_lane_coordinates: list[tuple[int, int, int, bool]] = []
    global_offset = 0
    for shape_order, allocation in enumerate(allocations):
        for local_lane in range(allocation):
            expected_lane_coordinates.append(
                (shape_order, local_lane, global_offset + local_lane, local_lane == 0)
            )
        global_offset += allocation
    actual_lane_coordinates = [
        (
            lane["shape_order"],
            lane["local_lane"],
            lane["global_lane"],
            lane["greedy"],
        )
        for lane in lanes
    ]
    if actual_lane_coordinates != expected_lane_coordinates:
        raise ValueError(
            "lane rows do not follow frozen shape/local/global allocation order"
        )
    for lane in lanes:
        shape = pair["accepted_shapes"][lane["shape_order"]]
        if any(
            lane[field] != shape[field]
            for field in (
                "shape_order",
                "shape_proposal_id",
                "proposed_height",
                "proposed_width",
            )
        ):
            raise ValueError("lane row does not bind its accepted shape")
        if lane["duplicate_class"] != lane["output_key"]:
            raise ValueError("duplicate_class must be the canonical output_key")

    if len(encoders) != (1 if allocations else 0):
        raise ValueError(
            "encoder ledger must contain exactly one call for K>0 and none for K=0"
        )
    expected_decoder_coordinates: list[tuple[int, str, int, int, list[int]]] = []
    global_offset = 0
    microbatch = _contract_value("INFERENCE_MICROBATCH")
    for shape_order, allocation in enumerate(allocations):
        proposal_id = pair["accepted_shapes"][shape_order]["shape_proposal_id"]
        for step in range(1, _contract_value("DENOISING_STEPS") + 1):
            for batch_index, start in enumerate(range(0, allocation, microbatch)):
                size = min(microbatch, allocation - start)
                global_lanes = list(
                    range(global_offset + start, global_offset + start + size)
                )
                expected_decoder_coordinates.append(
                    (shape_order, proposal_id, step, batch_index, global_lanes)
                )
        global_offset += allocation
    actual_decoder_coordinates = [
        (
            row["shape_order"],
            row["shape_proposal_id"],
            row["step"],
            row["batch_index_within_shape"],
            row["global_lanes"],
        )
        for row in decoders
    ]
    if actual_decoder_coordinates != expected_decoder_coordinates:
        raise ValueError(
            "decoder ledger does not follow frozen shape/step/microbatch order"
        )
    if encoders:
        encoder_call_id = encoders[0]["call_id"]
        if any(
            row["cached_memory_descriptor"]["encoder_call_id"] != encoder_call_id
            or row["cached_memory_descriptor"]["memory_length"]
            != encoders[0]["unpadded_memory_length"]
            or row["cached_memory_descriptor"]["dtype"] != encoders[0]["cache_dtype"]
            for row in decoders
        ):
            raise ValueError(
                "decoder cached-memory descriptors do not bind encoder cache"
            )

    calls_by_lane: dict[int, list[str]] = {lane["global_lane"]: [] for lane in lanes}
    for decoder in decoders:
        for global_lane in decoder["global_lanes"]:
            calls_by_lane[global_lane].append(decoder["call_id"])
    for lane in lanes:
        if lane["batch_forward_call_ids"] != calls_by_lane[lane["global_lane"]]:
            raise ValueError("lane decoder call IDs do not close from batch ledger")

    valid_lanes = [lane for lane in lanes if lane["format_status"] == "VALID"]
    unique_outputs = len({lane["output_key"] for lane in valid_lanes})
    if pair["format_valid"] != len(valid_lanes) or pair["format_invalid"] != (
        len(lanes) - len(valid_lanes)
    ):
        raise ValueError("pair format counts do not close from lane rows")
    if pair["unique_outputs"] != unique_outputs or pair["duplicate_outputs"] != (
        len(valid_lanes) - unique_outputs
    ):
        raise ValueError("pair deduplication counts do not close from lane outputs")
    if pair["masked_token_predictions_total"] != sum(
        lane["masked_token_predictions"] for lane in lanes
    ):
        raise ValueError("pair masked-token total does not close from lanes")
    if pair["encoder_gpu_ns"] != sum(row["gpu_forward_ns"] for row in encoders):
        raise ValueError("pair encoder GPU time does not close from encoder ledger")
    if pair["decoder_gpu_ns"] != sum(row["gpu_forward_ns"] for row in decoders):
        raise ValueError("pair decoder GPU time does not close from decoder ledger")
    return pair


_POOL_REPLAY_RECEIPT_FIELDS = {
    "schema",
    "receipt_id",
    "status",
    "checkpoint_sha256",
    "blind_shape_sidecar_id",
    "blind_shape_rows_sha256",
    "case_set_id",
    "pair_order_sha256",
    "pair_cost_ids_sha256",
    "trace_sha256s_sha256",
    "lane_ids_sha256",
    "encoder_call_ids_sha256",
    "decoder_call_ids_sha256",
    "pair_count",
    "trace_count",
    "lane_count",
    "encoder_call_count",
    "decoder_call_count",
    "masked_token_predictions",
    "verifier_source_sha256",
    "raw_logit_hashes_verified",
    "float64_probabilities_verified",
    "seeded_cpu_rng_verified",
    "bf16_cache_identity_verified",
}


def _pool_replay_receipt_payload(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in receipt.items() if key != "receipt_id"}


def validate_pool_replay_receipt(receipt: object) -> dict[str, Any]:
    payload = _exact_fields(
        receipt, _POOL_REPLAY_RECEIPT_FIELDS, label="M04a pool replay receipt"
    )
    if payload["schema"] != POOL_REPLAY_RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported pool replay receipt schema")
    if payload["status"] != "VERIFIED":
        raise ValueError("pool replay receipt status must be VERIFIED")
    for field in (
        "receipt_id",
        "checkpoint_sha256",
        "blind_shape_sidecar_id",
        "blind_shape_rows_sha256",
        "case_set_id",
        "pair_order_sha256",
        "pair_cost_ids_sha256",
        "trace_sha256s_sha256",
        "lane_ids_sha256",
        "encoder_call_ids_sha256",
        "decoder_call_ids_sha256",
        "verifier_source_sha256",
    ):
        _sha256(payload[field], field=field)
    for field in (
        "pair_count",
        "trace_count",
        "lane_count",
        "encoder_call_count",
        "decoder_call_count",
        "masked_token_predictions",
    ):
        _strict_int(payload[field], field=field)
    for field in (
        "raw_logit_hashes_verified",
        "float64_probabilities_verified",
        "seeded_cpu_rng_verified",
        "bf16_cache_identity_verified",
    ):
        if payload[field] is not True:
            raise ValueError(f"complete replay receipt requires {field}=true")
    if payload["trace_count"] != payload["lane_count"]:
        raise ValueError("replay receipt requires one trace per lane")
    if payload["receipt_id"] != _canonical_sha256(
        _pool_replay_receipt_payload(payload)
    ):
        raise ValueError("pool replay receipt_id does not match semantic content")
    return payload


def _rows_for_pair(
    rows: Sequence[Mapping[str, Any]], task_id: str, test_index: int
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        row
        for row in rows
        if row.get("blind_task_id") == task_id and row.get("test_index") == test_index
    )


def validate_pool_evidence(
    *,
    shape_sidecar: BlindInputShapeSidecar,
    checkpoint_sha256: str,
    setup_row: Mapping[str, Any],
    pair_rows: Sequence[Mapping[str, Any]],
    lane_rows: Sequence[Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]],
    encoder_rows: Sequence[Mapping[str, Any]],
    decoder_rows: Sequence[Mapping[str, Any]],
    replay_decoder_call: Callable[[Mapping[str, Any]], Mapping[int, object]],
    replay_cpu_uniforms: Callable[[int, int], Sequence[object]],
    verifier_source_sha256: str,
    fixture_mode: bool = False,
) -> dict[str, Any]:
    """Verify the entire pool against one checkpoint and return a replay receipt.

    ``replay_decoder_call`` must execute the recorded decoder call from the bound
    checkpoint and return ``{global_lane: masked_float64_logits}``.  The separate
    ``replay_cpu_uniforms`` callback must instantiate the frozen CPU torch generator
    from ``seed_u64`` and return its ordered float64 draws.  Keeping these callbacks
    outside this pure module preserves its no-PyTorch import boundary while making a
    complete production receipt impossible without neural replay.
    """

    if type(fixture_mode) is not bool:
        raise TypeError("fixture_mode must be boolean")
    if not fixture_mode:
        raise ValueError(
            "callback-based replay is fixture-only; production replay must use the "
            "checkpoint-loading neural verifier"
        )
    if not isinstance(shape_sidecar, BlindInputShapeSidecar):
        raise TypeError("shape_sidecar must be a BlindInputShapeSidecar")
    if shape_sidecar.fixture_mode and not fixture_mode:
        raise ValueError("fixture sidecar requires explicit fixture_mode=True")
    sidecar = validate_blind_input_shape_sidecar(
        shape_sidecar.manifest,
        shape_sidecar.rows,
        fixture_mode=fixture_mode,
    )
    checkpoint = _sha256(checkpoint_sha256, field="checkpoint_sha256")
    verifier_sha = _sha256(verifier_source_sha256, field="verifier_source_sha256")
    setup = validate_pool_setup_cost_row(setup_row)
    if setup["checkpoint_sha256"] != checkpoint:
        raise ValueError("pool setup does not bind the expected checkpoint")
    if not callable(replay_decoder_call) or not callable(replay_cpu_uniforms):
        raise TypeError("pool replay requires decoder and seeded CPU RNG callbacks")
    for name, value in (
        ("pair_rows", pair_rows),
        ("lane_rows", lane_rows),
        ("traces", traces),
        ("encoder_rows", encoder_rows),
        ("decoder_rows", decoder_rows),
    ):
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"{name} must be an ordered sequence")

    expected_pairs = [(row["blind_task_id"], row["test_index"]) for row in sidecar.rows]
    normalized_pairs = tuple(validate_pair_cost_row(row) for row in pair_rows)
    if [
        (row["blind_task_id"], row["test_index"]) for row in normalized_pairs
    ] != expected_pairs:
        raise ValueError(
            "pool pair rows do not preserve the frozen blind sidecar order"
        )

    normalized_lanes = tuple(validate_lane_row(row) for row in lane_rows)
    normalized_encoders = tuple(
        validate_encoder_forward_row(row) for row in encoder_rows
    )
    normalized_decoders = tuple(
        validate_decoder_forward_row(row) for row in decoder_rows
    )
    if len(traces) != len(normalized_lanes):
        raise ValueError("pool requires one trace for every lane")
    normalized_traces = tuple(dict(trace) for trace in traces)
    for trace in normalized_traces:
        validate_lane_trace(trace)

    expected_lane_pair_order = [
        pair
        for pair, pair_row in zip(expected_pairs, normalized_pairs)
        for _ in range(pair_row["raw_lanes"])
    ]
    if [
        (row["blind_task_id"], row["test_index"]) for row in normalized_lanes
    ] != expected_lane_pair_order:
        raise ValueError("pool lane rows do not preserve pair/lane order")
    if [
        (row["blind_task_id"], row["test_index"]) for row in normalized_traces
    ] != expected_lane_pair_order:
        raise ValueError("pool traces do not preserve pair/lane order")

    replayed_steps: dict[tuple[str, int, int], list[object | None]] = {
        (trace["blind_task_id"], trace["test_index"], trace["global_lane"]): [None]
        * _contract_value("DENOISING_STEPS")
        for trace in normalized_traces
    }
    masked_token_predictions = 0
    bf16_pair_count = 0
    for sidecar_row, pair in zip(sidecar.rows, normalized_pairs):
        task_id = pair["blind_task_id"]
        test_index = pair["test_index"]
        if pair["accepted_shapes"] != sidecar_row["accepted_shapes"]:
            raise ValueError("pair accepted shapes differ from frozen blind sidecar")
        pair_lanes = _rows_for_pair(normalized_lanes, task_id, test_index)
        pair_traces = _rows_for_pair(normalized_traces, task_id, test_index)
        pair_encoders = _rows_for_pair(normalized_encoders, task_id, test_index)
        pair_decoders = _rows_for_pair(normalized_decoders, task_id, test_index)
        validate_pair_evidence(
            pair,
            lane_rows=pair_lanes,
            traces=pair_traces,
            encoder_rows=pair_encoders,
            decoder_rows=pair_decoders,
        )
        if any(row["checkpoint_sha256"] != checkpoint for row in pair_lanes):
            raise ValueError("pool lane binds a checkpoint other than the expected one")
        if pair_encoders:
            encoder = pair_encoders[0]
            if encoder["cache_dtype"] != "bfloat16":
                raise ValueError("production task-memory cache must be BF16")
            if encoder["ordered_grid_descriptors"][-1]["grid_key"] != grid_key(
                as_grid(sidecar_row["test_input"])
            ):
                raise ValueError("encoder query descriptor differs from blind sidecar")
            expected_cache = {
                "encoder_call_id": encoder["call_id"],
                "cache_key": _canonical_sha256(
                    {
                        "checkpoint_sha256": checkpoint,
                        "blind_task_id": task_id,
                        "test_index": test_index,
                        "memory_length": encoder["unpadded_memory_length"],
                        "dtype": "bfloat16",
                    }
                ),
                "memory_length": encoder["unpadded_memory_length"],
                "dtype": "bfloat16",
            }
            if any(
                decoder["cached_memory_descriptor"] != expected_cache
                for decoder in pair_decoders
            ):
                raise ValueError(
                    "every decoder call must use the identical bound BF16 task cache"
                )
            bf16_pair_count += 1
        elif pair_decoders:
            raise ValueError("decoder calls cannot exist without a pair encoder cache")

        trace_by_lane = {trace["global_lane"]: trace for trace in pair_traces}
        for decoder in pair_decoders:
            replayed = replay_decoder_call(decoder)
            if not isinstance(replayed, Mapping) or set(replayed) != set(
                decoder["global_lanes"]
            ):
                raise ValueError(
                    "decoder replay must return exactly every recorded global lane"
                )
            for global_lane in decoder["global_lanes"]:
                trace = trace_by_lane[global_lane]
                step_index = decoder["step"] - 1
                if (
                    replayed_steps[(task_id, test_index, global_lane)][step_index]
                    is not None
                ):
                    raise ValueError(
                        "decoder replay supplied a lane step more than once"
                    )
                expected_rows = len(
                    trace["steps"][step_index]["masked_indices_at_entry"]
                )
                replayed_steps[(task_id, test_index, global_lane)][step_index] = (
                    _normalize_replayed_logits(
                        replayed[global_lane],
                        expected_rows=expected_rows,
                        label=(
                            f"decoder replay {decoder['call_id']} lane {global_lane}"
                        ),
                    )
                )

        for trace in pair_traces:
            key = (task_id, test_index, trace["global_lane"])
            step_logits = replayed_steps[key]
            if any(value is None for value in step_logits):
                raise ValueError("neural replay did not cover every trace step")
            structural = validate_lane_trace(trace)
            uniforms = replay_cpu_uniforms(trace["seed_u64"], structural.rng_draw_count)
            validate_lane_trace_replay(
                trace,
                replayed_logits_by_step=step_logits,
                replayed_uniforms=uniforms,
            )
            masked_token_predictions += structural.masked_token_predictions

    pair_order_payload = [
        {
            "blind_task_id": row["blind_task_id"],
            "test_index": row["test_index"],
            "row_id": row["row_id"],
        }
        for row in sidecar.rows
    ]
    receipt: dict[str, Any] = {
        "schema": POOL_REPLAY_RECEIPT_SCHEMA_VERSION,
        "status": "VERIFIED",
        "checkpoint_sha256": checkpoint,
        "blind_shape_sidecar_id": sidecar.manifest["sidecar_id"],
        "blind_shape_rows_sha256": sidecar.manifest["rows_sha256"],
        "case_set_id": sidecar.manifest["case_set_id"],
        "pair_order_sha256": _canonical_sha256(pair_order_payload),
        "pair_cost_ids_sha256": _canonical_sha256(
            [row["pair_cost_id"] for row in normalized_pairs]
        ),
        "trace_sha256s_sha256": _canonical_sha256(
            [lane_trace_sha256(row) for row in normalized_traces]
        ),
        "lane_ids_sha256": _canonical_sha256(
            [row["lane_id"] for row in normalized_lanes]
        ),
        "encoder_call_ids_sha256": _canonical_sha256(
            [row["call_id"] for row in normalized_encoders]
        ),
        "decoder_call_ids_sha256": _canonical_sha256(
            [row["call_id"] for row in normalized_decoders]
        ),
        "pair_count": len(normalized_pairs),
        "trace_count": len(normalized_traces),
        "lane_count": len(normalized_lanes),
        "encoder_call_count": len(normalized_encoders),
        "decoder_call_count": len(normalized_decoders),
        "masked_token_predictions": masked_token_predictions,
        "verifier_source_sha256": verifier_sha,
        "raw_logit_hashes_verified": True,
        "float64_probabilities_verified": True,
        "seeded_cpu_rng_verified": True,
        "bf16_cache_identity_verified": bf16_pair_count
        == sum(row["accepted_shape_count"] > 0 for row in normalized_pairs),
    }
    receipt["receipt_id"] = _canonical_sha256(_pool_replay_receipt_payload(receipt))
    return validate_pool_replay_receipt(receipt)


_POOL_SUMMARY_MEASUREMENT_FIELDS = {
    "setup_wall_time_ns",
    "sequential_pair_wall_time_ns",
    "finalization_wall_time_ns",
    "pool_wall_time_ns",
    "encoder_gpu_ns",
    "decoder_gpu_ns",
    "total_gpu_ns",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
}

_POOL_SUMMARY_FIELDS = {
    "schema",
    "summary_id",
    "setup_id",
    "pair_cost_ids",
    "pair_count",
    *_POOL_SUMMARY_MEASUREMENT_FIELDS,
}


def _pool_summary_semantic_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return _semantic_row_payload(
        row,
        identity_field="summary_id",
        measurement_fields=_POOL_SUMMARY_MEASUREMENT_FIELDS,
    )


def make_pool_cost_summary(
    setup_row: Mapping[str, Any],
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    finalization_wall_time_ns: int,
    cuda_peak_allocated_bytes: int,
    cuda_peak_reserved_bytes: int,
) -> dict[str, Any]:
    setup = validate_pool_setup_cost_row(setup_row)
    pairs = tuple(validate_pair_cost_row(row) for row in pair_rows)
    row: dict[str, Any] = {
        "schema": POOL_COST_SUMMARY_SCHEMA_VERSION,
        "setup_id": setup["setup_id"],
        "pair_cost_ids": [pair["pair_cost_id"] for pair in pairs],
        "pair_count": len(pairs),
        "setup_wall_time_ns": setup["setup_wall_time_ns"],
        "sequential_pair_wall_time_ns": sum(
            pair["pair_wall_time_ns"] for pair in pairs
        ),
        "finalization_wall_time_ns": finalization_wall_time_ns,
        "pool_wall_time_ns": setup["setup_wall_time_ns"]
        + sum(pair["pair_wall_time_ns"] for pair in pairs)
        + finalization_wall_time_ns,
        "encoder_gpu_ns": sum(pair["encoder_gpu_ns"] for pair in pairs),
        "decoder_gpu_ns": sum(pair["decoder_gpu_ns"] for pair in pairs),
        "total_gpu_ns": sum(
            pair["encoder_gpu_ns"] + pair["decoder_gpu_ns"] for pair in pairs
        ),
        "cuda_peak_allocated_bytes": cuda_peak_allocated_bytes,
        "cuda_peak_reserved_bytes": cuda_peak_reserved_bytes,
    }
    row["summary_id"] = _canonical_sha256(_pool_summary_semantic_payload(row))
    return validate_pool_cost_summary(row, setup_row=setup, pair_rows=pairs)


def validate_pool_cost_summary(
    row: object,
    *,
    setup_row: Mapping[str, Any],
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = _exact_fields(row, _POOL_SUMMARY_FIELDS, label="M04a pool-cost summary")
    if payload["schema"] != POOL_COST_SUMMARY_SCHEMA_VERSION:
        raise ValueError("unsupported M04a pool-cost summary schema")
    setup = validate_pool_setup_cost_row(setup_row)
    pairs = tuple(validate_pair_cost_row(item) for item in pair_rows)
    if payload["setup_id"] != setup["setup_id"]:
        raise ValueError("pool summary setup_id mismatch")
    if payload["pair_cost_ids"] != [pair["pair_cost_id"] for pair in pairs]:
        raise ValueError("pool summary pair IDs do not preserve pair order")
    if payload["pair_count"] != len(pairs):
        raise ValueError("pool summary pair_count mismatch")
    expected = {
        "setup_wall_time_ns": setup["setup_wall_time_ns"],
        "sequential_pair_wall_time_ns": sum(
            pair["pair_wall_time_ns"] for pair in pairs
        ),
        "encoder_gpu_ns": sum(pair["encoder_gpu_ns"] for pair in pairs),
        "decoder_gpu_ns": sum(pair["decoder_gpu_ns"] for pair in pairs),
    }
    expected["total_gpu_ns"] = expected["encoder_gpu_ns"] + expected["decoder_gpu_ns"]
    expected["cuda_peak_allocated_bytes"] = max(
        [setup["cuda_peak_allocated_bytes"]]
        + [pair["cuda_peak_allocated_bytes"] for pair in pairs]
    )
    expected["cuda_peak_reserved_bytes"] = max(
        [setup["cuda_peak_reserved_bytes"]]
        + [pair["cuda_peak_reserved_bytes"] for pair in pairs]
    )
    for field, value in expected.items():
        if payload[field] != value:
            raise ValueError(f"pool summary {field} does not close")
    _validate_measurements(payload, _POOL_SUMMARY_MEASUREMENT_FIELDS)
    if payload["pool_wall_time_ns"] != (
        payload["setup_wall_time_ns"]
        + payload["sequential_pair_wall_time_ns"]
        + payload["finalization_wall_time_ns"]
    ):
        raise ValueError("pool endpoint wall time does not close")
    expected_id = _canonical_sha256(_pool_summary_semantic_payload(payload))
    if payload["summary_id"] != expected_id:
        raise ValueError("pool summary_id does not match semantic content")
    return payload


_TRAINING_COUNT_FIELDS = {
    "optimizer_updates",
    "microbatches",
    "arc2_episodes",
    "rearc_episodes",
    "encoder_forward_calls",
    "decoder_forward_calls",
    "backward_calls",
    "validation_episode_calls",
    "validation_encoder_forward_calls",
    "validation_decoder_forward_calls",
    "checkpoint_writes",
    "masked_token_predictions",
}

_TRAINING_MEASUREMENT_FIELDS = {
    "cuda_event_ns",
    "cpu_time_ns",
    "wall_time_ns",
    "checkpoint_io_bytes",
    "checkpoint_io_ns",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
}

_TRAINING_LEDGER_FIELDS = {
    "schema",
    "event_id",
    "phase",
    "event_index",
    "optimizer_step",
    "validation_pass_index",
    "episode_index",
    "checkpoint_index",
    "resume_segment_index",
    *_TRAINING_COUNT_FIELDS,
    *_TRAINING_MEASUREMENT_FIELDS,
}

_TRAINING_PHASES = {
    "preflight_update",
    "primary_update",
    "validation_episode",
    "checkpoint_operation",
    "resume_segment",
}


def training_cost_event_semantic_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return _semantic_row_payload(
        row,
        identity_field="event_id",
        measurement_fields=_TRAINING_MEASUREMENT_FIELDS,
    )


def make_training_cost_ledger_row(
    *,
    phase: str,
    event_index: int,
    optimizer_step: int | None = None,
    validation_pass_index: int | None = None,
    episode_index: int | None = None,
    checkpoint_index: int | None = None,
    resume_segment_index: int | None = None,
    optimizer_updates: int = 0,
    microbatches: int = 0,
    arc2_episodes: int = 0,
    rearc_episodes: int = 0,
    encoder_forward_calls: int = 0,
    decoder_forward_calls: int = 0,
    backward_calls: int = 0,
    validation_episode_calls: int = 0,
    validation_encoder_forward_calls: int = 0,
    validation_decoder_forward_calls: int = 0,
    checkpoint_writes: int = 0,
    masked_token_predictions: int = 0,
    cuda_event_ns: int = 0,
    cpu_time_ns: int = 0,
    wall_time_ns: int = 0,
    checkpoint_io_bytes: int = 0,
    checkpoint_io_ns: int = 0,
    cuda_peak_allocated_bytes: int = 0,
    cuda_peak_reserved_bytes: int = 0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema": _contract_value("TRAINING_COST_LEDGER_SCHEMA_VERSION"),
        "phase": phase,
        "event_index": event_index,
        "optimizer_step": optimizer_step,
        "validation_pass_index": validation_pass_index,
        "episode_index": episode_index,
        "checkpoint_index": checkpoint_index,
        "resume_segment_index": resume_segment_index,
        "optimizer_updates": optimizer_updates,
        "microbatches": microbatches,
        "arc2_episodes": arc2_episodes,
        "rearc_episodes": rearc_episodes,
        "encoder_forward_calls": encoder_forward_calls,
        "decoder_forward_calls": decoder_forward_calls,
        "backward_calls": backward_calls,
        "validation_episode_calls": validation_episode_calls,
        "validation_encoder_forward_calls": validation_encoder_forward_calls,
        "validation_decoder_forward_calls": validation_decoder_forward_calls,
        "checkpoint_writes": checkpoint_writes,
        "masked_token_predictions": masked_token_predictions,
        "cuda_event_ns": cuda_event_ns,
        "cpu_time_ns": cpu_time_ns,
        "wall_time_ns": wall_time_ns,
        "checkpoint_io_bytes": checkpoint_io_bytes,
        "checkpoint_io_ns": checkpoint_io_ns,
        "cuda_peak_allocated_bytes": cuda_peak_allocated_bytes,
        "cuda_peak_reserved_bytes": cuda_peak_reserved_bytes,
    }
    row["event_id"] = _canonical_sha256(training_cost_event_semantic_payload(row))
    return validate_training_cost_ledger_row(row)


def validate_training_cost_ledger_row(row: object) -> dict[str, Any]:
    payload = _exact_fields(
        row, _TRAINING_LEDGER_FIELDS, label="M04a training-cost ledger row"
    )
    if payload["schema"] != _contract_value("TRAINING_COST_LEDGER_SCHEMA_VERSION"):
        raise ValueError("unsupported training-cost ledger schema")
    if payload["phase"] not in _TRAINING_PHASES:
        raise ValueError("unsupported training-cost phase")
    _strict_int(payload["event_index"], field="event_index")
    identity_fields = {
        "optimizer_step",
        "validation_pass_index",
        "episode_index",
        "checkpoint_index",
        "resume_segment_index",
    }
    for field in identity_fields:
        if payload[field] is not None:
            _strict_int(payload[field], field=field)
    required_by_phase = {
        "preflight_update": {"optimizer_step"},
        "primary_update": {"optimizer_step"},
        "validation_episode": {"validation_pass_index", "episode_index"},
        "checkpoint_operation": {"checkpoint_index"},
        "resume_segment": {
            "optimizer_step",
            "checkpoint_index",
            "resume_segment_index",
        },
    }
    required = required_by_phase[payload["phase"]]
    if {field for field in identity_fields if payload[field] is not None} != required:
        raise ValueError("training ledger identity fields do not match phase")
    for field in _TRAINING_COUNT_FIELDS:
        _strict_int(payload[field], field=field)
    if payload["phase"] == "resume_segment":
        if not 1 <= payload["checkpoint_index"] < VALIDATION_PASS_COUNT:
            raise ValueError(
                "resume must bind checkpoint "
                f"1..{VALIDATION_PASS_COUNT - 1}"
            )
        if (
            payload["optimizer_step"]
            != payload["checkpoint_index"] * VALIDATION_INTERVAL
        ):
            raise ValueError("resume optimizer step must match its checkpoint")
        if any(payload[field] != 0 for field in _TRAINING_COUNT_FIELDS):
            raise ValueError("resume rows must perform zero semantic work")
    _validate_measurements(payload, _TRAINING_MEASUREMENT_FIELDS)
    if payload["cuda_event_ns"] > payload["wall_time_ns"]:
        raise ValueError("training CUDA-event time exceeds event wall time")
    if payload["checkpoint_io_ns"] > payload["wall_time_ns"]:
        raise ValueError("checkpoint I/O time exceeds event wall time")
    expected = _canonical_sha256(training_cost_event_semantic_payload(payload))
    if payload["event_id"] != expected:
        raise ValueError("training event_id does not match semantic content")
    return payload


def validate_training_cost_ledger(
    rows: Sequence[Mapping[str, Any]],
    *,
    validation_episodes_per_pass: int,
    fresh_setup_wall_time_ns: int,
    expected_masked_token_predictions: int | None = None,
    lock_interval: Mapping[str, Any] | None = None,
    preflight_endpoint_wall_time_ns: int | None = None,
) -> dict[str, int | str]:
    """Validate the exact preflight/primary/validation/checkpoint/resume ledger."""

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("training ledger rows must be an ordered sequence")
    normalized = tuple(validate_training_cost_ledger_row(row) for row in rows)
    if [row["event_index"] for row in normalized] != list(range(len(normalized))):
        raise ValueError("training ledger event_index must be consecutive")
    validation_episodes_per_pass = _strict_int(
        validation_episodes_per_pass,
        field="validation_episodes_per_pass",
        minimum=1,
    )
    fresh_setup_wall_time_ns = _strict_int(
        fresh_setup_wall_time_ns, field="fresh_setup_wall_time_ns"
    )
    preflight = [row for row in normalized if row["phase"] == "preflight_update"]
    primary = [row for row in normalized if row["phase"] == "primary_update"]
    validation = [row for row in normalized if row["phase"] == "validation_episode"]
    checkpoints = [row for row in normalized if row["phase"] == "checkpoint_operation"]
    resumes = [row for row in normalized if row["phase"] == "resume_segment"]
    optimizer_updates = OPTIMIZER_UPDATES
    accumulation = PREFLIGHT_MICROBATCHES_PER_UPDATE
    validation_passes = VALIDATION_PASS_COUNT
    if len(preflight) != 100 or [row["optimizer_step"] for row in preflight] != list(
        range(1, 101)
    ):
        raise ValueError("preflight ledger must contain exactly ordered updates 1..100")
    expected_preflight_per_row = {
        "optimizer_updates": 1,
        "microbatches": accumulation,
        "arc2_episodes": 0,
        "rearc_episodes": 0,
        "encoder_forward_calls": accumulation,
        "decoder_forward_calls": accumulation,
        "backward_calls": accumulation,
        "validation_episode_calls": 0,
        "validation_encoder_forward_calls": 0,
        "validation_decoder_forward_calls": 0,
        "checkpoint_writes": 0,
    }
    if any(
        row[field] != expected
        for row in preflight
        for field, expected in expected_preflight_per_row.items()
    ) or any(row["masked_token_predictions"] < accumulation for row in preflight):
        raise ValueError("preflight update counts do not match the 100-step fixture")
    if len(primary) != optimizer_updates or [
        row["optimizer_step"] for row in primary
    ] != list(range(1, optimizer_updates + 1)):
        raise ValueError(
            "primary ledger must contain exactly ordered optimizer steps "
            f"1..{OPTIMIZER_UPDATES}"
        )
    expected_primary_per_row = {
        "optimizer_updates": 1,
        "microbatches": accumulation,
        "arc2_episodes": accumulation // 2,
        "rearc_episodes": accumulation // 2,
        "encoder_forward_calls": accumulation,
        "decoder_forward_calls": accumulation,
        "backward_calls": accumulation,
        "validation_episode_calls": 0,
        "validation_encoder_forward_calls": 0,
        "validation_decoder_forward_calls": 0,
        "checkpoint_writes": 0,
    }
    if any(
        row[field] != expected
        for row in primary
        for field, expected in expected_primary_per_row.items()
    ):
        raise ValueError("primary training row counts do not match frozen accumulation")
    if any(row["masked_token_predictions"] < accumulation for row in primary):
        raise ValueError(
            "every primary microbatch must predict at least one masked token"
        )
    expected_validation_coordinates = [
        (pass_index, episode_index)
        for pass_index in range(1, validation_passes + 1)
        for episode_index in range(validation_episodes_per_pass)
    ]
    if [
        (row["validation_pass_index"], row["episode_index"]) for row in validation
    ] != expected_validation_coordinates:
        raise ValueError(
            "validation episode rows do not close to "
            f"{VALIDATION_PASS_COUNT} ordered passes"
        )
    validation_zero_fields = {
        "optimizer_updates",
        "microbatches",
        "arc2_episodes",
        "rearc_episodes",
        "backward_calls",
        "checkpoint_writes",
    }
    if any(
        any(row[field] != 0 for field in validation_zero_fields)
        or row["encoder_forward_calls"] != 0
        or row["decoder_forward_calls"] != 0
        or row["validation_episode_calls"] != 1
        or row["validation_encoder_forward_calls"] != 1
        or row["validation_decoder_forward_calls"] != 1
        for row in validation
    ):
        raise ValueError("validation episode call counts do not close")
    if any(row["masked_token_predictions"] <= 0 for row in validation):
        raise ValueError("every validation episode must predict a nonempty fixed mask")
    if len(checkpoints) != validation_passes or [
        row["checkpoint_index"] for row in checkpoints
    ] != list(range(1, validation_passes + 1)):
        raise ValueError(
            "checkpoint ledger must contain exactly "
            f"{VALIDATION_PASS_COUNT} ordered operations"
        )
    if any(
        row["checkpoint_writes"] != 1
        or any(
            row[field] != 0 for field in _TRAINING_COUNT_FIELDS - {"checkpoint_writes"}
        )
        for row in checkpoints
    ):
        raise ValueError("checkpoint operation counts do not close")

    if [row["resume_segment_index"] for row in resumes] != list(
        range(1, len(resumes) + 1)
    ):
        raise ValueError("resume segment indices must be consecutive and one-based")
    checkpoint_event_index = {
        row["checkpoint_index"]: row["event_index"] for row in checkpoints
    }
    primary_event_index = {row["optimizer_step"]: row["event_index"] for row in primary}
    previous_checkpoint = 0
    for row in resumes:
        checkpoint_index = row["checkpoint_index"]
        optimizer_step = row["optimizer_step"]
        if not 1 <= checkpoint_index < validation_passes:
            raise ValueError("resume must bind a non-final completed checkpoint")
        if optimizer_step != checkpoint_index * VALIDATION_INTERVAL:
            raise ValueError("resume optimizer step does not match checkpoint index")
        if checkpoint_index < previous_checkpoint:
            raise ValueError("resume checkpoints must be nondecreasing")
        previous_checkpoint = checkpoint_index
        if not (
            checkpoint_event_index[checkpoint_index]
            < row["event_index"]
            < primary_event_index[optimizer_step + 1]
        ):
            raise ValueError(
                "resume cost row must occur after its checkpoint and before the next update"
            )
        if any(row[field] != 0 for field in _TRAINING_COUNT_FIELDS):
            raise ValueError(
                "resume rows are cost-only and must perform zero semantic work"
            )

    expected_semantic_order: list[tuple[str, int, int | None]] = [
        ("preflight_update", step, None) for step in range(1, 101)
    ]
    for step in range(1, optimizer_updates + 1):
        expected_semantic_order.append(("primary_update", step, None))
        if step % VALIDATION_INTERVAL == 0:
            pass_index = step // VALIDATION_INTERVAL
            expected_semantic_order.extend(
                ("validation_episode", pass_index, episode_index)
                for episode_index in range(validation_episodes_per_pass)
            )
            expected_semantic_order.append(("checkpoint_operation", pass_index, None))
    actual_semantic_order: list[tuple[str, int, int | None]] = []
    for row in normalized:
        if row["phase"] == "resume_segment":
            continue
        if row["phase"] in {"preflight_update", "primary_update"}:
            actual_semantic_order.append((row["phase"], row["optimizer_step"], None))
        elif row["phase"] == "validation_episode":
            actual_semantic_order.append(
                (
                    row["phase"],
                    row["validation_pass_index"],
                    row["episode_index"],
                )
            )
        else:
            actual_semantic_order.append((row["phase"], row["checkpoint_index"], None))
    if actual_semantic_order != expected_semantic_order:
        raise ValueError(
            "training ledger phases do not follow update/validation/checkpoint order"
        )

    semantic_rows = primary + validation + checkpoints
    closure = {
        field: sum(row[field] for row in semantic_rows)
        for field in _TRAINING_COUNT_FIELDS
    }
    closure["validation_passes"] = validation_passes
    expected_closure = {
        "optimizer_updates": optimizer_updates,
        "microbatches": optimizer_updates * accumulation,
        "arc2_episodes": optimizer_updates * accumulation // 2,
        "rearc_episodes": optimizer_updates * accumulation // 2,
        "encoder_forward_calls": optimizer_updates * accumulation,
        "decoder_forward_calls": optimizer_updates * accumulation,
        "backward_calls": optimizer_updates * accumulation,
        "validation_episode_calls": validation_passes * validation_episodes_per_pass,
        "validation_encoder_forward_calls": validation_passes
        * validation_episodes_per_pass,
        "validation_decoder_forward_calls": validation_passes
        * validation_episodes_per_pass,
        "checkpoint_writes": validation_passes,
    }
    for field, expected in expected_closure.items():
        if closure[field] != expected:
            raise ValueError(f"training cost closure mismatch for {field}")
    if expected_masked_token_predictions is not None:
        expected_masked_token_predictions = _strict_int(
            expected_masked_token_predictions,
            field="expected_masked_token_predictions",
        )
        if closure["masked_token_predictions"] != expected_masked_token_predictions:
            raise ValueError("masked-token total does not close from episode manifests")
    preflight_update_wall_time_ns = sum(row["wall_time_ns"] for row in preflight)
    if preflight_endpoint_wall_time_ns is None:
        preflight_wall_time_ns = preflight_update_wall_time_ns
    else:
        preflight_wall_time_ns = _strict_int(
            preflight_endpoint_wall_time_ns,
            field="preflight_endpoint_wall_time_ns",
            minimum=1,
        )
        if preflight_wall_time_ns < preflight_update_wall_time_ns:
            raise ValueError("preflight endpoint omits one or more update intervals")
    phase_wall = {
        "preflight_wall_time_ns": preflight_wall_time_ns,
        "primary_training_wall_time_ns": sum(row["wall_time_ns"] for row in primary),
        "validation_wall_time_ns": sum(row["wall_time_ns"] for row in validation),
        "checkpoint_wall_time_ns": sum(row["wall_time_ns"] for row in checkpoints),
        "resume_setup_wall_time_ns": fresh_setup_wall_time_ns
        + sum(row["wall_time_ns"] for row in resumes),
    }
    phase_wall_time_ns = sum(phase_wall.values())
    if lock_interval is None:
        gpu_lock_wall_time_ns = phase_wall_time_ns
    else:
        measured_interval = validate_training_lock_interval(
            lock_interval, expected_resume_segments=len(resumes)
        )
        gpu_lock_wall_time_ns = measured_interval["gpu_lock_wall_time_ns"]
        if phase_wall_time_ns > gpu_lock_wall_time_ns:
            raise ValueError(
                "training event phases exceed handshake-to-final-selection lock endpoints"
            )
    if gpu_lock_wall_time_ns > GPU_HOUR_BUDGET_NS:
        raise ValueError("training ledger exceeds the frozen 24 GPU-hour cap")
    closure.update(
        {
            "preflight_optimizer_updates": len(preflight),
            "preflight_microbatches": sum(row["microbatches"] for row in preflight),
            "preflight_encoder_forward_calls": sum(
                row["encoder_forward_calls"] for row in preflight
            ),
            "preflight_decoder_forward_calls": sum(
                row["decoder_forward_calls"] for row in preflight
            ),
            "preflight_backward_calls": sum(row["backward_calls"] for row in preflight),
            "resume_segments": len(resumes),
            **phase_wall,
            "gpu_lock_wall_time_ns": gpu_lock_wall_time_ns,
            "budget_limit_ns": GPU_HOUR_BUDGET_NS,
            "budget_remaining_ns": GPU_HOUR_BUDGET_NS - gpu_lock_wall_time_ns,
            "budget_status": "WITHIN_BUDGET",
        }
    )
    return closure


_TRAINING_COST_SUMMARY_FIELDS = {
    "schema",
    "summary_id",
    "ledger_sha256",
    "closure",
    "timing",
    "lock_interval",
}


def validate_training_cost_summary(summary: object) -> dict[str, Any]:
    payload = _exact_fields(
        summary, _TRAINING_COST_SUMMARY_FIELDS, label="M04a training-cost summary"
    )
    if payload["schema"] != TRAINING_COST_SUMMARY_SCHEMA_VERSION:
        raise ValueError("unsupported training-cost summary schema")
    _sha256(payload["summary_id"], field="summary_id")
    _sha256(payload["ledger_sha256"], field="ledger_sha256")
    closure = _validate_training_artifact_closure(payload["closure"])
    timing = _validate_artifact_timing("training", payload["timing"])
    lock_interval = validate_training_lock_interval(
        payload["lock_interval"],
        expected_resume_segments=closure["resume_segments"],
    )
    if lock_interval["gpu_lock_wall_time_ns"] != timing["gpu_lock_wall_time_ns"]:
        raise ValueError("training timing does not bind its measured lock endpoints")
    normalized = {
        "schema": payload["schema"],
        "summary_id": payload["summary_id"],
        "ledger_sha256": payload["ledger_sha256"],
        "closure": closure,
        "timing": timing,
        "lock_interval": lock_interval,
    }
    expected_id = _canonical_sha256(
        {key: value for key, value in normalized.items() if key != "summary_id"}
    )
    if payload["summary_id"] != expected_id:
        raise ValueError("training-cost summary_id does not match semantic content")
    return normalized


def make_training_cost_summary(
    *,
    ledger_sha256: str,
    closure: Mapping[str, Any],
    timing: Mapping[str, int],
    lock_interval: Mapping[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema": TRAINING_COST_SUMMARY_SCHEMA_VERSION,
        "ledger_sha256": ledger_sha256,
        "closure": dict(closure),
        "timing": dict(timing),
        "lock_interval": dict(lock_interval),
    }
    summary["summary_id"] = _canonical_sha256(summary)
    return validate_training_cost_summary(summary)


__all__ = [
    "BLIND_INPUT_ROW_SCHEMA_VERSION",
    "BLIND_SHAPE_SIDECAR_SCHEMA_VERSION",
    "BlindInputShapeSidecar",
    "CHECKPOINT_MANIFEST_SCHEMA_VERSION",
    "CHECKPOINT_SELECTION_RULE",
    "CAMPAIGN_OVERHEAD_PROBE_SCHEMA_VERSION",
    "CAMPAIGN_OVERHEAD_PROJECTION_SCHEMA_VERSION",
    "EVALUATION_ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "EVALUATION_LINEAGE_FIELDS",
    "FILE_BUNDLE_SCHEMA_VERSION",
    "FIXED_BLIND_ARTIFACT_MANIFEST_SHA256",
    "FIXED_BLIND_CASE_SET_ID",
    "FIXED_BLIND_SHAPE_ROWS_SHA256",
    "FIXED_BLIND_SHAPE_SIDECAR_ID",
    "FIXED_BLIND_TASKS_SHA256",
    "FIXED_BLIND_TASK_COUNT",
    "FIXED_BLIND_TEST_PAIR_COUNT",
    "FIXED_DSL_POOL_ARTIFACT_MANIFEST_SHA256",
    "FIXED_DSL_POOL_CONTENT_ID",
    "FIXED_DSL_POOL_SPEC_ID",
    "FIXED_DSL_SEMANTICS_VERSION",
    "FIXED_ZERO_CANDIDATE_TASK_COUNT",
    "LaneTraceSummary",
    "LOCK_HANDSHAKE_ARTIFACT_SCHEMA_VERSION",
    "M04aArtifactBundle",
    "MASK_SENTINEL",
    "NO_SHAPE_PROPOSAL",
    "OPTIMIZER_SCHEDULE_STATE_SCHEMA_VERSION",
    "POOL_ARTIFACT_CLOSURE_FIELDS",
    "POOL_ARTIFACT_FILES",
    "POOL_ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "POOL_COST_SUMMARY_SCHEMA_VERSION",
    "POOL_LINEAGE_FIELDS",
    "POOL_REPLAY_RECEIPT_SCHEMA_VERSION",
    "PREFLIGHT_CHECKPOINT_COST_PROBE_SCHEMA_VERSION",
    "PREFLIGHT_FINAL_SELECTION_PROBE_SCHEMA_VERSION",
    "PREFLIGHT_FRESH_RECONSTRUCTION_PROBE_SCHEMA_VERSION",
    "PREFLIGHT_LOCK_HANDSHAKE_PROBE_SCHEMA_VERSION",
    "PREFLIGHT_POST_VALIDATION_MARGIN_NS",
    "PREFLIGHT_PROJECTION_METHOD",
    "PREFLIGHT_PROJECTION_SCHEMA_VERSION",
    "PREFLIGHT_REPORT_SCHEMA_VERSION",
    "PREFLIGHT_SELECTED_METRIC_SUMMARY_SCHEMA_VERSION",
    "PREFLIGHT_VALIDATION_COMMITMENT_SCHEMA_VERSION",
    "SHAPE_READY",
    "ShapeSelection",
    "TRAINING_ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "TRAINING_ARTIFACT_CLOSURE_FIELDS",
    "TRAINING_ARTIFACT_FILES",
    "TRAINING_LINEAGE_FIELDS",
    "TRAINING_COST_SUMMARY_SCHEMA_VERSION",
    "TRAINING_LOCK_INTERVAL_SCHEMA_VERSION",
    "VALIDATION_AGGREGATION_SEMANTICS",
    "VALIDATION_METRIC_SCHEMA_VERSION",
    "SELECTED_CHECKPOINT_MANIFEST_SCHEMA_VERSION",
    "VerifiedBundle",
    "assert_oracle_free_payload",
    "build_artifact_manifest",
    "build_evaluation_artifact_manifest",
    "build_m04a_blind_input_sidecar",
    "build_pool_artifact_manifest",
    "build_training_artifact_manifest",
    "expected_pair_cost_payload",
    "lane_semantic_payload",
    "lane_trace_sha256",
    "make_decoder_forward_row",
    "make_encoder_forward_row",
    "make_lane_row",
    "make_pair_cost_row",
    "make_pool_cost_summary",
    "make_pool_setup_cost_row",
    "make_checkpoint_manifest_row",
    "make_selected_checkpoint_manifest",
    "make_training_cost_ledger_row",
    "make_training_cost_summary",
    "make_training_lock_interval",
    "make_validation_metric_row",
    "pair_cost_semantic_payload",
    "pool_setup_semantic_payload",
    "python_runtime_identity_sha256",
    "publish_blind_input_shape_sidecar",
    "publish_evaluation_artifact_bundle",
    "publish_m04a_artifact_bundle",
    "publish_pool_artifact_bundle",
    "publish_training_artifact_bundle",
    "read_blind_input_shape_sidecar",
    "read_closed_world_bundle",
    "read_committed_training_artifact_bundle",
    "read_m04a_artifact_bundle",
    "select_blind_shapes",
    "sequential_fp32_mean",
    "training_cost_event_semantic_payload",
    "validate_artifact_manifest",
    "validate_blind_input_row",
    "validate_blind_input_shape_sidecar",
    "validate_decoder_forward_row",
    "validate_checkpoint_manifest_row",
    "validate_encoder_forward_row",
    "validate_environment_manifest_artifact",
    "validate_lane_row",
    "validate_lane_trace",
    "validate_lane_trace_replay",
    "validate_pair_cost_row",
    "validate_pair_evidence",
    "validate_pool_cost_summary",
    "validate_pool_evidence",
    "validate_pool_replay_receipt",
    "validate_pool_setup_cost_row",
    "validate_preflight_artifact",
    "validate_campaign_overhead_cost_probe_report",
    "validate_lock_handshake_artifact",
    "validate_selected_checkpoint_manifest",
    "validate_training_checkpoint_artifacts",
    "validate_training_cost_ledger",
    "validate_training_cost_ledger_row",
    "validate_training_cost_summary",
    "validate_training_lock_interval",
    "validate_optimizer_schedule_state",
    "validate_validation_metric_row",
    "verify_test_source_snapshot_zip",
]
