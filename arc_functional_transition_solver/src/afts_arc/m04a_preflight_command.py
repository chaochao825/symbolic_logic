"""Production orchestration and diagnostic publication for ``preflight-m04a``.

This module is deliberately Torch-free at import time.  The CUDA runtime and
exact preflight implementation are loaded only after every pure, externally
committed input has passed validation.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import stat
import sys
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .m04a_contract import (
    DENOISING_STEPS,
    GRADIENT_ACCUMULATION,
    MAX_GRID_SIDE,
    OPTIMIZER_UPDATES,
    OUTPUT_COLOR_COUNT,
    VALIDATION_PASS_COUNT,
    canonical_sha256,
    mask_count_trace,
)
from .m04a_evidence import (
    MAX_HANDSHAKE_TO_PREFLIGHT_NS,
    PREFLIGHT_FIXTURE_SCHEMA_VERSION,
    PREFLIGHT_INFERENCE_SCHEMA_VERSION,
    PREFLIGHT_POST_VALIDATION_MARGIN_NS,
    PREFLIGHT_PROJECTION_METHOD,
    PREFLIGHT_PROJECTION_SCHEMA_VERSION,
    read_closed_world_bundle,
    validate_campaign_overhead_cost_probe_report,
    validate_environment_manifest_artifact,
    validate_preflight_artifact,
    validate_training_cost_ledger_row,
    verify_test_source_snapshot_zip,
)
from .m04a_launch_plan import (
    LaunchPlanArtifact,
    read_launch_plan_artifact,
    validate_launch_plan_payload,
)
from .m04a_python_runtime_lock import (
    PythonRuntimeLockArtifact,
    read_python_runtime_lock_artifact,
    validate_imported_torch_runtime,
    validate_live_python_runtime_lock,
    validate_python_runtime_lock_payload,
    canonical_python_runtime_lock_bytes,
)
from .m04a_lock_handshake import (
    LockHandshakeArtifact,
    read_lock_handshake_artifact,
    validate_lock_handshake_payload,
)
from .m04a_train_contract import (
    CAMPAIGN_GPU_BUDGET_NS,
    PREFLIGHT_UPDATES,
    learning_rate_for_update,
    training_config_sha256,
)
from .m04a_validation_manifest import (
    ValidationManifestCommitment,
    build_validation_manifest_commitment,
    read_validation_episode_manifest,
)
from .manifest import (
    runtime_source_fingerprint,
    serialize_json,
    serialize_jsonl,
    verify_source_snapshot_zip,
)


PREFLIGHT_GATE_ARTIFACT_KIND = "diagnostic_preflight_gate_not_training_campaign"
PREFLIGHT_GATE_SCHEMA_VERSION = "afts-m04a-diagnostic-preflight-gate/v0.2"
PREFLIGHT_GATE_MANIFEST = "preflight_gate_manifest.json"
PREFLIGHT_GATE_FILES = frozenset(
    {
        "environment_manifest.json",
        "launch_plan.json",
        "lock_handshake_artifact.json",
        "preflight.json",
        "preflight_diagnostic_checkpoint.pt",
        "preflight_overhead_cost_report.json",
        "preflight_training_cost_ledger.jsonl",
        "python-runtime-lock.json",
    }
)
PREFLIGHT_FAILURE_ARTIFACT_KIND = (
    "diagnostic_preflight_failure_not_training_campaign"
)
PREFLIGHT_FAILURE_SCHEMA_VERSION = "afts-m04a-diagnostic-preflight-failure/v0.1"
PREFLIGHT_FAILURE_MANIFEST = "preflight_failure_manifest.json"
PREFLIGHT_FAILURE_REPORT_SCHEMA_VERSION = (
    "afts-grid-cmlm-preflight-failure-report/v0.2"
)
PREFLIGHT_FAILURE_DIRECTORY = "preflight-failure"
PREFLIGHT_FAILURE_BASE_FILES = frozenset(
    {
        "environment_manifest.json",
        "launch_plan.json",
        "lock_handshake_artifact.json",
        "preflight_failure.json",
        "preflight_training_cost_ledger.jsonl",
        "python-runtime-lock.json",
    }
)
PREFLIGHT_FAILURE_FULL_FILES = frozenset(
    {
        *PREFLIGHT_FAILURE_BASE_FILES,
        "preflight_diagnostic_checkpoint.pt",
        "preflight_inference_summary.json",
        "preflight_overhead_cost_report.json",
        "preflight_training_summary.json",
    }
)
_PREFLIGHT_FAILURE_REPORT_FIELDS = frozenset(
    {
        "schema",
        "report_id",
        "status",
        "failure_code",
        "evidence_completeness",
        "fixture_id",
        "runtime",
        "validation_manifest_commitment",
        "lock_handshake_artifact_sha256",
        "preflight_started_perf_counter_ns",
        "completed_updates",
        "ledger_row_count",
        "ledger_rows_sha256",
        "training_summary",
        "inference_summary",
        "overhead_cost_probe_id",
        "budget_projection",
        "diagnostic_checkpoint_commitment",
        "dataset_file_reads",
        "training_checkpoint_writes",
        "diagnostic_checkpoint_writes",
        "fallback_used",
        "training_evidence_eligible",
        "same_run_retry_allowed",
    }
)
_PREFLIGHT_TRAINING_SUMMARY_FIELDS = frozenset(
    {
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
)
_PREFLIGHT_INFERENCE_SUMMARY_FIELDS = frozenset(
    {
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
)
FILE_BUNDLE_SCHEMA_VERSION = "afts-m04a-file-bundle/v0.1"
CONDA_EXPLICIT_BASENAME = "conda-explicit.txt"
MAX_VISIBLE_FILE_BYTES = 512 * 1024 * 1024
MAX_VISIBLE_TOTAL_BYTES = 1024 * 1024 * 1024
_GPU_UUID = re.compile(
    r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REPARSE_POINT_ATTRIBUTE = 0x400


@dataclass(frozen=True, slots=True)
class _TorchDependencies:
    configure_deterministic_cuda: Callable[[], Any]
    environment_manifest: Callable[..., dict[str, object]]
    run_exact_preflight: Callable[..., Any]
    release_held_gpu_lock: Callable[[Any], None]
    preflight_failure_type: type[BaseException]
    imported_torch: object


def _load_torch_dependencies() -> _TorchDependencies:
    """Import every Torch-owning module only at the execution boundary."""

    from .m04a_preflight import PreflightFailure, run_exact_preflight
    from .m04a_torch_runtime import (
        configure_deterministic_cuda,
        environment_manifest,
    )
    from .m04a_lock_handshake import release_held_gpu_lock
    import torch

    return _TorchDependencies(
        configure_deterministic_cuda=configure_deterministic_cuda,
        environment_manifest=environment_manifest,
        run_exact_preflight=run_exact_preflight,
        release_held_gpu_lock=release_held_gpu_lock,
        preflight_failure_type=PreflightFailure,
        imported_torch=torch,
    )


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TypeError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _is_link_or_reparse(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE
    )


def _absolute_lexical(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _assert_existing_nonsymlink_chain(path: Path, *, label: str) -> None:
    absolute = _absolute_lexical(path)
    current = Path(absolute.anchor)
    if not current.exists() or _is_link_or_reparse(current):
        raise ValueError(f"{label} root is unavailable or linked")
    for component in absolute.parts[1:]:
        current = current / component
        if not os.path.lexists(current):
            raise FileNotFoundError(current)
        if _is_link_or_reparse(current):
            raise ValueError(f"{label} traverses a symlink or reparse point")


def _read_committed_regular(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
    maximum_bytes: int = MAX_VISIBLE_FILE_BYTES,
) -> bytes:
    expected = _sha256(expected_sha256, field=f"expected {label} SHA-256")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOINHERIT", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if before.st_size < 0 or before.st_size > maximum_bytes:
            raise ValueError(f"{label} exceeds its byte limit")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(f"{label} became short while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError(f"{label} grew while reading")
        after = os.fstat(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        if any(
            getattr(before, field) != getattr(after, field) for field in stable_fields
        ):
            raise RuntimeError(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    snapshot = b"".join(chunks)
    if hashlib.sha256(snapshot).hexdigest() != expected:
        raise ValueError(f"{label} differs from its launch-plan commitment")
    return snapshot


def _visible_input_snapshots(
    visible_root: str | Path,
    *,
    expected_input_artifacts: Mapping[str, str],
    conda_explicit_sha256: str,
) -> tuple[Path, dict[str, str], dict[str, bytes]]:
    """Verify the fixed 17-plus-conda flat input root without self-discovery."""

    root = _absolute_lexical(visible_root)
    _assert_existing_nonsymlink_chain(root, label="visible root")
    if not root.is_dir():
        raise ValueError("visible root must be an existing directory")
    expected = dict(sorted(expected_input_artifacts.items()))
    if len(expected) != 17 or CONDA_EXPLICIT_BASENAME in expected:
        raise ValueError("launch plan must bind the exact 17 M04a input artifacts")
    for name, digest in expected.items():
        relative = PurePosixPath(name)
        if len(relative.parts) != 1 or relative.as_posix() != name:
            raise ValueError("M04a preflight input artifacts must be flat safe names")
        _sha256(digest, field=f"expected_input_artifacts[{name!r}]")
    expected[CONDA_EXPLICIT_BASENAME] = _sha256(
        conda_explicit_sha256, field="conda_explicit_sha256"
    )

    before = root.stat(follow_symlinks=False)
    actual_names: set[str] = set()
    for entry in os.scandir(root):
        if entry.is_symlink():
            raise ValueError("visible root contains a symlink or reparse point")
        entry_path = root / entry.name
        if _is_link_or_reparse(entry_path) or not entry.is_file(follow_symlinks=False):
            raise ValueError("visible root must contain only committed regular files")
        actual_names.add(entry.name)
    after = root.stat(follow_symlinks=False)
    if not os.path.samestat(before, after):
        raise RuntimeError("visible root identity changed while enumerating")
    if actual_names != set(expected):
        raise ValueError(
            "visible root is not the fixed 17-plus-conda closed world: "
            f"missing={sorted(set(expected) - actual_names)}, "
            f"extra={sorted(actual_names - set(expected))}"
        )

    snapshots: dict[str, bytes] = {}
    total_bytes = 0
    for name, digest in expected.items():
        snapshot = _read_committed_regular(
            root / name,
            expected_sha256=digest,
            label=f"visible input {name}",
        )
        total_bytes += len(snapshot)
        if total_bytes > MAX_VISIBLE_TOTAL_BYTES:
            raise ValueError("visible input snapshots exceed the total byte limit")
        snapshots[name] = snapshot
    final = root.stat(follow_symlinks=False)
    if not os.path.samestat(before, final):
        raise RuntimeError("visible root identity changed while reading")
    return root, expected, snapshots


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _failure_output_path(output: Path) -> Path:
    failure = output.parent / PREFLIGHT_FAILURE_DIRECTORY
    if failure == output:
        raise ValueError("preflight gate output cannot use the reserved failure name")
    return failure


def _validate_path_layout(
    *,
    plan: Mapping[str, Any],
    visible_root: Path,
    launch_plan_path: str | Path,
    python_runtime_lock_path: str | Path,
    lock_handshake_path: str | Path,
    validation_manifest_dir: str | Path,
    launcher_path: str | Path,
    cost_probe_scratch_dir: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path, Path, Path]:
    run_root = _absolute_lexical(str(plan["run_root"]))
    _assert_existing_nonsymlink_chain(run_root, label="run root")
    if not run_root.is_dir():
        raise ValueError("launch-plan run_root must already exist")
    if _path_is_within(run_root, visible_root) or _path_is_within(
        visible_root, run_root
    ):
        raise ValueError("run_root and visible root must be disjoint")

    external_files = {
        "launch plan": _absolute_lexical(launch_plan_path),
        "lock handshake": _absolute_lexical(lock_handshake_path),
    }
    for label, path in external_files.items():
        _assert_existing_nonsymlink_chain(path, label=label)
        if _path_is_within(path, visible_root):
            raise ValueError(f"{label} must be outside the visible root")
    validation_dir = _absolute_lexical(validation_manifest_dir)
    _assert_existing_nonsymlink_chain(validation_dir, label="validation manifest")
    if not validation_dir.is_dir() or _path_is_within(validation_dir, visible_root):
        raise ValueError("validation manifest bundle must be outside the visible root")

    launcher = _absolute_lexical(launcher_path)
    _assert_existing_nonsymlink_chain(launcher, label="launcher")
    if not launcher.is_file() or launcher != visible_root / "remote_launcher.py":
        raise ValueError("launcher must be the visible-root remote_launcher.py")
    runtime_lock = _absolute_lexical(python_runtime_lock_path)
    _assert_existing_nonsymlink_chain(runtime_lock, label="Python runtime lock")
    if (
        not runtime_lock.is_file()
        or runtime_lock != visible_root / "python-runtime-lock.json"
    ):
        raise ValueError(
            "Python runtime lock must be visible-root/python-runtime-lock.json"
        )

    scratch = _absolute_lexical(cost_probe_scratch_dir)
    output = _absolute_lexical(output_dir)
    failure_output = _failure_output_path(output)
    for label, target in (
        ("cost-probe scratch", scratch),
        ("output", output),
        ("failure output", failure_output),
    ):
        _assert_existing_nonsymlink_chain(target.parent, label=f"{label} parent")
        if not target.parent.is_dir() or not _path_is_within(target, run_root):
            raise ValueError(f"{label} must be a fresh child of launch-plan run_root")
        if _path_is_within(target, validation_dir) or _path_is_within(
            validation_dir, target
        ):
            raise ValueError(f"{label} must be disjoint from the validation bundle")
        if os.path.lexists(target):
            raise FileExistsError(f"{label} path already exists: {target}")
    for left_label, left, right_label, right in (
        ("cost-probe scratch", scratch, "output", output),
        ("cost-probe scratch", scratch, "failure output", failure_output),
        ("output", output, "failure output", failure_output),
    ):
        if (
            left == right
            or _path_is_within(left, right)
            or _path_is_within(right, left)
        ):
            raise ValueError(f"{left_label} and {right_label} paths must be disjoint")
    return run_root, scratch, output, launcher


def _assert_publication_parent_unchanged(
    *,
    run_root: Path,
    output: Path,
    expected_run_root: os.stat_result,
    expected_output_parent: os.stat_result,
) -> None:
    """Recheck the long-lived output boundary immediately after CUDA work."""

    _assert_existing_nonsymlink_chain(run_root, label="run root")
    _assert_existing_nonsymlink_chain(output.parent, label="output parent")
    current_root = run_root.stat(follow_symlinks=False)
    current_parent = output.parent.stat(follow_symlinks=False)
    if not os.path.samestat(current_root, expected_run_root) or not os.path.samestat(
        current_parent, expected_output_parent
    ):
        raise RuntimeError("run/output directory identity changed during preflight")
    if not stat.S_ISDIR(current_parent.st_mode) or not _path_is_within(
        output, run_root
    ):
        raise ValueError("preflight output escaped the committed run_root")
    if os.name == "posix" and current_parent.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise PermissionError(
            "preflight output parent must not be group/world writable"
        )
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite preflight gate: {output}")


def _validate_process_identity(plan: Mapping[str, Any], *, gpu_uuid: str) -> str:
    if not isinstance(gpu_uuid, str) or _GPU_UUID.fullmatch(gpu_uuid) is None:
        raise ValueError("--gpu-uuid must be one canonical lowercase full GPU UUID")
    expected_environment = {
        "AFTS_M04A_RUN_ID": plan["run_id"],
        "AFTS_M04A_GPU_UUID": gpu_uuid,
        "CUDA_VISIBLE_DEVICES": gpu_uuid,
    }
    if any(
        os.environ.get(name) != value for name, value in expected_environment.items()
    ):
        raise RuntimeError("launcher environment differs from launch-plan/GPU identity")
    return gpu_uuid


def _runtime_lock_live_kwargs(
    *,
    plan: Mapping[str, Any],
    runtime_lock: PythonRuntimeLockArtifact,
) -> dict[str, object]:
    raw = os.environ.get("AFTS_M04A_BOUND_IMPORT_ROOTS_JSON")
    try:
        bound_roots = json.loads(raw or "")
    except json.JSONDecodeError as exc:
        raise ValueError("bound import-root record is invalid JSON") from exc
    logical_roots = plan["ordered_import_roots"]
    if (
        not isinstance(bound_roots, list)
        or len(bound_roots) != len(logical_roots)
        or any(
            not isinstance(root, str)
            or re.fullmatch(r"/proc/self/fd/([0-9]+)", root) is None
            for root in bound_roots
        )
    ):
        raise ValueError("bound import-root record differs from launch plan")
    site_path = runtime_lock.payload["environment"]["site_packages_path"]
    site_indices = [
        index for index, logical in enumerate(logical_roots) if logical == site_path
    ]
    if len(site_indices) != 1:
        raise ValueError("runtime lock site-packages is not one logical import root")
    descriptor = int(bound_roots[site_indices[0]].rsplit("/", 1)[1])
    if descriptor < 3:
        raise ValueError("bound site-packages descriptor aliases standard I/O")
    return {
        "site_packages_fd": descriptor,
        "ordered_import_roots": logical_roots,
        "bound_import_roots": bound_roots,
    }


def _validation_commitment_from_plan(
    plan: Mapping[str, Any],
    *,
    validation_manifest_dir: str | Path,
    visible_snapshots: Mapping[str, bytes],
) -> ValidationManifestCommitment:
    expected = plan["validation_manifest_commitment"]
    manifest = read_validation_episode_manifest(
        validation_manifest_dir,
        expected_artifact_manifest_sha256=expected["outer_artifact_manifest_sha256"],
    )
    commitment = build_validation_manifest_commitment(
        manifest,
        expected_artifact_manifest_sha256=expected["outer_artifact_manifest_sha256"],
        expected_jsonl_sha256=expected["jsonl_sha256"],
    )
    if commitment.to_json_dict() != expected:
        raise ValueError("validation commitment differs from the launch plan")
    if (
        manifest.jsonl_bytes != visible_snapshots["validation_episode_manifest.jsonl"]
        or serialize_json(manifest.summary)
        != visible_snapshots["validation_episode_manifest_summary.json"]
    ):
        raise ValueError("visible validation copies differ from the committed bundle")
    return commitment


def _bind_environment_to_plan(
    environment: object,
    *,
    plan: Mapping[str, Any],
    gpu_uuid: str,
    visible_root: Path,
    visible_files: Mapping[str, str],
    launcher: Path,
    python_runtime_lock: PythonRuntimeLockArtifact,
    runtime_lock_payload: Mapping[str, Any],
) -> dict[str, object]:
    if not isinstance(environment, dict):
        raise TypeError("environment_manifest must return a dictionary")
    locked_python = runtime_lock_payload["python"]
    expected = {
        "run_id": plan["run_id"],
        "gpu_uuid": gpu_uuid,
        "launcher_path": str(launcher.resolve()),
        "launcher_sha256": plan["expected_input_artifacts"]["remote_launcher.py"],
        "ordered_import_roots": plan["ordered_import_roots"],
        "conda_explicit_path": str((visible_root / CONDA_EXPLICIT_BASENAME).resolve()),
        "conda_explicit_sha256": plan["conda_explicit_sha256"],
        "visible_root": str(visible_root.resolve()),
        "runtime_source_sha256": plan["runtime_source_fingerprint_sha256"],
        "test_source_sha256": plan["test_source_fingerprint_sha256"],
        "visible_files": dict(sorted(visible_files.items())),
        "python_runtime_lock_sha256": python_runtime_lock.artifact_sha256,
        "python_runtime_lock_id": runtime_lock_payload["runtime_lock_id"],
        "python_implementation": locked_python["implementation"],
        "python_version": locked_python["version"],
        "python_executable": locked_python["executable_lexical_path"],
        "python_executable_sha256": locked_python["executable_sha256"],
        "python_executable_bytes": locked_python["executable_bytes"],
    }
    if any(environment.get(field) != value for field, value in expected.items()):
        raise ValueError(
            "environment manifest differs from retained launch commitments"
        )
    return dict(environment)


def _strict_json_bytes(content: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key in {label}: {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite constant in {label}: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON in {label}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object")
    return payload


def _strict_nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{field} must be a nonnegative integer")
    return value


def _expected_preflight_fixture_id() -> str:
    demonstration_count = 10
    encoder_grid_count = 2 * demonstration_count + 1
    grid_token_count = MAX_GRID_SIDE * MAX_GRID_SIDE + 1
    payload = {
        "schema": PREFLIGHT_FIXTURE_SCHEMA_VERSION,
        "grid_side": MAX_GRID_SIDE,
        "target_cell_count": MAX_GRID_SIDE * MAX_GRID_SIDE,
        "demonstration_count": demonstration_count,
        "encoder_grid_count": encoder_grid_count,
        "encoder_grid_token_count": grid_token_count,
        "expected_memory_length": encoder_grid_count * grid_token_count,
        "demonstrations": [
            {
                "pair_index": index,
                "input_constant_color": index,
                "output_constant_color": (index + 1) % OUTPUT_COLOR_COUNT,
                "height": MAX_GRID_SIDE,
                "width": MAX_GRID_SIDE,
            }
            for index in range(demonstration_count)
        ],
        "query_constant_color": 0,
        "target_constant_color": 1,
        "training_mask_policy": "all_900_cells_every_microbatch",
        "training_updates": PREFLIGHT_UPDATES,
        "microbatches_per_update": GRADIENT_ACCUMULATION,
        "inference_lanes": 8,
        "inference_steps": DENOISING_STEPS,
    }
    return canonical_sha256(payload)


def _validate_inference_summary_payload(summary: object) -> dict[str, Any]:
    if (
        not isinstance(summary, dict)
        or set(summary) != _PREFLIGHT_INFERENCE_SUMMARY_FIELDS
    ):
        raise ValueError("budget failure inference fields do not match exact schema")
    lane_count = 8
    demonstrations = 10
    grid_token_count = MAX_GRID_SIDE * MAX_GRID_SIDE + 1
    encoder_grid_count = 2 * demonstrations + 1
    masked_cells = MAX_GRID_SIDE * MAX_GRID_SIDE
    trace = list(mask_count_trace(masked_cells))
    expected = {
        "schema": PREFLIGHT_INFERENCE_SCHEMA_VERSION,
        "lane_count": lane_count,
        "denoising_steps": DENOISING_STEPS,
        "encoder_batch_calls": 1,
        "decoder_batch_calls": DENOISING_STEPS,
        "sample_equivalent_forward_calls": lane_count * DENOISING_STEPS,
        "masked_token_predictions": lane_count * sum(trace[:-1]),
        "padded_encoder_batch_shape": [encoder_grid_count, grid_token_count],
        "unpadded_memory_length": encoder_grid_count * grid_token_count,
        "cache_dtype": "bfloat16",
        "decoder_batch_sizes": [lane_count] * DENOISING_STEPS,
        "mask_count_trace": trace,
    }
    if any(summary.get(field) != value for field, value in expected.items()):
        raise ValueError("budget failure inference semantic closure drifted")
    lane_traces = summary.get("lane_trace_sha256")
    final_outputs = summary.get("final_output_keys")
    if (
        not isinstance(lane_traces, list)
        or len(lane_traces) != lane_count
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in lane_traces
        )
        or not isinstance(final_outputs, list)
        or len(final_outputs) != lane_count
        or any(not isinstance(value, str) or not value for value in final_outputs)
        or summary.get("unique_outputs") != len(set(final_outputs))
    ):
        raise ValueError("budget failure inference lane closure drifted")
    measurement_fields = (
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
    )
    for field in measurement_fields:
        _strict_nonnegative_int(summary[field], field=field)
    if (
        summary["encoder_gpu_ns"] > summary["encoder_wall_time_ns"]
        or summary["decoder_gpu_ns"] > summary["decoder_wall_time_ns"]
        or (
            summary["h2d_ns"]
            + summary["d2h_ns"]
            + summary["encoder_wall_time_ns"]
            + summary["decoder_wall_time_ns"]
            + summary["cpu_sampling_ns"]
            + summary["hashing_ns"]
        )
        > summary["inference_wall_time_ns"]
    ):
        raise ValueError("budget failure inference timings do not close")
    semantic = dict(summary)
    inference_id = semantic.pop("inference_id", None)
    if inference_id != canonical_sha256(semantic):
        raise ValueError("budget failure inference summary identity mismatch")
    return dict(summary)


def _validate_failure_report_payload(
    report: object,
    ledger_rows: object,
    *,
    run_id: str,
    gpu_uuid: str,
    validation_manifest_commitment: Mapping[str, Any],
    lock_handshake_artifact_sha256: str,
    lock_handshake_payload: Mapping[str, Any],
    config_sha256: str,
    runtime_source_sha256: str,
    test_source_sha256: str,
    overhead_report: object | None,
    training_summary: object | None,
    inference_summary: object | None,
    diagnostic_checkpoint: bytes | None,
) -> dict[str, Any]:
    """Pure closed-world replay for a terminal preflight-failure payload."""

    if not isinstance(report, dict) or set(report) != _PREFLIGHT_FAILURE_REPORT_FIELDS:
        raise ValueError("preflight failure report fields do not match exact schema")
    if report.get("schema") != PREFLIGHT_FAILURE_REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported preflight failure report schema")
    failure_code = report.get("failure_code")
    if (
        failure_code not in {
            "OOM",
            "BUDGET_EXCEEDED",
            "NONFINITE_LOGITS",
            "PROJECTION_INCOMPLETE",
        }
        or report.get("status") != failure_code
        or report.get("dataset_file_reads") != 0
        or report.get("training_checkpoint_writes") != 0
        or report.get("diagnostic_checkpoint_writes")
        != (1 if failure_code == "BUDGET_EXCEEDED" else 0)
        or report.get("fallback_used") is not False
        or report.get("training_evidence_eligible") is not False
        or report.get("same_run_retry_allowed") is not False
    ):
        raise ValueError("preflight failure disposition is invalid")
    if report.get("fixture_id") != _expected_preflight_fixture_id():
        raise ValueError("preflight failure fixture_id drifted from frozen fixture")
    semantic = dict(report)
    claimed_report_id = semantic.pop("report_id", None)
    if claimed_report_id != canonical_sha256(semantic):
        raise ValueError("preflight failure report_id mismatch")
    if not isinstance(ledger_rows, (list, tuple)):
        raise TypeError("preflight failure ledger must be a JSONL row list")
    ledger = [validate_training_cost_ledger_row(row) for row in ledger_rows]
    completed = _strict_nonnegative_int(
        report.get("completed_updates"), field="completed_updates"
    )
    if (
        completed != len(ledger)
        or completed > PREFLIGHT_UPDATES
        or report.get("ledger_row_count") != len(ledger)
        or report.get("ledger_rows_sha256") != canonical_sha256(ledger)
        or [row["event_index"] for row in ledger] != list(range(len(ledger)))
        or any(row["phase"] != "preflight_update" for row in ledger)
    ):
        raise ValueError("preflight failure ledger commitment does not close")
    expected_counts = {
        "optimizer_updates": 1,
        "microbatches": GRADIENT_ACCUMULATION,
        "arc2_episodes": 0,
        "rearc_episodes": 0,
        "encoder_forward_calls": GRADIENT_ACCUMULATION,
        "decoder_forward_calls": GRADIENT_ACCUMULATION,
        "backward_calls": GRADIENT_ACCUMULATION,
        "validation_episode_calls": 0,
        "validation_encoder_forward_calls": 0,
        "validation_decoder_forward_calls": 0,
        "checkpoint_writes": 0,
        "masked_token_predictions": (
            GRADIENT_ACCUMULATION * MAX_GRID_SIDE * MAX_GRID_SIDE
        ),
    }
    for index, row in enumerate(ledger):
        if row["optimizer_step"] != index + 1 or any(
            row[field] != value for field, value in expected_counts.items()
        ):
            raise ValueError("preflight failure ledger semantic counts drifted")

    full = failure_code == "BUDGET_EXCEEDED"
    if report.get("evidence_completeness") != (
        "full_budget_probe" if full else "partial"
    ):
        raise ValueError("preflight failure completeness state drifted")
    attachments = (
        overhead_report,
        training_summary,
        inference_summary,
        diagnostic_checkpoint,
    )
    if not full:
        nullable = (
            "runtime",
            "validation_manifest_commitment",
            "lock_handshake_artifact_sha256",
            "preflight_started_perf_counter_ns",
            "training_summary",
            "inference_summary",
            "overhead_cost_probe_id",
            "budget_projection",
            "diagnostic_checkpoint_commitment",
        )
        if any(report[field] is not None for field in nullable) or any(
            value is not None for value in attachments
        ):
            raise ValueError("partial preflight failure contains full-probe evidence")
        return dict(report)

    if len(ledger) != PREFLIGHT_UPDATES:
        raise ValueError("budget failure requires the complete preflight ledger")
    runtime = report.get("runtime")
    if runtime != {
        "run_id": run_id,
        "gpu_uuid": gpu_uuid,
        "logical_device_index": 0,
    }:
        raise ValueError("budget failure runtime differs from committed launch")
    if (
        report.get("validation_manifest_commitment")
        != dict(validation_manifest_commitment)
        or report.get("lock_handshake_artifact_sha256")
        != lock_handshake_artifact_sha256
    ):
        raise ValueError("budget failure differs from external commitments")
    preflight_started = _strict_nonnegative_int(
        report.get("preflight_started_perf_counter_ns"),
        field="preflight_started_perf_counter_ns",
    )
    if preflight_started < 1:
        raise ValueError("preflight failure start timestamp must be positive")
    if (
        not isinstance(training_summary, dict)
        or set(training_summary) != _PREFLIGHT_TRAINING_SUMMARY_FIELDS
        or report.get("training_summary") != training_summary
        or not isinstance(inference_summary, dict)
        or set(inference_summary) != _PREFLIGHT_INFERENCE_SUMMARY_FIELDS
        or report.get("inference_summary") != inference_summary
        or type(diagnostic_checkpoint) is not bytes
        or not isinstance(overhead_report, dict)
    ):
        raise ValueError("budget failure full-probe attachments are incomplete")
    if (
        training_summary.get("updates") != PREFLIGHT_UPDATES
        or training_summary.get("microbatches_per_update")
        != GRADIENT_ACCUMULATION
        or training_summary.get("total_microbatches")
        != PREFLIGHT_UPDATES * GRADIENT_ACCUMULATION
        or training_summary.get("masked_tokens_per_microbatch")
        != MAX_GRID_SIDE * MAX_GRID_SIDE
        or training_summary.get("masked_token_predictions")
        != (
            PREFLIGHT_UPDATES
            * GRADIENT_ACCUMULATION
            * MAX_GRID_SIDE
            * MAX_GRID_SIDE
        )
        or training_summary.get("encoder_forward_calls")
        != PREFLIGHT_UPDATES * GRADIENT_ACCUMULATION
        or training_summary.get("decoder_forward_calls")
        != PREFLIGHT_UPDATES * GRADIENT_ACCUMULATION
        or training_summary.get("backward_calls")
        != PREFLIGHT_UPDATES * GRADIENT_ACCUMULATION
        or training_summary.get("ledger_row_count") != PREFLIGHT_UPDATES
        or training_summary.get("ledger_rows_sha256") != canonical_sha256(ledger)
        or training_summary.get("total_update_wall_ns")
        != sum(int(row["wall_time_ns"]) for row in ledger)
        or training_summary.get("total_update_cuda_event_ns")
        != sum(int(row["cuda_event_ns"]) for row in ledger)
        or training_summary.get("peak_allocated_bytes")
        != max(int(row["cuda_peak_allocated_bytes"]) for row in ledger)
        or training_summary.get("peak_reserved_bytes")
        != max(int(row["cuda_peak_reserved_bytes"]) for row in ledger)
    ):
        raise ValueError("budget failure training summary does not close from ledger")
    for field in (
        "learning_rate_hex",
        "mean_masked_cell_ce_hex",
        "gradient_norm_before_clip_hex",
    ):
        values = training_summary[field]
        if not isinstance(values, list) or len(values) != PREFLIGHT_UPDATES:
            raise ValueError(f"budget failure {field} trace is incomplete")
        try:
            parsed = [float.fromhex(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"budget failure {field} trace is invalid") from exc
        if any(not math.isfinite(value) or value < 0.0 for value in parsed):
            raise ValueError(f"budget failure {field} trace is negative")
    if training_summary["learning_rate_hex"] != [
        learning_rate_for_update(step).hex()
        for step in range(1, PREFLIGHT_UPDATES + 1)
    ]:
        raise ValueError("budget failure learning-rate trace drifted")
    _validate_inference_summary_payload(inference_summary)
    cost = validate_campaign_overhead_cost_probe_report(
        overhead_report,
        expected_run_id=run_id,
        expected_config_sha256=config_sha256,
        expected_runtime_source_sha256=runtime_source_sha256,
        expected_test_source_sha256=test_source_sha256,
        diagnostic_checkpoint_snapshot=diagnostic_checkpoint,
    )
    if report.get("overhead_cost_probe_id") != cost["probe_id"]:
        raise ValueError("budget failure does not bind its overhead report")
    lock_probe = cost["lock_handshake_probe"]
    expected_lock_bindings = {
        "run_id": lock_handshake_payload.get("run_id"),
        "acquisition_started_perf_counter_ns": lock_handshake_payload.get(
            "acquisition_started_perf_counter_ns"
        ),
        "handshake_completed_perf_counter_ns": lock_handshake_payload.get(
            "handshake_completed_perf_counter_ns"
        ),
        "wall_ns": lock_handshake_payload.get("wall_ns"),
    }
    if any(
        lock_probe.get(field) != value
        for field, value in expected_lock_bindings.items()
    ):
        raise ValueError("budget failure cost probe differs from lock handshake")
    checkpoint_probe = cost["checkpoint_cost_probe"]
    expected_checkpoint_commitment = {
        "artifact_filename": cost["checkpoint_artifact_filename"],
        "artifact_status": cost["checkpoint_artifact_status"],
        "artifact_disposition": checkpoint_probe["artifact_disposition"],
        "sha256": checkpoint_probe["checkpoint_sha256"],
        "bytes": checkpoint_probe["checkpoint_bytes"],
        "selectable_checkpoint_created": False,
    }
    if report.get("diagnostic_checkpoint_commitment") != (
        expected_checkpoint_commitment
    ):
        raise ValueError("budget failure checkpoint commitment drifted")
    projection = report.get("budget_projection")
    projection_fields = {
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
    if not isinstance(projection, dict) or set(projection) != projection_fields:
        raise ValueError("budget failure projection fields do not match exact schema")
    if (
        projection["schema"] != PREFLIGHT_PROJECTION_SCHEMA_VERSION
        or projection["projection_method"] != PREFLIGHT_PROJECTION_METHOD
    ):
        raise ValueError("budget failure projection version/method drifted")
    nonnegative_projection_fields = projection_fields - {
        "schema",
        "projection_id",
        "projection_method",
        "lock_handshake_probe_id",
        "fresh_reconstruction_probe_id",
        "checkpoint_cost_probe_id",
        "final_selection_probe_id",
        "budget_remaining_ns",
        "budget_status",
    }
    for field in nonnegative_projection_fields:
        _strict_nonnegative_int(projection[field], field=field)
    if type(projection["budget_remaining_ns"]) is not int:
        raise TypeError("budget_remaining_ns must be an integer")
    for field in (
        "measured_preflight_training_wall_ns",
        "measured_preflight_total_wall_ns",
        "mean_training_update_wall_ns",
        "mean_training_microbatch_wall_ns",
        "validation_episodes_per_pass",
    ):
        if projection[field] < 1:
            raise ValueError(f"budget failure {field} must be positive")
    if (
        projection["measured_handshake_to_preflight_start_ns"]
        > MAX_HANDSHAKE_TO_PREFLIGHT_NS
    ):
        raise ValueError("budget failure lock handshake is too old")
    projection_semantic = dict(projection)
    projection_id = projection_semantic.pop("projection_id")
    if projection_id != canonical_sha256(projection_semantic):
        raise ValueError("budget failure projection_id mismatch")
    training_wall = int(training_summary["total_update_wall_ns"])
    mean_update = (training_wall + PREFLIGHT_UPDATES - 1) // PREFLIGHT_UPDATES
    microbatch_count = PREFLIGHT_UPDATES * GRADIENT_ACCUMULATION
    mean_microbatch = (training_wall + microbatch_count - 1) // microbatch_count
    validation_episodes = int(validation_manifest_commitment["row_count"])
    primary_wall = mean_update * OPTIMIZER_UPDATES
    validation_calls = VALIDATION_PASS_COUNT * validation_episodes
    validation_wall = mean_microbatch * validation_calls
    checkpoint_wall = (
        VALIDATION_PASS_COUNT * int(checkpoint_probe["roundtrip_wall_ns"])
    )
    expected_projection_bindings = {
        "measured_preflight_training_wall_ns": training_wall,
        "mean_training_update_wall_ns": mean_update,
        "mean_training_microbatch_wall_ns": mean_microbatch,
        "lock_handshake_probe_id": cost["lock_handshake_probe"]["probe_id"],
        "measured_lock_setup_handshake_wall_ns": cost["lock_handshake_probe"][
            "wall_ns"
        ],
        "measured_handshake_to_preflight_start_ns": (
            preflight_started
            - int(lock_probe["handshake_completed_perf_counter_ns"])
        ),
        "fresh_reconstruction_probe_id": cost["fresh_reconstruction_probe"][
            "probe_id"
        ],
        "measured_fresh_reconstruction_wall_ns": cost[
            "fresh_reconstruction_probe"
        ]["total_wall_ns"],
        "checkpoint_cost_probe_id": checkpoint_probe["probe_id"],
        "measured_checkpoint_roundtrip_wall_ns": checkpoint_probe[
            "roundtrip_wall_ns"
        ],
        "final_selection_probe_id": cost["final_selection_probe"]["probe_id"],
        "measured_final_selection_wall_ns": cost["final_selection_probe"][
            "selection_wall_ns"
        ],
        "projected_primary_updates": OPTIMIZER_UPDATES,
        "projected_primary_wall_ns": primary_wall,
        "projected_validation_passes": VALIDATION_PASS_COUNT,
        "validation_episodes_per_pass": validation_episodes,
        "projected_validation_episode_calls": validation_calls,
        "projected_validation_wall_ns": validation_wall,
        "projected_checkpoint_writes": VALIDATION_PASS_COUNT,
        "projected_checkpoint_wall_ns": checkpoint_wall,
        "projected_final_selection_wall_ns": cost["final_selection_probe"][
            "selection_wall_ns"
        ],
        "budget_limit_ns": CAMPAIGN_GPU_BUDGET_NS,
        "budget_status": "BUDGET_EXCEEDED",
        "post_preflight_validation_margin_ns": (
            PREFLIGHT_POST_VALIDATION_MARGIN_NS
        ),
    }
    if any(
        projection.get(field) != value
        for field, value in expected_projection_bindings.items()
    ):
        raise ValueError("budget failure projection differs from measured parents")
    projected_total = (
        int(cost["lock_handshake_probe"]["wall_ns"])
        + int(projection["measured_handshake_to_preflight_start_ns"])
        + int(projection["measured_preflight_total_wall_ns"])
        + int(cost["fresh_reconstruction_probe"]["total_wall_ns"])
        + primary_wall
        + validation_wall
        + checkpoint_wall
        + int(cost["final_selection_probe"]["selection_wall_ns"])
        + int(projection["post_preflight_validation_margin_ns"])
    )
    if (
        int(projection["measured_preflight_total_wall_ns"]) < training_wall
        or projection["measured_handshake_to_preflight_start_ns"] < 0
        or projection["projected_campaign_wall_ns"] != projected_total
        or projection["budget_remaining_ns"]
        != CAMPAIGN_GPU_BUDGET_NS - projected_total
        or projected_total <= CAMPAIGN_GPU_BUDGET_NS
    ):
        raise ValueError("budget failure projection arithmetic does not close")
    return dict(report)


def _gate_manifest_payload(
    artifacts: Mapping[str, bytes],
    *,
    run_id: str,
    launch_plan: LaunchPlanArtifact,
    lock_artifact: LockHandshakeArtifact,
    preflight_report: Mapping[str, Any],
    overhead_report: Mapping[str, Any],
    python_runtime_lock: PythonRuntimeLockArtifact,
) -> dict[str, Any]:
    if set(artifacts) != PREFLIGHT_GATE_FILES:
        raise ValueError("diagnostic preflight payload file set drifted")
    files = [
        {
            "path": name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "rows": content.count(b"\n") if name.endswith(".jsonl") else None,
        }
        for name, content in sorted(artifacts.items())
    ]
    semantic: dict[str, Any] = {
        "schema": PREFLIGHT_GATE_SCHEMA_VERSION,
        "artifact_kind": PREFLIGHT_GATE_ARTIFACT_KIND,
        "disposition": PREFLIGHT_GATE_ARTIFACT_KIND,
        "run_id": run_id,
        "launch_plan_artifact_sha256": launch_plan.artifact_sha256,
        "launch_plan_id": launch_plan.payload["launch_plan_id"],
        "lock_handshake_artifact_sha256": lock_artifact.artifact_sha256,
        "lock_handshake_id": lock_artifact.payload["handshake_id"],
        "python_runtime_lock_artifact_sha256": python_runtime_lock.artifact_sha256,
        "python_runtime_lock_id": python_runtime_lock.payload["runtime_lock_id"],
        "preflight_report_id": preflight_report["report_id"],
        "overhead_cost_probe_id": overhead_report["probe_id"],
        "training_evidence_eligible": False,
        "requires_same_process_exact_preflight_rerun": True,
        "lock_release_policy": "release_after_atomic_gate_publication",
        "files": files,
    }
    return {**semantic, "gate_id": canonical_sha256(semantic)}


def _failure_manifest_payload(
    artifacts: Mapping[str, bytes],
    *,
    run_id: str,
    launch_plan: LaunchPlanArtifact,
    lock_artifact: LockHandshakeArtifact,
    failure_report: Mapping[str, Any],
    python_runtime_lock: PythonRuntimeLockArtifact,
) -> dict[str, Any]:
    completeness = failure_report["evidence_completeness"]
    expected_files = (
        PREFLIGHT_FAILURE_FULL_FILES
        if completeness == "full_budget_probe"
        else PREFLIGHT_FAILURE_BASE_FILES
    )
    if set(artifacts) != expected_files:
        raise ValueError("diagnostic preflight-failure payload file set drifted")
    files = [
        {
            "path": name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "rows": content.count(b"\n") if name.endswith(".jsonl") else None,
        }
        for name, content in sorted(artifacts.items())
    ]
    semantic: dict[str, Any] = {
        "schema": PREFLIGHT_FAILURE_SCHEMA_VERSION,
        "artifact_kind": PREFLIGHT_FAILURE_ARTIFACT_KIND,
        "disposition": PREFLIGHT_FAILURE_ARTIFACT_KIND,
        "run_id": run_id,
        "failure_code": failure_report["failure_code"],
        "evidence_completeness": completeness,
        "launch_plan_artifact_sha256": launch_plan.artifact_sha256,
        "launch_plan_id": launch_plan.payload["launch_plan_id"],
        "lock_handshake_artifact_sha256": lock_artifact.artifact_sha256,
        "lock_handshake_id": lock_artifact.payload["handshake_id"],
        "python_runtime_lock_artifact_sha256": python_runtime_lock.artifact_sha256,
        "python_runtime_lock_id": python_runtime_lock.payload["runtime_lock_id"],
        "failure_report_id": failure_report["report_id"],
        "overhead_cost_probe_id": failure_report["overhead_cost_probe_id"],
        "training_evidence_eligible": False,
        "diagnostic_checkpoint_selectable": False,
        "same_run_retry_allowed": False,
        "same_run_retry_policy": "forbidden_after_terminal_preflight_failure",
        "preflight_gate_created": False,
        "lock_release_policy": (
            "release_attempted_by_failed_exact_preflight_process_exit_backstop"
        ),
        "files": files,
    }
    return {**semantic, "failure_id": canonical_sha256(semantic)}


def _write_new_fsync(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _write_new_fsync_at(directory_fd: int, *, name: str, content: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing diagnostic gate")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_directory_at(parent_fd: int, *, name: str) -> int:
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(f"publication entry is not a real directory: {name}")
    return descriptor


def _read_exact_regular_at(directory_fd: int, *, name: str, expected: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size != len(expected):
            raise ValueError(f"published gate file metadata drifted: {name}")
        chunks: list[bytes] = []
        remaining = len(expected) + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    if b"".join(chunks) != expected:
        raise ValueError(f"published gate file bytes drifted: {name}")


def _verify_flat_directory_at(
    directory_fd: int, *, expected_files: Mapping[str, bytes]
) -> None:
    names = os.listdir(directory_fd)
    if len(names) != len(set(names)) or set(names) != set(expected_files):
        raise ValueError("published gate directory is not the exact flat closed world")
    for name, expected in expected_files.items():
        child = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(child.st_mode):
            raise ValueError(f"published gate contains a non-regular entry: {name}")
        _read_exact_regular_at(directory_fd, name=name, expected=expected)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_no_replace(source: Path, target: Path) -> None:
    if os.name == "posix":
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError(
                "atomic no-replace directory publication needs renameat2"
            )
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
                raise FileExistsError(target)
            raise OSError(error, os.strerror(error), target)
        return
    source.rename(target)


def _rename_directory_no_replace_at(
    parent_fd: int, *, source_name: str, target_name: str
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace directory publication needs renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_fd,
        os.fsencode(source_name),
        parent_fd,
        os.fsencode(target_name),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(target_name)
        raise OSError(error, os.strerror(error), target_name)


def _publish_preflight_gate(
    output_dir: Path,
    *,
    artifacts: Mapping[str, bytes],
    run_id: str,
    launch_plan: LaunchPlanArtifact,
    lock_artifact: LockHandshakeArtifact,
    preflight_report: Mapping[str, Any],
    overhead_report: Mapping[str, Any],
    python_runtime_lock: PythonRuntimeLockArtifact,
    expected_output_parent: os.stat_result,
) -> dict[str, Any]:
    target = _absolute_lexical(output_dir)
    _assert_existing_nonsymlink_chain(target.parent, label="output parent")
    payloads = dict(artifacts)
    gate = _gate_manifest_payload(
        payloads,
        run_id=run_id,
        launch_plan=launch_plan,
        lock_artifact=lock_artifact,
        preflight_report=preflight_report,
        overhead_report=overhead_report,
        python_runtime_lock=python_runtime_lock,
    )
    payloads[PREFLIGHT_GATE_MANIFEST] = serialize_json(gate)
    metadata = {
        name: {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "rows": content.count(b"\n") if name.endswith(".jsonl") else None,
        }
        for name, content in sorted(payloads.items())
    }
    outer = {
        "schema": FILE_BUNDLE_SCHEMA_VERSION,
        "bundle_status": "complete",
        "artifact_kind": PREFLIGHT_GATE_ARTIFACT_KIND,
        "semantic_manifest": PREFLIGHT_GATE_MANIFEST,
        "bundle_id": gate["gate_id"],
        "artifacts": metadata,
    }
    disk_files = {**payloads, "artifact_manifest.json": serialize_json(outer)}

    if os.name == "posix":
        parent_fd = os.open(
            target.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        staging_name = f".{target.name}.staging-{uuid.uuid4().hex}"
        published = False
        try:
            parent_info = os.fstat(parent_fd)
            if not os.path.samestat(parent_info, expected_output_parent):
                raise RuntimeError("output parent identity changed before publication")
            try:
                os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError(f"refusing to overwrite preflight gate: {target}")
            os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
            staging_fd = _open_directory_at(parent_fd, name=staging_name)
            try:
                for name, content in sorted(disk_files.items()):
                    _write_new_fsync_at(staging_fd, name=name, content=content)
                _verify_flat_directory_at(staging_fd, expected_files=disk_files)
                os.fsync(staging_fd)
            finally:
                os.close(staging_fd)
            _rename_directory_no_replace_at(
                parent_fd,
                source_name=staging_name,
                target_name=target.name,
            )
            published = True
            os.fsync(parent_fd)
            target_fd = _open_directory_at(parent_fd, name=target.name)
            try:
                _verify_flat_directory_at(target_fd, expected_files=disk_files)
            finally:
                os.close(target_fd)
            return gate
        except Exception as exc:
            retained = target.name if published else staging_name
            raise RuntimeError(
                "diagnostic preflight gate publication failed; retained at "
                f"{target.parent / retained}: {exc}"
            ) from exc
        finally:
            os.close(parent_fd)

    if not os.path.samestat(
        target.parent.stat(follow_symlinks=False), expected_output_parent
    ):
        raise RuntimeError("output parent identity changed before publication")
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite preflight gate: {target}")
    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700, exist_ok=False)
    try:
        for name, content in sorted(disk_files.items()):
            _write_new_fsync(staging / name, content)
        read_preflight_gate_bundle(staging)
        _rename_directory_no_replace(staging, target)
        published_gate = read_preflight_gate_bundle(target)
        if published_gate != gate:
            raise RuntimeError("published preflight gate differs from staged semantics")
        return published_gate
    except Exception as exc:
        raise RuntimeError(
            f"diagnostic preflight gate staging failed; preserved at {staging}: {exc}"
        ) from exc


def _publish_preflight_failure(
    output_dir: Path,
    *,
    artifacts: Mapping[str, bytes],
    run_id: str,
    launch_plan: LaunchPlanArtifact,
    lock_artifact: LockHandshakeArtifact,
    failure_report: Mapping[str, Any],
    python_runtime_lock: PythonRuntimeLockArtifact,
    expected_output_parent: os.stat_result,
) -> dict[str, Any]:
    """Exclusively publish terminal failure evidence; never create a PASS gate."""

    target = _absolute_lexical(output_dir)
    _assert_existing_nonsymlink_chain(target.parent, label="failure-output parent")
    payloads = dict(artifacts)
    manifest = _failure_manifest_payload(
        payloads,
        run_id=run_id,
        launch_plan=launch_plan,
        lock_artifact=lock_artifact,
        failure_report=failure_report,
        python_runtime_lock=python_runtime_lock,
    )
    payloads[PREFLIGHT_FAILURE_MANIFEST] = serialize_json(manifest)
    metadata = {
        name: {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "rows": content.count(b"\n") if name.endswith(".jsonl") else None,
        }
        for name, content in sorted(payloads.items())
    }
    outer = {
        "schema": FILE_BUNDLE_SCHEMA_VERSION,
        "bundle_status": "complete",
        "artifact_kind": PREFLIGHT_FAILURE_ARTIFACT_KIND,
        "semantic_manifest": PREFLIGHT_FAILURE_MANIFEST,
        "bundle_id": manifest["failure_id"],
        "artifacts": metadata,
    }
    disk_files = {**payloads, "artifact_manifest.json": serialize_json(outer)}

    if os.name == "posix":
        parent_fd = os.open(
            target.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        staging_name = f".{target.name}.staging-{uuid.uuid4().hex}"
        published = False
        try:
            parent_info = os.fstat(parent_fd)
            if not os.path.samestat(parent_info, expected_output_parent):
                raise RuntimeError("failure-output parent identity changed")
            try:
                os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError(
                    f"refusing to overwrite preflight failure: {target}"
                )
            os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
            staging_fd = _open_directory_at(parent_fd, name=staging_name)
            try:
                for name, content in sorted(disk_files.items()):
                    _write_new_fsync_at(staging_fd, name=name, content=content)
                _verify_flat_directory_at(staging_fd, expected_files=disk_files)
                os.fsync(staging_fd)
            finally:
                os.close(staging_fd)
            _rename_directory_no_replace_at(
                parent_fd,
                source_name=staging_name,
                target_name=target.name,
            )
            published = True
            os.fsync(parent_fd)
            target_fd = _open_directory_at(parent_fd, name=target.name)
            try:
                _verify_flat_directory_at(target_fd, expected_files=disk_files)
            finally:
                os.close(target_fd)
            return manifest
        except Exception as exc:
            retained = target.name if published else staging_name
            raise RuntimeError(
                "diagnostic preflight-failure publication failed; retained at "
                f"{target.parent / retained}: {exc}"
            ) from exc
        finally:
            os.close(parent_fd)

    if not os.path.samestat(
        target.parent.stat(follow_symlinks=False), expected_output_parent
    ):
        raise RuntimeError("failure-output parent identity changed")
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite preflight failure: {target}")
    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700, exist_ok=False)
    try:
        for name, content in sorted(disk_files.items()):
            _write_new_fsync(staging / name, content)
        read_preflight_failure_bundle(staging)
        _rename_directory_no_replace(staging, target)
        published_manifest = read_preflight_failure_bundle(target)
        if published_manifest != manifest:
            raise RuntimeError(
                "published preflight failure differs from staged semantics"
            )
        return published_manifest
    except Exception as exc:
        raise RuntimeError(
            "diagnostic preflight-failure staging failed; preserved at "
            f"{staging}: {exc}"
        ) from exc


def read_preflight_gate_bundle(directory: str | Path) -> dict[str, Any]:
    """Read the diagnostic gate while refusing any training-evidence upgrade."""

    bundle = read_closed_world_bundle(directory)
    outer = bundle.artifact_manifest
    if (
        outer.get("schema") != FILE_BUNDLE_SCHEMA_VERSION
        or outer.get("artifact_kind") != PREFLIGHT_GATE_ARTIFACT_KIND
        or outer.get("semantic_manifest") != PREFLIGHT_GATE_MANIFEST
    ):
        raise ValueError("bundle is not a diagnostic M04a preflight gate")
    expected_names = {*PREFLIGHT_GATE_FILES, PREFLIGHT_GATE_MANIFEST}
    if set(bundle.artifacts) != expected_names:
        raise ValueError("diagnostic preflight gate is not closed-world")
    gate_bytes = bundle.read_bytes(PREFLIGHT_GATE_MANIFEST)
    gate = _strict_json_bytes(gate_bytes, label=PREFLIGHT_GATE_MANIFEST)
    fields = {
        "schema",
        "artifact_kind",
        "disposition",
        "run_id",
        "launch_plan_artifact_sha256",
        "launch_plan_id",
        "lock_handshake_artifact_sha256",
        "lock_handshake_id",
        "python_runtime_lock_artifact_sha256",
        "python_runtime_lock_id",
        "preflight_report_id",
        "overhead_cost_probe_id",
        "training_evidence_eligible",
        "requires_same_process_exact_preflight_rerun",
        "lock_release_policy",
        "files",
        "gate_id",
    }
    if set(gate) != fields or serialize_json(gate) != gate_bytes:
        raise ValueError("diagnostic preflight gate manifest is not canonical/exact")
    if (
        gate["schema"] != PREFLIGHT_GATE_SCHEMA_VERSION
        or gate["artifact_kind"] != PREFLIGHT_GATE_ARTIFACT_KIND
        or gate["disposition"] != PREFLIGHT_GATE_ARTIFACT_KIND
        or gate["training_evidence_eligible"] is not False
        or gate["requires_same_process_exact_preflight_rerun"] is not True
        or gate["lock_release_policy"] != "release_after_atomic_gate_publication"
    ):
        raise ValueError("diagnostic preflight disposition drifted")
    for field in (
        "launch_plan_artifact_sha256",
        "launch_plan_id",
        "lock_handshake_artifact_sha256",
        "lock_handshake_id",
        "python_runtime_lock_artifact_sha256",
        "python_runtime_lock_id",
        "preflight_report_id",
        "overhead_cost_probe_id",
        "gate_id",
    ):
        _sha256(gate[field], field=field)
    semantic = dict(gate)
    claimed_id = semantic.pop("gate_id")
    if claimed_id != canonical_sha256(semantic) or outer["bundle_id"] != claimed_id:
        raise ValueError("diagnostic preflight gate identity mismatch")
    rows = gate["files"]
    if not isinstance(rows, list) or len(rows) != len(PREFLIGHT_GATE_FILES):
        raise ValueError("diagnostic preflight gate file rows are incomplete")
    expected_rows = [
        {
            "path": name,
            "sha256": bundle.artifacts[name]["sha256"],
            "bytes": bundle.artifacts[name]["bytes"],
            "rows": bundle.artifacts[name]["rows"],
        }
        for name in sorted(PREFLIGHT_GATE_FILES)
    ]
    if rows != expected_rows:
        raise ValueError("diagnostic preflight gate file rows do not close")
    report = bundle.read_json("preflight.json")
    cost = bundle.read_json("preflight_overhead_cost_report.json")
    ledger = bundle.read_jsonl("preflight_training_cost_ledger.jsonl")
    diagnostic = bundle.read_bytes("preflight_diagnostic_checkpoint.pt")
    launch_bytes = bundle.read_bytes("launch_plan.json")
    lock_bytes = bundle.read_bytes("lock_handshake_artifact.json")
    environment_bytes = bundle.read_bytes("environment_manifest.json")
    runtime_lock_bytes = bundle.read_bytes("python-runtime-lock.json")
    launch = validate_launch_plan_payload(
        _strict_json_bytes(launch_bytes, label="launch_plan.json")
    )
    lock = validate_lock_handshake_payload(
        _strict_json_bytes(lock_bytes, label="lock_handshake_artifact.json")
    )
    environment = _strict_json_bytes(
        environment_bytes, label="environment_manifest.json"
    )
    runtime_lock = validate_python_runtime_lock_payload(
        _strict_json_bytes(runtime_lock_bytes, label="python-runtime-lock.json")
    )
    validate_environment_manifest_artifact(
        environment,
        launch_plan=launch,
        expected_gpu_uuid=lock["gpu_uuid"],
        expected_launcher_sha256=launch["expected_input_artifacts"][
            "remote_launcher.py"
        ],
        python_runtime_lock=runtime_lock,
    )
    expected_visible_files = dict(launch["expected_input_artifacts"])
    expected_visible_files[CONDA_EXPLICIT_BASENAME] = launch["conda_explicit_sha256"]
    report_runtime = report.get("runtime")
    if (
        serialize_json(launch) != launch_bytes
        or serialize_json(lock) != lock_bytes
        or serialize_json(environment) != environment_bytes
        or canonical_python_runtime_lock_bytes(runtime_lock) != runtime_lock_bytes
        or serialize_json(report) != bundle.read_bytes("preflight.json")
        or serialize_json(cost)
        != bundle.read_bytes("preflight_overhead_cost_report.json")
        or serialize_jsonl(ledger)
        != bundle.read_bytes("preflight_training_cost_ledger.jsonl")
        or hashlib.sha256(launch_bytes).hexdigest()
        != gate["launch_plan_artifact_sha256"]
        or launch["launch_plan_id"] != gate["launch_plan_id"]
        or hashlib.sha256(lock_bytes).hexdigest()
        != gate["lock_handshake_artifact_sha256"]
        or lock["handshake_id"] != gate["lock_handshake_id"]
        or hashlib.sha256(runtime_lock_bytes).hexdigest()
        != gate["python_runtime_lock_artifact_sha256"]
        or runtime_lock["runtime_lock_id"] != gate["python_runtime_lock_id"]
        or gate["python_runtime_lock_artifact_sha256"]
        != launch["expected_input_artifacts"]["python-runtime-lock.json"]
        or launch["run_id"] != gate["run_id"]
        or lock["run_id"] != gate["run_id"]
        or lock["launch_plan_sha256"] != gate["launch_plan_artifact_sha256"]
        or lock["launcher_sha256"]
        != launch["expected_input_artifacts"]["remote_launcher.py"]
        or lock["attempt_nonce"] != launch["attempt_nonce"]
        or lock["lock_path"]
        != (
            PurePosixPath(launch["remote_project_root"])
            / "locks"
            / f"gpu-{lock['gpu_uuid']}.lock"
        ).as_posix()
        or environment.get("run_id") != gate["run_id"]
        or environment.get("gpu_uuid") != lock["gpu_uuid"]
        or environment.get("launcher_sha256")
        != launch["expected_input_artifacts"]["remote_launcher.py"]
        or environment.get("runtime_source_sha256")
        != launch["runtime_source_fingerprint_sha256"]
        or environment.get("test_source_sha256")
        != launch["test_source_fingerprint_sha256"]
        or environment.get("conda_explicit_sha256") != launch["conda_explicit_sha256"]
        or environment.get("ordered_import_roots") != launch["ordered_import_roots"]
        or environment.get("visible_files") != expected_visible_files
        or report.get("status") != "PASS"
        or report.get("report_id") != gate["preflight_report_id"]
        or not isinstance(report_runtime, dict)
        or report_runtime.get("run_id") != gate["run_id"]
        or report_runtime.get("gpu_uuid") != lock["gpu_uuid"]
        or report.get("overhead_cost_probe_id") != gate["overhead_cost_probe_id"]
        or report.get("lock_handshake_artifact_sha256")
        != gate["lock_handshake_artifact_sha256"]
        or cost.get("status") != "PASS"
        or cost.get("probe_id") != gate["overhead_cost_probe_id"]
        or cost.get("run_id") != gate["run_id"]
        or len(ledger) != 100
        or len(diagnostic)
        != cost.get("checkpoint_cost_probe", {}).get("checkpoint_bytes")
        or hashlib.sha256(diagnostic).hexdigest()
        != cost.get("checkpoint_cost_probe", {}).get("checkpoint_sha256")
    ):
        raise ValueError("diagnostic preflight payload closure failed")
    commitment = launch["validation_manifest_commitment"]
    validate_preflight_artifact(
        report,
        ledger,
        validation_episodes_per_pass=commitment["row_count"],
        expected_validation_outer_manifest_sha256=commitment[
            "outer_artifact_manifest_sha256"
        ],
        expected_validation_jsonl_sha256=commitment["jsonl_sha256"],
        expected_validation_summary_id=commitment["summary_id"],
        overhead_cost_probe_report=cost,
        diagnostic_checkpoint_snapshot=diagnostic,
        lock_handshake_artifact=lock,
        lock_handshake_artifact_bytes=lock_bytes,
        expected_lock_handshake_artifact_sha256=gate["lock_handshake_artifact_sha256"],
        expected_config_sha256=launch["training_config_sha256"],
        expected_runtime_source_sha256=launch["runtime_source_fingerprint_sha256"],
        expected_test_source_sha256=launch["test_source_fingerprint_sha256"],
        expected_launcher_sha256=launch["expected_input_artifacts"][
            "remote_launcher.py"
        ],
        expected_launch_plan_sha256=gate["launch_plan_artifact_sha256"],
        expected_remote_project_root=launch["remote_project_root"],
        expected_attempt_nonce=launch["attempt_nonce"],
    )
    return gate


def read_preflight_failure_bundle(directory: str | Path) -> dict[str, Any]:
    """Read and replay a terminal failure bundle without treating it as a gate."""

    bundle = read_closed_world_bundle(directory)
    outer = bundle.artifact_manifest
    if (
        outer.get("schema") != FILE_BUNDLE_SCHEMA_VERSION
        or outer.get("artifact_kind") != PREFLIGHT_FAILURE_ARTIFACT_KIND
        or outer.get("semantic_manifest") != PREFLIGHT_FAILURE_MANIFEST
    ):
        raise ValueError("bundle is not a diagnostic M04a preflight failure")
    manifest_bytes = bundle.read_bytes(PREFLIGHT_FAILURE_MANIFEST)
    manifest = _strict_json_bytes(
        manifest_bytes, label=PREFLIGHT_FAILURE_MANIFEST
    )
    fields = {
        "schema",
        "artifact_kind",
        "disposition",
        "run_id",
        "failure_code",
        "evidence_completeness",
        "launch_plan_artifact_sha256",
        "launch_plan_id",
        "lock_handshake_artifact_sha256",
        "lock_handshake_id",
        "python_runtime_lock_artifact_sha256",
        "python_runtime_lock_id",
        "failure_report_id",
        "overhead_cost_probe_id",
        "training_evidence_eligible",
        "diagnostic_checkpoint_selectable",
        "same_run_retry_allowed",
        "same_run_retry_policy",
        "preflight_gate_created",
        "lock_release_policy",
        "files",
        "failure_id",
    }
    if set(manifest) != fields or serialize_json(manifest) != manifest_bytes:
        raise ValueError("diagnostic preflight-failure manifest is not canonical/exact")
    completeness = manifest.get("evidence_completeness")
    if completeness not in {"partial", "full_budget_probe"}:
        raise ValueError("diagnostic preflight-failure completeness is invalid")
    expected_payload_files = (
        PREFLIGHT_FAILURE_FULL_FILES
        if completeness == "full_budget_probe"
        else PREFLIGHT_FAILURE_BASE_FILES
    )
    expected_names = {*expected_payload_files, PREFLIGHT_FAILURE_MANIFEST}
    if set(bundle.artifacts) != expected_names:
        raise ValueError("diagnostic preflight failure is not closed-world")
    if (
        manifest["schema"] != PREFLIGHT_FAILURE_SCHEMA_VERSION
        or manifest["artifact_kind"] != PREFLIGHT_FAILURE_ARTIFACT_KIND
        or manifest["disposition"] != PREFLIGHT_FAILURE_ARTIFACT_KIND
        or manifest["training_evidence_eligible"] is not False
        or manifest["diagnostic_checkpoint_selectable"] is not False
        or manifest["same_run_retry_allowed"] is not False
        or manifest["same_run_retry_policy"]
        != "forbidden_after_terminal_preflight_failure"
        or manifest["preflight_gate_created"] is not False
        or manifest["lock_release_policy"]
        != "release_attempted_by_failed_exact_preflight_process_exit_backstop"
    ):
        raise ValueError("diagnostic preflight-failure disposition drifted")
    for field in (
        "launch_plan_artifact_sha256",
        "launch_plan_id",
        "lock_handshake_artifact_sha256",
        "lock_handshake_id",
        "python_runtime_lock_artifact_sha256",
        "python_runtime_lock_id",
        "failure_report_id",
        "failure_id",
    ):
        _sha256(manifest[field], field=field)
    if completeness == "full_budget_probe":
        _sha256(manifest["overhead_cost_probe_id"], field="overhead_cost_probe_id")
    elif manifest["overhead_cost_probe_id"] is not None:
        raise ValueError("partial failure cannot claim an overhead probe")
    semantic = dict(manifest)
    claimed_id = semantic.pop("failure_id")
    if claimed_id != canonical_sha256(semantic) or outer["bundle_id"] != claimed_id:
        raise ValueError("diagnostic preflight-failure identity mismatch")
    rows = manifest["files"]
    if not isinstance(rows, list) or len(rows) != len(expected_payload_files):
        raise ValueError("diagnostic preflight-failure file rows are incomplete")
    expected_rows = [
        {
            "path": name,
            "sha256": bundle.artifacts[name]["sha256"],
            "bytes": bundle.artifacts[name]["bytes"],
            "rows": bundle.artifacts[name]["rows"],
        }
        for name in sorted(expected_payload_files)
    ]
    if rows != expected_rows:
        raise ValueError("diagnostic preflight-failure file rows do not close")

    launch_bytes = bundle.read_bytes("launch_plan.json")
    lock_bytes = bundle.read_bytes("lock_handshake_artifact.json")
    environment_bytes = bundle.read_bytes("environment_manifest.json")
    runtime_lock_bytes = bundle.read_bytes("python-runtime-lock.json")
    report_bytes = bundle.read_bytes("preflight_failure.json")
    ledger_bytes = bundle.read_bytes("preflight_training_cost_ledger.jsonl")
    launch = validate_launch_plan_payload(
        _strict_json_bytes(launch_bytes, label="launch_plan.json")
    )
    lock = validate_lock_handshake_payload(
        _strict_json_bytes(lock_bytes, label="lock_handshake_artifact.json")
    )
    environment = _strict_json_bytes(
        environment_bytes, label="environment_manifest.json"
    )
    runtime_lock = validate_python_runtime_lock_payload(
        _strict_json_bytes(runtime_lock_bytes, label="python-runtime-lock.json")
    )
    report = _strict_json_bytes(report_bytes, label="preflight_failure.json")
    ledger = bundle.read_jsonl("preflight_training_cost_ledger.jsonl")
    validate_environment_manifest_artifact(
        environment,
        launch_plan=launch,
        expected_gpu_uuid=lock["gpu_uuid"],
        expected_launcher_sha256=launch["expected_input_artifacts"][
            "remote_launcher.py"
        ],
        python_runtime_lock=runtime_lock,
    )
    expected_visible_files = dict(launch["expected_input_artifacts"])
    expected_visible_files[CONDA_EXPLICIT_BASENAME] = launch["conda_explicit_sha256"]
    if (
        serialize_json(launch) != launch_bytes
        or serialize_json(lock) != lock_bytes
        or serialize_json(environment) != environment_bytes
        or canonical_python_runtime_lock_bytes(runtime_lock) != runtime_lock_bytes
        or serialize_json(report) != report_bytes
        or serialize_jsonl(ledger) != ledger_bytes
        or hashlib.sha256(launch_bytes).hexdigest()
        != manifest["launch_plan_artifact_sha256"]
        or launch["launch_plan_id"] != manifest["launch_plan_id"]
        or hashlib.sha256(lock_bytes).hexdigest()
        != manifest["lock_handshake_artifact_sha256"]
        or lock["handshake_id"] != manifest["lock_handshake_id"]
        or hashlib.sha256(runtime_lock_bytes).hexdigest()
        != manifest["python_runtime_lock_artifact_sha256"]
        or runtime_lock["runtime_lock_id"] != manifest["python_runtime_lock_id"]
        or manifest["python_runtime_lock_artifact_sha256"]
        != launch["expected_input_artifacts"]["python-runtime-lock.json"]
        or launch["run_id"] != manifest["run_id"]
        or lock["run_id"] != manifest["run_id"]
        or lock["launch_plan_sha256"]
        != manifest["launch_plan_artifact_sha256"]
        or lock["launcher_sha256"]
        != launch["expected_input_artifacts"]["remote_launcher.py"]
        or lock["attempt_nonce"] != launch["attempt_nonce"]
        or lock["lock_path"]
        != (
            PurePosixPath(launch["remote_project_root"])
            / "locks"
            / f"gpu-{lock['gpu_uuid']}.lock"
        ).as_posix()
        or environment.get("run_id") != manifest["run_id"]
        or environment.get("gpu_uuid") != lock["gpu_uuid"]
        or environment.get("launcher_sha256")
        != launch["expected_input_artifacts"]["remote_launcher.py"]
        or environment.get("runtime_source_sha256")
        != launch["runtime_source_fingerprint_sha256"]
        or environment.get("test_source_sha256")
        != launch["test_source_fingerprint_sha256"]
        or environment.get("conda_explicit_sha256")
        != launch["conda_explicit_sha256"]
        or environment.get("ordered_import_roots") != launch["ordered_import_roots"]
        or environment.get("visible_files") != expected_visible_files
        or report.get("report_id") != manifest["failure_report_id"]
        or report.get("failure_code") != manifest["failure_code"]
        or report.get("evidence_completeness") != completeness
    ):
        raise ValueError("diagnostic preflight-failure parent closure failed")

    overhead: dict[str, Any] | None = None
    training: dict[str, Any] | None = None
    inference: dict[str, Any] | None = None
    diagnostic: bytes | None = None
    if completeness == "full_budget_probe":
        overhead_bytes = bundle.read_bytes("preflight_overhead_cost_report.json")
        training_bytes = bundle.read_bytes("preflight_training_summary.json")
        inference_bytes = bundle.read_bytes("preflight_inference_summary.json")
        overhead = _strict_json_bytes(
            overhead_bytes, label="preflight_overhead_cost_report.json"
        )
        training = _strict_json_bytes(
            training_bytes, label="preflight_training_summary.json"
        )
        inference = _strict_json_bytes(
            inference_bytes, label="preflight_inference_summary.json"
        )
        diagnostic = bundle.read_bytes("preflight_diagnostic_checkpoint.pt")
        if (
            serialize_json(overhead) != overhead_bytes
            or serialize_json(training) != training_bytes
            or serialize_json(inference) != inference_bytes
            or overhead.get("probe_id") != manifest["overhead_cost_probe_id"]
        ):
            raise ValueError("full budget-failure attachment serialization drifted")
    _validate_failure_report_payload(
        report,
        ledger,
        run_id=manifest["run_id"],
        gpu_uuid=lock["gpu_uuid"],
        validation_manifest_commitment=launch["validation_manifest_commitment"],
        lock_handshake_artifact_sha256=manifest[
            "lock_handshake_artifact_sha256"
        ],
        lock_handshake_payload=lock,
        config_sha256=launch["training_config_sha256"],
        runtime_source_sha256=launch["runtime_source_fingerprint_sha256"],
        test_source_sha256=launch["test_source_fingerprint_sha256"],
        overhead_report=overhead,
        training_summary=training,
        inference_summary=inference,
        diagnostic_checkpoint=diagnostic,
    )
    return manifest


def run_preflight_command(
    *,
    launch_plan_path: str | Path,
    launch_plan_sha256: str,
    python_runtime_lock_path: str | Path,
    python_runtime_lock_sha256: str,
    validation_manifest_dir: str | Path,
    lock_handshake_path: str | Path,
    lock_handshake_sha256: str,
    gpu_uuid: str,
    visible_root: str | Path,
    launcher_path: str | Path,
    cost_probe_scratch_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate committed inputs, run exact CUDA preflight, and publish a gate."""

    launch_artifact = read_launch_plan_artifact(
        launch_plan_path,
        expected_artifact_sha256=launch_plan_sha256,
    )
    plan = launch_artifact.payload
    if plan["training_config_sha256"] != training_config_sha256():
        raise ValueError("launch plan differs from the frozen training config")
    gpu = _validate_process_identity(plan, gpu_uuid=gpu_uuid)
    root, visible_files, visible_snapshots = _visible_input_snapshots(
        visible_root,
        expected_input_artifacts=plan["expected_input_artifacts"],
        conda_explicit_sha256=plan["conda_explicit_sha256"],
    )
    run_root, scratch, output, launcher = _validate_path_layout(
        plan=plan,
        visible_root=root,
        launch_plan_path=launch_plan_path,
        python_runtime_lock_path=python_runtime_lock_path,
        lock_handshake_path=lock_handshake_path,
        validation_manifest_dir=validation_manifest_dir,
        launcher_path=launcher_path,
        cost_probe_scratch_dir=cost_probe_scratch_dir,
        output_dir=output_dir,
    )
    run_root_identity = run_root.stat(follow_symlinks=False)
    output_parent_identity = output.parent.stat(follow_symlinks=False)
    launcher_snapshot = _read_committed_regular(
        launcher,
        expected_sha256=plan["expected_input_artifacts"]["remote_launcher.py"],
        label="executing M04a launcher",
        maximum_bytes=1024 * 1024,
    )
    if (
        os.environ.get("AFTS_EVIDENCE_LAUNCHER") is None
        or _absolute_lexical(os.environ["AFTS_EVIDENCE_LAUNCHER"]) != launcher
    ):
        raise RuntimeError("--launcher-path differs from the executing launcher")
    if launcher_snapshot != visible_snapshots["remote_launcher.py"]:
        raise ValueError("executing launcher differs from the visible reviewed copy")

    expected_runtime_lock_sha256 = plan["expected_input_artifacts"][
        "python-runtime-lock.json"
    ]
    if python_runtime_lock_sha256 != expected_runtime_lock_sha256:
        raise ValueError("runtime-lock CLI SHA differs from launch plan")
    python_runtime_lock = read_python_runtime_lock_artifact(
        python_runtime_lock_path,
        expected_artifact_sha256=expected_runtime_lock_sha256,
    )
    if python_runtime_lock.snapshot != visible_snapshots["python-runtime-lock.json"]:
        raise ValueError("runtime-lock reader snapshot differs from visible input")
    runtime_lock_live_kwargs = _runtime_lock_live_kwargs(
        plan=plan, runtime_lock=python_runtime_lock
    )
    fresh_runtime_lock = validate_live_python_runtime_lock(
        python_runtime_lock,
        expected_artifact_sha256=expected_runtime_lock_sha256,
        **runtime_lock_live_kwargs,
    )

    runtime_fingerprint = plan["runtime_source_fingerprint_sha256"]
    tests_fingerprint = plan["test_source_fingerprint_sha256"]
    verify_source_snapshot_zip(
        visible_snapshots["reviewed_runtime_source.zip"],
        expected_fingerprint_sha256=runtime_fingerprint,
    )
    verify_test_source_snapshot_zip(
        visible_snapshots["reviewed_test_snapshot.zip"],
        expected_fingerprint_sha256=tests_fingerprint,
    )
    if runtime_source_fingerprint() != runtime_fingerprint:
        raise RuntimeError("executing runtime source differs from the launch plan")

    validation_commitment = _validation_commitment_from_plan(
        plan,
        validation_manifest_dir=validation_manifest_dir,
        visible_snapshots=visible_snapshots,
    )
    lock_artifact = read_lock_handshake_artifact(
        lock_handshake_path,
        expected_artifact_sha256=lock_handshake_sha256,
        expected_run_id=plan["run_id"],
        expected_gpu_uuid=gpu,
        expected_launcher_sha256=plan["expected_input_artifacts"]["remote_launcher.py"],
        expected_launch_plan_sha256=launch_artifact.artifact_sha256,
        expected_remote_project_root=plan["remote_project_root"],
        expected_attempt_nonce=plan["attempt_nonce"],
    )

    dependencies = _load_torch_dependencies()
    validate_imported_torch_runtime(
        python_runtime_lock,
        expected_artifact_sha256=expected_runtime_lock_sha256,
        imported_torch=dependencies.imported_torch,
    )
    attestation = dependencies.configure_deterministic_cuda()
    environment = dependencies.environment_manifest(
        attestation=attestation,
        launcher_path=launcher,
        runtime_source_sha256=runtime_fingerprint,
        test_source_sha256=tests_fingerprint,
        visible_root=root,
        expected_visible_files=visible_files,
        conda_explicit_path=root / CONDA_EXPLICIT_BASENAME,
        expected_conda_explicit_sha256=plan["conda_explicit_sha256"],
        python_runtime_lock_sha256=expected_runtime_lock_sha256,
        python_runtime_lock_id=fresh_runtime_lock["runtime_lock_id"],
        python_implementation=fresh_runtime_lock["python"]["implementation"],
        python_version=fresh_runtime_lock["python"]["version"],
        python_executable_sha256=fresh_runtime_lock["python"]["executable_sha256"],
        python_executable_bytes=fresh_runtime_lock["python"]["executable_bytes"],
    )
    environment = _bind_environment_to_plan(
        environment,
        plan=plan,
        gpu_uuid=gpu,
        visible_root=root,
        visible_files=visible_files,
        launcher=launcher,
        python_runtime_lock=python_runtime_lock,
        runtime_lock_payload=fresh_runtime_lock,
    )
    validate_environment_manifest_artifact(
        environment,
        launch_plan=plan,
        expected_gpu_uuid=gpu,
        expected_launcher_sha256=plan["expected_input_artifacts"]["remote_launcher.py"],
        python_runtime_lock=fresh_runtime_lock,
    )

    result: Any | None = None
    published: dict[str, Any] | None = None
    try:
        try:
            result = dependencies.run_exact_preflight(
                runtime_attestation=attestation,
                validation_manifest_commitment=validation_commitment,
                launch_plan_artifact=launch_artifact,
                run_root=run_root,
                cost_probe_scratch_dir=scratch,
                config_sha256=plan["training_config_sha256"],
                runtime_source_sha256=runtime_fingerprint,
                test_source_sha256=tests_fingerprint,
                lock_handshake_artifact=lock_artifact,
            )
        except dependencies.preflight_failure_type as failure:
            failure_report = getattr(failure, "report", None)
            failure_ledger = list(getattr(failure, "ledger_rows", ()))
            overhead = getattr(failure, "overhead_cost_probe_report", None)
            training = getattr(failure, "training_summary", None)
            inference = getattr(failure, "inference_summary", None)
            diagnostic = getattr(
                failure, "diagnostic_checkpoint_snapshot", None
            )
            _validate_failure_report_payload(
                failure_report,
                failure_ledger,
                run_id=plan["run_id"],
                gpu_uuid=gpu,
                validation_manifest_commitment=plan[
                    "validation_manifest_commitment"
                ],
                lock_handshake_artifact_sha256=lock_artifact.artifact_sha256,
                lock_handshake_payload=lock_artifact.payload,
                config_sha256=plan["training_config_sha256"],
                runtime_source_sha256=runtime_fingerprint,
                test_source_sha256=tests_fingerprint,
                overhead_report=overhead,
                training_summary=training,
                inference_summary=inference,
                diagnostic_checkpoint=diagnostic,
            )
            final_root, final_visible_files, final_visible_snapshots = (
                _visible_input_snapshots(
                    root,
                    expected_input_artifacts=plan["expected_input_artifacts"],
                    conda_explicit_sha256=plan["conda_explicit_sha256"],
                )
            )
            if (
                not os.path.samefile(final_root, root)
                or final_visible_files != visible_files
                or final_visible_snapshots != visible_snapshots
                or runtime_source_fingerprint() != runtime_fingerprint
            ):
                raise RuntimeError(
                    "committed source/input snapshots changed during failed preflight"
                )
            validate_live_python_runtime_lock(
                python_runtime_lock,
                expected_artifact_sha256=expected_runtime_lock_sha256,
                **runtime_lock_live_kwargs,
            )
            validate_imported_torch_runtime(
                python_runtime_lock,
                expected_artifact_sha256=expected_runtime_lock_sha256,
                imported_torch=dependencies.imported_torch,
            )
            failure_output = _failure_output_path(output)
            _assert_publication_parent_unchanged(
                run_root=run_root,
                output=output,
                expected_run_root=run_root_identity,
                expected_output_parent=output_parent_identity,
            )
            _assert_publication_parent_unchanged(
                run_root=run_root,
                output=failure_output,
                expected_run_root=run_root_identity,
                expected_output_parent=output_parent_identity,
            )
            failure_artifacts = {
                "environment_manifest.json": serialize_json(environment),
                "launch_plan.json": launch_artifact.snapshot,
                "lock_handshake_artifact.json": lock_artifact.snapshot,
                "preflight_failure.json": serialize_json(failure_report),
                "preflight_training_cost_ledger.jsonl": serialize_jsonl(
                    failure_ledger
                ),
                "python-runtime-lock.json": python_runtime_lock.snapshot,
            }
            if failure_report["evidence_completeness"] == "full_budget_probe":
                failure_artifacts.update(
                    {
                        "preflight_diagnostic_checkpoint.pt": diagnostic,
                        "preflight_inference_summary.json": serialize_json(
                            inference
                        ),
                        "preflight_overhead_cost_report.json": serialize_json(
                            overhead
                        ),
                        "preflight_training_summary.json": serialize_json(
                            training
                        ),
                    }
                )
            failure_manifest = _publish_preflight_failure(
                failure_output,
                artifacts=failure_artifacts,
                run_id=plan["run_id"],
                launch_plan=launch_artifact,
                lock_artifact=lock_artifact,
                failure_report=failure_report,
                python_runtime_lock=python_runtime_lock,
                expected_output_parent=output_parent_identity,
            )
            setattr(failure, "published_failure_dir", str(failure_output))
            setattr(failure, "published_failure_id", failure_manifest["failure_id"])
            raise
        report = result.report
        ledger_rows = tuple(result.ledger_rows)
        overhead_report = result.overhead_cost_probe_report
        diagnostic = result.diagnostic_checkpoint_snapshot
        if (
            not isinstance(report, dict)
            or report.get("status") != "PASS"
            or not isinstance(overhead_report, dict)
            or overhead_report.get("status") != "PASS"
            or len(ledger_rows) != 100
            or type(diagnostic) is not bytes
            or result.held_lock_handshake.artifact.snapshot != lock_artifact.snapshot
        ):
            raise ValueError("exact preflight result is not publishable")
        final_root, final_visible_files, final_visible_snapshots = (
            _visible_input_snapshots(
                root,
                expected_input_artifacts=plan["expected_input_artifacts"],
                conda_explicit_sha256=plan["conda_explicit_sha256"],
            )
        )
        if (
            not os.path.samefile(final_root, root)
            or final_visible_files != visible_files
            or final_visible_snapshots != visible_snapshots
            or runtime_source_fingerprint() != runtime_fingerprint
        ):
            raise RuntimeError(
                "committed source/input snapshots changed during preflight"
            )
        validate_live_python_runtime_lock(
            python_runtime_lock,
            expected_artifact_sha256=expected_runtime_lock_sha256,
            **runtime_lock_live_kwargs,
        )
        validate_imported_torch_runtime(
            python_runtime_lock,
            expected_artifact_sha256=expected_runtime_lock_sha256,
            imported_torch=dependencies.imported_torch,
        )
        _assert_publication_parent_unchanged(
            run_root=run_root,
            output=output,
            expected_run_root=run_root_identity,
            expected_output_parent=output_parent_identity,
        )
        artifacts = {
            "environment_manifest.json": serialize_json(environment),
            "launch_plan.json": launch_artifact.snapshot,
            "lock_handshake_artifact.json": lock_artifact.snapshot,
            "preflight.json": serialize_json(report),
            "preflight_diagnostic_checkpoint.pt": diagnostic,
            "preflight_overhead_cost_report.json": serialize_json(overhead_report),
            "preflight_training_cost_ledger.jsonl": serialize_jsonl(ledger_rows),
            "python-runtime-lock.json": python_runtime_lock.snapshot,
        }
        published = _publish_preflight_gate(
            output,
            artifacts=artifacts,
            run_id=plan["run_id"],
            launch_plan=launch_artifact,
            lock_artifact=lock_artifact,
            preflight_report=report,
            overhead_report=overhead_report,
            python_runtime_lock=python_runtime_lock,
            expected_output_parent=output_parent_identity,
        )
    finally:
        if result is not None:
            active_exception = sys.exc_info()[1]
            try:
                dependencies.release_held_gpu_lock(result.held_lock_handshake)
            except BaseException as release_error:
                if active_exception is None:
                    raise
                raise RuntimeError(
                    "preflight failed and held GPU-lock release also failed: "
                    f"{release_error!r}"
                ) from active_exception
    if published is None:
        raise RuntimeError(
            "preflight command ended without a published diagnostic gate"
        )
    return {
        "status": "published_diagnostic_preflight_gate",
        "disposition": PREFLIGHT_GATE_ARTIFACT_KIND,
        "output_dir": str(output),
        "run_id": plan["run_id"],
        "gate_id": published["gate_id"],
        "launch_plan_artifact_sha256": launch_artifact.artifact_sha256,
        "lock_handshake_artifact_sha256": lock_artifact.artifact_sha256,
        "training_evidence_eligible": False,
        "requires_same_process_exact_preflight_rerun": True,
    }


__all__ = [
    "CONDA_EXPLICIT_BASENAME",
    "PREFLIGHT_FAILURE_ARTIFACT_KIND",
    "PREFLIGHT_FAILURE_DIRECTORY",
    "PREFLIGHT_FAILURE_MANIFEST",
    "PREFLIGHT_FAILURE_SCHEMA_VERSION",
    "PREFLIGHT_GATE_ARTIFACT_KIND",
    "PREFLIGHT_GATE_FILES",
    "PREFLIGHT_GATE_MANIFEST",
    "PREFLIGHT_GATE_SCHEMA_VERSION",
    "read_preflight_failure_bundle",
    "read_preflight_gate_bundle",
    "run_preflight_command",
]
