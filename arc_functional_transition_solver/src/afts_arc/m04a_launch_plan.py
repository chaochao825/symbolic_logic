"""Pure-Python schema and externally committed reader for M04a launch plans.

The launch plan is the immutable parent of the remote launcher, lock handshake,
and training evidence.  Its artifact SHA-256 commits the canonical file bytes;
``launch_plan_id`` separately commits the plan's semantic JSON payload.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .m04a_contract import canonical_sha256
from .m04a_train_contract import training_config_sha256
from .manifest import serialize_json


LAUNCH_PLAN_SCHEMA_VERSION = "afts-m04a-launch-plan/v0.2"
VALIDATION_MANIFEST_COMMITMENT_SCHEMA_VERSION = (
    "afts-m04a-validation-manifest-commitment/v0.1"
)
MAX_LAUNCH_PLAN_BYTES = 256 * 1024
ORDERED_IMPORT_ROOT_COUNTS = frozenset({2, 3})

EXPECTED_INPUT_ARTIFACT_PATHS = frozenset(
    {
        "data_split_manifest.json",
        "frozen_contract.md",
        "model_config.json",
        "ordered_fold_ids.json",
        "parameter_count.json",
        "python-runtime-lock.json",
        "quarantine_parent_ids.json",
        "remote_launcher.py",
        "reviewed_runtime_source.zip",
        "reviewed_test_snapshot.zip",
        "sanitized_shard_manifest.json",
        "schema_config_manifest.json",
        "seed_policy.json",
        "short_exact_replay_fixture.json",
        "validation_episode_manifest.jsonl",
        "validation_episode_manifest_summary.json",
        "validation_episode_outer_manifest.json",
    }
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_REPARSE_POINT_ATTRIBUTE = 0x400
_WINDOWS_FILE_SHARE_READ = 0x00000001
_WINDOWS_FILE_SHARE_WRITE = 0x00000002
_WINDOWS_FILE_SHARE_DELETE = 0x00000004
_WINDOWS_OPEN_EXISTING = 3
_WINDOWS_OPEN_REPARSE_POINT = 0x00200000
_WINDOWS_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_READER_TOKEN = object()
_PLAN_FIELDS = {
    "schema",
    "launch_plan_id",
    "attempt_nonce",
    "run_id",
    "remote_project_root",
    "run_root",
    "expected_input_artifacts",
    "runtime_source_fingerprint_sha256",
    "test_source_fingerprint_sha256",
    "training_config_sha256",
    "validation_manifest_commitment",
    "conda_explicit_sha256",
    "ordered_import_roots",
}
_VALIDATION_COMMITMENT_FIELDS = {
    "schema",
    "commitment_id",
    "outer_artifact_manifest_sha256",
    "jsonl_sha256",
    "summary_id",
    "row_count",
}


def _sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TypeError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _strict_int(value: object, *, field_name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(f"{field_name} must be an integer >= {minimum}")
    return value


def _lexical_posix_absolute(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise TypeError(f"{field_name} must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or path.anchor != "/"
        or path.as_posix() != value
        or value == "/"
        or value.endswith("/")
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise ValueError(f"{field_name} must be an absolute lexical POSIX path")
    return value


def _validate_validation_commitment(payload: object) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != _VALIDATION_COMMITMENT_FIELDS:
        raise ValueError(
            "validation_manifest_commitment fields do not match the exact schema"
        )
    normalized = dict(payload)
    if normalized["schema"] != VALIDATION_MANIFEST_COMMITMENT_SCHEMA_VERSION:
        raise ValueError("unsupported validation_manifest_commitment schema")
    _sha256(
        normalized["outer_artifact_manifest_sha256"],
        field_name="validation_manifest_commitment.outer_artifact_manifest_sha256",
    )
    _sha256(
        normalized["jsonl_sha256"],
        field_name="validation_manifest_commitment.jsonl_sha256",
    )
    _sha256(
        normalized["summary_id"], field_name="validation_manifest_commitment.summary_id"
    )
    _strict_int(
        normalized["row_count"],
        field_name="validation_manifest_commitment.row_count",
        minimum=1,
    )
    claimed_id = _sha256(
        normalized["commitment_id"],
        field_name="validation_manifest_commitment.commitment_id",
    )
    semantic = dict(normalized)
    semantic.pop("commitment_id")
    if claimed_id != canonical_sha256(semantic):
        raise ValueError("validation_manifest_commitment commitment_id mismatch")
    return normalized


def validate_launch_plan_payload(payload: object) -> dict[str, Any]:
    """Validate and defensively copy one exact v0.1 semantic payload."""

    if type(payload) is not dict or set(payload) != _PLAN_FIELDS:
        raise ValueError("launch-plan fields do not match the exact schema")
    normalized: dict[str, Any] = dict(payload)
    if normalized["schema"] != LAUNCH_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported launch-plan schema")

    _sha256(normalized["attempt_nonce"], field_name="attempt_nonce")

    run_id = normalized["run_id"]
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run_id must be one safe POSIX path component")
    remote_root = _lexical_posix_absolute(
        normalized["remote_project_root"], field_name="remote_project_root"
    )
    run_root = _lexical_posix_absolute(normalized["run_root"], field_name="run_root")
    expected_run_root = (PurePosixPath(remote_root) / "runs" / run_id).as_posix()
    if run_root != expected_run_root:
        raise ValueError("run_root must equal remote_project_root/runs/run_id")

    artifacts = normalized["expected_input_artifacts"]
    if type(artifacts) is not dict or set(artifacts) != EXPECTED_INPUT_ARTIFACT_PATHS:
        raise ValueError("expected_input_artifacts paths do not match the frozen set")
    normalized_artifacts: dict[str, str] = {}
    for name in sorted(EXPECTED_INPUT_ARTIFACT_PATHS):
        normalized_artifacts[name] = _sha256(
            artifacts[name], field_name=f"expected_input_artifacts[{name!r}]"
        )
    normalized["expected_input_artifacts"] = normalized_artifacts

    for field_name in (
        "runtime_source_fingerprint_sha256",
        "test_source_fingerprint_sha256",
        "training_config_sha256",
        "conda_explicit_sha256",
    ):
        _sha256(normalized[field_name], field_name=field_name)
    if normalized["training_config_sha256"] != training_config_sha256():
        raise ValueError("training_config_sha256 differs from the frozen config")

    commitment = _validate_validation_commitment(
        normalized["validation_manifest_commitment"]
    )
    if (
        commitment["outer_artifact_manifest_sha256"]
        != normalized_artifacts["validation_episode_outer_manifest.json"]
        or commitment["jsonl_sha256"]
        != normalized_artifacts["validation_episode_manifest.jsonl"]
    ):
        raise ValueError(
            "validation_manifest_commitment differs from expected input artifacts"
        )
    normalized["validation_manifest_commitment"] = commitment

    roots = normalized["ordered_import_roots"]
    if type(roots) is not list or len(roots) not in ORDERED_IMPORT_ROOT_COUNTS:
        raise ValueError("ordered_import_roots must contain exactly two or three paths")
    normalized_roots = [
        _lexical_posix_absolute(value, field_name=f"ordered_import_roots[{index}]")
        for index, value in enumerate(roots)
    ]
    if len(set(normalized_roots)) != len(normalized_roots):
        raise ValueError("ordered_import_roots must not contain duplicates")
    if normalized_roots[0] != f"{remote_root}/src":
        raise ValueError("ordered_import_roots must place reviewed source first")
    normalized["ordered_import_roots"] = normalized_roots

    claimed_id = _sha256(normalized["launch_plan_id"], field_name="launch_plan_id")
    semantic = dict(normalized)
    semantic.pop("launch_plan_id")
    if claimed_id != canonical_sha256(semantic):
        raise ValueError("launch_plan_id does not match semantic content")
    return normalized


def build_launch_plan(
    *,
    attempt_nonce: str,
    run_id: str,
    remote_project_root: str,
    run_root: str,
    expected_input_artifacts: Mapping[str, str],
    runtime_source_fingerprint_sha256: str,
    test_source_fingerprint_sha256: str,
    training_config_sha256: str,
    validation_manifest_commitment: Mapping[str, object],
    conda_explicit_sha256: str,
    ordered_import_roots: Sequence[str],
) -> dict[str, Any]:
    """Build one canonical semantic payload and derive its content ID."""

    if not isinstance(expected_input_artifacts, Mapping):
        raise TypeError("expected_input_artifacts must be a mapping")
    if not isinstance(validation_manifest_commitment, Mapping):
        raise TypeError("validation_manifest_commitment must be a mapping")
    if isinstance(ordered_import_roots, (str, bytes)) or not isinstance(
        ordered_import_roots, Sequence
    ):
        raise TypeError("ordered_import_roots must be a sequence")
    semantic: dict[str, Any] = {
        "schema": LAUNCH_PLAN_SCHEMA_VERSION,
        "attempt_nonce": attempt_nonce,
        "run_id": run_id,
        "remote_project_root": remote_project_root,
        "run_root": run_root,
        "expected_input_artifacts": dict(expected_input_artifacts),
        "runtime_source_fingerprint_sha256": runtime_source_fingerprint_sha256,
        "test_source_fingerprint_sha256": test_source_fingerprint_sha256,
        "training_config_sha256": training_config_sha256,
        "validation_manifest_commitment": dict(validation_manifest_commitment),
        "conda_explicit_sha256": conda_explicit_sha256,
        "ordered_import_roots": list(ordered_import_roots),
    }
    payload = {**semantic, "launch_plan_id": canonical_sha256(semantic)}
    return validate_launch_plan_payload(payload)


def canonical_launch_plan_bytes(payload: object) -> bytes:
    """Return canonical file bytes after a complete semantic validation."""

    return serialize_json(validate_launch_plan_payload(payload))


def _path_has_reparse_point(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE
    )


def _assert_nonsymlink_chain(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        if not os.path.lexists(current):
            raise FileNotFoundError(current)
        if _path_has_reparse_point(current):
            raise ValueError("launch-plan path traverses a symlink or reparse point")


def _create_windows_component_handle(
    create_file: Any,
    path: Path,
    *,
    desired_access: int,
    is_leaf: bool,
) -> Any:
    share_mode = _WINDOWS_FILE_SHARE_READ | _WINDOWS_FILE_SHARE_WRITE
    if is_leaf:
        # A verified leaf handle remains bound to the same file after a rename.
        # Directory handles deliberately omit FILE_SHARE_DELETE so ancestors
        # cannot be rename-swapped while the path is still being traversed.
        share_mode |= _WINDOWS_FILE_SHARE_DELETE
    return create_file(
        os.fspath(path),
        desired_access,
        share_mode,
        None,
        _WINDOWS_OPEN_EXISTING,
        _WINDOWS_OPEN_REPARSE_POINT | _WINDOWS_FILE_FLAG_BACKUP_SEMANTICS,
        None,
    )


def _open_windows_regular_nofollow(absolute: Path) -> int:
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
                ctypes.get_last_error(), "cannot resolve launch-plan handle path"
            )
        buffer = ctypes.create_unicode_buffer(size + 1)
        written = get_final_path(handle, buffer, len(buffer), 0)
        if written == 0 or written >= len(buffer):
            raise OSError(
                ctypes.get_last_error(), "cannot read launch-plan handle path"
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
                desired_access=generic_read if is_leaf else file_read_attributes,
                is_leaf=is_leaf,
            )
            if handle == invalid_handle:
                raise OSError(
                    ctypes.get_last_error(),
                    "cannot open launch-plan path component",
                )
            information = _ByHandleFileInformation()
            if not get_file_information(handle, ctypes.byref(information)):
                error = ctypes.get_last_error()
                close_handle(handle)
                raise OSError(error, "cannot inspect launch-plan path component")
            if information.attributes & reparse_attribute:
                close_handle(handle)
                raise ValueError(
                    "launch-plan path traverses a symlink or reparse point"
                )
            if is_leaf == bool(information.attributes & directory_attribute):
                close_handle(handle)
                if is_leaf:
                    raise ValueError("launch-plan artifact must be a regular file")
                raise ValueError("launch-plan ancestor must be a real directory")
            try:
                current_canonical = canonical_handle_path(handle)
            except BaseException:
                close_handle(handle)
                raise
            if previous_canonical is not None and os.path.normcase(
                os.path.normpath(os.path.dirname(current_canonical))
            ) != os.path.normcase(os.path.normpath(previous_canonical)):
                close_handle(handle)
                raise ValueError("launch-plan handle escaped a verified ancestor")
            previous_canonical = current_canonical
            if is_leaf:
                leaf_handle = int(handle)
            else:
                held_directories.append(int(handle))
        if leaf_handle is None:
            raise ValueError("launch-plan artifact must be a regular file")
        descriptor = msvcrt.open_osfhandle(
            leaf_handle, os.O_RDONLY | getattr(os, "O_BINARY", 0)
        )
        leaf_handle = None
        return descriptor
    finally:
        if leaf_handle is not None:
            close_handle(leaf_handle)
        for handle in reversed(held_directories):
            close_handle(handle)


def _open_regular_nofollow(path: Path) -> int:
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOINHERIT", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if os.name == "nt":
        return _open_windows_regular_nofollow(absolute)

    if os.name == "posix":
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        current_fd = os.open(absolute.anchor, directory_flags)
        try:
            for component in absolute.parts[1:-1]:
                next_fd = os.open(component, directory_flags, dir_fd=current_fd)
                os.close(current_fd)
                current_fd = next_fd
            return os.open(absolute.parts[-1], flags, dir_fd=current_fd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ValueError(
                    "launch-plan path contains a symlink or reparse point"
                ) from exc
            raise
        finally:
            os.close(current_fd)

    _assert_nonsymlink_chain(absolute)
    if not stat.S_ISREG(absolute.lstat().st_mode):
        raise ValueError("launch-plan artifact must be a regular file")
    return os.open(absolute, flags)


def _read_regular_snapshot(path: str | Path, *, expected_sha256: str) -> bytes:
    expected = _sha256(expected_sha256, field_name="expected_artifact_sha256")
    descriptor = _open_regular_nofollow(Path(path))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("launch-plan artifact must be a regular file")
        if before.st_size <= 0 or before.st_size > MAX_LAUNCH_PLAN_BYTES:
            raise ValueError("launch-plan artifact exceeds the bounded size")
        chunks: list[bytes] = []
        remaining = MAX_LAUNCH_PLAN_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
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
            raise RuntimeError("launch-plan artifact changed while reading")
    finally:
        os.close(descriptor)
    snapshot = b"".join(chunks)
    if len(snapshot) != before.st_size:
        raise ValueError("launch-plan artifact size changed while reading")
    if hashlib.sha256(snapshot).hexdigest() != expected:
        raise ValueError("launch-plan artifact external SHA-256 mismatch")
    return snapshot


def _decode_strict_json(snapshot: bytes) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate launch-plan JSON key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite launch-plan JSON constant: {value}")

    try:
        return json.loads(
            snapshot.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("launch-plan artifact is not strict UTF-8 JSON") from exc


@dataclass(frozen=True, slots=True)
class LaunchPlanArtifact:
    artifact_sha256: str
    snapshot: bytes = field(repr=False)
    _reader_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._reader_token is not _READER_TOKEN:
            raise TypeError("LaunchPlanArtifact must come from the committed reader")
        digest = _sha256(self.artifact_sha256, field_name="artifact_sha256")
        if (
            type(self.snapshot) is not bytes
            or hashlib.sha256(self.snapshot).hexdigest() != digest
        ):
            raise ValueError("launch-plan immutable snapshot mismatch")
        normalized = validate_launch_plan_payload(_decode_strict_json(self.snapshot))
        if serialize_json(normalized) != self.snapshot:
            raise ValueError("launch-plan artifact is not canonically serialized")

    @property
    def payload(self) -> dict[str, Any]:
        return validate_launch_plan_payload(_decode_strict_json(self.snapshot))


def read_launch_plan_artifact(
    path: str | Path, *, expected_artifact_sha256: str
) -> LaunchPlanArtifact:
    """Read one immutable canonical snapshot bound to a separately supplied SHA."""

    snapshot = _read_regular_snapshot(path, expected_sha256=expected_artifact_sha256)
    normalized = validate_launch_plan_payload(_decode_strict_json(snapshot))
    if serialize_json(normalized) != snapshot:
        raise ValueError("launch-plan artifact is not canonically serialized")
    return LaunchPlanArtifact(
        artifact_sha256=expected_artifact_sha256,
        snapshot=snapshot,
        _reader_token=_READER_TOKEN,
    )


__all__ = [
    "EXPECTED_INPUT_ARTIFACT_PATHS",
    "LAUNCH_PLAN_SCHEMA_VERSION",
    "MAX_LAUNCH_PLAN_BYTES",
    "ORDERED_IMPORT_ROOT_COUNTS",
    "LaunchPlanArtifact",
    "build_launch_plan",
    "canonical_launch_plan_bytes",
    "read_launch_plan_artifact",
    "validate_launch_plan_payload",
]
