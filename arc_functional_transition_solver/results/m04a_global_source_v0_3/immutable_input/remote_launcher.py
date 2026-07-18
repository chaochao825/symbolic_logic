"""Production POSIX launcher for one externally committed M04a attempt.

The launcher validates an immutable launch plan, selects and locks one idle GPU,
emits the v0.2 inherited-flock handshake, and then ``execve`` replaces this same
PID with the isolated Python bootstrap.  It never launches the neural process as
a child and never kills or evicts another GPU process.
"""

from __future__ import annotations

import errno
import hashlib
import io
import json
import os
import re
import selectors
import stat
import subprocess
import sys
import sysconfig
import time
import types
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


_NEURAL_COMMANDS = frozenset({"preflight-m04a"})
_FROZEN_PYTHON_VERSION = (3, 10, 20)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GPU_UUID = re.compile(
    r"GPU-([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\Z"
)
_BOOT_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
_CANONICAL_DECIMAL = re.compile(r"0|[1-9][0-9]*\Z")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

_LAUNCH_PLAN_PATH_ENV = "AFTS_M04A_LAUNCH_PLAN_JSON"
_LAUNCH_PLAN_SHA_ENV = "AFTS_M04A_LAUNCH_PLAN_SHA256"
_HANDSHAKE_SCHEMA = "afts-m04a-held-gpu-lock-handshake/v0.2"
_HANDSHAKE_SOURCE = "reviewed_launcher_inherited_posix_flock"
_HANDSHAKE_NAME = "lock_handshake_artifact.json"
_LAUNCH_PLAN_SCHEMA = "afts-m04a-launch-plan/v0.2"
_VALIDATION_COMMITMENT_SCHEMA = "afts-m04a-validation-manifest-commitment/v0.1"
_FROZEN_TRAINING_CONFIG_SHA256 = (
    "9cb23b21c3fee6f04551ceda1d42abde51c62a961d3ebdcac0242237a465080f"
)
_NVIDIA_SMI = "/usr/bin/nvidia-smi"
_GPU_QUERY = (
    _NVIDIA_SMI,
    "--query-gpu=uuid,utilization.gpu,memory.used",
    "--format=csv,noheader,nounits",
)
_PROCESS_QUERY = (
    _NVIDIA_SMI,
    "--query-compute-apps=gpu_uuid,pid",
    "--format=csv,noheader,nounits",
)

_MAX_LAUNCHER_BYTES = 2 * 1024 * 1024
_MAX_HANDSHAKE_BYTES = 64 * 1024
_MAX_PROBE_OUTPUT_BYTES = 256 * 1024
_MAX_VALIDATION_METADATA_BYTES = 16 * 1024 * 1024
_MAX_VALIDATION_JSONL_BYTES = 512 * 1024 * 1024
_MAX_RUNTIME_SOURCE_ZIP_BYTES = 128 * 1024 * 1024
_MAX_RUNTIME_SOURCE_ENTRY_BYTES = 16 * 1024 * 1024
_MAX_RUNTIME_SOURCE_TOTAL_BYTES = 64 * 1024 * 1024
_MAX_RUNTIME_SOURCE_ENTRY_COUNT = 1_000
_MAX_INPUT_ARTIFACT_BYTES = 512 * 1024 * 1024
_MAX_PROJECTED_ARTIFACT_BYTES = 1 << 50
_MAX_MINIMUM_FREE_INODES = 1 << 50
_MINIMUM_AVAILABLE_BYTES = 30 * (1 << 30)
_ARTIFACT_RESERVE_BYTES = 20 * (1 << 30)
_MAX_PROC_BYTES = 64 * 1024
_GPU_MEMORY_LIMIT_MIB = 256
_IMPORT_FD_BASE = 200
_EXPECTED_INPUT_PATHS = frozenset(
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


@dataclass(frozen=True, slots=True)
class _LaunchArguments:
    child_argv: tuple[str, ...]
    projected_artifact_bytes: int
    minimum_free_inodes: int


@dataclass(frozen=True, slots=True)
class _GpuRow:
    uuid: str
    utilization_percent: int
    memory_used_mib: int
    compute_pids: tuple[int, ...]

    @property
    def idle(self) -> bool:
        return (
            not self.compute_pids
            and self.utilization_percent == 0
            and self.memory_used_mib <= _GPU_MEMORY_LIMIT_MIB
        )


@dataclass(frozen=True, slots=True)
class _CapacityObservation:
    available_bytes: int
    free_inodes: int
    required_bytes: int


@dataclass(frozen=True, slots=True)
class _BoundImportRoots:
    logical_roots: tuple[str, ...]
    bound_roots: tuple[str, ...]
    descriptors: tuple[int, ...]


def _strict_decimal(
    value: object, *, field_name: str, minimum: int, maximum: int
) -> int:
    if not isinstance(value, str) or _CANONICAL_DECIMAL.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a canonical non-negative decimal")
    parsed = int(value)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{field_name} is outside its frozen bounds")
    return parsed


def _parse_launcher_arguments(argv: Sequence[str]) -> _LaunchArguments:
    if len(argv) < 2 or argv[1] not in _NEURAL_COMMANDS:
        allowed = ", ".join(sorted(_NEURAL_COMMANDS))
        raise SystemExit(
            f"a reviewed M04a neural command is required; allowed: {allowed}"
        )
    values: dict[str, int] = {}
    child = [argv[1]]
    flag_specs = {
        "--projected-artifact-bytes": (0, _MAX_PROJECTED_ARTIFACT_BYTES),
        "--minimum-free-inodes": (1, _MAX_MINIMUM_FREE_INODES),
    }
    index = 2
    while index < len(argv):
        argument = argv[index]
        if argument not in flag_specs:
            raise ValueError(f"unreviewed child argument is forbidden: {argument}")
        if argument in values:
            raise ValueError(f"duplicate launcher argument: {argument}")
        if index + 1 >= len(argv):
            raise ValueError(f"missing value for launcher argument: {argument}")
        minimum, maximum = flag_specs[argument]
        values[argument] = _strict_decimal(
            argv[index + 1],
            field_name=argument,
            minimum=minimum,
            maximum=maximum,
        )
        index += 2
    missing = [name for name in flag_specs if name not in values]
    if missing:
        raise ValueError(f"missing reviewed launcher arguments: {', '.join(missing)}")
    return _LaunchArguments(
        child_argv=tuple(child),
        projected_artifact_bytes=values["--projected-artifact-bytes"],
        minimum_free_inodes=values["--minimum-free-inodes"],
    )


def _sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _lexical_posix_absolute(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError(f"{field_name} must be a non-empty POSIX path")
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


def _external_plan_coordinates(
    environ: Mapping[str, str],
) -> tuple[Path, str, str]:
    raw_path = _lexical_posix_absolute(
        environ.get(_LAUNCH_PLAN_PATH_ENV), field_name=_LAUNCH_PLAN_PATH_ENV
    )
    expected_sha256 = _sha256(
        environ.get(_LAUNCH_PLAN_SHA_ENV), field_name=_LAUNCH_PLAN_SHA_ENV
    )
    posix = PurePosixPath(raw_path)
    if posix.name != "launch_plan.json" or posix.parent.name == "input":
        raise ValueError(
            "launch plan must be the exact remote_project_root/launch_plan.json"
        )
    remote_root = posix.parent.as_posix()
    _lexical_posix_absolute(remote_root, field_name="derived_remote_project_root")
    return Path(raw_path), expected_sha256, remote_root


def _open_existing_directory_nofollow(path: str | Path) -> int:
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    current_fd = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=current_fd)
            if not stat.S_ISDIR(os.fstat(next_fd).st_mode):
                os.close(next_fd)
                raise ValueError(f"non-directory path component in {absolute}")
            os.close(current_fd)
            current_fd = next_fd
        result = current_fd
        current_fd = -1
        return result
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(f"directory path traverses a symlink: {absolute}") from exc
        raise
    finally:
        if current_fd >= 0:
            os.close(current_fd)


def _read_bounded_regular_posix(path: str | Path, *, maximum_bytes: int) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    directory_fd = _open_existing_directory_nofollow(absolute.parent)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(absolute.name, flags, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"expected a regular file: {absolute}")
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise ValueError(f"file exceeds its bounded size: {absolute}")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
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
            raise RuntimeError(f"file changed during bounded read: {absolute}")
    finally:
        os.close(descriptor)
    snapshot = b"".join(chunks)
    if len(snapshot) != before.st_size:
        raise ValueError(f"file size changed during bounded read: {absolute}")
    return snapshot


def _read_verified_regular_at(
    directory_fd: int,
    *,
    name: str,
    expected_sha256: str,
    maximum_bytes: int,
) -> bytes:
    if name not in _EXPECTED_INPUT_PATHS:
        raise ValueError("input snapshot name is outside the frozen set")
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise ValueError(f"input artifact is not a bounded regular file: {name}")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
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
            raise RuntimeError(f"input artifact changed while reading: {name}")
    finally:
        os.close(descriptor)
    snapshot = b"".join(chunks)
    if len(snapshot) != before.st_size:
        raise ValueError(f"input artifact byte count changed: {name}")
    if hashlib.sha256(snapshot).hexdigest() != _sha256(
        expected_sha256, field_name=f"expected_input_artifacts[{name!r}]"
    ):
        raise ValueError(f"input artifact differs from launch plan: {name}")
    return snapshot


def _verify_regular_sha_at(
    directory_fd: int,
    *,
    name: str,
    expected_sha256: str,
    maximum_bytes: int,
) -> None:
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError("input artifact name is unsafe")
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise ValueError(f"input artifact is not a bounded regular file: {name}")
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(4 * 1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(f"input artifact became short: {name}")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError(f"input artifact grew while hashing: {name}")
        after = os.fstat(descriptor)
        if (
            not os.path.samestat(before, after)
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise RuntimeError(f"input artifact changed while hashing: {name}")
    finally:
        os.close(descriptor)
    if digest.hexdigest() != _sha256(
        expected_sha256, field_name=f"input artifact SHA-256 {name}"
    ):
        raise ValueError(f"input artifact differs from launch plan: {name}")


def _verify_pre_gpu_input_closed_world(
    input_fd: int, *, plan: Mapping[str, Any]
) -> None:
    expected_names = {*_EXPECTED_INPUT_PATHS, "conda-explicit.txt"}
    actual_names = os.listdir(input_fd)
    if (
        len(actual_names) != len(set(actual_names))
        or set(actual_names) != expected_names
    ):
        raise ValueError("visible input is not the exact 17-plus-conda closed world")
    artifacts = plan["expected_input_artifacts"]
    for name in sorted(_EXPECTED_INPUT_PATHS):
        _verify_regular_sha_at(
            input_fd,
            name=name,
            expected_sha256=artifacts[name],
            maximum_bytes=_MAX_INPUT_ARTIFACT_BYTES,
        )
    _verify_regular_sha_at(
        input_fd,
        name="conda-explicit.txt",
        expected_sha256=plan["conda_explicit_sha256"],
        maximum_bytes=_MAX_VALIDATION_METADATA_BYTES,
    )


def _named_bytes_fingerprint(items: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, content in sorted(items.items()):
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _reviewed_runtime_materials(
    input_fd: int, *, expected_zip_sha256: str, expected_fingerprint_sha256: str
) -> dict[str, bytes]:
    snapshot = _read_verified_regular_at(
        input_fd,
        name="reviewed_runtime_source.zip",
        expected_sha256=expected_zip_sha256,
        maximum_bytes=_MAX_RUNTIME_SOURCE_ZIP_BYTES,
    )
    materials: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(snapshot), mode="r") as archive:
            infos = archive.infolist()
            if not infos or len(infos) > _MAX_RUNTIME_SOURCE_ENTRY_COUNT:
                raise ValueError("runtime source ZIP entry count is invalid")
            total = 0
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
                    or name in materials
                ):
                    raise ValueError(f"unsafe reviewed runtime source entry: {name!r}")
                allowed = (
                    name == "pyproject.toml"
                    or name
                    in {"scripts/afts_arc_evidence.py", "scripts/afts_arc_m04a.py"}
                    or (
                        len(relative.parts) >= 2
                        and relative.parts[0] == "afts_arc"
                        and name.endswith(".py")
                    )
                )
                total += info.file_size
                if (
                    not allowed
                    or info.file_size > _MAX_RUNTIME_SOURCE_ENTRY_BYTES
                    or total > _MAX_RUNTIME_SOURCE_TOTAL_BYTES
                ):
                    raise ValueError(
                        "reviewed runtime source ZIP violates closed bounds"
                    )
                content = archive.read(info)
                if len(content) != info.file_size:
                    raise ValueError("reviewed runtime source ZIP entry became short")
                materials[name] = content
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        raise ValueError("invalid reviewed runtime source ZIP") from exc
    if _named_bytes_fingerprint(materials) != _sha256(
        expected_fingerprint_sha256,
        field_name="runtime_source_fingerprint_sha256",
    ):
        raise ValueError("reviewed runtime ZIP differs from named-byte fingerprint")
    if "afts_arc/m04a_python_runtime_lock.py" not in materials:
        raise ValueError("reviewed runtime ZIP lacks the runtime-lock validator")
    return materials


def _read_source_tree(source_fd: int) -> tuple[dict[str, bytes], frozenset[str]]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    pending: list[tuple[int, str]] = [(os.dup(source_fd), "")]
    materials: dict[str, bytes] = {}
    directories: set[str] = set()
    entry_count = 0
    total = 0
    try:
        while pending:
            directory_fd, prefix = pending.pop()
            try:
                names = sorted(os.listdir(directory_fd))
                for name in names:
                    entry_count += 1
                    if entry_count > _MAX_RUNTIME_SOURCE_ENTRY_COUNT * 2:
                        raise ValueError("reviewed source tree exceeds entry bound")
                    relative = f"{prefix}/{name}" if prefix else name
                    info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode):
                        raise ValueError("reviewed source tree contains a symlink")
                    if stat.S_ISDIR(info.st_mode):
                        directories.add(relative)
                        pending.append(
                            (os.open(name, flags, dir_fd=directory_fd), relative)
                        )
                        continue
                    if not stat.S_ISREG(info.st_mode) or not relative.endswith(".py"):
                        raise ValueError("reviewed source tree contains an extra entry")
                    descriptor = os.open(name, file_flags, dir_fd=directory_fd)
                    try:
                        before = os.fstat(descriptor)
                        if before.st_size > _MAX_RUNTIME_SOURCE_ENTRY_BYTES:
                            raise ValueError("reviewed source file exceeds byte bound")
                        chunks: list[bytes] = []
                        remaining = before.st_size
                        while remaining:
                            chunk = os.read(descriptor, min(1024 * 1024, remaining))
                            if not chunk:
                                raise RuntimeError("reviewed source file became short")
                            chunks.append(chunk)
                            remaining -= len(chunk)
                        if os.read(descriptor, 1):
                            raise RuntimeError("reviewed source file grew")
                        after = os.fstat(descriptor)
                        if (
                            not os.path.samestat(before, after)
                            or before.st_size != after.st_size
                            or before.st_mtime_ns != after.st_mtime_ns
                        ):
                            raise RuntimeError("reviewed source file changed")
                    finally:
                        os.close(descriptor)
                    content = b"".join(chunks)
                    total += len(content)
                    if total > _MAX_RUNTIME_SOURCE_TOTAL_BYTES:
                        raise ValueError(
                            "reviewed source tree exceeds total byte bound"
                        )
                    materials[relative] = content
            finally:
                os.close(directory_fd)
    except BaseException:
        for directory_fd, _ in pending:
            os.close(directory_fd)
        raise
    return materials, frozenset(directories)


def _load_reviewed_runtime_lock_module(
    *,
    input_fd: int,
    source_fd: int,
    expected_artifacts: Mapping[str, str],
    expected_runtime_fingerprint_sha256: str,
) -> types.ModuleType:
    materials = _reviewed_runtime_materials(
        input_fd,
        expected_zip_sha256=expected_artifacts["reviewed_runtime_source.zip"],
        expected_fingerprint_sha256=expected_runtime_fingerprint_sha256,
    )
    expected_source = {
        name: content
        for name, content in materials.items()
        if name.startswith("afts_arc/")
    }
    expected_directories: set[str] = set()
    for name in expected_source:
        parent = PurePosixPath(name).parent
        while parent.as_posix() != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    live_source, live_directories = _read_source_tree(source_fd)
    if live_source != expected_source or live_directories != frozenset(
        expected_directories
    ):
        raise ValueError("live source tree differs from reviewed runtime snapshot")
    module_name = "_afts_reviewed_m04a_python_runtime_lock"
    if module_name in sys.modules:
        raise RuntimeError("reviewed runtime-lock validator was preloaded")
    module = types.ModuleType(module_name)
    module.__file__ = (
        "reviewed_runtime_source.zip!/afts_arc/m04a_python_runtime_lock.py"
    )
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        source = materials["afts_arc/m04a_python_runtime_lock.py"]
        code = compile(source, module.__file__, "exec", dont_inherit=True, optimize=0)
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    required = {
        "read_python_runtime_lock_artifact",
        "validate_live_python_runtime_lock",
    }
    if any(not callable(getattr(module, name, None)) for name in required):
        raise ValueError("reviewed runtime-lock module lacks required validators")
    return module


def _validate_pre_gpu_python_runtime(
    *,
    module: types.ModuleType,
    plan: Mapping[str, Any],
    remote_root: str,
    bound_imports: _BoundImportRoots,
) -> Any:
    expected_sha256 = plan["expected_input_artifacts"]["python-runtime-lock.json"]
    artifact = module.read_python_runtime_lock_artifact(
        f"{remote_root}/input/python-runtime-lock.json",
        expected_artifact_sha256=expected_sha256,
    )
    locked_site = artifact.payload["environment"]["site_packages_path"]
    matching_indices = [
        index
        for index, logical in enumerate(bound_imports.logical_roots)
        if logical == locked_site
    ]
    if len(matching_indices) != 1:
        raise ValueError("locked site-packages is not one exact isolated import root")
    module.validate_live_python_runtime_lock(
        artifact,
        expected_artifact_sha256=expected_sha256,
        site_packages_fd=bound_imports.descriptors[matching_indices[0]],
        ordered_import_roots=bound_imports.logical_roots,
        bound_import_roots=bound_imports.bound_roots,
    )
    return artifact


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _serialize_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


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
        raise ValueError("launch plan is not strict UTF-8 JSON") from exc


def _strict_positive_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise TypeError(f"{field_name} must be a positive integer")
    return value


def _validate_validation_commitment(payload: object) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != _VALIDATION_COMMITMENT_FIELDS:
        raise ValueError("validation commitment fields do not match the exact schema")
    normalized = dict(payload)
    if normalized["schema"] != _VALIDATION_COMMITMENT_SCHEMA:
        raise ValueError("unsupported validation commitment schema")
    for field_name in (
        "outer_artifact_manifest_sha256",
        "jsonl_sha256",
        "summary_id",
        "commitment_id",
    ):
        _sha256(normalized[field_name], field_name=f"commitment.{field_name}")
    _strict_positive_int(normalized["row_count"], field_name="commitment.row_count")
    semantic = dict(normalized)
    claimed = semantic.pop("commitment_id")
    if claimed != _canonical_sha256(semantic):
        raise ValueError("validation commitment ID mismatch")
    return normalized


def _validate_launch_plan(payload: object) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) != _PLAN_FIELDS:
        raise ValueError("launch-plan fields do not match the exact schema")
    normalized: dict[str, Any] = dict(payload)
    if normalized["schema"] != _LAUNCH_PLAN_SCHEMA:
        raise ValueError("unsupported launch-plan schema")
    _sha256(normalized["attempt_nonce"], field_name="attempt_nonce")
    run_id = normalized["run_id"]
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run_id is not one safe path component")
    remote_root = _lexical_posix_absolute(
        normalized["remote_project_root"], field_name="remote_project_root"
    )
    run_root = _lexical_posix_absolute(normalized["run_root"], field_name="run_root")
    if run_root != (PurePosixPath(remote_root) / "runs" / run_id).as_posix():
        raise ValueError("run_root differs from remote_project_root/runs/run_id")
    artifacts = normalized["expected_input_artifacts"]
    if type(artifacts) is not dict or set(artifacts) != _EXPECTED_INPUT_PATHS:
        raise ValueError("launch-plan input artifacts differ from the frozen set")
    normalized_artifacts = {
        name: _sha256(artifacts[name], field_name=f"artifact[{name!r}]")
        for name in sorted(_EXPECTED_INPUT_PATHS)
    }
    normalized["expected_input_artifacts"] = normalized_artifacts
    for field_name in (
        "runtime_source_fingerprint_sha256",
        "test_source_fingerprint_sha256",
        "training_config_sha256",
        "conda_explicit_sha256",
    ):
        _sha256(normalized[field_name], field_name=field_name)
    if normalized["training_config_sha256"] != _FROZEN_TRAINING_CONFIG_SHA256:
        raise ValueError("launch plan differs from the frozen training config")
    commitment = _validate_validation_commitment(
        normalized["validation_manifest_commitment"]
    )
    if (
        commitment["outer_artifact_manifest_sha256"]
        != normalized_artifacts["validation_episode_outer_manifest.json"]
        or commitment["jsonl_sha256"]
        != normalized_artifacts["validation_episode_manifest.jsonl"]
    ):
        raise ValueError("validation commitment differs from launch-plan inputs")
    normalized["validation_manifest_commitment"] = commitment
    roots = normalized["ordered_import_roots"]
    if type(roots) is not list or len(roots) not in {2, 3}:
        raise ValueError("ordered import roots must contain exactly two or three paths")
    normalized_roots = [
        _lexical_posix_absolute(root, field_name=f"ordered_import_roots[{index}]")
        for index, root in enumerate(roots)
    ]
    if len(set(normalized_roots)) != len(normalized_roots):
        raise ValueError("ordered import roots contain duplicates")
    if normalized_roots[0] != f"{remote_root}/src":
        raise ValueError("ordered import roots must place reviewed source first")
    normalized["ordered_import_roots"] = normalized_roots
    semantic = dict(normalized)
    claimed_id = _sha256(semantic.pop("launch_plan_id"), field_name="launch_plan_id")
    if claimed_id != _canonical_sha256(semantic):
        raise ValueError("launch-plan ID mismatch")
    return normalized


def _load_committed_launch_plan(
    plan_path: Path,
    *,
    expected_sha256: str,
    derived_remote_root: str,
) -> dict[str, Any]:
    snapshot = _read_bounded_regular_posix(plan_path, maximum_bytes=256 * 1024)
    if hashlib.sha256(snapshot).hexdigest() != _sha256(
        expected_sha256, field_name="expected launch-plan SHA"
    ):
        raise ValueError("launch plan differs from its external SHA-256")
    normalized = _validate_launch_plan(_decode_strict_json(snapshot))
    if _serialize_json(normalized) != snapshot:
        raise ValueError("launch plan is not canonically serialized")
    if normalized["remote_project_root"] != derived_remote_root:
        raise ValueError("launch plan root differs from its external path")
    return normalized


def _validate_validation_view_snapshots(
    *, outer_bytes: bytes, jsonl_bytes: bytes, summary_bytes: bytes
) -> None:
    outer = _decode_strict_json(outer_bytes)
    summary = _decode_strict_json(summary_bytes)
    if type(outer) is not dict or set(outer) != {
        "schema_version",
        "bundle_status",
        "run_id",
        "artifacts",
    }:
        raise ValueError(
            "validation outer manifest fields do not match the exact schema"
        )
    if (
        type(outer["schema_version"]) is not int
        or outer["schema_version"] != 1
        or outer["bundle_status"] != "complete"
    ):
        raise ValueError("validation outer manifest is unsupported or incomplete")
    outer_run_id = _sha256(outer["run_id"], field_name="validation outer run_id")
    artifacts = outer["artifacts"]
    expected_names = {
        "validation_episode_manifest.jsonl",
        "validation_episode_manifest_summary.json",
    }
    if type(artifacts) is not dict or set(artifacts) != expected_names:
        raise ValueError("validation outer manifest artifact set is not closed-world")
    snapshots = {
        "validation_episode_manifest.jsonl": jsonl_bytes,
        "validation_episode_manifest_summary.json": summary_bytes,
    }
    jsonl_row_count = jsonl_bytes.count(b"\n")
    if (
        not jsonl_bytes
        or jsonl_bytes.startswith(b"\n")
        or not jsonl_bytes.endswith(b"\n")
        or b"\n\n" in jsonl_bytes
    ):
        raise ValueError("validation JSONL does not contain canonical nonblank rows")
    for name, snapshot in snapshots.items():
        metadata = artifacts[name]
        if type(metadata) is not dict or set(metadata) != {"sha256", "bytes", "rows"}:
            raise ValueError(f"validation outer metadata fields mismatch: {name}")
        if _sha256(metadata["sha256"], field_name=f"{name}.sha256") != (
            hashlib.sha256(snapshot).hexdigest()
        ):
            raise ValueError(f"validation outer SHA-256 mismatch: {name}")
        if type(metadata["bytes"]) is not int or metadata["bytes"] != len(snapshot):
            raise ValueError(f"validation outer byte count mismatch: {name}")
        if name.endswith(".jsonl"):
            if type(metadata["rows"]) is not int or metadata["rows"] != jsonl_row_count:
                raise ValueError("validation outer JSONL row count mismatch")
        elif metadata["rows"] is not None:
            raise ValueError("validation outer summary rows must be null")
    if type(summary) is not dict:
        raise TypeError("validation summary must be a JSON object")
    required_summary = {"summary_id", "jsonl_sha256", "jsonl_bytes", "row_count"}
    if not required_summary.issubset(summary):
        raise ValueError("validation summary lacks closure fields")
    if _serialize_json(summary) != summary_bytes:
        raise ValueError("validation summary is not canonically serialized")
    if (
        _sha256(summary["summary_id"], field_name="validation summary_id")
        != outer_run_id
        or _sha256(summary["jsonl_sha256"], field_name="validation jsonl_sha256")
        != hashlib.sha256(jsonl_bytes).hexdigest()
        or type(summary["jsonl_bytes"]) is not int
        or summary["jsonl_bytes"] != len(jsonl_bytes)
        or type(summary["row_count"]) is not int
        or summary["row_count"] != jsonl_row_count
    ):
        raise ValueError("validation summary does not close over outer/JSONL bytes")
    if _serialize_json(outer) != outer_bytes:
        raise ValueError("validation outer manifest is not canonically serialized")


def _write_new_readonly_file(directory_fd: int, *, name: str, content: bytes) -> None:
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
        _write_all(descriptor, content)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_back_exact_regular_at(
    directory_fd: int, *, name: str, expected: bytes
) -> bytes:
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
            raise ValueError(f"materialized validation file metadata mismatch: {name}")
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
    actual = b"".join(chunks)
    if actual != expected:
        raise ValueError(f"materialized validation file bytes mismatch: {name}")
    return actual


def _verify_materialized_validation_view(
    view_fd: int, *, expected: Mapping[str, bytes]
) -> None:
    names = os.listdir(view_fd)
    if len(names) > 3 or set(names) != set(expected):
        raise ValueError("materialized validation view is not closed-world")
    read_back: dict[str, bytes] = {}
    for name, expected_bytes in expected.items():
        metadata = os.stat(name, dir_fd=view_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o400
        ):
            raise ValueError(f"materialized validation file is not readonly: {name}")
        read_back[name] = _read_back_exact_regular_at(
            view_fd, name=name, expected=expected_bytes
        )
    _validate_validation_view_snapshots(
        outer_bytes=read_back["artifact_manifest.json"],
        jsonl_bytes=read_back["validation_episode_manifest.jsonl"],
        summary_bytes=read_back["validation_episode_manifest_summary.json"],
    )


def _materialize_validation_view(
    *,
    input_fd: int,
    run_root_fd: int,
    run_root: str,
    expected_artifacts: Mapping[str, str],
) -> str:
    outer_source_name = "validation_episode_outer_manifest.json"
    jsonl_name = "validation_episode_manifest.jsonl"
    summary_name = "validation_episode_manifest_summary.json"
    outer_bytes = _read_verified_regular_at(
        input_fd,
        name=outer_source_name,
        expected_sha256=expected_artifacts[outer_source_name],
        maximum_bytes=_MAX_VALIDATION_METADATA_BYTES,
    )
    jsonl_bytes = _read_verified_regular_at(
        input_fd,
        name=jsonl_name,
        expected_sha256=expected_artifacts[jsonl_name],
        maximum_bytes=_MAX_VALIDATION_JSONL_BYTES,
    )
    summary_bytes = _read_verified_regular_at(
        input_fd,
        name=summary_name,
        expected_sha256=expected_artifacts[summary_name],
        maximum_bytes=_MAX_VALIDATION_METADATA_BYTES,
    )
    _validate_validation_view_snapshots(
        outer_bytes=outer_bytes,
        jsonl_bytes=jsonl_bytes,
        summary_bytes=summary_bytes,
    )
    view_name = "validation-manifest"
    os.mkdir(view_name, mode=0o700, dir_fd=run_root_fd)
    view_fd = os.open(
        view_name,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=run_root_fd,
    )
    try:
        if not stat.S_ISDIR(os.fstat(view_fd).st_mode):
            raise RuntimeError("validation view is not a real directory")
        _write_new_readonly_file(view_fd, name=jsonl_name, content=jsonl_bytes)
        _write_new_readonly_file(view_fd, name=summary_name, content=summary_bytes)
        _write_new_readonly_file(
            view_fd, name="artifact_manifest.json", content=outer_bytes
        )
        _verify_materialized_validation_view(
            view_fd,
            expected={
                "artifact_manifest.json": outer_bytes,
                jsonl_name: jsonl_bytes,
                summary_name: summary_bytes,
            },
        )
        os.fchmod(view_fd, 0o500)
        os.fsync(view_fd)
    finally:
        os.close(view_fd)
    os.fsync(run_root_fd)
    return f"{run_root}/{view_name}"


def _run_bounded_command(arguments: Sequence[str]) -> tuple[bytes, bytes]:
    environment = {"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"}
    process = subprocess.Popen(
        list(arguments),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        close_fds=True,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        process.wait()
        raise RuntimeError("nvidia-smi probe pipes were not created")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    output = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + 10.0
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("nvidia-smi probe exceeded 10 seconds")
            events = selector.select(remaining)
            if not events:
                raise TimeoutError("nvidia-smi probe exceeded 10 seconds")
            for key, _ in events:
                chunk = os.read(key.fileobj.fileno(), 8192)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                output[key.data].extend(chunk)
                if sum(len(value) for value in output.values()) > (
                    _MAX_PROBE_OUTPUT_BYTES
                ):
                    raise RuntimeError("nvidia-smi output exceeded the bounded limit")
        return_code = process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
    stdout = bytes(output["stdout"])
    stderr = bytes(output["stderr"])
    if return_code != 0:
        detail = stderr.decode("ascii", errors="strict").strip()
        raise RuntimeError(f"nvidia-smi probe failed ({return_code}): {detail}")
    return stdout, stderr


def _ascii_lines(content: bytes, *, label: str) -> list[str]:
    try:
        text = content.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} output is not ASCII") from exc
    return [line for line in text.splitlines() if line.strip()]


def _normalize_gpu_uuid(value: str) -> str:
    match = _GPU_UUID.fullmatch(value)
    if match is None:
        raise ValueError("nvidia-smi returned a non-canonical full GPU UUID")
    return f"GPU-{match.group(1).lower()}"


def _parse_process_rows(content: bytes) -> dict[str, tuple[int, ...]]:
    by_gpu: dict[str, list[int]] = {}
    seen: set[tuple[str, int]] = set()
    for line in _ascii_lines(content, label="compute-process"):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise ValueError("nvidia-smi returned a malformed compute-process row")
        gpu_uuid = _normalize_gpu_uuid(fields[0])
        pid = _strict_decimal(
            fields[1], field_name="compute process PID", minimum=1, maximum=2**31 - 1
        )
        identity = (gpu_uuid, pid)
        if identity in seen:
            raise ValueError("nvidia-smi returned a duplicate compute-process row")
        seen.add(identity)
        by_gpu.setdefault(gpu_uuid, []).append(pid)
    return {gpu: tuple(sorted(pids)) for gpu, pids in by_gpu.items()}


def _parse_gpu_rows(
    content: bytes, *, processes: Mapping[str, tuple[int, ...]]
) -> tuple[_GpuRow, ...]:
    rows: list[_GpuRow] = []
    seen: set[str] = set()
    for line in _ascii_lines(content, label="GPU inventory"):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise ValueError("nvidia-smi returned a malformed GPU inventory row")
        gpu_uuid = _normalize_gpu_uuid(fields[0])
        if gpu_uuid in seen:
            raise ValueError("nvidia-smi returned a duplicate GPU UUID")
        seen.add(gpu_uuid)
        utilization = _strict_decimal(
            fields[1], field_name="GPU utilization", minimum=0, maximum=100
        )
        memory = _strict_decimal(
            fields[2], field_name="GPU memory.used", minimum=0, maximum=2**63 - 1
        )
        rows.append(
            _GpuRow(
                uuid=gpu_uuid,
                utilization_percent=utilization,
                memory_used_mib=memory,
                compute_pids=processes.get(gpu_uuid, ()),
            )
        )
    if not rows:
        raise RuntimeError("nvidia-smi returned no physical GPUs")
    unknown_process_gpus = set(processes) - seen
    if unknown_process_gpus:
        raise ValueError("compute-process rows name a GPU outside the inventory")
    return tuple(rows)


def _query_gpu_snapshot(
    runner: Callable[[Sequence[str]], tuple[bytes, bytes]] = _run_bounded_command,
) -> tuple[_GpuRow, ...]:
    before_bytes, before_stderr = runner(_PROCESS_QUERY)
    gpu_bytes, gpu_stderr = runner(_GPU_QUERY)
    after_bytes, after_stderr = runner(_PROCESS_QUERY)
    if before_stderr.strip() or gpu_stderr.strip() or after_stderr.strip():
        raise RuntimeError("nvidia-smi emitted unexpected stderr on a successful query")
    before = _parse_process_rows(before_bytes)
    after = _parse_process_rows(after_bytes)
    if before != after:
        raise RuntimeError(
            "compute-process inventory changed during GPU eligibility query"
        )
    return _parse_gpu_rows(gpu_bytes, processes=after)


def _select_idle_gpu(rows: Sequence[_GpuRow]) -> str:
    eligible = [row.uuid for row in rows if row.idle]
    if not eligible:
        raise RuntimeError("NO_IDLE_GPU")
    return eligible[0]


def _require_selected_gpu_idle(rows: Sequence[_GpuRow], *, gpu_uuid: str) -> None:
    matching = [row for row in rows if row.uuid == gpu_uuid]
    if len(matching) != 1 or not matching[0].idle:
        raise RuntimeError("SELECTED_GPU_NO_LONGER_IDLE")


def _capacity_observation(
    file_system: object,
    *,
    projected_artifact_bytes: int,
    minimum_free_inodes: int,
) -> _CapacityObservation:
    projected = projected_artifact_bytes
    inodes_required = minimum_free_inodes
    if (
        type(projected) is not int
        or not 0 <= projected <= _MAX_PROJECTED_ARTIFACT_BYTES
    ):
        raise TypeError("projected_artifact_bytes is outside its frozen bounds")
    if type(inodes_required) is not int or not (
        1 <= inodes_required <= _MAX_MINIMUM_FREE_INODES
    ):
        raise TypeError("minimum_free_inodes is outside its frozen bounds")
    fragment_size = getattr(file_system, "f_frsize", None)
    available_fragments = getattr(file_system, "f_bavail", None)
    free_inodes = getattr(file_system, "f_favail", None)
    if (
        type(fragment_size) is not int
        or fragment_size <= 0
        or type(available_fragments) is not int
        or available_fragments < 0
        or type(free_inodes) is not int
        or free_inodes < 0
    ):
        raise RuntimeError("filesystem capacity counters are unavailable")
    available_bytes = fragment_size * available_fragments
    required_bytes = max(_MINIMUM_AVAILABLE_BYTES, projected + _ARTIFACT_RESERVE_BYTES)
    if available_bytes < required_bytes:
        raise RuntimeError("INSUFFICIENT_FREE_BYTES")
    if free_inodes < inodes_required:
        raise RuntimeError("INSUFFICIENT_FREE_INODES")
    return _CapacityObservation(
        available_bytes=available_bytes,
        free_inodes=free_inodes,
        required_bytes=required_bytes,
    )


def _check_capacity_fd(
    descriptor: int,
    *,
    projected_artifact_bytes: int,
    minimum_free_inodes: int,
) -> _CapacityObservation:
    return _capacity_observation(
        os.fstatvfs(descriptor),
        projected_artifact_bytes=projected_artifact_bytes,
        minimum_free_inodes=minimum_free_inodes,
    )


def _open_project_directories(remote_root: str) -> dict[str, int]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    opened = {"root": _open_existing_directory_nofollow(remote_root)}
    try:
        for name, component in (
            ("input", "input"),
            ("source", "src"),
            ("runs", "runs"),
            ("locks", "locks"),
        ):
            descriptor = os.open(component, flags, dir_fd=opened["root"])
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                os.close(descriptor)
                raise ValueError(
                    f"remote project child is not a directory: {component}"
                )
            opened[name] = descriptor
        return opened
    except BaseException:
        for descriptor in opened.values():
            os.close(descriptor)
        raise


def _create_run_root(runs_fd: int, *, run_id: str) -> int:
    os.mkdir(run_id, mode=0o700, dir_fd=runs_fd)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(run_id, flags, dir_fd=runs_fd)
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise RuntimeError("exclusive run root is not a real directory")
    return descriptor


def _acquire_gpu_lock(locks_fd: int, *, gpu_uuid: str) -> tuple[int, int, int]:
    import fcntl

    name = f"gpu-{gpu_uuid}.lock"
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, 0o600, dir_fd=locks_fd)
    try:
        if descriptor < 3:
            replacement = fcntl.fcntl(descriptor, fcntl.F_DUPFD_CLOEXEC, 3)
            os.close(descriptor)
            descriptor = replacement
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or type(opened.st_dev) is not int
            or opened.st_dev < 0
            or type(opened.st_ino) is not int
            or opened.st_ino < 1
        ):
            raise ValueError("GPU lock path is not a regular file")
        path_stat = os.stat(name, dir_fd=locks_fd, follow_symlinks=False)
        if not os.path.samestat(opened, path_stat):
            raise ValueError("GPU lock path changed during open")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("GPU_LOCK_BUSY") from exc
        return descriptor, int(opened.st_dev), int(opened.st_ino)
    except BaseException:
        os.close(descriptor)
        raise


def _read_proc_ascii(path: str, *, maximum_bytes: int) -> str:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(4096, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    content = b"".join(chunks)
    if len(content) > maximum_bytes:
        raise RuntimeError(f"bounded proc read exceeded for {path}")
    try:
        return content.decode("ascii")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"proc record is not ASCII: {path}") from exc


def _linux_boot_id() -> str:
    value = _read_proc_ascii(
        "/proc/sys/kernel/random/boot_id", maximum_bytes=128
    ).strip()
    if _BOOT_ID.fullmatch(value) is None:
        raise RuntimeError("Linux boot_id is malformed")
    return value


def _linux_process_start_ticks(pid: int) -> int:
    raw = _read_proc_ascii(f"/proc/{pid}/stat", maximum_bytes=_MAX_PROC_BYTES)
    close = raw.rfind(")")
    if close < 0:
        raise RuntimeError("Linux process stat is malformed")
    fields = raw[close + 2 :].split()
    if len(fields) <= 19:
        raise RuntimeError("Linux process stat lacks starttime")
    return _strict_decimal(
        fields[19],
        field_name="process start ticks",
        minimum=1,
        maximum=2**63 - 1,
    )


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while publishing lock handshake")
        view = view[written:]


def _write_handshake_artifact(
    run_root_fd: int,
    *,
    run_root: str,
    payload: Mapping[str, Any],
) -> tuple[str, str]:
    content = _serialize_json(dict(payload))
    if not 0 < len(content) <= _MAX_HANDSHAKE_BYTES:
        raise ValueError("lock-handshake artifact exceeds the bounded size")
    descriptor = os.open(
        _HANDSHAKE_NAME,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=run_root_fd,
    )
    try:
        _write_all(descriptor, content)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(run_root_fd)
    return f"{run_root}/{_HANDSHAKE_NAME}", hashlib.sha256(content).hexdigest()


def _duplicate_directory_to_fixed_fd(source_fd: int, *, target_fd: int) -> int:
    import fcntl

    try:
        os.fstat(target_fd)
    except OSError as exc:
        if exc.errno != errno.EBADF:
            raise
    else:
        raise RuntimeError(f"fixed import descriptor is already occupied: {target_fd}")
    duplicated = fcntl.fcntl(source_fd, fcntl.F_DUPFD_CLOEXEC, target_fd)
    if duplicated != target_fd:
        os.close(duplicated)
        raise RuntimeError(f"fixed import descriptor allocation raced: {target_fd}")
    source_info = os.fstat(source_fd)
    target_info = os.fstat(duplicated)
    if not stat.S_ISDIR(target_info.st_mode) or not os.path.samestat(
        source_info, target_info
    ):
        os.close(duplicated)
        raise RuntimeError("fixed import descriptor identity mismatch")
    return duplicated


def _bind_import_roots(remote_root: str, *, source_fd: int) -> _BoundImportRoots:
    raw = (
        f"{remote_root}/src",
        os.path.abspath(sysconfig.get_path("purelib")),
        os.path.abspath(sysconfig.get_path("platlib")),
    )
    logical_roots = tuple(dict.fromkeys(raw))
    if len(logical_roots) not in {2, 3}:
        raise RuntimeError(
            "isolated import roots must contain exactly two or three paths"
        )
    bound: list[int] = []
    try:
        for index, root in enumerate(logical_roots):
            _lexical_posix_absolute(root, field_name=f"computed_import_roots[{index}]")
            close_source = root != f"{remote_root}/src"
            descriptor = (
                _open_existing_directory_nofollow(root) if close_source else source_fd
            )
            try:
                bound.append(
                    _duplicate_directory_to_fixed_fd(
                        descriptor, target_fd=_IMPORT_FD_BASE + index
                    )
                )
            finally:
                if close_source:
                    os.close(descriptor)
    except BaseException:
        for descriptor in bound:
            os.close(descriptor)
        raise
    bound_roots = tuple(f"/proc/self/fd/{descriptor}" for descriptor in bound)
    return _BoundImportRoots(
        logical_roots=logical_roots,
        bound_roots=bound_roots,
        descriptors=tuple(bound),
    )


def _reviewed_launcher_sha256(
    launcher_path: str,
    *,
    remote_root: str,
    expected_sha256: str,
) -> str:
    lexical = _lexical_posix_absolute(launcher_path, field_name="launcher_path")
    allowed = {
        f"{remote_root}/input/remote_launcher.py",
    }
    if lexical not in allowed:
        raise ValueError("launcher path is not an exact reviewed remote location")
    snapshot = _read_bounded_regular_posix(lexical, maximum_bytes=_MAX_LAUNCHER_BYTES)
    digest = hashlib.sha256(snapshot).hexdigest()
    if digest != _sha256(expected_sha256, field_name="expected remote launcher SHA"):
        raise ValueError("executing launcher differs from launch plan")
    return digest


def _seal_exec_descriptors(inherited_fds: frozenset[int]) -> None:
    entries = os.listdir("/proc/self/fd")
    if len(entries) > 4096:
        raise RuntimeError("process has too many descriptors for launch sealing")
    for name in entries:
        if not name.isdecimal():
            raise RuntimeError("/proc/self/fd contained a malformed descriptor")
        descriptor = int(name)
        if descriptor < 3 or descriptor in inherited_fds:
            continue
        try:
            os.set_inheritable(descriptor, False)
        except OSError as exc:
            if exc.errno != errno.EBADF:
                raise
    for descriptor in sorted(inherited_fds):
        if descriptor < 3:
            raise RuntimeError("inherited descriptor must not alias standard I/O")
        os.set_inheritable(descriptor, True)


def _child_environment(
    _inherited: Mapping[str, str],
    *,
    plan: Mapping[str, Any],
    plan_path: str,
    plan_sha256: str,
    launcher_path: str,
    launcher_sha256: str,
    input_root: str,
    validation_manifest_dir: str,
    gpu_uuid: str,
    lock_fd: int,
    handshake_path: str,
    handshake_sha256: str,
    import_roots: Sequence[str],
    bound_import_roots: Sequence[str],
    python_runtime_lock_id: str,
    arguments: _LaunchArguments,
    before_capacity: _CapacityObservation,
    after_capacity: _CapacityObservation,
) -> dict[str, str]:
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
    run_root = str(plan["run_root"])
    artifacts = plan["expected_input_artifacts"]
    environment.update(
        {
            "PYTHONPYCACHEPREFIX": (
                f"{run_root}/afts-m04a-pycache-{str(plan['attempt_nonce'])[:16]}"
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "AFTS_EVIDENCE_FRESH_SOURCE_LOADER": "1",
            "AFTS_EVIDENCE_LAUNCHER": f"{input_root}/remote_launcher.py",
            "AFTS_M04A_VISIBLE_LAUNCHER_PATH": f"{input_root}/remote_launcher.py",
            "AFTS_M04A_TORCH_MODE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": gpu_uuid,
            "AFTS_M04A_RUN_ID": str(plan["run_id"]),
            "AFTS_M04A_RUN_ROOT": run_root,
            "AFTS_M04A_GPU_UUID": gpu_uuid,
            "AFTS_M04A_INHERITED_LOCK_FD": str(lock_fd),
            "AFTS_M04A_LOCK_HANDSHAKE_ARTIFACT_JSON": handshake_path,
            "AFTS_M04A_LOCK_HANDSHAKE_ARTIFACT_SHA256": handshake_sha256,
            "AFTS_M04A_REMOTE_LAUNCHER_SHA256": launcher_sha256,
            "AFTS_M04A_LAUNCH_PLAN_JSON": plan_path,
            "AFTS_M04A_LAUNCH_PLAN_SHA256": plan_sha256,
            "AFTS_M04A_REMOTE_PROJECT_ROOT": str(plan["remote_project_root"]),
            "AFTS_M04A_ATTEMPT_NONCE": str(plan["attempt_nonce"]),
            "AFTS_M04A_INPUT_ROOT": input_root,
            "AFTS_M04A_VALIDATION_MANIFEST_DIR": validation_manifest_dir,
            "AFTS_M04A_VALIDATION_OUTER_SHA256": artifacts[
                "validation_episode_outer_manifest.json"
            ],
            "AFTS_M04A_VALIDATION_JSONL_SHA256": artifacts[
                "validation_episode_manifest.jsonl"
            ],
            "AFTS_M04A_RUNTIME_SOURCE_SHA256": str(
                plan["runtime_source_fingerprint_sha256"]
            ),
            "AFTS_M04A_TEST_SOURCE_SHA256": str(plan["test_source_fingerprint_sha256"]),
            "AFTS_M04A_CONDA_EXPLICIT_SHA256": str(plan["conda_explicit_sha256"]),
            "AFTS_M04A_PYTHON_RUNTIME_LOCK_JSON": (
                f"{input_root}/python-runtime-lock.json"
            ),
            "AFTS_M04A_PYTHON_RUNTIME_LOCK_SHA256": str(
                artifacts["python-runtime-lock.json"]
            ),
            "AFTS_M04A_PYTHON_RUNTIME_LOCK_ID": python_runtime_lock_id,
            "AFTS_M04A_COST_PROBE_SCRATCH_DIR": f"{run_root}/preflight-cost-probe",
            "AFTS_M04A_IMPORT_ROOTS_JSON": json.dumps(
                list(import_roots), separators=(",", ":")
            ),
            "AFTS_M04A_BOUND_IMPORT_ROOTS_JSON": json.dumps(
                list(bound_import_roots), separators=(",", ":")
            ),
            "AFTS_M04A_PROJECTED_ARTIFACT_BYTES": str(
                arguments.projected_artifact_bytes
            ),
            "AFTS_M04A_MINIMUM_FREE_INODES": str(arguments.minimum_free_inodes),
            "AFTS_M04A_REQUIRED_FREE_BYTES": str(after_capacity.required_bytes),
            "AFTS_M04A_AVAILABLE_BYTES_BEFORE_LOCK": str(
                before_capacity.available_bytes
            ),
            "AFTS_M04A_FREE_INODES_BEFORE_LOCK": str(before_capacity.free_inodes),
            "AFTS_M04A_AVAILABLE_BYTES_AFTER_LOCK": str(after_capacity.available_bytes),
            "AFTS_M04A_FREE_INODES_AFTER_LOCK": str(after_capacity.free_inodes),
        }
    )
    if arguments.child_argv[0] == "preflight-m04a":
        environment["AFTS_RUN_M04A_FULL_PREFLIGHT_TEST"] = "1"
    return environment


def _isolated_exec_argv(
    *, import_roots: Sequence[str], pycache_prefix: str, child_argv: Sequence[str]
) -> list[str]:
    bootstrap = (
        "import runpy,sys; "
        "assert sys.implementation.name=='cpython' and sys.version_info[:3]==(3,10,20),'M04a requires CPython 3.10.20'; "
        "count=int(sys.argv.pop(1)); "
        "roots=[sys.argv.pop(1) for _ in range(count)]; "
        "pycache_prefix=sys.argv.pop(1); "
        "sys.path.extend(roots); "
        "sys.pycache_prefix=pycache_prefix; "
        "sys.dont_write_bytecode=True; "
        "runpy.run_module('afts_arc',run_name='__main__',alter_sys=True)"
    )
    return [
        sys.executable,
        "-I",
        "-B",
        "-S",
        "-c",
        bootstrap,
        str(len(import_roots)),
        *import_roots,
        pycache_prefix,
        *child_argv,
    ]


def _preflight_cli_argv(
    *,
    plan_path: str,
    plan_sha256: str,
    validation_manifest_dir: str,
    handshake_path: str,
    handshake_sha256: str,
    gpu_uuid: str,
    remote_root: str,
    run_root: str,
    python_runtime_lock_sha256: str,
) -> tuple[str, ...]:
    return (
        "preflight-m04a",
        "--launch-plan",
        plan_path,
        "--launch-plan-sha256",
        plan_sha256,
        "--python-runtime-lock",
        f"{remote_root}/input/python-runtime-lock.json",
        "--python-runtime-lock-sha256",
        python_runtime_lock_sha256,
        "--validation-manifest-dir",
        validation_manifest_dir,
        "--lock-handshake-artifact",
        handshake_path,
        "--lock-handshake-artifact-sha256",
        handshake_sha256,
        "--gpu-uuid",
        gpu_uuid,
        "--visible-root",
        f"{remote_root}/input",
        "--launcher-path",
        f"{remote_root}/input/remote_launcher.py",
        "--cost-probe-scratch-dir",
        f"{run_root}/preflight-cost-probe",
        "--output-dir",
        f"{run_root}/preflight-gate",
    )


def _run_production_launcher(
    arguments: _LaunchArguments,
    *,
    inherited_environment: Mapping[str, str],
    launcher_path: str,
) -> None:
    plan_path, plan_sha256, derived_root = _external_plan_coordinates(
        inherited_environment
    )
    plan = _load_committed_launch_plan(
        plan_path,
        expected_sha256=plan_sha256,
        derived_remote_root=derived_root,
    )
    launcher_sha256 = _reviewed_launcher_sha256(
        launcher_path,
        remote_root=derived_root,
        expected_sha256=plan["expected_input_artifacts"]["remote_launcher.py"],
    )
    directories = _open_project_directories(derived_root)
    bound_imports: _BoundImportRoots | None = None
    runtime_lock_artifact: Any | None = None
    run_root_fd: int | None = None
    lock_fd: int | None = None
    try:
        bound_imports = _bind_import_roots(
            derived_root, source_fd=directories["source"]
        )
        if list(bound_imports.logical_roots) != plan["ordered_import_roots"]:
            raise ValueError("logical isolated import roots differ from launch plan")
        _verify_pre_gpu_input_closed_world(directories["input"], plan=plan)
        runtime_lock_module = _load_reviewed_runtime_lock_module(
            input_fd=directories["input"],
            source_fd=directories["source"],
            expected_artifacts=plan["expected_input_artifacts"],
            expected_runtime_fingerprint_sha256=plan[
                "runtime_source_fingerprint_sha256"
            ],
        )
        runtime_lock_artifact = _validate_pre_gpu_python_runtime(
            module=runtime_lock_module,
            plan=plan,
            remote_root=derived_root,
            bound_imports=bound_imports,
        )
        _check_capacity_fd(
            directories["runs"],
            projected_artifact_bytes=arguments.projected_artifact_bytes,
            minimum_free_inodes=arguments.minimum_free_inodes,
        )
        gpu_uuid = _select_idle_gpu(_query_gpu_snapshot())
        run_root_fd = _create_run_root(directories["runs"], run_id=str(plan["run_id"]))
        validation_manifest_dir = _materialize_validation_view(
            input_fd=directories["input"],
            run_root_fd=run_root_fd,
            run_root=str(plan["run_root"]),
            expected_artifacts=plan["expected_input_artifacts"],
        )
        before_capacity = _check_capacity_fd(
            run_root_fd,
            projected_artifact_bytes=arguments.projected_artifact_bytes,
            minimum_free_inodes=arguments.minimum_free_inodes,
        )
        acquisition_started = time.perf_counter_ns()
        lock_fd, lock_device, lock_inode = _acquire_gpu_lock(
            directories["locks"], gpu_uuid=gpu_uuid
        )
        after_capacity = _check_capacity_fd(
            run_root_fd,
            projected_artifact_bytes=arguments.projected_artifact_bytes,
            minimum_free_inodes=arguments.minimum_free_inodes,
        )
        pid = os.getpid()
        boot_id = _linux_boot_id()
        process_start_ticks = _linux_process_start_ticks(pid)
        _require_selected_gpu_idle(_query_gpu_snapshot(), gpu_uuid=gpu_uuid)
        eligibility_rechecked = time.perf_counter_ns()
        handshake_completed = time.perf_counter_ns()
        if not acquisition_started < eligibility_rechecked <= handshake_completed:
            raise RuntimeError("lock-handshake monotonic endpoints do not close")
        os.set_inheritable(lock_fd, True)
        semantic: dict[str, Any] = {
            "schema": _HANDSHAKE_SCHEMA,
            "run_id": plan["run_id"],
            "gpu_uuid": gpu_uuid,
            "lock_path": f"{derived_root}/locks/gpu-{gpu_uuid}.lock",
            "lock_st_dev": lock_device,
            "lock_st_ino": lock_inode,
            "lock_holder_pid": pid,
            "lock_holder_start_ticks": process_start_ticks,
            "inherited_lock_fd": lock_fd,
            "attempt_nonce": plan["attempt_nonce"],
            "boot_id": boot_id,
            "acquisition_started_perf_counter_ns": acquisition_started,
            "eligibility_rechecked_perf_counter_ns": eligibility_rechecked,
            "handshake_completed_perf_counter_ns": handshake_completed,
            "wall_ns": handshake_completed - acquisition_started,
            "launcher_sha256": launcher_sha256,
            "launch_plan_sha256": plan_sha256,
            "source": _HANDSHAKE_SOURCE,
        }
        handshake = {**semantic, "handshake_id": _canonical_sha256(semantic)}
        handshake_path, handshake_sha256 = _write_handshake_artifact(
            run_root_fd, run_root=str(plan["run_root"]), payload=handshake
        )
        environment = _child_environment(
            inherited_environment,
            plan=plan,
            plan_path=plan_path.as_posix(),
            plan_sha256=plan_sha256,
            launcher_path=launcher_path,
            launcher_sha256=launcher_sha256,
            input_root=f"{derived_root}/input",
            validation_manifest_dir=validation_manifest_dir,
            gpu_uuid=gpu_uuid,
            lock_fd=lock_fd,
            handshake_path=handshake_path,
            handshake_sha256=handshake_sha256,
            import_roots=bound_imports.logical_roots,
            bound_import_roots=bound_imports.bound_roots,
            python_runtime_lock_id=str(
                runtime_lock_artifact.payload["runtime_lock_id"]
            ),
            arguments=arguments,
            before_capacity=before_capacity,
            after_capacity=after_capacity,
        )
        child_argv = _preflight_cli_argv(
            plan_path=plan_path.as_posix(),
            plan_sha256=plan_sha256,
            validation_manifest_dir=validation_manifest_dir,
            handshake_path=handshake_path,
            handshake_sha256=handshake_sha256,
            gpu_uuid=gpu_uuid,
            remote_root=derived_root,
            run_root=str(plan["run_root"]),
            python_runtime_lock_sha256=plan["expected_input_artifacts"][
                "python-runtime-lock.json"
            ],
        )
        _seal_exec_descriptors(frozenset({lock_fd, *bound_imports.descriptors}))
        exec_argv = _isolated_exec_argv(
            import_roots=bound_imports.bound_roots,
            pycache_prefix=environment["PYTHONPYCACHEPREFIX"],
            child_argv=child_argv,
        )
        os.execve(sys.executable, exec_argv, environment)
        raise RuntimeError("isolated M04a execve unexpectedly returned")
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if run_root_fd is not None:
            os.close(run_root_fd)
        if bound_imports is not None:
            for descriptor in bound_imports.descriptors:
                os.close(descriptor)
        for descriptor in directories.values():
            os.close(descriptor)


def _initial_isolation_satisfied() -> bool:
    return bool(
        sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode
    )


def _reexec_initial_isolation(launcher_path: str, argv: Sequence[str]) -> None:
    _read_bounded_regular_posix(launcher_path, maximum_bytes=_MAX_LAUNCHER_BYTES)
    _seal_exec_descriptors(frozenset())
    environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        _LAUNCH_PLAN_PATH_ENV: os.environ[_LAUNCH_PLAN_PATH_ENV],
        _LAUNCH_PLAN_SHA_ENV: os.environ[_LAUNCH_PLAN_SHA_ENV],
    }
    isolated_argv = [
        sys.executable,
        "-I",
        "-B",
        "-S",
        launcher_path,
        *argv[1:],
    ]
    os.execve(sys.executable, isolated_argv, environment)
    raise RuntimeError("initial isolation execve unexpectedly returned")


def main() -> int:
    arguments = _parse_launcher_arguments(sys.argv)
    if (
        sys.implementation.name != "cpython"
        or sys.version_info[:3] != _FROZEN_PYTHON_VERSION
    ):
        raise RuntimeError(
            "M04a requires exact CPython 3.10.20; "
            f"found {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
    if os.name != "posix":
        raise RuntimeError("production M04a GPU locking requires POSIX")
    launcher_path = Path(os.path.abspath(__file__)).as_posix()
    if not _initial_isolation_satisfied():
        _reexec_initial_isolation(launcher_path, sys.argv)
    _run_production_launcher(
        arguments,
        inherited_environment=dict(os.environ),
        launcher_path=launcher_path,
    )
    raise RuntimeError("production M04a launcher unexpectedly returned")


if __name__ == "__main__":
    raise SystemExit(main())
