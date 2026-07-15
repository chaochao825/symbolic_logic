"""Canonical, Torch-free provenance lock for the M04a Python runtime.

The lock is intentionally built without importing :mod:`torch`.  It commits the
exact CPython executable, every installed distribution, and a file-by-file
verification of the dependency closure that can affect the neural runtime.
Production launchers must validate the live environment before querying CUDA or
creating a run directory.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import email.parser
import hashlib
import io
import json
import os
import platform
import re
import stat
import sys
import sysconfig
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


PYTHON_RUNTIME_LOCK_SCHEMA_VERSION = "afts-m04a-python-runtime-lock/v0.1"
PYTHON_RUNTIME_LOCK_ALGORITHM_SCHEMA_VERSION = (
    "afts-m04a-python-runtime-lock-algorithm/v0.1"
)
PYTHON_RUNTIME_LOCK_FILENAME = "python-runtime-lock.json"
MAX_RUNTIME_LOCK_BYTES = 32 * 1024 * 1024
MAX_DISTRIBUTION_COUNT = 8_192
MAX_METADATA_BYTES = 16 * 1024 * 1024
MAX_RECORD_BYTES = 32 * 1024 * 1024
MAX_RECORD_ENTRY_COUNT = 250_000
MAX_CRITICAL_DISTRIBUTION_COUNT = 4_096
MAX_CRITICAL_FILE_BYTES = 8 * 1024 * 1024 * 1024
MAX_CRITICAL_TOTAL_BYTES = 64 * 1024 * 1024 * 1024
MAX_REQUIREMENT_COUNT_PER_DISTRIBUTION = 16_384
FROZEN_MIXBIT_CRITICAL_RECORD_ENTRY_COUNT = 19_800
FROZEN_MIXBIT_CRITICAL_UNHASHED_ENTRY_COUNT = 4_688
FROZEN_MIXBIT_CRITICAL_PYC_ENTRY_COUNT = 4_661
FROZEN_MIXBIT_CRITICAL_VERIFIED_FILE_BYTES = 7_231_033_865

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DISTRIBUTION_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?\Z")
_REQUIREMENT_NAME = re.compile(r"\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")
_VERSION_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+!-]{0,511}\Z")
_REPARSE_POINT_ATTRIBUTE = 0x400
_READER_TOKEN = object()

_LOCK_FIELDS = {
    "schema",
    "runtime_lock_id",
    "algorithm",
    "environment",
    "python",
    "distributions",
    "critical_distribution_names",
    "critical_distributions",
    "torch_module",
}
_ALGORITHM_FIELDS = {
    "schema",
    "distribution_name_normalization",
    "critical_closure",
    "record_path_normalization",
    "record_verification",
    "record_unhashed_allowlist",
    "torch_origin_resolution",
}
_ENVIRONMENT_FIELDS = {
    "prefix_path",
    "site_packages_path",
    "site_packages_environment_relative",
    "marker_environment",
}
_MARKER_ENVIRONMENT_FIELDS = {
    "python_version",
    "python_full_version",
    "os_name",
    "sys_platform",
    "platform_system",
    "platform_machine",
    "platform_python_implementation",
    "implementation_name",
    "implementation_version",
    "platform_release",
    "platform_version",
    "extra",
}
_PYTHON_FIELDS = {
    "implementation",
    "version",
    "version_info",
    "executable_lexical_path",
    "executable_symlink_chain",
    "executable_resolved_path",
    "executable_device",
    "executable_inode",
    "executable_sha256",
    "executable_bytes",
}
_DISTRIBUTION_FIELDS = {
    "name",
    "version",
    "metadata_directory",
    "metadata_sha256",
    "metadata_bytes",
}
_CRITICAL_DISTRIBUTION_FIELDS = {
    "name",
    "version",
    "metadata_directory",
    "record_path",
    "record_sha256",
    "record_bytes",
    "record_entry_count",
    "hashed_file_count",
    "unhashed_entry_paths",
    "verified_files",
    "verified_file_bytes",
    "verified_files_sha256",
    "requires_dist_names",
    "active_requires_dist_names",
    "requires_dist_entries",
}
_VERIFIED_FILE_FIELDS = {
    "record_reference",
    "path",
    "declared_sha256",
    "declared_bytes",
    "actual_sha256",
    "actual_bytes",
}
_TORCH_MODULE_FIELDS = {
    "distribution_name",
    "module_version",
    "cuda_version",
    "module_init_path",
    "module_init_sha256",
    "module_init_bytes",
    "version_file_path",
    "version_file_sha256",
    "version_file_bytes",
    "git_version",
}
_ALGORITHM = {
    "schema": PYTHON_RUNTIME_LOCK_ALGORITHM_SCHEMA_VERSION,
    "distribution_name_normalization": (
        "pep503_lowercase_and_collapse_runs_of_hyphen_underscore_dot_to_hyphen"
    ),
    "critical_closure": (
        "roots_torch_triton_all_installed_nvidia_hyphen_star_cuda_hyphen_star_"
        "then_recursive_installed_requires_dist_lexical_names/v0.1"
    ),
    "record_path_normalization": (
        "resolve_record_reference_from_site_packages_to_environment_relative_"
        "reject_escape_store_no_dot_or_dotdot/v0.1"
    ),
    "record_verification": (
        "sha256_record_then_reopen_every_hashed_entry_no_follow_and_compare_"
        "declared_sha256_size_to_actual_sha256_size_canonical_rows/v0.1"
    ),
    "record_unhashed_allowlist": (
        "exact_record_self_plus_exact_record_listed_dot_pyc_paths_with_independent_"
        "actual_sha256_and_size/v0.1"
    ),
    "torch_origin_resolution": (
        "exact_locked_torch_slash_init_dot_py_and_literal_git_version_without_import"
        "/v0.1"
    ),
}

# Frozen by the read-only 237/mixbit audit.  Artifact hashes are deliberately
# not hard-coded: the builder derives every hash from the live target files.
FROZEN_MIXBIT_CRITICAL_DISTRIBUTIONS = {
    "cuda-bindings": "12.9.4",
    "cuda-pathfinder": "1.4.2",
    "filelock": "3.25.2",
    "fsspec": "2023.10.0",
    "jinja2": "3.1.6",
    "markupsafe": "3.0.3",
    "mpmath": "1.3.0",
    "networkx": "3.4.2",
    "nvidia-cublas-cu12": "12.8.4.1",
    "nvidia-cuda-cupti-cu12": "12.8.90",
    "nvidia-cuda-nvrtc-cu12": "12.8.93",
    "nvidia-cuda-runtime-cu12": "12.8.90",
    "nvidia-cudnn-cu12": "9.10.2.21",
    "nvidia-cufft-cu12": "11.3.3.83",
    "nvidia-cufile-cu12": "1.13.1.3",
    "nvidia-curand-cu12": "10.3.9.90",
    "nvidia-cusolver-cu12": "11.7.3.90",
    "nvidia-cusparse-cu12": "12.5.8.93",
    "nvidia-cusparselt-cu12": "0.7.1",
    "nvidia-nccl-cu12": "2.27.5",
    "nvidia-nvjitlink-cu12": "12.8.93",
    "nvidia-nvshmem-cu12": "3.4.5",
    "nvidia-nvtx-cu12": "12.8.90",
    "sympy": "1.14.0",
    "torch": "2.10.0",
    "triton": "3.6.0",
    "typing-extensions": "4.15.0",
}

FROZEN_MIXBIT_MARKER_ENVIRONMENT = {
    "python_version": "3.10",
    "python_full_version": "3.10.20",
    "os_name": "posix",
    "sys_platform": "linux",
    "platform_system": "Linux",
    "platform_machine": "x86_64",
    "platform_python_implementation": "CPython",
    "implementation_name": "cpython",
    "implementation_version": "3.10.20",
    "platform_release": "5.4.0-42-generic",
    "platform_version": "#46-Ubuntu SMP Fri Jul 10 00:24:02 UTC 2020",
    "extra": "",
}

_ALLOWED_DOTDOT_RECORD_REFERENCES = {
    "sympy": frozenset({"../../../bin/isympy", "../../../share/man/man1/isympy.1"}),
    "torch": frozenset({"../../../bin/torchfrtrace", "../../../bin/torchrun"}),
    "triton": frozenset({"../../../bin/proton", "../../../bin/proton-viewer"}),
}


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


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


def _sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TypeError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _strict_int(
    value: object,
    *,
    field_name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(f"{field_name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field_name} exceeds {maximum}")
    return value


def _exact_dict(value: object, fields: set[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields do not match the exact schema")
    return dict(value)


def canonical_distribution_name(value: object) -> str:
    """Return the PEP 503 canonical name used by the closure algorithm."""

    if not isinstance(value, str) or _DISTRIBUTION_NAME.fullmatch(value) is None:
        raise ValueError("distribution name is malformed")
    return re.sub(r"[-_.]+", "-", value).lower()


def _version(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _VERSION_COMPONENT.fullmatch(value) is None:
        raise ValueError(f"{field_name} is malformed")
    return value


def _lexical_absolute_path(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise TypeError(f"{field_name} must be a non-empty absolute path")
    path = Path(value)
    if (
        not path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise ValueError(f"{field_name} must be a lexical absolute path")
    return value


def _payload_absolute_path(
    value: object, *, field_name: str
) -> tuple[str, PurePosixPath | PureWindowsPath]:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise TypeError(f"{field_name} must be a non-empty absolute path")
    if value.startswith("/"):
        path: PurePosixPath | PureWindowsPath = PurePosixPath(value)
        if (
            not path.is_absolute()
            or path.as_posix() != value
            or value == "/"
            or value.endswith("/")
            or any(part in {"", ".", ".."} for part in path.parts[1:])
        ):
            raise ValueError(f"{field_name} must be a lexical POSIX path")
        return "posix", path
    path = PureWindowsPath(value)
    if (
        not path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise ValueError(f"{field_name} must be a lexical absolute path")
    return "windows", path


def _payload_require_within(
    child: PurePosixPath | PureWindowsPath,
    parent: PurePosixPath | PureWindowsPath,
    *,
    field_name: str,
) -> None:
    if type(child) is not type(parent):
        raise ValueError(f"{field_name} path flavor differs from environment prefix")
    try:
        child.relative_to(parent)
    except ValueError:
        raise ValueError(f"{field_name} must be inside environment prefix") from None


def _safe_relative_path(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise TypeError(f"{field_name} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or ":" in value
    ):
        raise ValueError(f"{field_name} must be a safe POSIX relative path")
    return value


def _path_is_link(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE
    )


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_mode == right.st_mode
    )


def _absolute_no_expand(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_nonsymlink_chain(path: Path, *, label: str) -> None:
    absolute = _absolute_no_expand(path)
    if not absolute.is_absolute():
        raise ValueError(f"{label} must be absolute")
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        try:
            info = current.lstat()
        except FileNotFoundError:
            raise FileNotFoundError(current) from None
        if _path_is_link(info):
            raise ValueError(f"{label} traverses a symlink or reparse point")


def _stable_fields(info: os.stat_result) -> tuple[int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


class _AnchoredRoot:
    """A directory identity used for no-follow, bounded, descriptor reads."""

    def __init__(self, path: str | Path, *, descriptor: int | None, label: str):
        self.path = _absolute_no_expand(path)
        _assert_nonsymlink_chain(self.path, label=label)
        if not self.path.is_dir():
            raise NotADirectoryError(self.path)
        self.label = label
        self._descriptor: int | None = None
        self._use_dir_fd = os.name == "posix"
        if descriptor is not None:
            if type(descriptor) is not int or descriptor < 0:
                raise TypeError(f"{label} descriptor must be a nonnegative integer")
            duplicate = os.dup(descriptor)
            try:
                supplied = os.fstat(duplicate)
                if not stat.S_ISDIR(supplied.st_mode):
                    raise ValueError(f"{label} descriptor is not a directory")
                actual = os.stat(self.path, follow_symlinks=False)
                if not _same_identity(supplied, actual):
                    raise ValueError(
                        f"{label} descriptor differs from its logical path"
                    )
            except BaseException:
                os.close(duplicate)
                raise
            self._descriptor = duplicate
        elif self._use_dir_fd:
            flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            self._descriptor = os.open(self.path, flags)
        self._initial = self.stat_root()

    def __enter__(self) -> _AnchoredRoot:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None

    def stat_root(self) -> os.stat_result:
        if self._descriptor is not None:
            return os.fstat(self._descriptor)
        return os.stat(self.path, follow_symlinks=False)

    def assert_stable(self) -> None:
        current = self.stat_root()
        if not _same_identity(self._initial, current):
            raise RuntimeError(f"{self.label} directory identity changed")
        path_info = os.stat(self.path, follow_symlinks=False)
        if not _same_identity(current, path_info):
            raise RuntimeError(f"{self.label} path was replaced")

    def _open_directory_fd(self, relative: str) -> tuple[int, list[int]]:
        if self._descriptor is None:
            raise OSError("directory descriptors are unavailable")
        if relative == "":
            duplicate = os.dup(self._descriptor)
            return duplicate, [duplicate]
        _safe_relative_path(relative, field_name="directory path")
        opened: list[int] = []
        current = os.dup(self._descriptor)
        opened.append(current)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            for component in PurePosixPath(relative).parts:
                following = os.open(component, flags, dir_fd=current)
                info = os.fstat(following)
                if not stat.S_ISDIR(info.st_mode):
                    os.close(following)
                    raise NotADirectoryError(relative)
                opened.append(following)
                current = following
        except BaseException:
            for descriptor in reversed(opened):
                os.close(descriptor)
            raise
        return current, opened

    def list_names(self, relative: str = "") -> list[str]:
        if relative:
            _safe_relative_path(relative, field_name="directory path")
        if self._descriptor is not None and self._use_dir_fd:
            final, opened = self._open_directory_fd(relative)
            try:
                names = os.listdir(final)
            finally:
                for descriptor in reversed(opened):
                    os.close(descriptor)
        else:
            directory = self.path.joinpath(*PurePosixPath(relative).parts)
            _assert_nonsymlink_chain(directory, label=self.label)
            names = os.listdir(directory)
        if any(not isinstance(name, str) or name in {"", ".", ".."} for name in names):
            raise ValueError(f"{self.label} contains a malformed directory entry")
        return sorted(names)

    def lstat(self, relative: str) -> os.stat_result:
        _safe_relative_path(relative, field_name="entry path")
        parent = PurePosixPath(relative).parent.as_posix()
        if parent == ".":
            parent = ""
        name = PurePosixPath(relative).name
        if self._descriptor is not None and self._use_dir_fd:
            _, opened = self._open_directory_fd(parent)
            directory = opened[-1]
            try:
                return os.stat(name, dir_fd=directory, follow_symlinks=False)
            finally:
                for descriptor in reversed(opened):
                    os.close(descriptor)
        path = self.path.joinpath(*PurePosixPath(relative).parts)
        _assert_nonsymlink_chain(path.parent, label=self.label)
        return path.lstat()

    def read_regular(
        self,
        relative: str,
        *,
        maximum_bytes: int,
        label: str,
    ) -> tuple[bytes, os.stat_result]:
        _safe_relative_path(relative, field_name=f"{label} path")
        parent = PurePosixPath(relative).parent.as_posix()
        if parent == ".":
            parent = ""
        name = PurePosixPath(relative).name
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOINHERIT", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        opened: list[int] = []
        if self._descriptor is not None and self._use_dir_fd:
            _, opened = self._open_directory_fd(parent)
            descriptor = os.open(name, flags, dir_fd=opened[-1])
        else:
            path = self.path.joinpath(*PurePosixPath(relative).parts)
            _assert_nonsymlink_chain(path.parent, label=self.label)
            before_path = path.lstat()
            if _path_is_link(before_path):
                raise ValueError(f"{label} is a symlink or reparse point")
            descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"{label} must be a regular file")
            if before.st_size < 0 or before.st_size > maximum_bytes:
                raise ValueError(f"{label} exceeds its byte bound")
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
            if _stable_fields(before) != _stable_fields(after):
                raise RuntimeError(f"{label} changed while reading")
        finally:
            os.close(descriptor)
            for parent_descriptor in reversed(opened):
                os.close(parent_descriptor)
        return b"".join(chunks), before

    def hash_regular(
        self,
        relative: str,
        *,
        maximum_bytes: int,
        label: str,
    ) -> tuple[str, int, os.stat_result]:
        """Hash a regular file without retaining its potentially huge payload."""

        _safe_relative_path(relative, field_name=f"{label} path")
        parent = PurePosixPath(relative).parent.as_posix()
        if parent == ".":
            parent = ""
        name = PurePosixPath(relative).name
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOINHERIT", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        opened: list[int] = []
        if self._descriptor is not None and self._use_dir_fd:
            _, opened = self._open_directory_fd(parent)
            descriptor = os.open(name, flags, dir_fd=opened[-1])
        else:
            path = self.path.joinpath(*PurePosixPath(relative).parts)
            _assert_nonsymlink_chain(path.parent, label=self.label)
            before_path = path.lstat()
            if _path_is_link(before_path):
                raise ValueError(f"{label} is a symlink or reparse point")
            descriptor = os.open(path, flags)
        digest = hashlib.sha256()
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"{label} must be a regular file")
            if before.st_size < 0 or before.st_size > maximum_bytes:
                raise ValueError(f"{label} exceeds its byte bound")
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(4 * 1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeError(f"{label} became short while hashing")
                digest.update(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise RuntimeError(f"{label} grew while hashing")
            after = os.fstat(descriptor)
            if _stable_fields(before) != _stable_fields(after):
                raise RuntimeError(f"{label} changed while hashing")
        finally:
            os.close(descriptor)
            for parent_descriptor in reversed(opened):
                os.close(parent_descriptor)
        return digest.hexdigest(), before.st_size, before

    def regular_tree_paths(
        self, relative: str, *, maximum_entries: int, label: str
    ) -> list[str]:
        _safe_relative_path(relative, field_name=f"{label} root")
        pending = [relative]
        files: list[str] = []
        seen_directories = 0
        while pending:
            directory = pending.pop()
            seen_directories += 1
            if seen_directories + len(files) > maximum_entries:
                raise ValueError(f"{label} tree exceeds entry bound")
            for name in self.list_names(directory):
                child = f"{directory}/{name}"
                info = self.lstat(child)
                if _path_is_link(info):
                    raise ValueError(
                        f"{label} tree contains a symlink or reparse point"
                    )
                if stat.S_ISDIR(info.st_mode):
                    pending.append(child)
                elif stat.S_ISREG(info.st_mode):
                    files.append(child)
                else:
                    raise ValueError(f"{label} tree contains a non-regular entry")
                if seen_directories + len(pending) + len(files) > maximum_entries:
                    raise ValueError(f"{label} tree exceeds entry bound")
        return sorted(files)


def _environment_relative(
    path: Path, *, environment_prefix: Path, field_name: str
) -> str:
    try:
        relative = path.relative_to(environment_prefix)
    except ValueError:
        raise ValueError(
            f"{field_name} must be inside the environment prefix"
        ) from None
    return _safe_relative_path(relative.as_posix(), field_name=field_name)


def _normalize_record_reference(
    raw_path: str,
    *,
    distribution_name: str,
    site_packages_environment_relative: str,
) -> str:
    if (
        not raw_path
        or "\x00" in raw_path
        or "\\" in raw_path
        or PurePosixPath(raw_path).is_absolute()
        or ":" in raw_path
    ):
        raise ValueError("RECORD path is absolute or malformed")
    contains_dotdot = ".." in raw_path.split("/")
    allowed_dotdot = _ALLOWED_DOTDOT_RECORD_REFERENCES.get(
        distribution_name, frozenset()
    )
    if contains_dotdot and raw_path not in allowed_dotdot:
        raise ValueError(
            f"{distribution_name} RECORD path uses non-allowlisted dotdot traversal"
        )
    components = list(PurePosixPath(site_packages_environment_relative).parts)
    for component in raw_path.split("/"):
        if component in {"", "."}:
            raise ValueError("RECORD path contains an empty or dot component")
        if component == "..":
            if not components:
                raise ValueError("RECORD path escapes the environment prefix")
            components.pop()
            continue
        components.append(component)
    if not components:
        raise ValueError("RECORD path resolves to the environment root")
    return _safe_relative_path(
        "/".join(components), field_name="normalized RECORD path"
    )


def _decode_metadata(content: bytes, *, label: str) -> Any:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    return email.parser.Parser().parsestr(text, headersonly=True)


def _marker_environment() -> dict[str, str]:
    implementation = sys.implementation.version
    implementation_version = (
        f"{implementation.major}.{implementation.minor}.{implementation.micro}"
    )
    if implementation.releaselevel != "final":
        level = {"alpha": "a", "beta": "b", "candidate": "rc"}.get(
            implementation.releaselevel
        )
        if level is None:
            raise ValueError("unsupported CPython implementation release level")
        implementation_version += f"{level}{implementation.serial}"
    return {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "python_full_version": platform.python_version(),
        "os_name": os.name,
        "sys_platform": sys.platform,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "platform_python_implementation": platform.python_implementation(),
        "implementation_name": sys.implementation.name,
        "implementation_version": implementation_version,
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "extra": "",
    }


@dataclass(frozen=True, slots=True)
class _MarkerToken:
    kind: str
    value: str


def _tokenize_marker(marker: str) -> list[_MarkerToken]:
    tokens: list[_MarkerToken] = []
    index = 0
    while index < len(marker):
        if marker[index].isspace():
            index += 1
            continue
        if marker[index] in "()":
            tokens.append(_MarkerToken(marker[index], marker[index]))
            index += 1
            continue
        if marker[index] in "'\"":
            quote = marker[index]
            start = index
            index += 1
            escaped = False
            while index < len(marker):
                char = marker[index]
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    index += 1
                    break
                index += 1
            else:
                raise ValueError("unterminated requirement marker string")
            literal = marker[start:index]
            try:
                value = ast.literal_eval(literal)
            except (SyntaxError, ValueError) as exc:
                raise ValueError("invalid requirement marker string") from exc
            if not isinstance(value, str) or len(value) > 1_024:
                raise ValueError("requirement marker literal is invalid")
            tokens.append(_MarkerToken("string", value))
            continue
        operator = next(
            (
                candidate
                for candidate in ("===", "~=", "==", "!=", "<=", ">=", "<", ">")
                if marker.startswith(candidate, index)
            ),
            None,
        )
        if operator is not None:
            tokens.append(_MarkerToken("operator", operator))
            index += len(operator)
            continue
        match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", marker[index:])
        if match is None:
            raise ValueError("unsupported requirement marker syntax")
        word = match.group(0)
        tokens.append(_MarkerToken("word", word))
        index += len(word)
    if len(tokens) > 256:
        raise ValueError("requirement marker exceeds token bound")
    return tokens


def _simple_version(value: str) -> tuple[tuple[int, ...], int, int]:
    match = re.fullmatch(
        r"[vV]?(\d+(?:\.\d+)*)(?:(a|b|rc)(\d+)?)?(?:(post|dev)(\d+)?)?",
        value,
    )
    if match is None:
        raise ValueError(f"unsupported version marker operand: {value!r}")
    release = tuple(int(part) for part in match.group(1).split("."))
    prerelease = match.group(2)
    suffix = match.group(4)
    if prerelease == "a":
        rank = -2
    elif prerelease == "b":
        rank = -1
    elif prerelease == "rc":
        rank = 0
    elif suffix == "dev":
        rank = -3
    elif suffix == "post":
        rank = 2
    else:
        rank = 1
    number = int(match.group(3) or match.group(5) or 0)
    return release, rank, number


def _compare_marker_values(
    left: str, operator: str, right: str, *, version: bool
) -> bool:
    if operator in {"in", "not in"}:
        result = left in right
        return not result if operator == "not in" else result
    if operator == "===":
        return left == right
    if version:
        left_parsed = _simple_version(left)
        right_parsed = _simple_version(right)
        width = max(len(left_parsed[0]), len(right_parsed[0]))
        left_value = (
            left_parsed[0] + (0,) * (width - len(left_parsed[0])),
            left_parsed[1],
            left_parsed[2],
        )
        right_value = (
            right_parsed[0] + (0,) * (width - len(right_parsed[0])),
            right_parsed[1],
            right_parsed[2],
        )
    else:
        left_value = left
        right_value = right
    if operator == "==":
        return left_value == right_value
    if operator == "!=":
        return left_value != right_value
    if operator == "<":
        return left_value < right_value
    if operator == "<=":
        return left_value <= right_value
    if operator == ">":
        return left_value > right_value
    if operator == ">=":
        return left_value >= right_value
    if operator == "~=":
        right_release = _simple_version(right)[0]
        left_release = _simple_version(left)[0]
        return (
            left_value >= right_value
            and left_release[: max(1, len(right_release) - 1)]
            == right_release[: max(1, len(right_release) - 1)]
        )
    raise ValueError(f"unsupported requirement marker operator: {operator}")


class _MarkerParser:
    def __init__(self, tokens: Sequence[_MarkerToken], environment: Mapping[str, str]):
        self.tokens = tokens
        self.environment = environment
        self.index = 0

    def _peek(self, value: str | None = None) -> bool:
        if self.index >= len(self.tokens):
            return False
        return value is None or self.tokens[self.index].value == value

    def _take(self, value: str | None = None) -> _MarkerToken:
        if not self._peek(value):
            raise ValueError("unsupported requirement marker grammar")
        token = self.tokens[self.index]
        self.index += 1
        return token

    def parse(self) -> bool:
        result = self._parse_or()
        if self.index != len(self.tokens):
            raise ValueError("trailing requirement marker tokens")
        return result

    def _parse_or(self) -> bool:
        result = self._parse_and()
        while self._peek("or"):
            self._take("or")
            following = self._parse_and()
            result = result or following
        return result

    def _parse_and(self) -> bool:
        result = self._parse_atom()
        while self._peek("and"):
            self._take("and")
            following = self._parse_atom()
            result = result and following
        return result

    def _parse_atom(self) -> bool:
        if self._peek("("):
            self._take("(")
            result = self._parse_or()
            self._take(")")
            return result
        left_token = self._take()
        if left_token.kind not in {"word", "string"}:
            raise ValueError("requirement marker operand is invalid")
        if self._peek("not"):
            self._take("not")
            self._take("in")
            operator = "not in"
        elif self._peek("in"):
            self._take("in")
            operator = "in"
        else:
            operator_token = self._take()
            if operator_token.kind != "operator":
                raise ValueError("requirement marker comparison operator is invalid")
            operator = operator_token.value
        right_token = self._take()
        if right_token.kind not in {"word", "string"}:
            raise ValueError("requirement marker operand is invalid")

        def resolve(token: _MarkerToken) -> tuple[str, str | None]:
            if token.kind == "string":
                return token.value, None
            if token.value not in self.environment:
                raise ValueError(
                    f"unsupported requirement marker variable: {token.value}"
                )
            return self.environment[token.value], token.value

        left, left_variable = resolve(left_token)
        right, right_variable = resolve(right_token)
        version_variables = {
            "python_version",
            "python_full_version",
            "implementation_version",
        }
        return _compare_marker_values(
            left,
            operator,
            right,
            version=(
                left_variable in version_variables
                or right_variable in version_variables
            ),
        )


def _evaluate_marker(marker: str, environment: Mapping[str, str]) -> bool:
    if not marker or len(marker) > 8_192:
        raise ValueError("requirement marker is empty or exceeds its bound")
    return _MarkerParser(_tokenize_marker(marker), environment).parse()


def _split_requirement_marker(value: str) -> tuple[str, str | None]:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote is not None:
            escaped = True
            continue
        if char in "'\"":
            quote = None if quote == char else char if quote is None else quote
            continue
        if char == ";" and quote is None:
            return value[:index], value[index + 1 :].strip()
    if quote is not None:
        raise ValueError("unterminated quote in Requires-Dist")
    return value, None


def _requires_dist_names_from_values(
    values: Sequence[str], *, marker_environment: Mapping[str, str]
) -> tuple[list[str], list[str], list[str]]:
    values = list(values)
    if len(values) > MAX_REQUIREMENT_COUNT_PER_DISTRIBUTION:
        raise ValueError("distribution requirement count exceeds its bound")
    if any(
        not isinstance(value, str) or not value or len(value) > 8_192
        for value in values
    ):
        raise ValueError("Requires-Dist entry is malformed or exceeds its bound")
    names: set[str] = set()
    active_names: set[str] = set()
    for value in values:
        requirement, marker = _split_requirement_marker(value)
        match = _REQUIREMENT_NAME.match(requirement)
        if match is None:
            raise ValueError(f"cannot parse Requires-Dist name: {value!r}")
        name = canonical_distribution_name(match.group(1))
        names.add(name)
        if marker is None or _evaluate_marker(marker, marker_environment):
            active_names.add(name)
    return sorted(names), sorted(active_names), sorted(values)


def _requires_dist_names(
    message: Any, *, marker_environment: Mapping[str, str]
) -> tuple[list[str], list[str], list[str]]:
    return _requires_dist_names_from_values(
        list(message.get_all("Requires-Dist", failobj=[])),
        marker_environment=marker_environment,
    )


@dataclass(frozen=True, slots=True)
class _DistributionState:
    name: str
    version: str
    metadata_directory: str
    metadata_environment_relative: str
    metadata_sha256: str
    metadata_bytes: int
    requires_dist_names: tuple[str, ...]
    active_requires_dist_names: tuple[str, ...]
    requires_dist_entries: tuple[str, ...]

    def public_inventory(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "metadata_directory": self.metadata_directory,
            "metadata_sha256": self.metadata_sha256,
            "metadata_bytes": self.metadata_bytes,
        }


def _metadata_filename(directory_name: str) -> str:
    if directory_name.endswith(".dist-info"):
        return "METADATA"
    if directory_name.endswith(".egg-info"):
        return "PKG-INFO"
    raise ValueError("unsupported distribution metadata directory")


def _scan_distributions(
    *,
    site_root: _AnchoredRoot,
    environment_root: _AnchoredRoot,
    site_environment_relative: str,
) -> dict[str, _DistributionState]:
    before = site_root.stat_root()
    names = site_root.list_names()
    metadata_directories: list[str] = []
    for name in names:
        if not (name.endswith(".dist-info") or name.endswith(".egg-info")):
            continue
        _safe_relative_path(name, field_name="metadata directory")
        info = site_root.lstat(name)
        if _path_is_link(info) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("distribution metadata entry must be a real directory")
        metadata_directories.append(name)
    if len(metadata_directories) > MAX_DISTRIBUTION_COUNT:
        raise ValueError("installed distribution count exceeds its bound")

    found: dict[str, _DistributionState] = {}
    for directory_name in metadata_directories:
        metadata_filename = _metadata_filename(directory_name)
        relative = f"{site_environment_relative}/{directory_name}/{metadata_filename}"
        content, _ = environment_root.read_regular(
            relative,
            maximum_bytes=MAX_METADATA_BYTES,
            label=f"{directory_name}/{metadata_filename}",
        )
        message = _decode_metadata(content, label=relative)
        raw_name = message.get("Name")
        raw_version = message.get("Version")
        name = canonical_distribution_name(raw_name)
        version = _version(raw_version, field_name=f"{name} version")
        if name in found:
            raise ValueError(f"duplicate installed distribution canonical name: {name}")
        requirement_entries = list(message.get_all("Requires-Dist", failobj=[]))
        if len(requirement_entries) > MAX_REQUIREMENT_COUNT_PER_DISTRIBUTION or any(
            not isinstance(entry, str) or not entry or len(entry) > 8_192
            for entry in requirement_entries
        ):
            raise ValueError("Requires-Dist inventory is malformed or exceeds bounds")
        found[name] = _DistributionState(
            name=name,
            version=version,
            metadata_directory=directory_name,
            metadata_environment_relative=f"{site_environment_relative}/{directory_name}",
            metadata_sha256=hashlib.sha256(content).hexdigest(),
            metadata_bytes=len(content),
            requires_dist_names=(),
            active_requires_dist_names=(),
            requires_dist_entries=tuple(sorted(requirement_entries)),
        )
    if site_root.list_names() != names or _stable_fields(
        site_root.stat_root()
    ) != _stable_fields(before):
        raise RuntimeError("site-packages changed while inventorying distributions")
    return found


def _critical_closure(
    distributions: dict[str, _DistributionState],
    *,
    marker_environment: Mapping[str, str],
) -> list[str]:
    for required_root in ("torch", "triton"):
        if required_root not in distributions:
            raise ValueError(
                f"required critical distribution is missing: {required_root}"
            )
    pending = {
        name
        for name in distributions
        if name in {"torch", "triton"}
        or name.startswith("nvidia-")
        or name.startswith("cuda-")
    }
    critical: set[str] = set()
    while pending:
        name = min(pending)
        pending.remove(name)
        if name in critical:
            continue
        state = distributions[name]
        requirements, active, entries = _requires_dist_names_from_values(
            state.requires_dist_entries,
            marker_environment=marker_environment,
        )
        distributions[name] = replace(
            state,
            requires_dist_names=tuple(requirements),
            active_requires_dist_names=tuple(active),
            requires_dist_entries=tuple(entries),
        )
        critical.add(name)
        for dependency in active:
            if dependency not in distributions:
                raise ValueError(
                    f"active dependency {dependency} of {name} is not installed"
                )
            if dependency not in critical:
                pending.add(dependency)
        if len(critical) + len(pending) > MAX_CRITICAL_DISTRIBUTION_COUNT:
            raise ValueError("critical distribution closure exceeds its bound")
    closure = sorted(critical)
    expected_names = sorted(FROZEN_MIXBIT_CRITICAL_DISTRIBUTIONS)
    if closure != expected_names:
        raise ValueError("active critical closure differs from frozen mixbit audit")
    if any(
        distributions[name].version != FROZEN_MIXBIT_CRITICAL_DISTRIBUTIONS[name]
        for name in closure
    ):
        raise ValueError(
            "critical distribution versions differ from frozen mixbit audit"
        )
    return closure


def _record_declared_sha256(value: str) -> str:
    if not value.startswith("sha256="):
        raise ValueError("RECORD entries must use sha256 declarations")
    encoded = value.removeprefix("sha256=")
    if not encoded or re.fullmatch(r"[A-Za-z0-9_-]+", encoded) is None:
        raise ValueError("RECORD sha256 declaration is malformed")
    try:
        decoded = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("RECORD sha256 declaration is malformed") from exc
    if len(decoded) != hashlib.sha256().digest_size:
        raise ValueError("RECORD sha256 declaration has the wrong digest size")
    return decoded.hex()


def _verify_record(
    distribution: _DistributionState,
    *,
    environment_root: _AnchoredRoot,
    site_environment_relative: str,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    if not distribution.metadata_directory.endswith(".dist-info"):
        raise ValueError(
            f"critical distribution {distribution.name} lacks dist-info RECORD"
        )
    record_path = f"{distribution.metadata_environment_relative}/RECORD"
    record_content, _ = environment_root.read_regular(
        record_path,
        maximum_bytes=MAX_RECORD_BYTES,
        label=f"{distribution.name} RECORD",
    )
    try:
        record_text = record_content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{distribution.name} RECORD is not UTF-8") from exc
    rows = list(csv.reader(io.StringIO(record_text, newline="")))
    if not rows or len(rows) > MAX_RECORD_ENTRY_COUNT:
        raise ValueError(f"{distribution.name} RECORD entry count is invalid")

    verified_rows: list[dict[str, object]] = []
    by_path: dict[str, dict[str, object]] = {}
    unhashed: list[str] = []
    found_dotdot: set[str] = set()
    total_bytes = 0
    for index, row in enumerate(rows):
        if len(row) != 3:
            raise ValueError(
                f"{distribution.name} RECORD row {index} must have three columns"
            )
        normalized_path = _normalize_record_reference(
            row[0],
            distribution_name=distribution.name,
            site_packages_environment_relative=site_environment_relative,
        )
        if ".." in row[0].split("/"):
            found_dotdot.add(row[0])
        if normalized_path in by_path:
            raise ValueError(f"{distribution.name} RECORD contains a duplicate path")
        declared_hash, declared_size = row[1], row[2]
        if not declared_hash and not declared_size:
            if normalized_path != record_path and not (
                normalized_path.endswith(".pyc")
                and "/__pycache__/" in f"/{normalized_path}"
            ):
                raise ValueError(
                    f"{distribution.name} RECORD has a non-allowlisted unhashed entry"
                )
            unhashed.append(normalized_path)
            if normalized_path == record_path:
                actual_sha256 = hashlib.sha256(record_content).hexdigest()
                actual_bytes = len(record_content)
            else:
                actual_sha256, actual_bytes, _ = environment_root.hash_regular(
                    normalized_path,
                    maximum_bytes=MAX_CRITICAL_FILE_BYTES,
                    label=f"{distribution.name} file {normalized_path}",
                )
            entry = {
                "record_reference": row[0],
                "path": normalized_path,
                "declared_sha256": None,
                "declared_bytes": None,
                "actual_sha256": actual_sha256,
                "actual_bytes": actual_bytes,
            }
        else:
            if not declared_hash or not declared_size:
                raise ValueError(
                    f"{distribution.name} RECORD hash and size must be both present"
                )
            declared_sha256 = _record_declared_sha256(declared_hash)
            if re.fullmatch(r"0|[1-9][0-9]*", declared_size) is None:
                raise ValueError(f"{distribution.name} RECORD size is malformed")
            declared_bytes = int(declared_size)
            if declared_bytes > MAX_CRITICAL_FILE_BYTES:
                raise ValueError(f"{distribution.name} RECORD file exceeds byte bound")
            actual_sha256, actual_bytes, _ = environment_root.hash_regular(
                normalized_path,
                maximum_bytes=MAX_CRITICAL_FILE_BYTES,
                label=f"{distribution.name} file {normalized_path}",
            )
            if actual_sha256 != declared_sha256 or actual_bytes != declared_bytes:
                raise ValueError(
                    f"{distribution.name} file differs from its RECORD declaration: "
                    f"{normalized_path}"
                )
            entry = {
                "record_reference": row[0],
                "path": normalized_path,
                "declared_sha256": declared_sha256,
                "declared_bytes": declared_bytes,
                "actual_sha256": actual_sha256,
                "actual_bytes": actual_bytes,
            }
        total_bytes += int(entry["actual_bytes"])
        if total_bytes > MAX_CRITICAL_TOTAL_BYTES:
            raise ValueError("critical distribution verification exceeds byte bound")
        verified_rows.append(entry)
        by_path[normalized_path] = entry
    expected_dotdot = set(
        _ALLOWED_DOTDOT_RECORD_REFERENCES.get(distribution.name, frozenset())
    )
    if found_dotdot != expected_dotdot:
        raise ValueError(
            f"{distribution.name} RECORD dotdot references differ from the frozen audit"
        )
    if record_path not in unhashed or unhashed.count(record_path) != 1:
        raise ValueError(
            f"{distribution.name} RECORD self-entry is missing or duplicated"
        )
    metadata_files = environment_root.regular_tree_paths(
        distribution.metadata_environment_relative,
        maximum_entries=MAX_RECORD_ENTRY_COUNT,
        label=f"{distribution.name} metadata",
    )
    if any(path not in by_path for path in metadata_files):
        raise ValueError(
            f"{distribution.name} metadata contains an extra file outside RECORD"
        )
    verified_rows.sort(key=lambda item: str(item["path"]))
    unhashed.sort()
    return (
        {
            "name": distribution.name,
            "version": distribution.version,
            "metadata_directory": distribution.metadata_directory,
            "record_path": record_path,
            "record_sha256": hashlib.sha256(record_content).hexdigest(),
            "record_bytes": len(record_content),
            "record_entry_count": len(rows),
            "hashed_file_count": len(rows) - len(unhashed),
            "unhashed_entry_paths": unhashed,
            "verified_files": verified_rows,
            "verified_file_bytes": total_bytes,
            "verified_files_sha256": _canonical_sha256(verified_rows),
            "requires_dist_names": list(distribution.requires_dist_names),
            "active_requires_dist_names": list(distribution.active_requires_dist_names),
            "requires_dist_entries": list(distribution.requires_dist_entries),
        },
        by_path,
    )


def _literal_torch_version_values(content: bytes) -> tuple[str, str, str]:
    try:
        source = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("torch/version.py is not UTF-8") from exc
    try:
        tree = ast.parse(source, filename="torch/version.py", mode="exec")
    except SyntaxError as exc:
        raise ValueError("torch/version.py cannot be parsed safely") from exc
    values: dict[str, str] = {}
    wanted = {"__version__", "cuda", "git_version"}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        matched = [
            target.id
            for target in targets
            if isinstance(target, ast.Name) and target.id in wanted
        ]
        if not matched:
            continue
        value = node.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            raise ValueError("torch version identity must use literal strings")
        for name in matched:
            if name in values:
                raise ValueError(f"torch {name} is assigned more than once")
            values[name] = value.value
    if set(values) != wanted or any(
        not value or len(value) > 512 for value in values.values()
    ):
        raise ValueError("torch version identity literals are incomplete")
    return values["__version__"], values["cuda"], values["git_version"]


def _torch_module_record(
    *,
    torch_files: Mapping[str, Mapping[str, object]],
    environment_root: _AnchoredRoot,
    site_environment_relative: str,
) -> dict[str, object]:
    module_init = f"{site_environment_relative}/torch/__init__.py"
    version_file = f"{site_environment_relative}/torch/version.py"
    stray_module = f"{site_environment_relative}/torch.py"
    try:
        stray_info = environment_root.lstat(stray_module)
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISREG(stray_info.st_mode) or _path_is_link(stray_info):
            raise ValueError("stray torch.py could shadow the locked torch package")
    if module_init not in torch_files or version_file not in torch_files:
        raise ValueError("torch RECORD does not own its exact module origin files")
    for path in (module_init, version_file):
        if torch_files[path]["declared_sha256"] is None:
            raise ValueError("torch module origin files must be hashed in RECORD")
    version_content, _ = environment_root.read_regular(
        version_file,
        maximum_bytes=MAX_METADATA_BYTES,
        label="torch/version.py",
    )
    init_entry = torch_files[module_init]
    version_entry = torch_files[version_file]
    module_version, cuda_version, git_version = _literal_torch_version_values(
        version_content
    )
    if module_version != "2.10.0+cu128" or cuda_version != "12.8":
        raise ValueError("torch module/CUDA versions differ from frozen mixbit runtime")
    return {
        "distribution_name": "torch",
        "module_version": module_version,
        "cuda_version": cuda_version,
        "module_init_path": module_init,
        "module_init_sha256": init_entry["actual_sha256"],
        "module_init_bytes": init_entry["actual_bytes"],
        "version_file_path": version_file,
        "version_file_sha256": version_entry["actual_sha256"],
        "version_file_bytes": version_entry["actual_bytes"],
        "git_version": git_version,
    }


def _assert_cross_owned_file_identities(
    critical_records: Sequence[Mapping[str, object]],
) -> None:
    identities: dict[str, tuple[object, object]] = {}
    for distribution in critical_records:
        files = distribution.get("verified_files")
        if not isinstance(files, list):
            raise TypeError("critical verified_files must be a list")
        for row in files:
            if not isinstance(row, Mapping):
                raise TypeError("critical verified file row must be a mapping")
            path = row.get("path")
            if not isinstance(path, str):
                raise TypeError("critical verified file path must be text")
            identity = (row.get("actual_sha256"), row.get("actual_bytes"))
            previous = identities.setdefault(path, identity)
            if previous != identity:
                raise ValueError(
                    "cross-owned critical file has contradictory actual identities"
                )


def _runtime_executable_lexical_path() -> Path:
    return _absolute_no_expand(sys.executable)


def _resolve_interpreter_executable(
    lexical: Path, *, environment_prefix: Path
) -> tuple[Path, list[dict[str, str]]]:
    _environment_relative(
        lexical,
        environment_prefix=environment_prefix,
        field_name="CPython executable lexical path",
    )
    _assert_nonsymlink_chain(lexical.parent, label="CPython executable parent")
    allowed_parents = {environment_prefix, environment_prefix / "bin"}
    if lexical.parent not in allowed_parents:
        raise ValueError("CPython executable must be at the environment root or bin")
    info = lexical.lstat()
    if _path_is_link(info):
        if lexical.parent != environment_prefix / "bin":
            raise ValueError(
                "CPython executable symlink must be inside environment bin"
            )
        target = os.readlink(lexical)
        target_path = PurePosixPath(target)
        if (
            not target
            or target_path.is_absolute()
            or len(target_path.parts) != 1
            or target_path.parts[0] in {"", ".", ".."}
            or "\\" in target
            or ":" in target
        ):
            raise ValueError(
                "CPython executable symlink target is not one safe basename"
            )
        resolved = lexical.parent / target
        resolved_info = resolved.lstat()
        if _path_is_link(resolved_info) or not stat.S_ISREG(resolved_info.st_mode):
            raise ValueError(
                "CPython executable symlink must resolve once to a regular file"
            )
        return resolved, [{"path": str(lexical), "target": target}]
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(
            "CPython executable must be a regular file or controlled symlink"
        )
    return lexical, []


def _runtime_version_info() -> list[object]:
    info = sys.version_info
    return [info.major, info.minor, info.micro, info.releaselevel, info.serial]


def _assert_proc_self_exe_identity(
    resolved_path: Path, resolved_info: os.stat_result
) -> None:
    if os.name != "posix":
        return
    proc_path = Path("/proc/self/exe")
    proc_info = os.stat(proc_path)
    if not _same_identity(proc_info, resolved_info):
        raise ValueError("sys.executable target differs from /proc/self/exe")
    if Path(os.path.realpath(proc_path)) != resolved_path:
        raise ValueError("resolved CPython path differs from /proc/self/exe")


def _build_runtime_lock(
    *,
    environment_prefix: str | Path,
    site_packages_path: str | Path,
    environment_prefix_fd: int | None,
    site_packages_fd: int | None,
) -> dict[str, Any]:
    prefix = _absolute_no_expand(environment_prefix)
    site_packages = _absolute_no_expand(site_packages_path)
    prefix_text = _lexical_absolute_path(str(prefix), field_name="environment prefix")
    site_text = _lexical_absolute_path(str(site_packages), field_name="site-packages")
    site_relative = _environment_relative(
        site_packages,
        environment_prefix=prefix,
        field_name="site-packages environment-relative path",
    )
    if platform.python_implementation() != "CPython":
        raise ValueError("M04a runtime lock requires CPython")
    executable_lexical = _runtime_executable_lexical_path()
    executable, symlink_chain = _resolve_interpreter_executable(
        executable_lexical, environment_prefix=prefix
    )
    executable_relative = _environment_relative(
        executable,
        environment_prefix=prefix,
        field_name="CPython executable",
    )

    with (
        _AnchoredRoot(
            prefix, descriptor=environment_prefix_fd, label="environment prefix"
        ) as environment_root,
        _AnchoredRoot(
            site_packages, descriptor=site_packages_fd, label="site-packages"
        ) as site_root,
    ):
        expected_site_info = environment_root.lstat(site_relative)
        if not stat.S_ISDIR(expected_site_info.st_mode) or _path_is_link(
            expected_site_info
        ):
            raise ValueError("site-packages is not a real directory in the environment")
        if not _same_identity(expected_site_info, site_root.stat_root()):
            raise ValueError("site-packages descriptor is not bound to the environment")
        executable_sha256, executable_bytes, executable_info = (
            environment_root.hash_regular(
                executable_relative,
                maximum_bytes=MAX_CRITICAL_FILE_BYTES,
                label="CPython executable",
            )
        )
        _assert_proc_self_exe_identity(executable, executable_info)
        marker_environment = _marker_environment()
        distributions = _scan_distributions(
            site_root=site_root,
            environment_root=environment_root,
            site_environment_relative=site_relative,
        )
        critical_names = _critical_closure(
            distributions, marker_environment=marker_environment
        )
        critical_records: list[dict[str, object]] = []
        torch_files: dict[str, dict[str, object]] | None = None
        critical_total = 0
        for name in critical_names:
            record, verified = _verify_record(
                distributions[name],
                environment_root=environment_root,
                site_environment_relative=site_relative,
            )
            critical_total += int(record["verified_file_bytes"])
            if critical_total > MAX_CRITICAL_TOTAL_BYTES:
                raise ValueError("critical closure verification exceeds byte bound")
            critical_records.append(record)
            if name == "torch":
                torch_files = verified
        _assert_cross_owned_file_identities(critical_records)
        if torch_files is None:
            raise RuntimeError("critical closure lost torch")
        torch_module = _torch_module_record(
            torch_files=torch_files,
            environment_root=environment_root,
            site_environment_relative=site_relative,
        )
        environment_root.assert_stable()
        site_root.assert_stable()

    semantic: dict[str, Any] = {
        "schema": PYTHON_RUNTIME_LOCK_SCHEMA_VERSION,
        "algorithm": dict(_ALGORITHM),
        "environment": {
            "prefix_path": prefix_text,
            "site_packages_path": site_text,
            "site_packages_environment_relative": site_relative,
            "marker_environment": marker_environment,
        },
        "python": {
            "implementation": "CPython",
            "version": platform.python_version(),
            "version_info": _runtime_version_info(),
            "executable_lexical_path": str(executable_lexical),
            "executable_symlink_chain": symlink_chain,
            "executable_resolved_path": str(executable),
            "executable_device": executable_info.st_dev,
            "executable_inode": executable_info.st_ino,
            "executable_sha256": executable_sha256,
            "executable_bytes": executable_bytes,
        },
        "distributions": [
            distributions[name].public_inventory() for name in sorted(distributions)
        ],
        "critical_distribution_names": critical_names,
        "critical_distributions": critical_records,
        "torch_module": torch_module,
    }
    payload = {**semantic, "runtime_lock_id": _canonical_sha256(semantic)}
    return validate_python_runtime_lock_payload(payload)


def build_python_runtime_lock(
    *,
    environment_prefix: str | Path | None = None,
    site_packages_path: str | Path | None = None,
    environment_prefix_fd: int | None = None,
    site_packages_fd: int | None = None,
) -> dict[str, Any]:
    """Build the lock from a live environment without importing Torch.

    On Linux, callers may pass already-bound directory descriptors.  The
    logical paths remain part of the lock and must name those exact descriptor
    identities.
    """

    prefix = sys.prefix if environment_prefix is None else environment_prefix
    if site_packages_path is None:
        purelib = sysconfig.get_paths()["purelib"]
        platlib = sysconfig.get_paths()["platlib"]
        if os.path.normcase(os.path.abspath(purelib)) != os.path.normcase(
            os.path.abspath(platlib)
        ):
            raise ValueError(
                "purelib and platlib differ; pass the production distribution root explicitly"
            )
        site_packages_path = purelib
    return _build_runtime_lock(
        environment_prefix=prefix,
        site_packages_path=site_packages_path,
        environment_prefix_fd=environment_prefix_fd,
        site_packages_fd=site_packages_fd,
    )


def _validate_distribution_inventory(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value or len(value) > MAX_DISTRIBUTION_COUNT:
        raise ValueError("distribution inventory count is invalid")
    normalized: list[dict[str, object]] = []
    previous = ""
    metadata_directories: set[str] = set()
    for index, raw in enumerate(value):
        item = _exact_dict(raw, _DISTRIBUTION_FIELDS, label="distribution inventory")
        name = canonical_distribution_name(item["name"])
        if item["name"] != name or name <= previous:
            raise ValueError(
                "distribution inventory names must be unique canonical order"
            )
        previous = name
        _version(item["version"], field_name=f"distributions[{index}].version")
        directory = _safe_relative_path(
            item["metadata_directory"],
            field_name=f"distributions[{index}].metadata_directory",
        )
        if "/" in directory or not directory.endswith((".dist-info", ".egg-info")):
            raise ValueError("metadata_directory must be one direct metadata directory")
        if directory in metadata_directories:
            raise ValueError("installed distributions reuse one metadata_directory")
        metadata_directories.add(directory)
        _sha256(item["metadata_sha256"], field_name="metadata_sha256")
        _strict_int(
            item["metadata_bytes"],
            field_name="metadata_bytes",
            minimum=1,
            maximum=MAX_METADATA_BYTES,
        )
        normalized.append(dict(item))
    return normalized


def _validate_requirement_names(value: object, *, field_name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) > MAX_REQUIREMENT_COUNT_PER_DISTRIBUTION
    ):
        raise ValueError(f"{field_name} count is invalid")
    names: list[str] = []
    previous = ""
    for raw_name in value:
        name = canonical_distribution_name(raw_name)
        if raw_name != name or name <= previous:
            raise ValueError(f"{field_name} must contain unique canonical names")
        names.append(name)
        previous = name
    return names


def _validate_verified_files(
    value: object,
    *,
    distribution_name: str,
    site_environment_relative: str,
    record_path: str,
    record_sha256: str,
    record_bytes: int,
) -> tuple[list[dict[str, object]], list[str], int, int]:
    if not isinstance(value, list) or not value or len(value) > MAX_RECORD_ENTRY_COUNT:
        raise ValueError("verified file manifest count is invalid")
    normalized: list[dict[str, object]] = []
    unhashed: list[str] = []
    hashed_count = 0
    total_bytes = 0
    previous = ""
    dotdot_references: set[str] = set()
    for index, raw in enumerate(value):
        item = _exact_dict(raw, _VERIFIED_FILE_FIELDS, label="verified file")
        reference = item["record_reference"]
        if not isinstance(reference, str):
            raise TypeError("verified file record_reference must be text")
        path = _normalize_record_reference(
            reference,
            distribution_name=distribution_name,
            site_packages_environment_relative=site_environment_relative,
        )
        if item["path"] != path or path <= previous:
            raise ValueError("verified files must have unique normalized path order")
        previous = path
        if ".." in reference.split("/"):
            dotdot_references.add(reference)
        actual_sha256 = _sha256(
            item["actual_sha256"],
            field_name=f"verified_files[{index}].actual_sha256",
        )
        actual_bytes = _strict_int(
            item["actual_bytes"],
            field_name=f"verified_files[{index}].actual_bytes",
            maximum=MAX_CRITICAL_FILE_BYTES,
        )
        total_bytes += actual_bytes
        if total_bytes > MAX_CRITICAL_TOTAL_BYTES:
            raise ValueError("verified file manifest exceeds byte bound")
        declared_sha256 = item["declared_sha256"]
        declared_bytes = item["declared_bytes"]
        if declared_sha256 is None and declared_bytes is None:
            if path != record_path and not (
                path.endswith(".pyc") and "/__pycache__/" in f"/{path}"
            ):
                raise ValueError(
                    "unhashed verified file is outside the exact allowlist"
                )
            unhashed.append(path)
        elif declared_sha256 is None or declared_bytes is None:
            raise ValueError(
                "declared hash and bytes must be both null or both present"
            )
        else:
            declared_sha256 = _sha256(
                declared_sha256,
                field_name=f"verified_files[{index}].declared_sha256",
            )
            declared_bytes = _strict_int(
                declared_bytes,
                field_name=f"verified_files[{index}].declared_bytes",
                maximum=MAX_CRITICAL_FILE_BYTES,
            )
            if declared_sha256 != actual_sha256 or declared_bytes != actual_bytes:
                raise ValueError(
                    "declared and independently measured file identity differ"
                )
            hashed_count += 1
        if path == record_path and (
            actual_sha256 != record_sha256 or actual_bytes != record_bytes
        ):
            raise ValueError("RECORD self-entry differs from committed RECORD bytes")
        normalized.append(dict(item))
    if set(unhashed).isdisjoint({record_path}) or unhashed.count(record_path) != 1:
        raise ValueError("verified file manifest lacks one unhashed RECORD self-entry")
    if dotdot_references != set(
        _ALLOWED_DOTDOT_RECORD_REFERENCES.get(distribution_name, frozenset())
    ):
        raise ValueError("verified file dotdot references differ from the frozen audit")
    return normalized, sorted(unhashed), hashed_count, total_bytes


def _validate_critical_distributions(
    value: object,
    *,
    names: Sequence[str],
    inventory: Mapping[str, Mapping[str, object]],
    site_environment_relative: str,
    marker_environment: Mapping[str, str],
) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != len(names):
        raise ValueError("critical distribution records differ from closure names")
    normalized: list[dict[str, object]] = []
    total = 0
    for index, raw in enumerate(value):
        item = _exact_dict(
            raw,
            _CRITICAL_DISTRIBUTION_FIELDS,
            label="critical distribution",
        )
        name = canonical_distribution_name(item["name"])
        if item["name"] != name or name != names[index]:
            raise ValueError("critical distributions are not in exact closure order")
        if name not in inventory:
            raise ValueError("critical distribution is absent from inventory")
        if (
            item["version"] != inventory[name]["version"]
            or item["metadata_directory"] != inventory[name]["metadata_directory"]
        ):
            raise ValueError("critical distribution identity differs from inventory")
        record_path = _safe_relative_path(item["record_path"], field_name="record_path")
        expected_suffix = f"/{item['metadata_directory']}/RECORD"
        if not record_path.endswith(expected_suffix):
            raise ValueError("critical RECORD path differs from metadata directory")
        count = _strict_int(
            item["record_entry_count"],
            field_name="record_entry_count",
            minimum=1,
            maximum=MAX_RECORD_ENTRY_COUNT,
        )
        record_sha256 = _sha256(item["record_sha256"], field_name="record_sha256")
        record_bytes = _strict_int(
            item["record_bytes"],
            field_name="record_bytes",
            minimum=1,
            maximum=MAX_RECORD_BYTES,
        )
        verified_files, unhashed, hashed_count, recomputed_bytes = (
            _validate_verified_files(
                item["verified_files"],
                distribution_name=name,
                site_environment_relative=site_environment_relative,
                record_path=record_path,
                record_sha256=record_sha256,
                record_bytes=record_bytes,
            )
        )
        if len(verified_files) != count:
            raise ValueError("verified file rows differ from RECORD entry count")
        if (
            _strict_int(
                item["hashed_file_count"],
                field_name="hashed_file_count",
                maximum=MAX_RECORD_ENTRY_COUNT,
            )
            != hashed_count
        ):
            raise ValueError("hashed file count differs from verified file rows")
        if item["unhashed_entry_paths"] != unhashed:
            raise ValueError("unhashed path inventory differs from verified file rows")
        verified_bytes = _strict_int(
            item["verified_file_bytes"],
            field_name="verified_file_bytes",
            minimum=1,
            maximum=MAX_CRITICAL_TOTAL_BYTES,
        )
        total += verified_bytes
        if total > MAX_CRITICAL_TOTAL_BYTES:
            raise ValueError("critical verified bytes exceed the closure bound")
        if verified_bytes != recomputed_bytes:
            raise ValueError("verified byte total differs from verified file rows")
        if _sha256(
            item["verified_files_sha256"], field_name="verified_files_sha256"
        ) != _canonical_sha256(verified_files):
            raise ValueError("verified file aggregate SHA-256 mismatch")
        requirements = _validate_requirement_names(
            item["requires_dist_names"],
            field_name=f"critical_distributions[{index}].requires_dist_names",
        )
        active = _validate_requirement_names(
            item["active_requires_dist_names"],
            field_name=f"critical_distributions[{index}].active_requires_dist_names",
        )
        if any(dependency not in inventory for dependency in active):
            raise ValueError("active critical dependency is absent from inventory")
        if not set(active).issubset(requirements):
            raise ValueError(
                "active critical dependencies are not a requirement subset"
            )
        entries = item["requires_dist_entries"]
        if (
            not isinstance(entries, list)
            or len(entries) > MAX_REQUIREMENT_COUNT_PER_DISTRIBUTION
            or entries != sorted(entries)
            or any(
                not isinstance(entry, str) or not entry or len(entry) > 8_192
                for entry in entries
            )
        ):
            raise ValueError("critical Requires-Dist entries are not canonical")
        derived_requirements, derived_active, derived_entries = (
            _requires_dist_names_from_values(
                entries, marker_environment=marker_environment
            )
        )
        if (
            requirements != derived_requirements
            or active != derived_active
            or entries != derived_entries
        ):
            raise ValueError("critical dependency names differ from marker evaluation")
        metadata_path = (
            f"{site_environment_relative}/{item['metadata_directory']}/METADATA"
        )
        metadata_rows = [row for row in verified_files if row["path"] == metadata_path]
        if (
            len(metadata_rows) != 1
            or metadata_rows[0]["actual_sha256"] != inventory[name]["metadata_sha256"]
            or metadata_rows[0]["actual_bytes"] != inventory[name]["metadata_bytes"]
        ):
            raise ValueError("critical METADATA inventory differs from its RECORD row")
        item["verified_files"] = verified_files
        normalized.append(dict(item))
    return normalized


def validate_python_runtime_lock_payload(payload: object) -> dict[str, Any]:
    """Purely validate and defensively copy one exact runtime-lock payload."""

    normalized = _exact_dict(payload, _LOCK_FIELDS, label="Python runtime lock")
    if normalized["schema"] != PYTHON_RUNTIME_LOCK_SCHEMA_VERSION:
        raise ValueError("unsupported Python runtime-lock schema")
    algorithm = _exact_dict(
        normalized["algorithm"], _ALGORITHM_FIELDS, label="algorithm"
    )
    if algorithm != _ALGORITHM:
        raise ValueError("Python runtime-lock algorithm drifted")
    environment = _exact_dict(
        normalized["environment"], _ENVIRONMENT_FIELDS, label="environment"
    )
    prefix = environment["prefix_path"]
    prefix_flavor, prefix_path = _payload_absolute_path(
        prefix, field_name="prefix_path"
    )
    site = environment["site_packages_path"]
    site_flavor, site_path = _payload_absolute_path(
        site, field_name="site_packages_path"
    )
    site_relative = _safe_relative_path(
        environment["site_packages_environment_relative"],
        field_name="site_packages_environment_relative",
    )
    if (
        prefix_flavor != site_flavor
        or prefix_path.joinpath(*PurePosixPath(site_relative).parts) != site_path
    ):
        raise ValueError("site-packages logical and environment-relative paths differ")
    marker_environment = _exact_dict(
        environment["marker_environment"],
        _MARKER_ENVIRONMENT_FIELDS,
        label="marker_environment",
    )
    if any(
        not isinstance(value, str) or len(value) > 4_096
        for value in marker_environment.values()
    ):
        raise ValueError("marker environment values must be bounded strings")
    if any(
        marker_environment.get(name) != value
        for name, value in FROZEN_MIXBIT_MARKER_ENVIRONMENT.items()
    ):
        raise ValueError(
            "marker environment differs from frozen mixbit CPython runtime"
        )
    environment["marker_environment"] = marker_environment
    python = _exact_dict(normalized["python"], _PYTHON_FIELDS, label="python")
    if python["implementation"] != "CPython":
        raise ValueError("runtime lock requires CPython")
    if python["version"] != "3.10.20":
        raise ValueError("runtime lock requires exact CPython 3.10.20")
    version_info = python["version_info"]
    if (
        not isinstance(version_info, list)
        or len(version_info) != 5
        or any(type(value) is not int or value < 0 for value in version_info[:3])
        or version_info[3] not in {"alpha", "beta", "candidate", "final"}
        or type(version_info[4]) is not int
        or version_info[4] < 0
    ):
        raise ValueError("python.version_info is malformed")
    if version_info != [3, 10, 20, "final", 0]:
        raise ValueError("runtime lock requires exact CPython version_info")
    lexical_flavor, lexical_path = _payload_absolute_path(
        python["executable_lexical_path"],
        field_name="python.executable_lexical_path",
    )
    resolved_flavor, resolved_path = _payload_absolute_path(
        python["executable_resolved_path"],
        field_name="python.executable_resolved_path",
    )
    if lexical_flavor != prefix_flavor or resolved_flavor != prefix_flavor:
        raise ValueError("CPython executable path flavor differs from environment")
    _payload_require_within(
        lexical_path,
        prefix_path,
        field_name="python.executable_lexical_path",
    )
    _payload_require_within(
        resolved_path,
        prefix_path,
        field_name="python.executable_resolved_path",
    )
    symlink_chain = python["executable_symlink_chain"]
    if symlink_chain == []:
        if lexical_path != resolved_path:
            raise ValueError("unlinked CPython executable paths must be identical")
    elif isinstance(symlink_chain, list) and len(symlink_chain) == 1:
        link = _exact_dict(
            symlink_chain[0], {"path", "target"}, label="executable symlink"
        )
        target = link["target"]
        if (
            link["path"] != str(lexical_path)
            or not isinstance(target, str)
            or not target
            or "\x00" in target
            or "\\" in target
            or ":" in target
            or PurePosixPath(target).is_absolute()
            or len(PurePosixPath(target).parts) != 1
            or target in {".", ".."}
            or lexical_path.parent / target != resolved_path
            or lexical_path.parent != prefix_path / "bin"
        ):
            raise ValueError("CPython executable symlink chain is malformed")
        python["executable_symlink_chain"] = [link]
    else:
        raise ValueError("CPython executable allows at most one controlled symlink")
    _strict_int(python["executable_device"], field_name="python.executable_device")
    _strict_int(
        python["executable_inode"], field_name="python.executable_inode", minimum=1
    )
    _sha256(python["executable_sha256"], field_name="python.executable_sha256")
    _strict_int(
        python["executable_bytes"],
        field_name="python.executable_bytes",
        minimum=1,
        maximum=MAX_CRITICAL_FILE_BYTES,
    )
    distributions = _validate_distribution_inventory(normalized["distributions"])
    inventory = {str(item["name"]): item for item in distributions}
    critical_names = _validate_requirement_names(
        normalized["critical_distribution_names"],
        field_name="critical_distribution_names",
    )
    if critical_names != sorted(FROZEN_MIXBIT_CRITICAL_DISTRIBUTIONS):
        raise ValueError("critical closure differs from exact frozen mixbit set")
    if any(
        inventory[name]["version"] != version
        for name, version in FROZEN_MIXBIT_CRITICAL_DISTRIBUTIONS.items()
    ):
        raise ValueError("critical inventory versions differ from frozen mixbit set")
    critical = _validate_critical_distributions(
        normalized["critical_distributions"],
        names=critical_names,
        inventory=inventory,
        site_environment_relative=site_relative,
        marker_environment=marker_environment,
    )
    _assert_cross_owned_file_identities(critical)
    record_entry_count = sum(int(item["record_entry_count"]) for item in critical)
    unhashed_paths = [
        path for item in critical for path in item["unhashed_entry_paths"]
    ]
    verified_file_bytes = sum(int(item["verified_file_bytes"]) for item in critical)
    if (
        record_entry_count != FROZEN_MIXBIT_CRITICAL_RECORD_ENTRY_COUNT
        or len(unhashed_paths) != FROZEN_MIXBIT_CRITICAL_UNHASHED_ENTRY_COUNT
        or sum(path.endswith(".pyc") for path in unhashed_paths)
        != FROZEN_MIXBIT_CRITICAL_PYC_ENTRY_COUNT
        or verified_file_bytes != FROZEN_MIXBIT_CRITICAL_VERIFIED_FILE_BYTES
    ):
        raise ValueError("critical file inventory differs from frozen mixbit audit")
    active_by_name = {
        str(item["name"]): list(item["active_requires_dist_names"]) for item in critical
    }
    reached = {
        name
        for name in critical_names
        if name in {"torch", "triton"}
        or name.startswith("nvidia-")
        or name.startswith("cuda-")
    }
    pending = list(reached)
    while pending:
        name = pending.pop()
        for dependency in active_by_name[name]:
            if dependency not in reached:
                reached.add(dependency)
                pending.append(dependency)
    if reached != set(critical_names):
        raise ValueError("active dependency edges do not derive the critical closure")
    torch_module = _exact_dict(
        normalized["torch_module"], _TORCH_MODULE_FIELDS, label="torch_module"
    )
    if torch_module["distribution_name"] != "torch":
        raise ValueError("torch module must be owned by the torch distribution")
    if (
        torch_module["module_version"] != "2.10.0+cu128"
        or torch_module["cuda_version"] != "12.8"
    ):
        raise ValueError("torch module/CUDA versions differ from frozen mixbit runtime")
    expected_module = f"{site_relative}/torch/__init__.py"
    expected_version = f"{site_relative}/torch/version.py"
    if (
        torch_module["module_init_path"] != expected_module
        or torch_module["version_file_path"] != expected_version
    ):
        raise ValueError("torch module origin differs from the locked distribution")
    for prefix_name in ("module_init", "version_file"):
        _sha256(
            torch_module[f"{prefix_name}_sha256"],
            field_name=f"torch_module.{prefix_name}_sha256",
        )
        _strict_int(
            torch_module[f"{prefix_name}_bytes"],
            field_name=f"torch_module.{prefix_name}_bytes",
            minimum=1,
            maximum=MAX_METADATA_BYTES,
        )
    git_version = torch_module["git_version"]
    if not isinstance(git_version, str) or not git_version or len(git_version) > 512:
        raise ValueError("torch git_version is malformed")
    torch_distribution = next(item for item in critical if item["name"] == "torch")
    torch_files = {item["path"]: item for item in torch_distribution["verified_files"]}
    for file_prefix, expected_path in (
        ("module_init", expected_module),
        ("version_file", expected_version),
    ):
        row = torch_files.get(expected_path)
        if (
            row is None
            or row["declared_sha256"] is None
            or row["actual_sha256"] != torch_module[f"{file_prefix}_sha256"]
            or row["actual_bytes"] != torch_module[f"{file_prefix}_bytes"]
        ):
            raise ValueError("torch module identity differs from its owned RECORD row")
    semantic = dict(normalized)
    claimed_id = _sha256(semantic.pop("runtime_lock_id"), field_name="runtime_lock_id")
    if claimed_id != _canonical_sha256(semantic):
        raise ValueError("runtime_lock_id does not match semantic content")
    return {
        **semantic,
        "runtime_lock_id": claimed_id,
        "algorithm": algorithm,
        "environment": environment,
        "python": python,
        "distributions": distributions,
        "critical_distribution_names": critical_names,
        "critical_distributions": critical,
        "torch_module": torch_module,
    }


def canonical_python_runtime_lock_bytes(payload: object) -> bytes:
    return _serialize_json(validate_python_runtime_lock_payload(payload))


def _strict_json_bytes(content: bytes) -> object:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Python runtime lock is not UTF-8") from exc

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("Python runtime lock is invalid JSON") from exc


def _read_external_regular(path: Path, *, maximum_bytes: int) -> bytes:
    absolute = _absolute_no_expand(path)
    _assert_nonsymlink_chain(absolute.parent, label="runtime-lock parent")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOINHERIT", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(absolute, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Python runtime lock must be a regular file")
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise ValueError("Python runtime lock byte size is invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError("Python runtime lock became short")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError("Python runtime lock grew while reading")
        after = os.fstat(descriptor)
        if _stable_fields(before) != _stable_fields(after):
            raise RuntimeError("Python runtime lock changed while reading")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


@dataclass(frozen=True, slots=True)
class PythonRuntimeLockArtifact:
    snapshot: bytes
    payload: dict[str, Any]
    artifact_sha256: str
    _token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _READER_TOKEN:
            raise TypeError(
                "PythonRuntimeLockArtifact must come from the external reader"
            )
        validated = validate_python_runtime_lock_payload(self.payload)
        if canonical_python_runtime_lock_bytes(validated) != self.snapshot:
            raise ValueError("Python runtime lock snapshot is not canonical")
        if hashlib.sha256(self.snapshot).hexdigest() != self.artifact_sha256:
            raise ValueError("Python runtime lock artifact SHA-256 mismatch")


def read_python_runtime_lock_artifact(
    path: str | Path,
    *,
    expected_artifact_sha256: str,
) -> PythonRuntimeLockArtifact:
    expected = _sha256(expected_artifact_sha256, field_name="expected_artifact_sha256")
    snapshot = _read_external_regular(Path(path), maximum_bytes=MAX_RUNTIME_LOCK_BYTES)
    actual = hashlib.sha256(snapshot).hexdigest()
    if actual != expected:
        raise ValueError("Python runtime lock differs from its external commitment")
    payload = validate_python_runtime_lock_payload(_strict_json_bytes(snapshot))
    if canonical_python_runtime_lock_bytes(payload) != snapshot:
        raise ValueError("Python runtime lock bytes are not canonical")
    return PythonRuntimeLockArtifact(
        snapshot=snapshot,
        payload=payload,
        artifact_sha256=actual,
        _token=_READER_TOKEN,
    )


def _fresh_payload_from_artifact(
    artifact: PythonRuntimeLockArtifact, *, expected_artifact_sha256: str
) -> dict[str, Any]:
    if type(artifact) is not PythonRuntimeLockArtifact:
        raise TypeError("runtime validation requires an externally read lock artifact")
    expected = _sha256(expected_artifact_sha256, field_name="expected_artifact_sha256")
    if (
        artifact.artifact_sha256 != expected
        or hashlib.sha256(artifact.snapshot).hexdigest() != expected
    ):
        raise ValueError("runtime-lock snapshot differs from external commitment")
    fresh = validate_python_runtime_lock_payload(_strict_json_bytes(artifact.snapshot))
    if canonical_python_runtime_lock_bytes(fresh) != artifact.snapshot:
        raise ValueError("runtime-lock snapshot is no longer canonical")
    if artifact.payload != fresh:
        raise ValueError("runtime-lock payload was mutated after external read")
    return fresh


def _assert_no_torch_shadow_roots(
    *,
    expected_site_packages: str,
    ordered_import_roots: Sequence[str | Path],
) -> None:
    protected_top_levels = {"torch", "triton", "nvidia", "cuda"}

    def shadows(name: str) -> bool:
        return any(
            name == protected or name.startswith(f"{protected}.")
            for protected in protected_top_levels
        )

    expected = _absolute_no_expand(expected_site_packages)
    seen_expected = False
    for index, raw_root in enumerate(ordered_import_roots):
        root = _absolute_no_expand(raw_root)
        _assert_nonsymlink_chain(root, label=f"import root {index}")
        if root == expected:
            if seen_expected:
                raise ValueError("site-packages appears more than once in import roots")
            seen_expected = True
            continue
        if seen_expected:
            continue
        if root.is_dir():
            top_level = os.listdir(root)
            if len(top_level) > MAX_DISTRIBUTION_COUNT:
                raise ValueError("import root exceeds top-level entry bound")
            if any(shadows(name) for name in top_level):
                raise ValueError(
                    f"import root {index} contains a torch shadow before site-packages"
                )
        elif root.is_file():
            # Reviewed runtime roots can be ZIP snapshots.  Search names without
            # importing zipimport or executing a path hook.
            import zipfile

            if not zipfile.is_zipfile(root):
                raise ValueError(
                    "non-directory import root before site-packages is not ZIP"
                )
            with zipfile.ZipFile(root, "r") as archive:
                names = set(archive.namelist())
            top_level = {name.split("/", 1)[0] for name in names if name}
            if any(shadows(name) for name in top_level):
                raise ValueError(
                    f"import root {index} contains a torch shadow before site-packages"
                )
        else:
            raise FileNotFoundError(root)
    if not seen_expected:
        raise ValueError("ordered import roots omit the locked site-packages root")


def _assert_no_critical_shadow_fds(
    descriptors: Sequence[int], *, site_index: int
) -> None:
    protected_top_levels = {"torch", "triton", "nvidia", "cuda"}
    for index, descriptor in enumerate(descriptors):
        if type(descriptor) is not int or descriptor < 3:
            raise ValueError("bound import descriptor aliases standard I/O")
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("bound import descriptor is not a directory")
        if index >= site_index:
            continue
        names = os.listdir(descriptor)
        if len(names) > MAX_DISTRIBUTION_COUNT:
            raise ValueError("bound import root exceeds top-level entry bound")
        if any(
            name == protected or name.startswith(f"{protected}.")
            for name in names
            for protected in protected_top_levels
        ):
            raise ValueError("bound import root contains a critical package shadow")


def validate_live_python_runtime_lock(
    artifact: PythonRuntimeLockArtifact,
    *,
    expected_artifact_sha256: str,
    environment_prefix_fd: int | None = None,
    site_packages_fd: int | None = None,
    ordered_import_roots: Sequence[str | Path] | None = None,
    bound_import_roots: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Rebuild and compare the live runtime against an externally bound lock."""

    locked = _fresh_payload_from_artifact(
        artifact, expected_artifact_sha256=expected_artifact_sha256
    )
    environment = locked["environment"]
    live = build_python_runtime_lock(
        environment_prefix=environment["prefix_path"],
        site_packages_path=environment["site_packages_path"],
        environment_prefix_fd=environment_prefix_fd,
        site_packages_fd=site_packages_fd,
    )
    if live != locked:
        raise ValueError("live Python runtime differs from python-runtime-lock.json")
    if ordered_import_roots is not None:
        if isinstance(ordered_import_roots, (str, bytes)):
            raise TypeError("ordered_import_roots must be a path sequence")
        scan_roots = ordered_import_roots
        expected_site = environment["site_packages_path"]
        if bound_import_roots is not None:
            if isinstance(bound_import_roots, (str, bytes)) or len(
                bound_import_roots
            ) != len(ordered_import_roots):
                raise ValueError("bound import roots differ from logical root count")
            descriptors: list[int] = []
            for logical, bound in zip(
                ordered_import_roots, bound_import_roots, strict=True
            ):
                match = re.fullmatch(r"/proc/self/fd/([0-9]+)", os.fspath(bound))
                if match is None:
                    raise ValueError("bound import root is not one exact proc-fd path")
                descriptors.append(int(match.group(1)))
                if not os.path.samefile(logical, bound):
                    raise ValueError(
                        "bound import root identity differs from logical path"
                    )
            site_indices = [
                index
                for index, logical in enumerate(ordered_import_roots)
                if _absolute_no_expand(logical)
                == _absolute_no_expand(environment["site_packages_path"])
            ]
            if len(site_indices) != 1:
                raise ValueError(
                    "logical import roots do not contain one locked site root"
                )
            _assert_no_critical_shadow_fds(descriptors, site_index=site_indices[0])
        else:
            _assert_no_torch_shadow_roots(
                expected_site_packages=expected_site,
                ordered_import_roots=scan_roots,
            )
    elif bound_import_roots is not None:
        raise ValueError("bound import roots require logical ordered_import_roots")
    return locked


def validate_imported_torch_runtime(
    artifact: PythonRuntimeLockArtifact,
    *,
    expected_artifact_sha256: str,
    imported_torch: object,
) -> dict[str, Any]:
    """Bind an already imported Torch module to its locked on-disk origin.

    The caller owns the import.  Keeping this function argument-based preserves
    the module's Torch-free import boundary.
    """

    payload = _fresh_payload_from_artifact(
        artifact, expected_artifact_sha256=expected_artifact_sha256
    )
    environment = payload["environment"]
    torch_identity = payload["torch_module"]
    expected_init = Path(environment["prefix_path"]).joinpath(
        *PurePosixPath(torch_identity["module_init_path"]).parts
    )
    expected_package = expected_init.parent
    raw_file = getattr(imported_torch, "__file__", None)
    spec = getattr(imported_torch, "__spec__", None)
    raw_origin = getattr(spec, "origin", None)
    raw_locations = getattr(spec, "submodule_search_locations", None)
    raw_path = getattr(imported_torch, "__path__", None)
    if (
        not isinstance(raw_file, str)
        or not isinstance(raw_origin, str)
        or raw_origin != raw_file
        or not isinstance(raw_locations, Sequence)
        or isinstance(raw_locations, (str, bytes))
        or list(raw_locations) != list(raw_path or [])
        or len(raw_locations) != 1
        or not isinstance(raw_locations[0], str)
    ):
        raise ValueError("imported Torch module/spec origin metadata is malformed")
    origin = _absolute_no_expand(raw_file)
    package_location = _absolute_no_expand(raw_locations[0])
    logical_origin = str(expected_init)
    logical_package = str(expected_package)
    bound_origin_match = re.fullmatch(
        r"/proc/self/fd/([0-9]+)/torch/__init__\.py", raw_file
    )
    bound_package_match = re.fullmatch(
        r"/proc/self/fd/([0-9]+)/torch", raw_locations[0]
    )
    if not (
        (raw_file == logical_origin and raw_locations[0] == logical_package)
        or (
            bound_origin_match is not None
            and bound_package_match is not None
            and bound_origin_match.group(1) == bound_package_match.group(1)
        )
    ):
        raise ValueError("imported Torch raw origin is not one exact locked form")
    try:
        origin_matches = os.path.samefile(origin, expected_init)
        package_matches = os.path.samefile(package_location, expected_package)
    except OSError as exc:
        raise ValueError(
            "imported Torch origin is not an existing locked path"
        ) from exc
    if not origin_matches or not package_matches:
        raise ValueError(
            "imported Torch resolved origin differs from locked distribution"
        )
    if getattr(imported_torch, "__version__", None) != torch_identity["module_version"]:
        raise ValueError("imported Torch version differs from python-runtime-lock.json")
    version_module = getattr(imported_torch, "version", None)
    if (
        getattr(version_module, "cuda", None) != torch_identity["cuda_version"]
        or getattr(version_module, "git_version", None) != torch_identity["git_version"]
    ):
        raise ValueError("imported Torch CUDA/git identity differs from runtime lock")
    with _AnchoredRoot(
        environment["prefix_path"], descriptor=None, label="environment prefix"
    ) as root:
        actual_sha256, actual_bytes, _ = root.hash_regular(
            torch_identity["module_init_path"],
            maximum_bytes=MAX_METADATA_BYTES,
            label="imported torch/__init__.py",
        )
        version_sha256, version_bytes, _ = root.hash_regular(
            torch_identity["version_file_path"],
            maximum_bytes=MAX_METADATA_BYTES,
            label="imported torch/version.py",
        )
    if (
        actual_sha256 != torch_identity["module_init_sha256"]
        or actual_bytes != torch_identity["module_init_bytes"]
        or version_sha256 != torch_identity["version_file_sha256"]
        or version_bytes != torch_identity["version_file_bytes"]
    ):
        raise ValueError("imported Torch origin changed after runtime-lock validation")
    return payload


def write_python_runtime_lock_exclusive(
    path: str | Path, payload: object
) -> tuple[str, int]:
    content = canonical_python_runtime_lock_bytes(payload)
    target = _absolute_no_expand(path)
    _assert_nonsymlink_chain(target.parent, label="runtime-lock output parent")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOINHERIT", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(target, flags, 0o600)
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError("short write while publishing Python runtime lock")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(content).hexdigest(), len(content)


def _cmd_build(args: argparse.Namespace) -> int:
    payload = build_python_runtime_lock(
        environment_prefix=args.environment_prefix,
        site_packages_path=args.site_packages_root,
    )
    content = canonical_python_runtime_lock_bytes(payload)
    if args.output is None:
        sys.stdout.buffer.write(content)
        sys.stdout.buffer.flush()
    else:
        digest, byte_count = write_python_runtime_lock_exclusive(args.output, payload)
        print(
            json.dumps(
                {
                    "status": "published_python_runtime_lock",
                    "path": str(_absolute_no_expand(args.output)),
                    "artifact_sha256": digest,
                    "artifact_bytes": byte_count,
                    "runtime_lock_id": payload["runtime_lock_id"],
                },
                sort_keys=True,
            )
        )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    artifact = read_python_runtime_lock_artifact(
        args.runtime_lock,
        expected_artifact_sha256=args.runtime_lock_sha256,
    )
    validate_live_python_runtime_lock(
        artifact,
        expected_artifact_sha256=args.runtime_lock_sha256,
    )
    print(
        json.dumps(
            {
                "status": "verified_live_python_runtime_lock",
                "artifact_sha256": artifact.artifact_sha256,
                "runtime_lock_id": artifact.payload["runtime_lock_id"],
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m afts_arc.m04a_python_runtime_lock")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser(
        "build", help="build python-runtime-lock.json without importing Torch"
    )
    build.add_argument("--environment-prefix")
    build.add_argument("--site-packages-root")
    build.add_argument("--output")
    build.set_defaults(handler=_cmd_build)
    verify = subparsers.add_parser(
        "verify", help="verify an external lock and the live Python environment"
    )
    verify.add_argument("--runtime-lock", required=True)
    verify.add_argument("--runtime-lock-sha256", required=True)
    verify.set_defaults(handler=_cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


__all__ = [
    "MAX_RUNTIME_LOCK_BYTES",
    "PYTHON_RUNTIME_LOCK_ALGORITHM_SCHEMA_VERSION",
    "PYTHON_RUNTIME_LOCK_FILENAME",
    "PYTHON_RUNTIME_LOCK_SCHEMA_VERSION",
    "PythonRuntimeLockArtifact",
    "build_python_runtime_lock",
    "canonical_distribution_name",
    "canonical_python_runtime_lock_bytes",
    "main",
    "read_python_runtime_lock_artifact",
    "validate_live_python_runtime_lock",
    "validate_imported_torch_runtime",
    "validate_python_runtime_lock_payload",
    "write_python_runtime_lock_exclusive",
]


if __name__ == "__main__":
    raise SystemExit(main())
