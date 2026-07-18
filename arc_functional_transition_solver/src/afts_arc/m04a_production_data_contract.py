"""Frozen byte commitments for the production M04a training data tree.

The diagnostic preflight intentionally sees no dataset files.  A formal training
campaign additionally receives one reviewed, read-only ``data`` directory.  This
module gives that directory a closed-world identity without widening the flat
17-file preflight input root or changing the launch-plan v0.2 schema.

The launch plan commits the reviewed runtime-source fingerprint and the reviewed
launcher bytes.  Both contain this exact table.  The launcher verifies the table
before its first GPU query; the campaign verifies it again before opening the mmap
readers.  The three artifact-manifest digests are then supplied to the existing
production readers, which replay their internal manifests and mmap contents.
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

from .m04a_contract import canonical_sha256


PRODUCTION_DATA_SCHEMA_VERSION = "afts-m04a-production-data-closure/v0.1"
PRODUCTION_DATA_FD_BASE = 300
SANITIZED_DIRECTORY = "sanitized_split"
TRAINING_CACHE_DIRECTORY = "rearc_train_cache"
VALIDATION_CACHE_DIRECTORY = "rearc_validation_cache"

SANITIZED_ARTIFACT_MANIFEST_SHA256 = (
    "35c69b8f8c077d6bae6d18f31910fdeaad621edaeb348a65a3e354bba6915ded"
)
TRAINING_CACHE_ARTIFACT_MANIFEST_SHA256 = (
    "8a952d832b4f2926f3ad46731734f29e6ad07ea4174226271698a3f7b3862ad4"
)
VALIDATION_CACHE_ARTIFACT_MANIFEST_SHA256 = (
    "aacac7f63877d21aeea9c58d4cb20bac6afee2710a920aa9b6f144112a8929f5"
)

_FILE_ROWS: dict[str, tuple[int, str]] = {
    "rearc_train_cache/artifact_manifest.json": (
        581,
        TRAINING_CACHE_ARTIFACT_MANIFEST_SHA256,
    ),
    "rearc_train_cache/cache_manifest.json": (
        9_486,
        "fe344217dcb69c4ddd2a0bc265f973a4c1c364420b3ab78237af1d410eef0e55",
    ),
    "rearc_train_cache/data.bin": (
        151_261_206,
        "297c18e1592d2947589ce87379636074fade30cc6cb9e84b4ead79671ae1accf",
    ),
    "rearc_train_cache/index.bin": (
        2_098_161,
        "a988ee1a7a29742d03627cc8b25ef77ffe202e63398289dff4c77e33e738e356",
    ),
    "rearc_validation_cache/artifact_manifest.json": (
        579,
        VALIDATION_CACHE_ARTIFACT_MANIFEST_SHA256,
    ),
    "rearc_validation_cache/cache_manifest.json": (
        2_739,
        "a307e17193866f4a3c4d731604a2b509256f072c6f18dd0ecb5281b43809b5ee",
    ),
    "rearc_validation_cache/data.bin": (
        29_730_899,
        "1595047815ab036e21f8037b870417d7f80f105d1f2efced6915e4691928d515",
    ),
    "rearc_validation_cache/index.bin": (
        408_473,
        "41f6ac79bbdae42d4284a99ded1e7c66453c6ad54e2bf2b4c9c5b2a8dcef104f",
    ),
    "sanitized_split/arc2_train.zip": (
        315_633,
        "a821a0810ee1e476216f3eff008063a938f1b5e8b6f5e0e72bca5e02b660597f",
    ),
    "sanitized_split/arc2_validation.zip": (
        64_999,
        "a65ed63789f929ac24078418f66d512b3f46b7d9ee322f3118d72b90e6c5d187",
    ),
    "sanitized_split/artifact_manifest.json": (
        923,
        SANITIZED_ARTIFACT_MANIFEST_SHA256,
    ),
    "sanitized_split/data_split_manifest.json": (
        346_867,
        "4e099a5590e6777e6896b0eb0704c38d1c6c7b8e4d44f615f4d2e07464ed57fd",
    ),
    "sanitized_split/rearc_train.zip": (
        26_567_832,
        "7738d77143efabf60b9252efc912e1b2ab3ebbddaa52d5903db9c202724607ab",
    ),
    "sanitized_split/rearc_validation.zip": (
        5_070_885,
        "ebaeb4f512f84b3fca8f379ca5eeb9e90650db855f01da3b841d711bdacbccd5",
    ),
}

PRODUCTION_DATA_FILES: Mapping[str, tuple[int, str]] = MappingProxyType(
    dict(sorted(_FILE_ROWS.items()))
)
PRODUCTION_DATA_TOTAL_BYTES = sum(row[0] for row in PRODUCTION_DATA_FILES.values())
PRODUCTION_DATA_FILE_COUNT = len(PRODUCTION_DATA_FILES)
_EXPECTED_DIRECTORIES = frozenset(
    {SANITIZED_DIRECTORY, TRAINING_CACHE_DIRECTORY, VALIDATION_CACHE_DIRECTORY}
)
_REPARSE_POINT_ATTRIBUTE = 0x400


def production_data_manifest_payload() -> dict[str, Any]:
    semantic: dict[str, Any] = {
        "schema": PRODUCTION_DATA_SCHEMA_VERSION,
        "file_count": PRODUCTION_DATA_FILE_COUNT,
        "total_bytes": PRODUCTION_DATA_TOTAL_BYTES,
        "sanitized_artifact_manifest_sha256": (
            SANITIZED_ARTIFACT_MANIFEST_SHA256
        ),
        "training_cache_artifact_manifest_sha256": (
            TRAINING_CACHE_ARTIFACT_MANIFEST_SHA256
        ),
        "validation_cache_artifact_manifest_sha256": (
            VALIDATION_CACHE_ARTIFACT_MANIFEST_SHA256
        ),
        "files": [
            {"path": path, "bytes": row[0], "sha256": row[1]}
            for path, row in PRODUCTION_DATA_FILES.items()
        ],
    }
    return {**semantic, "closure_id": canonical_sha256(semantic)}


PRODUCTION_DATA_CLOSURE_ID = str(production_data_manifest_payload()["closure_id"])


def _is_link_or_reparse(path: Path) -> bool:
    information = path.lstat()
    return stat.S_ISLNK(information.st_mode) or bool(
        getattr(information, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE
    )


def _absolute_lexical(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _assert_existing_nonsymlink_chain(path: Path) -> None:
    absolute = _absolute_lexical(path)
    current = Path(absolute.anchor)
    if not current.exists() or _is_link_or_reparse(current):
        raise ValueError("production data root anchor is unavailable or linked")
    for component in absolute.parts[1:]:
        current = current / component
        if not os.path.lexists(current):
            raise FileNotFoundError(current)
        if _is_link_or_reparse(current):
            raise ValueError("production data path traverses a link or reparse point")


def _read_and_hash_regular(path: Path, *, expected_bytes: int) -> str:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOINHERIT", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
            raise ValueError("production data file byte count differs from contract")
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(4 * 1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError("production data file became short while hashing")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError("production data file grew while hashing")
        after = os.fstat(descriptor)
        if any(
            getattr(before, field) != getattr(after, field)
            for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        ):
            raise RuntimeError("production data file changed while hashing")
        path_stat = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(path_stat.st_mode) or not os.path.samestat(
            after, path_stat
        ):
            raise RuntimeError("production data pathname identity changed")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ProductionDataBinding:
    root: Path
    sanitized_bundle_dir: Path
    training_rearc_cache_dir: Path
    validation_rearc_cache_dir: Path
    closure_id: str
    sealed_descriptor_rows: tuple[tuple[str, int, int, int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.closure_id != PRODUCTION_DATA_CLOSURE_ID:
            raise ValueError("production data binding has another closure identity")
        expected = {
            "sanitized_bundle_dir": self.root / SANITIZED_DIRECTORY,
            "training_rearc_cache_dir": self.root / TRAINING_CACHE_DIRECTORY,
            "validation_rearc_cache_dir": self.root / VALIDATION_CACHE_DIRECTORY,
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise ValueError(f"production data binding {field} escaped its root")
        if self.sealed_descriptor_rows:
            expected_paths = tuple(sorted(PRODUCTION_DATA_FILES))
            observed_paths = tuple(row[0] for row in self.sealed_descriptor_rows)
            observed_descriptors = tuple(row[1] for row in self.sealed_descriptor_rows)
            expected_descriptors = tuple(
                PRODUCTION_DATA_FD_BASE + index
                for index in range(len(expected_paths))
            )
            if observed_paths != expected_paths:
                raise ValueError("sealed production data paths are not canonical")
            if observed_descriptors != expected_descriptors:
                raise ValueError("sealed production data descriptors are not canonical")
            if any(
                type(device) is not int
                or type(inode) is not int
                or type(byte_count) is not int
                or byte_count != PRODUCTION_DATA_FILES[path][0]
                for path, _, device, inode, byte_count in self.sealed_descriptor_rows
            ):
                raise ValueError("sealed production data identities are malformed")

    @property
    def is_sealed_snapshot(self) -> bool:
        return bool(self.sealed_descriptor_rows)

    def artifact_source(self, relative: str) -> Path | int:
        if relative not in PRODUCTION_DATA_FILES:
            raise KeyError(relative)
        if self.sealed_descriptor_rows:
            for path, descriptor, _, _, _ in self.sealed_descriptor_rows:
                if path == relative:
                    return descriptor
            raise RuntimeError("sealed production data binding lost an artifact")
        return self.root.joinpath(*PurePosixPath(relative).parts)

    def directory_artifacts(self, directory: str) -> Mapping[str, Path | int]:
        if directory not in _EXPECTED_DIRECTORIES:
            raise KeyError(directory)
        return MappingProxyType(
            {
                PurePosixPath(relative).name: self.artifact_source(relative)
                for relative in PRODUCTION_DATA_FILES
                if PurePosixPath(relative).parent.as_posix() == directory
            }
        )


def _required_memfd_seals() -> int:
    if os.name != "posix":
        raise RuntimeError("sealed production data requires POSIX")
    import fcntl

    return (
        fcntl.F_SEAL_SEAL
        | fcntl.F_SEAL_SHRINK
        | fcntl.F_SEAL_GROW
        | fcntl.F_SEAL_WRITE
    )


def _assert_sealed_descriptor(
    descriptor: int,
    *,
    expected_bytes: int,
    expected_identity: tuple[int, int] | None = None,
) -> os.stat_result:
    import fcntl

    if type(descriptor) is not int or descriptor < 3:
        raise ValueError("production data descriptor is invalid")
    information = os.fstat(descriptor)
    if not stat.S_ISREG(information.st_mode) or information.st_size != expected_bytes:
        raise ValueError("sealed production data descriptor has wrong size or type")
    if expected_identity is not None and (
        information.st_dev,
        information.st_ino,
    ) != expected_identity:
        raise RuntimeError("sealed production data descriptor identity changed")
    if not os.get_inheritable(descriptor):
        raise RuntimeError("sealed production data descriptor was not inherited")
    if fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) != _required_memfd_seals():
        raise RuntimeError("production data descriptor is not exactly sealed")
    return information


def _sha256_pread(descriptor: int, *, expected_bytes: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < expected_bytes:
        block = os.pread(descriptor, min(4 * 1024 * 1024, expected_bytes - offset), offset)
        if not block:
            raise RuntimeError("sealed production data descriptor became short")
        digest.update(block)
        offset += len(block)
    if os.pread(descriptor, 1, expected_bytes):
        raise RuntimeError("sealed production data descriptor grew")
    return digest.hexdigest()


def validate_sealed_production_data_snapshot(
    root: str | Path,
) -> ProductionDataBinding:
    """Bind the fixed inherited sealed-memfd table used by formal training."""

    absolute = _absolute_lexical(root)
    _assert_existing_nonsymlink_chain(absolute)
    if not absolute.is_dir():
        raise ValueError("production data logical root must be an existing directory")
    rows: list[tuple[str, int, int, int, int]] = []
    for index, (relative, (expected_bytes, expected_sha256)) in enumerate(
        PRODUCTION_DATA_FILES.items()
    ):
        descriptor = PRODUCTION_DATA_FD_BASE + index
        information = _assert_sealed_descriptor(
            descriptor, expected_bytes=expected_bytes
        )
        if _sha256_pread(descriptor, expected_bytes=expected_bytes) != expected_sha256:
            raise ValueError(f"sealed production data SHA-256 mismatch: {relative}")
        after = _assert_sealed_descriptor(
            descriptor,
            expected_bytes=expected_bytes,
            expected_identity=(information.st_dev, information.st_ino),
        )
        rows.append(
            (
                relative,
                descriptor,
                after.st_dev,
                after.st_ino,
                after.st_size,
            )
        )
    return ProductionDataBinding(
        root=absolute,
        sanitized_bundle_dir=absolute / SANITIZED_DIRECTORY,
        training_rearc_cache_dir=absolute / TRAINING_CACHE_DIRECTORY,
        validation_rearc_cache_dir=absolute / VALIDATION_CACHE_DIRECTORY,
        closure_id=PRODUCTION_DATA_CLOSURE_ID,
        sealed_descriptor_rows=tuple(rows),
    )


def revalidate_production_data_binding(
    binding: ProductionDataBinding,
) -> ProductionDataBinding:
    """Fail closed if a previously bound filesystem or sealed view drifted."""

    if type(binding) is not ProductionDataBinding:
        raise TypeError("production data binding must be exact")
    if not binding.is_sealed_snapshot:
        replay = validate_production_data_root(binding.root)
        if replay != binding:
            raise RuntimeError("production data filesystem binding changed")
        return binding
    for relative, descriptor, device, inode, expected_bytes in (
        binding.sealed_descriptor_rows
    ):
        if expected_bytes != PRODUCTION_DATA_FILES[relative][0]:
            raise RuntimeError("sealed production data byte contract drifted")
        _assert_sealed_descriptor(
            descriptor,
            expected_bytes=expected_bytes,
            expected_identity=(device, inode),
        )
    return binding


def validate_production_data_root(root: str | Path) -> ProductionDataBinding:
    """Hash and bind the exact production data tree without following links."""

    absolute = _absolute_lexical(root)
    _assert_existing_nonsymlink_chain(absolute)
    if not absolute.is_dir():
        raise ValueError("production data root must be an existing directory")
    root_before = absolute.stat(follow_symlinks=False)
    if not stat.S_ISDIR(root_before.st_mode):
        raise ValueError("production data root must be a real directory")

    actual_directories: set[str] = set()
    actual_files: set[str] = set()
    for entry in os.scandir(absolute):
        entry_path = absolute / entry.name
        if entry.is_symlink() or _is_link_or_reparse(entry_path):
            raise ValueError("production data root contains a linked entry")
        if not entry.is_dir(follow_symlinks=False):
            raise ValueError("production data root may contain only fixed directories")
        actual_directories.add(entry.name)
        directory_before = entry_path.stat(follow_symlinks=False)
        for child in os.scandir(entry_path):
            child_path = entry_path / child.name
            if (
                child.is_symlink()
                or _is_link_or_reparse(child_path)
                or not child.is_file(follow_symlinks=False)
            ):
                raise ValueError("production data directories require regular files")
            relative = PurePosixPath(entry.name, child.name).as_posix()
            actual_files.add(relative)
        directory_after = entry_path.stat(follow_symlinks=False)
        if not os.path.samestat(directory_before, directory_after):
            raise RuntimeError("production data directory identity changed")
    if actual_directories != _EXPECTED_DIRECTORIES:
        raise ValueError("production data directories do not match the frozen set")
    if actual_files != set(PRODUCTION_DATA_FILES):
        raise ValueError("production data files do not match the frozen closed world")

    for relative, (expected_bytes, expected_sha256) in PRODUCTION_DATA_FILES.items():
        path = absolute.joinpath(*PurePosixPath(relative).parts)
        actual_sha256 = _read_and_hash_regular(path, expected_bytes=expected_bytes)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"production data SHA-256 mismatch: {relative}")
    root_after = absolute.stat(follow_symlinks=False)
    if not os.path.samestat(root_before, root_after):
        raise RuntimeError("production data root identity changed while binding")
    return ProductionDataBinding(
        root=absolute,
        sanitized_bundle_dir=absolute / SANITIZED_DIRECTORY,
        training_rearc_cache_dir=absolute / TRAINING_CACHE_DIRECTORY,
        validation_rearc_cache_dir=absolute / VALIDATION_CACHE_DIRECTORY,
        closure_id=PRODUCTION_DATA_CLOSURE_ID,
    )


__all__ = [
    "PRODUCTION_DATA_CLOSURE_ID",
    "PRODUCTION_DATA_FD_BASE",
    "PRODUCTION_DATA_FILE_COUNT",
    "PRODUCTION_DATA_FILES",
    "PRODUCTION_DATA_SCHEMA_VERSION",
    "PRODUCTION_DATA_TOTAL_BYTES",
    "ProductionDataBinding",
    "SANITIZED_ARTIFACT_MANIFEST_SHA256",
    "SANITIZED_DIRECTORY",
    "TRAINING_CACHE_ARTIFACT_MANIFEST_SHA256",
    "TRAINING_CACHE_DIRECTORY",
    "VALIDATION_CACHE_ARTIFACT_MANIFEST_SHA256",
    "VALIDATION_CACHE_DIRECTORY",
    "production_data_manifest_payload",
    "revalidate_production_data_binding",
    "validate_sealed_production_data_snapshot",
    "validate_production_data_root",
]
