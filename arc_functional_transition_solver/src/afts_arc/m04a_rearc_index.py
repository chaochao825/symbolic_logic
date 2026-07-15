"""Deterministic mmap cache derived only from sanitized M04a ReARC shards."""

from __future__ import annotations

import hashlib
import ctypes
import errno
import io
import json
import mmap
import os
import re
import stat
import struct
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping, Sequence

from .m04a_data import (
    ArtifactSource,
    M04AExample,
    ReARCExampleSource,
    load_sanitized_m04a_data,
)
from .manifest import serialize_json


CACHE_SEMANTICS_VERSION = "afts-m04a-rearc-mmap-cache/v0.1"
CACHE_MANIFEST_SCHEMA = "afts.m04a-rearc-mmap-cache-manifest/v1"
SANITIZED_SPLIT_SCHEMA = "afts.m04a-sanitized-split/v1"
DATA_MAGIC = b"AFTSRD1\0"
INDEX_MAGIC = b"AFTSRI1\0"
EXAMPLES_PER_PARENT = 1000
_ARTIFACT_NAMES = {
    "arc2_train.zip",
    "arc2_validation.zip",
    "rearc_train.zip",
    "rearc_validation.zip",
    "data_split_manifest.json",
}
_CACHE_NAMES = {
    "artifact_manifest.json",
    "cache_manifest.json",
    "data.bin",
    "index.bin",
}
_HEX8_RE = re.compile(r"[0-9a-f]{8}")
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_REARC_ZIP_BYTES = 1024 * 1024 * 1024
_MAX_REARC_MEMBER_BYTES = 16 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 1000


def _resolve_without_symlinks(
    value: str | Path, *, must_exist: bool
) -> Path:
    absolute = Path(os.path.abspath(os.fspath(Path(value).expanduser())))
    cursor = absolute if absolute.exists() else absolute.parent
    while True:
        try:
            metadata = os.lstat(cursor)
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"symlinked path component is forbidden: {cursor}")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    return absolute.resolve(strict=must_exist)


def _artifact_name(source: ArtifactSource) -> str:
    if type(source) is int:
        return f"sealed-production-data-fd-{source}"
    return Path(source).name


def _open_regular_fd(path: ArtifactSource) -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOINHERIT", 0)
    if type(path) is int:
        if path < 3:
            raise ValueError("sealed artifact descriptor must be >= 3")
        descriptor = os.open(f"/proc/self/fd/{path}", flags)
    else:
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"artifact is not a regular file: {_artifact_name(path)}")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _sha256_fd(descriptor: int, *, maximum_bytes: int | None = None) -> tuple[str, int]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    count = 0
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        count += len(block)
        if maximum_bytes is not None and count > maximum_bytes:
            raise ValueError("artifact exceeds its byte limit")
        digest.update(block)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest(), count


def _read_regular_bytes(path: ArtifactSource, *, maximum_bytes: int) -> bytes:
    descriptor = _open_regular_fd(path)
    try:
        metadata = os.fstat(descriptor)
        if metadata.st_size > maximum_bytes:
            raise ValueError(
                f"artifact exceeds its byte limit: {_artifact_name(path)}"
            )
        chunks: list[bytes] = []
        count = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - count))
            if not block:
                break
            chunks.append(block)
            count += len(block)
            if count > maximum_bytes:
                raise ValueError(
                    f"artifact exceeds its byte limit: {_artifact_name(path)}"
                )
        if os.fstat(descriptor).st_size != count:
            raise ValueError(
                f"artifact changed while being read: {_artifact_name(path)}"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _open_verified_mmap(
    path: ArtifactSource, *, expected_sha256: str, expected_bytes: int
) -> tuple[int, mmap.mmap]:
    descriptor = _open_regular_fd(path)
    try:
        before = os.fstat(descriptor)
        actual_sha256, actual_bytes = _sha256_fd(descriptor)
        after = os.fstat(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise ValueError(
                f"artifact changed while being hashed: {_artifact_name(path)}"
            )
        if actual_sha256 != expected_sha256 or actual_bytes != expected_bytes:
            raise ValueError(
                f"artifact hash/size mismatch: {_artifact_name(path)}"
            )
        mapped = mmap.mmap(descriptor, 0, access=mmap.ACCESS_READ)
        return descriptor, mapped
    except Exception:
        os.close(descriptor)
        raise


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_no_replace(source: Path, target: Path) -> None:
    if os.name == "posix":
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("renameat2(RENAME_NOREPLACE) is required")
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
                raise FileExistsError(f"refusing to overwrite ReARC mmap cache: {target}")
            raise OSError(error, os.strerror(error), str(target))
        return
    os.rename(source, target)


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


def _sha256_file(path: ArtifactSource) -> tuple[str, int]:
    descriptor = _open_regular_fd(path)
    try:
        return _sha256_fd(descriptor)
    finally:
        os.close(descriptor)


def _strict_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be lowercase SHA-256")
    return value


def _strict_uint(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _closed_world_files(root: Path, expected: set[str]) -> None:
    try:
        root_metadata = os.lstat(root)
    except FileNotFoundError as exc:
        raise ValueError(f"directory not found: {root}") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError(f"directory not found: {root}")
    observed: set[str] = set()
    for path in root.iterdir():
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"symlinked artifact is forbidden: {path.name}")
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"non-regular artifact is forbidden: {path.name}")
        observed.add(path.name)
    if observed != expected:
        raise ValueError(
            f"closed-world file set mismatch: expected={sorted(expected)}, observed={sorted(observed)}"
        )


def _validate_artifact_manifest(
    root: Path,
    *,
    expected_artifacts: set[str],
    artifact_sources: Mapping[str, ArtifactSource] | None = None,
) -> tuple[dict[str, Any], str]:
    if artifact_sources is not None and set(artifact_sources) != {
        "artifact_manifest.json",
        *expected_artifacts,
    }:
        raise ValueError("sealed artifact-manifest sources are not closed-world")

    def source(name: str) -> ArtifactSource:
        if artifact_sources is None:
            return root / name
        return artifact_sources[name]

    manifest_bytes = _read_regular_bytes(
        source("artifact_manifest.json"), maximum_bytes=_MAX_JSON_BYTES
    )
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    payload = _strict_json(manifest_bytes, label="artifact_manifest.json")
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "bundle_status",
        "run_id",
        "artifacts",
    }:
        raise ValueError("artifact manifest has an invalid schema")
    if payload["schema_version"] != 1 or payload["bundle_status"] != "complete":
        raise ValueError("artifact manifest is not a complete v1 bundle")
    if not isinstance(payload["run_id"], str) or not payload["run_id"]:
        raise ValueError("artifact manifest run_id must be non-empty")
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise ValueError("artifact manifest artifact set mismatch")
    for name in sorted(expected_artifacts):
        row = artifacts[name]
        if not isinstance(row, dict) or set(row) != {"sha256", "bytes", "rows"}:
            raise ValueError(f"invalid artifact metadata for {name}")
        expected_sha = _strict_sha(row["sha256"], field=f"{name}.sha256")
        expected_bytes = _strict_uint(row["bytes"], field=f"{name}.bytes")
        actual_sha, actual_bytes = _sha256_file(source(name))
        if actual_sha != expected_sha or actual_bytes != expected_bytes:
            raise ValueError(f"artifact hash/size mismatch: {name}")
    return payload, manifest_sha


def _validate_zip_info(info: zipfile.ZipInfo, *, expected_name: str) -> None:
    name = info.filename
    path = PurePosixPath(name)
    if (
        name != expected_name
        or "\\" in name
        or ":" in name
        or path.is_absolute()
        or path.as_posix() != name
        or any(part in {"", ".", ".."} for part in path.parts)
        or info.is_dir()
        or stat.S_ISLNK(info.external_attr >> 16)
        or info.flag_bits & 0x1
        or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
        or info.file_size <= 0
        or info.file_size > _MAX_REARC_MEMBER_BYTES
        or info.compress_size <= 0
        or info.file_size > info.compress_size * _MAX_COMPRESSION_RATIO
    ):
        raise ValueError(f"unsafe or unexpected sanitized ReARC member: {name!r}")


def _grid(value: object, *, label: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list) or not value or len(value) > 30:
        raise ValueError(f"{label} must have 1-30 rows")
    width: int | None = None
    rows: list[tuple[int, ...]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or not row or len(row) > 30:
            raise ValueError(f"{label}[{row_index}] must have 1-30 cells")
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


def _examples(content: bytes, *, label: str) -> tuple[M04AExample, ...]:
    payload = _strict_json(content, label=label)
    if not isinstance(payload, list) or len(payload) != EXAMPLES_PER_PARENT:
        raise ValueError(f"{label} must contain exactly {EXAMPLES_PER_PARENT} examples")
    result: list[M04AExample] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or set(item) != {"input", "output"}:
            raise ValueError(f"{label}[{index}] must contain exactly input and output")
        result.append(
            M04AExample.create(
                _grid(item["input"], label=f"{label}[{index}].input"),
                _grid(item["output"], label=f"{label}[{index}].output"),
            )
        )
    return tuple(result)


def _load_sanitized_shard(
    sanitized_root: Path,
    *,
    fold: str,
    expected_artifact_manifest_sha256: str,
) -> tuple[bytes, tuple[str, ...], Mapping[str, Mapping[str, object]], str, dict[str, Any]]:
    _closed_world_files(
        sanitized_root,
        {"artifact_manifest.json", *_ARTIFACT_NAMES},
    )
    _, sanitized_artifact_sha = _validate_artifact_manifest(
        sanitized_root,
        expected_artifacts=_ARTIFACT_NAMES,
    )
    if sanitized_artifact_sha != _strict_sha(
        expected_artifact_manifest_sha256,
        field="expected_artifact_manifest_sha256",
    ):
        raise ValueError("sanitized artifact-manifest external commitment mismatch")
    split_bytes = _read_regular_bytes(
        sanitized_root / "data_split_manifest.json", maximum_bytes=_MAX_JSON_BYTES
    )
    split = _strict_json(split_bytes, label="data_split_manifest.json")
    if not isinstance(split, dict) or split.get("schema") != SANITIZED_SPLIT_SCHEMA:
        raise ValueError("unsupported sanitized split manifest")
    if split.get("protected_payloads_absent") is not True:
        raise ValueError("sanitized split does not assert protected payload absence")
    allowlists = split.get("ordered_allowlists")
    shards = split.get("shards")
    if not isinstance(allowlists, dict) or not isinstance(shards, dict):
        raise ValueError("sanitized split is missing allowlists or shards")
    allowlist_key = f"rearc_{fold}"
    artifact_name = f"rearc_{fold}.zip"
    raw_ids = allowlists.get(allowlist_key)
    shard = shards.get(artifact_name)
    if not isinstance(raw_ids, list) or not isinstance(shard, dict):
        raise ValueError(f"sanitized split is missing {allowlist_key}")
    parent_ids = tuple(raw_ids)
    if (
        not parent_ids
        or parent_ids != tuple(sorted(parent_ids))
        or len(parent_ids) != len(set(parent_ids))
        or any(not isinstance(item, str) or _HEX8_RE.fullmatch(item) is None for item in parent_ids)
    ):
        raise ValueError("sanitized ReARC parent allowlist must be non-empty, sorted, unique 8-hex")
    if shard.get("source") != "rearc" or shard.get("fold") != fold:
        raise ValueError("sanitized ReARC shard source/fold mismatch")
    if shard.get("member_prefix") != f"rearc/{fold}/":
        raise ValueError("sanitized ReARC shard member prefix mismatch")
    if shard.get("member_count") != len(parent_ids):
        raise ValueError("sanitized ReARC shard member count mismatch")
    if shard.get("ordered_source_parent_ids") != list(parent_ids):
        raise ValueError("sanitized ReARC shard parent order mismatch")
    members = shard.get("members")
    if not isinstance(members, list) or len(members) != len(parent_ids):
        raise ValueError("sanitized ReARC shard member metadata mismatch")
    member_rows: dict[str, Mapping[str, object]] = {}
    for source_id, row in zip(parent_ids, members):
        if not isinstance(row, dict):
            raise ValueError("sanitized ReARC member metadata must be an object")
        expected_path = f"rearc/{fold}/{source_id}.json"
        if (
            row.get("path") != expected_path
            or row.get("source_parent_id") != source_id
            or not isinstance(row.get("semantic_parent_id"), str)
            or _HEX8_RE.fullmatch(row["semantic_parent_id"]) is None
        ):
            raise ValueError("sanitized ReARC member metadata identity mismatch")
        _strict_sha(row.get("sha256"), field=f"{source_id}.sha256")
        _strict_uint(row.get("bytes"), field=f"{source_id}.bytes", minimum=1)
        member_rows[source_id] = row
    zip_bytes = _read_regular_bytes(
        sanitized_root / artifact_name, maximum_bytes=_MAX_REARC_ZIP_BYTES
    )
    if hashlib.sha256(zip_bytes).hexdigest() != _strict_sha(
        shard.get("zip_sha256"), field=f"{artifact_name}.zip_sha256"
    ):
        raise ValueError("sanitized ReARC ZIP hash differs from split manifest")
    if len(zip_bytes) != _strict_uint(
        shard.get("zip_bytes"), field=f"{artifact_name}.zip_bytes", minimum=1
    ):
        raise ValueError("sanitized ReARC ZIP size differs from split manifest")
    return zip_bytes, parent_ids, member_rows, sanitized_artifact_sha, split


def _write_grid(handle: BinaryIO, grid: Sequence[Sequence[int]]) -> None:
    height = len(grid)
    width = len(grid[0])
    handle.write(struct.pack("<BB", height, width))
    handle.write(bytes(color for row in grid for color in row))


def _data_header() -> bytes:
    schema = CACHE_SEMANTICS_VERSION.encode("ascii")
    return DATA_MAGIC + struct.pack("<H", len(schema)) + schema


def _index_prefix(*, parent_count: int, data_size: int) -> bytes:
    schema = CACHE_SEMANTICS_VERSION.encode("ascii")
    return (
        INDEX_MAGIC
        + struct.pack("<H", len(schema))
        + schema
        + struct.pack("<IIQ", parent_count, EXAMPLES_PER_PARENT, data_size)
    )


def _artifact_manifest_payload(
    artifacts: Mapping[str, tuple[str, int]], *, run_id: str
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "bundle_status": "complete",
        "run_id": run_id,
        "artifacts": {
            name: {
                "sha256": metadata[0],
                "bytes": metadata[1],
                "rows": None,
            }
            for name, metadata in sorted(artifacts.items())
        },
    }


def _publish_cache(
    target: Path,
    *,
    data_staging: Path,
    index_bytes: bytes,
    cache_manifest_bytes: bytes,
    data_sha256: str,
    data_bytes: int,
    run_id: str,
) -> tuple[dict[str, object], str]:
    index_path = target / "index.bin"
    cache_manifest_path = target / "cache_manifest.json"
    with index_path.open("xb") as handle:
        handle.write(index_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    with cache_manifest_path.open("xb") as handle:
        handle.write(cache_manifest_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    if data_staging != target / "data.bin":
        raise RuntimeError("data staging path mismatch")
    artifacts = {
        "data.bin": (data_sha256, data_bytes),
        "index.bin": (hashlib.sha256(index_bytes).hexdigest(), len(index_bytes)),
        "cache_manifest.json": (
            hashlib.sha256(cache_manifest_bytes).hexdigest(),
            len(cache_manifest_bytes),
        ),
    }
    manifest = _artifact_manifest_payload(artifacts, run_id=run_id)
    manifest_bytes = serialize_json(manifest)
    with (target / "artifact_manifest.json").open("xb") as handle:
        handle.write(manifest_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(target)
    return manifest, hashlib.sha256(manifest_bytes).hexdigest()


def build_rearc_mmap_cache(
    *,
    sanitized_split_dir: str | Path,
    fold: str,
    output_dir: str | Path,
    expected_sanitized_artifact_manifest_sha256: str,
    allow_nonproduction_fixture: bool = False,
) -> dict[str, object]:
    """Build one train or validation mmap cache from a sanitized ReARC shard."""

    if fold not in {"train", "validation"}:
        raise ValueError("fold must be train or validation")
    if type(allow_nonproduction_fixture) is not bool:
        raise TypeError("allow_nonproduction_fixture must be bool")
    source = _resolve_without_symlinks(sanitized_split_dir, must_exist=True)
    target = _resolve_without_symlinks(output_dir, must_exist=False)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite ReARC mmap cache: {target}")
    if target == source or target.is_relative_to(source):
        raise ValueError("cache output must be physically outside the sanitized split root")

    expected_sanitized_sha = _strict_sha(
        expected_sanitized_artifact_manifest_sha256,
        field="expected_sanitized_artifact_manifest_sha256",
    )
    if not allow_nonproduction_fixture:
        load_sanitized_m04a_data(
            source,
            require_production_counts=True,
            expected_artifact_manifest_sha256=expected_sanitized_sha,
        )
    zip_bytes, parent_ids, member_rows, sanitized_artifact_sha, _ = _load_sanitized_shard(
        source,
        fold=fold,
        expected_artifact_manifest_sha256=expected_sanitized_sha,
    )
    semantic_parent_ids = tuple(
        str(member_rows[parent_id]["semantic_parent_id"]) for parent_id in parent_ids
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(exist_ok=False)
    offsets: list[int] = []
    data_path = staging / "data.bin"
    try:
        with data_path.open("xb") as data_handle, zipfile.ZipFile(
            io.BytesIO(zip_bytes), mode="r"
        ) as archive:
            header = _data_header()
            data_handle.write(header)
            offsets.append(len(header))
            infos = archive.infolist()
            if len(infos) != len(parent_ids):
                raise ValueError("sanitized ReARC ZIP member count mismatch")
            info_by_name: dict[str, zipfile.ZipInfo] = {}
            for info in infos:
                if info.filename in info_by_name:
                    raise ValueError(f"duplicate sanitized ReARC ZIP member: {info.filename}")
                info_by_name[info.filename] = info
            for source_id in parent_ids:
                expected_name = f"rearc/{fold}/{source_id}.json"
                info = info_by_name.get(expected_name)
                if info is None:
                    raise ValueError(f"sanitized ReARC ZIP missing member: {expected_name}")
                _validate_zip_info(info, expected_name=expected_name)
                content = archive.read(info)
                metadata = member_rows[source_id]
                if (
                    hashlib.sha256(content).hexdigest() != metadata["sha256"]
                    or len(content) != metadata["bytes"]
                ):
                    raise ValueError(f"sanitized ReARC member hash/size mismatch: {source_id}")
                for example in _examples(content, label=expected_name):
                    _write_grid(data_handle, example.input_grid)
                    _write_grid(data_handle, example.output_grid)
                    offsets.append(data_handle.tell())
            if set(info_by_name) != {
                f"rearc/{fold}/{source_id}.json" for source_id in parent_ids
            }:
                raise ValueError("sanitized ReARC ZIP contains an extra member")
            data_handle.flush()
            os.fsync(data_handle.fileno())

        data_sha, data_size = _sha256_file(data_path)
        expected_offset_count = len(parent_ids) * EXAMPLES_PER_PARENT + 1
        if len(offsets) != expected_offset_count or offsets[-1] != data_size:
            raise RuntimeError("derived ReARC data offsets do not close")
        prefix = _index_prefix(parent_count=len(parent_ids), data_size=data_size)
        ids_blob = b"".join(parent_id.encode("ascii") for parent_id in parent_ids)
        offsets_blob = b"".join(struct.pack("<Q", offset) for offset in offsets)
        index_bytes = prefix + ids_blob + offsets_blob
        index_sha = hashlib.sha256(index_bytes).hexdigest()
        input_zip_name = f"rearc_{fold}.zip"
        cache_manifest = {
            "schema": CACHE_MANIFEST_SCHEMA,
            "cache_semantics_version": CACHE_SEMANTICS_VERSION,
            "fold": fold,
            "production_contract": not allow_nonproduction_fixture,
            "examples_per_parent": EXAMPLES_PER_PARENT,
            "parent_count": len(parent_ids),
            "ordered_parent_ids": list(parent_ids),
            "ordered_semantic_parent_ids": list(semantic_parent_ids),
            "total_example_count": len(parent_ids) * EXAMPLES_PER_PARENT,
            "input_sanitized_artifact_manifest_sha256": sanitized_artifact_sha,
            "input_shard_artifact_name": input_zip_name,
            "input_shard_sha256": hashlib.sha256(zip_bytes).hexdigest(),
            "data": {
                "path": "data.bin",
                "magic_hex": DATA_MAGIC.hex(),
                "bytes": data_size,
                "sha256": data_sha,
            },
            "index": {
                "path": "index.bin",
                "magic_hex": INDEX_MAGIC.hex(),
                "bytes": len(index_bytes),
                "sha256": index_sha,
                "offset_encoding": "little_endian_uint64_absolute",
                "offset_count": len(offsets),
            },
            "grid_encoding": "uint8_height_width_then_row_major_uint8_colors",
        }
        cache_manifest_bytes = serialize_json(cache_manifest)
        run_id = hashlib.sha256(
            b"afts-m04a-rearc-mmap-cache-bundle/v0.1\0"
            + hashlib.sha256(cache_manifest_bytes).digest()
            + bytes.fromhex(data_sha)
            + bytes.fromhex(index_sha)
        ).hexdigest()[:20]
        artifact_manifest, artifact_manifest_sha = _publish_cache(
            staging,
            data_staging=data_path,
            index_bytes=index_bytes,
            cache_manifest_bytes=cache_manifest_bytes,
            data_sha256=data_sha,
            data_bytes=data_size,
            run_id=run_id,
        )
        _closed_world_files(staging, _CACHE_NAMES)
        _validate_artifact_manifest(
            staging,
            expected_artifacts={"cache_manifest.json", "data.bin", "index.bin"},
        )
        if not allow_nonproduction_fixture:
            for path in staging.iterdir():
                path.chmod(0o444)
            staging.chmod(0o555)
        _rename_directory_no_replace(staging, target)
        _fsync_directory(target.parent)
        return {
            "status": "published_rearc_mmap_cache",
            "output_dir": str(target),
            "fold": fold,
            "parent_count": len(parent_ids),
            "total_example_count": len(parent_ids) * EXAMPLES_PER_PARENT,
            "input_sanitized_artifact_manifest_sha256": sanitized_artifact_sha,
            "artifact_manifest_sha256": artifact_manifest_sha,
            "artifact_manifest": artifact_manifest,
        }
    except Exception as exc:
        raise RuntimeError(f"ReARC mmap cache build failed; preserved at {staging}: {exc}") from exc


def _parse_data_header(data: mmap.mmap) -> int:
    if len(data) < len(DATA_MAGIC) + 2 or data[: len(DATA_MAGIC)] != DATA_MAGIC:
        raise ValueError("invalid ReARC data magic")
    schema_length = struct.unpack_from("<H", data, len(DATA_MAGIC))[0]
    end = len(DATA_MAGIC) + 2 + schema_length
    if end > len(data) or data[len(DATA_MAGIC) + 2 : end] != CACHE_SEMANTICS_VERSION.encode(
        "ascii"
    ):
        raise ValueError("invalid ReARC data schema")
    return end


def _parse_index_header(index: mmap.mmap) -> tuple[int, int, int, int]:
    if len(index) < len(INDEX_MAGIC) + 2 or index[: len(INDEX_MAGIC)] != INDEX_MAGIC:
        raise ValueError("invalid ReARC index magic")
    schema_length = struct.unpack_from("<H", index, len(INDEX_MAGIC))[0]
    cursor = len(INDEX_MAGIC) + 2
    schema_end = cursor + schema_length
    if schema_end + struct.calcsize("<IIQ") > len(index):
        raise ValueError("truncated ReARC index header")
    if index[cursor:schema_end] != CACHE_SEMANTICS_VERSION.encode("ascii"):
        raise ValueError("invalid ReARC index schema")
    parent_count, examples_per_parent, data_size = struct.unpack_from(
        "<IIQ", index, schema_end
    )
    return schema_end + struct.calcsize("<IIQ"), parent_count, examples_per_parent, data_size


def _record_end(data: mmap.mmap, start: int) -> int:
    cursor = start
    for _ in range(2):
        if cursor + 2 > len(data):
            raise ValueError("truncated ReARC grid header")
        height, width = struct.unpack_from("<BB", data, cursor)
        if not 1 <= height <= 30 or not 1 <= width <= 30:
            raise ValueError("invalid ReARC cached grid dimensions")
        payload_start = cursor + 2
        cursor = payload_start + height * width
        if cursor > len(data):
            raise ValueError("truncated ReARC cached grid payload")
        if any(color > 9 for color in data[payload_start:cursor]):
            raise ValueError("cached ReARC grid contains a color outside 0-9")
    return cursor


def _decode_grid(data: mmap.mmap, cursor: int) -> tuple[tuple[tuple[int, ...], ...], int]:
    height, width = struct.unpack_from("<BB", data, cursor)
    cursor += 2
    end = cursor + height * width
    payload = data[cursor:end]
    if any(color > 9 for color in payload):
        raise ValueError("cached ReARC grid contains a color outside 0-9")
    rows = tuple(
        tuple(payload[row * width : (row + 1) * width]) for row in range(height)
    )
    return rows, end


class MMapReARCExampleSource(ReARCExampleSource):
    """Read-only, O(1)-indexed ReARC example source backed by two mmaps."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        expected_artifact_manifest_sha256: str,
        expected_sanitized_artifact_manifest_sha256: str,
        allow_nonproduction_fixture: bool = False,
        artifact_sources: Mapping[str, ArtifactSource] | None = None,
    ) -> None:
        if artifact_sources is None:
            self._root = _resolve_without_symlinks(cache_dir, must_exist=True)
            sources: dict[str, ArtifactSource] | None = None
        else:
            if set(artifact_sources) != _CACHE_NAMES:
                raise ValueError("sealed ReARC cache sources are not closed-world")
            self._root = Path(os.path.abspath(os.fspath(Path(cache_dir).expanduser())))
            sources = dict(artifact_sources)
            if any(type(source) is bool for source in sources.values()):
                raise TypeError("sealed ReARC cache source is invalid")
        expected_commitment = _strict_sha(
            expected_artifact_manifest_sha256,
            field="expected_artifact_manifest_sha256",
        )
        if sources is None:
            _closed_world_files(self._root, _CACHE_NAMES)
        _, actual_commitment = _validate_artifact_manifest(
            self._root,
            expected_artifacts={"cache_manifest.json", "data.bin", "index.bin"},
            artifact_sources=sources,
        )
        if actual_commitment != expected_commitment:
            raise ValueError("ReARC cache artifact-manifest commitment mismatch")
        manifest = _strict_json(
            _read_regular_bytes(
                (
                    self._root / "cache_manifest.json"
                    if sources is None
                    else sources["cache_manifest.json"]
                ),
                maximum_bytes=_MAX_JSON_BYTES,
            ),
            label="cache_manifest.json",
        )
        expected_manifest_fields = {
            "schema",
            "cache_semantics_version",
            "fold",
            "production_contract",
            "examples_per_parent",
            "parent_count",
            "ordered_parent_ids",
            "ordered_semantic_parent_ids",
            "total_example_count",
            "input_sanitized_artifact_manifest_sha256",
            "input_shard_artifact_name",
            "input_shard_sha256",
            "data",
            "index",
            "grid_encoding",
        }
        if not isinstance(manifest, dict) or set(manifest) != expected_manifest_fields:
            raise ValueError("invalid ReARC cache manifest schema")
        if (
            manifest["schema"] != CACHE_MANIFEST_SCHEMA
            or manifest["cache_semantics_version"] != CACHE_SEMANTICS_VERSION
            or manifest["fold"] not in {"train", "validation"}
            or manifest["examples_per_parent"] != EXAMPLES_PER_PARENT
            or type(manifest["production_contract"]) is not bool
        ):
            raise ValueError("unsupported ReARC cache manifest")
        if type(allow_nonproduction_fixture) is not bool:
            raise TypeError("allow_nonproduction_fixture must be bool")
        if not allow_nonproduction_fixture and manifest["production_contract"] is not True:
            raise ValueError("production reader rejects a nonproduction ReARC cache")
        input_commitment = _strict_sha(
            manifest["input_sanitized_artifact_manifest_sha256"],
            field="input sanitized artifact commitment",
        )
        if input_commitment != _strict_sha(
            expected_sanitized_artifact_manifest_sha256,
            field="expected_sanitized_artifact_manifest_sha256",
        ):
            raise ValueError(
                "ReARC cache input sanitized artifact manifest commitment mismatch"
            )
        raw_parent_ids = manifest["ordered_parent_ids"]
        raw_semantic_parent_ids = manifest["ordered_semantic_parent_ids"]
        if not isinstance(raw_parent_ids, list) or not isinstance(
            raw_semantic_parent_ids, list
        ):
            raise ValueError("ReARC cache parent IDs must be lists")
        parent_ids = tuple(raw_parent_ids)
        semantic_parent_ids = tuple(raw_semantic_parent_ids)
        if (
            not parent_ids
            or parent_ids != tuple(sorted(parent_ids))
            or len(parent_ids) != len(set(parent_ids))
            or any(not isinstance(item, str) or _HEX8_RE.fullmatch(item) is None for item in parent_ids)
            or len(semantic_parent_ids) != len(parent_ids)
            or any(
                not isinstance(item, str) or _HEX8_RE.fullmatch(item) is None
                for item in semantic_parent_ids
            )
        ):
            raise ValueError("invalid ordered ReARC cache parent identities")
        if manifest["parent_count"] != len(parent_ids) or manifest["total_example_count"] != (
            len(parent_ids) * EXAMPLES_PER_PARENT
        ):
            raise ValueError("ReARC cache manifest counts do not close")
        if not allow_nonproduction_fixture:
            expected_parent_count = 262 if manifest["fold"] == "train" else 51
            if len(parent_ids) != expected_parent_count:
                raise ValueError("ReARC cache does not match frozen production parent count")
        data_row = manifest["data"]
        index_row = manifest["index"]
        if not isinstance(data_row, dict) or not isinstance(index_row, dict):
            raise ValueError("ReARC cache file descriptors must be objects")
        if (
            data_row.get("path") != "data.bin"
            or data_row.get("magic_hex") != DATA_MAGIC.hex()
            or index_row.get("path") != "index.bin"
            or index_row.get("magic_hex") != INDEX_MAGIC.hex()
            or index_row.get("offset_encoding") != "little_endian_uint64_absolute"
        ):
            raise ValueError("ReARC cache file descriptor mismatch")
        expected_data_sha = _strict_sha(data_row.get("sha256"), field="data.sha256")
        expected_index_sha = _strict_sha(index_row.get("sha256"), field="index.sha256")
        expected_data_bytes = _strict_uint(data_row.get("bytes"), field="data.bytes", minimum=1)
        expected_index_bytes = _strict_uint(index_row.get("bytes"), field="index.bytes", minimum=1)
        try:
            self._data_fd, self._data = _open_verified_mmap(
                self._root / "data.bin" if sources is None else sources["data.bin"],
                expected_sha256=expected_data_sha,
                expected_bytes=expected_data_bytes,
            )
            self._index_fd, self._index = _open_verified_mmap(
                self._root / "index.bin" if sources is None else sources["index.bin"],
                expected_sha256=expected_index_sha,
                expected_bytes=expected_index_bytes,
            )
            data_payload_start = _parse_data_header(self._data)
            index_cursor, parent_count, examples_per_parent, header_data_size = (
                _parse_index_header(self._index)
            )
            if (
                parent_count != len(parent_ids)
                or examples_per_parent != EXAMPLES_PER_PARENT
                or header_data_size != len(self._data)
            ):
                raise ValueError("ReARC index header counts do not close")
            ids_end = index_cursor + parent_count * 8
            offset_count = parent_count * EXAMPLES_PER_PARENT + 1
            expected_index_size = ids_end + offset_count * 8
            if ids_end > len(self._index) or len(self._index) != expected_index_size:
                raise ValueError("ReARC index has an invalid size")
            index_ids = tuple(
                bytes(self._index[index_cursor + item * 8 : index_cursor + (item + 1) * 8]).decode(
                    "ascii"
                )
                for item in range(parent_count)
            )
            if index_ids != parent_ids:
                raise ValueError("ReARC index parent IDs differ from manifest")
            self._offset_base = ids_end
            previous = None
            for flat_index in range(offset_count):
                offset = struct.unpack_from("<Q", self._index, ids_end + flat_index * 8)[0]
                if (
                    (flat_index == 0 and offset != data_payload_start)
                    or offset > len(self._data)
                    or (previous is not None and offset <= previous)
                ):
                    raise ValueError("invalid ReARC cache offset table")
                if previous is not None and _record_end(self._data, previous) != offset:
                    raise ValueError("ReARC cache offset does not match record boundary")
                previous = offset
            if previous != len(self._data):
                raise ValueError("ReARC final offset does not close data file")
            if index_row.get("offset_count") != offset_count:
                raise ValueError("ReARC cache manifest offset count mismatch")
        except Exception:
            self.close()
            raise
        self._parent_ids = parent_ids
        self._parent_index = {parent_id: index for index, parent_id in enumerate(parent_ids)}
        self._semantic_parent_ids = dict(zip(parent_ids, semantic_parent_ids))
        self._closed = False
        self._fold = str(manifest["fold"])
        self._artifact_manifest_sha256 = actual_commitment
        self._input_sanitized_artifact_manifest_sha256 = input_commitment

    @property
    def parent_ids(self) -> tuple[str, ...]:
        return self._parent_ids

    @property
    def input_sanitized_artifact_manifest_sha256(self) -> str:
        return self._input_sanitized_artifact_manifest_sha256

    @property
    def fold(self) -> str:
        return self._fold

    @property
    def artifact_manifest_sha256(self) -> str:
        return self._artifact_manifest_sha256

    def example(self, parent_id: str, example_index: int) -> M04AExample:
        if self._closed:
            raise RuntimeError("ReARC mmap source is closed")
        try:
            parent_index = self._parent_index[parent_id]
        except (KeyError, TypeError) as exc:
            raise KeyError(f"unknown ReARC parent ID: {parent_id!r}") from exc
        if type(example_index) is not int or not 0 <= example_index < EXAMPLES_PER_PARENT:
            raise IndexError("ReARC example index must be in 0..999")
        flat_index = parent_index * EXAMPLES_PER_PARENT + example_index
        start = struct.unpack_from("<Q", self._index, self._offset_base + flat_index * 8)[0]
        end = struct.unpack_from("<Q", self._index, self._offset_base + (flat_index + 1) * 8)[0]
        input_grid, cursor = _decode_grid(self._data, start)
        output_grid, cursor = _decode_grid(self._data, cursor)
        if cursor != end:
            raise ValueError("ReARC cache record endpoint mismatch")
        return M04AExample.create(input_grid, output_grid)

    def semantic_parent_id(self, parent_id: str) -> str:
        if self._closed:
            raise RuntimeError("ReARC mmap source is closed")
        try:
            return self._semantic_parent_ids[parent_id]
        except (KeyError, TypeError) as exc:
            raise KeyError(f"unknown ReARC parent ID: {parent_id!r}") from exc

    def close(self) -> None:
        data = getattr(self, "_data", None)
        index = getattr(self, "_index", None)
        if data is not None:
            data.close()
            self._data = None
        if index is not None:
            index.close()
            self._index = None
        data_fd = getattr(self, "_data_fd", None)
        index_fd = getattr(self, "_index_fd", None)
        if data_fd is not None:
            os.close(data_fd)
            self._data_fd = None
        if index_fd is not None:
            os.close(index_fd)
            self._index_fd = None
        self._closed = True

    def __enter__(self) -> "MMapReARCExampleSource":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = [
    "CACHE_MANIFEST_SCHEMA",
    "CACHE_SEMANTICS_VERSION",
    "DATA_MAGIC",
    "EXAMPLES_PER_PARENT",
    "INDEX_MAGIC",
    "MMapReARCExampleSource",
    "build_rearc_mmap_cache",
]
