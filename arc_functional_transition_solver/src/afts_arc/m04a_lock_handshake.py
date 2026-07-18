"""Externally committed, inherited-flock attestation for the M04a launcher.

The JSON artifact alone is only a statement.  Production preflight must pair an
externally committed canonical artifact with the still-open POSIX file descriptor
that the reviewed launcher acquired before it exec'd the training process.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .m04a_contract import canonical_sha256
from .manifest import serialize_json


LOCK_HANDSHAKE_ARTIFACT_SCHEMA_VERSION = (
    "afts-m04a-held-gpu-lock-handshake/v0.2"
)
LOCK_HANDSHAKE_SOURCE = "reviewed_launcher_inherited_posix_flock"
MAX_LOCK_HANDSHAKE_BYTES = 64 * 1024
MAX_HANDSHAKE_TO_PREFLIGHT_NS = 10 * 60 * 1_000_000_000

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GPU_UUID = re.compile(
    r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
_BOOT_ID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
_REPARSE_POINT_ATTRIBUTE = 0x400
_READER_TOKEN = object()
_HELD_TOKEN = object()
_FIELDS = {
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


def _strict_int(value: object, *, field_name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(f"{field_name} must be an integer >= {minimum}")
    return value


def _sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TypeError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _nonempty(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field_name} must be a non-empty string")
    return value


def _path_has_reparse_point(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE
    )


def _assert_nonsymlink_chain(path: Path) -> None:
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    anchor = Path(absolute.anchor)
    current = anchor
    for component in absolute.parts[1:]:
        current = current / component
        if not os.path.lexists(current):
            raise ValueError("lock-handshake artifact path does not exist")
        if _path_has_reparse_point(current):
            raise ValueError("lock-handshake artifact traverses a reparse point")


def _read_regular_snapshot(path: Path, *, expected_sha256: str) -> bytes:
    expected = _sha256(expected_sha256, field_name="expected_artifact_sha256")
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    flags = (
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
        current_fd = os.open(absolute.anchor, directory_flags)
        try:
            for component in absolute.parts[1:-1]:
                next_fd = os.open(component, directory_flags, dir_fd=current_fd)
                os.close(current_fd)
                current_fd = next_fd
            descriptor = os.open(absolute.parts[-1], flags, dir_fd=current_fd)
        finally:
            os.close(current_fd)
    else:
        _assert_nonsymlink_chain(absolute)
        descriptor = os.open(absolute, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_LOCK_HANDSHAKE_BYTES:
            raise ValueError("lock-handshake artifact must be a bounded regular file")
        chunks: list[bytes] = []
        remaining = MAX_LOCK_HANDSHAKE_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(8192, remaining))
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
            raise RuntimeError("lock-handshake artifact changed while reading")
    finally:
        os.close(descriptor)
    snapshot = b"".join(chunks)
    if len(snapshot) != before.st_size:
        raise ValueError("lock-handshake artifact size changed while reading")
    if hashlib.sha256(snapshot).hexdigest() != expected:
        raise ValueError("lock-handshake artifact external SHA-256 mismatch")
    return snapshot


def _decode_strict_json(snapshot: bytes) -> object:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate lock-handshake JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        return json.loads(
            snapshot.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("lock-handshake artifact is not strict UTF-8 JSON") from exc


def validate_lock_handshake_payload(payload: object) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) != _FIELDS:
        raise ValueError("lock-handshake artifact fields do not match the exact schema")
    normalized = dict(payload)
    if normalized["schema"] != LOCK_HANDSHAKE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported lock-handshake artifact schema")
    _nonempty(normalized["run_id"], field_name="run_id")
    if not isinstance(normalized["gpu_uuid"], str) or _GPU_UUID.fullmatch(
        normalized["gpu_uuid"]
    ) is None:
        raise ValueError("gpu_uuid must be a full NVIDIA GPU UUID")
    lock_path = _nonempty(normalized["lock_path"], field_name="lock_path")
    parsed_path = PurePosixPath(lock_path)
    if (
        not parsed_path.is_absolute()
        or ".." in parsed_path.parts
        or parsed_path.as_posix() != lock_path
    ):
        raise ValueError("lock_path must be an absolute lexical POSIX path")
    _strict_int(normalized["lock_st_dev"], field_name="lock_st_dev")
    _strict_int(normalized["lock_st_ino"], field_name="lock_st_ino", minimum=1)
    _strict_int(normalized["lock_holder_pid"], field_name="lock_holder_pid", minimum=1)
    _strict_int(
        normalized["lock_holder_start_ticks"],
        field_name="lock_holder_start_ticks",
        minimum=1,
    )
    _strict_int(
        normalized["inherited_lock_fd"],
        field_name="inherited_lock_fd",
        minimum=3,
    )
    _sha256(normalized["attempt_nonce"], field_name="attempt_nonce")
    if not isinstance(normalized["boot_id"], str) or _BOOT_ID.fullmatch(
        normalized["boot_id"]
    ) is None:
        raise ValueError("boot_id must be a lowercase Linux boot UUID")
    started = _strict_int(
        normalized["acquisition_started_perf_counter_ns"],
        field_name="acquisition_started_perf_counter_ns",
    )
    rechecked = _strict_int(
        normalized["eligibility_rechecked_perf_counter_ns"],
        field_name="eligibility_rechecked_perf_counter_ns",
        minimum=1,
    )
    completed = _strict_int(
        normalized["handshake_completed_perf_counter_ns"],
        field_name="handshake_completed_perf_counter_ns",
        minimum=1,
    )
    wall = _strict_int(normalized["wall_ns"], field_name="wall_ns", minimum=1)
    if not started < rechecked <= completed or wall != completed - started:
        raise ValueError("lock-handshake monotonic endpoints do not close")
    _sha256(normalized["launcher_sha256"], field_name="launcher_sha256")
    _sha256(normalized["launch_plan_sha256"], field_name="launch_plan_sha256")
    if normalized["source"] != LOCK_HANDSHAKE_SOURCE:
        raise ValueError("lock-handshake source is not the reviewed launcher")
    semantic = dict(normalized)
    identifier = semantic.pop("handshake_id")
    if identifier != canonical_sha256(semantic):
        raise ValueError("lock-handshake handshake_id mismatch")
    return normalized


@dataclass(frozen=True, slots=True)
class LockHandshakeArtifact:
    artifact_sha256: str
    snapshot: bytes = field(repr=False)
    expected_run_id: str
    expected_gpu_uuid: str
    expected_launcher_sha256: str
    expected_launch_plan_sha256: str
    expected_remote_project_root: str
    expected_attempt_nonce: str = field(repr=False)
    _reader_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._reader_token is not _READER_TOKEN:
            raise TypeError("lock-handshake artifact must come from the committed reader")
        digest = _sha256(self.artifact_sha256, field_name="artifact_sha256")
        if type(self.snapshot) is not bytes or hashlib.sha256(self.snapshot).hexdigest() != digest:
            raise ValueError("lock-handshake immutable snapshot mismatch")
        payload = _decode_strict_json(self.snapshot)
        normalized = validate_lock_handshake_payload(payload)
        if serialize_json(normalized) != self.snapshot:
            raise ValueError("lock-handshake artifact is not canonically serialized")
        expected_gpu = _nonempty(
            self.expected_gpu_uuid, field_name="expected_gpu_uuid"
        )
        remote_root = PurePosixPath(
            _nonempty(
                self.expected_remote_project_root,
                field_name="expected_remote_project_root",
            )
        )
        if (
            not remote_root.is_absolute()
            or ".." in remote_root.parts
            or remote_root.as_posix() != self.expected_remote_project_root
        ):
            raise ValueError("expected_remote_project_root must be absolute lexical POSIX")
        expected = {
            "run_id": _nonempty(self.expected_run_id, field_name="expected_run_id"),
            "gpu_uuid": expected_gpu,
            "lock_path": (
                remote_root / "locks" / f"gpu-{expected_gpu}.lock"
            ).as_posix(),
            "launcher_sha256": _sha256(
                self.expected_launcher_sha256,
                field_name="expected_launcher_sha256",
            ),
            "launch_plan_sha256": _sha256(
                self.expected_launch_plan_sha256,
                field_name="expected_launch_plan_sha256",
            ),
            "attempt_nonce": _sha256(
                self.expected_attempt_nonce, field_name="expected_attempt_nonce"
            ),
        }
        if any(normalized[field] != value for field, value in expected.items()):
            raise ValueError("lock-handshake snapshot differs from retained commitments")

    @property
    def payload(self) -> dict[str, Any]:
        payload = _decode_strict_json(self.snapshot)
        return validate_lock_handshake_payload(payload)


def read_lock_handshake_artifact(
    path: str | Path,
    *,
    expected_artifact_sha256: str,
    expected_run_id: str,
    expected_gpu_uuid: str,
    expected_launcher_sha256: str,
    expected_launch_plan_sha256: str,
    expected_remote_project_root: str,
    expected_attempt_nonce: str,
) -> LockHandshakeArtifact:
    artifact_path = Path(path).expanduser()
    snapshot = _read_regular_snapshot(
        artifact_path, expected_sha256=expected_artifact_sha256
    )
    payload = _decode_strict_json(snapshot)
    normalized = validate_lock_handshake_payload(payload)
    expected_gpu = _nonempty(expected_gpu_uuid, field_name="expected_gpu_uuid")
    remote_root = PurePosixPath(
        _nonempty(
            expected_remote_project_root,
            field_name="expected_remote_project_root",
        )
    )
    if (
        not remote_root.is_absolute()
        or ".." in remote_root.parts
        or remote_root.as_posix() != expected_remote_project_root
    ):
        raise ValueError("expected_remote_project_root must be absolute lexical POSIX")
    expected = {
        "run_id": _nonempty(expected_run_id, field_name="expected_run_id"),
        "gpu_uuid": expected_gpu,
        "lock_path": (
            remote_root / "locks" / f"gpu-{expected_gpu}.lock"
        ).as_posix(),
        "launcher_sha256": _sha256(
            expected_launcher_sha256, field_name="expected_launcher_sha256"
        ),
        "launch_plan_sha256": _sha256(
            expected_launch_plan_sha256, field_name="expected_launch_plan_sha256"
        ),
        "attempt_nonce": _sha256(
            expected_attempt_nonce, field_name="expected_attempt_nonce"
        ),
    }
    if any(normalized[field] != value for field, value in expected.items()):
        raise ValueError("lock-handshake artifact differs from external launch commitments")
    return LockHandshakeArtifact(
        artifact_sha256=expected_artifact_sha256,
        snapshot=snapshot,
        expected_run_id=expected["run_id"],
        expected_gpu_uuid=expected["gpu_uuid"],
        expected_launcher_sha256=expected["launcher_sha256"],
        expected_launch_plan_sha256=expected["launch_plan_sha256"],
        expected_remote_project_root=expected_remote_project_root,
        expected_attempt_nonce=expected["attempt_nonce"],
        _reader_token=_READER_TOKEN,
    )


def _linux_boot_id() -> str:
    if os.name != "posix":
        raise RuntimeError("GPU lock freshness requires Linux /proc")
    value = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip()
    if _BOOT_ID.fullmatch(value) is None:
        raise RuntimeError("Linux boot_id is unavailable or malformed")
    return value


def _linux_process_start_ticks(pid: int) -> int:
    process_id = _strict_int(pid, field_name="pid", minimum=1)
    raw = Path(f"/proc/{process_id}/stat").read_text(encoding="ascii")
    close = raw.rfind(")")
    if close < 0:
        raise RuntimeError("Linux process stat is malformed")
    fields = raw[close + 2 :].split()
    if len(fields) <= 19:
        raise RuntimeError("Linux process stat lacks starttime")
    return _strict_int(int(fields[19]), field_name="process_start_ticks", minimum=1)


def _fdinfo_proves_holder(payload: dict[str, Any], descriptor: int) -> None:
    """Prove this specific FD shares the launcher-held flock description."""

    fd_stat = os.fstat(descriptor)
    expected = (
        os.major(fd_stat.st_dev),
        os.minor(fd_stat.st_dev),
        int(fd_stat.st_ino),
    )
    holder_pid = int(payload["lock_holder_pid"])
    with Path(f"/proc/self/fdinfo/{descriptor}").open(
        "r", encoding="ascii"
    ) as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number > 4096:
                raise RuntimeError("descriptor fdinfo exceeded the bounded record limit")
            fields = line.split()
            if (
                len(fields) < 7
                or fields[0] != "lock:"
                or fields[2:5] != ["FLOCK", "ADVISORY", "WRITE"]
            ):
                continue
            try:
                pid = int(fields[5])
                major_text, minor_text, inode_text = fields[6].split(":", 2)
                identity = (
                    int(major_text, 16),
                    int(minor_text, 16),
                    int(inode_text),
                )
            except (ValueError, IndexError):
                continue
            if pid == holder_pid and identity == expected:
                return
    raise ValueError("descriptor fdinfo does not prove this FD carries the declared flock")


@dataclass(frozen=True, slots=True)
class HeldGPULockHandshake:
    artifact: LockHandshakeArtifact
    preflight_started_perf_counter_ns: int
    _owned_fd: _OwnedLockFD = field(repr=False, compare=False)
    _held_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._held_token is not _HELD_TOKEN:
            raise TypeError("held GPU lock must come from POSIX FD attestation")
        if self.artifact._reader_token is not _READER_TOKEN:
            raise ValueError("held GPU lock lacks its committed reader parent")
        _validate_preflight_start(
            self.artifact.payload, self.preflight_started_perf_counter_ns
        )
        if type(self._owned_fd) is not _OwnedLockFD:
            raise TypeError("held GPU lock lacks its owned descriptor lifecycle")
        assert_held_gpu_lock(self)


def _validate_preflight_start(payload: dict[str, Any], value: int) -> int:
    started = _strict_int(
        value,
        field_name="preflight_started_perf_counter_ns",
        minimum=1,
    )
    completed = int(payload["handshake_completed_perf_counter_ns"])
    if started < completed:
        raise ValueError("preflight started before the lock handshake completed")
    if started - completed > MAX_HANDSHAKE_TO_PREFLIGHT_NS:
        raise ValueError("lock handshake is too old for this preflight attempt")
    return started


def _assert_posix_fd_matches(payload: dict[str, Any], *, descriptor: int) -> None:
    if os.name != "posix":
        raise RuntimeError("production GPU lock attestation requires POSIX flock")
    if payload["lock_holder_pid"] != os.getpid():
        raise ValueError("reviewed launcher must exec training in the lock-holder PID")
    if payload["boot_id"] != _linux_boot_id():
        raise ValueError("lock handshake belongs to another Linux boot")
    if payload["lock_holder_start_ticks"] != _linux_process_start_ticks(os.getpid()):
        raise ValueError("lock handshake belongs to another process lifetime")
    fd_stat = os.fstat(descriptor)
    if (
        not stat.S_ISREG(fd_stat.st_mode)
        or fd_stat.st_dev != payload["lock_st_dev"]
        or fd_stat.st_ino != payload["lock_st_ino"]
    ):
        raise ValueError("inherited lock FD identity differs from the handshake")
    lock_path = Path(payload["lock_path"])
    _assert_nonsymlink_chain(lock_path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    path_fd = os.open(lock_path, flags)
    try:
        path_stat = os.fstat(path_fd)
        if not stat.S_ISREG(path_stat.st_mode):
            raise ValueError("GPU lock path is not a regular file")
        if not os.path.samestat(path_stat, fd_stat):
            raise ValueError("lock path and inherited FD identify different files")
    finally:
        os.close(path_fd)
    _fdinfo_proves_holder(payload, descriptor)


class _OwnedLockFD:
    """Idempotently own one dup of the inherited open-file description."""

    __slots__ = ("_closed", "_descriptor", "_mutex")

    def __init__(self, descriptor: int, *, token: object) -> None:
        if token is not _HELD_TOKEN:
            raise TypeError("owned GPU-lock descriptor is internal")
        self._descriptor = _strict_int(
            descriptor, field_name="owned_lock_fd", minimum=0
        )
        self._closed = False
        self._mutex = threading.Lock()

    def assert_matches(self, payload: dict[str, Any]) -> None:
        with self._mutex:
            if self._closed:
                raise RuntimeError("owned GPU-lock descriptor has been released")
            _assert_posix_fd_matches(payload, descriptor=self._descriptor)

    def close(self) -> None:
        with self._mutex:
            if self._closed:
                return
            os.close(self._descriptor)
            self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def attest_inherited_gpu_lock(
    artifact: LockHandshakeArtifact,
    *,
    preflight_started_perf_counter_ns: int,
) -> HeldGPULockHandshake:
    if type(artifact) is not LockHandshakeArtifact or artifact._reader_token is not _READER_TOKEN:
        raise TypeError("GPU lock attestation requires a reader-produced artifact")
    payload = artifact.payload
    inherited_fd = int(payload["inherited_lock_fd"])
    _assert_posix_fd_matches(payload, descriptor=inherited_fd)
    try:
        owned_fd = os.dup(inherited_fd)
    except BaseException:
        try:
            os.close(inherited_fd)
        except OSError:
            pass
        raise
    try:
        os.set_inheritable(owned_fd, False)
    except BaseException:
        for descriptor in (owned_fd, inherited_fd):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise
    try:
        os.close(inherited_fd)
    except BaseException:
        try:
            os.close(owned_fd)
        except OSError:
            pass
        raise
    lifecycle = _OwnedLockFD(owned_fd, token=_HELD_TOKEN)
    try:
        return HeldGPULockHandshake(
            artifact=artifact,
            preflight_started_perf_counter_ns=preflight_started_perf_counter_ns,
            _owned_fd=lifecycle,
            _held_token=_HELD_TOKEN,
        )
    except BaseException:
        lifecycle.close()
        raise


def assert_held_gpu_lock(handshake: HeldGPULockHandshake) -> dict[str, Any]:
    if type(handshake) is not HeldGPULockHandshake or handshake._held_token is not _HELD_TOKEN:
        raise TypeError("expected a POSIX-attested held GPU lock")
    payload = handshake.artifact.payload
    handshake._owned_fd.assert_matches(payload)
    return dict(payload)


def release_held_gpu_lock(handshake: HeldGPULockHandshake) -> None:
    """Release the owned duplicate exactly once after the terminal lock endpoint."""

    if type(handshake) is not HeldGPULockHandshake or handshake._held_token is not _HELD_TOKEN:
        raise TypeError("expected a POSIX-attested held GPU lock")
    handshake._owned_fd.close()


def make_cost_probe_lock_handshake_from_artifact(
    artifact: LockHandshakeArtifact,
) -> dict[str, Any]:
    """Purely derive the legacy timing child from a committed full artifact."""

    if type(artifact) is not LockHandshakeArtifact or artifact._reader_token is not _READER_TOKEN:
        raise TypeError("cost-probe lock child requires a reader-produced artifact")
    payload = artifact.payload
    child: dict[str, Any] = {
        "schema": "afts-grid-cmlm-preflight-lock-handshake-probe/v0.1",
        "run_id": payload["run_id"],
        "acquisition_started_perf_counter_ns": payload[
            "acquisition_started_perf_counter_ns"
        ],
        "handshake_completed_perf_counter_ns": payload[
            "handshake_completed_perf_counter_ns"
        ],
        "wall_ns": payload["wall_ns"],
        "source": "verified_launcher_lock_handshake",
    }
    child["probe_id"] = canonical_sha256(child)
    return child


def make_cost_probe_lock_handshake(
    handshake: HeldGPULockHandshake,
) -> dict[str, Any]:
    """Derive the legacy timing child from a still-held committed lock parent."""

    assert_held_gpu_lock(handshake)
    return make_cost_probe_lock_handshake_from_artifact(handshake.artifact)


__all__ = [
    "HeldGPULockHandshake",
    "LOCK_HANDSHAKE_ARTIFACT_SCHEMA_VERSION",
    "LOCK_HANDSHAKE_SOURCE",
    "MAX_HANDSHAKE_TO_PREFLIGHT_NS",
    "LockHandshakeArtifact",
    "assert_held_gpu_lock",
    "attest_inherited_gpu_lock",
    "make_cost_probe_lock_handshake",
    "make_cost_probe_lock_handshake_from_artifact",
    "read_lock_handshake_artifact",
    "release_held_gpu_lock",
    "validate_lock_handshake_payload",
]
