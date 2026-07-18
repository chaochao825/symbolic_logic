from __future__ import annotations

import hashlib
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from afts_arc import m04a_lock_handshake as lock_handshake
from afts_arc.m04a_contract import canonical_sha256
from afts_arc.m04a_lock_handshake import (
    LOCK_HANDSHAKE_ARTIFACT_SCHEMA_VERSION,
    LOCK_HANDSHAKE_SOURCE,
    assert_held_gpu_lock,
    attest_inherited_gpu_lock,
    make_cost_probe_lock_handshake,
    make_cost_probe_lock_handshake_from_artifact,
    read_lock_handshake_artifact,
    release_held_gpu_lock,
    validate_lock_handshake_payload,
)
from afts_arc.manifest import serialize_json


GPU_UUID = "GPU-00000000-0000-0000-0000-000000000001"
ATTEMPT_NONCE = "c" * 64


def _boot_id() -> str:
    if os.name != "posix":
        return "00000000-0000-0000-0000-000000000001"
    return Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip()


def _process_start_ticks() -> int:
    if os.name != "posix":
        return 1
    raw = Path(f"/proc/{os.getpid()}/stat").read_text(encoding="ascii")
    fields = raw[raw.rfind(")") + 2 :].split()
    return int(fields[19])


def _payload(
    lock_path: str,
    *,
    descriptor: int,
    device: int,
    inode: int,
    completed_ns: int = 110,
) -> dict[str, object]:
    semantic: dict[str, object] = {
        "schema": LOCK_HANDSHAKE_ARTIFACT_SCHEMA_VERSION,
        "run_id": "m04a-test-run",
        "gpu_uuid": GPU_UUID,
        "lock_path": lock_path,
        "lock_st_dev": device,
        "lock_st_ino": inode,
        "lock_holder_pid": os.getpid(),
        "lock_holder_start_ticks": _process_start_ticks(),
        "inherited_lock_fd": descriptor,
        "attempt_nonce": ATTEMPT_NONCE,
        "boot_id": _boot_id(),
        "acquisition_started_perf_counter_ns": completed_ns - 10,
        "eligibility_rechecked_perf_counter_ns": completed_ns - 5,
        "handshake_completed_perf_counter_ns": completed_ns,
        "wall_ns": 10,
        "launcher_sha256": "a" * 64,
        "launch_plan_sha256": "b" * 64,
        "source": LOCK_HANDSHAKE_SOURCE,
    }
    return {**semantic, "handshake_id": canonical_sha256(semantic)}


def _read(path: Path, content: bytes, *, remote_root: str):
    return read_lock_handshake_artifact(
        path,
        expected_artifact_sha256=hashlib.sha256(content).hexdigest(),
        expected_run_id="m04a-test-run",
        expected_gpu_uuid=GPU_UUID,
        expected_launcher_sha256="a" * 64,
        expected_launch_plan_sha256="b" * 64,
        expected_remote_project_root=remote_root,
        expected_attempt_nonce=ATTEMPT_NONCE,
    )


class M04aLockHandshakeTests(unittest.TestCase):
    def test_gpu_uuid_requires_canonical_lowercase_full_uuid(self) -> None:
        for invalid in (
            "GPU-1234-abcd",
            "GPU-00000000-0000-0000-0000-00000000000A",
        ):
            payload = _payload(
                "/srv/afts/locks/gpu-invalid.lock",
                descriptor=9,
                device=1,
                inode=2,
            )
            payload["gpu_uuid"] = invalid
            semantic = dict(payload)
            semantic.pop("handshake_id")
            payload["handshake_id"] = canonical_sha256(semantic)
            with self.assertRaisesRegex(ValueError, "full NVIDIA GPU UUID"):
                validate_lock_handshake_payload(payload)

    def test_attestation_transfers_the_original_descriptor_into_its_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dummy_lock = root / "gpu.lock"
            dummy_lock.write_bytes(b"")
            info = dummy_lock.stat()
            remote_root = "/srv/afts"
            content = serialize_json(
                _payload(
                    f"{remote_root}/locks/gpu-{GPU_UUID}.lock",
                    descriptor=9,
                    device=info.st_dev,
                    inode=info.st_ino,
                )
            )
            artifact_path = root / "handshake.json"
            artifact_path.write_bytes(content)
            artifact = _read(artifact_path, content, remote_root=remote_root)
            sentinel = object()
            with (
                mock.patch.object(lock_handshake, "_assert_posix_fd_matches"),
                mock.patch.object(lock_handshake.os, "dup", return_value=17),
                mock.patch.object(lock_handshake.os, "set_inheritable") as inheritable,
                mock.patch.object(lock_handshake.os, "close") as closed,
                mock.patch.object(
                    lock_handshake,
                    "HeldGPULockHandshake",
                    return_value=sentinel,
                ),
            ):
                result = attest_inherited_gpu_lock(
                    artifact, preflight_started_perf_counter_ns=111
                )
            self.assertIs(result, sentinel)
            inheritable.assert_called_once_with(17, False)
            closed.assert_any_call(9)

    def test_attestation_closes_verified_original_when_duplication_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dummy_lock = root / "gpu.lock"
            dummy_lock.write_bytes(b"")
            info = dummy_lock.stat()
            remote_root = "/srv/afts"
            content = serialize_json(
                _payload(
                    f"{remote_root}/locks/gpu-{GPU_UUID}.lock",
                    descriptor=9,
                    device=info.st_dev,
                    inode=info.st_ino,
                )
            )
            artifact_path = root / "handshake.json"
            artifact_path.write_bytes(content)
            artifact = _read(artifact_path, content, remote_root=remote_root)
            with (
                mock.patch.object(lock_handshake, "_assert_posix_fd_matches"),
                mock.patch.object(
                    lock_handshake.os, "dup", side_effect=OSError("dup failed")
                ),
                mock.patch.object(lock_handshake.os, "close") as closed,
            ):
                with self.assertRaisesRegex(OSError, "dup failed"):
                    attest_inherited_gpu_lock(
                        artifact, preflight_started_perf_counter_ns=111
                    )
            closed.assert_called_once_with(9)

    def test_attestation_closes_both_descriptors_when_cloexec_setup_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dummy_lock = root / "gpu.lock"
            dummy_lock.write_bytes(b"")
            info = dummy_lock.stat()
            remote_root = "/srv/afts"
            content = serialize_json(
                _payload(
                    f"{remote_root}/locks/gpu-{GPU_UUID}.lock",
                    descriptor=9,
                    device=info.st_dev,
                    inode=info.st_ino,
                )
            )
            artifact_path = root / "handshake.json"
            artifact_path.write_bytes(content)
            artifact = _read(artifact_path, content, remote_root=remote_root)
            with (
                mock.patch.object(lock_handshake, "_assert_posix_fd_matches"),
                mock.patch.object(lock_handshake.os, "dup", return_value=17),
                mock.patch.object(
                    lock_handshake.os,
                    "set_inheritable",
                    side_effect=OSError("cloexec failed"),
                ),
                mock.patch.object(lock_handshake.os, "close") as closed,
            ):
                with self.assertRaisesRegex(OSError, "cloexec failed"):
                    attest_inherited_gpu_lock(
                        artifact, preflight_started_perf_counter_ns=111
                    )
            self.assertEqual(closed.call_args_list, [mock.call(17), mock.call(9)])

    def test_stale_preflight_releases_transferred_lock_descriptors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dummy_lock = root / "gpu.lock"
            dummy_lock.write_bytes(b"")
            info = dummy_lock.stat()
            remote_root = "/srv/afts"
            content = serialize_json(
                _payload(
                    f"{remote_root}/locks/gpu-{GPU_UUID}.lock",
                    descriptor=9,
                    device=info.st_dev,
                    inode=info.st_ino,
                )
            )
            artifact_path = root / "handshake.json"
            artifact_path.write_bytes(content)
            artifact = _read(artifact_path, content, remote_root=remote_root)
            with (
                mock.patch.object(lock_handshake, "_assert_posix_fd_matches"),
                mock.patch.object(lock_handshake.os, "dup", return_value=17),
                mock.patch.object(lock_handshake.os, "set_inheritable"),
                mock.patch.object(lock_handshake.os, "close") as closed,
            ):
                with self.assertRaisesRegex(ValueError, "before the lock"):
                    attest_inherited_gpu_lock(
                        artifact, preflight_started_perf_counter_ns=109
                    )
            self.assertEqual(closed.call_args_list, [mock.call(9), mock.call(17)])

    def test_external_hash_and_immutable_snapshot_bind_all_launch_parents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dummy_lock = root / "gpu.lock"
            dummy_lock.write_bytes(b"")
            info = dummy_lock.stat()
            remote_root = "/srv/afts"
            lock_path = f"{remote_root}/locks/gpu-{GPU_UUID}.lock"
            content = serialize_json(
                _payload(lock_path, descriptor=9, device=info.st_dev, inode=info.st_ino)
            )
            artifact_path = root / "handshake.json"
            artifact_path.write_bytes(content)
            artifact = _read(artifact_path, content, remote_root=remote_root)
            first = artifact.payload
            first["run_id"] = "mutated-copy"
            self.assertEqual(artifact.payload["run_id"], "m04a-test-run")
            self.assertEqual(artifact.snapshot, content)

            with self.assertRaisesRegex(ValueError, "external SHA"):
                read_lock_handshake_artifact(
                    artifact_path,
                    expected_artifact_sha256="0" * 64,
                    expected_run_id="m04a-test-run",
                    expected_gpu_uuid=GPU_UUID,
                    expected_launcher_sha256="a" * 64,
                    expected_launch_plan_sha256="b" * 64,
                    expected_remote_project_root=remote_root,
                    expected_attempt_nonce=ATTEMPT_NONCE,
                )
            with self.assertRaisesRegex(ValueError, "external launch"):
                read_lock_handshake_artifact(
                    artifact_path,
                    expected_artifact_sha256=hashlib.sha256(content).hexdigest(),
                    expected_run_id="another-run",
                    expected_gpu_uuid=GPU_UUID,
                    expected_launcher_sha256="a" * 64,
                    expected_launch_plan_sha256="b" * 64,
                    expected_remote_project_root=remote_root,
                    expected_attempt_nonce=ATTEMPT_NONCE,
                )

            wrong = _payload(
                f"{remote_root}/locks/not-the-gpu-lock.lock",
                descriptor=9,
                device=info.st_dev,
                inode=info.st_ino,
            )
            wrong_content = serialize_json(wrong)
            artifact_path.write_bytes(wrong_content)
            with self.assertRaisesRegex(ValueError, "external launch"):
                _read(artifact_path, wrong_content, remote_root=remote_root)

    @unittest.skipUnless(os.name == "posix", "production lock attestation requires POSIX")
    def test_inherited_flock_fd_is_live_and_precedes_preflight(self) -> None:
        import fcntl

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locks = root / "locks"
            locks.mkdir()
            lock_path = locks / f"gpu-{GPU_UUID}.lock"
            descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            original_open = True
            held = None
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                info = os.fstat(descriptor)
                completed_ns = time.perf_counter_ns()
                content = serialize_json(
                    _payload(
                        lock_path.as_posix(),
                        descriptor=descriptor,
                        device=info.st_dev,
                        inode=info.st_ino,
                        completed_ns=completed_ns,
                    )
                )
                artifact_path = root / "handshake.json"
                artifact_path.write_bytes(content)
                artifact = _read(
                    artifact_path, content, remote_root=root.as_posix()
                )
                held = attest_inherited_gpu_lock(
                    artifact,
                    preflight_started_perf_counter_ns=completed_ns + 1,
                )
                self.assertEqual(assert_held_gpu_lock(held), artifact.payload)
                self.assertEqual(
                    make_cost_probe_lock_handshake(held),
                    make_cost_probe_lock_handshake_from_artifact(artifact),
                )
                original_open = False
                with self.assertRaises(OSError):
                    os.fstat(descriptor)
                self.assertEqual(assert_held_gpu_lock(held), artifact.payload)
                release_held_gpu_lock(held)
                release_held_gpu_lock(held)
                with self.assertRaisesRegex(RuntimeError, "released"):
                    assert_held_gpu_lock(held)
            finally:
                if held is not None:
                    release_held_gpu_lock(held)
                if original_open:
                    os.close(descriptor)

    @unittest.skipUnless(os.name == "posix", "production lock attestation requires POSIX")
    def test_unlocked_descriptor_cannot_be_promoted_by_the_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locks = root / "locks"
            locks.mkdir()
            lock_path = locks / f"gpu-{GPU_UUID}.lock"
            descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                info = os.fstat(descriptor)
                completed_ns = time.perf_counter_ns()
                content = serialize_json(
                    _payload(
                        lock_path.as_posix(),
                        descriptor=descriptor,
                        device=info.st_dev,
                        inode=info.st_ino,
                        completed_ns=completed_ns,
                    )
                )
                artifact_path = root / "handshake.json"
                artifact_path.write_bytes(content)
                artifact = _read(
                    artifact_path, content, remote_root=root.as_posix()
                )
                with self.assertRaisesRegex(ValueError, "fdinfo"):
                    attest_inherited_gpu_lock(
                        artifact,
                        preflight_started_perf_counter_ns=completed_ns + 1,
                    )
            finally:
                os.close(descriptor)

    @unittest.skipUnless(os.name == "posix", "production lock attestation requires POSIX")
    def test_same_inode_unlocked_fd_cannot_borrow_another_fd_lock(self) -> None:
        import fcntl

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locks = root / "locks"
            locks.mkdir()
            lock_path = locks / f"gpu-{GPU_UUID}.lock"
            holder_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            wrong_fd = os.open(lock_path, os.O_RDWR)
            try:
                fcntl.flock(holder_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                info = os.fstat(wrong_fd)
                completed_ns = time.perf_counter_ns()
                content = serialize_json(
                    _payload(
                        lock_path.as_posix(),
                        descriptor=wrong_fd,
                        device=info.st_dev,
                        inode=info.st_ino,
                        completed_ns=completed_ns,
                    )
                )
                artifact_path = root / "handshake.json"
                artifact_path.write_bytes(content)
                artifact = _read(
                    artifact_path, content, remote_root=root.as_posix()
                )
                with self.assertRaisesRegex(ValueError, "specific FD|fdinfo|carries"):
                    attest_inherited_gpu_lock(
                        artifact,
                        preflight_started_perf_counter_ns=completed_ns + 1,
                    )
            finally:
                os.close(wrong_fd)
                os.close(holder_fd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
