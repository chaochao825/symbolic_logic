"""Explicit CPU-safe tests for the frozen M04a Torch runtime boundary."""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from afts_arc import m04a_torch_runtime as runtime


_GPU_A = "GPU-aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
_GPU_B = "GPU-11111111-2222-4333-8444-555555555555"
_CONFIG_SHA256 = "a" * 64
_RUNTIME_SHA256 = "b" * 64
_TEST_SHA256 = "c" * 64


class _CUuuid:
    __module__ = "torch._C"

    def __init__(self, value: str, *, rendered: str | None = None) -> None:
        compact = value.removeprefix("GPU-").replace("-", "")
        self.bytes = list(bytes.fromhex(compact))
        self.rendered = value.removeprefix("GPU-") if rendered is None else rendered

    def __str__(self) -> str:
        return self.rendered


def _checkpoint_payload() -> dict[str, object]:
    return {
        "schema": runtime.CHECKPOINT_SCHEMA_VERSION,
        "model_semantics_version": runtime.MODEL_SEMANTICS_VERSION,
        "optimizer_step": 7,
        "model_state": {},
        "optimizer_state": {},
        "rng_state": {
            "schema": runtime.RNG_STATE_SCHEMA_VERSION,
            "optimizer_step": 7,
            "torch_cpu": runtime.torch.arange(4, dtype=runtime.torch.uint8),
            "torch_cuda": [],
        },
        "config_sha256": _CONFIG_SHA256,
        "runtime_source_sha256": _RUNTIME_SHA256,
        "test_source_sha256": _TEST_SHA256,
    }


class M04aRuntimeUuidBindingTests(unittest.TestCase):
    def test_ordered_import_roots_accept_exact_deduplicated_two_or_three(self) -> None:
        for roots in (
            ["/project/src", "/site"],
            ["/project/src", "/pure", "/plat"],
        ):
            with self.subTest(roots=roots):
                self.assertEqual(
                    runtime._validate_ordered_import_roots(
                        roots,
                        bound_import_roots=[
                            f"/proc/self/fd/{200 + index}"
                            for index in range(len(roots))
                        ],
                        ordered_sys_path=[
                            "/stdlib",
                            *[
                                f"/proc/self/fd/{200 + index}"
                                for index in range(len(roots))
                            ],
                        ],
                    ),
                    roots,
                )
        for roots in (
            ["/one"],
            [f"/root/{index}" for index in range(4)],
            ["/project/src", "/project/src"],
            ["relative", "/site"],
        ):
            with self.subTest(invalid_roots=roots), self.assertRaises(ValueError):
                runtime._validate_ordered_import_roots(
                    roots,
                    bound_import_roots=[
                        f"/proc/self/fd/{200 + index}" for index in range(len(roots))
                    ],
                    ordered_sys_path=[
                        "/stdlib",
                        *[
                            f"/proc/self/fd/{200 + index}"
                            for index in range(len(roots))
                        ],
                    ],
                )

    def test_pure_binding_closes_canonical_single_visible_cuda_zero(self) -> None:
        self.assertEqual(
            runtime.bind_single_visible_cuda_uuid(
                configured_gpu_uuid=_GPU_A.upper(),
                cuda_visible_devices=_GPU_A,
                visible_device_count=1,
                current_device_index=0,
                logical_device_uuid=_GPU_A.removeprefix("GPU-"),
            ),
            _GPU_A,
        )

    def test_pure_binding_rejects_arbitrary_visible_or_logical_gpu(self) -> None:
        common = {
            "configured_gpu_uuid": _GPU_A,
            "cuda_visible_devices": _GPU_A,
            "visible_device_count": 1,
            "current_device_index": 0,
            "logical_device_uuid": _GPU_A,
        }
        for changed, message in (
            ({"cuda_visible_devices": _GPU_B}, "CUDA_VISIBLE_DEVICES"),
            ({"logical_device_uuid": _GPU_B}, "logical cuda:0 UUID"),
            ({"visible_device_count": 2}, "exactly one"),
            ({"current_device_index": 1}, "device 0"),
        ):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(RuntimeError, message):
                    runtime.bind_single_visible_cuda_uuid(**(common | changed))

    def test_torch_probe_binds_properties_of_logical_cuda_zero(self) -> None:
        environment = {
            "AFTS_M04A_GPU_UUID": _GPU_A,
            "CUDA_VISIBLE_DEVICES": _GPU_A,
        }
        with (
            patch.dict(os.environ, environment, clear=False),
            patch.object(runtime.torch.cuda, "device_count", return_value=1),
            patch.object(runtime.torch.cuda, "current_device", return_value=0),
            patch.object(
                runtime.torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(uuid=_GPU_A.removeprefix("GPU-")),
            ) as properties,
        ):
            self.assertEqual(runtime._bind_torch_logical_cuda_zero_uuid(), _GPU_A)
        properties.assert_called_once_with(0)

    def test_torch_probe_binds_exact_frozen_torch_cuuid(self) -> None:
        environment = {
            "AFTS_M04A_GPU_UUID": _GPU_A,
            "CUDA_VISIBLE_DEVICES": _GPU_A,
        }
        logical_uuid = _CUuuid(_GPU_A)
        with (
            patch.dict(os.environ, environment, clear=False),
            patch.object(runtime.torch._C, "_CUuuid", _CUuuid, create=True),
            patch.object(runtime.torch.cuda, "device_count", return_value=1),
            patch.object(runtime.torch.cuda, "current_device", return_value=0),
            patch.object(
                runtime.torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(uuid=logical_uuid),
            ),
        ):
            self.assertIs(type(logical_uuid), runtime.torch._C._CUuuid)
            self.assertEqual(runtime._bind_torch_logical_cuda_zero_uuid(), _GPU_A)

    def test_torch_cuuid_adapter_rejects_impostors_or_inconsistent_bytes(self) -> None:
        class SpoofedCUuuid:
            __module__ = "torch._C"
            __qualname__ = "_CUuuid"

        class SubclassedCUuuid(_CUuuid):
            pass

        with patch.object(runtime.torch._C, "_CUuuid", _CUuuid, create=True):
            with self.assertRaises(TypeError):
                runtime._torch_cuda_device_uuid_text(
                    SimpleNamespace(bytes=list(bytes.fromhex("00" * 16)))
                )
            with self.assertRaises(TypeError):
                runtime._torch_cuda_device_uuid_text(SpoofedCUuuid())
            with self.assertRaises(TypeError):
                runtime._torch_cuda_device_uuid_text(SubclassedCUuuid(_GPU_A))
            valid_raw = _CUuuid(_GPU_A).bytes
            for invalid_raw in (
                tuple(valid_raw),
                bytes(valid_raw),
                range(16),
                valid_raw[:-1],
                [*valid_raw, 0],
                [True, *valid_raw[1:]],
                [-1, *valid_raw[1:]],
                [256, *valid_raw[1:]],
                [1.0, *valid_raw[1:]],
            ):
                with self.subTest(invalid_raw=invalid_raw):
                    malformed = _CUuuid(_GPU_A)
                    malformed.bytes = invalid_raw
                    with self.assertRaises(TypeError):
                        runtime._torch_cuda_device_uuid_text(malformed)
            with self.assertRaises(ValueError):
                runtime._torch_cuda_device_uuid_text(_CUuuid(_GPU_A, rendered=_GPU_B))
            for invalid_text in (
                "MIG-aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee/1/0",
                "aaaaaaaa–bbbb-4ccc-8ddd-eeeeeeeeeeee",
            ):
                with self.subTest(invalid_text=invalid_text):
                    with self.assertRaises(ValueError):
                        runtime._torch_cuda_device_uuid_text(
                            _CUuuid(_GPU_A, rendered=invalid_text)
                        )

    def test_torch_cuuid_adapter_propagates_property_or_string_failures(self) -> None:
        class ExplodingBytes:
            @property
            def bytes(self) -> list[int]:
                raise RuntimeError("bytes unavailable")

        class ExplodingString:
            bytes = _CUuuid(_GPU_A).bytes

            def __str__(self) -> str:
                raise RuntimeError("string unavailable")

        for uuid_type, message in (
            (ExplodingBytes, "bytes unavailable"),
            (ExplodingString, "string unavailable"),
        ):
            with self.subTest(uuid_type=uuid_type):
                with (
                    patch.object(runtime.torch._C, "_CUuuid", uuid_type, create=True),
                    self.assertRaisesRegex(RuntimeError, message),
                ):
                    runtime._torch_cuda_device_uuid_text(uuid_type())

    def test_torch_probe_fails_closed_on_logical_uuid_mismatch_or_absence(self) -> None:
        environment = {
            "AFTS_M04A_GPU_UUID": _GPU_A,
            "CUDA_VISIBLE_DEVICES": _GPU_A,
        }
        for properties, error in (
            (SimpleNamespace(uuid=_GPU_B), RuntimeError),
            (SimpleNamespace(), RuntimeError),
        ):
            with self.subTest(properties=properties):
                with (
                    patch.dict(os.environ, environment, clear=False),
                    patch.object(runtime.torch.cuda, "device_count", return_value=1),
                    patch.object(runtime.torch.cuda, "current_device", return_value=0),
                    patch.object(
                        runtime.torch.cuda,
                        "get_device_properties",
                        return_value=properties,
                    ),
                    self.assertRaises(error),
                ):
                    runtime._bind_torch_logical_cuda_zero_uuid()


class M04aCheckpointHandleTests(unittest.TestCase):
    def test_hash_and_deserialization_share_exactly_one_path_open(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "checkpoint.pt"
            runtime.torch.save(_checkpoint_payload(), checkpoint)
            expected_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            original_path_open = Path.open
            original_torch_load = runtime.torch.load
            checkpoint_open_count = 0
            load_sources: list[object] = []
            replacement = io.BytesIO()
            replacement_payload = _checkpoint_payload()
            replacement_payload["optimizer_step"] = 8
            replacement_payload["rng_state"]["optimizer_step"] = 8
            runtime.torch.save(replacement_payload, replacement)

            def tracked_path_open(path: Path, *args: object, **kwargs: object):
                nonlocal checkpoint_open_count
                if path == checkpoint.absolute() and args[:1] == ("rb",):
                    checkpoint_open_count += 1
                return original_path_open(path, *args, **kwargs)

            def tracked_torch_load(source: object, *args: object, **kwargs: object):
                load_sources.append(source)
                checkpoint.write_bytes(replacement.getvalue())
                return original_torch_load(source, *args, **kwargs)

            with (
                patch.object(Path, "open", new=tracked_path_open),
                patch.object(runtime.torch, "load", side_effect=tracked_torch_load),
                patch.object(
                    runtime,
                    "_sha256_file",
                    side_effect=AssertionError("load_checkpoint reopened the path"),
                ),
            ):
                payload = runtime.load_checkpoint(
                    checkpoint,
                    expected_sha256=expected_sha256,
                    expected_optimizer_step=7,
                    expected_config_sha256=_CONFIG_SHA256,
                    expected_runtime_source_sha256=_RUNTIME_SHA256,
                    expected_test_source_sha256=_TEST_SHA256,
                    expected_bytes=checkpoint.stat().st_size,
                )

            self.assertEqual(payload["optimizer_step"], 7)
            self.assertEqual(checkpoint_open_count, 1)
            self.assertEqual(len(load_sources), 1)
            self.assertIsInstance(load_sources[0], io.BytesIO)
            self.assertFalse(isinstance(load_sources[0], (str, os.PathLike)))


if __name__ == "__main__":
    unittest.main()
