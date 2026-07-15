"""Optional PyTorch runtime controls for the frozen M04a neural stages.

Nothing in the default package import graph imports this module.  Neural CLI
handlers load it lazily inside a process whose launcher has already set the frozen
cuBLAS workspace configuration and isolated import paths.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import re
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

import torch

from .m04a_contract import MODEL_SEMANTICS_VERSION, OPTIMIZER_UPDATES, TRAINING_SEED


CUBLAS_WORKSPACE_CONFIG = ":4096:8"
CHECKPOINT_SCHEMA_VERSION = "afts-grid-cmlm-checkpoint/v0.1"
RNG_STATE_SCHEMA_VERSION = "afts-grid-cmlm-rng-state/v0.1"
ENVIRONMENT_SCHEMA_VERSION = "afts-m04a-environment/v0.3"
FROZEN_PYTHON_VERSION = (3, 10, 20)
FROZEN_TORCH_VERSION = "2.10.0+cu128"
FROZEN_TORCH_CUDA_VERSION = "12.8"
FROZEN_CUDNN_VERSION = 91_002
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
_RUNTIME_ATTESTATION_TOKEN = object()
_NVIDIA_GPU_UUID_PATTERN = re.compile(
    r"(?:GPU-)?([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)

_T = TypeVar("_T")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_nvidia_gpu_uuid(value: object) -> str:
    """Return the one canonical full-GPU UUID representation used by M04a."""

    if not isinstance(value, str):
        raise TypeError("NVIDIA GPU UUID must be a string")
    match = _NVIDIA_GPU_UUID_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("NVIDIA GPU UUID must be a full non-MIG UUID")
    return f"GPU-{match.group(1).lower()}"


def _torch_cuda_device_uuid_text(value: object) -> str:
    """Read the frozen Torch ``_CUuuid`` without accepting arbitrary objects."""

    if isinstance(value, str):
        return value
    expected_type = getattr(torch._C, "_CUuuid", None)
    if not isinstance(expected_type, type) or type(value) is not expected_type:
        raise TypeError("PyTorch CUDA UUID must be a string or exact torch._C._CUuuid")
    raw = getattr(value, "bytes", None)
    if (
        type(raw) is not list
        or len(raw) != 16
        or any(type(item) is not int or not 0 <= item <= 255 for item in raw)
    ):
        raise TypeError("PyTorch _CUuuid bytes must be 16 exact byte integers")
    compact = bytes(raw).hex()
    from_bytes = (
        f"{compact[:8]}-{compact[8:12]}-{compact[12:16]}-"
        f"{compact[16:20]}-{compact[20:]}"
    )
    if normalize_nvidia_gpu_uuid(str(value)) != normalize_nvidia_gpu_uuid(from_bytes):
        raise ValueError("PyTorch _CUuuid string and bytes disagree")
    return from_bytes


def bind_single_visible_cuda_uuid(
    *,
    configured_gpu_uuid: object,
    cuda_visible_devices: object,
    visible_device_count: object,
    current_device_index: object,
    logical_device_uuid: object,
) -> str:
    """Purely close the launch UUID against PyTorch logical ``cuda:0`` metadata."""

    configured = normalize_nvidia_gpu_uuid(configured_gpu_uuid)
    visible_selector = normalize_nvidia_gpu_uuid(cuda_visible_devices)
    logical = normalize_nvidia_gpu_uuid(logical_device_uuid)
    if type(visible_device_count) is not int or visible_device_count != 1:
        raise RuntimeError("M04a requires exactly one visible CUDA device")
    if type(current_device_index) is not int or current_device_index != 0:
        raise RuntimeError("M04a requires logical CUDA device 0 to be current")
    if visible_selector != configured:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES does not select the configured GPU UUID"
        )
    if logical != configured:
        raise RuntimeError(
            "PyTorch logical cuda:0 UUID differs from the configured GPU UUID"
        )
    return configured


def _validate_ordered_import_roots(
    import_roots: object,
    *,
    bound_import_roots: object,
    ordered_sys_path: object,
    require_live_fds: bool = False,
) -> list[str]:
    if type(import_roots) is not list or len(import_roots) not in {2, 3}:
        raise ValueError(
            "isolated import roots must contain exactly two or three paths"
        )
    normalized: list[str] = []
    for path in import_roots:
        if not isinstance(path, str) or not path:
            raise ValueError("isolated import roots must be non-empty strings")
        parsed = PurePosixPath(path)
        if (
            not parsed.is_absolute()
            or parsed.anchor != "/"
            or parsed.as_posix() != path
            or path == "/"
            or path.endswith("/")
            or any(part in {"", ".", ".."} for part in parsed.parts[1:])
        ):
            raise ValueError(
                "isolated import roots must be absolute lexical POSIX paths"
            )
        normalized.append(path)
    if len(set(normalized)) != len(normalized):
        raise ValueError("isolated logical import roots must not contain duplicates")
    expected = [f"/proc/self/fd/{200 + index}" for index in range(len(normalized))]
    if bound_import_roots != expected:
        raise ValueError("bound import roots must be the fixed inherited FD paths")
    if (
        not isinstance(ordered_sys_path, list)
        or ordered_sys_path[-len(normalized) :] != expected
    ):
        raise ValueError("ordered isolated import roots differ from sys.path")
    if require_live_fds:
        for index, path in enumerate(expected):
            descriptor = 200 + index
            info = os.fstat(descriptor)
            path_info = os.stat(path)
            if (
                not stat.S_ISDIR(info.st_mode)
                or not os.path.samestat(info, path_info)
                or not os.get_inheritable(descriptor)
                or Path(path).resolve().as_posix() != normalized[index]
            ):
                raise ValueError("fixed import descriptor is not live and inherited")
    return normalized


def _bind_torch_logical_cuda_zero_uuid() -> str:
    properties = torch.cuda.get_device_properties(0)
    try:
        return bind_single_visible_cuda_uuid(
            configured_gpu_uuid=os.environ.get("AFTS_M04A_GPU_UUID"),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            visible_device_count=torch.cuda.device_count(),
            current_device_index=torch.cuda.current_device(),
            logical_device_uuid=_torch_cuda_device_uuid_text(
                getattr(properties, "uuid", None)
            ),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "PyTorch logical cuda:0 did not expose a valid full GPU UUID"
        ) from exc


def assert_launcher_environment() -> None:
    """Fail before the first CUDA tensor if the reviewed launcher was bypassed."""

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            f"CUBLAS_WORKSPACE_CONFIG must already equal {CUBLAS_WORKSPACE_CONFIG}"
        )
    if (
        sys.implementation.name != "cpython"
        or sys.version_info[:3] != FROZEN_PYTHON_VERSION
    ):
        raise RuntimeError("M04a requires exact CPython 3.10.20")
    if torch.__version__ != FROZEN_TORCH_VERSION:
        raise RuntimeError(
            f"M04a requires torch {FROZEN_TORCH_VERSION}; found {torch.__version__}"
        )
    if torch.version.cuda != FROZEN_TORCH_CUDA_VERSION:
        raise RuntimeError(
            f"M04a requires the PyTorch CUDA 12.8 runtime; found {torch.version.cuda}"
        )
    if torch.backends.cudnn.version() != FROZEN_CUDNN_VERSION:
        raise RuntimeError(
            f"M04a requires cuDNN {FROZEN_CUDNN_VERSION}; "
            f"found {torch.backends.cudnn.version()}"
        )
    if os.environ.get("AFTS_M04A_TORCH_MODE") != "1":
        raise RuntimeError("AFTS_M04A_TORCH_MODE=1 is required")
    if os.environ.get("AFTS_EVIDENCE_FRESH_SOURCE_LOADER") != "1":
        raise RuntimeError("fresh isolated source loader marker is required")
    if not os.environ.get("AFTS_M04A_RUN_ID"):
        raise RuntimeError("AFTS_M04A_RUN_ID is required")
    if not os.environ.get("AFTS_M04A_GPU_UUID"):
        raise RuntimeError("AFTS_M04A_GPU_UUID is required")
    try:
        configured_gpu_uuid = normalize_nvidia_gpu_uuid(
            os.environ["AFTS_M04A_GPU_UUID"]
        )
        visible_gpu_uuid = normalize_nvidia_gpu_uuid(
            os.environ.get("CUDA_VISIBLE_DEVICES")
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("M04a requires full non-MIG GPU UUID selectors") from exc
    if visible_gpu_uuid != configured_gpu_uuid:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be the configured full GPU UUID")


@dataclass(frozen=True, slots=True)
class FrozenRuntimeAttestation:
    run_id: str
    gpu_uuid: str
    logical_device_index: int
    _token: object

    def __post_init__(self) -> None:
        if self._token is not _RUNTIME_ATTESTATION_TOKEN:
            raise ValueError("runtime attestation cannot be constructed externally")
        if (
            not self.run_id
            or self.logical_device_index != 0
            or self.gpu_uuid != normalize_nvidia_gpu_uuid(self.gpu_uuid)
        ):
            raise ValueError("invalid frozen runtime attestation")


def validate_runtime_attestation(value: object) -> FrozenRuntimeAttestation:
    if not isinstance(value, FrozenRuntimeAttestation) or value._token is not (
        _RUNTIME_ATTESTATION_TOKEN
    ):
        raise TypeError("a configure_deterministic_cuda attestation is required")
    assert_launcher_environment()
    actual_gpu_uuid = _bind_torch_logical_cuda_zero_uuid()
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("deterministic algorithms are no longer enabled")
    if torch.backends.cudnn.benchmark:
        raise RuntimeError("cuDNN benchmark is no longer disabled")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32 is no longer disabled")
    if (
        torch.backends.cuda.flash_sdp_enabled()
        or torch.backends.cuda.mem_efficient_sdp_enabled()
        or not torch.backends.cuda.math_sdp_enabled()
    ):
        raise RuntimeError("the mathematical SDPA policy is no longer active")
    if (
        value.run_id != os.environ.get("AFTS_M04A_RUN_ID")
        or value.gpu_uuid != actual_gpu_uuid
    ):
        raise RuntimeError("runtime attestation environment identity changed")
    return value


def configure_deterministic_cuda(
    *, seed: int = TRAINING_SEED
) -> FrozenRuntimeAttestation:
    """Apply the exact deterministic CUDA policy before model construction."""

    assert_launcher_environment()
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if not torch.cuda.is_available():
        raise RuntimeError("M04a neural stages require CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one GPU must be visible to the M04a process")
    gpu_uuid = _bind_torch_logical_cuda_zero_uuid()
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected GPU must support BF16")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("deterministic algorithms were not enabled")
    if torch.backends.cudnn.benchmark:
        raise RuntimeError("cuDNN benchmark must remain disabled")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32 must remain disabled")
    if torch.backends.cuda.flash_sdp_enabled():
        raise RuntimeError("flash SDPA must remain disabled")
    if torch.backends.cuda.mem_efficient_sdp_enabled():
        raise RuntimeError("memory-efficient SDPA must remain disabled")
    if not torch.backends.cuda.math_sdp_enabled():
        raise RuntimeError("mathematical SDPA must remain enabled")
    if torch.cuda.current_device() != 0:
        raise RuntimeError("the sole visible CUDA device must have logical index 0")
    return FrozenRuntimeAttestation(
        run_id=os.environ["AFTS_M04A_RUN_ID"],
        gpu_uuid=gpu_uuid,
        logical_device_index=0,
        _token=_RUNTIME_ATTESTATION_TOKEN,
    )


def bf16_autocast() -> AbstractContextManager[Any]:
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)


@dataclass(frozen=True, slots=True)
class CUDATiming:
    gpu_ns: int
    wall_ns: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int

    def __post_init__(self) -> None:
        for field, value in asdict(self).items():
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")


def timed_cuda_call(function: Callable[[], _T]) -> tuple[_T, CUDATiming]:
    """Measure one nonoverlapping GPU phase with the frozen synchronization boundary."""

    if not callable(function):
        raise TypeError("function must be callable")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA timing requires an available GPU")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter_ns()
    start_event.record()
    result = function()
    end_event.record()
    end_event.synchronize()
    wall_end = time.perf_counter_ns()
    gpu_ns = int(round(start_event.elapsed_time(end_event) * 1_000_000.0))
    timing = CUDATiming(
        gpu_ns=gpu_ns,
        wall_ns=wall_end - wall_start,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
    )
    return result, timing


def capture_rng_state(*, optimizer_step: int) -> dict[str, object]:
    if type(optimizer_step) is not int or optimizer_step < 0:
        raise ValueError("optimizer_step must be a non-negative integer")
    return {
        "schema": RNG_STATE_SCHEMA_VERSION,
        "optimizer_step": optimizer_step,
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def restore_rng_state(payload: Mapping[str, object]) -> int:
    if payload.get("schema") != RNG_STATE_SCHEMA_VERSION:
        raise ValueError("RNG state schema mismatch")
    optimizer_step = payload.get("optimizer_step")
    if type(optimizer_step) is not int or optimizer_step < 0:
        raise ValueError("invalid RNG optimizer_step")
    cpu_state = payload.get("torch_cpu")
    cuda_states = payload.get("torch_cuda")
    if not isinstance(cpu_state, torch.Tensor) or cpu_state.dtype != torch.uint8:
        raise TypeError("invalid CPU RNG tensor")
    if (
        not isinstance(cuda_states, list)
        or len(cuda_states) != torch.cuda.device_count()
    ):
        raise ValueError("CUDA RNG state count does not match visible devices")
    if any(
        not isinstance(state, torch.Tensor) or state.dtype != torch.uint8
        for state in cuda_states
    ):
        raise TypeError("invalid CUDA RNG tensor")
    torch.set_rng_state(cpu_state.cpu())
    torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
    return optimizer_step


def save_checkpoint_new(
    path: str | Path,
    *,
    optimizer_step: int,
    model_state: Mapping[str, object],
    optimizer_state: Mapping[str, object],
    config_sha256: str,
    runtime_source_sha256: str,
    test_source_sha256: str,
) -> dict[str, object]:
    """Serialize a resume checkpoint through an exclusive file handle."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if type(optimizer_step) is not int or not 0 <= optimizer_step <= OPTIMIZER_UPDATES:
        raise ValueError(f"optimizer_step must be in 0..{OPTIMIZER_UPDATES}")
    if not isinstance(model_state, Mapping) or not isinstance(optimizer_state, Mapping):
        raise TypeError("model and optimizer states must be mappings")
    for name, digest in (
        ("config_sha256", config_sha256),
        ("runtime_source_sha256", runtime_source_sha256),
        ("test_source_sha256", test_source_sha256),
    ):
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{name} must be a SHA-256 hex digest")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(f"{name} must be hexadecimal") from exc
    payload = {
        "schema": CHECKPOINT_SCHEMA_VERSION,
        "model_semantics_version": MODEL_SEMANTICS_VERSION,
        "optimizer_step": optimizer_step,
        "model_state": dict(model_state),
        "optimizer_state": dict(optimizer_state),
        "rng_state": capture_rng_state(optimizer_step=optimizer_step),
        "config_sha256": config_sha256.lower(),
        "runtime_source_sha256": runtime_source_sha256.lower(),
        "test_source_sha256": test_source_sha256.lower(),
    }
    with target.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "schema": CHECKPOINT_SCHEMA_VERSION,
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": _sha256_file(target),
        "optimizer_step": optimizer_step,
    }


def load_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_optimizer_step: int,
    expected_config_sha256: str,
    expected_runtime_source_sha256: str,
    expected_test_source_sha256: str,
    expected_bytes: int | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    source = Path(path).expanduser().absolute()
    identities = {
        "expected_sha256": expected_sha256,
        "expected_config_sha256": expected_config_sha256,
        "expected_runtime_source_sha256": expected_runtime_source_sha256,
        "expected_test_source_sha256": expected_test_source_sha256,
    }
    for field, digest in identities.items():
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{field} must be lowercase SHA-256")
    if (
        type(expected_optimizer_step) is not int
        or not 0 <= expected_optimizer_step <= OPTIMIZER_UPDATES
    ):
        raise ValueError(
            f"expected_optimizer_step must be in 0..{OPTIMIZER_UPDATES}"
        )
    if expected_bytes is not None and (
        type(expected_bytes) is not int or expected_bytes <= 0
    ):
        raise ValueError("expected_bytes must be a positive integer when supplied")
    if source.is_symlink():
        raise ValueError("checkpoint path cannot be a symlink")
    with source.open("rb") as handle:
        opened_stat = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened_stat.st_mode):
            raise ValueError("checkpoint must be a regular file")
        if opened_stat.st_size <= 0 or opened_stat.st_size > MAX_CHECKPOINT_BYTES:
            raise ValueError("checkpoint size exceeds the frozen host-memory bound")
        if expected_bytes is not None and opened_stat.st_size != expected_bytes:
            raise ValueError("checkpoint byte count differs from the artifact manifest")
        path_stat = source.stat(follow_symlinks=False)
        if not stat.S_ISREG(path_stat.st_mode) or not os.path.samestat(
            opened_stat, path_stat
        ):
            raise RuntimeError("checkpoint path identity changed while opening")

        checkpoint_bytes = handle.read(opened_stat.st_size + 1)
        if len(checkpoint_bytes) != opened_stat.st_size:
            raise RuntimeError("checkpoint byte count changed while reading")
        if hashlib.sha256(checkpoint_bytes).hexdigest() != expected_sha256:
            raise ValueError("checkpoint SHA-256 mismatch")
        final_stat = os.fstat(handle.fileno())
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        if any(
            getattr(final_stat, field) != getattr(opened_stat, field)
            for field in stable_fields
        ):
            raise RuntimeError("checkpoint file changed while reading immutable bytes")
    payload = torch.load(
        io.BytesIO(checkpoint_bytes),
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise TypeError("checkpoint root must be a mapping")
    if payload.get("schema") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema mismatch")
    if payload.get("model_semantics_version") != MODEL_SEMANTICS_VERSION:
        raise ValueError("checkpoint model semantics mismatch")
    required = {
        "schema",
        "model_semantics_version",
        "optimizer_step",
        "model_state",
        "optimizer_state",
        "rng_state",
        "config_sha256",
        "runtime_source_sha256",
        "test_source_sha256",
    }
    if set(payload) != required:
        raise ValueError("checkpoint fields do not match the frozen schema")
    if payload["optimizer_step"] != expected_optimizer_step:
        raise ValueError("checkpoint optimizer step mismatch")
    for field, expected in (
        ("config_sha256", expected_config_sha256),
        ("runtime_source_sha256", expected_runtime_source_sha256),
        ("test_source_sha256", expected_test_source_sha256),
    ):
        if payload[field] != expected:
            raise ValueError(f"checkpoint {field} mismatch")
    if not isinstance(payload["model_state"], dict) or not isinstance(
        payload["optimizer_state"], dict
    ):
        raise TypeError("checkpoint model/optimizer states must be dictionaries")
    rng_state = payload["rng_state"]
    if not isinstance(rng_state, dict) or set(rng_state) != {
        "schema",
        "optimizer_step",
        "torch_cpu",
        "torch_cuda",
    }:
        raise ValueError("checkpoint RNG state schema mismatch")
    if (
        rng_state.get("schema") != RNG_STATE_SCHEMA_VERSION
        or rng_state.get("optimizer_step") != expected_optimizer_step
    ):
        raise ValueError("checkpoint RNG state identity mismatch")
    cpu_state = rng_state.get("torch_cpu")
    cuda_states = rng_state.get("torch_cuda")
    if not isinstance(cpu_state, torch.Tensor) or cpu_state.dtype != torch.uint8:
        raise TypeError("checkpoint CPU RNG state is invalid")
    if not isinstance(cuda_states, list) or any(
        not isinstance(state, torch.Tensor) or state.dtype != torch.uint8
        for state in cuda_states
    ):
        raise TypeError("checkpoint CUDA RNG states are invalid")
    return payload


def environment_manifest(
    *,
    attestation: FrozenRuntimeAttestation,
    launcher_path: str | Path,
    runtime_source_sha256: str,
    test_source_sha256: str,
    visible_root: str | Path,
    expected_visible_files: Mapping[str, str],
    conda_explicit_path: str | Path,
    expected_conda_explicit_sha256: str,
    python_runtime_lock_sha256: str,
    python_runtime_lock_id: str,
    python_implementation: str,
    python_version: str,
    python_executable_sha256: str,
    python_executable_bytes: int,
) -> dict[str, object]:
    """Capture the neural runtime after launcher and data visibility checks."""

    validate_runtime_attestation(attestation)
    launcher_requested = Path(launcher_path).expanduser()
    if launcher_requested.is_symlink():
        raise ValueError("neural launcher cannot be a symlink")
    launcher = launcher_requested.resolve()
    if not launcher.is_file() or not stat.S_ISREG(launcher.stat().st_mode):
        raise FileNotFoundError(launcher)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("environment manifest requires exactly one visible GPU")
    properties = torch.cuda.get_device_properties(0)
    requested_visible_root = Path(visible_root).expanduser()
    if requested_visible_root.is_symlink():
        raise ValueError("visible input root cannot be a symlink")
    inventory_root = requested_visible_root.resolve()
    if not inventory_root.is_dir():
        raise ValueError("visible input root must be a directory")
    files: dict[str, str] = {}
    for path in sorted(inventory_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"visible input contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError(f"visible input contains a non-regular file: {path}")
        relative = path.relative_to(inventory_root).as_posix()
        files[relative] = _sha256_file(path)
        if len(files) > 10_000:
            raise ValueError("visible input inventory exceeds 10000 files")
    if not isinstance(expected_visible_files, Mapping):
        raise TypeError("expected visible inventory must be a mapping")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for name, digest in expected_visible_files.items()
    ):
        raise ValueError("expected visible inventory is malformed")
    expected_files = dict(sorted(expected_visible_files.items()))
    if not expected_files:
        raise ValueError("expected visible inventory must not be empty")
    if files != expected_files:
        raise ValueError("actual visible file inventory differs from launch plan")

    conda_requested = Path(conda_explicit_path).expanduser()
    if conda_requested.is_symlink():
        raise ValueError("conda explicit list cannot be a symlink")
    conda_explicit = conda_requested.resolve()
    if not conda_explicit.is_file():
        raise FileNotFoundError(conda_explicit)
    if (
        not isinstance(expected_conda_explicit_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_conda_explicit_sha256) is None
    ):
        raise ValueError("expected conda explicit digest must be SHA-256")
    conda_sha256 = _sha256_file(conda_explicit)
    if conda_sha256 != expected_conda_explicit_sha256:
        raise ValueError("conda explicit list differs from launch plan")
    if not conda_explicit.is_relative_to(inventory_root):
        raise ValueError("conda explicit list must be inside the visible input root")
    for field, digest in (
        ("runtime_source_sha256", runtime_source_sha256),
        ("test_source_sha256", test_source_sha256),
        ("python_runtime_lock_sha256", python_runtime_lock_sha256),
        ("python_runtime_lock_id", python_runtime_lock_id),
    ):
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{field} must be lowercase SHA-256")
    if (
        python_implementation != platform.python_implementation()
        or python_version != platform.python_version()
        or sys.version.split(maxsplit=1)[0] != python_version
    ):
        raise ValueError("live Python implementation/version differs from runtime lock")
    executable = Path(sys.executable)
    if (
        not executable.is_file()
        or type(python_executable_bytes) is not int
        or python_executable_bytes <= 0
        or executable.stat().st_size != python_executable_bytes
        or not isinstance(python_executable_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", python_executable_sha256) is None
        or _sha256_file(executable) != python_executable_sha256
    ):
        raise ValueError("live Python executable differs from runtime lock")

    raw_import_roots = os.environ.get("AFTS_M04A_IMPORT_ROOTS_JSON")
    raw_bound_import_roots = os.environ.get("AFTS_M04A_BOUND_IMPORT_ROOTS_JSON")
    try:
        import_roots = json.loads(raw_import_roots or "")
        bound_import_roots = json.loads(raw_bound_import_roots or "")
    except json.JSONDecodeError as exc:
        raise ValueError("launcher import-root record is invalid") from exc
    try:
        import_roots = _validate_ordered_import_roots(
            import_roots,
            bound_import_roots=bound_import_roots,
            ordered_sys_path=sys.path,
            require_live_fds=True,
        )
    except ValueError as exc:
        raise ValueError(
            "ordered isolated import roots differ from launcher record"
        ) from exc

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        raise RuntimeError("nvidia-smi driver query failed")
    raw_driver_rows = [
        tuple(part.strip() for part in line.split(",", 1))
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    driver_rows: list[tuple[str, str]] = []
    for row in raw_driver_rows:
        if len(row) != 2 or not row[1]:
            raise RuntimeError("nvidia-smi returned a malformed driver row")
        try:
            row_uuid = normalize_nvidia_gpu_uuid(row[0])
        except (TypeError, ValueError) as exc:
            raise RuntimeError("nvidia-smi returned a malformed GPU UUID") from exc
        driver_rows.append((row_uuid, row[1]))
    matching_driver_rows = [
        row for row in driver_rows if row[0] == attestation.gpu_uuid
    ]
    if len(matching_driver_rows) != 1 or len(matching_driver_rows[0]) != 2:
        raise RuntimeError("selected GPU UUID does not bind one driver row")
    driver_version = matching_driver_rows[0][1]
    return {
        "schema": ENVIRONMENT_SCHEMA_VERSION,
        "python": sys.version,
        "python_implementation": python_implementation,
        "python_version": python_version,
        "python_executable": sys.executable,
        "python_executable_sha256": python_executable_sha256,
        "python_executable_bytes": python_executable_bytes,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_driver": driver_version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_uuid": attestation.gpu_uuid,
        "run_id": os.environ["AFTS_M04A_RUN_ID"],
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "gpu": {
            "name": properties.name,
            "total_memory": int(properties.total_memory),
            "major": int(properties.major),
            "minor": int(properties.minor),
        },
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp": torch.backends.cuda.math_sdp_enabled(),
        "launcher_path": str(launcher),
        "launcher_sha256": _sha256_file(launcher),
        "ordered_sys_path": list(sys.path),
        "ordered_import_roots": import_roots,
        "conda_explicit_path": str(conda_explicit),
        "conda_explicit_sha256": conda_sha256,
        "visible_root": str(inventory_root),
        "runtime_source_sha256": runtime_source_sha256,
        "test_source_sha256": test_source_sha256,
        "visible_files": files,
        "python_runtime_lock_sha256": python_runtime_lock_sha256,
        "python_runtime_lock_id": python_runtime_lock_id,
    }


def canonical_environment_sha256(payload: Mapping[str, object]) -> str:
    serialized = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDATiming",
    "ENVIRONMENT_SCHEMA_VERSION",
    "FROZEN_CUDNN_VERSION",
    "FROZEN_PYTHON_VERSION",
    "FROZEN_TORCH_CUDA_VERSION",
    "FROZEN_TORCH_VERSION",
    "MAX_CHECKPOINT_BYTES",
    "FrozenRuntimeAttestation",
    "RNG_STATE_SCHEMA_VERSION",
    "assert_launcher_environment",
    "bind_single_visible_cuda_uuid",
    "bf16_autocast",
    "canonical_environment_sha256",
    "capture_rng_state",
    "configure_deterministic_cuda",
    "environment_manifest",
    "load_checkpoint",
    "normalize_nvidia_gpu_uuid",
    "restore_rng_state",
    "save_checkpoint_new",
    "timed_cuda_call",
    "validate_runtime_attestation",
]
