"""Frozen deployment sampler for the M04a Grid-CMLM.

This is an explicit PyTorch boundary.  The primary entry points accept only an
oracle-free :class:`BlindTask`, its validated blind shape-sidecar row, and an
already-loaded checkpoint model.  A CPU mode exists solely for the explicit fake
model suite; real deployment requires one CUDA BF16 task-memory cache per test pair.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import stat
import time
import weakref
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

import torch
from torch import Tensor

from .blind import BlindTask
from .grid import as_grid, grid_key, grid_to_lists
from .m04a_contract import (
    DENOISING_STEPS,
    INFERENCE_MICROBATCH,
    LANE_TRACE_SCHEMA_VERSION,
    MODEL_PARAMETER_COUNT,
    MODEL_SEMANTICS_VERSION,
    OPTIMIZER_UPDATES,
    OUTPUT_COLOR_COUNT,
    canonical_sha256,
    float64_sequence_sha256,
    lane_seed,
    mask_count_trace,
    shape_lane_allocations,
)
from .m04a_evidence import (
    BlindInputShapeSidecar,
    make_decoder_forward_row,
    make_encoder_forward_row,
    make_lane_row,
    make_pair_cost_row,
    validate_blind_input_row,
    validate_blind_input_shape_sidecar,
    validate_pair_evidence,
)
from .m04a_model import (
    MASK_TOKEN_ID,
    EncodedTaskMemory,
    GridCMLM,
    TokenBatch,
    parameter_state_sha256,
    trainable_parameter_count,
    tokenize_target_batch,
    tokenize_task_memory,
)
from .m04a_pool_staging import M04APoolPairStagingSink
from .m04a_torch_runtime import (
    FrozenRuntimeAttestation,
    load_checkpoint,
    validate_runtime_attestation,
)
from .manifest import serialize_jsonl


NONFINITE_LOGITS = "NONFINITE_LOGITS"
_MASK = "MASK"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_VERIFIED_CHECKPOINT_MODEL_TOKEN = object()
_FIXTURE_CHECKPOINT_MODEL_TOKEN = object()
_POOL_INTEGRITY_ATTESTATION_TOKEN = object()
_PRODUCTION_POOL_APPEND_PAIR = M04APoolPairStagingSink.append_pair
_PRODUCTION_POOL_FINALIZE = M04APoolPairStagingSink.finalize
_PRODUCTION_POOL_ABORT = M04APoolPairStagingSink.abort
_PRODUCTION_POOL_CLOSE_STREAMS = M04APoolPairStagingSink._close_streams
_PRODUCTION_POOL_STREAM_FILES = (
    "lane_traces.jsonl",
    "candidate_rows.jsonl",
    "encoder_forward_ledger.jsonl",
    "batch_forward_ledger.jsonl",
)
_SHA256_HASH_TYPE = type(hashlib.sha256())
_PRODUCTION_POOL_SINK_METHODS = tuple(
    (
        ("append_pair", _PRODUCTION_POOL_APPEND_PAIR),
        ("finalize", _PRODUCTION_POOL_FINALIZE),
        ("abort", _PRODUCTION_POOL_ABORT),
        ("_close_streams", _PRODUCTION_POOL_CLOSE_STREAMS),
    )
)
_MODEL_PARAMETER_VALUE_SEALS: dict[
    int,
    tuple[
        weakref.ReferenceType[GridCMLM],
        tuple[tuple[str, Tensor], ...],
        object,
    ],
] = {}


class NonFiniteLogitsError(RuntimeError):
    """A decoder produced a non-finite value; no pair result is returned."""

    failure_code = NONFINITE_LOGITS


class PoolSamplingAborted(RuntimeError):
    """The whole pool is incomplete and exposes no partial pair results."""

    failure_code = NONFINITE_LOGITS


class PoolArtifactSink(Protocol):
    """Exclusive incomplete-staging sink used by production pool generation."""

    def append_pair(
        self,
        *,
        blind_task_id: str,
        test_index: int,
        materials: Mapping[str, bytes],
    ) -> Mapping[str, object]: ...

    def finalize(
        self, *, pair_cost_rows: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, object]: ...

    def abort(self, *, failure_code: str) -> None: ...


def _require_exact_production_pool_sink(
    artifact_sink: object,
) -> M04APoolPairStagingSink:
    if type(artifact_sink) is not M04APoolPairStagingSink:
        raise TypeError(
            "production sampling requires the exact reviewed pool staging sink"
        )
    for method_name, expected_function in _PRODUCTION_POOL_SINK_METHODS:
        bound_method = getattr(artifact_sink, method_name, None)
        if (
            getattr(bound_method, "__self__", None) is not artifact_sink
            or getattr(bound_method, "__func__", None) is not expected_function
        ):
            raise TypeError(
                "production pool staging sink methods cannot be overridden"
            )
    state = object.__getattribute__(artifact_sink, "__dict__")
    if set(state) != {
        "_root",
        "_expected_pairs",
        "_next_pair",
        "_closed",
        "_aborted",
        "_handles",
        "_digests",
        "_bytes",
        "_rows",
    }:
        raise TypeError("production pool staging sink state fields changed")
    root = state["_root"]
    expected_pairs = state["_expected_pairs"]
    if (
        type(root) is not type(Path())
        or not root.is_absolute()
        or not root.is_dir()
        or type(expected_pairs) is not tuple
        or not expected_pairs
        or any(
            type(pair) is not tuple
            or len(pair) != 2
            or not isinstance(pair[0], str)
            or not pair[0]
            or type(pair[1]) is not int
            or pair[1] < 0
            for pair in expected_pairs
        )
        or len(expected_pairs) != len(set(expected_pairs))
        or type(state["_next_pair"]) is not int
        or not 0 <= state["_next_pair"] <= len(expected_pairs)
        or state["_closed"] is not False
        or state["_aborted"] is not False
    ):
        raise TypeError("production pool staging sink scalar state changed")
    handles = state["_handles"]
    digests = state["_digests"]
    byte_counts = state["_bytes"]
    row_counts = state["_rows"]
    expected_names = set(_PRODUCTION_POOL_STREAM_FILES)
    if any(
        type(mapping) is not dict or set(mapping) != expected_names
        for mapping in (handles, digests, byte_counts, row_counts)
    ):
        raise TypeError("production pool staging sink stream maps changed")
    for name in _PRODUCTION_POOL_STREAM_FILES:
        handle = handles[name]
        if (
            type(handle) is not io.BufferedWriter
            or handle.closed
            or not handle.writable()
            or Path(handle.name).resolve() != root / name
        ):
            raise TypeError("production pool staging sink file handle changed")
        opened_stat = os.fstat(handle.fileno())
        path_stat = (root / name).stat(follow_symlinks=False)
        if not stat.S_ISREG(opened_stat.st_mode) or not os.path.samestat(
            opened_stat, path_stat
        ):
            raise TypeError("production pool staging sink file identity changed")
        if type(digests[name]) is not _SHA256_HASH_TYPE:
            raise TypeError("production pool staging sink digest changed")
        if (
            type(byte_counts[name]) is not int
            or byte_counts[name] < 0
            or type(row_counts[name]) is not int
            or row_counts[name] < 0
        ):
            raise TypeError("production pool staging sink counters changed")
    return artifact_sink


def _sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TypeError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _seal_model_parameter_values(
    model: GridCMLM,
) -> tuple[tuple[str, Tensor], ...]:
    """Clone exact parameter and buffer values on their execution device."""

    return tuple(
        (f"parameter:{name}", parameter.detach().clone())
        for name, parameter in model.named_parameters()
    ) + tuple(
        (f"buffer:{name}", buffer.detach().clone())
        for name, buffer in model.named_buffers()
    )


def _parameter_value_seal_is_independent(
    model: GridCMLM,
    seal: tuple[tuple[str, Tensor], ...],
) -> bool:
    state_tensors = tuple(
        (f"parameter:{name}", parameter)
        for name, parameter in model.named_parameters()
    ) + tuple(
        (f"buffer:{name}", buffer)
        for name, buffer in model.named_buffers()
    )
    if len(state_tensors) != len(seal):
        return False
    for (name, parameter), sealed_entry in zip(state_tensors, seal):
        if (
            not isinstance(sealed_entry, tuple)
            or len(sealed_entry) != 2
            or sealed_entry[0] != name
            or not isinstance(sealed_entry[1], Tensor)
        ):
            return False
        sealed = sealed_entry[1]
        if (
            sealed is parameter
            or sealed.data_ptr() == parameter.data_ptr()
            or tuple(sealed.shape) != tuple(parameter.shape)
            or sealed.dtype != parameter.dtype
            or sealed.device != parameter.device
            or sealed.requires_grad
        ):
            return False
    return True


def _model_parameter_values_match_seal(
    model: GridCMLM,
    seal: tuple[tuple[str, Tensor], ...],
) -> bool:
    """Compare exact values on-device and return one aggregate boolean to host."""

    if not _parameter_value_seal_is_independent(model, seal):
        return False
    state_tensors = tuple(model.named_parameters()) + tuple(model.named_buffers())
    aggregate: Tensor | None = None
    for (_, current), (_, sealed) in zip(state_tensors, seal):
        matches = torch.eq(current.detach(), sealed).all()
        aggregate = matches if aggregate is None else torch.logical_and(
            aggregate, matches
        )
    return aggregate is not None and bool(aggregate.item())


def _register_model_parameter_value_seal(model: GridCMLM) -> object:
    """Keep mutable device clones in a module-private weak model registry."""

    seal = _seal_model_parameter_values(model)
    if not _parameter_value_seal_is_independent(model, seal):
        raise RuntimeError("could not create an independent parameter value seal")
    key = id(model)
    token = object()

    def discard(reference: weakref.ReferenceType[GridCMLM]) -> None:
        current = _MODEL_PARAMETER_VALUE_SEALS.get(key)
        if current is not None and current[0] is reference and current[2] is token:
            _MODEL_PARAMETER_VALUE_SEALS.pop(key, None)

    reference = weakref.ref(model, discard)
    _MODEL_PARAMETER_VALUE_SEALS[key] = (reference, seal, token)
    return token


def _registered_model_parameter_values_match(
    model: GridCMLM, token: object
) -> bool:
    registered = _MODEL_PARAMETER_VALUE_SEALS.get(id(model))
    if (
        registered is None
        or registered[0]() is not model
        or registered[2] is not token
    ):
        return False
    return _model_parameter_values_match_seal(model, registered[1])


@dataclass(frozen=True, slots=True)
class CheckpointModel:
    """A loaded model paired with the immutable weight artifact identity."""

    model: Any
    checkpoint_sha256: str
    model_semantics_version: str = MODEL_SEMANTICS_VERSION
    weight_state_sha256: str | None = None
    optimizer_step: int | None = None
    fixture_mode: bool = False
    _parameter_guard: tuple[tuple[object, ...], ...] = field(
        default=(), repr=False, compare=False
    )
    _execution_guard: tuple[tuple[object, ...], ...] = field(
        default=(), repr=False, compare=False
    )
    _parameter_value_registration_token: object = field(
        default=None, repr=False, compare=False
    )
    _construction_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _sha256(self.checkpoint_sha256, field_name="checkpoint_sha256")
        if self.model_semantics_version != MODEL_SEMANTICS_VERSION:
            raise ValueError("checkpoint model semantics do not match frozen M04a")
        if type(self.fixture_mode) is not bool:
            raise TypeError("fixture_mode must be boolean")
        if self.fixture_mode:
            if self._construction_token is not _FIXTURE_CHECKPOINT_MODEL_TOKEN:
                raise TypeError("fixture checkpoint models require the fixture constructor")
            if self.weight_state_sha256 is not None or self.optimizer_step is not None:
                raise ValueError("fixture checkpoint models cannot claim loaded weight state")
            for method in ("eval", "encode_task_memory", "decode_target"):
                if not callable(getattr(self.model, method, None)):
                    raise TypeError(f"fixture checkpoint model must provide {method}()")
            return
        if self._construction_token is not _VERIFIED_CHECKPOINT_MODEL_TOKEN:
            raise TypeError("production checkpoint models must come from load_checkpoint_model")
        if type(self.model) is not GridCMLM:
            raise TypeError("production checkpoint model must be the exact GridCMLM class")
        _sha256(self.weight_state_sha256, field_name="weight_state_sha256")
        if (
            type(self.optimizer_step) is not int
            or not 0 <= self.optimizer_step <= OPTIMIZER_UPDATES
        ):
            raise ValueError(f"optimizer_step must be in 0..{OPTIMIZER_UPDATES}")
        if trainable_parameter_count(self.model) != MODEL_PARAMETER_COUNT:
            raise ValueError("loaded checkpoint model parameter count does not close")
        if self._parameter_guard != _model_parameter_guard(self.model):
            raise ValueError("loaded checkpoint model parameter guard mismatch")
        if self._execution_guard != _model_execution_guard(self.model):
            raise ValueError("loaded checkpoint model execution guard mismatch")
        if not _registered_model_parameter_values_match(
            self.model, self._parameter_value_registration_token
        ):
            raise ValueError("loaded checkpoint model values differ from their device seal")
        if parameter_state_sha256(self.model) != self.weight_state_sha256:
            raise ValueError("loaded checkpoint weight values do not match attestation")

    @classmethod
    def fixture(cls, model: Any, *, checkpoint_sha256: str) -> "CheckpointModel":
        """Construct a CPU-suite fake that can never enter the CUDA path."""

        return cls(
            model=model,
            checkpoint_sha256=checkpoint_sha256,
            fixture_mode=True,
            _construction_token=_FIXTURE_CHECKPOINT_MODEL_TOKEN,
        )

    def assert_intact_for_runtime(
        self,
        runtime: "SamplerRuntime",
        *,
        verify_host_weight_sha256: bool = True,
    ) -> None:
        if type(verify_host_weight_sha256) is not bool:
            raise TypeError("verify_host_weight_sha256 must be boolean")
        if self.fixture_mode:
            if not runtime.allow_cpu_test or runtime.device.type != "cpu":
                raise ValueError("fixture checkpoint model is forbidden in CUDA sampling")
            return
        if runtime.allow_cpu_test or runtime.device.type != "cuda":
            raise ValueError("verified production checkpoint requires the CUDA runtime")
        if type(self.model) is not GridCMLM:
            raise TypeError("loaded checkpoint model class changed after verification")
        if self._parameter_guard != _model_parameter_guard(self.model):
            raise RuntimeError("loaded checkpoint model parameters changed after verification")
        if self._execution_guard != _model_execution_guard(self.model):
            raise RuntimeError("loaded checkpoint model execution path changed after verification")
        if not _registered_model_parameter_values_match(
            self.model, self._parameter_value_registration_token
        ):
            raise RuntimeError("loaded checkpoint model values differ from their device seal")
        if verify_host_weight_sha256 and (
            parameter_state_sha256(self.model) != self.weight_state_sha256
        ):
            raise RuntimeError("loaded checkpoint weight values changed after verification")


def _model_parameter_guard(model: GridCMLM) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            name,
            id(parameter),
            parameter.data_ptr(),
            tuple(parameter.shape),
            str(parameter.dtype),
            str(parameter.device),
            bool(parameter.requires_grad),
            int(parameter._version),
        )
        for name, parameter in model.named_parameters()
    )


def _execution_attribute_guard(
    value: object,
    *,
    depth: int = 0,
    seen: frozenset[int] = frozenset(),
) -> tuple[object, ...]:
    if value is None or type(value) in {bool, int, str, bytes}:
        return ("literal", type(value), value)
    if type(value) is float:
        return ("float", value.hex())
    if isinstance(value, Tensor):
        return (
            "tensor",
            type(value),
            id(value),
            value.data_ptr(),
            tuple(value.shape),
            str(value.dtype),
            str(value.device),
            int(value._version),
        )
    if isinstance(value, torch.nn.Module):
        return ("module", type(value), id(value))
    if type(value).__module__ == "torch":
        return ("torch-value", type(value), str(value))
    if callable(value):
        return ("callable", type(value), id(value))
    identity = id(value)
    if identity in seen or depth >= 8:
        return ("opaque", type(value), identity)
    nested_seen = seen | {identity}
    if isinstance(value, Mapping):
        guarded_items = tuple(
            sorted(
                (
                    (
                        _execution_attribute_guard(
                            key, depth=depth + 1, seen=nested_seen
                        ),
                        _execution_attribute_guard(
                            item, depth=depth + 1, seen=nested_seen
                        ),
                    )
                    for key, item in value.items()
                ),
                key=repr,
            )
        )
        return ("mapping", type(value), guarded_items)
    if isinstance(value, (tuple, list)):
        return (
            "sequence",
            type(value),
            tuple(
                _execution_attribute_guard(
                    item, depth=depth + 1, seen=nested_seen
                )
                for item in value
            ),
        )
    if isinstance(value, (set, frozenset)):
        guarded_items = tuple(
            sorted(
                (
                    _execution_attribute_guard(
                        item, depth=depth + 1, seen=nested_seen
                    )
                    for item in value
                ),
                key=repr,
            )
        )
        return ("set", type(value), guarded_items)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        return (
            "object",
            type(value),
            identity,
            _execution_attribute_guard(
                attributes, depth=depth + 1, seen=nested_seen
            ),
        )
    return ("opaque", type(value), identity)


def _model_execution_guard(model: GridCMLM) -> tuple[tuple[object, ...], ...]:
    guard: list[tuple[object, ...]] = [
        (
            "__api__",
            method_name,
            id(getattr(type(model), method_name)),
            method_name in model.__dict__,
        )
        for method_name in (
            "embed",
            "_embed_tensor_snapshot",
            "encode_grid_batch",
            "encode_task_memory",
            "decode_target",
            "forward",
        )
    ]
    for name, module in model.named_modules():
        guard.append(
            (
                name,
                type(module),
                id(type(module).forward),
                "forward" in module.__dict__,
                len(module._forward_pre_hooks),
                len(module._forward_hooks),
                len(module._backward_pre_hooks),
                len(module._backward_hooks),
                tuple(
                    (
                        attribute_name,
                        _execution_attribute_guard(attribute_value),
                    )
                    for attribute_name, attribute_value in sorted(
                        module.__dict__.items()
                    )
                ),
            )
        )
    return tuple(guard)


def load_checkpoint_model(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_optimizer_step: int,
    expected_config_sha256: str,
    expected_runtime_source_sha256: str,
    expected_test_source_sha256: str,
    expected_checkpoint_bytes: int,
    runtime: "SamplerRuntime",
) -> CheckpointModel:
    """Load exact checkpoint bytes into the exact frozen architecture."""

    if not isinstance(runtime, SamplerRuntime) or runtime.device.type != "cuda":
        raise ValueError("production checkpoint loading requires SamplerRuntime.cuda")
    validate_runtime_attestation(runtime.attestation)
    payload = load_checkpoint(
        path,
        expected_sha256=expected_sha256,
        expected_optimizer_step=expected_optimizer_step,
        expected_config_sha256=expected_config_sha256,
        expected_runtime_source_sha256=expected_runtime_source_sha256,
        expected_test_source_sha256=expected_test_source_sha256,
        expected_bytes=expected_checkpoint_bytes,
        map_location="cpu",
    )
    model = GridCMLM()
    incompatible = model.load_state_dict(payload["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("checkpoint state does not exactly match GridCMLM")
    if trainable_parameter_count(model) != MODEL_PARAMETER_COUNT:
        raise ValueError("checkpoint model parameter count does not close")
    state_sha256 = parameter_state_sha256(model)
    model.to(device=runtime.device)
    model.eval()
    guard = _model_parameter_guard(model)
    execution_guard = _model_execution_guard(model)
    parameter_value_registration_token = _register_model_parameter_value_seal(model)
    return CheckpointModel(
        model=model,
        checkpoint_sha256=expected_sha256,
        weight_state_sha256=state_sha256,
        optimizer_step=expected_optimizer_step,
        fixture_mode=False,
        _parameter_guard=guard,
        _execution_guard=execution_guard,
        _parameter_value_registration_token=parameter_value_registration_token,
        _construction_token=_VERIFIED_CHECKPOINT_MODEL_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class SamplerRuntime:
    """Execution boundary; CPU is opt-in and reserved for deterministic fakes."""

    device: torch.device
    allow_cpu_test: bool = False
    attestation: FrozenRuntimeAttestation | None = None

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type != "cuda" and not self.allow_cpu_test:
            raise ValueError("primary M04a sampling requires a CUDA device")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA runtime is unavailable")
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.type == "cuda":
            validate_runtime_attestation(self.attestation)
            if device.index != 0:
                raise ValueError("primary M04a sampling requires logical CUDA device 0")
        elif self.attestation is not None:
            raise ValueError("CPU test runtime cannot carry a CUDA attestation")
        object.__setattr__(self, "device", device)

    @classmethod
    def cuda(
        cls,
        attestation: FrozenRuntimeAttestation,
        device: str | torch.device = "cuda:0",
    ) -> "SamplerRuntime":
        return cls(
            device=torch.device(device),
            allow_cpu_test=False,
            attestation=attestation,
        )

    @classmethod
    def cpu_test(cls) -> "SamplerRuntime":
        return cls(device=torch.device("cpu"), allow_cpu_test=True)


@dataclass(frozen=True, slots=True)
class PairSamplingEvidence:
    blind_task_id: str
    test_index: int
    traces: tuple[dict[str, Any], ...]
    lane_rows: tuple[dict[str, Any], ...]
    encoder_rows: tuple[dict[str, Any], ...]
    decoder_rows: tuple[dict[str, Any], ...]
    pair_cost_row: dict[str, Any]
    artifact_receipt: Mapping[str, object] | None = None
    fixture_mode: bool = False

    def __post_init__(self) -> None:
        if type(self.fixture_mode) is not bool:
            raise TypeError("fixture_mode must be boolean")
        if self.fixture_mode and self.artifact_receipt is not None:
            raise ValueError("fixture pair evidence cannot carry an artifact receipt")
        validate_pair_evidence(
            self.pair_cost_row,
            lane_rows=self.lane_rows,
            traces=self.traces,
            encoder_rows=self.encoder_rows,
            decoder_rows=self.decoder_rows,
        )


@dataclass(frozen=True, slots=True)
class PoolSamplingEvidence:
    pairs: tuple[PairSamplingEvidence, ...]
    traces: tuple[dict[str, Any], ...]
    lane_rows: tuple[dict[str, Any], ...]
    encoder_rows: tuple[dict[str, Any], ...]
    decoder_rows: tuple[dict[str, Any], ...]
    pair_cost_rows: tuple[dict[str, Any], ...]
    fixture_mode: bool = False

    def __post_init__(self) -> None:
        if type(self.fixture_mode) is not bool:
            raise TypeError("fixture_mode must be boolean")
        if any(pair.fixture_mode != self.fixture_mode for pair in self.pairs):
            raise ValueError("pool fixture_mode differs from its pair evidence")

    @classmethod
    def from_pairs(
        cls, pairs: Sequence[PairSamplingEvidence]
    ) -> "PoolSamplingEvidence":
        normalized = tuple(pairs)
        fixture_modes = {pair.fixture_mode for pair in normalized}
        if len(fixture_modes) > 1:
            raise ValueError("pool cannot mix fixture and production pair evidence")
        return cls(
            pairs=normalized,
            traces=tuple(trace for pair in normalized for trace in pair.traces),
            lane_rows=tuple(row for pair in normalized for row in pair.lane_rows),
            encoder_rows=tuple(row for pair in normalized for row in pair.encoder_rows),
            decoder_rows=tuple(row for pair in normalized for row in pair.decoder_rows),
            pair_cost_rows=tuple(pair.pair_cost_row for pair in normalized),
            fixture_mode=next(iter(fixture_modes), False),
        )


@dataclass(frozen=True, slots=True)
class StreamedPoolSamplingEvidence:
    pair_cost_rows: tuple[dict[str, Any], ...]
    pair_artifact_receipts: tuple[Mapping[str, object], ...]
    final_artifact_receipt: Mapping[str, object]

    def __post_init__(self) -> None:
        if len(self.pair_cost_rows) != len(self.pair_artifact_receipts):
            raise ValueError("streamed pool receipts must close one-to-one with pairs")


@dataclass(frozen=True, slots=True)
class _ForwardResult:
    value: Any
    gpu_forward_ns: int
    wall_time_ns: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int


@dataclass(slots=True)
class _LaneState:
    shape_order: int
    shape_proposal_id: str
    height: int
    width: int
    global_lane: int
    local_lane: int
    greedy: bool
    seed_u64: int
    generator: torch.Generator
    cells: list[int | str]
    started_ns: int
    steps: list[dict[str, Any]] = field(default_factory=list)
    decoder_call_ids: list[str] = field(default_factory=list)
    cpu_sampling_ns: int = 0


def _model_device(model: Any) -> torch.device:
    declared = getattr(model, "device", None)
    if declared is not None:
        return torch.device(declared)
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(iter(parameters())).device
        except StopIteration:
            pass
    raise ValueError("checkpoint model device cannot be determined")


def _dtype_name(dtype: torch.dtype) -> str:
    names = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }
    try:
        return names[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported task-memory cache dtype: {dtype}") from exc


def _memory_tensor(memory: Any) -> Tensor:
    tensor = getattr(memory, "memory", None)
    if not isinstance(tensor, Tensor) or tensor.ndim != 3 or tensor.shape[0] != 1:
        raise TypeError("encoded task memory must expose a [1, length, width] tensor")
    return tensor


def _memory_length(memory: Any) -> int:
    declared = getattr(memory, "memory_length", None)
    value = declared() if callable(declared) else declared
    if value is None:
        value = int(_memory_tensor(memory).shape[1])
    if type(value) is not int or value <= 0:
        raise ValueError("encoded task memory length must be positive")
    return value


@contextmanager
def _frozen_inference_context(device: torch.device) -> Iterator[None]:
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    old_matmul_tf32: bool | None = None
    old_cudnn_tf32: bool | None = None
    if device.type == "cuda":
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        if device.type == "cuda":
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with torch.cuda.device(device):
                with sdpa_kernel(backends=[SDPBackend.MATH]), torch.inference_mode():
                    yield
        else:
            with torch.inference_mode():
                yield
    finally:
        if old_matmul_tf32 is not None and old_cudnn_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def _measure_forward(
    call: Callable[[], Any], *, device: torch.device
) -> _ForwardResult:
    if device.type != "cuda":
        started = time.perf_counter_ns()
        value = call()
        return _ForwardResult(
            value=value,
            gpu_forward_ns=0,
            wall_time_ns=time.perf_counter_ns() - started,
            peak_allocated_bytes=0,
            peak_reserved_bytes=0,
        )
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    started = time.perf_counter_ns()
    start_event.record()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        value = call()
    end_event.record()
    end_event.synchronize()
    wall_time_ns = time.perf_counter_ns() - started
    return _ForwardResult(
        value=value,
        gpu_forward_ns=int(round(start_event.elapsed_time(end_event) * 1_000_000)),
        wall_time_ns=wall_time_ns,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
    )


def _move_token_batch(
    batch: TokenBatch, *, device: torch.device
) -> tuple[TokenBatch, int]:
    if device.type != "cuda":
        return batch, 0
    torch.cuda.synchronize(device)
    started = time.perf_counter_ns()
    moved = batch.to(device, non_blocking=False)
    torch.cuda.synchronize(device)
    return moved, time.perf_counter_ns() - started


def _logits_to_cpu_float64(
    logits: Tensor, *, device: torch.device
) -> tuple[Tensor, int]:
    if device.type != "cuda":
        return logits.detach().to(device="cpu", dtype=torch.float64).contiguous(), 0
    torch.cuda.synchronize(device)
    started = time.perf_counter_ns()
    cpu = logits.detach().to(device="cpu", dtype=torch.float64).contiguous()
    torch.cuda.synchronize(device)
    return cpu, time.perf_counter_ns() - started


def _current_cuda_peaks(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda":
        return 0, 0
    return (
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )


def _validate_pair_inputs(
    blind_task: object, sidecar_row: object
) -> tuple[BlindTask, dict[str, Any], int]:
    if not isinstance(blind_task, BlindTask):
        raise TypeError("sampler accepts only an oracle-free BlindTask")
    row = validate_blind_input_row(sidecar_row)
    if row["blind_task_id"] != blind_task.task_id or row[
        "blind_content_sha256"
    ] != blind_task.blind_content_sha256:
        raise ValueError("shape sidecar row does not bind the supplied blind task")
    test_index = row["test_index"]
    if test_index >= len(blind_task.test_inputs):
        raise ValueError("shape sidecar test_index is outside the blind task")
    if as_grid(row["test_input"]) != blind_task.test_inputs[test_index]:
        raise ValueError("shape sidecar query input differs from the blind task")
    return blind_task, row, test_index


def _ordered_grid_descriptors(
    task: BlindTask, *, test_index: int
) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(task.train):
        if pair.output is None:
            raise ValueError("blind demonstration output is missing")
        for role, grid in (("demo_input", pair.input), ("demo_output", pair.output)):
            descriptors.append(
                {
                    "role": role,
                    "pair_index": pair_index,
                    "height": len(grid),
                    "width": len(grid[0]),
                    "grid_key": grid_key(grid),
                }
            )
    query = task.test_inputs[test_index]
    descriptors.append(
        {
            "role": "query_input",
            "pair_index": test_index,
            "height": len(query),
            "width": len(query[0]),
            "grid_key": grid_key(query),
        }
    )
    return descriptors


def _target_grid(lane: _LaneState) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            MASK_TOKEN_ID
            if lane.cells[row * lane.width + column] == _MASK
            else int(lane.cells[row * lane.width + column])
            for column in range(lane.width)
        )
        for row in range(lane.height)
    )


def _final_grid(lane: _LaneState) -> tuple[tuple[int, ...], ...]:
    if any(value == _MASK for value in lane.cells):
        raise RuntimeError("lane did not commit every target cell")
    return as_grid(
        [
            [int(lane.cells[row * lane.width + column]) for column in range(lane.width)]
            for row in range(lane.height)
        ]
    )


def _inverse_cdf(probabilities: Sequence[float], draw: float) -> int:
    cumulative = 0.0
    for color in range(OUTPUT_COLOR_COUNT):
        cumulative += float(probabilities[color])
        if cumulative > draw:
            return color
    return OUTPUT_COLOR_COUNT - 1


def _sample_lane_step(
    lane: _LaneState,
    *,
    step: int,
    logits: Tensor,
) -> tuple[dict[str, Any], int]:
    entry = [index for index, value in enumerate(lane.cells) if value == _MASK]
    schedule = mask_count_trace(lane.height * lane.width)
    if len(entry) != schedule[step - 1]:
        raise RuntimeError("lane mask state diverged from the frozen schedule")
    if tuple(logits.shape) != (lane.height * lane.width, OUTPUT_COLOR_COUNT):
        raise ValueError("lane logits must have exact [all_cells, 10] shape")
    selected_logits = logits[entry]
    hash_started = time.perf_counter_ns()
    logits_sha256 = float64_sequence_sha256(
        selected_logits.reshape(-1).tolist()
    )
    hashing_ns = time.perf_counter_ns() - hash_started

    sampling_started = time.perf_counter_ns()
    log_probability_rows = torch.log_softmax(selected_logits, dim=-1).tolist()
    probability_rows = torch.softmax(selected_logits, dim=-1).tolist()
    provisional: list[dict[str, Any]] = []
    confidences: list[tuple[float, int]] = []
    for row_index, linear_index in enumerate(entry):
        log_probabilities = log_probability_rows[row_index]
        probabilities = probability_rows[row_index]
        if lane.greedy:
            color = max(
                range(OUTPUT_COLOR_COUNT), key=lambda index: probabilities[index]
            )
            uniform: float | None = None
        else:
            uniform = float(
                torch.rand(
                    (), dtype=torch.float64, generator=lane.generator, device="cpu"
                ).item()
            )
            color = _inverse_cdf(probabilities, uniform)
        probability = float(probabilities[color])
        log_probability = float(log_probabilities[color])
        provisional.append(
            {
                "linear_index": linear_index,
                "row": linear_index // lane.width,
                "column": linear_index % lane.width,
                "chosen_color": color,
                "probability_hex": probability.hex(),
                "log_probability_hex": log_probability.hex(),
                "uniform_hex": None if uniform is None else uniform.hex(),
            }
        )
        confidences.append((probability, linear_index))
    retained = schedule[step]
    remasked_set = {
        linear_index
        for _, linear_index in sorted(confidences, key=lambda item: (item[0], item[1]))[
            :retained
        ]
    }
    predictions: list[dict[str, Any]] = []
    for prediction in provisional:
        linear_index = prediction["linear_index"]
        is_remasked = linear_index in remasked_set
        predictions.append(
            {**prediction, "provisional_remasked": is_remasked}
        )
        lane.cells[linear_index] = (
            _MASK if is_remasked else prediction["chosen_color"]
        )
    lane.cpu_sampling_ns += time.perf_counter_ns() - sampling_started
    remasked = sorted(remasked_set)
    return (
        {
            "step": step,
            "masked_indices_at_entry": entry,
            "logits_float64_le_sha256": logits_sha256,
            "predictions": predictions,
            "remasked_indices": remasked,
            "state_after_step": list(lane.cells),
        },
        hashing_ns,
    )


def _lane_trace(
    lane: _LaneState, *, blind_task_id: str, test_index: int
) -> dict[str, Any]:
    output = _final_grid(lane)
    return {
        "schema": LANE_TRACE_SCHEMA_VERSION,
        "blind_task_id": blind_task_id,
        "test_index": test_index,
        "shape_order": lane.shape_order,
        "shape_proposal_id": lane.shape_proposal_id,
        "proposed_height": lane.height,
        "proposed_width": lane.width,
        "global_lane": lane.global_lane,
        "local_lane": lane.local_lane,
        "greedy": lane.greedy,
        "seed_u64": lane.seed_u64,
        "initial_masked_indices": list(range(lane.height * lane.width)),
        "steps": lane.steps,
        "final_grid": grid_to_lists(output),
        "output_key": grid_key(output),
    }


def _cache_for_runtime(memory: Any, runtime: SamplerRuntime) -> Any:
    if runtime.allow_cpu_test:
        tensor = _memory_tensor(memory)
        if tensor.device.type != "cpu":
            raise ValueError("CPU test memory must remain on CPU")
        return memory
    if not isinstance(memory, EncodedTaskMemory):
        raise TypeError("primary sampler requires EncodedTaskMemory from GridCMLM")
    cache = memory.to_bf16_cuda_cache()
    tensor = _memory_tensor(cache)
    if (
        tensor.device != runtime.device
        or tensor.dtype != torch.bfloat16
        or not cache.is_bf16_cuda_cache
    ):
        raise ValueError("primary task-memory cache must be contiguous CUDA BF16")
    return cache


def _serialize_pair_materials(
    *,
    traces: Sequence[dict[str, Any]],
    lanes: Sequence[dict[str, Any]],
    encoders: Sequence[dict[str, Any]],
    decoders: Sequence[dict[str, Any]],
) -> dict[str, bytes]:
    return {
        "lane_traces.jsonl": serialize_jsonl(traces),
        "candidate_rows.jsonl": serialize_jsonl(lanes),
        "encoder_forward_ledger.jsonl": serialize_jsonl(encoders),
        "batch_forward_ledger.jsonl": serialize_jsonl(decoders),
    }


def sample_blind_test_pair(
    blind_task: BlindTask,
    shape_sidecar_row: Mapping[str, Any],
    checkpoint_model: CheckpointModel,
    *,
    runtime: SamplerRuntime,
    artifact_sink: PoolArtifactSink | None = None,
    fixture_sidecar: BlindInputShapeSidecar | None = None,
    _pool_integrity_attestation: object = None,
) -> PairSamplingEvidence:
    """Sample one blind test pair and return only fully closed evidence.

    ``NonFiniteLogitsError`` returns no partial result.  Pool callers must use
    :func:`sample_blind_pool`, which converts it into a whole-pool abort.
    """

    task, row, test_index = _validate_pair_inputs(blind_task, shape_sidecar_row)
    if not isinstance(checkpoint_model, CheckpointModel):
        raise TypeError("checkpoint_model must be a CheckpointModel")
    if not isinstance(runtime, SamplerRuntime):
        raise TypeError("runtime must be a SamplerRuntime")
    in_attested_pool = _pool_integrity_attestation is _POOL_INTEGRITY_ATTESTATION_TOKEN
    if _pool_integrity_attestation is not None and not in_attested_pool:
        raise ValueError("invalid pool integrity attestation")
    checkpoint_model.assert_intact_for_runtime(
        runtime, verify_host_weight_sha256=not in_attested_pool
    )
    if runtime.allow_cpu_test:
        if (
            not checkpoint_model.fixture_mode
            or not isinstance(fixture_sidecar, BlindInputShapeSidecar)
            or not fixture_sidecar.fixture_mode
        ):
            raise ValueError(
                "CPU fixture sampling requires an explicit fixture sidecar binding"
            )
        validated_fixture = validate_blind_input_shape_sidecar(
            fixture_sidecar.manifest,
            fixture_sidecar.rows,
            fixture_mode=True,
        )
        if row not in validated_fixture.rows:
            raise ValueError("fixture row is not declared by the fixture sidecar")
        if artifact_sink is not None:
            raise ValueError("fixture sampling cannot write pool artifact staging")
    else:
        if fixture_sidecar is not None or checkpoint_model.fixture_mode:
            raise ValueError("fixture sidecars/models are forbidden in production sampling")
        if artifact_sink is not None:
            _require_exact_production_pool_sink(artifact_sink)
    model = checkpoint_model.model
    if _model_device(model) != runtime.device:
        raise ValueError("checkpoint model and sampler runtime devices differ")
    if runtime.device.type == "cuda":
        validate_runtime_attestation(runtime.attestation)
        with torch.cuda.device(runtime.device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("primary M04a sampling requires CUDA BF16 support")

    accepted_shapes = [dict(shape) for shape in row["accepted_shapes"]]
    allocations = shape_lane_allocations(len(accepted_shapes))
    if not accepted_shapes:
        pair_started = time.perf_counter_ns()
        serialization_started = time.perf_counter_ns()
        materials = _serialize_pair_materials(
            traces=(), lanes=(), encoders=(), decoders=()
        )
        receipt = (
            None
            if artifact_sink is None
            else _PRODUCTION_POOL_APPEND_PAIR(
                _require_exact_production_pool_sink(artifact_sink),
                blind_task_id=task.task_id,
                test_index=test_index,
                materials=materials,
            )
        )
        serialization_ns = time.perf_counter_ns() - serialization_started
        pair_wall_time_ns = time.perf_counter_ns() - pair_started
        pair_cost = make_pair_cost_row(
            blind_task_id=task.task_id,
            test_index=test_index,
            accepted_shapes=(),
            lane_ids=(),
            encoder_call_ids=(),
            decoder_call_ids=(),
            pair_wall_time_ns=pair_wall_time_ns,
            serialization_ns=serialization_ns,
        )
        checkpoint_model.assert_intact_for_runtime(
            runtime, verify_host_weight_sha256=not in_attested_pool
        )
        return PairSamplingEvidence(
            blind_task_id=task.task_id,
            test_index=test_index,
            traces=(),
            lane_rows=(),
            encoder_rows=(),
            decoder_rows=(),
            pair_cost_row=pair_cost,
            artifact_receipt=receipt,
            fixture_mode=runtime.allow_cpu_test,
        )

    demonstrations = tuple((pair.input, pair.output) for pair in task.train)
    query_input = task.test_inputs[test_index]
    task_batch_cpu = tokenize_task_memory(demonstrations, query_input, device="cpu")
    previous_training = getattr(model, "training", None)
    model.eval()
    traces: list[dict[str, Any]] = []
    lane_rows: list[dict[str, Any]] = []
    encoder_rows: list[dict[str, Any]] = []
    decoder_rows: list[dict[str, Any]] = []
    h2d_ns = 0
    d2h_ns = 0
    hashing_ns = 0
    pair_peak_allocated = 0
    pair_peak_reserved = 0
    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
        torch.cuda.reset_peak_memory_stats(runtime.device)
    pair_started = time.perf_counter_ns()

    def update_pair_peaks() -> None:
        nonlocal pair_peak_allocated, pair_peak_reserved
        allocated, reserved = _current_cuda_peaks(runtime.device)
        pair_peak_allocated = max(pair_peak_allocated, allocated)
        pair_peak_reserved = max(pair_peak_reserved, reserved)

    try:
        with _frozen_inference_context(runtime.device):
            task_batch, transfer_ns = _move_token_batch(
                task_batch_cpu, device=runtime.device
            )
            h2d_ns += transfer_ns
            update_pair_peaks()
            encoder_measurement = _measure_forward(
                lambda: model.encode_task_memory(task_batch), device=runtime.device
            )
            cache = _cache_for_runtime(encoder_measurement.value, runtime)
            update_pair_peaks()
            cache_tensor = _memory_tensor(cache)
            cache_dtype = _dtype_name(cache_tensor.dtype)
            memory_length = _memory_length(cache)
            pair_peak_allocated = max(
                pair_peak_allocated, encoder_measurement.peak_allocated_bytes
            )
            pair_peak_reserved = max(
                pair_peak_reserved, encoder_measurement.peak_reserved_bytes
            )
            hash_started = time.perf_counter_ns()
            grid_descriptors = _ordered_grid_descriptors(
                task, test_index=test_index
            )
            encoder_row = make_encoder_forward_row(
                blind_task_id=task.task_id,
                test_index=test_index,
                ordered_grid_descriptors=grid_descriptors,
                padded_batch_shape=list(task_batch_cpu.token_ids.shape),
                unpadded_memory_length=memory_length,
                cache_dtype=cache_dtype,
                gpu_forward_ns=encoder_measurement.gpu_forward_ns,
                wall_time_ns=encoder_measurement.wall_time_ns,
                cuda_peak_allocated_bytes=encoder_measurement.peak_allocated_bytes,
                cuda_peak_reserved_bytes=encoder_measurement.peak_reserved_bytes,
            )
            hashing_ns += time.perf_counter_ns() - hash_started
            encoder_rows.append(encoder_row)
            hash_started = time.perf_counter_ns()
            cache_descriptor = {
                "encoder_call_id": encoder_row["call_id"],
                "cache_key": canonical_sha256(
                    {
                        "checkpoint_sha256": checkpoint_model.checkpoint_sha256,
                        "blind_task_id": task.task_id,
                        "test_index": test_index,
                        "memory_length": memory_length,
                        "dtype": cache_dtype,
                    }
                ),
                "memory_length": memory_length,
                "dtype": cache_dtype,
            }
            hashing_ns += time.perf_counter_ns() - hash_started

            global_offset = 0
            for shape_order, (shape, lane_count) in enumerate(
                zip(accepted_shapes, allocations)
            ):
                height = shape["proposed_height"]
                width = shape["proposed_width"]
                shape_started = time.perf_counter_ns()
                lanes: list[_LaneState] = []
                for local_lane in range(lane_count):
                    seed = lane_seed(
                        task.task_id,
                        test_index,
                        shape["shape_proposal_id"],
                        local_lane,
                    )
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(seed)
                    lanes.append(
                        _LaneState(
                            shape_order=shape_order,
                            shape_proposal_id=shape["shape_proposal_id"],
                            height=height,
                            width=width,
                            global_lane=global_offset + local_lane,
                            local_lane=local_lane,
                            greedy=local_lane == 0,
                            seed_u64=seed,
                            generator=generator,
                            cells=[_MASK] * (height * width),
                            started_ns=shape_started,
                        )
                    )
                for step in range(1, DENOISING_STEPS + 1):
                    for batch_index, batch_start in enumerate(
                        range(0, lane_count, INFERENCE_MICROBATCH)
                    ):
                        batch_lanes = lanes[
                            batch_start : batch_start + INFERENCE_MICROBATCH
                        ]
                        targets_cpu = tokenize_target_batch(
                            tuple(_target_grid(lane) for lane in batch_lanes),
                            device="cpu",
                        )
                        targets, transfer_ns = _move_token_batch(
                            targets_cpu, device=runtime.device
                        )
                        h2d_ns += transfer_ns
                        update_pair_peaks()
                        decoder_measurement = _measure_forward(
                            lambda targets=targets: model.decode_target(targets, cache),
                            device=runtime.device,
                        )
                        decoded = decoder_measurement.value
                        logits = getattr(decoded, "logits", None)
                        expected_shape = (
                            len(batch_lanes),
                            height * width,
                            OUTPUT_COLOR_COUNT,
                        )
                        if not isinstance(logits, Tensor) or tuple(logits.shape) != expected_shape:
                            raise ValueError(
                                "decoder logits must have exact [batch, cells, 10] shape"
                            )
                        logits_cpu, transfer_ns = _logits_to_cpu_float64(
                            logits, device=runtime.device
                        )
                        d2h_ns += transfer_ns
                        update_pair_peaks()
                        if not bool(torch.isfinite(logits_cpu).all().item()):
                            raise NonFiniteLogitsError(
                                f"{NONFINITE_LOGITS}: {task.task_id}[{test_index}] "
                                f"shape={shape_order} step={step} batch={batch_index}"
                            )
                        pair_peak_allocated = max(
                            pair_peak_allocated,
                            decoder_measurement.peak_allocated_bytes,
                        )
                        pair_peak_reserved = max(
                            pair_peak_reserved,
                            decoder_measurement.peak_reserved_bytes,
                        )
                        hash_started = time.perf_counter_ns()
                        decoder_row = make_decoder_forward_row(
                            blind_task_id=task.task_id,
                            test_index=test_index,
                            shape_order=shape_order,
                            shape_proposal_id=shape["shape_proposal_id"],
                            step=step,
                            batch_index_within_shape=batch_index,
                            global_lanes=[lane.global_lane for lane in batch_lanes],
                            cached_memory_descriptor=cache_descriptor,
                            gpu_forward_ns=decoder_measurement.gpu_forward_ns,
                            wall_time_ns=decoder_measurement.wall_time_ns,
                            cuda_peak_allocated_bytes=decoder_measurement.peak_allocated_bytes,
                            cuda_peak_reserved_bytes=decoder_measurement.peak_reserved_bytes,
                        )
                        hashing_ns += time.perf_counter_ns() - hash_started
                        decoder_rows.append(decoder_row)
                        for lane_offset, lane in enumerate(batch_lanes):
                            lane.decoder_call_ids.append(decoder_row["call_id"])
                            trace_step, step_hashing_ns = _sample_lane_step(
                                lane,
                                step=step,
                                logits=logits_cpu[lane_offset],
                            )
                            hashing_ns += step_hashing_ns
                            lane.steps.append(trace_step)
                shape_completed_ns = time.perf_counter_ns()
                for lane in lanes:
                    hash_started = time.perf_counter_ns()
                    trace = _lane_trace(
                        lane, blind_task_id=task.task_id, test_index=test_index
                    )
                    lane_row = make_lane_row(
                        trace,
                        checkpoint_sha256=checkpoint_model.checkpoint_sha256,
                        batch_forward_call_ids=lane.decoder_call_ids,
                        cpu_sampling_ns=lane.cpu_sampling_ns,
                        lane_wall_time_ns=shape_completed_ns - lane.started_ns,
                    )
                    hashing_ns += time.perf_counter_ns() - hash_started
                    traces.append(trace)
                    lane_rows.append(lane_row)
                global_offset += lane_count
    finally:
        if previous_training is True and callable(getattr(model, "train", None)):
            model.train(True)
    checkpoint_model.assert_intact_for_runtime(
        runtime, verify_host_weight_sha256=not in_attested_pool
    )

    serialization_started = time.perf_counter_ns()
    materials = _serialize_pair_materials(
        traces=traces,
        lanes=lane_rows,
        encoders=encoder_rows,
        decoders=decoder_rows,
    )
    receipt = (
        None
        if artifact_sink is None
        else _PRODUCTION_POOL_APPEND_PAIR(
            _require_exact_production_pool_sink(artifact_sink),
            blind_task_id=task.task_id,
            test_index=test_index,
            materials=materials,
        )
    )
    serialization_ns = time.perf_counter_ns() - serialization_started
    pair_wall_time_ns = time.perf_counter_ns() - pair_started
    unique_outputs = len({row["output_key"] for row in lane_rows})
    pair_cost = make_pair_cost_row(
        blind_task_id=task.task_id,
        test_index=test_index,
        accepted_shapes=accepted_shapes,
        lane_ids=[lane["lane_id"] for lane in lane_rows],
        encoder_call_ids=[encoder["call_id"] for encoder in encoder_rows],
        decoder_call_ids=[decoder["call_id"] for decoder in decoder_rows],
        unique_outputs=unique_outputs,
        pair_wall_time_ns=pair_wall_time_ns,
        h2d_ns=h2d_ns,
        d2h_ns=d2h_ns,
        encoder_gpu_ns=sum(row["gpu_forward_ns"] for row in encoder_rows),
        decoder_gpu_ns=sum(row["gpu_forward_ns"] for row in decoder_rows),
        cpu_sampling_ns=sum(row["cpu_sampling_ns"] for row in lane_rows),
        hashing_ns=hashing_ns,
        serialization_ns=serialization_ns,
        cuda_peak_allocated_bytes=pair_peak_allocated,
        cuda_peak_reserved_bytes=pair_peak_reserved,
    )
    return PairSamplingEvidence(
        blind_task_id=task.task_id,
        test_index=test_index,
        traces=tuple(traces),
        lane_rows=tuple(lane_rows),
        encoder_rows=tuple(encoder_rows),
        decoder_rows=tuple(decoder_rows),
        pair_cost_row=pair_cost,
        artifact_receipt=receipt,
        fixture_mode=runtime.allow_cpu_test,
    )


def _finalize_streamed_pool(
    checkpoint_model: CheckpointModel,
    runtime: SamplerRuntime,
    artifact_sink: M04APoolPairStagingSink,
    *,
    pair_cost_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, object]:
    """Close staging only after the final device seal and host hash both match."""

    try:
        checkpoint_model.assert_intact_for_runtime(
            runtime, verify_host_weight_sha256=True
        )
        return _PRODUCTION_POOL_FINALIZE(
            _require_exact_production_pool_sink(artifact_sink),
            pair_cost_rows=pair_cost_rows,
        )
    except Exception:
        _PRODUCTION_POOL_ABORT(
            artifact_sink, failure_code="POOL_FINALIZATION_FAILED"
        )
        raise


def sample_blind_pool(
    blind_tasks: Sequence[BlindTask],
    shape_sidecar: BlindInputShapeSidecar,
    checkpoint_model: CheckpointModel,
    *,
    runtime: SamplerRuntime,
    artifact_sink: PoolArtifactSink | None = None,
) -> PoolSamplingEvidence | StreamedPoolSamplingEvidence:
    """Sample a complete closed-world sidecar, aborting atomically on non-finite logits."""

    if isinstance(blind_tasks, (str, bytes)) or not isinstance(blind_tasks, Sequence):
        raise TypeError("blind_tasks must be an ordered sequence")
    tasks = tuple(blind_tasks)
    if not tasks or any(not isinstance(task, BlindTask) for task in tasks):
        raise TypeError("blind_tasks must contain only BlindTask objects")
    if not isinstance(shape_sidecar, BlindInputShapeSidecar):
        raise TypeError("shape_sidecar must be a BlindInputShapeSidecar")
    if not isinstance(runtime, SamplerRuntime):
        raise TypeError("runtime must be a SamplerRuntime")
    if not isinstance(checkpoint_model, CheckpointModel):
        raise TypeError("checkpoint_model must be a CheckpointModel")
    if shape_sidecar.fixture_mode != runtime.allow_cpu_test:
        raise ValueError("fixture sidecar/runtime modes must match exactly")
    if runtime.allow_cpu_test and artifact_sink is not None:
        raise ValueError("fixture pools cannot write production artifact staging")
    if runtime.device.type == "cuda" and artifact_sink is None:
        raise ValueError("primary CUDA pool generation requires an incomplete-staging sink")
    if runtime.device.type == "cuda":
        _require_exact_production_pool_sink(artifact_sink)
    sidecar = validate_blind_input_shape_sidecar(
        shape_sidecar.manifest,
        shape_sidecar.rows,
        fixture_mode=shape_sidecar.fixture_mode,
    )
    checkpoint_model.assert_intact_for_runtime(
        runtime, verify_host_weight_sha256=True
    )
    task_by_id = {task.task_id: task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("blind task IDs must be unique")
    sidecar_task_ids = {row["blind_task_id"] for row in sidecar.rows}
    if set(task_by_id) != sidecar_task_ids:
        raise ValueError("blind task set does not exactly match shape sidecar")
    for task_id, task in task_by_id.items():
        indices = [
            row["test_index"]
            for row in sidecar.rows
            if row["blind_task_id"] == task_id
        ]
        if indices != list(range(len(task.test_inputs))):
            raise ValueError(
                "shape sidecar must contain every blind test pair exactly once"
            )
    completed: list[PairSamplingEvidence] = []
    pair_cost_rows: list[dict[str, Any]] = []
    pair_receipts: list[Mapping[str, object]] = []
    try:
        for row in sidecar.rows:
            pair = sample_blind_test_pair(
                task_by_id[row["blind_task_id"]],
                row,
                checkpoint_model,
                runtime=runtime,
                artifact_sink=artifact_sink,
                fixture_sidecar=sidecar if runtime.allow_cpu_test else None,
                _pool_integrity_attestation=_POOL_INTEGRITY_ATTESTATION_TOKEN,
            )
            pair_cost_rows.append(pair.pair_cost_row)
            if artifact_sink is None:
                completed.append(pair)
            else:
                if pair.artifact_receipt is None:
                    raise RuntimeError("pool staging sink did not return a pair receipt")
                pair_receipts.append(pair.artifact_receipt)
    except NonFiniteLogitsError as exc:
        completed.clear()
        if artifact_sink is not None:
            _PRODUCTION_POOL_ABORT(artifact_sink, failure_code=NONFINITE_LOGITS)
        raise PoolSamplingAborted(f"{NONFINITE_LOGITS}: entire pool incomplete") from exc
    except Exception:
        completed.clear()
        if artifact_sink is not None:
            _PRODUCTION_POOL_ABORT(
                artifact_sink, failure_code="POOL_GENERATION_FAILED"
            )
        raise
    if artifact_sink is not None:
        final_receipt = _finalize_streamed_pool(
            checkpoint_model,
            runtime,
            artifact_sink,
            pair_cost_rows=pair_cost_rows,
        )
        return StreamedPoolSamplingEvidence(
            pair_cost_rows=tuple(pair_cost_rows),
            pair_artifact_receipts=tuple(pair_receipts),
            final_artifact_receipt=final_receipt,
        )
    checkpoint_model.assert_intact_for_runtime(
        runtime, verify_host_weight_sha256=True
    )
    return PoolSamplingEvidence.from_pairs(completed)


__all__ = [
    "CheckpointModel",
    "NONFINITE_LOGITS",
    "NonFiniteLogitsError",
    "PairSamplingEvidence",
    "PoolSamplingAborted",
    "PoolSamplingEvidence",
    "PoolArtifactSink",
    "SamplerRuntime",
    "StreamedPoolSamplingEvidence",
    "load_checkpoint_model",
    "sample_blind_pool",
    "sample_blind_test_pair",
]
