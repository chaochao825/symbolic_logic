"""PyTorch implementation of the frozen M04a Grid-CMLM architecture.

This module is an explicit optional-runtime boundary: importing it requires
PyTorch, while the package root, symbolic solver, CLI parser, and contract layer do
not import it.  Model construction always happens on CPU, performs the frozen
independent reset, and only then may callers move the model to CUDA.
"""

from __future__ import annotations

import hashlib
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from .m04a_contract import (
    ATTENTION_HEADS,
    DECODER_LAYER_COUNT,
    D_MODEL,
    DROPOUT_PROBABILITY,
    ENCODER_LAYER_COUNT,
    FFN_WIDTH,
    LAYER_NORM_EPSILON,
    MAX_DEMONSTRATIONS,
    MAX_GRID_SIDE,
    MODEL_PARAMETER_COUNT,
    MODEL_SEMANTICS_VERSION,
    MODEL_STAGE,
    OUTPUT_COLOR_COUNT,
    PAIR_SLOT_EMBEDDING_COUNT,
    POSITION_EMBEDDING_COUNT,
    QUERY_TARGET_PAIR_SLOT,
    ROLE_EMBEDDING_COUNT,
    TOKEN_EMBEDDING_COUNT,
    TRAINING_SEED,
)


COLOR_TOKEN_COUNT = 10
MASK_TOKEN_ID = 10
PAD_TOKEN_ID = 11
GRID_CLS_TOKEN_ID = 12

DEMO_INPUT_ROLE_ID = 0
DEMO_OUTPUT_ROLE_ID = 1
QUERY_INPUT_ROLE_ID = 2
TARGET_ROLE_ID = 3

GRID_CLS_POSITION_ID = 30
RESERVED_SIZE_ID = 0
_TOKEN_BATCH_ATTESTATION_TOKEN = object()
_TOKEN_BATCH_ATTESTATION_SCHEMA = b"afts-token-batch-attestation/v0.1"


def _strict_int(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(f"{field} must be an integer >= {minimum}")
    return value


def _normalize_token_grid(
    value: object,
    *,
    field: str,
    maximum_token: int,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a non-empty rectangular sequence")
    rows = tuple(value)
    if not rows or len(rows) > MAX_GRID_SIDE:
        raise ValueError(f"{field} height must be in 1..{MAX_GRID_SIDE}")
    normalized: list[tuple[int, ...]] = []
    width: int | None = None
    for row_index, row in enumerate(rows):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise TypeError(f"{field}[{row_index}] must be a sequence")
        cells = tuple(row)
        if not cells:
            raise ValueError(f"{field}[{row_index}] must not be empty")
        if width is None:
            width = len(cells)
            if width > MAX_GRID_SIDE:
                raise ValueError(f"{field} width must be in 1..{MAX_GRID_SIDE}")
        elif len(cells) != width:
            raise ValueError(f"{field} must be rectangular")
        for column_index, token in enumerate(cells):
            if type(token) is not int or not 0 <= token <= maximum_token:
                raise TypeError(
                    f"{field}[{row_index}][{column_index}] must be an integer token "
                    f"in 0..{maximum_token}"
                )
        normalized.append(cells)
    return tuple(normalized)


def _cpu_any(value: Tensor) -> bool:
    """Inspect validated construction tensors without synchronizing a GPU."""

    if value.device.type != "cpu":
        raise RuntimeError("semantic tensor validation is restricted to CPU tensors")
    return bool(torch.any(value).item())


def _token_batch_cpu_fingerprint(
    tensors: Sequence[Tensor],
    *,
    lengths: tuple[int, ...],
    semantic_kind: str,
) -> str:
    """Hash one validated CPU snapshot without ever inspecting CUDA values."""

    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise RuntimeError("TokenBatch content fingerprints are restricted to CPU")
    digest = hashlib.sha256()

    def frame(value: bytes | memoryview) -> None:
        digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
        digest.update(value)

    frame(_TOKEN_BATCH_ATTESTATION_SCHEMA)
    frame(semantic_kind.encode("ascii"))
    frame(",".join(str(length) for length in lengths).encode("ascii"))
    for tensor in tensors:
        contiguous = tensor.detach().contiguous()
        frame(str(contiguous.dtype).encode("ascii"))
        frame(",".join(str(dimension) for dimension in contiguous.shape).encode("ascii"))
        frame(memoryview(contiguous.numpy()).cast("B"))
    return digest.hexdigest()


def _version_tracked_tensor_copies(
    tensors: Sequence[Tensor],
    *,
    device: torch.device | str | None = None,
    non_blocking: bool = False,
) -> tuple[Tensor, ...]:
    """Create no-grad copies whose mutation versions remain attestable.

    ``torch.inference_mode`` makes tensors created inside it omit their version
    counters.  TokenBatch transfer attestations intentionally depend on those
    counters, so this narrowly scoped helper restores ordinary tensor creation
    even when a production sampler encloses the transfer in inference mode.
    """

    with torch.inference_mode(False), torch.no_grad():
        if device is None:
            copies = tuple(tensor.detach().clone() for tensor in tensors)
        else:
            copies = tuple(
                tensor.to(device=device, non_blocking=non_blocking)
                for tensor in tensors
            )
    # This is both a postcondition and a fail-closed compatibility check for the
    # exact PyTorch runtime: inference tensors raise here instead of weakening
    # the transfer attestation.
    tuple(int(tensor._version) for tensor in copies)
    return copies


@dataclass(frozen=True, slots=True)
class _CPUValidationAttestation:
    content_sha256: str
    _token: object

    def __post_init__(self) -> None:
        if (
            self._token is not _TOKEN_BATCH_ATTESTATION_TOKEN
            or len(self.content_sha256) != 64
        ):
            raise ValueError("invalid CPU TokenBatch attestation")


class _WeakReferenceable:
    __slots__ = ("__weakref__",)


@dataclass(frozen=True, slots=True, eq=False)
class _CUDATransferAttestation(_WeakReferenceable):
    source_content_sha256: str
    tensor_references: tuple[Tensor, ...] = field(repr=False, compare=False)
    tensor_versions: tuple[int, ...]
    lengths: tuple[int, ...]
    semantic_kind: str
    shape: tuple[int, int]
    device: str
    _token: object

    def __post_init__(self) -> None:
        if (
            self._token is not _TOKEN_BATCH_ATTESTATION_TOKEN
            or len(self.source_content_sha256) != 64
            or len(self.tensor_references) != 8
            or len(self.tensor_versions) != 8
        ):
            raise ValueError("invalid CUDA TokenBatch transfer attestation")


def _make_cuda_sealed_tensor_registry() -> tuple[Any, Any, Any]:
    registry = weakref.WeakKeyDictionary()

    def register(
        attestation: _CUDATransferAttestation,
        tensors: tuple[Tensor, ...],
    ) -> None:
        if len(tensors) != 8:
            raise ValueError("CUDA TokenBatch seal must contain eight tensors")
        if attestation in registry:
            raise RuntimeError("CUDA TokenBatch seal registration is one-shot")
        registry[attestation] = tensors

    def contains(attestation: _CUDATransferAttestation) -> bool:
        return attestation in registry

    def snapshot(
        attestation: _CUDATransferAttestation,
    ) -> tuple[Tensor, ...]:
        tensors = registry.get(attestation)
        if not isinstance(tensors, tuple) or len(tensors) != 8:
            raise RuntimeError("CUDA TokenBatch lacks a private tensor seal")
        return tuple(tensor.detach().clone() for tensor in tensors)

    return register, contains, snapshot


(
    _register_cuda_sealed_tensors,
    _has_cuda_sealed_tensors,
    _snapshot_cuda_sealed_tensors,
) = _make_cuda_sealed_tensor_registry()


@dataclass(frozen=True, slots=True)
class TokenBatch:
    """Seven embedding indices plus an explicit key/query padding mask."""

    token_ids: Tensor
    row_ids: Tensor
    column_ids: Tensor
    height_ids: Tensor
    width_ids: Tensor
    role_ids: Tensor
    pair_slot_ids: Tensor
    padding_mask: Tensor
    lengths: tuple[int, ...]
    semantic_kind: str
    _validation_attestation: object = field(
        default=None, repr=False, compare=False
    )

    def _all_tensors(self) -> tuple[Tensor, ...]:
        return (
            self.token_ids,
            self.row_ids,
            self.column_ids,
            self.height_ids,
            self.width_ids,
            self.role_ids,
            self.pair_slot_ids,
            self.padding_mask,
        )

    def _require_current_cpu_attestation(self) -> _CPUValidationAttestation:
        attestation = self._validation_attestation
        if not isinstance(attestation, _CPUValidationAttestation) or (
            attestation._token is not _TOKEN_BATCH_ATTESTATION_TOKEN
        ):
            raise RuntimeError("TokenBatch lacks a valid CPU content attestation")
        current_sha256 = _token_batch_cpu_fingerprint(
            self._all_tensors(),
            lengths=self.lengths,
            semantic_kind=self.semantic_kind,
        )
        if current_sha256 != attestation.content_sha256:
            raise RuntimeError(
                "TokenBatch CPU tensor content changed after semantic validation"
            )
        return attestation

    def _tensors_for_model_consumption(self) -> tuple[Tensor, ...]:
        """Return content-sealed tensors; CUDA callers never expose this snapshot."""

        if self.device.type == "cpu":
            self._require_current_cpu_attestation()
            return self._all_tensors()
        return _snapshot_cuda_sealed_tensors(
            self._require_current_cuda_attestation()
        )

    def _require_current_cuda_attestation(self) -> _CUDATransferAttestation:
        """Close CUDA provenance using Python/Tensor identities, never ``.item()``."""

        attestation = self._validation_attestation
        tensors = self._all_tensors()
        if (
            not isinstance(attestation, _CUDATransferAttestation)
            or attestation._token is not _TOKEN_BATCH_ATTESTATION_TOKEN
            or not _has_cuda_sealed_tensors(attestation)
            or any(
                reference is not tensor
                for reference, tensor in zip(attestation.tensor_references, tensors)
            )
            or attestation.tensor_versions
            != tuple(int(tensor._version) for tensor in tensors)
            or attestation.lengths != self.lengths
            or attestation.semantic_kind != self.semantic_kind
            or attestation.shape != tuple(self.token_ids.shape)
            or attestation.device != str(self.device)
        ):
            raise RuntimeError("CUDA TokenBatch does not match its validated transfer")
        return attestation

    def __post_init__(self) -> None:
        index_tensors = (
            self.token_ids,
            self.row_ids,
            self.column_ids,
            self.height_ids,
            self.width_ids,
            self.role_ids,
            self.pair_slot_ids,
        )
        if any(not isinstance(tensor, Tensor) for tensor in (*index_tensors, self.padding_mask)):
            raise TypeError("TokenBatch fields must be torch tensors")
        shape = self.token_ids.shape
        if len(shape) != 2 or shape[0] == 0 or shape[1] == 0:
            raise ValueError("TokenBatch tensors must have non-empty [batch, sequence] shape")
        if any(tensor.shape != shape for tensor in (*index_tensors, self.padding_mask)):
            raise ValueError("all TokenBatch tensors must have identical shape")
        if any(tensor.device != self.token_ids.device for tensor in (*index_tensors, self.padding_mask)):
            raise ValueError("all TokenBatch tensors must share one device")
        if any(tensor.dtype != torch.long for tensor in index_tensors):
            raise TypeError("TokenBatch embedding indices must use torch.long")
        if self.padding_mask.dtype != torch.bool:
            raise TypeError("TokenBatch padding_mask must use torch.bool")
        if self.semantic_kind not in {"encoder", "target"}:
            raise ValueError("TokenBatch semantic_kind must be encoder or target")
        lengths = tuple(self.lengths)
        if len(lengths) != shape[0] or any(
            type(length) is not int or not 1 <= length <= shape[1] for length in lengths
        ):
            raise ValueError("TokenBatch lengths must close against the padded tensor shape")
        if self.token_ids.device.type not in {"cpu", "cuda"}:
            raise ValueError("TokenBatch supports only CPU or CUDA tensors")
        if self.token_ids.device.type == "cpu":
            expected_mask = torch.arange(shape[1]).unsqueeze(0) >= torch.tensor(
                lengths
            ).unsqueeze(1)
            if _cpu_any(expected_mask != self.padding_mask):
                raise ValueError("TokenBatch padding mask is not the suffix implied by lengths")
            if _cpu_any(self.token_ids.masked_select(self.padding_mask) != PAD_TOKEN_ID):
                raise ValueError("padding positions must carry the PAD token")
            range_checks = (
                (self.token_ids, 0, TOKEN_EMBEDDING_COUNT - 1, "token"),
                (self.row_ids, 0, POSITION_EMBEDDING_COUNT - 1, "row"),
                (self.column_ids, 0, POSITION_EMBEDDING_COUNT - 1, "column"),
                (self.height_ids, 0, POSITION_EMBEDDING_COUNT - 1, "height"),
                (self.width_ids, 0, POSITION_EMBEDDING_COUNT - 1, "width"),
                (self.role_ids, 0, ROLE_EMBEDDING_COUNT - 1, "role"),
                (self.pair_slot_ids, 0, PAIR_SLOT_EMBEDDING_COUNT - 1, "pair slot"),
            )
            for tensor, lower, upper, label in range_checks:
                if _cpu_any((tensor < lower) | (tensor > upper)):
                    raise ValueError(f"TokenBatch {label} indices exceed their embedding table")
            for tensor, expected, label in (
                (self.row_ids, GRID_CLS_POSITION_ID, "row"),
                (self.column_ids, GRID_CLS_POSITION_ID, "column"),
                (self.height_ids, RESERVED_SIZE_ID, "height"),
                (self.width_ids, RESERVED_SIZE_ID, "width"),
            ):
                if _cpu_any(tensor.masked_select(self.padding_mask) != expected):
                    raise ValueError(f"TokenBatch PAD {label} indices are not canonical")
            if _cpu_any(self.role_ids != self.role_ids[:, :1]):
                raise ValueError("each TokenBatch sequence must use one canonical role")
            if _cpu_any(self.pair_slot_ids != self.pair_slot_ids[:, :1]):
                raise ValueError("each TokenBatch sequence must use one canonical pair slot")
            valid = ~self.padding_mask
            if self.semantic_kind == "encoder":
                if _cpu_any(self.token_ids[:, 0] != GRID_CLS_TOKEN_ID):
                    raise ValueError("every encoder grid sequence must begin with GRID_CLS")
                valid_roles = self.role_ids.masked_select(valid)
                valid_slots = self.pair_slot_ids.masked_select(valid)
                demo = (valid_roles == DEMO_INPUT_ROLE_ID) | (
                    valid_roles == DEMO_OUTPUT_ROLE_ID
                )
                query = valid_roles == QUERY_INPUT_ROLE_ID
                if _cpu_any(~(demo | query)):
                    raise ValueError("only demo/query roles can enter the task encoder")
                if _cpu_any(demo & (valid_slots >= MAX_DEMONSTRATIONS)):
                    raise ValueError("demo encoder roles require pair slots 0..9")
                if _cpu_any(query & (valid_slots != QUERY_TARGET_PAIR_SLOT)):
                    raise ValueError("query-input encoder role requires pair slot 15")
                roles = tuple(int(value) for value in self.role_ids[:, 0].tolist())
                slots = tuple(
                    int(value) for value in self.pair_slot_ids[:, 0].tolist()
                )
                if len(roles) % 2 != 1:
                    raise ValueError("encoder sequence count must be demo pairs plus query")
                demonstration_count = (len(roles) - 1) // 2
                if not 1 <= demonstration_count <= MAX_DEMONSTRATIONS:
                    raise ValueError("encoder requires one to ten ordered demonstrations")
                expected_roles = tuple(
                    role
                    for _ in range(demonstration_count)
                    for role in (DEMO_INPUT_ROLE_ID, DEMO_OUTPUT_ROLE_ID)
                ) + (QUERY_INPUT_ROLE_ID,)
                expected_slots = tuple(
                    slot
                    for slot in range(demonstration_count)
                    for _ in range(2)
                ) + (QUERY_TARGET_PAIR_SLOT,)
                if roles != expected_roles or slots != expected_slots:
                    raise ValueError(
                        "encoder grids must be [demo_in0,demo_out0,...,query15]"
                    )
            else:
                if _cpu_any(self.role_ids.masked_select(valid) != TARGET_ROLE_ID):
                    raise ValueError("decoder tokens must use the target role")
                if _cpu_any(
                    self.pair_slot_ids.masked_select(valid) != QUERY_TARGET_PAIR_SLOT
                ):
                    raise ValueError("decoder tokens must use query/target pair slot 15")
                if _cpu_any(self.token_ids.masked_select(valid) > MASK_TOKEN_ID):
                    raise ValueError("decoder cells may contain only colors or MASK")
            content_sha256 = _token_batch_cpu_fingerprint(
                (*index_tensors, self.padding_mask),
                lengths=lengths,
                semantic_kind=self.semantic_kind,
            )
            object.__setattr__(
                self,
                "_validation_attestation",
                _CPUValidationAttestation(
                    content_sha256=content_sha256,
                    _token=_TOKEN_BATCH_ATTESTATION_TOKEN,
                ),
            )
        object.__setattr__(self, "lengths", lengths)
        if self.token_ids.device.type == "cuda":
            try:
                self._require_current_cuda_attestation()
            except RuntimeError as exc:
                raise ValueError(
                    "CUDA TokenBatch tensors must originate from one bound "
                    "CPU-validated transfer"
                ) from exc

    @property
    def batch_size(self) -> int:
        return int(self.token_ids.shape[0])

    @property
    def padded_length(self) -> int:
        return int(self.token_ids.shape[1])

    @property
    def device(self) -> torch.device:
        return self.token_ids.device

    @property
    def has_padding(self) -> bool:
        return any(length != self.padded_length for length in self.lengths)

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "TokenBatch":
        target = torch.device(device)
        if target.type not in {"cpu", "cuda"}:
            raise ValueError("TokenBatch supports only CPU or CUDA transfers")

        source_tensors = self._all_tensors()
        if self.device.type == "cpu":
            cpu_attestation = self._require_current_cpu_attestation()
            if target.type == "cpu":
                return self

            # Copy a private CPU snapshot, then re-hash it.  A concurrent or prior
            # mutation of the publicly reachable source tensors cannot reach CUDA.
            snapshot = _version_tracked_tensor_copies(source_tensors)
            snapshot_sha256 = _token_batch_cpu_fingerprint(
                snapshot,
                lengths=self.lengths,
                semantic_kind=self.semantic_kind,
            )
            if snapshot_sha256 != cpu_attestation.content_sha256:
                raise RuntimeError(
                    "TokenBatch CPU tensor content changed during CUDA transfer"
                )
            transferred = _version_tracked_tensor_copies(
                snapshot,
                device=target,
                non_blocking=non_blocking,
            )
            sealed_tensors = _version_tracked_tensor_copies(transferred)
            cuda_attestation: object = _CUDATransferAttestation(
                source_content_sha256=snapshot_sha256,
                tensor_references=transferred,
                tensor_versions=tuple(int(tensor._version) for tensor in transferred),
                lengths=self.lengths,
                semantic_kind=self.semantic_kind,
                shape=tuple(transferred[0].shape),
                device=str(transferred[0].device),
                _token=_TOKEN_BATCH_ATTESTATION_TOKEN,
            )
            _register_cuda_sealed_tensors(cuda_attestation, sealed_tensors)
        else:
            cuda_source_attestation = self._require_current_cuda_attestation()
            if target.type == "cuda":
                if target.index is None or target.index == self.device.index:
                    return self
                raise ValueError("cross-device CUDA TokenBatch transfer is forbidden")
            transferred = _version_tracked_tensor_copies(
                _snapshot_cuda_sealed_tensors(cuda_source_attestation),
                device=target,
                non_blocking=non_blocking,
            )
            cuda_attestation = None

        return TokenBatch(
            token_ids=transferred[0],
            row_ids=transferred[1],
            column_ids=transferred[2],
            height_ids=transferred[3],
            width_ids=transferred[4],
            role_ids=transferred[5],
            pair_slot_ids=transferred[6],
            padding_mask=transferred[7],
            lengths=self.lengths,
            semantic_kind=self.semantic_kind,
            _validation_attestation=cuda_attestation,
        )

    def right_pad_to(self, padded_length: int) -> "TokenBatch":
        """Add semantic PAD suffixes, primarily for padding-invariance tests."""

        if self.device.type != "cpu":
            raise RuntimeError("right_pad_to is restricted to validated CPU batches")
        self._require_current_cpu_attestation()

        padded_length = _strict_int(padded_length, field="padded_length", minimum=1)
        if padded_length < self.padded_length:
            raise ValueError("padded_length cannot shrink a TokenBatch")
        if padded_length == self.padded_length:
            return self
        extra = padded_length - self.padded_length
        shape = (self.batch_size, extra)
        device = self.device
        enclosing_roles = self.role_ids[:, :1].expand(-1, extra)
        enclosing_slots = self.pair_slot_ids[:, :1].expand(-1, extra)

        def filled(value: int) -> Tensor:
            return torch.full(shape, value, dtype=torch.long, device=device)

        return TokenBatch(
            token_ids=torch.cat((self.token_ids, filled(PAD_TOKEN_ID)), dim=1),
            row_ids=torch.cat((self.row_ids, filled(GRID_CLS_POSITION_ID)), dim=1),
            column_ids=torch.cat((self.column_ids, filled(GRID_CLS_POSITION_ID)), dim=1),
            height_ids=torch.cat((self.height_ids, filled(RESERVED_SIZE_ID)), dim=1),
            width_ids=torch.cat((self.width_ids, filled(RESERVED_SIZE_ID)), dim=1),
            role_ids=torch.cat((self.role_ids, enclosing_roles), dim=1),
            pair_slot_ids=torch.cat((self.pair_slot_ids, enclosing_slots), dim=1),
            padding_mask=torch.cat(
                (
                    self.padding_mask,
                    torch.ones(shape, dtype=torch.bool, device=device),
                ),
                dim=1,
            ),
            lengths=self.lengths,
            semantic_kind=self.semantic_kind,
        )


def _build_token_batch(
    sequences: Sequence[dict[str, list[int]]],
    *,
    device: torch.device | str | None,
    semantic_kind: str,
) -> TokenBatch:
    if not sequences:
        raise ValueError("at least one token sequence is required")
    lengths = tuple(len(sequence["token"]) for sequence in sequences)
    padded_length = max(lengths)
    fields = ("token", "row", "column", "height", "width", "role", "slot")
    values: dict[str, list[list[int]]] = {field: [] for field in fields}
    padding_rows: list[list[bool]] = []
    for sequence_index, sequence in enumerate(sequences):
        if set(sequence) != set(fields):
            raise ValueError(f"sequence {sequence_index} has invalid embedding fields")
        length = lengths[sequence_index]
        if any(len(sequence[field]) != length for field in fields):
            raise ValueError(f"sequence {sequence_index} embedding fields do not align")
        extra = padded_length - length
        role = sequence["role"][0]
        slot = sequence["slot"][0]
        padding_values = {
            "token": PAD_TOKEN_ID,
            "row": GRID_CLS_POSITION_ID,
            "column": GRID_CLS_POSITION_ID,
            "height": RESERVED_SIZE_ID,
            "width": RESERVED_SIZE_ID,
            "role": role,
            "slot": slot,
        }
        for field_name in fields:
            values[field_name].append(
                sequence[field_name] + [padding_values[field_name]] * extra
            )
        padding_rows.append([False] * length + [True] * extra)

    def index_tensor(field: str) -> Tensor:
        return torch.tensor(values[field], dtype=torch.long, device="cpu")

    cpu_batch = TokenBatch(
        token_ids=index_tensor("token"),
        row_ids=index_tensor("row"),
        column_ids=index_tensor("column"),
        height_ids=index_tensor("height"),
        width_ids=index_tensor("width"),
        role_ids=index_tensor("role"),
        pair_slot_ids=index_tensor("slot"),
        padding_mask=torch.tensor(padding_rows, dtype=torch.bool, device="cpu"),
        lengths=lengths,
        semantic_kind=semantic_kind,
    )
    if device is None or torch.device(device).type == "cpu":
        return cpu_batch
    return cpu_batch.to(device)


def tokenize_encoder_grid_batch(
    grids: Sequence[object],
    roles: Sequence[int],
    pair_slots: Sequence[int],
    *,
    device: torch.device | str | None = None,
) -> TokenBatch:
    if isinstance(grids, (str, bytes)) or not isinstance(grids, Sequence) or not grids:
        raise TypeError("grids must be a non-empty sequence")
    if len(roles) != len(grids) or len(pair_slots) != len(grids):
        raise ValueError("roles and pair slots must align with grids")
    sequences: list[dict[str, list[int]]] = []
    for grid_index, (raw_grid, role, pair_slot) in enumerate(zip(grids, roles, pair_slots)):
        grid = _normalize_token_grid(
            raw_grid, field=f"grids[{grid_index}]", maximum_token=COLOR_TOKEN_COUNT - 1
        )
        role = _strict_int(role, field=f"roles[{grid_index}]")
        pair_slot = _strict_int(pair_slot, field=f"pair_slots[{grid_index}]")
        if role not in {DEMO_INPUT_ROLE_ID, DEMO_OUTPUT_ROLE_ID, QUERY_INPUT_ROLE_ID}:
            raise ValueError("encoder roles must be demo input/output or query input")
        if pair_slot >= PAIR_SLOT_EMBEDDING_COUNT:
            raise ValueError("pair slot exceeds the embedding table")
        height = len(grid)
        width = len(grid[0])
        tokens = [GRID_CLS_TOKEN_ID]
        rows = [GRID_CLS_POSITION_ID]
        columns = [GRID_CLS_POSITION_ID]
        for row_index, row in enumerate(grid):
            for column_index, token in enumerate(row):
                tokens.append(token)
                rows.append(row_index)
                columns.append(column_index)
        length = len(tokens)
        sequences.append(
            {
                "token": tokens,
                "row": rows,
                "column": columns,
                "height": [height] * length,
                "width": [width] * length,
                "role": [role] * length,
                "slot": [pair_slot] * length,
            }
        )
    return _build_token_batch(sequences, device=device, semantic_kind="encoder")


def _demonstration_grids(item: object, *, index: int) -> tuple[object, object]:
    if isinstance(item, Mapping):
        if set(item) != {"input", "output"}:
            raise ValueError(f"demonstrations[{index}] must contain exactly input and output")
        return item["input"], item["output"]
    if hasattr(item, "input_grid") and hasattr(item, "output_grid"):
        return getattr(item, "input_grid"), getattr(item, "output_grid")
    if isinstance(item, (str, bytes)) or not isinstance(item, Sequence) or len(item) != 2:
        raise TypeError(f"demonstrations[{index}] must be an input/output pair")
    return item[0], item[1]


def tokenize_task_memory(
    demonstrations: Sequence[object],
    query_input: object,
    *,
    device: torch.device | str | None = None,
) -> TokenBatch:
    if isinstance(demonstrations, (str, bytes)) or not isinstance(
        demonstrations, Sequence
    ):
        raise TypeError("demonstrations must be a sequence")
    demonstration_count = len(demonstrations)
    if not 1 <= demonstration_count <= MAX_DEMONSTRATIONS:
        raise ValueError(f"task requires 1..{MAX_DEMONSTRATIONS} demonstrations")
    grids: list[object] = []
    roles: list[int] = []
    slots: list[int] = []
    for index, item in enumerate(demonstrations):
        input_grid, output_grid = _demonstration_grids(item, index=index)
        grids.extend((input_grid, output_grid))
        roles.extend((DEMO_INPUT_ROLE_ID, DEMO_OUTPUT_ROLE_ID))
        slots.extend((index, index))
    grids.append(query_input)
    roles.append(QUERY_INPUT_ROLE_ID)
    slots.append(QUERY_TARGET_PAIR_SLOT)
    return tokenize_encoder_grid_batch(grids, roles, slots, device=device)


def tokenize_target_batch(
    targets: Sequence[object],
    *,
    device: torch.device | str | None = None,
) -> TokenBatch:
    if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence) or not targets:
        raise TypeError("targets must be a non-empty sequence")
    sequences: list[dict[str, list[int]]] = []
    for target_index, raw_grid in enumerate(targets):
        grid = _normalize_token_grid(
            raw_grid, field=f"targets[{target_index}]", maximum_token=MASK_TOKEN_ID
        )
        height = len(grid)
        width = len(grid[0])
        tokens: list[int] = []
        rows: list[int] = []
        columns: list[int] = []
        for row_index, row in enumerate(grid):
            for column_index, token in enumerate(row):
                tokens.append(token)
                rows.append(row_index)
                columns.append(column_index)
        length = len(tokens)
        sequences.append(
            {
                "token": tokens,
                "row": rows,
                "column": columns,
                "height": [height] * length,
                "width": [width] * length,
                "role": [TARGET_ROLE_ID] * length,
                "slot": [QUERY_TARGET_PAIR_SLOT] * length,
            }
        )
    return _build_token_batch(sequences, device=device, semantic_kind="target")


def mask_target_grid(target: object, masked_linear_indices: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    grid = _normalize_token_grid(target, field="target", maximum_token=COLOR_TOKEN_COUNT - 1)
    if isinstance(masked_linear_indices, (str, bytes)) or not isinstance(
        masked_linear_indices, Sequence
    ):
        raise TypeError("masked_linear_indices must be a sequence")
    masked = tuple(masked_linear_indices)
    cell_count = len(grid) * len(grid[0])
    if not masked or any(type(index) is not int for index in masked):
        raise TypeError("masked_linear_indices must contain integers")
    if masked != tuple(sorted(set(masked))) or masked[0] < 0 or masked[-1] >= cell_count:
        raise ValueError("masked_linear_indices must be sorted, unique, and in bounds")
    masked_set = set(masked)
    width = len(grid[0])
    return tuple(
        tuple(
            MASK_TOKEN_ID if row_index * width + column_index in masked_set else token
            for column_index, token in enumerate(row)
        )
        for row_index, row in enumerate(grid)
    )


def _zero_padding(hidden: Tensor, padding_mask: Tensor, *, has_padding: bool) -> Tensor:
    if not has_padding:
        return hidden
    return hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class GridEncoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            D_MODEL,
            ATTENTION_HEADS,
            dropout=DROPOUT_PROBABILITY,
            bias=True,
            batch_first=True,
        )
        self.linear1 = nn.Linear(D_MODEL, FFN_WIDTH, bias=True)
        self.dropout = nn.Dropout(DROPOUT_PROBABILITY)
        self.linear2 = nn.Linear(FFN_WIDTH, D_MODEL, bias=True)
        self.norm1 = nn.LayerNorm(D_MODEL, eps=LAYER_NORM_EPSILON, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(D_MODEL, eps=LAYER_NORM_EPSILON, elementwise_affine=True)
        self.dropout1 = nn.Dropout(DROPOUT_PROBABILITY)
        self.dropout2 = nn.Dropout(DROPOUT_PROBABILITY)
        self.activation = nn.GELU(approximate="none")

    def forward(
        self, hidden: Tensor, *, padding_mask: Tensor, has_padding: bool
    ) -> Tensor:
        normalized = self.norm1(hidden)
        attended = self.self_attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask if has_padding else None,
            need_weights=False,
            is_causal=False,
        )[0]
        hidden = _zero_padding(
            hidden + self.dropout1(attended), padding_mask, has_padding=has_padding
        )
        feed_forward = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(hidden)))))
        return _zero_padding(
            hidden + self.dropout2(feed_forward), padding_mask, has_padding=has_padding
        )


class GridDecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            D_MODEL,
            ATTENTION_HEADS,
            dropout=DROPOUT_PROBABILITY,
            bias=True,
            batch_first=True,
        )
        self.multihead_attn = nn.MultiheadAttention(
            D_MODEL,
            ATTENTION_HEADS,
            dropout=DROPOUT_PROBABILITY,
            bias=True,
            batch_first=True,
        )
        self.linear1 = nn.Linear(D_MODEL, FFN_WIDTH, bias=True)
        self.dropout = nn.Dropout(DROPOUT_PROBABILITY)
        self.linear2 = nn.Linear(FFN_WIDTH, D_MODEL, bias=True)
        self.norm1 = nn.LayerNorm(D_MODEL, eps=LAYER_NORM_EPSILON, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(D_MODEL, eps=LAYER_NORM_EPSILON, elementwise_affine=True)
        self.norm3 = nn.LayerNorm(D_MODEL, eps=LAYER_NORM_EPSILON, elementwise_affine=True)
        self.dropout1 = nn.Dropout(DROPOUT_PROBABILITY)
        self.dropout2 = nn.Dropout(DROPOUT_PROBABILITY)
        self.dropout3 = nn.Dropout(DROPOUT_PROBABILITY)
        self.activation = nn.GELU(approximate="none")

    def forward(
        self,
        hidden: Tensor,
        memory: Tensor,
        *,
        target_padding_mask: Tensor,
        target_lengths: tuple[int, ...],
        has_target_padding: bool,
        memory_key_padding_mask: Tensor,
    ) -> Tensor:
        normalized = self.norm1(hidden)
        self_attended = self.self_attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=target_padding_mask if has_target_padding else None,
            need_weights=False,
            is_causal=False,
        )[0]
        hidden = _zero_padding(
            hidden + self.dropout1(self_attended),
            target_padding_mask,
            has_padding=has_target_padding,
        )
        normalized = self.norm2(hidden)
        if has_target_padding:
            # MHA has only a key-padding mask.  Process each valid target prefix so
            # decoder PAD positions never become cross-attention queries.
            cross_rows: list[Tensor] = []
            padded_length = hidden.shape[1]
            for batch_index, valid_length in enumerate(target_lengths):
                row = self.multihead_attn(
                    normalized[batch_index : batch_index + 1, :valid_length],
                    memory[batch_index : batch_index + 1],
                    memory[batch_index : batch_index + 1],
                    key_padding_mask=memory_key_padding_mask[
                        batch_index : batch_index + 1
                    ],
                    need_weights=False,
                    is_causal=False,
                )[0]
                if valid_length < padded_length:
                    row = torch.cat(
                        (
                            row,
                            row.new_zeros((1, padded_length - valid_length, D_MODEL)),
                        ),
                        dim=1,
                    )
                cross_rows.append(row)
            cross_attended = torch.cat(cross_rows, dim=0)
        else:
            cross_attended = self.multihead_attn(
                normalized,
                memory,
                memory,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,
                is_causal=False,
            )[0]
        hidden = _zero_padding(
            hidden + self.dropout2(cross_attended),
            target_padding_mask,
            has_padding=has_target_padding,
        )
        feed_forward = self.linear2(self.dropout(self.activation(self.linear1(self.norm3(hidden)))))
        return _zero_padding(
            hidden + self.dropout3(feed_forward),
            target_padding_mask,
            has_padding=has_target_padding,
        )


def _reset_multihead_attention(module: nn.MultiheadAttention) -> None:
    if module.in_proj_weight is not None:
        nn.init.xavier_uniform_(module.in_proj_weight)
    else:
        for weight in (module.q_proj_weight, module.k_proj_weight, module.v_proj_weight):
            if weight is None:
                raise RuntimeError("MHA projection weights are incomplete")
            nn.init.xavier_uniform_(weight)
    if module.in_proj_bias is not None:
        nn.init.zeros_(module.in_proj_bias)
    nn.init.xavier_uniform_(module.out_proj.weight)
    if module.out_proj.bias is not None:
        nn.init.zeros_(module.out_proj.bias)


def _reset_linear(module: nn.Linear) -> None:
    nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _reset_layer_norm(module: nn.LayerNorm) -> None:
    if module.elementwise_affine:
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


@dataclass(frozen=True, slots=True)
class EncodedTaskMemory:
    memory: Tensor
    key_padding_mask: Tensor
    grid_lengths: tuple[int, ...]
    is_bf16_cuda_cache: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.memory, Tensor) or not isinstance(self.key_padding_mask, Tensor):
            raise TypeError("task memory fields must be tensors")
        if self.memory.ndim != 3 or self.memory.shape[0] != 1 or self.memory.shape[2] != D_MODEL:
            raise ValueError("task memory must have shape [1, memory_length, d_model]")
        if self.key_padding_mask.shape != self.memory.shape[:2]:
            raise ValueError("task memory key mask shape mismatch")
        if self.key_padding_mask.dtype != torch.bool:
            raise TypeError("task memory key mask must use torch.bool")
        if self.key_padding_mask.device != self.memory.device:
            raise ValueError("task memory and key mask must share one device")
        lengths = tuple(self.grid_lengths)
        if not lengths or any(type(length) is not int or length <= 0 for length in lengths):
            raise ValueError("grid_lengths must be a non-empty tuple of positive integers")
        if sum(lengths) != self.memory.shape[1]:
            raise ValueError("grid_lengths do not close against task memory length")
        if self.key_padding_mask.device.type == "cpu" and _cpu_any(
            self.key_padding_mask
        ):
            raise ValueError("concatenated task memory must have PAD stripped")
        if self.is_bf16_cuda_cache and (
            not self.memory.is_cuda or self.memory.dtype != torch.bfloat16
        ):
            raise ValueError("deployment cache must be CUDA BF16")
        object.__setattr__(self, "grid_lengths", lengths)

    @property
    def memory_length(self) -> int:
        return int(self.memory.shape[1])

    def to_bf16_cuda_cache(self) -> "EncodedTaskMemory":
        if not self.memory.is_cuda:
            raise ValueError("deployment cache requires CUDA task memory")
        return EncodedTaskMemory(
            memory=self.memory.detach().to(dtype=torch.bfloat16).contiguous(),
            key_padding_mask=self.key_padding_mask.detach().to(dtype=torch.bool).contiguous(),
            grid_lengths=self.grid_lengths,
            is_bf16_cuda_cache=True,
        )

    def expand_for_batch(self, batch_size: int) -> tuple[Tensor, Tensor]:
        batch_size = _strict_int(batch_size, field="batch_size", minimum=1)
        memory = self.memory.squeeze(0)
        key_padding_mask = self.key_padding_mask.squeeze(0)
        return (
            memory.unsqueeze(0).expand(batch_size, -1, -1).contiguous(),
            key_padding_mask.unsqueeze(0).expand(batch_size, -1).contiguous(),
        )

@dataclass(frozen=True, slots=True)
class DecodedTarget:
    logits: Tensor
    hidden_states: Tensor
    padding_mask: Tensor
    lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.logits.ndim != 3 or self.logits.shape[-1] != OUTPUT_COLOR_COUNT:
            raise ValueError("decoder logits must have shape [batch, sequence, 10]")
        if self.hidden_states.shape != (*self.logits.shape[:2], D_MODEL):
            raise ValueError("decoder hidden-state shape mismatch")
        if self.padding_mask.shape != self.logits.shape[:2] or self.padding_mask.dtype != torch.bool:
            raise ValueError("decoder padding-mask shape or dtype mismatch")
        if self.logits.device != self.hidden_states.device or self.logits.device != self.padding_mask.device:
            raise ValueError("decoder outputs must share one device")
        lengths = tuple(self.lengths)
        if len(lengths) != self.logits.shape[0]:
            raise ValueError("decoder lengths do not close against batch size")
        object.__setattr__(self, "lengths", lengths)


class GridCMLM(nn.Module):
    """The exact 8,733,706-parameter M04a model."""

    def __init__(self) -> None:
        super().__init__()
        # Frozen construction order.
        self.token_embedding = nn.Embedding(TOKEN_EMBEDDING_COUNT, D_MODEL)
        self.row_embedding = nn.Embedding(POSITION_EMBEDDING_COUNT, D_MODEL)
        self.column_embedding = nn.Embedding(POSITION_EMBEDDING_COUNT, D_MODEL)
        self.height_embedding = nn.Embedding(POSITION_EMBEDDING_COUNT, D_MODEL)
        self.width_embedding = nn.Embedding(POSITION_EMBEDDING_COUNT, D_MODEL)
        self.role_embedding = nn.Embedding(ROLE_EMBEDDING_COUNT, D_MODEL)
        self.pair_slot_embedding = nn.Embedding(PAIR_SLOT_EMBEDDING_COUNT, D_MODEL)
        self.encoder_layers = nn.ModuleList(
            GridEncoderLayer() for _ in range(ENCODER_LAYER_COUNT)
        )
        self.decoder_layers = nn.ModuleList(
            GridDecoderLayer() for _ in range(DECODER_LAYER_COUNT)
        )
        self.encoder_final_norm = nn.LayerNorm(
            D_MODEL, eps=LAYER_NORM_EPSILON, elementwise_affine=True
        )
        self.decoder_final_norm = nn.LayerNorm(
            D_MODEL, eps=LAYER_NORM_EPSILON, elementwise_affine=True
        )
        self.output_head = nn.Linear(D_MODEL, OUTPUT_COLOR_COUNT, bias=True)
        self.reset_contract_parameters()
        actual = trainable_parameter_count(self)
        if actual != MODEL_PARAMETER_COUNT:
            raise RuntimeError(
                f"Grid-CMLM parameter closure mismatch: {actual} != {MODEL_PARAMETER_COUNT}"
            )

    def reset_contract_parameters(self) -> None:
        torch.manual_seed(TRAINING_SEED)
        with torch.no_grad():
            for embedding in (
                self.token_embedding,
                self.row_embedding,
                self.column_embedding,
                self.height_embedding,
                self.width_embedding,
                self.role_embedding,
                self.pair_slot_embedding,
            ):
                nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
            for layer in self.encoder_layers:
                _reset_multihead_attention(layer.self_attn)
                _reset_linear(layer.linear1)
                _reset_linear(layer.linear2)
                _reset_layer_norm(layer.norm1)
                _reset_layer_norm(layer.norm2)
            for layer in self.decoder_layers:
                _reset_multihead_attention(layer.self_attn)
                _reset_multihead_attention(layer.multihead_attn)
                _reset_linear(layer.linear1)
                _reset_linear(layer.linear2)
                _reset_layer_norm(layer.norm1)
                _reset_layer_norm(layer.norm2)
                _reset_layer_norm(layer.norm3)
            _reset_layer_norm(self.encoder_final_norm)
            _reset_layer_norm(self.decoder_final_norm)
            _reset_linear(self.output_head)

    def _embed_tensor_snapshot(
        self,
        tensors: tuple[Tensor, ...],
        *,
        has_padding: bool,
    ) -> Tensor:
        (
            token_ids,
            row_ids,
            column_ids,
            height_ids,
            width_ids,
            role_ids,
            pair_slot_ids,
            padding_mask,
        ) = tensors
        hidden = (
            self.token_embedding(token_ids)
            + self.row_embedding(row_ids)
            + self.column_embedding(column_ids)
            + self.height_embedding(height_ids)
            + self.width_embedding(width_ids)
            + self.role_embedding(role_ids)
            + self.pair_slot_embedding(pair_slot_ids)
        )
        return _zero_padding(hidden, padding_mask, has_padding=has_padding)

    def embed(self, batch: TokenBatch) -> Tensor:
        if not isinstance(batch, TokenBatch):
            raise TypeError("GridCMLM.embed requires a TokenBatch")
        tensors = batch._tensors_for_model_consumption()
        return self._embed_tensor_snapshot(
            tensors, has_padding=batch.has_padding
        )

    def encode_grid_batch(self, batch: TokenBatch) -> Tensor:
        if batch.semantic_kind != "encoder":
            raise ValueError("encode_grid_batch requires an encoder TokenBatch")
        tensors = batch._tensors_for_model_consumption()
        padding_mask = tensors[-1]
        hidden = self._embed_tensor_snapshot(
            tensors, has_padding=batch.has_padding
        )
        for layer in self.encoder_layers:
            hidden = layer(
                hidden,
                padding_mask=padding_mask,
                has_padding=batch.has_padding,
            )
        hidden = self.encoder_final_norm(hidden)
        return _zero_padding(
            hidden, padding_mask, has_padding=batch.has_padding
        )

    def encode_task_memory(self, batch: TokenBatch) -> EncodedTaskMemory:
        encoded_grids = self.encode_grid_batch(batch)
        unpadded = [
            encoded_grids[index, :length]
            for index, length in enumerate(batch.lengths)
        ]
        memory = torch.cat(unpadded, dim=0).unsqueeze(0).contiguous()
        key_padding_mask = torch.zeros(
            (1, memory.shape[1]), dtype=torch.bool, device=memory.device
        )
        return EncodedTaskMemory(
            memory=memory,
            key_padding_mask=key_padding_mask,
            grid_lengths=batch.lengths,
        )

    def decode_target(
        self,
        batch: TokenBatch,
        task_memory: EncodedTaskMemory,
    ) -> DecodedTarget:
        if batch.device != task_memory.memory.device:
            raise ValueError("target tokens and task memory must share one device")
        if batch.semantic_kind != "target":
            raise ValueError("decode_target requires a target TokenBatch")
        tensors = batch._tensors_for_model_consumption()
        target_padding_mask = tensors[-1]
        memory, memory_key_padding_mask = task_memory.expand_for_batch(
            batch.batch_size
        )
        hidden = self._embed_tensor_snapshot(
            tensors, has_padding=batch.has_padding
        )
        for layer in self.decoder_layers:
            hidden = layer(
                hidden,
                memory,
                target_padding_mask=target_padding_mask,
                target_lengths=batch.lengths,
                has_target_padding=batch.has_padding,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        hidden = _zero_padding(
            self.decoder_final_norm(hidden),
            target_padding_mask,
            has_padding=batch.has_padding,
        )
        if batch.has_padding:
            logit_rows: list[Tensor] = []
            for batch_index, valid_length in enumerate(batch.lengths):
                valid_logits = self.output_head(
                    hidden[batch_index : batch_index + 1, :valid_length]
                )
                if valid_length < batch.padded_length:
                    valid_logits = torch.cat(
                        (
                            valid_logits,
                            valid_logits.new_zeros(
                                (
                                    1,
                                    batch.padded_length - valid_length,
                                    OUTPUT_COLOR_COUNT,
                                )
                            ),
                        ),
                        dim=1,
                    )
                logit_rows.append(valid_logits)
            logits = torch.cat(logit_rows, dim=0)
        else:
            logits = self.output_head(hidden)
        return DecodedTarget(
            logits=logits,
            hidden_states=hidden,
            # Never expose the CUDA TokenBatch's private sealed padding tensor.
            # Decoded outputs are caller-owned and may be mutated or retained.
            padding_mask=target_padding_mask.detach().clone(),
            lengths=batch.lengths,
        )

    def forward(self, task_batch: TokenBatch, target_batch: TokenBatch) -> DecodedTarget:
        return self.decode_target(target_batch, self.encode_task_memory(task_batch))


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def parameter_state_sha256(module: nn.Module) -> str:
    """Hash ordered parameter names, metadata, and exact raw CPU tensor bytes."""

    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        cpu = parameter.detach().cpu().contiguous()
        metadata = f"{name}\0{cpu.dtype}\0{tuple(cpu.shape)}".encode("ascii")
        raw = memoryview(cpu.numpy()).cast("B")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def model_config() -> dict[str, Any]:
    return {
        "model_semantics_version": MODEL_SEMANTICS_VERSION,
        "model_stage": MODEL_STAGE,
        "d_model": D_MODEL,
        "attention_heads": ATTENTION_HEADS,
        "ffn_width": FFN_WIDTH,
        "encoder_layers": ENCODER_LAYER_COUNT,
        "decoder_layers": DECODER_LAYER_COUNT,
        "dropout": DROPOUT_PROBABILITY,
        "layer_norm_epsilon": LAYER_NORM_EPSILON,
        "token_embeddings": TOKEN_EMBEDDING_COUNT,
        "position_embeddings": POSITION_EMBEDDING_COUNT,
        "role_embeddings": ROLE_EMBEDDING_COUNT,
        "pair_slot_embeddings": PAIR_SLOT_EMBEDDING_COUNT,
        "output_colors": OUTPUT_COLOR_COUNT,
        "parameter_count": MODEL_PARAMETER_COUNT,
        "initialization_seed": TRAINING_SEED,
        "target_has_grid_cls": False,
        "embedding_combination": "direct_unscaled_sum",
        "norm_first": True,
        "causal_decoder": False,
    }


__all__ = [
    "COLOR_TOKEN_COUNT",
    "DEMO_INPUT_ROLE_ID",
    "DEMO_OUTPUT_ROLE_ID",
    "DecodedTarget",
    "EncodedTaskMemory",
    "GRID_CLS_POSITION_ID",
    "GRID_CLS_TOKEN_ID",
    "GridCMLM",
    "GridDecoderLayer",
    "GridEncoderLayer",
    "MASK_TOKEN_ID",
    "PAD_TOKEN_ID",
    "QUERY_INPUT_ROLE_ID",
    "RESERVED_SIZE_ID",
    "TARGET_ROLE_ID",
    "TokenBatch",
    "mask_target_grid",
    "model_config",
    "parameter_state_sha256",
    "tokenize_encoder_grid_batch",
    "tokenize_target_batch",
    "tokenize_task_memory",
    "trainable_parameter_count",
]
