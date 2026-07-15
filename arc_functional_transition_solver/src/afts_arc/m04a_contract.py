"""Pure-Python semantic and accounting contract for the M04a Grid-CMLM.

This module deliberately has no optional-runtime imports.  In particular, it can
be imported on the local evidence host without PyTorch being installed.  Neural
modules consume these constants and validators, rather than re-encoding the
frozen schedule, seed, orbit, or cost semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias


# Frozen semantic identities.
MODEL_SEMANTICS_VERSION = "afts-grid-cmlm/v0.1"
MODEL_STAGE = "M04a_global_masked_grid_generation"
SAMPLER_SEMANTICS_VERSION = "afts-maskgit-cosine/v0.1"
DATA_FOLDS_SEMANTICS_VERSION = "afts-m04a-parent-folds/v0.1"
EPISODE_SEMANTICS_VERSION = "afts-grid-cmlm-episodes/v0.1"
VALIDATION_SEMANTICS_VERSION = "afts-grid-cmlm-validation/v0.1"
REARC_PARENT_SPLIT_SEMANTICS_VERSION = "m04a-rearc-parent-split/v0.1"
TASK_ORBIT_SEMANTICS_VERSION = "afts-task-d4-color-orbit/v0.1"
SAMPLE_ORBIT_SEMANTICS_VERSION = "afts-sample-d4-color-orbit/v0.1"
OUTPUT_SHAPE_SEMANTICS_VERSION = "afts-output-shape/v0.1"

# Frozen row and ledger schemas.
TRAINING_COST_LEDGER_SCHEMA_VERSION = "afts-training-cost-ledger/v0.1"
LANE_ROW_SCHEMA_VERSION = "afts-m04a-lane/v0.1"
LANE_TRACE_SCHEMA_VERSION = "afts-maskgit-trace/v0.1"
ENCODER_FORWARD_LEDGER_SCHEMA_VERSION = "afts-encoder-forward-ledger/v0.1"
DECODER_FORWARD_LEDGER_SCHEMA_VERSION = "afts-decoder-forward-ledger/v0.1"
PAIR_COST_SCHEMA_VERSION = "afts-m04a-pair-cost/v0.1"
POOL_SETUP_COST_SCHEMA_VERSION = "afts-m04a-pool-setup-cost/v0.1"

# Frozen data-stream and model constants.
TRAINING_SEED = 20260711
OPTIMIZER_UPDATES = 20_000
GRADIENT_ACCUMULATION = 16
VALIDATION_INTERVAL = 2_000
VALIDATION_PASS_COUNT = OPTIMIZER_UPDATES // VALIDATION_INTERVAL
MAX_GRID_SIDE = 30
MAX_GRID_CELLS = MAX_GRID_SIDE * MAX_GRID_SIDE
MAX_DEMONSTRATIONS = 10
D_MODEL = 256
ATTENTION_HEADS = 8
FFN_WIDTH = 1_024
ENCODER_LAYER_COUNT = 3
DECODER_LAYER_COUNT = 6
TOKEN_EMBEDDING_COUNT = 13
POSITION_EMBEDDING_COUNT = 31
ROLE_EMBEDDING_COUNT = 4
PAIR_SLOT_EMBEDDING_COUNT = 16
QUERY_TARGET_PAIR_SLOT = 15
OUTPUT_COLOR_COUNT = 10
LAYER_NORM_EPSILON = 1e-5
DROPOUT_PROBABILITY = 0.1
DENOISING_STEPS = 12
LANES_PER_PAIR = 64
MAX_ACCEPTED_SHAPES = 4
INFERENCE_MICROBATCH = 8

ARC2_SPLIT_SALT = "afts-e01-public-train-split-v1"
PUBLIC_SMOKE_SELECTION_SALT = "afts-e01-public-train-development-smoke-v1"


Grid: TypeAlias = tuple[tuple[int, ...], ...]


def _strict_int(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(f"{field} must be an integer >= {minimum}")
    return value


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field} must be a non-empty string")
    return value


def _normalize_grid(value: object, *, field: str = "grid") -> Grid:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a non-empty rectangular sequence")
    rows = tuple(value)
    if not rows:
        raise ValueError(f"{field} must not be empty")
    normalized: list[tuple[int, ...]] = []
    width: int | None = None
    for row_index, row in enumerate(rows):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise TypeError(f"{field}[{row_index}] must be a non-empty sequence")
        cells = tuple(row)
        if not cells:
            raise ValueError(f"{field}[{row_index}] must not be empty")
        if width is None:
            width = len(cells)
        elif len(cells) != width:
            raise ValueError(f"{field} must be rectangular")
        for column_index, cell in enumerate(cells):
            if type(cell) is not int or not 0 <= cell <= 9:
                raise TypeError(
                    f"{field}[{row_index}][{column_index}] must be an integer color 0-9"
                )
        normalized.append(cells)
    # Orbit audits must still canonicalize the over-size descendants that the
    # separate dimension audit later quarantines.  The model/schedule boundary is
    # therefore enforced by their consumers, not by this semantic grid parser.
    if width is None:  # Defensive; non-empty rows make this unreachable.
        raise RuntimeError(f"{field} width was not established")
    return tuple(normalized)


def canonical_json_bytes(payload: object) -> bytes:
    """Serialize with the exact canonical JSON settings used by M04a traces."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def trace_sha256(trace: Mapping[str, Any]) -> str:
    if not isinstance(trace, Mapping):
        raise TypeError("trace must be a mapping")
    if trace.get("schema") != LANE_TRACE_SCHEMA_VERSION:
        raise ValueError("trace schema does not match the frozen M04a trace schema")
    return canonical_sha256(dict(trace))


def float64_sequence_sha256(values: Iterable[int | float]) -> str:
    """Hash finite float64 values as concatenated little-endian IEEE-754 bytes."""

    digest = hashlib.sha256()
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"values[{index}] must be numeric")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"values[{index}] must be finite")
        digest.update(struct.pack("<d", normalized))
    return digest.hexdigest()


def _rotate90(grid: Grid) -> Grid:
    return tuple(tuple(row) for row in zip(*grid[::-1]))


def _rotate180(grid: Grid) -> Grid:
    return tuple(tuple(reversed(row)) for row in reversed(grid))


def _rotate270(grid: Grid) -> Grid:
    return tuple(tuple(row) for row in zip(*grid))[::-1]


def _reflect_left_to_right(grid: Grid) -> Grid:
    return tuple(tuple(reversed(row)) for row in grid)


D4_INDEX_NAMES: tuple[str, ...] = (
    "identity",
    "rotate90",
    "rotate180",
    "rotate270",
    "reflect_left_to_right",
    "reflect_left_to_right_rotate90",
    "reflect_left_to_right_rotate180",
    "reflect_left_to_right_rotate270",
)


def apply_indexed_d4(grid: object, index: int) -> Grid:
    """Apply the exact indexed D4 convention without symmetry deduplication."""

    normalized = _normalize_grid(grid)
    index = _strict_int(index, field="index")
    if index >= 8:
        raise ValueError("index must be in 0..7")
    if index < 4:
        reflected = normalized
        rotations = index
    else:
        reflected = _reflect_left_to_right(normalized)
        rotations = index - 4
    return (reflected, _rotate90(reflected), _rotate180(reflected), _rotate270(reflected))[
        rotations
    ]


@dataclass(frozen=True, slots=True)
class OrbitPair:
    input_grid: Grid
    output_grid: Grid

    @classmethod
    def create(cls, input_grid: object, output_grid: object) -> "OrbitPair":
        return cls(
            input_grid=_normalize_grid(input_grid, field="input"),
            output_grid=_normalize_grid(output_grid, field="output"),
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_grid", _normalize_grid(self.input_grid, field="input"))
        object.__setattr__(
            self, "output_grid", _normalize_grid(self.output_grid, field="output")
        )


def _normalize_pairs(value: object, *, field: str) -> tuple[OrbitPair, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a sequence of input/output pairs")
    result: list[OrbitPair] = []
    for index, item in enumerate(value):
        if isinstance(item, OrbitPair):
            result.append(item)
            continue
        if isinstance(item, Mapping):
            if set(item) != {"input", "output"}:
                raise ValueError(f"{field}[{index}] must contain exactly input and output")
            result.append(OrbitPair.create(item["input"], item["output"]))
            continue
        if isinstance(item, (str, bytes)) or not isinstance(item, Sequence) or len(item) != 2:
            raise TypeError(f"{field}[{index}] must be an input/output pair")
        result.append(OrbitPair.create(item[0], item[1]))
    return tuple(result)


def _grid_payload(grid: Grid) -> dict[str, object]:
    return {
        "shape": [len(grid), len(grid[0])],
        "cells": [list(row) for row in grid],
    }


def _canonicalize_grid_colors(grid: Grid, mapping: dict[int, int]) -> Grid:
    rows: list[tuple[int, ...]] = []
    for row in grid:
        canonical_row: list[int] = []
        for color in row:
            if color not in mapping:
                mapping[color] = len(mapping)
            canonical_row.append(mapping[color])
        rows.append(tuple(canonical_row))
    return tuple(rows)


def sample_orbit_serializations(input_grid: object, output_grid: object) -> tuple[bytes, ...]:
    pair = OrbitPair.create(input_grid, output_grid)
    serializations: list[bytes] = []
    for d4_index in range(8):
        transformed_input = apply_indexed_d4(pair.input_grid, d4_index)
        transformed_output = apply_indexed_d4(pair.output_grid, d4_index)
        color_mapping: dict[int, int] = {}
        canonical_input = _canonicalize_grid_colors(transformed_input, color_mapping)
        canonical_output = _canonicalize_grid_colors(transformed_output, color_mapping)
        serializations.append(
            canonical_json_bytes(
                {
                    "schema": SAMPLE_ORBIT_SEMANTICS_VERSION,
                    "input": _grid_payload(canonical_input),
                    "output": _grid_payload(canonical_output),
                }
            )
        )
    return tuple(serializations)


def sample_orbit_id(input_grid: object, output_grid: object) -> str:
    return hashlib.sha256(min(sample_orbit_serializations(input_grid, output_grid))).hexdigest()


def task_orbit_serializations(
    train_pairs: object, test_pairs: object
) -> tuple[bytes, ...]:
    train = _normalize_pairs(train_pairs, field="train_pairs")
    test = _normalize_pairs(test_pairs, field="test_pairs")
    if not train or not test:
        raise ValueError("task orbit requires non-empty train and test pairs")

    serializations: list[bytes] = []
    for d4_index in range(8):
        color_mapping: dict[int, int] = {}

        def canonical_pair(pair: OrbitPair) -> dict[str, object]:
            transformed_input = apply_indexed_d4(pair.input_grid, d4_index)
            transformed_output = apply_indexed_d4(pair.output_grid, d4_index)
            canonical_input = _canonicalize_grid_colors(transformed_input, color_mapping)
            canonical_output = _canonicalize_grid_colors(transformed_output, color_mapping)
            return {
                "input": _grid_payload(canonical_input),
                "output": _grid_payload(canonical_output),
            }

        serializations.append(
            canonical_json_bytes(
                {
                    "schema": TASK_ORBIT_SEMANTICS_VERSION,
                    "train": [canonical_pair(pair) for pair in train],
                    "test": [canonical_pair(pair) for pair in test],
                }
            )
        )
    return tuple(serializations)


def task_orbit_id(train_pairs: object, test_pairs: object) -> str:
    return hashlib.sha256(min(task_orbit_serializations(train_pairs, test_pairs))).hexdigest()


def _sha256_seed(payload: bytes) -> int:
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def episode_seed(
    optimizer_step: int,
    microbatch_slot: int,
) -> int:
    optimizer_step = _strict_int(optimizer_step, field="optimizer_step")
    microbatch_slot = _strict_int(microbatch_slot, field="microbatch_slot")
    if optimizer_step >= OPTIMIZER_UPDATES:
        raise ValueError(f"optimizer_step must be in 0..{OPTIMIZER_UPDATES - 1}")
    if microbatch_slot >= GRADIENT_ACCUMULATION:
        raise ValueError(f"microbatch_slot must be in 0..{GRADIENT_ACCUMULATION - 1}")
    global_microbatch = GRADIENT_ACCUMULATION * optimizer_step + microbatch_slot
    payload = (
        f"{EPISODE_SEMANTICS_VERSION}\0{TRAINING_SEED}\0{global_microbatch}"
    ).encode("ascii")
    return _sha256_seed(payload)


def lane_seed(
    blind_task_id: str,
    test_index: int,
    shape_proposal_id: str,
    local_lane: int,
) -> int:
    blind_task_id = _nonempty_string(blind_task_id, field="blind_task_id")
    test_index = _strict_int(test_index, field="test_index")
    shape_proposal_id = _nonempty_string(shape_proposal_id, field="shape_proposal_id")
    local_lane = _strict_int(local_lane, field="local_lane")
    if local_lane >= LANES_PER_PAIR:
        raise ValueError(f"local_lane must be in 0..{LANES_PER_PAIR - 1}")
    payload = (
        f"{SAMPLER_SEMANTICS_VERSION}\0{blind_task_id}\0{test_index}\0"
        f"{shape_proposal_id}\0{local_lane}"
    ).encode("utf-8")
    return _sha256_seed(payload)


def validation_episode_seed(parent_id: str, target_descriptor: str, view_index: int) -> int:
    parent_id = _nonempty_string(parent_id, field="parent_id")
    target_descriptor = _nonempty_string(target_descriptor, field="target_descriptor")
    view_index = _strict_int(view_index, field="view_index")
    if view_index >= 4:
        raise ValueError("view_index must be in 0..3")
    payload = (
        f"{VALIDATION_SEMANTICS_VERSION}\0{parent_id}\0{target_descriptor}\0{view_index}"
    ).encode("utf-8")
    return _sha256_seed(payload)


def rearc_parent_split_bucket(parent_task_id: str) -> int:
    parent_task_id = _nonempty_string(parent_task_id, field="parent_task_id")
    payload = f"{REARC_PARENT_SPLIT_SEMANTICS_VERSION}\0{parent_task_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big", signed=False) % 100


_SHAPE_LANE_ALLOCATIONS: dict[int, tuple[int, ...]] = {
    0: (),
    1: (64,),
    2: (32, 32),
    3: (22, 21, 21),
    4: (16, 16, 16, 16),
}


def shape_lane_allocations(accepted_shape_count: int) -> tuple[int, ...]:
    accepted_shape_count = _strict_int(
        accepted_shape_count, field="accepted_shape_count"
    )
    try:
        return _SHAPE_LANE_ALLOCATIONS[accepted_shape_count]
    except KeyError as exc:
        raise ValueError("accepted_shape_count must be in 0..4") from exc


@dataclass(frozen=True, slots=True)
class LanePlanEntry:
    shape_order: int
    local_lane: int
    global_lane: int
    batch_index_within_shape: int
    greedy: bool


def lane_plan(accepted_shape_count: int) -> tuple[LanePlanEntry, ...]:
    allocations = shape_lane_allocations(accepted_shape_count)
    result: list[LanePlanEntry] = []
    global_offset = 0
    for shape_order, lane_count in enumerate(allocations):
        for local_lane in range(lane_count):
            result.append(
                LanePlanEntry(
                    shape_order=shape_order,
                    local_lane=local_lane,
                    global_lane=global_offset + local_lane,
                    batch_index_within_shape=local_lane // INFERENCE_MICROBATCH,
                    greedy=local_lane == 0,
                )
            )
        global_offset += lane_count
    return tuple(result)


def decoder_batch_calls(accepted_shape_count: int) -> int:
    return DENOISING_STEPS * sum(
        math.ceil(count / INFERENCE_MICROBATCH)
        for count in shape_lane_allocations(accepted_shape_count)
    )


def retained_mask_count(cell_count: int, step: int) -> int:
    cell_count = _strict_int(cell_count, field="cell_count", minimum=1)
    step = _strict_int(step, field="step", minimum=1)
    if cell_count > MAX_GRID_CELLS:
        raise ValueError(f"cell_count must be <= {MAX_GRID_CELLS}")
    if step > DENOISING_STEPS:
        raise ValueError(f"step must be in 1..{DENOISING_STEPS}")
    if step == DENOISING_STEPS:
        return 0
    if step == 8:
        return (cell_count + 1) // 2
    return math.ceil(cell_count * math.cos(math.pi * step / 24))


def mask_count_trace(cell_count: int) -> tuple[int, ...]:
    cell_count = _strict_int(cell_count, field="cell_count", minimum=1)
    if cell_count > MAX_GRID_CELLS:
        raise ValueError(f"cell_count must be <= {MAX_GRID_CELLS}")
    trace = (cell_count,) + tuple(
        retained_mask_count(cell_count, step) for step in range(1, DENOISING_STEPS + 1)
    )
    if any(after > before for before, after in zip(trace, trace[1:])):
        raise RuntimeError("frozen mask schedule is not monotonically non-increasing")
    return trace


def masked_token_predictions(cell_count: int) -> int:
    return sum(mask_count_trace(cell_count)[:DENOISING_STEPS])


def mask_schedule_table() -> tuple[tuple[int, ...], ...]:
    return tuple(mask_count_trace(cell_count) for cell_count in range(1, MAX_GRID_CELLS + 1))


def mask_schedule_table_sha256() -> str:
    return canonical_sha256(mask_schedule_table())


# Pinned by tests after deriving the complete N=1..900 table from the frozen rule.
MASK_SCHEDULE_TABLE_SHA256 = (
    "fe74fb990c24252dcd3d798d4303aba869b2e4403e49d6d22e2fb4a609f350c6"
)


@dataclass(frozen=True, slots=True)
class ModelParameterClosure:
    one_mha: int
    one_ffn: int
    one_encoder_layer: int
    encoder_layers: int
    one_decoder_layer: int
    decoder_layers: int
    transformer_core: int
    embeddings: int
    final_layer_norms: int
    output_head: int
    total: int

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            _strict_int(getattr(self, field_name), field=field_name)
        if self.encoder_layers != ENCODER_LAYER_COUNT * self.one_encoder_layer:
            raise ValueError("encoder layer parameter closure mismatch")
        if self.decoder_layers != DECODER_LAYER_COUNT * self.one_decoder_layer:
            raise ValueError("decoder layer parameter closure mismatch")
        if self.transformer_core != self.encoder_layers + self.decoder_layers:
            raise ValueError("Transformer core parameter closure mismatch")
        if self.total != (
            self.transformer_core
            + self.embeddings
            + self.final_layer_norms
            + self.output_head
        ):
            raise ValueError("total model parameter closure mismatch")


def model_parameter_closure() -> ModelParameterClosure:
    d_model = D_MODEL
    ffn_width = FFN_WIDTH
    one_layer_norm = 2 * d_model
    one_mha = 3 * d_model * d_model + 3 * d_model + d_model * d_model + d_model
    one_ffn = (
        d_model * ffn_width + ffn_width + ffn_width * d_model + d_model
    )
    one_encoder_layer = one_mha + one_ffn + 2 * one_layer_norm
    one_decoder_layer = 2 * one_mha + one_ffn + 3 * one_layer_norm
    encoder_layers = ENCODER_LAYER_COUNT * one_encoder_layer
    decoder_layers = DECODER_LAYER_COUNT * one_decoder_layer
    transformer_core = encoder_layers + decoder_layers
    embedding_rows = (
        TOKEN_EMBEDDING_COUNT
        + 4 * POSITION_EMBEDDING_COUNT
        + ROLE_EMBEDDING_COUNT
        + PAIR_SLOT_EMBEDDING_COUNT
    )
    embeddings = embedding_rows * d_model
    final_layer_norms = 2 * one_layer_norm
    output_head = d_model * OUTPUT_COLOR_COUNT + OUTPUT_COLOR_COUNT
    total = transformer_core + embeddings + final_layer_norms + output_head
    return ModelParameterClosure(
        one_mha=one_mha,
        one_ffn=one_ffn,
        one_encoder_layer=one_encoder_layer,
        encoder_layers=encoder_layers,
        one_decoder_layer=one_decoder_layer,
        decoder_layers=decoder_layers,
        transformer_core=transformer_core,
        embeddings=embeddings,
        final_layer_norms=final_layer_norms,
        output_head=output_head,
        total=total,
    )


MODEL_PARAMETER_CLOSURE = model_parameter_closure()
MODEL_PARAMETER_COUNT = MODEL_PARAMETER_CLOSURE.total


@dataclass(frozen=True, slots=True)
class TrainingCostClosure:
    optimizer_updates: int
    microbatches: int
    arc2_episodes: int
    rearc_episodes: int
    encoder_forward_calls: int
    decoder_forward_calls: int
    backward_calls: int
    validation_passes: int
    validation_episodes_per_pass: int
    validation_episode_calls: int
    validation_encoder_forward_calls: int
    validation_decoder_forward_calls: int
    checkpoint_writes: int

    def __post_init__(self) -> None:
        validate_training_cost_closure(self)

    @classmethod
    def expected(cls, *, validation_episodes_per_pass: int) -> "TrainingCostClosure":
        validation_episodes_per_pass = _strict_int(
            validation_episodes_per_pass,
            field="validation_episodes_per_pass",
            minimum=1,
        )
        validation_calls = VALIDATION_PASS_COUNT * validation_episodes_per_pass
        microbatches = OPTIMIZER_UPDATES * GRADIENT_ACCUMULATION
        return cls(
            optimizer_updates=OPTIMIZER_UPDATES,
            microbatches=microbatches,
            arc2_episodes=microbatches // 2,
            rearc_episodes=microbatches // 2,
            encoder_forward_calls=microbatches,
            decoder_forward_calls=microbatches,
            backward_calls=microbatches,
            validation_passes=VALIDATION_PASS_COUNT,
            validation_episodes_per_pass=validation_episodes_per_pass,
            validation_episode_calls=validation_calls,
            validation_encoder_forward_calls=validation_calls,
            validation_decoder_forward_calls=validation_calls,
            checkpoint_writes=VALIDATION_PASS_COUNT,
        )


def validate_training_cost_closure(closure: TrainingCostClosure) -> TrainingCostClosure:
    if not isinstance(closure, TrainingCostClosure):
        raise TypeError("closure must be a TrainingCostClosure")
    for field_name in closure.__dataclass_fields__:
        minimum = 1 if field_name == "validation_episodes_per_pass" else 0
        _strict_int(getattr(closure, field_name), field=field_name, minimum=minimum)
    expected_microbatches = OPTIMIZER_UPDATES * GRADIENT_ACCUMULATION
    expected_validation_calls = (
        VALIDATION_PASS_COUNT * closure.validation_episodes_per_pass
    )
    expected = {
        "optimizer_updates": OPTIMIZER_UPDATES,
        "microbatches": expected_microbatches,
        "arc2_episodes": expected_microbatches // 2,
        "rearc_episodes": expected_microbatches // 2,
        "encoder_forward_calls": expected_microbatches,
        "decoder_forward_calls": expected_microbatches,
        "backward_calls": expected_microbatches,
        "validation_passes": VALIDATION_PASS_COUNT,
        "validation_episode_calls": expected_validation_calls,
        "validation_encoder_forward_calls": expected_validation_calls,
        "validation_decoder_forward_calls": expected_validation_calls,
        "checkpoint_writes": VALIDATION_PASS_COUNT,
    }
    mismatches = {
        name: (getattr(closure, name), value)
        for name, value in expected.items()
        if getattr(closure, name) != value
    }
    if mismatches:
        raise ValueError(f"primary training cost closure mismatch: {mismatches}")
    return closure


@dataclass(frozen=True, slots=True)
class PairCostClosure:
    accepted_shape_count: int
    shape_lane_allocations: tuple[int, ...]
    shape_cell_counts: tuple[int, ...]
    raw_lanes: int
    sample_equivalent_forward_calls: int
    encoder_batch_calls: int
    decoder_batch_calls: int
    total_actual_batch_calls: int
    mask_traces: tuple[tuple[int, ...], ...]
    masked_token_predictions_per_lane: tuple[int, ...]
    format_valid: int
    format_invalid: int
    unique_outputs: int
    duplicate_outputs: int
    candidate_rows: int

    def __post_init__(self) -> None:
        validate_pair_cost_closure(self)

    @classmethod
    def expected(
        cls,
        *,
        shape_cell_counts: Sequence[int],
        format_invalid: int = 0,
        unique_outputs: int | None = None,
    ) -> "PairCostClosure":
        if isinstance(shape_cell_counts, (str, bytes)) or not isinstance(
            shape_cell_counts, Sequence
        ):
            raise TypeError("shape_cell_counts must be a sequence")
        accepted_shape_count = len(shape_cell_counts)
        allocations = shape_lane_allocations(accepted_shape_count)
        normalized_cell_counts: list[int] = []
        traces: list[tuple[int, ...]] = []
        predictions: list[int] = []
        for shape_order, cell_count in enumerate(shape_cell_counts):
            cell_count = _strict_int(
                cell_count, field=f"shape_cell_counts[{shape_order}]", minimum=1
            )
            if cell_count > MAX_GRID_CELLS:
                raise ValueError("shape cell count exceeds the 30-by-30 model boundary")
            normalized_cell_counts.append(cell_count)
            trace = mask_count_trace(cell_count)
            traces.extend([trace] * allocations[shape_order])
            predictions.extend([sum(trace[:DENOISING_STEPS])] * allocations[shape_order])
        raw_lanes = sum(allocations)
        format_invalid = _strict_int(format_invalid, field="format_invalid")
        if format_invalid > raw_lanes:
            raise ValueError("format_invalid exceeds raw lanes")
        format_valid = raw_lanes - format_invalid
        if unique_outputs is None:
            unique_outputs = format_valid
        unique_outputs = _strict_int(unique_outputs, field="unique_outputs")
        if unique_outputs > format_valid:
            raise ValueError("unique_outputs exceeds valid lanes")
        actual_decoder_calls = decoder_batch_calls(accepted_shape_count)
        encoder_calls = 1 if accepted_shape_count > 0 else 0
        return cls(
            accepted_shape_count=accepted_shape_count,
            shape_lane_allocations=allocations,
            shape_cell_counts=tuple(normalized_cell_counts),
            raw_lanes=raw_lanes,
            sample_equivalent_forward_calls=raw_lanes * DENOISING_STEPS,
            encoder_batch_calls=encoder_calls,
            decoder_batch_calls=actual_decoder_calls,
            total_actual_batch_calls=encoder_calls + actual_decoder_calls,
            mask_traces=tuple(traces),
            masked_token_predictions_per_lane=tuple(predictions),
            format_valid=format_valid,
            format_invalid=format_invalid,
            unique_outputs=unique_outputs,
            duplicate_outputs=format_valid - unique_outputs,
            candidate_rows=raw_lanes,
        )


def validate_pair_cost_closure(closure: PairCostClosure) -> PairCostClosure:
    if not isinstance(closure, PairCostClosure):
        raise TypeError("closure must be a PairCostClosure")
    accepted_shape_count = _strict_int(
        closure.accepted_shape_count, field="accepted_shape_count"
    )
    expected_allocations = shape_lane_allocations(accepted_shape_count)
    if type(closure.shape_lane_allocations) is not tuple or any(
        type(value) is not int for value in closure.shape_lane_allocations
    ):
        raise TypeError("shape_lane_allocations must be a tuple of integers")
    if closure.shape_lane_allocations != expected_allocations:
        raise ValueError("shape lane allocations do not match the frozen table")
    if (
        type(closure.shape_cell_counts) is not tuple
        or len(closure.shape_cell_counts) != accepted_shape_count
        or any(
            type(value) is not int or not 1 <= value <= MAX_GRID_CELLS
            for value in closure.shape_cell_counts
        )
    ):
        raise ValueError("shape_cell_counts must bind one 1..900 count per shape")

    numeric_fields = (
        "raw_lanes",
        "sample_equivalent_forward_calls",
        "encoder_batch_calls",
        "decoder_batch_calls",
        "total_actual_batch_calls",
        "format_valid",
        "format_invalid",
        "unique_outputs",
        "duplicate_outputs",
        "candidate_rows",
    )
    for field_name in numeric_fields:
        _strict_int(getattr(closure, field_name), field=field_name)

    expected_raw_lanes = sum(expected_allocations)
    expected_encoder_calls = 1 if accepted_shape_count > 0 else 0
    expected_decoder_calls = decoder_batch_calls(accepted_shape_count)
    expected_scalars = {
        "raw_lanes": expected_raw_lanes,
        "sample_equivalent_forward_calls": expected_raw_lanes * DENOISING_STEPS,
        "encoder_batch_calls": expected_encoder_calls,
        "decoder_batch_calls": expected_decoder_calls,
        "total_actual_batch_calls": expected_encoder_calls + expected_decoder_calls,
        "candidate_rows": expected_raw_lanes,
    }
    for field_name, expected in expected_scalars.items():
        if getattr(closure, field_name) != expected:
            raise ValueError(f"{field_name} does not close: expected {expected}")
    if closure.format_valid + closure.format_invalid != expected_raw_lanes:
        raise ValueError("format validity counts do not close to raw lanes")
    if closure.unique_outputs + closure.duplicate_outputs != closure.format_valid:
        raise ValueError("output deduplication counts do not close to valid lanes")

    if type(closure.mask_traces) is not tuple or len(closure.mask_traces) != expected_raw_lanes:
        raise ValueError("one mask trace is required for every raw lane")
    if (
        type(closure.masked_token_predictions_per_lane) is not tuple
        or len(closure.masked_token_predictions_per_lane) != expected_raw_lanes
    ):
        raise ValueError("one masked-token count is required for every raw lane")
    expected_lane_cell_counts = tuple(
        cell_count
        for cell_count, allocation in zip(
            closure.shape_cell_counts, expected_allocations
        )
        for _ in range(allocation)
    )
    for lane, (trace, predictions, expected_cell_count) in enumerate(
        zip(
            closure.mask_traces,
            closure.masked_token_predictions_per_lane,
            expected_lane_cell_counts,
        )
    ):
        if type(trace) is not tuple or len(trace) != DENOISING_STEPS + 1:
            raise ValueError(f"mask trace {lane} must have 13 counts")
        if any(type(value) is not int or value < 0 for value in trace):
            raise TypeError(f"mask trace {lane} must contain non-negative integers")
        if trace[0] != expected_cell_count:
            raise ValueError(f"mask trace {lane} does not bind its declared shape size")
        if trace != mask_count_trace(trace[0]):
            raise ValueError(f"mask trace {lane} does not match the frozen schedule")
        if type(predictions) is not int or predictions != sum(trace[:DENOISING_STEPS]):
            raise ValueError(f"masked-token count {lane} does not close from its trace")
    return closure


__all__ = [
    "ARC2_SPLIT_SALT",
    "ATTENTION_HEADS",
    "DATA_FOLDS_SEMANTICS_VERSION",
    "DECODER_LAYER_COUNT",
    "D4_INDEX_NAMES",
    "DECODER_FORWARD_LEDGER_SCHEMA_VERSION",
    "DENOISING_STEPS",
    "D_MODEL",
    "DROPOUT_PROBABILITY",
    "ENCODER_LAYER_COUNT",
    "ENCODER_FORWARD_LEDGER_SCHEMA_VERSION",
    "EPISODE_SEMANTICS_VERSION",
    "FFN_WIDTH",
    "GRADIENT_ACCUMULATION",
    "INFERENCE_MICROBATCH",
    "LANES_PER_PAIR",
    "LANE_ROW_SCHEMA_VERSION",
    "LANE_TRACE_SCHEMA_VERSION",
    "LanePlanEntry",
    "LAYER_NORM_EPSILON",
    "MASK_SCHEDULE_TABLE_SHA256",
    "MAX_ACCEPTED_SHAPES",
    "MAX_DEMONSTRATIONS",
    "MAX_GRID_CELLS",
    "MAX_GRID_SIDE",
    "MODEL_PARAMETER_CLOSURE",
    "MODEL_PARAMETER_COUNT",
    "MODEL_SEMANTICS_VERSION",
    "MODEL_STAGE",
    "ModelParameterClosure",
    "OPTIMIZER_UPDATES",
    "OUTPUT_COLOR_COUNT",
    "OUTPUT_SHAPE_SEMANTICS_VERSION",
    "OrbitPair",
    "PAIR_COST_SCHEMA_VERSION",
    "POOL_SETUP_COST_SCHEMA_VERSION",
    "POSITION_EMBEDDING_COUNT",
    "PUBLIC_SMOKE_SELECTION_SALT",
    "PAIR_SLOT_EMBEDDING_COUNT",
    "PairCostClosure",
    "QUERY_TARGET_PAIR_SLOT",
    "REARC_PARENT_SPLIT_SEMANTICS_VERSION",
    "SAMPLE_ORBIT_SEMANTICS_VERSION",
    "SAMPLER_SEMANTICS_VERSION",
    "TASK_ORBIT_SEMANTICS_VERSION",
    "TOKEN_EMBEDDING_COUNT",
    "TRAINING_COST_LEDGER_SCHEMA_VERSION",
    "TRAINING_SEED",
    "TrainingCostClosure",
    "ROLE_EMBEDDING_COUNT",
    "VALIDATION_INTERVAL",
    "VALIDATION_PASS_COUNT",
    "VALIDATION_SEMANTICS_VERSION",
    "apply_indexed_d4",
    "canonical_json_bytes",
    "canonical_sha256",
    "decoder_batch_calls",
    "episode_seed",
    "float64_sequence_sha256",
    "lane_plan",
    "lane_seed",
    "mask_count_trace",
    "mask_schedule_table",
    "mask_schedule_table_sha256",
    "masked_token_predictions",
    "model_parameter_closure",
    "rearc_parent_split_bucket",
    "retained_mask_count",
    "sample_orbit_id",
    "sample_orbit_serializations",
    "shape_lane_allocations",
    "task_orbit_id",
    "task_orbit_serializations",
    "trace_sha256",
    "validate_pair_cost_closure",
    "validate_training_cost_closure",
    "validation_episode_seed",
]
