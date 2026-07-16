"""Categorical cellular automata and constructive Boolean codec bridges for ARC.

The learner is deliberately small and auditable.  It selects one deterministic
spatially shared lookup table from demonstrations only.  The compiled binary4
and one-hot paths are constructive equality/multiplexer networks; their gate
counts are declared upper bounds, not synthesis minima.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
import time
from typing import Iterable, Sequence

import numpy as np

from arc_data import ARC_COLOR_COUNT, ArcExample


BOUNDARY_COLOR = 10
BINARY4_BITS = 4
ONEHOT_INPUT_STATES = 11
ONEHOT_OUTPUT_STATES = 10


def _categorical_grid(value: object, name: str = "grid") -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or min(array.shape, default=0) == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional grid")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integer colors")
    if not bool(np.all((0 <= array) & (array < ARC_COLOR_COUNT))):
        raise ValueError(f"{name} colors must be between 0 and 9")
    return array.astype(np.uint8, copy=False)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def elias_delta_bits(value: int) -> int:
    """Length of an Elias-delta code for a positive integer."""
    value = _positive_integer(value, "value")
    width = value.bit_length()
    return width + 2 * (width.bit_length() - 1)


@dataclass(frozen=True)
class NeighborhoodSpec:
    name: str
    offsets: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.offsets or self.offsets[0] != (0, 0):
            raise ValueError("every neighborhood must contain center as its first offset")
        if len(set(self.offsets)) != len(self.offsets):
            raise ValueError("neighborhood offsets must be unique")
        if any(max(abs(row), abs(column)) > 2 for row, column in self.offsets):
            raise ValueError("the frozen ARC grammar is bounded to radius two")


def frozen_neighborhood_specs() -> tuple[NeighborhoodSpec, ...]:
    """Return the predeclared ARC local-rule grammar in stable order."""
    center = NeighborhoodSpec("center", ((0, 0),))
    wires = tuple(
        NeighborhoodSpec(
            f"center_wire_{row:+d}_{column:+d}",
            ((0, 0), (row, column)),
        )
        for row in range(-2, 3)
        for column in range(-2, 3)
        if (row, column) != (0, 0)
    )
    cross1 = NeighborhoodSpec(
        "von_neumann_r1",
        ((0, 0), (-1, 0), (0, -1), (0, 1), (1, 0)),
    )
    moore1 = NeighborhoodSpec(
        "moore_r1",
        ((0, 0),)
        + tuple(
            (row, column)
            for row in range(-1, 2)
            for column in range(-1, 2)
            if (row, column) != (0, 0)
        ),
    )
    cross2 = NeighborhoodSpec(
        "von_neumann_r2",
        ((0, 0),)
        + tuple(
            (row, column)
            for row in range(-2, 3)
            for column in range(-2, 3)
            if (row, column) != (0, 0) and abs(row) + abs(column) <= 2
        ),
    )
    moore2 = NeighborhoodSpec(
        "moore_r2",
        ((0, 0),)
        + tuple(
            (row, column)
            for row in range(-2, 3)
            for column in range(-2, 3)
            if (row, column) != (0, 0)
        ),
    )
    return (center, *wires, cross1, moore1, cross2, moore2)


FROZEN_NEIGHBORHOODS = frozen_neighborhood_specs()


def extract_neighborhoods(grid: object, spec: NeighborhoodSpec) -> np.ndarray:
    """Return ``H x W x K`` categorical patches with a distinct boundary symbol."""
    grid = _categorical_grid(grid)
    height, width = grid.shape
    patches = np.full((height, width, len(spec.offsets)), BOUNDARY_COLOR, dtype=np.uint8)
    for feature, (row_delta, column_delta) in enumerate(spec.offsets):
        target_row_start = max(0, -row_delta)
        target_row_stop = min(height, height - row_delta)
        target_column_start = max(0, -column_delta)
        target_column_stop = min(width, width - column_delta)
        if target_row_start >= target_row_stop or target_column_start >= target_column_stop:
            continue
        patches[
            target_row_start:target_row_stop,
            target_column_start:target_column_stop,
            feature,
        ] = grid[
            target_row_start + row_delta : target_row_stop + row_delta,
            target_column_start + column_delta : target_column_stop + column_delta,
        ]
    return patches


def _rule_description_bits(spec: NeighborhoodSpec, entries: int) -> int:
    # family tag + neighborhood count/offsets + fallback tag + entry count;
    # every sparse exception stores K four-bit colors and one four-bit output.
    return int(
        4
        + elias_delta_bits(len(spec.offsets))
        + 6 * len(spec.offsets)
        + 1
        + elias_delta_bits(entries + 1)
        + entries * (BINARY4_BITS * len(spec.offsets) + BINARY4_BITS)
    )


@dataclass(frozen=True)
class SparseCategoricalRule:
    spec: NeighborhoodSpec
    entries: tuple[tuple[tuple[int, ...], int], ...]
    seen_patterns: frozenset[tuple[int, ...]]
    training_cells: int
    model_description_bits: int

    def __post_init__(self) -> None:
        if tuple(sorted(self.entries)) != self.entries:
            raise ValueError("rule entries must be sorted")
        if any(len(pattern) != len(self.spec.offsets) for pattern, _ in self.entries):
            raise ValueError("rule entry width disagrees with its neighborhood")

    @property
    def mapping(self) -> dict[tuple[int, ...], int]:
        return dict(self.entries)

    @property
    def exception_count(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class RuleFit:
    rule: SparseCategoricalRule | None
    collision_patterns: int
    shape_compatible: bool


def rule_from_transition_table(
    spec: NeighborhoodSpec,
    mapping: dict[tuple[int, ...], int],
) -> SparseCategoricalRule:
    """Build a sparse center-fallback rule from an explicit valid-domain table."""
    normalized: dict[tuple[int, ...], int] = {}
    for pattern, output in mapping.items():
        pattern = tuple(int(value) for value in pattern)
        output = int(output)
        if len(pattern) != len(spec.offsets):
            raise ValueError("transition pattern width disagrees with its neighborhood")
        if any(value < 0 or value > BOUNDARY_COLOR for value in pattern):
            raise ValueError("transition patterns may use colors 0..9 and boundary 10")
        if not 0 <= output < ARC_COLOR_COUNT:
            raise ValueError("transition outputs must be colors 0..9")
        normalized[pattern] = output
    exceptions = tuple(sorted((pattern, output) for pattern, output in normalized.items() if output != pattern[0]))
    return SparseCategoricalRule(
        spec=spec,
        entries=exceptions,
        seen_patterns=frozenset(normalized),
        training_cells=len(normalized),
        model_description_bits=_rule_description_bits(spec, len(exceptions)),
    )


def fit_sparse_rule(
    examples: Sequence[ArcExample],
    spec: NeighborhoodSpec,
) -> RuleFit:
    """Fit exact shared semantics with center-color fallback.

    A pattern is retained only when its output differs from its center.  If the
    same complete pattern requires two outputs, no deterministic rule exists.
    """
    if not examples:
        raise ValueError("at least one demonstration is required")
    targets: dict[tuple[int, ...], set[int]] = {}
    training_cells = 0
    for example in examples:
        source = _categorical_grid(example.input_grid, "input_grid")
        target = _categorical_grid(example.output_grid, "output_grid")
        if source.shape != target.shape:
            return RuleFit(None, 0, False)
        patches = extract_neighborhoods(source, spec).reshape(-1, len(spec.offsets))
        flat_target = target.reshape(-1)
        training_cells += len(flat_target)
        for patch, output in zip(patches, flat_target):
            key = tuple(int(value) for value in patch)
            targets.setdefault(key, set()).add(int(output))
    collisions = sum(len(outputs) > 1 for outputs in targets.values())
    if collisions:
        return RuleFit(None, collisions, True)
    exceptions = tuple(
        sorted(
            (pattern, next(iter(outputs)))
            for pattern, outputs in targets.items()
            if next(iter(outputs)) != pattern[0]
        )
    )
    rule = SparseCategoricalRule(
        spec=spec,
        entries=exceptions,
        seen_patterns=frozenset(targets),
        training_cells=training_cells,
        model_description_bits=_rule_description_bits(spec, len(exceptions)),
    )
    return RuleFit(rule, 0, True)


def predict_direct(rule: SparseCategoricalRule, grid: object) -> np.ndarray:
    grid = _categorical_grid(grid)
    patches = extract_neighborhoods(grid, rule.spec)
    result = grid.copy()
    mapping = rule.mapping
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            key = tuple(int(value) for value in patches[row, column])
            if key in mapping:
                result[row, column] = mapping[key]
    return result


def encode_binary4(colors: object) -> np.ndarray:
    array = np.asarray(colors)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("binary4 encoder requires integer colors")
    if not bool(np.all((0 <= array) & (array <= BOUNDARY_COLOR))):
        raise ValueError("binary4 encoder accepts colors 0 through boundary symbol 10")
    return ((array[..., None].astype(np.uint8) >> np.arange(BINARY4_BITS, dtype=np.uint8)) & 1).astype(np.uint8)


def decode_binary4(bits: object) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(bits)
    if array.shape[-1:] != (BINARY4_BITS,) or not bool(np.all((array == 0) | (array == 1))):
        raise ValueError("binary4 decoder requires a final four-bit axis")
    colors = np.sum(array.astype(np.uint8) << np.arange(BINARY4_BITS, dtype=np.uint8), axis=-1).astype(np.uint8)
    valid = colors < ARC_COLOR_COUNT
    return colors, valid


def encode_onehot(colors: object, *, include_boundary: bool = False) -> np.ndarray:
    array = np.asarray(colors)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("one-hot encoder requires integer colors")
    states = ONEHOT_INPUT_STATES if include_boundary else ONEHOT_OUTPUT_STATES
    if not bool(np.all((0 <= array) & (array < states))):
        raise ValueError(f"one-hot encoder accepts colors 0 through {states - 1}")
    return np.eye(states, dtype=np.uint8)[array.astype(np.int64)]


def decode_onehot(bits: object) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(bits)
    if array.shape[-1:] != (ONEHOT_OUTPUT_STATES,) or not bool(np.all((array == 0) | (array == 1))):
        raise ValueError("one-hot decoder requires a final ten-bit axis")
    valid = np.sum(array, axis=-1) == 1
    return np.argmax(array, axis=-1).astype(np.uint8), valid


def predict_binary4(rule: SparseCategoricalRule, grid: object) -> tuple[np.ndarray, np.ndarray]:
    """Execute the constructive binary equality/multiplexer network."""
    grid = _categorical_grid(grid)
    patches = extract_neighborhoods(grid, rule.spec)
    patch_bits = encode_binary4(patches)
    result_bits = encode_binary4(grid)
    for pattern, output in rule.entries:
        wanted = encode_binary4(np.asarray(pattern, dtype=np.uint8))
        match = np.all(patch_bits == wanted, axis=(-2, -1))
        output_bits = encode_binary4(np.asarray(output, dtype=np.uint8))
        result_bits = np.where(match[..., None], output_bits, result_bits)
    colors, valid = decode_binary4(result_bits)
    return colors, valid


def predict_onehot(rule: SparseCategoricalRule, grid: object) -> tuple[np.ndarray, np.ndarray]:
    """Execute the constructive one-hot equality/multiplexer network."""
    grid = _categorical_grid(grid)
    patches = extract_neighborhoods(grid, rule.spec)
    patch_bits = encode_onehot(patches, include_boundary=True)
    result_bits = encode_onehot(grid)
    for pattern, output in rule.entries:
        match = np.ones(grid.shape, dtype=bool)
        for feature, color in enumerate(pattern):
            match &= patch_bits[..., feature, int(color)].astype(bool)
        output_bits = encode_onehot(np.asarray(output, dtype=np.uint8))
        result_bits = np.where(match[..., None], output_bits, result_bits)
    colors, valid = decode_onehot(result_bits)
    return colors, valid


def support_fraction(rule: SparseCategoricalRule, grid: object) -> float:
    patches = extract_neighborhoods(grid, rule.spec).reshape(-1, len(rule.spec.offsets))
    supported = sum(tuple(int(value) for value in patch) in rule.seen_patterns for patch in patches)
    return float(supported / len(patches))


def compiled_gate_upper_bounds(rule: SparseCategoricalRule) -> dict[str, int]:
    """Return no-sharing equality/mux upper bounds for both public codecs."""
    k = len(rule.spec.offsets)
    binary_gates = 0
    onehot_gates = 0
    for pattern, _ in rule.entries:
        pattern_bits = encode_binary4(np.asarray(pattern, dtype=np.uint8))
        binary_literal_nots = int(np.sum(pattern_bits == 0))
        binary_match_ands = max(0, BINARY4_BITS * k - 1)
        binary_mux = 1 + 3 * BINARY4_BITS  # one shared NOT(match), four 2:1 muxes
        binary_gates += binary_literal_nots + binary_match_ands + binary_mux
        onehot_match_ands = max(0, k - 1)
        onehot_mux = 1 + 3 * ONEHOT_OUTPUT_STATES
        onehot_gates += onehot_match_ands + onehot_mux
    return {
        "binary4_gate_upper_bound": int(binary_gates),
        "onehot_gate_upper_bound": int(onehot_gates),
        "binary4_state_bits_per_cell": BINARY4_BITS,
        "onehot_state_bits_per_cell": ONEHOT_OUTPUT_STATES,
        "onehot_boundary_input_states": ONEHOT_INPUT_STATES,
    }


@dataclass(frozen=True)
class SelectedRule:
    rule: SparseCategoricalRule | None
    status: str
    candidates_considered: int
    exact_training_candidates: int
    collision_candidates: int
    lodo_pair_exact: float
    lodo_cell_accuracy: float
    search_seconds: float
    ranked_rules: tuple[SparseCategoricalRule, ...]


def _pair_metrics(references: Sequence[np.ndarray], predictions: Sequence[np.ndarray]) -> tuple[float, float]:
    if len(references) != len(predictions) or not references:
        raise ValueError("metrics require equal nonempty reference and prediction lists")
    pair_exact = []
    correct = 0
    cells = 0
    for reference, prediction in zip(references, predictions):
        if reference.shape != prediction.shape:
            pair_exact.append(False)
            continue
        pair_exact.append(bool(np.array_equal(reference, prediction)))
        correct += int(np.sum(reference == prediction))
        cells += int(reference.size)
    return float(np.mean(pair_exact)), float(correct / cells) if cells else 0.0


def _lodo_metrics(examples: Sequence[ArcExample], spec: NeighborhoodSpec) -> tuple[float, float]:
    if len(examples) < 2:
        return float("nan"), float("nan")
    references: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for held_out in range(len(examples)):
        training = tuple(example for index, example in enumerate(examples) if index != held_out)
        fit = fit_sparse_rule(training, spec)
        references.append(np.asarray(examples[held_out].output_grid))
        if fit.rule is None:
            predictions.append(np.asarray(examples[held_out].input_grid))
        else:
            predictions.append(predict_direct(fit.rule, examples[held_out].input_grid))
    return _pair_metrics(references, predictions)


def select_demo_rule(
    examples: Sequence[ArcExample],
    specs: Sequence[NeighborhoodSpec] = FROZEN_NEIGHBORHOODS,
) -> SelectedRule:
    """Select an exact-training rule using demonstration-only diagnostics."""
    started = time.perf_counter()
    if not examples:
        raise ValueError("at least one demonstration is required")
    if any(example.input_grid.shape != example.output_grid.shape for example in examples):
        return SelectedRule(None, "unsupported_shape_change", len(specs), 0, 0, math.nan, math.nan, time.perf_counter() - started, ())

    candidates: list[tuple[tuple[float, float, int, str], SparseCategoricalRule, float, float]] = []
    collisions = 0
    for spec in specs:
        fit = fit_sparse_rule(examples, spec)
        if fit.rule is None:
            collisions += int(fit.collision_patterns > 0)
            continue
        predictions = [predict_direct(fit.rule, example.input_grid) for example in examples]
        if not all(np.array_equal(prediction, example.output_grid) for prediction, example in zip(predictions, examples)):
            raise AssertionError("a fitted deterministic rule failed its own demonstrations")
        lodo_pair, lodo_cell = _lodo_metrics(examples, spec)
        # For a single demonstration there is no honest LODO signal; all exact
        # candidates tie at zero and MDL decides.
        pair_score = 0.0 if math.isnan(lodo_pair) else lodo_pair
        cell_score = 0.0 if math.isnan(lodo_cell) else lodo_cell
        score = (-pair_score, -cell_score, fit.rule.model_description_bits, spec.name)
        candidates.append((score, fit.rule, lodo_pair, lodo_cell))
    if not candidates:
        return SelectedRule(None, "no_deterministic_local_rule", len(specs), 0, collisions, math.nan, math.nan, time.perf_counter() - started, ())
    candidates.sort(key=lambda item: item[0])
    _, rule, lodo_pair, lodo_cell = candidates[0]
    return SelectedRule(
        rule,
        "selected",
        len(specs),
        len(candidates),
        collisions,
        lodo_pair,
        lodo_cell,
        time.perf_counter() - started,
        tuple(item[1] for item in candidates),
    )


def find_oracle_local_rule(
    examples: Sequence[ArcExample],
    specs: Sequence[NeighborhoodSpec] = FROZEN_NEIGHBORHOODS,
) -> SparseCategoricalRule | None:
    """Return the shortest train+test-compatible rule (diagnostic leakage)."""
    compatible: list[SparseCategoricalRule] = []
    for spec in specs:
        fit = fit_sparse_rule(examples, spec)
        if fit.rule is not None:
            compatible.append(fit.rule)
    if not compatible:
        return None
    return min(compatible, key=lambda rule: (rule.model_description_bits, rule.spec.name))


@dataclass(frozen=True)
class RolloutResult:
    terminal: np.ndarray
    steps_executed: int
    stop_reason: str
    trajectory: tuple[np.ndarray, ...]


def rollout_rule(
    rule: SparseCategoricalRule,
    initial_grid: object,
    max_steps: int,
    *,
    stop: str = "fixed_point",
) -> RolloutResult:
    """Run a categorical rule with explicit fixed-point/cycle accounting."""
    max_steps = _positive_integer(max_steps, "max_steps")
    if stop not in ("fixed_point", "cycle", "max_steps"):
        raise ValueError("stop must be fixed_point, cycle, or max_steps")
    state = _categorical_grid(initial_grid).copy()
    trajectory = [state.copy()]
    seen = {state.tobytes(): 0}
    for step in range(1, max_steps + 1):
        next_state = predict_direct(rule, state)
        trajectory.append(next_state.copy())
        if stop in ("fixed_point", "cycle") and np.array_equal(next_state, state):
            return RolloutResult(next_state, step, "fixed_point", tuple(trajectory))
        key = next_state.tobytes()
        if stop == "cycle" and key in seen:
            return RolloutResult(next_state, step, f"cycle_length_{step - seen[key]}", tuple(trajectory))
        seen[key] = step
        state = next_state
    return RolloutResult(state, max_steps, "max_steps", tuple(trajectory))
