"""Exact cellular-automata semantics and bounded accounting helpers.

This module is deliberately dependency-light: it contains NumPy reference
semantics, not a reproduction of the JAX DiffLogic-CA training stack and not a
parser for the DigitalJS circuit format used by the accompanying web article.

The one-dimensional rule convention follows elementary cellular automata.  A
neighborhood is interpreted as a little-endian integer whose least-significant
bit is the rightmost cell.  Consequently, for radius one, ``x0=right``,
``x1=center``, and ``x2=left``.  Rule 110 therefore has the ascending-code truth
table ``[0, 1, 1, 1, 0, 1, 1, 0]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np


RULE110 = 110
MAX_LUT_RADIUS = 9

# This is the operation order used by differentiable logic-gate networks.  An
# operation index is also the four-bit truth table written in the conventional
# input order 00, 01, 10, 11 (most-significant output bit first).
BINARY_GATE16_NAMES = (
    "FALSE",
    "AND",
    "A_AND_NOT_B",
    "A",
    "NOT_A_AND_B",
    "B",
    "XOR",
    "OR",
    "NOR",
    "XNOR",
    "NOT_B",
    "A_OR_NOT_B",
    "NOT_A",
    "NOT_A_OR_B",
    "NAND",
    "TRUE",
)

BINARY_GATE16_DISPLAY_NAMES = (
    "FALSE",
    "AND",
    "A AND (NOT B)",
    "A",
    "(NOT A) AND B",
    "B",
    "XOR",
    "OR",
    "NOR",
    "XNOR",
    "NOT B",
    "A OR (NOT B)",
    "NOT A",
    "(NOT A) OR B",
    "NAND",
    "TRUE",
)


def _binary_array(values: object, name: str) -> np.ndarray:
    """Validate before casting so values such as 256 and NaN cannot wrap."""
    array = np.asarray(values)
    if not bool(np.all((array == 0) | (array == 1))):
        raise ValueError(f"{name} must contain only binary values")
    return array.astype(np.uint8, copy=False)


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _validate_radius(radius: object) -> int:
    result = _nonnegative_integer(radius, "radius")
    if result > MAX_LUT_RADIUS:
        raise ValueError(
            f"radius is limited to {MAX_LUT_RADIUS}; larger complete LUTs are intentionally not materialized"
        )
    return result


def _validate_boundary(boundary: str, boundary_value: object) -> tuple[str, int]:
    if boundary not in ("periodic", "fixed"):
        raise ValueError("boundary must be 'periodic' or 'fixed'")
    value = int(_binary_array([boundary_value], "boundary_value")[0])
    return boundary, value


def radius_lut_values(rule: object, radius: int = 1) -> np.ndarray:
    """Decode a radius-``r`` rule integer into ascending-neighborhood outputs.

    There are ``2 ** (2*r + 1)`` neighborhood codes and therefore that many
    output bits in the rule integer.  Entry ``k`` is output bit ``k``.  The
    complete table is bounded to ``MAX_LUT_RADIUS`` to avoid accidental giant
    allocations.
    """
    radius = _validate_radius(radius)
    if isinstance(rule, (bool, np.bool_)) or not isinstance(rule, Integral):
        raise TypeError("rule must be an integer")
    rule = int(rule)
    if rule < 0:
        raise ValueError("rule must be nonnegative")
    table_size = 1 << (2 * radius + 1)
    if rule.bit_length() > table_size:
        raise ValueError(f"rule does not fit a radius-{radius} LUT")
    return np.fromiter(((rule >> code) & 1 for code in range(table_size)), dtype=np.uint8, count=table_size)


def _coerce_lut(rule_or_lut: object, radius: int) -> np.ndarray:
    radius = _validate_radius(radius)
    if isinstance(rule_or_lut, Integral) and not isinstance(rule_or_lut, (bool, np.bool_)):
        return radius_lut_values(rule_or_lut, radius)
    lut = _binary_array(rule_or_lut, "lut").reshape(-1)
    expected = 1 << (2 * radius + 1)
    if len(lut) != expected:
        raise ValueError(f"a radius-{radius} LUT must contain {expected} outputs")
    return lut


def encode_radius_neighborhoods(neighborhoods: object, radius: int | None = None) -> np.ndarray:
    """Encode explicit neighborhoods using rightmost-to-leftmost bit columns.

    The last axis must have odd width ``2*r+1``.  Column zero is the rightmost
    cell and is the least-significant bit; the center is column ``r``.  For ECA
    this is precisely ``[x0=right, x1=center, x2=left]``.
    """
    values = _binary_array(neighborhoods, "neighborhoods")
    if values.ndim == 0:
        raise ValueError("neighborhoods must have a final feature axis")
    width = int(values.shape[-1])
    inferred = (width - 1) // 2
    if width < 1 or width != 2 * inferred + 1:
        raise ValueError("neighborhood width must be a positive odd number")
    if radius is None:
        radius = inferred
    radius = _validate_radius(radius)
    if width != 2 * radius + 1:
        raise ValueError(f"expected {2 * radius + 1} neighborhood bits for radius {radius}")
    weights = np.left_shift(np.uint64(1), np.arange(width, dtype=np.uint64))
    return np.sum(values.astype(np.uint64) * weights, axis=-1, dtype=np.uint64)


def radius_lut_predict(neighborhoods: object, rule_or_lut: object, radius: int | None = None) -> np.ndarray:
    """Apply a rule integer or explicit LUT to a batch of local neighborhoods."""
    array = _binary_array(neighborhoods, "neighborhoods")
    if array.ndim == 0:
        raise ValueError("neighborhoods must have a final feature axis")
    if radius is None:
        width = int(array.shape[-1])
        if width < 1 or width % 2 != 1:
            raise ValueError("cannot infer radius from an even or empty neighborhood width")
        radius = (width - 1) // 2
    radius = _validate_radius(radius)
    codes = encode_radius_neighborhoods(array, radius)
    lut = _coerce_lut(rule_or_lut, radius)
    return lut[codes].astype(np.uint8, copy=False)


def radius_neighborhood_codes(
    state: object,
    radius: int = 1,
    *,
    boundary: str = "periodic",
    boundary_value: int = 0,
) -> np.ndarray:
    """Return local codes with the lattice on the final array axis.

    Any leading axes are independent batch axes and are preserved.
    """
    state = _binary_array(state, "state")
    if state.ndim < 1 or state.shape[-1] == 0 or state.size == 0:
        raise ValueError("state must have a nonempty final lattice axis")
    radius = _validate_radius(radius)
    boundary, boundary_value = _validate_boundary(boundary, boundary_value)
    lattice_size = state.shape[-1]
    codes = np.zeros(state.shape, dtype=np.uint64)
    indices = np.arange(lattice_size)
    for bit in range(2 * radius + 1):
        offset = radius - bit
        if boundary == "periodic":
            neighbor = np.roll(state, -offset, axis=-1)
        else:
            source = indices + offset
            valid = (source >= 0) & (source < lattice_size)
            neighbor = np.full(state.shape, boundary_value, dtype=np.uint8)
            neighbor[..., valid] = state[..., source[valid]]
        codes |= neighbor.astype(np.uint64) << np.uint64(bit)
    return codes


def radius_lut_step(
    state: object,
    rule_or_lut: object,
    radius: int = 1,
    *,
    boundary: str = "periodic",
    boundary_value: int = 0,
) -> np.ndarray:
    """Perform one synchronous LUT update along the final lattice axis."""
    radius = _validate_radius(radius)
    lut = _coerce_lut(rule_or_lut, radius)
    codes = radius_neighborhood_codes(
        state,
        radius,
        boundary=boundary,
        boundary_value=boundary_value,
    )
    return lut[codes].astype(np.uint8, copy=False)


def radius_lut_rollout(
    initial_state: object,
    rule_or_lut: object,
    steps: int,
    radius: int = 1,
    *,
    boundary: str = "periodic",
    boundary_value: int = 0,
) -> np.ndarray:
    """Roll out batched synchronous states, returning ``t=0`` through ``t=steps``.

    The final input axis is the one-dimensional lattice.  All leading axes are
    independent batch axes; the returned time axis is prepended.
    """
    steps = _nonnegative_integer(steps, "steps")
    state = _binary_array(initial_state, "initial_state")
    if state.ndim < 1 or state.shape[-1] == 0 or state.size == 0:
        raise ValueError("initial_state must have a nonempty final lattice axis")
    radius = _validate_radius(radius)
    lut = _coerce_lut(rule_or_lut, radius)
    trajectory = np.empty((steps + 1, *state.shape), dtype=np.uint8)
    trajectory[0] = state
    for step in range(steps):
        state = radius_lut_step(
            state,
            lut,
            radius,
            boundary=boundary,
            boundary_value=boundary_value,
        )
        trajectory[step + 1] = state
    return trajectory


def eca_lut(rule: object) -> np.ndarray:
    """Return the eight outputs of an elementary cellular-automaton rule."""
    return radius_lut_values(rule, radius=1)


def eca_predict(neighborhoods: object, rule: object) -> np.ndarray:
    """Apply an ECA rule to explicit ``[right, center, left]`` neighborhoods."""
    return radius_lut_predict(neighborhoods, rule, radius=1)


def eca_step(
    state: object,
    rule: object,
    *,
    boundary: str = "periodic",
    boundary_value: int = 0,
) -> np.ndarray:
    return radius_lut_step(state, rule, radius=1, boundary=boundary, boundary_value=boundary_value)


def eca_rollout(
    initial_state: object,
    rule: object,
    steps: int,
    *,
    boundary: str = "periodic",
    boundary_value: int = 0,
) -> np.ndarray:
    return radius_lut_rollout(
        initial_state,
        rule,
        steps,
        radius=1,
        boundary=boundary,
        boundary_value=boundary_value,
    )


def game_of_life_next_from_patches(patches: object) -> np.ndarray:
    """Evaluate Conway's Game of Life on explicit 3x3 or flattened patches."""
    patches = _binary_array(patches, "patches")
    if patches.ndim >= 2 and patches.shape[-2:] == (3, 3):
        center = patches[..., 1, 1]
        neighbors = patches.sum(axis=(-2, -1), dtype=np.int16) - center
    elif patches.ndim >= 1 and patches.shape[-1] == 9:
        center = patches[..., 4]
        neighbors = patches.sum(axis=-1, dtype=np.int16) - center
    else:
        raise ValueError("patches must end in shape (3, 3) or a flattened length-9 axis")
    return ((neighbors == 3) | ((center == 1) & (neighbors == 2))).astype(np.uint8)


def moore_patches(
    grid: object,
    *,
    boundary: str = "fixed",
    boundary_value: int = 0,
) -> np.ndarray:
    """Extract row-major 3x3 Moore patches as an ``(H, W, 9)`` array.

    Patch index four is the center.  This public helper lets local classifiers
    consume exactly the same neighborhoods as the Game-of-Life oracle.
    """
    grid = _binary_array(grid, "grid")
    if grid.ndim != 2 or min(grid.shape) == 0:
        raise ValueError("grid must be a nonempty two-dimensional binary array")
    boundary, boundary_value = _validate_boundary(boundary, boundary_value)
    if boundary == "periodic":
        padded = np.pad(grid, 1, mode="wrap")
    else:
        padded = np.pad(grid, 1, mode="constant", constant_values=boundary_value)
    height, width = grid.shape
    return np.stack(
        [
            padded[row : row + height, column : column + width]
            for row in range(3)
            for column in range(3)
        ],
        axis=-1,
    ).astype(np.uint8, copy=False)


def game_of_life_step(
    grid: object,
    *,
    boundary: str = "fixed",
    boundary_value: int = 0,
) -> np.ndarray:
    """Perform one synchronous Conway's Game of Life update."""
    grid = _binary_array(grid, "grid")
    if grid.ndim != 2 or min(grid.shape) == 0:
        raise ValueError("grid must be a nonempty two-dimensional binary array")
    return game_of_life_next_from_patches(
        moore_patches(grid, boundary=boundary, boundary_value=boundary_value)
    )


def game_of_life_rollout(
    initial_grid: object,
    steps: int,
    *,
    boundary: str = "fixed",
    boundary_value: int = 0,
) -> np.ndarray:
    """Roll out Game of Life, returning ``t=0`` through ``t=steps``."""
    steps = _nonnegative_integer(steps, "steps")
    grid = _binary_array(initial_grid, "initial_grid")
    if grid.ndim != 2 or min(grid.shape) == 0:
        raise ValueError("initial_grid must be a nonempty two-dimensional binary array")
    trajectory = np.empty((steps + 1, *grid.shape), dtype=np.uint8)
    trajectory[0] = grid
    for step in range(steps):
        grid = game_of_life_step(grid, boundary=boundary, boundary_value=boundary_value)
        trajectory[step + 1] = grid
    return trajectory


def _resolve_gate16(gate: object) -> int:
    if isinstance(gate, Integral) and not isinstance(gate, (bool, np.bool_)):
        index = int(gate)
    elif isinstance(gate, str):
        if gate in BINARY_GATE16_NAMES:
            index = BINARY_GATE16_NAMES.index(gate)
        elif gate in BINARY_GATE16_DISPLAY_NAMES:
            index = BINARY_GATE16_DISPLAY_NAMES.index(gate)
        else:
            raise ValueError(f"unknown binary gate: {gate}")
    else:
        raise TypeError("gate must be an integer index or a public gate name")
    if not 0 <= index < 16:
        raise ValueError("gate index must be between 0 and 15")
    return index


def binary_gate16(gate: object, a: object, b: object) -> np.ndarray:
    """Evaluate one of all 16 crisp binary functions, with NumPy broadcasting."""
    index = _resolve_gate16(gate)
    a, b = np.broadcast_arrays(_binary_array(a, "a"), _binary_array(b, "b"))
    # Gate indices encode outputs for 00,01,10,11 from MSB to LSB.
    truth_position = 3 - (2 * a.astype(np.int8) + b.astype(np.int8))
    return ((index >> truth_position) & 1).astype(np.uint8)


def all_binary_gates16(a: object, b: object) -> np.ndarray:
    """Evaluate all 16 crisp gates; the final output axis follows public order."""
    a, b = np.broadcast_arrays(_binary_array(a, "a"), _binary_array(b, "b"))
    truth_position = 3 - (2 * a.astype(np.int8) + b.astype(np.int8))
    indices = np.arange(16, dtype=np.uint8)
    return ((indices >> truth_position[..., None]) & 1).astype(np.uint8)


def _wavefront_offsets(connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 4:
        return ((-1, 0), (0, -1), (0, 1), (1, 0))
    if connectivity == 8:
        return tuple(
            (row, column)
            for row in (-1, 0, 1)
            for column in (-1, 0, 1)
            if (row, column) != (0, 0)
        )
    raise ValueError("connectivity must be 4 or 8")


def boolean_wavefront_step(
    frontier: object,
    reached: object,
    passable: object,
    *,
    connectivity: int = 4,
) -> np.ndarray:
    """One synchronous Boolean OR/AND/NOT wavefront expansion without wrap."""
    frontier = _binary_array(frontier, "frontier").astype(bool)
    reached = _binary_array(reached, "reached").astype(bool)
    passable = _binary_array(passable, "passable").astype(bool)
    if frontier.ndim != 2 or frontier.shape != reached.shape or reached.shape != passable.shape:
        raise ValueError("frontier, reached, and passable must be equally shaped 2D grids")
    if np.any(frontier & ~reached):
        raise ValueError("frontier must be a subset of reached")
    height, width = frontier.shape
    padded = np.pad(frontier, 1, mode="constant")
    expanded = np.zeros_like(frontier)
    for row, column in _wavefront_offsets(connectivity):
        expanded |= padded[1 + row : 1 + row + height, 1 + column : 1 + column + width]
    return (expanded & passable & ~reached).astype(np.uint8)


def _coordinate(value: Sequence[int], shape: tuple[int, int], name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain a row and column")
    row, column = int(value[0]), int(value[1])
    if not (0 <= row < shape[0] and 0 <= column < shape[1]):
        raise ValueError(f"{name} is outside the grid")
    return row, column


@dataclass(frozen=True)
class BooleanWavefrontResult:
    reached: np.ndarray
    distance: np.ndarray
    iterations: int
    target_reached: bool
    target_distance: int
    path: tuple[tuple[int, int], ...]


def boolean_wavefront_pathfind(
    passable: object,
    source: Sequence[int],
    target: Sequence[int],
    *,
    connectivity: int = 4,
    max_steps: int | None = None,
) -> BooleanWavefrontResult:
    """Find a shortest unweighted grid path by synchronous Boolean wavefronts."""
    passable = _binary_array(passable, "passable")
    if passable.ndim != 2 or min(passable.shape) == 0:
        raise ValueError("passable must be a nonempty two-dimensional binary grid")
    source = _coordinate(source, passable.shape, "source")
    target = _coordinate(target, passable.shape, "target")
    _wavefront_offsets(connectivity)  # validate even for source == target
    if not bool(passable[source]):
        raise ValueError("source must be passable")
    if max_steps is None:
        max_steps = int(passable.size - 1)
    max_steps = _nonnegative_integer(max_steps, "max_steps")

    reached = np.zeros(passable.shape, dtype=np.uint8)
    frontier = np.zeros(passable.shape, dtype=np.uint8)
    distance = np.full(passable.shape, -1, dtype=np.int32)
    reached[source] = frontier[source] = 1
    distance[source] = 0
    iterations = 0
    while not bool(reached[target]) and bool(frontier.any()) and iterations < max_steps:
        next_frontier = boolean_wavefront_step(
            frontier,
            reached,
            passable,
            connectivity=connectivity,
        )
        iterations += 1
        if not bool(next_frontier.any()):
            frontier = next_frontier
            break
        distance[next_frontier.astype(bool)] = iterations
        reached |= next_frontier
        frontier = next_frontier

    target_reached = bool(reached[target])
    target_distance = int(distance[target]) if target_reached else -1
    path: list[tuple[int, int]] = []
    if target_reached:
        current = target
        path.append(current)
        while current != source:
            wanted = int(distance[current]) - 1
            candidates = []
            for row_delta, column_delta in _wavefront_offsets(connectivity):
                candidate = (current[0] + row_delta, current[1] + column_delta)
                if (
                    0 <= candidate[0] < passable.shape[0]
                    and 0 <= candidate[1] < passable.shape[1]
                    and int(distance[candidate]) == wanted
                ):
                    candidates.append(candidate)
            if not candidates:
                raise AssertionError("distance field does not contain a predecessor")
            current = min(candidates)
            path.append(current)
        path.reverse()
    return BooleanWavefrontResult(
        reached=reached,
        distance=distance,
        iterations=iterations,
        target_reached=target_reached,
        target_distance=target_distance,
        path=tuple(path),
    )


def trajectory_metrics(reference: object, prediction: object) -> dict[str, float | int]:
    """Compare equal-shaped hard trajectories whose first axis is time."""
    reference = _binary_array(reference, "reference")
    prediction = _binary_array(prediction, "prediction")
    if reference.shape != prediction.shape:
        raise ValueError("reference and prediction trajectories must have equal shape")
    if reference.ndim < 2 or reference.shape[0] == 0 or int(np.prod(reference.shape[1:])) == 0:
        raise ValueError("trajectories need a nonempty time axis and at least one cell")
    difference = reference != prediction
    per_step_errors = difference.reshape(reference.shape[0], -1).sum(axis=1)
    cells_per_state = int(np.prod(reference.shape[1:]))
    divergent = np.flatnonzero(per_step_errors)
    return {
        "timepoints": int(reference.shape[0]),
        "steps": int(reference.shape[0] - 1),
        "cells_per_state": cells_per_state,
        "cell_accuracy": float(1.0 - difference.mean()),
        "exact_trajectory": float(not bool(difference.any())),
        "final_cell_accuracy": float(1.0 - per_step_errors[-1] / cells_per_state),
        "exact_final_state": float(per_step_errors[-1] == 0),
        "first_divergence_step": int(divergent[0]) if len(divergent) else -1,
        "exact_step_fraction": float(np.mean(per_step_errors == 0)),
        "mean_hamming_fraction": float(np.mean(per_step_errors / cells_per_state)),
        "max_hamming_fraction": float(np.max(per_step_errors / cells_per_state)),
    }


def ca_complexity_accounting(
    spatial_shape: Sequence[int],
    steps: int,
    gates_per_cell: int,
    *,
    state_bits_per_cell: int = 1,
    update_fraction: float = 1.0,
    model_description_bits: int = 0,
    total_network_gates: int | None = None,
    reachable_network_gates: int | None = None,
    active_network_gates: int | None = None,
) -> dict[str, float | int]:
    """Report static storage and dynamic evaluation counts for a CA rollout.

    ``gates_per_cell`` is the number actually evaluated by one hard cell update.
    ``active_network_gates`` may exclude pass-through gates and is therefore
    reported separately rather than silently used as an operation count.  The
    returned quantities are accounting identities, not hardware PPA estimates.
    """
    dimensions = tuple(_nonnegative_integer(value, "spatial dimension") for value in spatial_shape)
    if not dimensions or any(value == 0 for value in dimensions):
        raise ValueError("spatial_shape must contain positive dimensions")
    steps = _nonnegative_integer(steps, "steps")
    gates_per_cell = _nonnegative_integer(gates_per_cell, "gates_per_cell")
    state_bits_per_cell = _nonnegative_integer(state_bits_per_cell, "state_bits_per_cell")
    model_description_bits = _nonnegative_integer(model_description_bits, "model_description_bits")
    update_fraction = float(update_fraction)
    if not np.isfinite(update_fraction) or not 0.0 <= update_fraction <= 1.0:
        raise ValueError("update_fraction must lie in [0, 1]")

    optional = {
        "total_network_gates": total_network_gates,
        "reachable_network_gates": reachable_network_gates,
        "active_network_gates": active_network_gates,
    }
    normalized: dict[str, int] = {}
    for name, value in optional.items():
        normalized[name] = gates_per_cell if value is None else _nonnegative_integer(value, name)
    if normalized["reachable_network_gates"] > normalized["total_network_gates"]:
        raise ValueError("reachable_network_gates cannot exceed total_network_gates")
    if normalized["active_network_gates"] > normalized["reachable_network_gates"]:
        raise ValueError("active_network_gates cannot exceed reachable_network_gates")

    cells = int(np.prod(np.asarray(dimensions, dtype=object)))
    expected_updates = float(cells * steps * update_fraction)
    state_bits = cells * state_bits_per_cell
    amortized = float(model_description_bits / expected_updates) if expected_updates else 0.0
    return {
        "spatial_cells": cells,
        "steps": steps,
        "state_bits_per_cell": state_bits_per_cell,
        "single_state_storage_bits": state_bits,
        "synchronous_double_buffer_bits": 2 * state_bits,
        "update_fraction": update_fraction,
        "expected_cell_updates": expected_updates,
        "evaluated_gates_per_cell": gates_per_cell,
        "expected_gate_evaluations": float(expected_updates * gates_per_cell),
        "total_network_gates": normalized["total_network_gates"],
        "reachable_network_gates": normalized["reachable_network_gates"],
        "active_network_gates": normalized["active_network_gates"],
        "model_description_bits": model_description_bits,
        "amortized_model_bits_per_cell_update": amortized,
    }
