"""A frozen, bounded library of interpretable categorical CA programs.

These programs are a separate evidence level from sparse neighborhood-table
induction.  A demonstration search may choose a copy wire or a monotone
color-propagation rule, but it may not inspect held-out outputs or synthesize
arbitrary Python code.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
import time
from typing import Sequence

import numpy as np

from arc_data import ARC_COLOR_COUNT, ArcExample


PROGRAM_HORIZONS: tuple[int | str, ...] = (1, 2, 3, 4, 8, "fixed_point")
COPY_OFFSETS: tuple[tuple[int, int], ...] = tuple(
    (row, column)
    for row in range(-2, 3)
    for column in range(-2, 3)
    if (row, column) != (0, 0)
)


def _grid(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or min(array.shape, default=0) == 0:
        raise ValueError("grid must be a nonempty matrix")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("grid must contain integer colors")
    if not bool(np.all((0 <= array) & (array < ARC_COLOR_COUNT))):
        raise ValueError("grid colors must be 0..9")
    return array.astype(np.uint8, copy=False)


def _neighbors(mask: np.ndarray, connectivity: int) -> np.ndarray:
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    height, width = mask.shape
    padded = np.pad(mask, 1, mode="constant")
    offsets = (
        ((-1, 0), (0, -1), (0, 1), (1, 0))
        if connectivity == 4
        else tuple(
            (row, column)
            for row in (-1, 0, 1)
            for column in (-1, 0, 1)
            if (row, column) != (0, 0)
        )
    )
    result = np.zeros_like(mask, dtype=bool)
    for row, column in offsets:
        result |= padded[1 + row : 1 + row + height, 1 + column : 1 + column + width]
    return result


def _copy_offset(grid: np.ndarray, row_delta: int, column_delta: int, boundary: int | str) -> np.ndarray:
    height, width = grid.shape
    if boundary == "center":
        result = grid.copy()
    else:
        result = np.full_like(grid, int(boundary))
    target_row_start = max(0, -row_delta)
    target_row_stop = min(height, height - row_delta)
    target_column_start = max(0, -column_delta)
    target_column_stop = min(width, width - column_delta)
    result[target_row_start:target_row_stop, target_column_start:target_column_stop] = grid[
        target_row_start + row_delta : target_row_stop + row_delta,
        target_column_start + column_delta : target_column_stop + column_delta,
    ]
    return result


@dataclass(frozen=True)
class LocalCAProgram:
    kind: str
    params: tuple[object, ...]
    description_bits: int
    onehot_gates_per_cell_upper: int

    @property
    def name(self) -> str:
        return f"{self.kind}:" + ":".join(str(value) for value in self.params)

    def run(self, grid: object) -> tuple[np.ndarray, int, str]:
        state = _grid(grid).copy()
        if self.kind == "identity":
            return state, 1, "one_step"
        if self.kind == "copy_offset":
            row, column, boundary = self.params
            return _copy_offset(state, int(row), int(column), boundary), 1, "one_step"
        if self.kind != "propagate":
            raise ValueError(f"unknown program kind {self.kind}")
        trigger, susceptible, output, connectivity, horizon = self.params
        trigger, susceptible, output, connectivity = map(int, (trigger, susceptible, output, connectivity))
        # A monotone program can change at least one cell for ``state.size``
        # updates and still needs one final no-change update to certify closure.
        max_steps = state.size + 1 if horizon == "fixed_point" else int(horizon)
        for step in range(1, max_steps + 1):
            active = (state == trigger) | (state == output)
            changed = (state == susceptible) & _neighbors(active, connectivity)
            next_state = state.copy()
            next_state[changed] = output
            if horizon == "fixed_point" and np.array_equal(next_state, state):
                return state, step, "fixed_point"
            state = next_state
        return state, max_steps, "max_steps" if horizon != "fixed_point" else "fixed_point_budget"


@dataclass(frozen=True)
class ProgramSelection:
    program: LocalCAProgram | None
    ranked_programs: tuple[LocalCAProgram, ...]
    status: str
    candidates_evaluated: int
    exact_candidates: int
    search_seconds: float


def _modal_input_color(examples: Sequence[ArcExample]) -> int:
    counts = np.zeros(ARC_COLOR_COUNT, dtype=np.int64)
    for example in examples:
        counts += np.bincount(np.asarray(example.input_grid).reshape(-1), minlength=ARC_COLOR_COUNT)
    return int(np.argmax(counts))


def _candidate_programs(examples: Sequence[ArcExample]) -> list[LocalCAProgram]:
    candidates = [LocalCAProgram("identity", (), 4, 0)]
    modal = _modal_input_color(examples)
    for row, column in COPY_OFFSETS:
        # Center fallback at an edge needs a boundary-valid mux; a fixed padded
        # color can remain a pure routed wire in this abstract accounting.
        candidates.append(LocalCAProgram("copy_offset", (row, column, "center"), 18, 31))
        candidates.append(LocalCAProgram("copy_offset", (row, column, modal), 22, 0))

    changed_pairs = {
        (int(source), int(target))
        for example in examples
        if example.input_grid.shape == example.output_grid.shape
        for source, target in zip(example.input_grid.reshape(-1), example.output_grid.reshape(-1))
        if source != target
    }
    if len(changed_pairs) == 1:
        susceptible, output = next(iter(changed_pairs))
        colors = sorted(
            {
                int(value)
                for example in examples
                for grid in (example.input_grid, example.output_grid)
                for value in np.unique(grid)
            }
        )
        for trigger in colors:
            if trigger == susceptible:
                continue
            for connectivity in (4, 8):
                neighbors = connectivity
                active_test_gates = 0 if trigger == output else neighbors
                onehot_upper = active_test_gates + max(0, neighbors - 1) + 1 + 31
                for horizon in PROGRAM_HORIZONS:
                    horizon_bits = 4 if isinstance(horizon, int) else 2
                    candidates.append(
                        LocalCAProgram(
                            "propagate",
                            (trigger, susceptible, output, connectivity, horizon),
                            4 + 12 + 1 + horizon_bits,
                            onehot_upper,
                        )
                    )
    return candidates


def select_demo_program(examples: Sequence[ArcExample]) -> ProgramSelection:
    started = time.perf_counter()
    if not examples:
        raise ValueError("at least one demonstration is required")
    if any(example.input_grid.shape != example.output_grid.shape for example in examples):
        return ProgramSelection(None, (), "unsupported_shape_change", 0, 0, time.perf_counter() - started)
    candidates = _candidate_programs(examples)
    exact: list[LocalCAProgram] = []
    for program in candidates:
        if all(np.array_equal(program.run(example.input_grid)[0], example.output_grid) for example in examples):
            exact.append(program)
    exact.sort(key=lambda program: (program.description_bits, program.onehot_gates_per_cell_upper, program.name))
    if not exact:
        return ProgramSelection(None, (), "no_exact_program", len(candidates), 0, time.perf_counter() - started)
    return ProgramSelection(exact[0], tuple(exact), "selected", len(candidates), len(exact), time.perf_counter() - started)
