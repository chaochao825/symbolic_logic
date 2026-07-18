"""Deployable demonstration residuals for deterministic symbolic search."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .grid import Grid, as_grid

Coordinate = tuple[int, int]


@dataclass(frozen=True, slots=True)
class DemoResidual:
    pair_index: int
    predicted_shape: tuple[int, int] | None
    target_shape: tuple[int, int]
    execution_valid: bool
    shape_match: bool
    exact: bool
    overlap_matches: int
    comparison_cells: int
    mismatch_count: int
    mismatch_cells: tuple[Coordinate, ...]
    color_confusions: tuple[tuple[int, int, int], ...]
    invalid_code: str | None

    @property
    def agreement(self) -> float:
        return self.overlap_matches / self.comparison_cells


def compare_grids(
    predicted: Grid | None,
    target: Grid,
    *,
    pair_index: int,
    invalid_code: str | None = None,
) -> DemoResidual:
    reference = as_grid(target)
    target_shape = (len(reference), len(reference[0]))
    if predicted is None:
        cells = target_shape[0] * target_shape[1]
        return DemoResidual(
            pair_index=pair_index,
            predicted_shape=None,
            target_shape=target_shape,
            execution_valid=False,
            shape_match=False,
            exact=False,
            overlap_matches=0,
            comparison_cells=cells,
            mismatch_count=cells,
            mismatch_cells=(),
            color_confusions=(),
            invalid_code=invalid_code or "invalid_execution",
        )

    candidate = as_grid(predicted)
    predicted_shape = (len(candidate), len(candidate[0]))
    overlap_height = min(predicted_shape[0], target_shape[0])
    overlap_width = min(predicted_shape[1], target_shape[1])
    matches = 0
    mismatch_cells: list[Coordinate] = []
    confusions: Counter[tuple[int, int]] = Counter()
    for row in range(overlap_height):
        for column in range(overlap_width):
            if candidate[row][column] == reference[row][column]:
                matches += 1
            else:
                if predicted_shape == target_shape:
                    mismatch_cells.append((row, column))
                confusions[(candidate[row][column], reference[row][column])] += 1
    comparison_cells = max(
        predicted_shape[0] * predicted_shape[1],
        target_shape[0] * target_shape[1],
    )
    shape_match = predicted_shape == target_shape
    exact = shape_match and matches == comparison_cells
    return DemoResidual(
        pair_index=pair_index,
        predicted_shape=predicted_shape,
        target_shape=target_shape,
        execution_valid=True,
        shape_match=shape_match,
        exact=exact,
        overlap_matches=matches,
        comparison_cells=comparison_cells,
        mismatch_count=comparison_cells - matches,
        mismatch_cells=tuple(mismatch_cells),
        color_confusions=tuple(
            (predicted_color, target_color, count)
            for (predicted_color, target_color), count in sorted(confusions.items())
        ),
        invalid_code=None,
    )
