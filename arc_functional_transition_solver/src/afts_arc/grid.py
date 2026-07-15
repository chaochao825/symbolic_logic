"""Immutable ARC grid representation, validation, hashing, and basic transforms."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import TypeAlias

Grid: TypeAlias = tuple[tuple[int, ...], ...]
ARC_MAX_HEIGHT = 30
ARC_MAX_WIDTH = 30


class GridValidationError(ValueError):
    """Raised when a value is not a valid ARC grid."""


def as_grid(
    value: object,
    *,
    max_height: int = ARC_MAX_HEIGHT,
    max_width: int = ARC_MAX_WIDTH,
) -> Grid:
    """Validate and convert a nested sequence into an immutable ARC grid.

    ARC grids are non-empty rectangular matrices whose cells are integer symbols in
    ``[0, 9]``. Booleans are rejected even though ``bool`` subclasses ``int``.
    """

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise GridValidationError("grid must be a sequence of rows")
    if not value:
        raise GridValidationError("grid must contain at least one row")
    if len(value) > max_height:
        raise GridValidationError(
            f"grid height {len(value)} exceeds maximum {max_height}"
        )

    rows: list[tuple[int, ...]] = []
    width: int | None = None
    for row_index, row in enumerate(value):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise GridValidationError(f"row {row_index} must be a sequence")
        if not row:
            raise GridValidationError(f"row {row_index} must not be empty")
        if width is None:
            width = len(row)
            if width > max_width:
                raise GridValidationError(
                    f"grid width {width} exceeds maximum {max_width}"
                )
        elif len(row) != width:
            raise GridValidationError(
                f"grid is ragged: row 0 has width {width}, row {row_index} has width {len(row)}"
            )

        normalized: list[int] = []
        for column_index, cell in enumerate(row):
            if type(cell) is not int or not 0 <= cell <= 9:
                raise GridValidationError(
                    f"cell ({row_index}, {column_index}) must be an integer in [0, 9]"
                )
            normalized.append(cell)
        rows.append(tuple(normalized))

    return tuple(rows)


def grid_to_lists(grid: Grid) -> list[list[int]]:
    """Convert an immutable grid to the JSON-compatible ARC representation."""

    return [list(row) for row in grid]


def canonical_grid_json(grid: Grid) -> str:
    """Return a compact, deterministic JSON representation."""

    return json.dumps(grid_to_lists(grid), separators=(",", ":"), ensure_ascii=True)


def grid_key(grid: Grid) -> str:
    """Return a content hash used for exact-output deduplication."""

    return hashlib.sha256(canonical_grid_json(grid).encode("ascii")).hexdigest()


def rotate90(grid: Grid) -> Grid:
    """Rotate a grid clockwise by 90 degrees."""

    return tuple(tuple(row) for row in zip(*grid[::-1]))


def rotate180(grid: Grid) -> Grid:
    return tuple(tuple(reversed(row)) for row in reversed(grid))


def rotate270(grid: Grid) -> Grid:
    return tuple(tuple(row) for row in zip(*grid))[::-1]


def flip_horizontal(grid: Grid) -> Grid:
    """Reflect a grid across its vertical axis."""

    return tuple(tuple(reversed(row)) for row in grid)


def flip_vertical(grid: Grid) -> Grid:
    """Reflect a grid across its horizontal axis."""

    return tuple(reversed(grid))


def transpose(grid: Grid) -> Grid:
    return tuple(tuple(row) for row in zip(*grid))


def dihedral_variants(grid: Grid) -> tuple[tuple[str, Grid], ...]:
    """Return unique D4 transforms in deterministic order.

    Symmetric inputs can collapse several transforms to the same output. The first
    transform producing an output is retained so downstream coverage counts unique
    grids rather than transform labels.
    """

    transforms = (
        ("identity", grid),
        ("rotate90", rotate90(grid)),
        ("rotate180", rotate180(grid)),
        ("rotate270", rotate270(grid)),
        ("flip_horizontal", flip_horizontal(grid)),
        ("flip_vertical", flip_vertical(grid)),
        ("transpose", transpose(grid)),
        ("anti_transpose", rotate180(transpose(grid))),
    )
    unique: list[tuple[str, Grid]] = []
    seen: set[str] = set()
    for name, candidate in transforms:
        key = grid_key(candidate)
        if key not in seen:
            seen.add(key)
            unique.append((name, candidate))
    return tuple(unique)
