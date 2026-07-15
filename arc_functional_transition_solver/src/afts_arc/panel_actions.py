"""Typed actions over frozen M02b panel hypotheses."""

from __future__ import annotations

from enum import Enum
from typing import Callable

from .grid import (
    Grid,
    as_grid,
    flip_horizontal,
    flip_vertical,
    grid_key,
    rotate180,
    rotate270,
    rotate90,
    transpose,
)
from .panel import PanelHypothesis, parse_panels

PANEL_SEQUENCE_D4_SEMANTICS_VERSION = "afts-panel-sequence-d4/v0.1"
PANEL_SEQUENCE_D4_STAGE = "M05d_indexed_panel_sequence_d4"
PANEL_LATTICE_PERIODIC_SEMANTICS_VERSION = (
    "afts-panel-lattice-periodic/v0.1"
)
PANEL_LATTICE_PERIODIC_STAGE = "M05e_two_axis_periodic_panel_broadcast"


class D4Step(str, Enum):
    IDENTITY = "identity"
    ROTATE90 = "rotate90"
    ROTATE180 = "rotate180"
    ROTATE270 = "rotate270"
    FLIP_HORIZONTAL = "flip_horizontal"
    FLIP_VERTICAL = "flip_vertical"
    TRANSPOSE = "transpose"
    ANTI_TRANSPOSE = "anti_transpose"


class PanelSequenceD4Code(str, Enum):
    EMPTY_SELECTION = "empty_selection"
    NON_UNIQUE_SELECTION = "non_unique_selection"
    INCOMPATIBLE_PANEL_SHAPES = "incompatible_panel_shapes"


class PanelLatticePeriodicCode(str, Enum):
    EMPTY_SELECTION = "empty_selection"
    NON_UNIQUE_SELECTION = "non_unique_selection"
    INCOMPATIBLE_PANEL_SHAPES = "incompatible_panel_shapes"


class _PeriodicPanelLattice:
    __slots__ = (
        "hypothesis",
        "seed_index",
        "row_count",
        "column_count",
    )

    def __init__(
        self,
        *,
        hypothesis: PanelHypothesis,
        seed_index: int,
        row_count: int,
        column_count: int,
    ) -> None:
        self.hypothesis = hypothesis
        self.seed_index = seed_index
        self.row_count = row_count
        self.column_count = column_count


def _identity(grid: Grid) -> Grid:
    return grid


def _anti_transpose(grid: Grid) -> Grid:
    return rotate180(transpose(grid))


_D4_TRANSFORMS: dict[D4Step, Callable[[Grid], Grid]] = {
    D4Step.IDENTITY: _identity,
    D4Step.ROTATE90: rotate90,
    D4Step.ROTATE180: rotate180,
    D4Step.ROTATE270: rotate270,
    D4Step.FLIP_HORIZONTAL: flip_horizontal,
    D4Step.FLIP_VERTICAL: flip_vertical,
    D4Step.TRANSPOSE: transpose,
    D4Step.ANTI_TRANSPOSE: _anti_transpose,
}
_D4_ORDERS: dict[D4Step, int] = {
    D4Step.IDENTITY: 1,
    D4Step.ROTATE90: 4,
    D4Step.ROTATE180: 2,
    D4Step.ROTATE270: 4,
    D4Step.FLIP_HORIZONTAL: 2,
    D4Step.FLIP_VERTICAL: 2,
    D4Step.TRANSPOSE: 2,
    D4Step.ANTI_TRANSPOSE: 2,
}


def apply_d4_power(grid: Grid, *, step: D4Step | str, exponent: int) -> Grid:
    """Apply one D4 element to an integer power, including negative powers."""

    if type(exponent) is not int:
        raise TypeError("D4 exponent must be an integer")
    try:
        selected = D4Step(step)
    except (TypeError, ValueError) as exc:
        raise ValueError("step must be a valid D4 element") from exc
    output = as_grid(grid)
    transform = _D4_TRANSFORMS[selected]
    for _ in range(exponent % _D4_ORDERS[selected]):
        output = transform(output)
    return output


def reassemble_panels(
    grid: Grid,
    *,
    hypothesis: PanelHypothesis,
    replacements: tuple[Grid, ...],
) -> Grid:
    """Replace every panel interior while preserving the bound separators."""

    normalized = as_grid(grid)
    if not isinstance(hypothesis, PanelHypothesis):
        raise TypeError("hypothesis must be a PanelHypothesis")
    if hypothesis.grid_key != grid_key(normalized):
        raise ValueError("panel hypothesis is not bound to the supplied grid")
    normalized_replacements = tuple(as_grid(item) for item in replacements)
    if len(normalized_replacements) != len(hypothesis.panels):
        raise ValueError("replacement count must equal the panel count")
    values = [list(row) for row in normalized]
    for panel, replacement in zip(hypothesis.panels, normalized_replacements):
        if (
            len(replacement) != panel.bbox.height
            or len(replacement[0]) != panel.bbox.width
        ):
            raise ValueError("replacement shape does not match its panel bbox")
        for local_row, row in enumerate(replacement):
            for local_column, cell in enumerate(row):
                values[panel.bbox.top + local_row][
                    panel.bbox.left + local_column
                ] = cell
    return as_grid(values)


def _select_periodic_panel_lattice(
    grid: Grid, *, background: int
) -> tuple[_PeriodicPanelLattice | None, PanelLatticePeriodicCode | None]:
    lattices = tuple(
        hypothesis
        for hypothesis in parse_panels(grid).hypotheses
        if hypothesis.row_separator_bands
        and hypothesis.column_separator_bands
        and hypothesis.separator_color != background
    )
    if not lattices:
        return None, PanelLatticePeriodicCode.EMPTY_SELECTION

    valid: list[_PeriodicPanelLattice] = []
    saw_multiple_seeds = False
    saw_incompatible_shape = False
    for hypothesis in lattices:
        row_count = 1 + max(panel.row_index for panel in hypothesis.panels)
        column_count = 1 + max(
            panel.column_index for panel in hypothesis.panels
        )
        if row_count < 2 or column_count < 2:
            continue
        occupied = tuple(
            index
            for index, panel in enumerate(hypothesis.panels)
            if any(
                cell != background
                for row in panel.content
                for cell in row
            )
        )
        if len(occupied) > 1:
            saw_multiple_seeds = True
        if not occupied or len(occupied) != 1:
            continue

        seed_index = occupied[0]
        seed = hypothesis.panels[seed_index]
        nominal_height = seed.bbox.height
        nominal_width = seed.bbox.width
        row_heights = tuple(
            next(
                panel.bbox.height
                for panel in hypothesis.panels
                if panel.row_index == row_index
            )
            for row_index in range(row_count)
        )
        column_widths = tuple(
            next(
                panel.bbox.width
                for panel in hypothesis.panels
                if panel.column_index == column_index
            )
            for column_index in range(column_count)
        )
        suffix_compatible = (
            all(height == nominal_height for height in row_heights[:-1])
            and 1 <= row_heights[-1] <= nominal_height
            and all(width == nominal_width for width in column_widths[:-1])
            and 1 <= column_widths[-1] <= nominal_width
        )
        if not suffix_compatible:
            saw_incompatible_shape = True
            continue
        valid.append(
            _PeriodicPanelLattice(
                hypothesis=hypothesis,
                seed_index=seed_index,
                row_count=row_count,
                column_count=column_count,
            )
        )

    if len(valid) > 1:
        return None, PanelLatticePeriodicCode.NON_UNIQUE_SELECTION
    if not valid:
        if saw_multiple_seeds:
            return None, PanelLatticePeriodicCode.NON_UNIQUE_SELECTION
        if saw_incompatible_shape:
            return None, PanelLatticePeriodicCode.INCOMPATIBLE_PANEL_SHAPES
        return None, PanelLatticePeriodicCode.EMPTY_SELECTION
    return valid[0], None


def panel_lattice_period_bounds(
    grid: Grid, *, background: int
) -> tuple[int, int] | None:
    """Return maximum non-trivial periods for one eligible input lattice."""

    if type(background) is not int or not 0 <= background <= 9:
        raise ValueError("background must be an ARC color")
    selected, invalid = _select_periodic_panel_lattice(
        as_grid(grid), background=background
    )
    if invalid is not None or selected is None:
        return None
    return selected.row_count - 1, selected.column_count - 1


def broadcast_panel_lattice_periodic(
    grid: Grid,
    *,
    background: int,
    row_period: int,
    column_period: int,
) -> tuple[Grid | None, PanelLatticePeriodicCode | None]:
    """Broadcast one full seed over a two-axis congruence class.

    A shorter final panel row or column is the visible prefix of the nominal
    panel domain. Leading and interior ragged intervals are deliberately out of
    scope for this action.
    """

    if type(background) is not int or not 0 <= background <= 9:
        raise ValueError("background must be an ARC color")
    for name, value in (
        ("row_period", row_period),
        ("column_period", column_period),
    ):
        if type(value) is not int or not 1 <= value <= 30:
            raise ValueError(f"{name} must be an integer in [1, 30]")
    normalized = as_grid(grid)
    selected, invalid = _select_periodic_panel_lattice(
        normalized, background=background
    )
    if invalid is not None or selected is None:
        return None, invalid

    hypothesis = selected.hypothesis
    seed_panel = hypothesis.panels[selected.seed_index]
    replacements: list[Grid] = []
    for panel in hypothesis.panels:
        row_matches = (
            panel.row_index - seed_panel.row_index
        ) % row_period == 0
        column_matches = (
            panel.column_index - seed_panel.column_index
        ) % column_period == 0
        selected_destination = row_matches and column_matches
        if selected_destination:
            replacements.append(
                as_grid(
                    tuple(
                        row[: panel.bbox.width]
                        for row in seed_panel.content[: panel.bbox.height]
                    )
                )
            )
        else:
            replacements.append(panel.content)
    return (
        reassemble_panels(
            normalized,
            hypothesis=hypothesis,
            replacements=tuple(replacements),
        ),
        None,
    )


def broadcast_panel_sequence_d4(
    grid: Grid,
    *,
    background: int,
    step: D4Step | str,
) -> tuple[Grid | None, PanelSequenceD4Code | None]:
    """Broadcast one seed through a single-axis panel sequence by D4 powers."""

    if type(background) is not int or not 0 <= background <= 9:
        raise ValueError("background must be an ARC color")
    try:
        selected_step = D4Step(step)
    except (TypeError, ValueError) as exc:
        raise ValueError("step must be a valid D4 element") from exc
    normalized = as_grid(grid)
    sequences = tuple(
        hypothesis
        for hypothesis in parse_panels(normalized).hypotheses
        if (
            bool(hypothesis.row_separator_bands)
            ^ bool(hypothesis.column_separator_bands)
        )
        and hypothesis.separator_color != background
    )
    if not sequences:
        return None, PanelSequenceD4Code.EMPTY_SELECTION
    valid: list[tuple[PanelHypothesis, tuple[Grid, ...]]] = []
    saw_multiple_seeds = False
    saw_incompatible_shape = False
    for hypothesis in sequences:
        occupied = tuple(
            index
            for index, panel in enumerate(hypothesis.panels)
            if any(cell != background for row in panel.content for cell in row)
        )
        if len(occupied) > 1:
            saw_multiple_seeds = True
        if len(
            {
                (panel.bbox.height, panel.bbox.width)
                for panel in hypothesis.panels
            }
        ) != 1:
            saw_incompatible_shape = True
            continue
        if not occupied:
            continue
        if len(occupied) != 1:
            continue
        seed_index = occupied[0]
        seed = hypothesis.panels[seed_index].content
        replacements = tuple(
            apply_d4_power(
                seed,
                step=selected_step,
                exponent=index - seed_index,
            )
            for index in range(len(hypothesis.panels))
        )
        if any(
            len(replacement) != panel.bbox.height
            or len(replacement[0]) != panel.bbox.width
            for panel, replacement in zip(hypothesis.panels, replacements)
        ):
            saw_incompatible_shape = True
            continue
        valid.append((hypothesis, replacements))
    if len(valid) > 1:
        return None, PanelSequenceD4Code.NON_UNIQUE_SELECTION
    if not valid:
        if saw_multiple_seeds:
            return None, PanelSequenceD4Code.NON_UNIQUE_SELECTION
        if saw_incompatible_shape:
            return None, PanelSequenceD4Code.INCOMPATIBLE_PANEL_SHAPES
        return None, PanelSequenceD4Code.EMPTY_SELECTION
    selected, replacements = valid[0]
    return (
        reassemble_panels(
            normalized,
            hypothesis=selected,
            replacements=replacements,
        ),
        None,
    )
