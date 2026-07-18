"""Independent full-span separator and panel views for M02b."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum

from .grid import ARC_MAX_HEIGHT, ARC_MAX_WIDTH, Grid, as_grid, grid_key, grid_to_lists
from .parse import BoundingBox

PANEL_PARSER_SEMANTICS_VERSION = "afts-full-span-panels/v0.1"
PANEL_PARSER_STAGE = "M02b_full_span_separator_panels"


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SeparatorBand:
    start: int
    end: int

    def __post_init__(self) -> None:
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or self.start < 0
            or self.end < self.start
        ):
            raise ValueError("separator band must be a non-negative closed interval")

    def to_json_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: object) -> "SeparatorBand":
        if not isinstance(payload, dict) or set(payload) != {"start", "end"}:
            raise ValueError("separator band must contain exactly start and end")
        return cls(start=payload["start"], end=payload["end"])


@dataclass(frozen=True, slots=True)
class PanelView:
    panel_id: str
    row_index: int
    column_index: int
    bbox: BoundingBox
    content: Grid
    content_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.panel_id, str) or not self.panel_id:
            raise TypeError("panel_id must be a non-empty string")
        if type(self.row_index) is not int or self.row_index < 0:
            raise ValueError("panel row_index must be a non-negative integer")
        if type(self.column_index) is not int or self.column_index < 0:
            raise ValueError("panel column_index must be a non-negative integer")
        if not isinstance(self.bbox, BoundingBox):
            raise TypeError("panel bbox must be a BoundingBox")
        content = as_grid(self.content)
        if len(content) != self.bbox.height or len(content[0]) != self.bbox.width:
            raise ValueError("panel content shape does not match its bbox")
        if self.content_key != grid_key(content):
            raise ValueError("panel content_key does not match panel content")
        object.__setattr__(self, "content", content)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "panel_id": self.panel_id,
            "row_index": self.row_index,
            "column_index": self.column_index,
            "bbox": self.bbox.to_json_dict(),
            "content": grid_to_lists(self.content),
            "content_key": self.content_key,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "PanelView":
        expected = {
            "panel_id",
            "row_index",
            "column_index",
            "bbox",
            "content",
            "content_key",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("panel view has missing or unknown fields")
        return cls(
            panel_id=payload["panel_id"],
            row_index=payload["row_index"],
            column_index=payload["column_index"],
            bbox=BoundingBox.from_json_dict(payload["bbox"]),
            content=as_grid(payload["content"]),
            content_key=payload["content_key"],
        )


def _validate_bands(
    bands: tuple[SeparatorBand, ...], *, limit: int, axis: str
) -> None:
    if any(not isinstance(band, SeparatorBand) for band in bands):
        raise TypeError(f"{axis} separator bands must contain SeparatorBand values")
    if tuple(sorted(bands, key=lambda item: item.start)) != bands:
        raise ValueError(f"{axis} separator bands must be sorted")
    previous_end = -1
    for band in bands:
        if band.end >= limit:
            raise ValueError(f"{axis} separator band exceeds the grid")
        if band.start <= previous_end + 1 and previous_end >= 0:
            raise ValueError(f"{axis} separator bands must be disjoint and merged")
        previous_end = band.end


def _complement_intervals(
    *, limit: int, bands: tuple[SeparatorBand, ...]
) -> tuple[tuple[int, int], ...]:
    intervals: list[tuple[int, int]] = []
    cursor = 0
    for band in bands:
        if cursor < band.start:
            intervals.append((cursor, band.start - 1))
        cursor = band.end + 1
    if cursor < limit:
        intervals.append((cursor, limit - 1))
    return tuple(intervals)


def _panel_id_payload(
    *,
    grid_key_value: str,
    separator_color: int,
    panel: PanelView,
) -> dict[str, object]:
    return {
        "panel_parser_semantics_version": PANEL_PARSER_SEMANTICS_VERSION,
        "grid_key": grid_key_value,
        "separator_color": separator_color,
        "row_index": panel.row_index,
        "column_index": panel.column_index,
        "bbox": panel.bbox.to_json_dict(),
        "content": grid_to_lists(panel.content),
        "content_key": panel.content_key,
    }


def _hypothesis_id_payload(
    *,
    grid_key_value: str,
    grid_height: int,
    grid_width: int,
    separator_color: int,
    row_separator_bands: tuple[SeparatorBand, ...],
    column_separator_bands: tuple[SeparatorBand, ...],
    panels: tuple[PanelView, ...],
) -> dict[str, object]:
    return {
        "panel_parser_semantics_version": PANEL_PARSER_SEMANTICS_VERSION,
        "grid_key": grid_key_value,
        "grid_height": grid_height,
        "grid_width": grid_width,
        "separator_color": separator_color,
        "row_separator_bands": [item.to_json_dict() for item in row_separator_bands],
        "column_separator_bands": [
            item.to_json_dict() for item in column_separator_bands
        ],
        "panels": [item.to_json_dict() for item in panels],
    }


@dataclass(frozen=True, slots=True)
class PanelHypothesis:
    parse_id: str
    grid_key: str
    grid_height: int
    grid_width: int
    separator_color: int
    row_separator_bands: tuple[SeparatorBand, ...]
    column_separator_bands: tuple[SeparatorBand, ...]
    panels: tuple[PanelView, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.grid_key, str) or len(self.grid_key) != 64:
            raise ValueError("panel hypothesis grid_key must be a SHA-256 hex string")
        try:
            int(self.grid_key, 16)
        except ValueError as exc:
            raise ValueError("panel hypothesis grid_key must be hexadecimal") from exc
        if (
            type(self.grid_height) is not int
            or not 1 <= self.grid_height <= ARC_MAX_HEIGHT
        ):
            raise ValueError("panel grid_height must be in [1, 30]")
        if (
            type(self.grid_width) is not int
            or not 1 <= self.grid_width <= ARC_MAX_WIDTH
        ):
            raise ValueError("panel grid_width must be in [1, 30]")
        if type(self.separator_color) is not int or not 0 <= self.separator_color <= 9:
            raise ValueError("separator_color must be an ARC color")
        row_bands = tuple(self.row_separator_bands)
        column_bands = tuple(self.column_separator_bands)
        if not row_bands and not column_bands:
            raise ValueError("panel hypothesis needs a row or column separator")
        _validate_bands(row_bands, limit=self.grid_height, axis="row")
        _validate_bands(column_bands, limit=self.grid_width, axis="column")
        row_intervals = _complement_intervals(
            limit=self.grid_height, bands=row_bands
        )
        column_intervals = _complement_intervals(
            limit=self.grid_width, bands=column_bands
        )
        expected_positions = tuple(
            (row_index, column_index, row_interval, column_interval)
            for row_index, row_interval in enumerate(row_intervals)
            for column_index, column_interval in enumerate(column_intervals)
        )
        panels = tuple(self.panels)
        if len(expected_positions) < 2 or len(panels) != len(expected_positions):
            raise ValueError("panel hypothesis must contain the full multi-panel lattice")
        if any(not isinstance(panel, PanelView) for panel in panels):
            raise TypeError("panels must contain PanelView values")
        reconstructed = [
            [self.separator_color for _ in range(self.grid_width)]
            for _ in range(self.grid_height)
        ]
        for panel, expected in zip(panels, expected_positions):
            row_index, column_index, row_interval, column_interval = expected
            expected_bbox = BoundingBox(
                top=row_interval[0],
                left=column_interval[0],
                bottom=row_interval[1],
                right=column_interval[1],
            )
            if (
                panel.row_index != row_index
                or panel.column_index != column_index
                or panel.bbox != expected_bbox
            ):
                raise ValueError("panels must be the canonical row-major lattice")
            expected_panel_id = _canonical_hash(
                _panel_id_payload(
                    grid_key_value=self.grid_key,
                    separator_color=self.separator_color,
                    panel=panel,
                )
            )[:20]
            if panel.panel_id != expected_panel_id:
                raise ValueError("panel_id does not match canonical panel content")
            for local_row, values in enumerate(panel.content):
                for local_column, cell in enumerate(values):
                    reconstructed[panel.bbox.top + local_row][
                        panel.bbox.left + local_column
                    ] = cell
        if grid_key(as_grid(reconstructed)) != self.grid_key:
            raise ValueError("panel lattice does not reconstruct the bound grid")
        expected_parse_id = _canonical_hash(
            _hypothesis_id_payload(
                grid_key_value=self.grid_key,
                grid_height=self.grid_height,
                grid_width=self.grid_width,
                separator_color=self.separator_color,
                row_separator_bands=row_bands,
                column_separator_bands=column_bands,
                panels=panels,
            )
        )[:20]
        if self.parse_id != expected_parse_id:
            raise ValueError("parse_id does not match canonical panel hypothesis")
        object.__setattr__(self, "row_separator_bands", row_bands)
        object.__setattr__(self, "column_separator_bands", column_bands)
        object.__setattr__(self, "panels", panels)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "parse_id": self.parse_id,
            "grid_key": self.grid_key,
            "grid_height": self.grid_height,
            "grid_width": self.grid_width,
            "separator_color": self.separator_color,
            "row_separator_bands": [
                item.to_json_dict() for item in self.row_separator_bands
            ],
            "column_separator_bands": [
                item.to_json_dict() for item in self.column_separator_bands
            ],
            "panels": [item.to_json_dict() for item in self.panels],
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "PanelHypothesis":
        expected = {
            "parse_id",
            "grid_key",
            "grid_height",
            "grid_width",
            "separator_color",
            "row_separator_bands",
            "column_separator_bands",
            "panels",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("panel hypothesis has missing or unknown fields")
        for field in ("row_separator_bands", "column_separator_bands", "panels"):
            if not isinstance(payload[field], list):
                raise TypeError(f"panel hypothesis {field} must be a list")
        return cls(
            parse_id=payload["parse_id"],
            grid_key=payload["grid_key"],
            grid_height=payload["grid_height"],
            grid_width=payload["grid_width"],
            separator_color=payload["separator_color"],
            row_separator_bands=tuple(
                SeparatorBand.from_json_dict(item)
                for item in payload["row_separator_bands"]
            ),
            column_separator_bands=tuple(
                SeparatorBand.from_json_dict(item)
                for item in payload["column_separator_bands"]
            ),
            panels=tuple(PanelView.from_json_dict(item) for item in payload["panels"]),
        )


@dataclass(frozen=True, slots=True)
class PanelParseBundle:
    grid_key: str
    hypotheses: tuple[PanelHypothesis, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.grid_key, str) or len(self.grid_key) != 64:
            raise ValueError("panel bundle grid_key must be a SHA-256 hex string")
        try:
            int(self.grid_key, 16)
        except ValueError as exc:
            raise ValueError("panel bundle grid_key must be hexadecimal") from exc
        hypotheses = tuple(self.hypotheses)
        if any(not isinstance(item, PanelHypothesis) for item in hypotheses):
            raise TypeError("panel hypotheses must contain PanelHypothesis values")
        if any(item.grid_key != self.grid_key for item in hypotheses):
            raise ValueError("all panel hypotheses must bind the bundle grid_key")
        if tuple(sorted(hypotheses, key=lambda item: item.separator_color)) != hypotheses:
            raise ValueError("panel hypotheses must be sorted by separator color")
        if len({item.separator_color for item in hypotheses}) != len(hypotheses):
            raise ValueError("panel hypotheses must have unique separator colors")
        if len({item.parse_id for item in hypotheses}) != len(hypotheses):
            raise ValueError("panel hypotheses must have unique parse IDs")
        object.__setattr__(self, "hypotheses", hypotheses)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "grid_key": self.grid_key,
            "hypotheses": [item.to_json_dict() for item in self.hypotheses],
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "PanelParseBundle":
        if not isinstance(payload, dict) or set(payload) != {"grid_key", "hypotheses"}:
            raise ValueError("panel parse bundle has missing or unknown fields")
        if not isinstance(payload["hypotheses"], list):
            raise TypeError("panel parse hypotheses must be a list")
        return cls(
            grid_key=payload["grid_key"],
            hypotheses=tuple(
                PanelHypothesis.from_json_dict(item) for item in payload["hypotheses"]
            ),
        )


def _merge_indices(indices: tuple[int, ...]) -> tuple[SeparatorBand, ...]:
    if not indices:
        return ()
    bands: list[SeparatorBand] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        bands.append(SeparatorBand(start=start, end=previous))
        start = previous = index
    bands.append(SeparatorBand(start=start, end=previous))
    return tuple(bands)


def _make_hypothesis(
    grid: Grid,
    *,
    separator_color: int,
    row_bands: tuple[SeparatorBand, ...],
    column_bands: tuple[SeparatorBand, ...],
) -> PanelHypothesis:
    source_key = grid_key(grid)
    row_intervals = _complement_intervals(limit=len(grid), bands=row_bands)
    column_intervals = _complement_intervals(
        limit=len(grid[0]), bands=column_bands
    )
    panels: list[PanelView] = []
    for row_index, (top, bottom) in enumerate(row_intervals):
        for column_index, (left, right) in enumerate(column_intervals):
            content = tuple(
                tuple(row[left : right + 1]) for row in grid[top : bottom + 1]
            )
            bbox = BoundingBox(top=top, left=left, bottom=bottom, right=right)
            provisional = PanelView(
                panel_id="pending",
                row_index=row_index,
                column_index=column_index,
                bbox=bbox,
                content=content,
                content_key=grid_key(content),
            )
            panels.append(
                PanelView(
                    panel_id=_canonical_hash(
                        _panel_id_payload(
                            grid_key_value=source_key,
                            separator_color=separator_color,
                            panel=provisional,
                        )
                    )[:20],
                    row_index=row_index,
                    column_index=column_index,
                    bbox=bbox,
                    content=content,
                    content_key=grid_key(content),
                )
            )
    normalized_panels = tuple(panels)
    payload = _hypothesis_id_payload(
        grid_key_value=source_key,
        grid_height=len(grid),
        grid_width=len(grid[0]),
        separator_color=separator_color,
        row_separator_bands=row_bands,
        column_separator_bands=column_bands,
        panels=normalized_panels,
    )
    return PanelHypothesis(
        parse_id=_canonical_hash(payload)[:20],
        grid_key=source_key,
        grid_height=len(grid),
        grid_width=len(grid[0]),
        separator_color=separator_color,
        row_separator_bands=row_bands,
        column_separator_bands=column_bands,
        panels=normalized_panels,
    )


def parse_panels(grid: Grid) -> PanelParseBundle:
    """Enumerate full-span same-color separator hypotheses deterministically."""

    normalized = as_grid(grid)
    height, width = len(normalized), len(normalized[0])
    colors = sorted({cell for row in normalized for cell in row})
    hypotheses: list[PanelHypothesis] = []
    for color in colors:
        full_rows = tuple(
            row for row in range(height) if all(cell == color for cell in normalized[row])
        )
        full_columns = tuple(
            column
            for column in range(width)
            if all(normalized[row][column] == color for row in range(height))
        )
        row_bands = _merge_indices(full_rows)
        column_bands = _merge_indices(full_columns)
        if not row_bands and not column_bands:
            continue
        row_intervals = _complement_intervals(limit=height, bands=row_bands)
        column_intervals = _complement_intervals(limit=width, bands=column_bands)
        if len(row_intervals) * len(column_intervals) < 2:
            continue
        hypotheses.append(
            _make_hypothesis(
                normalized,
                separator_color=color,
                row_bands=row_bands,
                column_bands=column_bands,
            )
        )
    return PanelParseBundle(grid_key=grid_key(normalized), hypotheses=tuple(hypotheses))


class PanelOverlayCode(str, Enum):
    EMPTY_SELECTION = "empty_selection"
    NON_UNIQUE_SELECTION = "non_unique_selection"
    INCOMPATIBLE_PANEL_SHAPES = "incompatible_panel_shapes"


def overlay_panel_grid(
    grid: Grid, *, background: int
) -> tuple[Grid | None, PanelOverlayCode | None]:
    """Overlay one uniquely eligible cross-lattice in canonical row-major order."""

    if type(background) is not int or not 0 <= background <= 9:
        raise ValueError("background must be an ARC color")
    cross_lattices = tuple(
        hypothesis
        for hypothesis in parse_panels(grid).hypotheses
        if hypothesis.row_separator_bands
        and hypothesis.column_separator_bands
        and hypothesis.separator_color != background
    )
    if not cross_lattices:
        return None, PanelOverlayCode.EMPTY_SELECTION
    compatible = tuple(
        hypothesis
        for hypothesis in cross_lattices
        if len({(panel.bbox.height, panel.bbox.width) for panel in hypothesis.panels})
        == 1
    )
    if not compatible:
        return None, PanelOverlayCode.INCOMPATIBLE_PANEL_SHAPES
    if len(compatible) != 1:
        return None, PanelOverlayCode.NON_UNIQUE_SELECTION
    selected = compatible[0]
    height = selected.panels[0].bbox.height
    width = selected.panels[0].bbox.width
    output = [[background for _ in range(width)] for _ in range(height)]
    for panel in selected.panels:
        for row, values in enumerate(panel.content):
            for column, cell in enumerate(values):
                if output[row][column] == background and cell != background:
                    output[row][column] = cell
    return as_grid(output), None
