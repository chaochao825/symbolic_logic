"""Deterministic multi-view ARC parsing with immutable provenance records."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable

from .grid import Grid, as_grid, grid_key

Coordinate = tuple[int, int]
PARSER_SEMANTICS_VERSION = "afts-multiview-cc/v0.1"


class ColorMode(str, Enum):
    SINGLE_COLOR = "single_color"
    MULTICOLOR_FOREGROUND = "multicolor_foreground"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    top: int
    left: int
    bottom: int
    right: int

    def __post_init__(self) -> None:
        values = (self.top, self.left, self.bottom, self.right)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("bounding-box coordinates must be non-negative integers")
        if self.bottom < self.top or self.right < self.left:
            raise ValueError("bounding box must have non-negative extent")

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1

    @property
    def width(self) -> int:
        return self.right - self.left + 1

    def contains(self, other: "BoundingBox") -> bool:
        return (
            self.top <= other.top
            and self.left <= other.left
            and self.bottom >= other.bottom
            and self.right >= other.right
        )

    def to_json_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: object) -> "BoundingBox":
        if not isinstance(payload, dict) or set(payload) != {
            "top",
            "left",
            "bottom",
            "right",
        }:
            raise ValueError("bounding box must contain exactly four coordinates")
        return cls(
            top=payload["top"],
            left=payload["left"],
            bottom=payload["bottom"],
            right=payload["right"],
        )


@dataclass(frozen=True, slots=True)
class ObjectView:
    object_id: str
    pixels: tuple[Coordinate, ...]
    color_counts: tuple[tuple[int, int], ...]
    bbox: BoundingBox
    area: int
    hole_count: int

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id must not be empty")
        if not self.pixels or tuple(sorted(self.pixels)) != self.pixels:
            raise ValueError("pixels must be a non-empty sorted tuple")
        if len(set(self.pixels)) != len(self.pixels):
            raise ValueError("pixels must be unique")
        if any(
            type(row) is not int
            or type(column) is not int
            or row < 0
            or column < 0
            for row, column in self.pixels
        ):
            raise ValueError("object pixels must be non-negative integer coordinates")
        if self.area != len(self.pixels):
            raise ValueError("area must equal the number of pixels")
        if self.hole_count < 0:
            raise ValueError("hole_count must be non-negative")
        if (
            not self.color_counts
            or tuple(sorted(self.color_counts)) != self.color_counts
            or len({color for color, _ in self.color_counts}) != len(self.color_counts)
            or any(
                type(color) is not int
                or not 0 <= color <= 9
                or type(count) is not int
                or count <= 0
                for color, count in self.color_counts
            )
            or sum(count for _, count in self.color_counts) != self.area
        ):
            raise ValueError("color counts must cover every object pixel")
        expected_bbox = _bbox(self.pixels)
        if self.bbox != expected_bbox:
            raise ValueError("object bbox does not match its pixels")
        expected_id = _canonical_hash(
            {
                "pixels": self.pixels,
                "colors": list(self.color_counts),
                "bbox": asdict(self.bbox),
            }
        )[:20]
        if self.object_id != expected_id:
            raise ValueError("object_id does not match canonical object content")

    @property
    def colors(self) -> tuple[int, ...]:
        return tuple(color for color, _ in self.color_counts)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "pixels": [list(item) for item in self.pixels],
            "color_counts": [list(item) for item in self.color_counts],
            "bbox": self.bbox.to_json_dict(),
            "area": self.area,
            "hole_count": self.hole_count,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "ObjectView":
        expected = {
            "object_id",
            "pixels",
            "color_counts",
            "bbox",
            "area",
            "hole_count",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("object view has missing or unknown fields")
        pixels = payload["pixels"]
        counts = payload["color_counts"]
        if not isinstance(pixels, list) or any(
            not isinstance(item, list) or len(item) != 2 for item in pixels
        ):
            raise TypeError("object pixels must be coordinate pairs")
        if not isinstance(counts, list) or any(
            not isinstance(item, list) or len(item) != 2 for item in counts
        ):
            raise TypeError("object color_counts must be pairs")
        return cls(
            object_id=payload["object_id"],
            pixels=tuple((item[0], item[1]) for item in pixels),
            color_counts=tuple((item[0], item[1]) for item in counts),
            bbox=BoundingBox.from_json_dict(payload["bbox"]),
            area=payload["area"],
            hole_count=payload["hole_count"],
        )


@dataclass(frozen=True, slots=True)
class ObjectRelation:
    first_object_id: str
    second_object_id: str
    touching_4: bool
    touching_8: bool
    first_bbox_contains_second: bool
    second_bbox_contains_first: bool
    row_ranges_overlap: bool
    column_ranges_overlap: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.first_object_id, str)
            or not self.first_object_id
            or not isinstance(self.second_object_id, str)
            or not self.second_object_id
            or self.first_object_id == self.second_object_id
        ):
            raise ValueError("object relation endpoints must be distinct IDs")
        for name in (
            "touching_4",
            "touching_8",
            "first_bbox_contains_second",
            "second_bbox_contains_first",
            "row_ranges_overlap",
            "column_ranges_overlap",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"relation field {name} must be boolean")

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: object) -> "ObjectRelation":
        expected = {
            "first_object_id",
            "second_object_id",
            "touching_4",
            "touching_8",
            "first_bbox_contains_second",
            "second_bbox_contains_first",
            "row_ranges_overlap",
            "column_ranges_overlap",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("object relation has missing or unknown fields")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ParseHypothesis:
    parse_id: str
    grid_key: str
    background_color: int | None
    connectivity: int
    color_mode: ColorMode
    objects: tuple[ObjectView, ...]
    relations: tuple[ObjectRelation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.color_mode, ColorMode):
            raise TypeError("color_mode must be a ColorMode")
        if self.connectivity not in {4, 8}:
            raise ValueError("connectivity must be 4 or 8")
        if self.background_color is not None and (
            type(self.background_color) is not int or not 0 <= self.background_color <= 9
        ):
            raise ValueError("background_color must be an ARC color")
        if tuple(sorted(self.objects, key=lambda item: item.object_id)) != self.objects:
            raise ValueError("objects must be sorted by object_id")
        if len({item.object_id for item in self.objects}) != len(self.objects):
            raise ValueError("objects must have unique content IDs")
        object_ids = {item.object_id for item in self.objects}
        relation_pairs: set[tuple[str, str]] = set()
        for relation in self.relations:
            pair = (relation.first_object_id, relation.second_object_id)
            if (
                relation.first_object_id not in object_ids
                or relation.second_object_id not in object_ids
                or relation.first_object_id >= relation.second_object_id
                or pair in relation_pairs
            ):
                raise ValueError("relations must be unique ordered references to objects")
            if relation.touching_4 and not relation.touching_8:
                raise ValueError("4-touching objects must also be 8-touching")
            relation_pairs.add(pair)
        if self.relations != _relations(self.objects):
            raise ValueError("relations do not match the canonical sparse object relations")
        expected_id = _canonical_hash(
            {
                "parser_semantics_version": PARSER_SEMANTICS_VERSION,
                "grid_key": self.grid_key,
                "background": self.background_color,
                "connectivity": self.connectivity,
                "color_mode": self.color_mode.value,
                "objects": [asdict(item) for item in self.objects],
            }
        )[:20]
        if self.parse_id != expected_id:
            raise ValueError("parse_id does not match canonical hypothesis content")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "parse_id": self.parse_id,
            "grid_key": self.grid_key,
            "background_color": self.background_color,
            "connectivity": self.connectivity,
            "color_mode": self.color_mode.value,
            "objects": [item.to_json_dict() for item in self.objects],
            "relations": [item.to_json_dict() for item in self.relations],
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "ParseHypothesis":
        expected = {
            "parse_id",
            "grid_key",
            "background_color",
            "connectivity",
            "color_mode",
            "objects",
            "relations",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("parse hypothesis has missing or unknown fields")
        if not isinstance(payload["objects"], list) or not isinstance(
            payload["relations"], list
        ):
            raise TypeError("parse objects and relations must be lists")
        return cls(
            parse_id=payload["parse_id"],
            grid_key=payload["grid_key"],
            background_color=payload["background_color"],
            connectivity=payload["connectivity"],
            color_mode=ColorMode(payload["color_mode"]),
            objects=tuple(ObjectView.from_json_dict(item) for item in payload["objects"]),
            relations=tuple(
                ObjectRelation.from_json_dict(item) for item in payload["relations"]
            ),
        )


@dataclass(frozen=True, slots=True)
class GridParseBundle:
    grid_key: str
    background_candidates: tuple[int | None, ...]
    hypotheses: tuple[ParseHypothesis, ...]

    def __post_init__(self) -> None:
        if not self.background_candidates or len(set(self.background_candidates)) != len(
            self.background_candidates
        ):
            raise ValueError("background candidates must be non-empty and unique")
        if not self.hypotheses:
            raise ValueError("parse bundle must contain hypotheses")
        if any(item.grid_key != self.grid_key for item in self.hypotheses):
            raise ValueError("all hypotheses must bind the bundle grid_key")
        if len({item.parse_id for item in self.hypotheses}) != len(self.hypotheses):
            raise ValueError("parse hypotheses must have unique content IDs")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "grid_key": self.grid_key,
            "background_candidates": list(self.background_candidates),
            "hypotheses": [item.to_json_dict() for item in self.hypotheses],
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "GridParseBundle":
        expected = {"grid_key", "background_candidates", "hypotheses"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("parse bundle has missing or unknown fields")
        if not isinstance(payload["background_candidates"], list) or not isinstance(
            payload["hypotheses"], list
        ):
            raise TypeError("parse bundle candidates and hypotheses must be lists")
        return cls(
            grid_key=payload["grid_key"],
            background_candidates=tuple(payload["background_candidates"]),
            hypotheses=tuple(
                ParseHypothesis.from_json_dict(item) for item in payload["hypotheses"]
            ),
        )


_DIRECTIONS_4: tuple[Coordinate, ...] = ((-1, 0), (0, -1), (0, 1), (1, 0))
_DIRECTIONS_8: tuple[Coordinate, ...] = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def background_hypotheses(
    grid: Grid,
    *,
    include_none: bool = True,
    max_backgrounds: int | None = None,
) -> tuple[int | None, ...]:
    """Return deterministic alternate backgrounds ordered by simple evidence."""

    normalized = as_grid(grid)
    if max_backgrounds is not None and max_backgrounds <= 0:
        raise ValueError("max_backgrounds must be positive")
    counts = Counter(cell for row in normalized for cell in row)
    height = len(normalized)
    width = len(normalized[0])
    border = [
        normalized[row][column]
        for row in range(height)
        for column in range(width)
        if row in {0, height - 1} or column in {0, width - 1}
    ]
    border_counts = Counter(border)
    ordered: list[int | None] = [None] if include_none else []

    def append(color: int) -> None:
        if color not in ordered:
            ordered.append(color)

    append(min(counts, key=lambda color: (-counts[color], color)))
    append(min(border_counts, key=lambda color: (-border_counts[color], color)))
    if 0 in counts:
        append(0)
    for color in sorted(counts, key=lambda item: (-counts[item], item)):
        append(color)
    if max_backgrounds is not None:
        ordered = ordered[:max_backgrounds]
    return tuple(ordered)


def _neighbors(
    coordinate: Coordinate, *, height: int, width: int, connectivity: int
) -> Iterable[Coordinate]:
    directions = _DIRECTIONS_4 if connectivity == 4 else _DIRECTIONS_8
    row, column = coordinate
    for dr, dc in directions:
        neighbor = (row + dr, column + dc)
        if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width:
            yield neighbor


def _components(
    grid: Grid,
    *,
    background: int | None,
    connectivity: int,
    color_mode: ColorMode,
) -> tuple[tuple[Coordinate, ...], ...]:
    height = len(grid)
    width = len(grid[0])
    foreground = {
        (row, column)
        for row in range(height)
        for column in range(width)
        if background is None or grid[row][column] != background
    }
    unseen = set(foreground)
    components: list[tuple[Coordinate, ...]] = []
    while unseen:
        start = min(unseen)
        target_color = grid[start[0]][start[1]]
        queue: deque[Coordinate] = deque((start,))
        unseen.remove(start)
        component: list[Coordinate] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in _neighbors(
                current, height=height, width=width, connectivity=connectivity
            ):
                if neighbor not in unseen:
                    continue
                if (
                    color_mode is ColorMode.SINGLE_COLOR
                    and grid[neighbor[0]][neighbor[1]] != target_color
                ):
                    continue
                unseen.remove(neighbor)
                queue.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _bbox(pixels: tuple[Coordinate, ...]) -> BoundingBox:
    rows = tuple(row for row, _ in pixels)
    columns = tuple(column for _, column in pixels)
    return BoundingBox(min(rows), min(columns), max(rows), max(columns))


def _hole_count(
    pixels: tuple[Coordinate, ...], bbox: BoundingBox, *, object_connectivity: int
) -> int:
    occupied = set(pixels)
    empty = {
        (row, column)
        for row in range(bbox.top, bbox.bottom + 1)
        for column in range(bbox.left, bbox.right + 1)
        if (row, column) not in occupied
    }
    holes = 0
    while empty:
        start = min(empty)
        queue: deque[Coordinate] = deque((start,))
        empty.remove(start)
        touches_boundary = False
        while queue:
            row, column = queue.popleft()
            touches_boundary |= (
                row in {bbox.top, bbox.bottom} or column in {bbox.left, bbox.right}
            )
            hole_directions = _DIRECTIONS_8 if object_connectivity == 4 else _DIRECTIONS_4
            for dr, dc in hole_directions:
                neighbor = (row + dr, column + dc)
                if neighbor in empty:
                    empty.remove(neighbor)
                    queue.append(neighbor)
        holes += int(not touches_boundary)
    return holes


def _make_object(
    grid: Grid, pixels: tuple[Coordinate, ...], *, connectivity: int
) -> ObjectView:
    bbox = _bbox(pixels)
    counts = Counter(grid[row][column] for row, column in pixels)
    payload = {
        "pixels": pixels,
        "colors": sorted(counts.items()),
        "bbox": asdict(bbox),
    }
    return ObjectView(
        object_id=_canonical_hash(payload)[:20],
        pixels=pixels,
        color_counts=tuple(sorted(counts.items())),
        bbox=bbox,
        area=len(pixels),
        hole_count=_hole_count(pixels, bbox, object_connectivity=connectivity),
    )


def _touches(first: ObjectView, second: ObjectView, connectivity: int) -> bool:
    second_pixels = set(second.pixels)
    directions = _DIRECTIONS_4 if connectivity == 4 else _DIRECTIONS_8
    return any(
        (row + dr, column + dc) in second_pixels
        for row, column in first.pixels
        for dr, dc in directions
    )


def _relations(objects: tuple[ObjectView, ...]) -> tuple[ObjectRelation, ...]:
    relations: list[ObjectRelation] = []
    for first_index, first in enumerate(objects):
        for second in objects[first_index + 1 :]:
            touching_4 = _touches(first, second, 4)
            touching_8 = _touches(first, second, 8)
            first_contains = first.bbox.contains(second.bbox)
            second_contains = second.bbox.contains(first.bbox)
            if touching_8 or first_contains or second_contains:
                relations.append(
                    ObjectRelation(
                        first_object_id=first.object_id,
                        second_object_id=second.object_id,
                        touching_4=touching_4,
                        touching_8=touching_8,
                        first_bbox_contains_second=first_contains,
                        second_bbox_contains_first=second_contains,
                        row_ranges_overlap=not (
                            first.bbox.bottom < second.bbox.top
                            or second.bbox.bottom < first.bbox.top
                        ),
                        column_ranges_overlap=not (
                            first.bbox.right < second.bbox.left
                            or second.bbox.right < first.bbox.left
                        ),
                    )
                )
    return tuple(relations)


def parse_grid(
    grid: Grid,
    *,
    backgrounds: Iterable[int | None] | None = None,
    connectivities: tuple[int, ...] = (4, 8),
    color_modes: tuple[ColorMode, ...] = (
        ColorMode.SINGLE_COLOR,
        ColorMode.MULTICOLOR_FOREGROUND,
    ),
    max_backgrounds: int | None = None,
) -> GridParseBundle:
    normalized = as_grid(grid)
    if any(connectivity not in {4, 8} for connectivity in connectivities):
        raise ValueError("connectivities may contain only 4 and 8")
    background_values = (
        background_hypotheses(normalized, max_backgrounds=max_backgrounds)
        if backgrounds is None
        else tuple(dict.fromkeys(backgrounds))
    )
    if not background_values or any(
        color is not None and (type(color) is not int or not 0 <= color <= 9)
        for color in background_values
    ):
        raise ValueError("backgrounds must contain ARC colors")

    hypotheses: list[ParseHypothesis] = []
    grid_fingerprint = grid_key(normalized)
    for background in background_values:
        for connectivity in connectivities:
            for color_mode in color_modes:
                objects = tuple(
                    sorted(
                        (
                            _make_object(
                                normalized, component, connectivity=connectivity
                            )
                            for component in _components(
                                normalized,
                                background=background,
                                connectivity=connectivity,
                                color_mode=color_mode,
                            )
                        ),
                        key=lambda item: item.object_id,
                    )
                )
                payload = {
                    "parser_semantics_version": PARSER_SEMANTICS_VERSION,
                    "grid_key": grid_fingerprint,
                    "background": background,
                    "connectivity": connectivity,
                    "color_mode": color_mode.value,
                    "objects": [asdict(item) for item in objects],
                }
                hypotheses.append(
                    ParseHypothesis(
                        parse_id=_canonical_hash(payload)[:20],
                        grid_key=grid_fingerprint,
                        background_color=background,
                        connectivity=connectivity,
                        color_mode=color_mode,
                        objects=objects,
                        relations=_relations(objects),
                    )
                )
    return GridParseBundle(
        grid_key=grid_fingerprint,
        background_candidates=background_values,
        hypotheses=tuple(hypotheses),
    )
