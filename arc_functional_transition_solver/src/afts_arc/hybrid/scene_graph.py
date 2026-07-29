"""Typed scene graphs and replayable multi-stage object programs.

This module is deliberately independent from the online controller.  It turns a
grid into deterministic connected-component or color-group objects, annotates
object correspondences, and executes a small typed AST.  The AST is broad enough
to represent crop, copy, count, arrange, and compose operations, while its
enumerator remains bounded and demonstration-derived.

Only :class:`~afts_arc.blind.BlindTask` is accepted by enumeration.  Therefore
demonstration outputs may constrain program fields, but query outputs are never
available to proposal generation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..blind import BlindTask
from ..grid import Grid, as_grid


SCENE_PIPELINE_DSL_VERSION = "afts-object-code-dsl/v0.2"
SCENE_AST_VERSION = "afts-object-code-scene-ast/v0.1"

SCENE_GROUPINGS = (
    "monochrome_components",
    "foreground_components",
    "color_groups",
)
SCENE_SELECTORS = (
    "all",
    "largest_area",
    "smallest_area",
    "largest_bbox",
    "smallest_bbox",
    "topmost",
    "bottommost",
    "leftmost",
    "rightmost",
    "most_holes",
    "least_holes",
    "unique_signature",
    "most_peers",
)
SCENE_CORRESPONDENCE_FEATURES = (
    "shape",
    "size",
    "topology",
    "relative_position",
)
SCENE_OPERATORS = ("crop", "copy", "count", "arrange", "compose")
SCENE_CANVAS_MODES = ("bbox", "input", "tight", "fixed", "count_line")
SCENE_RENDER_MODES = ("source_crop", "selected_only", "objects", "solid")
SCENE_D4_TRANSFORMS = (
    "identity",
    "rotate90",
    "rotate180",
    "rotate270",
    "flip_horizontal",
    "flip_vertical",
    "transpose",
    "anti_transpose",
)

Coordinate = tuple[int, int]
BoundingBox = tuple[int, int, int, int]

_DIRECTIONS = {
    4: ((-1, 0), (0, -1), (0, 1), (1, 0)),
    8: tuple(
        (dr, dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr, dc) != (0, 0)
    ),
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _content_id(value: object, *, length: int = 20) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()[:length]


def _arc_color(value: object, *, field_name: str) -> int:
    if type(value) is not int or not 0 <= value <= 9:
        raise ValueError(f"{field_name} must be an ARC color")
    return value


def _strict_int(
    value: object, *, field_name: str, minimum: int, maximum: int
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(
            f"{field_name} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def _bounding_box(cells: Sequence[Coordinate]) -> BoundingBox:
    if not cells:
        raise ValueError("an object must contain at least one cell")
    return (
        min(row for row, _ in cells),
        min(column for _, column in cells),
        max(row for row, _ in cells),
        max(column for _, column in cells),
    )


def _normalized_shape(cells: Sequence[Coordinate]) -> tuple[Coordinate, ...]:
    top, left, _, _ = _bounding_box(cells)
    return tuple(sorted((row - top, column - left) for row, column in cells))


def _transform_coordinate(coordinate: Coordinate, transform: str) -> Coordinate:
    row, column = coordinate
    mapping = {
        "identity": (row, column),
        "rotate90": (column, -row),
        "rotate180": (-row, -column),
        "rotate270": (-column, row),
        "flip_horizontal": (row, -column),
        "flip_vertical": (-row, column),
        "transpose": (column, row),
        "anti_transpose": (-column, -row),
    }
    try:
        return mapping[transform]
    except KeyError as exc:
        raise ValueError("unknown scene D4 transform") from exc


def _canonical_d4_shape(cells: Sequence[Coordinate]) -> tuple[Coordinate, ...]:
    normalized = _normalized_shape(cells)
    variants = []
    for transform in SCENE_D4_TRANSFORMS:
        transformed = tuple(
            _transform_coordinate(coordinate, transform) for coordinate in normalized
        )
        top = min(row for row, _ in transformed)
        left = min(column for _, column in transformed)
        variants.append(
            tuple(sorted((row - top, column - left) for row, column in transformed))
        )
    return min(variants)


def _transform_grid(grid: Grid, transform: str) -> Grid:
    normalized = as_grid(grid)
    height, width = len(normalized), len(normalized[0])
    cells = {
        _transform_coordinate((row, column), transform): normalized[row][column]
        for row in range(height)
        for column in range(width)
    }
    top = min(row for row, _ in cells)
    left = min(column for _, column in cells)
    bottom = max(row for row, _ in cells)
    right = max(column for _, column in cells)
    canvas = [[0 for _ in range(right - left + 1)] for _ in range(bottom - top + 1)]
    for (row, column), color in cells.items():
        canvas[row - top][column - left] = color
    return as_grid(canvas)


def _hole_count(cells: Sequence[Coordinate]) -> int:
    top, left, bottom, right = _bounding_box(cells)
    occupied = set(cells)
    unseen = {
        (row, column)
        for row in range(top, bottom + 1)
        for column in range(left, right + 1)
        if (row, column) not in occupied
    }
    holes = 0
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue: deque[Coordinate] = deque((start,))
        touches_boundary = False
        while queue:
            row, column = queue.popleft()
            touches_boundary |= row in {top, bottom} or column in {left, right}
            for dr, dc in _DIRECTIONS[4]:
                neighbor = (row + dr, column + dc)
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        holes += int(not touches_boundary)
    return holes


@dataclass(frozen=True, slots=True)
class SceneObject:
    """One deterministic object extracted from a grid."""

    object_id: str
    index: int
    cells: tuple[Coordinate, ...]
    colors: tuple[int, ...]
    bounding_box: BoundingBox
    normalized_shape: tuple[Coordinate, ...]
    canonical_d4_shape: tuple[Coordinate, ...]
    hole_count: int
    touches_border: bool

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("scene object index must be non-negative")
        if not self.cells or tuple(sorted(set(self.cells))) != self.cells:
            raise ValueError("scene object cells must be non-empty, unique, and sorted")
        if tuple(sorted(set(self.colors))) != self.colors:
            raise ValueError("scene object colors must be unique and sorted")
        if self.bounding_box != _bounding_box(self.cells):
            raise ValueError("scene object bounding box does not match its cells")
        if self.normalized_shape != _normalized_shape(self.cells):
            raise ValueError("scene object normalized shape does not match its cells")
        if self.canonical_d4_shape != _canonical_d4_shape(self.cells):
            raise ValueError("scene object canonical shape does not match its cells")
        if type(self.hole_count) is not int or self.hole_count < 0:
            raise ValueError("scene object hole count must be non-negative")
        if type(self.touches_border) is not bool:
            raise TypeError("scene object border flag must be boolean")

    @property
    def area(self) -> int:
        return len(self.cells)

    @property
    def height(self) -> int:
        top, _, bottom, _ = self.bounding_box
        return bottom - top + 1

    @property
    def width(self) -> int:
        _, left, _, right = self.bounding_box
        return right - left + 1

    @property
    def top(self) -> int:
        return self.bounding_box[0]

    @property
    def left(self) -> int:
        return self.bounding_box[1]

    @property
    def bottom(self) -> int:
        return self.bounding_box[2]

    @property
    def right(self) -> int:
        return self.bounding_box[3]


@dataclass(frozen=True, slots=True)
class SceneRelation:
    source_index: int
    target_index: int
    left_of: bool
    above: bool
    bbox_contains: bool
    target_bbox_contains: bool
    chebyshev_distance: int


@dataclass(frozen=True, slots=True)
class SceneGraph:
    height: int
    width: int
    background: int
    connectivity: int
    grouping: str
    objects: tuple[SceneObject, ...]
    relations: tuple[SceneRelation, ...]

    def __post_init__(self) -> None:
        _strict_int(self.height, field_name="scene height", minimum=1, maximum=30)
        _strict_int(self.width, field_name="scene width", minimum=1, maximum=30)
        _arc_color(self.background, field_name="scene background")
        if self.connectivity not in {4, 8}:
            raise ValueError("scene connectivity must be 4 or 8")
        if self.grouping not in SCENE_GROUPINGS:
            raise ValueError("unknown scene grouping")
        if tuple(item.index for item in self.objects) != tuple(range(len(self.objects))):
            raise ValueError("scene object indices must be dense and ordered")


def _component_groups(
    grid: Grid, *, background: int, connectivity: int, grouping: str
) -> tuple[tuple[Coordinate, ...], ...]:
    height, width = len(grid), len(grid[0])
    if grouping == "color_groups":
        return tuple(
            tuple(
                (row, column)
                for row in range(height)
                for column in range(width)
                if grid[row][column] == color
            )
            for color in sorted(
                {cell for row in grid for cell in row if cell != background}
            )
        )

    unseen = {
        (row, column)
        for row in range(height)
        for column in range(width)
        if grid[row][column] != background
    }
    groups: list[tuple[Coordinate, ...]] = []
    while unseen:
        start = min(unseen)
        start_color = grid[start[0]][start[1]]
        unseen.remove(start)
        queue: deque[Coordinate] = deque((start,))
        cells: list[Coordinate] = []
        while queue:
            row, column = queue.popleft()
            cells.append((row, column))
            for dr, dc in _DIRECTIONS[connectivity]:
                neighbor = (row + dr, column + dc)
                if neighbor not in unseen:
                    continue
                if (
                    grouping == "monochrome_components"
                    and grid[neighbor[0]][neighbor[1]] != start_color
                ):
                    continue
                unseen.remove(neighbor)
                queue.append(neighbor)
        groups.append(tuple(sorted(cells)))
    return tuple(sorted(groups, key=lambda item: (min(item), len(item), item)))


def _bbox_distance(first: SceneObject, second: SceneObject) -> int:
    row_gap = max(first.top - second.bottom - 1, second.top - first.bottom - 1, 0)
    column_gap = max(
        first.left - second.right - 1,
        second.left - first.right - 1,
        0,
    )
    return max(row_gap, column_gap)


def extract_scene_graph(
    grid: Grid,
    *,
    background: int,
    connectivity: int,
    grouping: str,
) -> SceneGraph:
    """Extract a deterministic scene graph from an ARC grid."""

    normalized = as_grid(grid)
    _arc_color(background, field_name="background")
    if type(connectivity) is not int or connectivity not in {4, 8}:
        raise ValueError("object connectivity must be 4 or 8")
    if grouping not in SCENE_GROUPINGS:
        raise ValueError("unknown scene grouping")
    height, width = len(normalized), len(normalized[0])
    groups = _component_groups(
        normalized,
        background=background,
        connectivity=connectivity,
        grouping=grouping,
    )
    objects = []
    for index, cells in enumerate(groups):
        colors = tuple(sorted({normalized[row][column] for row, column in cells}))
        box = _bounding_box(cells)
        payload = {
            "index": index,
            "cells": [list(item) for item in cells],
            "colors": list(colors),
        }
        objects.append(
            SceneObject(
                object_id=_content_id(payload),
                index=index,
                cells=cells,
                colors=colors,
                bounding_box=box,
                normalized_shape=_normalized_shape(cells),
                canonical_d4_shape=_canonical_d4_shape(cells),
                hole_count=_hole_count(cells),
                touches_border=(
                    box[0] == 0
                    or box[1] == 0
                    or box[2] == height - 1
                    or box[3] == width - 1
                ),
            )
        )
    relations = tuple(
        SceneRelation(
            source_index=first.index,
            target_index=second.index,
            left_of=first.right < second.left,
            above=first.bottom < second.top,
            bbox_contains=(
                first.top <= second.top
                and first.left <= second.left
                and first.bottom >= second.bottom
                and first.right >= second.right
            ),
            target_bbox_contains=(
                second.top <= first.top
                and second.left <= first.left
                and second.bottom >= first.bottom
                and second.right >= first.right
            ),
            chebyshev_distance=_bbox_distance(first, second),
        )
        for first in objects
        for second in objects
        if first.index != second.index
    )
    return SceneGraph(
        height,
        width,
        background,
        connectivity,
        grouping,
        tuple(objects),
        relations,
    )


@dataclass(frozen=True, slots=True)
class ParseObjectsNode:
    background: int
    connectivity: int
    grouping: str

    def __post_init__(self) -> None:
        _arc_color(self.background, field_name="parse background")
        if type(self.connectivity) is not int or self.connectivity not in {4, 8}:
            raise ValueError("parse connectivity must be 4 or 8")
        if self.grouping not in SCENE_GROUPINGS:
            raise ValueError("unknown parse grouping")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "op": "parse_objects",
            "background": self.background,
            "connectivity": self.connectivity,
            "grouping": self.grouping,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "ParseObjectsNode":
        expected = {"op", "background", "connectivity", "grouping"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("parse node has missing or unknown fields")
        if payload["op"] != "parse_objects":
            raise ValueError("scene parse node has the wrong operator")
        return cls(payload["background"], payload["connectivity"], payload["grouping"])


@dataclass(frozen=True, slots=True)
class CorrespondObjectsNode:
    policy: str = "none"
    features: tuple[str, ...] = ()
    d4_invariant: bool = True

    def __post_init__(self) -> None:
        if self.policy not in {"none", "equivalence"}:
            raise ValueError("unknown object correspondence policy")
        if tuple(sorted(set(self.features))) != self.features:
            raise ValueError("correspondence features must be unique and sorted")
        if any(item not in SCENE_CORRESPONDENCE_FEATURES for item in self.features):
            raise ValueError("unknown object correspondence feature")
        if self.policy == "none" and self.features:
            raise ValueError("disabled correspondence cannot carry features")
        if self.policy == "equivalence" and not self.features:
            raise ValueError("equivalence correspondence needs at least one feature")
        if type(self.d4_invariant) is not bool:
            raise TypeError("D4-invariant correspondence flag must be boolean")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "op": "correspond_objects",
            "policy": self.policy,
            "features": list(self.features),
            "d4_invariant": self.d4_invariant,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "CorrespondObjectsNode":
        expected = {"op", "policy", "features", "d4_invariant"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("correspondence node has missing or unknown fields")
        if payload["op"] != "correspond_objects":
            raise ValueError("scene correspondence node has the wrong operator")
        features = payload["features"]
        if not isinstance(features, list) or any(
            not isinstance(item, str) for item in features
        ):
            raise ValueError("correspondence features must be a list of strings")
        return cls(payload["policy"], tuple(features), payload["d4_invariant"])


@dataclass(frozen=True, slots=True)
class SelectObjectsNode:
    role: str

    def __post_init__(self) -> None:
        if self.role not in SCENE_SELECTORS:
            raise ValueError("unknown scene object role")

    def to_json_dict(self) -> dict[str, object]:
        return {"op": "select_objects", "role": self.role}

    @classmethod
    def from_json_dict(cls, payload: object) -> "SelectObjectsNode":
        expected = {"op", "role"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("selection node has missing or unknown fields")
        if payload["op"] != "select_objects":
            raise ValueError("scene selection node has the wrong operator")
        return cls(payload["role"])


@dataclass(frozen=True, slots=True)
class ObjectOperationNode:
    operator: str
    transform: str = "identity"
    axis: str | None = None
    spacing: int = 0
    repeat_rule: str | None = None
    output_color: int | None = None

    def __post_init__(self) -> None:
        if self.operator not in SCENE_OPERATORS:
            raise ValueError("unknown scene object operator")
        if self.transform not in SCENE_D4_TRANSFORMS:
            raise ValueError("unknown scene object transform")
        if self.axis not in {None, "row", "column"}:
            raise ValueError("scene operation axis must be row or column")
        _strict_int(self.spacing, field_name="scene spacing", minimum=0, maximum=2)
        if self.repeat_rule not in {None, "selected", "scene_objects", "peers"}:
            raise ValueError("unknown scene repeat rule")
        if self.output_color is not None:
            _arc_color(self.output_color, field_name="operation output color")
        if self.operator == "crop":
            if (
                self.axis is not None
                or self.repeat_rule is not None
                or self.output_color is not None
            ):
                raise ValueError("crop cannot carry axis, repeat, or output-color fields")
        elif self.operator == "count":
            if self.axis is None or self.repeat_rule not in {"selected", "scene_objects", "peers"}:
                raise ValueError("count requires an axis and repeat rule")
            if self.output_color is None:
                raise ValueError("count requires an output color")
        elif self.operator == "copy":
            if self.axis is None or self.repeat_rule not in {"scene_objects", "peers"}:
                raise ValueError("copy requires an axis and repeat rule")
            if self.output_color is not None:
                raise ValueError("copy preserves object colors")
        elif self.operator == "arrange":
            if self.axis is None or self.repeat_rule is not None or self.output_color is not None:
                raise ValueError("arrange requires only an axis")
        elif self.operator == "compose" and (
            self.axis is not None or self.repeat_rule is not None or self.output_color is not None
        ):
            raise ValueError("compose cannot carry axis, repeat, or output-color fields")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "op": "object_operation",
            "operator": self.operator,
            "transform": self.transform,
            "axis": self.axis,
            "spacing": self.spacing,
            "repeat_rule": self.repeat_rule,
            "output_color": self.output_color,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "ObjectOperationNode":
        expected = {
            "op",
            "operator",
            "transform",
            "axis",
            "spacing",
            "repeat_rule",
            "output_color",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("operation node has missing or unknown fields")
        if payload["op"] != "object_operation":
            raise ValueError("scene operation node has the wrong operator")
        return cls(
            payload["operator"],
            payload["transform"],
            payload["axis"],
            payload["spacing"],
            payload["repeat_rule"],
            payload["output_color"],
        )


@dataclass(frozen=True, slots=True)
class CanvasNode:
    mode: str
    background: int
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)
    height: int | None = None
    width: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in SCENE_CANVAS_MODES:
            raise ValueError("unknown scene canvas mode")
        _arc_color(self.background, field_name="canvas background")
        if len(self.padding) != 4:
            raise ValueError("canvas padding must contain top, bottom, left, right")
        for index, value in enumerate(self.padding):
            _strict_int(
                value,
                field_name=f"canvas padding {index}",
                minimum=0,
                maximum=2,
            )
        if (self.height is None) != (self.width is None):
            raise ValueError("fixed canvas height and width must appear together")
        if self.height is not None:
            _strict_int(self.height, field_name="canvas height", minimum=1, maximum=30)
            _strict_int(self.width, field_name="canvas width", minimum=1, maximum=30)
        if self.mode == "fixed" and self.height is None:
            raise ValueError("fixed canvas requires explicit height and width")
        if self.mode != "fixed" and self.height is not None:
            raise ValueError("only a fixed canvas may carry height and width")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "op": "infer_canvas",
            "mode": self.mode,
            "background": self.background,
            "padding": list(self.padding),
            "height": self.height,
            "width": self.width,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "CanvasNode":
        expected = {"op", "mode", "background", "padding", "height", "width"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("canvas node has missing or unknown fields")
        if payload["op"] != "infer_canvas":
            raise ValueError("scene canvas node has the wrong operator")
        padding = payload["padding"]
        if not isinstance(padding, list) or len(padding) != 4:
            raise ValueError("canvas padding must be a four-item list")
        return cls(
            payload["mode"],
            payload["background"],
            tuple(padding),
            payload["height"],
            payload["width"],
        )


@dataclass(frozen=True, slots=True)
class RenderObjectsNode:
    mode: str
    conflict_policy: str = "reject"

    def __post_init__(self) -> None:
        if self.mode not in SCENE_RENDER_MODES:
            raise ValueError("unknown scene render mode")
        if self.conflict_policy not in {"reject", "first", "last"}:
            raise ValueError("unknown scene render conflict policy")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "op": "render_objects",
            "mode": self.mode,
            "conflict_policy": self.conflict_policy,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "RenderObjectsNode":
        expected = {"op", "mode", "conflict_policy"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("render node has missing or unknown fields")
        if payload["op"] != "render_objects":
            raise ValueError("scene render node has the wrong operator")
        return cls(payload["mode"], payload["conflict_policy"])


@dataclass(frozen=True, slots=True)
class ScenePipelineProgram:
    parse: ParseObjectsNode
    correspond: CorrespondObjectsNode
    select: SelectObjectsNode
    operate: ObjectOperationNode
    canvas: CanvasNode
    render: RenderObjectsNode

    def __post_init__(self) -> None:
        operator = self.operate.operator
        allowed_canvas = {
            "crop": {"bbox", "fixed"},
            "copy": {"tight", "input", "fixed"},
            "count": {"count_line", "fixed"},
            "arrange": {"tight", "fixed"},
            "compose": {"tight", "fixed"},
        }[operator]
        if self.canvas.mode not in allowed_canvas:
            raise ValueError(f"{operator} cannot use canvas mode {self.canvas.mode}")
        allowed_render = {
            "crop": {"source_crop", "selected_only"},
            "copy": {"objects"},
            "count": {"solid"},
            "arrange": {"objects"},
            "compose": {"objects"},
        }[operator]
        if self.render.mode not in allowed_render:
            raise ValueError(f"{operator} cannot use render mode {self.render.mode}")
        if self.select.role in {"unique_signature", "most_peers"}:
            if self.correspond.policy != "equivalence":
                raise ValueError("signature roles require equivalence correspondence")
        elif self.correspond.policy != "none":
            raise ValueError("ordinary roles must not add unused correspondence")

    @property
    def description_bits(self) -> int:
        feature_bits = len(self.correspond.features)
        fixed_bits = 10 if self.canvas.mode == "fixed" else 0
        return 34 + feature_bits + 2 * sum(self.canvas.padding) + fixed_bits

    @property
    def functional_trace(self) -> tuple[str, ...]:
        features = "+".join(self.correspond.features) or "none"
        return (
            (
                "node:parse:objects:"
                f"bg={self.parse.background}:c={self.parse.connectivity}:"
                f"group={self.parse.grouping}"
            ),
            (
                "node:correspond:"
                f"policy={self.correspond.policy}:features={features}:"
                f"d4={int(self.correspond.d4_invariant)}"
            ),
            f"node:select:role={self.select.role}",
            (
                "node:operate:"
                f"op={self.operate.operator}:transform={self.operate.transform}:"
                f"axis={self.operate.axis}:spacing={self.operate.spacing}:"
                f"repeat={self.operate.repeat_rule}"
            ),
            (
                "node:canvas:"
                f"mode={self.canvas.mode}:bg={self.canvas.background}:"
                f"padding={','.join(str(item) for item in self.canvas.padding)}:"
                f"size={self.canvas.height}x{self.canvas.width}"
            ),
            (
                "node:render:"
                f"mode={self.render.mode}:conflict={self.render.conflict_policy}"
            ),
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "object_code_dsl_version": SCENE_PIPELINE_DSL_VERSION,
            "kind": "scene_pipeline",
            "scene_ast_version": SCENE_AST_VERSION,
            "ast": {
                "parse": self.parse.to_json_dict(),
                "correspond": self.correspond.to_json_dict(),
                "select": self.select.to_json_dict(),
                "operate": self.operate.to_json_dict(),
                "canvas": self.canvas.to_json_dict(),
                "render": self.render.to_json_dict(),
            },
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "ScenePipelineProgram":
        expected = {
            "object_code_dsl_version",
            "kind",
            "scene_ast_version",
            "ast",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("scene pipeline has missing or unknown fields")
        if payload["object_code_dsl_version"] != SCENE_PIPELINE_DSL_VERSION:
            raise ValueError("unsupported scene pipeline DSL version")
        if payload["scene_ast_version"] != SCENE_AST_VERSION:
            raise ValueError("unsupported scene AST version")
        if payload["kind"] != "scene_pipeline":
            raise ValueError("program kind is not scene_pipeline")
        ast = payload["ast"]
        ast_expected = {"parse", "correspond", "select", "operate", "canvas", "render"}
        if not isinstance(ast, Mapping) or set(ast) != ast_expected:
            raise ValueError("scene AST has missing or unknown nodes")
        return cls(
            ParseObjectsNode.from_json_dict(ast["parse"]),
            CorrespondObjectsNode.from_json_dict(ast["correspond"]),
            SelectObjectsNode.from_json_dict(ast["select"]),
            ObjectOperationNode.from_json_dict(ast["operate"]),
            CanvasNode.from_json_dict(ast["canvas"]),
            RenderObjectsNode.from_json_dict(ast["render"]),
        )


@dataclass(frozen=True, slots=True)
class ObjectCorrespondence:
    class_id: str
    signature: tuple[object, ...]
    object_indices: tuple[int, ...]


def _relative_position_signature(
    scene: SceneGraph, object_: SceneObject
) -> tuple[object, ...]:
    center_row_numerator = object_.top + object_.bottom
    center_column_numerator = object_.left + object_.right

    def band(numerator: int, length: int) -> int:
        # Compare 3 * center against two times the canvas length without floats.
        scaled = 3 * numerator
        if scaled < 2 * length:
            return 0
        if scaled > 4 * length:
            return 2
        return 1

    return (
        band(center_row_numerator, scene.height),
        band(center_column_numerator, scene.width),
        object_.touches_border,
    )


def object_correspondences(
    scene: SceneGraph, node: CorrespondObjectsNode
) -> tuple[ObjectCorrespondence, ...]:
    """Group objects using explicit shape/size/topology/position features."""

    if node.policy == "none":
        return tuple(
            ObjectCorrespondence(
                item.object_id,
                ("identity", item.object_id),
                (item.index,),
            )
            for item in scene.objects
        )
    groups: dict[tuple[object, ...], list[int]] = {}
    for object_ in scene.objects:
        signature: list[object] = []
        for feature in node.features:
            if feature == "shape":
                signature.append(
                    object_.canonical_d4_shape
                    if node.d4_invariant
                    else object_.normalized_shape
                )
            elif feature == "size":
                dimensions = (
                    tuple(sorted((object_.height, object_.width)))
                    if node.d4_invariant
                    else (object_.height, object_.width)
                )
                signature.append((object_.area, *dimensions))
            elif feature == "topology":
                signature.append(
                    (object_.hole_count, len(object_.colors), object_.touches_border)
                )
            elif feature == "relative_position":
                signature.append(_relative_position_signature(scene, object_))
            else:  # defensive: construction already validates features
                raise ValueError("unknown correspondence feature")
        groups.setdefault(tuple(signature), []).append(object_.index)
    return tuple(
        ObjectCorrespondence(
            _content_id(
                {
                    "signature": signature,
                    "objects": indices,
                }
            ),
            signature,
            tuple(indices),
        )
        for signature, indices in sorted(
            groups.items(), key=lambda item: (_canonical_json(item[0]), item[1])
        )
    )


def _role_key(object_: SceneObject, role: str) -> tuple[object, ...]:
    location = (object_.top, object_.left, object_.index)
    keys: dict[str, tuple[object, ...]] = {
        "largest_area": (-object_.area, *location),
        "smallest_area": (object_.area, *location),
        "largest_bbox": (-(object_.height * object_.width), *location),
        "smallest_bbox": (object_.height * object_.width, *location),
        "topmost": (object_.top, object_.left, object_.index),
        "bottommost": (-object_.bottom, object_.left, object_.index),
        "leftmost": (object_.left, object_.top, object_.index),
        "rightmost": (-object_.right, object_.top, object_.index),
        "most_holes": (-object_.hole_count, *location),
        "least_holes": (object_.hole_count, *location),
    }
    try:
        return keys[role]
    except KeyError as exc:
        raise ValueError("role does not have an object-local ordering") from exc


def _select_objects(
    scene: SceneGraph,
    node: SelectObjectsNode,
    correspondences: Sequence[ObjectCorrespondence],
) -> tuple[SceneObject, ...]:
    if not scene.objects:
        return ()
    if node.role == "all":
        return scene.objects
    by_index = {item.index: item for item in scene.objects}
    if node.role == "unique_signature":
        indices = tuple(
            item.object_indices[0]
            for item in correspondences
            if len(item.object_indices) == 1
        )
        return () if not indices else (by_index[min(indices)],)
    if node.role == "most_peers":
        group = min(
            correspondences,
            key=lambda item: (-len(item.object_indices), item.object_indices),
        )
        return tuple(by_index[index] for index in group.object_indices)
    return (min(scene.objects, key=lambda item: _role_key(item, node.role)),)


@dataclass(frozen=True, slots=True)
class ExecutionTraceNode:
    node_id: str
    operator: str
    status: str
    details: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"ok", "invalid"}:
            raise ValueError("execution trace status must be ok or invalid")
        if tuple(sorted(self.details)) != self.details:
            raise ValueError("execution trace details must be canonically sorted")

    @classmethod
    def create(
        cls,
        node_id: str,
        operator: str,
        status: str,
        **details: object,
    ) -> "ExecutionTraceNode":
        return cls(node_id, operator, status, tuple(sorted(details.items())))


@dataclass(frozen=True, slots=True)
class ScenePipelineExecution:
    status: str
    output: Grid | None
    reason: str | None
    object_count: int
    correspondence_count: int
    node_trace: tuple[ExecutionTraceNode, ...]

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _object_crop(
    grid: Grid,
    object_: SceneObject,
    *,
    input_background: int,
) -> Grid:
    top, left, bottom, right = object_.bounding_box
    occupied = set(object_.cells)
    return as_grid(
        [
            [
                grid[row][column]
                if (row, column) in occupied
                else input_background
                for column in range(left, right + 1)
            ]
            for row in range(top, bottom + 1)
        ]
    )


def _replace_background(grid: Grid, source: int, target: int) -> Grid:
    if source == target:
        return grid
    return as_grid(
        [[target if cell == source else cell for cell in row] for row in grid]
    )


def _pack_grids(
    grids: Sequence[Grid],
    *,
    axis: str,
    spacing: int,
    background: int,
) -> Grid | None:
    if not grids:
        return None
    if axis == "row":
        height = max(len(grid) for grid in grids)
        width = sum(len(grid[0]) for grid in grids) + spacing * (len(grids) - 1)
        canvas = [[background for _ in range(width)] for _ in range(height)]
        offset = 0
        for grid in grids:
            for row, values in enumerate(grid):
                canvas[row][offset : offset + len(values)] = values
            offset += len(grid[0]) + spacing
    else:
        height = sum(len(grid) for grid in grids) + spacing * (len(grids) - 1)
        width = max(len(grid[0]) for grid in grids)
        canvas = [[background for _ in range(width)] for _ in range(height)]
        offset = 0
        for grid in grids:
            for row, values in enumerate(grid):
                canvas[offset + row][: len(values)] = values
            offset += len(grid) + spacing
    return as_grid(canvas)


def _fit_canvas(
    grid: Grid,
    canvas: CanvasNode,
    *,
    conflict_policy: str,
    source_height: int,
    source_width: int,
) -> Grid | None:
    if canvas.mode not in {"fixed", "input", "tight"}:
        return grid
    if canvas.mode == "fixed":
        assert canvas.height is not None and canvas.width is not None
        height, width = canvas.height, canvas.width
        top_offset = left_offset = 0
    elif canvas.mode == "input":
        height, width = source_height, source_width
        top_offset = left_offset = 0
    else:
        pad_top, pad_bottom, pad_left, pad_right = canvas.padding
        height = len(grid) + pad_top + pad_bottom
        width = len(grid[0]) + pad_left + pad_right
        top_offset, left_offset = pad_top, pad_left
    if len(grid) + top_offset > height or len(grid[0]) + left_offset > width:
        return None
    output = [
        [canvas.background for _ in range(width)]
        for _ in range(height)
    ]
    for row, values in enumerate(grid):
        for column, color in enumerate(values):
            target_row = row + top_offset
            target_column = column + left_offset
            observed = output[target_row][target_column]
            if (
                observed != canvas.background
                and color != canvas.background
                and observed != color
                and conflict_policy == "reject"
            ):
                return None
            if color != canvas.background or conflict_policy == "last":
                output[target_row][target_column] = color
    return as_grid(output)


def _correspondence_peer_count(
    selected: Sequence[SceneObject],
    correspondences: Sequence[ObjectCorrespondence],
) -> int:
    if not selected:
        return 0
    index = selected[0].index
    return next(
        (
            len(item.object_indices)
            for item in correspondences
            if index in item.object_indices
        ),
        1,
    )


def execute_scene_pipeline(
    program: ScenePipelineProgram,
    grid: Grid,
    *,
    precomputed_scene: SceneGraph | None = None,
) -> ScenePipelineExecution:
    """Replay a scene AST and retain a node-level execution trace."""

    normalized = as_grid(grid)
    trace: list[ExecutionTraceNode] = []
    try:
        scene = precomputed_scene
        if scene is None:
            scene = extract_scene_graph(
                normalized,
                background=program.parse.background,
                connectivity=program.parse.connectivity,
                grouping=program.parse.grouping,
            )
        elif (
            scene.height != len(normalized)
            or scene.width != len(normalized[0])
            or scene.background != program.parse.background
            or scene.connectivity != program.parse.connectivity
            or scene.grouping != program.parse.grouping
        ):
            raise ValueError("precomputed scene does not match the program contract")
        trace.append(
            ExecutionTraceNode.create(
                "parse",
                "parse_objects",
                "ok",
                object_count=len(scene.objects),
            )
        )
        if not scene.objects:
            trace.append(
                ExecutionTraceNode.create(
                    "select", "select_objects", "invalid", reason="no_scene_objects"
                )
            )
            return ScenePipelineExecution(
                "invalid", None, "no_scene_objects", 0, 0, tuple(trace)
            )

        correspondences = object_correspondences(scene, program.correspond)
        trace.append(
            ExecutionTraceNode.create(
                "correspond",
                "correspond_objects",
                "ok",
                class_count=len(correspondences),
                feature_count=len(program.correspond.features),
            )
        )
        selected = _select_objects(scene, program.select, correspondences)
        if not selected:
            trace.append(
                ExecutionTraceNode.create(
                    "select",
                    "select_objects",
                    "invalid",
                    reason="role_not_instantiated",
                )
            )
            return ScenePipelineExecution(
                "invalid",
                None,
                "role_not_instantiated",
                len(scene.objects),
                len(correspondences),
                tuple(trace),
            )
        trace.append(
            ExecutionTraceNode.create(
                "select",
                "select_objects",
                "ok",
                role=program.select.role,
                selected_count=len(selected),
            )
        )

        operator = program.operate.operator
        output: Grid | None
        if operator == "crop":
            cells = tuple(cell for object_ in selected for cell in object_.cells)
            top, left, bottom, right = _bounding_box(cells)
            pad_top, pad_bottom, pad_left, pad_right = program.canvas.padding
            top -= pad_top
            bottom += pad_bottom
            left -= pad_left
            right += pad_right
            if (
                top < 0
                or left < 0
                or bottom >= len(normalized)
                or right >= len(normalized[0])
            ):
                output = None
            elif program.render.mode == "source_crop":
                output = as_grid(
                    [row[left : right + 1] for row in normalized[top : bottom + 1]]
                )
            else:
                occupied = {cell for object_ in selected for cell in object_.cells}
                output = as_grid(
                    [
                        [
                            normalized[row][column]
                            if (row, column) in occupied
                            else program.parse.background
                            for column in range(left, right + 1)
                        ]
                        for row in range(top, bottom + 1)
                    ]
                )
            if output is not None:
                output = _transform_grid(output, program.operate.transform)
                output = _replace_background(
                    output, program.parse.background, program.canvas.background
                )
                output = _fit_canvas(
                    output,
                    program.canvas,
                    conflict_policy=program.render.conflict_policy,
                    source_height=len(normalized),
                    source_width=len(normalized[0]),
                )
        elif operator == "count":
            count = {
                "selected": len(selected),
                "scene_objects": len(scene.objects),
                "peers": _correspondence_peer_count(selected, correspondences),
            }[program.operate.repeat_rule]
            if count < 1:
                output = None
            elif program.canvas.mode == "count_line":
                if program.operate.axis == "row":
                    output = as_grid([[program.operate.output_color] * count])
                else:
                    output = as_grid([[program.operate.output_color] for _ in range(count)])
            else:
                assert program.canvas.height is not None and program.canvas.width is not None
                if count > program.canvas.height * program.canvas.width:
                    output = None
                else:
                    values = [program.operate.output_color] * count + [
                        program.canvas.background
                    ] * (program.canvas.height * program.canvas.width - count)
                    output = as_grid(
                        [
                            values[row * program.canvas.width : (row + 1) * program.canvas.width]
                            for row in range(program.canvas.height)
                        ]
                    )
            if output is not None:
                output = _transform_grid(output, program.operate.transform)
        else:
            object_grids = tuple(
                _transform_grid(
                    _replace_background(
                        _object_crop(
                            normalized,
                            object_,
                            input_background=program.parse.background,
                        ),
                        program.parse.background,
                        program.canvas.background,
                    ),
                    program.operate.transform,
                )
                for object_ in selected
            )
            if operator == "copy":
                repeat = (
                    len(scene.objects)
                    if program.operate.repeat_rule == "scene_objects"
                    else _correspondence_peer_count(selected, correspondences)
                )
                output = _pack_grids(
                    tuple(object_grids[0] for _ in range(repeat)),
                    axis=program.operate.axis,
                    spacing=program.operate.spacing,
                    background=program.canvas.background,
                )
            elif operator == "arrange":
                output = _pack_grids(
                    object_grids,
                    axis=program.operate.axis,
                    spacing=program.operate.spacing,
                    background=program.canvas.background,
                )
            else:
                height = max(len(item) for item in object_grids)
                width = max(len(item[0]) for item in object_grids)
                canvas = [
                    [program.canvas.background for _ in range(width)]
                    for _ in range(height)
                ]
                output = as_grid(canvas)
                for object_grid in object_grids:
                    mutable = [list(row) for row in output]
                    conflict = False
                    for row, values in enumerate(object_grid):
                        for column, color in enumerate(values):
                            if color == program.canvas.background:
                                continue
                            observed = mutable[row][column]
                            if (
                                observed != program.canvas.background
                                and observed != color
                                and program.render.conflict_policy == "reject"
                            ):
                                conflict = True
                                break
                            if (
                                observed == program.canvas.background
                                or program.render.conflict_policy == "last"
                            ):
                                mutable[row][column] = color
                        if conflict:
                            break
                    if conflict:
                        output = None
                        break
                    output = as_grid(mutable)
            if output is not None:
                output = _fit_canvas(
                    output,
                    program.canvas,
                    conflict_policy=program.render.conflict_policy,
                    source_height=len(normalized),
                    source_width=len(normalized[0]),
                )

        if output is None:
            trace.append(
                ExecutionTraceNode.create(
                    "operate",
                    operator,
                    "invalid",
                    reason="operation_or_canvas_invalid",
                )
            )
            return ScenePipelineExecution(
                "invalid",
                None,
                "operation_or_canvas_invalid",
                len(scene.objects),
                len(correspondences),
                tuple(trace),
            )
        trace.append(
            ExecutionTraceNode.create(
                "operate", operator, "ok", selected_count=len(selected)
            )
        )
        trace.append(
            ExecutionTraceNode.create(
                "canvas",
                "infer_canvas",
                "ok",
                height=len(output),
                mode=program.canvas.mode,
                width=len(output[0]),
            )
        )
        trace.append(
            ExecutionTraceNode.create(
                "render", "render_objects", "ok", mode=program.render.mode
            )
        )
        return ScenePipelineExecution(
            "ok",
            output,
            None,
            len(scene.objects),
            len(correspondences),
            tuple(trace),
        )
    except Exception:
        trace.append(
            ExecutionTraceNode.create(
                "internal", "scene_pipeline", "invalid", reason="internal_error"
            )
        )
        return ScenePipelineExecution(
            "invalid", None, "internal_error", 0, 0, tuple(trace)
        )


def _modal_color(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda color: (-counts[color], color))


def _scene_backgrounds(task: BlindTask, *, limit: int = 3) -> tuple[int, ...]:
    inputs = tuple(pair.input for pair in task.train) + task.test_inputs
    modal = Counter(_modal_color(grid) for grid in inputs)
    all_colors = Counter(cell for grid in inputs for row in grid for cell in row)
    return tuple(
        sorted(
            all_colors,
            key=lambda color: (-modal[color], -all_colors[color], color != 0, color),
        )[:limit]
    )


def _scene_output_backgrounds(
    task: BlindTask, *, limit: int = 3
) -> tuple[int, ...]:
    outputs = tuple(pair.output for pair in task.train)
    assert all(output is not None for output in outputs)
    modal = Counter(_modal_color(output) for output in outputs if output is not None)
    colors = Counter(
        cell
        for output in outputs
        if output is not None
        for row in output
        for cell in row
    )
    return tuple(
        sorted(colors, key=lambda color: (-modal[color], -colors[color], color))[:limit]
    )


def _parse_candidates(task: BlindTask) -> tuple[ParseObjectsNode, ...]:
    return tuple(
        ParseObjectsNode(background, connectivity, grouping)
        for background in _scene_backgrounds(task)
        for grouping in SCENE_GROUPINGS
        for connectivity in ((4,) if grouping == "color_groups" else (4, 8))
    )


def _correspondence_for_role(
    role: str,
) -> tuple[CorrespondObjectsNode, ...]:
    if role not in {"unique_signature", "most_peers"}:
        return (CorrespondObjectsNode(),)
    feature_sets = (
        ("shape",),
        ("size",),
        ("topology",),
        ("shape", "size", "topology"),
        ("relative_position", "shape", "size", "topology"),
    )
    return tuple(
        CorrespondObjectsNode("equivalence", tuple(sorted(features)), invariant)
        for features in feature_sets
        for invariant in (False, True)
    )


def _selected_bbox_shape(
    grid: Grid,
    parse: ParseObjectsNode,
    correspond: CorrespondObjectsNode,
    select: SelectObjectsNode,
) -> tuple[int, int] | None:
    scene = extract_scene_graph(
        grid,
        background=parse.background,
        connectivity=parse.connectivity,
        grouping=parse.grouping,
    )
    selected = _select_objects(
        scene, select, object_correspondences(scene, correspond)
    )
    if not selected:
        return None
    cells = tuple(cell for item in selected for cell in item.cells)
    top, left, bottom, right = _bounding_box(cells)
    return bottom - top + 1, right - left + 1


def _crop_programs(task: BlindTask) -> list[ScenePipelineProgram]:
    programs: list[ScenePipelineProgram] = []
    output_backgrounds = _scene_output_backgrounds(task)
    first_output = task.train[0].output
    assert first_output is not None
    output_height, output_width = len(first_output), len(first_output[0])
    for parse in _parse_candidates(task):
        for role in SCENE_SELECTORS:
            select = SelectObjectsNode(role)
            for correspond in _correspondence_for_role(role):
                shape = _selected_bbox_shape(
                    task.train[0].input, parse, correspond, select
                )
                if shape is None:
                    continue
                selected_height, selected_width = shape
                for transform in SCENE_D4_TRANSFORMS:
                    if transform in {
                        "rotate90",
                        "rotate270",
                        "transpose",
                        "anti_transpose",
                    }:
                        required_height, required_width = output_width, output_height
                    else:
                        required_height, required_width = output_height, output_width
                    row_padding = required_height - selected_height
                    column_padding = required_width - selected_width
                    if not 0 <= row_padding <= 4 or not 0 <= column_padding <= 4:
                        continue
                    paddings = tuple(
                        (top, row_padding - top, left, column_padding - left)
                        for top in range(row_padding + 1)
                        for left in range(column_padding + 1)
                        if max(
                            top,
                            row_padding - top,
                            left,
                            column_padding - left,
                        )
                        <= 2
                    )
                    for padding in paddings:
                        for background in output_backgrounds:
                            for render_mode in ("source_crop", "selected_only"):
                                programs.append(
                                    ScenePipelineProgram(
                                        parse,
                                        correspond,
                                        select,
                                        ObjectOperationNode("crop", transform),
                                        CanvasNode("bbox", background, padding),
                                        RenderObjectsNode(render_mode),
                                    )
                                )
    return programs


def _count_programs(task: BlindTask) -> list[ScenePipelineProgram]:
    programs: list[ScenePipelineProgram] = []
    output_backgrounds = _scene_output_backgrounds(task)
    output_colors = tuple(
        sorted(
            {
                cell
                for pair in task.train
                for row in (pair.output or ())
                for cell in row
            }
        )
    )
    for parse in _parse_candidates(task):
        for axis in ("row", "column"):
            for repeat_rule in ("selected", "scene_objects"):
                for output_color in output_colors:
                    for background in output_backgrounds:
                        if output_color == background:
                            continue
                        programs.append(
                            ScenePipelineProgram(
                                parse,
                                CorrespondObjectsNode(),
                                SelectObjectsNode("all"),
                                ObjectOperationNode(
                                    "count",
                                    "identity",
                                    axis,
                                    0,
                                    repeat_rule,
                                    output_color,
                                ),
                                CanvasNode("count_line", background),
                                RenderObjectsNode("solid"),
                            )
                        )
    return programs


def _composition_programs(task: BlindTask) -> list[ScenePipelineProgram]:
    programs: list[ScenePipelineProgram] = []
    for parse in _parse_candidates(task):
        for background in _scene_output_backgrounds(task):
            for axis in ("row", "column"):
                for spacing in (0, 1):
                    programs.append(
                        ScenePipelineProgram(
                            parse,
                            CorrespondObjectsNode(),
                            SelectObjectsNode("all"),
                            ObjectOperationNode(
                                "arrange", "identity", axis, spacing
                            ),
                            CanvasNode("tight", background),
                            RenderObjectsNode("objects"),
                        )
                    )
            for conflict_policy in ("reject", "first", "last"):
                programs.append(
                    ScenePipelineProgram(
                        parse,
                        CorrespondObjectsNode(),
                        SelectObjectsNode("all"),
                        ObjectOperationNode("compose"),
                        CanvasNode("tight", background),
                        RenderObjectsNode("objects", conflict_policy),
                    )
                )
            for role in (
                "largest_area",
                "smallest_area",
                "largest_bbox",
                "smallest_bbox",
            ):
                for axis in ("row", "column"):
                    for repeat_rule in ("scene_objects",):
                        programs.append(
                            ScenePipelineProgram(
                                parse,
                                CorrespondObjectsNode(),
                                SelectObjectsNode(role),
                                ObjectOperationNode(
                                    "copy",
                                    "identity",
                                    axis,
                                    0,
                                    repeat_rule,
                                ),
                                CanvasNode("tight", background),
                                RenderObjectsNode("objects"),
                            )
                        )
    return programs


def enumerate_scene_pipeline_programs(
    task: BlindTask,
) -> tuple[ScenePipelineProgram, ...]:
    """Enumerate a deterministic bounded scene-AST grammar."""

    if not isinstance(task, BlindTask):
        raise TypeError("scene pipeline synthesis accepts BlindTask only")
    programs = [
        *_crop_programs(task),
        *_count_programs(task),
        *_composition_programs(task),
    ]
    unique = {
        _canonical_json(program.to_json_dict()): program for program in programs
    }
    return tuple(unique[key] for key in sorted(unique))
