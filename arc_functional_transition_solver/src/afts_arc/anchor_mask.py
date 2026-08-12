"""Finite anchor-to-mask programs for one typed spatial color delta."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from .grid import Grid, as_grid
from .hybrid.scene_graph import SceneGraph


ANCHOR_MASK_NODE_ID = "post_anchor_mask_delta"
ANCHOR_MASK_NODE_TYPE = "anchor_rasterized_delta"
ANCHOR_COLORS = tuple(range(10))

_STENCIL_OFFSETS = {
    "north": ((-1, 0),),
    "south": ((1, 0),),
    "west": ((0, -1),),
    "east": ((0, 1),),
    "block_nw": ((-1, -1), (-1, 0), (0, -1), (0, 0)),
    "block_ne": ((-1, 0), (-1, 1), (0, 0), (0, 1)),
    "block_sw": ((0, -1), (0, 0), (1, -1), (1, 0)),
    "block_se": ((0, 0), (0, 1), (1, 0), (1, 1)),
}
_MASK_PARAMETERS = {
    "stencil": tuple(_STENCIL_OFFSETS),
    "axis_project": ("row", "column"),
    "bbox_fill": ("inset0", "inset1"),
}
ANCHOR_MASK_SPECS = tuple(
    (mask_kind, parameter)
    for mask_kind in ("stencil", "axis_project", "bbox_fill")
    for parameter in _MASK_PARAMETERS[mask_kind]
)
ANCHOR_MASK_PROGRAMS_PER_DELTA = len(ANCHOR_COLORS) * len(ANCHOR_MASK_SPECS)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _content_id(value: object, *, length: int = 24) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()[:length]


def _arc_color(value: object, *, field_name: str) -> int:
    if type(value) is not int or not 0 <= value <= 9:
        raise ValueError(f"{field_name} must be an ARC color")
    return value


@dataclass(frozen=True, slots=True)
class AnchorRasterizedDeltaNode:
    """Apply one color delta on a mask rasterized from persistent anchors."""

    anchor_color: int
    mask_kind: str
    mask_parameter: str
    source_color: int
    target_color: int

    def __post_init__(self) -> None:
        _arc_color(self.anchor_color, field_name="anchor color")
        _arc_color(self.source_color, field_name="delta source")
        _arc_color(self.target_color, field_name="delta target")
        if self.source_color == self.target_color:
            raise ValueError("delta source and target must differ")
        if self.mask_kind not in _MASK_PARAMETERS:
            raise ValueError("unknown anchor-mask rasterizer")
        if self.mask_parameter not in _MASK_PARAMETERS[self.mask_kind]:
            raise ValueError("mask parameter is incompatible with its rasterizer")

    @property
    def node_id(self) -> str:
        return _content_id(self.to_json_dict())

    @property
    def sort_key(self) -> tuple[object, ...]:
        return (
            self.anchor_color,
            ANCHOR_MASK_SPECS.index((self.mask_kind, self.mask_parameter)),
            self.source_color,
            self.target_color,
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "op": ANCHOR_MASK_NODE_TYPE,
            "anchor_color": self.anchor_color,
            "mask_kind": self.mask_kind,
            "mask_parameter": self.mask_parameter,
            "source_color": self.source_color,
            "target_color": self.target_color,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "AnchorRasterizedDeltaNode":
        expected = {
            "op",
            "anchor_color",
            "mask_kind",
            "mask_parameter",
            "source_color",
            "target_color",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("anchor-mask node has missing or unknown fields")
        if payload["op"] != ANCHOR_MASK_NODE_TYPE:
            raise ValueError("anchor-mask node has the wrong operator")
        return cls(
            payload["anchor_color"],
            payload["mask_kind"],
            payload["mask_parameter"],
            payload["source_color"],
            payload["target_color"],
        )


def anchor_mask_domain(
    source_color: int,
    target_color: int,
) -> tuple[AnchorRasterizedDeltaNode, ...]:
    """Return the complete finite mask language for one certified delta."""

    _arc_color(source_color, field_name="delta source")
    _arc_color(target_color, field_name="delta target")
    if source_color == target_color:
        raise ValueError("delta source and target must differ")
    return tuple(
        AnchorRasterizedDeltaNode(
            anchor_color,
            mask_kind,
            mask_parameter,
            source_color,
            target_color,
        )
        for anchor_color in ANCHOR_COLORS
        for mask_kind, mask_parameter in ANCHOR_MASK_SPECS
    )


def _validate_context(
    input_grid: Grid,
    parent_grid: Grid,
    scene: SceneGraph,
) -> tuple[Grid, Grid]:
    normalized_input = as_grid(input_grid)
    normalized_parent = as_grid(parent_grid)
    input_shape = (len(normalized_input), len(normalized_input[0]))
    parent_shape = (len(normalized_parent), len(normalized_parent[0]))
    if input_shape != parent_shape:
        raise ValueError("anchor-mask input and parent canvases must match")
    if (scene.height, scene.width) != input_shape:
        raise ValueError("anchor-mask scene and input canvases must match")
    return normalized_input, normalized_parent


def _anchor_coordinates(
    node: AnchorRasterizedDeltaNode,
    input_grid: Grid,
    scene: SceneGraph,
) -> tuple[tuple[int, int], ...]:
    if node.anchor_color == scene.background:
        return ()
    anchors = tuple(
        (row, column)
        for row in range(len(input_grid))
        for column in range(len(input_grid[0]))
        if input_grid[row][column] == node.anchor_color
    )
    scene_cells = {
        cell
        for scene_object in scene.objects
        for cell in scene_object.cells
    }
    if any(anchor not in scene_cells for anchor in anchors):
        raise ValueError("anchor color is not represented by the persistent scene")
    return anchors


def rasterize_anchor_mask(
    node: AnchorRasterizedDeltaNode,
    input_grid: Grid,
    parent_grid: Grid,
    scene: SceneGraph,
) -> tuple[tuple[int, int], ...]:
    """Rasterize one mask without inspecting a demonstration target."""

    if not isinstance(node, AnchorRasterizedDeltaNode):
        raise TypeError("anchor-mask rasterization requires its typed node")
    normalized_input, normalized_parent = _validate_context(
        input_grid,
        parent_grid,
        scene,
    )
    height, width = len(normalized_parent), len(normalized_parent[0])
    anchors = _anchor_coordinates(node, normalized_input, scene)
    coordinates: set[tuple[int, int]] = set()

    if node.mask_kind == "stencil":
        offsets = _STENCIL_OFFSETS[node.mask_parameter]
        coordinates.update(
            (row + dr, column + dc)
            for row, column in anchors
            for dr, dc in offsets
            if 0 <= row + dr < height and 0 <= column + dc < width
        )
    elif node.mask_kind == "axis_project":
        if node.mask_parameter == "row":
            coordinates.update(
                (row, column)
                for row in {item[0] for item in anchors}
                for column in range(width)
            )
        else:
            coordinates.update(
                (row, column)
                for column in {item[1] for item in anchors}
                for row in range(height)
            )
    else:
        if anchors:
            inset = 0 if node.mask_parameter == "inset0" else 1
            top = min(row for row, _ in anchors) + inset
            left = min(column for _, column in anchors) + inset
            bottom = max(row for row, _ in anchors) - inset
            right = max(column for _, column in anchors) - inset
            if top <= bottom and left <= right:
                coordinates.update(
                    (row, column)
                    for row in range(top, bottom + 1)
                    for column in range(left, right + 1)
                )
    return tuple(sorted(coordinates))


def effective_anchor_mask(
    node: AnchorRasterizedDeltaNode,
    input_grid: Grid,
    parent_grid: Grid,
    scene: SceneGraph,
) -> tuple[tuple[int, int], ...]:
    """Return mask cells on which the node can change the parent grid."""

    normalized_parent = as_grid(parent_grid)
    return tuple(
        coordinate
        for coordinate in rasterize_anchor_mask(
            node,
            input_grid,
            normalized_parent,
            scene,
        )
        if normalized_parent[coordinate[0]][coordinate[1]] == node.source_color
    )


def execute_anchor_rasterized_delta(
    node: AnchorRasterizedDeltaNode,
    input_grid: Grid,
    parent_grid: Grid,
    scene: SceneGraph,
) -> Grid:
    """Execute one deterministic proof-carrying support program."""

    normalized_parent = as_grid(parent_grid)
    changed = set(effective_anchor_mask(node, input_grid, normalized_parent, scene))
    return as_grid(
        [
            [
                node.target_color if (row, column) in changed else color
                for column, color in enumerate(values)
            ]
            for row, values in enumerate(normalized_parent)
        ]
    )
