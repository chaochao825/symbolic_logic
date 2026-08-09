"""Opt-in relational delta programs with typed, topology-checked edits.

The module is intentionally separate from the default object/code provider so
that adding this experimental representation cannot change legacy enumeration,
candidate identities, checkpoints, or published results.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations

from .blind import BlindTask
from .grid import Grid, as_grid
from .hybrid.scene_graph import ExecutionTraceNode, SceneObject, extract_scene_graph
from .hybrid.types import canonical_json


RELATIONAL_DELTA_DSL_VERSION = "afts-relational-delta-dsl/v0.2"
RELATIONAL_DELTA_CERTIFICATE_SCHEMA = "afts.relational-delta-certificate/v0.2"
RELATIONAL_DELTA_OPERATIONS = (
    "add",
    "erase_add",
    "recolor_component",
    "fill_relation_region",
)
RELATIONAL_DELTA_RELATIONS = (
    "actor_area_to_row_interiors",
    "component_to_horizontal_border",
    "support_bbox_centers",
    "support_enclosed_regions",
)
RELATIONAL_DELTA_TARGETS: dict[str, tuple[str, str]] = {
    "actor_area_to_row_interiors": ("region", "unique_actor_area_match"),
    "component_to_horizontal_border": ("component", "all"),
    "support_bbox_centers": ("coordinate", "all"),
    "support_enclosed_regions": ("region", "all"),
}
RELATIONAL_DELTA_NEAR_MISS_PRECISION = 0.5
RELATIONAL_DELTA_NEAR_MISS_RECALL = 0.5

Coordinate = tuple[int, int]
ColoredCoordinate = tuple[int, int, int]


def _arc_color(value: object, *, field_name: str) -> int:
    if type(value) is not int or not 0 <= value <= 9:
        raise ValueError(f"{field_name} must be an ARC color")
    return value


@dataclass(frozen=True, slots=True)
class DeltaParseNode:
    connectivity: int
    grouping: str = "monochrome_components"

    def __post_init__(self) -> None:
        if type(self.connectivity) is not int or self.connectivity not in {4, 8}:
            raise ValueError("delta parse connectivity must be 4 or 8")
        if self.grouping != "monochrome_components":
            raise ValueError("v0.2 delta programs require monochrome components")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "op": "parse_objects",
            "background": "modal",
            "connectivity": self.connectivity,
            "grouping": self.grouping,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "DeltaParseNode":
        expected = {"op", "background", "connectivity", "grouping"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("delta parse node has missing or unknown fields")
        if payload["op"] != "parse_objects" or payload["background"] != "modal":
            raise ValueError("unsupported delta parse policy")
        return cls(payload["connectivity"], payload["grouping"])


@dataclass(frozen=True, slots=True)
class RoleAssignmentNode:
    actor_role: str
    support_role: str

    def __post_init__(self) -> None:
        if self.actor_role not in {"minority_foreground", "only_foreground"}:
            raise ValueError("unknown delta actor role")
        if self.support_role not in {"majority_foreground", "canvas"}:
            raise ValueError("unknown delta support role")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "op": "assign_roles",
            "actor_role": self.actor_role,
            "support_role": self.support_role,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "RoleAssignmentNode":
        expected = {"op", "actor_role", "support_role"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("role node has missing or unknown fields")
        if payload["op"] != "assign_roles":
            raise ValueError("delta role node has the wrong operator")
        return cls(payload["actor_role"], payload["support_role"])


@dataclass(frozen=True, slots=True)
class NaryRelationNode:
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in RELATIONAL_DELTA_RELATIONS:
            raise ValueError("unknown n-ary relation kind")

    def to_json_dict(self) -> dict[str, object]:
        return {"op": "correspond_nary", "kind": self.kind}

    @classmethod
    def from_json_dict(cls, payload: object) -> "NaryRelationNode":
        expected = {"op", "kind"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("n-ary relation node has missing or unknown fields")
        if payload["op"] != "correspond_nary":
            raise ValueError("delta relation node has the wrong operator")
        return cls(payload["kind"])


@dataclass(frozen=True, slots=True)
class DeltaTargetNode:
    kind: str
    selector: str

    def __post_init__(self) -> None:
        if self.kind not in {"component", "coordinate", "region"}:
            raise ValueError("unknown delta target kind")
        if self.selector not in {"all", "unique_actor_area_match"}:
            raise ValueError("unknown delta target selector")

    def to_json_dict(self) -> dict[str, object]:
        return {"op": "select_target", "kind": self.kind, "selector": self.selector}

    @classmethod
    def from_json_dict(cls, payload: object) -> "DeltaTargetNode":
        expected = {"op", "kind", "selector"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("target node has missing or unknown fields")
        if payload["op"] != "select_target":
            raise ValueError("delta target node has the wrong operator")
        return cls(payload["kind"], payload["selector"])


@dataclass(frozen=True, slots=True)
class TypedDeltaMaskNode:
    erase: str
    write: str = "target_cells"

    def __post_init__(self) -> None:
        if self.erase not in {"none", "actor_cells"}:
            raise ValueError("unknown typed erase mask")
        if self.write != "target_cells":
            raise ValueError("v0.2 writes must use complete target cells")

    def to_json_dict(self) -> dict[str, object]:
        return {"op": "build_delta_mask", "erase": self.erase, "write": self.write}

    @classmethod
    def from_json_dict(cls, payload: object) -> "TypedDeltaMaskNode":
        expected = {"op", "erase", "write"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("delta-mask node has missing or unknown fields")
        if payload["op"] != "build_delta_mask":
            raise ValueError("delta-mask node has the wrong operator")
        return cls(payload["erase"], payload["write"])


@dataclass(frozen=True, slots=True)
class DeltaRenderNode:
    operation: str
    color_policy: str
    palette: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.operation not in RELATIONAL_DELTA_OPERATIONS:
            raise ValueError("unknown relational delta operation")
        if self.color_policy not in {"actor", "constant", "target_side_palette"}:
            raise ValueError("unknown relational delta color policy")
        for index, color in enumerate(self.palette):
            _arc_color(color, field_name=f"render palette {index}")
        required_palette_size = {
            "actor": 0,
            "constant": 1,
            "target_side_palette": 2,
        }[self.color_policy]
        if len(self.palette) != required_palette_size:
            raise ValueError("render palette size does not match color policy")
        if self.color_policy == "target_side_palette" and len(set(self.palette)) != 2:
            raise ValueError("left and right target colors must differ")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "op": "render_delta",
            "operation": self.operation,
            "color_policy": self.color_policy,
            "palette": list(self.palette),
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "DeltaRenderNode":
        expected = {"op", "operation", "color_policy", "palette"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("delta render node has missing or unknown fields")
        if payload["op"] != "render_delta":
            raise ValueError("delta render node has the wrong operator")
        palette = payload["palette"]
        if not isinstance(palette, list):
            raise TypeError("delta render palette must be a list")
        return cls(payload["operation"], payload["color_policy"], tuple(palette))


@dataclass(frozen=True, slots=True)
class RelationalDeltaProgram:
    parse: DeltaParseNode
    roles: RoleAssignmentNode
    relation: NaryRelationNode
    target: DeltaTargetNode
    mask: TypedDeltaMaskNode
    render: DeltaRenderNode

    def __post_init__(self) -> None:
        required_target = RELATIONAL_DELTA_TARGETS[self.relation.kind]
        if (self.target.kind, self.target.selector) != required_target:
            raise ValueError("target type does not match relation semantics")
        canvas_relation = self.relation.kind == "component_to_horizontal_border"
        if canvas_relation != (self.roles.support_role == "canvas"):
            raise ValueError("canvas support is exclusive to border correspondence")
        if self.render.operation == "add":
            if self.mask.erase != "none":
                raise ValueError("add cannot erase source cells")
        elif self.render.operation == "erase_add":
            if self.mask.erase != "actor_cells" or self.target.kind != "region":
                raise ValueError("erase_add requires actor erasure and a region target")
        elif self.render.operation == "recolor_component":
            if (
                self.mask.erase != "actor_cells"
                or self.target.kind != "component"
                or self.render.color_policy != "target_side_palette"
            ):
                raise ValueError("recolor_component requires component relocation and palette")
        elif self.mask.erase != "none" or self.target.kind != "region":
            raise ValueError("fill_relation_region requires a non-erasing region target")
        if self.render.operation != "recolor_component" and (
            self.render.color_policy == "target_side_palette"
        ):
            raise ValueError("side palette is exclusive to recolor_component")

    @property
    def description_bits(self) -> int:
        return 44 + 4 * len(self.render.palette)

    @property
    def functional_trace(self) -> tuple[str, ...]:
        return (
            f"node:parse:modal:c={self.parse.connectivity}:group={self.parse.grouping}",
            (
                "node:roles:"
                f"actor={self.roles.actor_role}:support={self.roles.support_role}"
            ),
            f"node:correspond:nary:kind={self.relation.kind}",
            f"node:target:kind={self.target.kind}:selector={self.target.selector}",
            f"node:mask:erase={self.mask.erase}:write={self.mask.write}",
            (
                "node:render:"
                f"op={self.render.operation}:color={self.render.color_policy}:"
                f"palette={','.join(str(color) for color in self.render.palette)}"
            ),
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "relational_delta_dsl_version": RELATIONAL_DELTA_DSL_VERSION,
            "kind": "relational_delta_program",
            "ast": {
                "parse": self.parse.to_json_dict(),
                "roles": self.roles.to_json_dict(),
                "relation": self.relation.to_json_dict(),
                "target": self.target.to_json_dict(),
                "mask": self.mask.to_json_dict(),
                "render": self.render.to_json_dict(),
            },
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "RelationalDeltaProgram":
        expected = {"relational_delta_dsl_version", "kind", "ast"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("relational delta program has missing or unknown fields")
        if payload["relational_delta_dsl_version"] != RELATIONAL_DELTA_DSL_VERSION:
            raise ValueError("unsupported relational delta DSL version")
        if payload["kind"] != "relational_delta_program":
            raise ValueError("program kind is not relational_delta_program")
        ast = payload["ast"]
        ast_expected = {"parse", "roles", "relation", "target", "mask", "render"}
        if not isinstance(ast, Mapping) or set(ast) != ast_expected:
            raise ValueError("relational delta AST has missing or unknown nodes")
        return cls(
            DeltaParseNode.from_json_dict(ast["parse"]),
            RoleAssignmentNode.from_json_dict(ast["roles"]),
            NaryRelationNode.from_json_dict(ast["relation"]),
            DeltaTargetNode.from_json_dict(ast["target"]),
            TypedDeltaMaskNode.from_json_dict(ast["mask"]),
            DeltaRenderNode.from_json_dict(ast["render"]),
        )


def relational_delta_program_id(program: RelationalDeltaProgram) -> str:
    return hashlib.sha256(
        canonical_json(program.to_json_dict()).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class RelationalCorrespondence:
    member_ids: tuple[str, ...]
    source_cells: tuple[Coordinate, ...]
    target_kind: str
    target_cells: tuple[Coordinate, ...]
    side: str | None = None

    def __post_init__(self) -> None:
        if len(self.member_ids) < 2:
            raise ValueError("relational correspondence must be at least binary")
        if self.target_kind not in {"component", "coordinate", "region"}:
            raise ValueError("correspondence has an unknown target kind")
        if tuple(sorted(set(self.source_cells))) != self.source_cells:
            raise ValueError("correspondence source cells must be unique and sorted")
        if tuple(sorted(set(self.target_cells))) != self.target_cells:
            raise ValueError("correspondence target cells must be unique and sorted")
        if not self.target_cells:
            raise ValueError("correspondence target must be non-empty")
        if self.side not in {None, "left", "right"}:
            raise ValueError("correspondence side must be left or right")


@dataclass(frozen=True, slots=True)
class RelationalDeltaExecution:
    status: str
    output: Grid | None
    reason: str | None
    background: int
    actor_color: int | None
    support_color: int | None
    correspondences: tuple[RelationalCorrespondence, ...]
    erase_mask: tuple[Coordinate, ...]
    write_mask: tuple[Coordinate, ...]
    writes: tuple[ColoredCoordinate, ...]
    topology_valid: bool
    node_trace: tuple[ExecutionTraceNode, ...]

    def __post_init__(self) -> None:
        if self.status not in {"ok", "invalid"}:
            raise ValueError("relational delta execution status is invalid")
        if (self.status == "ok") != (self.output is not None):
            raise ValueError("only successful delta executions carry output")
        if tuple(sorted(set(self.erase_mask))) != self.erase_mask:
            raise ValueError("erase mask must be unique and sorted")
        if tuple(sorted(set(self.write_mask))) != self.write_mask:
            raise ValueError("write mask must be unique and sorted")
        if tuple(sorted(self.writes)) != self.writes:
            raise ValueError("colored writes must be sorted")
        if tuple((row, column) for row, column, _ in self.writes) != self.write_mask:
            raise ValueError("write colors and write mask do not close")
        if self.status == "ok" and not self.topology_valid:
            raise ValueError("successful delta execution must satisfy topology")

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _modal_color(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda color: (-counts[color], color))


def _unique_extreme_color(
    counts: Counter[int], *, smallest: bool
) -> int | None:
    if not counts:
        return None
    target = min(counts.values()) if smallest else max(counts.values())
    colors = tuple(sorted(color for color, count in counts.items() if count == target))
    return colors[0] if len(colors) == 1 else None


def _assign_roles(
    program: RelationalDeltaProgram, grid: Grid, background: int
) -> tuple[int, int | None] | None:
    counts = Counter(
        cell for row in grid for cell in row if cell != background
    )
    if program.roles.actor_role == "only_foreground":
        actor = next(iter(counts)) if len(counts) == 1 else None
    else:
        actor = _unique_extreme_color(counts, smallest=True)
    if actor is None:
        return None
    if program.roles.support_role == "canvas":
        return actor, None
    support_counts = Counter(
        {color: count for color, count in counts.items() if color != actor}
    )
    support = _unique_extreme_color(support_counts, smallest=False)
    return None if support is None else (actor, support)


def _objects_with_color(
    objects: Sequence[SceneObject], color: int
) -> tuple[SceneObject, ...]:
    return tuple(item for item in objects if item.colors == (color,))


def _row_interior_region(
    object_: SceneObject, grid: Grid, background: int
) -> tuple[Coordinate, ...]:
    by_row: dict[int, list[int]] = {}
    for row, column in object_.cells:
        by_row.setdefault(row, []).append(column)
    cells = {
        (row, column)
        for row, columns in by_row.items()
        if len(columns) >= 2
        for column in range(min(columns) + 1, max(columns))
        if grid[row][column] == background
    }
    return tuple(sorted(cells))


def _normalized_shape(cells: Sequence[Coordinate]) -> tuple[Coordinate, ...]:
    top = min(row for row, _ in cells)
    left = min(column for _, column in cells)
    return tuple(sorted((row - top, column - left) for row, column in cells))


def _enclosed_regions(
    object_: SceneObject, grid: Grid, background: int
) -> tuple[tuple[Coordinate, ...], ...]:
    top, left, bottom, right = object_.bounding_box
    available = {
        (row, column)
        for row in range(top, bottom + 1)
        for column in range(left, right + 1)
        if grid[row][column] == background
    }
    regions = []
    while available:
        seed = min(available)
        available.remove(seed)
        frontier = [seed]
        cells = {seed}
        touches_box = False
        while frontier:
            row, column = frontier.pop()
            touches_box |= row in {top, bottom} or column in {left, right}
            for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                neighbor = (row + dr, column + dc)
                if neighbor in available:
                    available.remove(neighbor)
                    cells.add(neighbor)
                    frontier.append(neighbor)
        if not touches_box:
            regions.append(tuple(sorted(cells)))
    return tuple(sorted(regions))


def _instantiate_correspondences(
    program: RelationalDeltaProgram,
    grid: Grid,
    *,
    background: int,
    actor_color: int,
    support_color: int | None,
) -> tuple[RelationalCorrespondence, ...]:
    scene = extract_scene_graph(
        grid,
        background=background,
        connectivity=program.parse.connectivity,
        grouping=program.parse.grouping,
    )
    actors = _objects_with_color(scene.objects, actor_color)
    if not actors:
        return ()
    actor_cells = tuple(sorted(cell for item in actors for cell in item.cells))
    kind = program.relation.kind
    if kind == "component_to_horizontal_border":
        width = len(grid[0])
        correspondences = []
        for actor in actors:
            solid = actor.area == actor.height * actor.width
            side = "left" if solid or actor.hole_count > 0 else "right"
            shift = -actor.left if side == "left" else width - 1 - actor.right
            target = tuple(sorted((row, column + shift) for row, column in actor.cells))
            correspondences.append(
                RelationalCorrespondence(
                    (actor.object_id, "canvas:left", "canvas:right"),
                    tuple(sorted(actor.cells)),
                    "component",
                    target,
                    side,
                )
            )
        return tuple(correspondences)

    if support_color is None:
        return ()
    supports = _objects_with_color(scene.objects, support_color)
    if not supports:
        return ()
    if kind == "actor_area_to_row_interiors":
        matches = tuple(
            (support, region)
            for support in supports
            if (region := _row_interior_region(support, grid, background))
            and len(region) == len(actor_cells)
        )
        if len(matches) != 1:
            return ()
        support, region = matches[0]
        return (
            RelationalCorrespondence(
                (f"actor-color:{actor_color}", support.object_id),
                actor_cells,
                "region",
                region,
            ),
        )
    if kind == "support_bbox_centers":
        result = []
        for support in supports:
            row_sum = support.top + support.bottom
            column_sum = support.left + support.right
            if row_sum % 2 or column_sum % 2:
                continue
            center = (row_sum // 2, column_sum // 2)
            if grid[center[0]][center[1]] != background:
                continue
            result.append(
                RelationalCorrespondence(
                    (f"actor-color:{actor_color}", support.object_id),
                    actor_cells,
                    "coordinate",
                    (center,),
                )
            )
        return tuple(result)
    result = []
    for support in supports:
        for region_index, region in enumerate(
            _enclosed_regions(support, grid, background)
        ):
            result.append(
                RelationalCorrespondence(
                    (
                        f"actor-color:{actor_color}",
                        support.object_id,
                        f"hole:{region_index}",
                    ),
                    actor_cells,
                    "region",
                    region,
                )
            )
    return tuple(result)


def _invalid_execution(
    *,
    reason: str,
    background: int,
    actor_color: int | None,
    support_color: int | None,
    correspondences: Sequence[RelationalCorrespondence],
    trace: Sequence[ExecutionTraceNode],
) -> RelationalDeltaExecution:
    return RelationalDeltaExecution(
        "invalid",
        None,
        reason,
        background,
        actor_color,
        support_color,
        tuple(correspondences),
        (),
        (),
        (),
        False,
        tuple(trace),
    )


def execute_relational_delta(
    program: RelationalDeltaProgram, grid: Grid
) -> RelationalDeltaExecution:
    """Execute a typed delta AST without implicit fallback or partial writes."""

    normalized = as_grid(grid)
    background = _modal_color(normalized)
    trace = [
        ExecutionTraceNode.create(
            "parse",
            "parse_objects",
            "ok",
            background=background,
            connectivity=program.parse.connectivity,
        )
    ]
    roles = _assign_roles(program, normalized, background)
    if roles is None:
        trace.append(
            ExecutionTraceNode.create(
                "roles", "assign_roles", "invalid", reason="ambiguous_roles"
            )
        )
        return _invalid_execution(
            reason="ambiguous_roles",
            background=background,
            actor_color=None,
            support_color=None,
            correspondences=(),
            trace=trace,
        )
    actor_color, support_color = roles
    trace.append(
        ExecutionTraceNode.create(
            "roles",
            "assign_roles",
            "ok",
            actor_color=actor_color,
            support_color=support_color,
        )
    )
    correspondences = _instantiate_correspondences(
        program,
        normalized,
        background=background,
        actor_color=actor_color,
        support_color=support_color,
    )
    if not correspondences:
        trace.append(
            ExecutionTraceNode.create(
                "relation",
                "correspond_nary",
                "invalid",
                reason="no_unique_correspondence",
            )
        )
        return _invalid_execution(
            reason="no_unique_correspondence",
            background=background,
            actor_color=actor_color,
            support_color=support_color,
            correspondences=(),
            trace=trace,
        )
    trace.append(
        ExecutionTraceNode.create(
            "relation",
            "correspond_nary",
            "ok",
            correspondence_count=len(correspondences),
            maximum_arity=max(len(item.member_ids) for item in correspondences),
        )
    )
    trace.append(
        ExecutionTraceNode.create(
            "target",
            "select_target",
            "ok",
            kind=program.target.kind,
            target_count=len(correspondences),
        )
    )

    erase = (
        set(cell for item in correspondences for cell in item.source_cells)
        if program.mask.erase == "actor_cells"
        else set()
    )
    colored_writes: dict[Coordinate, int] = {}
    for item in correspondences:
        if program.render.color_policy == "actor":
            color = actor_color
        elif program.render.color_policy == "constant":
            color = program.render.palette[0]
        else:
            if item.side is None:
                return _invalid_execution(
                    reason="target_side_missing",
                    background=background,
                    actor_color=actor_color,
                    support_color=support_color,
                    correspondences=correspondences,
                    trace=trace,
                )
            color = program.render.palette[0 if item.side == "left" else 1]
        for coordinate in item.target_cells:
            incumbent = colored_writes.get(coordinate)
            if incumbent is not None and incumbent != color:
                return _invalid_execution(
                    reason="conflicting_target_colors",
                    background=background,
                    actor_color=actor_color,
                    support_color=support_color,
                    correspondences=correspondences,
                    trace=trace,
                )
            colored_writes[coordinate] = color

    write_mask = set(colored_writes)
    height, width = len(normalized), len(normalized[0])
    if any(
        not (0 <= row < height and 0 <= column < width)
        for row, column in erase | write_mask
    ):
        return _invalid_execution(
            reason="delta_out_of_bounds",
            background=background,
            actor_color=actor_color,
            support_color=support_color,
            correspondences=correspondences,
            trace=trace,
        )
    if any(
        normalized[row][column] != background and (row, column) not in erase
        for row, column in write_mask
    ):
        return _invalid_execution(
            reason="target_overwrites_preserved_object",
            background=background,
            actor_color=actor_color,
            support_color=support_color,
            correspondences=correspondences,
            trace=trace,
        )

    topology_valid = True
    if program.render.operation == "add":
        topology_valid = (
            not erase
            and write_mask
            == {cell for item in correspondences for cell in item.target_cells}
            and len(write_mask)
            == sum(len(item.target_cells) for item in correspondences)
        )
    elif program.render.operation == "erase_add":
        topology_valid = (
            bool(erase)
            and len(erase) == len(write_mask)
            and write_mask
            == {cell for item in correspondences for cell in item.target_cells}
        )
    elif program.render.operation == "recolor_component":
        topology_valid = all(
            _normalized_shape(item.source_cells)
            == _normalized_shape(item.target_cells)
            for item in correspondences
        ) and (
            len(erase)
            == len(write_mask)
            == sum(len(item.source_cells) for item in correspondences)
            == sum(len(item.target_cells) for item in correspondences)
        )
    else:
        topology_valid = (
            not erase
            and write_mask
            == {cell for item in correspondences for cell in item.target_cells}
            and len(write_mask)
            == sum(len(item.target_cells) for item in correspondences)
        )
    if not topology_valid:
        trace.append(
            ExecutionTraceNode.create(
                "mask", "build_delta_mask", "invalid", reason="topology_violation"
            )
        )
        return _invalid_execution(
            reason="topology_violation",
            background=background,
            actor_color=actor_color,
            support_color=support_color,
            correspondences=correspondences,
            trace=trace,
        )

    canvas = [list(row) for row in normalized]
    for row, column in erase:
        canvas[row][column] = background
    for (row, column), color in colored_writes.items():
        canvas[row][column] = color
    output = as_grid(canvas)
    if output == normalized:
        trace.append(
            ExecutionTraceNode.create(
                "render", "render_delta", "invalid", reason="empty_effective_delta"
            )
        )
        return _invalid_execution(
            reason="empty_effective_delta",
            background=background,
            actor_color=actor_color,
            support_color=support_color,
            correspondences=correspondences,
            trace=trace,
        )
    ordered_erase = tuple(sorted(erase))
    ordered_writes = tuple(
        (row, column, colored_writes[(row, column)])
        for row, column in sorted(write_mask)
    )
    trace.append(
        ExecutionTraceNode.create(
            "mask",
            "build_delta_mask",
            "ok",
            erase_count=len(ordered_erase),
            topology_valid=True,
            write_count=len(ordered_writes),
        )
    )
    trace.append(
        ExecutionTraceNode.create(
            "render",
            "render_delta",
            "ok",
            operation=program.render.operation,
        )
    )
    return RelationalDeltaExecution(
        "ok",
        output,
        None,
        background,
        actor_color,
        support_color,
        correspondences,
        ordered_erase,
        tuple((row, column) for row, column, _ in ordered_writes),
        ordered_writes,
        True,
        tuple(trace),
    )


def _positive_output_colors(task: BlindTask) -> tuple[int, ...]:
    input_counts: Counter[int] = Counter()
    output_counts: Counter[int] = Counter()
    for pair in task.train:
        input_counts.update(cell for row in pair.input for cell in row)
        output_counts.update(cell for row in pair.output for cell in row)
    positive = tuple(
        sorted(
            (color for color in output_counts if output_counts[color] > input_counts[color]),
            key=lambda color: (-(output_counts[color] - input_counts[color]), color),
        )
    )
    return positive[:4]


def enumerate_relational_delta_programs(
    task: BlindTask,
) -> tuple[RelationalDeltaProgram, ...]:
    """Enumerate the bounded v0.2 grammar from observable task content only."""

    if not isinstance(task, BlindTask):
        raise TypeError("relational delta synthesis accepts BlindTask only")
    programs = []
    for connectivity in (4, 8):
        parse = DeltaParseNode(connectivity)
        relational_roles = RoleAssignmentNode(
            "minority_foreground", "majority_foreground"
        )
        programs.extend(
            (
                RelationalDeltaProgram(
                    parse,
                    relational_roles,
                    NaryRelationNode("actor_area_to_row_interiors"),
                    DeltaTargetNode("region", "unique_actor_area_match"),
                    TypedDeltaMaskNode("actor_cells"),
                    DeltaRenderNode("erase_add", "actor"),
                ),
                RelationalDeltaProgram(
                    parse,
                    relational_roles,
                    NaryRelationNode("actor_area_to_row_interiors"),
                    DeltaTargetNode("region", "unique_actor_area_match"),
                    TypedDeltaMaskNode("none"),
                    DeltaRenderNode("add", "actor"),
                ),
                RelationalDeltaProgram(
                    parse,
                    relational_roles,
                    NaryRelationNode("actor_area_to_row_interiors"),
                    DeltaTargetNode("region", "unique_actor_area_match"),
                    TypedDeltaMaskNode("none"),
                    DeltaRenderNode("fill_relation_region", "actor"),
                ),
                RelationalDeltaProgram(
                    parse,
                    relational_roles,
                    NaryRelationNode("support_bbox_centers"),
                    DeltaTargetNode("coordinate", "all"),
                    TypedDeltaMaskNode("none"),
                    DeltaRenderNode("add", "actor"),
                ),
                RelationalDeltaProgram(
                    parse,
                    relational_roles,
                    NaryRelationNode("support_enclosed_regions"),
                    DeltaTargetNode("region", "all"),
                    TypedDeltaMaskNode("none"),
                    DeltaRenderNode("fill_relation_region", "actor"),
                ),
            )
        )
        for left_color, right_color in permutations(_positive_output_colors(task), 2):
            programs.append(
                RelationalDeltaProgram(
                    parse,
                    RoleAssignmentNode("only_foreground", "canvas"),
                    NaryRelationNode("component_to_horizontal_border"),
                    DeltaTargetNode("component", "all"),
                    TypedDeltaMaskNode("actor_cells"),
                    DeltaRenderNode(
                        "recolor_component",
                        "target_side_palette",
                        (left_color, right_color),
                    ),
                )
            )
    unique = {
        canonical_json(program.to_json_dict()): program for program in programs
    }
    return tuple(
        sorted(
            unique.values(),
            key=lambda program: (
                program.description_bits,
                relational_delta_program_id(program),
            ),
        )
    )


def _delta(first: Grid, second: Grid) -> set[Coordinate]:
    if (len(first), len(first[0])) != (len(second), len(second[0])):
        raise ValueError("delta comparison requires same-shape grids")
    return {
        (row, column)
        for row in range(len(first))
        for column in range(len(first[0]))
        if first[row][column] != second[row][column]
    }


@dataclass(frozen=True, slots=True)
class RelationalDeltaProgramScore:
    program: RelationalDeltaProgram
    demo_outputs: tuple[Grid | None, ...]
    exact_demo_count: int
    mismatch_count: int
    execution_valid: bool

    @property
    def all_demo_exact(self) -> bool:
        return self.execution_valid and self.exact_demo_count == len(self.demo_outputs)


def score_relational_delta_program(
    program: RelationalDeltaProgram, task: BlindTask
) -> RelationalDeltaProgramScore:
    outputs = []
    exact_count = 0
    mismatch_count = 0
    valid = True
    for pair in task.train:
        execution = execute_relational_delta(program, pair.input)
        output = execution.output if execution.ok else None
        outputs.append(output)
        if output is None:
            valid = False
            mismatch_count += len(pair.output) * len(pair.output[0])
        elif (len(output), len(output[0])) != (len(pair.output), len(pair.output[0])):
            mismatch_count += max(
                len(output) * len(output[0]),
                len(pair.output) * len(pair.output[0]),
            )
        else:
            mismatches = len(_delta(output, pair.output))
            mismatch_count += mismatches
            exact_count += int(mismatches == 0)
    return RelationalDeltaProgramScore(
        program,
        tuple(outputs),
        exact_count,
        mismatch_count,
        valid,
    )


@dataclass(frozen=True, slots=True)
class RelationalDeltaNearMissQuality:
    eligible: bool
    identity_mismatch_count: int
    parent_mismatch_count: int
    delta_precision: float
    delta_recall: float
    canvas_compatible: bool
    topology_valid: bool
    per_demo_not_worse: bool
    reason: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "eligible": self.eligible,
            "identity_mismatch_count": self.identity_mismatch_count,
            "parent_mismatch_count": self.parent_mismatch_count,
            "parent_improves_identity": (
                self.parent_mismatch_count < self.identity_mismatch_count
            ),
            "delta_precision": self.delta_precision,
            "delta_recall": self.delta_recall,
            "canvas_compatible": self.canvas_compatible,
            "topology_valid": self.topology_valid,
            "per_demo_not_worse": self.per_demo_not_worse,
            "reason": self.reason,
        }


def relational_delta_near_miss_quality(
    program: RelationalDeltaProgram,
    task: BlindTask,
    *,
    minimum_precision: float = RELATIONAL_DELTA_NEAR_MISS_PRECISION,
    minimum_recall: float = RELATIONAL_DELTA_NEAR_MISS_RECALL,
) -> RelationalDeltaNearMissQuality:
    canvas_compatible = all(
        (len(pair.input), len(pair.input[0]))
        == (len(pair.output), len(pair.output[0]))
        for pair in task.train
    )
    if not canvas_compatible:
        incompatible_cells = sum(
            max(
                len(pair.input) * len(pair.input[0]),
                len(pair.output) * len(pair.output[0]),
            )
            for pair in task.train
        )
        return RelationalDeltaNearMissQuality(
            False,
            incompatible_cells,
            incompatible_cells,
            0.0,
            0.0,
            False,
            False,
            False,
            "canvas_incompatible",
        )
    identity_mismatches = 0
    parent_mismatches = 0
    predicted_count = 0
    gold_count = 0
    intersection_count = 0
    topology_valid = True
    per_demo_not_worse = True
    execution_valid = True
    for pair in task.train:
        execution = execute_relational_delta(program, pair.input)
        if not execution.ok or execution.output is None:
            execution_valid = False
            topology_valid = False
            continue
        gold_delta = _delta(pair.input, pair.output)
        predicted_delta = _delta(pair.input, execution.output)
        identity_mismatch = len(gold_delta)
        parent_mismatch = len(_delta(execution.output, pair.output))
        identity_mismatches += identity_mismatch
        parent_mismatches += parent_mismatch
        predicted_count += len(predicted_delta)
        gold_count += len(gold_delta)
        intersection_count += len(predicted_delta & gold_delta)
        per_demo_not_worse &= parent_mismatch <= identity_mismatch
        topology_valid &= execution.topology_valid
    precision = intersection_count / predicted_count if predicted_count else 0.0
    recall = intersection_count / gold_count if gold_count else 0.0
    conditions = (
        execution_valid,
        parent_mismatches > 0,
        parent_mismatches < identity_mismatches,
        per_demo_not_worse,
        precision >= minimum_precision,
        recall >= minimum_recall,
        topology_valid,
    )
    reasons = (
        "invalid_execution",
        "parent_is_exact",
        "does_not_improve_identity",
        "worse_on_a_demo",
        "delta_precision_below_threshold",
        "delta_recall_below_threshold",
        "topology_invalid",
    )
    reason = "eligible" if all(conditions) else reasons[conditions.index(False)]
    return RelationalDeltaNearMissQuality(
        all(conditions),
        identity_mismatches,
        parent_mismatches,
        precision,
        recall,
        True,
        topology_valid,
        per_demo_not_worse,
        reason,
    )


RELATIONAL_DELTA_DIAGNOSIS: dict[str, tuple[str, tuple[str, ...]]] = {
    "role_assignment": (
        "object_rematch",
        ("parse.connectivity", "roles.actor_role", "roles.support_role"),
    ),
    "target_geometry": (
        "target_reinfer",
        ("parse.connectivity", "relation.kind", "target.kind", "target.selector"),
    ),
    "delta_mask": (
        "edit_delta_mask",
        ("mask.erase", "render.operation"),
    ),
    "render": (
        "edit_render",
        ("render.color_policy", "render.operation", "render.palette"),
    ),
}


@dataclass(frozen=True, slots=True)
class RelationalDeltaFailureCertificate:
    certificate_id: str
    task_id: str
    parent_program_id: str
    diagnosis: str
    action: str
    affected_slots: tuple[str, ...]
    evidence: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.diagnosis not in RELATIONAL_DELTA_DIAGNOSIS:
            raise ValueError("unknown relational delta diagnosis")
        action, slots = RELATIONAL_DELTA_DIAGNOSIS[self.diagnosis]
        if self.action != action or self.affected_slots != slots:
            raise ValueError("certificate action or slots do not match diagnosis")
        if self.certificate_id != self._content_id():
            raise ValueError("relational delta certificate ID does not match content")

    def _content(self) -> dict[str, object]:
        return {
            "schema": RELATIONAL_DELTA_CERTIFICATE_SCHEMA,
            "task_id": self.task_id,
            "parent_program_id": self.parent_program_id,
            "diagnosis": self.diagnosis,
            "action": self.action,
            "affected_slots": list(self.affected_slots),
            "evidence": dict(self.evidence),
        }

    def _content_id(self) -> str:
        return hashlib.sha256(canonical_json(self._content()).encode("ascii")).hexdigest()

    def to_json_dict(self) -> dict[str, object]:
        return {"certificate_id": self.certificate_id, **self._content()}


def compile_relational_delta_failure_certificate(
    *,
    task_id: str,
    task: BlindTask,
    parent: RelationalDeltaProgram,
) -> RelationalDeltaFailureCertificate:
    rows = []
    executions = tuple(
        execute_relational_delta(parent, pair.input) for pair in task.train
    )
    mask_exact = True
    for index, (pair, execution) in enumerate(
        zip(task.train, executions, strict=True)
    ):
        predicted_delta = (
            set() if execution.output is None else _delta(pair.input, execution.output)
        )
        gold_delta = _delta(pair.input, pair.output)
        mask_exact &= predicted_delta == gold_delta and bool(gold_delta)
        rows.append(
            {
                "demo_index": index,
                "execution_status": execution.status,
                "execution_reason": execution.reason,
                "predicted_delta": [list(item) for item in sorted(predicted_delta)],
                "gold_delta": [list(item) for item in sorted(gold_delta)],
                "topology_valid": execution.topology_valid,
            }
        )
    quality = relational_delta_near_miss_quality(parent, task)
    if any(execution.reason == "ambiguous_roles" for execution in executions):
        diagnosis = "role_assignment"
    elif mask_exact:
        diagnosis = "render"
    elif quality.delta_precision >= 0.5 and quality.delta_recall >= 0.5:
        diagnosis = "delta_mask"
    else:
        diagnosis = "target_geometry"
    action, slots = RELATIONAL_DELTA_DIAGNOSIS[diagnosis]
    evidence: dict[str, object] = {
        "demo_count": len(rows),
        "demo_residuals": rows,
        "near_miss_quality": quality.to_json_dict(),
        "query_gold_read": False,
        "existing_slots_only": True,
    }
    content = {
        "schema": RELATIONAL_DELTA_CERTIFICATE_SCHEMA,
        "task_id": task_id,
        "parent_program_id": relational_delta_program_id(parent),
        "diagnosis": diagnosis,
        "action": action,
        "affected_slots": list(slots),
        "evidence": evidence,
    }
    certificate_id = hashlib.sha256(canonical_json(content).encode("ascii")).hexdigest()
    return RelationalDeltaFailureCertificate(
        certificate_id,
        task_id,
        content["parent_program_id"],
        diagnosis,
        action,
        slots,
        evidence,
    )


def relational_delta_program_fields(
    program: RelationalDeltaProgram,
) -> dict[str, object]:
    return {
        "parse.connectivity": program.parse.connectivity,
        "roles.actor_role": program.roles.actor_role,
        "roles.support_role": program.roles.support_role,
        "relation.kind": program.relation.kind,
        "target.kind": program.target.kind,
        "target.selector": program.target.selector,
        "mask.erase": program.mask.erase,
        "render.operation": program.render.operation,
        "render.color_policy": program.render.color_policy,
        "render.palette": program.render.palette,
    }


@dataclass(frozen=True, slots=True)
class RelationalDeltaSlotEdit:
    slot: str
    program: RelationalDeltaProgram


def single_slot_relational_delta_variants(
    parent: RelationalDeltaProgram,
    *,
    allowed_slots: Sequence[str],
    candidate_programs: Sequence[RelationalDeltaProgram],
    existing_program_ids: Sequence[str] = (),
) -> tuple[RelationalDeltaSlotEdit, ...]:
    parent_fields = relational_delta_program_fields(parent)
    allowed = frozenset(allowed_slots)
    if not allowed or any(slot not in parent_fields for slot in allowed):
        raise ValueError("allowed slots must name existing relational delta fields")
    existing = frozenset(existing_program_ids)
    edits = []
    for program in candidate_programs:
        if relational_delta_program_id(program) in existing:
            continue
        fields = relational_delta_program_fields(program)
        differences = tuple(
            sorted(
                slot for slot in parent_fields if parent_fields[slot] != fields[slot]
            )
        )
        if len(differences) == 1 and differences[0] in allowed:
            edits.append(RelationalDeltaSlotEdit(differences[0], program))
    return tuple(
        sorted(
            edits,
            key=lambda edit: (edit.slot, relational_delta_program_id(edit.program)),
        )
    )
