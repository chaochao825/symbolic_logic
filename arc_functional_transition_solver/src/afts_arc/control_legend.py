"""Typed programs driven by an ordered color-pair control legend.

This opt-in DSL is isolated from every existing provider.  Its parser finds a
unique lane of isolated ordered color pairs, protects the control cells, and
uses the resulting sequence either to rewrite payload cells or to color the
exterior background of payload-object bounding boxes.
"""

from __future__ import annotations

import hashlib
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .blind import BlindTask
from .grid import Grid, as_grid
from .hybrid.scene_graph import ExecutionTraceNode
from .hybrid.types import canonical_json


CONTROL_LEGEND_DSL_VERSION = "afts-control-legend-dsl/v0.1"
CONTROL_LEGEND_OPERATIONS = (
    "ordered_rewrite_payload",
    "fill_exterior_bbox_background",
)
CONTROL_LEGEND_PROGRAM_CAP = 16

Coordinate = tuple[int, int]
ColorRule = tuple[int, int]
ColoredCoordinate = tuple[int, int, int]


def _arc_color(value: object, *, field_name: str) -> int:
    if type(value) is not int or not 0 <= value <= 9:
        raise ValueError(f"{field_name} must be an ARC color")
    return value


@dataclass(frozen=True, slots=True)
class LegendParseNode:
    pair_axis: str
    pair_direction: str
    sequence_direction: str
    selector: str = "unique_isolated_pair_lane"
    max_gap: int = 2

    def __post_init__(self) -> None:
        if self.pair_axis not in {"horizontal", "vertical"}:
            raise ValueError("legend pair axis must be horizontal or vertical")
        if self.pair_direction not in {"forward", "reverse"}:
            raise ValueError("legend pair direction must be forward or reverse")
        if self.sequence_direction not in {"forward", "reverse"}:
            raise ValueError("legend sequence direction must be forward or reverse")
        if self.selector != "unique_isolated_pair_lane":
            raise ValueError("unknown legend control selector")
        if type(self.max_gap) is not int or self.max_gap != 2:
            raise ValueError("v0.1 legend programs require max_gap=2")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "op": "parse_control_legend",
            "background": "modal",
            "pair_axis": self.pair_axis,
            "pair_direction": self.pair_direction,
            "sequence_direction": self.sequence_direction,
            "selector": self.selector,
            "max_gap": self.max_gap,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "LegendParseNode":
        expected = {
            "op",
            "background",
            "pair_axis",
            "pair_direction",
            "sequence_direction",
            "selector",
            "max_gap",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("legend parse node has missing or unknown fields")
        if payload["op"] != "parse_control_legend":
            raise ValueError("legend parse node has the wrong operator")
        if payload["background"] != "modal":
            raise ValueError("unsupported legend background hypothesis")
        return cls(
            payload["pair_axis"],
            payload["pair_direction"],
            payload["sequence_direction"],
            payload["selector"],
            payload["max_gap"],
        )


@dataclass(frozen=True, slots=True)
class OrderedColorCorrespondenceNode:
    relation: str = "ordered_color_rewrite"
    duplicate_policy: str = "collapse_adjacent"

    def __post_init__(self) -> None:
        if self.relation != "ordered_color_rewrite":
            raise ValueError("unknown legend correspondence relation")
        if self.duplicate_policy != "collapse_adjacent":
            raise ValueError("unknown legend duplicate policy")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "op": "correspond_ordered_colors",
            "relation": self.relation,
            "duplicate_policy": self.duplicate_policy,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "OrderedColorCorrespondenceNode":
        expected = {"op", "relation", "duplicate_policy"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("legend correspondence node has missing or unknown fields")
        if payload["op"] != "correspond_ordered_colors":
            raise ValueError("legend correspondence node has the wrong operator")
        return cls(payload["relation"], payload["duplicate_policy"])


@dataclass(frozen=True, slots=True)
class LegendPayloadNode:
    kind: str
    connectivity: int | None

    def __post_init__(self) -> None:
        if self.kind == "non_control_cells":
            if self.connectivity is not None:
                raise ValueError("cell payloads do not use object connectivity")
        elif self.kind == "source_color_components":
            if type(self.connectivity) is not int or self.connectivity != 4:
                raise ValueError("v0.1 component payloads require 4-connectivity")
        else:
            raise ValueError("unknown legend payload kind")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "op": "assign_payload",
            "kind": self.kind,
            "connectivity": self.connectivity,
            "exclude": "control_cells",
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "LegendPayloadNode":
        expected = {"op", "kind", "connectivity", "exclude"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("legend payload node has missing or unknown fields")
        if payload["op"] != "assign_payload":
            raise ValueError("legend payload node has the wrong operator")
        if payload["exclude"] != "control_cells":
            raise ValueError("legend payload must exclude control cells")
        return cls(payload["kind"], payload["connectivity"])


@dataclass(frozen=True, slots=True)
class LegendRenderNode:
    operation: str
    padding: int | None

    def __post_init__(self) -> None:
        if self.operation not in CONTROL_LEGEND_OPERATIONS:
            raise ValueError("unknown control-legend render operation")
        if self.operation == "ordered_rewrite_payload":
            if self.padding is not None:
                raise ValueError("ordered cell rewriting does not use padding")
        elif type(self.padding) is not int or self.padding != 1:
            raise ValueError("v0.1 exterior bbox rendering requires padding=1")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "op": "render_legend_payload",
            "operation": self.operation,
            "padding": self.padding,
            "write_policy": "background_only",
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "LegendRenderNode":
        expected = {"op", "operation", "padding", "write_policy"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("legend render node has missing or unknown fields")
        if payload["op"] != "render_legend_payload":
            raise ValueError("legend render node has the wrong operator")
        if payload["write_policy"] != "background_only":
            raise ValueError("unsupported legend write policy")
        return cls(payload["operation"], payload["padding"])


@dataclass(frozen=True, slots=True)
class ControlLegendProgram:
    parse: LegendParseNode
    correspondence: OrderedColorCorrespondenceNode
    payload: LegendPayloadNode
    render: LegendRenderNode

    def __post_init__(self) -> None:
        rewrite = self.render.operation == "ordered_rewrite_payload"
        if rewrite != (self.payload.kind == "non_control_cells"):
            raise ValueError("legend payload type does not match render semantics")

    @property
    def description_bits(self) -> int:
        return 44 if self.render.operation == "ordered_rewrite_payload" else 52

    @property
    def functional_trace(self) -> tuple[str, ...]:
        return (
            (
                "node:parse:legend:"
                f"axis={self.parse.pair_axis}:pair={self.parse.pair_direction}:"
                f"sequence={self.parse.sequence_direction}:gap={self.parse.max_gap}"
            ),
            (
                "node:correspond:ordered_colors:"
                f"duplicates={self.correspondence.duplicate_policy}"
            ),
            (
                "node:payload:"
                f"kind={self.payload.kind}:connectivity={self.payload.connectivity}"
            ),
            (
                "node:render:legend:"
                f"op={self.render.operation}:padding={self.render.padding}"
            ),
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "control_legend_dsl_version": CONTROL_LEGEND_DSL_VERSION,
            "kind": "control_legend_program",
            "ast": {
                "parse": self.parse.to_json_dict(),
                "correspondence": self.correspondence.to_json_dict(),
                "payload": self.payload.to_json_dict(),
                "render": self.render.to_json_dict(),
            },
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "ControlLegendProgram":
        expected = {"control_legend_dsl_version", "kind", "ast"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("control-legend program has missing or unknown fields")
        if payload["control_legend_dsl_version"] != CONTROL_LEGEND_DSL_VERSION:
            raise ValueError("unsupported control-legend DSL version")
        if payload["kind"] != "control_legend_program":
            raise ValueError("program kind is not control_legend_program")
        ast = payload["ast"]
        ast_expected = {"parse", "correspondence", "payload", "render"}
        if not isinstance(ast, Mapping) or set(ast) != ast_expected:
            raise ValueError("control-legend AST has missing or unknown nodes")
        return cls(
            LegendParseNode.from_json_dict(ast["parse"]),
            OrderedColorCorrespondenceNode.from_json_dict(ast["correspondence"]),
            LegendPayloadNode.from_json_dict(ast["payload"]),
            LegendRenderNode.from_json_dict(ast["render"]),
        )


def control_legend_program_id(program: ControlLegendProgram) -> str:
    return hashlib.sha256(
        canonical_json(program.to_json_dict()).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ParsedControlLegend:
    background: int
    pair_axis: str
    lane_coordinate: int
    ordered_rules: tuple[ColorRule, ...]
    control_cells: tuple[Coordinate, ...]
    raw_pair_count: int

    def __post_init__(self) -> None:
        _arc_color(self.background, field_name="legend background")
        if self.pair_axis not in {"horizontal", "vertical"}:
            raise ValueError("parsed legend has an invalid pair axis")
        if type(self.lane_coordinate) is not int or self.lane_coordinate < 0:
            raise ValueError("legend lane coordinate must be non-negative")
        if len(self.ordered_rules) < 2:
            raise ValueError("a parsed legend requires at least two rules")
        for index, rule in enumerate(self.ordered_rules):
            if not isinstance(rule, tuple) or len(rule) != 2:
                raise TypeError("legend rules must be ordered color pairs")
            source = _arc_color(rule[0], field_name=f"legend rule {index} source")
            target = _arc_color(rule[1], field_name=f"legend rule {index} target")
            if source == target:
                raise ValueError("legend rule source and target must differ")
            if self.background in {source, target}:
                raise ValueError("legend rules cannot contain the modal background")
        if tuple(sorted(set(self.control_cells))) != self.control_cells:
            raise ValueError("legend control cells must be unique and sorted")
        if type(self.raw_pair_count) is not int or self.raw_pair_count < len(
            self.ordered_rules
        ):
            raise ValueError("legend raw-pair count is inconsistent")
        if len(self.control_cells) != 2 * self.raw_pair_count:
            raise ValueError("legend control-cell count does not match pair count")


@dataclass(frozen=True, slots=True)
class _PairObservation:
    sequence_coordinate: int
    first: int
    second: int
    cells: tuple[Coordinate, Coordinate]


def _modal_color(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda color: (-counts[color], color))


def _horizontal_observations(
    grid: Grid, background: int, lane: int
) -> tuple[_PairObservation, ...]:
    height, width = len(grid), len(grid[0])
    observations = []
    for row in range(height):
        first = grid[row][lane]
        second = grid[row][lane + 1]
        isolated = (lane == 0 or grid[row][lane - 1] == background) and (
            lane + 2 == width or grid[row][lane + 2] == background
        )
        if (
            first != background
            and second != background
            and first != second
            and isolated
        ):
            observations.append(
                _PairObservation(
                    row,
                    first,
                    second,
                    ((row, lane), (row, lane + 1)),
                )
            )
    return tuple(observations)


def _vertical_observations(
    grid: Grid, background: int, lane: int
) -> tuple[_PairObservation, ...]:
    height, width = len(grid), len(grid[0])
    observations = []
    for column in range(width):
        first = grid[lane][column]
        second = grid[lane + 1][column]
        isolated = (lane == 0 or grid[lane - 1][column] == background) and (
            lane + 2 == height or grid[lane + 2][column] == background
        )
        if (
            first != background
            and second != background
            and first != second
            and isolated
        ):
            observations.append(
                _PairObservation(
                    column,
                    first,
                    second,
                    ((lane, column), (lane + 1, column)),
                )
            )
    return tuple(observations)


def _split_observation_groups(
    observations: Sequence[_PairObservation], *, max_gap: int
) -> tuple[tuple[_PairObservation, ...], ...]:
    groups: list[list[_PairObservation]] = []
    current: list[_PairObservation] = []
    for observation in observations:
        if current and (
            observation.sequence_coordinate - current[-1].sequence_coordinate > max_gap
        ):
            groups.append(current)
            current = []
        current.append(observation)
    if current:
        groups.append(current)
    return tuple(tuple(group) for group in groups)


def _legend_from_group(
    node: LegendParseNode,
    *,
    background: int,
    lane_coordinate: int,
    group: Sequence[_PairObservation],
) -> ParsedControlLegend | None:
    ordered = tuple(group)
    if node.sequence_direction == "reverse":
        ordered = tuple(reversed(ordered))
    rules: list[ColorRule] = []
    for observation in ordered:
        pair = (observation.first, observation.second)
        if node.pair_direction == "reverse":
            pair = (pair[1], pair[0])
        if not rules or rules[-1] != pair:
            rules.append(pair)
    if len(rules) < 2:
        return None
    control_cells = tuple(
        sorted(cell for observation in group for cell in observation.cells)
    )
    return ParsedControlLegend(
        background,
        node.pair_axis,
        lane_coordinate,
        tuple(rules),
        control_cells,
        len(group),
    )


def _parse_control_legend(
    node: LegendParseNode, grid: Grid
) -> tuple[ParsedControlLegend | None, str | None]:
    normalized = as_grid(grid)
    background = _modal_color(normalized)
    lane_count = (
        len(normalized[0]) - 1
        if node.pair_axis == "horizontal"
        else len(normalized) - 1
    )
    candidates = []
    for lane in range(lane_count):
        observations = (
            _horizontal_observations(normalized, background, lane)
            if node.pair_axis == "horizontal"
            else _vertical_observations(normalized, background, lane)
        )
        for group in _split_observation_groups(observations, max_gap=node.max_gap):
            candidate = _legend_from_group(
                node,
                background=background,
                lane_coordinate=lane,
                group=group,
            )
            if candidate is not None:
                candidates.append(candidate)
    if not candidates:
        return None, "no_control_lane"
    if len(candidates) != 1:
        return None, "ambiguous_control_lanes"
    return candidates[0], None


def parse_control_legend(node: LegendParseNode, grid: Grid) -> ParsedControlLegend:
    """Return the unique parsed legend or raise for absent/ambiguous control."""

    parsed, reason = _parse_control_legend(node, as_grid(grid))
    if parsed is None:
        raise ValueError(reason)
    return parsed


@dataclass(frozen=True, slots=True)
class ControlLegendExecution:
    status: str
    output: Grid | None
    reason: str | None
    background: int
    ordered_rules: tuple[ColorRule, ...]
    control_cells: tuple[Coordinate, ...]
    payload_component_count: int
    write_mask: tuple[Coordinate, ...]
    writes: tuple[ColoredCoordinate, ...]
    topology_valid: bool
    node_trace: tuple[ExecutionTraceNode, ...]

    def __post_init__(self) -> None:
        if self.status not in {"ok", "invalid"}:
            raise ValueError("control-legend execution status is invalid")
        if (self.status == "ok") != (self.output is not None):
            raise ValueError("only successful legend executions carry output")
        if tuple(sorted(set(self.control_cells))) != self.control_cells:
            raise ValueError("execution control cells must be unique and sorted")
        if type(self.payload_component_count) is not int or (
            self.payload_component_count < 0
        ):
            raise ValueError("payload component count must be non-negative")
        if tuple(sorted(set(self.write_mask))) != self.write_mask:
            raise ValueError("legend write mask must be unique and sorted")
        if tuple(sorted(self.writes)) != self.writes:
            raise ValueError("legend colored writes must be sorted")
        if tuple((row, column) for row, column, _ in self.writes) != self.write_mask:
            raise ValueError("legend writes and write mask do not close")
        if self.status == "ok" and not self.topology_valid:
            raise ValueError("successful legend execution must satisfy topology")

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _invalid_execution(
    *,
    reason: str,
    background: int,
    parsed: ParsedControlLegend | None,
    payload_component_count: int,
    trace: Sequence[ExecutionTraceNode],
) -> ControlLegendExecution:
    return ControlLegendExecution(
        "invalid",
        None,
        reason,
        background,
        () if parsed is None else parsed.ordered_rules,
        () if parsed is None else parsed.control_cells,
        payload_component_count,
        (),
        (),
        False,
        tuple(trace),
    )


def _source_colors(rules: Sequence[ColorRule]) -> tuple[int, ...]:
    ordered = []
    for source, _ in rules:
        if source not in ordered:
            ordered.append(source)
    return tuple(ordered)


def _resolve_color(color: int, rules: Sequence[ColorRule]) -> int:
    resolved = color
    for source, target in rules:
        if resolved == source:
            resolved = target
    return resolved


def _monochrome_components(
    grid: Grid,
    *,
    color: int,
    excluded: frozenset[Coordinate],
    connectivity: int,
) -> tuple[tuple[Coordinate, ...], ...]:
    if connectivity != 4:
        raise ValueError("v0.1 control-legend components require 4-connectivity")
    height, width = len(grid), len(grid[0])
    unseen = {
        (row, column)
        for row in range(height)
        for column in range(width)
        if grid[row][column] == color and (row, column) not in excluded
    }
    components = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue: deque[Coordinate] = deque((start,))
        cells = {start}
        while queue:
            row, column = queue.popleft()
            for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                neighbor = (row + dr, column + dc)
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    cells.add(neighbor)
                    queue.append(neighbor)
        components.append(tuple(sorted(cells)))
    return tuple(sorted(components))


def _exterior_bbox_background(
    grid: Grid,
    component: Sequence[Coordinate],
    *,
    background: int,
    padding: int,
) -> tuple[Coordinate, ...]:
    height, width = len(grid), len(grid[0])
    top = max(0, min(row for row, _ in component) - padding)
    left = max(0, min(column for _, column in component) - padding)
    bottom = min(height - 1, max(row for row, _ in component) + padding)
    right = min(width - 1, max(column for _, column in component) + padding)
    available = {
        (row, column)
        for row in range(top, bottom + 1)
        for column in range(left, right + 1)
        if grid[row][column] == background
    }
    boundary = {
        coordinate
        for coordinate in available
        if coordinate[0] in {top, bottom} or coordinate[1] in {left, right}
    }
    reachable = set(boundary)
    queue: deque[Coordinate] = deque(sorted(boundary))
    while queue:
        row, column = queue.popleft()
        for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            neighbor = (row + dr, column + dc)
            if neighbor in available and neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)
    return tuple(sorted(reachable))


def execute_control_legend(
    program: ControlLegendProgram, grid: Grid
) -> ControlLegendExecution:
    normalized = as_grid(grid)
    background = _modal_color(normalized)
    parsed, reason = _parse_control_legend(program.parse, normalized)
    trace = []
    if parsed is None:
        trace.append(
            ExecutionTraceNode.create(
                "parse", "parse_control_legend", "invalid", reason=reason
            )
        )
        return _invalid_execution(
            reason=reason or "legend_parse_failed",
            background=background,
            parsed=None,
            payload_component_count=0,
            trace=trace,
        )
    trace.append(
        ExecutionTraceNode.create(
            "parse",
            "parse_control_legend",
            "ok",
            pair_axis=parsed.pair_axis,
            raw_pair_count=parsed.raw_pair_count,
        )
    )
    trace.append(
        ExecutionTraceNode.create(
            "correspondence",
            "correspond_ordered_colors",
            "ok",
            rule_count=len(parsed.ordered_rules),
        )
    )
    control = frozenset(parsed.control_cells)
    colored_writes: dict[Coordinate, int] = {}
    payload_component_count = 0

    if program.render.operation == "ordered_rewrite_payload":
        canvas = [list(row) for row in normalized]
        for source, target in parsed.ordered_rules:
            for row in range(len(canvas)):
                for column in range(len(canvas[0])):
                    if (row, column) not in control and canvas[row][column] == source:
                        canvas[row][column] = target
        for row in range(len(canvas)):
            for column in range(len(canvas[0])):
                if canvas[row][column] != normalized[row][column]:
                    colored_writes[(row, column)] = canvas[row][column]
        trace.append(
            ExecutionTraceNode.create(
                "payload",
                "assign_payload",
                "ok",
                kind=program.payload.kind,
            )
        )
    else:
        if program.payload.connectivity is None or program.render.padding is None:
            raise AssertionError("component render lacks typed parameters")
        for source in _source_colors(parsed.ordered_rules):
            target = _resolve_color(source, parsed.ordered_rules)
            if target == source:
                continue
            components = _monochrome_components(
                normalized,
                color=source,
                excluded=control,
                connectivity=program.payload.connectivity,
            )
            payload_component_count += len(components)
            for component in components:
                region = _exterior_bbox_background(
                    normalized,
                    component,
                    background=background,
                    padding=program.render.padding,
                )
                for coordinate in region:
                    if (
                        coordinate in colored_writes
                        and colored_writes[coordinate] != target
                    ):
                        trace.append(
                            ExecutionTraceNode.create(
                                "mask",
                                "build_legend_mask",
                                "invalid",
                                reason="conflicting_payload_writes",
                            )
                        )
                        return _invalid_execution(
                            reason="conflicting_payload_writes",
                            background=background,
                            parsed=parsed,
                            payload_component_count=payload_component_count,
                            trace=trace,
                        )
                    colored_writes[coordinate] = target
        trace.append(
            ExecutionTraceNode.create(
                "payload",
                "assign_payload",
                "ok",
                component_count=payload_component_count,
                kind=program.payload.kind,
            )
        )
        canvas = [list(row) for row in normalized]
        for (row, column), color in colored_writes.items():
            canvas[row][column] = color

    if not colored_writes:
        trace.append(
            ExecutionTraceNode.create(
                "render",
                "render_legend_payload",
                "invalid",
                reason="empty_effective_delta",
            )
        )
        return _invalid_execution(
            reason="empty_effective_delta",
            background=background,
            parsed=parsed,
            payload_component_count=payload_component_count,
            trace=trace,
        )
    if any(coordinate in control for coordinate in colored_writes):
        raise AssertionError("control-legend renderer modified protected control cells")
    if any(
        normalized[row][column] != background for row, column in colored_writes
    ) and (program.render.operation == "fill_exterior_bbox_background"):
        raise AssertionError("exterior bbox renderer overwrote a preserved object")
    output = as_grid(canvas)
    writes = tuple(
        (row, column, colored_writes[(row, column)])
        for row, column in sorted(colored_writes)
    )
    trace.append(
        ExecutionTraceNode.create(
            "mask",
            "build_legend_mask",
            "ok",
            control_cell_count=len(control),
            write_count=len(writes),
        )
    )
    trace.append(
        ExecutionTraceNode.create(
            "render",
            "render_legend_payload",
            "ok",
            operation=program.render.operation,
        )
    )
    return ControlLegendExecution(
        "ok",
        output,
        None,
        background,
        parsed.ordered_rules,
        parsed.control_cells,
        payload_component_count,
        tuple((row, column) for row, column, _ in writes),
        writes,
        True,
        tuple(trace),
    )


def _program(
    *,
    pair_axis: str,
    pair_direction: str,
    sequence_direction: str,
    operation: str,
) -> ControlLegendProgram:
    if operation == "ordered_rewrite_payload":
        payload = LegendPayloadNode("non_control_cells", None)
        render = LegendRenderNode(operation, None)
    else:
        payload = LegendPayloadNode("source_color_components", 4)
        render = LegendRenderNode(operation, 1)
    return ControlLegendProgram(
        LegendParseNode(pair_axis, pair_direction, sequence_direction),
        OrderedColorCorrespondenceNode(),
        payload,
        render,
    )


def enumerate_control_legend_programs(
    task: BlindTask,
) -> tuple[ControlLegendProgram, ...]:
    """Enumerate the fixed v0.1 grammar from an oracle-free task view."""

    if not isinstance(task, BlindTask):
        raise TypeError("control-legend synthesis accepts BlindTask only")
    programs = tuple(
        _program(
            pair_axis=pair_axis,
            pair_direction=pair_direction,
            sequence_direction=sequence_direction,
            operation=operation,
        )
        for pair_axis in ("horizontal", "vertical")
        for pair_direction in ("forward", "reverse")
        for sequence_direction in ("forward", "reverse")
        for operation in CONTROL_LEGEND_OPERATIONS
    )
    if len(programs) != CONTROL_LEGEND_PROGRAM_CAP:
        raise AssertionError("control-legend grammar size changed")
    return tuple(
        sorted(
            programs,
            key=lambda program: (
                program.description_bits,
                control_legend_program_id(program),
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class ControlLegendProgramScore:
    program: ControlLegendProgram
    demo_outputs: tuple[Grid | None, ...]
    exact_demo_count: int
    mismatch_count: int
    execution_valid: bool

    @property
    def all_demo_exact(self) -> bool:
        return self.execution_valid and self.exact_demo_count == len(self.demo_outputs)


def score_control_legend_program(
    program: ControlLegendProgram, task: BlindTask
) -> ControlLegendProgramScore:
    if not isinstance(task, BlindTask):
        raise TypeError("control-legend scoring accepts BlindTask only")
    outputs = []
    exact_demo_count = 0
    mismatch_count = 0
    execution_valid = True
    for pair in task.train:
        execution = execute_control_legend(program, pair.input)
        output = execution.output if execution.ok else None
        outputs.append(output)
        if output is None:
            execution_valid = False
            mismatch_count += len(pair.output) * len(pair.output[0])
            continue
        if (len(output), len(output[0])) != (
            len(pair.output),
            len(pair.output[0]),
        ):
            mismatch_count += max(
                len(output) * len(output[0]),
                len(pair.output) * len(pair.output[0]),
            )
            continue
        mismatches = sum(
            output[row][column] != pair.output[row][column]
            for row in range(len(output))
            for column in range(len(output[0]))
        )
        mismatch_count += mismatches
        exact_demo_count += int(mismatches == 0)
    return ControlLegendProgramScore(
        program,
        tuple(outputs),
        exact_demo_count,
        mismatch_count,
        execution_valid,
    )


def synthesize_control_legend_programs(
    task: BlindTask, *, max_exact_programs: int = CONTROL_LEGEND_PROGRAM_CAP
) -> tuple[ControlLegendProgramScore, ...]:
    if type(max_exact_programs) is not int or not 1 <= max_exact_programs <= (
        CONTROL_LEGEND_PROGRAM_CAP
    ):
        raise ValueError("max_exact_programs is outside the v0.1 grammar cap")
    exact = tuple(
        score
        for program in enumerate_control_legend_programs(task)
        if (score := score_control_legend_program(program, task)).all_demo_exact
    )
    return tuple(
        sorted(
            exact,
            key=lambda score: (
                score.program.description_bits,
                control_legend_program_id(score.program),
            ),
        )[:max_exact_programs]
    )
