"""A small typed, deterministic ARC grid DSL used as a coverage baseline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .grid import (
    ARC_MAX_HEIGHT,
    ARC_MAX_WIDTH,
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
from .parse import ColorMode, ObjectView, parse_grid
from .panel import PanelOverlayCode, overlay_panel_grid
from .panel_actions import (
    D4Step,
    PanelLatticePeriodicCode,
    PanelSequenceD4Code,
    broadcast_panel_lattice_periodic,
    broadcast_panel_sequence_d4,
)
from .contact_actions import BBoxContactCode, paint_bbox_contacts

DSL_SEMANTICS_VERSION = "afts-grid-dsl/v0.6"


class ValueType(str, Enum):
    GRID = "Grid"
    COLOR = "Color"
    CONNECTIVITY = "Connectivity"
    COLOR_MODE = "ColorMode"
    COLOR_MAP = "ColorMap"
    POSITIVE_INT = "PositiveInt"
    D4_STEP = "D4Step"


class InvalidCode(str, Enum):
    EMPTY_SELECTION = "empty_selection"
    NON_UNIQUE_SELECTION = "non_unique_selection"
    OUTPUT_SHAPE_EXCEEDS_LIMIT = "output_shape_exceeds_limit"
    INCOMPATIBLE_PANEL_SHAPES = "incompatible_panel_shapes"
    TARGET_COLLISION = "target_collision"
    OCCLUDED_RAY = "occluded_ray"
    INCOMPATIBLE_RELATION_GEOMETRY = "incompatible_relation_geometry"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True, slots=True)
class PrimitiveSpec:
    name: str
    input_type: ValueType
    output_type: ValueType
    argument_types: tuple[tuple[str, ValueType], ...]


@dataclass(frozen=True, slots=True)
class Instruction:
    op: str
    arguments_json: str

    def __post_init__(self) -> None:
        spec = primitive_spec(self.op)
        arguments = _parse_arguments(self.arguments_json)
        normalized = _validate_arguments(spec, arguments)
        object.__setattr__(
            self,
            "arguments_json",
            json.dumps(
                normalized,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ),
        )

    @classmethod
    def create(cls, op: str, **arguments: object) -> "Instruction":
        return cls(
            op=op,
            arguments_json=json.dumps(
                arguments,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ),
        )

    @property
    def arguments(self) -> dict[str, Any]:
        return json.loads(self.arguments_json)

    def to_json_dict(self) -> dict[str, object]:
        return {"op": self.op, "arguments": self.arguments}


@dataclass(frozen=True, slots=True)
class Program:
    program_id: str
    instructions: tuple[Instruction, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.instructions, (tuple, list)) or not self.instructions:
            raise ValueError("program must contain at least one instruction")
        instructions = tuple(self.instructions)
        if any(not isinstance(item, Instruction) for item in instructions):
            raise TypeError("program instructions must be Instruction values")
        expected = _program_id(instructions)
        if self.program_id != expected:
            raise ValueError("program_id does not match canonical program content")
        object.__setattr__(self, "instructions", instructions)

    @classmethod
    def create(cls, instructions: tuple[Instruction, ...] | list[Instruction]) -> "Program":
        normalized = tuple(instructions)
        return cls(program_id=_program_id(normalized), instructions=normalized)

    @classmethod
    def from_json_dict(cls, payload: object) -> "Program":
        """Load a canonical program record and verify its content address."""

        if not isinstance(payload, dict):
            raise TypeError("program must be a JSON object")
        expected = {"dsl_semantics_version", "program_id", "instructions"}
        if set(payload) != expected:
            raise ValueError(
                f"program fields must be {sorted(expected)}, found {sorted(payload)}"
            )
        if payload["dsl_semantics_version"] != DSL_SEMANTICS_VERSION:
            raise ValueError("unsupported DSL semantics version")
        raw_instructions = payload["instructions"]
        if not isinstance(raw_instructions, list) or not raw_instructions:
            raise ValueError("program instructions must be a non-empty list")
        instructions: list[Instruction] = []
        for index, item in enumerate(raw_instructions):
            if not isinstance(item, dict) or set(item) != {"op", "arguments"}:
                raise ValueError(
                    f"instruction[{index}] must contain exactly op and arguments"
                )
            if not isinstance(item["arguments"], dict):
                raise TypeError(f"instruction[{index}] arguments must be an object")
            instructions.append(Instruction.create(item["op"], **item["arguments"]))
        return cls(program_id=payload["program_id"], instructions=tuple(instructions))

    @property
    def node_count(self) -> int:
        return len(self.instructions)

    @property
    def functional_trace(self) -> tuple[str, ...]:
        return tuple(instruction.op for instruction in self.instructions)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "dsl_semantics_version": DSL_SEMANTICS_VERSION,
            "program_id": self.program_id,
            "instructions": [item.to_json_dict() for item in self.instructions],
        }


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    status: str
    output: Grid | None
    invalid_code: InvalidCode | None
    failing_instruction_index: int | None
    semantic_trace: tuple[tuple[str, str], ...]

    @property
    def ok(self) -> bool:
        return self.status == "ok"


_SPECS: tuple[PrimitiveSpec, ...] = (
    PrimitiveSpec("identity", ValueType.GRID, ValueType.GRID, ()),
    PrimitiveSpec("rotate90", ValueType.GRID, ValueType.GRID, ()),
    PrimitiveSpec("rotate180", ValueType.GRID, ValueType.GRID, ()),
    PrimitiveSpec("rotate270", ValueType.GRID, ValueType.GRID, ()),
    PrimitiveSpec("flip_horizontal", ValueType.GRID, ValueType.GRID, ()),
    PrimitiveSpec("flip_vertical", ValueType.GRID, ValueType.GRID, ()),
    PrimitiveSpec("transpose", ValueType.GRID, ValueType.GRID, ()),
    PrimitiveSpec("anti_transpose", ValueType.GRID, ValueType.GRID, ()),
    PrimitiveSpec(
        "scale_pixels",
        ValueType.GRID,
        ValueType.GRID,
        (
            ("row_factor", ValueType.POSITIVE_INT),
            ("column_factor", ValueType.POSITIVE_INT),
        ),
    ),
    PrimitiveSpec(
        "tile_grid",
        ValueType.GRID,
        ValueType.GRID,
        (
            ("row_repeats", ValueType.POSITIVE_INT),
            ("column_repeats", ValueType.POSITIVE_INT),
        ),
    ),
    PrimitiveSpec(
        "overlay_panel_grid",
        ValueType.GRID,
        ValueType.GRID,
        (("background", ValueType.COLOR),),
    ),
    PrimitiveSpec(
        "broadcast_panel_sequence_d4",
        ValueType.GRID,
        ValueType.GRID,
        (("background", ValueType.COLOR), ("step", ValueType.D4_STEP)),
    ),
    PrimitiveSpec(
        "broadcast_panel_lattice_periodic",
        ValueType.GRID,
        ValueType.GRID,
        (
            ("background", ValueType.COLOR),
            ("row_period", ValueType.POSITIVE_INT),
            ("column_period", ValueType.POSITIVE_INT),
        ),
    ),
    PrimitiveSpec(
        "paint_bbox_contacts",
        ValueType.GRID,
        ValueType.GRID,
        (("background", ValueType.COLOR),),
    ),
    PrimitiveSpec(
        "recolor",
        ValueType.GRID,
        ValueType.GRID,
        (("old", ValueType.COLOR), ("new", ValueType.COLOR)),
    ),
    PrimitiveSpec(
        "map_colors",
        ValueType.GRID,
        ValueType.GRID,
        (("pairs", ValueType.COLOR_MAP),),
    ),
    PrimitiveSpec(
        "crop_non_background",
        ValueType.GRID,
        ValueType.GRID,
        (("background", ValueType.COLOR),),
    ),
    PrimitiveSpec(
        "crop_largest_object",
        ValueType.GRID,
        ValueType.GRID,
        (
            ("background", ValueType.COLOR),
            ("connectivity", ValueType.CONNECTIVITY),
            ("color_mode", ValueType.COLOR_MODE),
        ),
    ),
    PrimitiveSpec(
        "keep_largest_object",
        ValueType.GRID,
        ValueType.GRID,
        (
            ("background", ValueType.COLOR),
            ("connectivity", ValueType.CONNECTIVITY),
            ("color_mode", ValueType.COLOR_MODE),
        ),
    ),
    PrimitiveSpec(
        "render_foreground_bbox",
        ValueType.GRID,
        ValueType.GRID,
        (("background", ValueType.COLOR), ("color", ValueType.COLOR)),
    ),
)
_SPEC_BY_NAME = {spec.name: spec for spec in _SPECS}


def primitive_registry() -> tuple[PrimitiveSpec, ...]:
    return _SPECS


def primitive_spec(name: str) -> PrimitiveSpec:
    try:
        return _SPEC_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"unknown DSL primitive: {name}") from exc


def _parse_arguments(serialized: object) -> dict[str, Any]:
    if not isinstance(serialized, str):
        raise TypeError("arguments_json must be a string")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate argument key: {key!r}")
            result[key] = value
        return result

    parsed = json.loads(serialized, object_pairs_hook=reject_duplicate)
    if not isinstance(parsed, dict):
        raise TypeError("instruction arguments must be a JSON object")
    return parsed


def _validate_arguments(spec: PrimitiveSpec, arguments: dict[str, Any]) -> dict[str, Any]:
    expected = tuple(name for name, _ in spec.argument_types)
    if set(arguments) != set(expected):
        raise ValueError(
            f"{spec.name} requires arguments {expected}, found {tuple(sorted(arguments))}"
        )
    normalized: dict[str, Any] = {}
    for name, value_type in spec.argument_types:
        value = arguments[name]
        if value_type is ValueType.COLOR:
            if type(value) is not int or not 0 <= value <= 9:
                raise ValueError(f"{name} must be an ARC color")
            normalized[name] = value
        elif value_type is ValueType.CONNECTIVITY:
            if type(value) is not int or value not in {4, 8}:
                raise ValueError(f"{name} must be 4 or 8")
            normalized[name] = value
        elif value_type is ValueType.COLOR_MODE:
            try:
                normalized[name] = ColorMode(value).value
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a valid ColorMode") from exc
        elif value_type is ValueType.COLOR_MAP:
            if not isinstance(value, (tuple, list)):
                raise TypeError("color-map pairs must be an ordered list")
            pairs: list[list[int]] = []
            seen: set[int] = set()
            for pair in value:
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise TypeError("each color-map entry must contain two colors")
                old, new = pair
                if (
                    type(old) is not int
                    or type(new) is not int
                    or not 0 <= old <= 9
                    or not 0 <= new <= 9
                ):
                    raise ValueError("color-map entries must contain ARC colors")
                if old in seen:
                    raise ValueError("color-map source colors must be unique")
                seen.add(old)
                pairs.append([old, new])
            normalized[name] = sorted(pairs)
        elif value_type is ValueType.POSITIVE_INT:
            if type(value) is not int or not 1 <= value <= max(
                ARC_MAX_HEIGHT, ARC_MAX_WIDTH
            ):
                raise ValueError(f"{name} must be an integer in [1, 30]")
            normalized[name] = value
        elif value_type is ValueType.D4_STEP:
            try:
                normalized[name] = D4Step(value).value
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a valid D4 element") from exc
        else:
            raise AssertionError(f"unhandled argument type: {value_type}")
    return normalized


def _program_id(instructions: tuple[Instruction, ...]) -> str:
    payload = {
        "dsl_semantics_version": DSL_SEMANTICS_VERSION,
        "instructions": [instruction.to_json_dict() for instruction in instructions],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _crop(grid: Grid, *, top: int, left: int, bottom: int, right: int) -> Grid:
    return tuple(tuple(row[left : right + 1]) for row in grid[top : bottom + 1])


def _unique_largest(objects: tuple[ObjectView, ...]) -> tuple[ObjectView | None, InvalidCode | None]:
    if not objects:
        return None, InvalidCode.EMPTY_SELECTION
    largest_area = max(item.area for item in objects)
    largest = tuple(item for item in objects if item.area == largest_area)
    if len(largest) != 1:
        return None, InvalidCode.NON_UNIQUE_SELECTION
    return largest[0], None


def _object_view(grid: Grid, arguments: dict[str, Any]) -> tuple[ObjectView | None, InvalidCode | None]:
    hypothesis = parse_grid(
        grid,
        backgrounds=(arguments["background"],),
        connectivities=(arguments["connectivity"],),
        color_modes=(ColorMode(arguments["color_mode"]),),
        max_backgrounds=None,
    ).hypotheses[0]
    return _unique_largest(hypothesis.objects)


def _execute_instruction(
    instruction: Instruction, grid: Grid
) -> tuple[Grid | None, InvalidCode | None]:
    arguments = instruction.arguments
    op = instruction.op
    geometry: dict[str, Callable[[Grid], Grid]] = {
        "identity": lambda value: value,
        "rotate90": rotate90,
        "rotate180": rotate180,
        "rotate270": rotate270,
        "flip_horizontal": flip_horizontal,
        "flip_vertical": flip_vertical,
        "transpose": transpose,
        "anti_transpose": lambda value: rotate180(transpose(value)),
    }
    if op in geometry:
        return geometry[op](grid), None
    if op in {"scale_pixels", "tile_grid"}:
        if op == "scale_pixels":
            row_count = arguments["row_factor"]
            column_count = arguments["column_factor"]
        else:
            row_count = arguments["row_repeats"]
            column_count = arguments["column_repeats"]
        if (
            len(grid) * row_count > ARC_MAX_HEIGHT
            or len(grid[0]) * column_count > ARC_MAX_WIDTH
        ):
            return None, InvalidCode.OUTPUT_SHAPE_EXCEEDS_LIMIT
        if op == "scale_pixels":
            expanded_rows = tuple(
                tuple(cell for cell in row for _ in range(column_count))
                for row in grid
            )
            return tuple(
                row for row in expanded_rows for _ in range(row_count)
            ), None
        tiled_rows = tuple(row * column_count for row in grid)
        return tiled_rows * row_count, None
    if op == "overlay_panel_grid":
        output, invalid = overlay_panel_grid(
            grid, background=arguments["background"]
        )
        if invalid is None:
            return output, None
        invalid_map = {
            PanelOverlayCode.EMPTY_SELECTION: InvalidCode.EMPTY_SELECTION,
            PanelOverlayCode.NON_UNIQUE_SELECTION: InvalidCode.NON_UNIQUE_SELECTION,
            PanelOverlayCode.INCOMPATIBLE_PANEL_SHAPES: (
                InvalidCode.INCOMPATIBLE_PANEL_SHAPES
            ),
        }
        return None, invalid_map[invalid]
    if op == "broadcast_panel_sequence_d4":
        output, invalid = broadcast_panel_sequence_d4(
            grid,
            background=arguments["background"],
            step=arguments["step"],
        )
        if invalid is None:
            return output, None
        invalid_map = {
            PanelSequenceD4Code.EMPTY_SELECTION: InvalidCode.EMPTY_SELECTION,
            PanelSequenceD4Code.NON_UNIQUE_SELECTION: (
                InvalidCode.NON_UNIQUE_SELECTION
            ),
            PanelSequenceD4Code.INCOMPATIBLE_PANEL_SHAPES: (
                InvalidCode.INCOMPATIBLE_PANEL_SHAPES
            ),
        }
        return None, invalid_map[invalid]
    if op == "broadcast_panel_lattice_periodic":
        output, invalid = broadcast_panel_lattice_periodic(
            grid,
            background=arguments["background"],
            row_period=arguments["row_period"],
            column_period=arguments["column_period"],
        )
        if invalid is None:
            return output, None
        invalid_map = {
            PanelLatticePeriodicCode.EMPTY_SELECTION: (
                InvalidCode.EMPTY_SELECTION
            ),
            PanelLatticePeriodicCode.NON_UNIQUE_SELECTION: (
                InvalidCode.NON_UNIQUE_SELECTION
            ),
            PanelLatticePeriodicCode.INCOMPATIBLE_PANEL_SHAPES: (
                InvalidCode.INCOMPATIBLE_PANEL_SHAPES
            ),
        }
        return None, invalid_map[invalid]
    if op == "paint_bbox_contacts":
        output, invalid = paint_bbox_contacts(
            grid,
            background=arguments["background"],
        )
        if invalid is None:
            return output, None
        invalid_map = {
            BBoxContactCode.EMPTY_SELECTION: InvalidCode.EMPTY_SELECTION,
            BBoxContactCode.NON_UNIQUE_SELECTION: (
                InvalidCode.NON_UNIQUE_SELECTION
            ),
            BBoxContactCode.TARGET_COLLISION: InvalidCode.TARGET_COLLISION,
            BBoxContactCode.OCCLUDED_RAY: InvalidCode.OCCLUDED_RAY,
            BBoxContactCode.INCOMPATIBLE_RELATION_GEOMETRY: (
                InvalidCode.INCOMPATIBLE_RELATION_GEOMETRY
            ),
        }
        return None, invalid_map[invalid]
    if op == "recolor":
        return tuple(
            tuple(arguments["new"] if cell == arguments["old"] else cell for cell in row)
            for row in grid
        ), None
    if op == "map_colors":
        mapping = {old: new for old, new in arguments["pairs"]}
        return tuple(tuple(mapping.get(cell, cell) for cell in row) for row in grid), None
    if op == "crop_non_background":
        coordinates = tuple(
            (row, column)
            for row, values in enumerate(grid)
            for column, cell in enumerate(values)
            if cell != arguments["background"]
        )
        if not coordinates:
            return None, InvalidCode.EMPTY_SELECTION
        rows = tuple(row for row, _ in coordinates)
        columns = tuple(column for _, column in coordinates)
        return _crop(
            grid,
            top=min(rows),
            left=min(columns),
            bottom=max(rows),
            right=max(columns),
        ), None
    if op in {"crop_largest_object", "keep_largest_object"}:
        selected, invalid = _object_view(grid, arguments)
        if invalid is not None or selected is None:
            return None, invalid
        if op == "crop_largest_object":
            return _crop(
                grid,
                top=selected.bbox.top,
                left=selected.bbox.left,
                bottom=selected.bbox.bottom,
                right=selected.bbox.right,
            ), None
        output = [
            [arguments["background"] for _ in range(len(grid[0]))]
            for _ in range(len(grid))
        ]
        for row, column in selected.pixels:
            output[row][column] = grid[row][column]
        return as_grid(output), None
    if op == "render_foreground_bbox":
        coordinates = tuple(
            (row, column)
            for row, values in enumerate(grid)
            for column, cell in enumerate(values)
            if cell != arguments["background"]
        )
        if not coordinates:
            return None, InvalidCode.EMPTY_SELECTION
        rows = tuple(row for row, _ in coordinates)
        columns = tuple(column for _, column in coordinates)
        top, left, bottom, right = min(rows), min(columns), max(rows), max(columns)
        output = [
            [arguments["background"] for _ in range(len(grid[0]))]
            for _ in range(len(grid))
        ]
        for row in range(top, bottom + 1):
            for column in range(left, right + 1):
                output[row][column] = arguments["color"]
        return as_grid(output), None
    raise AssertionError(f"executor missing primitive: {op}")


def execute_program(program: Program, grid: Grid) -> ExecutionOutcome:
    current = as_grid(grid)
    trace: list[tuple[str, str]] = []
    try:
        for index, instruction in enumerate(program.instructions):
            output, invalid = _execute_instruction(instruction, current)
            if invalid is not None or output is None:
                trace.append((instruction.op, f"invalid:{invalid.value if invalid else 'unknown'}"))
                return ExecutionOutcome(
                    status="invalid",
                    output=None,
                    invalid_code=invalid or InvalidCode.INTERNAL_ERROR,
                    failing_instruction_index=index,
                    semantic_trace=tuple(trace),
                )
            current = as_grid(output)
            trace.append((instruction.op, grid_key(current)))
    except Exception:
        return ExecutionOutcome(
            status="invalid",
            output=None,
            invalid_code=InvalidCode.INTERNAL_ERROR,
            failing_instruction_index=len(trace),
            semantic_trace=tuple(trace),
        )
    return ExecutionOutcome(
        status="ok",
        output=current,
        invalid_code=None,
        failing_instruction_index=None,
        semantic_trace=tuple(trace),
    )
