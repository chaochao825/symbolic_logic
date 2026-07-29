"""Replayable object/code programs and typed failure-directed frontier changes.

The module is opt-in.  It does not change the legacy scene DSL, residual schema,
or default online solver.  Candidate generation may inspect demonstration outputs
but receives only :class:`~afts_arc.blind.BlindTask`, so query outputs are never in
scope.  Every emitted program is serialized, content addressed, reconstructed by
its verifier, and replayed on a whole task.

Two deliberately small program families provide the first auditable slice:

* ``role_stamp`` extracts a payload relative to a unique source anchor and stamps
  it at target anchors; and
* ``d4_label_completion`` transfers missing annotations between D4-congruent
  structural objects while preserving observed cells.

Near misses are retained so a failure certificate can name a type-legal,
representation-specific frontier action instead of merely reporting pixel error.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

from ..blind import BlindTask
from ..grid import Grid, as_grid, grid_key
from ..residual import compare_grids
from .control import (
    ControlAction,
    ResidualActionCompiler,
    ResidualCompilerConfig,
)
from .router import RouteDecision, TaskFeatures
from .scene_graph import (
    SCENE_AST_VERSION,
    SCENE_PIPELINE_DSL_VERSION,
    ExecutionTraceNode,
    SceneGraph,
    ScenePipelineExecution,
    ScenePipelineProgram,
    enumerate_scene_pipeline_programs,
    execute_scene_pipeline,
    extract_scene_graph,
)
from .types import (
    CandidateEvaluation,
    CandidateHypothesis,
    ProviderResult,
    canonical_json,
)

if TYPE_CHECKING:
    from .control import Blackboard


OBJECT_CODE_DSL_VERSION = "afts-object-code-dsl/v0.1"
OBJECT_CODE_DSL_VERSIONS = (
    OBJECT_CODE_DSL_VERSION,
    SCENE_PIPELINE_DSL_VERSION,
)
OBJECT_CODE_PROVIDER_VERSION = "afts-hybrid-object-code/v0.3"
FAILURE_CERTIFICATE_VERSION = "afts-object-code-failure-certificate/v0.1"

ROLE_STAMP_CANVASES = ("blank", "erase_roles", "copy", "crop")
D4_TRANSFORMS = (
    "identity",
    "rotate90",
    "rotate180",
    "rotate270",
    "flip_horizontal",
    "flip_vertical",
    "transpose",
    "anti_transpose",
)
TYPED_OBJECT_CODE_ACTIONS = frozenset(
    {"object_rematch", "canvas_reinfer", "fill_ast_hole", "reparse_background"}
)

Coordinate = tuple[int, int]


def _content_id(payload: object, *, length: int = 24) -> str:
    return hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()[:length]


def _arc_color(value: object, *, field_name: str) -> int:
    if type(value) is not int or not 0 <= value <= 9:
        raise ValueError(f"{field_name} must be an ARC color")
    return value


def _transform_offset(coordinate: Coordinate, transform: str) -> Coordinate:
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
        raise ValueError("unknown D4 transform") from exc


@dataclass(frozen=True, slots=True)
class RoleStampProgram:
    """Extract one colored payload relative to a role and stamp it at peers."""

    background: int
    payload_color: int
    source_anchor_color: int
    target_anchor_color: int
    transform: str = "identity"
    canvas_mode: str = "blank"

    def __post_init__(self) -> None:
        colors = tuple(
            _arc_color(getattr(self, name), field_name=name)
            for name in (
                "background",
                "payload_color",
                "source_anchor_color",
                "target_anchor_color",
            )
        )
        if len(set(colors)) != len(colors):
            raise ValueError("role-stamp colors must denote distinct roles")
        if self.transform not in D4_TRANSFORMS:
            raise ValueError("unknown role-stamp transform")
        if self.canvas_mode not in ROLE_STAMP_CANVASES:
            raise ValueError("unknown role-stamp canvas mode")

    @property
    def description_bits(self) -> int:
        return 16 + 3 + math.ceil(math.log2(len(ROLE_STAMP_CANVASES))) + 2

    @property
    def functional_trace(self) -> tuple[str, ...]:
        return (
            f"parse:roles:bg={self.background}",
            f"bind:source={self.source_anchor_color}:targets={self.target_anchor_color}",
            f"extract:payload={self.payload_color}",
            f"transform:{self.transform}",
            f"render:{self.canvas_mode}",
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "object_code_dsl_version": OBJECT_CODE_DSL_VERSION,
            "kind": "role_stamp",
            "background": self.background,
            "payload_color": self.payload_color,
            "source_anchor_color": self.source_anchor_color,
            "target_anchor_color": self.target_anchor_color,
            "transform": self.transform,
            "canvas_mode": self.canvas_mode,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "RoleStampProgram":
        expected = {
            "object_code_dsl_version",
            "kind",
            "background",
            "payload_color",
            "source_anchor_color",
            "target_anchor_color",
            "transform",
            "canvas_mode",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("role-stamp program has missing or unknown fields")
        if payload["object_code_dsl_version"] != OBJECT_CODE_DSL_VERSION:
            raise ValueError("unsupported object/code DSL version")
        if payload["kind"] != "role_stamp":
            raise ValueError("program kind is not role_stamp")
        return cls(
            background=payload["background"],
            payload_color=payload["payload_color"],
            source_anchor_color=payload["source_anchor_color"],
            target_anchor_color=payload["target_anchor_color"],
            transform=payload["transform"],
            canvas_mode=payload["canvas_mode"],
        )


@dataclass(frozen=True, slots=True)
class D4LabelCompletionProgram:
    """Copy annotations from a complete prototype to congruent structures."""

    background: int
    structure_color: int
    connectivity: int = 4
    attachment_radius: int = 1

    def __post_init__(self) -> None:
        _arc_color(self.background, field_name="background")
        _arc_color(self.structure_color, field_name="structure_color")
        if self.background == self.structure_color:
            raise ValueError("background and structure colors must differ")
        if type(self.connectivity) is not int or self.connectivity not in {4, 8}:
            raise ValueError("object connectivity must be 4 or 8")
        if type(self.attachment_radius) is not int or self.attachment_radius not in {
            1,
            2,
        }:
            raise ValueError("annotation attachment radius must be 1 or 2")

    @property
    def description_bits(self) -> int:
        return 8 + 1 + 1 + 3

    @property
    def functional_trace(self) -> tuple[str, ...]:
        return (
            f"parse:structure:bg={self.background}:color={self.structure_color}:c={self.connectivity}",
            f"match:d4_congruent:radius={self.attachment_radius}",
            "bind:most_annotated_prototype",
            "render:fill_background_only",
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "object_code_dsl_version": OBJECT_CODE_DSL_VERSION,
            "kind": "d4_label_completion",
            "background": self.background,
            "structure_color": self.structure_color,
            "connectivity": self.connectivity,
            "attachment_radius": self.attachment_radius,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "D4LabelCompletionProgram":
        expected = {
            "object_code_dsl_version",
            "kind",
            "background",
            "structure_color",
            "connectivity",
            "attachment_radius",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("D4 completion program has missing or unknown fields")
        if payload["object_code_dsl_version"] != OBJECT_CODE_DSL_VERSION:
            raise ValueError("unsupported object/code DSL version")
        if payload["kind"] != "d4_label_completion":
            raise ValueError("program kind is not d4_label_completion")
        return cls(
            background=payload["background"],
            structure_color=payload["structure_color"],
            connectivity=payload["connectivity"],
            attachment_radius=payload["attachment_radius"],
        )


ObjectCodeProgram: TypeAlias = (
    RoleStampProgram | D4LabelCompletionProgram | ScenePipelineProgram
)


def object_code_program_from_json(payload: object) -> ObjectCodeProgram:
    if not isinstance(payload, Mapping):
        raise TypeError("object/code program must be a JSON object")
    kind = payload.get("kind")
    if kind == "role_stamp":
        return RoleStampProgram.from_json_dict(payload)
    if kind == "d4_label_completion":
        return D4LabelCompletionProgram.from_json_dict(payload)
    if kind == "scene_pipeline":
        return ScenePipelineProgram.from_json_dict(payload)
    raise ValueError("unknown object/code program kind")


@dataclass(frozen=True, slots=True)
class ObjectCodeExecution:
    status: str
    output: Grid | None
    reason: str | None
    object_count: int = 0
    correspondence_count: int = 0
    node_trace: tuple[ExecutionTraceNode, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"ok", "invalid"}:
            raise ValueError("object/code execution status must be ok or invalid")
        if (self.status == "ok") != (self.output is not None):
            raise ValueError("only successful executions may carry an output")

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _execute_role_stamp(program: RoleStampProgram, grid: Grid) -> ObjectCodeExecution:
    normalized = as_grid(grid)
    height, width = len(normalized), len(normalized[0])
    sources = tuple(
        (row, column)
        for row in range(height)
        for column in range(width)
        if normalized[row][column] == program.source_anchor_color
    )
    targets = tuple(
        (row, column)
        for row in range(height)
        for column in range(width)
        if normalized[row][column] == program.target_anchor_color
    )
    payload = tuple(
        (row, column)
        for row in range(height)
        for column in range(width)
        if normalized[row][column] == program.payload_color
    )
    if len(sources) != 1:
        return ObjectCodeExecution("invalid", None, "source_anchor_not_unique")
    if not targets:
        return ObjectCodeExecution("invalid", None, "target_anchor_missing")
    if not payload:
        return ObjectCodeExecution("invalid", None, "payload_missing")

    source = sources[0]
    offsets = tuple(
        _transform_offset((row - source[0], column - source[1]), program.transform)
        for row, column in payload
    )
    stamped = tuple(
        (target[0] + offset[0], target[1] + offset[1])
        for target in targets
        for offset in offsets
    )
    if any(
        row < 0 or row >= height or column < 0 or column >= width
        for row, column in stamped
    ):
        return ObjectCodeExecution("invalid", None, "stamp_out_of_bounds")

    if program.canvas_mode in {"blank", "crop"}:
        canvas = [[program.background for _ in range(width)] for _ in range(height)]
    else:
        canvas = [list(row) for row in normalized]
        if program.canvas_mode == "erase_roles":
            roles = {
                program.payload_color,
                program.source_anchor_color,
                program.target_anchor_color,
            }
            for row in range(height):
                for column in range(width):
                    if canvas[row][column] in roles:
                        canvas[row][column] = program.background
    for row, column in stamped:
        canvas[row][column] = program.payload_color

    if program.canvas_mode == "crop":
        foreground = tuple(
            (row, column)
            for row in range(height)
            for column in range(width)
            if canvas[row][column] != program.background
        )
        if not foreground:
            return ObjectCodeExecution("invalid", None, "empty_crop")
        top = min(row for row, _ in foreground)
        bottom = max(row for row, _ in foreground)
        left = min(column for _, column in foreground)
        right = max(column for _, column in foreground)
        canvas = [row[left : right + 1] for row in canvas[top : bottom + 1]]
    return ObjectCodeExecution(
        "ok",
        as_grid(canvas),
        None,
        object_count=1 + len(targets),
        correspondence_count=len(targets),
    )


_DIRECTIONS = {
    4: ((-1, 0), (0, -1), (0, 1), (1, 0)),
    8: tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)),
}


def _color_components(
    grid: Grid, *, color: int, connectivity: int
) -> tuple[tuple[Coordinate, ...], ...]:
    height, width = len(grid), len(grid[0])
    unseen = {
        (row, column)
        for row in range(height)
        for column in range(width)
        if grid[row][column] == color
    }
    components: list[tuple[Coordinate, ...]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue: deque[Coordinate] = deque((start,))
        component: list[Coordinate] = []
        while queue:
            row, column = queue.popleft()
            component.append((row, column))
            for dr, dc in _DIRECTIONS[connectivity]:
                neighbor = (row + dr, column + dc)
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda item: (min(item), len(item), item)))


def _distance_to_component(
    coordinate: Coordinate, component: Sequence[Coordinate]
) -> int:
    row, column = coordinate
    return min(max(abs(row - rr), abs(column - cc)) for rr, cc in component)


def _assigned_annotations(
    grid: Grid,
    components: Sequence[tuple[Coordinate, ...]],
    *,
    background: int,
    structure_color: int,
    radius: int,
) -> tuple[dict[Coordinate, int], ...]:
    assigned = [dict() for _ in components]
    for row, values in enumerate(grid):
        for column, color in enumerate(values):
            if color in {background, structure_color}:
                continue
            distances = tuple(
                _distance_to_component((row, column), component)
                for component in components
            )
            best = min(distances, default=radius + 1)
            if best > radius or distances.count(best) != 1:
                continue
            assigned[distances.index(best)][(row, column)] = color
    return tuple(assigned)


def _normalized_shape(component: Sequence[Coordinate]) -> tuple[Coordinate, ...]:
    top = min(row for row, _ in component)
    left = min(column for _, column in component)
    return tuple(sorted((row - top, column - left) for row, column in component))


def _canonical_d4_shape(component: Sequence[Coordinate]) -> tuple[Coordinate, ...]:
    normalized = _normalized_shape(component)
    variants = []
    for transform in D4_TRANSFORMS:
        transformed = tuple(_transform_offset(item, transform) for item in normalized)
        min_row = min(row for row, _ in transformed)
        min_column = min(column for _, column in transformed)
        variants.append(
            tuple(
                sorted(
                    (row - min_row, column - min_column) for row, column in transformed
                )
            )
        )
    return min(variants)


def _transfer_for_transform(
    prototype: Sequence[Coordinate],
    prototype_annotations: Mapping[Coordinate, int],
    target: Sequence[Coordinate],
    transform: str,
) -> dict[Coordinate, int] | None:
    proto_top = min(row for row, _ in prototype)
    proto_left = min(column for _, column in prototype)
    relative_base = tuple(
        (row - proto_top, column - proto_left) for row, column in prototype
    )
    transformed_base = tuple(
        _transform_offset(item, transform) for item in relative_base
    )
    transformed_top = min(row for row, _ in transformed_base)
    transformed_left = min(column for _, column in transformed_base)
    target_top = min(row for row, _ in target)
    target_left = min(column for _, column in target)

    def place(coordinate: Coordinate) -> Coordinate:
        transformed = _transform_offset(
            (coordinate[0] - proto_top, coordinate[1] - proto_left), transform
        )
        return (
            transformed[0] - transformed_top + target_top,
            transformed[1] - transformed_left + target_left,
        )

    if {place(item) for item in prototype} != set(target):
        return None
    return {
        place(coordinate): color for coordinate, color in prototype_annotations.items()
    }


def _execute_d4_completion(
    program: D4LabelCompletionProgram, grid: Grid
) -> ObjectCodeExecution:
    normalized = as_grid(grid)
    components = _color_components(
        normalized,
        color=program.structure_color,
        connectivity=program.connectivity,
    )
    if len(components) < 2:
        return ObjectCodeExecution("invalid", None, "too_few_structural_objects")
    annotations = _assigned_annotations(
        normalized,
        components,
        background=program.background,
        structure_color=program.structure_color,
        radius=program.attachment_radius,
    )
    groups: dict[tuple[Coordinate, ...], list[int]] = {}
    for index, component in enumerate(components):
        groups.setdefault(_canonical_d4_shape(component), []).append(index)

    canvas = [list(row) for row in normalized]
    transfers = 0
    correspondences = 0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        annotation_counts = [len(annotations[index]) for index in indices]
        maximum = max(annotation_counts)
        prototypes = [
            index
            for index, count in zip(indices, annotation_counts)
            if count == maximum
        ]
        if maximum == 0 or len(prototypes) != 1:
            continue
        prototype_index = prototypes[0]
        prototype = components[prototype_index]
        prototype_annotations = annotations[prototype_index]
        for target_index in indices:
            if target_index == prototype_index:
                continue
            target_annotations = annotations[target_index]
            predictions: dict[
                tuple[tuple[int, int, int], ...], dict[Coordinate, int]
            ] = {}
            for transform in D4_TRANSFORMS:
                prediction = _transfer_for_transform(
                    prototype,
                    prototype_annotations,
                    components[target_index],
                    transform,
                )
                if prediction is None or any(
                    prediction.get(coordinate) != color
                    for coordinate, color in target_annotations.items()
                ):
                    continue
                signature = tuple(
                    sorted(
                        (row, column, color)
                        for (row, column), color in prediction.items()
                    )
                )
                predictions[signature] = prediction
            if len(predictions) != 1:
                continue
            prediction = next(iter(predictions.values()))
            height, width = len(normalized), len(normalized[0])
            if any(
                row < 0 or row >= height or column < 0 or column >= width
                for row, column in prediction
            ):
                continue
            conflict = False
            for (row, column), color in prediction.items():
                observed = canvas[row][column]
                if observed not in {program.background, color}:
                    conflict = True
                    break
            if conflict:
                continue
            before = sum(
                canvas[row][column] == program.background for row, column in prediction
            )
            for (row, column), color in prediction.items():
                if canvas[row][column] == program.background:
                    canvas[row][column] = color
            transfers += before
            correspondences += 1
    if transfers == 0:
        return ObjectCodeExecution(
            "invalid",
            None,
            "no_unambiguous_incomplete_congruent_motif",
            object_count=len(components),
        )
    return ObjectCodeExecution(
        "ok",
        as_grid(canvas),
        None,
        object_count=len(components),
        correspondence_count=correspondences,
    )


def execute_object_code_program(
    program: ObjectCodeProgram, grid: Grid
) -> ObjectCodeExecution:
    try:
        if isinstance(program, RoleStampProgram):
            return _execute_role_stamp(program, grid)
        if isinstance(program, D4LabelCompletionProgram):
            return _execute_d4_completion(program, grid)
        if isinstance(program, ScenePipelineProgram):
            result = execute_scene_pipeline(program, grid)
            return _object_code_execution_from_scene(result)
        raise TypeError("unknown object/code program type")
    except Exception:
        return ObjectCodeExecution("invalid", None, "internal_error")


def _object_code_execution_from_scene(
    result: ScenePipelineExecution,
) -> ObjectCodeExecution:
    return ObjectCodeExecution(
        result.status,
        result.output,
        result.reason,
        result.object_count,
        result.correspondence_count,
        result.node_trace,
    )


def _execute_scored_program(
    program: ObjectCodeProgram,
    grid: Grid,
    scene_cache: dict[tuple[object, ...], SceneGraph],
) -> ObjectCodeExecution:
    if not isinstance(program, ScenePipelineProgram):
        return execute_object_code_program(program, grid)
    key = (
        grid_key(grid),
        program.parse.background,
        program.parse.connectivity,
        program.parse.grouping,
    )
    scene = scene_cache.get(key)
    if scene is None:
        scene = extract_scene_graph(
            grid,
            background=program.parse.background,
            connectivity=program.parse.connectivity,
            grouping=program.parse.grouping,
        )
        scene_cache[key] = scene
    result = execute_scene_pipeline(program, grid, precomputed_scene=scene)
    return _object_code_execution_from_scene(result)


def _observable_colors(task: BlindTask) -> tuple[int, ...]:
    counts: Counter[int] = Counter()
    for pair in task.train:
        counts.update(cell for row in pair.input for cell in row)
        assert pair.output is not None
        counts.update(cell for row in pair.output for cell in row)
    for grid in task.test_inputs:
        counts.update(cell for row in grid for cell in row)
    return tuple(sorted(counts, key=lambda color: (-counts[color], color)))


def _candidate_backgrounds(task: BlindTask, *, limit: int = 3) -> tuple[int, ...]:
    colors = _observable_colors(task)
    input_grids = tuple(pair.input for pair in task.train) + task.test_inputs
    modal = Counter(
        min(
            Counter(cell for row in grid for cell in row),
            key=lambda color: (
                -Counter(cell for row in grid for cell in row)[color],
                color,
            ),
        )
        for grid in input_grids
    )
    ordered = sorted(colors, key=lambda color: (-modal[color], color != 0, color))
    return tuple(ordered[:limit])


def enumerate_object_code_programs(task: BlindTask) -> tuple[ObjectCodeProgram, ...]:
    """Enumerate a deterministic, task-derived grammar without query labels."""

    if not isinstance(task, BlindTask):
        raise TypeError("object/code synthesis accepts BlindTask only")
    colors = _observable_colors(task)
    demo_input_counts = tuple(
        Counter(cell for row in pair.input for cell in row) for pair in task.train
    )
    demo_output_counts = tuple(
        Counter(cell for row in (pair.output or ()) for cell in row)
        for pair in task.train
    )
    programs: list[ObjectCodeProgram] = []
    for background in _candidate_backgrounds(task):
        # Role persistence is a canvas decision, not a parsing prerequisite.
        # Prefer disappearing roles for the blank/erase lanes, but retain a
        # bounded set of persistent roles so the copy lane is reachable.
        source_colors = tuple(
            sorted(
                (
                    color
                    for color in colors
                    if color != background
                    and all(counts[color] == 1 for counts in demo_input_counts)
                ),
                key=lambda color: (
                    not all(counts[color] == 0 for counts in demo_output_counts),
                    sum(counts[color] for counts in demo_output_counts),
                    color,
                ),
            )[:3]
        )
        target_colors = tuple(
            sorted(
                (
                    color
                    for color in colors
                    if color != background
                    and all(counts[color] >= 1 for counts in demo_input_counts)
                ),
                key=lambda color: (
                    not all(counts[color] == 0 for counts in demo_output_counts),
                    sum(counts[color] for counts in demo_output_counts),
                    -sum(counts[color] for counts in demo_input_counts),
                    color,
                ),
            )[:4]
        )
        payload_colors = tuple(
            sorted(
                (
                    color
                    for color in colors
                    if color != background
                    and all(counts[color] >= 1 for counts in demo_input_counts)
                    and all(counts[color] >= 1 for counts in demo_output_counts)
                ),
                key=lambda color: (
                    -sum(counts[color] for counts in demo_output_counts),
                    -sum(counts[color] for counts in demo_input_counts),
                    color,
                ),
            )[:4]
        )
        for payload_color in payload_colors:
            for source_color in source_colors:
                for target_color in target_colors:
                    if (
                        len({background, payload_color, source_color, target_color})
                        != 4
                    ):
                        continue
                    for transform in D4_TRANSFORMS:
                        for canvas_mode in ROLE_STAMP_CANVASES:
                            programs.append(
                                RoleStampProgram(
                                    background,
                                    payload_color,
                                    source_color,
                                    target_color,
                                    transform,
                                    canvas_mode,
                                )
                            )

        structure_rank = sorted(
            (
                color
                for color in colors
                if color != background
                and all(counts[color] >= 2 for counts in demo_input_counts)
            ),
            key=lambda color: (
                -sum(counts[color] for counts in demo_input_counts),
                color,
            ),
        )[:4]
        for structure_color in structure_rank:
            for connectivity in (4, 8):
                for radius in (1, 2):
                    programs.append(
                        D4LabelCompletionProgram(
                            background,
                            structure_color,
                            connectivity,
                            radius,
                        )
                    )
    # Preserve the complete v0.2 grammar as an identical prefix.  This keeps
    # legacy trial order and capped-search behavior replayable while the v0.3
    # scene family is appended as an opt-in representation extension.
    legacy_unique = {
        canonical_json(program.to_json_dict()): program for program in programs
    }
    scene_programs = enumerate_scene_pipeline_programs(task)
    scene_unique = {
        canonical_json(program.to_json_dict()): program for program in scene_programs
    }
    return (
        *(legacy_unique[key] for key in sorted(legacy_unique)),
        *(scene_unique[key] for key in sorted(scene_unique)),
    )


@dataclass(frozen=True, slots=True)
class ObjectCodeProgramScore:
    program: ObjectCodeProgram
    demo_outputs: tuple[Grid | None, ...]
    exact_demo_count: int
    shape_match_count: int
    agreement: float
    mismatch_count: int
    execution_valid: bool

    @property
    def all_demo_exact(self) -> bool:
        return self.execution_valid and self.exact_demo_count == len(self.demo_outputs)


def _score_program(
    program: ObjectCodeProgram,
    task: BlindTask,
    scene_cache: dict[tuple[object, ...], SceneGraph],
) -> ObjectCodeProgramScore:
    outputs: list[Grid | None] = []
    exact_count = 0
    shape_count = 0
    matches = 0
    cells = 0
    mismatches = 0
    for index, pair in enumerate(task.train):
        result = _execute_scored_program(program, pair.input, scene_cache)
        output = result.output if result.ok else None
        outputs.append(output)
        residual = compare_grids(
            output,
            pair.output,
            pair_index=index,
            invalid_code=result.reason,
        )
        exact_count += int(residual.exact)
        shape_count += int(residual.shape_match)
        matches += residual.overlap_matches
        cells += residual.comparison_cells
        mismatches += residual.mismatch_count
    return ObjectCodeProgramScore(
        program,
        tuple(outputs),
        exact_count,
        shape_count,
        matches / cells if cells else 0.0,
        mismatches,
        all(output is not None for output in outputs),
    )


@dataclass(frozen=True, slots=True)
class ObjectCodeSynthesisResult:
    exact_scores: tuple[ObjectCodeProgramScore, ...]
    near_miss_scores: tuple[ObjectCodeProgramScore, ...]
    program_trial_count: int
    demo_execution_count: int
    query_execution_count: int
    invalid_program_count: int
    semantic_duplicate_count: int


def synthesize_object_code_programs(
    task: BlindTask,
    *,
    max_program_trials: int = 20_000,
    max_exact_programs: int = 32,
    max_near_misses: int = 8,
    minimum_near_miss_agreement: float = 0.2,
    programs: Sequence[ObjectCodeProgram] | None = None,
) -> ObjectCodeSynthesisResult:
    if not isinstance(task, BlindTask):
        raise TypeError("object/code synthesis accepts BlindTask only")
    for name, value in (
        ("max_program_trials", max_program_trials),
        ("max_exact_programs", max_exact_programs),
        ("max_near_misses", max_near_misses),
    ):
        if type(value) is not int or value < (1 if name != "max_near_misses" else 0):
            raise ValueError(f"{name} has an invalid bound")
    if not 0.0 <= minimum_near_miss_agreement <= 1.0:
        raise ValueError("minimum_near_miss_agreement must be in [0, 1]")

    # ``()`` is a meaningful, exhausted repair frontier.  Do not treat it as a
    # request for a cold restart; only ``None`` selects full enumeration.
    frontier = tuple(
        enumerate_object_code_programs(task) if programs is None else programs
    )[:max_program_trials]
    scene_cache: dict[tuple[object, ...], SceneGraph] = {}
    scores = tuple(_score_program(program, task, scene_cache) for program in frontier)
    exact_by_semantics: dict[tuple[object, ...], ObjectCodeProgramScore] = {}
    query_executions = 0
    duplicates = 0
    for score in scores:
        if not score.all_demo_exact:
            continue
        query_results = tuple(
            _execute_scored_program(score.program, grid, scene_cache)
            for grid in task.test_inputs
        )
        query_executions += len(query_results)
        if any(not result.ok for result in query_results):
            continue
        signature = (
            tuple(
                grid_key(output) for output in score.demo_outputs if output is not None
            ),
            tuple(
                grid_key(result.output)
                for result in query_results
                if result.output is not None
            ),
        )
        incumbent = exact_by_semantics.get(signature)
        key = (
            score.program.description_bits,
            canonical_json(score.program.to_json_dict()),
        )
        if incumbent is None or key < (
            incumbent.program.description_bits,
            canonical_json(incumbent.program.to_json_dict()),
        ):
            duplicates += int(incumbent is not None)
            exact_by_semantics[signature] = score
        else:
            duplicates += 1
    exact = tuple(
        sorted(
            exact_by_semantics.values(),
            key=lambda score: (
                score.program.description_bits,
                canonical_json(score.program.to_json_dict()),
            ),
        )[:max_exact_programs]
    )
    near = tuple(
        sorted(
            (
                score
                for score in scores
                if not score.all_demo_exact
                and score.agreement >= minimum_near_miss_agreement
            ),
            key=lambda score: (
                -score.exact_demo_count,
                -score.shape_match_count,
                -score.agreement,
                score.mismatch_count,
                score.program.description_bits,
                canonical_json(score.program.to_json_dict()),
            ),
        )[:max_near_misses]
    )
    return ObjectCodeSynthesisResult(
        exact,
        near,
        len(frontier),
        len(frontier) * len(task.train),
        query_executions,
        sum(not score.execution_valid for score in scores),
        duplicates,
    )


def make_object_code_hypothesis(
    program: ObjectCodeProgram,
    *,
    demo_exact: bool,
    parent_hypothesis_ids: Sequence[str] = (),
    control_operator: str | None = None,
    ast_holes: Sequence[str] = (),
) -> CandidateHypothesis:
    serialized = program.to_json_dict()
    holes = tuple(sorted(set(ast_holes)))
    valid_fields = set(_program_fields(program))
    if any(field not in valid_fields for field in holes):
        raise ValueError("AST holes must name concrete typed program slots")
    digest = hashlib.sha256(canonical_json(serialized).encode("ascii")).hexdigest()

    def replay(grid: Grid) -> object | None:
        result = execute_object_code_program(program, grid)
        return result.output if result.ok else None

    def hard_verify(task: BlindTask) -> bool:
        try:
            reconstructed = object_code_program_from_json(serialized)
            if reconstructed.to_json_dict() != serialized:
                return False
            for grid in tuple(pair.input for pair in task.train) + task.test_inputs:
                first = execute_object_code_program(program, grid)
                second = execute_object_code_program(reconstructed, grid)
                if first != second or not first.ok:
                    return False
            return True
        except Exception:
            return False

    trace = program.functional_trace
    if control_operator is not None:
        trace = (*trace, f"control:{control_operator}")
    return CandidateHypothesis.create(
        name=f"object_code:{serialized['kind']}:{digest[:14]}",
        source="object_code_dsl",
        source_version=OBJECT_CODE_PROVIDER_VERSION,
        route="code_llm",
        description_bits=program.description_bits + 2 * len(holes),
        verification_mode="replayable",
        functional_trace=trace,
        spec={
            "object_code_program": serialized,
            "object_code_program_id": digest,
            "provisional_ast_holes": list(holes),
        },
        metadata={
            "demo_exact_at_generation": demo_exact,
            "emission_lane": (
                "demo_exact_object_code" if demo_exact else "object_code_near_miss"
            ),
            "control_operator": control_operator,
            "ast_holes": list(holes),
            "scene_ast_version": (
                SCENE_AST_VERSION if isinstance(program, ScenePipelineProgram) else None
            ),
            # A provisional AST may be replayable for diagnosis but cannot pass
            # selection until every declared hole is filled.
            "support_gate_passed": not holes,
        },
        parent_hypothesis_ids=parent_hypothesis_ids,
        replay=replay,
        hard_verifier=hard_verify,
    )


def _program_from_candidate(candidate: CandidateHypothesis) -> ObjectCodeProgram | None:
    payload = candidate.spec.get("object_code_program")
    if payload is None:
        return None
    try:
        return object_code_program_from_json(payload)
    except (TypeError, ValueError):
        return None


def object_code_program_id(program: ObjectCodeProgram) -> str:
    """Return a provenance-independent content ID for one replayable program."""

    return hashlib.sha256(
        canonical_json(program.to_json_dict()).encode("ascii")
    ).hexdigest()


def _flatten_program_fields(
    value: object, *, prefix: str = ""
) -> dict[str, object]:
    if isinstance(value, Mapping):
        flattened: dict[str, object] = {}
        for key, child in value.items():
            if key in {
                "object_code_dsl_version",
                "kind",
                "scene_ast_version",
                "op",
            }:
                continue
            path = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten_program_fields(child, prefix=path))
        return flattened
    if isinstance(value, list):
        return {prefix: tuple(value)}
    return {prefix: value}


def _program_fields(program: ObjectCodeProgram) -> dict[str, object]:
    return _flatten_program_fields(program.to_json_dict())


def _repair_frontier(
    task: BlindTask,
    parent: ObjectCodeProgram | None,
    operator: str,
    *,
    holes: Sequence[str] = (),
) -> tuple[ObjectCodeProgram, ...]:
    if operator not in TYPED_OBJECT_CODE_ACTIONS:
        raise ValueError("unknown typed object/code action")
    candidates = enumerate_object_code_programs(task)
    if parent is None:
        if operator == "canvas_reinfer":
            return tuple(
                item
                for item in candidates
                if isinstance(item, (RoleStampProgram, ScenePipelineProgram))
            )
        if operator == "fill_ast_hole":
            return ()
        return candidates
    parent_values = _program_fields(parent)
    if operator == "object_rematch":
        if isinstance(parent, RoleStampProgram):
            allowed = {
                "payload_color",
                "source_anchor_color",
                "target_anchor_color",
                "transform",
            }
        elif isinstance(parent, D4LabelCompletionProgram):
            allowed = {"structure_color", "connectivity", "attachment_radius"}
        else:
            allowed = {
                "ast.parse.connectivity",
                "ast.parse.grouping",
                "ast.correspond.policy",
                "ast.correspond.features",
                "ast.correspond.d4_invariant",
                "ast.select.role",
                "ast.operate.transform",
            }
    elif operator == "canvas_reinfer":
        allowed = (
            {"canvas_mode"}
            if isinstance(parent, RoleStampProgram)
            else {
                "ast.canvas.mode",
                "ast.canvas.background",
                "ast.canvas.padding",
                "ast.canvas.height",
                "ast.canvas.width",
                "ast.render.mode",
            }
        )
    elif operator == "reparse_background":
        allowed = (
            {"background"}
            if not isinstance(parent, ScenePipelineProgram)
            else {"ast.parse.background"}
        )
    else:
        allowed = set(holes)
        if not allowed:
            return ()
    frontier = []
    for candidate in candidates:
        if type(candidate) is not type(parent):
            continue
        values = _program_fields(candidate)
        differences = {
            key for key in parent_values if parent_values[key] != values.get(key)
        }
        if differences and differences <= allowed:
            frontier.append(candidate)
    return tuple(frontier)


@dataclass(frozen=True, slots=True)
class FailureCertificate:
    certificate_id: str
    blind_task_id: str
    blind_content_sha256: str
    parent_hypothesis_id: str
    parent_source: str
    diagnosis: str
    recommended_action: str
    affected_slots: tuple[str, ...]
    evidence: tuple[tuple[str, object], ...]
    cross_representation: bool

    def __post_init__(self) -> None:
        if self.recommended_action not in TYPED_OBJECT_CODE_ACTIONS:
            raise ValueError("failure certificate recommends an illegal action")
        if tuple(sorted(set(self.affected_slots))) != self.affected_slots:
            raise ValueError("affected slots must be unique and sorted")
        if tuple(sorted(self.evidence)) != self.evidence:
            raise ValueError("certificate evidence must be canonically sorted")
        expected = _content_id(self._payload())
        if self.certificate_id != expected:
            raise ValueError("certificate ID does not match canonical content")

    def _payload(self) -> dict[str, object]:
        return {
            "schema": FAILURE_CERTIFICATE_VERSION,
            "blind_task_id": self.blind_task_id,
            "blind_content_sha256": self.blind_content_sha256,
            "parent_hypothesis_id": self.parent_hypothesis_id,
            "parent_source": self.parent_source,
            "diagnosis": self.diagnosis,
            "recommended_action": self.recommended_action,
            "affected_slots": list(self.affected_slots),
            "evidence": {key: value for key, value in self.evidence},
            "cross_representation": self.cross_representation,
        }

    @classmethod
    def create(
        cls,
        *,
        task: BlindTask,
        evaluation: CandidateEvaluation,
        diagnosis: str,
        recommended_action: str,
        affected_slots: Sequence[str],
        evidence: Mapping[str, object],
    ) -> "FailureCertificate":
        payload = {
            "schema": FAILURE_CERTIFICATE_VERSION,
            "blind_task_id": task.task_id,
            "blind_content_sha256": task.blind_content_sha256,
            "parent_hypothesis_id": evaluation.hypothesis.hypothesis_id,
            "parent_source": evaluation.hypothesis.source,
            "diagnosis": diagnosis,
            "recommended_action": recommended_action,
            "affected_slots": sorted(set(affected_slots)),
            "evidence": dict(evidence),
            "cross_representation": evaluation.hypothesis.source != "object_code_dsl",
        }
        return cls(
            _content_id(payload),
            task.task_id,
            task.blind_content_sha256,
            evaluation.hypothesis.hypothesis_id,
            evaluation.hypothesis.source,
            diagnosis,
            recommended_action,
            tuple(payload["affected_slots"]),
            tuple(sorted(dict(evidence).items())),
            payload["cross_representation"],
        )

    def to_json_dict(self) -> dict[str, object]:
        return {"certificate_id": self.certificate_id, **self._payload()}


def diagnose_object_code_failure(
    task: BlindTask, evaluation: CandidateEvaluation
) -> FailureCertificate:
    """Compile observed demo failure into one legal representation action.

    The diagnosis is intentionally structural and cheap: it does not run hidden
    counterfactual search.  Whether the recommended frontier actually contains an
    exact repair is measured separately, preventing diagnosis work from becoming
    an unaccounted oracle.
    """

    holes = tuple(
        sorted(
            value
            for value in evaluation.hypothesis.spec.get("provisional_ast_holes", [])
            if isinstance(value, str)
        )
    )
    if evaluation.demo_exact and not holes:
        raise ValueError("an exact demonstration fit has no failure to diagnose")
    residuals = evaluation.residuals
    execution_failures = sum(not item.execution_valid for item in residuals)
    shape_mismatches = sum(not item.shape_match for item in residuals)
    valid_shape_mismatches = sum(
        item.execution_valid and not item.shape_match for item in residuals
    )
    mismatch_count = sum(item.mismatch_count for item in residuals)
    program = _program_from_candidate(evaluation.hypothesis)
    if holes:
        diagnosis = "ast_hole"
        action = "fill_ast_hole"
        slots = holes
    elif valid_shape_mismatches:
        diagnosis = "canvas_contract_mismatch"
        action = "canvas_reinfer"
        slots = (
            (
                "ast.canvas.background",
                "ast.canvas.height",
                "ast.canvas.mode",
                "ast.canvas.padding",
                "ast.canvas.width",
                "ast.render.mode",
            )
            if isinstance(program, ScenePipelineProgram)
            else ("canvas_mode",)
        )
    else:
        modal_backgrounds = {
            min(
                counts := Counter(cell for row in pair.input for cell in row),
                key=lambda color: (-counts[color], color),
            )
            for pair in task.train
        }
        program_background = (
            program.parse.background
            if isinstance(program, ScenePipelineProgram)
            else getattr(program, "background", None)
        )
        if program is not None and program_background not in modal_backgrounds:
            diagnosis = "background_role_mismatch"
            action = "reparse_background"
            slots = (
                ("ast.parse.background",)
                if isinstance(program, ScenePipelineProgram)
                else ("background",)
            )
        elif execution_failures and program is None:
            diagnosis = "cross_representation_execution_failure"
            action = "reparse_background"
            slots = ("background",)
        else:
            diagnosis = (
                "object_parse_or_correspondence_failure"
                if execution_failures
                else "object_correspondence_mismatch"
            )
            action = "object_rematch"
            slots = (
                (
                    "payload_color",
                    "source_anchor_color",
                    "target_anchor_color",
                    "transform",
                )
                if isinstance(program, RoleStampProgram)
                else (
                    (
                        "ast.correspond.d4_invariant",
                        "ast.correspond.features",
                        "ast.correspond.policy",
                        "ast.operate.transform",
                        "ast.parse.connectivity",
                        "ast.parse.grouping",
                        "ast.select.role",
                    )
                    if isinstance(program, ScenePipelineProgram)
                    else ("attachment_radius", "connectivity", "structure_color")
                )
            )
    return FailureCertificate.create(
        task=task,
        evaluation=evaluation,
        diagnosis=diagnosis,
        recommended_action=action,
        affected_slots=slots,
        evidence={
            "demo_count": len(residuals),
            "execution_failure_count": execution_failures,
            "shape_mismatch_count": shape_mismatches,
            "valid_shape_mismatch_count": valid_shape_mismatches,
            "mismatch_count": mismatch_count,
        },
    )


@dataclass(slots=True)
class ObjectCodeProvider:
    max_program_trials: int = 20_000
    max_exact_programs: int = 32
    max_near_misses: int = 8
    minimum_near_miss_agreement: float = 0.2
    emit_exact_on_open: bool = True
    name: str = "object_code_dsl"
    route: str = "code_llm"
    strict_budget_contract: bool = field(default=False, init=False)
    supports_residual_actions: bool = field(default=True, init=False)
    supports_repeated_batches: bool = field(default=False, init=False)
    max_control_calls: int = field(default=2, init=False)
    parent_sensitive_operators: frozenset[str] = field(
        default=frozenset(
            {
                "open_hypothesis",
                "exception_resynthesize",
                *TYPED_OBJECT_CODE_ACTIONS,
            }
        ),
        init=False,
    )

    def __post_init__(self) -> None:
        if type(self.max_program_trials) is not int or self.max_program_trials < 1:
            raise ValueError("max_program_trials must be positive")
        if type(self.max_exact_programs) is not int or self.max_exact_programs < 1:
            raise ValueError("max_exact_programs must be positive")
        if type(self.max_near_misses) is not int or self.max_near_misses < 0:
            raise ValueError("max_near_misses must be non-negative")
        if not 0.0 <= self.minimum_near_miss_agreement <= 1.0:
            raise ValueError("minimum_near_miss_agreement must be in [0, 1]")

    def _result(
        self,
        task: BlindTask,
        *,
        programs: Sequence[ObjectCodeProgram] | None = None,
        existing_program_ids: Sequence[str] = (),
        parent_id: str | None = None,
        operator: str | None = None,
        holes: Sequence[str] = (),
        candidate_slots: int | None = None,
        allow_exact: bool = True,
    ) -> ProviderResult:
        existing_ids = frozenset(existing_program_ids)
        if any(
            not isinstance(program_id, str)
            or len(program_id) != 64
            or any(character not in "0123456789abcdef" for character in program_id)
            for program_id in existing_ids
        ):
            raise ValueError("existing object/code program IDs must be SHA256 strings")
        requested_frontier = (
            tuple(enumerate_object_code_programs(task))
            if programs is None and existing_ids
            else (None if programs is None else tuple(programs))
        )
        novel_programs = (
            None
            if requested_frontier is None
            else tuple(
                program
                for program in requested_frontier
                if object_code_program_id(program) not in existing_ids
            )
        )
        result = synthesize_object_code_programs(
            task,
            max_program_trials=self.max_program_trials,
            max_exact_programs=self.max_exact_programs,
            max_near_misses=self.max_near_misses,
            minimum_near_miss_agreement=self.minimum_near_miss_agreement,
            programs=novel_programs,
        )
        scores = (
            *(result.exact_scores if allow_exact else ()),
            *result.near_miss_scores,
        )
        candidates = tuple(
            make_object_code_hypothesis(
                score.program,
                demo_exact=score.all_demo_exact,
                parent_hypothesis_ids=((parent_id,) if parent_id else ()),
                control_operator=operator,
                ast_holes=(() if score.all_demo_exact else holes),
            )
            for score in scores
        )
        if candidate_slots is not None:
            candidates = candidates[:candidate_slots]
        emitted_program_ids = tuple(
            value
            for candidate in candidates
            if isinstance(
                value := candidate.spec.get("object_code_program_id"), str
            )
        )
        novel_frontier_count = sum(
            program_id not in existing_ids for program_id in emitted_program_ids
        )
        diagnostics: dict[str, object] = {
            "object_code_dsl_version": OBJECT_CODE_DSL_VERSION,
            "object_code_dsl_versions": list(OBJECT_CODE_DSL_VERSIONS),
            "action_operator": operator,
            "parent_hypothesis_id": parent_id,
            "exact_program_count": len(result.exact_scores),
            "near_miss_program_count": len(result.near_miss_scores),
            "semantic_duplicates_removed": result.semantic_duplicate_count,
            "invalid_program_count": result.invalid_program_count,
            "candidate_slot_limit": candidate_slots,
            "emitted_candidate_count": len(candidates),
            "requested_frontier_program_count": (
                None if requested_frontier is None else len(requested_frontier)
            ),
            "novel_program_trial_count": result.program_trial_count,
            "novel_frontier_count": novel_frontier_count,
            "frontier_changed": novel_frontier_count > 0,
            "frontier_change_basis": "content_addressed_object_code_program_id",
            "native_cost": {
                "object_code_program_trials": result.program_trial_count,
                "demo_object_code_executions": result.demo_execution_count,
                "query_object_code_executions": result.query_execution_count,
            },
            "native_cost_status": "executed",
        }
        if not candidates:
            reason = (
                "no_novel_object_code_frontier"
                if requested_frontier is not None
                and requested_frontier
                and not novel_programs
                else "no_object_code_candidate_in_frontier"
            )
            return ProviderResult.abstained(
                self.name,
                self.route,
                reason,
                diagnostics,
            )
        if novel_frontier_count < 1:
            return ProviderResult.abstained(
                self.name,
                self.route,
                "no_novel_object_code_frontier",
                diagnostics,
            )
        return ProviderResult.ok(self.name, self.route, candidates, diagnostics)

    def propose(
        self, task: BlindTask, features: TaskFeatures, decision: RouteDecision
    ) -> ProviderResult:
        del features, decision
        return self._result(task, allow_exact=self.emit_exact_on_open)

    def act(
        self,
        task: BlindTask,
        features: TaskFeatures,
        decision: RouteDecision,
        blackboard: "Blackboard",
        action: ControlAction,
    ) -> ProviderResult:
        del features, decision
        if (
            action.kind != "propose"
            or action.actor != self.name
            or action.route != self.route
        ):
            raise ValueError("object/code provider received an incompatible action")
        if action.operator == "open_hypothesis":
            if action.parent_hypothesis_id is not None:
                raise ValueError("initial object/code action cannot bind a parent")
            return self._result(
                task,
                operator=action.operator,
                existing_program_ids=tuple(
                    object_code_program_id(program)
                    for item in blackboard.evaluations
                    if (program := _program_from_candidate(item.hypothesis))
                    is not None
                ),
                candidate_slots=action.budget.candidate_slots,
                allow_exact=self.emit_exact_on_open,
            )
        if action.operator not in TYPED_OBJECT_CODE_ACTIONS:
            raise ValueError("object/code provider received an untyped residual action")
        parent_evaluation = next(
            (
                item
                for item in blackboard.evaluations
                if item.hypothesis.hypothesis_id == action.parent_hypothesis_id
            ),
            None,
        )
        if parent_evaluation is None:
            raise ValueError("typed object/code action requires an observed parent")
        parent = _program_from_candidate(parent_evaluation.hypothesis)
        holes = tuple(
            value
            for value in parent_evaluation.hypothesis.spec.get(
                "provisional_ast_holes", []
            )
            if isinstance(value, str)
        )
        frontier = _repair_frontier(task, parent, action.operator, holes=holes)
        existing_program_ids = tuple(
            object_code_program_id(program)
            for item in blackboard.evaluations
            if (program := _program_from_candidate(item.hypothesis)) is not None
        )
        return self._result(
            task,
            programs=frontier,
            existing_program_ids=existing_program_ids,
            parent_id=parent_evaluation.hypothesis.hypothesis_id,
            operator=action.operator,
            holes=(),
            candidate_slots=action.budget.candidate_slots,
            allow_exact=True,
        )


class ObjectCodeResidualCompiler:
    """Versioned adapter that replaces coarse code actions with typed actions."""

    def __init__(self, config: ResidualCompilerConfig | None = None) -> None:
        self.base = ResidualActionCompiler(config)

    def compile(
        self,
        task: BlindTask,
        blackboard: "Blackboard",
        providers: Sequence[object],
        *,
        max_selected_hypotheses: int,
    ) -> tuple[ControlAction, ...]:
        actions = self.base.compile(
            task,
            blackboard,
            providers,
            max_selected_hypotheses=max_selected_hypotheses,
        )
        object_provider_names = {
            getattr(provider, "name")
            for provider in providers
            if isinstance(provider, ObjectCodeProvider)
        }
        evaluation_by_id = {
            item.hypothesis.hypothesis_id: item for item in blackboard.evaluations
        }
        typed: list[ControlAction] = []
        for action in actions:
            if (
                action.kind != "propose"
                or action.actor not in object_provider_names
                or action.parent_hypothesis_id is None
            ):
                typed.append(action)
                continue
            evaluation = evaluation_by_id.get(action.parent_hypothesis_id)
            if evaluation is None or evaluation.demo_exact:
                typed.append(action)
                continue
            certificate = diagnose_object_code_failure(task, evaluation)
            typed.append(
                ControlAction.create(
                    state_id=action.state_id,
                    kind=action.kind,
                    actor=action.actor,
                    route=action.route,
                    operator=certificate.recommended_action,
                    parent_hypothesis_id=action.parent_hypothesis_id,
                    evidence_signal_ids=action.evidence_signal_ids,
                    reason_codes=tuple(
                        sorted(
                            {
                                *action.reason_codes,
                                f"failure_certificate:{certificate.certificate_id}",
                                f"typed_diagnosis:{certificate.diagnosis}",
                            }
                        )
                    ),
                    budget=action.budget,
                    priority=action.priority - 5,
                )
            )
        unique = {action.action_id: action for action in typed}
        return tuple(
            sorted(
                unique.values(), key=lambda action: (action.priority, action.action_id)
            )
        )


def typed_repair_frontier(
    task: BlindTask,
    evaluation: CandidateEvaluation,
    *,
    existing_programs: Sequence[ObjectCodeProgram] = (),
) -> tuple[FailureCertificate, tuple[ObjectCodeProgram, ...]]:
    """Return only typed repair programs absent from a frozen parent pool."""

    certificate = diagnose_object_code_failure(task, evaluation)
    parent = _program_from_candidate(evaluation.hypothesis)
    holes = tuple(
        value
        for value in evaluation.hypothesis.spec.get("provisional_ast_holes", [])
        if isinstance(value, str)
    )
    frontier = _repair_frontier(
        task,
        parent,
        certificate.recommended_action,
        holes=holes,
    )
    existing_ids = {object_code_program_id(program) for program in existing_programs}
    return certificate, tuple(
        program
        for program in frontier
        if object_code_program_id(program) not in existing_ids
    )
