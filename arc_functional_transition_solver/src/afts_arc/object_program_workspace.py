"""Typed object-correspondence workspace and executable rematch programs.

The implementation is opt-in.  It does not alter the legacy object/code or
relational-delta enumerators, so historical candidate identities and numerical
results remain unchanged.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

from afts_arc.blind import BlindTask
from afts_arc.experiment_safety import canonical_sha256
from afts_arc.grid import Grid, as_grid
from afts_arc.hybrid.scene_graph import (
    SCENE_GROUPINGS,
    ExecutionTraceNode,
    ParseObjectsNode,
    SceneGraph,
    extract_scene_graph,
)


OBJECT_PROGRAM_WORKSPACE_DSL_VERSION = "afts-object-program-workspace-dsl/v1"
OBJECT_REMATCH_CERTIFICATE_SCHEMA = "afts.object-rematch-certificate/v1"
OBJECT_ASSIGNMENT_DISTRIBUTION_SCHEMA = (
    "afts.object-assignment-distribution/v1"
)
OBJECT_REMATCH_SLOT = "ast.correspond.selector"

BASE_OBJECT_SELECTORS = (
    "all",
    "min_area",
    "max_area",
    "closest_border",
    "farthest_border",
)
RELATIONAL_OBJECT_SELECTORS = (
    "rarest_shape",
    "most_common_shape",
    "rarest_size",
    "most_common_size",
    "rarest_topology",
    "most_common_topology",
    "least_related",
    "most_related",
    "border_touching",
    "interior",
)
OBJECT_ROLE_SELECTORS = BASE_OBJECT_SELECTORS + RELATIONAL_OBJECT_SELECTORS
OBJECT_EFFECT_OPERATIONS = ("recolor_component", "erase_component")

SELECTOR_FEATURES: dict[str, tuple[str, ...]] = {
    "all": (),
    "min_area": ("area",),
    "max_area": ("area",),
    "closest_border": ("border_distance",),
    "farthest_border": ("border_distance",),
    "rarest_shape": ("same_shape_peers",),
    "most_common_shape": ("same_shape_peers",),
    "rarest_size": ("same_size_peers",),
    "most_common_size": ("same_size_peers",),
    "rarest_topology": ("same_topology_peers",),
    "most_common_topology": ("same_topology_peers",),
    "least_related": ("relation_degree",),
    "most_related": ("relation_degree",),
    "border_touching": ("touches_border",),
    "interior": ("touches_border",),
}

Coordinate = tuple[int, int]
ColoredCoordinate = tuple[int, int, int]
WeightedGrid = tuple[Grid, int]


def _arc_color(value: object, *, field: str) -> int:
    if type(value) is not int or not 0 <= value <= 9:
        raise ValueError(f"{field} must be an ARC color")
    return value


def _modal_color(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda color: (-counts[color], color))


def _shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def _delta(first: Grid, second: Grid) -> set[Coordinate]:
    if _shape(first) != _shape(second):
        raise ValueError("object-rematch delta requires equal canvas shapes")
    return {
        (row, column)
        for row in range(len(first))
        for column in range(len(first[0]))
        if first[row][column] != second[row][column]
    }


def _parse_id(parse: ParseObjectsNode) -> str:
    return canonical_sha256(parse.to_json_dict())


@dataclass(frozen=True, slots=True)
class ObjectRelationProfile:
    """Intrinsic and n-ary relational features for one object slot."""

    object_id: str
    index: int
    area: int
    same_shape_peers: int
    same_size_peers: int
    same_topology_peers: int
    near_peers: int
    aligned_peers: int
    containment_peers: int
    relation_degree: int
    border_distance: int
    touches_border: bool

    def to_json_dict(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "index": self.index,
            "area": self.area,
            "same_shape_peers": self.same_shape_peers,
            "same_size_peers": self.same_size_peers,
            "same_topology_peers": self.same_topology_peers,
            "near_peers": self.near_peers,
            "aligned_peers": self.aligned_peers,
            "containment_peers": self.containment_peers,
            "relation_degree": self.relation_degree,
            "border_distance": self.border_distance,
            "touches_border": self.touches_border,
        }


@lru_cache(maxsize=32768)
def relation_profiles(scene: SceneGraph) -> tuple[ObjectRelationProfile, ...]:
    """Compute deterministic n-ary profiles over the complete object set."""

    relations_by_source: dict[int, list[object]] = {
        object_.index: [] for object_ in scene.objects
    }
    for relation in scene.relations:
        relations_by_source[relation.source_index].append(relation)
    profiles = []
    for object_ in scene.objects:
        same_shape = sum(
            peer.index != object_.index
            and peer.canonical_d4_shape == object_.canonical_d4_shape
            for peer in scene.objects
        )
        same_size = sum(
            peer.index != object_.index
            and (peer.area, peer.height, peer.width)
            == (object_.area, object_.height, object_.width)
            for peer in scene.objects
        )
        same_topology = sum(
            peer.index != object_.index
            and (peer.hole_count, peer.touches_border)
            == (object_.hole_count, object_.touches_border)
            for peer in scene.objects
        )
        near = 0
        aligned = 0
        containment = 0
        for relation in relations_by_source[object_.index]:
            peer = scene.objects[relation.target_index]
            near += int(relation.chebyshev_distance <= 1)
            row_overlap = not (
                object_.bottom < peer.top or peer.bottom < object_.top
            )
            column_overlap = not (
                object_.right < peer.left or peer.right < object_.left
            )
            aligned += int(row_overlap or column_overlap)
            containment += int(
                relation.bbox_contains or relation.target_bbox_contains
            )
        border_distance = min(
            object_.top,
            object_.left,
            scene.height - 1 - object_.bottom,
            scene.width - 1 - object_.right,
        )
        profiles.append(
            ObjectRelationProfile(
                object_.object_id,
                object_.index,
                object_.area,
                same_shape,
                same_size,
                same_topology,
                near,
                aligned,
                containment,
                near + aligned + containment,
                border_distance,
                object_.touches_border,
            )
        )
    return tuple(profiles)


@lru_cache(maxsize=32768)
def _workspace_scene(grid: Grid, parse: ParseObjectsNode) -> SceneGraph:
    return extract_scene_graph(
        grid,
        background=parse.background,
        connectivity=parse.connectivity,
        grouping=parse.grouping,
    )


@dataclass(frozen=True, slots=True)
class NaryObjectCorrespondenceNode:
    selector: str

    def __post_init__(self) -> None:
        if self.selector not in OBJECT_ROLE_SELECTORS:
            raise ValueError("unknown object-rematch selector")

    @property
    def required_features(self) -> tuple[str, ...]:
        return SELECTOR_FEATURES[self.selector]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "op": "correspond_nary",
            "selector": self.selector,
            "features": list(self.required_features),
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "NaryObjectCorrespondenceNode":
        expected = {"op", "selector", "features"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("object correspondence node fields differ")
        if payload["op"] != "correspond_nary":
            raise ValueError("object correspondence node has the wrong operator")
        features = payload["features"]
        if not isinstance(features, list) or any(
            not isinstance(feature, str) for feature in features
        ):
            raise TypeError("object correspondence features must be strings")
        node = cls(payload["selector"])
        if tuple(features) != node.required_features:
            raise ValueError("selector features do not match frozen semantics")
        return node


@dataclass(frozen=True, slots=True)
class ObjectEffectNode:
    operation: str
    target_color: int | None

    def __post_init__(self) -> None:
        if self.operation not in OBJECT_EFFECT_OPERATIONS:
            raise ValueError("unknown object-rematch effect")
        if self.operation == "erase_component":
            if self.target_color is not None:
                raise ValueError("erase effect cannot carry a target color")
        else:
            _arc_color(self.target_color, field="effect target color")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "op": "apply_complete_object_effect",
            "operation": self.operation,
            "target_color": self.target_color,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "ObjectEffectNode":
        expected = {"op", "operation", "target_color"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("object effect node fields differ")
        if payload["op"] != "apply_complete_object_effect":
            raise ValueError("object effect node has the wrong operator")
        return cls(payload["operation"], payload["target_color"])


@dataclass(frozen=True, slots=True)
class ObjectRematchProgram:
    parse: ParseObjectsNode
    correspond: NaryObjectCorrespondenceNode
    effect: ObjectEffectNode

    @property
    def description_bits(self) -> int:
        selector_cost = 0 if self.correspond.selector in BASE_OBJECT_SELECTORS else 8
        feature_cost = 3 * len(self.correspond.required_features)
        color_cost = 0 if self.effect.target_color is None else 4
        return 24 + selector_cost + feature_cost + color_cost

    @property
    def program_id(self) -> str:
        return canonical_sha256(self.to_json_dict())

    def to_json_dict(self) -> dict[str, object]:
        return {
            "dsl_version": OBJECT_PROGRAM_WORKSPACE_DSL_VERSION,
            "parse": self.parse.to_json_dict(),
            "correspond": self.correspond.to_json_dict(),
            "effect": self.effect.to_json_dict(),
            "canvas": {"op": "preserve_input_canvas"},
            "render": {
                "op": "render_complete_objects",
                "conflict_policy": "reject",
            },
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "ObjectRematchProgram":
        expected = {
            "dsl_version",
            "parse",
            "correspond",
            "effect",
            "canvas",
            "render",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("object-rematch program fields differ")
        if payload["dsl_version"] != OBJECT_PROGRAM_WORKSPACE_DSL_VERSION:
            raise ValueError("unsupported object-program workspace DSL")
        if payload["canvas"] != {"op": "preserve_input_canvas"}:
            raise ValueError("v1 object-rematch programs are same-canvas only")
        if payload["render"] != {
            "op": "render_complete_objects",
            "conflict_policy": "reject",
        }:
            raise ValueError("unsupported object-rematch render policy")
        return cls(
            ParseObjectsNode.from_json_dict(payload["parse"]),
            NaryObjectCorrespondenceNode.from_json_dict(payload["correspond"]),
            ObjectEffectNode.from_json_dict(payload["effect"]),
        )


def _profile_value(profile: ObjectRelationProfile, feature: str) -> int:
    values = {
        "area": profile.area,
        "same_shape_peers": profile.same_shape_peers,
        "same_size_peers": profile.same_size_peers,
        "same_topology_peers": profile.same_topology_peers,
        "relation_degree": profile.relation_degree,
        "border_distance": profile.border_distance,
    }
    return values[feature]


def select_object_ids(
    scene: SceneGraph,
    node: NaryObjectCorrespondenceNode,
) -> tuple[str, ...]:
    """Instantiate one selector as a complete n-ary object assignment."""

    profiles = relation_profiles(scene)
    if not profiles:
        return ()
    selector = node.selector
    if selector == "all":
        selected = profiles
    elif selector == "border_touching":
        selected = tuple(profile for profile in profiles if profile.touches_border)
    elif selector == "interior":
        selected = tuple(
            profile for profile in profiles if not profile.touches_border
        )
    else:
        feature = node.required_features[0]
        values = tuple(_profile_value(profile, feature) for profile in profiles)
        maximize = selector.startswith("max") or selector.startswith("most")
        maximize |= selector == "farthest_border"
        target = max(values) if maximize else min(values)
        selected = tuple(
            profile
            for profile in profiles
            if _profile_value(profile, feature) == target
        )
    return tuple(sorted(profile.object_id for profile in selected))


@dataclass(frozen=True, slots=True)
class ObjectRematchExecution:
    status: str
    output: Grid | None
    reason: str | None
    parse_id: str
    selected_object_ids: tuple[str, ...]
    write_mask: tuple[Coordinate, ...]
    writes: tuple[ColoredCoordinate, ...]
    profiles: tuple[ObjectRelationProfile, ...]
    node_trace: tuple[ExecutionTraceNode, ...]

    def __post_init__(self) -> None:
        if self.status not in {"ok", "invalid"}:
            raise ValueError("object-rematch execution status is invalid")
        if (self.status == "ok") != (self.output is not None):
            raise ValueError("only successful object-rematch executions have output")
        if tuple(sorted(set(self.selected_object_ids))) != self.selected_object_ids:
            raise ValueError("selected object IDs must be unique and sorted")
        if tuple(sorted(set(self.write_mask))) != self.write_mask:
            raise ValueError("object-rematch write mask must be unique and sorted")
        if tuple(sorted(self.writes)) != self.writes:
            raise ValueError("object-rematch colored writes must be sorted")
        if tuple((row, column) for row, column, _ in self.writes) != self.write_mask:
            raise ValueError("object-rematch write colors do not close over the mask")

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _invalid_execution(
    *,
    reason: str,
    parse: ParseObjectsNode,
    selected_object_ids: Sequence[str],
    profiles: Sequence[ObjectRelationProfile],
    trace: Sequence[ExecutionTraceNode],
) -> ObjectRematchExecution:
    return ObjectRematchExecution(
        "invalid",
        None,
        reason,
        _parse_id(parse),
        tuple(sorted(selected_object_ids)),
        (),
        (),
        tuple(profiles),
        tuple(trace),
    )


def execute_object_rematch(
    program: ObjectRematchProgram,
    grid: Grid,
) -> ObjectRematchExecution:
    """Execute a serialized object-rematch program without partial writes."""

    normalized = as_grid(grid)
    trace = []
    scene = _workspace_scene(normalized, program.parse)
    profiles = relation_profiles(scene)
    trace.append(
        ExecutionTraceNode.create(
            "parse",
            "parse_objects",
            "ok",
            object_count=len(scene.objects),
            parse_id=_parse_id(program.parse),
        )
    )
    trace.append(
        ExecutionTraceNode.create(
            "relation_graph",
            "build_nary_relation_graph",
            "ok",
            object_count=len(profiles),
            relation_count=len(scene.relations),
        )
    )
    selected_ids = select_object_ids(scene, program.correspond)
    if not selected_ids:
        trace.append(
            ExecutionTraceNode.create(
                "correspond",
                "correspond_nary",
                "invalid",
                reason="empty_object_assignment",
                selector=program.correspond.selector,
            )
        )
        return _invalid_execution(
            reason="empty_object_assignment",
            parse=program.parse,
            selected_object_ids=(),
            profiles=profiles,
            trace=trace,
        )
    trace.append(
        ExecutionTraceNode.create(
            "correspond",
            "correspond_nary",
            "ok",
            selected_count=len(selected_ids),
            selector=program.correspond.selector,
        )
    )
    selected = tuple(
        object_ for object_ in scene.objects if object_.object_id in selected_ids
    )
    write_mask = tuple(
        sorted(cell for object_ in selected for cell in object_.cells)
    )
    color = (
        program.parse.background
        if program.effect.operation == "erase_component"
        else program.effect.target_color
    )
    if color is None:
        raise AssertionError("validated recolor effect has no target color")
    writes = tuple((row, column, color) for row, column in write_mask)
    canvas = [list(row) for row in normalized]
    for row, column, target_color in writes:
        canvas[row][column] = target_color
    output = as_grid(canvas)
    if output == normalized:
        trace.append(
            ExecutionTraceNode.create(
                "effect",
                program.effect.operation,
                "invalid",
                reason="empty_effective_delta",
            )
        )
        return _invalid_execution(
            reason="empty_effective_delta",
            parse=program.parse,
            selected_object_ids=selected_ids,
            profiles=profiles,
            trace=trace,
        )
    trace.append(
        ExecutionTraceNode.create(
            "effect",
            program.effect.operation,
            "ok",
            complete_object_count=len(selected),
            write_count=len(write_mask),
        )
    )
    trace.append(
        ExecutionTraceNode.create(
            "render",
            "render_complete_objects",
            "ok",
            height=len(output),
            width=len(output[0]),
        )
    )
    return ObjectRematchExecution(
        "ok",
        output,
        None,
        _parse_id(program.parse),
        selected_ids,
        write_mask,
        writes,
        profiles,
        tuple(trace),
    )


def _candidate_backgrounds(task: BlindTask, *, limit: int = 2) -> tuple[int, ...]:
    grids = tuple(pair.input for pair in task.train) + task.test_inputs
    modal_counts = Counter(_modal_color(grid) for grid in grids)
    color_counts = Counter(cell for grid in grids for row in grid for cell in row)
    return tuple(
        sorted(
            color_counts,
            key=lambda color: (
                -modal_counts[color],
                -color_counts[color],
                color != 0,
                color,
            ),
        )[:limit]
    )


def _parse_candidates(task: BlindTask) -> tuple[ParseObjectsNode, ...]:
    return tuple(
        ParseObjectsNode(background, connectivity, grouping)
        for background in _candidate_backgrounds(task)
        for grouping in SCENE_GROUPINGS
        for connectivity in ((4,) if grouping == "color_groups" else (4, 8))
    )


def _effect_candidates(
    task: BlindTask,
    parse: ParseObjectsNode,
) -> tuple[ObjectEffectNode, ...]:
    output_colors = tuple(
        sorted(
            {
                cell
                for pair in task.train
                for row in pair.output
                for cell in row
            }
        )
    )
    effects = [ObjectEffectNode("erase_component", None)]
    effects.extend(
        ObjectEffectNode("recolor_component", color)
        for color in output_colors
        if color != parse.background
    )
    return tuple(effects)


def enumerate_object_rematch_programs(
    task: BlindTask,
) -> tuple[ObjectRematchProgram, ...]:
    """Enumerate the bounded v1 grammar from blind task observations."""

    if not isinstance(task, BlindTask):
        raise TypeError("object-program workspace accepts BlindTask only")
    programs = tuple(
        ObjectRematchProgram(
            parse,
            NaryObjectCorrespondenceNode(selector),
            effect,
        )
        for parse in _parse_candidates(task)
        for effect in _effect_candidates(task, parse)
        for selector in OBJECT_ROLE_SELECTORS
    )
    unique = {program.program_id: program for program in programs}
    return tuple(
        sorted(
            unique.values(),
            key=lambda program: (
                program.description_bits,
                program.program_id,
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class ObjectRematchProgramScore:
    program: ObjectRematchProgram
    demo_outputs: tuple[Grid | None, ...]
    exact_demo_count: int
    mismatch_count: int
    execution_valid: bool

    @property
    def all_demo_exact(self) -> bool:
        return self.execution_valid and self.exact_demo_count == len(self.demo_outputs)


def score_object_rematch_program(
    program: ObjectRematchProgram,
    task: BlindTask,
) -> ObjectRematchProgramScore:
    outputs = []
    exact_count = 0
    mismatch_count = 0
    valid = True
    for pair in task.train:
        execution = execute_object_rematch(program, pair.input)
        output = execution.output if execution.ok else None
        outputs.append(output)
        if output is None:
            valid = False
            mismatch_count += len(pair.output) * len(pair.output[0])
        elif _shape(output) != _shape(pair.output):
            mismatch_count += max(
                len(output) * len(output[0]),
                len(pair.output) * len(pair.output[0]),
            )
        else:
            mismatches = len(_delta(output, pair.output))
            mismatch_count += mismatches
            exact_count += int(mismatches == 0)
    return ObjectRematchProgramScore(
        program,
        tuple(outputs),
        exact_count,
        mismatch_count,
        valid,
    )


def synthesize_exact_object_rematch_programs(
    task: BlindTask,
) -> tuple[ObjectRematchProgram, ...]:
    return tuple(
        score.program
        for score in (
            score_object_rematch_program(program, task)
            for program in enumerate_object_rematch_programs(task)
        )
        if score.all_demo_exact
    )


def infer_effect_assignment(
    program: ObjectRematchProgram,
    source: Grid,
    target: Grid,
) -> tuple[str, ...] | None:
    """Map one complete output grid to a complete source-object assignment."""

    normalized_source = as_grid(source)
    normalized_target = as_grid(target)
    if _shape(normalized_source) != _shape(normalized_target):
        return None
    scene = _workspace_scene(normalized_source, program.parse)
    target_color = (
        program.parse.background
        if program.effect.operation == "erase_component"
        else program.effect.target_color
    )
    if target_color is None:
        raise AssertionError("validated recolor effect has no target color")
    assignment = tuple(
        sorted(
            object_.object_id
            for object_ in scene.objects
            if any(
                normalized_source[row][column] != target_color
                for row, column in object_.cells
            )
            and all(
                normalized_target[row][column] == target_color
                for row, column in object_.cells
            )
        )
    )
    if not assignment:
        return None
    canvas = [list(row) for row in normalized_source]
    assigned = set(assignment)
    for object_ in scene.objects:
        if object_.object_id not in assigned:
            continue
        for row, column in object_.cells:
            canvas[row][column] = target_color
    return assignment if as_grid(canvas) == normalized_target else None


@dataclass(frozen=True, slots=True)
class ObjectAssignmentMass:
    object_ids: tuple[str, ...]
    weight: int

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.object_ids))) != self.object_ids:
            raise ValueError("posterior assignment IDs must be unique and sorted")
        if type(self.weight) is not int or self.weight <= 0:
            raise ValueError("posterior assignment weight must be positive")

    def to_json_dict(self) -> dict[str, object]:
        return {"object_ids": list(self.object_ids), "weight": self.weight}


@dataclass(frozen=True, slots=True)
class ObjectAssignmentDistribution:
    distribution_id: str
    parse_id: str
    effect: Mapping[str, object]
    total_weight: int
    represented_weight: int
    rejected_weight: int
    assignments: tuple[ObjectAssignmentMass, ...]

    def __post_init__(self) -> None:
        if self.total_weight != self.represented_weight + self.rejected_weight:
            raise ValueError("posterior assignment weights do not close")
        if self.represented_weight != sum(item.weight for item in self.assignments):
            raise ValueError("posterior assignment masses do not close")
        if self.distribution_id != canonical_sha256(self._content()):
            raise ValueError("posterior assignment distribution ID differs")

    def _content(self) -> dict[str, object]:
        return {
            "schema": OBJECT_ASSIGNMENT_DISTRIBUTION_SCHEMA,
            "parse_id": self.parse_id,
            "effect": dict(self.effect),
            "total_weight": self.total_weight,
            "represented_weight": self.represented_weight,
            "rejected_weight": self.rejected_weight,
            "assignments": [item.to_json_dict() for item in self.assignments],
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"distribution_id": self.distribution_id, **self._content()}

    def weight_for(self, object_ids: Sequence[str]) -> int:
        normalized = tuple(sorted(object_ids))
        return sum(
            item.weight for item in self.assignments if item.object_ids == normalized
        )


def posterior_assignment_distribution(
    program: ObjectRematchProgram,
    source: Grid,
    weighted_samples: Sequence[WeightedGrid],
) -> ObjectAssignmentDistribution:
    """Map whole-grid posterior mass to joint object assignments."""

    if not weighted_samples:
        raise ValueError("object assignment posterior must not be empty")
    masses: Counter[tuple[str, ...]] = Counter()
    total_weight = 0
    rejected_weight = 0
    for raw_sample, weight in weighted_samples:
        if type(weight) is not int or weight <= 0:
            raise ValueError("posterior sample weights must be positive integers")
        sample = as_grid(raw_sample)
        total_weight += weight
        assignment = infer_effect_assignment(program, source, sample)
        if assignment is None:
            rejected_weight += weight
        else:
            masses[assignment] += weight
    assignments = tuple(
        ObjectAssignmentMass(object_ids, weight)
        for object_ids, weight in sorted(
            masses.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )
    content = {
        "schema": OBJECT_ASSIGNMENT_DISTRIBUTION_SCHEMA,
        "parse_id": _parse_id(program.parse),
        "effect": program.effect.to_json_dict(),
        "total_weight": total_weight,
        "represented_weight": total_weight - rejected_weight,
        "rejected_weight": rejected_weight,
        "assignments": [item.to_json_dict() for item in assignments],
    }
    return ObjectAssignmentDistribution(
        canonical_sha256(content),
        content["parse_id"],
        content["effect"],
        total_weight,
        total_weight - rejected_weight,
        rejected_weight,
        assignments,
    )


@dataclass(frozen=True, slots=True)
class ObjectRematchNearMissQuality:
    eligible: bool
    identity_mismatch_count: int
    parent_mismatch_count: int
    delta_precision: float
    delta_recall: float
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
            "per_demo_not_worse": self.per_demo_not_worse,
            "reason": self.reason,
        }


def object_rematch_near_miss_quality(
    program: ObjectRematchProgram,
    task: BlindTask,
    *,
    minimum_precision: float = 0.5,
    minimum_recall: float = 0.25,
) -> ObjectRematchNearMissQuality:
    identity_mismatches = 0
    parent_mismatches = 0
    predicted_count = 0
    gold_count = 0
    intersection_count = 0
    per_demo_not_worse = True
    execution_valid = True
    for pair in task.train:
        if _shape(pair.input) != _shape(pair.output):
            return ObjectRematchNearMissQuality(
                False,
                0,
                0,
                0.0,
                0.0,
                False,
                "canvas_incompatible",
            )
        execution = execute_object_rematch(program, pair.input)
        if not execution.ok or execution.output is None:
            execution_valid = False
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
    precision = intersection_count / predicted_count if predicted_count else 0.0
    recall = intersection_count / gold_count if gold_count else 0.0
    conditions = (
        execution_valid,
        parent_mismatches > 0,
        parent_mismatches < identity_mismatches,
        per_demo_not_worse,
        precision >= minimum_precision,
        recall >= minimum_recall,
    )
    reasons = (
        "invalid_execution",
        "parent_is_exact",
        "does_not_improve_identity",
        "worse_on_a_demo",
        "delta_precision_below_threshold",
        "delta_recall_below_threshold",
    )
    reason = "eligible" if all(conditions) else reasons[conditions.index(False)]
    return ObjectRematchNearMissQuality(
        all(conditions),
        identity_mismatches,
        parent_mismatches,
        precision,
        recall,
        per_demo_not_worse,
        reason,
    )


@dataclass(frozen=True, slots=True)
class ObjectRematchCertificate:
    certificate_id: str
    task_id: str
    parent_program_id: str
    mode: str
    action: str
    affected_slots: tuple[str, ...]
    evidence: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.mode not in {"residual_only", "joint_posterior"}:
            raise ValueError("unknown object-rematch certificate mode")
        if self.action != "object_rematch":
            raise ValueError("object-rematch certificate action differs")
        if self.affected_slots != (OBJECT_REMATCH_SLOT,):
            raise ValueError("object-rematch certificate slot differs")
        if self.certificate_id != canonical_sha256(self._content()):
            raise ValueError("object-rematch certificate ID differs")

    def _content(self) -> dict[str, object]:
        return {
            "schema": OBJECT_REMATCH_CERTIFICATE_SCHEMA,
            "task_id": self.task_id,
            "parent_program_id": self.parent_program_id,
            "mode": self.mode,
            "action": self.action,
            "affected_slots": list(self.affected_slots),
            "evidence": dict(self.evidence),
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"certificate_id": self.certificate_id, **self._content()}


def compile_object_rematch_certificate(
    *,
    task_id: str,
    task: BlindTask,
    parent: ObjectRematchProgram,
    mode: str,
    posterior_views: Sequence[Sequence[WeightedGrid]],
) -> ObjectRematchCertificate:
    """Compile demo residuals into one typed correspondence-slot action."""

    if mode not in {"residual_only", "joint_posterior"}:
        raise ValueError("unknown object-rematch certificate mode")
    if mode == "residual_only" and posterior_views:
        raise ValueError("residual-only certificate cannot receive posterior views")
    if mode == "joint_posterior" and len(posterior_views) != len(task.train):
        raise ValueError("joint posterior views must align with demonstrations")
    rows = []
    has_residual = False
    for demo_index, pair in enumerate(task.train):
        execution = execute_object_rematch(parent, pair.input)
        if not execution.ok or execution.output is None:
            raise ValueError("object-rematch certificate requires a valid parent")
        gold_assignment = infer_effect_assignment(parent, pair.input, pair.output)
        if gold_assignment is None:
            raise ValueError("gold demo is not representable by parent parse/effect")
        parent_assignment = execution.selected_object_ids
        has_residual |= parent_assignment != gold_assignment
        posterior = None
        gold_support = 0
        if mode == "joint_posterior":
            posterior = posterior_assignment_distribution(
                parent,
                pair.input,
                posterior_views[demo_index],
            )
            gold_support = posterior.weight_for(gold_assignment)
        rows.append(
            {
                "demo_index": demo_index,
                "parent_assignment": list(parent_assignment),
                "gold_assignment": list(gold_assignment),
                "parent_write_mask": [list(cell) for cell in execution.write_mask],
                "gold_delta_mask": [
                    list(cell) for cell in sorted(_delta(pair.input, pair.output))
                ],
                "posterior_gold_assignment_weight": gold_support,
                "posterior_distribution": (
                    None if posterior is None else posterior.to_json_dict()
                ),
            }
        )
    if not has_residual:
        raise ValueError("exact parent has no object-rematch residual")
    quality = object_rematch_near_miss_quality(parent, task)
    evidence: dict[str, object] = {
        "demo_residuals": rows,
        "near_miss_quality": quality.to_json_dict(),
        "existing_slots_only": True,
        "joint_assignment_mass_only": mode == "joint_posterior",
        "pixel_marginal_used": False,
        "query_gold_read": False,
    }
    content = {
        "schema": OBJECT_REMATCH_CERTIFICATE_SCHEMA,
        "task_id": task_id,
        "parent_program_id": parent.program_id,
        "mode": mode,
        "action": "object_rematch",
        "affected_slots": [OBJECT_REMATCH_SLOT],
        "evidence": evidence,
    }
    return ObjectRematchCertificate(
        canonical_sha256(content),
        task_id,
        parent.program_id,
        mode,
        "object_rematch",
        (OBJECT_REMATCH_SLOT,),
        evidence,
    )


@dataclass(frozen=True, slots=True)
class ObjectRematchEdit:
    slot: str
    program: ObjectRematchProgram


def object_rematch_variants(
    parent: ObjectRematchProgram,
    *,
    candidate_programs: Sequence[ObjectRematchProgram],
    existing_program_ids: Sequence[str],
) -> tuple[ObjectRematchEdit, ...]:
    """Return novel children that differ in exactly the correspondence slot."""

    existing = frozenset(existing_program_ids)
    edits = []
    for program in candidate_programs:
        if program.program_id in existing:
            continue
        if program.parse != parent.parse or program.effect != parent.effect:
            continue
        if program.correspond == parent.correspond:
            continue
        edits.append(ObjectRematchEdit(OBJECT_REMATCH_SLOT, program))
    return tuple(
        sorted(
            edits,
            key=lambda edit: (
                edit.program.description_bits,
                edit.program.program_id,
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class RankedObjectRematchEdit:
    edit: ObjectRematchEdit
    demo_exact: bool
    demo_mismatch_count: int
    posterior_assignment_weight: int
    posterior_total_weight: int
    query_output: Grid | None


def rank_object_rematch_variants(
    *,
    task: BlindTask,
    parent: ObjectRematchProgram,
    certificate: ObjectRematchCertificate,
    candidate_programs: Sequence[ObjectRematchProgram],
    existing_program_ids: Sequence[str],
    query_index: int,
    query_posterior: Sequence[WeightedGrid],
    use_posterior: bool,
) -> tuple[RankedObjectRematchEdit, ...]:
    """Rank legal child programs using joint assignment mass, never pixels."""

    if certificate.parent_program_id != parent.program_id:
        raise ValueError("certificate and parent program differ")
    if certificate.action != "object_rematch":
        raise ValueError("certificate does not authorize object rematching")
    if type(query_index) is not int or not 0 <= query_index < len(task.test_inputs):
        raise ValueError("query index is outside the blind task")
    if use_posterior and not query_posterior:
        raise ValueError("posterior-ranked repair requires a posterior view")
    variants = object_rematch_variants(
        parent,
        candidate_programs=candidate_programs,
        existing_program_ids=existing_program_ids,
    )
    query_input = task.test_inputs[query_index]
    ranked = []
    for edit in variants:
        score = score_object_rematch_program(edit.program, task)
        execution = execute_object_rematch(edit.program, query_input)
        query_output = execution.output if execution.ok else None
        posterior_weight = 0
        posterior_total = 0
        if use_posterior:
            distribution = posterior_assignment_distribution(
                edit.program,
                query_input,
                query_posterior,
            )
            posterior_total = distribution.total_weight
            if execution.ok:
                posterior_weight = distribution.weight_for(
                    execution.selected_object_ids
                )
        ranked.append(
            RankedObjectRematchEdit(
                edit,
                score.all_demo_exact,
                score.mismatch_count,
                posterior_weight,
                posterior_total,
                query_output,
            )
        )
    return tuple(
        sorted(
            ranked,
            key=lambda item: (
                not item.demo_exact,
                item.demo_mismatch_count,
                -item.posterior_assignment_weight if use_posterior else 0,
                item.edit.program.description_bits,
                item.edit.program.program_id,
            ),
        )
    )


def base_prefix_programs(
    programs: Sequence[ObjectRematchProgram],
) -> tuple[ObjectRematchProgram, ...]:
    return tuple(
        program
        for program in programs
        if program.correspond.selector in BASE_OBJECT_SELECTORS
    )
