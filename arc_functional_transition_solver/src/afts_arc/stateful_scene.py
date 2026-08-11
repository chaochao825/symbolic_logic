"""Persistent scene state and dependency-aware replay for scene programs.

The historical scene executor intentionally returns only its output and a small
diagnostic trace.  This opt-in adapter retains content-addressed object and
relation identities plus the state produced by each AST node.  A typed rewrite
can therefore reuse the unaffected prefix and execute only the dependent
suffix.  Existing providers and their numerical behavior are not modified.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .experiment_safety import canonical_sha256
from .grid import Grid, as_grid, grid_key
from .hybrid.scene_graph import (
    CanvasNode,
    ObjectCorrespondence,
    ObjectOperationNode,
    ParseObjectsNode,
    SceneGraph,
    SceneObject,
    ScenePipelineProgram,
    _bounding_box,
    _correspondence_peer_count,
    _fit_canvas,
    _object_crop,
    _pack_grids,
    _replace_background,
    _select_objects,
    _transform_grid,
    extract_scene_graph,
    object_correspondences,
)


STATEFUL_SCENE_SCHEMA = "afts.stateful-scene-execution/v1"
PERSISTENT_SCENE_SCHEMA = "afts.persistent-scene-state/v1"
STATEFUL_TRACE_SCHEMA = "afts.stateful-scene-trace-node/v1"
STATEFUL_NODE_ORDER = ("parse", "assignment", "operate", "canvas", "render")

Coordinate = tuple[int, int]
BoundingBox = tuple[int, int, int, int]


def _program_id(program: ScenePipelineProgram) -> str:
    return canonical_sha256(program.to_json_dict())


def _json_value(value: object) -> object:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def scene_program_node_payloads(
    program: ScenePipelineProgram,
) -> dict[str, object]:
    """Return the five independently replayable typed AST regions."""

    if not isinstance(program, ScenePipelineProgram):
        raise TypeError("stateful scene replay requires a scene pipeline program")
    return {
        "parse": program.parse.to_json_dict(),
        "assignment": {
            "correspond": program.correspond.to_json_dict(),
            "select": program.select.to_json_dict(),
        },
        "operate": program.operate.to_json_dict(),
        "canvas": program.canvas.to_json_dict(),
        "render": program.render.to_json_dict(),
    }


def scene_program_node_differences(
    parent: ScenePipelineProgram,
    child: ScenePipelineProgram,
) -> tuple[str, ...]:
    parent_nodes = scene_program_node_payloads(parent)
    child_nodes = scene_program_node_payloads(child)
    return tuple(
        node_id
        for node_id in STATEFUL_NODE_ORDER
        if parent_nodes[node_id] != child_nodes[node_id]
    )


def affected_scene_subtree(node_id: str) -> tuple[str, ...]:
    if node_id not in STATEFUL_NODE_ORDER:
        raise ValueError("unknown stateful scene node")
    return STATEFUL_NODE_ORDER[STATEFUL_NODE_ORDER.index(node_id) :]


@dataclass(frozen=True, slots=True)
class PersistentObjectRecord:
    entity_id: str
    source_object_id: str
    index: int
    cells: tuple[Coordinate, ...]
    colors: tuple[int, ...]
    bounding_box: BoundingBox
    ancestor_entity_ids: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "entity_id": self.entity_id,
            "source_object_id": self.source_object_id,
            "index": self.index,
            "cells": [list(cell) for cell in self.cells],
            "colors": list(self.colors),
            "bounding_box": list(self.bounding_box),
            "ancestor_entity_ids": list(self.ancestor_entity_ids),
        }


@dataclass(frozen=True, slots=True)
class PersistentRelationRecord:
    relation_id: str
    source_entity_id: str
    target_entity_id: str
    left_of: bool
    above: bool
    bbox_contains: bool
    target_bbox_contains: bool
    chebyshev_distance: int

    def to_json_dict(self) -> dict[str, object]:
        return {
            "relation_id": self.relation_id,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "left_of": self.left_of,
            "above": self.above,
            "bbox_contains": self.bbox_contains,
            "target_bbox_contains": self.target_bbox_contains,
            "chebyshev_distance": self.chebyshev_distance,
        }


@dataclass(frozen=True, slots=True)
class PersistentSceneState:
    state_id: str
    grid_id: str
    parse_id: str
    scene: SceneGraph
    objects: tuple[PersistentObjectRecord, ...]
    relations: tuple[PersistentRelationRecord, ...]

    @classmethod
    def create(
        cls,
        grid: Grid,
        parse: ParseObjectsNode,
        *,
        previous: "PersistentSceneState | None" = None,
    ) -> "PersistentSceneState":
        normalized = as_grid(grid)
        scene = extract_scene_graph(
            normalized,
            background=parse.background,
            connectivity=parse.connectivity,
            grouping=parse.grouping,
        )
        input_id = grid_key(normalized)
        previous_objects = () if previous is None else previous.objects
        records = []
        for object_ in scene.objects:
            entity_payload = {
                "grid_id": input_id,
                "cells": [list(cell) for cell in object_.cells],
                "colors": list(object_.colors),
            }
            cell_set = frozenset(object_.cells)
            ancestors = tuple(
                sorted(
                    record.entity_id
                    for record in previous_objects
                    if cell_set.intersection(record.cells)
                )
            )
            records.append(
                PersistentObjectRecord(
                    canonical_sha256(entity_payload),
                    object_.object_id,
                    object_.index,
                    object_.cells,
                    object_.colors,
                    object_.bounding_box,
                    ancestors,
                )
            )
        objects = tuple(records)
        entity_by_index = {record.index: record.entity_id for record in objects}
        relations = []
        for relation in scene.relations:
            payload = {
                "source_entity_id": entity_by_index[relation.source_index],
                "target_entity_id": entity_by_index[relation.target_index],
                "left_of": relation.left_of,
                "above": relation.above,
                "bbox_contains": relation.bbox_contains,
                "target_bbox_contains": relation.target_bbox_contains,
                "chebyshev_distance": relation.chebyshev_distance,
            }
            relations.append(
                PersistentRelationRecord(
                    canonical_sha256(payload),
                    payload["source_entity_id"],
                    payload["target_entity_id"],
                    payload["left_of"],
                    payload["above"],
                    payload["bbox_contains"],
                    payload["target_bbox_contains"],
                    payload["chebyshev_distance"],
                )
            )
        ordered_relations = tuple(sorted(relations, key=lambda item: item.relation_id))
        parse_id = canonical_sha256(parse.to_json_dict())
        content = {
            "schema": PERSISTENT_SCENE_SCHEMA,
            "grid_id": input_id,
            "parse_id": parse_id,
            "objects": [record.to_json_dict() for record in objects],
            "relations": [record.to_json_dict() for record in ordered_relations],
        }
        return cls(
            canonical_sha256(content),
            input_id,
            parse_id,
            scene,
            objects,
            ordered_relations,
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": PERSISTENT_SCENE_SCHEMA,
            "state_id": self.state_id,
            "grid_id": self.grid_id,
            "parse_id": self.parse_id,
            "objects": [record.to_json_dict() for record in self.objects],
            "relations": [record.to_json_dict() for record in self.relations],
        }


@dataclass(frozen=True, slots=True)
class AssignmentState:
    state_id: str
    correspondences: tuple[ObjectCorrespondence, ...]
    selected_objects: tuple[SceneObject, ...]
    selected_entity_ids: tuple[str, ...]
    correspondence_entity_ids: tuple[tuple[str, ...], ...]

    @classmethod
    def create(
        cls,
        program: ScenePipelineProgram,
        parse_state: PersistentSceneState,
    ) -> "AssignmentState":
        correspondences = object_correspondences(parse_state.scene, program.correspond)
        selected = _select_objects(parse_state.scene, program.select, correspondences)
        entity_by_index = {
            record.index: record.entity_id for record in parse_state.objects
        }
        selected_ids = tuple(
            sorted(entity_by_index[object_.index] for object_ in selected)
        )
        correspondence_ids = tuple(
            tuple(entity_by_index[index] for index in item.object_indices)
            for item in correspondences
        )
        content = {
            "parse_state_id": parse_state.state_id,
            "assignment": scene_program_node_payloads(program)["assignment"],
            "selected_entity_ids": list(selected_ids),
            "correspondence_entity_ids": [list(item) for item in correspondence_ids],
        }
        return cls(
            canonical_sha256(content),
            correspondences,
            selected,
            selected_ids,
            correspondence_ids,
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "state_id": self.state_id,
            "selected_entity_ids": list(self.selected_entity_ids),
            "correspondence_entity_ids": [
                list(item) for item in self.correspondence_entity_ids
            ],
        }


@dataclass(frozen=True, slots=True)
class OperationState:
    state_id: str
    node: ObjectOperationNode
    assignment_state_id: str

    @classmethod
    def create(
        cls,
        node: ObjectOperationNode,
        assignment: AssignmentState,
    ) -> "OperationState":
        content = {
            "assignment_state_id": assignment.state_id,
            "node": node.to_json_dict(),
        }
        return cls(canonical_sha256(content), node, assignment.state_id)


@dataclass(frozen=True, slots=True)
class CanvasState:
    state_id: str
    node: CanvasNode
    operation_state_id: str

    @classmethod
    def create(cls, node: CanvasNode, operation: OperationState) -> "CanvasState":
        content = {
            "operation_state_id": operation.state_id,
            "node": node.to_json_dict(),
        }
        return cls(canonical_sha256(content), node, operation.state_id)


@dataclass(frozen=True, slots=True)
class StatefulTraceNode:
    trace_id: str
    node_id: str
    state_id: str | None
    status: str
    reused: bool
    details_json: str

    @classmethod
    def create(
        cls,
        *,
        node_id: str,
        state_id: str | None,
        status: str,
        reused: bool,
        details: Mapping[str, object],
    ) -> "StatefulTraceNode":
        if node_id not in STATEFUL_NODE_ORDER:
            raise ValueError("unknown stateful trace node")
        if status not in {"ok", "invalid"}:
            raise ValueError("stateful trace status differs")
        if type(reused) is not bool:
            raise TypeError("stateful trace reuse flag must be boolean")
        normalized_details = _json_value(dict(details))
        details_json = json.dumps(
            normalized_details,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        content = {
            "schema": STATEFUL_TRACE_SCHEMA,
            "node_id": node_id,
            "state_id": state_id,
            "status": status,
            "reused": reused,
            "details": normalized_details,
        }
        return cls(
            canonical_sha256(content),
            node_id,
            state_id,
            status,
            reused,
            details_json,
        )

    @property
    def details(self) -> dict[str, object]:
        return json.loads(self.details_json)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": STATEFUL_TRACE_SCHEMA,
            "trace_id": self.trace_id,
            "node_id": self.node_id,
            "state_id": self.state_id,
            "status": self.status,
            "reused": self.reused,
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class StatefulSceneExecution:
    execution_id: str
    program: ScenePipelineProgram
    input_grid_id: str
    status: str
    output: Grid | None
    reason: str | None
    parse_state: PersistentSceneState
    assignment_state: AssignmentState | None
    operation_state: OperationState | None
    canvas_state: CanvasState | None
    trace: tuple[StatefulTraceNode, ...]
    reused_node_ids: tuple[str, ...]
    executed_node_ids: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": STATEFUL_SCENE_SCHEMA,
            "execution_id": self.execution_id,
            "program_id": _program_id(self.program),
            "input_grid_id": self.input_grid_id,
            "status": self.status,
            "output_grid_id": None if self.output is None else grid_key(self.output),
            "reason": self.reason,
            "parse_state": self.parse_state.to_json_dict(),
            "assignment_state": (
                None
                if self.assignment_state is None
                else self.assignment_state.to_json_dict()
            ),
            "operation_state_id": (
                None if self.operation_state is None else self.operation_state.state_id
            ),
            "canvas_state_id": (
                None if self.canvas_state is None else self.canvas_state.state_id
            ),
            "trace": [item.to_json_dict() for item in self.trace],
            "reused_node_ids": list(self.reused_node_ids),
            "executed_node_ids": list(self.executed_node_ids),
        }


def _render_output(
    program: ScenePipelineProgram,
    grid: Grid,
    parse_state: PersistentSceneState,
    assignment: AssignmentState,
) -> Grid | None:
    """Render with the historical scene semantics, isolated for suffix replay."""

    normalized = as_grid(grid)
    scene = parse_state.scene
    selected = assignment.selected_objects
    correspondences = assignment.correspondences
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
        counts = {
            "selected": len(selected),
            "scene_objects": len(scene.objects),
            "peers": _correspondence_peer_count(selected, correspondences),
        }
        count = counts[program.operate.repeat_rule]
        if count < 1:
            output = None
        elif program.canvas.mode == "count_line":
            if program.operate.axis == "row":
                output = as_grid([[program.operate.output_color] * count])
            else:
                output = as_grid([[program.operate.output_color] for _ in range(count)])
        else:
            if program.canvas.height is None or program.canvas.width is None:
                raise AssertionError("fixed count canvas lost its dimensions")
            if count > program.canvas.height * program.canvas.width:
                output = None
            else:
                values = [program.operate.output_color] * count + [
                    program.canvas.background
                ] * (program.canvas.height * program.canvas.width - count)
                output = as_grid(
                    [
                        values[
                            row * program.canvas.width : (row + 1)
                            * program.canvas.width
                        ]
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
            output = as_grid(
                [
                    [program.canvas.background for _ in range(width)]
                    for _ in range(height)
                ]
            )
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
    return output


def _execution(
    *,
    program: ScenePipelineProgram,
    grid: Grid,
    status: str,
    output: Grid | None,
    reason: str | None,
    parse_state: PersistentSceneState,
    assignment_state: AssignmentState | None,
    operation_state: OperationState | None,
    canvas_state: CanvasState | None,
    trace: Sequence[StatefulTraceNode],
    reused: Sequence[str],
    executed: Sequence[str],
) -> StatefulSceneExecution:
    content = {
        "schema": STATEFUL_SCENE_SCHEMA,
        "program_id": _program_id(program),
        "input_grid_id": grid_key(grid),
        "status": status,
        "output_grid_id": None if output is None else grid_key(output),
        "reason": reason,
        "parse_state_id": parse_state.state_id,
        "assignment_state_id": (
            None if assignment_state is None else assignment_state.state_id
        ),
        "operation_state_id": (
            None if operation_state is None else operation_state.state_id
        ),
        "canvas_state_id": None if canvas_state is None else canvas_state.state_id,
        "trace_ids": [item.trace_id for item in trace],
        "reused_node_ids": list(reused),
        "executed_node_ids": list(executed),
    }
    return StatefulSceneExecution(
        canonical_sha256(content),
        program,
        grid_key(grid),
        status,
        output,
        reason,
        parse_state,
        assignment_state,
        operation_state,
        canvas_state,
        tuple(trace),
        tuple(reused),
        tuple(executed),
    )


def execute_stateful_scene_pipeline(
    program: ScenePipelineProgram,
    grid: Grid,
    *,
    prior: StatefulSceneExecution | None = None,
    changed_node: str | None = None,
) -> StatefulSceneExecution:
    """Execute once or replay the dependency suffix of one typed AST rewrite."""

    if not isinstance(program, ScenePipelineProgram):
        raise TypeError("stateful scene execution requires a scene pipeline program")
    normalized = as_grid(grid)
    input_id = grid_key(normalized)
    if prior is None:
        if changed_node is not None:
            raise ValueError("changed_node requires a parent execution")
        reuse_before: tuple[str, ...] = ()
    else:
        if changed_node not in STATEFUL_NODE_ORDER:
            raise ValueError("typed replay requires one known changed node")
        if prior.input_grid_id != input_id:
            raise ValueError("parent execution belongs to another input grid")
        differences = scene_program_node_differences(prior.program, program)
        if differences != (changed_node,):
            raise ValueError("rewrite must change exactly its declared AST node")
        reuse_before = STATEFUL_NODE_ORDER[: STATEFUL_NODE_ORDER.index(changed_node)]

    trace: list[StatefulTraceNode] = []
    reused: list[str] = []
    executed: list[str] = []

    if "parse" in reuse_before:
        if prior is None:
            raise AssertionError("parse reuse lost its parent execution")
        parse_state = prior.parse_state
        reused.append("parse")
        parse_reused = True
    else:
        previous = (
            prior.parse_state if prior is not None and changed_node == "parse" else None
        )
        parse_state = PersistentSceneState.create(
            normalized, program.parse, previous=previous
        )
        executed.append("parse")
        parse_reused = False
    trace.append(
        StatefulTraceNode.create(
            node_id="parse",
            state_id=parse_state.state_id,
            status="ok",
            reused=parse_reused,
            details={
                "object_count": len(parse_state.objects),
                "relation_count": len(parse_state.relations),
            },
        )
    )

    if not parse_state.objects:
        executed.append("assignment")
        trace.append(
            StatefulTraceNode.create(
                node_id="assignment",
                state_id=None,
                status="invalid",
                reused=False,
                details={"reason": "no_scene_objects"},
            )
        )
        return _execution(
            program=program,
            grid=normalized,
            status="invalid",
            output=None,
            reason="no_scene_objects",
            parse_state=parse_state,
            assignment_state=None,
            operation_state=None,
            canvas_state=None,
            trace=trace,
            reused=reused,
            executed=executed,
        )

    if "assignment" in reuse_before:
        if prior is None or prior.assignment_state is None:
            raise ValueError("parent trace cannot satisfy assignment reuse")
        assignment = prior.assignment_state
        reused.append("assignment")
        assignment_reused = True
    else:
        assignment = AssignmentState.create(program, parse_state)
        executed.append("assignment")
        assignment_reused = False
    if not assignment.selected_objects:
        trace.append(
            StatefulTraceNode.create(
                node_id="assignment",
                state_id=assignment.state_id,
                status="invalid",
                reused=assignment_reused,
                details={"reason": "role_not_instantiated"},
            )
        )
        return _execution(
            program=program,
            grid=normalized,
            status="invalid",
            output=None,
            reason="role_not_instantiated",
            parse_state=parse_state,
            assignment_state=assignment,
            operation_state=None,
            canvas_state=None,
            trace=trace,
            reused=reused,
            executed=executed,
        )
    trace.append(
        StatefulTraceNode.create(
            node_id="assignment",
            state_id=assignment.state_id,
            status="ok",
            reused=assignment_reused,
            details={
                "correspondence_count": len(assignment.correspondences),
                "selected_entity_ids": list(assignment.selected_entity_ids),
            },
        )
    )

    if "operate" in reuse_before:
        if prior is None or prior.operation_state is None:
            raise ValueError("parent trace cannot satisfy operation reuse")
        operation = prior.operation_state
        reused.append("operate")
        operation_reused = True
    else:
        operation = OperationState.create(program.operate, assignment)
        executed.append("operate")
        operation_reused = False
    trace.append(
        StatefulTraceNode.create(
            node_id="operate",
            state_id=operation.state_id,
            status="ok",
            reused=operation_reused,
            details={"operator": program.operate.operator},
        )
    )

    if "canvas" in reuse_before:
        if prior is None or prior.canvas_state is None:
            raise ValueError("parent trace cannot satisfy canvas reuse")
        canvas = prior.canvas_state
        reused.append("canvas")
        canvas_reused = True
    else:
        canvas = CanvasState.create(program.canvas, operation)
        executed.append("canvas")
        canvas_reused = False
    trace.append(
        StatefulTraceNode.create(
            node_id="canvas",
            state_id=canvas.state_id,
            status="ok",
            reused=canvas_reused,
            details={"mode": program.canvas.mode},
        )
    )

    output = _render_output(program, normalized, parse_state, assignment)
    executed.append("render")
    render_state_id = canonical_sha256(
        {
            "canvas_state_id": canvas.state_id,
            "render": program.render.to_json_dict(),
            "output_grid_id": None if output is None else grid_key(output),
        }
    )
    trace.append(
        StatefulTraceNode.create(
            node_id="render",
            state_id=render_state_id,
            status="invalid" if output is None else "ok",
            reused=False,
            details={
                "mode": program.render.mode,
                "reason": None if output is not None else "operation_or_canvas_invalid",
            },
        )
    )
    return _execution(
        program=program,
        grid=normalized,
        status="invalid" if output is None else "ok",
        output=output,
        reason=None if output is not None else "operation_or_canvas_invalid",
        parse_state=parse_state,
        assignment_state=assignment,
        operation_state=operation,
        canvas_state=canvas,
        trace=trace,
        reused=reused,
        executed=executed,
    )
