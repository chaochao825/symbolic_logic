"""Exact cell provenance and one finite provenance-aligned effect language.

The historical scene executor is left untouched.  This module is an opt-in
adapter for supported crop parents: it records an exact inverse map from every
rendered cell to its visible source cell, then uses that map to compile one
typed post-render effect from demonstration residuals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .anchor_mask import ANCHOR_MASK_SPECS, _STENCIL_OFFSETS
from .experiment_safety import canonical_sha256
from .grid import Grid, as_grid, grid_key, grid_to_lists
from .hybrid.scene_graph import ScenePipelineProgram, _transform_coordinate
from .stateful_scene import (
    PersistentEntityId,
    PersistentRelationId,
    StatefulSceneExecution,
    execute_stateful_scene_pipeline,
)


CELL_PROVENANCE_SCHEMA = "afts.cell-provenance/v1"
PROVENANCED_SCENE_SCHEMA = "afts.provenanced-scene-execution/v1"
EFFECT_SUMMARY_SCHEMA = "afts.effect-summary/v1"
RELATIONAL_EFFECT_NODE_TYPE = "provenance_relational_effect"
RELATIONAL_FAILURE_CORE_SCHEMA = "afts.relational-failure-core/v1"
RELATIONAL_EFFECT_PROVIDER_VERSION = "afts-provenance-relational-effect/v0.4"

CELL_PROVENANCE_ORIGINS = (
    "source_entity",
    "source_background",
    "render_background",
)
RELATIONAL_EFFECT_PROGRAMS_PER_DELTA = 10 * len(ANCHOR_MASK_SPECS)

Coordinate = tuple[int, int]


def _arc_color(value: object, *, field: str) -> int:
    if type(value) is not int or not 0 <= value <= 9:
        raise ValueError(f"{field} must be an ARC color")
    return value


def _coordinate(value: object, *, field: str) -> Coordinate:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(item) is not int or item < 0 for item in value)
    ):
        raise ValueError(f"{field} must be a non-negative grid coordinate")
    return value[0], value[1]


@dataclass(frozen=True, slots=True)
class CellProvenance:
    """One exact output-cell lineage record.

    `source_coordinate` is absent when `selected_only` rendering synthesizes a
    background cell instead of copying the geometrically corresponding input
    cell.  A visible parser-background cell retains its source coordinate and
    color but intentionally carries no entity ID.
    """

    provenance_id: str
    output_coordinate: Coordinate
    source_coordinate: Coordinate | None
    source_color: int | None
    origin_kind: str
    entity_ids: tuple[PersistentEntityId, ...]
    relation_ids: tuple[PersistentRelationId, ...]
    producer_node_id: str
    exact: bool

    @classmethod
    def create(
        cls,
        *,
        output_coordinate: Coordinate,
        source_coordinate: Coordinate | None,
        source_color: int | None,
        origin_kind: str,
        entity_ids: Sequence[PersistentEntityId],
        relation_ids: Sequence[PersistentRelationId],
        producer_node_id: str = "render",
        exact: bool = True,
    ) -> "CellProvenance":
        output = _coordinate(output_coordinate, field="output coordinate")
        source = (
            None
            if source_coordinate is None
            else _coordinate(source_coordinate, field="source coordinate")
        )
        if source_color is not None:
            _arc_color(source_color, field="source color")
        if origin_kind not in CELL_PROVENANCE_ORIGINS:
            raise ValueError("unknown cell provenance origin")
        if origin_kind == "render_background":
            if source is not None or source_color is not None or entity_ids:
                raise ValueError("render background cannot claim a source cell")
        elif source is None or source_color is None:
            raise ValueError("source provenance requires a coordinate and color")
        if origin_kind == "source_entity" and not entity_ids:
            raise ValueError("entity provenance requires an entity ID")
        if origin_kind == "source_background" and entity_ids:
            raise ValueError("source background cannot claim an entity ID")
        ordered_entities = tuple(sorted(set(entity_ids)))
        ordered_relations = tuple(sorted(set(relation_ids)))
        if not isinstance(producer_node_id, str) or not producer_node_id:
            raise ValueError("provenance producer node must be non-empty")
        if type(exact) is not bool:
            raise TypeError("provenance exactness must be boolean")
        content = {
            "schema": CELL_PROVENANCE_SCHEMA,
            "output_coordinate": list(output),
            "source_coordinate": None if source is None else list(source),
            "source_color": source_color,
            "origin_kind": origin_kind,
            "entity_ids": list(ordered_entities),
            "relation_ids": list(ordered_relations),
            "producer_node_id": producer_node_id,
            "exact": exact,
        }
        return cls(
            canonical_sha256(content),
            output,
            source,
            source_color,
            origin_kind,
            ordered_entities,
            ordered_relations,
            producer_node_id,
            exact,
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": CELL_PROVENANCE_SCHEMA,
            "provenance_id": self.provenance_id,
            "output_coordinate": list(self.output_coordinate),
            "source_coordinate": (
                None if self.source_coordinate is None else list(self.source_coordinate)
            ),
            "source_color": self.source_color,
            "origin_kind": self.origin_kind,
            "entity_ids": list(self.entity_ids),
            "relation_ids": list(self.relation_ids),
            "producer_node_id": self.producer_node_id,
            "exact": self.exact,
        }


@dataclass(frozen=True, slots=True)
class ProvenancedSceneExecution:
    provenance_execution_id: str
    parent_execution: StatefulSceneExecution
    status: str
    reason: str | None
    cells: tuple[CellProvenance, ...]

    @classmethod
    def create(
        cls,
        *,
        parent_execution: StatefulSceneExecution,
        status: str,
        reason: str | None,
        cells: Sequence[CellProvenance],
    ) -> "ProvenancedSceneExecution":
        if status not in {"exact", "unsupported", "invalid_parent"}:
            raise ValueError("unknown provenanced execution status")
        ordered = tuple(sorted(cells, key=lambda item: item.output_coordinate))
        if len({item.output_coordinate for item in ordered}) != len(ordered):
            raise ValueError("cell provenance repeats an output coordinate")
        if status == "exact":
            if reason is not None or not parent_execution.ok or parent_execution.output is None:
                raise ValueError("exact provenance requires a valid parent output")
            expected = tuple(
                (row, column)
                for row in range(len(parent_execution.output))
                for column in range(len(parent_execution.output[0]))
            )
            if tuple(item.output_coordinate for item in ordered) != expected:
                raise ValueError("exact provenance does not cover the parent output")
            if any(not item.exact for item in ordered):
                raise ValueError("exact execution contains inexact cell provenance")
        elif reason is None or ordered:
            raise ValueError("non-exact provenance requires one reason and no cells")
        content = {
            "schema": PROVENANCED_SCENE_SCHEMA,
            "parent_execution_id": parent_execution.execution_id,
            "status": status,
            "reason": reason,
            "cell_provenance_ids": [item.provenance_id for item in ordered],
        }
        return cls(
            canonical_sha256(content),
            parent_execution,
            status,
            reason,
            ordered,
        )

    @property
    def exact(self) -> bool:
        return self.status == "exact"

    @property
    def output(self) -> Grid | None:
        return self.parent_execution.output

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": PROVENANCED_SCENE_SCHEMA,
            "provenance_execution_id": self.provenance_execution_id,
            "parent_execution_id": self.parent_execution.execution_id,
            "status": self.status,
            "reason": self.reason,
            "cells": [item.to_json_dict() for item in self.cells],
        }


def _unsupported(
    parent: StatefulSceneExecution,
    reason: str,
) -> ProvenancedSceneExecution:
    return ProvenancedSceneExecution.create(
        parent_execution=parent,
        status="invalid_parent" if not parent.ok else "unsupported",
        reason=reason,
        cells=(),
    )


def execute_scene_with_cell_provenance(
    program: ScenePipelineProgram,
    grid: Grid,
) -> ProvenancedSceneExecution:
    """Run the historical parent and add exact provenance when supported."""

    if not isinstance(program, ScenePipelineProgram):
        raise TypeError("cell provenance requires a scene pipeline program")
    normalized = as_grid(grid)
    parent = execute_stateful_scene_pipeline(program, normalized)
    if not parent.ok or parent.output is None:
        return _unsupported(parent, parent.reason or "parent_execution_invalid")
    if program.operate.operator != "crop":
        return _unsupported(parent, "provenance_unsupported_operator")
    if program.canvas.mode != "bbox":
        return _unsupported(parent, "provenance_unsupported_canvas")
    if program.render.mode not in {"source_crop", "selected_only"}:
        return _unsupported(parent, "provenance_unsupported_render")
    if parent.assignment_state is None:
        raise AssertionError("valid crop execution lost its assignment state")

    selected = parent.assignment_state.selected_objects
    selected_cells = tuple(cell for object_ in selected for cell in object_.cells)
    if not selected_cells:
        raise AssertionError("valid crop execution lost its selected cells")
    pad_top, pad_bottom, pad_left, pad_right = program.canvas.padding
    top = min(row for row, _ in selected_cells) - pad_top
    left = min(column for _, column in selected_cells) - pad_left
    bottom = max(row for row, _ in selected_cells) + pad_bottom
    right = max(column for _, column in selected_cells) + pad_right
    source_coordinates = tuple(
        (row, column)
        for row in range(top, bottom + 1)
        for column in range(left, right + 1)
    )
    transformed = tuple(
        _transform_coordinate((row - top, column - left), program.operate.transform)
        for row, column in source_coordinates
    )
    transformed_top = min(row for row, _ in transformed)
    transformed_left = min(column for _, column in transformed)
    occupied = frozenset(selected_cells)

    entity_ids_by_cell: dict[Coordinate, list[PersistentEntityId]] = {}
    for record in parent.parse_state.objects:
        for coordinate in record.cells:
            entity_ids_by_cell.setdefault(coordinate, []).append(record.entity_id)
    records = []
    for source, transformed_coordinate in zip(source_coordinates, transformed):
        output_coordinate = (
            transformed_coordinate[0] - transformed_top,
            transformed_coordinate[1] - transformed_left,
        )
        visible = program.render.mode == "source_crop" or source in occupied
        if visible:
            source_color = normalized[source[0]][source[1]]
            entity_ids = tuple(sorted(entity_ids_by_cell.get(source, ())))
            origin_kind = "source_entity" if entity_ids else "source_background"
            rendered_color = (
                program.canvas.background
                if source_color == program.parse.background
                else source_color
            )
            record = CellProvenance.create(
                output_coordinate=output_coordinate,
                source_coordinate=source,
                source_color=source_color,
                origin_kind=origin_kind,
                entity_ids=entity_ids,
                relation_ids=(),
            )
        else:
            rendered_color = program.canvas.background
            record = CellProvenance.create(
                output_coordinate=output_coordinate,
                source_coordinate=None,
                source_color=None,
                origin_kind="render_background",
                entity_ids=(),
                relation_ids=(),
            )
        observed = parent.output[output_coordinate[0]][output_coordinate[1]]
        if observed != rendered_color:
            raise AssertionError("cell provenance disagrees with historical rendering")
        records.append(record)
    return ProvenancedSceneExecution.create(
        parent_execution=parent,
        status="exact",
        reason=None,
        cells=records,
    )


def lesion_entity_links(
    execution: ProvenancedSceneExecution,
) -> ProvenancedSceneExecution:
    """Remove semantic entity/relation links while retaining exact source cells."""

    if not execution.exact:
        raise ValueError("entity lesion requires exact cell provenance")
    cells = tuple(
        CellProvenance.create(
            output_coordinate=item.output_coordinate,
            source_coordinate=item.source_coordinate,
            source_color=item.source_color,
            origin_kind=(
                "source_background"
                if item.origin_kind == "source_entity"
                else item.origin_kind
            ),
            entity_ids=(),
            relation_ids=(),
            producer_node_id=item.producer_node_id,
            exact=item.exact,
        )
        for item in execution.cells
    )
    return ProvenancedSceneExecution.create(
        parent_execution=execution.parent_execution,
        status="exact",
        reason=None,
        cells=cells,
    )


@dataclass(frozen=True, slots=True)
class RelationalEffectNode:
    anchor_color: int
    effect_kind: str
    effect_parameter: str
    source_color: int
    target_color: int

    def __post_init__(self) -> None:
        _arc_color(self.anchor_color, field="anchor color")
        _arc_color(self.source_color, field="effect source color")
        _arc_color(self.target_color, field="effect target color")
        if self.source_color == self.target_color:
            raise ValueError("effect source and target colors must differ")
        if (self.effect_kind, self.effect_parameter) not in ANCHOR_MASK_SPECS:
            raise ValueError("effect is outside the frozen relational language")

    @property
    def node_id(self) -> str:
        return canonical_sha256(self.to_json_dict())

    @property
    def sort_key(self) -> tuple[object, ...]:
        return (
            self.anchor_color,
            ANCHOR_MASK_SPECS.index((self.effect_kind, self.effect_parameter)),
            self.source_color,
            self.target_color,
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "op": RELATIONAL_EFFECT_NODE_TYPE,
            "anchor_color": self.anchor_color,
            "effect_kind": self.effect_kind,
            "effect_parameter": self.effect_parameter,
            "source_color": self.source_color,
            "target_color": self.target_color,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "RelationalEffectNode":
        expected = {
            "op",
            "anchor_color",
            "effect_kind",
            "effect_parameter",
            "source_color",
            "target_color",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("relational effect has missing or unknown fields")
        if payload["op"] != RELATIONAL_EFFECT_NODE_TYPE:
            raise ValueError("relational effect has the wrong operator")
        return cls(
            payload["anchor_color"],
            payload["effect_kind"],
            payload["effect_parameter"],
            payload["source_color"],
            payload["target_color"],
        )


def relational_effect_domain(
    source_color: int,
    target_color: int,
) -> tuple[RelationalEffectNode, ...]:
    _arc_color(source_color, field="effect source color")
    _arc_color(target_color, field="effect target color")
    if source_color == target_color:
        raise ValueError("effect source and target colors must differ")
    domain = tuple(
        RelationalEffectNode(
            anchor_color,
            effect_kind,
            effect_parameter,
            source_color,
            target_color,
        )
        for anchor_color in range(10)
        for effect_kind, effect_parameter in ANCHOR_MASK_SPECS
    )
    if len(domain) != RELATIONAL_EFFECT_PROGRAMS_PER_DELTA:
        raise AssertionError("relational effect domain changed size")
    return domain


def provenance_anchor_coordinates(
    node: RelationalEffectNode,
    execution: ProvenancedSceneExecution,
) -> tuple[Coordinate, ...]:
    if not isinstance(node, RelationalEffectNode):
        raise TypeError("anchor lookup requires a relational effect node")
    if not execution.exact:
        raise ValueError("anchor lookup requires exact cell provenance")
    return tuple(
        item.output_coordinate
        for item in execution.cells
        if item.source_coordinate is not None and item.source_color == node.anchor_color
    )


def rasterize_relational_effect(
    node: RelationalEffectNode,
    execution: ProvenancedSceneExecution,
) -> tuple[Coordinate, ...]:
    if not execution.exact or execution.output is None:
        raise ValueError("effect rasterization requires exact cell provenance")
    height, width = len(execution.output), len(execution.output[0])
    anchors = provenance_anchor_coordinates(node, execution)
    coordinates: set[Coordinate] = set()
    if node.effect_kind == "stencil":
        offsets = _STENCIL_OFFSETS[node.effect_parameter]
        coordinates.update(
            (row + dr, column + dc)
            for row, column in anchors
            for dr, dc in offsets
            if 0 <= row + dr < height and 0 <= column + dc < width
        )
    elif node.effect_kind == "axis_project":
        if node.effect_parameter == "row":
            coordinates.update(
                (row, column)
                for row in {coordinate[0] for coordinate in anchors}
                for column in range(width)
            )
        else:
            coordinates.update(
                (row, column)
                for column in {coordinate[1] for coordinate in anchors}
                for row in range(height)
            )
    elif anchors:
        inset = 0 if node.effect_parameter == "inset0" else 1
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


def effective_relational_footprint(
    node: RelationalEffectNode,
    execution: ProvenancedSceneExecution,
) -> tuple[Coordinate, ...]:
    if execution.output is None:
        raise ValueError("effect footprint requires a parent output")
    return tuple(
        coordinate
        for coordinate in rasterize_relational_effect(node, execution)
        if execution.output[coordinate[0]][coordinate[1]] == node.source_color
    )


def execute_relational_effect(
    node: RelationalEffectNode,
    execution: ProvenancedSceneExecution,
) -> Grid:
    if execution.output is None:
        raise ValueError("relational effect requires a parent output")
    changed = set(effective_relational_footprint(node, execution))
    return as_grid(
        [
            [
                node.target_color if (row, column) in changed else color
                for column, color in enumerate(values)
            ]
            for row, values in enumerate(execution.output)
        ]
    )


@dataclass(frozen=True, slots=True)
class EffectSummary:
    summary_id: str
    provenance_execution_id: str
    status: str
    parent_shape: tuple[int, int] | None
    target_shape: tuple[int, int]
    delta_pairs: tuple[tuple[int, int], ...]
    changed_cells: tuple[Coordinate, ...]
    source_backed_cells: tuple[Coordinate, ...]
    entity_backed_cells: tuple[Coordinate, ...]
    affected_entity_ids: tuple[PersistentEntityId, ...]
    affected_relation_set_id: str
    affected_relation_count: int
    affected_relation_id_sample: tuple[PersistentRelationId, ...]

    @classmethod
    def create(
        cls,
        execution: ProvenancedSceneExecution,
        target: Grid,
    ) -> "EffectSummary":
        expected = as_grid(target)
        target_shape = len(expected), len(expected[0])
        parent = execution.output
        if not execution.exact or parent is None:
            status = "provenance_unsupported"
            parent_shape = None if parent is None else (len(parent), len(parent[0]))
            delta_pairs: tuple[tuple[int, int], ...] = ()
            changed: tuple[Coordinate, ...] = ()
            source_backed: tuple[Coordinate, ...] = ()
            entity_backed: tuple[Coordinate, ...] = ()
            entity_ids: tuple[PersistentEntityId, ...] = ()
            relation_ids: tuple[PersistentRelationId, ...] = ()
        else:
            parent_shape = len(parent), len(parent[0])
            if parent_shape != target_shape:
                status = "shape_incompatible"
                delta_pairs = ()
                changed = ()
            else:
                changed = tuple(
                    (row, column)
                    for row in range(target_shape[0])
                    for column in range(target_shape[1])
                    if parent[row][column] != expected[row][column]
                )
                delta_pairs = tuple(
                    sorted(
                        {
                            (parent[row][column], expected[row][column])
                            for row, column in changed
                        }
                    )
                )
                if not changed:
                    status = "base_exact"
                elif len(delta_pairs) == 1:
                    status = "single_delta"
                else:
                    status = "multi_delta"
            by_coordinate = {item.output_coordinate: item for item in execution.cells}
            source_backed = tuple(
                coordinate
                for coordinate in changed
                if by_coordinate[coordinate].source_coordinate is not None
            )
            entity_backed = tuple(
                coordinate
                for coordinate in changed
                if by_coordinate[coordinate].entity_ids
            )
            entity_ids = tuple(
                sorted(
                    {
                        entity_id
                        for coordinate in changed
                        for entity_id in by_coordinate[coordinate].entity_ids
                    }
                )
            )
            affected_entity_set = frozenset(entity_ids)
            relation_ids = tuple(
                relation.relation_id
                for relation in execution.parent_execution.parse_state.relations
                if relation.source_entity_id in affected_entity_set
                and relation.target_entity_id in affected_entity_set
            )
        relation_set_id = canonical_sha256(list(relation_ids))
        relation_sample = relation_ids[:16]
        content = {
            "schema": EFFECT_SUMMARY_SCHEMA,
            "provenance_execution_id": execution.provenance_execution_id,
            "status": status,
            "parent_shape": None if parent_shape is None else list(parent_shape),
            "target_shape": list(target_shape),
            "delta_pairs": [list(item) for item in delta_pairs],
            "changed_cells": [list(item) for item in changed],
            "source_backed_cells": [list(item) for item in source_backed],
            "entity_backed_cells": [list(item) for item in entity_backed],
            "affected_entity_ids": list(entity_ids),
            "affected_relation_set_id": relation_set_id,
            "affected_relation_count": len(relation_ids),
            "affected_relation_id_sample": list(relation_sample),
        }
        return cls(
            canonical_sha256(content),
            execution.provenance_execution_id,
            status,
            parent_shape,
            target_shape,
            delta_pairs,
            changed,
            source_backed,
            entity_backed,
            entity_ids,
            relation_set_id,
            len(relation_ids),
            relation_sample,
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": EFFECT_SUMMARY_SCHEMA,
            "summary_id": self.summary_id,
            "provenance_execution_id": self.provenance_execution_id,
            "status": self.status,
            "parent_shape": None if self.parent_shape is None else list(self.parent_shape),
            "target_shape": list(self.target_shape),
            "delta_pairs": [list(item) for item in self.delta_pairs],
            "changed_cells": [list(item) for item in self.changed_cells],
            "source_backed_cells": [list(item) for item in self.source_backed_cells],
            "entity_backed_cells": [list(item) for item in self.entity_backed_cells],
            "affected_entity_ids": list(self.affected_entity_ids),
            "affected_relation_set_id": self.affected_relation_set_id,
            "affected_relation_count": self.affected_relation_count,
            "affected_relation_id_sample": list(self.affected_relation_id_sample),
        }


def relational_effect_demo_domain(
    execution: ProvenancedSceneExecution,
    target: Grid,
    *,
    delta_pair: tuple[int, int] | None = None,
) -> tuple[RelationalEffectNode, ...]:
    summary = EffectSummary.create(execution, target)
    if summary.status == "single_delta":
        observed_delta = summary.delta_pairs[0]
        if delta_pair is not None and delta_pair != observed_delta:
            return ()
        source_color, target_color = observed_delta
    elif summary.status == "base_exact" and delta_pair is not None:
        source_color, target_color = delta_pair
    else:
        return ()
    expected = as_grid(target)
    return tuple(
        node
        for node in relational_effect_domain(source_color, target_color)
        if execute_relational_effect(node, execution) == expected
    )


@dataclass(frozen=True, slots=True)
class RelationalFailureCore:
    core_id: str
    blind_task_id: str
    status: str
    provenance_execution_ids: tuple[str, ...]
    effect_summaries: tuple[EffectSummary, ...]
    candidates: tuple[RelationalEffectNode, ...]
    proof_obligations: tuple[str, ...]
    failure_kind: str | None

    @property
    def identified(self) -> bool:
        return self.status == "identified"

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": RELATIONAL_FAILURE_CORE_SCHEMA,
            "core_id": self.core_id,
            "blind_task_id": self.blind_task_id,
            "status": self.status,
            "provenance_execution_ids": list(self.provenance_execution_ids),
            "effect_summaries": [item.to_json_dict() for item in self.effect_summaries],
            "candidates": [item.to_json_dict() for item in self.candidates],
            "proof_obligations": list(self.proof_obligations),
            "failure_kind": self.failure_kind,
        }


def compile_relational_failure_core(
    *,
    blind_task_id: str,
    executions: Sequence[ProvenancedSceneExecution],
    targets: Sequence[Grid],
) -> RelationalFailureCore:
    """Compile demonstration failures into one finite typed version space."""

    if not isinstance(blind_task_id, str) or not blind_task_id:
        raise ValueError("failure core requires a blind task ID")
    if not executions or len(executions) != len(targets):
        raise ValueError("failure core requires aligned demonstrations")
    summaries = tuple(
        EffectSummary.create(execution, target)
        for execution, target in zip(executions, targets)
    )
    statuses = {summary.status for summary in summaries}
    candidates: tuple[RelationalEffectNode, ...] = ()
    if statuses == {"base_exact"}:
        status = "base_exact"
        failure_kind = None
        obligations: tuple[str, ...] = ()
    elif "provenance_unsupported" in statuses:
        status = "unreachable"
        failure_kind = "provenance_unsupported"
        obligations = ("provide_exact_cell_provenance",)
    elif "shape_incompatible" in statuses:
        status = "unreachable"
        failure_kind = "shape_incompatible"
        obligations = ("reinfer_canvas",)
    elif "multi_delta" in statuses:
        status = "unreachable"
        failure_kind = "delta_semantics_inconsistent"
        obligations = ("expand_effect_delta_type",)
    else:
        delta_pairs = {
            summary.delta_pairs[0]
            for summary in summaries
            if summary.status == "single_delta"
        }
        if len(delta_pairs) != 1:
            status = "unreachable"
            failure_kind = "cross_demo_delta_inconsistent"
            obligations = ("rebind_color_roles",)
        else:
            delta_pair = next(iter(delta_pairs))
            domains = tuple(
                set(
                    relational_effect_demo_domain(
                        execution,
                        target,
                        delta_pair=delta_pair,
                    )
                )
                for execution, target in zip(executions, targets)
            )
            joint = set.intersection(*domains)
            candidates = tuple(sorted(joint, key=lambda item: item.sort_key))
            if candidates:
                status = "identified"
                failure_kind = None
                obligations = (
                    "insert_typed_relational_effect",
                    "replay_affected_render_suffix",
                    "verify_all_demonstrations_exact",
                )
            else:
                status = "unreachable"
                failure_kind = "relation_effect_language_unreachable"
                obligations = ("recruit_new_typed_effect_family",)
    content = {
        "schema": RELATIONAL_FAILURE_CORE_SCHEMA,
        "blind_task_id": blind_task_id,
        "status": status,
        "provenance_execution_ids": [
            execution.provenance_execution_id for execution in executions
        ],
        "effect_summary_ids": [summary.summary_id for summary in summaries],
        "candidate_ids": [candidate.node_id for candidate in candidates],
        "proof_obligations": list(obligations),
        "failure_kind": failure_kind,
    }
    return RelationalFailureCore(
        canonical_sha256(content),
        blind_task_id,
        status,
        tuple(execution.provenance_execution_id for execution in executions),
        summaries,
        candidates,
        obligations,
        failure_kind,
    )


def _global_recolor(grid: Grid, source_color: int, target_color: int) -> Grid:
    return as_grid(
        [
            [target_color if color == source_color else color for color in row]
            for row in grid
        ]
    )


def _component_recolor_outputs(
    grid: Grid,
    source_color: int,
    target_color: int,
) -> tuple[Grid, ...]:
    normalized = as_grid(grid)
    height, width = len(normalized), len(normalized[0])
    outputs = set()
    source_cells = {
        (row, column)
        for row in range(height)
        for column in range(width)
        if normalized[row][column] == source_color
    }
    for connectivity in (4, 8):
        directions = (
            ((-1, 0), (1, 0), (0, -1), (0, 1))
            if connectivity == 4
            else tuple(
                (dr, dc)
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0)
            )
        )
        unseen = set(source_cells)
        while unseen:
            start = min(unseen)
            unseen.remove(start)
            frontier = [start]
            component = {start}
            while frontier:
                row, column = frontier.pop()
                for dr, dc in directions:
                    neighbor = row + dr, column + dc
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.add(neighbor)
                        frontier.append(neighbor)
            outputs.add(
                as_grid(
                    [
                        [
                            target_color if (row, column) in component else color
                            for column, color in enumerate(values)
                        ]
                        for row, values in enumerate(normalized)
                    ]
                )
            )
    return tuple(sorted(outputs, key=grid_key))


def _persistent_entity_outputs(
    execution: ProvenancedSceneExecution,
    source_color: int,
    target_color: int,
) -> tuple[Grid, ...]:
    if not execution.exact or execution.output is None:
        return ()
    outputs = set()
    entity_ids = sorted(
        {entity_id for item in execution.cells for entity_id in item.entity_ids}
    )
    for entity_id in entity_ids:
        changed = {
            item.output_coordinate
            for item in execution.cells
            if entity_id in item.entity_ids
            and execution.output[item.output_coordinate[0]][item.output_coordinate[1]]
            == source_color
        }
        outputs.add(
            as_grid(
                [
                    [
                        target_color if (row, column) in changed else color
                        for column, color in enumerate(values)
                    ]
                    for row, values in enumerate(execution.output)
                ]
            )
        )
    return tuple(sorted(outputs, key=grid_key))


def leave_one_demo_out_relational_effects(
    *,
    blind_task_id: str,
    program: ScenePipelineProgram,
    inputs: Sequence[Grid],
    targets: Sequence[Grid],
) -> dict[str, object]:
    """Strict LODO: every surviving program must agree and be held-out exact."""

    if len(inputs) != len(targets) or len(inputs) < 2:
        raise ValueError("strict LODO requires at least two aligned demonstrations")
    executions = tuple(
        execute_scene_with_cell_provenance(program, input_grid)
        for input_grid in inputs
    )
    folds = []
    for holdout_index, (holdout_execution, holdout_target) in enumerate(
        zip(executions, targets)
    ):
        training_executions = tuple(
            execution
            for index, execution in enumerate(executions)
            if index != holdout_index
        )
        training_targets = tuple(
            target for index, target in enumerate(targets) if index != holdout_index
        )
        core = compile_relational_failure_core(
            blind_task_id=blind_task_id,
            executions=training_executions,
            targets=training_targets,
        )
        predicted = (
            set()
            if not core.identified or not holdout_execution.exact
            else {
                execute_relational_effect(candidate, holdout_execution)
                for candidate in core.candidates
            }
        )
        unanimous = len(predicted) == 1
        exact = unanimous and next(iter(predicted)) == as_grid(holdout_target)
        parent = holdout_execution.output
        identity_exact = parent is not None and parent == as_grid(holdout_target)
        global_exact = False
        component_exact = False
        persistent_entity_exact = False
        if core.identified and parent is not None:
            deltas = {
                (candidate.source_color, candidate.target_color)
                for candidate in core.candidates
            }
            if len(deltas) != 1:
                raise AssertionError("one failure core changed delta semantics")
            source_color, target_color = next(iter(deltas))
            global_exact = (
                _global_recolor(parent, source_color, target_color)
                == as_grid(holdout_target)
            )
            component_exact = any(
                output == as_grid(holdout_target)
                for output in _component_recolor_outputs(
                    parent,
                    source_color,
                    target_color,
                )
            )
            persistent_entity_exact = any(
                output == as_grid(holdout_target)
                for output in _persistent_entity_outputs(
                    holdout_execution,
                    source_color,
                    target_color,
                )
            )
        folds.append(
            {
                "holdout_index": holdout_index,
                "core_id": core.core_id,
                "identified": core.identified,
                "version_space_size": len(core.candidates),
                "prediction_count": len(predicted),
                "unanimous": unanimous,
                "exact": exact,
                "identity_exact": identity_exact,
                "global_recolor_exact": global_exact,
                "component_recolor_exact": component_exact,
                "persistent_entity_exact": persistent_entity_exact,
                "failure_kind": core.failure_kind,
            }
        )
    fold_count = len(folds)
    return {
        "schema": "afts.provenance-relational-effect-lodo/v1",
        "blind_task_id": blind_task_id,
        "fold_count": fold_count,
        "identified_fold_count": sum(int(fold["identified"]) for fold in folds),
        "unanimous_fold_count": sum(int(fold["unanimous"]) for fold in folds),
        "exact_fold_count": sum(int(fold["exact"]) for fold in folds),
        "identity_exact_fold_count": sum(
            int(fold["identity_exact"]) for fold in folds
        ),
        "global_recolor_exact_fold_count": sum(
            int(fold["global_recolor_exact"]) for fold in folds
        ),
        "component_recolor_exact_fold_count": sum(
            int(fold["component_recolor_exact"]) for fold in folds
        ),
        "persistent_entity_exact_fold_count": sum(
            int(fold["persistent_entity_exact"]) for fold in folds
        ),
        "strict_all_folds_exact": all(fold["exact"] for fold in folds),
        "identity_strict_all_folds_exact": all(
            fold["identity_exact"] for fold in folds
        ),
        "global_recolor_strict_all_folds_exact": all(
            fold["global_recolor_exact"] for fold in folds
        ),
        "component_recolor_strict_all_folds_exact": all(
            fold["component_recolor_exact"] for fold in folds
        ),
        "persistent_entity_strict_all_folds_exact": all(
            fold["persistent_entity_exact"] for fold in folds
        ),
        "folds": folds,
    }


def candidate_output_bundle(
    core: RelationalFailureCore,
    executions: Sequence[ProvenancedSceneExecution],
) -> tuple[tuple[Grid, ...], ...]:
    """Return unique, content-ordered query bundles from one identified core."""

    if not core.identified:
        return ()
    if not executions or any(not execution.exact for execution in executions):
        return ()
    bundles = {
        tuple(execute_relational_effect(candidate, execution) for execution in executions)
        for candidate in core.candidates
    }
    return tuple(
        sorted(
            bundles,
            key=lambda bundle: tuple(grid_key(output) for output in bundle),
        )
    )


def effect_bundle_json(bundle: Sequence[Grid]) -> list[list[list[int]]]:
    return [grid_to_lists(output) for output in bundle]
