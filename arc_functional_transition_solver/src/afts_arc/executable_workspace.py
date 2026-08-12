"""Typed partial programs and proof obligations for an executable workspace.

The first workspace contract intentionally reuses the established scene-pipeline
executor.  A sketch may leave any existing node as a typed hole, but it cannot
silently change the legacy program or execution semantics.  Abstract execution
therefore has three outcomes:

* ``incomplete``: the type graph is valid and names legal holes to fill;
* ``exact``: the materialized legacy program matches every demonstration;
* ``incompatible``: a complete program fails and emits a typed rewrite
  obligation backed by its execution trace or observable grid mismatch.

The module accepts :class:`~afts_arc.blind.BlindTask` only.  Query outputs are
therefore structurally unavailable to diagnosis and proposal control.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from .blind import BlindTask
from .grid import Grid, as_grid, grid_to_lists
from .hybrid.scene_graph import (
    CanvasNode,
    CorrespondObjectsNode,
    ExecutionTraceNode,
    ObjectOperationNode,
    ParseObjectsNode,
    RenderObjectsNode,
    ScenePipelineProgram,
    SelectObjectsNode,
    execute_scene_pipeline,
)


TYPED_SKETCH_SCHEMA_VERSION = "afts.typed-program-sketch/v0.1"
TYPED_SKETCH_TOPOLOGY_SCHEMA_VERSION = "afts.typed-program-sketch/v0.2"
ABSTRACT_EXECUTION_SCHEMA_VERSION = "afts.abstract-execution/v0.1"
RECOLOR_REACHABILITY_SCHEMA_VERSION = "afts.recolor-reachability/v0.1"
INPUT_GRID_NODE_ID = "$input"
RECOLOR_NODE_ID = "post_recolor"

SCENE_PIPELINE_NODE_TYPES = (
    "parse",
    "correspond",
    "select",
    "operate",
    "canvas",
    "render",
)
RECOLOR_NODE_TYPE = "recolor_grid"
SKETCH_NODE_TYPES = SCENE_PIPELINE_NODE_TYPES + (RECOLOR_NODE_TYPE,)
RECOLOR_TOPOLOGY_NODE_TYPES = SCENE_PIPELINE_NODE_TYPES + (RECOLOR_NODE_TYPE,)


@dataclass(frozen=True, slots=True)
class RecolorGridNode:
    """Replace one ARC color globally while preserving grid shape."""

    source_color: int
    target_color: int

    def __post_init__(self) -> None:
        if type(self.source_color) is not int or not 0 <= self.source_color <= 9:
            raise ValueError("recolor source must be an ARC color")
        if type(self.target_color) is not int or not 0 <= self.target_color <= 9:
            raise ValueError("recolor target must be an ARC color")
        if self.source_color == self.target_color:
            raise ValueError("recolor source and target must differ")

    @property
    def node_id(self) -> str:
        return _content_id(self.to_json_dict())

    def to_json_dict(self) -> dict[str, object]:
        return {
            "op": RECOLOR_NODE_TYPE,
            "source_color": self.source_color,
            "target_color": self.target_color,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "RecolorGridNode":
        expected = {"op", "source_color", "target_color"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("recolor node has missing or unknown fields")
        if payload["op"] != RECOLOR_NODE_TYPE:
            raise ValueError("recolor node has the wrong operator")
        return cls(payload["source_color"], payload["target_color"])


def execute_recolor_grid(node: RecolorGridNode, grid: Grid) -> Grid:
    """Execute one deterministic global recolor."""

    if not isinstance(node, RecolorGridNode):
        raise TypeError("recolor execution requires a RecolorGridNode")
    normalized = as_grid(grid)
    return as_grid(
        [
            [
                node.target_color if color == node.source_color else color
                for color in row
            ]
            for row in normalized
        ]
    )


RECOLOR_NODE_DOMAIN = tuple(
    RecolorGridNode(source_color, target_color)
    for source_color in range(10)
    for target_color in range(10)
    if source_color != target_color
)

_NODE_INPUT_TYPES = {
    "parse": ("grid",),
    "correspond": ("scene",),
    "select": ("scene", "correspondence_set"),
    "operate": ("grid", "scene", "object_set"),
    "canvas": ("operation_plan",),
    "render": ("grid", "scene", "object_set", "operation_plan", "canvas"),
    RECOLOR_NODE_TYPE: ("grid",),
}
_NODE_OUTPUT_TYPES = {
    "parse": "scene",
    "correspond": "correspondence_set",
    "select": "object_set",
    "operate": "operation_plan",
    "canvas": "canvas",
    "render": "grid",
    RECOLOR_NODE_TYPE: "grid",
}
_NODE_BINDING_CLASSES = {
    "parse": ParseObjectsNode,
    "correspond": CorrespondObjectsNode,
    "select": SelectObjectsNode,
    "operate": ObjectOperationNode,
    "canvas": CanvasNode,
    "render": RenderObjectsNode,
    RECOLOR_NODE_TYPE: RecolorGridNode,
}

PROOF_OBLIGATION_TYPES = (
    "fill_typed_hole",
    "reparse",
    "selection_conflict",
    "production_mismatch",
    "canvas_mismatch",
    "render_mismatch",
    "postprocess_mismatch",
    "insert_typed_node",
    "representation_unreachable",
    "execution_conflict",
    "executor_failure",
)
_OBLIGATION_LEGAL_ACTIONS = {
    "fill_typed_hole": ("fill_typed_hole",),
    "reparse": ("reparse_background", "replace_parse"),
    "selection_conflict": ("object_rematch", "replace_selector"),
    "production_mismatch": (
        "object_rematch",
        "replace_selector",
        "replace_operation",
        "insert_typed_node",
    ),
    "canvas_mismatch": ("canvas_reinfer",),
    "render_mismatch": ("replace_render",),
    "postprocess_mismatch": ("replace_recolor",),
    "insert_typed_node": ("insert_typed_node",),
    "representation_unreachable": ("expand_representation",),
    "execution_conflict": ("replace_operation", "canvas_reinfer"),
    "executor_failure": ("inspect_executor",),
}


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


def _validate_identifier(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be a non-empty canonical string")
    return value


def _validate_evidence_value(value: object) -> None:
    if type(value) not in {bool, int, str}:
        raise TypeError("proof-obligation evidence must contain JSON scalar values")


@dataclass(frozen=True, slots=True)
class TypedHole:
    """One unresolved node binding with an explicit required node type."""

    hole_id: str
    expected_node_type: str
    constraints: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.hole_id, field_name="typed hole ID")
        if self.expected_node_type not in SKETCH_NODE_TYPES:
            raise ValueError("typed hole has an unknown expected node type")
        if tuple(sorted(self.constraints)) != self.constraints:
            raise ValueError("typed-hole constraints must be canonically sorted")
        keys = tuple(key for key, _ in self.constraints)
        if len(keys) != len(set(keys)):
            raise ValueError("typed-hole constraint keys must be unique")
        for key, value in self.constraints:
            _validate_identifier(key, field_name="typed-hole constraint key")
            _validate_evidence_value(value)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "hole_id": self.hole_id,
            "expected_node_type": self.expected_node_type,
            "constraints": [[key, value] for key, value in self.constraints],
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "TypedHole":
        expected = {"hole_id", "expected_node_type", "constraints"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("typed hole has missing or unknown fields")
        constraints = payload["constraints"]
        if not isinstance(constraints, list) or any(
            not isinstance(item, list) or len(item) != 2 for item in constraints
        ):
            raise ValueError("typed-hole constraints must be key-value pairs")
        return cls(
            payload["hole_id"],
            payload["expected_node_type"],
            tuple((item[0], item[1]) for item in constraints),
        )


SceneNodeBinding = (
    ParseObjectsNode
    | CorrespondObjectsNode
    | SelectObjectsNode
    | ObjectOperationNode
    | CanvasNode
    | RenderObjectsNode
    | RecolorGridNode
)


@dataclass(frozen=True, slots=True)
class TypedSketchNode:
    """One topologically ordered node in a typed partial program."""

    node_id: str
    node_type: str
    input_ids: tuple[str, ...]
    binding: SceneNodeBinding | TypedHole

    def __post_init__(self) -> None:
        _validate_identifier(self.node_id, field_name="sketch node ID")
        if self.node_id == INPUT_GRID_NODE_ID:
            raise ValueError("the reserved input-grid ID cannot name a sketch node")
        if self.node_type not in SKETCH_NODE_TYPES:
            raise ValueError("sketch node has an unknown type")
        if not isinstance(self.input_ids, tuple) or any(
            not isinstance(item, str) for item in self.input_ids
        ):
            raise TypeError("sketch node inputs must be a tuple of strings")
        if len(self.input_ids) != len(set(self.input_ids)):
            raise ValueError("sketch node inputs must be unique")
        if isinstance(self.binding, TypedHole):
            if self.binding.expected_node_type != self.node_type:
                raise TypeError("typed hole does not match its sketch node type")
        elif not isinstance(self.binding, _NODE_BINDING_CLASSES[self.node_type]):
            raise TypeError("concrete binding does not match its sketch node type")

    @property
    def output_type(self) -> str:
        return _NODE_OUTPUT_TYPES[self.node_type]

    def to_json_dict(self) -> dict[str, object]:
        if isinstance(self.binding, TypedHole):
            binding = {"kind": "hole", "value": self.binding.to_json_dict()}
        else:
            binding = {"kind": "concrete", "value": self.binding.to_json_dict()}
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "input_ids": list(self.input_ids),
            "binding": binding,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "TypedSketchNode":
        expected = {"node_id", "node_type", "input_ids", "binding"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("typed sketch node has missing or unknown fields")
        node_type = payload["node_type"]
        if node_type not in SKETCH_NODE_TYPES:
            raise ValueError("typed sketch node has an unknown type")
        input_ids = payload["input_ids"]
        if not isinstance(input_ids, list) or any(
            not isinstance(item, str) for item in input_ids
        ):
            raise ValueError("typed sketch node inputs must be a list of strings")
        binding_payload = payload["binding"]
        if not isinstance(binding_payload, Mapping) or set(binding_payload) != {
            "kind",
            "value",
        }:
            raise ValueError("typed sketch binding has missing or unknown fields")
        if binding_payload["kind"] == "hole":
            binding: SceneNodeBinding | TypedHole = TypedHole.from_json_dict(
                binding_payload["value"]
            )
        elif binding_payload["kind"] == "concrete":
            binding = _NODE_BINDING_CLASSES[node_type].from_json_dict(
                binding_payload["value"]
            )
        else:
            raise ValueError("typed sketch binding has an unknown kind")
        return cls(payload["node_id"], node_type, tuple(input_ids), binding)


@dataclass(frozen=True, slots=True)
class TypedProgramSketch:
    """A content-addressed, type-checked partial scene program."""

    nodes: tuple[TypedSketchNode, ...]
    output_node_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.nodes, tuple) or not self.nodes:
            raise ValueError("typed program sketch requires an ordered node tuple")
        if any(not isinstance(node, TypedSketchNode) for node in self.nodes):
            raise TypeError("typed program sketch contains a non-node value")
        node_ids = tuple(node.node_id for node in self.nodes)
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("typed program sketch node IDs must be unique")
        _validate_identifier(self.output_node_id, field_name="sketch output node ID")

        value_types = {INPUT_GRID_NODE_ID: "grid"}
        for node in self.nodes:
            if any(input_id not in value_types for input_id in node.input_ids):
                raise ValueError("typed sketch nodes must be dependency ordered")
            observed_inputs = tuple(value_types[input_id] for input_id in node.input_ids)
            expected_inputs = _NODE_INPUT_TYPES[node.node_type]
            if observed_inputs != expected_inputs:
                raise TypeError(
                    f"{node.node_type} requires inputs {expected_inputs}, "
                    f"found {observed_inputs}"
                )
            value_types[node.node_id] = node.output_type
        if self.output_node_id not in value_types:
            raise ValueError("typed sketch output node does not exist")
        if value_types[self.output_node_id] != "grid":
            raise TypeError("typed sketch output must have grid type")

        hole_ids = tuple(hole.hole_id for hole in self.holes)
        if len(hole_ids) != len(set(hole_ids)):
            raise ValueError("typed program sketch hole IDs must be unique")
        node_types = tuple(node.node_type for node in self.nodes)
        if node_types == SCENE_PIPELINE_NODE_TYPES and (
            self.output_node_id != self.nodes[-1].node_id
        ):
            raise ValueError("scene pipeline output must be its render node")
        if node_types == RECOLOR_TOPOLOGY_NODE_TYPES:
            recolor_node = self.nodes[-1]
            if recolor_node.node_id != RECOLOR_NODE_ID:
                raise ValueError("recolor topology must use the canonical node ID")
            if recolor_node.input_ids != (self.nodes[-2].node_id,):
                raise ValueError("recolor topology must extend the render output edge")
            if self.output_node_id != recolor_node.node_id:
                raise ValueError("recolor topology output must be the appended node")
        prefix_complete = len(self.nodes) >= len(SCENE_PIPELINE_NODE_TYPES) and not any(
            isinstance(node.binding, TypedHole)
            for node in self.nodes[: len(SCENE_PIPELINE_NODE_TYPES)]
        )
        if prefix_complete and node_types in {
            SCENE_PIPELINE_NODE_TYPES,
            RECOLOR_TOPOLOGY_NODE_TYPES,
        }:
            # The legacy program constructor is also the refinement-type checker:
            # it rejects individually typed nodes whose cross-node contracts are
            # incompatible (for example, a signature selector without matching
            # correspondence features).
            self.materialize_scene_pipeline_prefix()

    @property
    def holes(self) -> tuple[TypedHole, ...]:
        return tuple(
            node.binding
            for node in self.nodes
            if isinstance(node.binding, TypedHole)
        )

    @property
    def complete(self) -> bool:
        return not self.holes

    @property
    def sketch_id(self) -> str:
        return _content_id(self.to_json_dict())

    def to_json_dict(self) -> dict[str, object]:
        schema = (
            TYPED_SKETCH_TOPOLOGY_SCHEMA_VERSION
            if any(node.node_type == RECOLOR_NODE_TYPE for node in self.nodes)
            else TYPED_SKETCH_SCHEMA_VERSION
        )
        return {
            "schema": schema,
            "nodes": [node.to_json_dict() for node in self.nodes],
            "output_node_id": self.output_node_id,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "TypedProgramSketch":
        expected = {"schema", "nodes", "output_node_id"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("typed program sketch has missing or unknown fields")
        if payload["schema"] not in {
            TYPED_SKETCH_SCHEMA_VERSION,
            TYPED_SKETCH_TOPOLOGY_SCHEMA_VERSION,
        }:
            raise ValueError("unsupported typed program sketch schema")
        nodes = payload["nodes"]
        if not isinstance(nodes, list):
            raise ValueError("typed program sketch nodes must be a list")
        sketch = cls(
            tuple(TypedSketchNode.from_json_dict(node) for node in nodes),
            payload["output_node_id"],
        )
        has_recolor = any(
            node.node_type == RECOLOR_NODE_TYPE for node in sketch.nodes
        )
        if (
            payload["schema"] == TYPED_SKETCH_SCHEMA_VERSION and has_recolor
        ) or (
            payload["schema"] == TYPED_SKETCH_TOPOLOGY_SCHEMA_VERSION
            and not has_recolor
        ):
            raise ValueError("typed sketch topology does not match its schema")
        return sketch

    @classmethod
    def from_scene_pipeline(
        cls,
        program: ScenePipelineProgram,
        *,
        holes: tuple[TypedHole, ...] = (),
    ) -> "TypedProgramSketch":
        if not isinstance(program, ScenePipelineProgram):
            raise TypeError("typed sketch source must be a scene pipeline")
        if any(not isinstance(hole, TypedHole) for hole in holes):
            raise TypeError("scene pipeline holes must contain TypedHole values")
        holes_by_type = {hole.expected_node_type: hole for hole in holes}
        if len(holes_by_type) != len(holes):
            raise ValueError("scene pipeline accepts at most one hole per node type")

        bindings: dict[str, SceneNodeBinding | TypedHole] = {
            "parse": program.parse,
            "correspond": program.correspond,
            "select": program.select,
            "operate": program.operate,
            "canvas": program.canvas,
            "render": program.render,
        }
        for node_type, hole in holes_by_type.items():
            bindings[node_type] = hole
        return cls(
            (
                TypedSketchNode("parse", "parse", (INPUT_GRID_NODE_ID,), bindings["parse"]),
                TypedSketchNode("correspond", "correspond", ("parse",), bindings["correspond"]),
                TypedSketchNode("select", "select", ("parse", "correspond"), bindings["select"]),
                TypedSketchNode(
                    "operate",
                    "operate",
                    (INPUT_GRID_NODE_ID, "parse", "select"),
                    bindings["operate"],
                ),
                TypedSketchNode("canvas", "canvas", ("operate",), bindings["canvas"]),
                TypedSketchNode(
                    "render",
                    "render",
                    (INPUT_GRID_NODE_ID, "parse", "select", "operate", "canvas"),
                    bindings["render"],
                ),
            ),
            "render",
        )

    def fill_hole(
        self,
        hole_id: str,
        binding: SceneNodeBinding,
    ) -> "TypedProgramSketch":
        _validate_identifier(hole_id, field_name="typed hole ID")
        matches = tuple(
            (index, node)
            for index, node in enumerate(self.nodes)
            if isinstance(node.binding, TypedHole) and node.binding.hole_id == hole_id
        )
        if len(matches) != 1:
            raise ValueError("typed hole ID must identify exactly one unresolved node")
        index, node = matches[0]
        replacement = TypedSketchNode(
            node.node_id,
            node.node_type,
            node.input_ids,
            binding,
        )
        nodes = self.nodes[:index] + (replacement,) + self.nodes[index + 1 :]
        return TypedProgramSketch(nodes, self.output_node_id)

    def materialize_scene_pipeline(self) -> ScenePipelineProgram:
        if not self.complete:
            raise ValueError("an incomplete typed sketch cannot be materialized")
        if tuple(node.node_type for node in self.nodes) != SCENE_PIPELINE_NODE_TYPES:
            raise ValueError("legacy materialization requires the canonical scene pipeline")
        return self.materialize_scene_pipeline_prefix()

    def materialize_scene_pipeline_prefix(self) -> ScenePipelineProgram:
        if len(self.nodes) < len(SCENE_PIPELINE_NODE_TYPES) or tuple(
            node.node_type for node in self.nodes[: len(SCENE_PIPELINE_NODE_TYPES)]
        ) != SCENE_PIPELINE_NODE_TYPES:
            raise ValueError("sketch does not start with the canonical scene pipeline")
        bindings = tuple(
            node.binding for node in self.nodes[: len(SCENE_PIPELINE_NODE_TYPES)]
        )
        if any(isinstance(binding, TypedHole) for binding in bindings):
            raise ValueError("scene pipeline prefix contains an unresolved hole")
        if not isinstance(bindings[0], ParseObjectsNode):
            raise TypeError("parse binding has the wrong concrete type")
        if not isinstance(bindings[1], CorrespondObjectsNode):
            raise TypeError("correspondence binding has the wrong concrete type")
        if not isinstance(bindings[2], SelectObjectsNode):
            raise TypeError("selector binding has the wrong concrete type")
        if not isinstance(bindings[3], ObjectOperationNode):
            raise TypeError("operation binding has the wrong concrete type")
        if not isinstance(bindings[4], CanvasNode):
            raise TypeError("canvas binding has the wrong concrete type")
        if not isinstance(bindings[5], RenderObjectsNode):
            raise TypeError("render binding has the wrong concrete type")
        return ScenePipelineProgram(
            bindings[0],
            bindings[1],
            bindings[2],
            bindings[3],
            bindings[4],
            bindings[5],
        )

    @property
    def topology(self) -> str:
        node_types = tuple(node.node_type for node in self.nodes)
        if node_types == SCENE_PIPELINE_NODE_TYPES:
            return "scene_pipeline"
        if node_types == RECOLOR_TOPOLOGY_NODE_TYPES:
            return "scene_pipeline_recolor"
        return "unsupported"

    @property
    def recolor_binding(self) -> RecolorGridNode | TypedHole | None:
        if self.topology == "scene_pipeline":
            return None
        if self.topology != "scene_pipeline_recolor":
            raise ValueError("sketch does not have a supported recolor topology")
        binding = self.nodes[len(SCENE_PIPELINE_NODE_TYPES)].binding
        if not isinstance(binding, (RecolorGridNode, TypedHole)):
            raise TypeError("recolor topology has the wrong postprocess binding")
        return binding


@dataclass(frozen=True, slots=True)
class ProofObligation:
    """A typed, replayable reason why the current sketch is not yet accepted."""

    obligation_type: str
    target_node_ids: tuple[str, ...]
    target_node_types: tuple[str, ...]
    demo_indices: tuple[int, ...]
    evidence: tuple[tuple[str, object], ...] = ()
    hole_id: str | None = None

    def __post_init__(self) -> None:
        if self.obligation_type not in PROOF_OBLIGATION_TYPES:
            raise ValueError("unknown proof-obligation type")
        if len(self.target_node_ids) != len(self.target_node_types):
            raise ValueError("proof-obligation node IDs and types must align")
        if any(item not in SKETCH_NODE_TYPES for item in self.target_node_types):
            raise ValueError("proof obligation targets an unknown node type")
        if len(self.target_node_ids) != len(set(self.target_node_ids)):
            raise ValueError("proof-obligation target nodes must be unique")
        if tuple(sorted(set(self.demo_indices))) != self.demo_indices:
            raise ValueError("proof-obligation demo indices must be unique and sorted")
        if any(type(index) is not int or index < 0 for index in self.demo_indices):
            raise ValueError("proof-obligation demo indices must be non-negative")
        if tuple(sorted(self.evidence)) != self.evidence:
            raise ValueError("proof-obligation evidence must be canonically sorted")
        evidence_keys = tuple(key for key, _ in self.evidence)
        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError("proof-obligation evidence keys must be unique")
        for key, value in self.evidence:
            _validate_identifier(key, field_name="proof-obligation evidence key")
            _validate_evidence_value(value)
        if self.obligation_type == "fill_typed_hole":
            if self.hole_id is None:
                raise ValueError("fill-hole obligation requires a typed hole ID")
            _validate_identifier(self.hole_id, field_name="proof-obligation hole ID")
        elif self.hole_id is not None:
            raise ValueError("only a fill-hole obligation may name a typed hole")

    @property
    def legal_actions(self) -> tuple[str, ...]:
        return _OBLIGATION_LEGAL_ACTIONS[self.obligation_type]

    @property
    def obligation_id(self) -> str:
        return _content_id(self.to_json_dict())

    def to_json_dict(self) -> dict[str, object]:
        return {
            "obligation_type": self.obligation_type,
            "target_node_ids": list(self.target_node_ids),
            "target_node_types": list(self.target_node_types),
            "demo_indices": list(self.demo_indices),
            "evidence": [[key, value] for key, value in self.evidence],
            "hole_id": self.hole_id,
            "legal_actions": list(self.legal_actions),
        }


@dataclass(frozen=True, slots=True)
class AbstractDemoExecution:
    demo_index: int
    status: str
    output: Grid | None
    reason: str | None
    node_trace: tuple[ExecutionTraceNode, ...]
    shape_matches: bool | None
    pixel_mismatch_count: int | None

    def __post_init__(self) -> None:
        if type(self.demo_index) is not int or self.demo_index < 0:
            raise ValueError("abstract demo index must be non-negative")
        if self.status not in {"exact", "incompatible"}:
            raise ValueError("abstract demo status must be exact or incompatible")
        if self.status == "exact":
            if self.output is None or self.reason is not None:
                raise ValueError("exact abstract demo execution has inconsistent fields")
            if self.shape_matches is not True or self.pixel_mismatch_count != 0:
                raise ValueError("exact abstract demo execution must have zero mismatch")
        if self.status == "incompatible" and self.reason is None:
            raise ValueError("incompatible abstract demo execution requires a reason")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "demo_index": self.demo_index,
            "status": self.status,
            "output": None if self.output is None else grid_to_lists(self.output),
            "reason": self.reason,
            "node_trace": [
                {
                    "node_id": node.node_id,
                    "operator": node.operator,
                    "status": node.status,
                    "details": [[key, value] for key, value in node.details],
                }
                for node in self.node_trace
            ],
            "shape_matches": self.shape_matches,
            "pixel_mismatch_count": self.pixel_mismatch_count,
        }


@dataclass(frozen=True, slots=True)
class AbstractExecutionResult:
    blind_task_id: str
    sketch_id: str
    status: str
    demo_executions: tuple[AbstractDemoExecution, ...]
    obligations: tuple[ProofObligation, ...]

    def __post_init__(self) -> None:
        _validate_identifier(self.blind_task_id, field_name="blind task ID")
        _validate_identifier(self.sketch_id, field_name="typed sketch ID")
        if self.status not in {"incomplete", "exact", "incompatible"}:
            raise ValueError("unknown abstract execution status")
        if self.status == "incomplete" and (
            self.demo_executions or not self.obligations
        ):
            raise ValueError("incomplete abstract execution must contain only obligations")
        if self.status == "exact" and (
            not self.demo_executions
            or self.obligations
            or any(item.status != "exact" for item in self.demo_executions)
        ):
            raise ValueError("exact abstract execution has inconsistent evidence")
        if self.status == "incompatible" and (
            not self.demo_executions or not self.obligations
        ):
            raise ValueError("incompatible abstract execution requires evidence")

    @property
    def exact(self) -> bool:
        return self.status == "exact"

    @property
    def result_id(self) -> str:
        return _content_id(self.to_json_dict())

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": ABSTRACT_EXECUTION_SCHEMA_VERSION,
            "blind_task_id": self.blind_task_id,
            "sketch_id": self.sketch_id,
            "status": self.status,
            "demo_executions": [item.to_json_dict() for item in self.demo_executions],
            "obligations": [item.to_json_dict() for item in self.obligations],
        }


@dataclass(frozen=True, slots=True)
class TypedSketchExecution:
    """Concrete execution of one complete supported sketch topology."""

    status: str
    output: Grid | None
    reason: str | None
    node_trace: tuple[ExecutionTraceNode, ...]

    def __post_init__(self) -> None:
        if self.status not in {"ok", "invalid"}:
            raise ValueError("typed sketch execution status must be ok or invalid")
        if self.status == "ok" and (self.output is None or self.reason is not None):
            raise ValueError("successful typed sketch execution has inconsistent fields")
        if self.status == "invalid" and self.reason is None:
            raise ValueError("invalid typed sketch execution requires a reason")

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def execute_typed_sketch(
    sketch: TypedProgramSketch,
    grid: Grid,
) -> TypedSketchExecution:
    """Execute a legacy or one-recolor sketch without changing prefix semantics."""

    if not isinstance(sketch, TypedProgramSketch):
        raise TypeError("typed execution requires a TypedProgramSketch")
    if not sketch.complete:
        raise ValueError("an incomplete typed sketch cannot execute")
    if sketch.topology not in {"scene_pipeline", "scene_pipeline_recolor"}:
        raise ValueError("typed execution does not support this sketch topology")

    program = sketch.materialize_scene_pipeline_prefix()
    prefix = execute_scene_pipeline(program, grid)
    if not prefix.ok:
        if prefix.reason is None:
            raise ValueError("invalid scene execution is missing its reason")
        return TypedSketchExecution(
            "invalid",
            None,
            prefix.reason,
            prefix.node_trace,
        )
    if prefix.output is None:
        raise ValueError("successful scene execution is missing its output")

    output = prefix.output
    trace = prefix.node_trace
    binding = sketch.recolor_binding
    if binding is not None:
        if not isinstance(binding, RecolorGridNode):
            raise TypeError("complete recolor topology has an unresolved binding")
        output = execute_recolor_grid(binding, output)
        trace = trace + (
            ExecutionTraceNode.create(
                RECOLOR_NODE_ID,
                RECOLOR_NODE_TYPE,
                "ok",
                source_color=binding.source_color,
                target_color=binding.target_color,
            ),
        )
    return TypedSketchExecution("ok", output, None, trace)


def _node_ids_by_type(sketch: TypedProgramSketch) -> dict[str, str]:
    if sketch.topology not in {"scene_pipeline", "scene_pipeline_recolor"}:
        raise ValueError("abstract execution requires a supported canonical topology")
    node_ids = {node.node_type: node.node_id for node in sketch.nodes}
    if any(node_type not in node_ids for node_type in SCENE_PIPELINE_NODE_TYPES):
        raise ValueError("abstract execution requires every scene-pipeline node")
    return node_ids


def _proof_obligation(
    obligation_type: str,
    *,
    node_ids: dict[str, str],
    node_types: tuple[str, ...],
    demo_indices: tuple[int, ...],
    hole_id: str | None = None,
    **evidence: object,
) -> ProofObligation:
    return ProofObligation(
        obligation_type,
        tuple(node_ids[node_type] for node_type in node_types),
        node_types,
        demo_indices,
        tuple(sorted(evidence.items())),
        hole_id,
    )


def _support_mask(grid: Grid, *, background: int) -> tuple[tuple[bool, ...], ...]:
    return tuple(tuple(cell != background for cell in row) for row in grid)


def _mismatch_obligation(
    *,
    sketch: TypedProgramSketch,
    program: ScenePipelineProgram,
    demo_index: int,
    predicted: Grid,
    expected: Grid,
) -> tuple[ProofObligation, bool, int | None, str]:
    node_ids = _node_ids_by_type(sketch)
    shape_matches = (
        len(predicted) == len(expected) and len(predicted[0]) == len(expected[0])
    )
    if not shape_matches:
        return (
            _proof_obligation(
                "canvas_mismatch",
                node_ids=node_ids,
                node_types=("canvas",),
                demo_indices=(demo_index,),
                predicted_height=len(predicted),
                predicted_width=len(predicted[0]),
                expected_height=len(expected),
                expected_width=len(expected[0]),
            ),
            False,
            None,
            "shape_mismatch",
        )

    mismatch_count = sum(
        predicted[row][column] != expected[row][column]
        for row in range(len(expected))
        for column in range(len(expected[0]))
    )
    common_evidence = {
        "pixel_mismatch_count": mismatch_count,
        "height": len(expected),
        "width": len(expected[0]),
    }
    if sketch.topology == "scene_pipeline_recolor":
        obligation = _proof_obligation(
            "postprocess_mismatch",
            node_ids=node_ids,
            node_types=(RECOLOR_NODE_TYPE,),
            demo_indices=(demo_index,),
            **common_evidence,
        )
        reason = "postprocess_mismatch"
    elif _support_mask(predicted, background=program.canvas.background) == _support_mask(
        expected,
        background=program.canvas.background,
    ):
        obligation = _proof_obligation(
            "render_mismatch",
            node_ids=node_ids,
            node_types=("render",),
            demo_indices=(demo_index,),
            **common_evidence,
        )
        reason = "render_mismatch"
    else:
        obligation = _proof_obligation(
            "production_mismatch",
            node_ids=node_ids,
            node_types=("correspond", "select", "operate"),
            demo_indices=(demo_index,),
            **common_evidence,
        )
        reason = "production_mismatch"
    return obligation, True, mismatch_count, reason


def _invalid_execution_obligation(
    *,
    sketch: TypedProgramSketch,
    demo_index: int,
    reason: str,
) -> ProofObligation:
    node_ids = _node_ids_by_type(sketch)
    if reason == "no_scene_objects":
        return _proof_obligation(
            "reparse",
            node_ids=node_ids,
            node_types=("parse",),
            demo_indices=(demo_index,),
            execution_reason=reason,
        )
    if reason == "role_not_instantiated":
        return _proof_obligation(
            "selection_conflict",
            node_ids=node_ids,
            node_types=("correspond", "select"),
            demo_indices=(demo_index,),
            execution_reason=reason,
        )
    if reason == "operation_or_canvas_invalid":
        return _proof_obligation(
            "execution_conflict",
            node_ids=node_ids,
            node_types=("operate", "canvas"),
            demo_indices=(demo_index,),
            execution_reason=reason,
        )
    return ProofObligation(
        "executor_failure",
        (),
        (),
        (demo_index,),
        (("execution_reason", reason),),
    )


def abstract_execute(
    sketch: TypedProgramSketch,
    task: BlindTask,
) -> AbstractExecutionResult:
    """Diagnose a typed sketch using demonstration evidence only."""

    if not isinstance(sketch, TypedProgramSketch):
        raise TypeError("abstract execution requires a TypedProgramSketch")
    if not isinstance(task, BlindTask):
        raise TypeError("abstract execution requires an oracle-free BlindTask")

    if not sketch.complete:
        node_ids = {node.node_type: node.node_id for node in sketch.nodes}
        demo_indices = tuple(range(len(task.train)))
        obligations = tuple(
            _proof_obligation(
                "fill_typed_hole",
                node_ids=node_ids,
                node_types=(hole.expected_node_type,),
                demo_indices=demo_indices,
                hole_id=hole.hole_id,
                parent_sketch_id=sketch.sketch_id,
                required_node_type=hole.expected_node_type,
            )
            for hole in sketch.holes
        )
        return AbstractExecutionResult(
            task.task_id,
            sketch.sketch_id,
            "incomplete",
            (),
            obligations,
        )

    program = sketch.materialize_scene_pipeline_prefix()
    demo_executions: list[AbstractDemoExecution] = []
    obligations: list[ProofObligation] = []
    for demo_index, pair in enumerate(task.train):
        if pair.output is None:
            raise ValueError("blind task demonstration is missing its output")
        execution = execute_typed_sketch(sketch, pair.input)
        if not execution.ok:
            reason = execution.reason
            if reason is None:
                raise ValueError("invalid scene execution is missing its reason")
            obligation = _invalid_execution_obligation(
                sketch=sketch,
                demo_index=demo_index,
                reason=reason,
            )
            obligations.append(obligation)
            demo_executions.append(
                AbstractDemoExecution(
                    demo_index,
                    "incompatible",
                    None,
                    reason,
                    execution.node_trace,
                    None,
                    None,
                )
            )
            continue
        output = execution.output
        if output is None:
            raise ValueError("successful scene execution is missing its output")
        if output == pair.output:
            demo_executions.append(
                AbstractDemoExecution(
                    demo_index,
                    "exact",
                    output,
                    None,
                    execution.node_trace,
                    True,
                    0,
                )
            )
            continue
        obligation, shape_matches, mismatch_count, reason = _mismatch_obligation(
            sketch=sketch,
            program=program,
            demo_index=demo_index,
            predicted=output,
            expected=pair.output,
        )
        obligations.append(obligation)
        demo_executions.append(
            AbstractDemoExecution(
                demo_index,
                "incompatible",
                output,
                reason,
                execution.node_trace,
                shape_matches,
                mismatch_count,
            )
        )

    status = "exact" if not obligations else "incompatible"
    return AbstractExecutionResult(
        task.task_id,
        sketch.sketch_id,
        status,
        tuple(demo_executions),
        tuple(obligations),
    )


@dataclass(frozen=True, slots=True)
class RecolorDemoReachability:
    """One demonstration's exact abstract domain for a single recolor node."""

    demo_index: int
    base_status: str
    base_output: Grid | None
    candidates: tuple[RecolorGridNode, ...]
    reason: str | None

    def __post_init__(self) -> None:
        if type(self.demo_index) is not int or self.demo_index < 0:
            raise ValueError("recolor demo index must be non-negative")
        if self.base_status not in {"exact", "inexact", "invalid"}:
            raise ValueError("unknown recolor demo base status")
        expected_order = tuple(
            sorted(
                set(self.candidates),
                key=lambda node: (node.source_color, node.target_color),
            )
        )
        if self.candidates != expected_order:
            raise ValueError("recolor demo candidates must be unique and ordered")
        if self.base_status == "invalid":
            if self.base_output is not None or self.candidates or self.reason is None:
                raise ValueError("invalid recolor demo has inconsistent evidence")
        elif self.base_output is None:
            raise ValueError("valid recolor demo requires a base output")
        elif self.base_status == "exact" and self.reason is not None:
            raise ValueError("base-exact recolor demo must not carry a failure reason")
        elif self.base_status == "inexact" and not self.candidates and self.reason is None:
            raise ValueError("unreachable recolor demo requires a failure reason")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "demo_index": self.demo_index,
            "base_status": self.base_status,
            "base_output": (
                None if self.base_output is None else grid_to_lists(self.base_output)
            ),
            "candidates": [node.to_json_dict() for node in self.candidates],
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RecolorReachabilityResult:
    """Joint demonstration reachability for one appended recolor node."""

    blind_task_id: str
    sketch_id: str
    status: str
    demos: tuple[RecolorDemoReachability, ...]
    candidates: tuple[RecolorGridNode, ...]
    obligations: tuple[ProofObligation, ...]

    def __post_init__(self) -> None:
        _validate_identifier(self.blind_task_id, field_name="blind task ID")
        _validate_identifier(self.sketch_id, field_name="typed sketch ID")
        if self.status not in {"base_exact", "reachable", "unreachable"}:
            raise ValueError("unknown recolor reachability status")
        if not self.demos:
            raise ValueError("recolor reachability requires demonstration evidence")
        if tuple(item.demo_index for item in self.demos) != tuple(
            range(len(self.demos))
        ):
            raise ValueError("recolor demonstrations must be dense and ordered")
        expected_order = tuple(
            sorted(
                set(self.candidates),
                key=lambda node: (node.source_color, node.target_color),
            )
        )
        if self.candidates != expected_order:
            raise ValueError("joint recolor candidates must be unique and ordered")
        if self.status == "base_exact":
            if self.candidates or self.obligations or any(
                demo.base_status != "exact" for demo in self.demos
            ):
                raise ValueError("base-exact reachability has inconsistent evidence")
        elif self.status == "reachable":
            if not self.candidates or len(self.obligations) != 1:
                raise ValueError("reachable recolor result requires one obligation")
            if self.obligations[0].obligation_type != "insert_typed_node":
                raise ValueError("reachable recolor result has the wrong obligation")
            if all(demo.base_status == "exact" for demo in self.demos):
                raise ValueError("recolor frontier must change an inexact demonstration")
            joint = set(RECOLOR_NODE_DOMAIN)
            for demo in self.demos:
                joint.intersection_update(demo.candidates)
            if set(self.candidates) != joint:
                raise ValueError("joint recolor candidates do not match demo domains")
            for candidate in self.candidates:
                if not any(
                    demo.base_output is not None
                    and execute_recolor_grid(candidate, demo.base_output)
                    != demo.base_output
                    for demo in self.demos
                ):
                    raise ValueError("recolor candidate does not change the demo frontier")
        elif self.candidates or len(self.obligations) != 1:
            raise ValueError("unreachable recolor result has inconsistent evidence")
        elif self.obligations[0].obligation_type != "representation_unreachable":
            raise ValueError("unreachable recolor result has the wrong obligation")

    @property
    def novel_frontier_count(self) -> int:
        return len(self.candidates) if self.status == "reachable" else 0

    @property
    def result_id(self) -> str:
        return _content_id(self.to_json_dict())

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": RECOLOR_REACHABILITY_SCHEMA_VERSION,
            "blind_task_id": self.blind_task_id,
            "sketch_id": self.sketch_id,
            "status": self.status,
            "demos": [demo.to_json_dict() for demo in self.demos],
            "candidates": [node.to_json_dict() for node in self.candidates],
            "obligations": [item.to_json_dict() for item in self.obligations],
            "novel_frontier_count": self.novel_frontier_count,
        }


def abstract_recolor_domain(
    predicted: Grid,
    expected: Grid,
) -> tuple[RecolorGridNode, ...]:
    """Return every single recolor whose concrete output equals ``expected``."""

    normalized_predicted = as_grid(predicted)
    normalized_expected = as_grid(expected)
    if len(normalized_predicted) != len(normalized_expected) or len(
        normalized_predicted[0]
    ) != len(normalized_expected[0]):
        return ()
    return tuple(
        node
        for node in RECOLOR_NODE_DOMAIN
        if all(
            normalized_expected[row][column]
            == (
                node.target_color
                if normalized_predicted[row][column] == node.source_color
                else normalized_predicted[row][column]
            )
            for row in range(len(normalized_expected))
            for column in range(len(normalized_expected[0]))
        )
    )


def analyze_recolor_reachability(
    sketch: TypedProgramSketch,
    task: BlindTask,
) -> RecolorReachabilityResult:
    """Prove joint reachability within the one-global-recolor language."""

    if not isinstance(sketch, TypedProgramSketch):
        raise TypeError("recolor reachability requires a TypedProgramSketch")
    if not isinstance(task, BlindTask):
        raise TypeError("recolor reachability requires an oracle-free BlindTask")
    if not sketch.complete or sketch.topology != "scene_pipeline":
        raise ValueError("recolor reachability requires a complete legacy sketch")

    demos: list[RecolorDemoReachability] = []
    shared_candidates = set(RECOLOR_NODE_DOMAIN)
    for demo_index, pair in enumerate(task.train):
        if pair.output is None:
            raise ValueError("blind task demonstration is missing its output")
        execution = execute_typed_sketch(sketch, pair.input)
        if not execution.ok:
            if execution.reason is None:
                raise ValueError("invalid typed execution is missing its reason")
            domain: tuple[RecolorGridNode, ...] = ()
            demos.append(
                RecolorDemoReachability(
                    demo_index,
                    "invalid",
                    None,
                    domain,
                    execution.reason,
                )
            )
            shared_candidates.clear()
            continue
        if execution.output is None:
            raise ValueError("successful typed execution is missing its output")
        domain = abstract_recolor_domain(execution.output, pair.output)
        shared_candidates.intersection_update(domain)
        base_exact = execution.output == pair.output
        if base_exact:
            reason = None
        elif len(execution.output) != len(pair.output) or len(execution.output[0]) != len(
            pair.output[0]
        ):
            reason = "shape_unreachable"
        elif not domain:
            reason = "single_recolor_unreachable"
        else:
            reason = None
        demos.append(
            RecolorDemoReachability(
                demo_index,
                "exact" if base_exact else "inexact",
                execution.output,
                domain,
                reason,
            )
        )

    demo_tuple = tuple(demos)
    if all(demo.base_status == "exact" for demo in demo_tuple):
        return RecolorReachabilityResult(
            task.task_id,
            sketch.sketch_id,
            "base_exact",
            demo_tuple,
            (),
            (),
        )

    candidates = tuple(
        sorted(
            shared_candidates,
            key=lambda node: (node.source_color, node.target_color),
        )
    )
    node_ids = _node_ids_by_type(sketch)
    demo_indices = tuple(range(len(task.train)))
    if candidates:
        obligation = _proof_obligation(
            "insert_typed_node",
            node_ids=node_ids,
            node_types=("render",),
            demo_indices=demo_indices,
            candidate_count=len(candidates),
            domain_size=len(RECOLOR_NODE_DOMAIN),
            inexact_demo_count=sum(
                demo.base_status == "inexact" for demo in demo_tuple
            ),
            novel_frontier_count=len(candidates),
            parent_sketch_id=sketch.sketch_id,
            required_node_type=RECOLOR_NODE_TYPE,
        )
        return RecolorReachabilityResult(
            task.task_id,
            sketch.sketch_id,
            "reachable",
            demo_tuple,
            candidates,
            (obligation,),
        )

    if any(demo.base_status == "invalid" for demo in demo_tuple):
        failure_kind = "base_execution_invalid"
    elif any(demo.reason == "shape_unreachable" for demo in demo_tuple):
        failure_kind = "shape_unreachable"
    elif all(demo.candidates for demo in demo_tuple):
        failure_kind = "cross_demo_inconsistent"
    else:
        failure_kind = "single_recolor_unreachable"
    obligation = _proof_obligation(
        "representation_unreachable",
        node_ids=node_ids,
        node_types=("render",),
        demo_indices=demo_indices,
        candidate_count=0,
        domain_size=len(RECOLOR_NODE_DOMAIN),
        failure_kind=failure_kind,
        parent_sketch_id=sketch.sketch_id,
        required_node_type=RECOLOR_NODE_TYPE,
    )
    return RecolorReachabilityResult(
        task.task_id,
        sketch.sketch_id,
        "unreachable",
        demo_tuple,
        (),
        (obligation,),
    )


def insert_recolor_hole(
    sketch: TypedProgramSketch,
    obligation: ProofObligation,
    *,
    hole_id: str,
) -> TypedProgramSketch:
    """Append the only topology node licensed by a recolor reachability proof."""

    if not isinstance(sketch, TypedProgramSketch):
        raise TypeError("recolor insertion requires a TypedProgramSketch")
    if not isinstance(obligation, ProofObligation):
        raise TypeError("recolor insertion requires a ProofObligation")
    if not sketch.complete or sketch.topology != "scene_pipeline":
        raise ValueError("recolor insertion requires a complete legacy sketch")
    if obligation.obligation_type != "insert_typed_node":
        raise ValueError("recolor insertion requires an insert-node obligation")
    if obligation.legal_actions != ("insert_typed_node",):
        raise ValueError("insert-node obligation has an inconsistent action set")
    if obligation.target_node_ids != (sketch.output_node_id,) or (
        obligation.target_node_types != ("render",)
    ):
        raise ValueError("insert-node obligation targets the wrong output edge")
    evidence = dict(obligation.evidence)
    if evidence["required_node_type"] != RECOLOR_NODE_TYPE:
        raise ValueError("insert-node obligation requests another node type")
    if type(evidence["novel_frontier_count"]) is not int or (
        evidence["novel_frontier_count"] < 1
    ):
        raise ValueError("insert-node obligation has no novel frontier")
    if evidence["parent_sketch_id"] != sketch.sketch_id:
        raise ValueError("insert-node obligation belongs to another parent sketch")

    hole = TypedHole(
        hole_id,
        RECOLOR_NODE_TYPE,
        (
            ("domain_size", len(RECOLOR_NODE_DOMAIN)),
            ("edge_source", sketch.output_node_id),
            ("operator_family", RECOLOR_NODE_TYPE),
            ("parent_sketch_id", sketch.sketch_id),
        ),
    )
    recolor_node = TypedSketchNode(
        RECOLOR_NODE_ID,
        RECOLOR_NODE_TYPE,
        (sketch.output_node_id,),
        hole,
    )
    return TypedProgramSketch(sketch.nodes + (recolor_node,), RECOLOR_NODE_ID)


def fill_obligation(
    sketch: TypedProgramSketch,
    obligation: ProofObligation,
    binding: SceneNodeBinding,
) -> TypedProgramSketch:
    """Apply the only legal action supported by a fill-hole obligation."""

    if not isinstance(obligation, ProofObligation):
        raise TypeError("typed sketch repair requires a ProofObligation")
    if obligation.obligation_type != "fill_typed_hole":
        raise ValueError("only a fill-hole obligation can bind a typed sketch node")
    if obligation.legal_actions != ("fill_typed_hole",):
        raise ValueError("fill-hole obligation has an inconsistent legal action set")
    if obligation.hole_id is None:
        raise ValueError("fill-hole obligation is missing its typed hole ID")
    evidence = dict(obligation.evidence)
    if evidence["parent_sketch_id"] != sketch.sketch_id:
        raise ValueError("fill-hole obligation belongs to another parent sketch")
    return sketch.fill_hole(obligation.hole_id, binding)
