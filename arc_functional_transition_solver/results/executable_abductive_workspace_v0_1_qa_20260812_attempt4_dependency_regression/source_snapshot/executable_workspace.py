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
from .grid import Grid, grid_to_lists
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
ABSTRACT_EXECUTION_SCHEMA_VERSION = "afts.abstract-execution/v0.1"
INPUT_GRID_NODE_ID = "$input"

SKETCH_NODE_TYPES = (
    "parse",
    "correspond",
    "select",
    "operate",
    "canvas",
    "render",
)

_NODE_INPUT_TYPES = {
    "parse": ("grid",),
    "correspond": ("scene",),
    "select": ("scene", "correspondence_set"),
    "operate": ("grid", "scene", "object_set"),
    "canvas": ("operation_plan",),
    "render": ("grid", "scene", "object_set", "operation_plan", "canvas"),
}
_NODE_OUTPUT_TYPES = {
    "parse": "scene",
    "correspond": "correspondence_set",
    "select": "object_set",
    "operate": "operation_plan",
    "canvas": "canvas",
    "render": "grid",
}
_NODE_BINDING_CLASSES = {
    "parse": ParseObjectsNode,
    "correspond": CorrespondObjectsNode,
    "select": SelectObjectsNode,
    "operate": ObjectOperationNode,
    "canvas": CanvasNode,
    "render": RenderObjectsNode,
}

PROOF_OBLIGATION_TYPES = (
    "fill_typed_hole",
    "reparse",
    "selection_conflict",
    "production_mismatch",
    "canvas_mismatch",
    "render_mismatch",
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
        if not hole_ids and tuple(node.node_type for node in self.nodes) == SKETCH_NODE_TYPES:
            # The legacy program constructor is also the refinement-type checker:
            # it rejects individually typed nodes whose cross-node contracts are
            # incompatible (for example, a signature selector without matching
            # correspondence features).
            self.materialize_scene_pipeline()

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
        return {
            "schema": TYPED_SKETCH_SCHEMA_VERSION,
            "nodes": [node.to_json_dict() for node in self.nodes],
            "output_node_id": self.output_node_id,
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "TypedProgramSketch":
        expected = {"schema", "nodes", "output_node_id"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("typed program sketch has missing or unknown fields")
        if payload["schema"] != TYPED_SKETCH_SCHEMA_VERSION:
            raise ValueError("unsupported typed program sketch schema")
        nodes = payload["nodes"]
        if not isinstance(nodes, list):
            raise ValueError("typed program sketch nodes must be a list")
        return cls(
            tuple(TypedSketchNode.from_json_dict(node) for node in nodes),
            payload["output_node_id"],
        )

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
        if tuple(node.node_type for node in self.nodes) != SKETCH_NODE_TYPES:
            raise ValueError("legacy materialization requires the canonical scene pipeline")
        bindings = tuple(node.binding for node in self.nodes)
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


def _node_ids_by_type(sketch: TypedProgramSketch) -> dict[str, str]:
    node_ids = {node.node_type: node.node_id for node in sketch.nodes}
    if set(node_ids) != set(SKETCH_NODE_TYPES):
        raise ValueError("abstract execution requires one canonical node of each type")
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
    if _support_mask(predicted, background=program.canvas.background) == _support_mask(
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

    program = sketch.materialize_scene_pipeline()
    demo_executions: list[AbstractDemoExecution] = []
    obligations: list[ProofObligation] = []
    for demo_index, pair in enumerate(task.train):
        if pair.output is None:
            raise ValueError("blind task demonstration is missing its output")
        execution = execute_scene_pipeline(program, pair.input)
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
    return sketch.fill_hole(obligation.hole_id, binding)
