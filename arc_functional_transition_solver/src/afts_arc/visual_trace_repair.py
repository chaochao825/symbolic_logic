"""Candidate-conditioned visual certificates for bounded scene-AST repair."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from afts_arc.blind import BlindTask
from afts_arc.experiment_safety import canonical_sha256
from afts_arc.grid import Grid, as_grid
from afts_arc.hybrid.object_code import object_code_program_id
from afts_arc.hybrid.scene_graph import (
    SceneGraph,
    SceneObject,
    ScenePipelineExecution,
    ScenePipelineProgram,
    enumerate_scene_pipeline_programs,
    extract_scene_graph,
)
from afts_arc.hybrid.types import canonical_json


VISUAL_TRACE_CERTIFICATE_SCHEMA = "afts.visual-trace-certificate/v1"
RELATIONAL_GROUPS = (
    "canvas",
    "background",
    "object_identity",
    "transform",
    "spatial_relation",
    "palette",
    "mask_render",
)

GROUP_ACTION_SLOTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "canvas": (
        "canvas_reinfer",
        (
            "ast.canvas.height",
            "ast.canvas.mode",
            "ast.canvas.padding",
            "ast.canvas.width",
        ),
    ),
    "background": (
        "reparse_background",
        ("ast.canvas.background", "ast.parse.background"),
    ),
    "object_identity": (
        "object_rematch",
        (
            "ast.correspond.d4_invariant",
            "ast.correspond.features",
            "ast.correspond.policy",
            "ast.parse.connectivity",
            "ast.parse.grouping",
            "ast.select.role",
        ),
    ),
    "transform": ("object_rematch", ("ast.operate.transform",)),
    "spatial_relation": (
        "object_rematch",
        (
            "ast.correspond.d4_invariant",
            "ast.correspond.features",
            "ast.correspond.policy",
            "ast.operate.axis",
            "ast.operate.operator",
            "ast.operate.repeat_rule",
            "ast.operate.spacing",
            "ast.select.role",
        ),
    ),
    "palette": (
        "object_rematch",
        (
            "ast.canvas.background",
            "ast.operate.output_color",
            "ast.parse.background",
        ),
    ),
    "mask_render": (
        "mask_rerender",
        ("ast.render.conflict_policy", "ast.render.mode"),
    ),
}


def _modal_color(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda color: (-counts[color], color))


def _object_structure(object_: SceneObject) -> tuple[object, ...]:
    return (
        object_.area,
        object_.height,
        object_.width,
        object_.hole_count,
        object_.canonical_d4_shape,
        object_.touches_border,
    )


def _object_orientation(object_: SceneObject) -> tuple[object, ...]:
    return (*_object_structure(object_), object_.normalized_shape)


def _scene_relational_signature(scene: SceneGraph) -> dict[str, object]:
    by_index = {object_.index: object_ for object_ in scene.objects}
    structures = tuple(sorted(_object_structure(object_) for object_ in scene.objects))
    orientations = tuple(
        sorted(_object_orientation(object_) for object_ in scene.objects)
    )
    palettes = tuple(
        sorted(
            (_object_structure(object_), object_.colors)
            for object_ in scene.objects
        )
    )
    relations = tuple(
        sorted(
            (
                _object_structure(by_index[relation.source_index]),
                _object_structure(by_index[relation.target_index]),
                relation.left_of,
                relation.above,
                relation.bbox_contains,
                relation.target_bbox_contains,
                min(relation.chebyshev_distance, 3),
            )
            for relation in scene.relations
        )
    )
    return {
        "structure": structures,
        "orientation": orientations,
        "palette": palettes,
        "relations": relations,
    }


def _output_scene(program: ScenePipelineProgram, grid: Grid) -> SceneGraph:
    return extract_scene_graph(
        grid,
        background=program.canvas.background,
        connectivity=program.parse.connectivity,
        grouping=program.parse.grouping,
    )


def relational_difference_groups(
    program: ScenePipelineProgram,
    parent_output: Grid,
    alternative_output: Grid,
) -> tuple[str, ...]:
    """Compare two complete grids through one candidate's relational ontology."""

    parent = as_grid(parent_output)
    alternative = as_grid(alternative_output)
    if parent == alternative:
        return ()
    groups: set[str] = set()
    if (len(parent), len(parent[0])) != (len(alternative), len(alternative[0])):
        groups.add("canvas")
    if _modal_color(parent) != _modal_color(alternative):
        groups.add("background")
    parent_signature = _scene_relational_signature(_output_scene(program, parent))
    alternative_signature = _scene_relational_signature(
        _output_scene(program, alternative)
    )
    if parent_signature["structure"] != alternative_signature["structure"]:
        groups.add("object_identity")
    if parent_signature["orientation"] != alternative_signature["orientation"]:
        groups.add("transform")
    if parent_signature["relations"] != alternative_signature["relations"]:
        groups.add("spatial_relation")
    if parent_signature["palette"] != alternative_signature["palette"]:
        groups.add("palette")
    if not groups:
        groups.add("mask_render")
    return tuple(group for group in RELATIONAL_GROUPS if group in groups)


def posterior_group_counts(
    program: ScenePipelineProgram,
    parent_output: Grid,
    samples: Sequence[Grid],
) -> dict[str, int]:
    """Count relation-group disagreements over complete posterior samples."""

    if not samples:
        raise ValueError("posterior evidence requires at least one complete sample")
    counts = {group: 0 for group in RELATIONAL_GROUPS}
    for sample in samples:
        for group in relational_difference_groups(program, parent_output, sample):
            counts[group] += 1
    return counts


def _trace_payload(
    executions: Sequence[ScenePipelineExecution],
) -> tuple[dict[str, object], ...]:
    expected_nodes = ("parse", "correspond", "select", "operate", "canvas", "render")
    payload = []
    for execution in executions:
        if not execution.ok or execution.output is None:
            raise ValueError("visual trace repair requires valid parent executions")
        by_id = {node.node_id: node for node in execution.node_trace}
        if tuple(node for node in expected_nodes if node not in by_id):
            raise ValueError("parent execution trace omits a required scene-AST node")
        payload.append(
            {
                "nodes": [
                    {
                        "node_id": node_id,
                        "operator": by_id[node_id].operator,
                        "status": by_id[node_id].status,
                        "details": {
                            key: value for key, value in by_id[node_id].details
                        },
                    }
                    for node_id in expected_nodes
                ],
                "object_count": execution.object_count,
                "correspondence_count": execution.correspondence_count,
            }
        )
    return tuple(payload)


@dataclass(frozen=True, slots=True)
class VisualTraceCertificate:
    certificate_id: str
    task_id: str
    parent_program_id: str
    mode: str
    diagnosis_group: str
    recommended_action: str
    affected_slots: tuple[str, ...]
    evidence: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.mode not in {"visual_lodo", "bridge_lesion"}:
            raise ValueError("unknown visual trace certificate mode")
        if self.diagnosis_group not in RELATIONAL_GROUPS:
            raise ValueError("unknown relational diagnosis group")
        expected_action, expected_slots = GROUP_ACTION_SLOTS[self.diagnosis_group]
        if self.recommended_action != expected_action:
            raise ValueError("certificate action does not match its diagnosis group")
        if self.affected_slots != expected_slots:
            raise ValueError("certificate slots do not match its diagnosis group")
        if self.certificate_id != canonical_sha256(self._content()):
            raise ValueError("visual trace certificate ID does not match content")

    def _content(self) -> dict[str, object]:
        return {
            "schema": VISUAL_TRACE_CERTIFICATE_SCHEMA,
            "task_id": self.task_id,
            "parent_program_id": self.parent_program_id,
            "mode": self.mode,
            "diagnosis_group": self.diagnosis_group,
            "recommended_action": self.recommended_action,
            "affected_slots": list(self.affected_slots),
            "evidence": dict(self.evidence),
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"certificate_id": self.certificate_id, **self._content()}


def compile_visual_trace_certificate(
    *,
    task_id: str,
    parent_program: ScenePipelineProgram,
    demo_executions: Sequence[ScenePipelineExecution],
    demo_gold_outputs: Sequence[Grid],
    lodo_samples: Sequence[Sequence[Grid]],
    query_execution: ScenePipelineExecution,
    query_samples: Sequence[Grid],
    mode: str,
) -> VisualTraceCertificate:
    """Compile LODO-supported demo residuals into one legal AST action."""

    if mode not in {"visual_lodo", "bridge_lesion"}:
        raise ValueError("unknown visual trace certificate mode")
    if not (
        len(demo_executions) == len(demo_gold_outputs) == len(lodo_samples)
    ):
        raise ValueError("demo executions, gold outputs, and LODO views must align")
    if not demo_executions:
        raise ValueError("visual trace certificate requires demonstrations")
    if not query_execution.ok or query_execution.output is None:
        raise ValueError("visual trace certificate requires a valid query execution")
    demo_outputs = tuple(execution.output for execution in demo_executions)
    if any(output is None for output in demo_outputs):
        raise ValueError("visual trace certificate parent output is missing")
    normalized_demo_outputs = tuple(
        as_grid(output) for output in demo_outputs if output is not None
    )
    normalized_gold = tuple(as_grid(output) for output in demo_gold_outputs)
    normalized_lodo = tuple(
        tuple(as_grid(sample) for sample in view) for view in lodo_samples
    )
    normalized_query_samples = tuple(as_grid(sample) for sample in query_samples)
    truth_groups = tuple(
        relational_difference_groups(parent_program, parent, gold)
        for parent, gold in zip(
            normalized_demo_outputs, normalized_gold, strict=True
        )
    )
    eligible = tuple(
        group for group in RELATIONAL_GROUPS if any(group in item for item in truth_groups)
    )
    if not eligible:
        raise ValueError("an exact parent has no demonstration residual to compile")
    lodo_counts = tuple(
        posterior_group_counts(parent_program, parent, samples)
        for parent, samples in zip(
            normalized_demo_outputs, normalized_lodo, strict=True
        )
    )
    query_counts = posterior_group_counts(
        parent_program,
        as_grid(query_execution.output),
        normalized_query_samples,
    )
    rows = []
    for group in eligible:
        affected = tuple(
            index for index, groups in enumerate(truth_groups) if group in groups
        )
        numerator = sum(lodo_counts[index][group] for index in affected)
        denominator = sum(len(normalized_lodo[index]) for index in affected)
        rows.append(
            {
                "group": group,
                "affected_demo_indices": list(affected),
                "affected_demo_count": len(affected),
                "lodo_support_count": numerator,
                "lodo_support_total": denominator,
                "query_support_count": query_counts[group],
                "query_support_total": len(normalized_query_samples),
            }
        )

    group_index = {group: index for index, group in enumerate(RELATIONAL_GROUPS)}

    def visual_key(row: Mapping[str, object]) -> tuple[object, ...]:
        return (
            Fraction(row["lodo_support_count"], row["lodo_support_total"]),
            row["affected_demo_count"],
            Fraction(row["query_support_count"], row["query_support_total"]),
            -group_index[row["group"]],
        )

    def lesion_key(row: Mapping[str, object]) -> tuple[object, ...]:
        return (
            row["affected_demo_count"],
            -group_index[row["group"]],
        )

    selected = max(rows, key=visual_key if mode == "visual_lodo" else lesion_key)
    selected_group = selected["group"]
    action, slots = GROUP_ACTION_SLOTS[selected_group]
    evidence: dict[str, object] = {
        "truth_groups_by_demo": [list(groups) for groups in truth_groups],
        "group_evidence": rows,
        "demo_trace": list(_trace_payload(demo_executions)),
        "query_trace": list(_trace_payload((query_execution,))),
        "complete_grid_posterior_only": True,
        "grid_synthesis": False,
    }
    content = {
        "schema": VISUAL_TRACE_CERTIFICATE_SCHEMA,
        "task_id": task_id,
        "parent_program_id": object_code_program_id(parent_program),
        "mode": mode,
        "diagnosis_group": selected_group,
        "recommended_action": action,
        "affected_slots": list(slots),
        "evidence": evidence,
    }
    return VisualTraceCertificate(
        canonical_sha256(content),
        task_id,
        content["parent_program_id"],
        mode,
        selected_group,
        action,
        slots,
        evidence,
    )


def _flatten_program_fields(
    value: object,
    *,
    prefix: str = "",
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


def scene_program_fields(program: ScenePipelineProgram) -> dict[str, object]:
    return _flatten_program_fields(program.to_json_dict())


@dataclass(frozen=True, slots=True)
class SceneSlotEdit:
    slot: str
    program: ScenePipelineProgram


def single_slot_scene_variants(
    task: BlindTask,
    parent: ScenePipelineProgram,
    *,
    allowed_slots: Sequence[str],
    existing_program_ids: Sequence[str] = (),
    candidate_programs: Sequence[ScenePipelineProgram] | None = None,
) -> tuple[SceneSlotEdit, ...]:
    """Enumerate content-novel programs changing one existing parent leaf slot."""

    allowed = frozenset(allowed_slots)
    parent_fields = scene_program_fields(parent)
    if not allowed or any(slot not in parent_fields for slot in allowed):
        raise ValueError("allowed edit slots must name existing parent fields")
    existing = frozenset(existing_program_ids)
    edits = []
    programs = (
        enumerate_scene_pipeline_programs(task)
        if candidate_programs is None
        else tuple(candidate_programs)
    )
    for program in programs:
        program_id = object_code_program_id(program)
        if program_id in existing:
            continue
        fields = scene_program_fields(program)
        if set(fields) != set(parent_fields):
            raise ValueError("scene program variants have inconsistent fields")
        differences = tuple(
            sorted(
                field
                for field in parent_fields
                if parent_fields[field] != fields[field]
            )
        )
        if len(differences) == 1 and differences[0] in allowed:
            edits.append(SceneSlotEdit(differences[0], program))
    return tuple(
        sorted(
            edits,
            key=lambda edit: (
                edit.slot,
                canonical_json(edit.program.to_json_dict()),
            ),
        )
    )


def query_posterior_rank_key(
    program: ScenePipelineProgram,
    output: Grid,
    samples: Sequence[Grid],
) -> tuple[object, ...]:
    """Rank executable outputs against complete query-posterior hypotheses."""

    normalized_output = as_grid(output)
    normalized_samples = tuple(as_grid(sample) for sample in samples)
    if not normalized_samples:
        raise ValueError("query posterior ranking requires samples")
    exact_count = sum(sample == normalized_output for sample in normalized_samples)
    relation_distance = sum(
        len(relational_difference_groups(program, normalized_output, sample))
        for sample in normalized_samples
    )
    return (
        -exact_count,
        Fraction(relation_distance, len(normalized_samples)),
        program.description_bits,
        object_code_program_id(program),
    )
