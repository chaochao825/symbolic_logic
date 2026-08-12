"""Exact version-space reasoning for one anchor-rasterized topology node."""

from __future__ import annotations

from dataclasses import dataclass

from .anchor_mask import (
    ANCHOR_MASK_NODE_ID,
    ANCHOR_MASK_NODE_TYPE,
    ANCHOR_MASK_PROGRAMS_PER_DELTA,
    AnchorRasterizedDeltaNode,
    anchor_mask_domain,
    execute_anchor_rasterized_delta,
)
from .blind import BlindTask
from .executable_workspace import (
    INPUT_GRID_NODE_ID,
    ProofObligation,
    TypedHole,
    TypedProgramSketch,
    TypedSketchNode,
    execute_recolor_grid,
    execute_typed_sketch,
    RecolorGridNode,
)
from .experiment_safety import canonical_sha256
from .grid import Grid, as_grid, grid_to_lists
from .hybrid.scene_graph import SceneGraph, extract_scene_graph


ANCHOR_MASK_REACHABILITY_SCHEMA = "afts.anchor-mask-reachability/v0.1"
ANCHOR_MASK_LODO_SCHEMA = "afts.anchor-mask-lodo/v0.1"


def _scene_for_sketch(sketch: TypedProgramSketch, grid: Grid) -> SceneGraph:
    program = sketch.materialize_scene_pipeline_prefix()
    return extract_scene_graph(
        grid,
        background=program.parse.background,
        connectivity=program.parse.connectivity,
        grouping=program.parse.grouping,
    )


def _delta_pair(parent: Grid, expected: Grid) -> tuple[int, int] | None:
    mappings = {
        (parent[row][column], expected[row][column])
        for row in range(len(expected))
        for column in range(len(expected[0]))
        if parent[row][column] != expected[row][column]
    }
    if len(mappings) != 1:
        return None
    return next(iter(mappings))


def abstract_anchor_mask_domain(
    *,
    input_grid: Grid,
    parent_grid: Grid,
    expected_grid: Grid,
    scene: SceneGraph,
    source_color: int,
    target_color: int,
) -> tuple[AnchorRasterizedDeltaNode, ...]:
    """Return every program in the finite language matching one example."""

    parent = as_grid(parent_grid)
    expected = as_grid(expected_grid)
    if (len(parent), len(parent[0])) != (len(expected), len(expected[0])):
        return ()
    candidates = tuple(
        node
        for node in anchor_mask_domain(source_color, target_color)
        if execute_anchor_rasterized_delta(node, input_grid, parent, scene) == expected
    )
    return tuple(sorted(candidates, key=lambda node: node.sort_key))


@dataclass(frozen=True, slots=True)
class AnchorMaskDemoReachability:
    demo_index: int
    base_status: str
    base_output: Grid | None
    delta_pair: tuple[int, int] | None
    candidates: tuple[AnchorRasterizedDeltaNode, ...]
    reason: str | None

    def __post_init__(self) -> None:
        if type(self.demo_index) is not int or self.demo_index < 0:
            raise ValueError("anchor-mask demo index must be non-negative")
        if self.base_status not in {"exact", "inexact", "invalid"}:
            raise ValueError("unknown anchor-mask base status")
        expected = tuple(sorted(set(self.candidates), key=lambda node: node.sort_key))
        if self.candidates != expected:
            raise ValueError("anchor-mask candidates must be unique and ordered")
        if self.base_status == "invalid":
            if self.base_output is not None or self.candidates or self.reason is None:
                raise ValueError("invalid anchor-mask demo has inconsistent evidence")
        elif self.base_output is None:
            raise ValueError("valid anchor-mask demo requires a base output")
        if self.delta_pair is not None:
            source_color, target_color = self.delta_pair
            if source_color == target_color:
                raise ValueError("anchor-mask delta pair must change color")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "demo_index": self.demo_index,
            "base_status": self.base_status,
            "base_output": (
                None if self.base_output is None else grid_to_lists(self.base_output)
            ),
            "delta_pair": (
                None if self.delta_pair is None else list(self.delta_pair)
            ),
            "candidates": [node.to_json_dict() for node in self.candidates],
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class AnchorMaskReachabilityResult:
    blind_task_id: str
    sketch_id: str
    status: str
    demos: tuple[AnchorMaskDemoReachability, ...]
    candidates: tuple[AnchorRasterizedDeltaNode, ...]
    obligations: tuple[ProofObligation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.blind_task_id, str) or not self.blind_task_id:
            raise ValueError("anchor-mask reachability requires a blind task ID")
        if not isinstance(self.sketch_id, str) or not self.sketch_id:
            raise ValueError("anchor-mask reachability requires a sketch ID")
        if self.status not in {"base_exact", "reachable", "unreachable"}:
            raise ValueError("unknown anchor-mask reachability status")
        if not self.demos or tuple(item.demo_index for item in self.demos) != tuple(
            range(len(self.demos))
        ):
            raise ValueError("anchor-mask demonstrations must be dense and ordered")
        expected = tuple(sorted(set(self.candidates), key=lambda node: node.sort_key))
        if self.candidates != expected:
            raise ValueError("joint anchor-mask candidates must be unique and ordered")
        if self.status == "base_exact":
            if (
                self.candidates
                or self.obligations
                or any(demo.base_status != "exact" for demo in self.demos)
            ):
                raise ValueError("base-exact anchor-mask result is inconsistent")
        elif self.status == "reachable":
            if not self.candidates or len(self.obligations) != 1:
                raise ValueError("reachable anchor-mask result requires one obligation")
            if self.obligations[0].obligation_type != "insert_typed_node":
                raise ValueError("reachable anchor-mask result has the wrong obligation")
            joint = set(anchor_mask_domain(
                self.candidates[0].source_color,
                self.candidates[0].target_color,
            ))
            for demo in self.demos:
                joint.intersection_update(demo.candidates)
            if set(self.candidates) != joint:
                raise ValueError("joint anchor-mask candidates do not match demo domains")
        elif self.candidates or len(self.obligations) != 1:
            raise ValueError("unreachable anchor-mask result is inconsistent")
        elif self.obligations[0].obligation_type != "representation_unreachable":
            raise ValueError("unreachable anchor-mask result has the wrong obligation")

    @property
    def novel_frontier_count(self) -> int:
        return len(self.candidates) if self.status == "reachable" else 0

    @property
    def result_id(self) -> str:
        return canonical_sha256(self.to_json_dict())

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": ANCHOR_MASK_REACHABILITY_SCHEMA,
            "blind_task_id": self.blind_task_id,
            "sketch_id": self.sketch_id,
            "status": self.status,
            "demos": [demo.to_json_dict() for demo in self.demos],
            "candidates": [node.to_json_dict() for node in self.candidates],
            "obligations": [item.to_json_dict() for item in self.obligations],
            "novel_frontier_count": self.novel_frontier_count,
        }


def _obligation(
    *,
    sketch: TypedProgramSketch,
    status: str,
    failure_kind: str | None,
    candidate_count: int,
    demo_count: int,
) -> ProofObligation:
    evidence: list[tuple[str, object]] = [
        ("candidate_count", candidate_count),
        ("domain_size", ANCHOR_MASK_PROGRAMS_PER_DELTA),
        ("parent_sketch_id", sketch.sketch_id),
        ("required_node_type", ANCHOR_MASK_NODE_TYPE),
    ]
    if failure_kind is not None:
        evidence.append(("failure_kind", failure_kind))
    if status == "reachable":
        evidence.append(("novel_frontier_count", candidate_count))
        obligation_type = "insert_typed_node"
    else:
        obligation_type = "representation_unreachable"
    return ProofObligation(
        obligation_type,
        (sketch.output_node_id,),
        ("render",),
        tuple(range(demo_count)),
        tuple(sorted(evidence)),
    )


def analyze_anchor_mask_reachability(
    sketch: TypedProgramSketch,
    task: BlindTask,
) -> AnchorMaskReachabilityResult:
    """Compute the exact cross-demo version space for the frozen mask language."""

    if not isinstance(sketch, TypedProgramSketch):
        raise TypeError("anchor-mask reachability requires a TypedProgramSketch")
    if not isinstance(task, BlindTask):
        raise TypeError("anchor-mask reachability requires an oracle-free BlindTask")
    if not sketch.complete or sketch.topology != "scene_pipeline":
        raise ValueError("anchor-mask reachability requires a complete legacy sketch")

    observations: list[tuple[int, Grid, Grid, SceneGraph, str, tuple[int, int] | None]] = []
    invalid_rows: list[AnchorMaskDemoReachability] = []
    inexact_deltas: list[tuple[int, int]] = []
    for demo_index, pair in enumerate(task.train):
        if pair.output is None:
            raise ValueError("blind demonstration is missing its output")
        execution = execute_typed_sketch(sketch, pair.input)
        if not execution.ok or execution.output is None:
            reason = execution.reason or "base_execution_invalid"
            invalid_rows.append(
                AnchorMaskDemoReachability(
                    demo_index,
                    "invalid",
                    None,
                    None,
                    (),
                    reason,
                )
            )
            continue
        parent = execution.output
        expected = pair.output
        if (len(parent), len(parent[0])) != (len(expected), len(expected[0])):
            invalid_rows.append(
                AnchorMaskDemoReachability(
                    demo_index,
                    "invalid",
                    None,
                    None,
                    (),
                    "shape_unreachable",
                )
            )
            continue
        if (len(pair.input), len(pair.input[0])) != (len(parent), len(parent[0])):
            invalid_rows.append(
                AnchorMaskDemoReachability(
                    demo_index,
                    "invalid",
                    None,
                    None,
                    (),
                    "canvas_incompatible",
                )
            )
            continue
        base_status = "exact" if parent == expected else "inexact"
        delta = None if base_status == "exact" else _delta_pair(parent, expected)
        if base_status == "inexact" and delta is None:
            invalid_rows.append(
                AnchorMaskDemoReachability(
                    demo_index,
                    "invalid",
                    None,
                    None,
                    (),
                    "multi_delta_unreachable",
                )
            )
            continue
        if delta is not None:
            inexact_deltas.append(delta)
        observations.append(
            (
                demo_index,
                parent,
                expected,
                _scene_for_sketch(sketch, pair.input),
                base_status,
                delta,
            )
        )

    if invalid_rows:
        invalid_by_index = {row.demo_index: row for row in invalid_rows}
        observed_by_index = {row[0]: row for row in observations}
        demos = tuple(
            invalid_by_index[index]
            if index in invalid_by_index
            else AnchorMaskDemoReachability(
                index,
                observed_by_index[index][4],
                observed_by_index[index][1],
                observed_by_index[index][5],
                (),
                None,
            )
            for index in range(len(task.train))
        )
        failure_kind = invalid_rows[0].reason or "base_execution_invalid"
        return AnchorMaskReachabilityResult(
            task.task_id,
            sketch.sketch_id,
            "unreachable",
            demos,
            (),
            (_obligation(
                sketch=sketch,
                status="unreachable",
                failure_kind=failure_kind,
                candidate_count=0,
                demo_count=len(task.train),
            ),),
        )

    if not inexact_deltas:
        demos = tuple(
            AnchorMaskDemoReachability(
                index,
                "exact",
                parent,
                None,
                (),
                None,
            )
            for index, parent, _, _, _, _ in observations
        )
        return AnchorMaskReachabilityResult(
            task.task_id,
            sketch.sketch_id,
            "base_exact",
            demos,
            (),
            (),
        )

    shared_deltas = set(inexact_deltas)
    if len(shared_deltas) != 1:
        demos = tuple(
            AnchorMaskDemoReachability(
                index,
                base_status,
                parent,
                delta,
                (),
                "cross_demo_delta_inconsistent" if delta is not None else None,
            )
            for index, parent, _, _, base_status, delta in observations
        )
        return AnchorMaskReachabilityResult(
            task.task_id,
            sketch.sketch_id,
            "unreachable",
            demos,
            (),
            (_obligation(
                sketch=sketch,
                status="unreachable",
                failure_kind="cross_demo_delta_inconsistent",
                candidate_count=0,
                demo_count=len(task.train),
            ),),
        )

    source_color, target_color = next(iter(shared_deltas))
    shared_candidates = set(anchor_mask_domain(source_color, target_color))
    demos_list: list[AnchorMaskDemoReachability] = []
    for index, parent, expected, scene, base_status, delta in observations:
        domain = abstract_anchor_mask_domain(
            input_grid=task.train[index].input,
            parent_grid=parent,
            expected_grid=expected,
            scene=scene,
            source_color=source_color,
            target_color=target_color,
        )
        shared_candidates.intersection_update(domain)
        demos_list.append(
            AnchorMaskDemoReachability(
                index,
                base_status,
                parent,
                delta,
                domain,
                None if domain else "anchor_mask_unreachable",
            )
        )
    candidates = tuple(sorted(shared_candidates, key=lambda node: node.sort_key))
    demos = tuple(demos_list)
    if candidates:
        return AnchorMaskReachabilityResult(
            task.task_id,
            sketch.sketch_id,
            "reachable",
            demos,
            candidates,
            (_obligation(
                sketch=sketch,
                status="reachable",
                failure_kind=None,
                candidate_count=len(candidates),
                demo_count=len(task.train),
            ),),
        )
    failure_kind = (
        "cross_demo_program_inconsistent"
        if all(demo.candidates for demo in demos)
        else "anchor_mask_unreachable"
    )
    return AnchorMaskReachabilityResult(
        task.task_id,
        sketch.sketch_id,
        "unreachable",
        demos,
        (),
        (_obligation(
            sketch=sketch,
            status="unreachable",
            failure_kind=failure_kind,
            candidate_count=0,
            demo_count=len(task.train),
        ),),
    )


def insert_anchor_mask_hole(
    sketch: TypedProgramSketch,
    obligation: ProofObligation,
    *,
    hole_id: str,
) -> TypedProgramSketch:
    """Insert the single typed node licensed by a non-empty version space."""

    if not isinstance(sketch, TypedProgramSketch):
        raise TypeError("anchor-mask insertion requires a TypedProgramSketch")
    if not isinstance(obligation, ProofObligation):
        raise TypeError("anchor-mask insertion requires a ProofObligation")
    if not sketch.complete or sketch.topology != "scene_pipeline":
        raise ValueError("anchor-mask insertion requires a complete legacy sketch")
    if obligation.obligation_type != "insert_typed_node":
        raise ValueError("anchor-mask insertion requires an insert-node obligation")
    evidence = dict(obligation.evidence)
    if evidence["required_node_type"] != ANCHOR_MASK_NODE_TYPE:
        raise ValueError("insert-node obligation requests another node type")
    if evidence["parent_sketch_id"] != sketch.sketch_id:
        raise ValueError("insert-node obligation belongs to another parent sketch")
    if evidence["novel_frontier_count"] < 1:
        raise ValueError("insert-node obligation has no novel frontier")
    hole = TypedHole(
        hole_id,
        ANCHOR_MASK_NODE_TYPE,
        (
            ("domain_size", ANCHOR_MASK_PROGRAMS_PER_DELTA),
            ("edge_parent_grid", sketch.output_node_id),
            ("edge_parse_scene", "parse"),
            ("edge_source_grid", INPUT_GRID_NODE_ID),
            ("parent_sketch_id", sketch.sketch_id),
        ),
    )
    node = TypedSketchNode(
        ANCHOR_MASK_NODE_ID,
        ANCHOR_MASK_NODE_TYPE,
        (sketch.output_node_id, INPUT_GRID_NODE_ID, "parse"),
        hole,
    )
    return TypedProgramSketch(sketch.nodes + (node,), ANCHOR_MASK_NODE_ID)


def _component_baseline_exact(
    *,
    parent_grid: Grid,
    expected_grid: Grid,
    scene: SceneGraph,
    source_color: int,
    target_color: int,
) -> bool:
    parent = as_grid(parent_grid)
    expected = as_grid(expected_grid)
    for anchor_color in range(10):
        selected = {
            cell
            for scene_object in scene.objects
            if anchor_color in scene_object.colors
            for cell in scene_object.cells
        }
        output = as_grid(
            [
                [
                    target_color
                    if (row, column) in selected and color == source_color
                    else color
                    for column, color in enumerate(values)
                ]
                for row, values in enumerate(parent)
            ]
        )
        if output == expected:
            return True
    return False


def leave_one_demo_out_anchor_masks(
    sketch: TypedProgramSketch,
    task: BlindTask,
) -> dict[str, object]:
    """Audit held-demo support prediction without changing query candidates."""

    if len(task.train) < 2:
        return {
            "schema": ANCHOR_MASK_LODO_SCHEMA,
            "fold_count": len(task.train),
            "predictable_fold_count": 0,
            "exact_fold_count": 0,
            "identity_exact_fold_count": 0,
            "global_recolor_exact_fold_count": 0,
            "component_recolor_exact_fold_count": 0,
            "strict_all_folds_exact": False,
            "folds": [],
        }

    folds = []
    for holdout_index, holdout in enumerate(task.train):
        subset = BlindTask.from_observations(
            train=tuple(
                pair for index, pair in enumerate(task.train) if index != holdout_index
            ),
            test_inputs=task.test_inputs,
        )
        inferred = analyze_anchor_mask_reachability(sketch, subset)
        parent_execution = execute_typed_sketch(sketch, holdout.input)
        if not parent_execution.ok or parent_execution.output is None:
            folds.append(
                {
                    "holdout_index": holdout_index,
                    "predictable": False,
                    "exact": False,
                    "identity_exact": False,
                    "global_recolor_exact": False,
                    "component_recolor_exact": False,
                    "version_space_size": 0,
                    "reason": "holdout_parent_invalid",
                }
            )
            continue
        if holdout.output is None:
            raise ValueError("blind holdout demonstration is missing its output")
        parent = parent_execution.output
        scene = _scene_for_sketch(sketch, holdout.input)
        predictable = inferred.status == "reachable"
        exact = predictable and any(
            execute_anchor_rasterized_delta(
                node,
                holdout.input,
                parent,
                scene,
            )
            == holdout.output
            for node in inferred.candidates
        )
        identity_exact = parent == holdout.output
        global_exact = False
        component_exact = False
        if predictable:
            deltas = {
                (node.source_color, node.target_color) for node in inferred.candidates
            }
            if len(deltas) != 1:
                raise ValueError("reachable mask version space changed delta semantics")
            source_color, target_color = next(iter(deltas))
            global_exact = (
                execute_recolor_grid(
                    RecolorGridNode(source_color, target_color),
                    parent,
                )
                == holdout.output
            )
            component_exact = _component_baseline_exact(
                parent_grid=parent,
                expected_grid=holdout.output,
                scene=scene,
                source_color=source_color,
                target_color=target_color,
            )
        folds.append(
            {
                "holdout_index": holdout_index,
                "predictable": predictable,
                "exact": exact,
                "identity_exact": identity_exact,
                "global_recolor_exact": global_exact,
                "component_recolor_exact": component_exact,
                "version_space_size": len(inferred.candidates),
                "reason": None if predictable else inferred.status,
            }
        )

    fold_count = len(folds)
    predictable_count = sum(int(fold["predictable"]) for fold in folds)
    exact_count = sum(int(fold["exact"]) for fold in folds)
    return {
        "schema": ANCHOR_MASK_LODO_SCHEMA,
        "fold_count": fold_count,
        "predictable_fold_count": predictable_count,
        "exact_fold_count": exact_count,
        "identity_exact_fold_count": sum(
            int(fold["identity_exact"]) for fold in folds
        ),
        "global_recolor_exact_fold_count": sum(
            int(fold["global_recolor_exact"]) for fold in folds
        ),
        "component_recolor_exact_fold_count": sum(
            int(fold["component_recolor_exact"]) for fold in folds
        ),
        "strict_all_folds_exact": (
            predictable_count == fold_count and exact_count == fold_count
        ),
        "folds": folds,
    }
