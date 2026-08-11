"""Node-local counterfactual repair over persistent scene execution state.

This v3 lane replaces raster-to-raster stage composition with one typed AST
rewrite.  The parent execution retains object and relation identities, the
demonstration residual names one failed node, and replay reuses every upstream
state outside the affected dependency subtree.  Query labels are never inputs
to diagnosis, enumeration, ranking, or candidate construction.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .blind import BlindTask
from .experiment_safety import canonical_sha256
from .grid import Grid, grid_key
from .hybrid.object_code import (
    ObjectCodeProgram,
    ObjectCodeProgramScore,
    execute_object_code_program,
    object_code_program_id,
    synthesize_object_code_programs,
)
from .hybrid.scene_graph import (
    ScenePipelineProgram,
    enumerate_scene_pipeline_programs,
    execute_scene_pipeline,
)
from .residual import compare_grids
from .stateful_scene import (
    STATEFUL_NODE_ORDER,
    StatefulSceneExecution,
    affected_scene_subtree,
    execute_stateful_scene_pipeline,
    scene_program_node_differences,
    scene_program_node_payloads,
)


STATEFUL_REWRITE_DSL_VERSION = "afts-stateful-object-graph-rewrite-dsl/v3"
STATEFUL_REWRITE_PROVIDER_VERSION = "afts-stateful-object-graph-rewrite/v3"
STATEFUL_REWRITE_CERTIFICATE_SCHEMA = "afts.stateful-node-failure/v1"
STATEFUL_REWRITE_SCHEMA = "afts.stateful-scene-rewrite/v1"

Coordinate = tuple[int, int]


def _bundle_id(outputs: Sequence[Grid]) -> str:
    return canonical_sha256([grid_key(output) for output in outputs])


def _grid_shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def _delta_coordinates(first: Grid, second: Grid) -> frozenset[Coordinate]:
    if _grid_shape(first) != _grid_shape(second):
        raise ValueError("delta coordinates require equal canvas shapes")
    return frozenset(
        (row, column)
        for row in range(len(first))
        for column in range(len(first[0]))
        if first[row][column] != second[row][column]
    )


def _delta_is_complete_object_union(
    execution: StatefulSceneExecution,
    delta: frozenset[Coordinate],
) -> bool:
    if not delta:
        return False
    covered: set[Coordinate] = set()
    for record in execution.parse_state.objects:
        cells = frozenset(record.cells)
        if cells and cells <= delta:
            covered.update(cells)
    return covered == set(delta)


@dataclass(frozen=True, slots=True)
class StatefulNodeFailureCertificate:
    certificate_id: str
    blind_task_id: str
    parent_program_id: str
    diagnosis: str
    node_id: str
    affected_subtree: tuple[str, ...]
    mismatch_counts: tuple[int, ...]
    shape_match: tuple[bool, ...]
    parent_execution_ids: tuple[str, ...]
    evidence_json: str

    @classmethod
    def create(
        cls,
        *,
        task: BlindTask,
        parent: ScenePipelineProgram,
        diagnosis: str,
        node_id: str,
        mismatch_counts: Sequence[int],
        shape_match: Sequence[bool],
        parent_execution_ids: Sequence[str],
        evidence: Mapping[str, object],
    ) -> "StatefulNodeFailureCertificate":
        if node_id not in STATEFUL_NODE_ORDER:
            raise ValueError("failure certificate names an unknown AST node")
        mismatch = tuple(mismatch_counts)
        shapes = tuple(shape_match)
        executions = tuple(parent_execution_ids)
        if (
            not mismatch
            or len(mismatch) != len(shapes)
            or len(mismatch) != len(executions)
        ):
            raise ValueError("failure certificate demonstration evidence differs")
        if any(type(value) is not int or value < 0 for value in mismatch):
            raise ValueError("failure mismatch counts must be non-negative integers")
        if any(type(value) is not bool for value in shapes):
            raise TypeError("failure shape flags must be boolean")
        normalized_evidence = json.loads(
            json.dumps(dict(evidence), sort_keys=True, allow_nan=False)
        )
        evidence_json = json.dumps(
            normalized_evidence,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        content = {
            "schema": STATEFUL_REWRITE_CERTIFICATE_SCHEMA,
            "blind_task_id": task.task_id,
            "parent_program_id": object_code_program_id(parent),
            "diagnosis": diagnosis,
            "node_id": node_id,
            "affected_subtree": list(affected_scene_subtree(node_id)),
            "mismatch_counts": list(mismatch),
            "shape_match": list(shapes),
            "parent_execution_ids": list(executions),
            "evidence": normalized_evidence,
        }
        return cls(
            canonical_sha256(content),
            task.task_id,
            object_code_program_id(parent),
            diagnosis,
            node_id,
            affected_scene_subtree(node_id),
            mismatch,
            shapes,
            executions,
            evidence_json,
        )

    @property
    def evidence(self) -> dict[str, object]:
        return json.loads(self.evidence_json)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": STATEFUL_REWRITE_CERTIFICATE_SCHEMA,
            "certificate_id": self.certificate_id,
            "blind_task_id": self.blind_task_id,
            "parent_program_id": self.parent_program_id,
            "diagnosis": self.diagnosis,
            "node_id": self.node_id,
            "affected_subtree": list(self.affected_subtree),
            "mismatch_counts": list(self.mismatch_counts),
            "shape_match": list(self.shape_match),
            "parent_execution_ids": list(self.parent_execution_ids),
            "evidence": self.evidence,
        }


def compile_stateful_node_failure(
    task: BlindTask,
    parent: ScenePipelineProgram,
    executions: Sequence[StatefulSceneExecution],
) -> StatefulNodeFailureCertificate:
    """Compile demonstration residuals into one typed AST-node diagnosis."""

    observed = tuple(executions)
    if len(observed) != len(task.train):
        raise ValueError("parent execution count differs from demonstrations")
    residuals = tuple(
        compare_grids(
            execution.output,
            pair.output,
            pair_index=index,
            invalid_code=execution.reason,
        )
        for index, (execution, pair) in enumerate(
            zip(observed, task.train, strict=True)
        )
    )
    if all(item.exact for item in residuals):
        raise ValueError("an exact parent has no node failure to compile")

    invalid = tuple(execution for execution in observed if not execution.ok)
    shape_match = tuple(item.shape_match for item in residuals)
    target_delta_sizes: list[int | None] = []
    parent_delta_sizes: list[int | None] = []
    complete_object_union: list[bool] = []
    if invalid:
        first = invalid[0]
        failed = next(item for item in first.trace if item.status == "invalid")
        node_id = "parse" if first.reason == "no_scene_objects" else failed.node_id
        diagnosis = f"execution_failure:{first.reason}"
    elif not all(shape_match):
        node_id = "canvas"
        diagnosis = "output_canvas_contract_mismatch"
    else:
        for execution, pair in zip(observed, task.train, strict=True):
            if execution.output is None:
                raise AssertionError("valid parent execution lost its output")
            if _grid_shape(pair.input) == _grid_shape(pair.output) and _grid_shape(
                pair.input
            ) == _grid_shape(execution.output):
                target_delta = _delta_coordinates(pair.input, pair.output)
                parent_delta = _delta_coordinates(pair.input, execution.output)
                target_delta_sizes.append(len(target_delta))
                parent_delta_sizes.append(len(parent_delta))
                complete_object_union.append(
                    _delta_is_complete_object_union(execution, target_delta)
                )
            else:
                target_delta_sizes.append(None)
                parent_delta_sizes.append(None)
                complete_object_union.append(False)
        masks_equal = all(
            target is not None and target == parent
            for target, parent in zip(
                target_delta_sizes, parent_delta_sizes, strict=True
            )
        )
        if masks_equal and any(
            value for value in target_delta_sizes if value is not None
        ):
            node_id = "operate"
            diagnosis = "correct_support_wrong_object_operation"
        elif all(complete_object_union):
            node_id = "assignment"
            diagnosis = "wrong_object_or_relation_assignment"
        elif parent.operate.operator in {"crop", "copy", "count", "arrange", "compose"}:
            node_id = "assignment"
            diagnosis = "selection_conditioned_output_mismatch"
        else:
            node_id = "operate"
            diagnosis = "object_operation_mismatch"

    return StatefulNodeFailureCertificate.create(
        task=task,
        parent=parent,
        diagnosis=diagnosis,
        node_id=node_id,
        mismatch_counts=tuple(item.mismatch_count for item in residuals),
        shape_match=shape_match,
        parent_execution_ids=tuple(item.execution_id for item in observed),
        evidence={
            "complete_object_union": complete_object_union,
            "execution_failure_count": len(invalid),
            "parent_delta_sizes": parent_delta_sizes,
            "target_delta_sizes": target_delta_sizes,
            "trace_state_ids": [
                [node.state_id for node in execution.trace] for execution in observed
            ],
        },
    )


@dataclass(frozen=True, slots=True)
class StatefulSceneRewrite:
    rewrite_id: str
    parent_program_id: str
    certificate_id: str
    node_id: str
    affected_subtree: tuple[str, ...]
    program: ScenePipelineProgram

    @classmethod
    def create(
        cls,
        *,
        parent: ScenePipelineProgram,
        certificate: StatefulNodeFailureCertificate,
        program: ScenePipelineProgram,
    ) -> "StatefulSceneRewrite":
        parent_id = object_code_program_id(parent)
        if certificate.parent_program_id != parent_id:
            raise ValueError("certificate belongs to another parent program")
        differences = scene_program_node_differences(parent, program)
        if differences != (certificate.node_id,):
            raise ValueError("counterfactual must change exactly the diagnosed node")
        content = {
            "schema": STATEFUL_REWRITE_SCHEMA,
            "dsl_version": STATEFUL_REWRITE_DSL_VERSION,
            "parent_program_id": parent_id,
            "certificate_id": certificate.certificate_id,
            "node_id": certificate.node_id,
            "affected_subtree": list(certificate.affected_subtree),
            "program": program.to_json_dict(),
        }
        return cls(
            canonical_sha256(content),
            parent_id,
            certificate.certificate_id,
            certificate.node_id,
            certificate.affected_subtree,
            program,
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": STATEFUL_REWRITE_SCHEMA,
            "rewrite_id": self.rewrite_id,
            "dsl_version": STATEFUL_REWRITE_DSL_VERSION,
            "parent_program_id": self.parent_program_id,
            "certificate_id": self.certificate_id,
            "node_id": self.node_id,
            "affected_subtree": list(self.affected_subtree),
            "program": self.program.to_json_dict(),
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "StatefulSceneRewrite":
        if not isinstance(payload, Mapping):
            raise TypeError("stateful rewrite must be an object")
        expected = {
            "schema",
            "rewrite_id",
            "dsl_version",
            "parent_program_id",
            "certificate_id",
            "node_id",
            "affected_subtree",
            "program",
        }
        if set(payload) != expected:
            raise ValueError("stateful rewrite fields differ")
        if payload["schema"] != STATEFUL_REWRITE_SCHEMA:
            raise ValueError("unsupported stateful rewrite schema")
        if payload["dsl_version"] != STATEFUL_REWRITE_DSL_VERSION:
            raise ValueError("unsupported stateful rewrite DSL")
        node_id = payload["node_id"]
        if node_id not in STATEFUL_NODE_ORDER:
            raise ValueError("stateful rewrite node differs")
        subtree = payload["affected_subtree"]
        if not isinstance(subtree, list) or tuple(subtree) != affected_scene_subtree(
            node_id
        ):
            raise ValueError("stateful rewrite dependency subtree differs")
        program = ScenePipelineProgram.from_json_dict(payload["program"])
        content = {
            "schema": STATEFUL_REWRITE_SCHEMA,
            "dsl_version": STATEFUL_REWRITE_DSL_VERSION,
            "parent_program_id": payload["parent_program_id"],
            "certificate_id": payload["certificate_id"],
            "node_id": node_id,
            "affected_subtree": subtree,
            "program": program.to_json_dict(),
        }
        rebuilt = cls(
            canonical_sha256(content),
            payload["parent_program_id"],
            payload["certificate_id"],
            node_id,
            tuple(subtree),
            program,
        )
        if payload["rewrite_id"] != rebuilt.rewrite_id:
            raise ValueError("stateful rewrite ID differs from canonical content")
        return rebuilt


def enumerate_stateful_scene_rewrites(
    parent: ScenePipelineProgram,
    certificate: StatefulNodeFailureCertificate,
    *,
    candidate_programs: Sequence[ScenePipelineProgram],
    max_trials: int,
) -> tuple[StatefulSceneRewrite, ...]:
    if type(max_trials) is not int or max_trials < 1:
        raise ValueError("rewrite trial bound must be positive")
    eligible = {
        object_code_program_id(program): program
        for program in candidate_programs
        if scene_program_node_differences(parent, program) == (certificate.node_id,)
    }
    ordered = sorted(
        eligible.values(),
        key=lambda program: (
            program.description_bits,
            canonical_sha256(scene_program_node_payloads(program)[certificate.node_id]),
            object_code_program_id(program),
        ),
    )[:max_trials]
    return tuple(
        StatefulSceneRewrite.create(
            parent=parent,
            certificate=certificate,
            program=program,
        )
        for program in ordered
    )


def execute_stateful_scene_rewrite(
    rewrite: StatefulSceneRewrite,
    grid: Grid,
    *,
    parent_execution: StatefulSceneExecution | None,
) -> StatefulSceneExecution:
    """Execute a rewrite only when its content-addressed bridge is present."""

    if parent_execution is None:
        raise ValueError("stateful rewrite requires its executable parent trace")
    if object_code_program_id(parent_execution.program) != rewrite.parent_program_id:
        raise ValueError("stateful rewrite parent trace differs")
    return execute_stateful_scene_pipeline(
        rewrite.program,
        grid,
        prior=parent_execution,
        changed_node=rewrite.node_id,
    )


@dataclass(frozen=True, slots=True)
class StatefulRewriteCandidate:
    rewrite: StatefulSceneRewrite
    certificate: StatefulNodeFailureCertificate
    demo_outputs: tuple[Grid, ...]
    query_outputs: tuple[Grid, ...]
    parent_trace_ids: tuple[str, ...]

    @property
    def output_bundle_id(self) -> str:
        return _bundle_id(self.query_outputs)


@dataclass(frozen=True, slots=True)
class StatefulRewriteSynthesisResult:
    baseline_programs: tuple[ObjectCodeProgram, ...]
    baseline_output_bundle_ids: tuple[str, ...]
    certificates: tuple[StatefulNodeFailureCertificate, ...]
    candidates: tuple[StatefulRewriteCandidate, ...]
    novel_candidates: tuple[StatefulRewriteCandidate, ...]
    first_stage_program_trials: int
    rewrite_program_trials: int
    rewrite_padding_trials: int
    parent_count: int
    demo_node_executions: tuple[tuple[str, int], ...]
    query_node_executions: tuple[tuple[str, int], ...]
    reused_demo_nodes: tuple[tuple[str, int], ...]


def _execute_object_code_bundle(
    program: ObjectCodeProgram,
    inputs: Sequence[Grid],
) -> tuple[Grid, ...] | None:
    outputs = []
    for grid in inputs:
        execution = execute_object_code_program(program, grid)
        if not execution.ok or execution.output is None:
            return None
        outputs.append(execution.output)
    return tuple(outputs)


def _stateful_parent_executions(
    program: ScenePipelineProgram,
    inputs: Sequence[Grid],
) -> tuple[StatefulSceneExecution, ...] | None:
    executions = tuple(
        execute_stateful_scene_pipeline(program, grid) for grid in inputs
    )
    if any(not execution.ok for execution in executions):
        return None
    return executions


def _verify_rewrite_execution(
    rewrite: StatefulSceneRewrite,
    grid: Grid,
    parent_execution: StatefulSceneExecution,
    replayed: StatefulSceneExecution,
) -> None:
    round_trip = StatefulSceneRewrite.from_json_dict(rewrite.to_json_dict())
    if round_trip != rewrite:
        raise AssertionError("stateful rewrite AST round-trip differs")
    fresh = execute_stateful_scene_pipeline(rewrite.program, grid)
    legacy = execute_scene_pipeline(rewrite.program, grid)
    if not replayed.ok or replayed.output is None:
        raise AssertionError("candidate replay became invalid")
    if fresh.output != replayed.output or legacy.output != replayed.output:
        raise AssertionError("suffix replay differs from full scene execution")
    expected_reuse = STATEFUL_NODE_ORDER[: STATEFUL_NODE_ORDER.index(rewrite.node_id)]
    if replayed.reused_node_ids != expected_reuse:
        raise AssertionError("suffix replay reused the wrong dependency prefix")
    if replayed.executed_node_ids != affected_scene_subtree(rewrite.node_id):
        raise AssertionError("suffix replay executed the wrong dependency subtree")
    if parent_execution.input_grid_id != replayed.input_grid_id:
        raise AssertionError("suffix replay changed the input identity")


def _parent_sort_key(score: ObjectCodeProgramScore) -> tuple[object, ...]:
    program = score.program
    if not isinstance(program, ScenePipelineProgram):
        raise TypeError("stateful parent ordering requires a scene program")
    return (
        -score.exact_demo_count,
        -score.shape_match_count,
        score.mismatch_count,
        -score.agreement,
        program.description_bits,
        object_code_program_id(program),
    )


def synthesize_stateful_object_graph_rewrites(
    task: BlindTask,
    *,
    max_first_stage_trials: int = 2_000,
    max_parents: int = 4,
    max_rewrite_trials: int = 512,
    max_candidates: int = 32,
    first_stage_programs: Sequence[ObjectCodeProgram] | None = None,
    rewrite_programs: Sequence[ScenePipelineProgram] | None = None,
) -> StatefulRewriteSynthesisResult:
    """Search one diagnosed node while preserving all unaffected trace state."""

    if not isinstance(task, BlindTask):
        raise TypeError("stateful rewrite synthesis accepts BlindTask only")
    for name, value in (
        ("max_first_stage_trials", max_first_stage_trials),
        ("max_parents", max_parents),
        ("max_rewrite_trials", max_rewrite_trials),
        ("max_candidates", max_candidates),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    first = synthesize_object_code_programs(
        task,
        max_program_trials=max_first_stage_trials,
        max_exact_programs=max_first_stage_trials,
        max_near_misses=max_first_stage_trials,
        minimum_near_miss_agreement=0.0,
        programs=first_stage_programs,
    )
    baseline_programs = tuple(score.program for score in first.exact_scores)
    baseline_bundle_ids = []
    for program in baseline_programs:
        outputs = _execute_object_code_bundle(program, task.test_inputs)
        if outputs is None:
            raise AssertionError("demo-exact baseline failed on query input")
        baseline_bundle_ids.append(_bundle_id(outputs))
    baseline_ids = frozenset(baseline_bundle_ids)

    parent_scores = sorted(
        (
            score
            for score in first.near_miss_scores
            if isinstance(score.program, ScenePipelineProgram)
            and score.execution_valid
            and not score.all_demo_exact
        ),
        key=_parent_sort_key,
    )
    grammar = tuple(
        enumerate_scene_pipeline_programs(task)
        if rewrite_programs is None
        else rewrite_programs
    )
    certificates = []
    candidates_by_rewrite: dict[str, StatefulRewriteCandidate] = {}
    rewrite_trials = 0
    parent_count = 0
    demo_nodes: Counter[str] = Counter()
    query_nodes: Counter[str] = Counter()
    reused_demo: Counter[str] = Counter()

    for score in parent_scores:
        if parent_count >= max_parents:
            break
        parent = score.program
        if not isinstance(parent, ScenePipelineProgram):
            raise AssertionError("filtered parent lost its scene-program type")
        demo_parent = _stateful_parent_executions(
            parent, tuple(pair.input for pair in task.train)
        )
        query_parent = _stateful_parent_executions(parent, task.test_inputs)
        if demo_parent is None or query_parent is None:
            continue
        certificate = compile_stateful_node_failure(task, parent, demo_parent)
        rewrites = enumerate_stateful_scene_rewrites(
            parent,
            certificate,
            candidate_programs=grammar,
            max_trials=max_rewrite_trials,
        )
        certificates.append(certificate)
        parent_count += 1
        rewrite_trials += len(rewrites)
        for rewrite in rewrites:
            demo_replays = tuple(
                execute_stateful_scene_rewrite(
                    rewrite,
                    pair.input,
                    parent_execution=parent_execution,
                )
                for pair, parent_execution in zip(task.train, demo_parent, strict=True)
            )
            for execution in demo_replays:
                demo_nodes.update(execution.executed_node_ids)
                reused_demo.update(execution.reused_node_ids)
            if any(
                not execution.ok or execution.output is None
                for execution in demo_replays
            ):
                continue
            demo_outputs = tuple(
                execution.output
                for execution in demo_replays
                if execution.output is not None
            )
            if demo_outputs != tuple(pair.output for pair in task.train):
                continue
            query_replays = tuple(
                execute_stateful_scene_rewrite(
                    rewrite,
                    grid,
                    parent_execution=parent_execution,
                )
                for grid, parent_execution in zip(
                    task.test_inputs, query_parent, strict=True
                )
            )
            for execution in query_replays:
                query_nodes.update(execution.executed_node_ids)
            if any(
                not execution.ok or execution.output is None
                for execution in query_replays
            ):
                continue
            for grid, parent_execution, replayed in zip(
                tuple(pair.input for pair in task.train) + task.test_inputs,
                demo_parent + query_parent,
                demo_replays + query_replays,
                strict=True,
            ):
                _verify_rewrite_execution(rewrite, grid, parent_execution, replayed)
            query_outputs = tuple(
                execution.output
                for execution in query_replays
                if execution.output is not None
            )
            candidates_by_rewrite[rewrite.rewrite_id] = StatefulRewriteCandidate(
                rewrite,
                certificate,
                demo_outputs,
                query_outputs,
                tuple(
                    execution.execution_id for execution in demo_parent + query_parent
                ),
            )

    ordered_candidates = sorted(
        candidates_by_rewrite.values(),
        key=lambda candidate: (
            candidate.rewrite.program.description_bits,
            candidate.rewrite.rewrite_id,
        ),
    )[:max_candidates]
    by_output: dict[str, StatefulRewriteCandidate] = {}
    for candidate in ordered_candidates:
        incumbent = by_output.get(candidate.output_bundle_id)
        if incumbent is None or (
            candidate.rewrite.program.description_bits,
            candidate.rewrite.rewrite_id,
        ) < (
            incumbent.rewrite.program.description_bits,
            incumbent.rewrite.rewrite_id,
        ):
            by_output[candidate.output_bundle_id] = candidate
    candidates = tuple(by_output[key] for key in sorted(by_output))
    novel = tuple(
        candidate
        for candidate in candidates
        if candidate.output_bundle_id not in baseline_ids
    )
    reserved = max_parents * max_rewrite_trials
    return StatefulRewriteSynthesisResult(
        baseline_programs,
        tuple(sorted(set(baseline_bundle_ids))),
        tuple(certificates),
        candidates,
        novel,
        first.program_trial_count,
        rewrite_trials,
        reserved - rewrite_trials,
        parent_count,
        tuple(sorted(demo_nodes.items())),
        tuple(sorted(query_nodes.items())),
        tuple(sorted(reused_demo.items())),
    )
