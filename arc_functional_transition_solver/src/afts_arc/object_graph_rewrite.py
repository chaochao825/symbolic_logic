"""Trace-conditioned, two-stage object-program composition.

Object–Program Workspace v1 could change one selector but could not move into a
new representation region on natural tasks.  This opt-in v2 bridge treats an
execution-valid first-stage near miss as an object/program state, compiles its
demonstration residual into a typed second-stage hole, and synthesizes only that
hole.  Query outputs are never inputs to synthesis or ranking.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .blind import BlindTask
from .cognitive_workspace import FailureSignal
from .grid import Grid, grid_key
from .hybrid.object_code import (
    ObjectCodeProgram,
    ObjectCodeProgramScore,
    execute_object_code_program,
    object_code_program_from_json,
    object_code_program_id,
    synthesize_object_code_programs,
)
from .hybrid.types import CandidateHypothesis, canonical_json
from .residual import compare_grids
from .task import ARCPair


OBJECT_GRAPH_REWRITE_DSL_VERSION = "afts-object-graph-rewrite-dsl/v2"
OBJECT_GRAPH_REWRITE_PROVIDER_VERSION = "afts-object-graph-rewrite/v2"
OBJECT_GRAPH_REWRITE_CERTIFICATE_VERSION = "afts-object-graph-rewrite-certificate/v2"


def _content_id(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()


def _bundle_id(outputs: Sequence[Grid]) -> str:
    return _content_id([grid_key(output) for output in outputs])


@dataclass(frozen=True, slots=True)
class ObjectGraphRewriteProgram:
    """A typed two-stage AST whose second stage closes a residual-derived hole."""

    first_stage: ObjectCodeProgram
    second_stage: ObjectCodeProgram

    @property
    def program_id(self) -> str:
        return _content_id(self.to_json_dict())

    @property
    def description_bits(self) -> int:
        return self.first_stage.description_bits + self.second_stage.description_bits + 8

    @property
    def functional_trace(self) -> tuple[str, ...]:
        return (
            "workspace:load_executable_parent",
            *(f"stage0:{item}" for item in self.first_stage.functional_trace),
            "bridge:failure_certificate->stages[1]",
            *(f"stage1:{item}" for item in self.second_stage.functional_trace),
            "verify:demo_exact_composition",
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "dsl_version": OBJECT_GRAPH_REWRITE_DSL_VERSION,
            "kind": "object_graph_rewrite_composition",
            "stages": [
                self.first_stage.to_json_dict(),
                self.second_stage.to_json_dict(),
            ],
            "hole_contract": {
                "filled_slot": "stages[1]",
                "input_type": "object_code_execution_grid",
                "output_type": "arc_grid",
            },
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "ObjectGraphRewriteProgram":
        if not isinstance(payload, Mapping):
            raise TypeError("object-graph rewrite program must be an object")
        expected = {"dsl_version", "kind", "stages", "hole_contract"}
        if set(payload) != expected:
            raise ValueError("object-graph rewrite program fields differ")
        if payload["dsl_version"] != OBJECT_GRAPH_REWRITE_DSL_VERSION:
            raise ValueError("unsupported object-graph rewrite DSL")
        if payload["kind"] != "object_graph_rewrite_composition":
            raise ValueError("object-graph rewrite program kind differs")
        if payload["hole_contract"] != {
            "filled_slot": "stages[1]",
            "input_type": "object_code_execution_grid",
            "output_type": "arc_grid",
        }:
            raise ValueError("object-graph rewrite hole contract differs")
        stages = payload["stages"]
        if not isinstance(stages, list) or len(stages) != 2:
            raise ValueError("object-graph rewrite program requires two stages")
        return cls(
            object_code_program_from_json(stages[0]),
            object_code_program_from_json(stages[1]),
        )


@dataclass(frozen=True, slots=True)
class ObjectGraphRewriteExecution:
    status: str
    output: Grid | None
    reason: str | None
    completed_stage_count: int

    def __post_init__(self) -> None:
        if self.status not in {"ok", "invalid"}:
            raise ValueError("composition execution status differs")
        if (self.status == "ok") != (self.output is not None):
            raise ValueError("only successful composition execution has output")
        if type(self.completed_stage_count) is not int or not 0 <= self.completed_stage_count <= 2:
            raise ValueError("completed stage count differs")

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def execute_object_graph_rewrite(
    program: ObjectGraphRewriteProgram, grid: Grid
) -> ObjectGraphRewriteExecution:
    if not isinstance(program, ObjectGraphRewriteProgram):
        raise TypeError("object-graph rewrite execution requires a typed program")
    first = execute_object_code_program(program.first_stage, grid)
    if not first.ok or first.output is None:
        return ObjectGraphRewriteExecution("invalid", None, "stage0_invalid", 0)
    second = execute_object_code_program(program.second_stage, first.output)
    if not second.ok or second.output is None:
        return ObjectGraphRewriteExecution("invalid", None, "stage1_invalid", 1)
    return ObjectGraphRewriteExecution("ok", second.output, None, 2)


@dataclass(frozen=True, slots=True)
class ObjectGraphRewriteCertificate:
    certificate_id: str
    blind_task_id: str
    parent_program_id: str
    mismatch_counts: tuple[int, ...]
    shape_match: tuple[bool, ...]
    affected_slots: tuple[str, ...] = ("stages[1]",)

    @classmethod
    def from_parent_score(
        cls,
        task: BlindTask,
        parent: ObjectCodeProgramScore,
    ) -> "ObjectGraphRewriteCertificate":
        if parent.all_demo_exact or not parent.execution_valid:
            raise ValueError("composition parent must be an execution-valid near miss")
        residuals = tuple(
            compare_grids(
                output,
                pair.output,
                pair_index=index,
                invalid_code=None,
            )
            for index, (output, pair) in enumerate(zip(parent.demo_outputs, task.train, strict=True))
        )
        mismatch_counts = tuple(item.mismatch_count for item in residuals)
        shape_match = tuple(item.shape_match for item in residuals)
        payload = {
            "schema": OBJECT_GRAPH_REWRITE_CERTIFICATE_VERSION,
            "blind_task_id": task.task_id,
            "parent_program_id": object_code_program_id(parent.program),
            "mismatch_counts": list(mismatch_counts),
            "shape_match": list(shape_match),
            "affected_slots": ["stages[1]"],
        }
        return cls(
            _content_id(payload),
            task.task_id,
            object_code_program_id(parent.program),
            mismatch_counts,
            shape_match,
        )

    def to_failure_signal(self) -> FailureSignal:
        return FailureSignal.create(
            certificate_type=(
                "canvas_then_ast_hole"
                if not all(self.shape_match)
                else "residual_ast_hole"
            ),
            current_representation="object_code_program",
            target_representation="object_program_composition",
            affected_slots=self.affected_slots,
            evidence_ids=(self.certificate_id,),
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": OBJECT_GRAPH_REWRITE_CERTIFICATE_VERSION,
            "certificate_id": self.certificate_id,
            "blind_task_id": self.blind_task_id,
            "parent_program_id": self.parent_program_id,
            "mismatch_counts": list(self.mismatch_counts),
            "shape_match": list(self.shape_match),
            "affected_slots": list(self.affected_slots),
        }


@dataclass(frozen=True, slots=True)
class ObjectGraphRewriteCandidate:
    program: ObjectGraphRewriteProgram
    certificate: ObjectGraphRewriteCertificate
    demo_outputs: tuple[Grid, ...]
    query_outputs: tuple[Grid, ...]

    @property
    def output_bundle_id(self) -> str:
        return _bundle_id(self.query_outputs)


@dataclass(frozen=True, slots=True)
class ObjectGraphRewriteSynthesisResult:
    baseline_programs: tuple[ObjectCodeProgram, ...]
    baseline_output_bundle_ids: tuple[str, ...]
    parent_certificates: tuple[ObjectGraphRewriteCertificate, ...]
    candidates: tuple[ObjectGraphRewriteCandidate, ...]
    novel_candidates: tuple[ObjectGraphRewriteCandidate, ...]
    first_stage_program_trials: int
    second_stage_program_trials: int
    parent_count: int
    second_stage_padding_trials: int


def _execute_bundle(
    program: ObjectCodeProgram, inputs: Sequence[Grid]
) -> tuple[Grid, ...] | None:
    outputs = []
    for grid in inputs:
        result = execute_object_code_program(program, grid)
        if not result.ok or result.output is None:
            return None
        outputs.append(result.output)
    return tuple(outputs)


def _execute_composition_bundle(
    program: ObjectGraphRewriteProgram, inputs: Sequence[Grid]
) -> tuple[Grid, ...] | None:
    outputs = []
    for grid in inputs:
        result = execute_object_graph_rewrite(program, grid)
        if not result.ok or result.output is None:
            return None
        outputs.append(result.output)
    return tuple(outputs)


def _intermediate_task(
    task: BlindTask,
    parent: ObjectCodeProgramScore,
) -> tuple[BlindTask, tuple[Grid, ...]] | None:
    if any(output is None for output in parent.demo_outputs):
        return None
    query_outputs = _execute_bundle(parent.program, task.test_inputs)
    if query_outputs is None:
        return None
    demo_outputs = tuple(
        output for output in parent.demo_outputs if output is not None
    )
    residual_task = BlindTask.from_observations(
        train=tuple(
            ARCPair(intermediate, pair.output)
            for intermediate, pair in zip(demo_outputs, task.train, strict=True)
        ),
        test_inputs=query_outputs,
    )
    return residual_task, query_outputs


def synthesize_object_graph_rewrites(
    task: BlindTask,
    *,
    max_first_stage_trials: int = 2_000,
    max_parents: int = 4,
    max_second_stage_trials: int = 512,
    max_candidates: int = 32,
    first_stage_programs: Sequence[ObjectCodeProgram] | None = None,
    second_stage_programs: Sequence[ObjectCodeProgram] | None = None,
) -> ObjectGraphRewriteSynthesisResult:
    """Fill a typed second-stage hole from demonstration-only execution traces."""

    if not isinstance(task, BlindTask):
        raise TypeError("object-graph rewrite synthesis accepts BlindTask only")
    for name, value in (
        ("max_first_stage_trials", max_first_stage_trials),
        ("max_parents", max_parents),
        ("max_second_stage_trials", max_second_stage_trials),
        ("max_candidates", max_candidates),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    first = synthesize_object_code_programs(
        task,
        max_program_trials=max_first_stage_trials,
        max_exact_programs=max_first_stage_trials,
        max_near_misses=max_parents,
        minimum_near_miss_agreement=0.0,
        programs=first_stage_programs,
    )
    baseline_programs = tuple(score.program for score in first.exact_scores)
    baseline_bundle_ids = []
    for program in baseline_programs:
        query_outputs = _execute_bundle(program, task.test_inputs)
        if query_outputs is None:
            raise AssertionError("synthesis returned an invalid exact baseline")
        baseline_bundle_ids.append(_bundle_id(query_outputs))
    baseline_ids = frozenset(baseline_bundle_ids)

    parents = tuple(
        score
        for score in first.near_miss_scores
        if score.execution_valid and not score.all_demo_exact
    )[:max_parents]
    certificates = []
    candidates_by_program: dict[str, ObjectGraphRewriteCandidate] = {}
    second_trials = 0
    for parent in parents:
        intermediate = _intermediate_task(task, parent)
        if intermediate is None:
            continue
        residual_task, _ = intermediate
        certificate = ObjectGraphRewriteCertificate.from_parent_score(task, parent)
        certificates.append(certificate)
        second = synthesize_object_code_programs(
            residual_task,
            max_program_trials=max_second_stage_trials,
            max_exact_programs=max_candidates,
            max_near_misses=0,
            programs=second_stage_programs,
        )
        second_trials += second.program_trial_count
        for score in second.exact_scores:
            program = ObjectGraphRewriteProgram(parent.program, score.program)
            demo_outputs = _execute_composition_bundle(
                program, tuple(pair.input for pair in task.train)
            )
            query_outputs = _execute_composition_bundle(program, task.test_inputs)
            if demo_outputs is None or query_outputs is None:
                continue
            if demo_outputs != tuple(pair.output for pair in task.train):
                raise AssertionError("second-stage exact score did not close composition")
            candidates_by_program[program.program_id] = ObjectGraphRewriteCandidate(
                program,
                certificate,
                demo_outputs,
                query_outputs,
            )
    candidates = tuple(
        sorted(
            candidates_by_program.values(),
            key=lambda candidate: (
                candidate.program.description_bits,
                candidate.program.program_id,
            ),
        )[:max_candidates]
    )
    by_output: dict[str, ObjectGraphRewriteCandidate] = {}
    for candidate in candidates:
        incumbent = by_output.get(candidate.output_bundle_id)
        if incumbent is None or (
            candidate.program.description_bits,
            candidate.program.program_id,
        ) < (
            incumbent.program.description_bits,
            incumbent.program.program_id,
        ):
            by_output[candidate.output_bundle_id] = candidate
    deduplicated = tuple(
        by_output[key] for key in sorted(by_output)
    )
    novel = tuple(
        candidate
        for candidate in deduplicated
        if candidate.output_bundle_id not in baseline_ids
    )
    reserved_second_trials = max_parents * max_second_stage_trials
    return ObjectGraphRewriteSynthesisResult(
        baseline_programs,
        tuple(sorted(set(baseline_bundle_ids))),
        tuple(certificates),
        deduplicated,
        novel,
        first.program_trial_count,
        second_trials,
        len(parents),
        reserved_second_trials - second_trials,
    )


def make_object_graph_rewrite_hypothesis(
    candidate: ObjectGraphRewriteCandidate,
) -> CandidateHypothesis:
    program = candidate.program
    serialized = program.to_json_dict()

    def replay(grid: Grid) -> object | None:
        result = execute_object_graph_rewrite(program, grid)
        return result.output if result.ok else None

    def hard_verify(task: BlindTask) -> bool:
        reconstructed = ObjectGraphRewriteProgram.from_json_dict(serialized)
        if reconstructed.to_json_dict() != serialized:
            return False
        for grid in tuple(pair.input for pair in task.train) + task.test_inputs:
            first = execute_object_graph_rewrite(program, grid)
            second = execute_object_graph_rewrite(reconstructed, grid)
            if first != second or not first.ok:
                return False
        return True

    return CandidateHypothesis.create(
        name=f"object_graph_rewrite:{program.program_id[:16]}",
        source="object_graph_rewrite",
        source_version=OBJECT_GRAPH_REWRITE_PROVIDER_VERSION,
        route="dsl_program",
        description_bits=program.description_bits,
        verification_mode="replayable",
        functional_trace=program.functional_trace,
        spec={
            "program": serialized,
            "program_id": program.program_id,
            "certificate_id": candidate.certificate.certificate_id,
            "filled_holes": ["stages[1]"],
        },
        metadata={
            "demo_exact_at_generation": True,
            "support_gate_passed": True,
            "output_bundle_id": candidate.output_bundle_id,
        },
        replay=replay,
        hard_verifier=hard_verify,
        parent_hypothesis_ids=(candidate.certificate.parent_program_id,),
    )
