"""Bounded multi-node transitions over persistent scene execution state.

The v4 lane keeps the v3 parent grammar and native reservation fixed while
testing three query-blind search policies.  Every emitted candidate names all
changed AST regions, cites an executable parent trace, replays from the first
changed node, and agrees with a fresh execution before it can enter the output
frontier.
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
    object_code_program_id,
    synthesize_object_code_programs,
)
from .hybrid.scene_graph import (
    ScenePipelineProgram,
    enumerate_scene_pipeline_programs,
    execute_scene_pipeline,
)
from .residual import compare_grids
from .stateful_object_graph_rewrite import (
    StatefulNodeFailureCertificate,
    compile_stateful_node_failure,
)
from .stateful_scene import (
    STATEFUL_NODE_ORDER,
    StatefulSceneExecution,
    affected_scene_subtree,
    execute_stateful_scene_pipeline,
    execute_stateful_scene_transition,
    scene_program_node_differences,
    scene_program_node_payloads,
)


COUNTERFACTUAL_TRANSITION_DSL_VERSION = "afts-counterfactual-transition-dsl/v4"
COUNTERFACTUAL_TRANSITION_PROVIDER_VERSION = "afts-counterfactual-transition/v4"
COUNTERFACTUAL_TRANSITION_SCHEMA = "afts.counterfactual-transition/v1"
COUNTERFACTUAL_STRATEGIES = (
    "distance_tiered",
    "typed_semantic_bundles",
    "residual_beam",
)


def _bundle_id(outputs: Sequence[Grid]) -> str:
    return canonical_sha256([grid_key(output) for output in outputs])


def _program_sort_key(program: ScenePipelineProgram) -> tuple[object, ...]:
    return (
        program.description_bits,
        object_code_program_id(program),
    )


def _parent_sort_key(score: ObjectCodeProgramScore) -> tuple[object, ...]:
    program = score.program
    if not isinstance(program, ScenePipelineProgram):
        raise TypeError("transition parent ordering requires a scene program")
    return (
        -score.exact_demo_count,
        -score.shape_match_count,
        score.mismatch_count,
        -score.agreement,
        program.description_bits,
        object_code_program_id(program),
    )


def _canonical_changed_nodes(nodes: Sequence[str]) -> tuple[str, ...]:
    declared = tuple(nodes)
    if not declared or len(declared) > 3:
        raise ValueError("a counterfactual transition must change one to three nodes")
    if any(node_id not in STATEFUL_NODE_ORDER for node_id in declared):
        raise ValueError("counterfactual transition contains an unknown node")
    canonical = tuple(node_id for node_id in STATEFUL_NODE_ORDER if node_id in declared)
    if declared != canonical:
        raise ValueError("counterfactual transition nodes must be unique and ordered")
    return declared


def _semantic_bundles(node_id: str) -> tuple[tuple[str, ...], ...]:
    bundles = {
        "parse": (
            ("parse",),
            ("parse", "assignment"),
            ("parse", "canvas"),
            ("parse", "assignment", "operate"),
            ("parse", "canvas", "render"),
        ),
        "assignment": (
            ("assignment",),
            ("assignment", "operate"),
            ("parse", "assignment"),
            ("assignment", "canvas"),
            ("assignment", "operate", "canvas"),
        ),
        "operate": (
            ("operate",),
            ("assignment", "operate"),
            ("operate", "canvas"),
            ("operate", "render"),
            ("assignment", "operate", "canvas"),
        ),
        "canvas": (
            ("canvas",),
            ("operate", "canvas"),
            ("assignment", "canvas"),
            ("canvas", "render"),
            ("parse", "canvas"),
            ("operate", "canvas", "render"),
        ),
        "render": (
            ("render",),
            ("operate", "render"),
            ("canvas", "render"),
            ("assignment", "render"),
            ("operate", "canvas", "render"),
        ),
    }
    if node_id not in bundles:
        raise ValueError("semantic transition requires a known diagnosed node")
    return bundles[node_id]


def _round_robin_programs(
    groups: Sequence[Sequence[ScenePipelineProgram]],
    *,
    max_trials: int,
    schedule: Sequence[int] | None = None,
) -> tuple[ScenePipelineProgram, ...]:
    if type(max_trials) is not int or max_trials < 1:
        raise ValueError("transition trial bound must be positive")
    ordered_groups = tuple(tuple(sorted(group, key=_program_sort_key)) for group in groups)
    if not ordered_groups:
        return ()
    cycle = tuple(range(len(ordered_groups))) if schedule is None else tuple(schedule)
    if not cycle or any(index < 0 or index >= len(ordered_groups) for index in cycle):
        raise ValueError("round-robin schedule differs from its program groups")
    offsets = [0 for _ in ordered_groups]
    selected: list[ScenePipelineProgram] = []
    selected_ids: set[str] = set()
    while len(selected) < max_trials:
        progressed = False
        for index in cycle:
            group = ordered_groups[index]
            while offsets[index] < len(group):
                program = group[offsets[index]]
                offsets[index] += 1
                program_id = object_code_program_id(program)
                if program_id in selected_ids:
                    continue
                selected.append(program)
                selected_ids.add(program_id)
                progressed = True
                break
            if len(selected) >= max_trials:
                break
        if not progressed:
            break
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class CounterfactualTransition:
    transition_id: str
    strategy: str
    parent_program_id: str
    certificate_id: str
    changed_nodes: tuple[str, ...]
    affected_subtree: tuple[str, ...]
    program: ScenePipelineProgram

    @classmethod
    def create(
        cls,
        *,
        strategy: str,
        parent: ScenePipelineProgram,
        certificate: StatefulNodeFailureCertificate,
        program: ScenePipelineProgram,
    ) -> "CounterfactualTransition":
        if strategy not in COUNTERFACTUAL_STRATEGIES:
            raise ValueError("unknown counterfactual transition strategy")
        parent_id = object_code_program_id(parent)
        if certificate.parent_program_id != parent_id:
            raise ValueError("transition certificate belongs to another parent")
        changed_nodes = _canonical_changed_nodes(
            scene_program_node_differences(parent, program)
        )
        affected_subtree = affected_scene_subtree(changed_nodes[0])
        content = {
            "schema": COUNTERFACTUAL_TRANSITION_SCHEMA,
            "dsl_version": COUNTERFACTUAL_TRANSITION_DSL_VERSION,
            "strategy": strategy,
            "parent_program_id": parent_id,
            "certificate_id": certificate.certificate_id,
            "changed_nodes": list(changed_nodes),
            "affected_subtree": list(affected_subtree),
            "program": program.to_json_dict(),
        }
        return cls(
            canonical_sha256(content),
            strategy,
            parent_id,
            certificate.certificate_id,
            changed_nodes,
            affected_subtree,
            program,
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": COUNTERFACTUAL_TRANSITION_SCHEMA,
            "transition_id": self.transition_id,
            "dsl_version": COUNTERFACTUAL_TRANSITION_DSL_VERSION,
            "strategy": self.strategy,
            "parent_program_id": self.parent_program_id,
            "certificate_id": self.certificate_id,
            "changed_nodes": list(self.changed_nodes),
            "affected_subtree": list(self.affected_subtree),
            "program": self.program.to_json_dict(),
        }

    @classmethod
    def from_json_dict(cls, payload: object) -> "CounterfactualTransition":
        if not isinstance(payload, Mapping):
            raise TypeError("counterfactual transition must be an object")
        expected = {
            "schema",
            "transition_id",
            "dsl_version",
            "strategy",
            "parent_program_id",
            "certificate_id",
            "changed_nodes",
            "affected_subtree",
            "program",
        }
        if set(payload) != expected:
            raise ValueError("counterfactual transition fields differ")
        if payload["schema"] != COUNTERFACTUAL_TRANSITION_SCHEMA:
            raise ValueError("unsupported counterfactual transition schema")
        if payload["dsl_version"] != COUNTERFACTUAL_TRANSITION_DSL_VERSION:
            raise ValueError("unsupported counterfactual transition DSL")
        strategy = payload["strategy"]
        if strategy not in COUNTERFACTUAL_STRATEGIES:
            raise ValueError("counterfactual transition strategy differs")
        raw_nodes = payload["changed_nodes"]
        if not isinstance(raw_nodes, list):
            raise TypeError("counterfactual changed nodes must be an array")
        changed_nodes = _canonical_changed_nodes(raw_nodes)
        raw_subtree = payload["affected_subtree"]
        if not isinstance(raw_subtree, list):
            raise TypeError("counterfactual affected subtree must be an array")
        affected_subtree = affected_scene_subtree(changed_nodes[0])
        if tuple(raw_subtree) != affected_subtree:
            raise ValueError("counterfactual affected subtree differs")
        program = ScenePipelineProgram.from_json_dict(payload["program"])
        content = {
            "schema": COUNTERFACTUAL_TRANSITION_SCHEMA,
            "dsl_version": COUNTERFACTUAL_TRANSITION_DSL_VERSION,
            "strategy": strategy,
            "parent_program_id": payload["parent_program_id"],
            "certificate_id": payload["certificate_id"],
            "changed_nodes": list(changed_nodes),
            "affected_subtree": list(affected_subtree),
            "program": program.to_json_dict(),
        }
        rebuilt = cls(
            canonical_sha256(content),
            strategy,
            payload["parent_program_id"],
            payload["certificate_id"],
            changed_nodes,
            affected_subtree,
            program,
        )
        if payload["transition_id"] != rebuilt.transition_id:
            raise ValueError("counterfactual transition ID differs from content")
        return rebuilt


def execute_counterfactual_transition(
    transition: CounterfactualTransition,
    grid: Grid,
    *,
    parent_execution: StatefulSceneExecution | None,
) -> StatefulSceneExecution:
    if parent_execution is None:
        raise ValueError("counterfactual transition requires its parent trace")
    if object_code_program_id(parent_execution.program) != transition.parent_program_id:
        raise ValueError("counterfactual transition parent trace differs")
    return execute_stateful_scene_transition(
        transition.program,
        grid,
        prior=parent_execution,
        changed_nodes=transition.changed_nodes,
    )


@dataclass(frozen=True, slots=True)
class CounterfactualTransitionCandidate:
    transition: CounterfactualTransition
    certificate: StatefulNodeFailureCertificate
    demo_outputs: tuple[Grid, ...]
    query_outputs: tuple[Grid, ...]
    parent_trace_ids: tuple[str, ...]

    @property
    def output_bundle_id(self) -> str:
        return _bundle_id(self.query_outputs)


@dataclass(frozen=True, slots=True)
class CounterfactualTransitionArmResult:
    strategy: str
    candidates: tuple[CounterfactualTransitionCandidate, ...]
    program_trials: int
    padding_trials: int
    demo_node_executions: tuple[tuple[str, int], ...]
    query_node_executions: tuple[tuple[str, int], ...]
    reused_demo_nodes: tuple[tuple[str, int], ...]
    changed_node_sets: tuple[tuple[str, int], ...]
    improving_trial_count: int


@dataclass(frozen=True, slots=True)
class CounterfactualTransitionSynthesisResult:
    certificates: tuple[StatefulNodeFailureCertificate, ...]
    arms: tuple[CounterfactualTransitionArmResult, ...]
    first_stage_program_trials: int
    parent_count: int


@dataclass(frozen=True, slots=True)
class _ParentContext:
    program: ScenePipelineProgram
    certificate: StatefulNodeFailureCertificate
    demo_executions: tuple[StatefulSceneExecution, ...]
    query_executions: tuple[StatefulSceneExecution, ...]


@dataclass(frozen=True, slots=True)
class _TrialEvaluation:
    transition: CounterfactualTransition
    demo_executions: tuple[StatefulSceneExecution, ...]
    score: tuple[object, ...]
    exact: bool
    improves_parent: bool


@dataclass(frozen=True, slots=True)
class _ProgramDelta:
    program: ScenePipelineProgram
    changed_nodes: tuple[str, ...]
    node_payloads: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _GrammarProgram:
    program: ScenePipelineProgram
    node_payloads: Mapping[str, object]


def _program_deltas(
    parent: ScenePipelineProgram,
    grammar: Sequence[_GrammarProgram],
) -> tuple[_ProgramDelta, ...]:
    parent_nodes = scene_program_node_payloads(parent)
    deltas = []
    for item in grammar:
        child_nodes = item.node_payloads
        changed_nodes = tuple(
            node_id
            for node_id in STATEFUL_NODE_ORDER
            if parent_nodes[node_id] != child_nodes[node_id]
        )
        if 1 <= len(changed_nodes) <= 3:
            deltas.append(_ProgramDelta(item.program, changed_nodes, child_nodes))
    return tuple(deltas)


def _stateful_executions(
    program: ScenePipelineProgram,
    inputs: Sequence[Grid],
) -> tuple[StatefulSceneExecution, ...] | None:
    executions = tuple(execute_stateful_scene_pipeline(program, grid) for grid in inputs)
    if any(not execution.ok for execution in executions):
        return None
    return executions


def _residual_score(
    executions: Sequence[StatefulSceneExecution],
    task: BlindTask,
    program: ScenePipelineProgram,
) -> tuple[object, ...]:
    residuals = tuple(
        compare_grids(
            execution.output,
            pair.output,
            pair_index=index,
            invalid_code=execution.reason,
        )
        for index, (execution, pair) in enumerate(
            zip(executions, task.train, strict=True)
        )
    )
    return (
        -sum(item.exact for item in residuals),
        -sum(item.shape_match for item in residuals),
        sum(item.mismatch_count for item in residuals),
        sum(not execution.ok for execution in executions),
        program.description_bits,
        object_code_program_id(program),
    )


def _parent_residual_score(
    context: _ParentContext,
    task: BlindTask,
) -> tuple[object, ...]:
    return _residual_score(context.demo_executions, task, context.program)


def _evaluate_program(
    *,
    strategy: str,
    program: ScenePipelineProgram,
    context: _ParentContext,
    task: BlindTask,
) -> _TrialEvaluation:
    transition = CounterfactualTransition.create(
        strategy=strategy,
        parent=context.program,
        certificate=context.certificate,
        program=program,
    )
    executions = tuple(
        execute_counterfactual_transition(
            transition,
            pair.input,
            parent_execution=parent_execution,
        )
        for pair, parent_execution in zip(
            task.train, context.demo_executions, strict=True
        )
    )
    score = _residual_score(executions, task, program)
    exact = all(
        execution.ok and execution.output == pair.output
        for execution, pair in zip(executions, task.train, strict=True)
    )
    return _TrialEvaluation(
        transition,
        executions,
        score,
        exact,
        score[:4] < _parent_residual_score(context, task)[:4],
    )


def _verify_transition_execution(
    transition: CounterfactualTransition,
    grid: Grid,
    parent_execution: StatefulSceneExecution,
    replayed: StatefulSceneExecution,
) -> None:
    round_trip = CounterfactualTransition.from_json_dict(
        transition.to_json_dict()
    )
    if round_trip != transition:
        raise AssertionError("counterfactual transition round-trip differs")
    fresh = execute_stateful_scene_pipeline(transition.program, grid)
    legacy = execute_scene_pipeline(transition.program, grid)
    if not replayed.ok or replayed.output is None:
        raise AssertionError("counterfactual transition became invalid")
    if replayed.output != fresh.output or replayed.output != legacy.output:
        raise AssertionError("counterfactual suffix replay differs from full replay")
    first = transition.changed_nodes[0]
    expected_reuse = STATEFUL_NODE_ORDER[: STATEFUL_NODE_ORDER.index(first)]
    if replayed.reused_node_ids != expected_reuse:
        raise AssertionError("counterfactual transition reused the wrong prefix")
    if replayed.executed_node_ids != affected_scene_subtree(first):
        raise AssertionError("counterfactual transition executed the wrong suffix")
    if parent_execution.input_grid_id != replayed.input_grid_id:
        raise AssertionError("counterfactual transition changed input identity")


def _emit_candidate(
    evaluation: _TrialEvaluation,
    *,
    context: _ParentContext,
    task: BlindTask,
) -> tuple[CounterfactualTransitionCandidate | None, Counter[str]]:
    query_nodes: Counter[str] = Counter()
    if not evaluation.exact:
        return None, query_nodes
    query_executions = tuple(
        execute_counterfactual_transition(
            evaluation.transition,
            grid,
            parent_execution=parent_execution,
        )
        for grid, parent_execution in zip(
            task.test_inputs, context.query_executions, strict=True
        )
    )
    for execution in query_executions:
        query_nodes.update(execution.executed_node_ids)
    if any(not execution.ok or execution.output is None for execution in query_executions):
        return None, query_nodes
    for grid, parent_execution, replayed in zip(
        tuple(pair.input for pair in task.train) + task.test_inputs,
        context.demo_executions + context.query_executions,
        evaluation.demo_executions + query_executions,
        strict=True,
    ):
        _verify_transition_execution(
            evaluation.transition,
            grid,
            parent_execution,
            replayed,
        )
    demo_outputs = tuple(
        execution.output
        for execution in evaluation.demo_executions
        if execution.output is not None
    )
    query_outputs = tuple(
        execution.output for execution in query_executions if execution.output is not None
    )
    return (
        CounterfactualTransitionCandidate(
            evaluation.transition,
            context.certificate,
            demo_outputs,
            query_outputs,
            tuple(
                execution.execution_id
                for execution in context.demo_executions + context.query_executions
            ),
        ),
        query_nodes,
    )


def _distance_tiered_programs(
    deltas: Sequence[_ProgramDelta],
    *,
    max_trials: int,
) -> tuple[ScenePipelineProgram, ...]:
    groups = tuple(
        tuple(
            delta.program
            for delta in deltas
            if len(delta.changed_nodes) == distance
        )
        for distance in (1, 2, 3)
    )
    return _round_robin_programs(
        groups,
        max_trials=max_trials,
        schedule=(0, 1, 1, 2),
    )


def _semantic_programs(
    certificate: StatefulNodeFailureCertificate,
    deltas: Sequence[_ProgramDelta],
    *,
    max_trials: int,
) -> tuple[ScenePipelineProgram, ...]:
    bundles = _semantic_bundles(certificate.node_id)
    groups = tuple(
        tuple(
            delta.program
            for delta in deltas
            if delta.changed_nodes == changed_nodes
        )
        for changed_nodes in bundles
    )
    return _round_robin_programs(groups, max_trials=max_trials)


def _extends_partial(
    *,
    parent: ScenePipelineProgram,
    partial: ScenePipelineProgram,
    candidate: _ProgramDelta,
) -> bool:
    partial_nodes = scene_program_node_payloads(partial)
    partial_differences = scene_program_node_differences(parent, partial)
    return all(
        candidate.node_payloads[node_id] == partial_nodes[node_id]
        for node_id in partial_differences
    )


def _run_parent_arm(
    *,
    strategy: str,
    context: _ParentContext,
    task: BlindTask,
    deltas: Sequence[_ProgramDelta],
    max_trials: int,
) -> tuple[
    tuple[CounterfactualTransitionCandidate, ...],
    int,
    Counter[str],
    Counter[str],
    Counter[str],
    Counter[str],
    int,
]:
    if strategy == "distance_tiered":
        programs = _distance_tiered_programs(
            deltas, max_trials=max_trials
        )
        phase_one_count = len(programs)
    elif strategy == "typed_semantic_bundles":
        programs = _semantic_programs(
            context.certificate,
            deltas,
            max_trials=max_trials,
        )
        phase_one_count = len(programs)
    elif strategy == "residual_beam":
        first_budget = max(1, max_trials // 4)
        node_groups = tuple(
            tuple(
                delta.program
                for delta in deltas
                if delta.changed_nodes == (node_id,)
            )
            for node_id in STATEFUL_NODE_ORDER
        )
        diagnosed_index = STATEFUL_NODE_ORDER.index(context.certificate.node_id)
        schedule = (diagnosed_index,) + tuple(
            index for index in range(len(STATEFUL_NODE_ORDER)) if index != diagnosed_index
        )
        phase_one = _round_robin_programs(
            node_groups,
            max_trials=first_budget,
            schedule=schedule,
        )
        phase_one_evaluations = tuple(
            _evaluate_program(
                strategy=strategy,
                program=program,
                context=context,
                task=task,
            )
            for program in phase_one
        )
        beam = tuple(
            evaluation.transition.program
            for evaluation in sorted(
                phase_one_evaluations,
                key=lambda evaluation: evaluation.score,
            )[:8]
        )
        legal_bundles = frozenset(_semantic_bundles(context.certificate.node_id))
        extension_groups = tuple(
            tuple(
                candidate.program
                for candidate in deltas
                if candidate.changed_nodes in legal_bundles
                and len(candidate.changed_nodes) > 1
                and _extends_partial(
                    parent=context.program,
                    partial=partial,
                    candidate=candidate,
                )
            )
            for partial in beam
        )
        extensions = _round_robin_programs(
            extension_groups,
            max_trials=max_trials - len(phase_one),
        )
        programs = phase_one + extensions
        phase_one_count = len(phase_one)
    else:
        raise ValueError("unknown counterfactual transition strategy")

    if strategy == "residual_beam":
        evaluations = list(phase_one_evaluations)
        evaluations.extend(
            _evaluate_program(
                strategy=strategy,
                program=program,
                context=context,
                task=task,
            )
            for program in programs[phase_one_count:]
        )
    else:
        evaluations = [
            _evaluate_program(
                strategy=strategy,
                program=program,
                context=context,
                task=task,
            )
            for program in programs
        ]

    candidates = []
    demo_nodes: Counter[str] = Counter()
    query_nodes: Counter[str] = Counter()
    reused_demo: Counter[str] = Counter()
    changed_sets: Counter[str] = Counter()
    improving = 0
    for evaluation in evaluations:
        improving += int(evaluation.improves_parent)
        changed_sets["+".join(evaluation.transition.changed_nodes)] += 1
        for execution in evaluation.demo_executions:
            demo_nodes.update(execution.executed_node_ids)
            reused_demo.update(execution.reused_node_ids)
        candidate, emitted_query_nodes = _emit_candidate(
            evaluation,
            context=context,
            task=task,
        )
        query_nodes.update(emitted_query_nodes)
        if candidate is not None:
            candidates.append(candidate)
    return (
        tuple(candidates),
        len(evaluations),
        demo_nodes,
        query_nodes,
        reused_demo,
        changed_sets,
        improving,
    )


def synthesize_counterfactual_transition_arms(
    task: BlindTask,
    *,
    strategies: Sequence[str] = COUNTERFACTUAL_STRATEGIES,
    max_first_stage_trials: int = 2_000,
    max_parents: int = 4,
    max_transition_trials: int = 512,
    max_candidates: int = 32,
    first_stage_programs: Sequence[ObjectCodeProgram] | None = None,
    transition_programs: Sequence[ScenePipelineProgram] | None = None,
) -> CounterfactualTransitionSynthesisResult:
    """Run frozen multi-node arms over the same query-blind parent contexts."""

    if not isinstance(task, BlindTask):
        raise TypeError("counterfactual transition synthesis accepts BlindTask only")
    selected_strategies = tuple(strategies)
    if not selected_strategies or len(set(selected_strategies)) != len(
        selected_strategies
    ):
        raise ValueError("counterfactual transition strategies must be unique")
    if any(strategy not in COUNTERFACTUAL_STRATEGIES for strategy in selected_strategies):
        raise ValueError("counterfactual transition strategy differs")
    for name, value in (
        ("max_first_stage_trials", max_first_stage_trials),
        ("max_parents", max_parents),
        ("max_transition_trials", max_transition_trials),
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
    contexts = []
    for score in parent_scores:
        if len(contexts) >= max_parents:
            break
        parent = score.program
        if not isinstance(parent, ScenePipelineProgram):
            raise AssertionError("filtered transition parent lost its scene type")
        demo = _stateful_executions(parent, tuple(pair.input for pair in task.train))
        query = _stateful_executions(parent, task.test_inputs)
        if demo is None or query is None:
            continue
        contexts.append(
            _ParentContext(
                parent,
                compile_stateful_node_failure(task, parent, demo),
                demo,
                query,
            )
        )

    grammar_programs = tuple(
        enumerate_scene_pipeline_programs(task)
        if transition_programs is None
        else transition_programs
    )
    grammar = tuple(
        _GrammarProgram(program, scene_program_node_payloads(program))
        for program in grammar_programs
    )
    reserved_trials = max_parents * max_transition_trials
    deltas_by_parent = {
        object_code_program_id(context.program): _program_deltas(
            context.program, grammar
        )
        for context in contexts
    }
    arm_results = []
    for strategy in selected_strategies:
        candidates_by_transition: dict[str, CounterfactualTransitionCandidate] = {}
        program_trials = 0
        demo_nodes: Counter[str] = Counter()
        query_nodes: Counter[str] = Counter()
        reused_demo: Counter[str] = Counter()
        changed_sets: Counter[str] = Counter()
        improving = 0
        for context in contexts:
            (
                candidates,
                parent_trials,
                parent_demo_nodes,
                parent_query_nodes,
                parent_reused_demo,
                parent_changed_sets,
                parent_improving,
            ) = _run_parent_arm(
                strategy=strategy,
                context=context,
                task=task,
                deltas=deltas_by_parent[object_code_program_id(context.program)],
                max_trials=max_transition_trials,
            )
            program_trials += parent_trials
            demo_nodes.update(parent_demo_nodes)
            query_nodes.update(parent_query_nodes)
            reused_demo.update(parent_reused_demo)
            changed_sets.update(parent_changed_sets)
            improving += parent_improving
            for candidate in candidates:
                candidates_by_transition[candidate.transition.transition_id] = candidate
        ordered = sorted(
            candidates_by_transition.values(),
            key=lambda candidate: (
                candidate.transition.program.description_bits,
                candidate.transition.transition_id,
            ),
        )[:max_candidates]
        seen_outputs: set[str] = set()
        deduplicated = []
        for candidate in ordered:
            if candidate.output_bundle_id in seen_outputs:
                continue
            seen_outputs.add(candidate.output_bundle_id)
            deduplicated.append(candidate)
        arm_results.append(
            CounterfactualTransitionArmResult(
                strategy,
                tuple(deduplicated),
                program_trials,
                reserved_trials - program_trials,
                tuple(sorted(demo_nodes.items())),
                tuple(sorted(query_nodes.items())),
                tuple(sorted(reused_demo.items())),
                tuple(sorted(changed_sets.items())),
                improving,
            )
        )
    return CounterfactualTransitionSynthesisResult(
        tuple(context.certificate for context in contexts),
        tuple(arm_results),
        first.program_trial_count,
        len(contexts),
    )
