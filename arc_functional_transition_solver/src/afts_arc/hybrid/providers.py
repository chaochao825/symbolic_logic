"""Candidate providers for the functional ARC router.

No provider in this module starts training or calls a remote model implicitly.
LLM, masked-diffusion, and DiffLogic sources are explicit callbacks and otherwise
produce auditable abstention receipts.
"""

from __future__ import annotations

import importlib
import hashlib
import math
import re
from pathlib import Path
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from ..blind import BlindTask
from ..dsl import Instruction, Program, execute_program, primitive_registry
from ..grid import Grid
from ..search import (
    ProgramEvaluation,
    SearchConfig,
    evaluate_program,
    instruction_proposals,
    search_programs,
)
from .router import RouteDecision, TaskFeatures
from .types import CandidateHypothesis, ProviderResult, canonical_json

if TYPE_CHECKING:
    from .control import Blackboard, ControlAction


DSL_PROVIDER_VERSION = "afts-hybrid-dsl/v1"
CA_PROVIDER_VERSION = "afts-hybrid-ca/v1"
OPTIONAL_PROVIDER_VERSION = "afts-hybrid-external/v1"
DIFFLOGIC_GATE16_NAMES = (
    "FALSE",
    "AND",
    "A_AND_NOT_B",
    "A",
    "NOT_A_AND_B",
    "B",
    "XOR",
    "OR",
    "NOR",
    "XNOR",
    "NOT_B",
    "A_OR_NOT_B",
    "NOT_A",
    "NOT_A_OR_B",
    "NAND",
    "TRUE",
)


def _program_description_bits(program: Program) -> int:
    """Charge the typed grammar, not its verbose audit JSON serialization."""

    def elias_delta(value: int) -> int:
        width = value.bit_length()
        return width + 2 * (width.bit_length() - 1)

    def value_bits(value: object) -> int:
        if value is None:
            return 1
        if isinstance(value, bool):
            return 1
        if isinstance(value, int):
            return 1 + elias_delta(abs(value) + 1)
        if isinstance(value, str):
            encoded = value.encode("utf-8")
            return elias_delta(len(encoded) + 1) + 8 * len(encoded)
        if isinstance(value, (tuple, list)):
            return elias_delta(len(value) + 1) + sum(value_bits(item) for item in value)
        if isinstance(value, dict):
            return elias_delta(len(value) + 1) + sum(
                value_bits(key) + value_bits(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            )
        raise TypeError(f"unsupported DSL argument type for MDL: {type(value).__name__}")

    op_bits = max(1, math.ceil(math.log2(len(primitive_registry()))))
    return elias_delta(program.node_count) + sum(
        op_bits + sum(value_bits(value) for _, value in sorted(instruction.arguments.items()))
        for instruction in program.instructions
    )


def _dsl_hypothesis(
    program: Program,
    *,
    parent_hypothesis_id: str | None = None,
    control_operator: str | None = None,
    control_metadata: dict[str, object] | None = None,
) -> CandidateHypothesis:
    serialized = program.to_json_dict()

    def replay(grid: Grid) -> object | None:
        outcome = execute_program(program, grid)
        return outcome.output if outcome.ok else None

    def hard_verify(task: BlindTask) -> bool:
        reparsed = Program.from_json_dict(serialized)
        grids = tuple(pair.input for pair in task.train) + task.test_inputs
        return all(execute_program(reparsed, grid).ok for grid in grids)

    metadata: dict[str, object] = {"node_count": program.node_count}
    if control_operator is not None:
        metadata["control_operator"] = control_operator
    metadata.update(control_metadata or {})
    return CandidateHypothesis.create(
        name=f"dsl:{program.program_id}",
        source="typed_dsl",
        source_version=DSL_PROVIDER_VERSION,
        route="dsl_program",
        description_bits=_program_description_bits(program),
        verification_mode="replayable",
        functional_trace=program.functional_trace,
        spec={"program": serialized},
        metadata=metadata,
        parent_hypothesis_ids=(
            () if parent_hypothesis_id is None else (parent_hypothesis_id,)
        ),
        replay=replay,
        hard_verifier=hard_verify,
    )


@dataclass(slots=True)
class DslProgramProvider:
    search_config: SearchConfig = SearchConfig()
    include_repair_seeds: bool = True
    name: str = "typed_dsl"
    route: str = "dsl_program"
    strict_budget_contract: bool = field(default=False, init=False)
    supports_residual_actions: bool = field(default=True, init=False)
    supports_repeated_batches: bool = field(default=False, init=False)
    max_control_calls: int = field(default=3, init=False)
    parent_sensitive_operators: frozenset[str] = field(
        default=frozenset({"shape_resynthesize", "suffix_resynthesize"}),
        init=False,
    )

    @staticmethod
    def _semantically_diverse(
        evaluations: Sequence[ProgramEvaluation],
        *,
        limit: int,
        exact_lane: bool,
    ) -> tuple[tuple[ProgramEvaluation, ...], int]:
        """Keep the best program for each observable demo/query behavior."""

        if limit < 1:
            return (), 0
        if exact_lane:
            ranked = sorted(
                evaluations,
                key=lambda item: (
                    _program_description_bits(item.program),
                    item.program.program_id,
                ),
            )
        else:
            ranked = sorted(
                evaluations,
                key=lambda item: (
                    -item.exact_demo_count,
                    -item.shape_match_count,
                    -item.agreement,
                    _program_description_bits(item.program),
                    item.program.program_id,
                ),
            )
        selected: list[ProgramEvaluation] = []
        seen_semantics: set[tuple[object, ...]] = set()
        duplicate_count = 0
        for evaluation in ranked:
            signature = evaluation.semantic_signature()
            if signature in seen_semantics:
                duplicate_count += 1
                continue
            seen_semantics.add(signature)
            selected.append(evaluation)
            if len(selected) >= limit:
                break
        return tuple(selected), duplicate_count

    def propose(
        self,
        task: BlindTask,
        features: TaskFeatures,
        decision: RouteDecision,
    ) -> ProviderResult:
        if not isinstance(task, BlindTask):
            raise TypeError("DSL provider accepts BlindTask only")
        seed_evaluations: list[ProgramEvaluation] = []
        if self.include_repair_seeds:
            for op in (
                "identity",
                "rotate90",
                "rotate180",
                "rotate270",
                "flip_horizontal",
                "flip_vertical",
                "transpose",
                "anti_transpose",
            ):
                seed_evaluations.append(
                    evaluate_program(Program.create((Instruction.create(op),)), task)
                )
        result = search_programs(task, config=self.search_config)
        exact_by_program = {
            evaluation.program.program_id: evaluation
            for evaluation in (*result.exact_evaluations, *seed_evaluations)
            if evaluation.all_demo_exact
        }
        near_by_program = {
            evaluation.program.program_id: evaluation
            for evaluation in seed_evaluations
            if not evaluation.all_demo_exact
        }
        candidate_limit = decision.budget_for(self.route)
        exact, exact_semantic_duplicates = self._semantically_diverse(
            tuple(exact_by_program.values()),
            limit=candidate_limit,
            exact_lane=True,
        )
        near, near_semantic_duplicates = self._semantically_diverse(
            tuple(near_by_program.values()),
            limit=max(0, candidate_limit - len(exact)),
            exact_lane=False,
        )
        ordered = tuple(
            _dsl_hypothesis(
                evaluation.program,
                control_metadata={
                    "demo_exact": evaluation.all_demo_exact,
                    "emission_lane": (
                        "demo_exact" if evaluation.all_demo_exact else "near_miss_seed"
                    ),
                    "demo_exact_count": evaluation.exact_demo_count,
                    "demo_shape_match_count": evaluation.shape_match_count,
                    "demo_agreement": evaluation.agreement,
                },
            )
            for evaluation in (*exact, *near)
        )
        return ProviderResult.ok(
            self.name,
            self.route,
            ordered,
            {
                "exact_program_count": len(result.exact_evaluations),
                "expansions": result.expansions,
                "semantic_duplicates": result.semantic_duplicates,
                "repair_seed_count": int(self.include_repair_seeds) * 8,
                "repair_seed_demo_exact_count": sum(
                    item.all_demo_exact for item in seed_evaluations
                ),
                "emission_policy": "demo_exact_first_semantic_diversity_v2",
                "emitted_demo_exact_count": len(exact),
                "emitted_near_miss_count": len(near),
                "emission_semantic_duplicates_removed": (
                    exact_semantic_duplicates + near_semantic_duplicates
                ),
                "search_config": {
                    "max_depth": self.search_config.max_depth,
                    "beam_width": self.search_config.beam_width,
                    "max_instruction_options": self.search_config.max_instruction_options,
                    "max_exact_programs": self.search_config.max_exact_programs,
                },
            },
        )

    @staticmethod
    def _parent_candidate(
        blackboard: "Blackboard", parent_hypothesis_id: str | None
    ) -> CandidateHypothesis | None:
        if parent_hypothesis_id is None:
            return None
        return next(
            (
                item
                for item in blackboard.candidates
                if item.hypothesis_id == parent_hypothesis_id
            ),
            None,
        )

    @staticmethod
    def _parent_program(parent: CandidateHypothesis) -> Program | None:
        raw = parent.spec.get("program")
        if not isinstance(raw, dict):
            return None
        try:
            return Program.from_json_dict(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parent_score(
        blackboard: "Blackboard", parent_hypothesis_id: str
    ) -> tuple[int, int, float]:
        evaluation = next(
            item
            for item in blackboard.evaluations
            if item.hypothesis.hypothesis_id == parent_hypothesis_id
        )
        return (
            sum(item.exact for item in evaluation.residuals),
            sum(item.shape_match for item in evaluation.residuals),
            evaluation.agreement,
        )

    def _residual_programs(
        self,
        task: BlindTask,
        blackboard: "Blackboard",
        action: "ControlAction",
    ) -> ProviderResult:
        parent = self._parent_candidate(blackboard, action.parent_hypothesis_id)
        if parent is None or action.parent_hypothesis_id is None:
            return ProviderResult.abstained(
                self.name,
                self.route,
                "residual_parent_unavailable",
                {"action_operator": action.operator},
            )
        parent_program = self._parent_program(parent)
        options = instruction_proposals(task)[: self.search_config.max_instruction_options]
        programs: dict[str, Program] = {}
        if action.operator == "shape_resynthesize":
            for instruction in options:
                program = Program.create((instruction,))
                programs[program.program_id] = program
                if parent_program is not None and parent_program.node_count < 3:
                    appended = Program.create(
                        (*parent_program.instructions, instruction)
                    )
                    programs[appended.program_id] = appended
        elif action.operator == "suffix_resynthesize":
            for instruction in options:
                if instruction.op == "identity":
                    continue
                if parent_program is None:
                    program = Program.create((instruction,))
                    programs[program.program_id] = program
                    continue
                if parent_program.node_count < 3:
                    appended = Program.create(
                        (*parent_program.instructions, instruction)
                    )
                    programs[appended.program_id] = appended
                replaced = Program.create(
                    (*parent_program.instructions[:-1], instruction)
                )
                programs[replaced.program_id] = replaced
        else:
            raise ValueError("unsupported residual DSL operator")

        parent_score = self._parent_score(blackboard, action.parent_hypothesis_id)
        evaluated = tuple(evaluate_program(program, task) for program in programs.values())

        def score(item: object) -> tuple[int, int, float, int, str]:
            return (
                item.exact_demo_count,
                item.shape_match_count,
                item.agreement,
                -item.program.node_count,
                item.program.program_id,
            )

        improved = []
        for evaluation in evaluated:
            candidate_score = (
                evaluation.exact_demo_count,
                evaluation.shape_match_count,
                evaluation.agreement,
            )
            if action.operator == "shape_resynthesize":
                keep = (
                    evaluation.shape_match_count > parent_score[1]
                    or evaluation.all_demo_exact
                )
            else:
                keep = candidate_score > parent_score
            if keep:
                improved.append(evaluation)
        ranked = tuple(sorted(improved, key=score, reverse=True))
        selected = ranked[: action.budget.candidate_slots]
        candidates = tuple(
            _dsl_hypothesis(
                evaluation.program,
                parent_hypothesis_id=action.parent_hypothesis_id,
                control_operator=action.operator,
                control_metadata={
                    "improved_over_parent": True,
                    "evidence_signal_ids": list(action.evidence_signal_ids),
                },
            )
            for evaluation in selected
        )
        diagnostics = {
            "action_operator": action.operator,
            "parent_hypothesis_id": action.parent_hypothesis_id,
            "parent_representation": (
                "dsl_program" if parent_program is not None else parent.route
            ),
            "instruction_option_count": len(options),
            "program_trials": len(evaluated),
            "improving_program_count": len(improved),
            "native_cost": {
                "program_trials": len(evaluated),
                "demo_program_executions": len(evaluated) * len(task.train),
                "query_program_executions": len(evaluated) * len(task.test_inputs),
            },
        }
        if not candidates:
            return ProviderResult.abstained(
                self.name,
                self.route,
                "no_residual_improving_program",
                diagnostics,
            )
        return ProviderResult.ok(self.name, self.route, candidates, diagnostics)

    def act(
        self,
        task: BlindTask,
        features: TaskFeatures,
        decision: RouteDecision,
        blackboard: "Blackboard",
        action: "ControlAction",
    ) -> ProviderResult:
        if (
            action.kind != "propose"
            or action.actor != self.name
            or action.route != self.route
        ):
            raise ValueError("DSL provider received an action for another provider")
        if action.operator == "synthesize":
            raw = self.propose(task, features, decision)
            if raw.status != "ok":
                return raw
            selected = raw.candidates[: action.budget.candidate_slots]
            diagnostics = {
                **raw.diagnostics,
                "action_operator": action.operator,
                "candidate_slot_limit": action.budget.candidate_slots,
                "native_cost": {
                    "program_expansions": raw.diagnostics.get("expansions", 0),
                },
            }
            return ProviderResult.ok(self.name, self.route, selected, diagnostics)
        return self._residual_programs(task, blackboard, action)


def _load_ca_backend() -> dict[str, Any]:
    """Load inference-only root modules; never import arc_difflogic_train."""

    names = ("numpy", "arc_data", "arc_ca", "arc_ca_programs", "arc_difflogic_features")
    return {name: importlib.import_module(name) for name in names}


def _python_grid(array: object) -> list[list[int]]:
    values = array.tolist() if hasattr(array, "tolist") else array
    return [[int(cell) for cell in row] for row in values]


def _condition_candidate_on_action(
    candidate: CandidateHypothesis,
    action: "ControlAction",
) -> CandidateHypothesis:
    parents = tuple(
        dict.fromkeys(
            (
                *candidate.parent_hypothesis_ids,
                *((action.parent_hypothesis_id,) if action.parent_hypothesis_id else ()),
            )
        )
    )
    return CandidateHypothesis.create(
        name=candidate.name,
        source=candidate.source,
        source_version=candidate.source_version,
        route=candidate.route,
        description_bits=candidate.description_bits,
        verification_mode=candidate.verification_mode,
        functional_trace=(*candidate.functional_trace, f"control:{action.operator}"),
        spec=candidate.spec,
        metadata={
            **candidate.metadata,
            "control_operator": action.operator,
            "evidence_signal_ids": list(action.evidence_signal_ids),
        },
        parent_hypothesis_ids=parents,
        replay=candidate.replay,
        hard_verifier=candidate.hard_verifier,
        explicit_demo_outputs=candidate.explicit_demo_outputs,
        explicit_query_outputs=candidate.explicit_query_outputs,
    )


@dataclass(slots=True)
class SparseCAProvider:
    policies: tuple[str, ...] = ("none", "d4", "bgpad", "d4_bgpad")
    max_rules_per_policy: int = 2
    max_programs: int = 4
    minimum_query_support: float = 0.5
    minimum_foreground_support: float = 0.5
    name: str = "sparse_ca_d4_bgpad"
    route: str = "sparse_ca"
    strict_budget_contract: bool = field(default=False, init=False)
    supports_residual_actions: bool = field(default=True, init=False)
    supports_repeated_batches: bool = field(default=False, init=False)
    max_control_calls: int = field(default=2, init=False)
    parent_sensitive_operators: frozenset[str] = field(
        default=frozenset(), init=False
    )

    def __post_init__(self) -> None:
        allowed = {"none", "d4", "bgpad", "d4_bgpad"}
        if not self.policies or any(item not in allowed for item in self.policies):
            raise ValueError("unknown sparse CA policy")
        if self.max_rules_per_policy < 1 or self.max_programs < 0:
            raise ValueError("CA candidate budgets must be non-negative and non-empty")
        for name in ("minimum_query_support", "minimum_foreground_support"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def propose(
        self,
        task: BlindTask,
        features: TaskFeatures,
        decision: RouteDecision,
    ) -> ProviderResult:
        if not isinstance(task, BlindTask):
            raise TypeError("CA provider accepts BlindTask only")
        if not features.all_same_shape:
            return ProviderResult.abstained(
                self.name,
                self.route,
                "unsupported_shape_change",
                {"policies": list(self.policies)},
            )
        try:
            backend = _load_ca_backend()
        except (ImportError, ModuleNotFoundError) as exc:
            return ProviderResult.abstained(
                self.name,
                self.route,
                "backend_unavailable",
                {"error_type": type(exc).__name__},
            )
        np = backend["numpy"]
        ArcExample = backend["arc_data"].ArcExample
        select_demo_rule = backend["arc_ca"].select_demo_rule
        predict_direct = backend["arc_ca"].predict_direct
        predict_binary4 = backend["arc_ca"].predict_binary4
        predict_onehot = backend["arc_ca"].predict_onehot
        extract_neighborhoods = backend["arc_ca"].extract_neighborhoods
        NeighborhoodSpec = backend["arc_ca"].NeighborhoodSpec
        fit_sparse_rule = backend["arc_ca"].fit_sparse_rule
        compiled_gate_upper_bounds = backend["arc_ca"].compiled_gate_upper_bounds
        select_demo_program = backend["arc_ca_programs"].select_demo_program
        LocalCAProgram = backend["arc_ca_programs"].LocalCAProgram
        augment_examples = backend["arc_difflogic_features"].augment_examples
        modal_color = backend["arc_difflogic_features"].modal_color
        backend_source = Path(backend["arc_ca"].__file__).resolve()
        backend_sha256 = hashlib.sha256(backend_source.read_bytes()).hexdigest()
        program_backend_source = Path(backend["arc_ca_programs"].__file__).resolve()
        program_backend_sha256 = hashlib.sha256(program_backend_source.read_bytes()).hexdigest()

        examples = tuple(
            ArcExample(
                np.asarray(pair.input, dtype=np.uint8),
                np.asarray(pair.output, dtype=np.uint8),
            )
            for pair in task.train
        )
        hypotheses: dict[str, CandidateHypothesis] = {}
        policy_status: dict[str, str] = {}
        # Four deployment policies require a valid two-bit prefix code.
        policy_bits = {"none": 2, "d4": 2, "bgpad": 2, "d4_bgpad": 2}

        for policy in self.policies:
            selected = select_demo_rule(augment_examples(examples, policy))
            if selected.rule is None:
                policy_status[policy] = selected.status
                continue
            emitted = 0
            for rule in selected.ranked_rules[: self.max_rules_per_policy]:
                def prepare(grid: object, selected_policy: str = policy) -> object:
                    array = np.asarray(grid, dtype=np.uint8)
                    if selected_policy in ("bgpad", "d4_bgpad"):
                        array = np.pad(array, 1, constant_values=modal_color(array))
                    return array

                def deploy(
                    grid: object,
                    selected_rule: object = rule,
                    selected_policy: str = policy,
                ) -> object:
                    prepared = prepare(grid, selected_policy)
                    output = predict_direct(selected_rule, prepared)
                    return output[1:-1, 1:-1] if selected_policy in ("bgpad", "d4_bgpad") else output

                if not all(
                    np.array_equal(deploy(example.input_grid), example.output_grid)
                    for example in examples
                ):
                    continue
                entries = [
                    {"pattern": [int(value) for value in pattern], "output": int(output)}
                    for pattern, output in rule.entries
                ]
                seen_patterns = [
                    [int(value) for value in pattern]
                    for pattern in sorted(rule.seen_patterns)
                ]
                rule_record = {
                    "policy": policy,
                    "neighborhood": rule.spec.name,
                    "offsets": [list(offset) for offset in rule.spec.offsets],
                    "entries": entries,
                    "seen_pattern_count": len(seen_patterns),
                    "seen_patterns_sha256": hashlib.sha256(
                        canonical_json(seen_patterns).encode("ascii")
                    ).hexdigest(),
                    "model_description_bits": int(rule.model_description_bits),
                    "backend_source_sha256": backend_sha256,
                    "support_policy": {
                        "minimum_query_support": self.minimum_query_support,
                        "minimum_foreground_support": self.minimum_foreground_support,
                    },
                }

                def support_stats(selected_rule: object, grids: Sequence[object], selected_policy: str) -> dict[str, object]:
                    total_cells = 0
                    unsupported_cells = 0
                    foreground_cells = 0
                    unsupported_foreground_cells = 0
                    for grid in grids:
                        original = np.asarray(grid, dtype=np.uint8)
                        background = modal_color(original)
                        prepared = prepare(original, selected_policy)
                        patches = extract_neighborhoods(prepared, selected_rule.spec)
                        active = np.ones(prepared.shape, dtype=bool)
                        if selected_policy in ("bgpad", "d4_bgpad"):
                            active[:] = False
                            active[1:-1, 1:-1] = True
                        foreground = active & (prepared != background)
                        supported = np.zeros(prepared.shape, dtype=bool)
                        for row in range(prepared.shape[0]):
                            for column in range(prepared.shape[1]):
                                pattern = tuple(int(value) for value in patches[row, column])
                                supported[row, column] = pattern in selected_rule.seen_patterns
                        total_cells += int(np.sum(active))
                        unsupported_cells += int(np.sum(active & ~supported))
                        foreground_cells += int(np.sum(foreground))
                        unsupported_foreground_cells += int(np.sum(foreground & ~supported))
                    query_support = (
                        1.0 - unsupported_cells / total_cells if total_cells else 0.0
                    )
                    foreground_support = (
                        1.0 - unsupported_foreground_cells / foreground_cells
                        if foreground_cells
                        else 1.0
                    )
                    return {
                        "query_support": query_support,
                        "foreground_query_support": foreground_support,
                        "query_cell_count": total_cells,
                        "unsupported_query_cells": unsupported_cells,
                        "foreground_query_cell_count": foreground_cells,
                        "unsupported_foreground_query_cells": unsupported_foreground_cells,
                    }

                stats = support_stats(rule, task.test_inputs, policy)
                support_gate_passed = bool(
                    stats["query_support"] >= self.minimum_query_support
                    and stats["foreground_query_support"] >= self.minimum_foreground_support
                )

                def rebuild_rule(
                    blind: BlindTask,
                    record: dict[str, object] = rule_record,
                ) -> object:
                    spec = NeighborhoodSpec(
                        str(record["neighborhood"]),
                        tuple(tuple(int(value) for value in offset) for offset in record["offsets"]),
                    )
                    blind_examples = tuple(
                        ArcExample(
                            np.asarray(pair.input, dtype=np.uint8),
                            np.asarray(pair.output, dtype=np.uint8),
                        )
                        for pair in blind.train
                    )
                    fitted = fit_sparse_rule(
                        augment_examples(blind_examples, str(record["policy"])),
                        spec,
                    )
                    if fitted.rule is None:
                        raise ValueError("serialized sparse rule cannot be refit from demonstrations")
                    rebuilt = fitted.rule
                    rebuilt_entries = [
                        {"pattern": [int(value) for value in pattern], "output": int(output)}
                        for pattern, output in rebuilt.entries
                    ]
                    rebuilt_seen = [
                        [int(value) for value in pattern]
                        for pattern in sorted(rebuilt.seen_patterns)
                    ]
                    if rebuilt_entries != record["entries"]:
                        raise ValueError("serialized sparse entries differ from demonstration refit")
                    if len(rebuilt_seen) != record["seen_pattern_count"]:
                        raise ValueError("serialized sparse seen-pattern count differs from refit")
                    if hashlib.sha256(canonical_json(rebuilt_seen).encode("ascii")).hexdigest() != record["seen_patterns_sha256"]:
                        raise ValueError("serialized sparse seen-pattern digest differs from refit")
                    return rebuilt

                def hard_verify(
                    blind: BlindTask,
                    selected_rule: object = rule,
                    selected_policy: str = policy,
                    expected_stats: dict[str, object] = dict(stats),
                    expected_gate: bool = support_gate_passed,
                    record: dict[str, object] = dict(rule_record),
                ) -> bool:
                    try:
                        if hashlib.sha256(backend_source.read_bytes()).hexdigest() != record["backend_source_sha256"]:
                            return False
                        rebuilt = rebuild_rule(blind, record)
                        if rebuilt.model_description_bits != record["model_description_bits"]:
                            return False
                        for grid in tuple(pair.input for pair in blind.train) + blind.test_inputs:
                            prepared = prepare(grid, selected_policy)
                            original_direct = predict_direct(selected_rule, prepared)
                            direct = predict_direct(rebuilt, prepared)
                            binary, valid_binary = predict_binary4(rebuilt, prepared)
                            onehot, valid_onehot = predict_onehot(rebuilt, prepared)
                            if not (
                                bool(np.all(valid_binary))
                                and bool(np.all(valid_onehot))
                                and np.array_equal(original_direct, direct)
                                and np.array_equal(direct, binary)
                                and np.array_equal(direct, onehot)
                            ):
                                return False
                        observed_stats = support_stats(rebuilt, blind.test_inputs, selected_policy)
                        if observed_stats != expected_stats:
                            return False
                        support_policy = record["support_policy"]
                        observed_gate = bool(
                            observed_stats["query_support"]
                            >= support_policy["minimum_query_support"]
                            and observed_stats["foreground_query_support"]
                            >= support_policy["minimum_foreground_support"]
                        )
                        return observed_gate is expected_gate
                    except Exception:
                        return False

                candidate = CandidateHypothesis.create(
                    name=f"sparse_ca:{policy}:{rule.spec.name}",
                    source="sparse_categorical_ca",
                    source_version=CA_PROVIDER_VERSION,
                    route=self.route,
                    description_bits=int(rule.model_description_bits + policy_bits[policy]),
                    verification_mode="replayable",
                    functional_trace=("sparse_ca", f"augmentation:{policy}", rule.spec.name),
                    spec=rule_record,
                    metadata={
                        **stats,
                        "support_gate_passed": support_gate_passed,
                        "minimum_query_support": self.minimum_query_support,
                        "minimum_foreground_support": self.minimum_foreground_support,
                        "training_cells": int(rule.training_cells),
                        "gate_upper_bounds": compiled_gate_upper_bounds(rule),
                    },
                    replay=lambda grid, fn=deploy: _python_grid(fn(grid)),
                    hard_verifier=hard_verify,
                )
                hypotheses[candidate.hypothesis_id] = candidate
                emitted += 1
            policy_candidates = [
                item
                for item in hypotheses.values()
                if item.source == "sparse_categorical_ca" and item.spec.get("policy") == policy
            ]
            if any(item.metadata.get("support_gate_passed") is True for item in policy_candidates):
                policy_status[policy] = "selected"
            elif emitted:
                policy_status[policy] = "insufficient_query_support"
            else:
                policy_status[policy] = "rejected_original_demo_mismatch"

        program_selection = select_demo_program(examples)
        for program in program_selection.ranked_programs[: self.max_programs]:
            program_record = {
                "kind": program.kind,
                "params": list(program.params),
                "description_bits": int(program.description_bits),
                "onehot_gates_per_cell_upper": int(program.onehot_gates_per_cell_upper),
                "backend_source_sha256": program_backend_sha256,
            }

            def program_replay(grid: object, selected_program: object = program) -> object:
                return _python_grid(selected_program.run(grid)[0])

            def program_verify(
                blind: BlindTask,
                selected_program: object = program,
                record: dict[str, object] = dict(program_record),
            ) -> bool:
                try:
                    if hashlib.sha256(program_backend_source.read_bytes()).hexdigest() != record["backend_source_sha256"]:
                        return False
                    rebuilt = LocalCAProgram(
                        str(record["kind"]),
                        tuple(record["params"]),
                        int(record["description_bits"]),
                        int(record["onehot_gates_per_cell_upper"]),
                    )
                    for grid in tuple(pair.input for pair in blind.train) + blind.test_inputs:
                        first = selected_program.run(grid)[0]
                        second = rebuilt.run(grid)[0]
                        if not np.array_equal(first, second):
                            return False
                        if not bool(np.all((0 <= first) & (first <= 9))):
                            return False
                    return True
                except Exception:
                    return False

            candidate = CandidateHypothesis.create(
                name=f"bounded_ca_program:{program.name}",
                source="bounded_ca_program",
                source_version=CA_PROVIDER_VERSION,
                route=self.route,
                description_bits=int(program.description_bits),
                verification_mode="replayable",
                functional_trace=("bounded_ca_program", program.kind, program.name),
                spec=program_record,
                metadata={"query_support": 1.0},
                replay=program_replay,
                hard_verifier=program_verify,
            )
            hypotheses[candidate.hypothesis_id] = candidate

        ordered = tuple(
            sorted(
                hypotheses.values(),
                key=lambda item: (
                    item.source == "sparse_categorical_ca"
                    and item.metadata.get("support_gate_passed") is not True,
                    item.description_bits,
                    item.hypothesis_id,
                ),
            )[: decision.budget_for(self.route)]
        )
        return ProviderResult.ok(
            self.name,
            self.route,
            ordered,
            {
                "policy_status": policy_status,
                "bounded_program_status": program_selection.status,
                "bounded_program_exact_candidates": program_selection.exact_candidates,
            },
        )

    def act(
        self,
        task: BlindTask,
        features: TaskFeatures,
        decision: RouteDecision,
        blackboard: "Blackboard",
        action: "ControlAction",
    ) -> ProviderResult:
        if (
            action.kind != "propose"
            or action.actor != self.name
            or action.route != self.route
        ):
            raise ValueError("CA provider received an action for another provider")
        policy_family = {
            "local_transition_search": ("none", "bgpad"),
            "d4_bgpad_search": ("d4", "d4_bgpad"),
        }[action.operator]
        selected_policies = tuple(
            policy for policy in self.policies if policy in policy_family
        )
        if not selected_policies:
            return ProviderResult.abstained(
                self.name,
                self.route,
                "operator_policy_family_unavailable",
                {
                    "action_operator": action.operator,
                    "configured_policies": list(self.policies),
                },
            )
        delegated = SparseCAProvider(
            policies=selected_policies,
            max_rules_per_policy=self.max_rules_per_policy,
            max_programs=(
                self.max_programs
                if action.operator == "local_transition_search"
                else 0
            ),
            minimum_query_support=self.minimum_query_support,
            minimum_foreground_support=self.minimum_foreground_support,
            name=self.name,
            route=self.route,
        )
        raw = delegated.propose(task, features, decision)
        if raw.status != "ok":
            return raw
        conditioned = tuple(
            _condition_candidate_on_action(candidate, action)
            for candidate in raw.candidates[: action.budget.candidate_slots]
        )
        diagnostics = {
            **raw.diagnostics,
            "action_operator": action.operator,
            "parent_hypothesis_id": action.parent_hypothesis_id,
            "selected_policy_family": list(selected_policies),
            "candidate_slot_limit": action.budget.candidate_slots,
            "native_cost": {
                "policy_fits": len(selected_policies),
                "max_rules_per_policy": self.max_rules_per_policy,
                "bounded_program_trials_enabled": (
                    action.operator == "local_transition_search"
                    and self.max_programs > 0
                ),
            },
        }
        return ProviderResult.ok(self.name, self.route, conditioned, diagnostics)


ExternalCallback = Callable[[BlindTask, TaskFeatures, RouteDecision], Sequence[CandidateHypothesis]]


def _content_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _validate_difflogic_bundle(spec: dict[str, object], task: BlindTask) -> int:
    """Validate a self-contained circuit manifest and return its charged bits."""

    required = {
        "artifact_sha256",
        "source_commit",
        "feature_schema_version",
        "config",
        "augmentation_policy",
        "horizon",
        "hard_circuit",
        "hard_circuit_sha256",
        "blind_content_sha256",
    }
    missing = required - set(spec)
    if missing:
        raise ValueError(f"DiffLogic hard bundle is incomplete: {sorted(missing)}")
    if (
        not isinstance(spec["source_commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", spec["source_commit"]) is None
        or not isinstance(spec["feature_schema_version"], str)
        or not spec["feature_schema_version"]
        or spec["augmentation_policy"] not in {"none", "d4", "bgpad", "d4_bgpad"}
        or spec["blind_content_sha256"] != task.blind_content_sha256
    ):
        raise ValueError("DiffLogic hard provenance fields are invalid")

    config = spec["config"]
    if not isinstance(config, dict):
        raise ValueError("DiffLogic hard config must be an object")
    required_config = {"model_kind", "state_encoding", "feature_count", "max_steps"}
    if required_config - set(config):
        raise ValueError("DiffLogic hard config is incomplete")
    if (
        config["model_kind"] != "difflogic_hard_circuit"
        or not isinstance(config["state_encoding"], str)
        or not config["state_encoding"]
        or not _positive_int(config["feature_count"])
        or not _positive_int(config["max_steps"])
        or not _positive_int(spec["horizon"])
        or spec["horizon"] > config["max_steps"]
    ):
        raise ValueError("DiffLogic hard config fields are invalid")

    circuit = spec["hard_circuit"]
    if not isinstance(circuit, dict):
        raise ValueError("DiffLogic hard circuit must be an object")
    required_circuit = {"input_count", "output_count", "gates", "outputs"}
    if required_circuit - set(circuit):
        raise ValueError("DiffLogic hard circuit is incomplete")
    input_count = circuit["input_count"]
    output_count = circuit["output_count"]
    gates = circuit["gates"]
    outputs = circuit["outputs"]
    if (
        not _positive_int(input_count)
        or not _positive_int(output_count)
        or input_count != config["feature_count"]
        or not isinstance(gates, list)
        or not isinstance(outputs, list)
        or len(outputs) != output_count
    ):
        raise ValueError("DiffLogic hard circuit dimensions are invalid")
    for gate_index, gate in enumerate(gates):
        if not isinstance(gate, dict) or set(gate) != {"op", "inputs"}:
            raise ValueError("DiffLogic gates require exactly op and inputs")
        references = gate["inputs"]
        operation = gate["op"]
        valid_operation = (
            isinstance(operation, str) and operation in DIFFLOGIC_GATE16_NAMES
        ) or (
            type(operation) is int
            and 0 <= operation < len(DIFFLOGIC_GATE16_NAMES)
        )
        if (
            not valid_operation
            or not isinstance(references, list)
            or len(references) != 2
            or any(
                type(reference) is not int
                or not 0 <= reference < input_count + gate_index
                for reference in references
            )
        ):
            raise ValueError("DiffLogic gate fields or references are invalid")
    if any(
        type(reference) is not int
        or not 0 <= reference < input_count + len(gates)
        for reference in outputs
    ):
        raise ValueError("DiffLogic output references are invalid")

    circuit_sha256 = _content_sha256(circuit)
    if spec["hard_circuit_sha256"] != circuit_sha256:
        raise ValueError("DiffLogic hard circuit digest does not match its payload")
    artifact_payload = {
        key: value for key, value in spec.items() if key != "artifact_sha256"
    }
    if spec["artifact_sha256"] != _content_sha256(artifact_payload):
        raise ValueError("DiffLogic artifact digest does not match its manifest")
    return 8 * len(canonical_json(circuit).encode("ascii"))


@dataclass(slots=True)
class OptionalCallbackProvider:
    name: str
    route: str
    callback: ExternalCallback | None = None
    absent_reason: str = "not_configured"

    def propose(
        self,
        task: BlindTask,
        features: TaskFeatures,
        decision: RouteDecision,
    ) -> ProviderResult:
        if not isinstance(task, BlindTask):
            raise TypeError("external providers accept BlindTask only")
        if self.callback is None:
            return ProviderResult.abstained(
                self.name,
                self.route,
                self.absent_reason,
                {"framework_started_training": False, "callback_invoked": False},
            )
        raw_candidates = tuple(self.callback(task, features, decision))
        allowed_modes = {
            "masked_diffusion": {"grid_only", "replayable"},
            "code_llm": {"replayable"},
            "difflogic_hard": {"hard_circuit"},
        }[self.route]
        candidates: list[CandidateHypothesis] = []
        scoring_fields = {
            "query_support",
            "foreground_query_support",
            "query_cell_count",
            "unsupported_query_cells",
            "foreground_query_cell_count",
            "unsupported_foreground_query_cells",
            "support_gate_passed",
        }
        rejected: list[dict[str, object]] = []
        for index, item in enumerate(raw_candidates):
            try:
                if not isinstance(item, CandidateHypothesis):
                    raise TypeError("external callback item is not a CandidateHypothesis")
                if item.route != self.route:
                    raise ValueError("external candidate route does not match its provider")
                if item.verification_mode not in allowed_modes:
                    raise ValueError(
                        "external candidate verification mode is not allowed for this provider"
                    )
                spec = item.spec
                artifact_sha256 = spec.get("artifact_sha256")
                if (
                    not isinstance(artifact_sha256, str)
                    or re.fullmatch(r"[0-9a-f]{64}", artifact_sha256) is None
                ):
                    raise ValueError(
                        "external candidate spec requires a lowercase artifact_sha256"
                    )
                circuit_bits = 0
                if self.route == "difflogic_hard":
                    circuit_bits = _validate_difflogic_bundle(spec, task)
                if item.verification_mode != "grid_only" and item.hard_verifier is None:
                    raise ValueError(
                        "external executable candidates require an independent verifier"
                    )
                explicit_cells = sum(
                    len(grid) * len(grid[0])
                    for bundle in (item.explicit_demo_outputs, item.explicit_query_outputs)
                    for grid in (bundle or ())
                )
                external_floor = 16 + 8 * len(item.spec_json.encode("ascii"))
                if item.verification_mode == "grid_only":
                    external_floor += 4 * explicit_cells
                metadata = {
                    key: value
                    for key, value in item.metadata.items()
                    if key not in scoring_fields
                }
                metadata.update(
                    {
                        "external_provider": self.name,
                        "external_declared_source": item.source,
                        "external_declared_source_version": item.source_version,
                        "external_description_floor_bits": external_floor,
                        "hard_circuit_payload_bits": circuit_bits,
                        "verifier_trust_boundary": "configured_external_integration",
                    }
                )
                candidates.append(
                    CandidateHypothesis.create(
                        name=item.name,
                        source=self.name,
                        source_version=OPTIONAL_PROVIDER_VERSION,
                        route=item.route,
                        description_bits=max(item.description_bits, external_floor),
                        verification_mode=item.verification_mode,
                        functional_trace=(f"external:{self.name}", *item.functional_trace),
                        spec=spec,
                        metadata=metadata,
                        parent_hypothesis_ids=item.parent_hypothesis_ids,
                        replay=item.replay,
                        hard_verifier=item.hard_verifier,
                        explicit_demo_outputs=item.explicit_demo_outputs,
                        explicit_query_outputs=item.explicit_query_outputs,
                    )
                )
            except (TypeError, ValueError) as exc:
                rejected.append(
                    {
                        "index": index,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        ordered = tuple(
            sorted(
                candidates,
                key=lambda item: (item.description_bits, item.hypothesis_id),
            )[: decision.budget_for(self.route)]
        )
        diagnostics = {
            "framework_started_training": False,
            "callback_invoked": True,
            "callback_is_trusted_integration_boundary": True,
            "reported_candidate_count": len(raw_candidates),
            "invalid_candidate_count": len(rejected),
            "invalid_candidates": rejected,
        }
        if raw_candidates and not ordered:
            return ProviderResult.error(
                self.name,
                self.route,
                "all_candidates_invalid",
                diagnostics,
            )
        return ProviderResult.ok(
            self.name,
            self.route,
            ordered,
            diagnostics,
        )


class MaskedDiffusionProvider(OptionalCallbackProvider):
    def __init__(self, callback: ExternalCallback | None = None) -> None:
        super().__init__("masked_diffusion", "masked_diffusion", callback, "not_configured")


class CodeModelProvider(OptionalCallbackProvider):
    def __init__(self, callback: ExternalCallback | None = None) -> None:
        super().__init__("code_llm", "code_llm", callback, "not_configured")


class DiffLogicHardProvider(OptionalCallbackProvider):
    """Inference-only hard-circuit source; a missing export is an abstention."""

    def __init__(self, callback: ExternalCallback | None = None) -> None:
        super().__init__("difflogic_hard", "difflogic_hard", callback, "missing_hard_circuit")
