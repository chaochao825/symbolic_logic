"""Demo-exact gate, hard replay checks, MDL accounting, and bundle dedup."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..blind import BlindTask
from ..grid import Grid, as_grid
from ..residual import compare_grids
from .types import CandidateEvaluation, CandidateHypothesis, canonical_json


ROUTE_DESCRIPTION_BITS = {
    "dsl_program": 3,
    "sparse_ca": 3,
    "difflogic_hard": 4,
    "code_llm": 4,
    "masked_diffusion": 4,
    "residual_repair": 4,
}


def _normalize_prediction(value: object | None) -> Grid | None:
    if value is None:
        return None
    try:
        normalized = value.tolist() if hasattr(value, "tolist") else value
        return as_grid(normalized)
    except (TypeError, ValueError):
        return None


def _replay_bundle(
    candidate: CandidateHypothesis,
    task: BlindTask,
) -> tuple[tuple[Grid | None, ...], tuple[Grid | None, ...]]:
    if candidate.replay is None:
        demos = candidate.explicit_demo_outputs or ()
        queries = candidate.explicit_query_outputs or ()
        if len(demos) != len(task.train) or len(queries) != len(task.test_inputs):
            return (
                tuple(None for _ in task.train),
                tuple(None for _ in task.test_inputs),
            )
        return tuple(demos), tuple(queries)

    def run(grid: Grid) -> Grid | None:
        try:
            return _normalize_prediction(candidate.replay(grid))
        except Exception:
            return None

    return (
        tuple(run(pair.input) for pair in task.train),
        tuple(run(grid) for grid in task.test_inputs),
    )


def _residual_description_bits(residuals: Sequence[object]) -> int:
    bits = 0
    for residual in residuals:
        cells = max(1, residual.comparison_cells)
        position_bits = max(1, math.ceil(math.log2(cells + 1)))
        bits += 1
        if not residual.execution_valid:
            bits += 4 * cells
            continue
        if not residual.shape_match:
            bits += 12
        if residual.mismatch_count:
            bits += position_bits + residual.mismatch_count * (position_bits + 4)
    return bits


def _support_description_bits(candidate: CandidateHypothesis) -> int:
    unsupported = candidate.metadata.get("unsupported_query_cells")
    total = candidate.metadata.get("query_cell_count")
    if (
        type(unsupported) is int
        and type(total) is int
        and 0 <= unsupported <= total
        and total > 0
    ):
        position_bits = max(1, math.ceil(math.log2(total + 1)))
        return unsupported * (position_bits + 4)
    raw = candidate.metadata.get("query_support")
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        return 0
    support = min(1.0, max(0.0, float(raw)))
    if support >= 1.0:
        return 0
    if support <= 0.0:
        return 16
    return min(16, math.ceil(-math.log2(support)))


def evaluate_hypothesis(
    candidate: CandidateHypothesis,
    task: BlindTask,
) -> CandidateEvaluation:
    if not isinstance(task, BlindTask):
        raise TypeError("verification accepts BlindTask only")
    demo_outputs, query_outputs = _replay_bundle(candidate, task)
    residuals = tuple(
        compare_grids(
            predicted,
            pair.output,
            pair_index=index,
            invalid_code="candidate_execution_failed" if predicted is None else None,
        )
        for index, (predicted, pair) in enumerate(zip(demo_outputs, task.train))
    )
    execution_valid = (
        len(demo_outputs) == len(task.train)
        and len(query_outputs) == len(task.test_inputs)
        and all(item is not None for item in (*demo_outputs, *query_outputs))
    )
    demo_exact = execution_valid and all(item.exact for item in residuals)
    hard_verified = False
    if (
        execution_valid
        and candidate.verification_mode in {"replayable", "hard_circuit"}
        and candidate.hard_verifier is not None
    ):
        try:
            hard_verified = candidate.hard_verifier(task) is True
        except Exception:
            hard_verified = False

    if not execution_valid:
        rejection = "execution_failed"
    elif not demo_exact:
        rejection = "demo_mismatch"
    elif candidate.metadata.get("support_gate_passed") is False:
        rejection = "insufficient_query_support"
    elif candidate.verification_mode == "grid_only" or candidate.hard_verifier is None:
        rejection = "missing_hard_verifier"
    elif not hard_verified:
        rejection = "hard_verification_failed"
    else:
        rejection = None

    route_bits = ROUTE_DESCRIPTION_BITS.get(candidate.route, 6)
    residual_bits = _residual_description_bits(residuals)
    support_bits = _support_description_bits(candidate)
    return CandidateEvaluation(
        hypothesis=candidate,
        demo_outputs=demo_outputs,
        query_outputs=query_outputs,
        residuals=residuals,
        demo_exact=demo_exact,
        hard_verified=hard_verified,
        rejection_reason=rejection,
        route_bits=route_bits,
        residual_bits=residual_bits,
        support_bits=support_bits,
        total_mdl_bits=(
            candidate.description_bits + route_bits + residual_bits + support_bits
        ),
    )


def evaluate_hypotheses(
    candidates: Sequence[CandidateHypothesis],
    task: BlindTask,
) -> tuple[CandidateEvaluation, ...]:
    unique, collisions = evaluate_hypotheses_isolated(candidates, task)
    if collisions:
        if "canonical_payload_conflict" in collisions.values():
            raise ValueError("hypothesis ID collision with different canonical payload")
        raise ValueError(
            "one canonical replay artifact produced inconsistent bundle semantics"
        )
    return unique


def evaluate_hypotheses_isolated(
    candidates: Sequence[CandidateHypothesis],
    task: BlindTask,
) -> tuple[tuple[CandidateEvaluation, ...], dict[str, str]]:
    """Evaluate once, quarantine every member of a conflicting artifact-ID group.

    The strict public helper above still raises on an identity/semantic collision.
    The orchestrator uses this partitioning form so a faulty provider cannot abort
    unrelated specialists or win by being observed first.
    """

    evaluated = tuple(evaluate_hypothesis(candidate, task) for candidate in candidates)
    grouped: dict[str, list[CandidateEvaluation]] = {}
    for evaluation in evaluated:
        grouped.setdefault(evaluation.hypothesis.hypothesis_id, []).append(evaluation)

    unique: dict[str, CandidateEvaluation] = {}
    collisions: dict[str, str] = {}
    for hypothesis_id in sorted(grouped):
        group = grouped[hypothesis_id]
        incumbent = group[0]
        canonical_payloads = {
            canonical_json(evaluation.hypothesis.to_json_dict())
            for evaluation in group
        }
        semantics = {
            (
                evaluation.demo_outputs,
                evaluation.query_outputs,
                evaluation.hard_verified,
            )
            for evaluation in group
        }
        reason = (
            "canonical_payload_conflict"
            if len(canonical_payloads) > 1
            else "replay_semantics_conflict"
            if len(semantics) > 1
            else None
        )
        if reason is not None:
            collisions[hypothesis_id] = reason
        else:
            unique[hypothesis_id] = incumbent
    return tuple(unique[key] for key in sorted(unique)), collisions


def rank_verified(
    evaluations: Sequence[CandidateEvaluation],
    *,
    limit: int = 2,
) -> tuple[CandidateEvaluation, ...]:
    if type(limit) is not int or limit < 1:
        raise ValueError("selection limit must be a positive integer")
    ranked = sorted(
        (item for item in evaluations if item.eligible),
        key=lambda item: (
            item.total_mdl_bits,
            item.hypothesis.description_bits,
            item.hypothesis.source,
            item.hypothesis.hypothesis_id,
        ),
    )
    selected: list[CandidateEvaluation] = []
    signatures: set[tuple[Grid, ...]] = set()
    for evaluation in ranked:
        signature = evaluation.query_signature()
        if signature is None or signature in signatures:
            continue
        signatures.add(signature)
        selected.append(evaluation)
        if len(selected) == limit:
            break
    return tuple(selected)
