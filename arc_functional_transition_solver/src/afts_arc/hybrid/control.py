"""Typed, content-addressed control primitives for online ARC search.

The controller uses normalized compute units (NCU), not wall-clock time or an
unverifiable FLOP estimate.  One non-stop action reserves one control unit plus
one unit for every candidate slot.  The full reservation is charged even when a
provider abstains, so unused capacity cannot be recycled differently by two
policies in a matched-budget comparison.

Only providers that explicitly declare ``strict_budget_contract`` (notably the
frozen candidate-pool adapter below) make a run suitable for strict controller
comparisons.  Legacy providers remain usable, but are labelled controller-only.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ..blind import BlindTask
from .router import ROUTES, RouteDecision, TaskFeatures
from .types import (
    CandidateEvaluation,
    CandidateHypothesis,
    ProviderResult,
    RepairReceipt,
    canonical_json,
)
from .verification import rank_verified


CONTROL_SCHEMA_VERSION = "afts.online-control/v1"
PROVIDER_OPERATORS: dict[str, frozenset[str]] = {
    "dsl_program": frozenset(
        {"synthesize", "shape_resynthesize", "suffix_resynthesize"}
    ),
    "code_llm": frozenset(
        {"open_hypothesis", "exception_resynthesize", "counterfactual_judge"}
    ),
    "sparse_ca": frozenset({"local_transition_search", "d4_bgpad_search"}),
    "difflogic_hard": frozenset({"hard_circuit_search"}),
    "masked_diffusion": frozenset({"global_sample", "masked_inpaint"}),
}
DEFAULT_PROVIDER_OPERATOR: dict[str, str] = {
    "dsl_program": "synthesize",
    "code_llm": "open_hypothesis",
    "sparse_ca": "local_transition_search",
    "difflogic_hard": "hard_circuit_search",
    "masked_diffusion": "global_sample",
}
REPAIR_OPERATORS = frozenset({"global_color_map", "local_transition"})


def _content_id(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()[:24]


@dataclass(frozen=True, slots=True)
class BudgetVector:
    """A multi-dimensional reservation in normalized controller units."""

    compute_units: int = 0
    controller_steps: int = 0
    provider_calls: int = 0
    repair_attempts: int = 0
    candidate_slots: int = 0

    def __post_init__(self) -> None:
        for name in (
            "compute_units",
            "controller_steps",
            "provider_calls",
            "repair_attempts",
            "candidate_slots",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"budget field {name} must be a non-negative integer")

    def __add__(self, other: "BudgetVector") -> "BudgetVector":
        if not isinstance(other, BudgetVector):
            return NotImplemented
        return BudgetVector(
            self.compute_units + other.compute_units,
            self.controller_steps + other.controller_steps,
            self.provider_calls + other.provider_calls,
            self.repair_attempts + other.repair_attempts,
            self.candidate_slots + other.candidate_slots,
        )

    def __sub__(self, other: "BudgetVector") -> "BudgetVector":
        if not isinstance(other, BudgetVector):
            return NotImplemented
        values = tuple(
            getattr(self, name) - getattr(other, name)
            for name in (
                "compute_units",
                "controller_steps",
                "provider_calls",
                "repair_attempts",
                "candidate_slots",
            )
        )
        if any(value < 0 for value in values):
            raise ValueError("budget subtraction would produce a negative resource")
        return BudgetVector(*values)

    def fits_within(self, other: "BudgetVector") -> bool:
        return all(
            getattr(self, name) <= getattr(other, name)
            for name in (
                "compute_units",
                "controller_steps",
                "provider_calls",
                "repair_attempts",
                "candidate_slots",
            )
        )

    @property
    def is_zero(self) -> bool:
        return self == BudgetVector()

    def to_json_dict(self) -> dict[str, int]:
        return {
            "compute_units": self.compute_units,
            "controller_steps": self.controller_steps,
            "provider_calls": self.provider_calls,
            "repair_attempts": self.repair_attempts,
            "candidate_slots": self.candidate_slots,
        }


@dataclass(frozen=True, slots=True)
class BudgetLedger:
    limit: BudgetVector
    used: BudgetVector = BudgetVector()

    def __post_init__(self) -> None:
        if not self.used.fits_within(self.limit):
            raise ValueError("used budget exceeds its limit")

    @property
    def remaining(self) -> BudgetVector:
        return self.limit - self.used

    def can_reserve(self, reservation: BudgetVector) -> bool:
        return reservation.fits_within(self.remaining)

    def charge(self, reservation: BudgetVector) -> "BudgetLedger":
        if not self.can_reserve(reservation):
            raise ValueError("action reservation exceeds the remaining budget")
        return BudgetLedger(self.limit, self.used + reservation)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "accounting": "normalized_compute_units",
            "reservation_policy": "charge_full_slice_even_on_abstention",
            "limit": self.limit.to_json_dict(),
            "used": self.used.to_json_dict(),
            "remaining": self.remaining.to_json_dict(),
        }


def _action_payload(
    *,
    state_id: str,
    kind: str,
    actor: str,
    route: str | None,
    operator: str,
    parent_hypothesis_id: str | None,
    evidence_signal_ids: tuple[str, ...],
    reason_codes: tuple[str, ...],
    budget: BudgetVector,
    priority: int,
) -> dict[str, object]:
    return {
        "schema": CONTROL_SCHEMA_VERSION,
        "state_id": state_id,
        "kind": kind,
        "actor": actor,
        "route": route,
        "operator": operator,
        "parent_hypothesis_id": parent_hypothesis_id,
        "evidence_signal_ids": list(evidence_signal_ids),
        "reason_codes": list(reason_codes),
        "budget": budget.to_json_dict(),
        "priority": priority,
    }


@dataclass(frozen=True, slots=True)
class ControlAction:
    """A type-checked action compiled from one immutable blackboard state."""

    action_id: str
    state_id: str
    kind: str
    actor: str
    route: str | None
    operator: str
    parent_hypothesis_id: str | None
    evidence_signal_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    budget: BudgetVector
    priority: int

    def __post_init__(self) -> None:
        if self.kind not in {"propose", "repair", "stop"}:
            raise ValueError("action kind must be propose, repair, or stop")
        if not self.state_id or not self.actor or not self.operator:
            raise TypeError("action state, actor, and operator must be non-empty")
        if type(self.priority) is not int:
            raise TypeError("action priority must be an integer")
        if tuple(sorted(set(self.evidence_signal_ids))) != self.evidence_signal_ids:
            raise ValueError("evidence signal IDs must be unique and sorted")
        if tuple(sorted(set(self.reason_codes))) != self.reason_codes:
            raise ValueError("reason codes must be unique and sorted")

        if self.kind == "propose":
            if self.route not in PROVIDER_OPERATORS:
                raise ValueError("proposal action requires a registered provider route")
            if self.operator not in PROVIDER_OPERATORS[self.route]:
                raise ValueError("operator is not legal for the proposal route")
            if (
                self.budget.controller_steps != 1
                or self.budget.provider_calls != 1
                or self.budget.repair_attempts != 0
                or self.budget.candidate_slots < 1
                or self.budget.compute_units != 1 + self.budget.candidate_slots
            ):
                raise ValueError(
                    "proposal action has an invalid normalized budget slice"
                )
        elif self.kind == "repair":
            if self.actor != "residual_repair" or self.route != "residual_repair":
                raise ValueError(
                    "repair action must use the residual_repair actor and route"
                )
            if self.operator not in REPAIR_OPERATORS:
                raise ValueError("unknown repair operator")
            if not self.parent_hypothesis_id:
                raise ValueError("repair action requires a parent hypothesis")
            if (
                self.budget.controller_steps != 1
                or self.budget.provider_calls != 0
                or self.budget.repair_attempts != 1
                or self.budget.candidate_slots != 1
                or self.budget.compute_units != 2
            ):
                raise ValueError("repair action has an invalid normalized budget slice")
        else:
            if (
                self.actor != "controller"
                or self.route is not None
                or self.operator != "stop"
            ):
                raise ValueError("STOP must be emitted by the controller")
            if self.parent_hypothesis_id is not None or not self.budget.is_zero:
                raise ValueError("STOP cannot have a parent or consume budget")

        expected = _content_id(
            _action_payload(
                state_id=self.state_id,
                kind=self.kind,
                actor=self.actor,
                route=self.route,
                operator=self.operator,
                parent_hypothesis_id=self.parent_hypothesis_id,
                evidence_signal_ids=self.evidence_signal_ids,
                reason_codes=self.reason_codes,
                budget=self.budget,
                priority=self.priority,
            )
        )
        if self.action_id != expected:
            raise ValueError("action_id does not match canonical action content")

    @classmethod
    def create(
        cls,
        *,
        state_id: str,
        kind: str,
        actor: str,
        route: str | None,
        operator: str,
        parent_hypothesis_id: str | None = None,
        evidence_signal_ids: Sequence[str] = (),
        reason_codes: Sequence[str] = (),
        budget: BudgetVector = BudgetVector(),
        priority: int = 0,
    ) -> "ControlAction":
        evidence = tuple(sorted(set(evidence_signal_ids)))
        reasons = tuple(sorted(set(reason_codes)))
        payload = _action_payload(
            state_id=state_id,
            kind=kind,
            actor=actor,
            route=route,
            operator=operator,
            parent_hypothesis_id=parent_hypothesis_id,
            evidence_signal_ids=evidence,
            reason_codes=reasons,
            budget=budget,
            priority=priority,
        )
        return cls(
            _content_id(payload),
            state_id,
            kind,
            actor,
            route,
            operator,
            parent_hypothesis_id,
            evidence,
            reasons,
            budget,
            priority,
        )

    def to_json_dict(self) -> dict[str, object]:
        payload = _action_payload(
            state_id=self.state_id,
            kind=self.kind,
            actor=self.actor,
            route=self.route,
            operator=self.operator,
            parent_hypothesis_id=self.parent_hypothesis_id,
            evidence_signal_ids=self.evidence_signal_ids,
            reason_codes=self.reason_codes,
            budget=self.budget,
            priority=self.priority,
        )
        payload["action_id"] = self.action_id
        return payload


def _result_payload(
    *,
    action: ControlAction,
    status: str,
    reason: str | None,
    budget_fidelity: str,
    emitted_candidate_ids: tuple[str, ...],
    accepted_candidate_ids: tuple[str, ...],
    quarantined_candidate_ids: tuple[str, ...],
    actual_candidate_evaluations: int,
    provider_result: ProviderResult | None,
    repair_receipt: RepairReceipt | None,
) -> dict[str, object]:
    return {
        "schema": CONTROL_SCHEMA_VERSION,
        "action": action.to_json_dict(),
        "status": status,
        "reason": reason,
        "charged_budget": action.budget.to_json_dict(),
        "budget_fidelity": budget_fidelity,
        "emitted_candidate_ids": list(emitted_candidate_ids),
        "accepted_candidate_ids": list(accepted_candidate_ids),
        "quarantined_candidate_ids": list(quarantined_candidate_ids),
        "actual_candidate_evaluations": actual_candidate_evaluations,
        "provider_result": None
        if provider_result is None
        else provider_result.to_json_dict(),
        "repair_receipt": None
        if repair_receipt is None
        else repair_receipt.to_json_dict(),
    }


@dataclass(frozen=True, slots=True)
class ActionResult:
    result_id: str
    action: ControlAction
    status: str
    reason: str | None
    budget_fidelity: str
    emitted_candidate_ids: tuple[str, ...] = ()
    accepted_candidate_ids: tuple[str, ...] = ()
    quarantined_candidate_ids: tuple[str, ...] = ()
    actual_candidate_evaluations: int = 0
    provider_result: ProviderResult | None = field(default=None, compare=False)
    repair_receipt: RepairReceipt | None = None

    def __post_init__(self) -> None:
        if self.status not in {"ok", "abstained", "error", "stopped"}:
            raise ValueError("unknown action-result status")
        if self.budget_fidelity not in {
            "strict_replayable",
            "controller_only",
            "not_applicable",
        }:
            raise ValueError("unknown budget fidelity")
        if type(self.actual_candidate_evaluations) is not int or not (
            0 <= self.actual_candidate_evaluations <= self.action.budget.candidate_slots
        ):
            raise ValueError("actual evaluations exceed the reserved candidate slots")
        for values in (
            self.emitted_candidate_ids,
            self.accepted_candidate_ids,
            self.quarantined_candidate_ids,
        ):
            if tuple(sorted(set(values))) != values:
                raise ValueError(
                    "action-result candidate IDs must be unique and sorted"
                )
        if self.action.kind == "stop":
            if self.status != "stopped" or self.budget_fidelity != "not_applicable":
                raise ValueError("STOP requires a stopped/not-applicable receipt")
        elif self.status == "stopped":
            raise ValueError("only STOP may produce a stopped receipt")
        expected = _content_id(
            _result_payload(
                action=self.action,
                status=self.status,
                reason=self.reason,
                budget_fidelity=self.budget_fidelity,
                emitted_candidate_ids=self.emitted_candidate_ids,
                accepted_candidate_ids=self.accepted_candidate_ids,
                quarantined_candidate_ids=self.quarantined_candidate_ids,
                actual_candidate_evaluations=self.actual_candidate_evaluations,
                provider_result=self.provider_result,
                repair_receipt=self.repair_receipt,
            )
        )
        if self.result_id != expected:
            raise ValueError("result_id does not match canonical result content")

    @classmethod
    def create(
        cls,
        *,
        action: ControlAction,
        status: str,
        reason: str | None,
        budget_fidelity: str,
        emitted_candidate_ids: Sequence[str] = (),
        accepted_candidate_ids: Sequence[str] = (),
        quarantined_candidate_ids: Sequence[str] = (),
        actual_candidate_evaluations: int = 0,
        provider_result: ProviderResult | None = None,
        repair_receipt: RepairReceipt | None = None,
    ) -> "ActionResult":
        emitted = tuple(sorted(set(emitted_candidate_ids)))
        accepted = tuple(sorted(set(accepted_candidate_ids)))
        quarantined = tuple(sorted(set(quarantined_candidate_ids)))
        payload = _result_payload(
            action=action,
            status=status,
            reason=reason,
            budget_fidelity=budget_fidelity,
            emitted_candidate_ids=emitted,
            accepted_candidate_ids=accepted,
            quarantined_candidate_ids=quarantined,
            actual_candidate_evaluations=actual_candidate_evaluations,
            provider_result=provider_result,
            repair_receipt=repair_receipt,
        )
        return cls(
            _content_id(payload),
            action,
            status,
            reason,
            budget_fidelity,
            emitted,
            accepted,
            quarantined,
            actual_candidate_evaluations,
            provider_result,
            repair_receipt,
        )

    def to_json_dict(self) -> dict[str, object]:
        payload = _result_payload(
            action=self.action,
            status=self.status,
            reason=self.reason,
            budget_fidelity=self.budget_fidelity,
            emitted_candidate_ids=self.emitted_candidate_ids,
            accepted_candidate_ids=self.accepted_candidate_ids,
            quarantined_candidate_ids=self.quarantined_candidate_ids,
            actual_candidate_evaluations=self.actual_candidate_evaluations,
            provider_result=self.provider_result,
            repair_receipt=self.repair_receipt,
        )
        payload["result_id"] = self.result_id
        return payload


@dataclass(frozen=True, slots=True)
class ResidualSignal:
    signal_id: str
    hypothesis_id: str
    route: str
    rejection_reason: str | None
    execution_failure_count: int
    shape_mismatch_count: int
    mismatch_count: int
    comparison_cells: int
    agreement: float
    localized: bool
    unambiguous_global_color_map: bool
    low_query_support: bool
    exact_but_unverified: bool

    @classmethod
    def from_evaluation(
        cls,
        evaluation: CandidateEvaluation,
        task: BlindTask,
        *,
        localized_fraction: float,
    ) -> "ResidualSignal":
        execution_failures = sum(
            not item.execution_valid for item in evaluation.residuals
        )
        shape_mismatches = sum(not item.shape_match for item in evaluation.residuals)
        mismatch_count = sum(item.mismatch_count for item in evaluation.residuals)
        comparison_cells = sum(item.comparison_cells for item in evaluation.residuals)
        fraction = mismatch_count / comparison_cells if comparison_cells else 1.0
        localized = (
            mismatch_count > 0
            and execution_failures == 0
            and shape_mismatches == 0
            and fraction <= localized_fraction
        )

        targets: dict[int, set[int]] = {}
        mapping_valid = execution_failures == 0 and shape_mismatches == 0
        changed = False
        if mapping_valid:
            for predicted, pair in zip(evaluation.demo_outputs, task.train):
                if predicted is None or pair.output is None:
                    mapping_valid = False
                    break
                for predicted_row, target_row in zip(predicted, pair.output):
                    for source, target in zip(predicted_row, target_row):
                        targets.setdefault(source, set()).add(target)
                        changed |= source != target
        unambiguous = (
            mapping_valid
            and changed
            and all(len(values) == 1 for values in targets.values())
        )
        metadata = evaluation.hypothesis.metadata
        raw_support = metadata.get("query_support", 1.0)
        low_support = metadata.get("support_gate_passed") is False or (
            isinstance(raw_support, (int, float))
            and not isinstance(raw_support, bool)
            and float(raw_support) < 1.0
        )
        exact_unverified = evaluation.demo_exact and not evaluation.hard_verified
        payload = {
            "hypothesis_id": evaluation.hypothesis.hypothesis_id,
            "route": evaluation.hypothesis.route,
            "rejection_reason": evaluation.rejection_reason,
            "execution_failure_count": execution_failures,
            "shape_mismatch_count": shape_mismatches,
            "mismatch_count": mismatch_count,
            "comparison_cells": comparison_cells,
            "agreement": evaluation.agreement,
            "localized": localized,
            "unambiguous_global_color_map": unambiguous,
            "low_query_support": low_support,
            "exact_but_unverified": exact_unverified,
        }
        return cls(_content_id(payload), **payload)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "signal_id": self.signal_id,
            "hypothesis_id": self.hypothesis_id,
            "route": self.route,
            "rejection_reason": self.rejection_reason,
            "execution_failure_count": self.execution_failure_count,
            "shape_mismatch_count": self.shape_mismatch_count,
            "mismatch_count": self.mismatch_count,
            "comparison_cells": self.comparison_cells,
            "agreement": self.agreement,
            "localized": self.localized,
            "unambiguous_global_color_map": self.unambiguous_global_color_map,
            "low_query_support": self.low_query_support,
            "exact_but_unverified": self.exact_but_unverified,
        }


def _state_payload(
    *,
    blind_task_id: str,
    blind_content_sha256: str,
    features: TaskFeatures,
    route_decision: RouteDecision,
    budget: BudgetLedger,
    candidates: tuple[CandidateHypothesis, ...],
    evaluations: tuple[CandidateEvaluation, ...],
    provider_results: tuple[ProviderResult, ...],
    repair_receipts: tuple[RepairReceipt, ...],
    action_results: tuple[ActionResult, ...],
    residual_signals: tuple[ResidualSignal, ...],
    quarantined_candidate_ids: tuple[str, ...],
    stopped: bool,
    stop_reason: str | None,
) -> dict[str, object]:
    return {
        "schema": CONTROL_SCHEMA_VERSION,
        "blind_task_id": blind_task_id,
        "blind_content_sha256": blind_content_sha256,
        "features": features.to_json_dict(),
        "route_decision": route_decision.to_json_dict(),
        "budget": budget.to_json_dict(),
        "candidates": [item.to_json_dict() for item in candidates],
        "evaluations": [item.to_json_dict() for item in evaluations],
        "provider_results": [item.to_json_dict() for item in provider_results],
        "repair_receipts": [item.to_json_dict() for item in repair_receipts],
        "action_result_ids": [item.result_id for item in action_results],
        "residual_signals": [item.to_json_dict() for item in residual_signals],
        "quarantined_candidate_ids": list(quarantined_candidate_ids),
        "stopped": stopped,
        "stop_reason": stop_reason,
    }


@dataclass(frozen=True, slots=True)
class Blackboard:
    """Immutable solver state; every transition creates a new content ID."""

    state_id: str
    blind_task_id: str
    blind_content_sha256: str
    features: TaskFeatures
    route_decision: RouteDecision
    budget: BudgetLedger
    candidates: tuple[CandidateHypothesis, ...]
    evaluations: tuple[CandidateEvaluation, ...]
    provider_results: tuple[ProviderResult, ...]
    repair_receipts: tuple[RepairReceipt, ...]
    action_results: tuple[ActionResult, ...]
    residual_signals: tuple[ResidualSignal, ...]
    quarantined_candidate_ids: tuple[str, ...]
    stopped: bool = False
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        candidate_ids = tuple(item.hypothesis_id for item in self.candidates)
        evaluation_ids = tuple(
            item.hypothesis.hypothesis_id for item in self.evaluations
        )
        if candidate_ids != tuple(sorted(set(candidate_ids))):
            raise ValueError("blackboard candidates must have unique sorted IDs")
        if evaluation_ids != candidate_ids:
            raise ValueError(
                "blackboard candidate/evaluation ledgers must be ID-aligned"
            )
        if (
            tuple(sorted(set(self.quarantined_candidate_ids)))
            != self.quarantined_candidate_ids
        ):
            raise ValueError("quarantined candidate IDs must be unique and sorted")
        if set(candidate_ids) & set(self.quarantined_candidate_ids):
            raise ValueError(
                "quarantined candidates cannot remain on the active blackboard"
            )
        signal_ids = tuple(item.signal_id for item in self.residual_signals)
        if signal_ids != tuple(sorted(set(signal_ids))):
            raise ValueError("residual signals must have unique sorted IDs")
        expected = _content_id(
            _state_payload(
                blind_task_id=self.blind_task_id,
                blind_content_sha256=self.blind_content_sha256,
                features=self.features,
                route_decision=self.route_decision,
                budget=self.budget,
                candidates=self.candidates,
                evaluations=self.evaluations,
                provider_results=self.provider_results,
                repair_receipts=self.repair_receipts,
                action_results=self.action_results,
                residual_signals=self.residual_signals,
                quarantined_candidate_ids=self.quarantined_candidate_ids,
                stopped=self.stopped,
                stop_reason=self.stop_reason,
            )
        )
        if self.state_id != expected:
            raise ValueError("state_id does not match canonical blackboard content")

    @classmethod
    def create(
        cls,
        *,
        task: BlindTask,
        features: TaskFeatures,
        route_decision: RouteDecision,
        budget: BudgetLedger,
        candidates: Sequence[CandidateHypothesis] = (),
        evaluations: Sequence[CandidateEvaluation] = (),
        provider_results: Sequence[ProviderResult] = (),
        repair_receipts: Sequence[RepairReceipt] = (),
        action_results: Sequence[ActionResult] = (),
        quarantined_candidate_ids: Sequence[str] = (),
        localized_fraction: float = 0.35,
        stopped: bool = False,
        stop_reason: str | None = None,
    ) -> "Blackboard":
        ordered_candidates = tuple(
            sorted(candidates, key=lambda item: item.hypothesis_id)
        )
        ordered_evaluations = tuple(
            sorted(evaluations, key=lambda item: item.hypothesis.hypothesis_id)
        )
        signals = tuple(
            sorted(
                (
                    ResidualSignal.from_evaluation(
                        item,
                        task,
                        localized_fraction=localized_fraction,
                    )
                    for item in ordered_evaluations
                ),
                key=lambda item: item.signal_id,
            )
        )
        normalized_quarantine = tuple(sorted(set(quarantined_candidate_ids)))
        provider_tuple = tuple(provider_results)
        repair_tuple = tuple(repair_receipts)
        result_tuple = tuple(action_results)
        payload = _state_payload(
            blind_task_id=task.task_id,
            blind_content_sha256=task.blind_content_sha256,
            features=features,
            route_decision=route_decision,
            budget=budget,
            candidates=ordered_candidates,
            evaluations=ordered_evaluations,
            provider_results=provider_tuple,
            repair_receipts=repair_tuple,
            action_results=result_tuple,
            residual_signals=signals,
            quarantined_candidate_ids=normalized_quarantine,
            stopped=stopped,
            stop_reason=stop_reason,
        )
        return cls(
            _content_id(payload),
            task.task_id,
            task.blind_content_sha256,
            features,
            route_decision,
            budget,
            ordered_candidates,
            ordered_evaluations,
            provider_tuple,
            repair_tuple,
            result_tuple,
            signals,
            normalized_quarantine,
            stopped,
            stop_reason,
        )

    @property
    def seen_candidate_ids(self) -> frozenset[str]:
        emitted = {
            hypothesis_id
            for result in self.action_results
            for hypothesis_id in result.emitted_candidate_ids
        }
        return frozenset(
            emitted
            | {item.hypothesis_id for item in self.candidates}
            | set(self.quarantined_candidate_ids)
        )

    def selected(self, *, limit: int = 2) -> tuple[CandidateEvaluation, ...]:
        return rank_verified(self.evaluations, limit=limit)

    def to_json_dict(self, *, selected_limit: int = 2) -> dict[str, object]:
        return {
            "state_id": self.state_id,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
            "budget": self.budget.to_json_dict(),
            "candidate_ids": [item.hypothesis_id for item in self.candidates],
            "evaluation_ids": [
                item.hypothesis.hypothesis_id for item in self.evaluations
            ],
            "selected_hypothesis_ids": [
                item.hypothesis.hypothesis_id
                for item in self.selected(limit=selected_limit)
            ],
            "residual_signals": [item.to_json_dict() for item in self.residual_signals],
            "quarantined_candidate_ids": list(self.quarantined_candidate_ids),
            "action_result_ids": [item.result_id for item in self.action_results],
        }


class OnlineCandidateProvider(Protocol):
    name: str
    route: str
    strict_budget_contract: bool

    def act(
        self,
        task: BlindTask,
        features: TaskFeatures,
        decision: RouteDecision,
        blackboard: Blackboard,
        action: ControlAction,
    ) -> ProviderResult: ...


@dataclass(frozen=True, slots=True)
class FrozenActionBatch:
    """One immutable action-conditioned slice of a frozen candidate pool."""

    operator: str
    candidates: tuple[CandidateHypothesis, ...]
    parent_hypothesis_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operator, str) or not self.operator:
            raise TypeError("frozen action operator must be non-empty")
        if self.parent_hypothesis_id is not None and (
            not isinstance(self.parent_hypothesis_id, str)
            or not self.parent_hypothesis_id
        ):
            raise TypeError("frozen action parent must be None or a non-empty ID")
        canonical = tuple(
            sorted(
                self.candidates,
                key=lambda item: (item.description_bits, item.hypothesis_id),
            )
        )
        if any(not isinstance(item, CandidateHypothesis) for item in canonical):
            raise TypeError("frozen action batches require CandidateHypothesis values")
        if len({item.hypothesis_id for item in canonical}) != len(canonical):
            raise ValueError("a frozen action batch requires unique candidate IDs")
        object.__setattr__(self, "candidates", canonical)

    @property
    def batch_id(self) -> str:
        return _content_id(
            {
                "operator": self.operator,
                "parent_hypothesis_id": self.parent_hypothesis_id,
                "candidates": [item.to_json_dict() for item in self.candidates],
            }
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "operator": self.operator,
            "parent_hypothesis_id": self.parent_hypothesis_id,
            "candidate_ids": [item.hypothesis_id for item in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class FrozenCandidatePoolProvider:
    """Replay deterministic batches from a strict action-addressed pool."""

    candidates: tuple[CandidateHypothesis, ...]
    name: str
    route: str
    action_batches: tuple[FrozenActionBatch, ...] = ()
    strict_budget_contract: bool = field(default=True, init=False)
    supports_residual_actions: bool = field(default=True, init=False)
    supports_repeated_batches: bool = field(default=True, init=False)
    parent_sensitive_operators: frozenset[str] = field(
        default=frozenset(), init=False
    )

    def __post_init__(self) -> None:
        if not self.name or self.route not in ROUTES:
            raise ValueError("frozen pool requires a name and registered route")
        batches = tuple(self.action_batches)
        if any(not isinstance(item, FrozenActionBatch) for item in batches):
            raise TypeError("action_batches must contain FrozenActionBatch values")
        if not batches:
            batches = (
                FrozenActionBatch(
                    DEFAULT_PROVIDER_OPERATOR[self.route],
                    tuple(self.candidates),
                ),
            )
        if any(
            batch.operator not in PROVIDER_OPERATORS[self.route] for batch in batches
        ):
            raise ValueError("frozen action operator is not legal for the provider route")
        batch_keys = tuple(
            (batch.operator, batch.parent_hypothesis_id) for batch in batches
        )
        if len(set(batch_keys)) != len(batch_keys):
            raise ValueError("frozen action batches require unique action keys")

        supplied = (
            *self.candidates,
            *(candidate for batch in batches for candidate in batch.candidates),
        )
        payload_by_id: dict[str, str] = {}
        pooled: dict[str, CandidateHypothesis] = {}
        for candidate in supplied:
            if candidate.route != self.route:
                raise ValueError("frozen-pool candidate route mismatch")
            payload = canonical_json(candidate.to_json_dict())
            incumbent = payload_by_id.get(candidate.hypothesis_id)
            if incumbent is not None and incumbent != payload:
                raise ValueError("frozen-pool candidate IDs have conflicting payloads")
            payload_by_id[candidate.hypothesis_id] = payload
            pooled[candidate.hypothesis_id] = candidate
        canonical = tuple(
            sorted(
                pooled.values(),
                key=lambda item: (item.description_bits, item.hypothesis_id),
            )
        )
        object.__setattr__(self, "candidates", canonical)
        object.__setattr__(
            self,
            "parent_sensitive_operators",
            frozenset(
                batch.operator
                for batch in batches
                if batch.parent_hypothesis_id is not None
            ),
        )
        object.__setattr__(
            self,
            "action_batches",
            tuple(
                sorted(
                    batches,
                    key=lambda item: (
                        item.operator,
                        item.parent_hypothesis_id or "",
                        item.batch_id,
                    ),
                )
            ),
        )

    @classmethod
    def from_action_batches(
        cls,
        action_batches: Sequence[FrozenActionBatch],
        name: str,
        route: str,
    ) -> "FrozenCandidatePoolProvider":
        return cls((), name, route, tuple(action_batches))

    @property
    def pool_id(self) -> str:
        return _content_id(
            {
                "name": self.name,
                "route": self.route,
                "candidates": [item.to_json_dict() for item in self.candidates],
                "action_batches": [
                    item.to_json_dict() for item in self.action_batches
                ],
            }
        )

    @property
    def max_control_calls(self) -> int:
        return max(1, len(self.candidates), len(self.action_batches))

    def has_unseen_candidates(self, blackboard: Blackboard) -> bool:
        attempted = {
            (result.action.operator, result.action.parent_hypothesis_id)
            for result in blackboard.action_results
            if result.action.kind == "propose" and result.action.actor == self.name
        }
        for batch in self.action_batches:
            key = (batch.operator, batch.parent_hypothesis_id)
            if batch.candidates and any(
                item.hypothesis_id not in blackboard.seen_candidate_ids
                for item in batch.candidates
            ):
                return True
            if not batch.candidates and key not in attempted:
                return True
        return False

    def act(
        self,
        task: BlindTask,
        features: TaskFeatures,
        decision: RouteDecision,
        blackboard: Blackboard,
        action: ControlAction,
    ) -> ProviderResult:
        del task, features, decision
        if (
            action.kind != "propose"
            or action.actor != self.name
            or action.route != self.route
        ):
            raise ValueError("frozen pool received an action for another provider")
        matching = tuple(
            batch
            for batch in self.action_batches
            if batch.operator == action.operator
            and batch.parent_hypothesis_id == action.parent_hypothesis_id
        )
        available_keys = tuple(
            sorted(
                f"{batch.operator}:{batch.parent_hypothesis_id or '-'}"
                for batch in self.action_batches
            )
        )
        if not matching:
            return ProviderResult.abstained(
                self.name,
                self.route,
                "frozen_action_unavailable",
                {
                    "pool_id": self.pool_id,
                    "strict_budget_contract": True,
                    "requested_operator": action.operator,
                    "requested_parent_hypothesis_id": action.parent_hypothesis_id,
                    "available_action_keys": list(available_keys),
                },
            )
        unseen = tuple(
            item
            for batch in matching
            for item in batch.candidates
            if item.hypothesis_id not in blackboard.seen_candidate_ids
        )
        selected = tuple(
            sorted(
                {item.hypothesis_id: item for item in unseen}.values(),
                key=lambda item: (item.description_bits, item.hypothesis_id),
            )[: action.budget.candidate_slots]
        )
        diagnostics = {
            "pool_id": self.pool_id,
            "strict_budget_contract": True,
            "matched_batch_ids": [item.batch_id for item in matching],
            "requested_operator": action.operator,
            "requested_parent_hypothesis_id": action.parent_hypothesis_id,
            "available_before_action": len(unseen),
            "candidate_slot_limit": action.budget.candidate_slots,
            "remaining_after_action": len(unseen) - len(selected),
        }
        if not selected:
            return ProviderResult.abstained(
                self.name, self.route, "frozen_action_exhausted", diagnostics
            )
        return ProviderResult.ok(self.name, self.route, selected, diagnostics)


class ControlPolicy(Protocol):
    name: str

    def select(
        self,
        blackboard: Blackboard,
        actions: Sequence[ControlAction],
    ) -> ControlAction: ...


@dataclass(frozen=True, slots=True)
class ResidualFirstPolicy:
    name: str = "residual_first_v1"

    def select(
        self,
        blackboard: Blackboard,
        actions: Sequence[ControlAction],
    ) -> ControlAction:
        del blackboard
        available = tuple(actions)
        if not available:
            raise ValueError("policy requires at least one compiled action")
        non_stop = tuple(item for item in available if item.kind != "stop")
        pool = non_stop or available
        return min(pool, key=lambda item: (item.priority, item.action_id))


@dataclass(frozen=True, slots=True)
class CoverageAwareResidualPolicy:
    """Stateful exploration policy with demo-only representation preferences.

    The policy reserves early calls for compatible, previously untried sources,
    then uses semantic action novelty and typed residual priority to choose among
    repairs or repeated proposal actions.  It never reads a hidden test output.
    """

    name: str = "coverage_aware_residual_v2"

    @staticmethod
    def _compatible(blackboard: Blackboard, action: ControlAction) -> bool:
        if action.route == "sparse_ca":
            return blackboard.features.all_same_shape
        return True

    @staticmethod
    def _route_rank(blackboard: Blackboard, action: ControlAction) -> int:
        route = action.route or ""
        features = blackboard.features
        if features.shape_change:
            preferred = ("dsl_program", "code_llm", "masked_diffusion")
        elif features.d4_consistent:
            preferred = ("dsl_program", "sparse_ca", "difflogic_hard")
        elif features.all_same_shape:
            preferred = ("sparse_ca", "dsl_program", "difflogic_hard")
        else:
            preferred = blackboard.route_decision.route_order
        try:
            return preferred.index(route)
        except ValueError:
            return len(preferred) + blackboard.route_decision.priority_for(route)

    @staticmethod
    def _semantic_novelty_by_result(blackboard: Blackboard) -> dict[str, int]:
        evaluations = {
            item.hypothesis.hypothesis_id: item for item in blackboard.evaluations
        }
        seen: set[tuple[object, ...]] = set()
        novelty: dict[str, int] = {}
        for result in blackboard.action_results:
            count = 0
            for hypothesis_id in result.accepted_candidate_ids:
                evaluation = evaluations.get(hypothesis_id)
                if evaluation is None:
                    continue
                signature = (
                    evaluation.demo_outputs,
                    evaluation.query_outputs,
                    evaluation.hard_verified,
                    evaluation.rejection_reason,
                )
                if signature not in seen:
                    seen.add(signature)
                    count += 1
            novelty[result.result_id] = count
        return novelty

    def select(
        self,
        blackboard: Blackboard,
        actions: Sequence[ControlAction],
    ) -> ControlAction:
        available = tuple(actions)
        if not available:
            raise ValueError("policy requires at least one compiled action")
        proposals = tuple(
            item
            for item in available
            if item.kind == "propose" and self._compatible(blackboard, item)
        )
        calls: dict[str, int] = {}
        for result in blackboard.action_results:
            if result.action.kind == "propose":
                calls[result.action.actor] = calls.get(result.action.actor, 0) + 1

        untried = tuple(item for item in proposals if calls.get(item.actor, 0) == 0)
        if untried:
            return min(
                untried,
                key=lambda item: (
                    self._route_rank(blackboard, item),
                    item.priority,
                    item.actor,
                    item.action_id,
                ),
            )

        repairs = tuple(item for item in available if item.kind == "repair")
        if repairs:
            return min(repairs, key=lambda item: (item.priority, item.action_id))

        if proposals:
            novelty = self._semantic_novelty_by_result(blackboard)
            latest_by_operator: dict[tuple[str, str], ActionResult] = {}
            for result in blackboard.action_results:
                if result.action.kind == "propose":
                    latest_by_operator[(result.action.actor, result.action.operator)] = (
                        result
                    )

            def proposal_key(action: ControlAction) -> tuple[object, ...]:
                previous = latest_by_operator.get((action.actor, action.operator))
                no_semantic_gain = (
                    previous is not None and novelty.get(previous.result_id, 0) == 0
                )
                return (
                    no_semantic_gain,
                    calls.get(action.actor, 0),
                    self._route_rank(blackboard, action),
                    not bool(action.evidence_signal_ids),
                    action.priority,
                    action.actor,
                    action.action_id,
                )

            return min(proposals, key=proposal_key)

        stops = tuple(item for item in available if item.kind == "stop")
        if not stops:
            raise ValueError("compiled action set has no executable action or STOP")
        return min(stops, key=lambda item: item.action_id)


@dataclass(frozen=True, slots=True)
class FixedSchedulePolicy:
    """Matched-budget run-all baseline that defers repair until proposals finish."""

    schedule: tuple[str, ...] = ROUTES
    name: str = "fixed_schedule_v1"

    def __post_init__(self) -> None:
        if not self.schedule or len(set(self.schedule)) != len(self.schedule):
            raise ValueError("fixed schedule must contain unique provider names/routes")

    def select(
        self,
        blackboard: Blackboard,
        actions: Sequence[ControlAction],
    ) -> ControlAction:
        del blackboard
        available = tuple(actions)
        proposals = tuple(item for item in available if item.kind == "propose")
        for token in self.schedule:
            matching = tuple(
                item for item in proposals if item.actor == token or item.route == token
            )
            if matching:
                return min(matching, key=lambda item: (item.priority, item.action_id))
        if proposals:
            return min(proposals, key=lambda item: (item.actor, item.action_id))
        repairs = tuple(item for item in available if item.kind == "repair")
        if repairs:
            return min(repairs, key=lambda item: (item.priority, item.action_id))
        stops = tuple(item for item in available if item.kind == "stop")
        if not stops:
            raise ValueError("compiled action set has no executable action or STOP")
        return min(stops, key=lambda item: item.action_id)


@dataclass(frozen=True, slots=True)
class StaticRoutePolicy:
    """One-shot task router baseline that ignores residual priority bonuses."""

    name: str = "static_task_router_v1"

    def select(
        self,
        blackboard: Blackboard,
        actions: Sequence[ControlAction],
    ) -> ControlAction:
        available = tuple(actions)
        proposals = tuple(item for item in available if item.kind == "propose")
        if proposals:
            return min(
                proposals,
                key=lambda item: (
                    blackboard.route_decision.priority_for(item.route or ""),
                    item.actor,
                    item.action_id,
                ),
            )
        repairs = tuple(item for item in available if item.kind == "repair")
        if repairs:
            return min(repairs, key=lambda item: (item.priority, item.action_id))
        stops = tuple(item for item in available if item.kind == "stop")
        if not stops:
            raise ValueError("compiled action set has no executable action or STOP")
        return min(stops, key=lambda item: item.action_id)


@dataclass(frozen=True, slots=True)
class RoundRobinPolicy:
    """Explore the least-called provider before consuming another provider slice."""

    schedule: tuple[str, ...] = ROUTES
    name: str = "round_robin_v1"

    def __post_init__(self) -> None:
        if not self.schedule or len(set(self.schedule)) != len(self.schedule):
            raise ValueError("round-robin schedule must contain unique tokens")

    def select(
        self,
        blackboard: Blackboard,
        actions: Sequence[ControlAction],
    ) -> ControlAction:
        available = tuple(actions)
        proposals = tuple(item for item in available if item.kind == "propose")
        if proposals:
            calls: dict[str, int] = {}
            for result in blackboard.action_results:
                if result.action.kind == "propose":
                    calls[result.action.actor] = calls.get(result.action.actor, 0) + 1

            def schedule_rank(action: ControlAction) -> int:
                for index, token in enumerate(self.schedule):
                    if action.actor == token or action.route == token:
                        return index
                return len(self.schedule)

            return min(
                proposals,
                key=lambda item: (
                    calls.get(item.actor, 0),
                    schedule_rank(item),
                    item.actor,
                    item.action_id,
                ),
            )
        repairs = tuple(item for item in available if item.kind == "repair")
        if repairs:
            return min(repairs, key=lambda item: (item.priority, item.action_id))
        stops = tuple(item for item in available if item.kind == "stop")
        if not stops:
            raise ValueError("compiled action set has no executable action or STOP")
        return min(stops, key=lambda item: item.action_id)


@dataclass(frozen=True, slots=True)
class DeterministicRandomPolicy:
    """Oracle-free random-action baseline with content-stable sampling."""

    seed: int = 0
    name: str = "deterministic_random_v1"

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise TypeError("random-policy seed must be an integer")

    def select(
        self,
        blackboard: Blackboard,
        actions: Sequence[ControlAction],
    ) -> ControlAction:
        del blackboard
        available = tuple(actions)
        if not available:
            raise ValueError("policy requires at least one compiled action")
        non_stop = tuple(item for item in available if item.kind != "stop")
        pool = non_stop or available

        def random_key(action: ControlAction) -> tuple[str, str]:
            digest = hashlib.sha256(
                f"{self.seed}:{action.state_id}:{action.action_id}".encode("ascii")
            ).hexdigest()
            return digest, action.action_id

        return min(pool, key=random_key)


@dataclass(frozen=True, slots=True)
class ResidualCompilerConfig:
    provider_batch_size: int = 4
    minimum_repair_agreement: float = 0.5
    localized_residual_fraction: float = 0.35

    def __post_init__(self) -> None:
        if type(self.provider_batch_size) is not int or self.provider_batch_size < 1:
            raise ValueError("provider_batch_size must be positive")
        for name in ("minimum_repair_agreement", "localized_residual_fraction"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


class ResidualActionCompiler:
    """Compile observed failure structure into legal proposal/repair actions."""

    def __init__(self, config: ResidualCompilerConfig | None = None) -> None:
        self.config = config or ResidualCompilerConfig()

    @staticmethod
    def _attempt_key(action: ControlAction) -> tuple[object, ...]:
        return (
            action.kind,
            action.actor,
            action.operator,
            action.parent_hypothesis_id,
        )

    @staticmethod
    def _best_signal(blackboard: Blackboard) -> ResidualSignal | None:
        candidates = tuple(
            item
            for item in blackboard.residual_signals
            if item.mismatch_count > 0
            or item.execution_failure_count > 0
            or item.low_query_support
            or item.exact_but_unverified
        )
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (
                not item.unambiguous_global_color_map,
                not item.localized,
                item.execution_failure_count > 0,
                -item.agreement,
                item.mismatch_count,
                item.signal_id,
            ),
        )

    def _provider_slice(self, remaining: BudgetVector) -> BudgetVector | None:
        if remaining.controller_steps < 1 or remaining.provider_calls < 1:
            return None
        slots = min(
            self.config.provider_batch_size,
            remaining.candidate_slots,
            max(0, remaining.compute_units - 1),
        )
        if slots < 1:
            return None
        reservation = BudgetVector(
            compute_units=1 + slots,
            controller_steps=1,
            provider_calls=1,
            candidate_slots=slots,
        )
        return reservation if reservation.fits_within(remaining) else None

    @staticmethod
    def _repair_slice(remaining: BudgetVector) -> BudgetVector | None:
        reservation = BudgetVector(
            compute_units=2,
            controller_steps=1,
            repair_attempts=1,
            candidate_slots=1,
        )
        return reservation if reservation.fits_within(remaining) else None

    @staticmethod
    def _provider_operator(
        route: str,
        signal: ResidualSignal | None,
        features: TaskFeatures,
    ) -> tuple[str, tuple[str, ...], int]:
        if signal is None:
            defaults = {
                "dsl_program": "synthesize",
                "code_llm": "open_hypothesis",
                "sparse_ca": "d4_bgpad_search"
                if features.d4_consistent
                else "local_transition_search",
                "difflogic_hard": "hard_circuit_search",
                "masked_diffusion": "global_sample",
            }
            return defaults[route], ("initial_route",), 0

        if signal.execution_failure_count or signal.shape_mismatch_count:
            mapping = {
                "dsl_program": (
                    "shape_resynthesize",
                    "shape_or_execution_failure",
                    -60,
                ),
                "code_llm": (
                    "exception_resynthesize",
                    "shape_or_execution_failure",
                    -40,
                ),
                "masked_diffusion": (
                    "global_sample",
                    "shape_or_execution_failure",
                    -20,
                ),
                "sparse_ca": ("d4_bgpad_search", "shape_failure_symbolic_fallback", 40),
                "difflogic_hard": (
                    "hard_circuit_search",
                    "shape_failure_symbolic_fallback",
                    50,
                ),
            }
        elif signal.localized:
            mapping = {
                "masked_diffusion": ("masked_inpaint", "localized_demo_residual", -50),
                "sparse_ca": (
                    "local_transition_search",
                    "localized_demo_residual",
                    -45,
                ),
                "difflogic_hard": (
                    "hard_circuit_search",
                    "compressible_local_residual",
                    -35,
                ),
                "dsl_program": (
                    "suffix_resynthesize",
                    "localized_program_residual",
                    -25,
                ),
                "code_llm": ("exception_resynthesize", "localized_demo_residual", -15),
            }
        elif signal.low_query_support:
            mapping = {
                "dsl_program": ("synthesize", "low_query_support_switch", -45),
                "code_llm": ("exception_resynthesize", "low_query_support_switch", -30),
                "masked_diffusion": ("global_sample", "low_query_support_switch", -20),
                "sparse_ca": ("d4_bgpad_search", "low_query_support_retry", 30),
                "difflogic_hard": (
                    "hard_circuit_search",
                    "low_query_support_switch",
                    10,
                ),
            }
        elif signal.exact_but_unverified:
            mapping = {
                "dsl_program": (
                    "suffix_resynthesize",
                    "compile_unverified_exact_grid",
                    -50,
                ),
                "code_llm": (
                    "exception_resynthesize",
                    "compile_unverified_exact_grid",
                    -35,
                ),
                "difflogic_hard": (
                    "hard_circuit_search",
                    "compile_unverified_exact_grid",
                    -30,
                ),
                "masked_diffusion": (
                    "masked_inpaint",
                    "unverified_exact_refinement",
                    20,
                ),
                "sparse_ca": (
                    "local_transition_search",
                    "compile_unverified_exact_grid",
                    -10,
                ),
            }
        else:
            mapping = {
                "dsl_program": ("suffix_resynthesize", "unexplained_residual", -20),
                "code_llm": ("open_hypothesis", "unexplained_residual", -15),
                "masked_diffusion": ("global_sample", "unexplained_residual", -10),
                "sparse_ca": ("local_transition_search", "unexplained_residual", 10),
                "difflogic_hard": ("hard_circuit_search", "unexplained_residual", 5),
            }
        operator, reason, bonus = mapping[route]
        return operator, (reason, "cross_representation_control"), bonus

    def compile(
        self,
        task: BlindTask,
        blackboard: Blackboard,
        providers: Sequence[object],
        *,
        max_selected_hypotheses: int,
    ) -> tuple[ControlAction, ...]:
        if blackboard.stopped:
            raise ValueError("cannot compile actions for a stopped blackboard")
        selected = blackboard.selected(limit=max_selected_hypotheses)
        distinct_signatures = {
            item.query_signature()
            for item in selected
            if item.query_signature() is not None
        }
        if len(distinct_signatures) >= max_selected_hypotheses:
            return (
                ControlAction.create(
                    state_id=blackboard.state_id,
                    kind="stop",
                    actor="controller",
                    route=None,
                    operator="stop",
                    reason_codes=("pass_at_k_filled",),
                    priority=-10_000,
                ),
            )

        attempted = {
            self._attempt_key(result.action) for result in blackboard.action_results
        }
        remaining = blackboard.budget.remaining
        actions: list[ControlAction] = []
        repair_slice = self._repair_slice(remaining)
        signal_by_hypothesis = {
            item.hypothesis_id: item for item in blackboard.residual_signals
        }
        if repair_slice is not None:
            evaluations = tuple(
                sorted(
                    blackboard.evaluations,
                    key=lambda item: (
                        item.residual_bits,
                        -item.agreement,
                        item.hypothesis.hypothesis_id,
                    ),
                )
            )
            for rank, evaluation in enumerate(evaluations):
                signal = signal_by_hypothesis[evaluation.hypothesis.hypothesis_id]
                parent_legal = (
                    not evaluation.demo_exact
                    and evaluation.hard_verified
                    and evaluation.hypothesis.replay is not None
                    and evaluation.agreement >= self.config.minimum_repair_agreement
                    and signal.execution_failure_count == 0
                    and signal.shape_mismatch_count == 0
                )
                if not parent_legal:
                    continue
                operators: list[tuple[str, str]] = []
                if signal.unambiguous_global_color_map:
                    operators.append(
                        ("global_color_map", "unambiguous_color_confusion")
                    )
                if signal.localized and len(task.train) >= 2:
                    operators.append(
                        ("local_transition", "localized_high_support_candidate")
                    )
                for operator, reason in operators:
                    key = (
                        "repair",
                        "residual_repair",
                        operator,
                        evaluation.hypothesis.hypothesis_id,
                    )
                    if key in attempted:
                        continue
                    actions.append(
                        ControlAction.create(
                            state_id=blackboard.state_id,
                            kind="repair",
                            actor="residual_repair",
                            route="residual_repair",
                            operator=operator,
                            parent_hypothesis_id=evaluation.hypothesis.hypothesis_id,
                            evidence_signal_ids=(signal.signal_id,),
                            reason_codes=(reason, "demo_residual_compiled"),
                            budget=repair_slice,
                            priority=-100 + rank,
                        )
                    )

        provider_slice = self._provider_slice(remaining)
        best_signal = self._best_signal(blackboard)
        calls_by_provider: dict[str, int] = {}
        for result in blackboard.action_results:
            if result.action.kind == "propose":
                calls_by_provider[result.action.actor] = (
                    calls_by_provider.get(result.action.actor, 0) + 1
                )
        if provider_slice is not None:
            for provider_index, provider in enumerate(providers):
                provider_name = getattr(provider, "name")
                route = getattr(provider, "route")
                call_count = calls_by_provider.get(provider_name, 0)
                max_calls = getattr(provider, "max_control_calls", 1)
                if type(max_calls) is not int or max_calls < 1:
                    raise ValueError(
                        "provider max_control_calls must be a positive integer"
                    )
                if call_count >= max_calls:
                    continue
                has_unseen = getattr(provider, "has_unseen_candidates", None)
                if callable(has_unseen) and not has_unseen(blackboard):
                    continue
                if call_count and not bool(
                    getattr(provider, "supports_residual_actions", False)
                ):
                    continue
                operator, reasons, bonus = self._provider_operator(
                    route, best_signal, blackboard.features
                )
                parent_sensitive = getattr(
                    provider, "parent_sensitive_operators", None
                )
                parent_id = (
                    None
                    if best_signal is None
                    or (
                        parent_sensitive is not None
                        and operator not in parent_sensitive
                    )
                    else best_signal.hypothesis_id
                )
                evidence = () if best_signal is None else (best_signal.signal_id,)
                key = ("propose", provider_name, operator, parent_id)
                if key in attempted:
                    if not bool(getattr(provider, "supports_repeated_batches", False)):
                        continue
                    previous = tuple(
                        result
                        for result in blackboard.action_results
                        if self._attempt_key(result.action) == key
                    )
                    if not previous or (
                        previous[-1].status != "ok"
                        or not previous[-1].accepted_candidate_ids
                    ):
                        continue
                base = (
                    blackboard.route_decision.priority_for(route) * 100 + provider_index
                )
                actions.append(
                    ControlAction.create(
                        state_id=blackboard.state_id,
                        kind="propose",
                        actor=provider_name,
                        route=route,
                        operator=operator,
                        parent_hypothesis_id=parent_id,
                        evidence_signal_ids=evidence,
                        reason_codes=reasons,
                        budget=provider_slice,
                        priority=base + bonus,
                    )
                )

        if actions:
            stop_reasons = ("voluntary_stop_available",)
            stop_priority = 100_000
        else:
            exhausted = (
                remaining.compute_units < 2
                or remaining.controller_steps < 1
                or remaining.candidate_slots < 1
            )
            stop_reasons = (
                "compute_budget_exhausted" if exhausted else "no_legal_untried_action",
            )
            stop_priority = -10_000
        actions.append(
            ControlAction.create(
                state_id=blackboard.state_id,
                kind="stop",
                actor="controller",
                route=None,
                operator="stop",
                reason_codes=stop_reasons,
                priority=stop_priority,
            )
        )
        return tuple(sorted(actions, key=lambda item: (item.priority, item.action_id)))


def candidate_pool_batches(candidate_count: int, batch_size: int) -> int:
    """Public helper for constructing matched frozen-pool provider call caps."""

    if type(candidate_count) is not int or candidate_count < 0:
        raise ValueError("candidate_count must be non-negative")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    return math.ceil(candidate_count / batch_size)
