"""Stateful, budgeted, residual-driven orchestration for hybrid ARC solving."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..blind import BlindTask
from ..candidate import CandidateRecord
from .control import (
    ActionResult,
    Blackboard,
    BudgetLedger,
    BudgetVector,
    ControlAction,
    ControlPolicy,
    ResidualActionCompiler,
    ResidualCompilerConfig,
    ResidualFirstPolicy,
)
from .orchestrator import SolveReport
from .metareasoning import (
    NativeBudgetLedger,
    NativeCostContract,
    NativeCostVector,
)
from .providers import (
    CodeModelProvider,
    DiffLogicHardProvider,
    DslProgramProvider,
    ExternalCallback,
    MaskedDiffusionProvider,
    SparseCAProvider,
)
from .repair import global_color_map_repair, local_transition_repair
from .router import ROUTES, route_task
from .types import (
    CandidateEvaluation,
    CandidateHypothesis,
    CandidateProvider,
    ProviderResult,
    RepairReceipt,
    canonical_json,
)
from .verification import evaluate_hypotheses_isolated, rank_verified


_STRICT_PROVIDER_INTEGRITY_FAILURES = frozenset(
    {"budget_contract_violation", "provider_exception"}
)
_REPAIR_NATIVE_COSTS = {
    "residual_repair.native_cost.option_calls": 1.0,
    "residual_repair.native_cost.repair_attempts": 1.0,
}


def _provider_native_cost(result: ProviderResult | None) -> NativeCostVector | None:
    """Return a valid explicit observation, never an implicit zero fallback."""

    if result is None:
        return None
    diagnostics = result.diagnostics
    if "native_cost" not in diagnostics:
        return None
    raw_cost = diagnostics["native_cost"]
    if not isinstance(raw_cost, Mapping):
        return None
    try:
        return NativeCostVector.from_mapping(raw_cost)
    except (TypeError, ValueError):
        return None


def _strict_proposal_integrity(result: ActionResult) -> bool:
    if result.budget_fidelity != "strict_replayable":
        return False
    if result.reason in _STRICT_PROVIDER_INTEGRITY_FAILURES:
        return False
    provider_result = result.provider_result
    if provider_result is None:
        return False
    diagnostics = provider_result.diagnostics
    if "budget_contract_violation" in diagnostics:
        return False
    return _provider_native_cost(provider_result) is not None


def _declared_repair_native_cost(
    reservation: NativeCostVector,
) -> NativeCostVector:
    """Project fixed repair work onto dimensions declared by the contract."""

    declared = reservation.to_mapping()
    return NativeCostVector(
        tuple(
            (key, value)
            for key, value in _REPAIR_NATIVE_COSTS.items()
            if key in declared
        )
    )


def _declared_proposal_native_cost(
    result: ProviderResult | None,
    reservation: NativeCostVector,
    *,
    actor: str,
) -> NativeCostVector | None:
    """Add the known one-call action counter when its dimension is declared."""

    observed = _provider_native_cost(result)
    if observed is None:
        return None
    option_key = f"{actor}.native_cost.option_calls"
    if option_key not in reservation.to_mapping():
        return observed
    actual = observed.to_mapping()
    actual[option_key] = max(actual.get(option_key, 0.0), 1.0)
    return NativeCostVector(tuple(actual.items()))


@dataclass(frozen=True, slots=True)
class OnlineControlConfig:
    """Configuration shared by the controller and matched-budget baselines."""

    budget_limit: BudgetVector = BudgetVector(
        compute_units=96,
        controller_steps=16,
        provider_calls=8,
        repair_attempts=8,
        candidate_slots=80,
    )
    provider_batch_size: int = 4
    max_selected_hypotheses: int = 2
    minimum_repair_agreement: float = 0.5
    localized_residual_fraction: float = 0.35
    native_cost_contract: NativeCostContract | None = None
    native_budget_limit: NativeCostVector | None = None

    def __post_init__(self) -> None:
        if type(self.provider_batch_size) is not int or self.provider_batch_size < 1:
            raise ValueError("provider_batch_size must be positive")
        if (
            type(self.max_selected_hypotheses) is not int
            or self.max_selected_hypotheses < 1
        ):
            raise ValueError("max_selected_hypotheses must be positive")
        for name in ("minimum_repair_agreement", "localized_residual_fraction"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if (self.native_cost_contract is None) != (self.native_budget_limit is None):
            raise ValueError(
                "native cost contract and budget limit must be configured together"
            )


@dataclass(frozen=True, slots=True)
class OnlineSolveReport:
    """Final legacy-compatible result plus the complete controller trajectory."""

    base_report: SolveReport
    policy_name: str
    states: tuple[Blackboard, ...]
    native_budget: NativeBudgetLedger | None = None
    native_reservation_violation_count: int = 0
    native_masked_action_count: int = 0

    def __post_init__(self) -> None:
        if not self.states or not self.states[-1].stopped:
            raise ValueError("online report requires a terminal blackboard")
        if self.states[0].action_results:
            raise ValueError("initial blackboard must precede all actions")
        for name in (
            "native_reservation_violation_count",
            "native_masked_action_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for previous, current in zip(self.states, self.states[1:]):
            if len(current.action_results) != len(previous.action_results) + 1:
                raise ValueError(
                    "every blackboard transition must append one action result"
                )
            if current.action_results[:-1] != previous.action_results:
                raise ValueError("blackboard action history is not append-only")
            if current.action_results[-1].action.state_id != previous.state_id:
                raise ValueError("action is not bound to its predecessor blackboard")

    @property
    def status(self) -> str:
        return self.base_report.status

    @property
    def selected(self) -> tuple[CandidateEvaluation, ...]:
        return self.base_report.selected

    @property
    def provider_results(self) -> tuple[ProviderResult, ...]:
        return self.base_report.provider_results

    @property
    def initial_evaluations(self) -> tuple[CandidateEvaluation, ...]:
        return self.base_report.initial_evaluations

    @property
    def repaired_evaluations(self) -> tuple[CandidateEvaluation, ...]:
        return self.base_report.repaired_evaluations

    @property
    def repair_receipts(self) -> tuple[RepairReceipt, ...]:
        return self.base_report.repair_receipts

    @property
    def strict_budget_comparable(self) -> bool:
        proposal_results = tuple(
            item
            for item in self.states[-1].action_results
            if item.action.kind == "propose"
        )
        return bool(proposal_results) and all(
            _strict_proposal_integrity(item) for item in proposal_results
        )

    @property
    def native_budget_comparable(self) -> bool:
        return (
            self.native_budget is not None
            and self.strict_budget_comparable
            and self.native_reservation_violation_count == 0
        )

    def candidate_records(
        self, *, task_id: str | None = None
    ) -> tuple[CandidateRecord, ...]:
        return self.base_report.candidate_records(task_id=task_id)

    def lineage_candidate_records(
        self, *, task_id: str | None = None
    ) -> tuple[CandidateRecord, ...]:
        return self.base_report.lineage_candidate_records(task_id=task_id)

    def to_json_dict(self, *, task_id: str | None = None) -> dict[str, object]:
        report = self.base_report.to_json_dict(task_id=task_id)
        report["schema"] = "afts.online-functional-router-report/v1"
        final = self.states[-1]
        action_results = final.action_results
        actual_evaluations = sum(
            item.actual_candidate_evaluations for item in action_results
        )
        reserved_slots = final.budget.used.candidate_slots
        repaired_eligible = sum(item.eligible for item in self.repaired_evaluations)
        used_compute = final.budget.used.compute_units
        report["control"] = {
            "policy": self.policy_name,
            "budget_contract": {
                "unit": "normalized_compute_unit",
                "strict_budget_comparable": self.strict_budget_comparable,
                "strict_claim_scope": "frozen/replayable providers declaring strict_budget_contract",
                "legacy_provider_scope": "controller-only; not FLOP/wall-clock matched",
            },
            "native_budget_contract": {
                "active": self.native_budget is not None,
                "strict_native_budget_comparable": self.native_budget_comparable,
                "reservation_violation_count": (
                    self.native_reservation_violation_count
                ),
                "masked_action_count": self.native_masked_action_count,
                "ledger": (
                    None
                    if self.native_budget is None
                    else self.native_budget.to_json_dict()
                ),
            },
            "initial_state_id": self.states[0].state_id,
            "final_state_id": final.state_id,
            "stop_reason": final.stop_reason,
            "budget": final.budget.to_json_dict(),
            "trajectory": [
                item.to_json_dict(selected_limit=len(self.selected) or 1)
                for item in self.states
            ],
            "action_results": [item.to_json_dict() for item in action_results],
            "metrics": {
                "reserved_candidate_slots": reserved_slots,
                "actual_candidate_evaluations": actual_evaluations,
                "candidate_slot_utilization": (
                    actual_evaluations / reserved_slots if reserved_slots else 0.0
                ),
                "eligible_repair_count": repaired_eligible,
                "eligible_repairs_per_compute_unit": (
                    repaired_eligible / used_compute if used_compute else 0.0
                ),
                "selected_distinct_query_bundles": len(self.selected),
                "oracle_accessed_by_controller": False,
            },
        }
        return report


def _evaluation_semantics(evaluation: CandidateEvaluation) -> tuple[object, ...]:
    return (
        evaluation.demo_outputs,
        evaluation.query_outputs,
        evaluation.hard_verified,
        evaluation.rejection_reason,
    )


def _merge_candidate_batch(
    blackboard: Blackboard,
    candidates: Sequence[CandidateHypothesis],
    task: BlindTask,
) -> tuple[
    tuple[CandidateHypothesis, ...],
    tuple[CandidateEvaluation, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Evaluate only the new batch, then quarantine conflicts and descendants."""

    submitted = tuple(candidates)
    new_evaluations, within_batch_collisions = evaluate_hypotheses_isolated(
        submitted, task
    )
    candidate_by_id = {item.hypothesis_id: item for item in blackboard.candidates}
    evaluation_by_id = {
        item.hypothesis.hypothesis_id: item for item in blackboard.evaluations
    }
    conflicts = dict(within_batch_collisions)
    accepted: set[str] = set()

    for evaluation in new_evaluations:
        hypothesis_id = evaluation.hypothesis.hypothesis_id
        incumbent = evaluation_by_id.get(hypothesis_id)
        if incumbent is None:
            candidate_by_id[hypothesis_id] = evaluation.hypothesis
            evaluation_by_id[hypothesis_id] = evaluation
            accepted.add(hypothesis_id)
            continue
        canonical_conflict = canonical_json(
            incumbent.hypothesis.to_json_dict()
        ) != canonical_json(evaluation.hypothesis.to_json_dict())
        semantic_conflict = _evaluation_semantics(incumbent) != _evaluation_semantics(
            evaluation
        )
        if canonical_conflict or semantic_conflict:
            conflicts[hypothesis_id] = (
                "canonical_payload_conflict"
                if canonical_conflict
                else "replay_semantics_conflict"
            )

    quarantined = set(blackboard.quarantined_candidate_ids) | set(conflicts)
    changed = True
    while changed:
        changed = False
        for hypothesis_id, candidate in tuple(candidate_by_id.items()):
            if hypothesis_id in quarantined:
                continue
            if set(candidate.parent_hypothesis_ids) & quarantined:
                quarantined.add(hypothesis_id)
                changed = True

    for hypothesis_id in quarantined:
        candidate_by_id.pop(hypothesis_id, None)
        evaluation_by_id.pop(hypothesis_id, None)
        accepted.discard(hypothesis_id)

    active_ids = tuple(sorted(candidate_by_id))
    return (
        tuple(candidate_by_id[item] for item in active_ids),
        tuple(evaluation_by_id[item] for item in active_ids),
        tuple(sorted(accepted)),
        tuple(sorted(quarantined)),
    )


def _validated_provider_result(
    provider: object,
    result: object,
    action: ControlAction,
    *,
    strict_budget_contract: bool,
) -> ProviderResult:
    if not isinstance(result, ProviderResult):
        raise TypeError("provider returned a non-ProviderResult value")
    if result.provider != getattr(provider, "name") or result.route != getattr(
        provider, "route"
    ):
        raise ValueError("provider result identity does not match its provider")
    if any(item.route != result.route for item in result.candidates):
        raise ValueError("provider candidate route does not match its provider")
    if result.status != "ok":
        return result

    reported = len(result.candidates)
    diagnostics = result.diagnostics
    diagnostics.update(
        {
            "online_action_id": action.action_id,
            "central_candidate_slots": action.budget.candidate_slots,
            "reported_candidate_count": reported,
            "strict_budget_contract": strict_budget_contract,
        }
    )
    if strict_budget_contract and reported > action.budget.candidate_slots:
        diagnostics["budget_contract_violation"] = "candidate_slot_overrun"
        return ProviderResult.error(
            result.provider,
            result.route,
            "budget_contract_violation",
            diagnostics,
        )
    normalized = tuple(
        sorted(
            result.candidates,
            key=lambda item: (item.description_bits, item.hypothesis_id),
        )[: action.budget.candidate_slots]
    )
    return ProviderResult.ok(result.provider, result.route, normalized, diagnostics)


class OnlineFunctionalRouterSolver:
    """Execute one typed action at a time and recompile after every residual."""

    def __init__(
        self,
        *,
        providers: Sequence[CandidateProvider] | None = None,
        config: OnlineControlConfig | None = None,
        policy: ControlPolicy | None = None,
        masked_diffusion_callback: ExternalCallback | None = None,
        code_model_callback: ExternalCallback | None = None,
        difflogic_hard_callback: ExternalCallback | None = None,
    ) -> None:
        self.config = config or OnlineControlConfig()
        self.policy = policy or ResidualFirstPolicy()
        self.providers = tuple(
            providers
            if providers is not None
            else (
                DslProgramProvider(),
                SparseCAProvider(),
                CodeModelProvider(code_model_callback),
                DiffLogicHardProvider(difflogic_hard_callback),
                MaskedDiffusionProvider(masked_diffusion_callback),
            )
        )
        if not self.providers:
            raise ValueError("online solver requires at least one provider")
        for provider in self.providers:
            if not callable(getattr(provider, "propose", None)) and not callable(
                getattr(provider, "act", None)
            ):
                raise TypeError("providers must implement propose or act")
            if (
                not isinstance(getattr(provider, "name", None), str)
                or not provider.name
            ):
                raise TypeError("providers require a non-empty name")
            if (
                not isinstance(getattr(provider, "route", None), str)
                or not provider.route
            ):
                raise TypeError("providers require a non-empty route")
            if provider.route not in ROUTES:
                raise ValueError("provider route is not a registered specialist")
        names = tuple(provider.name for provider in self.providers)
        if len(set(names)) != len(names):
            raise ValueError("provider names must be unique")
        self.compiler = ResidualActionCompiler(
            ResidualCompilerConfig(
                provider_batch_size=self.config.provider_batch_size,
                minimum_repair_agreement=self.config.minimum_repair_agreement,
                localized_residual_fraction=self.config.localized_residual_fraction,
            )
        )

    def _provider_action(
        self,
        task: BlindTask,
        blackboard: Blackboard,
        action: ControlAction,
    ) -> tuple[
        ProviderResult,
        ActionResult,
        tuple[CandidateHypothesis, ...],
        tuple[CandidateEvaluation, ...],
        tuple[str, ...],
    ]:
        provider = next(item for item in self.providers if item.name == action.actor)
        strict = bool(getattr(provider, "strict_budget_contract", False))
        fidelity = "strict_replayable" if strict else "controller_only"
        try:
            act = getattr(provider, "act", None)
            if callable(act):
                raw = act(
                    task,
                    blackboard.features,
                    blackboard.route_decision,
                    blackboard,
                    action,
                )
            else:
                raw = provider.propose(
                    task, blackboard.features, blackboard.route_decision
                )
            result = _validated_provider_result(
                provider,
                raw,
                action,
                strict_budget_contract=strict,
            )
        except Exception as exc:
            result = ProviderResult.error(
                provider.name,
                provider.route,
                "provider_exception",
                {"error_type": type(exc).__name__, "message": str(exc)},
            )

        submitted = result.candidates
        candidates, evaluations, accepted, quarantined = _merge_candidate_batch(
            blackboard, submitted, task
        )
        newly_quarantined = tuple(
            sorted(set(quarantined) - set(blackboard.quarantined_candidate_ids))
        )
        if newly_quarantined:
            clean = tuple(
                item
                for item in result.candidates
                if item.hypothesis_id not in newly_quarantined
            )
            diagnostics = result.diagnostics
            diagnostics.update(
                {
                    "semantic_artifact_conflict": True,
                    "quarantined_candidate_ids": list(newly_quarantined),
                }
            )
            result = (
                ProviderResult.ok(result.provider, result.route, clean, diagnostics)
                if clean
                else ProviderResult.error(
                    result.provider,
                    result.route,
                    "semantic_artifact_conflict",
                    diagnostics,
                )
            )
        status = result.status
        receipt = ActionResult.create(
            action=action,
            status=status,
            reason=result.reason,
            budget_fidelity=fidelity,
            emitted_candidate_ids=(item.hypothesis_id for item in submitted),
            accepted_candidate_ids=accepted,
            quarantined_candidate_ids=newly_quarantined,
            actual_candidate_evaluations=len(submitted),
            provider_result=result,
        )
        return result, receipt, candidates, evaluations, quarantined

    def _repair_action(
        self,
        task: BlindTask,
        blackboard: Blackboard,
        action: ControlAction,
    ) -> tuple[
        RepairReceipt,
        ActionResult,
        tuple[CandidateHypothesis, ...],
        tuple[CandidateEvaluation, ...],
        tuple[str, ...],
    ]:
        evaluation = next(
            (
                item
                for item in blackboard.evaluations
                if item.hypothesis.hypothesis_id == action.parent_hypothesis_id
            ),
            None,
        )
        if evaluation is None:
            repair_receipt = RepairReceipt(
                action.parent_hypothesis_id or "missing",
                action.operator,
                "abstained",
                "parent_not_on_blackboard",
            )
            produced: tuple[CandidateHypothesis, ...] = ()
        else:
            strategy = {
                "global_color_map": global_color_map_repair,
                "local_transition": local_transition_repair,
            }[action.operator]
            produced, repair_receipt = strategy(evaluation, task)
        if len(produced) > action.budget.candidate_slots:
            raise RuntimeError("repair operator exceeded its typed candidate slot")
        candidates, evaluations, accepted, quarantined = _merge_candidate_batch(
            blackboard, produced, task
        )
        newly_quarantined = tuple(
            sorted(set(quarantined) - set(blackboard.quarantined_candidate_ids))
        )
        status = "ok" if repair_receipt.status == "produced" else "abstained"
        result = ActionResult.create(
            action=action,
            status=status,
            reason=repair_receipt.reason,
            budget_fidelity="strict_replayable",
            emitted_candidate_ids=(item.hypothesis_id for item in produced),
            accepted_candidate_ids=accepted,
            quarantined_candidate_ids=newly_quarantined,
            actual_candidate_evaluations=len(produced),
            repair_receipt=repair_receipt,
        )
        return repair_receipt, result, candidates, evaluations, quarantined

    def solve(self, task: BlindTask) -> OnlineSolveReport:
        if not isinstance(task, BlindTask):
            raise TypeError("OnlineFunctionalRouterSolver accepts BlindTask only")
        features, decision = route_task(task)
        blackboard = Blackboard.create(
            task=task,
            features=features,
            route_decision=decision,
            budget=BudgetLedger(self.config.budget_limit),
            localized_fraction=self.config.localized_residual_fraction,
        )
        states = [blackboard]
        native_contract = self.config.native_cost_contract
        native_accounting_active = (
            native_contract is not None
            and self.config.native_budget_limit is not None
            and all(
                bool(getattr(provider, "strict_budget_contract", False))
                for provider in self.providers
            )
        )
        native_budget = (
            NativeBudgetLedger(self.config.native_budget_limit)
            if native_accounting_active
            else None
        )
        native_violations = 0
        native_masked_actions = 0

        while not blackboard.stopped:
            actions = self.compiler.compile(
                task,
                blackboard,
                self.providers,
                max_selected_hypotheses=self.config.max_selected_hypotheses,
            )
            reservations: dict[str, NativeCostVector] = {}
            if native_budget is not None and native_contract is not None:
                compiled_non_stop = tuple(
                    item for item in actions if item.kind != "stop"
                )
                affordable: list[ControlAction] = []
                for candidate_action in compiled_non_stop:
                    reservation = native_contract.reservation_for(
                        candidate_action.actor, candidate_action.operator
                    )
                    if reservation is None or not native_budget.can_reserve(
                        reservation
                    ):
                        native_masked_actions += 1
                        continue
                    reservations[candidate_action.action_id] = reservation
                    affordable.append(candidate_action)
                if compiled_non_stop and not affordable:
                    actions = (
                        ControlAction.create(
                            state_id=blackboard.state_id,
                            kind="stop",
                            actor="controller",
                            route=None,
                            operator="stop",
                            reason_codes=("native_budget_exhausted",),
                            priority=-10_000,
                        ),
                    )
                else:
                    actions = (
                        *affordable,
                        *(item for item in actions if item.kind == "stop"),
                    )
            action = self.policy.select(blackboard, actions)
            compiled_by_id = {item.action_id: item for item in actions}
            if (
                action.action_id not in compiled_by_id
                or compiled_by_id[action.action_id] != action
            ):
                raise ValueError(
                    "control policy returned an action outside the legal mask"
                )
            if action.state_id != blackboard.state_id:
                raise ValueError("control policy returned an action for a stale state")

            if action.kind == "stop":
                result = ActionResult.create(
                    action=action,
                    status="stopped",
                    reason=action.reason_codes[0] if action.reason_codes else None,
                    budget_fidelity="not_applicable",
                )
                blackboard = Blackboard.create(
                    task=task,
                    features=features,
                    route_decision=decision,
                    budget=blackboard.budget,
                    candidates=blackboard.candidates,
                    evaluations=blackboard.evaluations,
                    provider_results=blackboard.provider_results,
                    repair_receipts=blackboard.repair_receipts,
                    action_results=(*blackboard.action_results, result),
                    quarantined_candidate_ids=blackboard.quarantined_candidate_ids,
                    localized_fraction=self.config.localized_residual_fraction,
                    stopped=True,
                    stop_reason=result.reason,
                )
                states.append(blackboard)
                break

            if not blackboard.budget.can_reserve(action.budget):
                raise AssertionError(
                    "compiler emitted an action outside the budget mask"
                )
            charged = blackboard.budget.charge(action.budget)
            native_reservation = reservations.get(action.action_id)
            if native_budget is not None:
                if native_reservation is None:
                    raise AssertionError(
                        "native-budget mask emitted an action without a reservation"
                    )
                native_budget = native_budget.charge(native_reservation)
            if action.kind == "propose":
                provider_result, result, candidates, evaluations, quarantined = (
                    self._provider_action(task, blackboard, action)
                )
                if native_reservation is not None:
                    actual_native = _declared_proposal_native_cost(
                        provider_result,
                        native_reservation,
                        actor=action.actor,
                    )
                    if actual_native is None or not actual_native.fits_within(
                        native_reservation
                    ):
                        native_violations += 1
                provider_results = (*blackboard.provider_results, provider_result)
                repair_receipts = blackboard.repair_receipts
            else:
                repair_receipt, result, candidates, evaluations, quarantined = (
                    self._repair_action(task, blackboard, action)
                )
                provider_results = blackboard.provider_results
                repair_receipts = (*blackboard.repair_receipts, repair_receipt)
                if native_reservation is not None:
                    actual_native = _declared_repair_native_cost(native_reservation)
                    if not actual_native.fits_within(native_reservation):
                        native_violations += 1

            blackboard = Blackboard.create(
                task=task,
                features=features,
                route_decision=decision,
                budget=charged,
                candidates=candidates,
                evaluations=evaluations,
                provider_results=provider_results,
                repair_receipts=repair_receipts,
                action_results=(*blackboard.action_results, result),
                quarantined_candidate_ids=quarantined,
                localized_fraction=self.config.localized_residual_fraction,
            )
            states.append(blackboard)

        final = states[-1]
        initial_evaluations = tuple(
            item
            for item in final.evaluations
            if item.hypothesis.route != "residual_repair"
        )
        repaired_evaluations = tuple(
            item
            for item in final.evaluations
            if item.hypothesis.route == "residual_repair"
        )
        selected = rank_verified(
            final.evaluations,
            limit=self.config.max_selected_hypotheses,
        )
        base = SolveReport(
            blind_task_id=task.task_id,
            blind_content_sha256=task.blind_content_sha256,
            features=features,
            route_decision=decision,
            provider_results=final.provider_results,
            initial_evaluations=initial_evaluations,
            repair_receipts=final.repair_receipts,
            repaired_evaluations=repaired_evaluations,
            selected=selected,
        )
        return OnlineSolveReport(
            base,
            self.policy.name,
            tuple(states),
            native_budget,
            native_violations,
            native_masked_actions,
        )
