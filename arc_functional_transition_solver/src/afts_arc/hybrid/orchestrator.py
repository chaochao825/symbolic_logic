"""End-to-end route -> propose -> verify -> repair -> select orchestration."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from ..blind import BlindTask
from ..candidate import CandidateRecord
from .providers import (
    CodeModelProvider,
    DiffLogicHardProvider,
    DslProgramProvider,
    ExternalCallback,
    MaskedDiffusionProvider,
    SparseCAProvider,
)
from .repair import generate_repairs
from .router import ROUTES, RouteDecision, TaskFeatures, route_task
from .types import (
    CandidateEvaluation,
    CandidateProvider,
    ProviderResult,
    RepairReceipt,
)
from .verification import (
    evaluate_hypotheses,
    evaluate_hypotheses_isolated,
    rank_verified,
)


@dataclass(frozen=True, slots=True)
class FunctionalRouterConfig:
    max_selected_hypotheses: int = 2
    max_repair_parents: int = 8
    minimum_repair_agreement: float = 0.5

    def __post_init__(self) -> None:
        if self.max_selected_hypotheses < 1 or self.max_repair_parents < 0:
            raise ValueError("solver candidate limits must be non-negative")
        if not 0.0 <= self.minimum_repair_agreement <= 1.0:
            raise ValueError("minimum_repair_agreement must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SolveReport:
    blind_task_id: str
    blind_content_sha256: str
    features: TaskFeatures
    route_decision: RouteDecision
    provider_results: tuple[ProviderResult, ...]
    initial_evaluations: tuple[CandidateEvaluation, ...]
    repair_receipts: tuple[RepairReceipt, ...]
    repaired_evaluations: tuple[CandidateEvaluation, ...]
    selected: tuple[CandidateEvaluation, ...]

    @property
    def status(self) -> str:
        return "solved" if self.selected else "abstained"

    def candidate_records(self, *, task_id: str | None = None) -> tuple[CandidateRecord, ...]:
        output_task_id = task_id or self.blind_task_id
        records: list[CandidateRecord] = []
        evaluation_by_id = {
            item.hypothesis.hypothesis_id: item
            for item in (*self.initial_evaluations, *self.repaired_evaluations)
        }

        def program_hash(evaluation: CandidateEvaluation) -> str:
            hypothesis = evaluation.hypothesis
            if hypothesis.source == "typed_dsl":
                program = hypothesis.spec.get("program")
                if isinstance(program, dict) and isinstance(program.get("program_id"), str):
                    return program["program_id"]
            return hypothesis.hypothesis_id

        def parameters(evaluation: CandidateEvaluation, rank: int | None) -> dict[str, object]:
            hypothesis = evaluation.hypothesis
            return {
                "blind_content_sha256": self.blind_content_sha256,
                "bundle_hypothesis_id": hypothesis.hypothesis_id,
                "bundle_rank": rank,
                "route": hypothesis.route,
                "verification_mode": hypothesis.verification_mode,
                "hard_verified": evaluation.hard_verified,
                "demo_exact": evaluation.demo_exact,
                "mdl": {
                    "route_bits": evaluation.route_bits,
                    "model_bits": hypothesis.description_bits,
                    "residual_bits": evaluation.residual_bits,
                    "support_bits": evaluation.support_bits,
                    "total_bits": evaluation.total_mdl_bits,
                },
                "spec": hypothesis.spec,
                "metadata": hypothesis.metadata,
                "parent_hypothesis_ids": list(hypothesis.parent_hypothesis_ids),
            }

        for rank, evaluation in enumerate(self.selected):
            hypothesis = evaluation.hypothesis
            for test_index, output in enumerate(evaluation.query_outputs):
                if output is None:
                    raise AssertionError("selected hypothesis has a missing query output")
                parent_candidate_ids: list[str] = []
                parent_candidate_records: list[dict[str, object]] = []
                for parent_hypothesis_id in hypothesis.parent_hypothesis_ids:
                    parent = evaluation_by_id.get(parent_hypothesis_id)
                    if parent is None or parent.query_outputs[test_index] is None:
                        raise AssertionError("repair parent is absent from the hypothesis ledger")
                    parent_record = CandidateRecord.create(
                        task_id=output_task_id,
                        test_index=test_index,
                        source_type=parent.hypothesis.source,
                        source_version=parent.hypothesis.source_version,
                        output=parent.query_outputs[test_index],
                        program_hash=program_hash(parent),
                        functional_trace=parent.hypothesis.functional_trace,
                        generation_parameters=parameters(parent, None),
                        evidence_status="verified_repair_parent",
                    )
                    parent_candidate_ids.append(parent_record.candidate_id)
                    parent_candidate_records.append(parent_record.to_json_dict())
                generation_parameters = parameters(evaluation, rank)
                generation_parameters["parent_candidate_records"] = parent_candidate_records
                records.append(
                    CandidateRecord.create(
                        task_id=output_task_id,
                        test_index=test_index,
                        source_type=hypothesis.source,
                        source_version=hypothesis.source_version,
                        output=output,
                        parent_candidate_ids=parent_candidate_ids,
                        program_hash=program_hash(evaluation),
                        functional_trace=hypothesis.functional_trace,
                        generation_parameters=generation_parameters,
                        evidence_status="demo_exact_hard_verified",
                    )
                )
        return tuple(records)

    def lineage_candidate_records(
        self, *, task_id: str | None = None
    ) -> tuple[CandidateRecord, ...]:
        """Return a parent-first, ID-closed ledger for the selected submissions."""

        ordered: dict[str, CandidateRecord] = {}
        for child in self.candidate_records(task_id=task_id):
            parameters = json.loads(child.generation_parameters_json)
            raw_parents = parameters.get("parent_candidate_records", [])
            if not isinstance(raw_parents, list):
                raise ValueError("parent_candidate_records must be a JSON list")
            parents = tuple(CandidateRecord.from_json_dict(item) for item in raw_parents)
            if tuple(item.candidate_id for item in parents) != child.parent_candidate_ids:
                raise ValueError("embedded parent records do not match parent_candidate_ids")
            for record in (*parents, child):
                incumbent = ordered.get(record.candidate_id)
                if incumbent is not None and incumbent != record:
                    raise ValueError("candidate lineage ID has inconsistent record content")
                ordered.setdefault(record.candidate_id, record)

        available: set[str] = set()
        for record in ordered.values():
            if not set(record.parent_candidate_ids).issubset(available):
                raise ValueError("candidate lineage is not parent-first and ID-closed")
            available.add(record.candidate_id)
        return tuple(ordered.values())

    def to_json_dict(self, *, task_id: str | None = None) -> dict[str, object]:
        display_task_id = task_id or self.blind_task_id
        return {
            "schema": "afts.functional-router-report/v1",
            "task_id": display_task_id,
            "blind_task_id": self.blind_task_id,
            "blind_content_sha256": self.blind_content_sha256,
            "status": self.status,
            "oracle_policy": "train_outputs_and_test_inputs_only",
            "features": self.features.to_json_dict(),
            "router": self.route_decision.to_json_dict(),
            "providers": [item.to_json_dict() for item in self.provider_results],
            "initial_evaluations": [item.to_json_dict() for item in self.initial_evaluations],
            "repairs": [item.to_json_dict() for item in self.repair_receipts],
            "repaired_evaluations": [item.to_json_dict() for item in self.repaired_evaluations],
            "selected_hypothesis_ids": [item.hypothesis.hypothesis_id for item in self.selected],
            "selected": [item.to_json_dict() for item in self.selected],
            "candidate_records": [
                item.to_json_dict() for item in self.candidate_records(task_id=display_task_id)
            ],
            "lineage_candidate_records": [
                item.to_json_dict()
                for item in self.lineage_candidate_records(task_id=display_task_id)
            ],
        }


class FunctionalRouterSolver:
    def __init__(
        self,
        *,
        providers: Sequence[CandidateProvider] | None = None,
        config: FunctionalRouterConfig | None = None,
        masked_diffusion_callback: ExternalCallback | None = None,
        code_model_callback: ExternalCallback | None = None,
        difflogic_hard_callback: ExternalCallback | None = None,
    ) -> None:
        self.config = config or FunctionalRouterConfig()
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
            raise ValueError("solver requires at least one candidate provider")
        for provider in self.providers:
            if not callable(getattr(provider, "propose", None)):
                raise TypeError("providers must implement propose")
            if not isinstance(getattr(provider, "name", None), str) or not provider.name:
                raise TypeError("providers require a non-empty name")
            if not isinstance(getattr(provider, "route", None), str) or not provider.route:
                raise TypeError("providers require a non-empty route")
            if provider.route not in ROUTES:
                raise ValueError("provider route is not a registered router specialist")
        provider_names = tuple(provider.name for provider in self.providers)
        if len(set(provider_names)) != len(provider_names):
            raise ValueError("provider names must be unique for unambiguous receipts")

    def solve(self, task: BlindTask) -> SolveReport:
        if not isinstance(task, BlindTask):
            raise TypeError("FunctionalRouterSolver accepts BlindTask only")
        features, decision = route_task(task)
        ordered_providers = tuple(
            sorted(
                enumerate(self.providers),
                key=lambda item: (
                    decision.priority_for(item[1].route),
                    item[0],
                ),
            )
        )
        provider_results: list[ProviderResult] = []
        candidates = []
        for _, provider in ordered_providers:
            try:
                result = provider.propose(task, features, decision)
                if not isinstance(result, ProviderResult):
                    raise TypeError("provider returned a non-ProviderResult value")
                if result.provider != provider.name or result.route != provider.route:
                    raise ValueError("provider result identity does not match its provider")
                if any(item.route != provider.route for item in result.candidates):
                    raise ValueError("provider candidate route does not match its provider")
                if result.status == "ok":
                    reported_count = len(result.candidates)
                    normalized = tuple(
                        sorted(
                            result.candidates,
                            key=lambda item: (item.description_bits, item.hypothesis_id),
                        )[: decision.budget_for(provider.route)]
                    )
                    diagnostics = result.diagnostics
                    diagnostics.update(
                        {
                            "central_budget": decision.budget_for(provider.route),
                            "reported_candidate_count": reported_count,
                        }
                    )
                    result = ProviderResult.ok(
                        provider.name,
                        provider.route,
                        normalized,
                        diagnostics,
                    )
            except Exception as exc:
                result = ProviderResult.error(
                    provider.name,
                    provider.route,
                    "provider_exception",
                    {"error_type": type(exc).__name__, "message": str(exc)},
                )
            provider_results.append(result)
            candidates.extend(result.candidates)

        initial_evaluations, collisions = evaluate_hypotheses_isolated(candidates, task)
        if collisions:
            for index, result in enumerate(provider_results):
                quarantined = tuple(
                    item.hypothesis_id
                    for item in result.candidates
                    if item.hypothesis_id in collisions
                )
                if not quarantined:
                    continue
                clean = tuple(
                    item
                    for item in result.candidates
                    if item.hypothesis_id not in collisions
                )
                diagnostics = result.diagnostics
                diagnostics.update(
                    {
                        "semantic_artifact_conflict": True,
                        "quarantined_candidate_ids": sorted(set(quarantined)),
                        "collision_reasons": {
                            hypothesis_id: collisions[hypothesis_id]
                            for hypothesis_id in sorted(set(quarantined))
                        },
                    }
                )
                provider_results[index] = (
                    ProviderResult.ok(result.provider, result.route, clean, diagnostics)
                    if clean
                    else ProviderResult.error(
                        result.provider,
                        result.route,
                        "semantic_artifact_conflict",
                        diagnostics,
                    )
                )
        repaired, receipts = generate_repairs(
            initial_evaluations,
            task,
            max_parents=self.config.max_repair_parents,
            minimum_agreement=self.config.minimum_repair_agreement,
        )
        repaired_evaluations = evaluate_hypotheses(repaired, task)
        selected = rank_verified(
            (*initial_evaluations, *repaired_evaluations),
            limit=self.config.max_selected_hypotheses,
        )
        return SolveReport(
            blind_task_id=task.task_id,
            blind_content_sha256=task.blind_content_sha256,
            features=features,
            route_decision=decision,
            provider_results=tuple(provider_results),
            initial_evaluations=initial_evaluations,
            repair_receipts=receipts,
            repaired_evaluations=repaired_evaluations,
            selected=selected,
        )
