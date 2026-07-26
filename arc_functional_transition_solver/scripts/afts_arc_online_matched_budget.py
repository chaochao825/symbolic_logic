"""Build and evaluate strict action-frozen ARC controller portfolios.

The script has two oracle-separated phases per task:

1. Discover action-conditioned grid-DSL/scene-DSL/CA batches without consulting
   test outputs.
2. Freeze their ``(provider, operator, parent) -> candidates`` mapping, replay
   every policy under one controller budget, and only then score with oracles.

No model training, model download, or remote model call is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _bootstrap() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    repository_root = project_root.parent
    for source_root in (repository_root / "src", project_root / "src"):
        resolved = str(source_root.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    return repository_root


REPOSITORY_ROOT = _bootstrap()

from afts_arc.blind import BlindTask  # noqa: E402
from afts_arc.experiment_safety import (  # noqa: E402
    ARC_EXPERIMENT_SOURCE_PATHS,
    ExperimentSafetyError,
    TaskFingerprint,
    atomic_write_json,
    candidate_dag_id,
    capture_git_source_provenance,
    file_sha256,
    runtime_metadata,
    validate_native_budget_profile,
    validate_pool_manifest,
    validate_summary,
    verify_source_provenance_unchanged,
)
from afts_arc.hybrid import (  # noqa: E402
    BudgetVector,
    CoverageAwareResidualPolicy,
    DeliberationSketch,
    DeterministicRandomPolicy,
    DslProgramProvider,
    FixedSchedulePolicy,
    FrozenActionBatch,
    FrozenCandidatePoolProvider,
    NativeCostContract,
    NativeCostVector,
    OnlineControlConfig,
    OnlineFunctionalRouterSolver,
    ResidualFirstPolicy,
    RoundRobinPolicy,
    SceneProgramProvider,
    SparseCAProvider,
    StaticRoutePolicy,
    StructuredDeliberationPolicy,
    aggregate_control_metrics,
    evaluate_online_report_with_oracle,
)
from afts_arc.hybrid.control import ControlPolicy  # noqa: E402
from afts_arc.hybrid.online import OnlineSolveReport  # noqa: E402
from afts_arc.hybrid.types import (  # noqa: E402
    CandidateHypothesis,
    ProviderResult,
    canonical_json,
)
from afts_arc.search import SearchConfig  # noqa: E402
from afts_arc.task import ARCTask, load_task_directory  # noqa: E402


SCHEMA_VERSION = "afts.online-matched-budget/v3"


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _merge_candidates(
    groups: Iterable[Iterable[CandidateHypothesis]],
) -> tuple[CandidateHypothesis, ...]:
    candidates: dict[str, CandidateHypothesis] = {}
    payloads: dict[str, str] = {}
    for group in groups:
        for candidate in group:
            payload = canonical_json(candidate.to_json_dict())
            incumbent = payloads.get(candidate.hypothesis_id)
            if incumbent is not None and incumbent != payload:
                raise ValueError(
                    "candidate ID collision while constructing frozen pool"
                )
            payloads[candidate.hypothesis_id] = payload
            candidates[candidate.hypothesis_id] = candidate
    return tuple(candidates[key] for key in sorted(candidates))


def _strict_native_cost(result: ProviderResult, *, actor: str) -> NativeCostVector:
    diagnostics = result.diagnostics
    if "native_cost" not in diagnostics:
        raise ExperimentSafetyError(
            f"missing actual native cost for frozen action {actor}"
        )
    return NativeCostVector.from_mapping(
        diagnostics["native_cost"], prefix=f"{actor}.native_cost"
    )


class RecordingProvider:
    """Cache one live provider result per typed action key for later replay."""

    strict_budget_contract = False

    def __init__(self, delegate: object) -> None:
        self.delegate = delegate
        self.name = str(getattr(delegate, "name"))
        self.route = str(getattr(delegate, "route"))
        self.supports_residual_actions = bool(
            getattr(delegate, "supports_residual_actions", False)
        )
        self.supports_repeated_batches = bool(
            getattr(delegate, "supports_repeated_batches", False)
        )
        self.max_control_calls = int(getattr(delegate, "max_control_calls", 1))
        self.parent_sensitive_operators = getattr(
            delegate, "parent_sensitive_operators", None
        )
        self.cache: dict[tuple[str, str | None], ProviderResult] = {}

    def act(self, task, features, decision, blackboard, action) -> ProviderResult:
        key = (action.operator, action.parent_hypothesis_id)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        act = getattr(self.delegate, "act", None)
        try:
            if callable(act):
                result = act(task, features, decision, blackboard, action)
            else:
                result = self.delegate.propose(task, features, decision)
        except Exception as exc:
            self.cache[key] = ProviderResult.error(
                self.name,
                self.route,
                "provider_exception",
                {
                    "exception_type": type(exc).__name__,
                    "native_cost_status": "unknown",
                },
            )
            raise
        if not isinstance(result, ProviderResult):
            raise TypeError("recording provider received a non-ProviderResult")
        self.cache[key] = result
        return result

    def frozen(self) -> FrozenCandidatePoolProvider:
        batches = tuple(
            FrozenActionBatch(
                operator,
                result.candidates,
                parent,
                _strict_native_cost(result, actor=self.name),
            )
            for (operator, parent), result in sorted(
                self.cache.items(), key=lambda item: (item[0][0], item[0][1] or "")
            )
        )
        return FrozenCandidatePoolProvider.from_action_batches(
            batches, self.name, self.route
        )

    def manifest(self) -> dict[str, object]:
        frozen = self.frozen()
        return {
            "provider": self.name,
            "route": self.route,
            "strict_live_budget_contract": False,
            "frozen_action_batches": [
                batch.to_json_dict() for batch in frozen.action_batches
            ],
            "actions": [
                {
                    "operator": operator,
                    "parent_hypothesis_id": parent,
                    "status": result.status,
                    "reason": result.reason,
                    "candidate_ids": [item.hypothesis_id for item in result.candidates],
                    "diagnostics": result.diagnostics,
                }
                for (operator, parent), result in sorted(
                    self.cache.items(),
                    key=lambda item: (item[0][0], item[0][1] or ""),
                )
            ],
        }


@dataclass(frozen=True, slots=True)
class PolicySpec:
    name: str
    factory: Callable[[], ControlPolicy]
    provider_names: tuple[str, ...]
    pool_scope: str


def _policy_specs() -> tuple[PolicySpec, ...]:
    dsl = "typed_dsl"
    ca = "sparse_ca_d4_bgpad"
    scene = "scene_predicate_dsl"
    both = (dsl, ca)
    expanded = (dsl, ca, scene)
    return (
        PolicySpec(
            "coverage_aware_v2",
            lambda: CoverageAwareResidualPolicy(name="coverage_aware_v2"),
            both,
            "base_union",
        ),
        PolicySpec(
            "coverage_aware_expanded",
            lambda: CoverageAwareResidualPolicy(name="coverage_aware_expanded"),
            expanded,
            "heterogeneous_union",
        ),
        PolicySpec(
            "summary_grounded",
            lambda: StructuredDeliberationPolicy(
                name="summary_grounded",
                use_grounding=True,
                use_adaptive_diversity=False,
                use_borderline_ucb=False,
            ),
            expanded,
            "heterogeneous_union",
        ),
        PolicySpec(
            "static_grounding_only",
            lambda: StructuredDeliberationPolicy(
                name="static_grounding_only",
                use_grounding=True,
                use_phase_control=False,
                use_adaptive_diversity=False,
                use_borderline_ucb=False,
            ),
            expanded,
            "heterogeneous_union",
        ),
        PolicySpec(
            "adaptive_diversity",
            lambda: StructuredDeliberationPolicy(
                name="adaptive_diversity",
                use_grounding=False,
                use_adaptive_diversity=True,
                use_borderline_ucb=False,
            ),
            expanded,
            "heterogeneous_union",
        ),
        PolicySpec(
            "structured_deliberation",
            lambda: StructuredDeliberationPolicy(
                name="structured_deliberation",
                use_grounding=True,
                use_adaptive_diversity=True,
                use_borderline_ucb=True,
            ),
            expanded,
            "heterogeneous_union",
        ),
        PolicySpec(
            "residual_first",
            lambda: ResidualFirstPolicy(name="residual_first"),
            expanded,
            "heterogeneous_union",
        ),
        PolicySpec(
            "fixed_dsl_first",
            lambda: FixedSchedulePolicy((dsl, ca, scene), name="fixed_dsl_first"),
            expanded,
            "heterogeneous_union",
        ),
        PolicySpec(
            "fixed_ca_first",
            lambda: FixedSchedulePolicy((ca, dsl, scene), name="fixed_ca_first"),
            expanded,
            "heterogeneous_union",
        ),
        PolicySpec(
            "round_robin",
            lambda: RoundRobinPolicy((dsl, ca, scene), name="round_robin"),
            expanded,
            "heterogeneous_union",
        ),
        PolicySpec(
            "static_task_router",
            lambda: StaticRoutePolicy(name="static_task_router"),
            expanded,
            "heterogeneous_union",
        ),
        PolicySpec(
            "random_seed_0",
            lambda: DeterministicRandomPolicy(seed=0, name="random_seed_0"),
            expanded,
            "heterogeneous_union",
        ),
        PolicySpec(
            "dsl_only",
            lambda: ResidualFirstPolicy(name="dsl_only"),
            (dsl,),
            "dsl_single_source",
        ),
        PolicySpec(
            "ca_only",
            lambda: ResidualFirstPolicy(name="ca_only"),
            (ca,),
            "ca_single_source",
        ),
        PolicySpec(
            "scene_only",
            lambda: ResidualFirstPolicy(name="scene_only"),
            (scene,),
            "scene_single_source",
        ),
    )


def _provider_subset(
    providers: Sequence[object], names: Sequence[str]
) -> tuple[object, ...]:
    by_name = {str(getattr(provider, "name")): provider for provider in providers}
    missing = set(names) - set(by_name)
    if missing:
        raise ValueError(f"policy requested missing providers: {sorted(missing)}")
    return tuple(by_name[name] for name in names)


def _report_summary(report: OnlineSolveReport) -> dict[str, object]:
    final = report.states[-1]
    evaluations = {item.hypothesis.hypothesis_id: item for item in final.evaluations}
    seen_semantics: set[tuple[object, ...]] = set()
    semantic_novelty: dict[str, int] = {}
    for result in final.action_results:
        novel = 0
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
            if signature not in seen_semantics:
                seen_semantics.add(signature)
                novel += 1
        semantic_novelty[result.result_id] = novel
    return {
        "status": report.status,
        "strict_budget_comparable": report.strict_budget_comparable,
        "native_budget_comparable": report.native_budget_comparable,
        "declared_native_actual_observations_complete": (
            report.strict_budget_comparable
        ),
        "native_budget": (
            None
            if report.native_budget is None
            else report.native_budget.to_json_dict()
        ),
        "native_reservation_violation_count": (
            report.native_reservation_violation_count
        ),
        "native_masked_action_count": report.native_masked_action_count,
        "stop_reason": final.stop_reason,
        "budget_used": final.budget.used.to_json_dict(),
        "selected_hypothesis_ids": [
            item.hypothesis.hypothesis_id for item in report.selected[:2]
        ],
        "emitted_candidate_count": sum(
            len(result.emitted_candidate_ids) for result in final.action_results
        ),
        "accepted_candidate_count": sum(
            len(result.accepted_candidate_ids) for result in final.action_results
        ),
        "semantic_novel_candidate_count": sum(semantic_novelty.values()),
        "deliberation_sketch": DeliberationSketch.from_blackboard(final).to_json_dict(),
        "actions": [
            {
                "kind": result.action.kind,
                "actor": result.action.actor,
                "route": result.action.route,
                "operator": result.action.operator,
                "parent_hypothesis_id": result.action.parent_hypothesis_id,
                "status": result.status,
                "reason": result.reason,
                "accepted_candidate_ids": list(result.accepted_candidate_ids),
                "semantic_novel_candidate_count": semantic_novelty[result.result_id],
                "budget_fidelity": result.budget_fidelity,
            }
            for result in final.action_results
        ],
    }


def _numeric_native_cost(
    recorders: Sequence[RecordingProvider], report: OnlineSolveReport
) -> dict[str, float]:
    by_name = {provider.name: provider for provider in recorders}
    repair_key = "residual_repair.native_cost.repair_attempts"
    totals: dict[str, float] = {repair_key: 0.0}

    def add(prefix: str, value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                add(f"{prefix}.{key}" if prefix else str(key), item)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            totals[prefix] = totals.get(prefix, 0.0) + float(value)

    for result in report.states[-1].action_results:
        action = result.action
        if action.kind == "repair":
            totals[repair_key] += 1.0
        if action.kind != "propose":
            continue
        replay_cost = (
            {}
            if result.provider_result is None
            else result.provider_result.diagnostics.get("native_cost", {})
        )
        if replay_cost:
            add("", replay_cost)
            continue
        recorder = by_name.get(action.actor)
        if recorder is not None:
            raw = recorder.cache.get((action.operator, action.parent_hypothesis_id))
            if raw is not None:
                add(
                    f"{action.actor}.native_cost",
                    raw.diagnostics.get("native_cost", {}),
                )
    return dict(sorted(totals.items()))


def _task_pool_manifest(
    task: ARCTask,
    recorders: Sequence[RecordingProvider],
    pool_candidates: Sequence[CandidateHypothesis],
    discovery: dict[str, OnlineSolveReport],
    frozen_replays: dict[str, OnlineSolveReport],
) -> dict[str, object]:
    blind = BlindTask.from_task(task)
    discovery_ids = {
        candidate.hypothesis_id
        for report in discovery.values()
        for candidate in report.states[-1].candidates
    }
    replay_ids = {
        candidate.hypothesis_id
        for report in frozen_replays.values()
        for candidate in report.states[-1].candidates
    }
    payload = {
        "schema": "afts.frozen-action-pool/v3",
        "task_id": task.task_id,
        "task_source_sha256": task.source_sha256,
        "blind_content_sha256": blind.blind_content_sha256,
        "candidate_count": len(pool_candidates),
        "candidates": [item.to_json_dict() for item in pool_candidates],
        "providers": [item.manifest() for item in recorders],
        "discovery_policies": {
            name: _report_summary(report) for name, report in sorted(discovery.items())
        },
        "frozen_replay_policies": {
            name: _report_summary(report)
            for name, report in sorted(frozen_replays.items())
        },
        "pool_closure": {
            "rule": "provider batches plus all oracle-free discovery/replay descendants",
            "discovery_candidate_count": len(discovery_ids),
            "replay_candidate_count": len(replay_ids),
            "replay_only_candidate_count": len(replay_ids - discovery_ids),
        },
        "oracle_used_during_discovery": False,
        "oracle_used_during_pool_closure": False,
    }
    pool_id = _sha256(payload)
    return {
        "pool_id": pool_id,
        "audit_manifest_id": pool_id,
        "candidate_dag_id": candidate_dag_id(payload),
        **payload,
    }


def _make_recorders(args: argparse.Namespace) -> tuple[RecordingProvider, ...]:
    return (
        RecordingProvider(
            DslProgramProvider(
                search_config=SearchConfig(
                    max_depth=args.dsl_max_depth,
                    beam_width=args.dsl_beam_width,
                    max_instruction_options=args.dsl_max_instruction_options,
                    max_exact_programs=args.dsl_max_exact_programs,
                ),
                include_repair_seeds=True,
            )
        ),
        RecordingProvider(
            SparseCAProvider(
                max_rules_per_policy=args.ca_max_rules_per_policy,
                max_programs=args.ca_max_programs,
            )
        ),
        RecordingProvider(
            SceneProgramProvider(max_exact_rules=args.scene_max_exact_rules)
        ),
    )


def _single_source_pool(
    spec: PolicySpec,
    recorders: Sequence[RecordingProvider],
    discovery_report: OnlineSolveReport,
) -> tuple[CandidateHypothesis, ...]:
    allowed = set(spec.provider_names)
    source_candidates = (
        candidate
        for recorder in recorders
        if recorder.name in allowed
        for result in recorder.cache.values()
        for candidate in result.candidates
    )
    return _merge_candidates(
        (
            source_candidates,
            discovery_report.states[-1].candidates,
        )
    )


def run_task(
    task: ARCTask,
    *,
    config: OnlineControlConfig,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    blind = BlindTask.from_task(task)
    specs = _policy_specs()
    recorders = _make_recorders(args)
    discovery: dict[str, OnlineSolveReport] = {}
    discovery_seconds: dict[str, float] = {}

    for spec in specs:
        started = time.perf_counter()
        report = OnlineFunctionalRouterSolver(
            providers=_provider_subset(recorders, spec.provider_names),
            config=config,
            policy=spec.factory(),
        ).solve(blind)
        discovery_seconds[spec.name] = time.perf_counter() - started
        discovery[spec.name] = report

    discovered_pool = _merge_candidates(
        (
            (
                candidate
                for recorder in recorders
                for result in recorder.cache.values()
                for candidate in result.candidates
            ),
            (
                candidate
                for report in discovery.values()
                for candidate in report.states[-1].candidates
            ),
        )
    )
    frozen_providers = tuple(recorder.frozen() for recorder in recorders)
    frozen_replays: dict[str, OnlineSolveReport] = {}
    replay_seconds_by_policy: dict[str, float] = {}
    for spec in specs:
        selected_providers = _provider_subset(frozen_providers, spec.provider_names)
        started = time.perf_counter()
        frozen_replays[spec.name] = OnlineFunctionalRouterSolver(
            providers=selected_providers,
            config=config,
            policy=spec.factory(),
        ).solve(blind)
        replay_seconds_by_policy[spec.name] = time.perf_counter() - started

    heterogeneous_pool = _merge_candidates(
        (
            discovered_pool,
            (
                candidate
                for report in frozen_replays.values()
                for candidate in report.states[-1].candidates
            ),
        )
    )
    pool_manifest = _task_pool_manifest(
        task, recorders, heterogeneous_pool, discovery, frozen_replays
    )
    pool_integrity = validate_pool_manifest(
        pool_manifest,
        expected_task=TaskFingerprint(
            task.task_id,
            task.source_sha256,
            blind.blind_content_sha256,
        ),
    )
    if pool_integrity.candidate_dag_id != pool_manifest["candidate_dag_id"]:
        raise AssertionError("candidate DAG identity changed during pool construction")
    pool_dir = output_dir / "pools"
    pool_dir.mkdir(parents=True, exist_ok=True)
    pool_path = pool_dir / f"{task.task_id}.{pool_manifest['pool_id'][:16]}.json"
    atomic_write_json(pool_path, pool_manifest)

    policy_results: dict[str, object] = {}
    metric_objects: dict[str, object] = {}
    for spec in specs:
        report = frozen_replays[spec.name]
        if spec.pool_scope == "heterogeneous_union":
            metric_pool = heterogeneous_pool
        else:
            metric_pool = _single_source_pool(spec, recorders, report)
        metrics = evaluate_online_report_with_oracle(
            report,
            task,
            pool_candidates=metric_pool,
            pool_scope=spec.pool_scope,
        )
        metric_objects[spec.name] = metrics
        policy_results[spec.name] = {
            **_report_summary(report),
            "metrics": metrics.to_json_dict(),
            "discovery_seconds": discovery_seconds[spec.name],
            "frozen_replay_seconds": replay_seconds_by_policy[spec.name],
            "native_cost_vector": _numeric_native_cost(recorders, report),
        }

    task_result = {
        "task_id": task.task_id,
        "task_source_sha256": task.source_sha256,
        "blind_content_sha256": blind.blind_content_sha256,
        "pool_id": pool_manifest["pool_id"],
        "candidate_dag_id": pool_manifest["candidate_dag_id"],
        "pool_manifest": str(pool_path.relative_to(output_dir)),
        "heterogeneous_pool_candidate_count": len(heterogeneous_pool),
        "policies": policy_results,
    }
    return task_result, metric_objects


def _select_tasks(
    tasks: Sequence[ARCTask],
    *,
    seed: int,
    offset: int,
    limit: int,
    task_ids: str | None,
) -> tuple[ARCTask, ...]:
    if task_ids:
        requested = tuple(item.strip() for item in task_ids.split(",") if item.strip())
        if len(set(requested)) != len(requested):
            raise ValueError("explicit task IDs must be unique")
        by_id = {task.task_id: task for task in tasks}
        missing = set(requested) - set(by_id)
        if missing:
            raise ValueError(f"unknown task IDs: {sorted(missing)}")
        return tuple(by_id[item] for item in requested)
    ordered = sorted(
        tasks,
        key=lambda task: hashlib.sha256(
            f"{seed}:{task.task_id}".encode("ascii")
        ).hexdigest(),
    )
    selected = tuple(ordered[offset : offset + limit])
    if len(selected) != limit:
        raise ValueError(
            f"requested offset/limit yields {len(selected)} tasks, expected {limit}"
        )
    return selected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", default="training")
    parser.add_argument(
        "--allow-nontraining-split",
        action="store_true",
        help="allow a noncanonical diagnostic run outside the training split",
    )
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="allow a noncanonical diagnostic run from modified source",
    )
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260726)
    parser.add_argument("--task-ids")
    parser.add_argument("--compute-units", type=int, default=11)
    parser.add_argument("--controller-steps", type=int, default=4)
    parser.add_argument("--provider-calls", type=int, default=3)
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--candidate-slots", type=int, default=7)
    parser.add_argument("--provider-batch-size", type=int, default=2)
    parser.add_argument("--dsl-max-depth", type=int, default=2)
    parser.add_argument("--dsl-beam-width", type=int, default=32)
    parser.add_argument("--dsl-max-instruction-options", type=int, default=32)
    parser.add_argument("--dsl-max-exact-programs", type=int, default=16)
    parser.add_argument("--ca-max-rules-per-policy", type=int, default=2)
    parser.add_argument("--ca-max-programs", type=int, default=4)
    parser.add_argument("--scene-max-exact-rules", type=int, default=64)
    parser.add_argument(
        "--native-budget-profile",
        type=Path,
        help="fit-block calibrated native-cost contract and vector budget",
    )
    return parser


def _load_native_budget_profile(
    path: Path | None,
    *,
    heldout_fingerprints: Sequence[TaskFingerprint] = (),
) -> tuple[
    NativeCostContract | None, NativeCostVector | None, dict[str, object] | None
]:
    if path is None:
        return None, None, None
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    integrity = validate_native_budget_profile(
        payload, heldout_fingerprints=heldout_fingerprints
    )
    contract = NativeCostContract.from_json_dict(payload.get("contract"))
    limit = NativeCostVector.from_json_dict(payload.get("budget_limit"))
    for reservation in contract.reservations:
        if not reservation.cost.fits_within(limit):
            raise ExperimentSafetyError(
                "native reservation does not fit within the profile budget limit"
            )
    return (
        contract,
        limit,
        {
            "path": str(resolved),
            "file_sha256": file_sha256(resolved),
            "schema": integrity.schema,
            "profile_id": integrity.profile_id,
            "contract_id": contract.contract_id,
            "fit_overlap_guard": integrity.overlap_guard,
            "reservation_rule": payload.get("reservation_rule"),
            "budget_rule": payload.get("budget_rule"),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_start = capture_git_source_provenance(
        REPOSITORY_ROOT, paths=ARC_EXPERIMENT_SOURCE_PATHS
    )
    if source_start.dirty and not args.allow_dirty_source:
        raise ExperimentSafetyError(
            "refusing experiment from a dirty source tree; commit first or use "
            "--allow-dirty-source for a noncanonical diagnostic"
        )
    if args.split != "training" and not args.allow_nontraining_split:
        raise ExperimentSafetyError(
            "non-training data are sealed in this development runner; use an "
            "explicit noncanonical override only when authorized"
        )
    split_root = (args.dataset_root / args.split).resolve()
    tasks = load_task_directory(split_root)
    selected = _select_tasks(
        tasks,
        seed=args.sample_seed,
        offset=args.sample_offset,
        limit=args.limit,
        task_ids=args.task_ids,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "summary.json").exists():
        raise ExperimentSafetyError(
            "refusing to replace an existing summary; choose a new output directory"
        )
    heldout_fingerprints = tuple(
        TaskFingerprint(
            task.task_id,
            task.source_sha256,
            BlindTask.from_task(task).blind_content_sha256,
        )
        for task in selected
    )
    native_contract, native_limit, native_profile = _load_native_budget_profile(
        args.native_budget_profile,
        heldout_fingerprints=heldout_fingerprints,
    )
    config = OnlineControlConfig(
        budget_limit=BudgetVector(
            compute_units=args.compute_units,
            controller_steps=args.controller_steps,
            provider_calls=args.provider_calls,
            repair_attempts=args.repair_attempts,
            candidate_slots=args.candidate_slots,
        ),
        provider_batch_size=args.provider_batch_size,
        max_selected_hypotheses=2,
        native_cost_contract=native_contract,
        native_budget_limit=native_limit,
    )
    started = time.perf_counter()
    task_results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    metrics_by_policy: dict[str, list[object]] = {
        spec.name: [] for spec in _policy_specs()
    }
    for index, task in enumerate(selected, start=1):
        try:
            task_result, metric_objects = run_task(
                task, config=config, args=args, output_dir=output_dir
            )
            task_results.append(task_result)
            for name, metrics in metric_objects.items():
                metrics_by_policy[name].append(metrics)
            print(
                json.dumps(
                    {
                        "completed": index,
                        "total": len(selected),
                        "task_id": task.task_id,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as exc:
            failures.append(
                {
                    "task_id": task.task_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            print(
                json.dumps(
                    {
                        "failed": index,
                        "total": len(selected),
                        "task_id": task.task_id,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    aggregates = {
        name: aggregate_control_metrics(items).to_json_dict()
        for name, items in metrics_by_policy.items()
    }
    native_cost_aggregates: dict[str, dict[str, float]] = {}
    native_reserved_cost_aggregates: dict[str, dict[str, float]] = {}
    timing_aggregates: dict[str, dict[str, float]] = {}
    for task_result in task_results:
        policies = task_result["policies"]
        for name, policy_result in policies.items():
            native = native_cost_aggregates.setdefault(name, {})
            for key, value in policy_result["native_cost_vector"].items():
                native[key] = native.get(key, 0.0) + float(value)
            reserved = native_reserved_cost_aggregates.setdefault(name, {})
            native_budget_payload = policy_result["native_budget"]
            if native_budget_payload is not None:
                for key, value in native_budget_payload["used"]["items"]:
                    reserved[key] = reserved.get(key, 0.0) + float(value)
            timing = timing_aggregates.setdefault(
                name, {"discovery_seconds": 0.0, "frozen_replay_seconds": 0.0}
            )
            timing["discovery_seconds"] += float(policy_result["discovery_seconds"])
            timing["frozen_replay_seconds"] += float(
                policy_result["frozen_replay_seconds"]
            )
    source_end = capture_git_source_provenance(
        REPOSITORY_ROOT, paths=ARC_EXPERIMENT_SOURCE_PATHS
    )
    verify_source_provenance_unchanged(source_start, source_end)
    publication_blockers: list[str] = []
    if source_start.dirty:
        publication_blockers.append("dirty_source_override")
    if args.split != "training":
        publication_blockers.append("nontraining_split_override")
    if failures:
        publication_blockers.append("incomplete_task_execution")
    if native_profile is not None and native_profile["fit_overlap_guard"] != "verified":
        publication_blockers.append("native_profile_fit_overlap_guard_unavailable")
    for task_result in task_results:
        for name, policy_result in task_result["policies"].items():
            if not policy_result["strict_budget_comparable"]:
                publication_blockers.append(
                    f"strict_budget_not_comparable:{task_result['task_id']}:{name}"
                )
            if (
                native_profile is not None
                and not policy_result["native_budget_comparable"]
            ):
                publication_blockers.append(
                    f"native_budget_not_comparable:{task_result['task_id']}:{name}"
                )
    publication_blockers = sorted(set(publication_blockers))
    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_start.head,
        "source_provenance": source_start.to_json_dict(),
        "runtime": runtime_metadata(("numpy",)),
        "argv": list(sys.argv[1:] if argv is None else argv),
        "training_started": False,
        "oracle_access": "posthoc_only_after_each_frozen_replay",
        "claim_scope": (
            "strict action-frozen NCU plus fit-calibrated native token bucket"
            if native_profile is not None
            else "strict action-frozen NCU comparison; native cost vectors are descriptive"
        ),
        "dataset": {
            "root": str(args.dataset_root.resolve()),
            "split": args.split,
            "selection": (
                "explicit_task_ids" if args.task_ids else "sha256_seeded_sample"
            ),
            "sample_seed": args.sample_seed,
            "sample_offset": args.sample_offset,
            "requested_task_count": len(selected),
            "completed_task_count": len(task_results),
            "failed_task_count": len(failures),
            "task_ids": [task.task_id for task in selected],
        },
        "controller_config": {
            "budget_limit": config.budget_limit.to_json_dict(),
            "provider_batch_size": config.provider_batch_size,
            "max_selected_hypotheses": config.max_selected_hypotheses,
            "minimum_repair_agreement": config.minimum_repair_agreement,
            "localized_residual_fraction": config.localized_residual_fraction,
            "native_budget_profile": native_profile,
            "native_budget_limit": (
                None if native_limit is None else native_limit.to_json_dict()
            ),
        },
        "provider_config": {
            "dsl": {
                "max_depth": args.dsl_max_depth,
                "beam_width": args.dsl_beam_width,
                "max_instruction_options": args.dsl_max_instruction_options,
                "max_exact_programs": args.dsl_max_exact_programs,
                "include_repair_seeds": True,
            },
            "sparse_ca": {
                "max_rules_per_policy": args.ca_max_rules_per_policy,
                "max_programs": args.ca_max_programs,
            },
            "scene_predicate_dsl": {
                "max_exact_rules": args.scene_max_exact_rules,
                "semantics_version": "afts-scene-predicate-dsl/v0.1",
            },
        },
        "policy_aggregates": aggregates,
        "policy_native_cost_aggregates": native_cost_aggregates,
        "policy_native_reserved_cost_aggregates": native_reserved_cost_aggregates,
        "policy_timing_aggregates": timing_aggregates,
        "tasks": task_results,
        "failures": failures,
        "publication_eligibility": {
            "eligible": not publication_blockers,
            "blockers": publication_blockers,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload["result_id"] = _sha256(payload)
    validate_summary(payload, required_split=None)
    atomic_write_json(output_dir / "summary.json", payload)
    print(
        json.dumps(
            {
                "summary": str(output_dir / "summary.json"),
                "result_id": payload["result_id"],
                "completed": len(task_results),
                "failed": len(failures),
            },
            sort_keys=True,
        )
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
