"""Build and evaluate strict action-frozen ARC controller portfolios.

The script has two oracle-separated phases per task:

1. Discover action-conditioned DSL/CA batches without consulting test outputs.
2. Freeze their ``(provider, operator, parent) -> candidates`` mapping, replay
   every policy under one controller budget, and only then score with oracles.

No model training, model download, or remote model call is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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
from afts_arc.hybrid import (  # noqa: E402
    BudgetVector,
    CoverageAwareResidualPolicy,
    DeterministicRandomPolicy,
    DslProgramProvider,
    FixedSchedulePolicy,
    FrozenActionBatch,
    FrozenCandidatePoolProvider,
    OnlineControlConfig,
    OnlineFunctionalRouterSolver,
    ResidualFirstPolicy,
    RoundRobinPolicy,
    SparseCAProvider,
    StaticRoutePolicy,
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


SCHEMA_VERSION = "afts.online-matched-budget/v2"


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


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
                raise ValueError("candidate ID collision while constructing frozen pool")
            payloads[candidate.hypothesis_id] = payload
            candidates[candidate.hypothesis_id] = candidate
    return tuple(candidates[key] for key in sorted(candidates))


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
        if callable(act):
            result = act(task, features, decision, blackboard, action)
        else:
            result = self.delegate.propose(task, features, decision)
        if not isinstance(result, ProviderResult):
            raise TypeError("recording provider received a non-ProviderResult")
        self.cache[key] = result
        return result

    def frozen(self) -> FrozenCandidatePoolProvider:
        batches = tuple(
            FrozenActionBatch(operator, result.candidates, parent)
            for (operator, parent), result in sorted(
                self.cache.items(), key=lambda item: (item[0][0], item[0][1] or "")
            )
        )
        return FrozenCandidatePoolProvider.from_action_batches(
            batches, self.name, self.route
        )

    def manifest(self) -> dict[str, object]:
        return {
            "provider": self.name,
            "route": self.route,
            "strict_live_budget_contract": False,
            "actions": [
                {
                    "operator": operator,
                    "parent_hypothesis_id": parent,
                    "status": result.status,
                    "reason": result.reason,
                    "candidate_ids": [
                        item.hypothesis_id for item in result.candidates
                    ],
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
    both = (dsl, ca)
    return (
        PolicySpec(
            "coverage_aware_v2",
            lambda: CoverageAwareResidualPolicy(name="coverage_aware_v2"),
            both,
            "heterogeneous_union",
        ),
        PolicySpec(
            "residual_first",
            lambda: ResidualFirstPolicy(name="residual_first"),
            both,
            "heterogeneous_union",
        ),
        PolicySpec(
            "fixed_dsl_first",
            lambda: FixedSchedulePolicy(
                (dsl, ca), name="fixed_dsl_first"
            ),
            both,
            "heterogeneous_union",
        ),
        PolicySpec(
            "fixed_ca_first",
            lambda: FixedSchedulePolicy(
                (ca, dsl), name="fixed_ca_first"
            ),
            both,
            "heterogeneous_union",
        ),
        PolicySpec(
            "round_robin",
            lambda: RoundRobinPolicy((dsl, ca), name="round_robin"),
            both,
            "heterogeneous_union",
        ),
        PolicySpec(
            "static_task_router",
            lambda: StaticRoutePolicy(name="static_task_router"),
            both,
            "heterogeneous_union",
        ),
        PolicySpec(
            "random_seed_0",
            lambda: DeterministicRandomPolicy(
                seed=0, name="random_seed_0"
            ),
            both,
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
    evaluations = {
        item.hypothesis.hypothesis_id: item for item in final.evaluations
    }
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
    totals: dict[str, float] = {"repair_attempts": 0.0}

    def add(prefix: str, value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                add(f"{prefix}.{key}" if prefix else str(key), item)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            totals[prefix] = totals.get(prefix, 0.0) + float(value)

    for result in report.states[-1].action_results:
        action = result.action
        if action.kind == "repair":
            totals["repair_attempts"] += 1.0
        if action.kind != "propose":
            continue
        recorder = by_name.get(action.actor)
        if recorder is None:
            continue
        raw = recorder.cache.get((action.operator, action.parent_hypothesis_id))
        if raw is not None:
            add(f"{action.actor}.native_cost", raw.diagnostics.get("native_cost", {}))
    return dict(sorted(totals.items()))


def _task_pool_manifest(
    task: ARCTask,
    recorders: Sequence[RecordingProvider],
    pool_candidates: Sequence[CandidateHypothesis],
    discovery: dict[str, OnlineSolveReport],
) -> dict[str, object]:
    blind = BlindTask.from_task(task)
    payload = {
        "schema": "afts.frozen-action-pool/v1",
        "task_id": task.task_id,
        "task_source_sha256": task.source_sha256,
        "blind_content_sha256": blind.blind_content_sha256,
        "candidate_count": len(pool_candidates),
        "candidates": [item.to_json_dict() for item in pool_candidates],
        "providers": [item.manifest() for item in recorders],
        "discovery_policies": {
            name: _report_summary(report) for name, report in sorted(discovery.items())
        },
        "oracle_used_during_discovery": False,
    }
    return {"pool_id": _sha256(payload), **payload}


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

    heterogeneous_pool = _merge_candidates(
        (
            (candidate for recorder in recorders for result in recorder.cache.values() for candidate in result.candidates),
            (candidate for report in discovery.values() for candidate in report.states[-1].candidates),
        )
    )
    pool_manifest = _task_pool_manifest(
        task, recorders, heterogeneous_pool, discovery
    )
    pool_dir = output_dir / "pools"
    pool_dir.mkdir(parents=True, exist_ok=True)
    pool_path = pool_dir / f"{task.task_id}.{pool_manifest['pool_id'][:16]}.json"
    pool_path.write_text(
        json.dumps(pool_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    frozen_providers = tuple(recorder.frozen() for recorder in recorders)
    policy_results: dict[str, object] = {}
    metric_objects: dict[str, object] = {}
    for spec in specs:
        selected_providers = _provider_subset(
            frozen_providers, spec.provider_names
        )
        started = time.perf_counter()
        report = OnlineFunctionalRouterSolver(
            providers=selected_providers,
            config=config,
            policy=spec.factory(),
        ).solve(blind)
        replay_seconds = time.perf_counter() - started
        if spec.pool_scope == "heterogeneous_union":
            metric_pool = heterogeneous_pool
        else:
            metric_pool = _single_source_pool(
                spec, recorders, discovery[spec.name]
            )
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
            "frozen_replay_seconds": replay_seconds,
            "native_cost_vector": _numeric_native_cost(recorders, report),
        }

    task_result = {
        "task_id": task.task_id,
        "task_source_sha256": task.source_sha256,
        "blind_content_sha256": blind.blind_content_sha256,
        "pool_id": pool_manifest["pool_id"],
        "pool_manifest": str(pool_path.relative_to(output_dir)),
        "heterogeneous_pool_candidate_count": len(heterogeneous_pool),
        "policies": policy_results,
    }
    return task_result, metric_objects


def _select_tasks(
    tasks: Sequence[ARCTask], *, seed: int, limit: int, task_ids: str | None
) -> tuple[ARCTask, ...]:
    if task_ids:
        requested = tuple(item.strip() for item in task_ids.split(",") if item.strip())
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
    return tuple(ordered[:limit])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", default="training")
    parser.add_argument("--limit", type=int, default=32)
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    split_root = (args.dataset_root / args.split).resolve()
    tasks = load_task_directory(split_root)
    selected = _select_tasks(
        tasks,
        seed=args.sample_seed,
        limit=args.limit,
        task_ids=args.task_ids,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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
    timing_aggregates: dict[str, dict[str, float]] = {}
    for task_result in task_results:
        policies = task_result["policies"]
        for name, policy_result in policies.items():
            native = native_cost_aggregates.setdefault(name, {})
            for key, value in policy_result["native_cost_vector"].items():
                native[key] = native.get(key, 0.0) + float(value)
            timing = timing_aggregates.setdefault(
                name, {"discovery_seconds": 0.0, "frozen_replay_seconds": 0.0}
            )
            timing["discovery_seconds"] += float(policy_result["discovery_seconds"])
            timing["frozen_replay_seconds"] += float(
                policy_result["frozen_replay_seconds"]
            )
    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git_head(),
        "training_started": False,
        "oracle_access": "posthoc_only_after_each_frozen_replay",
        "claim_scope": (
            "strict action-frozen NCU comparison; native cost vectors are descriptive"
        ),
        "dataset": {
            "root": str(args.dataset_root.resolve()),
            "split": args.split,
            "selection": (
                "explicit_task_ids" if args.task_ids else "sha256_seeded_sample"
            ),
            "sample_seed": args.sample_seed,
            "requested_task_count": len(selected),
            "completed_task_count": len(task_results),
            "failed_task_count": len(failures),
            "task_ids": [task.task_id for task in selected],
        },
        "controller_config": {
            "budget_limit": config.budget_limit.to_json_dict(),
            "provider_batch_size": config.provider_batch_size,
            "max_selected_hypotheses": config.max_selected_hypotheses,
        },
        "provider_config": {
            "dsl": {
                "max_depth": args.dsl_max_depth,
                "beam_width": args.dsl_beam_width,
                "max_instruction_options": args.dsl_max_instruction_options,
                "max_exact_programs": args.dsl_max_exact_programs,
            },
            "sparse_ca": {
                "max_rules_per_policy": args.ca_max_rules_per_policy,
                "max_programs": args.ca_max_programs,
            },
        },
        "policy_aggregates": aggregates,
        "policy_native_cost_aggregates": native_cost_aggregates,
        "policy_timing_aggregates": timing_aggregates,
        "tasks": task_results,
        "failures": failures,
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload["result_id"] = _sha256(payload)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
