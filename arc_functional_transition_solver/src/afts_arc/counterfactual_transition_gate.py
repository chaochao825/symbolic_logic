"""Query-blind freeze and post-freeze scoring for v4 transition arms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from .counterfactual_transition import (
    COUNTERFACTUAL_STRATEGIES,
    COUNTERFACTUAL_TRANSITION_DSL_VERSION,
    COUNTERFACTUAL_TRANSITION_PROVIDER_VERSION,
    CounterfactualTransitionCandidate,
    synthesize_counterfactual_transition_arms,
)
from .experiment_safety import canonical_sha256
from .grid import Grid, as_grid, grid_to_lists
from .hybrid.object_code import object_code_program_id
from .object_graph_rewrite_gate import blind_task_from_challenge


COUNTERFACTUAL_TRANSITION_FREEZE_SCHEMA = (
    "afts.counterfactual-transition-freeze/v4"
)
COUNTERFACTUAL_TRANSITION_RESULT_SCHEMA = (
    "afts.counterfactual-transition-result/v4"
)
SCIENTIFIC_LANES = (
    "controlled_semantic",
    "outcome_exposed_development",
    "prospective_reserve",
)


def _object(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{field} keys must be strings")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be an array")
    return value


def _validated_v3_tasks(
    v3_freeze: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    if v3_freeze["query_gold_read"] is not False:
        raise ValueError("v3 parent freeze is not query-blind")
    content = {key: value for key, value in v3_freeze.items() if key != "freeze_id"}
    if v3_freeze["freeze_id"] != canonical_sha256(content):
        raise ValueError("v3 parent freeze ID differs from canonical content")
    tasks = {}
    for raw_task in _sequence(v3_freeze["tasks"], field="v3 tasks"):
        task = _object(raw_task, field="v3 task")
        task_id = task["task_id"]
        if not isinstance(task_id, str) or not task_id:
            raise TypeError("v3 task ID must be a non-empty string")
        if task_id in tasks:
            raise ValueError("v3 parent freeze repeats a task ID")
        tasks[task_id] = task
    return tasks


def _candidate_row(
    candidate: CounterfactualTransitionCandidate,
) -> dict[str, object]:
    identity = {
        "provider_version": COUNTERFACTUAL_TRANSITION_PROVIDER_VERSION,
        "strategy": candidate.transition.strategy,
        "transition_id": candidate.transition.transition_id,
        "output_bundle_id": candidate.output_bundle_id,
    }
    return {
        "candidate_id": canonical_sha256(identity),
        **identity,
        "program_id": object_code_program_id(candidate.transition.program),
        "transition": candidate.transition.to_json_dict(),
        "certificate": candidate.certificate.to_json_dict(),
        "parent_trace_ids": list(candidate.parent_trace_ids),
        "query_outputs": [grid_to_lists(output) for output in candidate.query_outputs],
        "description_bits": candidate.transition.program.description_bits,
    }


@dataclass(frozen=True, slots=True)
class _FreezeTaskJob:
    task_id: str
    raw_task: Mapping[str, object]
    v3_task: Mapping[str, object]
    strategies: tuple[str, ...]
    max_first_stage_trials: int
    max_parents: int
    max_transition_trials: int
    max_candidates: int


@dataclass(frozen=True, slots=True)
class _FreezeTaskResult:
    row: dict[str, object]
    opportunity_counts: tuple[tuple[str, int], ...]
    novel_frontier_counts: tuple[tuple[str, int], ...]
    improving_trial_counts: tuple[tuple[str, int], ...]


def _freeze_task(job: _FreezeTaskJob) -> _FreezeTaskResult:
    task = blind_task_from_challenge(job.raw_task)
    if job.v3_task["blind_content_sha256"] != task.blind_content_sha256:
        raise ValueError("v3 and v4 blind task identities differ")
    synthesis = synthesize_counterfactual_transition_arms(
        task,
        strategies=job.strategies,
        max_first_stage_trials=job.max_first_stage_trials,
        max_parents=job.max_parents,
        max_transition_trials=job.max_transition_trials,
        max_candidates=job.max_candidates,
    )
    expected_certificate_ids = tuple(
        item["certificate_id"]
        for item in (
            _object(raw, field="v3 certificate")
            for raw in _sequence(job.v3_task["certificates"], field="v3 certificates")
        )
    )
    observed_certificate_ids = tuple(
        item.certificate_id for item in synthesis.certificates
    )
    if observed_certificate_ids != expected_certificate_ids:
        raise ValueError("v4 parent or certificate prefix drifted from v3")

    incumbent_rows = tuple(
        _object(item, field="v3 incumbent candidate")
        for item in _sequence(
            job.v3_task["incumbent_candidates"], field="v3 incumbent candidates"
        )
    )
    incumbent_output_ids = frozenset(
        str(row["output_bundle_id"]) for row in incumbent_rows
    )
    cold_rows = tuple(
        _object(item, field="v3 cold candidate")
        for item in _sequence(
            job.v3_task["cold_candidates"], field="v3 cold candidates"
        )
    )
    v3_native_cost = _object(job.v3_task["native_cost"], field="v3 native cost")
    cold_cost = _object(v3_native_cost["cold_arm"], field="v3 cold cost")
    reserved_trials = job.max_parents * job.max_transition_trials
    if cold_cost["program_trials"] != reserved_trials:
        raise ValueError("v4 reservation differs from the frozen v3 cold arm")

    opportunity_counts = {strategy: 0 for strategy in job.strategies}
    novel_frontier_counts = {strategy: 0 for strategy in job.strategies}
    improving_trial_counts = {strategy: 0 for strategy in job.strategies}
    arm_rows: dict[str, object] = {}
    for arm in synthesis.arms:
        candidate_rows = tuple(_candidate_row(candidate) for candidate in arm.candidates)
        novel_rows = tuple(
            row
            for row in candidate_rows
            if row["output_bundle_id"] not in incumbent_output_ids
        )
        opportunity = bool(novel_rows)
        opportunity_counts[arm.strategy] = int(opportunity)
        novel_frontier_counts[arm.strategy] = len(novel_rows)
        improving_trial_counts[arm.strategy] = arm.improving_trial_count
        if opportunity:
            terminal_reason = "novel_frontier_opportunity"
        elif synthesis.parent_count == 0:
            terminal_reason = "no_stateful_scene_parent"
        elif arm.program_trials == 0:
            terminal_reason = "no_legal_bounded_transition"
        elif candidate_rows:
            terminal_reason = "demo_exact_but_incumbent_duplicate"
        else:
            terminal_reason = "legal_transitions_but_no_demo_exact_candidate"
        arm_rows[arm.strategy] = {
            "strategy": arm.strategy,
            "terminal_reason": terminal_reason,
            "transition_candidates": list(candidate_rows),
            "novel_transition_candidate_ids": [
                row["candidate_id"] for row in novel_rows
            ],
            "frontier_opportunity": opportunity,
            "novel_frontier_count": len(novel_rows),
            "improving_trial_count": arm.improving_trial_count,
            "changed_node_sets": dict(arm.changed_node_sets),
            "trace_accounting": {
                "demo_node_executions": dict(arm.demo_node_executions),
                "query_node_executions": dict(arm.query_node_executions),
                "reused_demo_nodes": dict(arm.reused_demo_nodes),
            },
            "native_cost": {
                "program_trials": reserved_trials,
                "realized_program_trials": arm.program_trials,
                "padding_program_trials": arm.padding_trials,
                "reserved_demo_executions": reserved_trials * len(task.train),
                "reserved_query_executions": reserved_trials
                * len(task.test_inputs),
            },
        }
    row = {
        "task_id": job.task_id,
        "blind_task_id": task.task_id,
        "blind_content_sha256": task.blind_content_sha256,
        "demo_count": len(task.train),
        "query_count": len(task.test_inputs),
        "parent_count": synthesis.parent_count,
        "certificates": [item.to_json_dict() for item in synthesis.certificates],
        "incumbent_candidates": list(incumbent_rows),
        "frozen_cold_candidates": list(cold_rows),
        "frozen_cold_native_cost": dict(cold_cost),
        "arms": arm_rows,
        "shared_first_stage": {
            "program_trials": synthesis.first_stage_program_trials,
            "demo_executions": synthesis.first_stage_program_trials
            * len(task.train),
        },
    }
    return _FreezeTaskResult(
        row,
        tuple(sorted(opportunity_counts.items())),
        tuple(sorted(novel_frontier_counts.items())),
        tuple(sorted(improving_trial_counts.items())),
    )


def freeze_counterfactual_transition_opportunities(
    *,
    challenges: Mapping[str, object],
    v3_freeze: Mapping[str, object],
    cohort_id: str,
    scientific_lane: str,
    source_files: Mapping[str, str],
    strategies: Sequence[str] = COUNTERFACTUAL_STRATEGIES,
    max_first_stage_trials: int = 2_000,
    max_parents: int = 4,
    max_transition_trials: int = 512,
    max_candidates: int = 32,
    max_workers: int = 1,
) -> dict[str, object]:
    """Freeze all preregistered arms before opening development query labels."""

    if not challenges:
        raise ValueError("counterfactual transition cohort must not be empty")
    if scientific_lane not in SCIENTIFIC_LANES:
        raise ValueError("unknown counterfactual transition scientific lane")
    selected_strategies = tuple(strategies)
    if not selected_strategies or len(set(selected_strategies)) != len(
        selected_strategies
    ):
        raise ValueError("counterfactual strategies must be unique")
    if any(strategy not in COUNTERFACTUAL_STRATEGIES for strategy in selected_strategies):
        raise ValueError("unknown counterfactual transition strategy")
    if type(max_workers) is not int or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    v3_by_task = _validated_v3_tasks(v3_freeze)
    if set(v3_by_task) != set(challenges):
        raise ValueError("v3 parent freeze and v4 challenge task sets differ")

    opportunity_counts = {strategy: 0 for strategy in selected_strategies}
    novel_frontier_counts = {strategy: 0 for strategy in selected_strategies}
    improving_trial_counts = {strategy: 0 for strategy in selected_strategies}
    reserved_trials = max_parents * max_transition_trials
    jobs = tuple(
        _FreezeTaskJob(
            task_id,
            _object(challenges[task_id], field=f"challenge {task_id}"),
            v3_by_task[task_id],
            selected_strategies,
            max_first_stage_trials,
            max_parents,
            max_transition_trials,
            max_candidates,
        )
        for task_id in sorted(challenges)
    )
    if max_workers == 1:
        frozen_tasks = tuple(_freeze_task(job) for job in jobs)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            frozen_tasks = tuple(executor.map(_freeze_task, jobs))
    tasks = [result.row for result in frozen_tasks]
    for result in frozen_tasks:
        for strategy, value in result.opportunity_counts:
            opportunity_counts[strategy] += value
        for strategy, value in result.novel_frontier_counts:
            novel_frontier_counts[strategy] += value
        for strategy, value in result.improving_trial_counts:
            improving_trial_counts[strategy] += value

    content: dict[str, object] = {
        "schema": COUNTERFACTUAL_TRANSITION_FREEZE_SCHEMA,
        "provider_version": COUNTERFACTUAL_TRANSITION_PROVIDER_VERSION,
        "dsl_version": COUNTERFACTUAL_TRANSITION_DSL_VERSION,
        "cohort_id": cohort_id,
        "scientific_lane": scientific_lane,
        "parent_v3_freeze_id": v3_freeze["freeze_id"],
        "query_gold_read": False,
        "public_evaluation_read": False,
        "controller_training_started": False,
        "dependency_replay_required": True,
        "multi_arm_union_is_single_arm_comparable": False,
        "execution_workers": max_workers,
        "strategies": list(selected_strategies),
        "search_contract": {
            "max_first_stage_trials": max_first_stage_trials,
            "max_parents": max_parents,
            "max_transition_trials": max_transition_trials,
            "max_candidates": max_candidates,
            "reserved_trials_per_arm": reserved_trials,
        },
        "source_files": dict(sorted(source_files.items())),
        "task_count": len(tasks),
        "opportunity_counts": opportunity_counts,
        "novel_frontier_counts": novel_frontier_counts,
        "improving_trial_counts": improving_trial_counts,
        "tasks": tasks,
    }
    return {"freeze_id": canonical_sha256(content), **content}


def _candidate_exact(candidate: Mapping[str, object], gold: Sequence[Grid]) -> bool:
    outputs = tuple(
        as_grid(item)
        for item in _sequence(candidate["query_outputs"], field="candidate outputs")
    )
    return outputs == tuple(gold)


def score_counterfactual_transition_freeze(
    *,
    freeze: Mapping[str, object],
    solutions: Mapping[str, object],
    solution_source_sha256: str,
) -> dict[str, object]:
    """Read exposed query labels only after all v4 arms are frozen."""

    content = {key: value for key, value in freeze.items() if key != "freeze_id"}
    if freeze["freeze_id"] != canonical_sha256(content):
        raise ValueError("counterfactual transition freeze ID differs")
    if freeze["query_gold_read"] is not False:
        raise ValueError("counterfactual transition freeze is not query-blind")
    strategies = tuple(
        str(item) for item in _sequence(freeze["strategies"], field="strategies")
    )
    metrics = {
        strategy: {
            "transition_exact": 0,
            "transition_pass_at_2": 0,
            "unique_over_incumbent": 0,
            "unique_over_incumbent_and_cold": 0,
        }
        for strategy in strategies
    }
    incumbent_exact_count = 0
    cold_exact_count = 0
    union_exact_count = 0
    union_unique_count = 0
    task_rows = []
    for raw_task in _sequence(freeze["tasks"], field="freeze tasks"):
        task = _object(raw_task, field="freeze task")
        task_id = task["task_id"]
        if task_id not in solutions:
            raise ValueError("solution task set does not cover the v4 freeze")
        gold = tuple(
            as_grid(item)
            for item in _sequence(solutions[task_id], field="solution outputs")
        )
        incumbent = tuple(
            _object(item, field="incumbent candidate")
            for item in _sequence(
                task["incumbent_candidates"], field="incumbent candidates"
            )
        )
        cold = tuple(
            _object(item, field="cold candidate")
            for item in _sequence(
                task["frozen_cold_candidates"], field="frozen cold candidates"
            )
        )
        incumbent_exact = any(_candidate_exact(item, gold) for item in incumbent)
        cold_exact = any(_candidate_exact(item, gold) for item in cold)
        incumbent_exact_count += int(incumbent_exact)
        cold_exact_count += int(cold_exact)
        raw_arms = _object(task["arms"], field="task arms")
        arm_metrics = {}
        any_transition_exact = False
        for strategy in strategies:
            arm = _object(raw_arms[strategy], field=f"{strategy} arm")
            candidates = tuple(
                _object(item, field="transition candidate")
                for item in _sequence(
                    arm["transition_candidates"], field="transition candidates"
                )
            )
            transition_exact = any(
                _candidate_exact(item, gold) for item in candidates
            )
            pass_at_2 = any(
                _candidate_exact(item, gold) for item in candidates[:2]
            )
            unique_incumbent = transition_exact and not incumbent_exact
            unique_all = unique_incumbent and not cold_exact
            values = {
                "transition_exact": transition_exact,
                "transition_pass_at_2": pass_at_2,
                "unique_over_incumbent": unique_incumbent,
                "unique_over_incumbent_and_cold": unique_all,
            }
            arm_metrics[strategy] = values
            any_transition_exact = any_transition_exact or transition_exact
            for name, value in values.items():
                metrics[strategy][name] += int(value)
        union_exact_count += int(any_transition_exact)
        union_unique_count += int(any_transition_exact and not incumbent_exact and not cold_exact)
        task_rows.append(
            {
                "task_id": task_id,
                "incumbent_exact": incumbent_exact,
                "cold_exact": cold_exact,
                "arms": arm_metrics,
                "three_arm_oracle_union_exact": any_transition_exact,
            }
        )
    result_content: dict[str, object] = {
        "schema": COUNTERFACTUAL_TRANSITION_RESULT_SCHEMA,
        "freeze_id": freeze["freeze_id"],
        "solution_source_sha256": solution_source_sha256,
        "task_count": freeze["task_count"],
        "incumbent_exact": incumbent_exact_count,
        "frozen_cold_exact": cold_exact_count,
        "arm_metrics": metrics,
        "three_arm_oracle_union_exact": union_exact_count,
        "three_arm_oracle_union_unique": union_unique_count,
        "three_arm_union_uses_three_times_single_arm_reservation": True,
        "tasks": task_rows,
    }
    return {"result_id": canonical_sha256(result_content), **result_content}
