"""Post-hoc oracle metrics for online-controller experiments.

This module is deliberately outside the solver.  It accepts an already-frozen
``OnlineSolveReport`` and an oracle-bearing task only after control has stopped,
then checks that the task's blind hash matches the report before scoring it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..blind import BlindTask
from ..grid import Grid
from ..task import ARCTask
from .online import OnlineSolveReport
from .types import CandidateEvaluation, CandidateHypothesis, canonical_json
from .verification import evaluate_hypotheses


@dataclass(frozen=True, slots=True)
class TaskControlMetrics:
    task_id: str
    blind_content_sha256: str
    pool_scope: str
    strict_pool_metrics: bool
    pool_candidate_count: int
    observed_candidate_count: int
    selected_count: int
    pool_oracle_covered: bool
    pool_selectable_oracle_covered: bool
    observed_oracle_covered: bool
    observed_selectable_oracle_covered: bool
    pass_at_k: bool
    raw_pool_coverage_utilization: float | None
    pool_coverage_utilization: float | None
    exploration_recall: float | None
    selection_utilization: float | None
    correct_repair_candidate_count: int
    repair_recovered_task: bool
    compute_units: int
    correct_repair_candidates_per_compute_unit: float
    recovered_tasks_per_compute_unit: float

    @property
    def oracle_covered(self) -> bool:
        """Compatibility alias for selectable full-pool oracle coverage."""

        return self.pool_selectable_oracle_covered

    @property
    def oracle_coverage_utilization(self) -> float | None:
        """Compatibility alias for selectable full-pool utilization."""

        return self.pool_coverage_utilization

    @property
    def correct_repair_count(self) -> int:
        return self.correct_repair_candidate_count

    @property
    def correct_repairs_per_compute_unit(self) -> float:
        return self.correct_repair_candidates_per_compute_unit

    def to_json_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "blind_content_sha256": self.blind_content_sha256,
            "pool_scope": self.pool_scope,
            "strict_pool_metrics": self.strict_pool_metrics,
            "pool_candidate_count": self.pool_candidate_count,
            "observed_candidate_count": self.observed_candidate_count,
            "selected_count": self.selected_count,
            "pool_oracle_covered": self.pool_oracle_covered,
            "pool_selectable_oracle_covered": self.pool_selectable_oracle_covered,
            "observed_oracle_covered": self.observed_oracle_covered,
            "observed_selectable_oracle_covered": (
                self.observed_selectable_oracle_covered
            ),
            "pass_at_k": self.pass_at_k,
            "raw_pool_coverage_utilization": self.raw_pool_coverage_utilization,
            "pool_coverage_utilization": self.pool_coverage_utilization,
            "exploration_recall": self.exploration_recall,
            "selection_utilization": self.selection_utilization,
            "correct_repair_candidate_count": self.correct_repair_candidate_count,
            "repair_recovered_task": self.repair_recovered_task,
            "compute_units": self.compute_units,
            "correct_repair_candidates_per_compute_unit": (
                self.correct_repair_candidates_per_compute_unit
            ),
            "recovered_tasks_per_compute_unit": (
                self.recovered_tasks_per_compute_unit
            ),
        }


@dataclass(frozen=True, slots=True)
class AggregateControlMetrics:
    task_count: int
    strict_pool_task_count: int
    pool_oracle_covered_tasks: int
    pool_selectable_oracle_covered_tasks: int
    observed_oracle_covered_tasks: int
    observed_selectable_oracle_covered_tasks: int
    passed_tasks: int
    pass_rate: float
    raw_pool_coverage_utilization: float | None
    pool_coverage_utilization: float | None
    exploration_recall: float | None
    selection_utilization: float | None
    correct_repair_candidate_count: int
    repair_recovered_tasks: int
    compute_units: int
    correct_repair_candidates_per_compute_unit: float
    recovered_tasks_per_compute_unit: float

    @property
    def oracle_covered_tasks(self) -> int:
        return self.pool_selectable_oracle_covered_tasks

    @property
    def oracle_coverage_utilization(self) -> float | None:
        return self.pool_coverage_utilization

    @property
    def correct_repair_count(self) -> int:
        return self.correct_repair_candidate_count

    @property
    def correct_repairs_per_compute_unit(self) -> float:
        return self.correct_repair_candidates_per_compute_unit

    def to_json_dict(self) -> dict[str, object]:
        return {
            "task_count": self.task_count,
            "strict_pool_task_count": self.strict_pool_task_count,
            "pool_oracle_covered_tasks": self.pool_oracle_covered_tasks,
            "pool_selectable_oracle_covered_tasks": (
                self.pool_selectable_oracle_covered_tasks
            ),
            "observed_oracle_covered_tasks": self.observed_oracle_covered_tasks,
            "observed_selectable_oracle_covered_tasks": (
                self.observed_selectable_oracle_covered_tasks
            ),
            "passed_tasks": self.passed_tasks,
            "pass_rate": self.pass_rate,
            "raw_pool_coverage_utilization": self.raw_pool_coverage_utilization,
            "pool_coverage_utilization": self.pool_coverage_utilization,
            "exploration_recall": self.exploration_recall,
            "selection_utilization": self.selection_utilization,
            "correct_repair_candidate_count": self.correct_repair_candidate_count,
            "repair_recovered_tasks": self.repair_recovered_tasks,
            "compute_units": self.compute_units,
            "correct_repair_candidates_per_compute_unit": (
                self.correct_repair_candidates_per_compute_unit
            ),
            "recovered_tasks_per_compute_unit": (
                self.recovered_tasks_per_compute_unit
            ),
        }


def _oracle_bundle(task: ARCTask) -> tuple[Grid, ...]:
    outputs: list[Grid] = []
    for pair in task.test:
        if pair.output is None:
            raise ValueError("oracle metrics require every test output")
        outputs.append(pair.output)
    return tuple(outputs)


def evaluate_online_report_with_oracle(
    report: OnlineSolveReport,
    task: ARCTask,
    *,
    pool_candidates: Sequence[CandidateHypothesis] | None = None,
    pool_scope: str = "frozen_action_manifest",
) -> TaskControlMetrics:
    """Score a terminal report and, when supplied, its complete frozen pool.

    ``pool_candidates`` must contain every candidate that the compared policies
    were allowed to observe, including pre-generated repair batches.  Omitting
    it retains a clearly labelled observed-only fallback for legacy reports; a
    matched-budget coverage-utilization claim requires ``strict_pool_metrics``.
    """

    if not isinstance(report, OnlineSolveReport):
        raise TypeError("expected an OnlineSolveReport")
    if not isinstance(task, ARCTask):
        raise TypeError("expected an ARCTask")
    blind = BlindTask.from_task(task)
    if blind.blind_content_sha256 != report.base_report.blind_content_sha256:
        raise ValueError("oracle task does not match the report's blind content hash")
    oracle = _oracle_bundle(task)

    def correct(evaluation: CandidateEvaluation) -> bool:
        return (
            len(evaluation.query_outputs) == len(oracle)
            and all(output is not None for output in evaluation.query_outputs)
            and tuple(evaluation.query_outputs) == oracle
        )

    def selectable_correct(evaluation: CandidateEvaluation) -> bool:
        return evaluation.eligible and correct(evaluation)

    observed = (
        *report.initial_evaluations,
        *report.repaired_evaluations,
    )
    if pool_candidates is None:
        strict_pool_metrics = False
        resolved_pool_scope = "observed_only_fallback"
        pool_evaluations = tuple(observed)
    else:
        if not isinstance(pool_scope, str) or not pool_scope:
            raise TypeError("pool_scope must be a non-empty string")
        strict_pool_metrics = True
        resolved_pool_scope = pool_scope
        pool_evaluations = evaluate_hypotheses(tuple(pool_candidates), blind)
        pool_payloads = {
            item.hypothesis.hypothesis_id: canonical_json(
                item.hypothesis.to_json_dict()
            )
            for item in pool_evaluations
        }
        for evaluation in observed:
            hypothesis = evaluation.hypothesis
            payload = pool_payloads.get(hypothesis.hypothesis_id)
            if payload is None:
                raise ValueError(
                    "observed candidate is absent from the declared frozen pool"
                )
            if payload != canonical_json(hypothesis.to_json_dict()):
                raise ValueError(
                    "observed candidate conflicts with the frozen-pool payload"
                )

    pool_oracle_covered = any(correct(item) for item in pool_evaluations)
    pool_selectable_covered = any(
        selectable_correct(item) for item in pool_evaluations
    )
    observed_oracle_covered = any(correct(item) for item in observed)
    observed_selectable_covered = any(selectable_correct(item) for item in observed)
    passed = any(selectable_correct(item) for item in report.selected)
    correct_repairs = sum(
        selectable_correct(item) for item in report.repaired_evaluations
    )
    initial_correct = any(
        selectable_correct(item) for item in report.initial_evaluations
    )
    repair_recovered_task = bool(correct_repairs and not initial_correct)
    compute = report.states[-1].budget.used.compute_units
    return TaskControlMetrics(
        task_id=task.task_id,
        blind_content_sha256=blind.blind_content_sha256,
        pool_scope=resolved_pool_scope,
        strict_pool_metrics=strict_pool_metrics,
        pool_candidate_count=len(pool_evaluations),
        observed_candidate_count=len(observed),
        selected_count=len(report.selected),
        pool_oracle_covered=pool_oracle_covered,
        pool_selectable_oracle_covered=pool_selectable_covered,
        observed_oracle_covered=observed_oracle_covered,
        observed_selectable_oracle_covered=observed_selectable_covered,
        pass_at_k=passed,
        raw_pool_coverage_utilization=(
            1.0 if passed else 0.0 if pool_oracle_covered else None
        ),
        pool_coverage_utilization=(
            1.0 if passed else 0.0 if pool_selectable_covered else None
        ),
        exploration_recall=(
            1.0
            if observed_selectable_covered
            else 0.0
            if pool_selectable_covered
            else None
        ),
        selection_utilization=(
            1.0 if passed else 0.0 if observed_selectable_covered else None
        ),
        correct_repair_candidate_count=correct_repairs,
        repair_recovered_task=repair_recovered_task,
        compute_units=compute,
        correct_repair_candidates_per_compute_unit=(
            correct_repairs / compute if compute else 0.0
        ),
        recovered_tasks_per_compute_unit=(
            int(repair_recovered_task) / compute if compute else 0.0
        ),
    )


def aggregate_control_metrics(
    metrics: Sequence[TaskControlMetrics],
) -> AggregateControlMetrics:
    items = tuple(metrics)
    if any(not isinstance(item, TaskControlMetrics) for item in items):
        raise TypeError("aggregate inputs must be TaskControlMetrics")
    task_count = len(items)
    pool_raw = sum(item.pool_oracle_covered for item in items)
    pool_selectable = sum(item.pool_selectable_oracle_covered for item in items)
    observed_raw = sum(item.observed_oracle_covered for item in items)
    observed_selectable = sum(
        item.observed_selectable_oracle_covered for item in items
    )
    passed = sum(item.pass_at_k for item in items)
    repairs = sum(item.correct_repair_candidate_count for item in items)
    recovered_tasks = sum(item.repair_recovered_task for item in items)
    compute = sum(item.compute_units for item in items)
    return AggregateControlMetrics(
        task_count=task_count,
        strict_pool_task_count=sum(item.strict_pool_metrics for item in items),
        pool_oracle_covered_tasks=pool_raw,
        pool_selectable_oracle_covered_tasks=pool_selectable,
        observed_oracle_covered_tasks=observed_raw,
        observed_selectable_oracle_covered_tasks=observed_selectable,
        passed_tasks=passed,
        pass_rate=passed / task_count if task_count else 0.0,
        raw_pool_coverage_utilization=passed / pool_raw if pool_raw else None,
        pool_coverage_utilization=(
            passed / pool_selectable if pool_selectable else None
        ),
        exploration_recall=(
            observed_selectable / pool_selectable if pool_selectable else None
        ),
        selection_utilization=(
            passed / observed_selectable if observed_selectable else None
        ),
        correct_repair_candidate_count=repairs,
        repair_recovered_tasks=recovered_tasks,
        compute_units=compute,
        correct_repair_candidates_per_compute_unit=(
            repairs / compute if compute else 0.0
        ),
        recovered_tasks_per_compute_unit=(
            recovered_tasks / compute if compute else 0.0
        ),
    )
