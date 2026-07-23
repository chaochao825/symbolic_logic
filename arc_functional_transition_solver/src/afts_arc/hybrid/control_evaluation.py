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


@dataclass(frozen=True, slots=True)
class TaskControlMetrics:
    task_id: str
    blind_content_sha256: str
    observed_candidate_count: int
    selected_count: int
    oracle_covered: bool
    pass_at_k: bool
    oracle_coverage_utilization: float | None
    correct_repair_count: int
    compute_units: int
    correct_repairs_per_compute_unit: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "blind_content_sha256": self.blind_content_sha256,
            "observed_candidate_count": self.observed_candidate_count,
            "selected_count": self.selected_count,
            "oracle_covered": self.oracle_covered,
            "pass_at_k": self.pass_at_k,
            "oracle_coverage_utilization": self.oracle_coverage_utilization,
            "correct_repair_count": self.correct_repair_count,
            "compute_units": self.compute_units,
            "correct_repairs_per_compute_unit": self.correct_repairs_per_compute_unit,
        }


@dataclass(frozen=True, slots=True)
class AggregateControlMetrics:
    task_count: int
    oracle_covered_tasks: int
    passed_tasks: int
    pass_rate: float
    oracle_coverage_utilization: float | None
    correct_repair_count: int
    compute_units: int
    correct_repairs_per_compute_unit: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "task_count": self.task_count,
            "oracle_covered_tasks": self.oracle_covered_tasks,
            "passed_tasks": self.passed_tasks,
            "pass_rate": self.pass_rate,
            "oracle_coverage_utilization": self.oracle_coverage_utilization,
            "correct_repair_count": self.correct_repair_count,
            "compute_units": self.compute_units,
            "correct_repairs_per_compute_unit": self.correct_repairs_per_compute_unit,
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
) -> TaskControlMetrics:
    """Score only after solving; no oracle-bearing object reaches the controller."""

    if not isinstance(report, OnlineSolveReport):
        raise TypeError("expected an OnlineSolveReport")
    if not isinstance(task, ARCTask):
        raise TypeError("expected an ARCTask")
    blind = BlindTask.from_task(task)
    if blind.blind_content_sha256 != report.base_report.blind_content_sha256:
        raise ValueError("oracle task does not match the report's blind content hash")
    oracle = _oracle_bundle(task)

    def correct(evaluation) -> bool:
        return (
            len(evaluation.query_outputs) == len(oracle)
            and all(output is not None for output in evaluation.query_outputs)
            and tuple(evaluation.query_outputs) == oracle
        )

    evaluations = (
        *report.initial_evaluations,
        *report.repaired_evaluations,
    )
    oracle_covered = any(correct(item) for item in evaluations)
    passed = any(correct(item) for item in report.selected)
    correct_repairs = sum(correct(item) for item in report.repaired_evaluations)
    compute = report.states[-1].budget.used.compute_units
    return TaskControlMetrics(
        task_id=task.task_id,
        blind_content_sha256=blind.blind_content_sha256,
        observed_candidate_count=len(evaluations),
        selected_count=len(report.selected),
        oracle_covered=oracle_covered,
        pass_at_k=passed,
        oracle_coverage_utilization=(
            1.0 if passed else 0.0 if oracle_covered else None
        ),
        correct_repair_count=correct_repairs,
        compute_units=compute,
        correct_repairs_per_compute_unit=(
            correct_repairs / compute if compute else 0.0
        ),
    )


def aggregate_control_metrics(
    metrics: Sequence[TaskControlMetrics],
) -> AggregateControlMetrics:
    items = tuple(metrics)
    if any(not isinstance(item, TaskControlMetrics) for item in items):
        raise TypeError("aggregate inputs must be TaskControlMetrics")
    task_count = len(items)
    covered = sum(item.oracle_covered for item in items)
    passed = sum(item.pass_at_k for item in items)
    repairs = sum(item.correct_repair_count for item in items)
    compute = sum(item.compute_units for item in items)
    return AggregateControlMetrics(
        task_count=task_count,
        oracle_covered_tasks=covered,
        passed_tasks=passed,
        pass_rate=passed / task_count if task_count else 0.0,
        oracle_coverage_utilization=passed / covered if covered else None,
        correct_repair_count=repairs,
        compute_units=compute,
        correct_repairs_per_compute_unit=repairs / compute if compute else 0.0,
    )
