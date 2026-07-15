"""Official-style pair scoring plus stricter task-level diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .grid import Grid, as_grid
from .task import ARCTask

AttemptsByTest = Mapping[int, Sequence[Grid | object]]
SubmissionsByTask = Mapping[str, AttemptsByTest]


@dataclass(frozen=True, slots=True)
class PairScore:
    test_index: int
    pass_at_1: bool
    pass_at_2: bool
    attempts_available: int
    attempts_considered: int


@dataclass(frozen=True, slots=True)
class TaskScore:
    task_id: str
    pair_scores: tuple[PairScore, ...]
    official_pass_at_1: float
    official_pass_at_2: float
    strict_task_solved_at_1: bool
    strict_task_solved_at_2: bool


@dataclass(frozen=True, slots=True)
class DatasetScore:
    task_scores: tuple[TaskScore, ...]
    official_pass_at_1: float
    official_pass_at_2: float
    strict_task_solved_rate_at_1: float
    strict_task_solved_rate_at_2: float


def score_task(task: ARCTask, attempts: AttemptsByTest) -> TaskScore:
    """Score at most two attempts per test pair.

    The official ARC benchmarking implementation gives each task the fraction of its
    test pairs solved by any submitted attempt and averages those task fractions.
    Benchmark policy allows two attempts, so this function explicitly ignores attempts
    after the second. It additionally reports a strict all-test-pairs-solved diagnostic.
    """

    for test_index in attempts:
        if type(test_index) is not int:
            raise TypeError("attempt mapping keys must be integer test indices")
        if not 0 <= test_index < len(task.test):
            raise ValueError(
                f"attempt mapping contains out-of-range test index {test_index} for {task.task_id}"
            )

    pair_scores: list[PairScore] = []
    for test_index, pair in enumerate(task.test):
        if pair.output is None:
            raise ValueError(f"task {task.task_id} test[{test_index}] has no reference output")
        raw_attempts = attempts.get(test_index, ())
        considered = tuple(as_grid(value) for value in raw_attempts[:2])
        correct = tuple(candidate == pair.output for candidate in considered)
        pair_scores.append(
            PairScore(
                test_index=test_index,
                pass_at_1=bool(correct[:1] and correct[0]),
                pass_at_2=any(correct),
                attempts_available=len(raw_attempts),
                attempts_considered=len(considered),
            )
        )

    count = len(pair_scores)
    official_at_1 = sum(score.pass_at_1 for score in pair_scores) / count
    official_at_2 = sum(score.pass_at_2 for score in pair_scores) / count
    return TaskScore(
        task_id=task.task_id,
        pair_scores=tuple(pair_scores),
        official_pass_at_1=official_at_1,
        official_pass_at_2=official_at_2,
        strict_task_solved_at_1=all(score.pass_at_1 for score in pair_scores),
        strict_task_solved_at_2=all(score.pass_at_2 for score in pair_scores),
    )


def score_dataset(
    tasks: Sequence[ARCTask], submissions: SubmissionsByTask
) -> DatasetScore:
    if not tasks:
        raise ValueError("cannot score an empty task sequence")
    if any(not isinstance(task, ARCTask) for task in tasks):
        raise TypeError("tasks must contain only ARCTask values")
    task_ids = tuple(task.task_id for task in tasks)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task sequence contains duplicate task IDs")
    if any(not isinstance(task_id, str) for task_id in submissions):
        raise TypeError("submission mapping keys must be task ID strings")
    unknown = set(submissions) - set(task_ids)
    if unknown:
        raise ValueError(f"submissions contain unknown task IDs: {sorted(unknown)}")
    task_scores = tuple(
        score_task(task, submissions.get(task.task_id, {})) for task in tasks
    )
    count = len(task_scores)
    return DatasetScore(
        task_scores=task_scores,
        official_pass_at_1=sum(score.official_pass_at_1 for score in task_scores) / count,
        official_pass_at_2=sum(score.official_pass_at_2 for score in task_scores) / count,
        strict_task_solved_rate_at_1=sum(
            score.strict_task_solved_at_1 for score in task_scores
        )
        / count,
        strict_task_solved_rate_at_2=sum(
            score.strict_task_solved_at_2 for score in task_scores
        )
        / count,
    )
