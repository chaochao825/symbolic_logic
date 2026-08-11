"""Build a collision-free ARC cohort without selecting replacement tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from afts_arc.experiment_safety import canonical_sha256, file_sha256


COHORT_ELIGIBILITY_SCHEMA = "afts.arc-cohort-eligibility-amendment/v1"


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field} keys must be strings")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be an array")
    return value


def _grid(value: object, *, field: str) -> list[list[int]]:
    rows = _sequence(value, field=field)
    if not rows:
        raise ValueError(f"{field} must not be empty")
    width: int | None = None
    result: list[list[int]] = []
    for row_index, raw_row in enumerate(rows):
        row = _sequence(raw_row, field=f"{field}[{row_index}]")
        if not row or not all(isinstance(cell, int) for cell in row):
            raise ValueError(f"{field} must contain non-empty integer rows")
        if any(cell < 0 or cell > 9 for cell in row):
            raise ValueError(f"{field} contains a color outside 0..9")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"{field} must be rectangular")
        result.append(list(row))
    return result


def find_demo_query_input_collisions(
    challenges: Mapping[str, object],
) -> dict[str, list[dict[str, int]]]:
    """Return exact demo/query input collisions without reading query outputs."""

    collisions: dict[str, list[dict[str, int]]] = {}
    for task_id in sorted(challenges):
        task = _mapping(challenges[task_id], field=f"challenge[{task_id}]")
        if set(task) != {"train", "test"}:
            raise ValueError("challenge task must contain train and test only")
        train = _sequence(task["train"], field=f"challenge[{task_id}].train")
        test = _sequence(task["test"], field=f"challenge[{task_id}].test")
        demo_inputs: dict[str, list[int]] = {}
        for demo_index, raw_demo in enumerate(train):
            demo = _mapping(
                raw_demo, field=f"challenge[{task_id}].train[{demo_index}]"
            )
            if set(demo) != {"input", "output"}:
                raise ValueError("demonstration must contain input and output")
            grid = _grid(demo["input"], field="demonstration input")
            _grid(demo["output"], field="demonstration output")
            demo_inputs.setdefault(canonical_sha256(grid), []).append(demo_index)
        task_collisions: list[dict[str, int]] = []
        for query_index, raw_query in enumerate(test):
            query = _mapping(
                raw_query, field=f"challenge[{task_id}].test[{query_index}]"
            )
            if set(query) != {"input"}:
                raise ValueError("challenge query must contain input only")
            grid = _grid(query["input"], field="query input")
            for demo_index in demo_inputs.get(canonical_sha256(grid), []):
                task_collisions.append(
                    {"demo_index": demo_index, "query_index": query_index}
                )
        if task_collisions:
            collisions[task_id] = task_collisions
    return collisions


def build_collision_free_amendment(
    *,
    challenges: Mapping[str, object],
    solutions: Mapping[str, object],
    witnesses: Mapping[str, object],
    source_paths: Mapping[str, Path],
) -> dict[str, object]:
    """Exclude structurally ineligible tasks while preserving all other records."""

    task_ids = set(challenges)
    if set(solutions) != task_ids or set(witnesses) != task_ids:
        raise ValueError("challenge, solution, and witness task IDs must match")
    if set(source_paths) != {"challenges", "solutions", "witnesses"}:
        raise ValueError("source paths must identify all three cohort artifacts")

    collisions = find_demo_query_input_collisions(challenges)
    excluded_ids = set(collisions)
    eligible_ids = sorted(task_ids - excluded_ids)
    eligible_challenges = {task_id: challenges[task_id] for task_id in eligible_ids}
    eligible_solutions = {task_id: solutions[task_id] for task_id in eligible_ids}
    eligible_witnesses = {task_id: witnesses[task_id] for task_id in eligible_ids}
    for task_id in eligible_ids:
        task = _mapping(challenges[task_id], field=f"challenge[{task_id}]")
        query_count = len(_sequence(task["test"], field="challenge test"))
        if len(_sequence(solutions[task_id], field="solution outputs")) != query_count:
            raise ValueError("solution output count differs from challenge query count")

    content: dict[str, object] = {
        "schema": COHORT_ELIGIBILITY_SCHEMA,
        "criterion": "exclude_exact_demo_query_input_collision",
        "replacement_policy": "none",
        "query_gold_used_for_eligibility": False,
        "source_artifacts": {
            name: {
                "path": str(source_paths[name].resolve()),
                "sha256": file_sha256(source_paths[name]),
            }
            for name in sorted(source_paths)
        },
        "counts": {
            "source_tasks": len(task_ids),
            "eligible_tasks": len(eligible_ids),
            "excluded_tasks": len(excluded_ids),
        },
        "excluded": [
            {
                "task_id": task_id,
                "reason": "exact_demo_query_input_collision",
                "collision_indices": collisions[task_id],
            }
            for task_id in sorted(excluded_ids)
        ],
        "eligible_task_ids": eligible_ids,
    }
    return {
        "amendment": {
            "eligibility_decision_id": canonical_sha256(content),
            **content,
        },
        "challenges": eligible_challenges,
        "solutions": eligible_solutions,
        "witnesses": eligible_witnesses,
    }
