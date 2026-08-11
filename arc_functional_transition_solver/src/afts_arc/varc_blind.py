"""Construct query-gold-free ARC task files for the VARC provider."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from afts_arc.experiment_safety import canonical_sha256
from afts_arc.grid import as_grid, grid_to_lists


VARC_BLIND_SCHEMA = "afts.varc-query-blind-cohort/v1"
SAFE_TASK_ID = re.compile(r"[A-Za-z0-9_-]{1,200}")


def _grid(value: object, *, field: str) -> list[list[int]]:
    del field
    return grid_to_lists(as_grid(value))


def _object(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field} keys must be strings")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a JSON array")
    return value


def build_query_blind_tasks(
    *,
    challenges: Mapping[str, object],
    source_cohort_id: str,
    source_challenges_sha256: str,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """Return per-task sentinel payloads and their content-addressed manifest."""

    if not challenges:
        raise ValueError("challenges must not be empty")
    if not source_cohort_id:
        raise ValueError("source_cohort_id must not be empty")
    if not source_challenges_sha256:
        raise ValueError("source_challenges_sha256 must not be empty")
    blind_tasks: dict[str, dict[str, object]] = {}
    records: list[dict[str, object]] = []
    for task_id in sorted(challenges):
        if SAFE_TASK_ID.fullmatch(task_id) is None:
            raise ValueError(f"task_id is not safe for a provider filename: {task_id}")
        task = _object(challenges[task_id], field=f"challenges[{task_id}]")
        if set(task) != {"train", "test"}:
            raise ValueError(f"challenge {task_id} has missing or unknown fields")
        demonstrations = _sequence(task["train"], field=f"challenges[{task_id}].train")
        queries = _sequence(task["test"], field=f"challenges[{task_id}].test")
        if not demonstrations:
            raise ValueError(f"challenge {task_id} has no demonstrations")
        if not queries:
            raise ValueError(f"challenge {task_id} has no queries")
        normalized_demos: list[dict[str, object]] = []
        for pair_index, raw_pair in enumerate(demonstrations):
            pair = _object(raw_pair, field=f"challenges[{task_id}].train[{pair_index}]")
            if set(pair) != {"input", "output"}:
                raise ValueError(
                    f"demonstration {task_id}[{pair_index}] has missing or unknown fields"
                )
            normalized_demos.append(
                {
                    "input": _grid(
                        pair["input"],
                        field=f"challenges[{task_id}].train[{pair_index}].input",
                    ),
                    "output": _grid(
                        pair["output"],
                        field=f"challenges[{task_id}].train[{pair_index}].output",
                    ),
                }
            )
        normalized_queries: list[dict[str, object]] = []
        for query_index, raw_query in enumerate(queries):
            query = _object(
                raw_query, field=f"challenges[{task_id}].test[{query_index}]"
            )
            if set(query) != {"input"}:
                raise ValueError(
                    f"query {task_id}[{query_index}] must contain input only"
                )
            query_input = _grid(
                query["input"],
                field=f"challenges[{task_id}].test[{query_index}].input",
            )
            normalized_queries.append({"input": query_input, "output": query_input})
        blind = {"test": normalized_queries, "train": normalized_demos}
        blind_tasks[task_id] = blind
        records.append(
            {
                "blind_sha256": canonical_sha256(blind),
                "demonstration_count": len(normalized_demos),
                "query_count": len(normalized_queries),
                "source_task_sha256": canonical_sha256(task),
                "task_id": task_id,
            }
        )
    body: dict[str, object] = {
        "provider_visible_test_output": "exact copy of corresponding test input",
        "query_gold_read": False,
        "schema": VARC_BLIND_SCHEMA,
        "source_challenges_sha256": source_challenges_sha256,
        "source_cohort_id": source_cohort_id,
        "tasks": records,
    }
    manifest = {"blind_cohort_id": canonical_sha256(body), **body}
    return blind_tasks, manifest
