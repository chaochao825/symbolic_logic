"""Build a task-level exact-coverage matrix for two frozen providers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from afts_arc.experiment_safety import canonical_sha256


PROVIDER_OVERLAP_SCHEMA = "afts.provider-exact-overlap/v1"


def _object(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field} keys must be strings")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be an array")
    return value


def _task_hits(
    result: Mapping[str, object], *, endpoint: str, provider: str
) -> dict[str, bool]:
    rows: dict[str, bool] = {}
    for raw_task in _sequence(result["tasks"], field=f"{provider}.tasks"):
        task = _object(raw_task, field=f"{provider} task")
        task_id = task["task_id"]
        if not isinstance(task_id, str) or not task_id or task_id in rows:
            raise ValueError(f"{provider} task_id is invalid or duplicated")
        hits = _object(
            task["strict_task_hits"], field=f"{provider}[{task_id}].strict_task_hits"
        )
        if endpoint not in hits:
            raise ValueError(f"{provider} endpoint {endpoint} is absent")
        hit = hits[endpoint]
        if not isinstance(hit, bool):
            raise TypeError(f"{provider} strict task hit must be boolean")
        rows[task_id] = hit
    return rows


def build_provider_overlap(
    *,
    first_result: Mapping[str, object],
    second_result: Mapping[str, object],
    first_provider: str,
    second_provider: str,
    first_endpoint: str,
    second_endpoint: str,
    eligible_task_ids: Sequence[str],
    eligibility_amendment_id: str,
    source_files: Mapping[str, str],
) -> dict[str, object]:
    """Compare exact whole-task coverage on an explicitly eligible task set."""

    if not first_provider or not second_provider or first_provider == second_provider:
        raise ValueError("provider names must be non-empty and distinct")
    if not first_endpoint or not second_endpoint:
        raise ValueError("provider endpoints must not be empty")
    if not eligibility_amendment_id:
        raise ValueError("eligibility_amendment_id must not be empty")
    if not source_files or not all(source_files[name] for name in source_files):
        raise ValueError("source file hashes must not be empty")
    eligible = list(eligible_task_ids)
    if not eligible or len(eligible) != len(set(eligible)) or eligible != sorted(eligible):
        raise ValueError("eligible task IDs must be non-empty, unique, and sorted")

    first_hits = _task_hits(
        first_result, endpoint=first_endpoint, provider=first_provider
    )
    second_hits = _task_hits(
        second_result, endpoint=second_endpoint, provider=second_provider
    )
    eligible_set = set(eligible)
    for provider, hits in (
        (first_provider, first_hits),
        (second_provider, second_hits),
    ):
        missing = sorted(eligible_set - set(hits))
        if missing:
            raise ValueError(f"{provider} is missing eligible tasks: {missing}")

    contingency = {"both": 0, "first_only": 0, "second_only": 0, "neither": 0}
    task_rows: list[dict[str, object]] = []
    for task_id in eligible:
        first_hit = first_hits[task_id]
        second_hit = second_hits[task_id]
        if first_hit and second_hit:
            category = "both"
        elif first_hit:
            category = "first_only"
        elif second_hit:
            category = "second_only"
        else:
            category = "neither"
        contingency[category] += 1
        task_rows.append(
            {
                "category": category,
                "first_hit": first_hit,
                "second_hit": second_hit,
                "task_id": task_id,
            }
        )

    content: dict[str, object] = {
        "schema": PROVIDER_OVERLAP_SCHEMA,
        "eligibility_amendment_id": eligibility_amendment_id,
        "eligible_task_count": len(eligible),
        "providers": {
            "first": {"endpoint": first_endpoint, "name": first_provider},
            "second": {"endpoint": second_endpoint, "name": second_provider},
        },
        "coverage": {
            "first": contingency["both"] + contingency["first_only"],
            "second": contingency["both"] + contingency["second_only"],
            "union": len(eligible) - contingency["neither"],
        },
        "contingency": contingency,
        "ignored_ineligible_tasks": {
            "first": sorted(set(first_hits) - eligible_set),
            "second": sorted(set(second_hits) - eligible_set),
        },
        "source_files": dict(sorted(source_files.items())),
        "tasks": task_rows,
    }
    return {"overlap_id": canonical_sha256(content), **content}
