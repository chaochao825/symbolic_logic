"""Family-level audit for heterogeneous population results.

Task replicas from one generator family are correlated evidence.  This module
keeps task-level solver metrics unchanged while reporting the effective number
of families that contribute each complement signal.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

from afts_arc.experiment_safety import canonical_sha256


FAMILY_AUDIT_SCHEMA = "afts.heterogeneous-population-family-audit/v1"


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def _seal_family_index(seal: Mapping[str, object]) -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    family_sources: dict[str, str] = {}
    for raw_task in _list(seal["tasks"], "seal.tasks"):
        task = _mapping(raw_task, "seal task")
        task_id = _string(task["task_id"], "seal task_id")
        family_id = _string(task["family_id"], "seal family_id")
        family_source_id = _string(
            task["family_source_id"], "seal family_source_id"
        )
        if task_id in index:
            raise ValueError(f"duplicate seal task_id: {task_id}")
        previous_source = family_sources.setdefault(family_id, family_source_id)
        if previous_source != family_source_id:
            raise ValueError(f"family_id has multiple source identities: {family_id}")
        index[task_id] = (family_id, family_source_id)
    declared_count = seal["task_count"]
    if type(declared_count) is not int or declared_count != len(index):
        raise ValueError("seal task_count disagrees with unique task records")
    return index


def audit_population_families(
    *,
    population_result: Mapping[str, object],
    seal: Mapping[str, object],
    anchor_provider: str,
    recruited_provider: str,
    source_files: Mapping[str, str],
) -> dict[str, object]:
    """Aggregate exact task outcomes by an explicitly sealed family identity."""

    if anchor_provider == recruited_provider:
        raise ValueError("anchor and recruited providers must differ")
    if population_result["query_gold_read"] is not True:
        raise ValueError("population result must be an outcome-exposed score")
    if seal["query_gold_written"] is not False:
        raise ValueError("family identities must come from a blind seal")

    family_index = _seal_family_index(seal)
    result_tasks = _list(population_result["tasks"], "population_result.tasks")
    seen: set[str] = set()
    family_tasks: dict[str, list[dict[str, object]]] = defaultdict(list)
    task_pattern_counts: Counter[str] = Counter()
    signal_families: dict[str, set[str]] = defaultdict(set)

    for raw_task in result_tasks:
        task = _mapping(raw_task, "population result task")
        task_id = _string(task["task_id"], "population result task_id")
        if task_id in seen:
            raise ValueError(f"duplicate population task_id: {task_id}")
        seen.add(task_id)
        if task_id not in family_index:
            raise ValueError(f"population task absent from seal: {task_id}")
        family_id, family_source_id = family_index[task_id]
        provider_hits = _mapping(task["provider_hits"], "provider_hits")
        if anchor_provider not in provider_hits:
            raise ValueError(f"anchor provider absent for task: {task_id}")
        if recruited_provider not in provider_hits:
            raise ValueError(f"recruited provider absent for task: {task_id}")
        anchor_hit = _boolean(
            provider_hits[anchor_provider], "anchor provider task hit"
        )
        recruited_hit = _boolean(
            provider_hits[recruited_provider], "recruited provider task hit"
        )
        union_hit = _boolean(task["union_hit"], "union task hit")
        if (anchor_hit or recruited_hit) and not union_hit:
            raise ValueError(f"provider hit without union hit: {task_id}")

        if anchor_hit and recruited_hit:
            pattern = "both"
        elif anchor_hit:
            pattern = "anchor_only"
        elif recruited_hit:
            pattern = "recruited_only"
        else:
            pattern = "neither"
        task_pattern_counts[pattern] += 1

        signals: list[str] = []
        if pattern == "anchor_only":
            signals.append("anchor_exclusive")
        if pattern == "recruited_only":
            signals.append("recruited_exclusive")
        if union_hit and not anchor_hit:
            signals.append("anchor_miss_union_hit")
        if union_hit and not anchor_hit and not recruited_hit:
            signals.append("cross_provider_composed")
        if not union_hit:
            signals.append("union_miss")
        for signal in signals:
            signal_families[signal].add(family_id)

        family_tasks[family_id].append(
            {
                "anchor_hit": anchor_hit,
                "pattern": pattern,
                "recruited_hit": recruited_hit,
                "signals": signals,
                "task_id": task_id,
                "union_hit": union_hit,
            }
        )

    if seen != set(family_index):
        missing = sorted(set(family_index) - seen)
        raise ValueError(f"sealed tasks absent from population result: {missing}")

    family_rows: list[dict[str, object]] = []
    for family_id in sorted(family_tasks):
        rows = sorted(family_tasks[family_id], key=lambda row: str(row["task_id"]))
        patterns = Counter(str(row["pattern"]) for row in rows)
        signal_counts = Counter(
            signal for row in rows for signal in _list(row["signals"], "signals")
        )
        family_rows.append(
            {
                "family_id": family_id,
                "family_source_id": family_index[str(rows[0]["task_id"])][1],
                "signal_task_counts": dict(sorted(signal_counts.items())),
                "task_count": len(rows),
                "task_ids": [str(row["task_id"]) for row in rows],
                "task_pattern_counts": dict(sorted(patterns.items())),
            }
        )

    marginal_counts = [
        sum(
            "anchor_miss_union_hit" in _list(task["signals"], "signals")
            for task in family_tasks[str(row["family_id"])]
        )
        for row in family_rows
    ]
    marginal_task_count = sum(marginal_counts)
    maximum_cluster = max(marginal_counts, default=0)
    family_signal_counts = {
        signal: len(families) for signal, families in sorted(signal_families.items())
    }
    body: dict[str, object] = {
        "anchor_provider": anchor_provider,
        "family_count": len(family_rows),
        "family_signal_counts": family_signal_counts,
        "families": family_rows,
        "marginal_cluster_concentration": {
            "denominator": marginal_task_count,
            "numerator": maximum_cluster,
        },
        "marginal_task_count": marginal_task_count,
        "population_result_id": _string(
            population_result["result_id"], "population result_id"
        ),
        "query_gold_read": True,
        "recruited_provider": recruited_provider,
        "schema": FAMILY_AUDIT_SCHEMA,
        "seal_id": _string(seal["seal_id"], "seal_id"),
        "source_files": dict(sorted(source_files.items())),
        "task_count": len(result_tasks),
        "task_pattern_counts": dict(sorted(task_pattern_counts.items())),
    }
    return {"audit_id": canonical_sha256(body), **body}
