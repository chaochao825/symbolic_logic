from __future__ import annotations

from copy import deepcopy

import pytest

from afts_arc.experiment_safety import canonical_sha256
from afts_arc.population_family_audit import audit_population_families


def _seal() -> dict[str, object]:
    return {
        "query_gold_written": False,
        "seal_id": "seal",
        "task_count": 4,
        "tasks": [
            {
                "family_id": "family-a",
                "family_source_id": "source-a",
                "task_id": "task-a0",
            },
            {
                "family_id": "family-a",
                "family_source_id": "source-a",
                "task_id": "task-a1",
            },
            {
                "family_id": "family-b",
                "family_source_id": "source-b",
                "task_id": "task-b0",
            },
            {
                "family_id": "family-c",
                "family_source_id": "source-c",
                "task_id": "task-c0",
            },
        ],
    }


def _result() -> dict[str, object]:
    rows = [
        ("task-a0", False, True, True),
        ("task-a1", False, False, True),
        ("task-b0", True, False, True),
        ("task-c0", False, False, False),
    ]
    return {
        "query_gold_read": True,
        "result_id": "result",
        "tasks": [
            {
                "provider_hits": {"anchor": anchor, "recruited": recruited},
                "task_id": task_id,
                "union_hit": union,
            }
            for task_id, anchor, recruited, union in rows
        ],
    }


def test_family_audit_exposes_correlated_marginal_tasks() -> None:
    audit = audit_population_families(
        population_result=_result(),
        seal=_seal(),
        anchor_provider="anchor",
        recruited_provider="recruited",
        source_files={"population_result": "a" * 64, "seal": "b" * 64},
    )

    assert audit["task_pattern_counts"] == {
        "anchor_only": 1,
        "neither": 2,
        "recruited_only": 1,
    }
    assert audit["marginal_task_count"] == 2
    assert audit["family_signal_counts"] == {
        "anchor_exclusive": 1,
        "anchor_miss_union_hit": 1,
        "cross_provider_composed": 1,
        "recruited_exclusive": 1,
        "union_miss": 1,
    }
    assert audit["marginal_cluster_concentration"] == {
        "denominator": 2,
        "numerator": 2,
    }
    assert audit["audit_id"] == canonical_sha256(
        {key: value for key, value in audit.items() if key != "audit_id"}
    )


@pytest.mark.parametrize("failure", ["missing", "duplicate", "source_alias"])
def test_family_audit_fails_closed_on_invalid_family_binding(failure: str) -> None:
    seal = deepcopy(_seal())
    result = deepcopy(_result())
    if failure == "missing":
        result["tasks"].pop()
        match = "sealed tasks absent"
    elif failure == "duplicate":
        result["tasks"].append(deepcopy(result["tasks"][0]))
        match = "duplicate population task_id"
    else:
        seal["tasks"][1]["family_source_id"] = "different-source"
        match = "multiple source identities"

    with pytest.raises((TypeError, ValueError), match=match):
        audit_population_families(
            population_result=result,
            seal=seal,
            anchor_provider="anchor",
            recruited_provider="recruited",
            source_files={"population_result": "a" * 64, "seal": "b" * 64},
        )
