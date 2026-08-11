from __future__ import annotations

import pytest

from afts_arc.experiment_safety import canonical_sha256
from afts_arc.provider_overlap import build_provider_overlap


def _result(endpoint: str, hits: dict[str, bool]) -> dict[str, object]:
    return {
        "tasks": [
            {"task_id": task_id, "strict_task_hits": {endpoint: hit}}
            for task_id, hit in sorted(hits.items())
        ]
    }


def test_overlap_reports_unique_union_and_explicit_ignored_tasks() -> None:
    overlap = build_provider_overlap(
        first_result=_result("raw", {"a": True, "b": True, "c": False, "x": True}),
        second_result=_result("10", {"a": True, "b": False, "c": True}),
        first_provider="visual",
        second_provider="recursive",
        first_endpoint="raw",
        second_endpoint="10",
        eligible_task_ids=["a", "b", "c"],
        eligibility_amendment_id="amendment",
        source_files={"first": "a" * 64, "second": "b" * 64},
    )

    assert overlap["coverage"] == {"first": 2, "second": 2, "union": 3}
    assert overlap["contingency"] == {
        "both": 1,
        "first_only": 1,
        "second_only": 1,
        "neither": 0,
    }
    assert overlap["ignored_ineligible_tasks"] == {"first": ["x"], "second": []}
    assert overlap["overlap_id"] == canonical_sha256(
        {key: value for key, value in overlap.items() if key != "overlap_id"}
    )


def test_overlap_rejects_missing_eligible_task_or_unknown_endpoint() -> None:
    with pytest.raises(ValueError, match="missing eligible tasks"):
        build_provider_overlap(
            first_result=_result("raw", {"a": True}),
            second_result=_result("10", {"a": True, "b": False}),
            first_provider="visual",
            second_provider="recursive",
            first_endpoint="raw",
            second_endpoint="10",
            eligible_task_ids=["a", "b"],
            eligibility_amendment_id="amendment",
            source_files={"first": "a"},
        )

    with pytest.raises(ValueError, match="endpoint raw is absent"):
        build_provider_overlap(
            first_result=_result("1", {"a": True}),
            second_result=_result("10", {"a": True}),
            first_provider="visual",
            second_provider="recursive",
            first_endpoint="raw",
            second_endpoint="10",
            eligible_task_ids=["a"],
            eligibility_amendment_id="amendment",
            source_files={"first": "a"},
        )
