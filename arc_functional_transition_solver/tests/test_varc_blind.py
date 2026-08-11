from __future__ import annotations

import pytest

from afts_arc.experiment_safety import canonical_sha256
from afts_arc.varc_blind import build_query_blind_tasks


def _challenges() -> dict[str, object]:
    return {
        "arc_tgi_family_1": {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3, 4]]}],
        }
    }


def test_blind_cohort_is_deterministic_and_uses_input_copy_sentinel() -> None:
    first_tasks, first_manifest = build_query_blind_tasks(
        challenges=_challenges(),
        source_cohort_id="cohort",
        source_challenges_sha256="a" * 64,
    )
    second_tasks, second_manifest = build_query_blind_tasks(
        challenges=_challenges(),
        source_cohort_id="cohort",
        source_challenges_sha256="a" * 64,
    )

    assert first_tasks == second_tasks
    assert first_manifest == second_manifest
    assert first_tasks["arc_tgi_family_1"]["test"] == [
        {"input": [[3, 4]], "output": [[3, 4]]}
    ]
    assert first_manifest["query_gold_read"] is False
    assert first_manifest["blind_cohort_id"] == canonical_sha256(
        {
            key: value
            for key, value in first_manifest.items()
            if key != "blind_cohort_id"
        }
    )


def test_blind_cohort_rejects_query_gold_and_unsafe_task_id() -> None:
    gold = _challenges()
    gold["arc_tgi_family_1"]["test"][0]["output"] = [[9]]
    with pytest.raises(ValueError, match="must contain input only"):
        build_query_blind_tasks(
            challenges=gold,
            source_cohort_id="cohort",
            source_challenges_sha256="a" * 64,
        )

    unsafe = {"../escape": _challenges()["arc_tgi_family_1"]}
    with pytest.raises(ValueError, match="not safe"):
        build_query_blind_tasks(
            challenges=unsafe,
            source_cohort_id="cohort",
            source_challenges_sha256="a" * 64,
        )


def test_blind_cohort_validates_demo_and_query_grids() -> None:
    malformed = _challenges()
    malformed["arc_tgi_family_1"]["train"][0]["output"] = [[10]]
    with pytest.raises(ValueError, match=r"integer in \[0, 9\]"):
        build_query_blind_tasks(
            challenges=malformed,
            source_cohort_id="cohort",
            source_challenges_sha256="a" * 64,
        )
