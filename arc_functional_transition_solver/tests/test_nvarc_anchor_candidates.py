from __future__ import annotations

from copy import deepcopy

import pytest

from afts_arc.experiment_safety import canonical_sha256
from afts_arc.nvarc_anchor import (
    freeze_submission,
    freeze_submission_with_rejections,
    score_candidate_freeze,
    score_candidate_freeze_gate,
)


def _challenges() -> dict[str, object]:
    return {
        "task_a": {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]]}, {"input": [[4]]}],
        },
        "task_b": {
            "train": [{"input": [[5]], "output": [[6]]}],
            "test": [{"input": [[7]]}],
        },
    }


def _attempts(first: list[list[int]], second: list[list[int]]) -> dict[str, object]:
    return {f"attempt_{rank}": first if rank == 1 else second for rank in range(1, 11)}


def _submission() -> dict[str, object]:
    return {
        "task_a": [
            _attempts([[8]], [[0]]),
            _attempts([[0]], [[9]]),
        ],
        "task_b": [_attempts([[1]], [[0]])],
    }


def _freeze() -> dict[str, object]:
    return freeze_submission(
        challenges=_challenges(),
        submission=_submission(),
        cohort_id="cohort",
        anchor_run_id="run",
        source_files={"challenges": "a" * 64, "submission": "b" * 64},
    )


def test_freeze_is_deterministic_query_gold_free_and_deduplicated() -> None:
    first = _freeze()
    second = _freeze()

    assert first == second
    assert first["query_gold_read"] is False
    assert first["freeze_id"] == canonical_sha256(
        {key: value for key, value in first.items() if key != "freeze_id"}
    )
    task_a = first["tasks"][0]
    first_query = task_a["queries"][0]
    assert first_query["attempt_count"] == 10
    assert first_query["unique_candidate_count"] == 2
    assert first_query["candidates"][1]["ranks"] == list(range(2, 11))


def test_freeze_rejects_task_query_attempt_and_grid_contract_violations() -> None:
    missing_task = _submission()
    del missing_task["task_b"]
    with pytest.raises(ValueError, match="task set differs"):
        freeze_submission(
            challenges=_challenges(),
            submission=missing_task,
            cohort_id="cohort",
            anchor_run_id="run",
            source_files={"x": "digest"},
        )

    missing_query = _submission()
    missing_query["task_a"] = missing_query["task_a"][:1]
    with pytest.raises(ValueError, match="query count differs"):
        freeze_submission(
            challenges=_challenges(),
            submission=missing_query,
            cohort_id="cohort",
            anchor_run_id="run",
            source_files={"x": "digest"},
        )

    missing_attempt = _submission()
    del missing_attempt["task_a"][0]["attempt_10"]
    with pytest.raises(ValueError, match="exactly attempt_1"):
        freeze_submission(
            challenges=_challenges(),
            submission=missing_attempt,
            cohort_id="cohort",
            anchor_run_id="run",
            source_files={"x": "digest"},
        )

    invalid_grid = _submission()
    invalid_grid["task_a"][0]["attempt_1"] = [[10]]
    with pytest.raises(ValueError, match=r"integer in \[0, 9\]"):
        freeze_submission(
            challenges=_challenges(),
            submission=invalid_grid,
            cohort_id="cohort",
            anchor_run_id="run",
            source_files={"x": "digest"},
        )


def test_v2_freeze_records_invalid_rank_as_explicit_miss() -> None:
    submission = _submission()
    submission["task_a"][0]["attempt_4"] = []

    freeze = freeze_submission_with_rejections(
        challenges=_challenges(),
        submission=submission,
        cohort_id="cohort",
        anchor_run_id="run",
        source_files={"x": "digest"},
    )

    assert freeze["rejected_attempt_count"] == 1
    query = freeze["tasks"][0]["queries"][0]
    assert query["rejected_attempts"] == [
        {
            "rank": 4,
            "raw_output_sha256": canonical_sha256([]),
            "reason": "invalid_arc_grid",
        }
    ]
    result = score_candidate_freeze(
        freeze=freeze,
        solutions={"task_a": [[[8]], [[9]]], "task_b": [[[1]]]},
        solution_source_sha256="c" * 64,
    )
    assert result["metrics_by_k"]["10"]["strict_task_pass"]["correct"] == 2


def test_score_reports_strict_task_query_and_upstream_pair_metrics() -> None:
    result = score_candidate_freeze(
        freeze=_freeze(),
        solutions={"task_a": [[[8]], [[9]]], "task_b": [[[1]]]},
        solution_source_sha256="c" * 64,
    )

    pass_1 = result["metrics_by_k"]["1"]
    assert pass_1["strict_task_pass"] == {"correct": 1, "rate": 0.5, "total": 2}
    assert pass_1["query_pass"] == {
        "correct": 2,
        "rate": 2 / 3,
        "total": 3,
    }
    assert pass_1["upstream_task_mean_pair_pass"] == {
        "rate": 0.75,
        "task_count": 2,
    }
    pass_2 = result["metrics_by_k"]["2"]
    assert pass_2["strict_task_pass"] == {"correct": 2, "rate": 1.0, "total": 2}
    assert result["query_gold_read"] is True


def test_score_rejects_forged_freeze_and_solution_set() -> None:
    forged = deepcopy(_freeze())
    forged["tasks"][0]["queries"][0]["candidates"][0]["output"] = [[7]]
    with pytest.raises(ValueError, match="freeze_id"):
        score_candidate_freeze(
            freeze=forged,
            solutions={"task_a": [[[8]], [[9]]], "task_b": [[[1]]]},
            solution_source_sha256="c" * 64,
        )

    with pytest.raises(ValueError, match="task set differs"):
        score_candidate_freeze(
            freeze=_freeze(),
            solutions={"task_a": [[[8]], [[9]]]},
            solution_source_sha256="c" * 64,
        )


def test_gate_score_hides_task_rows_and_commits_to_full_result() -> None:
    solutions = {"task_a": [[[8]], [[9]]], "task_b": [[[1]]]}
    full = score_candidate_freeze(
        freeze=_freeze(),
        solutions=solutions,
        solution_source_sha256="c" * 64,
    )
    gate = score_candidate_freeze_gate(
        freeze=_freeze(),
        solutions=solutions,
        solution_source_sha256="c" * 64,
    )

    assert "tasks" not in gate
    assert gate["task_level_outcomes_exposed"] is False
    assert gate["full_result_id_commitment"] == full["result_id"]
    assert gate["metrics_by_k"] == full["metrics_by_k"]
