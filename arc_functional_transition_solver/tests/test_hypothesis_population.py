from __future__ import annotations

from copy import deepcopy

import pytest

from afts_arc.experiment_safety import canonical_sha256
from afts_arc.hypothesis_population import (
    HYPOTHESIS_POPULATION_WITH_ABSTENTIONS_SCHEMA,
    freeze_hypothesis_population,
    score_hypothesis_population,
    summarize_hypothesis_population_result,
)
from afts_arc.nvarc_anchor import freeze_submission
from afts_arc.varc_blind import build_query_blind_tasks
from afts_arc.varc_candidates import freeze_varc_predictions


def _challenges() -> dict[str, object]:
    return {
        "task_a": {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]]}],
        },
        "task_b": {
            "train": [{"input": [[2]], "output": [[3]]}],
            "test": [{"input": [[4]]}],
        },
        "task_c": {
            "train": [{"input": [[3]], "output": [[4]]}],
            "test": [{"input": [[5]]}],
        },
        "task_d": {
            "train": [{"input": [[4]], "output": [[5]]}],
            "test": [{"input": [[6]]}, {"input": [[7]]}],
        },
    }


def _attempts(first: int, second: int) -> dict[str, object]:
    return {
        f"attempt_{rank}": [[first if rank == 1 else second]]
        for rank in range(1, 11)
    }


def _recursive_freeze() -> dict[str, object]:
    submission = {
        "task_a": [_attempts(8, 0)],
        "task_b": [_attempts(0, 1)],
        "task_c": [_attempts(1, 0)],
        "task_d": [_attempts(2, 0), _attempts(0, 1)],
    }
    return freeze_submission(
        challenges=_challenges(),
        submission=submission,
        cohort_id="cohort",
        anchor_run_id="recursive-run",
        source_files={"challenges": "a" * 64, "submission": "b" * 64},
    )


def _visual_freeze() -> dict[str, object]:
    _, manifest = build_query_blind_tasks(
        challenges=_challenges(),
        source_cohort_id="source",
        source_challenges_sha256="a" * 64,
    )
    predictions = {
        "task_a": {"0": [[[0]], [[0]], [[1]]]},
        "task_b": {"0": [[[9]], [[9]], [[0]]]},
        "task_c": {"0": [[[1]], [[1]], [[0]]]},
        "task_d": {
            "0": [[[0]], [[1]]],
            "1": [[[3]], [[3]], [[0]]],
        },
    }
    return freeze_varc_predictions(
        challenges=_challenges(),
        blind_manifest=manifest,
        predictions=predictions,
        provider_contract={"checkpoint": "visual", "seed": 42},
        source_files={"challenges": "a" * 64, "predictions": "c" * 64},
    )


def _population() -> dict[str, object]:
    return freeze_hypothesis_population(
        cohort_id="cohort",
        provider_freezes={
            "recursive": _recursive_freeze(),
            "visual": _visual_freeze(),
        },
        provider_families={
            "recursive": "recursive-test-time-learning",
            "visual": "visual-test-time-training",
        },
        provider_freeze_sha256s={"recursive": "d" * 64, "visual": "e" * 64},
        provider_receipt_sha256s={"recursive": "f" * 64, "visual": "0" * 64},
    )


def _solutions() -> dict[str, object]:
    return {
        "task_a": [[[8]]],
        "task_b": [[[9]]],
        "task_c": [[[1]]],
        "task_d": [[[2]], [[3]]],
    }


def test_population_is_query_blind_deterministic_and_merges_exact_outputs() -> None:
    first = _population()
    second = _population()

    assert first == second
    assert first["query_gold_read"] is False
    assert first["controller_training_started"] is False
    assert first["population_id"] == canonical_sha256(
        {key: value for key, value in first.items() if key != "population_id"}
    )
    shared = [
        hypothesis
        for task in first["tasks"]
        for query in task["queries"]
        for hypothesis in query["hypotheses"]
        if hypothesis["provider_count"] == 2
    ]
    assert shared
    assert all(
        [row["provider_name"] for row in hypothesis["contributors"]]
        == ["recursive", "visual"]
        for hypothesis in shared
    )


def test_population_score_separates_complement_from_cross_provider_composition() -> None:
    result = score_hypothesis_population(
        population=_population(),
        solutions=_solutions(),
        solution_source_sha256="1" * 64,
    )

    metrics = result["metrics"]
    assert metrics["provider_strict_task_coverage"] == {
        "recursive": {"correct": 2, "total": 4},
        "visual": {"correct": 2, "total": 4},
    }
    assert metrics["exclusive_strict_task_coverage_by_provider"] == {
        "recursive": 1,
        "visual": 1,
    }
    assert metrics["union_strict_task_coverage"] == {"correct": 4, "total": 4}
    assert metrics["union_increment_over_best_single"] == 2
    assert metrics["cross_provider_composed_strict_task_coverage"] == 1
    assert metrics["union_query_coverage"] == {"correct": 5, "total": 5}
    assert result["pairwise_contingencies"]["recursive|visual"] == {
        "both": 1,
        "first_only": 1,
        "neither": 1,
        "second_only": 1,
    }


def test_population_rejects_homogeneous_or_forged_evidence() -> None:
    with pytest.raises(ValueError, match="distinct provider families"):
        freeze_hypothesis_population(
            cohort_id="cohort",
            provider_freezes={
                "recursive": _recursive_freeze(),
                "visual": _visual_freeze(),
            },
            provider_families={"recursive": "same", "visual": "same"},
            provider_freeze_sha256s={"recursive": "d" * 64, "visual": "e" * 64},
            provider_receipt_sha256s={"recursive": "f" * 64, "visual": "0" * 64},
        )

    forged = deepcopy(_population())
    forged["tasks"][0]["queries"][0]["hypotheses"][0]["output"] = [[7]]
    with pytest.raises(ValueError, match="population_id"):
        score_hypothesis_population(
            population=forged,
            solutions=_solutions(),
            solution_source_sha256="1" * 64,
        )


def test_population_summary_hides_task_rows_and_commits_to_full_result() -> None:
    result = score_hypothesis_population(
        population=_population(),
        solutions=_solutions(),
        solution_source_sha256="1" * 64,
    )
    summary = summarize_hypothesis_population_result(result)

    assert "tasks" not in summary
    assert summary["task_level_outcomes_exposed"] is False
    assert summary["full_result_id_commitment"] == result["result_id"]
    assert summary["metrics"] == result["metrics"]


def test_population_v2_preserves_provider_abstentions_as_failed_coverage() -> None:
    recursive = deepcopy(_recursive_freeze())
    recursive["tasks"] = [
        task for task in recursive["tasks"] if task["task_id"] != "task_b"
    ]
    body = {key: value for key, value in recursive.items() if key != "freeze_id"}
    recursive["freeze_id"] = canonical_sha256(body)
    population = freeze_hypothesis_population(
        cohort_id="cohort",
        provider_freezes={"recursive": recursive, "visual": _visual_freeze()},
        provider_families={
            "recursive": "recursive-test-time-learning",
            "visual": "visual-test-time-training",
        },
        provider_freeze_sha256s={"recursive": "d" * 64, "visual": "e" * 64},
        provider_receipt_sha256s={"recursive": "f" * 64, "visual": "0" * 64},
    )

    assert population["schema"] == HYPOTHESIS_POPULATION_WITH_ABSTENTIONS_SCHEMA
    assert population["aggregate"]["provider_abstention_task_count"] == {
        "recursive": 1,
        "visual": 0,
    }
    result = score_hypothesis_population(
        population=population,
        solutions=_solutions(),
        solution_source_sha256="1" * 64,
    )
    task_b = next(task for task in result["tasks"] if task["task_id"] == "task_b")
    assert task_b["provider_hits"] == {"recursive": False, "visual": True}
