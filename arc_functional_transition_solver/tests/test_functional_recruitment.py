from __future__ import annotations

from copy import deepcopy

import pytest

from afts_arc.experiment_safety import canonical_sha256
from afts_arc.functional_recruitment import (
    FUNCTIONAL_RECRUITMENT_PLAN_SCHEMA,
    FUNCTIONAL_RECRUITMENT_PLAN_WITH_ABSTENTIONS_SCHEMA,
    freeze_functional_recruitment_plan,
    score_functional_recruitment_plan,
    summarize_functional_recruitment_result,
)
from afts_arc.hypothesis_population import HYPOTHESIS_POPULATION_RESULT_SCHEMA
from afts_arc.nvarc_anchor import freeze_submission
from afts_arc.varc_run import VARC_RUN_RECEIPT_SCHEMA


TASK_IDS = tuple(f"task_{index:02d}" for index in range(10))


def _challenges() -> dict[str, object]:
    return {
        task_id: {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]]}],
        }
        for task_id in TASK_IDS
    }


def _attempts(top_support: int) -> dict[str, object]:
    return {
        f"attempt_{rank}": [[1 if rank <= top_support else 2]]
        for rank in range(1, 11)
    }


def _anchor_freeze() -> dict[str, object]:
    top_supports = [5, 6, 7, 10, 10, 10, 10, 10, 10, 10]
    submission = {
        task_id: [_attempts(top_support)]
        for task_id, top_support in zip(TASK_IDS, top_supports, strict=True)
    }
    return freeze_submission(
        challenges=_challenges(),
        submission=submission,
        cohort_id="cohort",
        anchor_run_id="anchor",
        source_files={"challenges": "a" * 64, "submission": "b" * 64},
    )


def _plan() -> dict[str, object]:
    return freeze_functional_recruitment_plan(
        anchor_freeze=_anchor_freeze(),
        cohort_id="cohort",
        anchor_provider_name="recursive",
        recruited_provider_name="visual",
        budget_percentages=(10, 20, 30, 50),
        random_seed_count=256,
        source_files={
            "anchor_candidate_freeze": "c" * 64,
            "anchor_cost_receipt": "d" * 64,
            "protocol": "e" * 64,
        },
    )


def _population_result() -> dict[str, object]:
    task_rows = []
    for index, task_id in enumerate(TASK_IDS):
        marginal = index < 3
        task_rows.append(
            {
                "provider_hits": {
                    "recursive": not marginal,
                    "visual": marginal,
                },
                "task_id": task_id,
                "union_hit": True,
            }
        )
    content: dict[str, object] = {
        "schema": HYPOTHESIS_POPULATION_RESULT_SCHEMA,
        "population_id": "population",
        "tasks": task_rows,
    }
    return {"result_id": canonical_sha256(content), **content}


def _run_receipt() -> dict[str, object]:
    content: dict[str, object] = {
        "schema": VARC_RUN_RECEIPT_SCHEMA,
        "controller_training_started": False,
        "query_blind_protocol": {"query_gold_read": False},
        "task_statuses": [
            {"elapsed_seconds": index + 1, "task_id": task_id}
            for index, task_id in enumerate(TASK_IDS)
        ],
    }
    return {"receipt_id": canonical_sha256(content), **content}


def test_plan_is_anchor_only_deterministic_and_orders_disagreement_first() -> None:
    first = _plan()
    second = _plan()

    assert first == second
    assert first["schema"] == FUNCTIONAL_RECRUITMENT_PLAN_SCHEMA
    assert first["query_gold_read"] is False
    assert first["recruited_provider_candidates_read"] is False
    assert all("anchor_abstained" not in row for row in first["diagnostics"])
    assert first["priority_order"][:3] == ["task_00", "task_01", "task_02"]
    assert first["budget_task_counts"] == {"10": 1, "20": 2, "30": 3, "50": 5}


def test_score_measures_recovered_complement_and_random_controls() -> None:
    result = score_functional_recruitment_plan(
        plan=_plan(),
        population_result=_population_result(),
        recruited_provider_run_receipt=_run_receipt(),
        recruited_provider_run_receipt_sha256="f" * 64,
    )

    by_budget = {row["activation_percentage"]: row for row in result["budgets"]}
    assert by_budget[10]["incremental_recoveries"] == 1
    assert by_budget[20]["incremental_recoveries"] == 2
    assert by_budget[30]["incremental_recoveries"] == 3
    assert by_budget[30]["anchor_plus_recruitment_strict_coverage"] == 10
    assert by_budget[30]["target_recall"] == {"denominator": 3, "numerator": 3}
    assert result["development_gate"]["passed"] is True


def test_summary_hides_task_and_random_seed_outcomes() -> None:
    result = score_functional_recruitment_plan(
        plan=_plan(),
        population_result=_population_result(),
        recruited_provider_run_receipt=_run_receipt(),
        recruited_provider_run_receipt_sha256="f" * 64,
    )
    summary = summarize_functional_recruitment_result(result)

    assert "marginal_task_ids" not in summary
    assert summary["task_level_outcomes_exposed"] is False
    assert summary["full_result_id_commitment"] == result["result_id"]
    assert all(
        "recovery_counts" not in row["random_recovery_distribution"]
        for row in summary["budgets"]
    )


def test_score_rejects_forged_plan_or_population_result() -> None:
    forged_plan = deepcopy(_plan())
    forged_plan["priority_order"].reverse()
    with pytest.raises(ValueError, match="plan_id"):
        score_functional_recruitment_plan(
            plan=forged_plan,
            population_result=_population_result(),
            recruited_provider_run_receipt=_run_receipt(),
            recruited_provider_run_receipt_sha256="f" * 64,
        )

    forged_result = deepcopy(_population_result())
    forged_result["tasks"][0]["union_hit"] = False
    with pytest.raises(ValueError, match="result_id"):
        score_functional_recruitment_plan(
            plan=_plan(),
            population_result=forged_result,
            recruited_provider_run_receipt=_run_receipt(),
            recruited_provider_run_receipt_sha256="f" * 64,
        )


def test_v2_plan_recruits_anchor_abstention_before_disagreement() -> None:
    missing_task = "task_missing"
    task_universe = tuple(sorted((*TASK_IDS, missing_task)))
    plan = freeze_functional_recruitment_plan(
        anchor_freeze=_anchor_freeze(),
        cohort_id="cohort",
        anchor_provider_name="recursive",
        recruited_provider_name="visual",
        budget_percentages=(10, 30),
        random_seed_count=16,
        source_files={
            "anchor_candidate_freeze": "c" * 64,
            "anchor_cost_receipt": "d" * 64,
            "protocol": "e" * 64,
            "task_universe_manifest": "f" * 64,
        },
        task_universe=task_universe,
    )

    assert plan["schema"] == FUNCTIONAL_RECRUITMENT_PLAN_WITH_ABSTENTIONS_SCHEMA
    assert plan["anchor_abstention_count"] == 1
    assert plan["priority_order"][0] == missing_task
    missing_diagnostic = next(
        row for row in plan["diagnostics"] if row["task_id"] == missing_task
    )
    assert missing_diagnostic["anchor_abstained"] is True
