from __future__ import annotations

import copy

import pytest

from afts_arc.experiment_safety import canonical_sha256
from afts_arc.relational_transducer_gate import (
    audit_relational_transducer_allocation_ablation,
    freeze_relational_transducer_gate,
    score_relational_transducer_allocation_ablation,
    score_relational_transducer_gate,
)


def _challenges() -> dict[str, object]:
    return {
        "a": {
            "train": [
                {
                    "input": [[0, 8, 4, 9]],
                    "output": [[0, 6, 6, 9]],
                }
            ],
            "test": [{"input": [[0, 4, 8, 9]]}],
        },
        "b": {
            "train": [{"input": [[0, 1]], "output": [[0, 1]]}],
            "test": [{"input": [[0, 1]]}],
        },
        "c": {
            "train": [{"input": [[0, 2]], "output": [[0, 2]]}],
            "test": [{"input": [[0, 2]]}],
        },
    }


def _visual() -> dict[str, object]:
    tasks = []
    for task_id, output, count in (
        ("a", [[0, 5, 5, 9]], 10),
        ("b", [[1, 0]], 10),
        ("c", [[2, 2]], 10),
    ):
        tasks.append(
            {
                "task_id": task_id,
                "queries": [
                    {
                        "candidates": [
                            {"output": output, "sample_count": count}
                        ]
                    }
                ],
            }
        )
    return {"freeze_id": "visual", "query_gold_read": False, "tasks": tasks}


def _recursive() -> dict[str, object]:
    return {
        "freeze_id": "recursive",
        "query_gold_read": False,
        "tasks": [
            {
                "task_id": task_id,
                "queries": [{"candidates": [{"output": output}]}],
            }
            for task_id, output in (
                ("a", [[0, 5, 4, 9]]),
                ("b", [[0, 1]]),
                ("c", [[0, 2]]),
            )
        ],
    }


def test_gate_is_query_blind_content_addressed_and_cost_matched() -> None:
    first = freeze_relational_transducer_gate(
        challenges=_challenges(),
        visual_freeze=_visual(),
        recursive_freeze=_recursive(),
        cohort_id="cohort",
        source_files={"protocol": "a" * 64},
    )
    second = freeze_relational_transducer_gate(
        challenges=_challenges(),
        visual_freeze=_visual(),
        recursive_freeze=_recursive(),
        cohort_id="cohort",
        source_files={"protocol": "a" * 64},
    )

    assert first == second
    assert first["query_gold_read"] is False
    assert first["freeze_id"] == canonical_sha256(
        {key: value for key, value in first.items() if key != "freeze_id"}
    )
    assert first["arms"]["visual_typed"]["native_cost"] == first["arms"][
        "cold_restart"
    ]["native_cost"]
    task_a = next(task for task in first["tasks"] if task["task_id"] == "a")
    assert task_a["frontier_changed"] is True
    assert any(
        candidate["query_outputs"] == [[[0, 6, 6, 9]]]
        for candidate in task_a["candidates"]
    )


def test_score_counts_only_recovery_beyond_frozen_union() -> None:
    freeze = freeze_relational_transducer_gate(
        challenges=_challenges(),
        visual_freeze=_visual(),
        recursive_freeze=_recursive(),
        cohort_id="cohort",
        source_files={"protocol": "a" * 64},
    )
    result = score_relational_transducer_gate(
        freeze=freeze,
        solutions={"a": [[[0, 6, 6, 9]]], "b": [[[0, 1]]], "c": [[[0, 2]]]},
        solution_source_sha256="b" * 64,
    )

    assert result["base_union_strict_coverage"] == 2
    assert result["unique_recoveries"]["static_family"] == 1
    assert result["confirm_gate"]["applicable"] is False
    assert result["result_id"] == canonical_sha256(
        {key: value for key, value in result.items() if key != "result_id"}
    )


def test_visual_posterior_ablation_changes_allocation_without_gold() -> None:
    freeze = freeze_relational_transducer_gate(
        challenges=_challenges(),
        visual_freeze=_visual(),
        recursive_freeze=_recursive(),
        cohort_id="cohort",
        source_files={"protocol": "a" * 64},
    )
    audit = audit_relational_transducer_allocation_ablation(
        freeze=freeze,
        protocol_sha256="c" * 64,
    )

    assert audit["query_gold_read"] is False
    assert audit["observed_visual_selected_task_ids"] == ["b"]
    assert audit["posterior_cleared"]["selected_task_ids"] == ["a"]
    assert audit["posterior_cleared"]["selected_set_changed"] is True
    assert audit["primary_gate_passed"] is True
    assert audit["ablation_id"] == canonical_sha256(
        {key: value for key, value in audit.items() if key != "ablation_id"}
    )


def test_allocation_outcome_attributes_recovery_after_frozen_ablation() -> None:
    freeze = freeze_relational_transducer_gate(
        challenges=_challenges(),
        visual_freeze=_visual(),
        recursive_freeze=_recursive(),
        cohort_id="cohort",
        source_files={"protocol": "a" * 64},
    )
    audit = audit_relational_transducer_allocation_ablation(
        freeze=freeze,
        protocol_sha256="c" * 64,
    )
    outcome = score_relational_transducer_allocation_ablation(
        freeze=freeze,
        ablation=audit,
        solutions={"a": [[[0, 6, 6, 9]]], "b": [[[0, 1]]], "c": [[[0, 2]]]},
        solution_source_sha256="b" * 64,
    )

    assert outcome["unique_recoveries"] == {
        "observed_visual": 0,
        "posterior_cleared": 1,
        "all_features_cleared": 1,
    }
    assert outcome["observed_minus_posterior_cleared_recoveries"] == -1
    assert outcome["posterior_performance_attribution_supported"] is False
    assert outcome["result_id"] == canonical_sha256(
        {key: value for key, value in outcome.items() if key != "result_id"}
    )


def test_allocation_outcome_rejects_rehashed_inconsistent_ablation() -> None:
    freeze = freeze_relational_transducer_gate(
        challenges=_challenges(),
        visual_freeze=_visual(),
        recursive_freeze=_recursive(),
        cohort_id="cohort",
        source_files={"protocol": "a" * 64},
    )
    audit = audit_relational_transducer_allocation_ablation(
        freeze=freeze,
        protocol_sha256="c" * 64,
    )
    tampered = copy.deepcopy(audit)
    tampered["posterior_cleared"]["selected_set_changed"] = False
    tampered["ablation_id"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "ablation_id"}
    )

    with pytest.raises(ValueError, match="comparison is inconsistent"):
        score_relational_transducer_allocation_ablation(
            freeze=freeze,
            ablation=tampered,
            solutions={
                "a": [[[0, 6, 6, 9]]],
                "b": [[[0, 1]]],
                "c": [[[0, 2]]],
            },
            solution_source_sha256="b" * 64,
        )
