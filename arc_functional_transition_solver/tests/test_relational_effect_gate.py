from __future__ import annotations

import pytest

from afts_arc.experiment_safety import canonical_sha256
from afts_arc.relational_effect_gate import (
    _sensor_summary,
    audit_relational_effect_sensor,
    evaluate_relational_effect_interventions,
)


def _sensor_task(
    index: int,
    *,
    method: bool,
    identity: bool = False,
    global_recolor: bool = False,
    component: bool = False,
) -> dict[str, object]:
    return {
        "task_id": f"task-{index}",
        "family_id": f"family-{index % 3}",
        "strict_lodo": method,
        "baseline_strict_lodo": {
            "identity": identity,
            "global": global_recolor,
            "component": component,
        },
    }


def test_sensor_gate_requires_family_diversity_and_baseline_advantage() -> None:
    passing = tuple(_sensor_task(index, method=index < 5) for index in range(6))
    summary = _sensor_summary(passing)

    assert summary["method_task_count"] == 5
    assert summary["method_family_count"] == 3
    assert summary["advantage_over_strongest_baseline"] == 5
    assert summary["unique_over_all_baselines_family_count"] == 3
    assert summary["passed"] is True

    tied = tuple(
        _sensor_task(index, method=index < 5, component=index < 5)
        for index in range(6)
    )
    tied_summary = _sensor_summary(tied)
    assert tied_summary["strongest_baseline"] == "component"
    assert tied_summary["advantage_over_strongest_baseline"] == 0
    assert tied_summary["passed"] is False


def test_sensor_gate_does_not_count_correlated_episodes_as_family_diversity() -> None:
    tasks = tuple(
        {
            **_sensor_task(index, method=True),
            "family_id": "one-family",
        }
        for index in range(6)
    )

    summary = _sensor_summary(tasks)

    assert summary["method_task_count"] == 6
    assert summary["method_family_count"] == 1
    assert summary["passed"] is False


def test_failure_audit_is_query_blind_and_marks_baseline_redundancy() -> None:
    summary = {
        "schema": "afts.effect-summary/v1",
        "summary_id": "summary",
        "provenance_execution_id": "execution",
        "status": "single_delta",
        "parent_shape": [2, 2],
        "target_shape": [2, 2],
        "delta_pairs": [[0, 2]],
        "changed_cells": [[0, 0], [0, 1]],
        "source_backed_cells": [[0, 0], [0, 1]],
        "entity_backed_cells": [[0, 0]],
        "affected_entity_ids": ["entity"],
        "affected_relation_set_id": "relation-set",
        "affected_relation_count": 0,
        "affected_relation_id_sample": [],
    }
    task = {
        "task_id": "task",
        "family_id": "family",
        "strict_lodo": True,
        "baseline_strict_lodo": {
            "identity": False,
            "global": True,
            "component": True,
        },
        "parents": [
            {
                "core": {
                    "status": "identified",
                    "failure_kind": None,
                    "effect_summaries": [summary],
                },
                "lodo": {
                    "strict_all_folds_exact": True,
                    "folds": [
                        {
                            "identified": True,
                            "unanimous": True,
                            "exact": True,
                            "failure_kind": None,
                        }
                    ],
                },
            }
        ],
    }
    content = {
        "query_gold_read": False,
        "controller_training_started": False,
        "tasks": [task],
    }
    freeze = {"freeze_id": canonical_sha256(content), **content}

    audit = audit_relational_effect_sensor(freeze)

    assert audit["terminal_reason_counts"] == {"strict_but_baseline_redundant": 1}
    assert audit["source_backed_recall"] == 1.0
    assert audit["entity_backed_recall"] == 0.5
    assert audit["query_gold_read"] is False

    changed = dict(freeze)
    changed["query_gold_read"] = True
    changed_content = {key: value for key, value in changed.items() if key != "freeze_id"}
    changed["freeze_id"] = canonical_sha256(changed_content)
    with pytest.raises(ValueError, match="query-blind"):
        audit_relational_effect_sensor(changed)


def test_residual_intervention_fails_closed_before_natural_utility_gate() -> None:
    freeze_content = {"tasks": []}
    freeze = {"freeze_id": canonical_sha256(freeze_content), **freeze_content}
    result_content = {
        "freeze_id": freeze["freeze_id"],
        "natural_utility_gate_passed": False,
    }
    result = {"result_id": canonical_sha256(result_content), **result_content}

    with pytest.raises(PermissionError, match="did not authorize"):
        evaluate_relational_effect_interventions(
            challenges={},
            freeze=freeze,
            result=result,
            solutions={},
        )
