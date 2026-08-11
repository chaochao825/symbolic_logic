from __future__ import annotations

import copy

import pytest

from afts_arc.counterfactual_transition_gate import (
    freeze_counterfactual_transition_opportunities,
    score_counterfactual_transition_freeze,
)
from afts_arc.object_graph_rewrite_gate import (
    freeze_object_graph_rewrite_opportunities,
)
from afts_arc.stateful_object_graph_rewrite_gate import (
    freeze_stateful_object_graph_rewrite_opportunities,
)


def _challenge() -> dict[str, object]:
    grid = [[0, 1, 0], [0, 0, 0], [0, 2, 0]]
    return {
        "task-a": {
            "train": [{"input": grid, "output": grid}],
            "test": [{"input": grid}],
        }
    }


def _v3_freeze() -> dict[str, object]:
    incumbent = freeze_object_graph_rewrite_opportunities(
        challenges=_challenge(),
        cohort_id="controlled-v4",
        scientific_lane="controlled_semantic",
        source_files={"protocol": "sha256"},
        max_first_stage_trials=1,
        max_parents=1,
        max_second_stage_trials=1,
        max_candidates=2,
    )
    return freeze_stateful_object_graph_rewrite_opportunities(
        challenges=_challenge(),
        incumbent_freeze=incumbent,
        cohort_id="controlled-v4",
        scientific_lane="controlled_semantic",
        source_files={"protocol": "sha256"},
        max_first_stage_trials=1,
        max_parents=1,
        max_rewrite_trials=1,
        max_candidates=2,
    )


def test_v4_freeze_is_deterministic_query_blind_and_per_arm_cost_matched() -> None:
    arguments = {
        "challenges": _challenge(),
        "v3_freeze": _v3_freeze(),
        "cohort_id": "controlled-v4",
        "scientific_lane": "controlled_semantic",
        "source_files": {"protocol": "sha256"},
        "max_first_stage_trials": 1,
        "max_parents": 1,
        "max_transition_trials": 1,
        "max_candidates": 2,
    }
    first = freeze_counterfactual_transition_opportunities(**arguments)
    second = freeze_counterfactual_transition_opportunities(**arguments)
    parallel = freeze_counterfactual_transition_opportunities(
        **arguments,
        max_workers=2,
    )

    assert first == second
    assert first["query_gold_read"] is False
    assert first["controller_training_started"] is False
    assert first["multi_arm_union_is_single_arm_comparable"] is False
    task = first["tasks"][0]
    for strategy in first["strategies"]:
        assert task["arms"][strategy]["native_cost"]["program_trials"] == 1
    assert task["frozen_cold_native_cost"]["program_trials"] == 1
    serial_content = {
        key: value
        for key, value in first.items()
        if key not in {"freeze_id", "execution_workers"}
    }
    parallel_content = {
        key: value
        for key, value in parallel.items()
        if key not in {"freeze_id", "execution_workers"}
    }
    assert serial_content == parallel_content


def test_v4_score_preserves_freeze_and_labels_union_as_three_arm_cost() -> None:
    freeze = freeze_counterfactual_transition_opportunities(
        challenges=_challenge(),
        v3_freeze=_v3_freeze(),
        cohort_id="controlled-v4",
        scientific_lane="controlled_semantic",
        source_files={"protocol": "sha256"},
        max_first_stage_trials=1,
        max_parents=1,
        max_transition_trials=1,
        max_candidates=2,
    )
    freeze_id = freeze["freeze_id"]
    result = score_counterfactual_transition_freeze(
        freeze=freeze,
        solutions={"task-a": [[[0, 1, 0], [0, 0, 0], [0, 2, 0]]]},
        solution_source_sha256="solution-sha256",
    )

    assert freeze["freeze_id"] == freeze_id
    assert result["freeze_id"] == freeze_id
    assert result["three_arm_union_uses_three_times_single_arm_reservation"] is True


def test_v4_rejects_tampered_v3_parent_freeze() -> None:
    v3_freeze = copy.deepcopy(_v3_freeze())
    v3_freeze["tasks"][0]["blind_content_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="v3 parent freeze ID differs"):
        freeze_counterfactual_transition_opportunities(
            challenges=_challenge(),
            v3_freeze=v3_freeze,
            cohort_id="controlled-v4",
            scientific_lane="controlled_semantic",
            source_files={"protocol": "sha256"},
            max_first_stage_trials=1,
            max_parents=1,
            max_transition_trials=1,
            max_candidates=2,
        )
