from __future__ import annotations

import copy

import pytest

from afts_arc.object_graph_rewrite_gate import (
    freeze_object_graph_rewrite_opportunities,
)
from afts_arc.stateful_object_graph_rewrite_gate import (
    freeze_stateful_object_graph_rewrite_opportunities,
    score_stateful_object_graph_rewrite_freeze,
)


def _challenge() -> dict[str, object]:
    grid = [[0, 1, 0], [0, 0, 0], [0, 2, 0]]
    return {
        "task-a": {
            "train": [{"input": grid, "output": grid}],
            "test": [{"input": grid}],
        }
    }


def _incumbent() -> dict[str, object]:
    return freeze_object_graph_rewrite_opportunities(
        challenges=_challenge(),
        cohort_id="controlled-v3",
        scientific_lane="controlled_semantic",
        source_files={"protocol": "sha256"},
        max_first_stage_trials=1,
        max_parents=1,
        max_second_stage_trials=1,
        max_candidates=2,
    )


def test_query_blind_stateful_freeze_is_deterministic_and_cost_matched() -> None:
    arguments = {
        "challenges": _challenge(),
        "incumbent_freeze": _incumbent(),
        "cohort_id": "controlled-v3",
        "scientific_lane": "controlled_semantic",
        "source_files": {"protocol": "sha256"},
        "max_first_stage_trials": 1,
        "max_parents": 1,
        "max_rewrite_trials": 1,
        "max_candidates": 2,
    }
    first = freeze_stateful_object_graph_rewrite_opportunities(**arguments)
    second = freeze_stateful_object_graph_rewrite_opportunities(**arguments)

    assert first == second
    assert first["query_gold_read"] is False
    assert first["controller_training_started"] is False
    assert first["dependency_replay_required"] is True
    task = first["tasks"][0]
    assert task["native_cost"]["rewrite_arm"]["program_trials"] == 1
    assert task["native_cost"]["cold_arm"]["program_trials"] == 1
    assert task["frontier_opportunity"] == (task["novel_frontier_count"] > 0)


def test_stateful_scoring_preserves_freeze_identity() -> None:
    freeze = freeze_stateful_object_graph_rewrite_opportunities(
        challenges=_challenge(),
        incumbent_freeze=_incumbent(),
        cohort_id="controlled-v3",
        scientific_lane="controlled_semantic",
        source_files={"protocol": "sha256"},
        max_first_stage_trials=1,
        max_parents=1,
        max_rewrite_trials=1,
        max_candidates=2,
    )
    freeze_id = freeze["freeze_id"]
    result = score_stateful_object_graph_rewrite_freeze(
        freeze=freeze,
        solutions={"task-a": [[[0, 1, 0], [0, 0, 0], [0, 2, 0]]]},
        solution_source_sha256="solution-sha256",
    )

    assert freeze["freeze_id"] == freeze_id
    assert result["freeze_id"] == freeze_id
    assert result["task_count"] == 1


def test_incumbent_identity_tampering_is_rejected() -> None:
    incumbent = copy.deepcopy(_incumbent())
    incumbent["tasks"][0]["blind_content_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="incumbent freeze ID differs"):
        freeze_stateful_object_graph_rewrite_opportunities(
            challenges=_challenge(),
            incumbent_freeze=incumbent,
            cohort_id="controlled-v3",
            scientific_lane="controlled_semantic",
            source_files={"protocol": "sha256"},
            max_first_stage_trials=1,
            max_parents=1,
            max_rewrite_trials=1,
            max_candidates=2,
        )
