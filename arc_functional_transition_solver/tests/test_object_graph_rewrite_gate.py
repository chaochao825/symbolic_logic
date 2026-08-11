from __future__ import annotations

from afts_arc.object_graph_rewrite_gate import (
    freeze_object_graph_rewrite_opportunities,
    score_object_graph_rewrite_freeze,
)


def _challenge() -> dict[str, object]:
    grid = [[0, 1, 0], [0, 0, 0], [0, 2, 0]]
    return {
        "task-a": {
            "train": [{"input": grid, "output": grid}],
            "test": [{"input": grid}],
        }
    }


def test_query_blind_freeze_is_deterministic_and_cost_matched() -> None:
    arguments = {
        "challenges": _challenge(),
        "cohort_id": "controlled-test",
        "scientific_lane": "controlled_semantic",
        "source_files": {"protocol": "sha256"},
        "max_first_stage_trials": 1,
        "max_parents": 1,
        "max_second_stage_trials": 1,
        "max_candidates": 2,
    }
    first = freeze_object_graph_rewrite_opportunities(**arguments)
    second = freeze_object_graph_rewrite_opportunities(**arguments)

    assert first == second
    assert first["query_gold_read"] is False
    assert first["controller_training_started"] is False
    task = first["tasks"][0]
    assert task["native_cost"]["composition_arm"]["program_trials"] == 1
    assert task["native_cost"]["cold_arm"]["program_trials"] == 1
    assert task["frontier_opportunity"] == (task["novel_frontier_count"] > 0)


def test_post_freeze_scoring_does_not_mutate_candidate_identity() -> None:
    freeze = freeze_object_graph_rewrite_opportunities(
        challenges=_challenge(),
        cohort_id="controlled-test",
        scientific_lane="controlled_semantic",
        source_files={"protocol": "sha256"},
        max_first_stage_trials=1,
        max_parents=1,
        max_second_stage_trials=1,
        max_candidates=2,
    )
    freeze_id = freeze["freeze_id"]
    result = score_object_graph_rewrite_freeze(
        freeze=freeze,
        solutions={"task-a": [[[0, 1, 0], [0, 0, 0], [0, 2, 0]]]},
        solution_source_sha256="solution-sha256",
    )

    assert freeze["freeze_id"] == freeze_id
    assert result["freeze_id"] == freeze_id
    assert result["task_count"] == 1
