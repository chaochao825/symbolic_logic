from __future__ import annotations

import copy

import pytest

from afts_arc.experiment_safety import canonical_sha256
from afts_arc.object_program_workspace_gate import (
    ARM_NAMES,
    freeze_object_program_workspace_gate,
    score_object_program_workspace_gate,
)


def _paint(
    height: int,
    width: int,
    objects: list[tuple[int, tuple[tuple[int, int], ...]]],
) -> list[list[int]]:
    grid = [[0 for _ in range(width)] for _ in range(height)]
    for color, cells in objects:
        for row, column in cells:
            grid[row][column] = color
    return grid


def _recolor(
    grid: list[list[int]],
    cells: set[tuple[int, int]],
    color: int,
) -> list[list[int]]:
    output = [row[:] for row in grid]
    for row, column in cells:
        output[row][column] = color
    return output


def _inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    first = _paint(
        9,
        9,
        [
            (2, ((2, 2), (3, 2), (3, 3))),
            (3, ((5, 5),)),
            (4, ((0, 1), (0, 2))),
            (4, ((8, 6), (8, 7))),
        ],
    )
    first_gold = _recolor(
        first,
        {(2, 2), (3, 2), (3, 3), (5, 5)},
        8,
    )
    second = _paint(
        10,
        10,
        [
            (5, ((2, 2), (2, 3), (3, 2), (3, 3))),
            (6, ((6, 6), (7, 6))),
            (4, ((0, 1),)),
            (4, ((0, 5),)),
            (4, ((9, 8),)),
        ],
    )
    second_gold = _recolor(
        second,
        {(2, 2), (2, 3), (3, 2), (3, 3), (6, 6), (7, 6)},
        8,
    )
    query = _paint(
        9,
        9,
        [
            (2, ((3, 3),)),
            (3, ((5, 5),)),
            (4, ((0, 1), (0, 2))),
        ],
    )
    query_gold = _recolor(query, {(3, 3), (5, 5)}, 8)
    query_distractor = _recolor(query, {(0, 1), (0, 2)}, 8)
    challenges = {
        "fixture": {
            "train": [
                {"input": first, "output": first_gold},
                {"input": second, "output": second_gold},
            ],
            "test": [{"input": query}],
        }
    }
    visual = {
        "freeze_id": "visual",
        "query_gold_read": False,
        "tasks": [
            {
                "task_id": "fixture",
                "queries": [
                    {
                        "candidates": [
                            {"output": query_gold, "sample_count": 8},
                            {"output": query_distractor, "sample_count": 2},
                        ]
                    }
                ],
            }
        ],
    }
    return challenges, visual, {"fixture": [query_gold]}


def test_freeze_is_query_blind_replayable_and_cost_matched() -> None:
    challenges, visual, _ = _inputs()
    first = freeze_object_program_workspace_gate(
        challenges=challenges,
        visual_freeze=visual,
        cohort_id="cohort",
        scientific_lane="outcome_exposed_development",
        source_files={"protocol": "a" * 64},
    )
    second = freeze_object_program_workspace_gate(
        challenges=challenges,
        visual_freeze=visual,
        cohort_id="cohort",
        scientific_lane="outcome_exposed_development",
        source_files={"protocol": "a" * 64},
    )

    assert first == second
    assert first["query_gold_read"] is False
    assert first["opportunity_count"] == 1
    assert first["freeze_id"] == canonical_sha256(
        {key: value for key, value in first.items() if key != "freeze_id"}
    )
    task = first["tasks"][0]
    assert task["selection_reason"] == "eligible"
    assert task["program_opportunity"] is True
    assert task["frontier_opportunity"] is True
    costs = [task["arms"][arm]["native_cost"] for arm in ARM_NAMES]
    assert all(cost == costs[0] for cost in costs)
    assert all(
        task["arms"][arm]["frontier_changed"]
        == (task["arms"][arm]["novel_frontier_count"] > 0)
        for arm in ARM_NAMES
    )


def test_score_attributes_recovery_only_after_candidate_freeze() -> None:
    challenges, visual, solutions = _inputs()
    freeze = freeze_object_program_workspace_gate(
        challenges=challenges,
        visual_freeze=visual,
        cohort_id="cohort",
        scientific_lane="outcome_exposed_development",
        source_files={"protocol": "a" * 64},
    )
    result = score_object_program_workspace_gate(
        freeze=freeze,
        solutions=solutions,
        solution_source_sha256="b" * 64,
    )

    assert result["query_gold_read_after_freeze"] is True
    assert result["strict_task_coverage"]["complete_grammar"] == 1
    assert result["strict_task_coverage"]["visual_typed"] == 1
    assert result["unique_recovery_over_base_prefix"]["visual_typed"] == 1
    assert result["development_screen"]["applicable"] is False
    assert result["result_id"] == canonical_sha256(
        {key: value for key, value in result.items() if key != "result_id"}
    )


def test_score_rejects_rehashed_inconsistent_freeze() -> None:
    challenges, visual, solutions = _inputs()
    freeze = freeze_object_program_workspace_gate(
        challenges=challenges,
        visual_freeze=visual,
        cohort_id="cohort",
        scientific_lane="outcome_exposed_development",
        source_files={"protocol": "a" * 64},
    )
    tampered = copy.deepcopy(freeze)
    tampered["query_gold_read"] = True
    tampered["freeze_id"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "freeze_id"}
    )

    with pytest.raises(ValueError, match="not query-gold-free"):
        score_object_program_workspace_gate(
            freeze=tampered,
            solutions=solutions,
            solution_source_sha256="b" * 64,
        )


def test_control_lane_cannot_authorize_generalization_claim() -> None:
    challenges, visual, solutions = _inputs()
    freeze = freeze_object_program_workspace_gate(
        challenges=challenges,
        visual_freeze=visual,
        cohort_id="control",
        scientific_lane="controlled_semantic",
        source_files={"protocol": "a" * 64},
    )
    result = score_object_program_workspace_gate(
        freeze=freeze,
        solutions=solutions,
        solution_source_sha256="b" * 64,
    )

    assert result["controlled_semantic_gate"]["applicable"] is True
    assert result["controlled_semantic_gate"]["licenses_generalization_claim"] is False
    assert result["development_screen"]["applicable"] is False
    assert result["development_screen"]["prospective_confirmation_authorized"] is False
