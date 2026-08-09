from __future__ import annotations

from afts_arc.blind import BlindTask
from afts_arc.grid import as_grid
from afts_arc.relational_mask import RelationalMaskProgram
from afts_arc.task import ARCPair
from scripts.afts_arc_relational_mask_confirmation_gate import (
    ACTION_PROGRAM_TRIALS,
    _deduplicate_candidates,
    _score_action_arm,
)


def _task() -> BlindTask:
    grid = [[8 for _ in range(7)] for _ in range(7)]
    for row, column in (
        (0, 3),
        (1, 3),
        (5, 3),
        (6, 3),
        (3, 0),
        (3, 1),
        (3, 5),
        (3, 6),
    ):
        grid[row][column] = 1
    output = [row[:] for row in grid]
    output[3][3] = 4
    return BlindTask.from_observations(
        train=(ARCPair(as_grid(grid), as_grid(output)),),
        test_inputs=(as_grid(grid),),
    )


def test_action_arm_reserves_equal_fixed_cost_with_padding() -> None:
    task = _task()
    exact = RelationalMaskProgram(8, 1, 4, "cardinal4", (2, 3), "copy")
    arm = _score_action_arm(
        programs=(exact,),
        padding_program=exact,
        task=task,
        source="fixture",
    )

    assert arm["reserved_program_trials"] == ACTION_PROGRAM_TRIALS
    assert arm["actual_program_trials"] == 1
    assert arm["padding_program_trials"] == ACTION_PROGRAM_TRIALS - 1
    assert arm["total_execution_count"] == ACTION_PROGRAM_TRIALS * 2
    assert arm["novel_frontier_count"] == 1
    assert len(arm["candidates"]) == 1


def test_candidate_deduplication_keeps_mdl_first() -> None:
    outputs = [[[1]]]
    candidates = (
        {
            "source": "z",
            "program_id": "b",
            "description_bits": 20,
            "program": {},
            "query_outputs": outputs,
        },
        {
            "source": "a",
            "program_id": "a",
            "description_bits": 10,
            "program": {},
            "query_outputs": outputs,
        },
    )

    assert _deduplicate_candidates(candidates) == (candidates[1],)


def test_candidate_tie_break_uses_program_id_before_source() -> None:
    outputs = [[[1]]]
    candidates = (
        {
            "source": "a_source",
            "program_id": "z_program",
            "description_bits": 10,
            "program": {},
            "query_outputs": outputs,
        },
        {
            "source": "z_source",
            "program_id": "a_program",
            "description_bits": 10,
            "program": {},
            "query_outputs": outputs,
        },
    )

    assert _deduplicate_candidates(candidates) == (candidates[1],)
