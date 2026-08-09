from __future__ import annotations

from pathlib import Path

import pytest

from afts_arc.blind import BlindTask
from afts_arc.grid import as_grid
from afts_arc.relational_delta import (
    DeltaParseNode,
    DeltaRenderNode,
    DeltaTargetNode,
    NaryRelationNode,
    RelationalDeltaProgram,
    RoleAssignmentNode,
    TypedDeltaMaskNode,
    relational_delta_program_id,
)
from afts_arc.task import ARCPair
from scripts.afts_arc_relational_delta_dev_gate import (
    ACTION_PROGRAM_TRIALS,
    EXPECTED_SEMANTIC_TEST_NAMES,
    _arm,
    _blind_task,
    _semantic_test_receipt,
)


def _recolor_task_and_programs() -> tuple[
    BlindTask, RelationalDeltaProgram, RelationalDeltaProgram
]:
    source = as_grid(
        [
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 1, 1, 4, 4, 4, 4],
            [4, 4, 1, 1, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 1, 1, 1, 4, 4, 4, 4],
            [4, 1, 4, 4, 4, 4, 4, 4],
            [4, 1, 1, 1, 4, 4, 4, 4],
        ]
    )
    target = as_grid(
        [
            [4, 4, 4, 4, 4, 4, 4, 4],
            [2, 2, 4, 4, 4, 4, 4, 4],
            [2, 2, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 3, 3, 3],
            [4, 4, 4, 4, 4, 3, 4, 4],
            [4, 4, 4, 4, 4, 3, 3, 3],
        ]
    )

    def program(palette: tuple[int, int]) -> RelationalDeltaProgram:
        return RelationalDeltaProgram(
            DeltaParseNode(4),
            RoleAssignmentNode("only_foreground", "canvas"),
            NaryRelationNode("component_to_horizontal_border"),
            DeltaTargetNode("component", "all"),
            TypedDeltaMaskNode("actor_cells"),
            DeltaRenderNode(
                "recolor_component", "target_side_palette", palette
            ),
        )

    task = BlindTask.from_observations(
        train=(ARCPair(source, target),),
        test_inputs=(source,),
    )
    return task, program((2, 3)), program((3, 2))


def test_action_arm_binds_novelty_to_content_ids_and_equal_reserved_cost() -> None:
    task, correct, parent = _recolor_task_and_programs()
    arm = _arm(
        programs=(correct,),
        padding_program=parent,
        task=task,
        source="controlled_fixture",
        initial_program_ids=(relational_delta_program_id(parent),),
    )
    assert arm["reserved_program_trials"] == ACTION_PROGRAM_TRIALS
    assert arm["actual_program_trials"] == 1
    assert arm["padding_program_trials"] == ACTION_PROGRAM_TRIALS - 1
    assert arm["total_execution_count"] == ACTION_PROGRAM_TRIALS * 2
    assert arm["novel_frontier_count"] == 1
    assert arm["frontier_changed"] is True
    assert len(arm["candidates"]) == 1


def test_blind_loader_rejects_a_non_sentinel_query_output() -> None:
    payload = {
        "train": [{"input": [[0]], "output": [[1]]}],
        "test": [{"input": [[0]], "output": [[1]]}],
    }
    with pytest.raises(ValueError, match="input-copy sentinel"):
        _blind_task(payload)


def test_semantic_receipt_is_hashed_and_fail_closed(tmp_path: Path) -> None:
    passing = tmp_path / "passing.xml"
    passing.write_text(
        '<testsuites><testsuite tests="9" failures="0" errors="0" skipped="0">'
        + "".join(
            f'<testcase name="{name}"/>'
            for name in sorted(EXPECTED_SEMANTIC_TEST_NAMES)
        )
        + "</testsuite></testsuites>",
        encoding="utf-8",
    )
    receipt = _semantic_test_receipt(passing)
    assert receipt["passed"] is True
    assert receipt["tests"] == 9
    assert len(receipt["report_sha256"]) == 64

    failing = tmp_path / "failing.xml"
    failing.write_text(
        '<testsuites><testsuite tests="9" failures="1" errors="0" skipped="0">'
        + "".join(
            f'<testcase name="{name}"/>'
            for name in sorted(EXPECTED_SEMANTIC_TEST_NAMES)
        )
        + "</testsuite></testsuites>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not pass"):
        _semantic_test_receipt(failing)
