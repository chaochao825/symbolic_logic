from __future__ import annotations

from afts_arc.blind import BlindTask
from afts_arc.grid import as_grid
from afts_arc.relational_delta import enumerate_relational_delta_programs
from afts_arc.task import ARCPair
from scripts.afts_arc_relational_delta_failure_matrix import (
    classify_terminal_failure,
)


def _task(source: list[list[int]], target: list[list[int]]) -> BlindTask:
    input_grid = as_grid(source)
    return BlindTask.from_observations(
        train=(ARCPair(input_grid, as_grid(target)),),
        test_inputs=(input_grid,),
    )


def _classify(task: BlindTask) -> dict[str, object]:
    return classify_terminal_failure(
        task,
        enumerate_relational_delta_programs(task),
    )


def test_failure_matrix_classifies_canvas_incompatibility_first() -> None:
    task = _task(
        [[0, 1, 0], [0, 0, 0]],
        [[1]],
    )

    row = _classify(task)

    assert row["primary_category"] == "canvas_incompatible"
    assert row["canvas_compatible"] is False


def test_failure_matrix_classifies_role_parse_failure() -> None:
    task = _task(
        [[0, 1, 0], [0, 2, 0], [0, 0, 0]],
        [[0, 3, 0], [0, 4, 0], [0, 0, 0]],
    )

    row = _classify(task)

    assert row["primary_category"] == "parse_failure"
    assert row["programs_reaching_roles_on_all_demos"] == 0


def test_failure_matrix_classifies_missing_relation() -> None:
    task = _task(
        [[0, 1, 0, 2, 0, 2, 0]],
        [[0, 3, 0, 2, 0, 2, 0]],
    )

    row = _classify(task)

    assert row["primary_category"] == "relation_missing"
    assert row["programs_reaching_roles_on_all_demos"] > 0
    assert row["programs_reaching_relation_on_all_demos"] == 0


def test_failure_matrix_classifies_typed_ast_failure() -> None:
    task = _task(
        [[0, 1, 0, 0, 1, 0]],
        [[2, 0, 0, 3, 0, 0]],
    )

    row = _classify(task)

    assert row["primary_category"] == "ast_insufficient"
    assert row["programs_reaching_relation_on_all_demos"] > 0
    assert row["programs_legal_on_all_demos"] == 0


def test_failure_matrix_classifies_legal_but_inexact() -> None:
    task = _task(
        [[0, 0, 1, 0, 0]],
        [[0, 2, 0, 3, 0]],
    )

    row = _classify(task)

    assert row["primary_category"] == "legal_but_inexact"
    assert row["programs_legal_on_all_demos"] > 0
    assert row["best_legal_program"] is not None
